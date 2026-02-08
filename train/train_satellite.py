"""
MaskablePPO 训练脚本 —— 卫星网络任务卸载环境
=============================================

功能：
  1. 使用 MaskablePPO（带动作掩码的 PPO）训练 SatelliteSingleAgentEnv
  2. 通过自定义 Callback 将环境指标（delay, energy, overflow, reward 各分量）写入 TensorBoard
  3. 使用 MaskableEvalCallback 定期评估并保存最佳模型
  4. 训练结束后运行推理演示

运行：
  cd train && python train_satellite.py

TensorBoard 可视化：
  tensorboard --logdir=./train/satellite_maskppo_logs/
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path

import numpy as np

# ── 确保 environment 包可导入 ──────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "environment"))

from environment.satellite_single_agent_env import SatelliteSingleAgentEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import TensorBoardOutputFormat


# =====================================================================
# 1. 自定义 Callback —— 记录环境特有指标到 TensorBoard
# =====================================================================
class SatelliteMetricsCallback(BaseCallback):
    """
    在每个 step 结束后，从 info dict 中提取卫星环境的核心指标，
    写入 SB3 logger（自动同步到 TensorBoard）。

    记录的指标：
      env/total_delay       — 单步总延迟 (s)
      env/total_energy      — 单步总能耗 (J)
      env/overflow_count    — 单步溢出（超时 + 缓冲区溢出）任务数
      env/num_tasks         — 单步处理的任务数
      env/episode_reward    — 完整 episode 累计奖励
      env/episode_length    — 完整 episode 步数
      env/avg_delay_per_task  — 单步每任务平均延迟
      env/avg_energy_per_task — 单步每任务平均能耗
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_count = 0

    def _on_step(self) -> bool:
        # infos 是一个 list (VecEnv 返回)，取第一个环境
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}

        total_delay = info.get("total_delay", 0.0)
        total_energy = info.get("total_energy", 0.0)
        overflow_count = info.get("overflow_count", 0)
        num_tasks = info.get("num_tasks", 0)
        reward = self.locals.get("rewards", [0.0])
        reward_val = float(reward[0]) if hasattr(reward, '__getitem__') else float(reward)

        # 逐步记录
        self.logger.record("env/step_reward", reward_val)
        self.logger.record("env/total_delay", total_delay)
        self.logger.record("env/total_energy", total_energy)
        self.logger.record("env/overflow_count", overflow_count)
        self.logger.record("env/num_tasks", num_tasks)

        if num_tasks > 0:
            self.logger.record("env/avg_delay_per_task", total_delay / num_tasks)
            self.logger.record("env/avg_energy_per_task", total_energy / num_tasks)

        # episode 累计
        self._episode_reward += reward_val
        self._episode_length += 1

        # 检测 episode 结束
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self._episode_count += 1
            self.logger.record("env/episode_reward", self._episode_reward)
            self.logger.record("env/episode_length", self._episode_length)
            self.logger.record("env/episode_count", self._episode_count)
            if self.verbose >= 1:
                print(f"[Episode {self._episode_count}] "
                      f"reward={self._episode_reward:.2f}, "
                      f"length={self._episode_length}")
            self._episode_reward = 0.0
            self._episode_length = 0

        return True


# =====================================================================
# 2. 环境工厂函数
# =====================================================================
def make_env(num_satellites: int = 4,
             num_users: int = 10,
             lambda0: float = 0.5,
             I_max: int | None = None,
             max_steps: int = 100,
             env_update_interval: int = 5,
             seed: int | None = None,
             ) -> SatelliteSingleAgentEnv:
    """创建并返回卫星任务卸载环境实例。"""
    env = SatelliteSingleAgentEnv(
        num_satellites=num_satellites,
        num_users=num_users,
        lambda0=lambda0,
        I_max=I_max,
        max_steps=max_steps,
        env_update_interval=env_update_interval,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


# =====================================================================
# 3. 训练主函数
# =====================================================================
def train(
    # ── 环境参数 ──
    num_satellites: int = 4,
    num_users: int = 10,
    lambda0: float = 0.5,
    I_max: int | None = 8,
    max_steps: int = 60,
    env_update_interval: int = 5,
    # ── 训练参数 ──
    total_timesteps: int = 200_000,
    n_steps: int = 1024,
    batch_size: int = 64,
    n_epochs: int = 10,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: float | None = None,
    # ── 评估参数 ──
    eval_freq: int = 5_000,
    n_eval_episodes: int = 5,
    # ── 日志 / 保存 ──
    log_dir: str = "./satellite_maskppo_logs/",
    save_dir: str = "./satellite_maskppo_models/",
    seed: int = 42,
    verbose: int = 1,
):
    """
    完整训练流程：
      1. 创建训练 & 评估环境
      2. 构建 MaskablePPO 模型
      3. 挂载自定义回调 + 评估回调
      4. 训练
      5. 保存最终模型
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # ── 创建环境 ──
    print("=" * 60)
    print("  创建训练环境 & 评估环境")
    print("=" * 60)
    train_env = make_env(num_satellites, num_users, lambda0, I_max, max_steps, env_update_interval, seed=seed)
    eval_env = make_env(num_satellites, num_users, lambda0, I_max, max_steps, env_update_interval, seed=seed + 1000)

    # ── 打印环境信息 ──
    print(f"\n  动作空间: {train_env.action_space}")
    print(f"  观测空间: {train_env.observation_space}")
    print(f"  I_max={train_env.I_max}, B={train_env.B}")
    print(f"  action_masks 维度: ({train_env.I_max * train_env.B},)")
    print(f"  MaskablePPO 期望的 mask 维度: ({sum(train_env.action_space.nvec)},)")
    assert train_env.I_max * train_env.B == sum(train_env.action_space.nvec), \
        "action_masks 维度与 MultiDiscrete nvec 不匹配！"
    print(f"  ✓ action_masks 维度与 MultiDiscrete(nvec) 完全匹配")

    # ===================== 构建模型 =====================
    print("\n" + "=" * 60)
    print("  构建 MaskablePPO 模型")
    print("=" * 60)

    model = MaskablePPO(
        policy=MaskableActorCriticPolicy,
        env=train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        tensorboard_log=log_dir,
        verbose=verbose,
        seed=seed,
    )

    print(f"  策略网络: {model.policy}")
    print(f"  设备: {model.device}")

    # ── 回调 ──
    metrics_callback = SatelliteMetricsCallback(verbose=verbose)

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=os.path.join(log_dir, "eval_results"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        use_masking=True,
        verbose=verbose,
    )

    callback = CallbackList([metrics_callback, eval_callback])

    # ── 训练 ──
    print("\n" + "=" * 60)
    print(f"  开始训练 (total_timesteps={total_timesteps})")
    print("=" * 60)

    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=1,
        tb_log_name="MaskablePPO_Satellite",
        use_masking=True,
    )
    elapsed = time.time() - t0
    print(f"\n  训练完成！用时 {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    # ── 保存最终模型 ──
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"  最终模型已保存到: {final_path}")

    # ── 最终评估 ──
    print("\n" + "=" * 60)
    print("  最终评估")
    print("=" * 60)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env,
        n_eval_episodes=20,
        deterministic=True,
    )
    print(f"  评估结果: mean_reward={mean_reward:.4f} ± {std_reward:.4f}")

    train_env.close()
    eval_env.close()

    print(f"\n  TensorBoard 可视化:")
    print(f"    tensorboard --logdir={os.path.abspath(log_dir)}")

    return model


# =====================================================================
# 4. 推理演示
# =====================================================================
def demo_inference(model_path: str, num_episodes: int = 3, **env_kwargs):
    """加载训练好的模型，进行推理演示。"""
    env = make_env(**env_kwargs)
    model = MaskablePPO.load(model_path)

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

        print(f"[Demo Episode {ep + 1}] steps={step}, reward={total_reward:.4f}")

    env.close()


# =====================================================================
# 5. 入口
# =====================================================================
if __name__ == "__main__":
    trained_model = train(
        # ── 环境 ──
        num_satellites=4,
        num_users=10,
        lambda0=0.3,
        I_max=6,
        max_steps=60, #单个 episode 的最大步数
        env_update_interval=5,
        # ── 训练 ──
        total_timesteps=200_000, #总训练步数
        n_steps=600, #每次 collect_rollouts 收集多少步经验后再更新策略
        batch_size=64, #每次更新策略时，从经验回放池中采样多少步经验
        n_epochs=10, #对同一批 rollout 数据重复训练几轮
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,         # GAE 偏差-方差平衡，经典值
        clip_range=0.2,          # PPO 经典值
        ent_coef=0.02,           # 动作空间大 → 适当提高探索，0.01~0.05
        target_kl=0.03,          # KL 散度早停
        # ── 评估 ──
        eval_freq=1_000,
        n_eval_episodes=5,
        # ── 日志 ──
        log_dir="./satellite_maskppo_logs/",
        save_dir="./satellite_maskppo_models/",
        seed=42,
        verbose=1,
    )
