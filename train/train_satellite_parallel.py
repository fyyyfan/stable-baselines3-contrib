"""
MaskablePPO 训练脚本 —— 卫星网络任务卸载环境（支持并行环境加速）
=================================================================

功能：
  1. 使用 MaskablePPO（带动作掩码的 PPO）训练 SatelliteSingleAgentEnv
  2. 支持 SubprocVecEnv（多进程真并行）和 DummyVecEnv（单进程顺序执行）
  3. 通过自定义 Callback 将环境指标写入 TensorBoard
  4. 使用 MaskableEvalCallback 定期评估并保存最佳模型

并行加速原理：
  - PPO 的 collect_rollouts 阶段需要与环境交互 n_steps 步
  - 使用 n_envs 个并行环境后，每一步同时收集 n_envs 条 transition
  - rollout buffer 大小变为 n_steps × n_envs，有效加速数据收集
  - 策略更新阶段（train）不受影响，仍在单 GPU/CPU 上执行

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
# SubprocVecEnv 子进程也需要能 import，将路径加到环境变量
os.environ["PYTHONPATH"] = str(ROOT_DIR / "environment") + os.pathsep + os.environ.get("PYTHONPATH", "")

from environment.satellite_single_agent_env import SatelliteSingleAgentEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


# =====================================================================
# 1. 自定义 Callback —— 记录环境特有指标到 TensorBoard（多环境适配）
# =====================================================================
class SatelliteMetricsCallback(BaseCallback):
    """
    适配多并行环境的指标记录回调。
    
    在 VecEnv 模式下：
      - infos 是长度为 n_envs 的列表，每个元素是一个 dict
      - rewards 是 shape (n_envs,) 的数组
      - dones 是 shape (n_envs,) 的数组
    
    策略：对所有环境的指标取平均值记录，episode 结束时逐环境追踪。
    """

    def __init__(self, n_envs: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.n_envs = n_envs
        # 每个环境独立追踪 episode 累计
        self._episode_rewards = [0.0] * n_envs
        self._episode_lengths = [0] * n_envs
        self._episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}] * self.n_envs)
        rewards = self.locals.get("rewards", np.zeros(self.n_envs))
        dones = self.locals.get("dones", np.zeros(self.n_envs, dtype=bool))

        # ── 聚合所有环境的逐步指标 ──
        delays = [info.get("total_delay", 0.0) for info in infos]
        energies = [info.get("total_energy", 0.0) for info in infos]
        overflows = [info.get("overflow_count", 0) for info in infos]
        num_tasks_list = [info.get("num_tasks", 0) for info in infos]

        self.logger.record("env/mean_step_reward", float(np.mean(rewards)))
        self.logger.record("env/mean_delay", float(np.mean(delays)))
        self.logger.record("env/mean_energy", float(np.mean(energies)))
        self.logger.record("env/mean_overflow", float(np.mean(overflows)))
        self.logger.record("env/mean_num_tasks", float(np.mean(num_tasks_list)))

        total_tasks = sum(num_tasks_list)
        if total_tasks > 0:
            self.logger.record("env/avg_delay_per_task",
                               sum(delays) / total_tasks)
            self.logger.record("env/avg_energy_per_task",
                               sum(energies) / total_tasks)

        # ── 逐环境追踪 episode ──
        for i in range(self.n_envs):
            self._episode_rewards[i] += float(rewards[i])
            self._episode_lengths[i] += 1

            if dones[i]:
                self._episode_count += 1
                self.logger.record("env/episode_reward", self._episode_rewards[i])
                self.logger.record("env/episode_length", self._episode_lengths[i])
                self.logger.record("env/episode_count", self._episode_count)
                if self.verbose >= 1:
                    print(f"[Episode {self._episode_count}] env#{i} "
                          f"reward={self._episode_rewards[i]:.2f}, "
                          f"length={self._episode_lengths[i]}")
                self._episode_rewards[i] = 0.0
                self._episode_lengths[i] = 0

        return True


# =====================================================================
# 2. 环境工厂函数
# =====================================================================
def make_single_env(num_satellites: int = 4,
                    num_users: int = 10,
                    lambda0: float = 0.5,
                    I_max: int | None = None,
                    max_steps: int = 100,
                    env_update_interval: int = 5,
                    verbose: bool = True) -> SatelliteSingleAgentEnv:
    """创建单个环境实例（用于非并行场景和推理演示）。"""
    return SatelliteSingleAgentEnv(
        num_satellites=num_satellites,
        num_users=num_users,
        lambda0=lambda0,
        I_max=I_max,
        max_steps=max_steps,
        env_update_interval=env_update_interval,
        verbose=verbose,
    )


def make_parallel_envs(
    n_envs: int,
    vec_env_cls: type = SubprocVecEnv,
    seed: int = 42,
    **env_kwargs,
):
    """
    创建并行向量化环境。

    Args:
        n_envs: 并行环境数量
        vec_env_cls: SubprocVecEnv（多进程）或 DummyVecEnv（单进程）
        seed: 随机种子基数，每个环境 seed = base + i
        **env_kwargs: 传递给 SatelliteSingleAgentEnv 的参数

    Returns:
        VecEnv 实例
    """
    # 训练时关闭环境内部打印
    env_kwargs.setdefault("verbose", False)

    vec_env = make_vec_env(
        env_id=SatelliteSingleAgentEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
    )
    return vec_env


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
    # ── 并行参数 ──
    n_envs: int = 4,
    vec_env_cls: str = "subproc",  # "subproc" 或 "dummy"
    # ── 训练参数 ──
    total_timesteps: int = 200_000,
    n_steps: int = 256,
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
    完整训练流程（支持并行环境加速）：
      1. 创建并行训练环境 & 单独评估环境
      2. 构建 MaskablePPO 模型
      3. 挂载自定义回调 + 评估回调
      4. 训练
      5. 保存最终模型

    关键并行参数关系：
      rollout_buffer_size = n_steps × n_envs
      每次策略更新使用的数据量 = n_steps × n_envs × n_epochs
      总迭代轮数 ≈ total_timesteps / (n_steps × n_envs)
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # ── 选择向量化方式 ──
    VecEnvCls = SubprocVecEnv if vec_env_cls == "subproc" else DummyVecEnv

    env_kwargs = dict(
        num_satellites=num_satellites,
        num_users=num_users,
        lambda0=lambda0,
        I_max=I_max,
        max_steps=max_steps,
        env_update_interval=env_update_interval,
    )

    # ── 创建并行训练环境 ──
    print("=" * 60)
    print(f"  创建训练环境: {n_envs} 个并行 ({VecEnvCls.__name__})")
    print("=" * 60)
    train_env = make_parallel_envs(
        n_envs=n_envs,
        vec_env_cls=VecEnvCls,
        seed=seed,
        **env_kwargs,
    )

    # ── 创建评估环境（单独 1 个，方便 MaskableEvalCallback） ──
    eval_env = make_parallel_envs(
        n_envs=1,
        vec_env_cls=DummyVecEnv,  # 评估用单进程即可
        seed=seed + 10000,
        **env_kwargs,
    )

    # ── 打印信息 ──
    print(f"\n  动作空间: {train_env.action_space}")
    print(f"  观测空间: {train_env.observation_space}")
    print(f"  并行环境数: {n_envs}")
    print(f"  每次 rollout 收集: n_steps={n_steps} × n_envs={n_envs} = {n_steps * n_envs} transitions")
    print(f"  总迭代轮数 ≈ {total_timesteps} / ({n_steps}×{n_envs}) = {total_timesteps // (n_steps * n_envs)}")

    # ── 构建模型 ──
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
    metrics_callback = SatelliteMetricsCallback(n_envs=n_envs, verbose=verbose)

    # eval_freq 是以 collect_rollouts 的 step 为单位的 (每步所有 n_envs 同时走一步)
    # 为了保证约每 eval_freq 个 timestep 评估一次，需除以 n_envs
    adjusted_eval_freq = max(eval_freq // n_envs, 1)

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=os.path.join(log_dir, "eval_results"),
        eval_freq=adjusted_eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        use_masking=True,
        verbose=verbose,
    )

    callback = CallbackList([metrics_callback, eval_callback])

    # ── 训练 ──
    print("\n" + "=" * 60)
    print(f"  开始训练 (total_timesteps={total_timesteps}, n_envs={n_envs})")
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
    print(f"  等效速度: {total_timesteps / elapsed:.0f} timesteps/s")

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
    """加载训练好的模型，进行推理演示（单环境，带详细打印）。"""
    env_kwargs.setdefault("verbose", True)  # 推理时开启详细打印
    env = make_single_env(**env_kwargs)

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
        max_steps=60,
        env_update_interval=5,
        # ── 并行 ──
        n_envs=4,                # 4 个并行环境，数据收集加速约 4 倍
        vec_env_cls="subproc",   # "subproc"=多进程真并行, "dummy"=单进程顺序
        # ── 训练 ──
        total_timesteps=600_000,
        n_steps=256,             # 每个环境收集 256 步，总 buffer = 256×4 = 1024
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        target_kl=0.03,
        # ── 评估 ──
        eval_freq=5_000,         # 约每 5000 个 timestep 评估一次
        n_eval_episodes=5,
        # ── 日志 ──
        log_dir="./satellite_maskppo_logs/",
        save_dir="./satellite_maskppo_models/",
        seed=42,
        verbose=1,
    )
