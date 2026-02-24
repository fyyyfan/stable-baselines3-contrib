"""
MaskablePPO + GNN 训练脚本 —— 异构 GAT 拓扑特征提取
=====================================================

功能：
  1. 使用 MaskablePPO + GNNFeaturesExtractor 训练 SatelliteGNNEnv
  2. GNN 编码器从异构二部图（task-sat-sat）中提取拓扑特征
  3. 支持 SubprocVecEnv / DummyVecEnv 并行加速
  4. 通过自定义 Callback 将环境指标写入 TensorBoard
  5. 使用 MaskableEvalCallback 定期评估并保存最佳模型

与 train_satellite_parallel.py 的区别：
  - 使用 SatelliteGNNEnv (Dict 观测空间) 代替 SatelliteSingleAgentEnv
  - 使用 "MultiInputPolicy" 代替 "MlpPolicy" 以支持 Dict 观测
  - 通过 policy_kwargs 注入 GNNFeaturesExtractor 作为特征提取器

运行：
  cd train && python train_satellite_GNN.py

指定 GPU 训练（例如使用第 0 号 GPU）：
  CUDA_VISIBLE_DEVICES=0 python train_satellite_GNN.py
  或在代码中调用 train(..., device="cuda:0")

TensorBoard 可视化：
  tensorboard --logdir=./satellite_maskppo_logs/
"""

from __future__ import annotations

import sys
import os
import torch
import platform
import time
from pathlib import Path

import numpy as np

# ── 确保 environment 和 train 包可导入 ─────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "environment"))
sys.path.insert(0, str(ROOT_DIR / "train"))
sys.path.insert(0, str(ROOT_DIR))
# SubprocVecEnv 子进程也需要能 import
os.environ["PYTHONPATH"] = (
    str(ROOT_DIR / "environment") + os.pathsep
    + str(ROOT_DIR / "train") + os.pathsep
    + str(ROOT_DIR) + os.pathsep
    + os.environ.get("PYTHONPATH", "")
)

from environment.satellite_env_gnn import SatelliteGNNEnv
from train.gnn_encoder import GNNFeaturesExtractor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, sync_envs_normalization


# =====================================================================
# 1. 自定义 Callback —— 评估环境：记录环境指标到 TensorBoard（多环境适配）
# =====================================================================
class SatelliteEvalCallback(MaskableEvalCallback):
    """
    扩展 MaskableEvalCallback，在评估时额外记录
    eval/avg_delay_per_task 和 eval/avg_energy_per_task。
    通过 evaluate_policy 的 callback 参数在每个评估 step 收集 info 指标，
    episode 结束时按任务数加权计算每任务平均延迟/能耗。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_ep_delays: list[float] = []
        self._eval_ep_energies: list[float] = []
        self._cur_delay: dict[int, float] = {}
        self._cur_energy: dict[int, float] = {}
        self._cur_tasks: dict[int, int] = {}

    def _collect_eval_metrics(self, locals_dict, globals_dict):
        """在 evaluate_policy 每个 step 后被调用，积累指标。"""
        self._log_success_callback(locals_dict, globals_dict)

        info = locals_dict["info"]
        done = locals_dict["done"]
        i = locals_dict["i"]

        self._cur_delay.setdefault(i, 0.0)
        self._cur_energy.setdefault(i, 0.0)
        self._cur_tasks.setdefault(i, 0)

        self._cur_delay[i] += info.get("total_delay", 0.0)
        self._cur_energy[i] += info.get("total_energy", 0.0)
        self._cur_tasks[i] += info.get("num_tasks", 0)

        if done:
            total_tasks = self._cur_tasks[i]
            if total_tasks > 0:
                self._eval_ep_delays.append(self._cur_delay[i] / total_tasks)
                self._eval_ep_energies.append(self._cur_energy[i] / total_tasks)
            self._cur_delay[i] = 0.0
            self._cur_energy[i] = 0.0
            self._cur_tasks[i] = 0

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way"
                    ) from e

            self._is_success_buffer = []
            self._eval_ep_delays = []
            self._eval_ep_energies = []
            self._cur_delay = {}
            self._cur_energy = {}
            self._cur_tasks = {}

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._collect_eval_metrics,
                use_masking=self.use_masking,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._eval_ep_delays) > 0:
                mean_delay = float(np.mean(self._eval_ep_delays))
                mean_energy = float(np.mean(self._eval_ep_energies))
                self.logger.record("eval/avg_delay_per_task", mean_delay)
                self.logger.record("eval/avg_energy_per_task", mean_energy)
                if self.verbose > 0:
                    print(
                        f"Eval avg_delay_per_task={mean_delay:.4f}, "
                        f"avg_energy_per_task={mean_energy:.4f}"
                    )

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_mean_reward = float(mean_reward)
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


# =====================================================================
# 1-b. 自定义 Callback —— 记录环境指标到 TensorBoard（多环境适配）
# =====================================================================
class SatelliteMetricsCallback(BaseCallback):
    """
    适配多并行环境的指标记录回调。

    在 VecEnv 模式下：
      - infos 是长度为 n_envs 的列表
      - rewards 是 shape (n_envs,) 的数组
      - dones 是 shape (n_envs,) 的数组

    策略：对所有环境的指标取平均值记录，episode 结束时逐环境追踪。
    """

    def __init__(self, n_envs: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.n_envs = n_envs
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
            self.logger.record(
                "env/avg_delay_per_task", sum(delays) / total_tasks
            )
            self.logger.record(
                "env/avg_energy_per_task", sum(energies) / total_tasks
            )

        # ── 逐环境追踪 episode ──
        for i in range(self.n_envs):
            self._episode_rewards[i] += float(rewards[i])
            self._episode_lengths[i] += 1

            if dones[i]:
                self._episode_count += 1
                self.logger.record("env/episode_reward",
                                   self._episode_rewards[i])
                self.logger.record("env/episode_length",
                                   self._episode_lengths[i])
                self.logger.record("env/episode_count",
                                   self._episode_count)
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
def make_gnn_env(
    num_satellites: int = 4,
    num_users: int = 10,
    lambda0: float = 0.5,
    I_max: int | None = None,
    max_steps: int = 100,
    env_update_interval: int = 5,
    verbose: bool = True,
) -> SatelliteGNNEnv:
    """创建单个 GNN 环境实例（用于推理演示）。"""
    return SatelliteGNNEnv(
        num_satellites=num_satellites,
        num_users=num_users,
        lambda0=lambda0,
        I_max=I_max,
        max_steps=max_steps,
        env_update_interval=env_update_interval,
        verbose=verbose,
    )


def make_parallel_gnn_envs(
    n_envs: int,
    vec_env_cls: type = SubprocVecEnv,
    seed: int = 42,
    **env_kwargs,
):
    """
    创建并行向量化 GNN 环境。

    Args:
        n_envs:      并行环境数量
        vec_env_cls: SubprocVecEnv（多进程）或 DummyVecEnv（单进程）
        seed:        随机种子
        **env_kwargs: 传递给 SatelliteGNNEnv 的参数
    """
    env_kwargs.setdefault("verbose", False)

    vec_env = make_vec_env(
        env_id=SatelliteGNNEnv,
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
    env_update_interval: int = 5, #更新网络拓扑的周期
    # ── 并行参数 ──
    n_envs: int = 4,
    vec_env_cls: str = "subproc",  # "subproc" 或 "dummy"
    # ── GNN 参数 ──
    gnn_features_dim: int = 128,
    gnn_hidden_dim: int = 64,
    gnn_num_layers: int = 2,
    gnn_num_heads: int = 4,
    gnn_dropout: float = 0.0,
    # ── 训练参数 ──
    total_timesteps: int = 600_000,
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
    eval_freq: int = 6_000,
    n_eval_episodes: int = 5,
    # ── 日志 / 保存 ──
    log_dir: str = "./satellite_maskppo_logs/",
    save_dir: str = "./satellite_maskppo_models_gnn/",
    seed: int = 42,
    verbose: int = 1,
    # ── 设备 ──
    device: str | None = "auto",  # "auto" | "cuda" | "cuda:0" | "cpu"
):
    """
    完整训练流程（MaskablePPO + GNN 特征提取）：
      1. 创建并行训练环境 (SatelliteGNNEnv) & 评估环境
      2. 构建 MaskablePPO (MultiInputPolicy + GNNFeaturesExtractor)
      3. 挂载自定义回调 + 评估回调
      4. 训练
      5. 保存最终模型
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

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
    print(f"  创建 GNN 训练环境: {n_envs} 个并行 ({VecEnvCls.__name__})")
    print("=" * 60)
    train_env = make_parallel_gnn_envs(
        n_envs=n_envs,
        vec_env_cls=VecEnvCls,
        seed=seed,
        **env_kwargs,
    )

    # ── 创建评估环境 ──
    eval_env = make_parallel_gnn_envs(
        n_envs=1,
        vec_env_cls=DummyVecEnv,
        seed=seed + 10000,
        **env_kwargs,
    )

    # ── 打印信息 ──
    print(f"\n  动作空间: {train_env.action_space}")
    print(f"  观测空间 (Dict):")
    obs_space = train_env.observation_space
    for key in obs_space.spaces:
        print(f"    {key}: {obs_space[key].shape}")
    print(f"  并行环境数: {n_envs}")
    print(f"  每次 rollout: n_steps={n_steps} × n_envs={n_envs}"
          f" = {n_steps * n_envs} transitions")

    # ── 构建 MaskablePPO + GNN ──
    print("\n" + "=" * 60)
    print("  构建 MaskablePPO + GNNFeaturesExtractor")
    print("=" * 60)

    policy_kwargs = dict(
        features_extractor_class=GNNFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=gnn_features_dim,
            hidden_dim=gnn_hidden_dim,
            num_gnn_layers=gnn_num_layers,
            num_heads=gnn_num_heads,
            edge_dim=2,       # [distance, rate]
            dropout=gnn_dropout,
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    model = MaskablePPO(
        policy="MultiInputPolicy",   # Dict 观测空间需使用 MultiInputPolicy
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
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=verbose,
        seed=seed,
        device=device,
    )

    print(f"  策略网络: {model.policy.__class__.__name__}")
    print(f"  特征提取器: {model.policy.features_extractor.__class__.__name__}")
    print(f"  GNN 参数: features_dim={gnn_features_dim}, "
          f"hidden_dim={gnn_hidden_dim}, "
          f"layers={gnn_num_layers}, heads={gnn_num_heads}")
    print(f"  设备: {model.device}")

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(
        p.numel() for p in model.policy.parameters() if p.requires_grad
    )
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    # ── 回调 ──
    metrics_callback = SatelliteMetricsCallback(
        n_envs=n_envs, verbose=verbose
    )

    adjusted_eval_freq = max(eval_freq // n_envs, 1)

    eval_callback = SatelliteEvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=os.path.join(log_dir, "eval_results_gnn"),
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
    print(f"  使用 GNN 特征提取器进行拓扑感知决策")
    print("=" * 60)

    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=1,
        tb_log_name="MaskablePPO_GNN_Satellite",
        use_masking=True,
    )
    elapsed = time.time() - t0
    print(f"\n  训练完成！用时 {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print(f"  等效速度: {total_timesteps / elapsed:.0f} timesteps/s")

    # ── 保存最终模型 ──
    final_path = os.path.join(save_dir, "final_model_gnn")
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
    """加载训练好的 GNN 模型，进行推理演示。"""
    env_kwargs.setdefault("verbose", True)
    env = make_gnn_env(**env_kwargs)

    model = MaskablePPO.load(model_path)

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        while not done:
            masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=masks,
                                      deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

        print(f"[Demo Episode {ep + 1}] steps={step}, "
              f"reward={total_reward:.4f}")

    env.close()


# =====================================================================
# 5. 入口
# =====================================================================
if __name__ == "__main__":
    trained_model = train(
        # ── 环境 ──
        num_satellites=4, #4;8
        num_users=20, #10
        lambda0=0.25, #0.3
        I_max=10, #6
        max_steps=60,
        env_update_interval=5,
        # ── 并行 ──
        n_envs=6, #4;6
        vec_env_cls="subproc",
        # ── GNN ──
        gnn_features_dim=128, #128;256
        gnn_hidden_dim=64, #64;128
        gnn_num_layers=2,
        gnn_num_heads=4,
        gnn_dropout=0.0,
        # ── 训练 ──
        total_timesteps=1000_000,
        n_steps=1024, #256;1024
        batch_size=256, #64;256
        n_epochs=5, #5;10
        learning_rate=3e-4, #3e-4, 1e-4
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02, #0.02;0.01
        target_kl=0.03,
        # ── 评估 ──
        eval_freq=10_000,
        n_eval_episodes=5,
        # ── 日志 ──
        log_dir="./satellite_maskppo_logs/",
        save_dir="./satellite_maskppo_models_gnn/",
        seed=42,
        verbose=1,
        # 指定设备: "auto"(有 GPU 则用 GPU), "cuda:0", "cuda:1", "cpu"
        # 注意: 使用 CUDA_VISIBLE_DEVICES=1 时，应设置为 "cuda:0" 或 "cuda"
        device="cuda:1",
    )
