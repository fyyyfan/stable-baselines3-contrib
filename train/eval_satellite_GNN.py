"""
MaskablePPO + GNN 模型评估脚本 —— 不同任务大小场景下的性能测试
================================================================

功能：
  1. 加载训练好的 MaskablePPO + GNN 模型
  2. 在不同任务大小（task_size）设置下创建评估环境
  3. 收集并对比各场景的平均延迟、平均能耗、平均奖励等指标
  4. 输出格式化的对比结果表格，并保存到 CSV 文件

运行：
  cd train && python eval_satellite_GNN.py

指定模型路径：
  python eval_satellite_GNN.py --model_path ./satellite_maskppo_models_gnn/best_model
"""

from __future__ import annotations

import sys
import os
import argparse
import time
from pathlib import Path
from functools import partial
from typing import Callable

import numpy as np

# ── 确保 environment 和 train 包可导入 ─────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "environment"))
sys.path.insert(0, str(ROOT_DIR / "train"))
sys.path.insert(0, str(ROOT_DIR))

from environment.satellite_env_gnn import SatelliteGNNEnv
from environment.coreForSat import Task, IoTDevice

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


# =====================================================================
# 1. 任务大小场景定义
# =====================================================================
TASK_SIZE_SCENARIOS = {
    "XS  (100-300 kbit)":  {"size_low": 1, "size_high": 3, "size_scale": 1e2},
    "S   (300-500 kbit)":  {"size_low": 3, "size_high": 5, "size_scale": 1e2},
    "M   (500-1000 kbit)": {"size_low": 5, "size_high": 10, "size_scale": 1e2},
    "L   (1000-2000 kbit)": {"size_low": 10, "size_high": 20, "size_scale": 1e2},
    "XL  (2000-4000 kbit)": {"size_low": 20, "size_high": 40, "size_scale": 1e2},
}


# =====================================================================
# 2. Monkey-patch 任务生成方法以控制任务大小
# =====================================================================
def _make_patched_generate_tasks(
    size_low: float,
    size_high: float,
    size_scale: float,
) -> Callable:
    """
    返回一个替代 IoTDevice.generate_tasks 的函数，
    使用自定义的任务大小范围。
    """
    def patched_generate_tasks(self: IoTDevice, tau: float = 1.0):
        num_arrivals = np.random.poisson(self.lambda0 * tau)
        for _ in range(num_arrivals):
            z_i = int(np.random.uniform(size_low, size_high)) * size_scale
            k_i = int(np.random.uniform(2, 10)) * 1e-4
            d_i = 3
            task = Task(size=z_i, cycles=k_i, max_delay=d_i)
            self.task_queue.append(task)

    return patched_generate_tasks


# =====================================================================
# 3. 单场景评估
# =====================================================================
def evaluate_scenario(
    model: MaskablePPO,
    env_kwargs: dict,
    scenario_name: str,
    size_low: float,
    size_high: float,
    size_scale: float,
    n_episodes: int = 20,
    seed: int = 10042,
) -> dict:
    """
    在给定任务大小场景下运行若干 episode，收集性能指标。

    Returns:
        dict: 包含各项汇总指标的字典
    """
    patched_fn = _make_patched_generate_tasks(size_low, size_high, size_scale)

    original_fn = IoTDevice.generate_tasks

    ep_rewards = []
    ep_delays = []
    ep_energies = []
    ep_overflows = []
    ep_lengths = []

    try:
        IoTDevice.generate_tasks = patched_fn

        np.random.seed(seed)

        for ep in range(n_episodes):
            env = SatelliteGNNEnv(**env_kwargs)
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            total_delay = 0.0
            total_energy = 0.0
            total_overflow = 0
            total_tasks = 0
            step_count = 0

            while not done:
                masks = get_action_masks(env)
                action, _ = model.predict(obs, action_masks=masks, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                total_delay += info.get("total_delay", 0.0)
                total_energy += info.get("total_energy", 0.0)
                total_overflow += info.get("overflow_count", 0)
                total_tasks += info.get("num_tasks", 0)
                step_count += 1
                done = terminated or truncated

            avg_delay = total_delay / max(total_tasks, 1)
            avg_energy = total_energy / max(total_tasks, 1)

            ep_rewards.append(total_reward)
            ep_delays.append(avg_delay)
            ep_energies.append(avg_energy)
            ep_overflows.append(total_overflow)
            ep_lengths.append(step_count)
            print(f"Episode {ep+1} completed")

            env.close()

    finally:
        IoTDevice.generate_tasks = original_fn

    return {
        "scenario": scenario_name,
        "reward_mean": float(np.mean(ep_rewards)),
        "reward_std": float(np.std(ep_rewards)),
        "delay_mean": float(np.mean(ep_delays)),
        "delay_std": float(np.std(ep_delays)),
        "energy_mean": float(np.mean(ep_energies)),
        "energy_std": float(np.std(ep_energies)),
        "overflow_mean": float(np.mean(ep_overflows)),
        "overflow_std": float(np.std(ep_overflows)),
        "ep_length_mean": float(np.mean(ep_lengths)),
        "n_episodes": n_episodes,
    }


# =====================================================================
# 4. 格式化输出
# =====================================================================
def print_results_table(results: list[dict]):
    """打印对比结果表格到终端。"""
    header = (
        f"{'场景':<25s} | {'平均奖励':>12s} | {'平均延迟(s)':>14s} | "
        f"{'平均能耗(J)':>14s} | {'溢出次数':>10s} | {'Episode长度':>12s}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  不同任务大小场景下的模型性能评估结果")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r['scenario']:<25s} | "
            f"{r['reward_mean']:>8.3f}±{r['reward_std']:<4.2f} | "
            f"{r['delay_mean']:>9.4f}±{r['delay_std']:<5.3f} | "
            f"{r['energy_mean']:>9.4f}±{r['energy_std']:<5.3f} | "
            f"{r['overflow_mean']:>6.1f}±{r['overflow_std']:<4.1f} | "
            f"{r['ep_length_mean']:>8.1f}"
        )

    print(sep)
    print(f"  每个场景评估 {results[0]['n_episodes']} 个 episode\n")


def save_results_csv(results: list[dict], output_path: str):
    """将结果保存为 CSV 文件。"""
    import csv

    fieldnames = [
        "scenario",
        "reward_mean", "reward_std",
        "delay_mean", "delay_std",
        "energy_mean", "energy_std",
        "overflow_mean", "overflow_std",
        "ep_length_mean", "n_episodes",
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"  评估结果已保存到: {output_path}")


# =====================================================================
# 5. 主函数
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="评估 MaskablePPO+GNN 模型在不同任务大小场景下的性能"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="./satellite_maskppo_models_gnn/best_model",
        help="模型文件路径 (不含 .zip 后缀)",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=10,
        help="每个场景评估的 episode 数量",
    )
    parser.add_argument(
        "--seed", type=int, default=10042,
        help="随机种子",
    )
    parser.add_argument(
        "--output_csv", type=str,
        default="./eval_results/task_size_eval.csv",
        help="评估结果 CSV 输出路径",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="推理设备: auto / cuda / cuda:0 / cpu",
    )
    # ── 环境参数（应与训练时保持一致） ──
    parser.add_argument("--num_satellites", type=int, default=4)
    parser.add_argument("--num_users", type=int, default=10)
    parser.add_argument("--lambda0", type=float, default=0.3)
    parser.add_argument("--I_max", type=int, default=6)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--env_update_interval", type=int, default=5)

    args = parser.parse_args()

    # ── 构建环境参数（与训练一致） ──
    env_kwargs = dict(
        num_satellites=args.num_satellites,
        num_users=args.num_users,
        lambda0=args.lambda0,
        I_max=args.I_max,
        max_steps=args.max_steps,
        env_update_interval=args.env_update_interval,
        verbose=False,
    )

    # ── 加载模型 ──
    print("=" * 60)
    print(f"  加载模型: {args.model_path}")
    print("=" * 60)
    model = MaskablePPO.load(args.model_path, device=args.device)
    print(f"  设备: {model.device}")
    print(f"  策略: {model.policy.__class__.__name__}")

    # ── 逐场景评估 ──
    all_results = []
    total_scenarios = len(TASK_SIZE_SCENARIOS)

    for idx, (scenario_name, scenario_cfg) in enumerate(TASK_SIZE_SCENARIOS.items(), 1):
        print(f"\n{'─' * 60}")
        print(f"  [{idx}/{total_scenarios}] 评估场景: {scenario_name}")
        print(f"{'─' * 60}")

        t0 = time.time()
        result = evaluate_scenario(
            model=model,
            env_kwargs=env_kwargs,
            scenario_name=scenario_name,
            size_low=scenario_cfg["size_low"],
            size_high=scenario_cfg["size_high"],
            size_scale=scenario_cfg["size_scale"],
            n_episodes=args.n_episodes,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        all_results.append(result)

        print(f"  完成 ({elapsed:.1f}s): "
              f"avg_delay={result['delay_mean']:.4f}s, "
              f"avg_energy={result['energy_mean']:.4f}J, "
              f"reward={result['reward_mean']:.3f}")

    # ── 输出汇总结果 ──
    print_results_table(all_results)
    save_results_csv(all_results, args.output_csv)


if __name__ == "__main__":
    main()
