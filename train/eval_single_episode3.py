"""
单回合实验脚本（随机策略 / 最大算力策略）
==========================================

目标：
  - 在 `SatelliteSingleAgentEnv` 中每个 step 生成动作
  - 支持两种策略：
      random  : 随机采样，非法动作自动置 0（本地执行）
      max_comp: 根据观测中的候选节点计算资源，为每个任务选择算力最大的合法节点
                本地算力来自 f_local，卫星算力来自 sat_comp_resource，
                云端算力来自 f_cloud（默认 5）
  - 记录每步延迟、能耗、奖励等指标，并保存 CSV + JSON

运行示例：
  cd train
  python eval_single_episode3.py --seed 10042 --max_steps 60
  
  python eval_single_episode3.py --seed 10042 --max_steps 60 --policy max_comp --f_cloud 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# ── 确保 environment 包可导入 ─────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "environment"))
sys.path.insert(0, str(ROOT_DIR))

from environment.satellite_single_agent_env import SatelliteSingleAgentEnv


@dataclass
class StepRecord:
    episode: int
    step: int
    action_raw_json: str
    action_exec_json: str
    replaced_count: int
    reward: float
    total_delay: float
    total_energy: float
    num_tasks: int
    overflow_count: int
    delay_per_task: float
    energy_per_task: float
    done: bool
    valid_action_count: int
    I_max: int
    B: int


def sample_random_action(i_max: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """为 MultiDiscrete([B] * I_max) 采样随机动作。"""
    return rng.integers(low=0, high=b, size=(i_max,), dtype=np.int64)


def sample_max_comp_action(
    obs: np.ndarray,
    mask_matrix: np.ndarray,
    i_max: int,
    b: int,
    obs_task_dim: int = 3,
    obs_candi_state_dim: int = 3,
    f_cloud: float = 5.0,
) -> np.ndarray:
    """最大算力策略：根据观测向量中各候选节点的计算资源，为每个任务选择算力最大的合法节点。

    观测向量布局（与 SatelliteSingleAgentEnv._get_obs() 一致）：
        前 I_max * obs_task_dim 维  : 任务特征（size, cycles, deadline）
        后 I_max * B * obs_candi_state_dim 维 : 候选节点特征
            每个候选节点 3 个特征：[距离, 传输速率, 计算资源]
            计算资源位置：offset + 2
                target=0 (本地)   -> f_local
                target=1..M (卫星) -> sat.comp_resource
                target=M+1 (云端)  -> f_cloud（obs 中的值；若为 0 则用参数 f_cloud 兜底）

    Args:
        obs           : 当前步环境返回的观测向量，shape=(obs_size,)
        mask_matrix   : 合法动作矩阵，shape=(I_max, B)，True 表示合法
        i_max         : 最大并发任务数
        b             : 每个任务的候选节点数（B = M+2）
        obs_task_dim  : 每个任务特征维度（默认 3）
        obs_candi_state_dim : 每个候选节点特征维度（默认 3）
        f_cloud       : 云端算力兜底值（当 obs 中云端算力为 0 时使用），默认 5.0

    Returns:
        action : shape=(I_max,), 每位为所选候选节点索引
    """
    action = np.zeros(i_max, dtype=np.int64)
    task_block_size = obs_task_dim * i_max  # 任务特征块总长度

    for task_idx in range(i_max):
        best_target = 0
        best_comp = -np.inf

        for target in range(b):
            # 该 (task, target) 是否合法
            if not bool(mask_matrix[task_idx, target]):
                continue

            # 计算资源在观测向量中的绝对位置
            candi_base = task_block_size + (task_idx * b + target) * obs_candi_state_dim
            comp = float(obs[candi_base + 2])  # 计算资源特征

            # 云端候选节点（target == b-1）
            if target == b - 1:
                comp = f_cloud

            if comp > best_comp:
                best_comp = comp
                best_target = target

        action[task_idx] = best_target

    return action


def enforce_mask_or_zero(action: np.ndarray, mask_matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """
    若动作非法，则将该任务位动作置为 0。
    约定 action==0 一定合法；若 mask 中出现异常，这里仍强制置 0。
    """
    fixed = action.copy()
    replaced = 0
    i_max, b = mask_matrix.shape
    for i in range(min(i_max, fixed.shape[0])):
        a = int(fixed[i])
        illegal = (a < 0 or a >= b or not bool(mask_matrix[i, a]))
        if illegal:
            fixed[i] = 0
            replaced += 1
    return fixed, replaced


def run_single_episode(
    env_kwargs: dict,
    seed: int,
    policy: str = "random",
    f_cloud: float = 5.0,
) -> tuple[list[StepRecord], dict]:
    """
    运行单回合评估。

    Args:
        env_kwargs : 传递给 SatelliteSingleAgentEnv 的关键字参数
        seed       : 随机种子
        policy     : 动作策略，可选 "random" 或 "max_comp"
        f_cloud    : max_comp 策略中云端算力兜底值，默认 5.0
    """
    env = SatelliteSingleAgentEnv(**env_kwargs)
    rng = np.random.default_rng(seed)

    try:
        obs, info = env.reset(seed=seed)
    except TypeError:
        obs, info = env.reset()

    records: list[StepRecord] = []
    done = False
    step_idx = 0

    ep_reward = 0.0
    ep_total_delay = 0.0
    ep_total_energy = 0.0
    ep_total_tasks = 0
    ep_total_overflow = 0

    t0 = time.time()
    while not done:
        flat_masks = np.asarray(env.action_masks(), dtype=bool)
        i_max = int(env.I_max)
        b = int(env.B)
        mask_matrix = flat_masks.reshape(i_max, b)

        if policy == "max_comp":
            action_raw = sample_max_comp_action(
                obs=obs,
                mask_matrix=mask_matrix,
                i_max=i_max,
                b=b,
                obs_task_dim=env.obs_task_dim,
                obs_candi_state_dim=env.obs_candi_state_dim,
                f_cloud=f_cloud,
            )
            # max_comp 策略本身已保证合法，replaced_count 理论为 0
            action_exec, replaced_count = enforce_mask_or_zero(action_raw, mask_matrix)
        else:
            action_raw = sample_random_action(i_max=i_max, b=b, rng=rng)
            action_exec, replaced_count = enforce_mask_or_zero(action_raw, mask_matrix)

        obs, reward, terminated, truncated, info = env.step(action_exec)
        done = bool(terminated or truncated)

        total_delay = float(info.get("total_delay", 0.0))
        total_energy = float(info.get("total_energy", 0.0))
        num_tasks = int(info.get("num_tasks", 0))
        overflow = int(info.get("overflow_count", 0))

        delay_per_task = total_delay / max(num_tasks, 1)
        energy_per_task = total_energy / max(num_tasks, 1)

        records.append(
            StepRecord(
                episode=0,
                step=step_idx,
                action_raw_json=json.dumps(action_raw.tolist(), ensure_ascii=False),
                action_exec_json=json.dumps(action_exec.tolist(), ensure_ascii=False),
                replaced_count=int(replaced_count),
                reward=float(reward),
                total_delay=total_delay,
                total_energy=total_energy,
                num_tasks=num_tasks,
                overflow_count=overflow,
                delay_per_task=float(delay_per_task),
                energy_per_task=float(energy_per_task),
                done=done,
                valid_action_count=int(np.sum(flat_masks)),
                I_max=i_max,
                B=b,
            )
        )

        ep_reward += float(reward)
        ep_total_delay += total_delay
        ep_total_energy += total_energy
        ep_total_tasks += num_tasks
        ep_total_overflow += overflow
        step_idx += 1

    elapsed = time.time() - t0
    summary = {
        "episode_reward": float(ep_reward),
        "episode_steps": int(step_idx),
        "total_delay": float(ep_total_delay),
        "total_energy": float(ep_total_energy),
        "total_tasks": int(ep_total_tasks),
        "total_overflow": int(ep_total_overflow),
        "avg_delay_per_task": float(ep_total_delay / max(ep_total_tasks, 1)),
        "avg_energy_per_task": float(ep_total_energy / max(ep_total_tasks, 1)),
        "total_replaced_actions": int(sum(r.replaced_count for r in records)),
        "walltime_sec": float(elapsed),
    }
    env.close()
    return records, summary


def save_step_records_csv(records: list[StepRecord], output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    rows = [asdict(r) for r in records]
    if not rows:
        raise RuntimeError("没有采集到任何 step 记录（episode 可能立即终止）。")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(obj: dict, output_json: str) -> None:
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="单回合动作评估（支持随机策略与最大算力策略）"
    )
    parser.add_argument("--seed", type=int, default=10042, help="随机种子")
    parser.add_argument("--output_dir", type=str, default="./eval_results_single_episode", help="输出目录")
    parser.add_argument("--run_name", type=str, default="", help="可选：自定义实验名称")
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        choices=["random", "max_comp"],
        help="动作策略：random（随机）或 max_comp（最大算力）",
    )
    parser.add_argument(
        "--f_cloud",
        type=float,
        default=5.0,
        help="max_comp 策略中云端算力兜底值（当 obs 中云端算力为 0 时使用），默认 5.0",
    )

    # 场景参数（与现有脚本保持一致）
    parser.add_argument("--num_satellites", type=int, default=4)
    parser.add_argument("--num_users", type=int, default=10)
    parser.add_argument("--lambda0", type=float, default=0.3)
    parser.add_argument("--I_max", type=int, default=6)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--env_update_interval", type=int, default=5)
    parser.add_argument("--verbose_env", action="store_true", help="环境 verbose 输出")
    args = parser.parse_args()

    env_kwargs = dict(
        num_satellites=args.num_satellites,
        num_users=args.num_users,
        lambda0=args.lambda0,
        I_max=args.I_max,
        max_steps=args.max_steps,
        env_update_interval=args.env_update_interval,
        verbose=bool(args.verbose_env),
    )

    records, summary = run_single_episode(
        env_kwargs=env_kwargs,
        seed=int(args.seed),
        policy=args.policy,
        f_cloud=float(args.f_cloud),
    )
    summary["episode"] = 0
    summary["seed"] = int(args.seed)

    policy_label = args.policy
    if args.policy == "max_comp":
        policy_label = f"max_comp(f_cloud={args.f_cloud})"

    print(
        f"[{policy_label}] steps={summary['episode_steps']}, "
        f"reward={summary['episode_reward']:.4f}, "
        f"avg_delay_per_task={summary['avg_delay_per_task']:.6f}, "
        f"avg_energy_per_task={summary['avg_energy_per_task']:.6f}, "
        f"replaced={summary['total_replaced_actions']}"
    )

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name.strip() or f"seed{args.seed}_{args.policy}_ep1_{ts}"
    steps_csv = os.path.join(args.output_dir, f"single_episode_steps_{args.policy}_{run_name}.csv")
    summary_json = os.path.join(args.output_dir, f"single_episode_summary_{args.policy}_{run_name}.json")

    policy_meta = args.policy
    if args.policy == "random":
        policy_meta = "random_with_mask_fix_to_zero"
    elif args.policy == "max_comp":
        policy_meta = f"max_comp_f_cloud_{args.f_cloud}"

    save_step_records_csv(records, steps_csv)
    save_json(
        {
            "run_name": run_name,
            "policy": policy_meta,
            "f_cloud": float(args.f_cloud) if args.policy == "max_comp" else None,
            "env_kwargs": env_kwargs,
            "episode": summary,
        },
        summary_json,
    )

    print("-" * 60)
    print(f"逐 step CSV: {steps_csv}")
    print(f"汇总 JSON : {summary_json}")


if __name__ == "__main__":
    main()