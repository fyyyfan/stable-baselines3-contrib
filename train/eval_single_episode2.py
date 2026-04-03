"""
单回合实验脚本（非 GNN）：逐 step 记录任务延迟与能耗
====================================================

实验场景与 `eval_single_episode.py` 保持一致：
  - 逐 step 记录 info 中的 total_delay / total_energy / num_tasks / overflow_count
  - 记录 reward、MultiDiscrete 动作向量（每个任务一个子动作）
  - 同时保存逐步 CSV + 汇总 JSON

区别：
  - 评估环境使用 `SatelliteSingleAgentEnv`（非 GNN 观测）

运行示例：
  cd train
  python eval_single_episode2.py \
    --model_path ./satellite_maskppo_models/final_model15.zip \
    --n_episodes 1 \
    --seed 10042 \
    --deterministic
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
from typing import Any

import numpy as np

# ── 确保 environment 包可导入 ─────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "environment"))
sys.path.insert(0, str(ROOT_DIR))

from environment.satellite_single_agent_env import SatelliteSingleAgentEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


@dataclass
class StepRecord:
    episode: int
    step: int
    action_json: str
    action_len: int
    reward: float
    total_delay: float
    total_energy: float
    num_tasks: int
    overflow_count: int
    delay_per_task: float
    energy_per_task: float
    done: bool
    # 动作掩码相关（便于 debug/复现实验）
    valid_action_count: int
    I_max: int
    B: int
    invalid_task_count: int
    action_is_valid: bool


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _action_to_list(action: Any) -> list[int]:
    if action is None:
        return []
    if isinstance(action, (list, tuple)):
        return [_safe_int(a) for a in action]
    if isinstance(action, np.ndarray):
        return [_safe_int(a) for a in action.flatten().tolist()]
    return [_safe_int(action)]


def resolve_model_path(model_path: str) -> str:
    """
    解析/纠正常见的模型路径写法，尽量减少相对路径与文件名误差导致的 FileNotFoundError。
    """
    p = Path(model_path)
    candidates: list[Path] = [p]

    # 补/去 .zip
    if p.suffix != ".zip":
        candidates.append(p.with_suffix(".zip"))
    else:
        candidates.append(p.with_suffix(""))

    # 如果是相对路径，再尝试从仓库根目录补 `train/`
    extra: list[Path] = []
    for c in candidates:
        if not c.is_absolute():
            extra.append(Path("train") / c)
    candidates.extend(extra)

    for c in candidates:
        if c.exists():
            return str(c)

    tried = "\n".join([f"- {c}" for c in candidates])
    raise FileNotFoundError(
        f"找不到模型文件：{model_path}\n已尝试以下路径：\n{tried}"
    )


def run_single_episode(
    model: MaskablePPO,
    env_kwargs: dict,
    seed: int,
    deterministic: bool = True,
) -> tuple[list[StepRecord], dict]:
    env = SatelliteSingleAgentEnv(**env_kwargs)
    try:
        obs, info = env.reset(seed=seed)
    except TypeError:
        obs, info = env.reset()

    records: list[StepRecord] = []
    done = False

    ep_reward = 0.0
    ep_total_delay = 0.0
    ep_total_energy = 0.0
    ep_total_tasks = 0
    ep_total_overflow = 0

    step_idx = 0
    t0 = time.time()

    while not done:
        masks = get_action_masks(env)
        action, _ = model.predict(
            obs,
            action_masks=masks,
            deterministic=deterministic,
        )
        action_list = _action_to_list(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        total_delay = float(info.get("total_delay", 0.0))
        total_energy = float(info.get("total_energy", 0.0))
        num_tasks = _safe_int(info.get("num_tasks", 0))
        overflow = _safe_int(info.get("overflow_count", 0))

        delay_per_task = total_delay / max(num_tasks, 1)
        energy_per_task = total_energy / max(num_tasks, 1)

        I_max = _safe_int(getattr(env, "I_max", len(action_list)), len(action_list))
        B = _safe_int(getattr(env, "B", 0), 0)
        valid_action_count = int(np.sum(masks)) if masks is not None else 0

        invalid_task_count = 0
        action_is_valid = True
        if masks is not None and B > 0 and I_max > 0:
            try:
                mask_matrix = np.asarray(masks, dtype=bool).reshape(I_max, B)
                check_n = min(num_tasks, len(action_list), I_max)
                for i in range(check_n):
                    ai = action_list[i]
                    if ai < 0 or ai >= B or not bool(mask_matrix[i, ai]):
                        invalid_task_count += 1
                action_is_valid = (invalid_task_count == 0)
            except Exception:
                invalid_task_count = 0
                action_is_valid = True

        records.append(
            StepRecord(
                episode=0,
                step=step_idx,
                action_json=json.dumps(action_list, ensure_ascii=False),
                action_len=int(len(action_list)),
                reward=float(reward),
                total_delay=total_delay,
                total_energy=total_energy,
                num_tasks=num_tasks,
                overflow_count=overflow,
                delay_per_task=float(delay_per_task),
                energy_per_task=float(energy_per_task),
                done=done,
                valid_action_count=valid_action_count,
                I_max=int(I_max),
                B=int(B),
                invalid_task_count=int(invalid_task_count),
                action_is_valid=action_is_valid,
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
        description="单回合逐 step 指标评估（非 GNN / SatelliteSingleAgentEnv）"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./satellite_maskppo_models/final_model15.zip",
        help="模型文件路径（不含 .zip 后缀）或 .zip 路径",
    )
    parser.add_argument("--n_episodes", type=int, default=1, help="评估回合数")
    parser.add_argument("--seed", type=int, default=10042, help="随机种子")
    parser.add_argument("--device", type=str, default="auto", help="推理设备: auto/cuda/cuda:0/cpu")
    parser.add_argument("--deterministic", action="store_true", help="使用确定性动作（推荐评估时打开）")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results_single_episode",
        help="输出目录（CSV+JSON）",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="可选：自定义本次实验名称（影响文件名）",
    )

    # ── 环境参数（与 GNN 版保持一致） ──
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

    # ── 加载模型 ──
    print("=" * 60)
    print(f"  加载模型: {args.model_path}")
    print("=" * 60)
    model_path = resolve_model_path(args.model_path)
    if model_path != args.model_path:
        print(f"  解析后的模型路径: {model_path}")
    model = MaskablePPO.load(model_path, device=args.device)
    print(f"  设备: {model.device}")

    all_summaries = []
    all_records: list[StepRecord] = []

    base_seed = int(args.seed)
    for ep in range(int(args.n_episodes)):
        records, summary = run_single_episode(
            model=model,
            env_kwargs=env_kwargs,
            seed=base_seed + ep,
            deterministic=bool(args.deterministic),
        )
        for r in records:
            r.episode = ep
        all_records.extend(records)
        summary["episode"] = ep
        summary["seed"] = base_seed + ep
        all_summaries.append(summary)

        print(
            f"[Episode {ep}] steps={summary['episode_steps']}, "
            f"reward={summary['episode_reward']:.4f}, "
            f"avg_delay_per_task={summary['avg_delay_per_task']:.6f}, "
            f"avg_energy_per_task={summary['avg_energy_per_task']:.6f}"
        )

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name.strip() or f"seed{args.seed}_ep{args.n_episodes}_{ts}"
    output_dir = args.output_dir

    steps_csv = os.path.join(output_dir, f"single_episode_steps_nognn_{run_name}.csv")
    summary_json = os.path.join(output_dir, f"single_episode_summary_nognn_{run_name}.json")

    save_step_records_csv(all_records, steps_csv)
    save_json(
        {
            "run_name": run_name,
            "model_path": str(model_path),
            "device": str(model.device),
            "deterministic": bool(args.deterministic),
            "env_kwargs": env_kwargs,
            "episodes": all_summaries,
        },
        summary_json,
    )

    print("-" * 60)
    print(f"逐 step CSV: {steps_csv}")
    print(f"汇总 JSON : {summary_json}")


if __name__ == "__main__":
    main()