"""CAPRA v1 监督数据生产入口（真实路径）。

主路径：
1. 从外部 rollout episodes JSONL 读取待挖掘数据。
2. 通过外部 env_factory 为每个 episode 构建真实环境。
3. 调用 core.mining 产出可直接用于训练查表的 supervision JSONL。

说明：
- demo/debug 路径仅用于排障，不作为默认行为。
- --debug_summary_path 仅输出 debug-only 诊断文件，不参与训练主路径。
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from experiments.robot.capra.adapters.env_adapter import EnvAdapter
from experiments.robot.capra.core.mining import MiningConfigV1, mine_episode_v1
from experiments.robot.capra.io.supervision_io import SupervisionRecord, write_supervision_jsonl


def _to_numpy_nested(value: Any) -> Any:
    if isinstance(value, list):
        try:
            return np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            return [_to_numpy_nested(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_numpy_nested(v) for k, v in value.items()}
    return value


def load_episodes_jsonl(path: str) -> List[Dict[str, Any]]:
    """读取 rollout episode JSONL（每行一个 episode）。"""
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"episodes_path 不存在: {path}")

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            timesteps = []
            for step in raw.get("timesteps", []):
                step_copy = dict(step)
                step_copy["base_action"] = np.asarray(step_copy["base_action"], dtype=np.float32)
                if "observation_input" in step_copy:
                    step_copy["observation_input"] = _to_numpy_nested(step_copy["observation_input"])
                if "info_before" in step_copy:
                    step_copy["info_before"] = _to_numpy_nested(step_copy["info_before"])
                timesteps.append(step_copy)
            raw["timesteps"] = timesteps
            rows.append(raw)

    return rows


def resolve_env_factory(factory_spec: str) -> Callable[[Dict[str, Any]], Any]:
    """解析 `module:function` 形式的环境工厂。"""
    if ":" not in factory_spec:
        raise ValueError("env_factory 必须是 module:function 形式")

    module_name, func_name = factory_spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, func_name, None)
    if not callable(factory):
        raise ValueError(f"env_factory 不可调用: {factory_spec}")
    return factory


def build_mining_debug_summary(records: List[SupervisionRecord]) -> Dict[str, Any]:
    """生成调试汇总（不参与训练主路径逻辑）。"""
    if not records:
        return {
            "num_records": 0,
            "mean_weight": 0.0,
            "max_weight": 0.0,
            "num_unique_sample_keys": 0,
        }

    weights = [float(r.weight) for r in records]
    return {
        "num_records": int(len(records)),
        "mean_weight": float(sum(weights) / len(weights)),
        "max_weight": float(max(weights)),
        "num_unique_sample_keys": int(len(set(r.sample_key for r in records))),
    }


def maybe_write_debug_summary(records: List[SupervisionRecord], debug_summary_path: Optional[str]) -> None:
    """按需写入 debug summary；默认不写。"""
    if not debug_summary_path:
        return

    summary = build_mining_debug_summary(records)
    path = Path(debug_summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)


# 功能：驱动 episode 循环并汇总所有监督记录。
# 用法：传入真实 env_adapter 与 rollout episodes。
def run_capra_mining(
    env_adapter: EnvAdapter,
    episodes: Iterable[Dict[str, Any]],
    output_path: str,
    cfg: MiningConfigV1 | None = None,
) -> List[SupervisionRecord]:
    """在 episode 集合上执行 CAPRA 监督挖掘并写盘。"""
    config = cfg or MiningConfigV1()
    all_records: List[SupervisionRecord] = []

    for episode_idx, ep in enumerate(episodes):
        instruction = str(ep.get("instruction", ""))
        dataset_name = str(ep.get("dataset_name", "")).strip().lower()
        timesteps = ep.get("timesteps", [])
        records = mine_episode_v1(
            env_adapter=env_adapter,
            instruction=instruction,
            timesteps=timesteps,
            cfg=config,
            episode_idx=episode_idx,
            dataset_name=dataset_name,
        )
        all_records.extend(records)

    write_supervision_jsonl(all_records, output_path)
    return all_records


def run_capra_mining_from_episodes_file(
    episodes_path: str,
    env_factory: Callable[[Dict[str, Any]], Any],
    output_path: str,
    cfg: MiningConfigV1 | None = None,
    debug_summary_path: Optional[str] = None,
) -> List[SupervisionRecord]:
    """真实主路径：episodes JSONL + env_factory -> supervision JSONL。"""
    episodes = load_episodes_jsonl(episodes_path)
    config = cfg or MiningConfigV1()

    all_records: List[SupervisionRecord] = []
    for episode_idx, ep in enumerate(episodes):
        env = env_factory(ep)
        env_adapter = EnvAdapter(env)
        instruction = str(ep.get("instruction", ""))
        dataset_name = str(ep.get("dataset_name", "")).strip().lower()
        records = mine_episode_v1(
            env_adapter=env_adapter,
            instruction=instruction,
            timesteps=ep.get("timesteps", []),
            cfg=config,
            episode_idx=episode_idx,
            dataset_name=dataset_name,
        )
        all_records.extend(records)

    write_supervision_jsonl(all_records, output_path)
    maybe_write_debug_summary(all_records, debug_summary_path)
    return all_records


def main() -> None:
    """命令行入口：默认真实数据生产路径。"""
    parser = argparse.ArgumentParser(description="Run CAPRA supervision mining on rollout episodes")
    parser.add_argument(
        "--episodes_path",
        type=str,
        required=True,
        help="Rollout episodes JSONL path (one episode per line)",
    )
    parser.add_argument(
        "--env_factory",
        type=str,
        required=True,
        help="Environment factory spec: module:function",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path("tmp") / "capra" / "mined_v1.jsonl"),
        help="Path to output supervision JSONL",
    )
    parser.add_argument(
        "--debug_summary_path",
        type=str,
        default="",
        help="Optional debug summary JSON path (non-training artifact)",
    )
    parser.add_argument("--epsilon_p", type=float, default=0.05)
    parser.add_argument("--delta_min", type=float, default=1e-6)
    parser.add_argument("--w_max", type=float, default=5.0)
    parser.add_argument("--short_horizon_steps", type=int, default=1)
    args = parser.parse_args()

    cfg = MiningConfigV1(
        epsilon_p=float(args.epsilon_p),
        delta_min=float(args.delta_min),
        w_max=float(args.w_max),
        short_horizon_steps=int(args.short_horizon_steps),
    )

    factory = resolve_env_factory(args.env_factory)
    records = run_capra_mining_from_episodes_file(
        episodes_path=args.episodes_path,
        env_factory=factory,
        output_path=args.output_path,
        cfg=cfg,
        debug_summary_path=args.debug_summary_path or None,
    )
    print(f"Wrote {len(records)} records to {args.output_path}")


if __name__ == "__main__":
    main()
