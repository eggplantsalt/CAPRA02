"""CAPRA v1 评测入口（默认 SafeLIBERO 主路径）。

主路径：
1. 默认走 SafeLIBERO 环境评测路径（safelibero_real）。
2. 可选走真实 LIBERO benchmark 评测（libero_real）。

调试路径（默认关闭）：
1. debug_tiny：使用内置 outcome 的最小指标联调。
2. debug_custom_split：仅做 custom split 适配探活。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from experiments.robot.capra.adapters.benchmark_adapters import (
    SafeLiberoSmokeConfig,
    smoke_run_custom_split,
    smoke_run_safelibero,
)
from experiments.robot.capra.evaluation.metrics import EpisodeOutcome, compute_metrics_v1
from experiments.robot.capra.io.supervision_io import read_supervision_jsonl


# 功能：按模式执行 CAPRA 评测；默认走 SafeLIBERO 主路径。
# 用法：CLI 或测试可调用本函数。
def run_capra_eval(
    pretrained_checkpoint: str = "",
    supervision_path: str | None = None,
    benchmark_mode: str = "safelibero_real",
    debug_mode: bool = False,
    safelibero_root: str = "vlsa-aegis/safelibero",
    task_suite_name: str = "safelibero_spatial",
    safety_level: str = "I",
    custom_split: str = "support-critical-neighbor",
    num_trials_per_task: int = 50,
    seed: int = 7,
    local_log_dir: str = "./experiments/logs",
) -> Dict[str, Any]:
    """执行 CAPRA 评测并返回结构化结果。"""
    # 兼容历史调用：将旧模式名称映射到当前模式名。
    legacy_mode_map = {
        "tiny": "debug_tiny",
        "safelibero": "safelibero_real",
        "debug_safelibero": "safelibero_real",
        "custom_split": "debug_custom_split",
    }
    legacy_mode_name = benchmark_mode if benchmark_mode in legacy_mode_map else ""
    if benchmark_mode in legacy_mode_map:
        benchmark_mode = legacy_mode_map[benchmark_mode]
        debug_mode = True

    if benchmark_mode == "safelibero_real":
        # SafeLIBERO 主路径：当前实现做环境注册与套件构造校验，返回可用性统计。
        # 注意：这里尚未接入完整策略 rollout 成功率评测，属于后续待接通能力。
        adapter = smoke_run_safelibero(
            SafeLiberoSmokeConfig(
                safelibero_root=safelibero_root,
                task_suite_name=task_suite_name,
                safety_level=safety_level,
            )
        )
        if not adapter.get("ok"):
            return {
                "mode": "safelibero_real",
                "adapter_ok": 0.0,
                "adapter": adapter,
            }
        return {
            "mode": "safelibero_real",
            "adapter_ok": 1.0,
            "task_suite_name": task_suite_name,
            "safety_level": str(safety_level).upper(),
            "num_tasks": float(adapter.get("num_tasks", 0.0)),
            "adapter": adapter,
        }

    if benchmark_mode == "libero_real":
        if not pretrained_checkpoint.strip():
            raise ValueError("libero_real 模式要求提供 --pretrained_checkpoint")

        # LIBERO 可选路径：直接复用上游真实评测入口。
        from experiments.robot.libero.run_libero_eval import GenerateConfig, eval_libero

        success_rate = eval_libero(
            GenerateConfig(
                pretrained_checkpoint=pretrained_checkpoint,
                task_suite_name=task_suite_name,
                num_trials_per_task=int(num_trials_per_task),
                seed=int(seed),
                local_log_dir=local_log_dir,
            )
        )
        return {
            "mode": "libero_real",
            "task_suite_name": task_suite_name,
            "success_rate": float(success_rate),
        }

    if not debug_mode:
        raise ValueError(
            "debug 模式默认关闭；如需 debug_tiny/debug_safelibero/debug_custom_split，请显式添加 --debug_mode"
        )

    if benchmark_mode == "debug_custom_split":
        # debug-only：检查延迟副作用 split 适配层状态。
        adapter = smoke_run_custom_split(custom_split)
        return {
            "adapter_ok": float(1 if adapter.get("ok") else 0),
            "mode": legacy_mode_name or "debug_custom_split",
            "adapter": adapter,
        }

    if benchmark_mode != "debug_tiny":
        raise ValueError(f"未知 benchmark_mode: {benchmark_mode}")

    # debug-only：tiny 指标聚合，仅用于本地排障。
    records = read_supervision_jsonl(supervision_path) if supervision_path else []
    outcomes = []
    for rec in records:
        md = rec.get("metadata", {})
        outcomes.append(
            EpisodeOutcome(
                success=bool(md.get("success", False)),
                displacement_total=float(md.get("displacement_total", 0.0)),
                non_target_displacement=float(md.get("non_target_displacement", 0.0)),
                severe_event=bool(md.get("severe_event", False)),
            )
        )

    if not outcomes:
        outcomes = [
            EpisodeOutcome(success=True, displacement_total=0.2, non_target_displacement=0.1, severe_event=False),
            EpisodeOutcome(success=False, displacement_total=0.4, non_target_displacement=0.3, severe_event=True),
        ]

    metrics = compute_metrics_v1(records=records, episode_outcomes=outcomes)
    metrics["mode"] = legacy_mode_name or "debug_tiny"
    return metrics


# 功能：持久化评测结果到磁盘。
# 用法：main 中调用，默认输出到 tmp/capra 目录。
def _write_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """将指标字典写入 JSON 文件。"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# 功能：参数解析 + 分支调度 + 结果写盘 + 终端打印。
# 用法：python -m experiments.robot.capra.pipelines.run_capra_eval ...
def main() -> None:
    """命令行入口：默认 SafeLIBERO 主路径。"""

    parser = argparse.ArgumentParser(description="Run CAPRA evaluation (default: SafeLIBERO main path)")
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="",
        help="Checkpoint path for LIBERO evaluation (required only in libero_real mode)",
    )
    parser.add_argument("--supervision_path", type=str, default="", help="Path to mined supervision JSONL (debug_tiny only)")
    parser.add_argument("--output_path", type=str, default="tmp/capra/eval_metrics_v1.json", help="Metric output JSON")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug-only modes")
    parser.add_argument(
        "--benchmark_mode",
        type=str,
        default="safelibero_real",
        choices=["safelibero_real", "libero_real", "debug_tiny", "debug_custom_split", "debug_safelibero"],
        help="Evaluation mode (default main path: safelibero_real)",
    )
    parser.add_argument("--safelibero_root", type=str, default="vlsa-aegis/safelibero", help="SafeLIBERO root (safelibero_real)")
    parser.add_argument("--task_suite_name", type=str, default="safelibero_spatial", help="Task suite for safelibero_real or libero_real")
    parser.add_argument("--safety_level", type=str, default="I", help="SafeLIBERO safety level (safelibero_real)")
    parser.add_argument(
        "--custom_split",
        type=str,
        default="support-critical-neighbor",
        help="Custom split name (debug_custom_split only)",
    )
    parser.add_argument("--num_trials_per_task", type=int, default=50, help="LIBERO rollouts per task (libero_real)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for real eval")
    parser.add_argument("--local_log_dir", type=str, default="./experiments/logs", help="Local log dir for real eval")
    args = parser.parse_args()

    metrics = run_capra_eval(
        pretrained_checkpoint=args.pretrained_checkpoint,
        supervision_path=args.supervision_path,
        benchmark_mode=args.benchmark_mode,
        debug_mode=args.debug_mode,
        safelibero_root=args.safelibero_root,
        task_suite_name=args.task_suite_name,
        safety_level=args.safety_level,
        custom_split=args.custom_split,
        num_trials_per_task=args.num_trials_per_task,
        seed=args.seed,
        local_log_dir=args.local_log_dir,
    )
    _write_metrics(metrics, args.output_path)
    print(json.dumps(metrics, ensure_ascii=True))


if __name__ == "__main__":
    main()
