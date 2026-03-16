"""Tiny/smoke CAPRA v1 evaluation runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from experiments.robot.capra.benchmark_adapters import (
    SafeLiberoSmokeConfig,
    smoke_run_custom_split,
    smoke_run_safelibero,
)
from experiments.robot.capra.metrics import EpisodeOutcome, compute_metrics_v1
from experiments.robot.capra.supervision_io import read_supervision_jsonl


def run_capra_eval(
    supervision_path: str | None,
    tiny: bool = False,
    benchmark_mode: str = "tiny",
    safelibero_root: str = "vlsa-aegis/safelibero",
    task_suite_name: str = "safelibero_spatial",
    safety_level: str = "I",
    custom_split: str = "support-critical-neighbor",
) -> Dict[str, Any]:
    """Run CAPRA v1 metric aggregation and return structured metrics."""
    if benchmark_mode == "safelibero":
        adapter = smoke_run_safelibero(
            SafeLiberoSmokeConfig(
                safelibero_root=safelibero_root,
                task_suite_name=task_suite_name,
                safety_level=safety_level,
            )
        )
        return {
            "adapter_ok": float(1 if adapter.get("ok") else 0),
            "num_tasks": float(adapter.get("num_tasks", 0.0)),
            "mode": "safelibero",
            "adapter": adapter,
        }

    if benchmark_mode == "custom_split":
        adapter = smoke_run_custom_split(custom_split)
        return {
            "adapter_ok": float(1 if adapter.get("ok") else 0),
            "mode": "custom_split",
            "adapter": adapter,
        }

    records = read_supervision_jsonl(supervision_path) if supervision_path else []

    if tiny:
        outcomes = [
            EpisodeOutcome(success=True, displacement_total=0.2, non_target_displacement=0.1, severe_event=False),
            EpisodeOutcome(success=False, displacement_total=0.4, non_target_displacement=0.3, severe_event=True),
        ]
    else:
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

    return compute_metrics_v1(records=records, episode_outcomes=outcomes)


def _write_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CAPRA v1 metric aggregation")
    parser.add_argument("--supervision_path", type=str, default="", help="Path to mined supervision JSONL")
    parser.add_argument("--output_path", type=str, default="tmp/capra/eval_metrics_v1.json", help="Metric output JSON")
    parser.add_argument("--tiny", action="store_true", help="Use tiny smoke episode outcomes")
    parser.add_argument(
        "--benchmark_mode",
        type=str,
        default="tiny",
        choices=["tiny", "safelibero", "custom_split"],
        help="Choose tiny metric mode or benchmark adapter smoke mode",
    )
    parser.add_argument("--safelibero_root", type=str, default="vlsa-aegis/safelibero", help="SafeLIBERO repository root")
    parser.add_argument("--task_suite_name", type=str, default="safelibero_spatial", help="SafeLIBERO suite name")
    parser.add_argument("--safety_level", type=str, default="I", help="SafeLIBERO safety level")
    parser.add_argument(
        "--custom_split",
        type=str,
        default="support-critical-neighbor",
        help="Custom split name (support-critical-neighbor or chain-reaction)",
    )
    args = parser.parse_args()

    metrics = run_capra_eval(
        supervision_path=args.supervision_path,
        tiny=args.tiny,
        benchmark_mode=args.benchmark_mode,
        safelibero_root=args.safelibero_root,
        task_suite_name=args.task_suite_name,
        safety_level=args.safety_level,
        custom_split=args.custom_split,
    )
    _write_metrics(metrics, args.output_path)
    print(json.dumps(metrics, ensure_ascii=True))


if __name__ == "__main__":
    main()
