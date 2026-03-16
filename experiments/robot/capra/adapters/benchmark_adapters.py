"""Thin benchmark adapters for CAPRA Phase 6."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import importlib
import sys


SUPPORTED_CUSTOM_SPLITS = {
    "support-critical-neighbor": "Delayed side-effect split focused on fragile support interactions.",
    "chain-reaction": "Delayed side-effect split focused on cascade side effects.",
}

SUPPORTED_SAFELIBERO_LEVELS = {"I", "II"}


@dataclass
class SafeLiberoSmokeConfig:
    safelibero_root: str = "vlsa-aegis/safelibero"
    task_suite_name: str = "safelibero_spatial"
    safety_level: str = "I"


def get_libero_utility_eval_command(pretrained_checkpoint: str, task_suite_name: str = "libero_spatial") -> str:
    """Return standard upstream LIBERO utility eval command without modifying upstream flow."""
    return (
        "python experiments/robot/libero/run_libero_eval.py "
        f"--pretrained_checkpoint {pretrained_checkpoint} "
        f"--task_suite_name {task_suite_name}"
    )


def smoke_run_safelibero(config: SafeLiberoSmokeConfig) -> Dict[str, Any]:
    """SafeLIBERO 烟雾检查：验证路径、注册表与套件构造是否可用。"""
    safety_level = str(config.safety_level).upper()
    if safety_level not in SUPPORTED_SAFELIBERO_LEVELS:
        return {
            "ok": False,
            "reason": f"Invalid safety level: {config.safety_level}",
            "supported_safety_levels": sorted(SUPPORTED_SAFELIBERO_LEVELS),
            "available_suites": [],
        }

    root = Path(config.safelibero_root)
    if not root.exists():
        return {
            "ok": False,
            "reason": f"SafeLIBERO root not found: {config.safelibero_root}",
            "available_suites": [],
        }

    # 同时尝试两个常见 Python 根路径：仓库根目录与其 libero 子目录。
    candidate_roots = [root.resolve(), (root / "libero").resolve()]
    for candidate in candidate_roots:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    benchmark = importlib.import_module("libero.libero.benchmark")
    bench_dict = benchmark.get_benchmark_dict()

    available = sorted(bench_dict.keys())
    if config.task_suite_name not in available:
        return {
            "ok": False,
            "reason": f"Suite not found: {config.task_suite_name}",
            "available_suites": available,
        }

    suite_ctor = bench_dict[config.task_suite_name]
    suite = suite_ctor(safety_level=safety_level)

    sample_task = None
    if suite.n_tasks > 0:
        task = suite.get_task(0)
        sample_task = {"name": task.name, "language": task.language}

    return {
        "ok": True,
        "suite": config.task_suite_name,
        "safety_level": safety_level,
        "num_tasks": int(suite.n_tasks),
        "sample_task": sample_task,
        "available_suites": available,
    }


def smoke_run_custom_split(split_name: str) -> Dict[str, Any]:
    """自定义 split 烟雾检查：仅允许 v1 已声明的高价值 split。"""
    if split_name not in SUPPORTED_CUSTOM_SPLITS:
        return {
            "ok": False,
            "reason": f"Unsupported split: {split_name}",
            "supported_splits": sorted(SUPPORTED_CUSTOM_SPLITS.keys()),
        }

    return {
        "ok": True,
        "split": split_name,
        "description": SUPPORTED_CUSTOM_SPLITS[split_name],
        "status": "smoke_ready",
    }
