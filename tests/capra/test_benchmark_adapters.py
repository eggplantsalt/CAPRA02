"""Tests for CAPRA v1 thin benchmark adapters."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.capra.adapters.benchmark_adapters import (
    get_libero_utility_eval_command,
    smoke_run_custom_split,
)
from experiments.robot.capra.pipelines.run_capra_eval import run_capra_eval


def test_libero_utility_command_is_upstream_path() -> None:
    cmd = get_libero_utility_eval_command("checkpoints/model.pt", task_suite_name="libero_spatial")
    assert "experiments/robot/libero/run_libero_eval.py" in cmd
    assert "--pretrained_checkpoint checkpoints/model.pt" in cmd
    assert "--task_suite_name libero_spatial" in cmd


def test_custom_split_smoke_supported_and_unsupported() -> None:
    ok_result = smoke_run_custom_split("support-critical-neighbor")
    assert ok_result["ok"] is True
    assert ok_result["split"] == "support-critical-neighbor"

    bad_result = smoke_run_custom_split("unknown-split")
    assert bad_result["ok"] is False
    assert "supported_splits" in bad_result


def test_run_capra_eval_custom_split_mode() -> None:
    metrics = run_capra_eval(
        supervision_path=None,
        benchmark_mode="custom_split",
        custom_split="chain-reaction",
    )
    assert metrics["adapter_ok"] == 1.0
    assert metrics["mode"] == "custom_split"
    assert metrics["adapter"]["split"] == "chain-reaction"


def test_run_capra_eval_safelibero_missing_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing_safelibero_root"
    metrics = run_capra_eval(
        supervision_path=None,
        benchmark_mode="safelibero",
        safelibero_root=str(missing_root),
        task_suite_name="safelibero_spatial",
        safety_level="I",
    )
    assert metrics["adapter_ok"] == 0.0
    assert metrics["mode"] == "safelibero"
    assert "reason" in metrics["adapter"]
