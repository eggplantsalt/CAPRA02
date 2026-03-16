"""Smoke tests for CAPRA finetune overlay entrypoint."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.capra.io.supervision_io import SupervisionRecord, write_supervision_jsonl


def _load_finetune_capra_module():
    module_path = REPO_ROOT / "vla-scripts" / "finetune_capra.py"
    spec = importlib.util.spec_from_file_location("finetune_capra", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tiny_smoke_logs_anchor_and_capra_losses() -> None:
    mod = _load_finetune_capra_module()
    logs = mod.run_tiny_capra_smoke(steps=2)

    assert "anchor_loss" in logs
    assert "capra_loss" in logs
    assert "total_loss" in logs
    assert logs["anchor_loss"] >= 0.0
    assert logs["capra_loss"] >= 0.0


def test_run_from_supervision_file_smoke(tmp_path: Path) -> None:
    mod = _load_finetune_capra_module()

    rec = SupervisionRecord(
        observation={"id": "obs-1"},
        instruction="move safely",
        base_action=[[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        safer_action=[[0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        weight=1.0,
        candidate_stats={"summary": {}},
        metadata={"episode_idx": 0, "step_idx": 0},
    )

    out = tmp_path / "supervision.jsonl"
    write_supervision_jsonl([rec], str(out))

    cfg = mod.FinetuneCapraConfig(
        supervision_path=str(out),
        lambda_capra=1.0,
        lr=1e-3,
        steps=1,
        tiny_smoke=False,
    )
    logs = mod.run_from_supervision_file(cfg)

    assert logs["anchor_loss"] >= 0.0
    assert logs["capra_loss"] >= 0.0
