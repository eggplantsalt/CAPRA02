"""Tests for CAPRA v1 metrics and eval runner."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.capra.metrics import EpisodeOutcome, compute_ear, compute_metrics_v1, compute_spir
from experiments.robot.capra.run_capra_eval import run_capra_eval
from experiments.robot.capra.supervision_io import SupervisionRecord, write_supervision_jsonl


def _toy_records() -> list[dict]:
    return [
        {
            "instruction": "a",
            "weight": 1.0,
            "candidate_stats": {
                "local_regret": 0.5,
                "summary": {
                    "base_index": 0,
                    "safer_index": 1,
                    "progress_preserving_indices": [0, 1],
                },
            },
        },
        {
            "instruction": "b",
            "weight": 1.0,
            "candidate_stats": {
                "local_regret": 0.0,
                "summary": {
                    "base_index": 0,
                    "safer_index": 0,
                    "progress_preserving_indices": [0],
                },
            },
        },
        {
            "instruction": "c",
            "weight": 1.0,
            "candidate_stats": {
                "local_regret": 2.0,
                "summary": {
                    "base_index": 0,
                    "safer_index": 1,
                    "progress_preserving_indices": [],
                },
            },
        },
    ]


def test_spir_and_ear_on_toy_records() -> None:
    records = _toy_records()
    spir = compute_spir(records)
    ear = compute_ear(records)

    # Effective steps are first two records only.
    assert spir == 0.5
    assert ear == 0.25


def test_compute_metrics_v1_with_episode_aggregation() -> None:
    records = _toy_records()
    outcomes = [
        EpisodeOutcome(success=True, displacement_total=0.2, non_target_displacement=0.1, severe_event=False),
        EpisodeOutcome(success=False, displacement_total=0.4, non_target_displacement=0.3, severe_event=True),
    ]
    metrics = compute_metrics_v1(records, outcomes)

    assert metrics["SPIR"] == 0.5
    assert metrics["EAR"] == 0.25
    assert metrics["success_rate"] == 0.5
    assert metrics["severe_event_rate"] == 0.5


def test_run_capra_eval_tiny_smoke(tmp_path: Path) -> None:
    recs = [
        SupervisionRecord(
            observation={"id": 1},
            instruction="move safely",
            base_action=[[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            safer_action=[[0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            weight=1.0,
            candidate_stats={
                "local_regret": 0.1,
                "summary": {
                    "base_index": 0,
                    "safer_index": 1,
                    "progress_preserving_indices": [0, 1],
                },
            },
            metadata={},
        )
    ]
    sup_path = tmp_path / "sup.jsonl"
    write_supervision_jsonl(recs, str(sup_path))

    metrics = run_capra_eval(str(sup_path), tiny=True)
    assert "SPIR" in metrics
    assert "EAR" in metrics
    assert "success_rate" in metrics
    assert "displacement_total_mean" in metrics
    assert "non_target_displacement_mean" in metrics
    assert "severe_event_rate" in metrics
