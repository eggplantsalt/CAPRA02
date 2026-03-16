"""Tests for CAPRA v1 progress feature computation and gate."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.capra.core.task_progress import compute_progress_features_v1, equivalent_progress_gate
from experiments.robot.capra.core.types import StateSignals


def test_compute_progress_features_v1_and_gate_pass() -> None:
    before = StateSignals(gripper_open_fraction=0.2)
    after_base = StateSignals(gripper_open_fraction=0.2)
    after_cand = StateSignals(gripper_open_fraction=0.2)

    base_feat = compute_progress_features_v1(
        before=before,
        after=after_base,
        done_after=False,
        target_dist_before=1.0,
        target_dist_after=0.6,
    )
    cand_feat = compute_progress_features_v1(
        before=before,
        after=after_cand,
        done_after=False,
        target_dist_before=1.0,
        target_dist_after=0.65,
    )

    assert base_feat.progress_after == -0.6
    assert cand_feat.progress_after == -0.65
    assert equivalent_progress_gate(cand_feat, base_feat, epsilon_p=0.1)


def test_equivalent_progress_gate_rejects_unrecoverable() -> None:
    before = StateSignals()
    after_base = StateSignals(unrecoverable=False)
    after_bad = StateSignals(unrecoverable=True)

    base_feat = compute_progress_features_v1(before, after_base, done_after=False, target_dist_before=1.0, target_dist_after=0.5)
    bad_feat = compute_progress_features_v1(before, after_bad, done_after=False, target_dist_before=1.0, target_dist_after=0.4)

    assert equivalent_progress_gate(bad_feat, base_feat, epsilon_p=0.2) is False
