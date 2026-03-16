"""Tests for CAPRA footprint v1 computation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.capra.footprint import compute_footprint_v1
from experiments.robot.capra.types import StateSignals


def test_compute_footprint_v1_with_object_displacement_and_severe_flags() -> None:
    before = StateSignals(
        object_positions={"obj": np.array([0.0, 0.0, 0.0], dtype=np.float32)},
        unrecoverable=False,
        toppled=False,
        support_broken=False,
    )
    after = StateSignals(
        object_positions={"obj": np.array([0.3, 0.4, 0.0], dtype=np.float32)},
        unrecoverable=True,
        toppled=False,
        support_broken=True,
    )

    fp = compute_footprint_v1(before, after)

    assert np.isclose(fp.displacement_total, 0.5)
    assert fp.severe_event_flags["unrecoverable"] is True
    assert fp.severe_event_flags["support_broken"] is True
    assert fp.severe_penalty == 15.0
    assert np.isclose(fp.total, 15.5)
