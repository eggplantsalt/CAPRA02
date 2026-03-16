"""Footprint v1 computation for CAPRA local candidate evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np

from experiments.robot.capra.core.types import FootprintV1, StateSignals


def _compute_displacement_total(before: StateSignals, after: StateSignals) -> float:
    keys = sorted(set(before.object_positions.keys()) & set(after.object_positions.keys()))
    if keys:
        total = 0.0
        for key in keys:
            total += float(np.linalg.norm(after.object_positions[key] - before.object_positions[key]))
        return total

    if before.ee_pos is not None and after.ee_pos is not None:
        return float(np.linalg.norm(after.ee_pos - before.ee_pos))

    return 0.0


def compute_footprint_v1(
    before: StateSignals,
    after: StateSignals,
    severe_penalties: Dict[str, float] | None = None,
) -> FootprintV1:
    """Compute footprint v1: displacement_total + severe_event penalties."""
    penalty_table = severe_penalties or {
        "unrecoverable": 10.0,
        "toppled": 5.0,
        "support_broken": 5.0,
    }

    displacement_total = _compute_displacement_total(before, after)

    severe_flags = {
        "unrecoverable": bool(after.unrecoverable),
        "toppled": bool(after.toppled),
        "support_broken": bool(after.support_broken),
    }

    severe_penalty = 0.0
    for key, enabled in severe_flags.items():
        if enabled:
            severe_penalty += float(penalty_table.get(key, 0.0))

    components = {
        "displacement_total": float(displacement_total),
        "severe_penalty": float(severe_penalty),
    }

    return FootprintV1(
        displacement_total=float(displacement_total),
        severe_event_flags=severe_flags,
        severe_penalty=float(severe_penalty),
        components=components,
    )
