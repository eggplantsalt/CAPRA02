"""Progress-preserving feature computation and gate for CAPRA v1."""

from __future__ import annotations

from typing import Optional

from experiments.robot.capra.types import ProgressFeaturesV1, StateSignals


def _to_progress_score(target_dist: Optional[float], done: bool) -> Optional[float]:
    if done:
        return 1.0
    if target_dist is None:
        return None
    return -float(target_dist)


def compute_progress_features_v1(
    before: StateSignals,
    after: StateSignals,
    done_after: bool = False,
    target_dist_before: Optional[float] = None,
    target_dist_after: Optional[float] = None,
) -> ProgressFeaturesV1:
    """Compute v1 progress features for progress-preserving filtering only."""
    progress_before = _to_progress_score(target_dist_before, done=False)
    progress_after = _to_progress_score(target_dist_after, done=done_after)

    progress_delta: Optional[float]
    if progress_before is not None and progress_after is not None:
        progress_delta = progress_after - progress_before
    else:
        progress_delta = None

    return ProgressFeaturesV1(
        progress_before=progress_before,
        progress_after=progress_after,
        progress_delta=progress_delta,
        done_after=bool(done_after),
        target_dist_before=target_dist_before,
        target_dist_after=target_dist_after,
        grasp_held=after.grasp_held,
        object_settled=after.sim_backed.get("object_settled") if after.sim_backed else None,
        unrecoverable=after.unrecoverable,
        gripper_open_fraction=after.gripper_open_fraction,
    )


def equivalent_progress_gate(
    candidate: ProgressFeaturesV1,
    base: ProgressFeaturesV1,
    epsilon_p: float = 0.0,
) -> bool:
    """Return whether candidate is progress-preserving relative to base."""
    if candidate.unrecoverable:
        return False

    if base.done_after and not candidate.done_after:
        return False

    if candidate.done_after and not base.done_after:
        return True

    if candidate.progress_after is None or base.progress_after is None:
        return True

    threshold = float(base.progress_after) - float(epsilon_p)
    return float(candidate.progress_after) >= threshold
