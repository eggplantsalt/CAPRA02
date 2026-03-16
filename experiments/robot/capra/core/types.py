"""Common CAPRA data structures for CAPRA overlay modules."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EnvSnapshot:
    """Serializable environment snapshot used for round-trip restore tests."""

    state: Any
    obs: Optional[Dict[str, Any]] = None


@dataclass
class StateSignals:
    """Unified state readout combining obs-backed and sim-backed fields."""

    ee_pos: Optional[np.ndarray] = None
    ee_quat: Optional[np.ndarray] = None
    joint_pos: Optional[np.ndarray] = None
    gripper_qpos: Optional[np.ndarray] = None
    gripper_open_fraction: Optional[float] = None

    object_positions: Dict[str, np.ndarray] = field(default_factory=dict)
    object_velocities: Dict[str, np.ndarray] = field(default_factory=dict)

    contact_count: Optional[int] = None
    grasp_held: Optional[bool] = None
    support_broken: Optional[bool] = None
    toppled: Optional[bool] = None
    unrecoverable: Optional[bool] = None

    severe_event_flags: Dict[str, bool] = field(default_factory=dict)

    obs_backed: Dict[str, Any] = field(default_factory=dict)
    sim_backed: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionProposal:
    """A local candidate action chunk around a base action chunk."""

    name: str
    action_chunk: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressFeaturesV1:
    """Progress-preserving features used by CAPRA v1 gate."""

    progress_before: Optional[float] = None
    progress_after: Optional[float] = None
    progress_delta: Optional[float] = None
    done_after: bool = False
    target_dist_before: Optional[float] = None
    target_dist_after: Optional[float] = None
    grasp_held: Optional[bool] = None
    object_settled: Optional[bool] = None
    unrecoverable: Optional[bool] = None
    gripper_open_fraction: Optional[float] = None


@dataclass
class FootprintV1:
    """Footprint v1 container with decomposed components."""

    displacement_total: float
    severe_event_flags: Dict[str, bool]
    severe_penalty: float
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return float(self.displacement_total + self.severe_penalty)


@dataclass
class CandidateEvalV1:
    """Evaluation result for one candidate action chunk."""

    name: str
    action_chunk: np.ndarray
    progress_features: ProgressFeaturesV1
    progress_preserving: bool
    footprint: FootprintV1
    info_after: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateSummaryV1:
    """Summary of candidate evaluations against a base action."""

    base_name: str
    base_index: int
    evaluations: List[CandidateEvalV1]
    progress_preserving_indices: List[int]
    safer_index: Optional[int]
    local_regret: float
