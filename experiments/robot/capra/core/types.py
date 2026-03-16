"""CAPRA overlay 共用数据结构定义。

这个文件用于集中声明 CAPRA 各层（adapters/core/io/evaluation）共享的数据容器。
好处是：
1. 跨模块字段命名一致，减少接口歧义。
2. 便于测试、序列化和类型检查。
3. 新成员可以通过本文件快速理解 CAPRA 关键中间产物。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EnvSnapshot:
    """可序列化环境快照，用于状态回滚和反事实评估。"""

    state: Any
    obs: Optional[Dict[str, Any]] = None


@dataclass
class StateSignals:
    """统一状态信号容器：同时承载 obs 来源与 sim 来源字段。"""

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
    """围绕 base action chunk 生成的一个局部候选动作。"""

    name: str
    action_chunk: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressFeaturesV1:
    """CAPRA v1 中用于 progress-preserving gate 的特征集合。"""

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
    """footprint v1 容器，保存总量与分量。"""

    displacement_total: float
    severe_event_flags: Dict[str, bool]
    severe_penalty: float
    components: Dict[str, float] = field(default_factory=dict)

    # 功能：返回 footprint v1 的总风险值。
    # 用法：下游直接比较 total 大小来选择更安全候选。
    @property
    def total(self) -> float:
        return float(self.displacement_total + self.severe_penalty)


@dataclass
class CandidateEvalV1:
    """单个候选动作评估结果。"""

    name: str
    action_chunk: np.ndarray
    progress_features: ProgressFeaturesV1
    progress_preserving: bool
    footprint: FootprintV1
    info_after: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateSummaryV1:
    """相对 base 动作的候选评估汇总结果。"""

    base_name: str
    base_index: int
    evaluations: List[CandidateEvalV1]
    progress_preserving_indices: List[int]
    safer_index: Optional[int]
    local_regret: float
