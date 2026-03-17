"""CAPRA 核心层轻量导出入口。

注意：
- 该文件保持轻量，避免在包初始化阶段触发 adapters/core 的循环导入。
- 需要具体算法实现时，请直接从对应子模块导入。
"""

from experiments.robot.capra.core.training_targets import (
    BatchSampleIdentity,
    BatchSupervisionCollation,
    SupervisionLookupEntry,
    SupervisionLookupIndex,
    build_batch_sample_identities,
    build_supervision_lookup_index,
    collate_training_targets,
    load_supervision_lookup_index,
    supervision_record_to_lookup,
)
from experiments.robot.capra.core.types import EnvSnapshot, FootprintV1, ProgressFeaturesV1, StateSignals

__all__ = [
    "BatchSampleIdentity",
    "BatchSupervisionCollation",
    "EnvSnapshot",
    "FootprintV1",
    "ProgressFeaturesV1",
    "StateSignals",
    "SupervisionLookupEntry",
    "SupervisionLookupIndex",
    "build_batch_sample_identities",
    "build_supervision_lookup_index",
    "collate_training_targets",
    "load_supervision_lookup_index",
    "supervision_record_to_lookup",
]
