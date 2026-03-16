"""CAPRA 核心算法层导出入口。

该层汇聚 CAPRA 的关键算法组件：
候选生成、progress 判定、footprint 计算、监督挖掘、训练目标构造。
"""

from experiments.robot.capra.core.footprint import compute_footprint_v1
from experiments.robot.capra.core.local_evaluator import evaluate_candidate_v1, evaluate_candidates_v1, summarise_candidate_results
from experiments.robot.capra.core.mining import MiningConfigV1, mine_episode_v1, mine_one_timestep_v1
from experiments.robot.capra.core.proposals import ProposalConfig, build_local_proposals
from experiments.robot.capra.core.task_progress import compute_progress_features_v1, equivalent_progress_gate
from experiments.robot.capra.core.training_targets import CapraTrainingTarget, collate_training_targets, supervision_record_to_target
from experiments.robot.capra.core.types import EnvSnapshot, FootprintV1, ProgressFeaturesV1, StateSignals

__all__ = [
    "CapraTrainingTarget",
    "EnvSnapshot",
    "FootprintV1",
    "MiningConfigV1",
    "ProgressFeaturesV1",
    "ProposalConfig",
    "StateSignals",
    "build_local_proposals",
    "collate_training_targets",
    "compute_footprint_v1",
    "compute_progress_features_v1",
    "equivalent_progress_gate",
    "evaluate_candidate_v1",
    "evaluate_candidates_v1",
    "mine_episode_v1",
    "mine_one_timestep_v1",
    "summarise_candidate_results",
    "supervision_record_to_target",
]
