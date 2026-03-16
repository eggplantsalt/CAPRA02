"""CAPRA overlay package."""

from experiments.robot.capra.adapters.env_adapter import EnvAdapter, get_sim
from experiments.robot.capra.adapters.state_api import read_state_signals
from experiments.robot.capra.core.footprint import compute_footprint_v1
from experiments.robot.capra.core.local_evaluator import evaluate_candidate_v1, evaluate_candidates_v1, summarise_candidate_results
from experiments.robot.capra.core.proposals import ProposalConfig, build_local_proposals
from experiments.robot.capra.core.task_progress import compute_progress_features_v1, equivalent_progress_gate
from experiments.robot.capra.core.types import EnvSnapshot, FootprintV1, ProgressFeaturesV1, StateSignals

__all__ = [
	"EnvAdapter",
	"EnvSnapshot",
	"FootprintV1",
	"ProgressFeaturesV1",
	"ProposalConfig",
	"StateSignals",
	"build_local_proposals",
	"compute_footprint_v1",
	"compute_progress_features_v1",
	"equivalent_progress_gate",
	"evaluate_candidate_v1",
	"evaluate_candidates_v1",
	"get_sim",
	"read_state_signals",
	"summarise_candidate_results",
]
