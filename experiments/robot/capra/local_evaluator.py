"""Local candidate evaluator for CAPRA v1."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from experiments.robot.capra.env_adapter import EnvAdapter
from experiments.robot.capra.footprint import compute_footprint_v1
from experiments.robot.capra.task_progress import compute_progress_features_v1, equivalent_progress_gate
from experiments.robot.capra.types import ActionProposal, CandidateEvalV1, CandidateSummaryV1
from experiments.robot.capra.state_api import read_state_signals


def _to_chunk(action_chunk: np.ndarray) -> np.ndarray:
	arr = np.asarray(action_chunk, dtype=np.float32)
	if arr.ndim == 1:
		return arr[None, :]
	if arr.ndim != 2:
		raise ValueError("action_chunk must have shape [T, D] or [D]")
	return arr


def _step_env(env: Any, action: np.ndarray):
	return env.step(action.tolist())


def evaluate_candidate_v1(
	env_adapter: EnvAdapter,
	proposal: ActionProposal,
	snapshot,
	before_obs: Optional[Dict[str, Any]],
	before_info: Optional[Dict[str, Any]],
	base_progress_features,
	short_horizon_steps: int,
	epsilon_p: float,
) -> CandidateEvalV1:
	"""Evaluate one candidate from the same snapshot with short counterfactual rollout."""
	env_adapter.restore(snapshot)
	env = env_adapter.env

	obs_before = before_obs
	if obs_before is None:
		get_observation = getattr(env, "get_observation", None)
		if callable(get_observation):
			obs_before = get_observation()

	before_signals = read_state_signals(env_adapter, obs=obs_before, info=before_info)

	actions = _to_chunk(proposal.action_chunk)
	horizon = min(max(short_horizon_steps, 1), actions.shape[0])

	obs_after = obs_before
	info_after: Dict[str, Any] = {}
	done_after = False
	for idx in range(horizon):
		obs_after, _reward, done_after, step_info = _step_env(env, actions[idx])
		info_after = step_info or {}
		if done_after:
			break

	after_signals = read_state_signals(env_adapter, obs=obs_after, info=info_after)

	progress_features = compute_progress_features_v1(
		before=before_signals,
		after=after_signals,
		done_after=bool(done_after),
		target_dist_before=before_info.get("target_dist") if before_info else None,
		target_dist_after=info_after.get("target_dist") if info_after else None,
	)

	progress_preserving = equivalent_progress_gate(
		candidate=progress_features,
		base=base_progress_features,
		epsilon_p=epsilon_p,
	)

	footprint = compute_footprint_v1(before=before_signals, after=after_signals)

	return CandidateEvalV1(
		name=proposal.name,
		action_chunk=actions,
		progress_features=progress_features,
		progress_preserving=progress_preserving,
		footprint=footprint,
		info_after=info_after,
	)


def evaluate_candidates_v1(
	env_adapter: EnvAdapter,
	proposals: List[ActionProposal],
	short_horizon_steps: int = 1,
	epsilon_p: float = 0.0,
	base_name: str = "base",
	obs_before: Optional[Dict[str, Any]] = None,
	info_before: Optional[Dict[str, Any]] = None,
) -> CandidateSummaryV1:
	"""Evaluate base action and local proposals using snapshot-based counterfactual rollouts."""
	if not proposals:
		raise ValueError("proposals must be non-empty")

	snapshot = env_adapter.snapshot(include_obs=False)

	if obs_before is None:
		get_observation = getattr(env_adapter.env, "get_observation", None)
		if callable(get_observation):
			obs_before = get_observation()

	base_idx = next((i for i, p in enumerate(proposals) if p.name == base_name), 0)
	base_eval = evaluate_candidate_v1(
		env_adapter=env_adapter,
		proposal=proposals[base_idx],
		snapshot=snapshot,
		before_obs=obs_before,
		before_info=info_before,
		base_progress_features=compute_progress_features_v1(
			before=read_state_signals(env_adapter, obs=obs_before, info=info_before),
			after=read_state_signals(env_adapter, obs=obs_before, info=info_before),
			done_after=False,
			target_dist_before=info_before.get("target_dist") if info_before else None,
			target_dist_after=info_before.get("target_dist") if info_before else None,
		),
		short_horizon_steps=short_horizon_steps,
		epsilon_p=epsilon_p,
	)

	base_progress_features = base_eval.progress_features
	evaluations: List[CandidateEvalV1] = []
	for proposal in proposals:
		evaluations.append(
			evaluate_candidate_v1(
				env_adapter=env_adapter,
				proposal=proposal,
				snapshot=snapshot,
				before_obs=obs_before,
				before_info=info_before,
				base_progress_features=base_progress_features,
				short_horizon_steps=short_horizon_steps,
				epsilon_p=epsilon_p,
			)
		)

	preserving_indices = [i for i, ev in enumerate(evaluations) if ev.progress_preserving]

	safer_index: Optional[int] = None
	local_regret = 0.0
	if preserving_indices:
		safer_index = min(preserving_indices, key=lambda idx: evaluations[idx].footprint.total)
		base_total = evaluations[base_idx].footprint.total
		safer_total = evaluations[safer_index].footprint.total
		local_regret = float(max(base_total - safer_total, 0.0))

	env_adapter.restore(snapshot)
	return CandidateSummaryV1(
		base_name=proposals[base_idx].name,
		base_index=base_idx,
		evaluations=evaluations,
		progress_preserving_indices=preserving_indices,
		safer_index=safer_index,
		local_regret=local_regret,
	)


def summarise_candidate_results(summary: CandidateSummaryV1) -> Dict[str, Any]:
	"""Convert candidate summary to a compact serializable dictionary."""
	candidates = []
	for idx, ev in enumerate(summary.evaluations):
		candidates.append(
			{
				"index": idx,
				"name": ev.name,
				"progress_preserving": ev.progress_preserving,
				"footprint_total": ev.footprint.total,
				"displacement_total": ev.footprint.displacement_total,
				"severe_penalty": ev.footprint.severe_penalty,
			}
		)

	return {
		"base_index": summary.base_index,
		"safer_index": summary.safer_index,
		"local_regret": summary.local_regret,
		"progress_preserving_indices": summary.progress_preserving_indices,
		"candidates": candidates,
	}
