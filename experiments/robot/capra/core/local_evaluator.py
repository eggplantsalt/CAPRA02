"""CAPRA v1 局部候选评估器。

这个文件是 CAPRA 挖掘链路的核心执行层，负责：
1. 在同一环境快照上逐个执行候选动作（反事实评估）。
2. 计算每个候选的 progress 特征与 footprint。
3. 找出 progress-preserving 集合内 footprint 最小的 safer 候选。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from experiments.robot.capra.adapters.env_adapter import EnvAdapter
from experiments.robot.capra.adapters.state_api import read_state_signals
from experiments.robot.capra.core.footprint import compute_footprint_v1
from experiments.robot.capra.core.task_progress import compute_progress_features_v1, equivalent_progress_gate
from experiments.robot.capra.core.types import ActionProposal, CandidateEvalV1, CandidateSummaryV1


# 功能：规范化动作形状为 [T, D]。
# 用法：所有候选动作进入评估前都要先调用。
def _to_chunk(action_chunk: np.ndarray) -> np.ndarray:
	arr = np.asarray(action_chunk, dtype=np.float32)
	if arr.ndim == 1:
		return arr[None, :]
	if arr.ndim != 2:
		raise ValueError("action_chunk must have shape [T, D] or [D]")
	return arr


# 功能：统一环境 step 调用，兼容 numpy 输入。
# 用法：传入单步动作向量，内部转换为 list 后执行。
def _step_env(env: Any, action: np.ndarray):
	return env.step(action.tolist())


# 功能：输出单候选评估结果（progress + footprint + info_after）。
# 用法：由 evaluate_candidates_v1 在循环中反复调用。
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
	"""从同一个快照出发，执行短视界反事实 rollout 并评估单个候选动作。"""
	env_adapter.restore(snapshot)
	env = env_adapter.env

	obs_before = before_obs
	if obs_before is None:
		get_observation = getattr(env, "get_observation", None)
		if callable(get_observation):
			obs_before = get_observation()

	before_signals = read_state_signals(env_adapter, obs=obs_before, info=before_info)

	# 统一动作形状为 [T, D]，便于后续按时间步执行。
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

	# 推进保持 gate 只回答“是否明显变差”，不做完整任务等价判定。
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


# 功能：完成“评估全部候选 -> 选择 safer -> 计算局部 regret”的完整流程。
# 用法：上游 mining 调用该函数以获取单时间步候选汇总。
def evaluate_candidates_v1(
	env_adapter: EnvAdapter,
	proposals: List[ActionProposal],
	short_horizon_steps: int = 1,
	epsilon_p: float = 0.0,
	base_name: str = "base",
	obs_before: Optional[Dict[str, Any]] = None,
	info_before: Optional[Dict[str, Any]] = None,
) -> CandidateSummaryV1:
	"""在同一初始快照上评估 base 与局部提案，输出候选汇总。"""
	if not proposals:
		raise ValueError("proposals must be non-empty")

	snapshot = env_adapter.snapshot(include_obs=False)

	if obs_before is None:
		get_observation = getattr(env_adapter.env, "get_observation", None)
		if callable(get_observation):
			obs_before = get_observation()

	# 优先按名称定位 base；若未找到，默认第 0 个候选。
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
		# 在 progress-preserving 集合内选择 footprint 最小的候选。
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


# 功能：把 dataclass 结构整理成 JSON 友好字典。
# 用法：挖掘记录写盘前调用，便于日志与离线分析。
def summarise_candidate_results(summary: CandidateSummaryV1) -> Dict[str, Any]:
	"""将候选汇总对象转成紧凑且可序列化的字典。"""
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
