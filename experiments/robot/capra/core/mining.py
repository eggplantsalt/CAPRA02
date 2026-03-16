"""CAPRA v1 supervision mining from local candidate evaluation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from experiments.robot.capra.adapters.env_adapter import EnvAdapter
from experiments.robot.capra.core.local_evaluator import evaluate_candidates_v1, summarise_candidate_results
from experiments.robot.capra.core.proposals import ProposalConfig, build_local_proposals
from experiments.robot.capra.io.supervision_io import SupervisionRecord


@dataclass
class MiningConfigV1:
    """CAPRA v1 挖掘配置。"""

    epsilon_p: float = 0.05
    delta_min: float = 1e-6
    w_max: float = 5.0
    short_horizon_steps: int = 1
    proposal_config: ProposalConfig = field(default_factory=ProposalConfig)


def _to_chunk(action_chunk: np.ndarray) -> np.ndarray:
    arr = np.asarray(action_chunk, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError("action_chunk must have shape [T, D] or [D]")
    return arr


def _to_serializable_obs(observation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if observation is None:
        return {}

    serial: Dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            serial[key] = value.tolist()
        elif isinstance(value, dict):
            nested = {}
            for nk, nv in value.items():
                nested[nk] = nv.tolist() if isinstance(nv, np.ndarray) else nv
            serial[key] = nested
        else:
            serial[key] = value
    return serial


def mine_one_timestep_v1(
    env_adapter: EnvAdapter,
    instruction: str,
    base_action_chunk: np.ndarray,
    cfg: Optional[MiningConfigV1] = None,
    episode_idx: int = 0,
    step_idx: int = 0,
    observation_input: Optional[Dict[str, Any]] = None,
    info_before: Optional[Dict[str, Any]] = None,
) -> Optional[SupervisionRecord]:
    """在单个时间步挖掘监督样本：仅当存在更安全候选且超过阈值时产出。"""
    config = cfg or MiningConfigV1()
    base_chunk = _to_chunk(base_action_chunk)

    proposals = build_local_proposals(base_chunk, config=config.proposal_config, include_base=True)
    summary_obj = evaluate_candidates_v1(
        env_adapter=env_adapter,
        proposals=proposals,
        short_horizon_steps=config.short_horizon_steps,
        epsilon_p=config.epsilon_p,
        base_name="base",
        obs_before=observation_input,
        info_before=info_before,
    )
    summary = summarise_candidate_results(summary_obj)

    safer_index = summary_obj.safer_index
    if safer_index is None:
        return None

    if safer_index == summary_obj.base_index:
        return None

    base_eval = summary_obj.evaluations[summary_obj.base_index]
    safer_eval = summary_obj.evaluations[safer_index]
    delta = float(base_eval.footprint.total - safer_eval.footprint.total)
    # Delta_t 不足阈值时跳过，避免引入噪声监督。
    if delta <= float(config.delta_min):
        return None

    # v1 权重采用 clip(Delta_t, 0, w_max)。
    weight = float(np.clip(delta, 0.0, config.w_max))

    return SupervisionRecord(
        observation=_to_serializable_obs(observation_input),
        instruction=instruction,
        base_action=np.asarray(base_eval.action_chunk).tolist(),
        safer_action=np.asarray(safer_eval.action_chunk).tolist(),
        weight=weight,
        candidate_stats={
            "summary": summary,
            "base_footprint": float(base_eval.footprint.total),
            "safer_footprint": float(safer_eval.footprint.total),
            "local_regret": float(delta),
        },
        metadata={
            "episode_idx": int(episode_idx),
            "step_idx": int(step_idx),
            "base_index": int(summary_obj.base_index),
            "safer_index": int(safer_index),
            "progress_preserving_indices": list(summary_obj.progress_preserving_indices),
        },
    )


def mine_episode_v1(
    env_adapter: EnvAdapter,
    instruction: str,
    timesteps: Iterable[Dict[str, Any]],
    cfg: Optional[MiningConfigV1] = None,
    episode_idx: int = 0,
) -> List[SupervisionRecord]:
    """遍历一个 episode 的多个时间步，收集所有有效 supervision。"""
    config = cfg or MiningConfigV1()
    records: List[SupervisionRecord] = []

    for step_idx, step in enumerate(timesteps):
        rec = mine_one_timestep_v1(
            env_adapter=env_adapter,
            instruction=instruction,
            base_action_chunk=np.asarray(step["base_action"], dtype=np.float32),
            cfg=config,
            episode_idx=episode_idx,
            step_idx=step_idx,
            observation_input=step.get("observation_input"),
            info_before=step.get("info_before"),
        )
        if rec is not None:
            records.append(rec)

    return records
