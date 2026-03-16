"""Unified state read API for CAPRA (obs-first, sim-backed)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from experiments.robot.capra.adapters.env_adapter import EnvAdapter
from experiments.robot.capra.core.types import StateSignals


def _as_numpy(value: Any) -> Optional[np.ndarray]:
	if value is None:
		return None
	return np.asarray(value)


def _safe_float(value: Any) -> Optional[float]:
	if value is None:
		return None
	try:
		return float(value)
	except (TypeError, ValueError):
		return None


def read_state_signals(
	env_adapter: EnvAdapter,
	obs: Optional[Dict[str, Any]] = None,
	info: Optional[Dict[str, Any]] = None,
) -> StateSignals:
	"""Read state signals from observation first, then enrich with simulator-backed dynamics."""
	signals = StateSignals()

	# OBS-backed signals (preferred for pose/state observations).
	if obs is not None:
		signals.ee_pos = _as_numpy(obs.get("robot0_eef_pos"))
		signals.ee_quat = _as_numpy(obs.get("robot0_eef_quat"))
		signals.joint_pos = _as_numpy(obs.get("robot0_joint_pos"))
		signals.gripper_qpos = _as_numpy(obs.get("robot0_gripper_qpos"))

		object_positions = obs.get("object_positions")
		if isinstance(object_positions, dict):
			signals.object_positions = {k: np.asarray(v) for k, v in object_positions.items()}

		object_velocities = obs.get("object_velocities")
		if isinstance(object_velocities, dict):
			signals.object_velocities = {k: np.asarray(v) for k, v in object_velocities.items()}

		if signals.gripper_qpos is not None and signals.gripper_qpos.size > 0:
			# Keep this normalized proxy simple for v1; tighten semantics in later phases if needed.
			signals.gripper_open_fraction = float(np.clip(np.mean(signals.gripper_qpos), 0.0, 1.0))

		signals.obs_backed["obs_keys"] = sorted(obs.keys())

	# SIM-backed signals (preferred for dynamic/contact cues).
	sim = env_adapter.get_sim()
	sim_data = getattr(sim, "data", None)

	if sim_data is not None:
		qvel = getattr(sim_data, "qvel", None)
		if qvel is not None:
			signals.sim_backed["qvel"] = np.asarray(qvel).copy()

		ncon = getattr(sim_data, "ncon", None)
		if ncon is not None:
			signals.contact_count = int(ncon)

	# Optional info-driven flags, if available from caller/env.
	if info is not None:
		signals.grasp_held = bool(info.get("grasp_held")) if "grasp_held" in info else None
		signals.support_broken = bool(info.get("support_broken")) if "support_broken" in info else None
		signals.toppled = bool(info.get("toppled")) if "toppled" in info else None
		signals.unrecoverable = bool(info.get("unrecoverable")) if "unrecoverable" in info else None

		info_positions = info.get("object_positions")
		if isinstance(info_positions, dict):
			signals.object_positions = {k: np.asarray(v) for k, v in info_positions.items()}

		info_velocities = info.get("object_velocities")
		if isinstance(info_velocities, dict):
			signals.object_velocities = {k: np.asarray(v) for k, v in info_velocities.items()}

		if "object_settled" in info:
			signals.sim_backed["object_settled"] = bool(info.get("object_settled"))

	# Severe flags are just a transport container in this phase.
	signals.severe_event_flags = {
		"toppled": bool(signals.toppled) if signals.toppled is not None else False,
		"support_broken": bool(signals.support_broken) if signals.support_broken is not None else False,
		"unrecoverable": bool(signals.unrecoverable) if signals.unrecoverable is not None else False,
	}

	# Keep place for frequently consumed scalar dynamics.
	signals.sim_backed["mean_abs_qvel"] = _safe_float(
		np.mean(np.abs(signals.sim_backed["qvel"])) if "qvel" in signals.sim_backed else None
	)

	return signals
