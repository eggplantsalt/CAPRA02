"""CAPRA 统一状态读取接口（obs-first, sim-backed）。

这个文件负责把环境中来源不同的状态字段汇总成统一对象 StateSignals：
1. 观测侧字段：末端位姿、关节、夹爪、对象位置等。
2. 仿真侧字段：速度、接触数等动态信息。
3. info 字段：抓取状态、翻倒、不可恢复等任务语义标记。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from experiments.robot.capra.adapters.env_adapter import EnvAdapter
from experiments.robot.capra.core.types import StateSignals


# 功能：将输入转换为 numpy 数组，空值保持 None。
# 用法：统一处理 obs/info 中可能存在的不同数据类型。
def _as_numpy(value: Any) -> Optional[np.ndarray]:
	if value is None:
		return None
	return np.asarray(value)


# 功能：安全地将输入转为 float，失败时返回 None。
# 用法：用于构造可选标量统计，避免类型异常中断流程。
def _safe_float(value: Any) -> Optional[float]:
	if value is None:
		return None
	try:
		return float(value)
	except (TypeError, ValueError):
		return None


# 功能：构建 CAPRA 下游统一消费的状态信号对象。
# 用法：local_evaluator 在候选动作执行前后各调用一次。
def read_state_signals(
	env_adapter: EnvAdapter,
	obs: Optional[Dict[str, Any]] = None,
	info: Optional[Dict[str, Any]] = None,
) -> StateSignals:
	"""先读观测信号，再补充仿真动态信号，生成统一 StateSignals。"""
	signals = StateSignals()

	# 观测侧信号优先：位置/姿态等直接观测字段可读性更强、稳定性更好。
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
			# v1 先使用简单归一化开合度代理，后续阶段再细化语义。
			signals.gripper_open_fraction = float(np.clip(np.mean(signals.gripper_qpos), 0.0, 1.0))

		signals.obs_backed["obs_keys"] = sorted(obs.keys())

	# 仿真侧信号补充：速度、接触数量等动态线索优先从 sim 中读取。
	sim = env_adapter.get_sim()
	sim_data = getattr(sim, "data", None)

	if sim_data is not None:
		qvel = getattr(sim_data, "qvel", None)
		if qvel is not None:
			signals.sim_backed["qvel"] = np.asarray(qvel).copy()

		ncon = getattr(sim_data, "ncon", None)
		if ncon is not None:
			signals.contact_count = int(ncon)

	# 若 info 提供了任务相关标记，则同步写入统一信号容器。
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

	# v1 阶段 severe flags 仅作传输容器，不做更复杂推理。
	signals.severe_event_flags = {
		"toppled": bool(signals.toppled) if signals.toppled is not None else False,
		"support_broken": bool(signals.support_broken) if signals.support_broken is not None else False,
		"unrecoverable": bool(signals.unrecoverable) if signals.unrecoverable is not None else False,
	}

	# 缓存常用标量动态特征，便于下游 footprint/progress 直接消费。
	signals.sim_backed["mean_abs_qvel"] = _safe_float(
		np.mean(np.abs(signals.sim_backed["qvel"])) if "qvel" in signals.sim_backed else None
	)

	return signals
