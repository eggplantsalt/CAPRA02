"""CAPRA 环境适配器：统一并约束 simulator 访问方式。

这个文件的目标是屏蔽不同环境封装差异，向 CAPRA 核心层提供稳定接口：
1. get_sim(): 唯一合法 sim 入口。
2. snapshot()/restore(): 反事实评估需要的状态快照机制。
3. capture_state()/restore_state(): 显式状态抓取与回滚。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Optional

import numpy as np

from experiments.robot.capra.core.types import EnvSnapshot


class EnvAdapter:
	"""轻量包装器：集中管理 CAPRA 逻辑对 env/sim 的访问。"""

	# 功能：初始化适配器并保存可选自定义回调。
	# 用法：传入 env，必要时注入 get_sim/get_state/set_state 自定义函数。
	def __init__(
		self,
		env: Any,
		get_sim_fn: Optional[Callable[[Any], Any]] = None,
		get_state_fn: Optional[Callable[[Any], Any]] = None,
		set_state_fn: Optional[Callable[[Any, Any], None]] = None,
	) -> None:
		self.env = env
		self._get_sim_fn = get_sim_fn
		self._get_state_fn = get_state_fn
		self._set_state_fn = set_state_fn

	# 功能：尽可能从不同包装层中找到可用 sim 句柄。
	# 用法：state_api、local_evaluator 等模块都应通过本函数拿 sim。
	def get_sim(self) -> Any:
		"""通过统一入口返回 simulator 句柄。"""
		if self._get_sim_fn is not None:
			sim = self._get_sim_fn(self.env)
			if sim is None:
				raise RuntimeError("Custom get_sim_fn returned None")
			return sim

		# 先尝试常见包装层，兼容不同环境封装风格。
		candidates = [self.env]
		for attr in ("_env", "env"):
			nested = getattr(self.env, attr, None)
			if nested is not None:
				candidates.append(nested)

		for obj in candidates:
			method = getattr(obj, "get_sim", None)
			if callable(method):
				sim = method()
				if sim is not None:
					return sim

			sim = getattr(obj, "sim", None)
			if sim is not None:
				return sim

		raise RuntimeError("Unable to resolve simulator handle from environment")

	# 功能：读取当前状态并深拷贝，保证后续修改不污染快照。
	# 用法：反事实 rollout 前调用一次并缓存。
	def capture_state(self) -> Any:
		"""抓取可恢复的环境/仿真状态。"""
		if self._get_state_fn is not None:
			return deepcopy(self._get_state_fn(self.env))

		sim = self.get_sim()

		get_state = getattr(sim, "get_state", None)
		if callable(get_state):
			return deepcopy(get_state())

		get_env_state = getattr(self.env, "get_state", None)
		if callable(get_env_state):
			return deepcopy(get_env_state())

		raise RuntimeError("No supported state capture path found")

	# 功能：按兼容路径优先级回滚状态。
	# 用法：每个候选评估前恢复到同一起点。
	def restore_state(self, state: Any) -> None:
		"""将环境/仿真状态恢复到指定快照。"""
		if self._set_state_fn is not None:
			self._set_state_fn(self.env, deepcopy(state))
			return

		sim = self.get_sim()
		state_copy = deepcopy(state)

		set_state_flat = getattr(sim, "set_state_from_flattened", None)
		if callable(set_state_flat):
			set_state_flat(np.asarray(state_copy))
			self._maybe_forward(sim)
			return

		set_state = getattr(sim, "set_state", None)
		if callable(set_state):
			set_state(state_copy)
			self._maybe_forward(sim)
			return

		env_set_state = getattr(self.env, "set_state", None)
		if callable(env_set_state):
			env_set_state(state_copy)
			return

		env_set_init_state = getattr(self.env, "set_init_state", None)
		if callable(env_set_init_state):
			env_set_init_state(state_copy)
			return

		raise RuntimeError("No supported state restore path found")

	# 功能：同时保存 state，可选附带 obs。
	# 用法：evaluate_candidates_v1 进入评估循环前调用。
	def snapshot(self, include_obs: bool = False) -> EnvSnapshot:
		"""抓取用于反事实评估的快照。"""
		obs = None
		if include_obs:
			get_observation = getattr(self.env, "get_observation", None)
			if callable(get_observation):
				obs = deepcopy(get_observation())

		return EnvSnapshot(state=self.capture_state(), obs=obs)

	# 功能：对外隐藏 restore_state 细节。
	# 用法：统一传入 EnvSnapshot 即可恢复。
	def restore(self, snapshot: EnvSnapshot) -> None:
		"""从快照恢复环境状态。"""
		self.restore_state(snapshot.state)

	# 功能：在状态恢复后触发 sim.forward，确保派生量同步。
	# 用法：仅在 sim 对象提供 forward 方法时调用。
	@staticmethod
	def _maybe_forward(sim: Any) -> None:
		forward = getattr(sim, "forward", None)
		if callable(forward):
			forward()


# 功能：提供函数式调用风格，兼容旧调用代码。
# 用法：get_sim(env) 等价于 EnvAdapter(env).get_sim()。
def get_sim(env: Any) -> Any:
	"""便捷函数：与统一 sim 访问接口保持一致。"""
	return EnvAdapter(env).get_sim()
