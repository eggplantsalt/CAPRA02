"""CAPRA environment adapter with sanctioned simulation access."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Optional

import numpy as np

from experiments.robot.capra.core.types import EnvSnapshot


class EnvAdapter:
	"""Thin wrapper that centralizes env/sim access for CAPRA logic."""

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

	def get_sim(self) -> Any:
		"""Return simulator handle via the single sanctioned access path."""
		if self._get_sim_fn is not None:
			sim = self._get_sim_fn(self.env)
			if sim is None:
				raise RuntimeError("Custom get_sim_fn returned None")
			return sim

		# Resolve common wrappers first.
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

	def capture_state(self) -> Any:
		"""Capture a restorable simulation/environment state."""
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

	def restore_state(self, state: Any) -> None:
		"""Restore environment/simulation state from a previous snapshot."""
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

	def snapshot(self, include_obs: bool = False) -> EnvSnapshot:
		"""Capture snapshot for counterfactual round-trip operations."""
		obs = None
		if include_obs:
			get_observation = getattr(self.env, "get_observation", None)
			if callable(get_observation):
				obs = deepcopy(get_observation())

		return EnvSnapshot(state=self.capture_state(), obs=obs)

	def restore(self, snapshot: EnvSnapshot) -> None:
		"""Restore environment from a previously captured snapshot."""
		self.restore_state(snapshot.state)

	@staticmethod
	def _maybe_forward(sim: Any) -> None:
		forward = getattr(sim, "forward", None)
		if callable(forward):
			forward()


def get_sim(env: Any) -> Any:
	"""Convenience function matching sanctioned sim access API."""
	return EnvAdapter(env).get_sim()
