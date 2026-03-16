"""Phase 1 tests for CAPRA env adapter infrastructure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.capra.env_adapter import EnvAdapter


class _FakeSim:
    def __init__(self) -> None:
        self._state = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        self.forward_calls = 0

    def get_state(self):
        return self._state.copy()

    def set_state_from_flattened(self, state):
        self._state = np.asarray(state, dtype=np.float64).copy()

    def forward(self):
        self.forward_calls += 1


class _FakeEnv:
    def __init__(self) -> None:
        self.sim = _FakeSim()

    def get_observation(self):
        return {"robot0_eef_pos": np.array([1.0, 2.0, 3.0], dtype=np.float64)}


def test_get_sim_returns_sanctioned_sim() -> None:
    env = _FakeEnv()
    adapter = EnvAdapter(env)

    assert adapter.get_sim() is env.sim


def test_snapshot_restore_round_trip() -> None:
    env = _FakeEnv()
    adapter = EnvAdapter(env)

    snapshot = adapter.snapshot(include_obs=True)

    env.sim.set_state_from_flattened([9.0, 9.0, 9.0])
    assert not np.allclose(env.sim.get_state(), snapshot.state)

    adapter.restore(snapshot)

    assert np.allclose(env.sim.get_state(), snapshot.state)
    assert env.sim.forward_calls >= 1
