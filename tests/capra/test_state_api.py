"""Phase 1 tests for CAPRA state API."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.capra.env_adapter import EnvAdapter
from experiments.robot.capra.state_api import read_state_signals


class _FakeData:
    def __init__(self) -> None:
        self.qvel = np.array([0.0, -0.2, 0.3], dtype=np.float64)
        self.ncon = 2


class _FakeSim:
    def __init__(self) -> None:
        self.data = _FakeData()

    def get_state(self):
        return np.array([1.0, 2.0, 3.0], dtype=np.float64)


class _FakeEnv:
    def __init__(self) -> None:
        self.sim = _FakeSim()


def test_read_state_signals_obs_first_and_sim_backed() -> None:
    env = _FakeEnv()
    adapter = EnvAdapter(env)

    obs = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "robot0_joint_pos": np.array([0.5, 0.6, 0.7], dtype=np.float64),
        "robot0_gripper_qpos": np.array([0.25], dtype=np.float64),
    }
    info = {"grasp_held": True, "toppled": False}

    signals = read_state_signals(adapter, obs=obs, info=info)

    assert np.allclose(signals.ee_pos, obs["robot0_eef_pos"])
    assert np.allclose(signals.ee_quat, obs["robot0_eef_quat"])
    assert np.allclose(signals.joint_pos, obs["robot0_joint_pos"])
    assert np.allclose(signals.gripper_qpos, obs["robot0_gripper_qpos"])
    assert signals.gripper_open_fraction == 0.25

    assert signals.contact_count == 2
    assert "qvel" in signals.sim_backed
    assert np.allclose(signals.sim_backed["qvel"], np.array([0.0, -0.2, 0.3], dtype=np.float64))

    assert signals.grasp_held is True
    assert signals.severe_event_flags["toppled"] is False
