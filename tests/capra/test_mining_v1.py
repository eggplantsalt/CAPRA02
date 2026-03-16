"""Tests for CAPRA v1 supervision mining."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.capra.adapters.env_adapter import EnvAdapter
from experiments.robot.capra.core.mining import MiningConfigV1, mine_episode_v1, mine_one_timestep_v1
from experiments.robot.capra.core.proposals import ProposalConfig
from experiments.robot.capra.io.supervision_io import read_supervision_jsonl, write_supervision_jsonl


class _FakeSim:
    def __init__(self) -> None:
        self._state = np.array([0.0, 0.0], dtype=np.float32)

    @property
    def data(self):
        class _Data:
            qvel = np.array([0.1, 0.0], dtype=np.float32)
            ncon = 0

        return _Data()

    def get_state(self):
        return self._state.copy()

    def set_state_from_flattened(self, state):
        self._state = np.asarray(state, dtype=np.float32).copy()

    def forward(self):
        return None


class _FakeEnv:
    def __init__(self) -> None:
        self.sim = _FakeSim()
        self.target = 1.0

    def get_observation(self):
        eef_x, obj_x = self.sim.get_state()
        return {
            "robot0_eef_pos": np.array([eef_x, 0.0, 0.0], dtype=np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_joint_pos": np.array([eef_x], dtype=np.float32),
            "robot0_gripper_qpos": np.array([0.5], dtype=np.float32),
            "object_positions": {"obj": np.array([obj_x, 0.0, 0.0], dtype=np.float32)},
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        eef_x, obj_x = self.sim.get_state()
        eef_x = eef_x + float(action[0])
        obj_x = obj_x + max(float(action[0]), 0.0) * 2.0
        self.sim.set_state_from_flattened([eef_x, obj_x])
        obs = self.get_observation()
        info = {
            "target_dist": abs(self.target - eef_x),
            "object_positions": {"obj": np.array([obj_x, 0.0, 0.0], dtype=np.float32)},
            "toppled": False,
            "support_broken": False,
            "unrecoverable": False,
        }
        return obs, 0.0, False, info


def _config() -> MiningConfigV1:
    return MiningConfigV1(
        epsilon_p=0.05,
        delta_min=1e-6,
        w_max=3.0,
        short_horizon_steps=1,
        proposal_config=ProposalConfig(speed_scales=(0.9,), num_gaussian=0, max_proposals=5),
    )


def test_mine_one_timestep_v1_returns_record_when_safer_exists() -> None:
    env_adapter = EnvAdapter(_FakeEnv())
    rec = mine_one_timestep_v1(
        env_adapter=env_adapter,
        instruction="move safely",
        base_action_chunk=np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        cfg=_config(),
        episode_idx=3,
        step_idx=1,
        info_before={"target_dist": 1.0},
    )

    assert rec is not None
    assert rec.weight > 0.0
    assert rec.instruction == "move safely"
    assert "summary" in rec.candidate_stats
    assert rec.metadata["episode_idx"] == 3


def test_mine_episode_v1_and_io_roundtrip(tmp_path: Path) -> None:
    env_adapter = EnvAdapter(_FakeEnv())
    records = mine_episode_v1(
        env_adapter=env_adapter,
        instruction="move safely",
        timesteps=[
            {
                "base_action": np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                "info_before": {"target_dist": 1.0},
            }
        ],
        cfg=_config(),
        episode_idx=0,
    )

    assert len(records) >= 1

    out_path = tmp_path / "mined.jsonl"
    count = write_supervision_jsonl(records, str(out_path))
    loaded = read_supervision_jsonl(str(out_path))

    assert count == len(records)
    assert len(loaded) == count
    assert loaded[0]["instruction"] == "move safely"
