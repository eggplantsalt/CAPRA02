"""Tests for CAPRA local candidate evaluator v1."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.capra.env_adapter import EnvAdapter
from experiments.robot.capra.local_evaluator import evaluate_candidates_v1, summarise_candidate_results
from experiments.robot.capra.proposals import ProposalConfig, build_local_proposals


class _FakeSim:
    def __init__(self) -> None:
        self._state = np.array([0.0, 0.0], dtype=np.float32)  # [eef_x, obj_x]

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
        done = False
        reward = 0.0
        return obs, reward, done, info


def test_local_evaluator_v1_finds_safer_candidate_when_available() -> None:
    env = _FakeEnv()
    adapter = EnvAdapter(env)

    base_action = np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    proposals = build_local_proposals(
        base_action,
        ProposalConfig(
            speed_scales=(0.9,),
            retract_delta=0.03,
            num_gaussian=0,
            max_proposals=5,
        ),
        include_base=True,
    )

    summary = evaluate_candidates_v1(
        env_adapter=adapter,
        proposals=proposals,
        short_horizon_steps=1,
        epsilon_p=0.05,
        base_name="base",
        info_before={"target_dist": 1.0},
    )

    compact = summarise_candidate_results(summary)

    assert compact["base_index"] == 0
    assert compact["safer_index"] is not None
    assert compact["local_regret"] >= 0.0
    assert len(compact["candidates"]) == len(proposals)

    if compact["safer_index"] != 0:
        base = compact["candidates"][0]["footprint_total"]
        safer = compact["candidates"][compact["safer_index"]]["footprint_total"]
        assert safer <= base
