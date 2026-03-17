"""End-to-end test: mining production path -> supervision JSONL -> training-side lookup hit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.capra.adapters.env_adapter import EnvAdapter
from experiments.robot.capra.core.mining import MiningConfigV1
from experiments.robot.capra.core.training_targets import collate_training_targets, load_supervision_lookup_index
from experiments.robot.capra.pipelines.run_capra_mining import run_capra_mining
from experiments.robot.capra.io.supervision_io import read_supervision_jsonl


class _ProdLikeSim:
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


class _ProdLikeEnv:
    def __init__(self) -> None:
        self.sim = _ProdLikeSim()
        self.target = 1.0

    def get_observation(self):
        eef_x, obj_x = self.sim.get_state()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[0, 0, 0] = int(max(0.0, min(255.0, eef_x * 100.0)))
        return {
            "robot0_eef_pos": np.array([eef_x, 0.0, 0.0], dtype=np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_joint_pos": np.array([eef_x], dtype=np.float32),
            "robot0_gripper_qpos": np.array([0.5], dtype=np.float32),
            "object_positions": {"obj": np.array([obj_x, 0.0, 0.0], dtype=np.float32)},
            "image_primary": image,
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


def test_mining_to_training_lookup_e2e(tmp_path: Path) -> None:
    env_adapter = EnvAdapter(_ProdLikeEnv())

    episodes = [
        {
            "instruction": "move near target while minimizing disturbance",
            "dataset_name": "libero_spatial",
            "timesteps": [
                {
                    "base_action": np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    "observation_input": env_adapter.env.get_observation(),
                    "info_before": {"target_dist": 1.0},
                    "step_idx": 0,
                }
            ],
        }
    ]

    out_path = tmp_path / "mined_v1.jsonl"
    records = run_capra_mining(
        env_adapter=env_adapter,
        episodes=episodes,
        output_path=str(out_path),
        cfg=MiningConfigV1(short_horizon_steps=1, delta_min=1e-6),
    )

    assert len(records) >= 1
    loaded_rows = read_supervision_jsonl(str(out_path))
    assert len(loaded_rows) >= 1

    sample_key = loaded_rows[0]["sample_key"]
    lookup = load_supervision_lookup_index(str(out_path), strict_key_only=True)

    batch = {
        "sample_keys": [sample_key, "unmatched-k"],
        "dataset_names": ["libero_spatial", "libero_spatial"],
        "instruction_texts": [
            "move near target while minimizing disturbance",
            "move near target while minimizing disturbance",
        ],
    }
    collation = collate_training_targets(batch=batch, supervision_index=lookup, device=torch.device("cpu"))

    assert collation.num_hits == 1
    assert collation.matched_batch_indices == [0]
    assert collation.matched_sample_keys == [sample_key]
