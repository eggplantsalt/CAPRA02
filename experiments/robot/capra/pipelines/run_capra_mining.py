"""Small-scale CAPRA v1 mining runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from experiments.robot.capra.adapters.env_adapter import EnvAdapter
from experiments.robot.capra.core.mining import MiningConfigV1, mine_episode_v1
from experiments.robot.capra.io.supervision_io import SupervisionRecord, write_supervision_jsonl


def run_capra_mining(
    env_adapter: EnvAdapter,
    episodes: Iterable[Dict[str, Any]],
    output_path: str,
    cfg: MiningConfigV1 | None = None,
) -> List[SupervisionRecord]:
    """Run CAPRA supervision mining over a small iterable of episodes."""
    config = cfg or MiningConfigV1()
    all_records: List[SupervisionRecord] = []

    for episode_idx, ep in enumerate(episodes):
        instruction = str(ep.get("instruction", ""))
        timesteps = ep.get("timesteps", [])
        records = mine_episode_v1(
            env_adapter=env_adapter,
            instruction=instruction,
            timesteps=timesteps,
            cfg=config,
            episode_idx=episode_idx,
        )
        all_records.extend(records)

    write_supervision_jsonl(all_records, output_path)
    return all_records


class _DemoSim:
    def __init__(self) -> None:
        self.state = np.array([0.0, 0.0], dtype=np.float32)

    def get_state(self):
        return self.state.copy()

    def set_state_from_flattened(self, state):
        self.state = np.asarray(state, dtype=np.float32).copy()

    @property
    def data(self):
        class _Data:
            qvel = np.array([0.1, 0.0], dtype=np.float32)
            ncon = 0

        return _Data()

    def forward(self):
        return None


class _DemoEnv:
    def __init__(self) -> None:
        self.sim = _DemoSim()
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


def _build_demo_episodes() -> List[Dict[str, Any]]:
    return [
        {
            "instruction": "move near target while minimizing disturbance",
            "timesteps": [
                {
                    "base_action": np.array([[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    "info_before": {"target_dist": 1.0},
                }
            ],
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small-scale CAPRA v1 supervision mining")
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(Path("tmp") / "capra" / "mined_v1.jsonl"),
        help="Path to output JSONL supervision records",
    )
    args = parser.parse_args()

    env_adapter = EnvAdapter(_DemoEnv())
    episodes = _build_demo_episodes()
    records = run_capra_mining(env_adapter=env_adapter, episodes=episodes, output_path=args.output_path)
    print(f"Wrote {len(records)} records to {args.output_path}")


if __name__ == "__main__":
    main()
