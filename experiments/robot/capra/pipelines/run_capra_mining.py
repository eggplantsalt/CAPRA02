"""CAPRA v1 小规模监督挖掘入口。

这个文件提供命令行可调用的 mining pipeline，
当前默认使用 demo 环境进行链路烟雾验证，用于：
1. 验证挖掘流程可跑通。
2. 生成 JSONL 监督样本供后续训练使用。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from experiments.robot.capra.adapters.env_adapter import EnvAdapter
from experiments.robot.capra.core.mining import MiningConfigV1, mine_episode_v1
from experiments.robot.capra.io.supervision_io import SupervisionRecord, write_supervision_jsonl


# 功能：驱动 episode 循环并汇总所有监督记录。
# 用法：可被 CLI 入口调用，也可被测试直接调用。
def run_capra_mining(
    env_adapter: EnvAdapter,
    episodes: Iterable[Dict[str, Any]],
    output_path: str,
    cfg: MiningConfigV1 | None = None,
) -> List[SupervisionRecord]:
    """在一个小规模 episode 集合上执行 CAPRA 监督挖掘。"""
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
    """用于烟雾验证的最小模拟器对象。"""

    # 功能：初始化演示仿真器状态。
    # 用法：_DemoEnv 构造时自动创建。
    def __init__(self) -> None:
        self.state = np.array([0.0, 0.0], dtype=np.float32)

    # 功能：读取当前仿真状态向量。
    # 用法：评估前保存状态与环境观测构造时调用。
    def get_state(self):
        return self.state.copy()

    # 功能：从扁平状态向量恢复仿真状态。
    # 用法：反事实回滚或 step 更新后写回状态。
    def set_state_from_flattened(self, state):
        self.state = np.asarray(state, dtype=np.float32).copy()

    # 功能：提供与真实 sim.data 类似的数据接口。
    # 用法：让 EnvAdapter/state_api 在 demo 环境下可直接复用。
    @property
    def data(self):
        class _Data:
            qvel = np.array([0.1, 0.0], dtype=np.float32)
            ncon = 0

        return _Data()

    # 功能：兼容 MuJoCo sim.forward 接口。
    # 用法：EnvAdapter 恢复状态后可安全调用。
    def forward(self):
        return None


class _DemoEnv:
    """用于烟雾验证的最小环境对象。"""

    # 功能：初始化 demo 环境和目标位置。
    # 用法：main 中构造 EnvAdapter 前调用。
    def __init__(self) -> None:
        self.sim = _DemoSim()
        self.target = 1.0

    # 功能：导出与训练环境同构的观测字典。
    # 用法：挖掘和评估流程读取当前状态时调用。
    def get_observation(self):
        eef_x, obj_x = self.sim.get_state()
        return {
            "robot0_eef_pos": np.array([eef_x, 0.0, 0.0], dtype=np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_joint_pos": np.array([eef_x], dtype=np.float32),
            "robot0_gripper_qpos": np.array([0.5], dtype=np.float32),
            "object_positions": {"obj": np.array([obj_x, 0.0, 0.0], dtype=np.float32)},
        }

    # 功能：执行一步简化动力学并返回 step 四元组。
    # 用法：local evaluator 在 demo 模式下调用。
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


    # 功能：构建最小可运行的演示 episode 列表。
    # 用法：main 默认使用该数据做一轮 smoke mining。
def _build_demo_episodes() -> List[Dict[str, Any]]:
    """构造演示用 episode 数据。"""

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


# 功能：解析命令行参数并执行一轮 demo 挖掘。
# 用法：python -m experiments.robot.capra.pipelines.run_capra_mining --output_path ...
def main() -> None:
    """命令行入口：运行 demo 挖掘并输出 JSONL。"""

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
