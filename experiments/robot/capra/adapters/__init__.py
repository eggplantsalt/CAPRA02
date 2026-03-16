"""CAPRA 适配器层导出入口。

本层用于统一对外暴露“与具体环境/benchmark 交互”相关能力，
包括环境状态访问、统一状态读取、benchmark 适配检查。
"""

from experiments.robot.capra.adapters.env_adapter import EnvAdapter, get_sim
from experiments.robot.capra.adapters.state_api import read_state_signals
from experiments.robot.capra.adapters.benchmark_adapters import (
    SafeLiberoSmokeConfig,
    get_libero_utility_eval_command,
    smoke_run_custom_split,
    smoke_run_safelibero,
)

__all__ = [
    "EnvAdapter",
    "SafeLiberoSmokeConfig",
    "get_libero_utility_eval_command",
    "get_sim",
    "read_state_signals",
    "smoke_run_custom_split",
    "smoke_run_safelibero",
]
