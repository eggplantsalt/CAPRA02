"""CAPRA 命令行流水线导出入口。

该层对外提供 mining/eval 两个命令行流程函数。
"""

from experiments.robot.capra.pipelines.run_capra_eval import run_capra_eval
from experiments.robot.capra.pipelines.run_capra_mining import run_capra_mining

__all__ = ["run_capra_eval", "run_capra_mining"]
