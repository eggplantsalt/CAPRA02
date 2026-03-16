"""CAPRA 命令行流水线入口层。"""

from experiments.robot.capra.pipelines.run_capra_eval import run_capra_eval
from experiments.robot.capra.pipelines.run_capra_mining import run_capra_mining

__all__ = ["run_capra_eval", "run_capra_mining"]
