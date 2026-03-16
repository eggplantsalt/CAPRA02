"""Phase 0 import smoke tests for CAPRA scaffold."""

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_capra_package_importable() -> None:
    module = importlib.import_module("experiments.robot.capra")
    assert module is not None


def test_capra_placeholder_modules_importable() -> None:
    module_names = [
           "experiments.robot.capra.adapters.env_adapter",
           "experiments.robot.capra.adapters.state_api",
           "experiments.robot.capra.core.local_evaluator",
           "experiments.robot.capra.core.mining",
    ]
    for name in module_names:
        module = importlib.import_module(name)
        assert module is not None
