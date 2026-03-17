"""Tests for CAPRA supervision alignment with real training batches."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.capra.core.training_targets import (
    build_supervision_lookup_index,
    collate_training_targets,
)


def _record(sample_key: str, weight: float, safer_action=None):
    return {
        "sample_key": sample_key,
        "lookup_key": sample_key,
        "instruction": "move safely",
        "safer_action": safer_action if safer_action is not None else [[0.1] * 7],
        "weight": weight,
        "align": {"dataset_name": "libero_spatial", "step_idx": 1},
    }


def test_supervision_matches_real_batch_sample_key() -> None:
    index = build_supervision_lookup_index([_record("k1", 1.0)], strict_key_only=True)
    batch = {
        "sample_keys": ["k1", "k2"],
        "dataset_names": ["libero_spatial", "libero_spatial"],
        "instruction_texts": ["move safely", "move safely"],
    }

    collation = collate_training_targets(batch=batch, supervision_index=index, device=torch.device("cpu"))

    assert collation.num_hits == 1
    assert collation.matched_batch_indices == [0]
    assert collation.matched_sample_keys == ["k1"]


def test_unmatched_samples_do_not_trigger_capra_loss() -> None:
    index = build_supervision_lookup_index([_record("k1", 1.0)], strict_key_only=True)
    batch = {
        "sample_keys": ["k2", "k3"],
        "dataset_names": ["libero_spatial", "libero_spatial"],
        "instruction_texts": ["move safely", "move safely"],
    }

    collation = collate_training_targets(batch=batch, supervision_index=index, device=torch.device("cpu"))

    assert collation.num_hits == 0
    assert collation.weights.numel() == 0


def test_duplicate_key_uses_max_weight_entry() -> None:
    index = build_supervision_lookup_index([
        _record("dup-k", 0.2, safer_action=[[0.2] * 7]),
        _record("dup-k", 1.5, safer_action=[[0.9] * 7]),
    ])
    batch = {
        "sample_keys": ["dup-k"],
        "dataset_names": ["libero_spatial"],
        "instruction_texts": ["move safely"],
    }

    collation = collate_training_targets(
        batch=batch,
        supervision_index=index,
        device=torch.device("cpu"),
        duplicate_strategy="max_weight",
    )

    assert collation.num_hits == 1
    assert float(collation.weights[0].item()) == 1.5
    assert torch.allclose(collation.safer_actions[0], torch.tensor([[0.9] * 7], dtype=torch.float32))


def test_missing_required_keys_are_skipped_safely() -> None:
    records = [
        _record("ok-k", 1.0),
        {"sample_key": "bad-k", "instruction": "move safely", "weight": 1.0},
    ]
    index = build_supervision_lookup_index(records, strict_key_only=True)

    assert "ok-k" in index.by_sample_key
    assert "bad-k" not in index.by_sample_key
