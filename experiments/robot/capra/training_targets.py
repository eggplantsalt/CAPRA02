"""Utilities for converting mined CAPRA supervision to training targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


@dataclass
class CapraTrainingTarget:
    """One CAPRA training target sample."""

    base_action: torch.Tensor
    safer_action: torch.Tensor
    weight: torch.Tensor
    instruction: str
    metadata: Dict[str, Any]


def _to_action_tensor(action_like: Any) -> torch.Tensor:
    arr = np.asarray(action_like, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError("Action must be shape [T, D] or [D]")
    return torch.from_numpy(arr)


def supervision_record_to_target(record: Dict[str, Any]) -> CapraTrainingTarget:
    """Convert one mined supervision record into a training target."""
    if "base_action" not in record or "safer_action" not in record:
        raise KeyError("Supervision record must include base_action and safer_action")

    weight = float(record.get("weight", 0.0))
    instruction = str(record.get("instruction", ""))
    metadata = dict(record.get("metadata", {}))

    return CapraTrainingTarget(
        base_action=_to_action_tensor(record["base_action"]),
        safer_action=_to_action_tensor(record["safer_action"]),
        weight=torch.tensor(weight, dtype=torch.float32),
        instruction=instruction,
        metadata=metadata,
    )


def collate_training_targets(records: Iterable[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate mined supervision records into batched training tensors."""
    targets: List[CapraTrainingTarget] = [supervision_record_to_target(rec) for rec in records]
    if not targets:
        raise ValueError("records must be non-empty")

    base_actions = torch.stack([t.base_action for t in targets], dim=0)
    safer_actions = torch.stack([t.safer_action for t in targets], dim=0)
    weights = torch.stack([t.weight for t in targets], dim=0)

    return {
        "base_actions": base_actions,
        "safer_actions": safer_actions,
        "weights": weights,
    }
