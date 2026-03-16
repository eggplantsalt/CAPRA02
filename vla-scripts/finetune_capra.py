"""CAPRA finetuning overlay entrypoint.

This keeps upstream finetune.py unchanged and adds CAPRA-specific target loading
plus anchor + weighted safer-target regression losses.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.capra.supervision_io import read_supervision_jsonl
from experiments.robot.capra.training_targets import collate_training_targets


@dataclass
class FinetuneCapraConfig:
    """Minimal CAPRA finetune config for overlay training entrypoint."""

    supervision_path: str = ""
    lambda_capra: float = 1.0
    lr: float = 1e-3
    steps: int = 1
    tiny_smoke: bool = False


def compute_anchor_loss(pred_actions: torch.Tensor, base_actions: torch.Tensor) -> torch.Tensor:
    """Anchor loss to keep model close to base policy behavior."""
    return torch.nn.functional.l1_loss(pred_actions, base_actions)


def compute_weighted_safer_loss(
    pred_actions: torch.Tensor,
    safer_actions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted safer-target regression loss for CAPRA."""
    abs_err = torch.abs(pred_actions - safer_actions).mean(dim=(-2, -1))
    return (abs_err * weights).mean()


def compute_capra_total_loss(
    pred_actions: torch.Tensor,
    base_actions: torch.Tensor,
    safer_actions: torch.Tensor,
    weights: torch.Tensor,
    lambda_capra: float,
) -> Dict[str, torch.Tensor]:
    """Compute anchor + weighted CAPRA regression losses."""
    anchor_loss = compute_anchor_loss(pred_actions, base_actions)
    capra_loss = compute_weighted_safer_loss(pred_actions, safer_actions, weights)
    total_loss = anchor_loss + float(lambda_capra) * capra_loss
    return {
        "anchor_loss": anchor_loss,
        "capra_loss": capra_loss,
        "total_loss": total_loss,
    }


def _build_tiny_model(action_shape: torch.Size) -> nn.Module:
    """Build tiny model for smoke execution on CPU."""
    t_steps, action_dim = int(action_shape[-2]), int(action_shape[-1])
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(t_steps * action_dim, t_steps * action_dim),
        nn.Unflatten(-1, (t_steps, action_dim)),
    )
    return model


def run_tiny_capra_smoke(
    supervision_records: Optional[List[Dict[str, Any]]] = None,
    steps: int = 2,
    lambda_capra: float = 1.0,
    lr: float = 1e-3,
) -> Dict[str, float]:
    """Run a tiny CAPRA training loop and return last-step losses."""
    if supervision_records is None:
        supervision_records = [
            {
                "instruction": "move safely",
                "base_action": [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                "safer_action": [[0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                "weight": 1.0,
                "metadata": {"episode_idx": 0, "step_idx": 0},
            }
        ]

    batch = collate_training_targets(supervision_records)
    base_actions = batch["base_actions"]
    safer_actions = batch["safer_actions"]
    weights = batch["weights"]

    model = _build_tiny_model(base_actions.shape)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    last_logs: Dict[str, float] = {}
    for step in range(int(steps)):
        pred_actions = model(base_actions)
        losses = compute_capra_total_loss(
            pred_actions=pred_actions,
            base_actions=base_actions,
            safer_actions=safer_actions,
            weights=weights,
            lambda_capra=lambda_capra,
        )

        optimizer.zero_grad()
        losses["total_loss"].backward()
        optimizer.step()

        last_logs = {
            "step": float(step),
            "anchor_loss": float(losses["anchor_loss"].detach().cpu().item()),
            "capra_loss": float(losses["capra_loss"].detach().cpu().item()),
            "total_loss": float(losses["total_loss"].detach().cpu().item()),
        }
        print(
            f"step={step} anchor_loss={last_logs['anchor_loss']:.6f} "
            f"capra_loss={last_logs['capra_loss']:.6f} total_loss={last_logs['total_loss']:.6f}"
        )

    return last_logs


def run_from_supervision_file(cfg: FinetuneCapraConfig) -> Dict[str, float]:
    """Run CAPRA finetune loop from mined supervision JSONL."""
    if not cfg.supervision_path:
        raise ValueError("supervision_path must be provided unless --tiny_smoke is used")

    rows = read_supervision_jsonl(cfg.supervision_path)
    if not rows:
        raise ValueError(f"No supervision records found in: {cfg.supervision_path}")

    return run_tiny_capra_smoke(
        supervision_records=rows,
        steps=cfg.steps,
        lambda_capra=cfg.lambda_capra,
        lr=cfg.lr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CAPRA overlay finetuning entrypoint")
    parser.add_argument("--supervision_path", type=str, default="", help="Path to mined supervision JSONL")
    parser.add_argument("--lambda_capra", type=float, default=1.0, help="Weight on CAPRA safer-target loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for tiny smoke loop")
    parser.add_argument("--steps", type=int, default=1, help="Number of tiny smoke optimization steps")
    parser.add_argument("--tiny_smoke", action="store_true", help="Run tiny in-memory smoke training")
    args = parser.parse_args()

    cfg = FinetuneCapraConfig(
        supervision_path=args.supervision_path,
        lambda_capra=args.lambda_capra,
        lr=args.lr,
        steps=args.steps,
        tiny_smoke=args.tiny_smoke,
    )

    if cfg.tiny_smoke:
        run_tiny_capra_smoke(steps=cfg.steps, lambda_capra=cfg.lambda_capra, lr=cfg.lr)
    else:
        run_from_supervision_file(cfg)


if __name__ == "__main__":
    main()
