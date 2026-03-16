"""Local action proposal generation for CAPRA v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from experiments.robot.capra.types import ActionProposal


@dataclass
class ProposalConfig:
    """Configuration for finite local proposal construction."""

    speed_scales: Iterable[float] = (0.8, 1.2)
    lift_delta: float = 0.03
    retract_delta: float = 0.03
    num_gaussian: int = 0
    gaussian_std: float = 0.01
    max_proposals: int = 8
    lift_axis: int = 2
    retract_axis: int = 0


def _ensure_chunk(base_action_chunk: np.ndarray) -> np.ndarray:
    arr = np.asarray(base_action_chunk, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError("base_action_chunk must have shape [T, D] or [D]")
    return arr


def build_local_proposals(
    base_action_chunk: np.ndarray,
    config: ProposalConfig | None = None,
    include_base: bool = True,
    rng: np.random.Generator | None = None,
) -> List[ActionProposal]:
    """Build a small, configurable local proposal set around a base action chunk."""
    cfg = config or ProposalConfig()
    base = _ensure_chunk(base_action_chunk)
    proposals: List[ActionProposal] = []

    def add(name: str, chunk: np.ndarray, metadata: dict | None = None) -> None:
        if len(proposals) >= cfg.max_proposals:
            return
        proposals.append(ActionProposal(name=name, action_chunk=chunk.astype(np.float32), metadata=metadata or {}))

    if include_base:
        add("base", base.copy(), {"kind": "base"})

    for scale in cfg.speed_scales:
        add(
            f"speed_{scale:.2f}",
            base * float(scale),
            {"kind": "speed_scale", "scale": float(scale)},
        )

    lift_chunk = base.copy()
    if cfg.lift_axis < lift_chunk.shape[1]:
        lift_chunk[:, cfg.lift_axis] += float(cfg.lift_delta)
    add("small_lift", lift_chunk, {"kind": "lift", "delta": float(cfg.lift_delta), "axis": cfg.lift_axis})

    retract_chunk = base.copy()
    if cfg.retract_axis < retract_chunk.shape[1]:
        retract_chunk[:, cfg.retract_axis] -= float(cfg.retract_delta)
    add(
        "small_retract",
        retract_chunk,
        {"kind": "retract", "delta": float(cfg.retract_delta), "axis": cfg.retract_axis},
    )

    if cfg.num_gaussian > 0:
        local_rng = rng or np.random.default_rng(0)
        for idx in range(cfg.num_gaussian):
            if len(proposals) >= cfg.max_proposals:
                break
            noise = local_rng.normal(0.0, cfg.gaussian_std, size=base.shape).astype(np.float32)
            add(
                f"gaussian_{idx}",
                base + noise,
                {"kind": "gaussian", "std": float(cfg.gaussian_std), "seeded": rng is None},
            )

    return proposals
