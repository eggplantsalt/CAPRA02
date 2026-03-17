"""Tests for CAPRA v1 proposal generation (prefix-local, gripper-safe)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.capra.core.proposals import ProposalConfig, build_local_proposals


def _base_chunk(t: int = 4, d: int = 7) -> np.ndarray:
    return np.asarray(
        [
            [0.20, -0.10, 0.05, 0.01, -0.02, 0.03, 0.80],
            [0.18, -0.08, 0.04, 0.01, -0.02, 0.03, 0.75],
            [0.16, -0.06, 0.03, 0.01, -0.02, 0.03, 0.70],
            [0.14, -0.04, 0.02, 0.01, -0.02, 0.03, 0.65],
        ],
        dtype=np.float32,
    )[:t, :d]


def _find_by_name(proposals, name: str):
    for p in proposals:
        if p.name == name:
            return p
    return None


def test_base_proposal_exists_by_default() -> None:
    proposals = build_local_proposals(_base_chunk(), ProposalConfig())
    names = [p.name for p in proposals]
    assert "base" in names


def test_prefix_speed_only_modifies_prefix_steps() -> None:
    base = _base_chunk()
    cfg = ProposalConfig(prefix_steps=2, speed_scales=(0.8,), num_gaussian=0)
    proposals = build_local_proposals(base, cfg)
    cand = _find_by_name(proposals, "speed_prefix_0.80")
    assert cand is not None

    # Prefix should change on mutable dims; tail should remain unchanged.
    assert not np.allclose(cand.action_chunk[:2, :6], base[:2, :6])
    assert np.allclose(cand.action_chunk[2:, :], base[2:, :])


def test_default_templates_keep_gripper_channel_unchanged() -> None:
    base = _base_chunk()
    cfg = ProposalConfig(prefix_steps=3, speed_scales=(0.8, 1.2), num_gaussian=0, gripper_index=6)
    proposals = build_local_proposals(base, cfg)

    for p in proposals:
        assert np.allclose(p.action_chunk[:, 6], base[:, 6])


def test_gaussian_respects_protected_dims() -> None:
    base = _base_chunk()
    cfg = ProposalConfig(
        prefix_steps=2,
        speed_scales=(),
        num_gaussian=2,
        gaussian_std=0.05,
        gripper_index=6,
        protected_dims=(5,),
        max_proposals=6,
    )
    proposals = build_local_proposals(base, cfg, rng=np.random.default_rng(123))
    gaussian = [p for p in proposals if p.name.startswith("gaussian_")]
    assert len(gaussian) == 2

    for p in gaussian:
        # protected dim 5 and gripper dim 6 should remain unchanged
        assert np.allclose(p.action_chunk[:, 5], base[:, 5])
        assert np.allclose(p.action_chunk[:, 6], base[:, 6])


def test_max_proposals_is_enforced() -> None:
    base = _base_chunk()
    cfg = ProposalConfig(
        prefix_steps=3,
        speed_scales=(0.8, 0.9, 1.1, 1.2),
        lateral_delta=0.02,
        num_gaussian=4,
        max_proposals=5,
    )
    proposals = build_local_proposals(base, cfg)
    assert len(proposals) == 5


def test_proposal_schema_compatible_with_downstream() -> None:
    base = _base_chunk()
    cfg = ProposalConfig(prefix_steps=2, speed_scales=(0.8,), num_gaussian=1)
    proposals = build_local_proposals(base, cfg, rng=np.random.default_rng(7))

    assert len(proposals) > 0
    for p in proposals:
        assert isinstance(p.name, str)
        assert p.action_chunk.shape == base.shape
        assert p.action_chunk.dtype == np.float32
        assert isinstance(p.metadata, dict)
