"""Smoke tests for CAPRA real overlay training entrypoint."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_finetune_capra_module():
    module_path = REPO_ROOT / "vla-scripts" / "finetune_capra.py"
    spec = importlib.util.spec_from_file_location("finetune_capra", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"finetune_capra module import skipped due to environment dependency issue: {exc}")
    return module


def test_load_openvla_checkpoint_path(monkeypatch) -> None:
    mod = _load_finetune_capra_module()

    called = {"processor": False, "model": False, "path": ""}

    class _FakeProcessor:
        pass

    class _FakeModel:
        def to(self, _device):
            return self

    def _fake_processor_from_pretrained(path, trust_remote_code=True):
        called["processor"] = True
        called["path"] = path
        assert trust_remote_code is True
        return _FakeProcessor()

    def _fake_model_from_pretrained(path, **kwargs):
        called["model"] = True
        called["path"] = path
        assert kwargs["trust_remote_code"] is True
        return _FakeModel()

    monkeypatch.setattr(mod.AutoProcessor, "from_pretrained", _fake_processor_from_pretrained)
    monkeypatch.setattr(mod.AutoModelForVision2Seq, "from_pretrained", _fake_model_from_pretrained)

    processor, model = mod.load_openvla_checkpoint("dummy/openvla", 0)

    assert processor is not None
    assert model is not None
    assert called["processor"] is True
    assert called["model"] is True
    assert called["path"] == "dummy/openvla"


def test_capra_collator_reads_alignment_fields() -> None:
    mod = _load_finetune_capra_module()

    collator = mod.CapraPaddedCollator(model_max_length=32, pad_token_id=0)

    inst1 = {
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "labels": torch.tensor([1, 2, 3], dtype=torch.long),
        "pixel_values": torch.randn(3, 32, 32),
        "actions": np.zeros((mod.NUM_ACTIONS_CHUNK, mod.ACTION_DIM), dtype=np.float32),
        "dataset_name": "libero_spatial",
        "instruction_text": "move safely",
        "sample_key": "k1",
        "align_meta": {"dataset_name": "libero_spatial", "step_idx": 1},
    }
    inst2 = {
        "input_ids": torch.tensor([1, 2], dtype=torch.long),
        "labels": torch.tensor([1, 2], dtype=torch.long),
        "pixel_values": torch.randn(3, 32, 32),
        "actions": np.ones((mod.NUM_ACTIONS_CHUNK, mod.ACTION_DIM), dtype=np.float32),
        "dataset_name": "libero_spatial",
        "instruction_text": "move safely",
        "sample_key": "k2",
        "align_meta": {"dataset_name": "libero_spatial", "step_idx": 2},
    }

    batch = collator([inst1, inst2])

    assert "sample_keys" in batch
    assert "align_meta" in batch
    assert "instruction_texts" in batch
    assert len(batch["sample_keys"]) == 2


def test_task_loss_and_capra_loss_composition(tmp_path: Path) -> None:
    mod = _load_finetune_capra_module()

    sup_path = tmp_path / "sup.jsonl"
    sup_path.write_text(
        '{"sample_key": "stable-k1", "lookup_key": "stable-k1", "instruction": "move safely", "safer_action": [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]], "weight": 1.0, "align": {"dataset_name": "libero_spatial", "step_idx": 1}}\n',
        encoding="utf-8",
    )
    sup_index = mod.load_supervision_lookup_index(str(sup_path), strict_key_only=True)

    expected_task = torch.tensor(0.25, requires_grad=True)

    def _fake_run_forward_pass(**_kwargs):
        return expected_task, {"loss_value": 0.25}

    original = mod.UPSTREAM.run_forward_pass
    mod.UPSTREAM.run_forward_pass = _fake_run_forward_pass
    try:
        task_loss, metrics = mod.compute_task_loss_on_main_batch(
            vla=object(),
            action_head=object(),
            proprio_projector=None,
            batch={"dummy": True},
            action_tokenizer=object(),
            device_id=0,
            use_proprio=False,
            use_film=False,
            num_patches=1,
        )
    finally:
        mod.UPSTREAM.run_forward_pass = original

    assert float(task_loss.detach().cpu().item()) == 0.25
    assert metrics["loss_value"] == 0.25

    predicted = torch.zeros((1, mod.NUM_ACTIONS_CHUNK, mod.ACTION_DIM), dtype=torch.float32)
    batch = {
        "sample_keys": ["stable-k1"],
        "dataset_names": ["libero_spatial"],
        "instruction_texts": ["move safely"],
    }
    collation = mod.collate_training_targets(batch=batch, supervision_index=sup_index, device=torch.device("cpu"))

    total_loss, capra_loss, hits = mod.compose_total_loss(
        task_loss=task_loss,
        predicted_actions=predicted,
        collation=collation,
        lambda_capra=1.0,
    )

    assert hits == 1
    assert float(capra_loss.detach().cpu().item()) > 0.0
    assert float(total_loss.detach().cpu().item()) > float(task_loss.detach().cpu().item())
