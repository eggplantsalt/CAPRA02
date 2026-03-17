"""CAPRA supervision 与主训练 batch 对齐工具。

本模块提供两层能力：
1. 构建 supervision 查表索引（基于稳定 sample_key）。
2. 在真实 RLDS/LIBERO 训练 batch 上做命中匹配与聚合，供 CAPRA loss 使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from experiments.robot.capra.io.supervision_io import (
    build_stable_sample_key,
    normalize_instruction,
    read_supervision_jsonl,
)


@dataclass
class SupervisionLookupEntry:
    """单条 supervision 的训练期查表结构。"""

    sample_key: str
    safer_action: torch.Tensor
    weight: float
    metadata: Dict[str, Any]
    candidate_stats: Dict[str, Any]
    provenance: Dict[str, Any]


@dataclass
class SupervisionLookupIndex:
    """训练期 supervision 查表索引。"""

    by_sample_key: Dict[str, List[SupervisionLookupEntry]]
    strict_key_only: bool = True


@dataclass
class BatchSupervisionCollation:
    """一个训练 batch 的 supervision 匹配结果。"""

    matched_batch_indices: List[int]
    matched_sample_keys: List[str]
    safer_actions: torch.Tensor
    weights: torch.Tensor
    num_hits: int


@dataclass
class BatchSampleIdentity:
    """主训练流样本身份字段。"""

    sample_key: str
    dataset_name: str
    instruction_norm: str
    episode_idx: Optional[int]
    step_idx: Optional[int]
    frame_fingerprint: str


def _to_action_tensor(action_like: Any) -> torch.Tensor:
    arr = np.asarray(action_like, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError("Action must be shape [T, D] or [D]")
    return torch.from_numpy(arr)


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def supervision_record_to_lookup(record: Dict[str, Any]) -> SupervisionLookupEntry:
    """把监督记录解析为训练期查表条目。"""
    if "safer_action" not in record:
        raise KeyError("Supervision record must include safer_action")

    sample_key = str(record.get("sample_key", "")).strip()
    if not sample_key:
        # 兼容旧记录：缺失 sample_key 时，尽量基于 align/metadata 重建稳定键。
        align = record.get("align", {}) if isinstance(record.get("align", {}), dict) else {}
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
        sample_key = build_stable_sample_key(
            dataset_name=str(align.get("dataset_name", metadata.get("dataset_name", ""))),
            instruction=str(record.get("instruction", "")),
            episode_idx=_as_int(align.get("episode_idx", metadata.get("episode_idx"))),
            step_idx=_as_int(align.get("step_idx", metadata.get("step_idx"))),
            frame_fingerprint=str(align.get("frame_fingerprint", "")),
            source_uid=str(align.get("source_uid", metadata.get("source_uid", ""))),
        )

    return SupervisionLookupEntry(
        sample_key=sample_key,
        safer_action=_to_action_tensor(record["safer_action"]),
        weight=float(record.get("weight", 1.0)),
        metadata=dict(record.get("metadata", {})) if isinstance(record.get("metadata", {}), dict) else {},
        candidate_stats=(
            dict(record.get("candidate_stats", {})) if isinstance(record.get("candidate_stats", {}), dict) else {}
        ),
        provenance=dict(record.get("provenance", {})) if isinstance(record.get("provenance", {}), dict) else {},
    )


def build_supervision_lookup_index(
    records: Iterable[Dict[str, Any]],
    strict_key_only: bool = True,
) -> SupervisionLookupIndex:
    """构建 supervision 查表索引。"""
    by_sample_key: Dict[str, List[SupervisionLookupEntry]] = {}

    for rec in records:
        try:
            entry = supervision_record_to_lookup(rec)
        except (KeyError, ValueError, TypeError):
            # 缺失关键字段或形状非法的记录直接跳过，避免中断主训练。
            continue
        if not entry.sample_key and strict_key_only:
            continue
        by_sample_key.setdefault(entry.sample_key, []).append(entry)

    return SupervisionLookupIndex(by_sample_key=by_sample_key, strict_key_only=bool(strict_key_only))


def load_supervision_lookup_index(
    supervision_path: str,
    strict_key_only: bool = True,
) -> SupervisionLookupIndex:
    """从 JSONL 读取并构建监督查表索引。"""
    records = read_supervision_jsonl(supervision_path)
    return build_supervision_lookup_index(records, strict_key_only=strict_key_only)


def build_batch_sample_identities(batch: Dict[str, Any]) -> List[BatchSampleIdentity]:
    """从主训练 batch 提取每条样本的稳定身份。"""
    sample_keys = [str(s) for s in batch.get("sample_keys", [])]
    dataset_names = [str(s).strip().lower() for s in batch.get("dataset_names", [""] * len(sample_keys))]
    instruction_texts = [str(s) for s in batch.get("instruction_texts", [""] * len(sample_keys))]

    align_meta_list = batch.get("align_meta", [])
    if not isinstance(align_meta_list, list):
        align_meta_list = []

    n = max(len(sample_keys), len(dataset_names), len(instruction_texts), len(align_meta_list))
    identities: List[BatchSampleIdentity] = []

    for idx in range(n):
        align_meta = align_meta_list[idx] if idx < len(align_meta_list) and isinstance(align_meta_list[idx], dict) else {}
        dataset_name = (
            dataset_names[idx]
            if idx < len(dataset_names) and dataset_names[idx]
            else str(align_meta.get("dataset_name", "")).strip().lower()
        )
        instruction = instruction_texts[idx] if idx < len(instruction_texts) else str(align_meta.get("instruction", ""))
        instruction_norm = normalize_instruction(instruction)
        episode_idx = _as_int(align_meta.get("episode_idx"))
        step_idx = _as_int(align_meta.get("step_idx"))
        frame_fingerprint = str(align_meta.get("frame_fingerprint", ""))
        source_uid = str(align_meta.get("source_uid", ""))

        sample_key = sample_keys[idx] if idx < len(sample_keys) and sample_keys[idx] else ""
        if not sample_key:
            # 训练 batch 未显式给 key 时，回退到同构键构造，尽量保持与 mining 侧一致。
            sample_key = build_stable_sample_key(
                dataset_name=dataset_name,
                instruction=instruction,
                episode_idx=episode_idx,
                step_idx=step_idx,
                frame_fingerprint=frame_fingerprint,
                source_uid=source_uid,
            )

        identities.append(
            BatchSampleIdentity(
                sample_key=sample_key,
                dataset_name=dataset_name,
                instruction_norm=instruction_norm,
                episode_idx=episode_idx,
                step_idx=step_idx,
                frame_fingerprint=frame_fingerprint,
            )
        )

    return identities


def _select_entry(entries: List[SupervisionLookupEntry], strategy: str = "max_weight") -> SupervisionLookupEntry:
    """多条 supervision 命中同一键时的聚合策略。"""
    if not entries:
        raise ValueError("entries must be non-empty")

    if strategy == "first":
        return entries[0]
    if strategy == "mean":
        base = entries[0]
        mean_weight = float(sum(e.weight for e in entries) / len(entries))
        return SupervisionLookupEntry(
            sample_key=base.sample_key,
            safer_action=base.safer_action,
            weight=mean_weight,
            metadata=base.metadata,
            candidate_stats=base.candidate_stats,
            provenance=base.provenance,
        )

    # 默认使用最大权重条目，强调更强监督信号。
    return max(entries, key=lambda e: float(e.weight))


def collate_training_targets(
    batch: Dict[str, Any],
    supervision_index: SupervisionLookupIndex,
    device: torch.device,
    duplicate_strategy: str = "max_weight",
) -> BatchSupervisionCollation:
    """在真实训练 batch 上执行 supervision 命中匹配并输出张量。"""
    identities = build_batch_sample_identities(batch)

    matched_indices: List[int] = []
    matched_keys: List[str] = []
    safer_actions: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []

    for idx, ident in enumerate(identities):
        # 核心对齐机制：严格 sample_key 精确匹配，不做 instruction 模糊匹配。
        entries = supervision_index.by_sample_key.get(ident.sample_key, [])
        if not entries:
            continue

        selected = _select_entry(entries, strategy=duplicate_strategy)
        matched_indices.append(idx)
        matched_keys.append(ident.sample_key)
        safer_actions.append(selected.safer_action.to(device=device))
        weights.append(torch.tensor(float(selected.weight), device=device, dtype=torch.float32))

    if not safer_actions:
        return BatchSupervisionCollation(
            matched_batch_indices=[],
            matched_sample_keys=[],
            safer_actions=torch.empty((0, 0, 0), device=device, dtype=torch.float32),
            weights=torch.empty((0,), device=device, dtype=torch.float32),
            num_hits=0,
        )

    # 对齐不同 T 长度：统一裁剪为 batch 内最短 chunk，保证训练稳定。
    min_t = min(int(t.shape[0]) for t in safer_actions)
    safer_actions = [t[:min_t] for t in safer_actions]

    return BatchSupervisionCollation(
        matched_batch_indices=matched_indices,
        matched_sample_keys=matched_keys,
        safer_actions=torch.stack(safer_actions, dim=0).to(torch.float32),
        weights=torch.stack(weights, dim=0),
        num_hits=len(matched_indices),
    )
