"""CAPRA 挖掘监督样本的 I/O 工具与数据结构。

这个文件提供两类能力：
1. SupervisionRecord：统一监督样本字段定义。
2. JSONL 读写：把监督样本写盘/读盘，供训练与评测复用。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


SCHEMA_VERSION = "capra.supervision.v2"


def normalize_instruction(text: str) -> str:
    """统一 instruction 归一化，避免空格和大小写造成键不一致。"""
    return " ".join(str(text).strip().lower().split())


def compute_observation_fingerprint(observation: Optional[Dict[str, Any]]) -> str:
    """对观测构建稳定指纹，用于跨流程样本对齐。"""
    if not observation:
        return ""

    h = hashlib.sha1()
    for key in sorted(observation.keys()):
        value = observation[key]
        h.update(key.encode("utf-8"))
        if isinstance(value, np.ndarray):
            h.update(value.tobytes())
        elif isinstance(value, dict):
            for nk in sorted(value.keys()):
                nv = value[nk]
                h.update(str(nk).encode("utf-8"))
                if isinstance(nv, np.ndarray):
                    h.update(nv.tobytes())
                else:
                    h.update(str(nv).encode("utf-8"))
        else:
            h.update(str(value).encode("utf-8"))
    return h.hexdigest()


def build_stable_sample_key(
    dataset_name: str,
    instruction: str,
    step_idx: Optional[int],
    episode_idx: Optional[int] = None,
    frame_fingerprint: str = "",
    source_uid: str = "",
) -> str:
    """构建 supervision 与主训练样本共用的稳定键。"""
    normalized_instruction = normalize_instruction(instruction)
    parts = [
        f"dataset={str(dataset_name).strip().lower()}",
        f"instruction={normalized_instruction}",
        f"episode={'' if episode_idx is None else int(episode_idx)}",
        f"step={'' if step_idx is None else int(step_idx)}",
        f"frame={frame_fingerprint}",
        f"uid={source_uid}",
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class SupervisionRecord:
    """一条 CAPRA 挖掘监督记录。"""

    sample_key: str
    lookup_key: str
    observation: Dict[str, Any]
    instruction: str
    base_action: List[Any]
    safer_action: List[Any]
    weight: float
    align: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    candidate_stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    # 功能：把 dataclass 样本转成普通字典，便于 JSON 序列化。
    # 用法：write_supervision_jsonl 内部调用。
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def upgrade_supervision_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """把旧版 supervision 记录升级到 v2 schema（向后兼容）。"""
    record = dict(raw)
    metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
    align = record.get("align", {}) if isinstance(record.get("align", {}), dict) else {}

    dataset_name = str(
        align.get("dataset_name", metadata.get("dataset_name", record.get("dataset_name", "")))
    ).strip().lower()
    instruction = str(record.get("instruction", ""))
    episode_idx = _as_int(align.get("episode_idx", metadata.get("episode_idx")))
    step_idx = _as_int(align.get("step_idx", metadata.get("step_idx")))
    frame_fingerprint = str(align.get("frame_fingerprint", "")).strip()
    source_uid = str(align.get("source_uid", metadata.get("source_uid", ""))).strip()

    if not frame_fingerprint:
        frame_fingerprint = compute_observation_fingerprint(record.get("observation", {}))

    sample_key = str(record.get("sample_key", "")).strip()
    if not sample_key:
        sample_key = build_stable_sample_key(
            dataset_name=dataset_name,
            instruction=instruction,
            episode_idx=episode_idx,
            step_idx=step_idx,
            frame_fingerprint=frame_fingerprint,
            source_uid=source_uid,
        )

    record["sample_key"] = sample_key
    record["lookup_key"] = str(record.get("lookup_key", sample_key)).strip() or sample_key
    record["align"] = {
        "dataset_name": dataset_name,
        "instruction_norm": normalize_instruction(instruction),
        "episode_idx": episode_idx,
        "step_idx": step_idx,
        "frame_fingerprint": frame_fingerprint,
        "source_uid": source_uid,
    }
    record["provenance"] = record.get("provenance", {}) if isinstance(record.get("provenance", {}), dict) else {}
    record["candidate_stats"] = (
        record.get("candidate_stats", {}) if isinstance(record.get("candidate_stats", {}), dict) else {}
    )
    record["metadata"] = metadata
    record["schema_version"] = str(record.get("schema_version", SCHEMA_VERSION))

    return record


# 功能：逐条写入 JSONL，并自动创建父目录。
# 用法：挖掘流程结束后调用，保存训练监督文件。
def write_supervision_jsonl(records: Iterable[SupervisionRecord], output_path: str) -> int:
    """将监督记录写入 JSONL，并返回写入条数。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            upgraded = upgrade_supervision_record(rec.to_dict())
            f.write(json.dumps(upgraded, ensure_ascii=True) + "\n")
            count += 1
    return count


# 功能：读取并解析 JSONL 为字典列表。
# 用法：训练入口 finetune_capra.py 调用本函数加载监督数据。
def read_supervision_jsonl(input_path: str) -> List[Dict[str, Any]]:
    """读取先前写入的 JSONL 监督文件。"""
    path = Path(input_path)
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(upgrade_supervision_record(json.loads(stripped)))
    return rows
