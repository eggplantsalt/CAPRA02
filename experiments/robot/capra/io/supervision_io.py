"""CAPRA 挖掘监督样本的 I/O 工具与数据结构。

这个文件提供两类能力：
1. SupervisionRecord：统一监督样本字段定义。
2. JSONL 读写：把监督样本写盘/读盘，供训练与评测复用。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class SupervisionRecord:
    """一条 CAPRA 挖掘监督记录。"""

    observation: Dict[str, Any]
    instruction: str
    base_action: List[Any]
    safer_action: List[Any]
    weight: float
    candidate_stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 功能：把 dataclass 样本转成普通字典，便于 JSON 序列化。
    # 用法：write_supervision_jsonl 内部调用。
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 功能：逐条写入 JSONL，并自动创建父目录。
# 用法：挖掘流程结束后调用，保存训练监督文件。
def write_supervision_jsonl(records: Iterable[SupervisionRecord], output_path: str) -> int:
    """将监督记录写入 JSONL，并返回写入条数。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=True) + "\n")
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
            rows.append(json.loads(stripped))
    return rows
