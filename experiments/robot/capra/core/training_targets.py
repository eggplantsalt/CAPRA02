"""将 CAPRA 挖掘监督样本转换为训练目标张量的工具函数。

这个文件处于“挖掘数据 -> 训练器输入”之间，负责：
1. 对单条监督记录做字段校验与张量化。
2. 将多条监督记录拼成批量张量。
3. 统一 batch 维/时序维/动作维形状约定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


@dataclass
class CapraTrainingTarget:
    """单条 CAPRA 训练样本。"""

    base_action: torch.Tensor
    safer_action: torch.Tensor
    weight: torch.Tensor
    instruction: str
    metadata: Dict[str, Any]


# 功能：将动作数组转换为 torch.Tensor，并规范为 [T, D]。
# 用法：单条监督记录转样本时内部调用。
def _to_action_tensor(action_like: Any) -> torch.Tensor:
    arr = np.asarray(action_like, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError("Action must be shape [T, D] or [D]")
    return torch.from_numpy(arr)


# 功能：完成字段读取、默认值处理与张量构造。
# 用法：collate_training_targets 会逐条调用该函数。
def supervision_record_to_target(record: Dict[str, Any]) -> CapraTrainingTarget:
    """将一条挖掘监督记录转换为训练样本对象。"""
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


# 功能：将 list[record] 组装为训练器可直接使用的 batched tensor 字典。
# 用法：finetune_capra.py 在读取 JSONL 后直接调用。
def collate_training_targets(records: Iterable[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """将多条监督记录拼接为批量训练张量。"""
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
