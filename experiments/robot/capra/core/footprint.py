"""CAPRA 局部候选评估中的 footprint v1 计算。

这个文件负责把“动作执行后的副作用”转换为可比较的标量风险。
v1 版本仅包含两部分：
1. displacement_total（位移总量）
2. severe_penalty（严重事件惩罚）
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from experiments.robot.capra.core.types import FootprintV1, StateSignals


# 功能：估计动作造成的位移总量。
# 用法：优先使用对象位置变化；若缺失则退化为末端执行器位移。
def _compute_displacement_total(before: StateSignals, after: StateSignals) -> float:
    keys = sorted(set(before.object_positions.keys()) & set(after.object_positions.keys()))
    if keys:
        total = 0.0
        for key in keys:
            total += float(np.linalg.norm(after.object_positions[key] - before.object_positions[key]))
        return total

    if before.ee_pos is not None and after.ee_pos is not None:
        return float(np.linalg.norm(after.ee_pos - before.ee_pos))

    return 0.0


# 功能：输出可直接比较的 footprint 对象，用于 safer 候选选择。
# 用法：local_evaluator 在每个候选 rollout 完成后调用。
def compute_footprint_v1(
    before: StateSignals,
    after: StateSignals,
    severe_penalties: Dict[str, float] | None = None,
) -> FootprintV1:
    """计算 footprint v1：总位移 + 严重事件惩罚。"""
    penalty_table = severe_penalties or {
        "unrecoverable": 10.0,
        "toppled": 5.0,
        "support_broken": 5.0,
    }

    displacement_total = _compute_displacement_total(before, after)

    severe_flags = {
        "unrecoverable": bool(after.unrecoverable),
        "toppled": bool(after.toppled),
        "support_broken": bool(after.support_broken),
    }

    # 严重事件惩罚按事件逐项累加，方便后续按事件类型拆分调参。
    severe_penalty = 0.0
    for key, enabled in severe_flags.items():
        if enabled:
            severe_penalty += float(penalty_table.get(key, 0.0))

    components = {
        "displacement_total": float(displacement_total),
        "severe_penalty": float(severe_penalty),
    }

    return FootprintV1(
        displacement_total=float(displacement_total),
        severe_event_flags=severe_flags,
        severe_penalty=float(severe_penalty),
        components=components,
    )
