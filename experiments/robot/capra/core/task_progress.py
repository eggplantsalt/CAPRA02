"""CAPRA v1 的 progress 特征计算与 gate 判定。

这个文件解决的问题是：
在局部候选动作中，如何快速筛掉“明显让任务变差”的动作，
从而保留 progress-preserving 候选供 footprint 比较。
"""

from __future__ import annotations

from typing import Optional

from experiments.robot.capra.core.types import ProgressFeaturesV1, StateSignals


# 功能：将目标距离和完成标记映射为统一的 progress 分数。
# 用法：done=True 时直接返回 1.0；否则使用负距离作为推进分数。
def _to_progress_score(target_dist: Optional[float], done: bool) -> Optional[float]:
    if done:
        return 1.0
    if target_dist is None:
        return None
    return -float(target_dist)


# 功能：聚合 before/after 关键信号，输出 gate 所需特征。
# 用法：由 local_evaluator 在每个候选评估后调用。
def compute_progress_features_v1(
    before: StateSignals,
    after: StateSignals,
    done_after: bool = False,
    target_dist_before: Optional[float] = None,
    target_dist_after: Optional[float] = None,
) -> ProgressFeaturesV1:
    """计算 v1 progress 特征，仅用于“是否明显变差”的过滤。"""
    progress_before = _to_progress_score(target_dist_before, done=False)
    progress_after = _to_progress_score(target_dist_after, done=done_after)

    progress_delta: Optional[float]
    if progress_before is not None and progress_after is not None:
        progress_delta = progress_after - progress_before
    else:
        progress_delta = None

    return ProgressFeaturesV1(
        progress_before=progress_before,
        progress_after=progress_after,
        progress_delta=progress_delta,
        done_after=bool(done_after),
        target_dist_before=target_dist_before,
        target_dist_after=target_dist_after,
        grasp_held=after.grasp_held,
        object_settled=after.sim_backed.get("object_settled") if after.sim_backed else None,
        unrecoverable=after.unrecoverable,
        gripper_open_fraction=after.gripper_open_fraction,
    )


# 功能：给出布尔判定，决定候选是否进入后续安全比较集合。
# 用法：传入 candidate/base 的 ProgressFeaturesV1 以及 epsilon_p 容忍度。
def equivalent_progress_gate(
    candidate: ProgressFeaturesV1,
    base: ProgressFeaturesV1,
    epsilon_p: float = 0.0,
) -> bool:
    """判断候选动作是否相对 base 保持任务推进能力。"""
    if candidate.unrecoverable:
        return False

    if base.done_after and not candidate.done_after:
        return False

    if candidate.done_after and not base.done_after:
        return True

    if candidate.progress_after is None or base.progress_after is None:
        return True

    threshold = float(base.progress_after) - float(epsilon_p)
    return float(candidate.progress_after) >= threshold
