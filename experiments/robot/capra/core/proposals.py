"""CAPRA v1 局部候选动作生成模块。

这个文件的职责是：
1. 接收 base policy 在当前时刻给出的动作块（action chunk）。
2. 围绕该 base 动作构造一个“有限、稳定、可配置”的局部候选集合。
3. 保证候选数量受控，便于后续反事实评估在可接受开销下运行。

典型使用流程：
- 上游先给出 base_action_chunk。
- 调用 build_local_proposals(...) 得到候选列表。
- 下游 local_evaluator 对候选逐个做短视界评估。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from experiments.robot.capra.core.types import ActionProposal


@dataclass
class ProposalConfig:
    """有限候选集生成配置。"""

    speed_scales: Iterable[float] = (0.8, 1.2)
    lift_delta: float = 0.03
    retract_delta: float = 0.03
    num_gaussian: int = 0
    gaussian_std: float = 0.01
    max_proposals: int = 8
    lift_axis: int = 2
    retract_axis: int = 0


# 功能：将输入动作统一转换为 [T, D] 形状。
# 用法：支持输入 [D] 或 [T, D]，其余形状会抛出异常。
def _ensure_chunk(base_action_chunk: np.ndarray) -> np.ndarray:
    arr = np.asarray(base_action_chunk, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError("base_action_chunk must have shape [T, D] or [D]")
    return arr


# 功能：生成包含 base、速度缩放、微抬升、微回撤、可选高斯扰动的候选集。
# 用法：传入 base_action_chunk，可选传入 ProposalConfig 自定义候选策略。
def build_local_proposals(
    base_action_chunk: np.ndarray,
    config: ProposalConfig | None = None,
    include_base: bool = True,
    rng: np.random.Generator | None = None,
) -> List[ActionProposal]:
    """围绕 base action chunk 构造一个小而稳定、可配置的局部候选集。"""
    cfg = config or ProposalConfig()
    base = _ensure_chunk(base_action_chunk)
    proposals: List[ActionProposal] = []

    # 功能：统一执行候选截断与结构化封装。
    # 用法：本函数为内部助手，仅在 build_local_proposals 内调用。
    def add(name: str, chunk: np.ndarray, metadata: dict | None = None) -> None:
        # 统一在这里做数量上限控制，避免上层忘记截断候选数量。
        if len(proposals) >= cfg.max_proposals:
            return
        proposals.append(ActionProposal(name=name, action_chunk=chunk.astype(np.float32), metadata=metadata or {}))

    if include_base:
        add("base", base.copy(), {"kind": "base"})

    for scale in cfg.speed_scales:
        # 速度缩放候选：保持动作方向不变，只改变幅度。
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
        # 高斯扰动候选：用于补充少量随机邻域探索。
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
