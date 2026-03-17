"""CAPRA v1 局部候选动作生成模块（前缀级主路径）。

这个文件的职责是：
1. 接收 base policy 在当前时刻给出的动作块（action chunk）。
2. 以“前缀级局部改写”为主，构造有限、稳定、可解释的候选集合。
3. 默认保护 gripper 与受保护维度，避免隐式缩放/噪声污染动作语义。

典型使用流程：
- 上游先给出 base_action_chunk。
- 调用 build_local_proposals(...) 得到候选列表。
- 下游 local_evaluator 对候选逐个做短视界评估。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

import numpy as np

from experiments.robot.capra.core.types import ActionProposal


@dataclass
class ProposalConfig:
    """有限候选集生成配置（默认以前缀级模板为主）。"""

    include_base: bool = True
    prefix_steps: int = 2
    speed_scales: Iterable[float] = (0.8, 1.2)
    lift_delta: float = 0.03
    retract_delta: float = 0.03
    lateral_delta: float = 0.0

    num_gaussian: int = 0
    gaussian_std: float = 0.01
    gaussian_prefix_only: bool = True

    max_proposals: int = 8

    axis_x: int = 0
    axis_y: int = 1
    axis_z: int = 2

    gripper_index: Optional[int] = 6
    protected_dims: Iterable[int] = field(default_factory=tuple)

    # whole-chunk 模板默认关闭，仅作为次级/调试补充。
    enable_whole_chunk_templates: bool = False
    whole_chunk_speed_scales: Iterable[float] = field(default_factory=tuple)


# 功能：将输入动作统一转换为 [T, D] 形状。
# 用法：支持输入 [D] 或 [T, D]，其余形状会抛出异常。
def _ensure_chunk(base_action_chunk: np.ndarray) -> np.ndarray:
    arr = np.asarray(base_action_chunk, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError("base_action_chunk must have shape [T, D] or [D]")
    return arr


def _is_valid_dim(dim: Optional[int], action_dim: int) -> bool:
    return dim is not None and 0 <= int(dim) < int(action_dim)


def _resolve_prefix_steps(chunk_len: int, prefix_steps: int) -> int:
    if chunk_len <= 0:
        return 0
    return max(1, min(int(prefix_steps), int(chunk_len)))


def _collect_protected_dims(cfg: ProposalConfig, action_dim: int) -> Set[int]:
    protected: Set[int] = set()
    if _is_valid_dim(cfg.gripper_index, action_dim):
        protected.add(int(cfg.gripper_index))
    for dim in cfg.protected_dims:
        if _is_valid_dim(dim, action_dim):
            protected.add(int(dim))
    return protected


def _mutable_dims(action_dim: int, protected_dims: Set[int]) -> List[int]:
    return [dim for dim in range(action_dim) if dim not in protected_dims]


# 功能：生成“前缀局部模板为主、whole-chunk/gaussian 为次级”的候选集。
# 用法：传入 base_action_chunk；可通过 ProposalConfig 调整 prefix_steps、include_base、保护维与次级模板开关。
def build_local_proposals(
    base_action_chunk: np.ndarray,
    config: ProposalConfig | None = None,
    include_base: Optional[bool] = None,
    rng: np.random.Generator | None = None,
) -> List[ActionProposal]:
    """围绕 base action chunk 构造前缀级局部候选集。"""
    cfg = config or ProposalConfig()
    base = _ensure_chunk(base_action_chunk)
    chunk_len, action_dim = int(base.shape[0]), int(base.shape[1])
    prefix_steps = _resolve_prefix_steps(chunk_len, cfg.prefix_steps)

    protected_dims = _collect_protected_dims(cfg, action_dim)
    mutable = _mutable_dims(action_dim, protected_dims)

    proposals: List[ActionProposal] = []

    # 功能：统一执行候选截断与结构化封装。
    # 用法：本函数为内部助手，仅在 build_local_proposals 内调用。
    def add(name: str, chunk: np.ndarray, metadata: dict | None = None) -> None:
        # 统一在这里做数量上限控制，避免上层忘记截断候选数量。
        if len(proposals) >= cfg.max_proposals:
            return
        proposals.append(ActionProposal(name=name, action_chunk=chunk.astype(np.float32), metadata=metadata or {}))

    def add_prefix_template(name: str, chunk: np.ndarray, metadata: dict | None = None) -> None:
        md = {
            "scope": "prefix",
            "prefix_steps": int(prefix_steps),
            "protected_dims": sorted(list(protected_dims)),
        }
        if metadata:
            md.update(metadata)
        add(name, chunk, md)

    # include_base=True 时保留 identity 候选，作为局部反事实比较基线。
    include_base_value = cfg.include_base if include_base is None else bool(include_base)
    if include_base_value:
        add(
            "base",
            base.copy(),
            {
                "kind": "identity",
                "scope": "identity",
                "protected_dims": sorted(list(protected_dims)),
            },
        )

    # 主模板 1：前缀速度缩放（仅非保护维度）。
    for scale in cfg.speed_scales:
        scaled = base.copy()
        if mutable:
            scaled[:prefix_steps, mutable] *= float(scale)
        add_prefix_template(
            f"speed_prefix_{scale:.2f}",
            scaled,
            {"kind": "speed_scale_prefix", "scale": float(scale)},
        )

    # 主模板 2：前缀抬升（z 轴）。
    if _is_valid_dim(cfg.axis_z, action_dim) and int(cfg.axis_z) not in protected_dims:
        lifted = base.copy()
        lifted[:prefix_steps, int(cfg.axis_z)] += float(cfg.lift_delta)
        add_prefix_template(
            "lift_prefix",
            lifted,
            {
                "kind": "lift_prefix",
                "axis": int(cfg.axis_z),
                "delta": float(cfg.lift_delta),
            },
        )

    # 主模板 3：前缀回撤（x 轴）。
    if _is_valid_dim(cfg.axis_x, action_dim) and int(cfg.axis_x) not in protected_dims:
        retracted = base.copy()
        retracted[:prefix_steps, int(cfg.axis_x)] -= float(cfg.retract_delta)
        add_prefix_template(
            "retract_prefix",
            retracted,
            {
                "kind": "retract_prefix",
                "axis": int(cfg.axis_x),
                "delta": float(cfg.retract_delta),
            },
        )

    # 主模板 4：lift_then_go（前缀前半抬升，后半恢复推进）。
    lift_then_go = base.copy()
    half = max(1, prefix_steps // 2)
    if _is_valid_dim(cfg.axis_z, action_dim) and int(cfg.axis_z) not in protected_dims:
        lift_then_go[:half, int(cfg.axis_z)] += float(cfg.lift_delta)
    if _is_valid_dim(cfg.axis_x, action_dim) and int(cfg.axis_x) not in protected_dims:
        lift_then_go[half:prefix_steps, int(cfg.axis_x)] += float(cfg.retract_delta) * 0.5
    if _is_valid_dim(cfg.axis_y, action_dim) and int(cfg.axis_y) not in protected_dims:
        lift_then_go[:half, int(cfg.axis_y)] *= 0.5
    add_prefix_template("lift_then_go", lift_then_go, {"kind": "lift_then_go", "split": int(half)})

    # 可选模板：前缀横向偏移。
    if float(cfg.lateral_delta) != 0.0 and _is_valid_dim(cfg.axis_y, action_dim) and int(cfg.axis_y) not in protected_dims:
        left = base.copy()
        right = base.copy()
        left[:prefix_steps, int(cfg.axis_y)] += float(cfg.lateral_delta)
        right[:prefix_steps, int(cfg.axis_y)] -= float(cfg.lateral_delta)
        add_prefix_template(
            "lateral_left_prefix",
            left,
            {
                "kind": "lateral_offset_prefix",
                "axis": int(cfg.axis_y),
                "delta": float(cfg.lateral_delta),
                "direction": "left",
            },
        )
        add_prefix_template(
            "lateral_right_prefix",
            right,
            {
                "kind": "lateral_offset_prefix",
                "axis": int(cfg.axis_y),
                "delta": float(cfg.lateral_delta),
                "direction": "right",
            },
        )

    # 次级模板：whole-chunk 变换（默认关闭，避免与前缀主路径混淆）。
    if cfg.enable_whole_chunk_templates:
        for scale in cfg.whole_chunk_speed_scales:
            whole_scaled = base.copy()
            if mutable:
                whole_scaled[:, mutable] *= float(scale)
            add(
                f"speed_whole_{scale:.2f}",
                whole_scaled,
                {
                    "kind": "speed_scale_whole_chunk",
                    "scale": float(scale),
                    "scope": "whole_chunk_secondary",
                    "protected_dims": sorted(list(protected_dims)),
                },
            )

    # 次级随机补充：仅在显式开启 num_gaussian 时加入。
    if cfg.num_gaussian > 0:
        # 高斯扰动候选：随机补充项（非主模板），默认不改保护维度。
        local_rng = rng or np.random.default_rng(0)
        row_end = prefix_steps if cfg.gaussian_prefix_only else chunk_len
        for idx in range(cfg.num_gaussian):
            if len(proposals) >= cfg.max_proposals:
                break
            gaussian = base.copy()
            if mutable:
                noise = local_rng.normal(0.0, cfg.gaussian_std, size=(row_end, len(mutable))).astype(np.float32)
                gaussian[:row_end, mutable] += noise
            add(
                f"gaussian_{idx}",
                gaussian,
                {
                    "kind": "gaussian_secondary",
                    "std": float(cfg.gaussian_std),
                    "scope": "prefix" if cfg.gaussian_prefix_only else "whole_chunk_secondary",
                    "seeded": rng is None,
                    "protected_dims": sorted(list(protected_dims)),
                },
            )

    return proposals
