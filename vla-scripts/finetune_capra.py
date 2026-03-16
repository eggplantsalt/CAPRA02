"""CAPRA 微调 overlay 入口。

该文件不改动 upstream 的 finetune.py，
仅增加 CAPRA 监督数据读取与 anchor + 加权 safer 回归损失。

当前实现定位：
1. 以 JSONL 监督样本为输入的轻量训练入口。
2. 提供 tiny smoke 回路用于快速验证损失与参数链路。
3. 提供可选 DeepSpeed 分支，支持多卡扩展验证。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.capra.io.supervision_io import read_supervision_jsonl
from experiments.robot.capra.core.training_targets import collate_training_targets


@dataclass
class FinetuneCapraConfig:
    """CAPRA 增量微调配置（最小可用集合）。"""

    supervision_path: str = ""
    lambda_capra: float = 1.0
    lr: float = 1e-3
    steps: int = 1
    tiny_smoke: bool = False
    use_deepspeed: bool = False
    deepspeed_config_path: str = ""
    deepspeed_zero_stage: int = 2
    deepspeed_grad_clipping: float = 1.0


# 功能：约束当前预测不要偏离 base policy 太多。
# 用法：作为总损失中的稳定项长期启用。
def compute_anchor_loss(pred_actions: torch.Tensor, base_actions: torch.Tensor) -> torch.Tensor:
    """锚点损失：约束模型不要偏离原始 base policy 太远。"""
    return torch.nn.functional.l1_loss(pred_actions, base_actions)


# 功能：计算加权 safer 回归误差。
# 用法：每条样本按其风险差值权重参与训练。
def compute_weighted_safer_loss(
    pred_actions: torch.Tensor,
    safer_actions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """加权 safer 回归损失：风险差值越大，监督权重越大。"""
    # 预测动作与 safer 动作形状为 [B, T, D]，先在动作维和时序维求平均，再按样本权重加权。
    abs_err = torch.abs(pred_actions - safer_actions).mean(dim=(-2, -1))
    return (abs_err * weights).mean()


# 功能：组合 CAPRA 训练三项输出（anchor/capra/total）。
# 用法：训练循环中统一调用并记录日志。
def compute_capra_total_loss(
    pred_actions: torch.Tensor,
    base_actions: torch.Tensor,
    safer_actions: torch.Tensor,
    weights: torch.Tensor,
    lambda_capra: float,
) -> Dict[str, torch.Tensor]:
    """组合 CAPRA 总损失：L = L_anchor + lambda * L_safer。"""
    anchor_loss = compute_anchor_loss(pred_actions, base_actions)
    capra_loss = compute_weighted_safer_loss(pred_actions, safer_actions, weights)
    total_loss = anchor_loss + float(lambda_capra) * capra_loss
    return {
        "anchor_loss": anchor_loss,
        "capra_loss": capra_loss,
        "total_loss": total_loss,
    }


# 功能：构建最小线性模型用于链路验证。
# 用法：tiny smoke 模式下快速检查损失与反向传播是否正常。
def _build_tiny_model(action_shape: torch.Size) -> nn.Module:
    """构建 tiny 线性模型，用于快速烟雾验证训练链路。"""
    t_steps, action_dim = int(action_shape[-2]), int(action_shape[-1])
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(t_steps * action_dim, t_steps * action_dim),
        nn.Unflatten(-1, (t_steps, action_dim)),
    )
    return model


# 功能：读取外部配置或构造默认 DeepSpeed 配置。
# 用法：DeepSpeed 分支初始化前调用。
def _load_deepspeed_config(cfg: FinetuneCapraConfig, batch_size: int) -> Dict[str, Any]:
    """加载 DeepSpeed 配置：优先外部 JSON，否则回退到内置最小配置。"""
    if cfg.deepspeed_config_path:
        config_path = Path(cfg.deepspeed_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"未找到 DeepSpeed 配置文件: {cfg.deepspeed_config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "train_batch_size": int(batch_size),
        "gradient_accumulation_steps": 1,
        "gradient_clipping": float(cfg.deepspeed_grad_clipping),
        "zero_optimization": {
            "stage": int(cfg.deepspeed_zero_stage),
        },
        "bf16": {"enabled": bool(torch.cuda.is_available())},
        "fp16": {"enabled": False},
    }


# 功能：执行最小训练回路并返回最后一步损失。
# 用法：既可纯内存运行，也可读取 JSONL 后复用该流程。
def run_tiny_capra_smoke(
    supervision_records: Optional[List[Dict[str, Any]]] = None,
    steps: int = 2,
    lambda_capra: float = 1.0,
    lr: float = 1e-3,
    use_deepspeed: bool = False,
    deepspeed_config_path: str = "",
    deepspeed_zero_stage: int = 2,
    deepspeed_grad_clipping: float = 1.0,
) -> Dict[str, float]:
    """运行一个最小 CAPRA 训练回路，并返回最后一步损失。"""
    if supervision_records is None:
        supervision_records = [
            {
                "instruction": "move safely",
                "base_action": [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                "safer_action": [[0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                "weight": 1.0,
                "metadata": {"episode_idx": 0, "step_idx": 0},
            }
        ]

    batch = collate_training_targets(supervision_records)
    base_actions = batch["base_actions"]
    safer_actions = batch["safer_actions"]
    weights = batch["weights"]

    # 统一设备放置，避免后续切换 DeepSpeed 分支时出现张量设备不一致。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_actions = base_actions.to(device)
    safer_actions = safer_actions.to(device)
    weights = weights.to(device)

    model = _build_tiny_model(base_actions.shape).to(device)
    optimizer: Optional[optim.Optimizer] = None
    engine: Optional[Any] = None

    ds_cfg = FinetuneCapraConfig(
        deepspeed_config_path=deepspeed_config_path,
        deepspeed_zero_stage=deepspeed_zero_stage,
        deepspeed_grad_clipping=deepspeed_grad_clipping,
    )
    if use_deepspeed:
        # DeepSpeed 只在 CUDA 环境下启用；纯 CPU 场景直接报错，避免静默回退。
        if not torch.cuda.is_available():
            raise RuntimeError("启用 DeepSpeed 需要 CUDA 设备")
        try:
            import deepspeed
        except ImportError as exc:
            raise ImportError("当前环境未安装 deepspeed，请先安装后再运行") from exc

        deepspeed_cfg = _load_deepspeed_config(ds_cfg, int(base_actions.shape[0]))
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=deepspeed_cfg,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    last_logs: Dict[str, float] = {}
    for step in range(int(steps)):
        if engine is not None:
            pred_actions = engine(base_actions)
        else:
            pred_actions = model(base_actions)
        losses = compute_capra_total_loss(
            pred_actions=pred_actions,
            base_actions=base_actions,
            safer_actions=safer_actions,
            weights=weights,
            lambda_capra=lambda_capra,
        )

        if engine is not None:
            # DeepSpeed 接管反向传播与参数更新，内部会处理 ZeRO 分片等细节。
            engine.backward(losses["total_loss"])
            engine.step()
        else:
            assert optimizer is not None
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()

        last_logs = {
            "step": float(step),
            "anchor_loss": float(losses["anchor_loss"].detach().cpu().item()),
            "capra_loss": float(losses["capra_loss"].detach().cpu().item()),
            "total_loss": float(losses["total_loss"].detach().cpu().item()),
            "deepspeed": float(1.0 if engine is not None else 0.0),
        }
        print(
            f"step={step} anchor_loss={last_logs['anchor_loss']:.6f} "
            f"capra_loss={last_logs['capra_loss']:.6f} total_loss={last_logs['total_loss']:.6f}"
        )

    return last_logs


# 功能：从监督文件读取样本并触发训练。
# 用法：CLI 非 tiny 模式默认调用本函数。
def run_from_supervision_file(cfg: FinetuneCapraConfig) -> Dict[str, float]:
    """从 mined JSONL 监督数据读取样本并执行 CAPRA 训练回路。"""
    if not cfg.supervision_path:
        raise ValueError("未提供 supervision_path；若只做烟雾验证请使用 --tiny_smoke")

    rows = read_supervision_jsonl(cfg.supervision_path)
    if not rows:
        raise ValueError(f"监督文件中没有可用记录: {cfg.supervision_path}")

    return run_tiny_capra_smoke(
        supervision_records=rows,
        steps=cfg.steps,
        lambda_capra=cfg.lambda_capra,
        lr=cfg.lr,
        use_deepspeed=cfg.use_deepspeed,
        deepspeed_config_path=cfg.deepspeed_config_path,
        deepspeed_zero_stage=cfg.deepspeed_zero_stage,
        deepspeed_grad_clipping=cfg.deepspeed_grad_clipping,
    )


# 功能：提供命令行入口，支持普通模式和 tiny smoke 模式。
# 用法：python vla-scripts/finetune_capra.py --supervision_path ...
def main() -> None:
    parser = argparse.ArgumentParser(description="CAPRA overlay 微调入口")
    parser.add_argument("--supervision_path", type=str, default="", help="挖掘监督 JSONL 路径")
    parser.add_argument("--lambda_capra", type=float, default=1.0, help="CAPRA safer 损失权重")
    parser.add_argument("--lr", type=float, default=1e-3, help="tiny 训练回路学习率")
    parser.add_argument("--steps", type=int, default=1, help="tiny 训练优化步数")
    parser.add_argument("--tiny_smoke", action="store_true", help="执行内存内 tiny 烟雾训练")
    parser.add_argument("--use_deepspeed", action="store_true", help="启用 DeepSpeed 训练引擎")
    parser.add_argument(
        "--deepspeed_config_path",
        type=str,
        default="",
        help="可选 DeepSpeed JSON 配置路径",
    )
    parser.add_argument(
        "--deepspeed_zero_stage",
        type=int,
        default=2,
        help="未提供外部配置时使用的 ZeRO stage",
    )
    parser.add_argument(
        "--deepspeed_grad_clipping",
        type=float,
        default=1.0,
        help="未提供外部配置时使用的梯度裁剪值",
    )
    parser.add_argument("--local_rank", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg = FinetuneCapraConfig(
        supervision_path=args.supervision_path,
        lambda_capra=args.lambda_capra,
        lr=args.lr,
        steps=args.steps,
        tiny_smoke=args.tiny_smoke,
        use_deepspeed=args.use_deepspeed,
        deepspeed_config_path=args.deepspeed_config_path,
        deepspeed_zero_stage=args.deepspeed_zero_stage,
        deepspeed_grad_clipping=args.deepspeed_grad_clipping,
    )

    if cfg.tiny_smoke:
        run_tiny_capra_smoke(
            steps=cfg.steps,
            lambda_capra=cfg.lambda_capra,
            lr=cfg.lr,
            use_deepspeed=cfg.use_deepspeed,
            deepspeed_config_path=cfg.deepspeed_config_path,
            deepspeed_zero_stage=cfg.deepspeed_zero_stage,
            deepspeed_grad_clipping=cfg.deepspeed_grad_clipping,
        )
    else:
        run_from_supervision_file(cfg)


if __name__ == "__main__":
    main()
