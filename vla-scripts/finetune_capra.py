"""CAPRA finetuning overlay entrypoint.

This keeps upstream finetune.py unchanged and adds CAPRA-specific target loading
plus anchor + weighted safer-target regression losses.
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
    """Minimal CAPRA finetune config for overlay training entrypoint."""

    supervision_path: str = ""
    lambda_capra: float = 1.0
    lr: float = 1e-3
    steps: int = 1
    tiny_smoke: bool = False
    use_deepspeed: bool = False
    deepspeed_config_path: str = ""
    deepspeed_zero_stage: int = 2
    deepspeed_grad_clipping: float = 1.0


def compute_anchor_loss(pred_actions: torch.Tensor, base_actions: torch.Tensor) -> torch.Tensor:
    """锚点损失：约束模型不要偏离原始 base policy 太远。"""
    return torch.nn.functional.l1_loss(pred_actions, base_actions)


def compute_weighted_safer_loss(
    pred_actions: torch.Tensor,
    safer_actions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """加权 safer 回归损失：风险差值越大，监督权重越大。"""
    # pred/safer 形状为 [B, T, D]，这里先在动作维度和时序维度求平均，再按样本权重加权。
    abs_err = torch.abs(pred_actions - safer_actions).mean(dim=(-2, -1))
    return (abs_err * weights).mean()


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


def _build_tiny_model(action_shape: torch.Size) -> nn.Module:
    """Build tiny model for smoke execution on CPU."""
    t_steps, action_dim = int(action_shape[-2]), int(action_shape[-1])
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(t_steps * action_dim, t_steps * action_dim),
        nn.Unflatten(-1, (t_steps, action_dim)),
    )
    return model


def _load_deepspeed_config(cfg: FinetuneCapraConfig, batch_size: int) -> Dict[str, Any]:
    """加载 DeepSpeed 配置：优先外部 JSON，否则回退到内置最小配置。"""
    if cfg.deepspeed_config_path:
        config_path = Path(cfg.deepspeed_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"DeepSpeed config not found: {cfg.deepspeed_config_path}")
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
        # DeepSpeed 只在 CUDA 环境下启用；CPU-only 场景直接报错，避免 silent fallback。
        if not torch.cuda.is_available():
            raise RuntimeError("DeepSpeed mode requires CUDA device")
        try:
            import deepspeed
        except ImportError as exc:
            raise ImportError("deepspeed is not installed, please install it first") from exc

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
            # DeepSpeed 接管 backward/step，内部会处理 ZeRO 分片与优化器更新。
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


def run_from_supervision_file(cfg: FinetuneCapraConfig) -> Dict[str, float]:
    """从 mined JSONL 监督数据读取样本并执行 CAPRA 训练回路。"""
    if not cfg.supervision_path:
        raise ValueError("supervision_path must be provided unless --tiny_smoke is used")

    rows = read_supervision_jsonl(cfg.supervision_path)
    if not rows:
        raise ValueError(f"No supervision records found in: {cfg.supervision_path}")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="CAPRA overlay finetuning entrypoint")
    parser.add_argument("--supervision_path", type=str, default="", help="Path to mined supervision JSONL")
    parser.add_argument("--lambda_capra", type=float, default=1.0, help="Weight on CAPRA safer-target loss")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for tiny smoke loop")
    parser.add_argument("--steps", type=int, default=1, help="Number of tiny smoke optimization steps")
    parser.add_argument("--tiny_smoke", action="store_true", help="Run tiny in-memory smoke training")
    parser.add_argument("--use_deepspeed", action="store_true", help="启用 DeepSpeed 训练引擎")
    parser.add_argument(
        "--deepspeed_config_path",
        type=str,
        default="",
        help="Optional DeepSpeed JSON config path",
    )
    parser.add_argument(
        "--deepspeed_zero_stage",
        type=int,
        default=2,
        help="ZeRO stage when no external DeepSpeed config is provided",
    )
    parser.add_argument(
        "--deepspeed_grad_clipping",
        type=float,
        default=1.0,
        help="Gradient clipping when no external DeepSpeed config is provided",
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
