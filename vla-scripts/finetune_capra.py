"""CAPRA 真实训练 overlay 入口（挂接 OpenVLA-OFT 上游骨架）。

设计目标：
1. 复用上游 `vla-scripts/finetune.py` 的真实模型/数据/训练基础设施。
2. 以 RLDS 主数据流的任务损失作为 `task_loss`（真实 anchor 来源）。
3. 以 CAPRA supervision JSONL 仅作为附加 sparse supervision，计算 `capra_loss`。
4. 默认路径即真实训练路径，不再以内存 toy/tiny loop 作为主实现。

说明：
- smoke/tiny 仅用于 tests 中的排障验证，不是训练主路径。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import draccus
except ModuleNotFoundError:
    # 测试环境中若未安装 draccus，则提供最小 no-op 装饰器回退。
    _fallback = types.ModuleType("draccus")

    def _wrap():
        def _decorator(fn):
            return fn

        return _decorator

    _fallback.wrap = _wrap
    sys.modules["draccus"] = _fallback
    import draccus
import torch
import torch.distributed as dist
import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

try:
    from peft import LoraConfig, get_peft_model
except ModuleNotFoundError:
    @dataclass
    class LoraConfig:  # type: ignore[no-redef]
        r: int
        lora_alpha: int
        lora_dropout: float
        target_modules: str
        init_lora_weights: str

    def get_peft_model(model, _config):  # type: ignore[no-redef]
        raise RuntimeError("未安装 peft，无法执行真实 LoRA 训练；请先安装 peft")

try:
    from accelerate import PartialState
except ModuleNotFoundError:
    class PartialState:  # type: ignore[override]
        def __init__(self):
            self.local_process_index = 0
            self.num_processes = 1
            self.is_main_process = True

    _accelerate_fallback = types.ModuleType("accelerate")
    _accelerate_fallback.PartialState = PartialState
    sys.modules["accelerate"] = _accelerate_fallback

try:
    import wandb
except ModuleNotFoundError:
    class _WandbFallback:
        @staticmethod
        def init(*args, **kwargs):
            return None

        @staticmethod
        def log(*args, **kwargs):
            return None

    wandb = _WandbFallback()  # type: ignore[assignment]

from experiments.robot.capra.core.training_targets import (
    BatchSupervisionCollation,
    SupervisionLookupIndex,
    collate_training_targets,
    load_supervision_lookup_index,
)
from experiments.robot.capra.io.supervision_io import (
    build_stable_sample_key,
    compute_observation_fingerprint,
    normalize_instruction,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE, NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics


def _load_upstream_finetune_module() -> Any:
    """动态加载上游 finetune.py，复用其训练骨架工具函数。"""
    upstream_path = Path(__file__).resolve().with_name("finetune.py")
    spec = importlib.util.spec_from_file_location("openvla_upstream_finetune", upstream_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载上游训练入口: {upstream_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UPSTREAM = _load_upstream_finetune_module()


@dataclass
class FinetuneCapraConfig(UPSTREAM.FinetuneConfig):
    """CAPRA overlay 配置：继承上游配置并追加 CAPRA 字段。"""

    supervision_path: str = ""
    lambda_capra: float = 1.0

    # DeepSpeed 仅作为真实训练循环可选后端（非 tiny 分支）。
    use_deepspeed: bool = False
    deepspeed_config_path: str = ""
    deepspeed_zero_stage: int = 2
    deepspeed_grad_clipping: float = 1.0


class CapraRLDSBatchTransform(RLDSBatchTransform):
    """在上游 RLDSBatchTransform 基础上保留原始语言指令文本。"""

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        out = super().__call__(rlds_batch)
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        dataset_raw = rlds_batch.get("dataset_name", "")
        dataset_name = dataset_raw.decode().lower() if isinstance(dataset_raw, (bytes, bytearray)) else str(dataset_raw).lower()

        timestep_raw = None
        if isinstance(rlds_batch.get("observation", {}), dict):
            timestep_raw = rlds_batch["observation"].get("timestep")
        step_idx = int(timestep_raw[0]) if timestep_raw is not None else None

        obs_for_fp = {}
        if isinstance(rlds_batch.get("observation", {}), dict) and "image_primary" in rlds_batch["observation"]:
            obs_for_fp["image_primary"] = rlds_batch["observation"]["image_primary"][0]
        frame_fingerprint = compute_observation_fingerprint(obs_for_fp)
        sample_key = build_stable_sample_key(
            dataset_name=dataset_name,
            instruction=lang,
            episode_idx=None,
            step_idx=step_idx,
            frame_fingerprint=frame_fingerprint,
            source_uid="",
        )

        out["instruction_text"] = lang
        out["sample_key"] = sample_key
        out["align_meta"] = {
            "dataset_name": dataset_name,
            "instruction": normalize_instruction(lang),
            "episode_idx": None,
            "step_idx": step_idx,
            "frame_fingerprint": frame_fingerprint,
            "source_uid": "",
        }
        return out


class CapraPaddedCollator(PaddedCollatorForActionPrediction):
    """扩展上游 collator，保留 instruction 文本用于 supervision 命中。"""

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        out = super().__call__(instances)
        if "instruction_text" in instances[0]:
            out["instruction_texts"] = [str(instance["instruction_text"]) for instance in instances]
        if "sample_key" in instances[0]:
            out["sample_keys"] = [str(instance["sample_key"]) for instance in instances]
        if "align_meta" in instances[0]:
            out["align_meta"] = [dict(instance["align_meta"]) for instance in instances]
        return out


def _module_of(model_or_wrapper: Any) -> Any:
    return model_or_wrapper.module if hasattr(model_or_wrapper, "module") else model_or_wrapper


def load_openvla_checkpoint(vla_path: str, device_id: int) -> Tuple[Any, Any]:
    """加载真实 OpenVLA processor 与 checkpoint 模型。"""
    processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_id)
    return processor, model


def _predict_actions_l1(
    vla: Any,
    action_head: Any,
    proprio_projector: Any,
    batch: Dict[str, Any],
    device_id: int,
    use_proprio: bool,
    use_film: bool,
    num_patches: int,
) -> torch.Tensor:
    """在真实模型上预测连续动作块（L1 regression 路径）。"""
    head = _module_of(action_head)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"] if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=use_film,
        )

    last_hidden_states = output.hidden_states[-1]
    text_hidden_states = last_hidden_states[:, num_patches:-1]

    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = UPSTREAM.get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = UPSTREAM.get_next_actions_mask(ground_truth_token_ids)

    batch_size = batch["input_ids"].shape[0]
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )
    return head.predict_action(actions_hidden_states)


def compute_capra_loss_from_batch(
    predicted_actions: torch.Tensor,
    collation: BatchSupervisionCollation,
) -> Tuple[torch.Tensor, int]:
    """根据严格 sample_key 命中结果计算 CAPRA 附加损失。"""
    if collation.num_hits <= 0:
        return torch.zeros((), dtype=predicted_actions.dtype, device=predicted_actions.device), 0

    losses: List[torch.Tensor] = []
    valid_weights: List[torch.Tensor] = []

    for i, batch_idx in enumerate(collation.matched_batch_indices):
        target = collation.safer_actions[i].to(device=predicted_actions.device, dtype=predicted_actions.dtype)
        pred = predicted_actions[batch_idx]

        if target.shape != pred.shape:
            if target.shape[0] > pred.shape[0]:
                target = target[: pred.shape[0]]
            elif target.shape[0] < pred.shape[0]:
                continue
            if target.shape[1] != pred.shape[1]:
                continue

        losses.append(torch.abs(pred - target).mean())
        valid_weights.append(collation.weights[i].to(device=predicted_actions.device, dtype=predicted_actions.dtype))

    if not losses:
        return torch.zeros((), dtype=predicted_actions.dtype, device=predicted_actions.device), 0

    stacked_losses = torch.stack(losses)
    stacked_weights = torch.stack(valid_weights).clamp_min(0.0)
    denom = stacked_weights.sum().clamp_min(1e-6)
    weighted = (stacked_losses * stacked_weights).sum() / denom
    return weighted, len(losses)


def compose_total_loss(
    task_loss: torch.Tensor,
    predicted_actions: torch.Tensor,
    collation: BatchSupervisionCollation,
    lambda_capra: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """组合真实 task_loss 与附加 capra_loss。"""
    capra_loss, capra_hits = compute_capra_loss_from_batch(
        predicted_actions=predicted_actions,
        collation=collation,
    )
    total_loss = task_loss + float(lambda_capra) * capra_loss
    return total_loss, capra_loss, capra_hits


def compute_task_loss_on_main_batch(
    vla: Any,
    action_head: Any,
    proprio_projector: Any,
    batch: Dict[str, Any],
    action_tokenizer: ActionTokenizer,
    device_id: int,
    use_proprio: bool,
    use_film: bool,
    num_patches: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """在 RLDS 主数据流 batch 上调用上游 forward，得到真实 task_loss。"""
    return UPSTREAM.run_forward_pass(
        vla=vla,
        action_head=action_head,
        noisy_action_projector=None,
        proprio_projector=proprio_projector,
        batch=batch,
        action_tokenizer=action_tokenizer,
        device_id=device_id,
        use_l1_regression=True,
        use_diffusion=False,
        use_proprio=use_proprio,
        use_film=use_film,
        num_patches=num_patches,
        compute_diffusion_l1=False,
        num_diffusion_steps_train=None,
    )


def _load_deepspeed_config(cfg: FinetuneCapraConfig, train_batch_size: int) -> Dict[str, Any]:
    if cfg.deepspeed_config_path:
        path = Path(cfg.deepspeed_config_path)
        if not path.exists():
            raise FileNotFoundError(f"DeepSpeed 配置不存在: {cfg.deepspeed_config_path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "train_batch_size": int(train_batch_size),
        "gradient_accumulation_steps": int(cfg.grad_accumulation_steps),
        "gradient_clipping": float(cfg.deepspeed_grad_clipping),
        "zero_optimization": {"stage": int(cfg.deepspeed_zero_stage)},
        "bf16": {"enabled": True},
        "fp16": {"enabled": False},
    }


@draccus.wrap()
def finetune_capra(cfg: FinetuneCapraConfig) -> None:
    """CAPRA 真实训练入口：task_loss（主数据流）+ capra_loss（附加 supervision）。"""
    assert cfg.use_lora, "CAPRA overlay 当前仅支持 LoRA 训练"
    assert cfg.use_l1_regression, "CAPRA overlay 当前要求 use_l1_regression=True"
    assert not cfg.use_diffusion, "CAPRA overlay 当前未实现 diffusion + CAPRA 联合损失"
    assert cfg.supervision_path, "必须提供 --supervision_path，且其仅作为附加 supervision"

    cfg.vla_path = cfg.vla_path.rstrip("/")
    run_id = f"{UPSTREAM.get_run_id(cfg)}+capra-lambda-{cfg.lambda_capra}"
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}")

    supervision_index: SupervisionLookupIndex = load_supervision_lookup_index(
        cfg.supervision_path,
        strict_key_only=True,
    )
    if distributed_state.is_main_process:
        print(f"Loaded CAPRA supervision keys: {len(supervision_index.by_sample_key)} from {cfg.supervision_path}")

    if UPSTREAM.model_is_on_hf_hub(cfg.vla_path):
        cfg.vla_path = UPSTREAM.snapshot_download(repo_id=cfg.vla_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    if distributed_state.is_main_process:
        UPSTREAM.update_auto_map(cfg.vla_path)
        UPSTREAM.check_model_logic_mismatch(cfg.vla_path)

    dist.barrier()

    processor, vla_raw = load_openvla_checkpoint(cfg.vla_path, device_id)

    vla_raw.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla_raw = get_peft_model(vla_raw, lora_config)
    vla_raw.print_trainable_parameters()

    if cfg.use_film:
        UPSTREAM.count_parameters(vla_raw.vision_backbone, "vla.vision_backbone (original)")
        vla_raw.model.vision_backbone = FiLMedPrismaticVisionBackbone(
            vision_backbone=vla_raw.model.vision_backbone,
            llm_dim=vla_raw.llm_dim,
        )
        UPSTREAM.count_parameters(vla_raw.vision_backbone, "vla.vision_backbone (post-wrap)")
        if cfg.resume:
            state_dict = UPSTREAM.load_checkpoint("vision_backbone", cfg.vla_path, cfg.resume_step)
            vla_raw.model.vision_backbone.load_state_dict(state_dict)
        vla_raw.model.vision_backbone = vla_raw.model.vision_backbone.to(device_id)

    if cfg.use_deepspeed:
        try:
            import deepspeed
        except ImportError as exc:
            raise ImportError("use_deepspeed=True 但环境未安装 deepspeed") from exc

        vla = vla_raw
    else:
        vla = UPSTREAM.wrap_ddp(vla_raw, device_id, find_unused=True)

    action_head = UPSTREAM.init_module(
        L1RegressionActionHead,
        "action_head",
        cfg,
        device_id,
        {"input_dim": _module_of(vla).llm_dim, "hidden_dim": _module_of(vla).llm_dim, "action_dim": ACTION_DIM},
        to_bf16=True,
    )

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = UPSTREAM.init_module(
            ProprioProjector,
            "proprio_projector",
            cfg,
            device_id,
            {"llm_dim": _module_of(vla).llm_dim, "proprio_dim": PROPRIO_DIM},
        )

    noisy_action_projector = None

    num_patches = _module_of(vla).vision_backbone.get_num_patches() * _module_of(vla).vision_backbone.get_num_images_in_input()
    if cfg.use_proprio:
        num_patches += 1

    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    trainable_params += [param for param in action_head.parameters() if param.requires_grad]
    if proprio_projector is not None:
        trainable_params += [param for param in proprio_projector.parameters() if param.requires_grad]

    optimizer = None
    scheduler = None
    if cfg.use_deepspeed:
        deepspeed_cfg = _load_deepspeed_config(cfg, train_batch_size=cfg.batch_size * distributed_state.num_processes)
        vla, optimizer, _, scheduler = deepspeed.initialize(
            model=vla,
            model_parameters=trainable_params,
            config=deepspeed_cfg,
        )
    else:
        optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
        scheduler = MultiStepLR(
            optimizer,
            milestones=[cfg.num_steps_before_decay],
            gamma=0.1,
        )

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    use_wrist_image = cfg.num_images_in_input > 1
    effective_shuffle_buffer_size = UPSTREAM.get_effective_shuffle_buffer_size(cfg)

    batch_transform = CapraRLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(_module_of(vla).config.image_sizes),
        shuffle_buffer_size=effective_shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    collator = CapraPaddedCollator(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    if distributed_state.is_main_process:
        print(
            "Detected constants:\n"
            f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
            f"\tACTION_DIM: {ACTION_DIM}\n"
            f"\tPROPRIO_DIM: {PROPRIO_DIM}\n"
            f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
        )

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        if optimizer is not None:
            optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            task_loss, task_metrics = compute_task_loss_on_main_batch(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
            )

            predicted_actions = _predict_actions_l1(
                vla=vla,
                action_head=action_head,
                proprio_projector=proprio_projector,
                batch=batch,
                device_id=device_id,
                use_proprio=cfg.use_proprio,
                use_film=cfg.use_film,
                num_patches=num_patches,
            )
            device = torch.device(f"cuda:{device_id}")
            collation = collate_training_targets(
                batch=batch,
                supervision_index=supervision_index,
                device=device,
                duplicate_strategy="max_weight",
            )
            total_loss, capra_loss, capra_hits = compose_total_loss(
                task_loss=task_loss,
                predicted_actions=predicted_actions,
                collation=collation,
                lambda_capra=cfg.lambda_capra,
            )

            if cfg.use_deepspeed:
                vla.backward(total_loss)
                vla.step()
                if batch_idx % cfg.grad_accumulation_steps == 0:
                    progress.update()
            else:
                assert optimizer is not None and scheduler is not None
                normalized_loss = total_loss / cfg.grad_accumulation_steps
                normalized_loss.backward()

                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    progress.update()

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx

            combined_metrics = {
                "task_loss": float(task_loss.detach().cpu().item()),
                "capra_loss": float(capra_loss.detach().cpu().item()),
                "total_loss": float(total_loss.detach().cpu().item()),
                "capra_hits": float(capra_hits),
            }
            combined_metrics.update({f"task_{k}": v for k, v in task_metrics.items()})

            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                wandb.log({f"CAPRA Train/{k}": v for k, v in combined_metrics.items()}, step=log_step)
                if scheduler is not None:
                    wandb.log({"CAPRA Train/Learning Rate": scheduler.get_last_lr()[0]}, step=log_step)

            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:
                UPSTREAM.save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    action_head=action_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )

            if log_step == cfg.max_steps:
                if distributed_state.is_main_process:
                    print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune_capra()
