# 多卡与 DeepSpeed 模板

本页统一给出 Baseline 多卡与 CAPRA DeepSpeed 的可复制命令。

## 1. Baseline 多卡（torchrun）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/runs \
  --shuffle_buffer_size 2000 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --max_steps 150005 \
  --save_freq 10000 \
  --lora_rank 32 \
  --wandb_entity your_entity \
  --wandb_project your_project
```

## 2. CAPRA DeepSpeed（单机多卡）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

export NUM_GPUS=4
bash scripts/capra/train/finetune_capra_v1_deepspeed.sh \
  tmp/capra/mined_v1.jsonl \
  openvla/openvla-7b \
  /path/to/rlds \
  libero_spatial_no_noops \
  /path/to/runs \
  1000 \
  1.0
```

## 3. CAPRA DeepSpeed（自定义配置）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

export NUM_GPUS=4
bash scripts/capra/train/finetune_capra_v1_deepspeed.sh \
  tmp/capra/mined_v1.jsonl \
  openvla/openvla-7b \
  /path/to/rlds \
  libero_spatial_no_noops \
  /path/to/runs \
  1000 \
  1.0 \
  5e-4 \
  /path/to/deepspeed_config.json
```

## 4. 推荐执行顺序

1. 先用 NUM_GPUS=1 跑通流程
2. 再扩到 NUM_GPUS=2
3. 最后扩到目标卡数
