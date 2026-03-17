# 多卡与 DeepSpeed 操作手册

## 1. 先判断你该用哪种方式

- Baseline 原生多卡：`torchrun + vla-scripts/finetune.py`
- CAPRA 多卡：优先走 `scripts/capra/train/finetune_capra_v1_deepspeed.sh`

## 2. Baseline 多卡模板（torchrun）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/runs \
  --shuffle_buffer_size 2000 \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --max_steps 150005 \
  --save_freq 10000
```

## 3. CAPRA DeepSpeed 模板（推荐）

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

## 4. 带自定义 DeepSpeed 配置

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

## 5. 从 1 卡扩到 N 卡的稳妥步骤

1. 单卡先跑通（确认损失和保存目录正常）。
2. 扩到 2 卡（确认不会死锁或 OOM）。
3. 扩到目标卡数。

## 6. 最常见 3 个问题

1. `deepspeed` 没装。
2. `NUM_GPUS` 和实际可见 GPU 数不一致。
3. batch 太大导致 OOM。
