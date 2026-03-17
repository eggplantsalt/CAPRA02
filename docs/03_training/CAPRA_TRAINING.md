# CAPRA 训练手册（一步一步）

入口文件：`vla-scripts/finetune_capra.py`

## 1. 先理解 CAPRA 训练的数据流

CAPRA 训练同时吃两条输入：

1. RLDS 主数据流
   - 提供主任务 `task_loss`
2. supervision JSONL
   - 仅在 sample_key 命中时提供 `capra_loss`

总损失：

- `total_loss = task_loss + lambda_capra * capra_loss`

## 2. 第一步：生成 supervision JSONL

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

bash scripts/capra/mine/mine_capra_v1.sh \
  /path/to/episodes.jsonl \
  your_pkg.your_env_factory:build_env \
  tmp/capra/mined_v1.jsonl
```

输出：`tmp/capra/mined_v1.jsonl`

## 3. 第二步：单机 CAPRA 训练

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python vla-scripts/finetune_capra.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/runs \
  --supervision_path tmp/capra/mined_v1.jsonl \
  --batch_size 1 \
  --max_steps 200 \
  --lambda_capra 1.0 \
  --learning_rate 5e-4
```

你应该在日志里看到：

- `task_loss`
- `capra_loss`
- `total_loss`
- `capra_hits`

## 4. 第三步：DeepSpeed 训练（可选）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

export NUM_GPUS=1
bash scripts/capra/train/finetune_capra_v1_deepspeed.sh \
  tmp/capra/mined_v1.jsonl \
  openvla/openvla-7b \
  /path/to/rlds \
  libero_spatial_no_noops \
  /path/to/runs \
  200 \
  1.0
```

自定义 deepspeed config：

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

export NUM_GPUS=2
bash scripts/capra/train/finetune_capra_v1_deepspeed.sh \
  tmp/capra/mined_v1.jsonl \
  openvla/openvla-7b \
  /path/to/rlds \
  libero_spatial_no_noops \
  /path/to/runs \
  500 \
  1.0 \
  5e-4 \
  /path/to/deepspeed_config.json
```

## 5. 必改参数清单（最常用）

- `--supervision_path`：CAPRA 监督文件。
- `--vla_path`：模型路径。
- `--data_root_dir`：RLDS 根目录。
- `--dataset_name`：RLDS 数据名。
- `--run_root_dir`：输出目录。
- `--max_steps`：总步数。
- `--lambda_capra`：CAPRA 附加损失系数。
- `--learning_rate`：学习率。

## 6. 可选参数（按需改）

- `--use_deepspeed`
- `--deepspeed_config_path`
- `--grad_accumulation_steps`
- `--wandb_entity`
- `--wandb_project`

## 7. 当前实现边界（请务必知道）

- 这不是 toy loop，已经挂接上游真实训练骨架。
- supervision JSONL 是附加监督，不替代 RLDS 主数据流。
- 命中 sample_key 才会产生 capra_loss；不命中时等价于纯主任务训练。
