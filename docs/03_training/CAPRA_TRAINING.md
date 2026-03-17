# CAPRA 训练指南（当前实现版本）

本文描述当前仓库里 CAPRA 已实现的训练能力：

1. 先挖掘 supervision JSONL
2. 用 finetune_capra.py 在真实 RLDS 主数据流上做 overlay 训练
3. 可选使用 DeepSpeed

## 1. 先挖掘监督数据

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

bash scripts/capra/mine/mine_capra_v1.sh \
  /path/to/episodes.jsonl \
  your_pkg.your_env_factory:build_env \
  tmp/capra/mined_v1.jsonl
```

输出文件：tmp/capra/mined_v1.jsonl

## 2. 单机训练（非 DeepSpeed）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python vla-scripts/finetune_capra.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/runs \
  --supervision_path tmp/capra/mined_v1.jsonl \
  --max_steps 200 \
  --lambda_capra 1.0 \
  --learning_rate 5e-4
```

## 3. DeepSpeed 训练（脚本封装）

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

带自定义 deepspeed JSON：

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

## 4. 关键参数与作用

- --supervision_path：监督 JSONL 路径
- --max_steps：优化步数
- --lambda_capra：safer loss 系数
- --learning_rate：学习率
- --use_deepspeed：启用 deepspeed（脚本已自动传）
- --deepspeed_config_path：deepspeed 外部配置路径
- --vla_path：OpenVLA checkpoint 路径
- --data_root_dir / --dataset_name：真实 RLDS 数据流输入
- --run_root_dir：checkpoint 与日志输出目录

## 5. 当前实现边界（非常重要）

- 当前 vla-scripts/finetune_capra.py 是 overlay 到上游真实训练骨架，不是独立 toy 回路。
- CAPRA supervision 是附加 sparse supervision，主损失来自 RLDS 主数据流 task_loss。
