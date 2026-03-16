# CAPRA 训练指南（当前实现版本）

本文描述当前仓库里 CAPRA 已实现的训练能力：

1. 先挖掘 supervision JSONL
2. 用 finetune_capra.py 做 anchor + safer 回归训练
3. 可选使用 DeepSpeed

## 1. 先挖掘监督数据

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

bash scripts/capra/mine/mine_capra_v1.sh tmp/capra/mined_v1.jsonl
```

输出文件：tmp/capra/mined_v1.jsonl

## 2. 单机训练（非 DeepSpeed）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python vla-scripts/finetune_capra.py \
  --supervision_path tmp/capra/mined_v1.jsonl \
  --steps 200 \
  --lambda_capra 1.0 \
  --lr 1e-4
```

## 3. DeepSpeed 训练（脚本封装）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

export NUM_GPUS=1
bash scripts/capra/train/finetune_capra_v1_deepspeed.sh \
  tmp/capra/mined_v1.jsonl \
  200 \
  1.0 \
  1e-4
```

带自定义 deepspeed JSON：

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

export NUM_GPUS=2
bash scripts/capra/train/finetune_capra_v1_deepspeed.sh \
  tmp/capra/mined_v1.jsonl \
  500 \
  1.0 \
  1e-4 \
  /path/to/deepspeed_config.json
```

## 4. 关键参数与作用

- --supervision_path：监督 JSONL 路径
- --steps：优化步数
- --lambda_capra：safer loss 系数
- --lr：学习率
- --use_deepspeed：启用 deepspeed（脚本已自动传）
- --deepspeed_config_path：deepspeed 外部配置路径

## 5. 当前实现边界（非常重要）

- 当前 vla-scripts/finetune_capra.py 是 overlay 的训练回路，不是完整替代上游 finetune.py。
- 当前训练目标聚焦 anchor_loss + capra_loss，不包含 DPO、KL 投影等未实现内容。
