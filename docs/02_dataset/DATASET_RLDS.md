# 数据集准备指南：RLDS（与当前代码严格对齐）

本文只描述当前仓库真实会读取到的字段与路径。

## 1. Baseline 训练脚本如何读取数据

入口：

- vla-scripts/finetune.py

关键参数：

- --data_root_dir：RLDS 根目录
- --dataset_name：数据集名称（例如 libero_spatial_no_noops）
- --shuffle_buffer_size：请求的 buffer 大小

重要实现细节：

- 当 dataset_name 包含 libero 或 safelibero 时，代码会自动把 shuffle_buffer_size 限制到不超过 2000。
- 该逻辑用于降低 RLDS 数据加载时的内存风险。

## 2. 推荐目录组织

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

mkdir -p tmp/datasets/rlds
```

示例：

- 如果你的 RLDS 在 /data/rlds
- 则训练命令里写 --data_root_dir /data/rlds

## 3. Baseline 训练参数模板（含 RLDS 路径）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /data/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /data/runs \
  --shuffle_buffer_size 2000 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --max_steps 1000 \
  --save_freq 100
```

## 4. CAPRA 挖掘数据与训练数据格式

当前 CAPRA v1 使用 JSONL 监督文件：

- 生成入口：scripts/capra/mine/mine_capra_v1.sh
- 读取入口：vla-scripts/finetune_capra.py

典型字段：

- observation
- instruction
- base_action
- safer_action
- weight
- candidate_stats
- metadata

你可以直接用以下方式生成并消费：

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

bash scripts/capra/mine/mine_capra_v1.sh tmp/capra/mined_v1.jsonl
python vla-scripts/finetune_capra.py --supervision_path tmp/capra/mined_v1.jsonl --steps 10
```
