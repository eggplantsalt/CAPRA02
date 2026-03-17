# 数据集准备指南：RLDS + CAPRA supervision

本文只描述“当前代码真实会读取的输入”。

## 1. 先搞清楚两类输入

本工程有两条输入流：

1. 主训练数据流（RLDS）
   - 给 `vla-scripts/finetune.py` 和 `vla-scripts/finetune_capra.py`
2. CAPRA 附加监督流（supervision JSONL）
   - 由 mining 生成，给 `vla-scripts/finetune_capra.py`

不要混淆：

- RLDS 是主数据流。
- supervision JSONL 只是 CAPRA 附加监督。

## 2. RLDS 路径参数怎么填

训练参数里你必须填写：

- `--data_root_dir`：RLDS 根目录
- `--dataset_name`：数据集名（例如 `libero_spatial_no_noops`）

示例：

- 你把数据放在 `/data/rlds`
- 则命令里写 `--data_root_dir /data/rlds`

## 3. 推荐目录布局（可直接照抄）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

mkdir -p tmp/datasets/rlds
mkdir -p tmp/capra
mkdir -p tmp/runs
```

## 4. RLDS 训练常用模板

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/runs \
  --shuffle_buffer_size 2000 \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --max_steps 1000
```

说明：

- 当前代码会对 `libero/safelibero` 数据名自动限流 `shuffle_buffer_size<=2000`。
- 建议你仍显式写 `--shuffle_buffer_size 2000`，避免多人协作时误配置。

## 5. CAPRA supervision JSONL 必备字段

按当前 schema（v2），一条记录至少包含：

- `sample_key`
- `lookup_key`
- `instruction`
- `safer_action`
- `weight`
- `align`
- `provenance`
- `candidate_stats`
- `metadata`

详细对齐见：`docs/CAPRA_SUPERVISION_ALIGNMENT.md`。

## 6. 从 episodes 到 supervision 的傻瓜命令

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

bash scripts/capra/mine/mine_capra_v1.sh \
  /path/to/episodes.jsonl \
  your_pkg.your_env_factory:build_env \
  tmp/capra/mined_v1.jsonl
```

## 7. 把 RLDS + supervision 一起喂给 CAPRA 训练

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
  --lambda_capra 1.0
```
