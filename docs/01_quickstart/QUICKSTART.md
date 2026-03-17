# Quickstart：零基础最短路径（当前仓库真实入口）

本文目标：让你在最短时间内完成三件事。

1. 环境激活并进入仓库
2. 验证上游 Baseline 训练/评测入口可调用
3. 验证 CAPRA 挖掘、训练、评测三条流水线入口可调用

注意：以下命令全部基于当前代码真实路径，不包含任何虚构参数。

## 1. 环境激活与目录切换

```bash
# 1) 激活你的 conda 环境（名称请替换为你的真实环境）
conda activate openvla-oft

# 2) 进入仓库根目录（请替换为你的真实路径）
cd /path/to/openvla-oft

# 3) 可选：确认 Python 与关键脚本存在
python --version
ls vla-scripts/finetune.py
ls vla-scripts/finetune_capra.py
```

## 2. Baseline 最小可用验证

### 2.1 Baseline 训练入口（单卡模板）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/runs \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --max_steps 10 \
  --save_freq 10 \
  --wandb_entity your_entity \
  --wandb_project your_project
```

说明：

- 训练入口使用 draccus 配置映射，参数名与 FinetuneConfig 一一对应。
- 当前仓库对 LIBERO/SafeLIBERO 数据名会自动限制 shuffle_buffer_size 上限为 2000。

### 2.2 Baseline 评测入口（LIBERO）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial
```

## 3. CAPRA 最小可用验证

### 3.1 监督挖掘（真实输入）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

bash scripts/capra/mine/mine_capra_v1.sh \
  /path/to/episodes.jsonl \
  your_pkg.your_env_factory:build_env \
  tmp/capra/mined_v1.jsonl
```

### 3.2 CAPRA 训练（真实 overlay）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python vla-scripts/finetune_capra.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/runs \
  --supervision_path tmp/capra/mined_v1.jsonl \
  --max_steps 5 \
  --lambda_capra 1.0 \
  --learning_rate 5e-4
```

### 3.3 CAPRA 评测（SafeLIBERO 主路径）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

bash scripts/capra/eval/eval_capra_v1.sh \
  tmp/capra/eval_metrics_v1.json \
  safelibero_spatial \
  I \
  vlsa-aegis/safelibero
```

说明：

- `run_capra_eval.py` 默认模式是 `safelibero_real`。
  - `safelibero_real` 当前输出环境可用性统计（如 `adapter_ok`、`num_tasks`），用于确认评测环境入口已接通。
- 如需基于 OpenVLA checkpoint 的 LIBERO 评测，可显式使用 `--benchmark_mode libero_real`。

## 4. DeepSpeed 最小模板（CAPRA）

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

如果你有自定义 deepspeed 配置文件：

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
