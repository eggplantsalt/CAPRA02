# 评测指南（Baseline + CAPRA）

这份文档把“评测怎么跑、结果怎么看”一次说清楚。

## 1. Baseline（LIBERO）评测

入口：`experiments/robot/libero/run_libero_eval.py`

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50 \
  --seed 7
```

常用 suite：

- libero_spatial
- libero_object
- libero_goal
- libero_10
- libero_90

## 2. CAPRA 默认评测（SafeLIBERO 主路径）

入口：`experiments/robot/capra/pipelines/run_capra_eval.py`

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --benchmark_mode safelibero_real \
  --task_suite_name safelibero_spatial \
  --safety_level I \
  --safelibero_root vlsa-aegis/safelibero \
  --output_path tmp/capra/eval_metrics_v1.json
```

当前该模式主要输出：

- `adapter_ok`
- `num_tasks`
- `sample_task`

说明：

- 该主路径当前用于确认 SafeLIBERO 环境与套件入口可用。
- 还不是完整策略 rollout success_rate 报告。

## 3. CAPRA 可选评测（LIBERO checkpoint）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --benchmark_mode libero_real \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50 \
  --seed 7 \
  --output_path tmp/capra/eval_metrics_libero.json
```

该模式输出关键字段：

- `success_rate`

## 4. CAPRA debug 模式（仅排障）

### 4.1 custom split 适配检查

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --benchmark_mode debug_custom_split \
  --debug_mode \
  --custom_split support-critical-neighbor \
  --output_path tmp/capra/eval_adapter_custom_split.json
```

可选 split：

- support-critical-neighbor
- chain-reaction

## 5. 脚本方式（新手推荐）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

bash scripts/capra/eval/eval_capra_v1.sh \
  tmp/capra/eval_metrics_v1.json \
  safelibero_spatial \
  I \
  vlsa-aegis/safelibero
```

## 6. 结果文件怎么看

```bash
cat tmp/capra/eval_metrics_v1.json
```

你至少先确认：

1. `mode` 是否符合预期。
2. 输出字段是否符合对应模式。
3. 路径参数是否被正确回写到 JSON。
