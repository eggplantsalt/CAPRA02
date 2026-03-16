# 评测指南（Baseline + CAPRA）

## 1. Baseline（LIBERO）评测

入口：experiments/robot/libero/run_libero_eval.py

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50 \
  --seed 7
```

可选 task_suite_name：

- libero_spatial
- libero_object
- libero_goal
- libero_10
- libero_90

## 2. CAPRA 评测（tiny 指标）

入口：experiments/robot/capra/pipelines/run_capra_eval.py

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --supervision_path tmp/capra/mined_v1.jsonl \
  --output_path tmp/capra/eval_metrics_v1.json \
  --benchmark_mode tiny \
  --tiny
```

## 3. CAPRA 评测（SafeLIBERO 适配检查）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --benchmark_mode safelibero \
  --safelibero_root vlsa-aegis/safelibero \
  --task_suite_name safelibero_spatial \
  --safety_level I \
  --output_path tmp/capra/eval_adapter_safelibero.json
```

## 4. CAPRA 评测（custom split 适配检查）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --benchmark_mode custom_split \
  --custom_split support-critical-neighbor \
  --output_path tmp/capra/eval_adapter_custom_split.json
```

可选 custom_split：

- support-critical-neighbor
- chain-reaction

## 5. 用封装脚本执行 CAPRA 评测

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

bash scripts/capra/eval/eval_capra_v1.sh \
  tmp/capra/mined_v1.jsonl \
  tmp/capra/eval_metrics_v1.json \
  tiny \
  safelibero_spatial \
  I \
  support-critical-neighbor \
  vlsa-aegis/safelibero
```
