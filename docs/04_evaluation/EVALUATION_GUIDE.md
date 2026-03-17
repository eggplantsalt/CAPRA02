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

## 2. CAPRA 评测（默认 SafeLIBERO 主路径）

入口：experiments/robot/capra/pipelines/run_capra_eval.py

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --output_path tmp/capra/eval_metrics_v1.json \
  --benchmark_mode safelibero_real \
  --task_suite_name safelibero_spatial \
  --safety_level I \
  --safelibero_root vlsa-aegis/safelibero
```

当前该模式返回：

- `adapter_ok`、`num_tasks`、`sample_task` 等环境可用性信息
- 不直接输出策略 rollout 的 success_rate

## 3. CAPRA 评测（可选 LIBERO checkpoint 路径）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --output_path tmp/capra/eval_metrics_libero.json \
  --benchmark_mode libero_real \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50 \
  --seed 7
```

## 4. CAPRA debug 评测（custom split 适配检查）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --benchmark_mode debug_custom_split \
  --debug_mode \
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
  tmp/capra/eval_metrics_v1.json \
  safelibero_spatial \
  I \
  vlsa-aegis/safelibero
```

说明：

- `safelibero_real` 是默认主路径。
- `safelibero_real` 当前定位是 SafeLIBERO 环境入口与套件校验主路径。
- `debug_tiny/debug_custom_split` 必须显式添加 `--debug_mode`，仅用于 debug，不作为论文主结果。
