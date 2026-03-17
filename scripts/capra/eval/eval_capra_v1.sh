#!/usr/bin/env bash
set -euo pipefail

# 默认主路径：SafeLIBERO 评测入口（当前返回环境可用性统计）。
# 参数顺序：
# 1 output_path（可选，默认 tmp/capra/eval_metrics_v1.json）
# 2 task_suite_name（可选，默认 safelibero_spatial）
# 3 safety_level（可选，默认 I）
# 4 safelibero_root（可选，默认 vlsa-aegis/safelibero）
OUTPUT_PATH="${1:-tmp/capra/eval_metrics_v1.json}"
TASK_SUITE_NAME="${2:-safelibero_spatial}"
SAFETY_LEVEL="${3:-I}"
SAFELIBERO_ROOT="${4:-vlsa-aegis/safelibero}"

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --output_path "${OUTPUT_PATH}" \
  --benchmark_mode "safelibero_real" \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --safety_level "${SAFETY_LEVEL}" \
  --safelibero_root "${SAFELIBERO_ROOT}"
