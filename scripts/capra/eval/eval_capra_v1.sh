#!/usr/bin/env bash
set -euo pipefail

# 参数顺序：
# 1 supervision_path  2 output_path  3 benchmark_mode  4 task_suite_name
# 5 safety_level      6 custom_split 7 safelibero_root
SUPERVISION_PATH="${1:-tmp/capra/mined_v1.jsonl}"
OUTPUT_PATH="${2:-tmp/capra/eval_metrics_v1.json}"
BENCHMARK_MODE="${3:-tiny}"
TASK_SUITE_NAME="${4:-safelibero_spatial}"
SAFETY_LEVEL="${5:-I}"
CUSTOM_SPLIT="${6:-support-critical-neighbor}"
SAFELIBERO_ROOT="${7:-vlsa-aegis/safelibero}"

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --supervision_path "${SUPERVISION_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --benchmark_mode "${BENCHMARK_MODE}" \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --safety_level "${SAFETY_LEVEL}" \
  --custom_split "${CUSTOM_SPLIT}" \
  --safelibero_root "${SAFELIBERO_ROOT}" \
  --tiny
