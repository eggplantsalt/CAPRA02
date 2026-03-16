#!/usr/bin/env bash
set -euo pipefail

# 参数顺序：
# 1 supervision_path 2 steps 3 lambda_capra 4 lr 5 deepspeed_config_path(可选)
SUPERVISION_PATH="${1:-tmp/capra/mined_v1.jsonl}"
STEPS="${2:-200}"
LAMBDA_CAPRA="${3:-1.0}"
LR="${4:-1e-4}"
DEEPSPEED_CONFIG_PATH="${5:-}"
# 通过环境变量控制 GPU 数量，默认单卡。
NUM_GPUS="${NUM_GPUS:-1}"

CMD=(
  deepspeed
  --num_gpus "${NUM_GPUS}"
  vla-scripts/finetune_capra.py
  --supervision_path "${SUPERVISION_PATH}"
  --steps "${STEPS}"
  --lambda_capra "${LAMBDA_CAPRA}"
  --lr "${LR}"
  --use_deepspeed
)

if [[ -n "${DEEPSPEED_CONFIG_PATH}" ]]; then
  CMD+=(--deepspeed_config_path "${DEEPSPEED_CONFIG_PATH}")
fi

"${CMD[@]}"
