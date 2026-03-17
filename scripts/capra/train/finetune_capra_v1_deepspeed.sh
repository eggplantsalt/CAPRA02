#!/usr/bin/env bash
set -euo pipefail

# 参数顺序：
# 1 supervision_path（必填）
# 2 vla_path（必填）
# 3 data_root_dir（必填）
# 4 dataset_name（必填）
# 5 run_root_dir（必填）
# 6 max_steps（可选，默认 200）
# 7 lambda_capra（可选，默认 1.0）
# 8 learning_rate（可选，默认 5e-4）
# 9 deepspeed_config_path（可选）
SUPERVISION_PATH="${1:-}"
VLA_PATH="${2:-}"
DATA_ROOT_DIR="${3:-}"
DATASET_NAME="${4:-}"
RUN_ROOT_DIR="${5:-}"
MAX_STEPS="${6:-200}"
LAMBDA_CAPRA="${7:-1.0}"
LEARNING_RATE="${8:-5e-4}"
DEEPSPEED_CONFIG_PATH="${9:-}"
# 通过环境变量控制 GPU 数量，默认单卡。
NUM_GPUS="${NUM_GPUS:-1}"

if [[ -z "${SUPERVISION_PATH}" || -z "${VLA_PATH}" || -z "${DATA_ROOT_DIR}" || -z "${DATASET_NAME}" || -z "${RUN_ROOT_DIR}" ]]; then
  echo "Usage: bash scripts/capra/train/finetune_capra_v1_deepspeed.sh <supervision_path> <vla_path> <data_root_dir> <dataset_name> <run_root_dir> [max_steps] [lambda_capra] [learning_rate] [deepspeed_config_path]"
  exit 1
fi

CMD=(
  deepspeed
  --num_gpus "${NUM_GPUS}"
  vla-scripts/finetune_capra.py
  --supervision_path "${SUPERVISION_PATH}"
  --vla_path "${VLA_PATH}"
  --data_root_dir "${DATA_ROOT_DIR}"
  --dataset_name "${DATASET_NAME}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --max_steps "${MAX_STEPS}"
  --lambda_capra "${LAMBDA_CAPRA}"
  --learning_rate "${LEARNING_RATE}"
  --use_deepspeed
)

if [[ -n "${DEEPSPEED_CONFIG_PATH}" ]]; then
  CMD+=(--deepspeed_config_path "${DEEPSPEED_CONFIG_PATH}")
fi

"${CMD[@]}"
