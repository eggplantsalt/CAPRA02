#!/usr/bin/env bash
set -euo pipefail

# 参数1：输出监督文件路径（默认写到 tmp/capra）
OUTPUT_PATH="${1:-tmp/capra/mined_v1.jsonl}"
python -m experiments.robot.capra.pipelines.run_capra_mining --output_path "${OUTPUT_PATH}"
