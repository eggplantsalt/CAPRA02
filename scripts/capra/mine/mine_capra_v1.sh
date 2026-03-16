#!/usr/bin/env bash
set -euo pipefail

OUTPUT_PATH="${1:-tmp/capra/mined_v1.jsonl}"
python -m experiments.robot.capra.pipelines.run_capra_mining --output_path "${OUTPUT_PATH}"
