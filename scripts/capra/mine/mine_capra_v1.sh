#!/usr/bin/env bash
set -euo pipefail

# CAPRA 挖掘脚本（主路径）：
# - 输入真实 rollout episodes JSONL 与 env_factory
# - 输出可用于训练查表的 supervision JSONL

# 参数：
# 1 episodes_path（rollout episode JSONL）
# 2 env_factory（module:function）
# 3 output_path（可选，默认 tmp/capra/mined_v1.jsonl）
EPISODES_PATH="${1:-}"
ENV_FACTORY="${2:-}"
OUTPUT_PATH="${3:-tmp/capra/mined_v1.jsonl}"

if [[ -z "${EPISODES_PATH}" || -z "${ENV_FACTORY}" ]]; then
	echo "Usage: bash scripts/capra/mine/mine_capra_v1.sh <episodes_path> <env_factory> [output_path]"
	echo "Example: bash scripts/capra/mine/mine_capra_v1.sh tmp/capra/episodes.jsonl my_pkg.env_factory:build_env tmp/capra/mined_v1.jsonl"
	exit 1
fi

python -m experiments.robot.capra.pipelines.run_capra_mining \
	--episodes_path "${EPISODES_PATH}" \
	--env_factory "${ENV_FACTORY}" \
	--output_path "${OUTPUT_PATH}"
