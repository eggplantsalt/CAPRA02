# SafeLIBERO 数据与参数指南

本文只覆盖当前仓库已经接入的真实能力：

1. SafeLIBERO 套件名与安全等级参数
2. CAPRA 评测适配器的调用方式
3. 常见路径设置

## 1. 当前支持的 SafeLIBERO 参数

评测入口：

- experiments/robot/capra/pipelines/run_capra_eval.py

参数：

- --benchmark_mode safelibero
- --safelibero_root（默认 vlsa-aegis/safelibero）
- --task_suite_name（默认 safelibero_spatial）
- --safety_level（I 或 II）

支持的 suite 名（来自 safelibero benchmark 注册）：

- safelibero_spatial
- safelibero_object
- safelibero_goal
- safelibero_long

## 2. 快速检查 SafeLIBERO 适配是否可用

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

## 3. 使用封装脚本调用（推荐）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

bash scripts/capra/eval/eval_capra_v1.sh \
  tmp/capra/mined_v1.jsonl \
  tmp/capra/eval_adapter_safelibero.json \
  safelibero \
  safelibero_spatial \
  I \
  support-critical-neighbor \
  vlsa-aegis/safelibero
```

说明：

- 脚本会统一透传 safelibero_root、suite、level。
- 脚本默认附带 --tiny 参数，但在 benchmark_mode=safelibero 时，逻辑分支会优先返回适配器结果。
