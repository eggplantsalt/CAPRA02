# SafeLIBERO 数据与参数指南

本文只覆盖当前仓库的 SafeLIBERO 主路径能力：

1. SafeLIBERO 套件名与安全等级参数
2. CAPRA 评测主入口调用方式
3. 常见路径设置

## 1. 当前支持的 SafeLIBERO 参数

评测入口：

- experiments/robot/capra/pipelines/run_capra_eval.py

参数：

- --benchmark_mode safelibero_real（默认主路径）
- --safelibero_root（默认 vlsa-aegis/safelibero）
- --task_suite_name（默认 safelibero_spatial）
- --safety_level（I 或 II）

支持的 suite 名（来自 safelibero benchmark 注册）：

- safelibero_spatial
- safelibero_object
- safelibero_goal
- safelibero_long

## 2. 执行 SafeLIBERO 主路径评测

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --benchmark_mode safelibero_real \
  --safelibero_root vlsa-aegis/safelibero \
  --task_suite_name safelibero_spatial \
  --safety_level I \
  --output_path tmp/capra/eval_metrics_v1.json
```

当前输出重点：

- `adapter_ok`：SafeLIBERO 根路径与套件注册是否可用
- `num_tasks`：当前 suite 注册的任务数
- `sample_task`：样例任务元信息（用于快速核对套件加载）

说明：

- `safelibero_real` 已是默认主路径，但当前实现仍是“环境与套件可用性主路径”，不是完整策略 rollout 成功率评测。

## 3. 使用封装脚本调用（推荐）

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

- 评测脚本默认走 `safelibero_real` 主路径。
- 如需 LIBERO checkpoint 评测，改用 `--benchmark_mode libero_real` 并传 `--pretrained_checkpoint`。
