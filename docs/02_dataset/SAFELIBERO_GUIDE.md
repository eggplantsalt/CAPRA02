# SafeLIBERO 指南（主路径版）

如果你当前实验主要基于 SafeLIBERO，这份文件就是你的主手册。

## 1. 当前代码支持什么

评测入口：

- `experiments/robot/capra/pipelines/run_capra_eval.py`

默认模式：

- `--benchmark_mode safelibero_real`

可配参数：

- `--safelibero_root`（默认 `vlsa-aegis/safelibero`）
- `--task_suite_name`（默认 `safelibero_spatial`）
- `--safety_level`（`I` 或 `II`）

支持 suite：

- `safelibero_spatial`
- `safelibero_object`
- `safelibero_goal`
- `safelibero_long`

## 2. 一步检查 SafeLIBERO 是否接通

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

看输出 JSON 里的字段：

- `adapter_ok=1`：路径与注册基本可用
- `num_tasks`：suite 任务数量
- `sample_task`：样例任务

## 3. 用脚本执行（推荐新手）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

bash scripts/capra/eval/eval_capra_v1.sh \
  tmp/capra/eval_metrics_v1.json \
  safelibero_spatial \
  I \
  vlsa-aegis/safelibero
```

## 4. 新手最常见配置错误

1. `safelibero_root` 指到了 `vlsa-aegis`，而不是 `vlsa-aegis/safelibero`。
2. `safety_level` 写成小写 i/ii。
3. `task_suite_name` 拼写错误。

## 5. 重要边界（避免误解）

- `safelibero_real` 现在是默认主路径。
- 但当前主路径输出的是“环境可用性统计”，还不是完整策略 rollout success_rate 报告。
- 如果你要拿 checkpoint 做完整 success_rate，对应当前可用的是 `libero_real` 分支。
