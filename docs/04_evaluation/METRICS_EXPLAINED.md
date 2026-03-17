# 指标与日志解释（严格按当前实现）

本文只解释当前代码已经产出的指标与日志。

## 1. CAPRA 训练日志（finetune_capra.py）

当前会打印：

- task_loss：主数据流任务损失（来自 RLDS batch）
- capra_loss：命中 supervision 时的附加 safer 回归损失
- total_loss：task_loss + lambda_capra * capra_loss

命令示例：

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python vla-scripts/finetune_capra.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/runs \
  --supervision_path tmp/capra/mined_v1.jsonl \
  --max_steps 20 \
  --lambda_capra 1.0 \
  --learning_rate 5e-4
```

## 2. CAPRA 评测输出（按模式区分）

### 2.1 debug_tiny（调试指标聚合）

调用 `compute_metrics_v1`，输出字段包括：

- SPIR：有效步中 base 不是最安全动作的比例
- EAR：有效步上的平均正 regret
- success_rate：episode 成功率
- displacement_total_mean：总位移均值
- non_target_displacement_mean：非目标位移均值
- severe_event_rate：严重事件比例
- num_effective_steps：有效时间步数
- num_episodes：episode 数

### 2.2 safelibero_real（默认主路径）

当前输出以环境可用性统计为主：

- adapter_ok
- num_tasks
- sample_task
- task_suite_name
- safety_level

说明：该模式当前不直接输出策略 rollout success_rate。

### 2.3 libero_real（可选路径）

复用上游 `run_libero_eval.py`，当前返回：

- success_rate
- task_suite_name

命令示例：

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --output_path tmp/capra/eval_metrics_v1.json \
  --benchmark_mode safelibero_real \
  --task_suite_name safelibero_spatial \
  --safety_level I \
  --safelibero_root vlsa-aegis/safelibero

cat tmp/capra/eval_metrics_v1.json
```

## 3. 关于 Loss_act、Loss_ret、lambda、epsilon 的说明

- 当前 CAPRA overlay 代码没有名为 Loss_act 或 Loss_ret 的日志字段。
- 当前实现里对应的训练系数是 lambda_capra（命令参数 --lambda_capra）。
- 当前实现里 progress gate 的阈值 epsilon_p 定义在 MiningConfigV1，但没有通过 CLI 暴露到 run_capra_mining.py。

结论：

- 你现在能直接监控的是 task_loss、capra_loss、total_loss，以及评测模式对应的输出字段。
- SPIR、EAR 当前属于 `debug_tiny` 指标聚合路径，不是 `safelibero_real` 默认输出。
- 如果后续要在主评测路径输出更多 CAPRA 指标，需要继续接入完整 rollout 结果并扩展 pipeline 输出。
