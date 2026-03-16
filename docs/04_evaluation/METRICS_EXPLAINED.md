# 指标与日志解释（严格按当前实现）

本文只解释当前代码已经产出的指标与日志。

## 1. CAPRA 训练日志（finetune_capra.py）

当前会打印：

- anchor_loss：锚点损失，约束模型不要偏离 base action
- capra_loss：加权 safer 回归损失
- total_loss：anchor_loss + lambda_capra * capra_loss

命令示例：

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python vla-scripts/finetune_capra.py \
  --supervision_path tmp/capra/mined_v1.jsonl \
  --steps 20 \
  --lambda_capra 1.0 \
  --lr 1e-4
```

## 2. CAPRA 评测指标（evaluation/metrics.py）

当前输出字段：

- SPIR：有效步中 base 不是最安全动作的比例
- EAR：有效步上的平均正 regret
- success_rate：episode 成功率
- displacement_total_mean：总位移均值
- non_target_displacement_mean：非目标位移均值
- severe_event_rate：严重事件比例
- num_effective_steps：有效时间步数
- num_episodes：episode 数

命令示例：

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --supervision_path tmp/capra/mined_v1.jsonl \
  --output_path tmp/capra/eval_metrics_v1.json \
  --benchmark_mode tiny \
  --tiny

cat tmp/capra/eval_metrics_v1.json
```

## 3. 关于 Loss_act、Loss_ret、lambda、epsilon 的说明

- 当前 CAPRA overlay 代码没有名为 Loss_act 或 Loss_ret 的日志字段。
- 当前实现里对应的训练系数是 lambda_capra（命令参数 --lambda_capra）。
- 当前实现里 progress gate 的阈值 epsilon_p 定义在 MiningConfigV1，但没有通过 CLI 暴露到 run_capra_mining.py。

结论：

- 你现在能直接监控的是 anchor_loss、capra_loss、total_loss、SPIR、EAR 等字段。
- 如果后续要在日志里显式输出 epsilon_p、delta_min、w_max，需要在 pipeline CLI 增加参数透传。
