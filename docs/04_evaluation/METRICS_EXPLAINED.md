# 指标与日志解读（新手版）

本文目标：你看到日志字段时，能立刻知道“它在表达什么”。

## 1. 训练日志字段（CAPRA）

入口：`vla-scripts/finetune_capra.py`

当前稳定字段：

- `task_loss`：RLDS 主数据流的任务损失。
- `capra_loss`：命中 supervision 样本时的附加损失。
- `total_loss`：总损失，`task_loss + lambda_capra * capra_loss`。
- `capra_hits`：当前 batch 命中 supervision 的样本数。

如何判断训练是否正常：

1. `task_loss` 非 NaN、可波动下降。
2. `capra_hits` 不是长期为 0（否则 supervision 没对齐上）。
3. `total_loss` 和 `task_loss/capra_loss` 数值量级相符。

## 2. 评测输出字段（按模式看）

### 2.1 `safelibero_real`（默认）

当前主要看：

- `adapter_ok`
- `num_tasks`
- `sample_task`

这组字段是“环境入口是否可用”的信号。

### 2.2 `libero_real`

当前主要看：

- `success_rate`

### 2.3 `debug_tiny`

会调用 `compute_metrics_v1`，输出：

- `SPIR`
- `EAR`
- `success_rate`
- `displacement_total_mean`
- `non_target_displacement_mean`
- `severe_event_rate`

## 3. 关于 Loss_act / Loss_ret / lambda / epsilon

你要求的这些术语里，当前代码状态是：

- `Loss_act`：当前代码没有这个字段。
- `Loss_ret`：当前代码没有这个字段。
- `lambda`：当前对应 `lambda_capra`（命令参数 `--lambda_capra`）。
- `epsilon`：当前对应 `epsilon_p`（定义在 mining 配置里，不是训练日志字段）。

因此你在当前实现中应该重点看：

- 训练：`task_loss/capra_loss/total_loss/capra_hits`
- 评测：按模式读取字段，不要跨模式套用。

## 4. 梯度冲突监控说明

当前代码没有显式输出“梯度冲突指数”或“梯度夹角”字段。

可替代的工程观察方式：

1. 看 `task_loss` 是否异常抖动或发散。
2. 看 `capra_loss` 在 `capra_hits>0` 时是否长期无法下降。
3. 看 `total_loss` 是否长期被某一项主导。

如果后续需要严格梯度冲突监控，需要在训练循环新增梯度统计日志。
