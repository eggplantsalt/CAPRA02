# CAPRA 真实工程化审计报告（当前代码状态）

审计日期：2026-03-17

审计范围：
- vla-scripts/finetune_capra.py
- experiments/robot/capra/
- scripts/capra/
- docs/（现有文档树）

审计基线：
- vla-scripts/finetune.py
- experiments/robot/libero/run_libero_eval.py

## 1. 总体结论

当前 CAPRA 已从早期 toy/scaffold 状态进入“主路径可运行 + 若干能力待补全”的工程阶段：

- 训练主路径：已接入上游真实训练骨架（RLDS 主 batch + CAPRA 附加 supervision）。
- mining 主路径：已是真实输入路径（episodes JSONL + env_factory）。
- proposals 主路径：已升级为 prefix-local、gripper-safe、模板化局部候选。
- eval 主路径：默认切到 SafeLIBERO 路径（safelibero_real），但当前输出仍以环境可用性统计为主，尚未接入完整 rollout 成功率评测。

## 2. 主路径与 debug/test-only 边界

主路径：
- 训练：vla-scripts/finetune_capra.py
- 挖掘：experiments/robot/capra/pipelines/run_capra_mining.py
- 评测：experiments/robot/capra/pipelines/run_capra_eval.py（默认 benchmark_mode=safelibero_real）

debug/test-only：
- run_capra_eval.py 的 debug_tiny、debug_custom_split
- tests/capra/* 下的单元测试与 smoke 测试
- run_capra_mining.py 的 --debug_summary_path 输出

## 3. 训练实现状态（finetune_capra）

当前真实行为：
- 复用上游训练骨架能力（模型加载、RLDSDataset、forward pass、checkpoint）。
- task_loss 来自 RLDS 主数据流。
- CAPRA loss 来自 supervision 命中样本上的附加 safer 回归。
- 总损失：total_loss = task_loss + lambda_capra * capra_loss。

边界：
- supervision JSONL 不是主数据流，只是附加 sparse supervision。

## 4. mining 实现状态

当前真实行为：
- 从 episodes JSONL 读取时序样本。
- 通过 env_factory 构造环境并执行局部反事实评估。
- 满足 supervision 条件时输出 SupervisionRecord。
- 写出可供训练严格查表使用的 sample_key/lookup_key 对齐记录。

不产出 supervision 的典型情况：
- progress-preserving 集为空；
- 局部 regret 未超过阈值 delta_min；
- 没有比 base 更安全的候选。

## 5. proposals 实现状态（重点）

source of truth：
- experiments/robot/capra/core/proposals.py
- 函数：build_local_proposals(...)

当前主路径：
- prefix-local templates（默认主路径）
- 不是默认 whole-chunk 扰动

主模板：
- speed_prefix_*
- lift_prefix
- retract_prefix
- lift_then_go
- lateral_left_prefix / lateral_right_prefix（lateral_delta 非 0 时启用）

关键语义：
- prefix_steps：只改 chunk 前缀步数，后缀保持 base。
- include_base：是否保留 identity 基线候选（base）。
- gripper/protected dims：通过 gripper_index + protected_dims 统一保护，不参与缩放或加噪。

次级候选：
- gaussian：次级补充、可选、默认关闭（num_gaussian=0），且默认不改保护维。
- whole-chunk 模板：次级、默认关闭（enable_whole_chunk_templates=False）。

当前边界：
- proposals 仍是规则模板，不是 learned proposal policy / policy sampling / buffer retrieval。

## 6. evaluation 实现状态

评测入口：
- experiments/robot/capra/pipelines/run_capra_eval.py

当前模式能力：
- safelibero_real（默认）：当前返回环境可用性统计（adapter_ok、num_tasks、sample_task）。
- libero_real（可选）：复用上游 run_libero_eval，返回 success_rate。
- debug_tiny/debug_custom_split：仅调试用途。

说明：
- 默认主路径已不是 tiny/smoke。
- 但 safelibero_real 仍未接入完整策略 rollout 指标闭环，这部分属于后续待实现。

## 7. deferred / future work

当前仍 deferred：
- learned proposal policy（替代规则模板）；
- full CAPRA 闭环调度（全局多轮 mining-train-eval）；
- safelibero_real 下的完整策略 rollout 成功率评测链路（当前主路径仍是环境校验输出）。

## 8. 与历史文档的关系

- docs/99_archive/ 下文档保留历史阶段结论，仅用于追溯。
- 现行执行口径以 docs/01~05 与当前代码实现为准。
