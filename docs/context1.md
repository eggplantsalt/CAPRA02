你现在不是在做 smoke test，也不是在做最小链路验证。

你的任务是把当前 CAPRA 代码库从 toy/scaffold/harness 状态，升级成真正可用于论文实验的 OpenVLA-OFT 工程实现。这里的“真正工程实现”有严格含义：

1. 必须接真实 OpenVLA-OFT 模型，而不是 toy model / linear model / fake model。
2. 必须复用或挂接上游 `vla-scripts/finetune.py` 的训练骨架，而不是自己另写一套独立 tiny training loop。
3. 必须接真实 LIBERO / RLDS 主数据流作为 task/anchor 来源。
4. CAPRA supervision JSONL 只能作为附加 sparse supervision，不能替代主数据流。
5. 所有 tiny smoke / in-memory fake loop / hand-crafted toy sample 默认不能作为主实现，只能移入 tests 或 debug 目录。
6. 不允许把“能跑通 loss”误当作“实现了真实训练”。
7. 每一步都以可用于 GPU 真实训练、可导出 checkpoint、可用于论文 benchmark 为标准。
8. 不要过度工程化；尽量 overlay 到 OpenVLA-OFT，而不是重写其训练框架。
9. 任何对 repo 的重要设计判断，都要基于实际阅读过的代码文件，而不是猜测。
10. 每阶段结束时，必须明确列出：当前真实实现了什么、仍然缺什么、哪些 smoke path 已被降级或删除。

你的默认目标不是“证明想法可行”，而是“交付真实工程实现”。

做任何修改前，请先读取并审查：
- `vla-scripts/finetune.py`
- `experiments/robot/libero/run_libero_eval.py`
- 当前 `vla-scripts/finetune_capra.py`
- `experiments/robot/capra/` 下所有训练相关文件
- `scripts/capra/` 下所有训练/挖掘脚本

不要自动扩展到下一阶段；每次只完成我明确要求的阶段任务。
