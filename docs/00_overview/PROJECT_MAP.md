# 工程目录导引（Step 2 草案）

> 该文档在 Step 3 会扩展为完整新手导览版。

## CAPRA 代码分层（已完成）

- `experiments/robot/capra/adapters/`：环境、状态、SafeLIBERO 适配
- `experiments/robot/capra/core/`：提案生成、局部评估、监督挖掘、训练目标
- `experiments/robot/capra/io/`：监督数据读写
- `experiments/robot/capra/evaluation/`：指标计算
- `experiments/robot/capra/pipelines/`：命令行流水线入口

## CAPRA 脚本分层（已完成）

- `scripts/capra/mine/`：监督挖掘脚本
- `scripts/capra/train/`：训练脚本（含 DeepSpeed）
- `scripts/capra/eval/`：评测脚本

## Upstream 保持不破坏原则

- 原生 OpenVLA-OFT 核心逻辑与主入口保持不破坏。
- CAPRA 作为 overlay 分层接入，不覆盖上游业务语义。
