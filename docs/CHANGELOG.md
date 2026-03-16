# 文档变更日志

> 说明：本日志记录对旧版 Markdown 的删除、重写、归档等操作决策与原因。

## 2026-03-16

### 删除

- 删除 `docs/CAPRA02.md`
  - 原因：与 `docs/90_reference/IDEA.md` 存在较大语义重叠，且阶段性指令内容已由代码与进度文档吸收，继续保留会造成新成员阅读混乱。

### 归档迁移（非删除）

- `docs/CAPRA_PLAN.md` -> `docs/99_archive/CAPRA_PLAN.md`
  - 原因：该文件主要记录早期阶段推进计划，属于历史过程文档。
- `docs/CAPRA_CONTEXT.md` -> `docs/99_archive/CAPRA_CONTEXT.md`
  - 原因：与全局约束及现行实现存在时间差，保留用于追溯。
- `docs/CAPRA_BENCHMARK_ADAPTERS.md` -> `docs/99_archive/CAPRA_BENCHMARK_ADAPTERS.md`
  - 原因：文件中的模块路径基于旧目录结构，已不再是最新入口。
- `docs/IDEA.md` -> `docs/90_reference/IDEA.md`
  - 原因：属于有效方法论与研究边界定义，纳入参考区长期保留。

### 新增

- 新建 `docs/README.md`
  - 作用：提供重构后的文档目录入口与维护约定。

## 2026-03-16（Step 3）

### 重写

- 重写 `docs/00_overview/PROJECT_MAP.md`
  - 原因：从“草案状态”升级为“当前真实代码结构”导览，补齐真实入口命令。
- 重写 `docs/README.md`
  - 原因：补充完整文档索引，确保 00_overview 至 05_troubleshooting 可直接导航。

### 新增

- `docs/01_quickstart/QUICKSTART.md`
- `docs/02_dataset/DATASET_RLDS.md`
- `docs/02_dataset/SAFELIBERO_GUIDE.md`
- `docs/03_training/BASELINE_TRAINING.md`
- `docs/03_training/CAPRA_TRAINING.md`
- `docs/03_training/MULTI_GPU_DEEPSPEED.md`
- `docs/04_evaluation/EVALUATION_GUIDE.md`
- `docs/04_evaluation/METRICS_EXPLAINED.md`
- `docs/05_troubleshooting/TROUBLESHOOTING.md`

说明：

- 以上文档均基于当前真实代码入口与参数编写。
- 每篇文档均提供可复制粘贴的命令模板（含 conda activate 与 cd）。
