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
