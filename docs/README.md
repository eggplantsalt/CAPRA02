# CAPRA 文档总览

本目录已按主题重构，后续所有中文教程将按以下结构维护。

## 当前目录结构

- `docs/00_overview/`：工程导览与目录索引
	- `PROJECT_MAP.md`
	- `DOCS_TREE_PLAN.md`
- `docs/01_quickstart/`：从零开始快速跑通
	- `QUICKSTART.md`
- `docs/02_dataset/`：数据集准备与 RLDS/SafeLIBERO
	- `DATASET_RLDS.md`
	- `SAFELIBERO_GUIDE.md`
- `docs/03_training/`：训练流程（Baseline 与 CAPRA）
	- `BASELINE_TRAINING.md`
	- `CAPRA_TRAINING.md`
	- `MULTI_GPU_DEEPSPEED.md`
- `docs/04_evaluation/`：评测流程与指标解读
	- `EVALUATION_GUIDE.md`
	- `METRICS_EXPLAINED.md`
- `docs/05_troubleshooting/`：常见问题与排障
	- `TROUBLESHOOTING.md`
- `docs/90_reference/`：有效参考文档
- `docs/99_archive/`：历史阶段文档（仅归档）

## 文档使用约定

- `docs/CONTEXT.md` 作为全局约束文件，保持原位且不修改。
- 归档区文档用于追溯历史决策，不作为默认执行手册。
- 新增教程优先采用“复制粘贴即可执行”的命令风格。

## 当前主路径速记

- CAPRA 训练主路径：`vla-scripts/finetune_capra.py`（RLDS task_loss + CAPRA 附加 supervision）。
- CAPRA 挖掘主路径：`experiments/robot/capra/pipelines/run_capra_mining.py`（episodes JSONL + env_factory）。
- CAPRA 评测默认主路径：`run_capra_eval.py --benchmark_mode safelibero_real`（当前输出 SafeLIBERO 环境可用性统计）。
- `debug_tiny`、`debug_custom_split` 属于调试路径，不作为论文主结果。
