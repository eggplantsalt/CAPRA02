# CAPRA 文档总入口（先看这里）

这套文档只讲两件事：

1. 当前代码已经实现了什么。
2. 你应该按什么顺序把实验跑起来。

如果你是第一次接触本仓库，建议按下面顺序阅读。

## 一、建议阅读顺序（零基础）

1. `docs/00_overview/PROJECT_MAP.md`
2. `docs/01_quickstart/QUICKSTART.md`
3. `docs/01_quickstart/FULL_EXPERIMENT_PLAYBOOK.md`
4. `docs/02_dataset/DATASET_RLDS.md`
5. `docs/03_training/BASELINE_TRAINING.md`
6. `docs/03_training/CAPRA_TRAINING.md`
7. `docs/04_evaluation/EVALUATION_GUIDE.md`
8. `docs/05_troubleshooting/TROUBLESHOOTING.md`

## 二、目录作用说明

- `docs/00_overview/`：先搞清楚工程结构、主路径和调试路径。
- `docs/01_quickstart/`：一条最短路径，先把命令跑通再深入。
- `docs/02_dataset/`：RLDS 与 SafeLIBERO 的路径和字段约定。
- `docs/03_training/`：Baseline 与 CAPRA 的训练操作手册。
- `docs/04_evaluation/`：评测命令、输出解释和结果解读。
- `docs/05_troubleshooting/`：常见报错处理模板。
- `docs/90_reference/`：方法论参考，不等于当前实现状态。
- `docs/99_archive/`：历史阶段记录，仅追溯，不作当前执行依据。

## 三、当前主路径（务必区分）

- 训练主路径：`vla-scripts/finetune_capra.py`
  - 真实 RLDS 主数据流 task_loss + CAPRA 附加 supervision。
- 挖掘主路径：`experiments/robot/capra/pipelines/run_capra_mining.py`
  - 输入 episodes JSONL + env_factory，输出 supervision JSONL。
- 评测默认主路径：`experiments/robot/capra/pipelines/run_capra_eval.py --benchmark_mode safelibero_real`
  - 当前输出 SafeLIBERO 环境可用性统计（不是完整 rollout 成功率）。

调试路径：

- `debug_tiny`
- `debug_custom_split`

## 四、使用约束

- `docs/CONTEXT.md` 是全局约束文件，不应修改。
- 文档描述与代码冲突时，以代码真实行为为准。
- 历史文档只追溯，不用于当前实验命令。
