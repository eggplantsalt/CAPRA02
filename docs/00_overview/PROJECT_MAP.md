# 工程目录导引（新同学必读）

这份文件回答一个核心问题：

“我应该改哪里，哪里不能乱改？”

## 1. 哪些是上游原生代码（尽量不改）

- `prismatic/`
- `vla-scripts/finetune.py`
- `experiments/robot/libero/run_libero_eval.py`
- `experiments/robot/openvla_utils.py`
- `experiments/robot/robot_utils.py`

原则：

- 这些文件属于上游主干；CAPRA 以 overlay 方式接入，优先少改或不改。

## 2. 哪些是 CAPRA 代码（主要维护区）

### 2.1 Python 模块分层

- `experiments/robot/capra/adapters/`
  - `env_adapter.py`：环境快照/回滚抽象。
  - `state_api.py`：从 obs/info/sim 读取统一状态信号。
  - `benchmark_adapters.py`：SafeLIBERO 环境可用性校验、custom split 校验。

- `experiments/robot/capra/core/`
  - `proposals.py`：候选动作块生成（prefix-local 主路径）。
  - `task_progress.py`：progress-preserving gate。
  - `footprint.py`：局部风险度量。
  - `local_evaluator.py`：同一快照下反事实候选评估。
  - `mining.py`：将候选评估结果挖掘为 supervision。
  - `training_targets.py`：训练 batch 与 supervision 严格 key 对齐。
  - `types.py`：核心数据结构。

- `experiments/robot/capra/io/`
  - `supervision_io.py`：supervision JSONL 读写、schema 升级。

- `experiments/robot/capra/evaluation/`
  - `metrics.py`：SPIR/EAR 与 episode 统计函数。

- `experiments/robot/capra/pipelines/`
  - `run_capra_mining.py`：挖掘入口。
  - `run_capra_eval.py`：评测入口。

### 2.2 Shell 脚本分层

- `scripts/capra/mine/mine_capra_v1.sh`
- `scripts/capra/train/finetune_capra_v1_deepspeed.sh`
- `scripts/capra/eval/eval_capra_v1.sh`

## 3. 一分钟理解主路径

- 训练主路径：`vla-scripts/finetune_capra.py`
  - 主损失来自 RLDS 主数据流。
  - CAPRA supervision 是附加监督。

- 挖掘主路径：`run_capra_mining.py`
  - episodes JSONL + env_factory -> supervision JSONL。

- 评测默认主路径：`run_capra_eval.py --benchmark_mode safelibero_real`
  - 当前返回 SafeLIBERO 环境可用性统计。

## 4. 必须区分的 debug/test-only 路径

- `run_capra_eval.py` 里的 `debug_tiny`、`debug_custom_split`。
- `tests/capra/` 里的单元测试和 smoke 用例。

这些路径用于调试与验证，不是论文主结果路径。

## 5. 入口速查命令

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

# Baseline 训练入口
python vla-scripts/finetune.py --help

# CAPRA 训练入口
python vla-scripts/finetune_capra.py --help

# CAPRA 挖掘入口
python -m experiments.robot.capra.pipelines.run_capra_mining --help

# CAPRA 评测入口
python -m experiments.robot.capra.pipelines.run_capra_eval --help
```
