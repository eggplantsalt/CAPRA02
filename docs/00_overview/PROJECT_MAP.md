# 工程目录导引（按当前真实结构）

本文给新同学一个明确边界：哪些是 upstream，哪些是 CAPRA overlay。

## 1. upstream 核心区域（尽量不改）

- prismatic/
- vla-scripts/finetune.py
- experiments/robot/libero/run_libero_eval.py
- experiments/robot/openvla_utils.py
- experiments/robot/robot_utils.py

## 2. CAPRA overlay 核心区域

### 2.1 Python 分层

- experiments/robot/capra/adapters/
	- env_adapter.py：统一环境与 sim 访问
	- state_api.py：obs-first, sim-backed 状态读取
	- benchmark_adapters.py：SafeLIBERO 环境可用性校验与 custom split 白名单
- experiments/robot/capra/core/
	- proposals.py：prefix-local 模板候选生成（含 gripper/protected dims 保护）
	- task_progress.py：progress-preserving gate
	- footprint.py：footprint v1 计算
	- local_evaluator.py：反事实短视界评估
	- mining.py：监督挖掘
	- training_targets.py：监督样本转训练张量
	- types.py：核心数据结构
- experiments/robot/capra/io/
	- supervision_io.py：JSONL 读写
- experiments/robot/capra/evaluation/
	- metrics.py：SPIR、EAR 与 episode 指标
- experiments/robot/capra/pipelines/
	- run_capra_mining.py：挖掘入口
	- run_capra_eval.py：评测入口（默认 safelibero_real）

### 2.2 Shell 分层

- scripts/capra/mine/mine_capra_v1.sh
- scripts/capra/train/finetune_capra_v1_deepspeed.sh
- scripts/capra/eval/eval_capra_v1.sh

## 3. 入口速查

### 3.1 Baseline

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

# 训练
python vla-scripts/finetune.py --help

# 评测
python experiments/robot/libero/run_libero_eval.py --help
```

### 3.2 CAPRA

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

# 挖掘
python -m experiments.robot.capra.pipelines.run_capra_mining --help

# 训练
python vla-scripts/finetune_capra.py --help

# 评测
python -m experiments.robot.capra.pipelines.run_capra_eval --help
```

## 4. 主路径与调试路径

- 主路径：
	- 训练：`vla-scripts/finetune_capra.py`（RLDS task_loss + CAPRA 附加 supervision）
	- 挖掘：`run_capra_mining.py`（episodes JSONL + env_factory）
	- 评测：`run_capra_eval.py --benchmark_mode safelibero_real`（当前返回 SafeLIBERO 环境可用性统计）
- debug/test-only：
	- `debug_tiny`、`debug_custom_split`
	- `tests/capra/*` 下单测与 smoke 测试
