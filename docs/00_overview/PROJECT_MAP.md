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
	- benchmark_adapters.py：SafeLIBERO 与 custom split 适配
- experiments/robot/capra/core/
	- proposals.py：局部动作候选生成
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
	- run_capra_eval.py：评测入口

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
