# CAPRA 全流程实验指导书（零基础从头到尾）

适用人群：

- 第一次接触本仓库。
- 对 CAPRA、SafeLIBERO、RLDS 都不熟。
- 希望按步骤抄命令就能跑完一次完整实验。

本文目标：

1. 从环境检查开始。
2. 完成挖掘、训练、评测。
3. 留下可复现实验产物（日志、模型目录、评测 JSON）。

---

## 0. 全流程总览（你将完成什么）

你会依次完成 9 个阶段：

1. 准备环境与目录。
2. 检查脚本入口是否可调用。
3. 准备输入数据（RLDS + episodes）。
4. 运行 CAPRA 挖掘，得到 supervision JSONL。
5. 跑 baseline 训练（可选但推荐）。
6. 跑 CAPRA 训练（核心阶段）。
7. 跑 SafeLIBERO 主路径评测。
8. 跑 LIBERO success_rate 评测（可选）。
9. 汇总结果并做一次最小结论。

如果你按本文执行，最后至少会得到：

- 一个 supervision 文件：`tmp/capra/mined_v1.jsonl`
- 一个训练输出目录：`/path/to/runs/...`
- 一个 SafeLIBERO 评测 JSON：`tmp/capra/eval_metrics_v1.json`
- 一个 LIBERO 评测 JSON（可选）：`tmp/capra/eval_metrics_libero.json`

---

## 1. 先准备 6 个路径（先改成你自己的）

在开始前，请先确定以下路径变量：

1. 仓库根目录：`/path/to/openvla-oft`
2. Conda 环境名：`openvla-oft`
3. RLDS 根目录：`/path/to/rlds`
4. episodes 输入文件：`/path/to/episodes.jsonl`
5. 训练输出目录：`/path/to/runs`
6. SafeLIBERO 根目录：`vlsa-aegis/safelibero`（相对仓库）

建议你把这 6 个路径写到自己的实验笔记里，后面只替换变量，不要临时硬改命令。

---

## 2. 阶段 A：环境检查（不通过就不要往下跑）

### A1. 激活环境并进入仓库

在终端执行：

    conda activate openvla-oft
    cd /path/to/openvla-oft

预期结果：

- 终端提示你已在 openvla-oft 环境。
- 当前目录是仓库根目录。

### A2. 检查 Python 和 Torch

执行：

    python --version
    python -c "import torch; print(torch.__version__)"

预期结果：

- 两条命令都成功。
- 第二条能打印 torch 版本。

### A3. 检查四个核心入口是否存在

执行：

    python vla-scripts/finetune.py --help
    python vla-scripts/finetune_capra.py --help
    python -m experiments.robot.capra.pipelines.run_capra_mining --help
    python -m experiments.robot.capra.pipelines.run_capra_eval --help

预期结果：

- 四条 help 都能打印。
- 没有 ModuleNotFoundError。

如果这里失败，先去排障文档处理，再继续：

- `docs/05_troubleshooting/TROUBLESHOOTING.md`

---

## 3. 阶段 B：输入数据准备（实验能否成功的关键）

### B1. 你需要两类输入

1. RLDS 数据：训练主数据流。
2. episodes JSONL：CAPRA 挖掘输入。

### B2. 快速自检（只查文件存在）

执行：

    python -c "import os; print('rlds_exists=', os.path.exists('/path/to/rlds'))"
    python -c "import os; print('episodes_exists=', os.path.exists('/path/to/episodes.jsonl'))"

预期结果：

- 两个布尔值都应为 True。

### B3. 常见误区

1. 只准备了 RLDS，忘记 episodes。
2. episodes 文件路径写错一个目录层级。
3. dataset_name 与 RLDS 实际内容不匹配。

---

## 4. 阶段 C：运行 CAPRA 挖掘（生成 supervision）

这是 CAPRA 的第一步核心产物，后面训练会依赖它。

### C1. 执行挖掘命令

如果你在 Linux/macOS 或 Git Bash 里，执行：

    conda activate openvla-oft
    cd /path/to/openvla-oft
    bash scripts/capra/mine/mine_capra_v1.sh /path/to/episodes.jsonl your_pkg.your_env_factory:build_env tmp/capra/mined_v1.jsonl

参数解释：

1. 第 1 个参数：episodes 输入文件。
2. 第 2 个参数：env factory，格式必须是 module:function。
3. 第 3 个参数：输出 supervision JSONL。

### C2. 结果检查

执行：

    python -c "import os; p='tmp/capra/mined_v1.jsonl'; print('exists=', os.path.exists(p), 'size=', os.path.getsize(p) if os.path.exists(p) else -1)"

预期结果：

- exists=True
- size 大于 0

### C3. 挖掘失败时先看哪里

1. env_factory 是否真的是 module:function。
2. episodes 是否可读取。
3. 输出目录是否有写权限。

---

## 5. 阶段 D：先跑 baseline（可选但强烈推荐）

这一步的作用是确认你的基础训练链路正常。

### D1. baseline 最小训练

执行：

    conda activate openvla-oft
    cd /path/to/openvla-oft
    python vla-scripts/finetune.py --vla_path openvla/openvla-7b --data_root_dir /path/to/rlds --dataset_name libero_spatial_no_noops --run_root_dir /path/to/runs --batch_size 1 --max_steps 20 --learning_rate 5e-4

预期结果：

- 命令可跑完。
- run_root_dir 下有新输出目录。

如果你时间紧，也可以跳过 D，直接进入 CAPRA 训练。

---

## 6. 阶段 E：CAPRA 训练（主实验）

### E1. 最小 CAPRA 训练命令

执行：

    conda activate openvla-oft
    cd /path/to/openvla-oft
    python vla-scripts/finetune_capra.py --vla_path openvla/openvla-7b --data_root_dir /path/to/rlds --dataset_name libero_spatial_no_noops --run_root_dir /path/to/runs --supervision_path tmp/capra/mined_v1.jsonl --batch_size 1 --max_steps 20 --lambda_capra 1.0 --learning_rate 5e-4

### E2. 你必须看到的日志字段

1. task_loss
2. capra_loss
3. total_loss
4. capra_hits

关键判断：

- 如果 capra_hits 长期为 0，说明 supervision 没对齐上。
- 如果 total_loss 异常爆炸，先减小学习率或缩小 batch。

### E3. 第一轮实验建议配置

1. max_steps 先用 20-100 跑通。
2. lambda_capra 先固定 1.0。
3. 只改一个变量，不要一次改三四个参数。

---

## 7. 阶段 F：SafeLIBERO 主路径评测

### F1. 评测命令（默认主路径）

执行：

    conda activate openvla-oft
    cd /path/to/openvla-oft
    bash scripts/capra/eval/eval_capra_v1.sh tmp/capra/eval_metrics_v1.json safelibero_spatial I vlsa-aegis/safelibero

### F2. 输出检查

执行：

    python -c "import json; p='tmp/capra/eval_metrics_v1.json'; d=json.load(open(p,'r',encoding='utf-8')); print(d)"

你应该关注：

1. mode 是否是 safelibero_real。
2. 是否有 adapter_ok。
3. 是否有 num_tasks 和 sample_task。

注意：

- safelibero_real 当前重点是环境与任务入口可用性，不是 rollout success_rate。

---

## 8. 阶段 G：LIBERO success_rate 评测（可选）

如果你要看 success_rate，使用 libero_real。

执行：

    conda activate openvla-oft
    cd /path/to/openvla-oft
    python -m experiments.robot.capra.pipelines.run_capra_eval --benchmark_mode libero_real --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --task_suite_name libero_spatial --num_trials_per_task 50 --seed 7 --output_path tmp/capra/eval_metrics_libero.json

检查：

    python -c "import json; p='tmp/capra/eval_metrics_libero.json'; d=json.load(open(p,'r',encoding='utf-8')); print('success_rate=', d.get('success_rate'))"

---

## 9. 阶段 H：多卡 DeepSpeed（进阶，可后做）

如果你已经单卡跑通，可以再做这一步。

执行示例：

    conda activate openvla-oft
    cd /path/to/openvla-oft
    export NUM_GPUS=2
    bash scripts/capra/train/finetune_capra_v1_deepspeed.sh tmp/capra/mined_v1.jsonl openvla/openvla-7b /path/to/rlds libero_spatial_no_noops /path/to/runs 500 1.0

首次多卡建议：

1. 先把 max_steps 设小（例如 50）。
2. 先确认能稳定起跑，再扩步数。

---

## 10. 阶段 I：写实验记录（强烈建议）

每次实验至少记录这 8 项：

1. 日期与实验编号。
2. 使用的数据路径。
3. supervision 文件路径和条目规模。
4. 关键训练参数（lr、lambda、max_steps、batch_size）。
5. 训练日志中的关键现象（是否出现 capra_hits）。
6. SafeLIBERO 输出 JSON 摘要。
7. LIBERO success_rate（如果跑了）。
8. 下一轮计划只改哪一个变量。

这样你能避免“过了一周忘记自己做过什么”的问题。

---

## 11. 最小毕业标准（你可以拿来当完成检查）

满足以下 6 条，就算你独立跑通了 CAPRA 基础实验：

1. 能成功激活环境并调用四个入口 help。
2. 能产出非空的 mined_v1.jsonl。
3. 能成功跑完 finetune_capra 最小配置。
4. 日志里出现 task_loss/capra_loss/total_loss/capra_hits。
5. 能产出 safelibero_real 的评测 JSON。
6. 能解释你这次实验的核心参数和结果。

---

## 12. 一张快速故障对照表

1. capra_loss 一直为 0：优先查 supervision_path 和 key 对齐。
2. 挖掘输出为空：优先查 episodes 路径和 env_factory。
3. SafeLIBERO 没有 success_rate：这是当前模式边界，不是命令错。
4. 多卡启动失败：先回到单卡验证环境。

详细排障请直接看：

- `docs/05_troubleshooting/TROUBLESHOOTING.md`

---

## 13. 推荐阅读顺序（做完本文后）

1. `docs/02_dataset/DATASET_RLDS.md`
2. `docs/03_training/CAPRA_TRAINING.md`
3. `docs/04_evaluation/EVALUATION_GUIDE.md`
4. `docs/04_evaluation/METRICS_EXPLAINED.md`

你完成这四篇后，基本可以独立维护实验流程。
