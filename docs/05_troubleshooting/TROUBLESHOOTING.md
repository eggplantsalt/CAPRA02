# 常见报错与排查

本文覆盖当前仓库最常见问题：环境、路径、显存、DeepSpeed、SafeLIBERO。

## 1. ModuleNotFoundError: experiments.robot.capra...

原因：不在仓库根目录执行，或环境没激活。

解决：

```bash
conda activate openvla-oft
cd /path/to/openvla-oft
python -m experiments.robot.capra.pipelines.run_capra_eval --help
```

## 2. supervision_path must be provided

原因：调用 finetune_capra.py 时没传监督文件。

解决：

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

bash scripts/capra/mine/mine_capra_v1.sh /path/to/episodes.jsonl your_pkg.your_env_factory:build_env tmp/capra/mined_v1.jsonl
python vla-scripts/finetune_capra.py --vla_path openvla/openvla-7b --data_root_dir /path/to/rlds --dataset_name libero_spatial_no_noops --run_root_dir /path/to/runs --supervision_path tmp/capra/mined_v1.jsonl --max_steps 10
```

## 3. DeepSpeed mode requires CUDA device
# 故障排查（傻瓜版）

目标：你只要照着这份文档做，就能把大部分常见问题定位到具体环节。

排查统一模板：

1. 报错是什么。
2. 去哪里看。
3. 改哪里。
4. 怎么复验。

## 1. `capra_loss` 一直是 0

报错/现象：

- 训练日志有 `task_loss`，但 `capra_loss` 长期为 0。
- `capra_hits` 经常是 0。

去哪里看：

1. 训练命令里的 `--supervision_path`。
2. 监督文件里 `sample_key` 是否真实存在。
3. 训练 batch 的样本 key 是否和 supervision key 同格式。

改哪里：

1. 确保 `--supervision_path` 指向你刚挖掘出的 JSONL。
2. 对齐 key 规则（不要一边带前缀一边不带）。

怎么复验：

1. 跑 20-50 steps 小实验。
2. 观察日志中 `capra_hits` 是否出现正值。
3. 观察 `capra_loss` 是否不再长期为 0。

## 2. supervision 命中率极低

报错/现象：

- 训练能跑，但 supervision 几乎不生效。

去哪里看：

1. `run_capra_mining.py` 输出的 supervision 条目数。
2. 挖掘过滤条件是否太严。

改哪里：

1. 先放宽挖掘阈值，确保先有覆盖。
2. 先保证“能命中”，再做“高质量筛选”。

怎么复验：

1. 对比新旧 supervision 文件条目数。
2. 再跑训练小实验，确认 `capra_hits` 增加。

## 3. SafeLIBERO 评测没有 `success_rate`

报错/现象：

- `safelibero_real` 输出里没有 `success_rate`。

去哪里看：

1. `run_capra_eval.py` 的模式分支。

解释（这是当前实现，不是你命令写错）：

1. `safelibero_real` 当前主打入口可用性信号。
2. 若要直接看 `success_rate`，使用 `libero_real`。

怎么复验：

1. 同样命令改 `--benchmark_mode libero_real` 再跑一次。
2. 输出 JSON 应该包含 `success_rate`。

## 4. DeepSpeed 多卡起不来

报错/现象：

- 初始化卡住或直接退出。

去哪里看：

1. `nvidia-smi`。
2. Python 里 `torch.cuda.is_available()` 和 `torch.cuda.device_count()`。
3. DeepSpeed 是否在当前环境安装。

改哪里：

1. 先修环境一致性，再谈训练参数。
2. 先单卡跑通，再切多卡。

怎么复验：

1. 单卡命令稳定跑通。
2. 多卡最小步数（20-50 step）可启动并产生日志。

## 5. SafeLIBERO 任务加载失败

报错/现象：

- 提示 task suite 不存在或资源找不到。

去哪里看：

1. `--safelibero_root` 路径。
2. `--task_suite_name` 是否拼写正确。

改哪里：

1. 先用文档里的最小 suite 验证路径。
2. 再替换成目标 suite。

怎么复验：

1. 运行评测命令后，输出 JSON 至少包含 `adapter_ok/num_tasks/sample_task`。

## 6. 你可以直接复制的“最小复现命令”

### 6.1 训练最小复现

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python vla-scripts/finetune_capra.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/runs \
  --supervision_path tmp/capra/mined_v1.jsonl \
  --max_steps 20 \
  --batch_size 2
```

### 6.2 评测最小复现（SafeLIBERO 主路径）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --output_path tmp/capra/eval_metrics_v1.json \
  --benchmark_mode safelibero_real \
  --task_suite_name safelibero_spatial \
  --safety_level I \
  --safelibero_root vlsa-aegis/safelibero
```

## 7. 通用排查顺序（避免绕路）

1. 先查路径和文件是否存在。
2. 再查 key 和字段是否对齐。
3. 先做最小规模跑通（单卡、少步数）。
4. 跑通后再加多卡和调参。
5. 最后再做指标解释与实验对比。
