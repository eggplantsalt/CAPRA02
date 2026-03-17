# Quickstart：零基础一条龙傻瓜流程

这份文档的目标是：

1. 不需要理解源码，也能按步骤把主流程跑起来。
2. 每一步都给“复制粘贴可执行”的命令模板。
3. 明确每一步的预期结果，便于快速定位卡点。

## 0. 你需要提前准备什么

至少准备以下路径：

- 仓库路径：`/path/to/openvla-oft`
- RLDS 路径：`/path/to/rlds`
- 训练输出路径：`/path/to/runs`
- 挖掘输入 episodes：`/path/to/episodes.jsonl`

## 1. 第一步：进入正确环境

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python --version
python -c "import torch; print(torch.__version__)"
```

预期：

- 不报错，能打印 Python 和 Torch 版本。

## 2. 第二步：先检查入口是否可调用

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python vla-scripts/finetune.py --help
python vla-scripts/finetune_capra.py --help
python -m experiments.robot.capra.pipelines.run_capra_mining --help
python -m experiments.robot.capra.pipelines.run_capra_eval --help
```

预期：

- 四条命令都能打印 help，不报 `ModuleNotFoundError`。

## 3. 第三步：跑最小 CAPRA 挖掘

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

bash scripts/capra/mine/mine_capra_v1.sh \
  /path/to/episodes.jsonl \
  your_pkg.your_env_factory:build_env \
  tmp/capra/mined_v1.jsonl
```

预期：

- 终端出现 `Wrote ... records to tmp/capra/mined_v1.jsonl`。
- `tmp/capra/mined_v1.jsonl` 文件存在且非空。

## 4. 第四步：跑最小 CAPRA 训练

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python vla-scripts/finetune_capra.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/runs \
  --supervision_path tmp/capra/mined_v1.jsonl \
  --batch_size 1 \
  --max_steps 20 \
  --lambda_capra 1.0 \
  --learning_rate 5e-4
```

预期：

- 日志中出现 `task_loss`、`capra_loss`、`total_loss`。
- 在 `--run_root_dir` 下能看到训练输出目录。

## 5. 第五步：跑 CAPRA 默认评测（SafeLIBERO 主路径）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

bash scripts/capra/eval/eval_capra_v1.sh \
  tmp/capra/eval_metrics_v1.json \
  safelibero_spatial \
  I \
  vlsa-aegis/safelibero
```

预期：

- 输出 JSON 中出现 `mode=safelibero_real`。
- 有 `adapter_ok`、`num_tasks`、`sample_task` 等字段。

重要说明：

- `safelibero_real` 当前是环境可用性主路径，不是完整 rollout 成功率评测。

## 6. 第六步：可选跑 LIBERO checkpoint 评测

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --benchmark_mode libero_real \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50 \
  --seed 7 \
  --output_path tmp/capra/eval_metrics_libero.json
```

预期：

- 输出 JSON 中出现 `success_rate` 字段。

## 7. 第七步：可选 DeepSpeed 模板

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

export NUM_GPUS=1
bash scripts/capra/train/finetune_capra_v1_deepspeed.sh \
  tmp/capra/mined_v1.jsonl \
  openvla/openvla-7b \
  /path/to/rlds \
  libero_spatial_no_noops \
  /path/to/runs \
  200 \
  1.0
```

## 8. 新手最容易犯的 5 个错误

1. 不在仓库根目录执行命令。
2. `--supervision_path` 没传或路径写错。
3. `env_factory` 不是 `module:function` 形式。
4. SafeLIBERO 路径不是 `vlsa-aegis/safelibero` 实际位置。
5. RLDS 路径和 dataset_name 对不上。

出错后直接看：`docs/05_troubleshooting/TROUBLESHOOTING.md`。
