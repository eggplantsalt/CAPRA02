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

原因：调用 finetune_capra.py 时没传监督文件，且没用 --tiny_smoke。

解决：

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

bash scripts/capra/mine/mine_capra_v1.sh tmp/capra/mined_v1.jsonl
python vla-scripts/finetune_capra.py --supervision_path tmp/capra/mined_v1.jsonl --steps 10
```

## 3. DeepSpeed mode requires CUDA device

原因：在 CPU 环境开启了 --use_deepspeed。

解决：

```bash
# 方案 A：不用 deepspeed
conda activate openvla-oft
cd /path/to/openvla-oft
python vla-scripts/finetune_capra.py --supervision_path tmp/capra/mined_v1.jsonl --steps 10

# 方案 B：切到有 GPU 的机器，再使用 deepspeed 脚本
conda activate openvla-oft
cd /path/to/openvla-oft
export NUM_GPUS=1
bash scripts/capra/train/finetune_capra_v1_deepspeed.sh tmp/capra/mined_v1.jsonl 200 1.0 1e-4
```

## 4. deepspeed is not installed

原因：环境缺少 deepspeed 包。

解决：

```bash
conda activate openvla-oft
pip install deepspeed
```

## 5. SafeLIBERO root not found

原因：--safelibero_root 路径错误。

解决：

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python -m experiments.robot.capra.pipelines.run_capra_eval \
  --benchmark_mode safelibero \
  --safelibero_root vlsa-aegis/safelibero \
  --task_suite_name safelibero_spatial \
  --safety_level I \
  --output_path tmp/capra/eval_adapter_safelibero.json
```

## 6. 显存不足或内存飙升

场景 A：Baseline 训练显存不足。

- 降低 --batch_size
- 关闭非必要开关（例如先不启用 diffusion）

场景 B：RLDS 数据加载占用过高。

- 显式传 --shuffle_buffer_size 2000
- 当前代码在 LIBERO/SafeLIBERO 数据名下也会自动限流到 2000

模板：

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/runs \
  --shuffle_buffer_size 2000 \
  --batch_size 1 \
  --max_steps 1000
```
