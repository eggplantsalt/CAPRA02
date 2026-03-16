# Baseline 训练指南（OpenVLA-OFT 原生入口）

本文只使用当前仓库真实存在的脚本与参数。

入口：vla-scripts/finetune.py

## 1. 单卡训练模板（可直接复制）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

python vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/runs \
  --shuffle_buffer_size 2000 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity your_entity \
  --wandb_project your_project \
  --run_id_note baseline_libero_spatial
```

## 2. 多卡训练模板（torchrun）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/runs \
  --shuffle_buffer_size 2000 \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 100000 \
  --max_steps 150005 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity your_entity \
  --wandb_project your_project \
  --run_id_note baseline_multi_gpu
```

## 3. 参数说明（高频必改）

- --data_root_dir：RLDS 根目录
- --dataset_name：任务数据名，例如 libero_spatial_no_noops
- --run_root_dir：checkpoint 与日志输出目录
- --batch_size：每卡 batch
- --max_steps：总优化步数
- --save_freq：保存频率

## 4. 与内存相关的注意点

- 当前代码在 dataset_name 包含 libero 或 safelibero 时，会自动限制 shuffle_buffer_size 上限为 2000。
- 建议仍显式传入 --shuffle_buffer_size 2000，避免团队成员误解默认值。
