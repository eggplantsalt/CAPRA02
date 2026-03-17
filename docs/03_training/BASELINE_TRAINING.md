# Baseline 训练手册（OpenVLA-OFT 原生）

入口文件：`vla-scripts/finetune.py`

本页只讲“原生 baseline 怎么稳定跑起来”。

## 1. 必改参数（先记住这 6 个）

- `--vla_path`
- `--data_root_dir`
- `--dataset_name`
- `--run_root_dir`
- `--batch_size`
- `--max_steps`

## 2. 单卡最小模板（建议第一跑）

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
  --learning_rate 5e-4 \
  --max_steps 1000 \
  --save_freq 100
```

## 3. 标准训练模板（单卡）

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
  --lora_rank 32 \
  --wandb_entity your_entity \
  --wandb_project your_project \
  --run_id_note baseline_libero_spatial
```

## 4. 多卡模板（torchrun）

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /path/to/runs \
  --shuffle_buffer_size 2000 \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --max_steps 150005 \
  --save_freq 10000
```

## 5. 建议的调参顺序

1. 固定 `vla_path/data_root_dir/dataset_name`。
2. 先调 `batch_size` 保证不 OOM。
3. 再调 `learning_rate` 与 `max_steps`。
4. 最后再开大卡数。

## 6. 内存与稳定性提醒

- 数据名包含 `libero/safelibero` 时，代码会自动把 `shuffle_buffer_size` 限到 2000。
- 建议命令里明确写 `--shuffle_buffer_size 2000`。
