# 文档树规划（Step 2 输出）

> 说明：这是文档结构规划快照；当前实现口径请以同目录 `PROJECT_MAP.md` 与 01~05 教程为准。

本文件用于固定 Step 2 的新文档树，Step 3 将在此基础上填充完整教程内容。

## 目标文档树

- `docs/00_overview/`
  - `PROJECT_MAP.md`：工程目录导引（CAPRA 与 upstream 边界）
- `docs/01_quickstart/`
  - `QUICKSTART.md`：零基础最短路径
- `docs/02_dataset/`
  - `DATASET_RLDS.md`：RLDS 格式与路径规则
  - `SAFELIBERO_GUIDE.md`：SafeLIBERO 接入说明
- `docs/03_training/`
  - `BASELINE_TRAINING.md`：上游 Baseline 训练
  - `CAPRA_TRAINING.md`：CAPRA 训练与配置详解
  - `MULTI_GPU_DEEPSPEED.md`：多卡/DeepSpeed 模板
- `docs/04_evaluation/`
  - `EVALUATION_GUIDE.md`：评测输入输出与 checkpoint 指定
  - `METRICS_EXPLAINED.md`：指标物理意义与日志解释
- `docs/05_troubleshooting/`
  - `TROUBLESHOOTING.md`：OOM/路径/环境常见报错排查

## 执行约束

- 所有新文档 100% 中文。
- 命令示例必须给出完整可复制版本。
- 与代码不一致时，以代码为准并回写文档。

## 快速核对命令

```bash
conda activate openvla-oft
cd /path/to/openvla-oft

# 查看当前 docs 目录树
find docs -maxdepth 2 -type f | sort
```
