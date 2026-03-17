# CAPRA Supervision 与主训练流对齐说明

## 1. supervision JSONL 的真实角色

CAPRA supervision JSONL 在当前实现中的角色是：
- 附加 sparse supervision（仅用于 CAPRA 附加损失）
- 不是主训练数据流
- 不能替代 RLDS/LIBERO 主 batch 的 task_loss 计算

主损失来源：
- `task_loss` 来自 RLDS 主数据流 batch（上游 forward）

附加损失来源：
- `capra_loss` 仅在主 batch 样本命中 supervision 索引时激活

---

## 2. 当前 supervision schema（v2）

文件：`experiments/robot/capra/io/supervision_io.py`

每条记录至少包含：
- `sample_key`: 稳定样本键（训练时主查表键）
- `lookup_key`: 可用于快速查表的键（当前等于 sample_key）
- `instruction`
- `safer_action`
- `weight`
- `align`: 对齐元信息（dataset_name, instruction_norm, episode_idx, step_idx, frame_fingerprint, source_uid）
- `provenance`: 记录该 supervision 的生成来源与配置
- `candidate_stats`: 可选候选统计信息
- `metadata`: 兼容字段
- `schema_version`

向后兼容：
- 旧记录读入时会自动升级为 v2（补齐键与对齐字段）

---

## 3. 稳定样本标识与严格对齐

### 3.1 训练时使用什么字段唯一标识样本？

主键：`sample_key`

构造方式：
- 基于 `dataset_name + instruction_norm + episode_idx + step_idx + frame_fingerprint + source_uid` 的稳定哈希。

实现函数：
- `build_stable_sample_key(...)`

### 3.2 如何与 LIBERO/RLDS 主数据流严格对齐？

训练入口：`vla-scripts/finetune_capra.py`

流程：
1. `CapraRLDSBatchTransform` 在每个 RLDS 样本上生成 `sample_key` 和 `align_meta`。
2. `CapraPaddedCollator` 将批内 `sample_keys` 一并输出。
3. `collate_training_targets(batch, supervision_index, ...)` 按 `sample_key` 查表命中 supervision。
4. 仅对命中样本计算 `capra_loss`。

说明：
- 当前主路径是“严格 key 匹配”，不是 instruction 文本模糊匹配。

### 3.3 episode/step、dataset key、frame id 的关系

当前键体系采用组合键：
- `dataset_name`: 数据域隔离
- `step_idx`: 时间步定位
- `frame_fingerprint`: 帧级稳定指纹（提高唯一性）
- `episode_idx`/`source_uid`: 若可获得则参与键构建

在缺少某些字段时仍能构键，但严格性会下降；建议 miner 与 trainer 都写入完整 align 字段。

---

## 4. 命中策略与未命中策略

文件：`experiments/robot/capra/core/training_targets.py`

### 4.1 一个 batch 命中多条 supervision 怎么处理？

- 同一 `sample_key` 对应多条记录时，默认策略 `max_weight`：选权重最大的条目。
- 可选策略：`first`、`mean`。

### 4.2 没命中 supervision 怎么处理？

- 直接跳过，不触发 CAPRA loss。
- `num_hits = 0` 时 `capra_loss = 0`，训练只走 `task_loss`。

### 4.3 缺失键/坏记录怎么处理？

- 索引构建时跳过无效记录（如缺少 `safer_action` 或形状非法）。
- 不中断主训练。

---

## 5. 训练时 CAPRA loss 激活条件

激活条件：
- 主 batch 内样本的 `sample_key` 在 supervision 索引中命中。

非激活条件：
- 未命中 supervision。

总损失：
- `total_loss = task_loss + lambda_capra * capra_loss`

其中：
- `task_loss`：主数据流真实任务损失
- `capra_loss`：命中 supervision 样本上的附加 safer 约束

---

## 6. 相关代码位置

- schema 与升级：`experiments/robot/capra/io/supervision_io.py`
- 查表与批内对齐：`experiments/robot/capra/core/training_targets.py`
- 训练入口接线：`vla-scripts/finetune_capra.py`
- 对齐测试：`tests/capra/test_supervision_alignment.py`
