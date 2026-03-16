你现在在一个“纯净 upstream openvla-oft 仓库”中工作。

你的目标不是重写 OpenVLA-OFT，而是在尽量不破坏 baseline 的前提下，以最小侵入 overlay 方式新增 CAPRA-v1。

## 1. 项目身份

本项目研究的是 manipulation-only VLA 的 intrinsic safety。
我们要解决的问题不是外接 safety shield，不是 memory 主模块，不是 mechanistic steering，也不是 generic DPO / preference alignment。

核心问题是：
当若干局部候选动作对任务推进差不多时，base VLA 为什么还会系统性地偏向更危险的动作？

我们把这个对象定义为：
local avoidable-risk regret

## 2. 当前必须实现的 v1 pipeline

首版只实现下面这条路径：

1. 用 base OpenVLA-OFT 在 simulator 中自然 rollout
2. 在每个时刻围绕 base action chunk 构造一个有限 local proposal set
3. 从同一 snapshot 出发，对每个候选做 short counterfactual rollout
4. 计算 progress features，并用 progress-preserving gate 过滤候选
5. 计算 footprint v1
6. 在 progress-preserving set 中找到更安全候选
7. 挖出 safer supervision records
8. 用 anchor + weighted safer-target regression 训练 CAPRA policy
9. 评测 SPIR / EAR / success / displacement / severe events

## 3. v1 的数学对象（给工程实现用）

对时刻 t 的 base action 记为 a_base。

候选集：
A_t = {a_base} union local proposals around a_base

progress-preserving set：
E_t = { a in A_t | G_t(a) >= G_t(a_base) - epsilon_p }

其中 G_t(a) 不是 full potential，不是 reward，只是 progress-preserving 判断所需的一组特征。

更安全候选：
a_star = argmin_{a in E_t} F_t(a)

局部 regret：
Delta_t = F_t(a_base) - F_t(a_star)

v1 有效 supervision 条件：
- E_t 非空
- Delta_t > delta_min

v1 训练权重：
w_t = clip(Delta_t, 0, w_max)

v1 训练目标：
L = L_anchor + lambda * w_t * L1(pred_action, a_star)

注意：
v1 不实现 full q-star KL projection，不实现 precursor attribution，不实现 DPO。

## 4. progress 与 footprint 的实现边界

### Progress v1
progress 只做“明显有没有变差”的 gate，不做 exact task equivalence oracle。
优先使用以下 features：
- progress_before
- progress_after
- progress_delta
- done_after
- target_dist_before / after（如果能稳定获得）
- grasp_held
- object_settled
- unrecoverable
- gripper_open_fraction

### Footprint v1
footprint 只保留：
- displacement_total
- severe_event_flags
- severe_penalty

默认：
F_t(a) = displacement_total(a) + severe_penalty(a)

不要在 v1 默认实现：
- contact impulse
- exact support-break modeling
- full 3-term footprint

## 5. 候选动作集的工程约束

OpenVLA-OFT 的 LIBERO 默认路径是 continuous action chunk + L1 regression。
因此不要假设存在可直接读取的离散动作 logits 或 policy density。

local proposal set 只能是有限集合，围绕 base action chunk 做局部扰动，例如：
- speed-scaled variants
- small lift variants
- small retract variants
- optional small Gaussian perturbations

保持 proposal 数量小、稳定、可配置。

## 6. 代码集成原则

严格使用 overlay 方式：
- 新逻辑放在 experiments/robot/capra/
- 新训练入口放在 vla-scripts/finetune_capra.py
- 新脚本放在 scripts/capra/
- 新测试放在 tests/capra/

尽量不动：
- experiments/robot/libero/run_libero_eval.py
- experiments/robot/openvla_utils.py
- experiments/robot/robot_utils.py
- vla-scripts/finetune.py
- prismatic/

如果训练 loss 注入无法避免，才允许对上游做最小 hook，并且必须在 progress_capra.md 里说明原因。

## 7. 信号访问规则

项目采用 obs-first, sim-backed。

- 位置/姿态类信号优先用 obs
- 速度、接触、抓取、支撑、topple 等动态信号优先用 sim-backed state_api

必须遵守：
- env_adapter.get_sim() 是唯一合法 sim 入口
- state_api.read_state_signals() 是统一状态读取层
- 业务代码不得直接乱写 env._env.env.sim 之类的内部路径

## 8. 当前明确不做什么

以下内容在 v1 中一律不要实现，除非用户明确要求：
- test-time shield / safety layer / QP / CBF
- memory 主模块
- generic DPO / pairwise preference 主训练线
- exact predicate hook
- full V_prog
- contact impulse
- precursor attribution
- Safety Alternative Buffer
- merged dataset builder
- heavy resume/checkpoint framework
- four procedural splits 的全量生成器
- real-world / ALOHA 主线
- backward-compatibility shims for old CAPRA code

这是一个干净 upstream repo，不要为了不存在的 legacy path 去写兼容层。

## 9. Benchmarks 与指标

utility benchmark：
- standard LIBERO suites（沿用 upstream OpenVLA-OFT protocol）

safety benchmark：
- SafeLIBERO（后续阶段再接）

v1 headline metrics：
- SPIR
- EAR
- success
- non-target displacement / displacement_total
- severe event rate

不要在 v1 里实现：
- EditGain
- LeadTime
- precursor metrics

## 10. 工作方式要求

- 先读代码，再改代码
- 对没打开过的文件，不要做断言
- 只完成当前阶段，不要擅自跨阶段继续
- 保持最小侵入，避免过度工程化
- 不要删除或重写无关代码
- 临时文件若仅用于试验，请在结束前清理
- 不要引入不必要的新抽象、新框架或新配置系统
- 如果 repo 实际结构与预期不一致，先记录到 docs/CAPRA_PLAN.md，再决定怎么接

## 11. 状态管理要求

每完成一个阶段，都必须更新：
- progress_capra.md：自由文本进度、已完成内容、剩余风险、下一步
- tests_capra.json：结构化记录测试状态

tests_capra.json 建议格式：
{
  "tests": [
    {"id": "capra_imports", "status": "not_started"},
    {"id": "env_adapter_roundtrip", "status": "not_started"},
    {"id": "state_api_signals", "status": "not_started"},
    {"id": "local_evaluator_v1", "status": "not_started"},
    {"id": "mining_v1_smoke", "status": "not_started"},
    {"id": "finetune_capra_smoke", "status": "not_started"},
    {"id": "metrics_v1", "status": "not_started"}
  ]
}

## 12. 每轮回复格式

每轮结束时请只给出：
1. 本轮改了哪些文件
2. 本轮跑了哪些测试，结果如何
3. 当前还剩什么风险或阻塞点
4. 我下一条最应该发给你的阶段 prompt 是什么

没有让我确认之前，不要自动进入下一阶段。
