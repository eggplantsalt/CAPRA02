这个文件是关于我们整个idea的定义

# CAPRA-v1 on OpenVLA-OFT

## 1. 项目一句话定义

我们研究的是 **manipulation-only VLA 的内生安全**。  
核心问题不是外接 safety shield，不是 memory 主模块，不是 mechanistic steering，也不是 generic preference alignment。  
我们要回答的问题是：

> 当若干局部候选动作对任务推进差不多时，为什么 base VLA 还会系统性地偏向更危险的那个动作？

我们把这种现象定义为 **local avoidable-risk regret**。

---

## 2. 我们真正要解决的现有问题

现有相关工作大致分两类：

1. **训练期约束优化路线**  
   例如 SafeVLA / ISA 这一类，把安全写进 CMDP / SafeRL 框架里。  
   这条线很重要，但它更像“显式约束下的整体安全优化”，而不是专门回答 manipulation VLA 的局部内生风险排序问题。

2. **测试时外接安全层路线**  
   例如 VLSA / AEGIS 这一类，在 base VLA 后面加 SC layer / safety filter / CBF-QP。  
   这条线能给 test-time safety correction，但它是 **external shield**，不是我们要的 intrinsic safety。

我们的切入点是：

> 不增加 test-time safety layer，单纯通过训练期局部替代动作挖掘与对齐，让模型本身更少偏向高风险动作。

---

## 3. 首版工作的核心 claim（必须收紧）

首版 paper 只 claim 这三件事：

### Claim 1
在 manipulation-only VLA 上，base OpenVLA-OFT 存在可测的 **local avoidable-risk regret**：  
即在局部 **progress-preserving neighborhood** 内，模型经常没有选 footprint 更小的动作。

### Claim 2
我们可以用 simulator-derived counterfactual local evaluation 自动挖出“更安全但不明显更差”的局部替代动作，而不依赖人工 safety label，也不增加 test-time 模块。

### Claim 3
把这些 safer local alternatives 蒸馏回 OpenVLA-OFT 后，可以在不显著伤害标准 LIBERO utility 的前提下，降低：
- SPIR
- EAR
- SafeLIBERO 上的外部安全风险
- delayed side-effect splits 上的长尾风险

---

## 4. 首版明确不做什么

首版 **不做**：

- 不做 attack / defense
- 不做 navigation + manipulation 混合主叙事
- 不做 memory 主模块
- 不做 mechanistic steering 主贡献
- 不做 generic DPO / pairwise preference 主方法
- 不做 test-time shield / QP / CBF / safety filter
- 不做 full CAPRA precursor attribution 主线
- 不做 exact predicate-based full progress function
- 不做 full 3-term footprint（尤其不把 contact impulse 当默认核心）
- 不做大规模新 benchmark
- 不做真机主结果
- 不 claim hard safety guarantee

---

## 5. 诚实边界（必须写清楚）

这篇工作的边界必须非常诚实：

1. **我们定义的是 progress-preserving neighborhood，不是 exact task equivalence。**
2. **我们只搜索有限候选集，因此测到的 avoidable risk 是真实 avoidable risk 的一个下界近似。**
3. **footprint 是 simulator proxy，不是“真实世界一切风险”的完美刻画。**
4. **我们不提供 hard safety guarantees；我们提供的是 measurable intrinsic risk reduction。**
5. **首版只针对 manipulation-only tabletop simulation。**

---

## 6. 首版最终方法：CAPRA-v1（implementable version）

科学总框架仍然叫 CAPRA。  
但首版工程目标不是 full CAPRA，而是一个可实现的 **CAPRA-v1**。

### 6.1 数据来源

先让 base OpenVLA-OFT 在 simulation 中自然 rollout。  
不使用人工 safety label。  
所有 supervision 来自 simulator / state signals / short counterfactual rollout。

每条 rollout 至少保存：
- observation
- language instruction
- base predicted action chunk
- environment snapshot
- selected state signals
- success / done
- object displacement related statistics
- severe events related statistics

---

### 6.2 候选动作集（关键：不要假设 policy density）

OpenVLA-OFT 是 continuous action chunk regression，不要假设它有可采样的解析动作分布。

所以在时刻 t，候选集定义为：

\[
\mathcal A_t = \{a_t^{base}\} \cup \mathcal P(a_t^{base})
\]

其中：
- \(a_t^{base}\) 是当前 base policy 输出
- \(\mathcal P(a_t^{base})\) 是围绕 base action chunk 的一个 **有限局部 proposal set**

首版 proposal set 只做小而稳定的局部扰动，例如：
- speed-scaled variants
- small lift variants
- small retract / jitter variants
- optional small Gaussian local perturbations

原则：
- 候选必须离 base action 足够近
- 不假设离散 logits
- 不假设 sample-K from policy density
- 不依赖 Safety Alternative Buffer

Safety Alternative Buffer 在首版中不是必须项，直接 deferred。

---

### 6.3 局部反事实评估：只做短视界

从同一 snapshot 出发，对每个候选动作做短视界 counterfactual rollout。  
首版只做 **short-horizon local evaluation**，不做长窗口 precursor replacement。

对每个候选动作 \(a\)，计算两个量：

#### (1) Progress signal
记为 \(G_t(a)\)

它不是 full potential，也不是 exact predicate count。  
它只是一个 **progress-preserving gate**，用来排除“明显把任务搞坏”的候选动作。

首版允许使用的 progress features：
- done / success after rollout
- progress delta
- target distance change（若可稳定获取）
- grasp held
- object settled
- unrecoverable
- gripper open fraction
- 其他 obs-first, sim-backed 的弱进度特征

然后定义：
\[
\mathcal E_t = \{a \in \mathcal A_t \mid G_t(a) \ge G_t(a_t^{base}) - \epsilon_p\}
\]

注意：
\(\mathcal E_t\) 是 **progress-preserving neighborhood**，不是 exact task-equivalent set。

#### (2) Footprint v1
记为 \(F_t(a)\)

首版只保留：
- `displacement_total`
- `severe_event_flags`
- `severe_penalty`

总足迹：
\[
F_t(a) = displacement\_total(a) + severe\_penalty(a)
\]

首版默认 **不启用 contact impulse 主项**。  
support-break 如果实现不稳，也不作为 headline metric，只保留为可选 severe flag。

---

### 6.4 局部 avoidable risk 定义

如果 \(\mathcal E_t\) 非空，则定义：

\[
a_t^\star = \arg\min_{a \in \mathcal E_t} F_t(a)
\]

\[
\Delta_t = F_t(a_t^{base}) - F_t(a_t^\star)
\]

其中：
- \(a_t^\star\) 是局部 progress-preserving 候选里更安全的动作
- \(\Delta_t\) 表示当前 base action 多承担了多少本来可以避免的局部风险

当：
- \(\mathcal E_t \neq \emptyset\)
- \(\Delta_t > \delta\)

时，我们就把这个时刻记为一个有效的 mined supervision 点。

---

### 6.5 训练目标：首版用 safer-candidate relabeling，不做 full KL projection

理论上，CAPRA 可以写成一个受约束局部风险投影。  
但首版实现不追求 full candidate-distribution KL。

首版直接做：

- **anchor loss**：保持原始任务能力
- **safer-candidate regression loss**：把模型往 \(a_t^\star\) 拉

训练目标写成：

\[
L = L_{anchor} + \lambda \sum_t w_t \, \|\pi_\theta(o_t, l_t) - a_t^\star\|_1
\]

其中：
\[
w_t = \mathrm{clip}(\Delta_t, 0, w_{max})
\]

也可以加激活条件：
- 只有当 \(\mathcal E_t\) 非空
- 且 \(\Delta_t > \delta\)
- 且 candidate 没有明显毁掉 progress

才激活 CAPRA loss。

首版不要强行实现：
- full \(q^\star\) candidate distribution projection
- policy density estimation
- pairwise ranking / DPO

这些都 deferred。

---

## 7. 工程接口原则

首版工程上只有一个真正核心的“source of truth”：

### Local Candidate Evaluator

它负责：
- snapshot save / restore
- short counterfactual rollout
- progress-preserving gate
- footprint v1
- candidate summary

同时，状态读取必须遵守：

- `env.get_sim()` 作为唯一合法 sim 入口
- `state_api.read_state_signals()` 作为统一状态读取层
- 业务模块不得到处直接写 simulator 内部路径

首版采用：
- **obs-first, sim-backed**
而不是 obs-only。

原因很简单：
仅靠 obs-only，很多关键动态信号不可靠，例如：
- object velocity
- topple detection
- contact-based support break
- grasp held

---

## 8. Benchmarks：首版只保留最必要的三层

### 8.1 主 utility benchmark
沿用 OpenVLA-OFT 官方标准 LIBERO protocol：
- LIBERO-Spatial
- LIBERO-Object
- LIBERO-Goal
- LIBERO-10 / LIBERO-Long subset

目的：
证明 CAPRA-v1 不明显破坏原本任务能力。

### 8.2 主 safety benchmark
使用 SafeLIBERO。

目的：
证明在一个现成的 safety-critical tabletop benchmark 上，
training-time intrinsic alignment 就能降低安全风险，
而不依赖 external shield。

### 8.3 额外 delayed-side-effect benchmark（首版只补两个）
不做 4 个全上，首版先只做两个最值钱的：

1. **Support-Critical-Neighbor**
   - 用来测试 fragile support / topple / edge instability

2. **Chain-Reaction**
   - 用来测试局部扰动引发级联副作用

原因：
这两个最直接支撑“delayed hazard / side effect”主叙事，  
而且相比 4 个全做，工程成本更可控。

可选 appendix / future：
- CollateralClutter
- OccludedRememberedHazard

---

## 9. 评测指标：首版只保留最硬的一组

### 主指标
- **SPIR**：Safety Preference Inversion Rate
- **EAR**：Expected Avoidable Risk

### utility 指标
- success rate on standard LIBERO

### 外部安全一致性指标
- non-target displacement total
- topple / severe event rate
- latency / throughput（若需要）

首版不要把这些放进主结果：
- precursor LeadTime
- Attribution EditGain
- support-break exact metric
- 一堆复杂机制指标

这些都容易把 paper 拉散。

---

## 10. 和现有工作的差异（给审稿人的一句话）

- 和 **SafeVLA / ISA** 不同：  
  我们不是 CMDP/SafeRL 的整体约束优化主线，也不是 navigation+manipulation 的 integrated setting；我们研究的是 manipulation-only 下的 **local avoidable-risk regret**。

- 和 **VLSA / AEGIS** 不同：  
  我们不加 test-time SC layer，不做 QP safety correction，不 claim hard safety guarantee；我们做的是 **training-time intrinsic alignment**。

- 和 memory / steering / preference 线不同：  
  我们不把 memory、mechanistic steering、generic DPO 当主方法；它们都不是首版贡献核心。

---

## 11. 首版最重要的 3 个实验问题

### Q1. Existence
base OpenVLA-OFT 是否真的存在 measurable local avoidable-risk regret？

### Q2. Effectiveness
CAPRA-v1 是否能在不明显伤害 standard LIBERO utility 的前提下，降低 SPIR / EAR / SafeLIBERO safety risk？

### Q3. Delayed side effects
这种局部风险对齐是否也能减少 support-critical / chain-reaction 这类 delayed side effects 的长尾风险？

---

## 12. 首版最重要的 3 个诚实 limitation

1. 我们只在有限局部 proposal set 中搜索 safer alternative。
2. progress-preserving gate 只是近似，不是 exact task equivalence oracle。
3. 我们不提供 hard safety guarantee，只报告 measurable risk reduction。

---

## 13. 实现优先级（必须严格按阶段走）

### Phase 0
repo 脚手架 + baseline 路径确认

### Phase 1
env adapter / state_api / snapshot round-trip

### Phase 2
local candidate evaluator
- short rollout
- progress gate
- footprint v1
- candidate summary

### Phase 3
mining pipeline
- 从 base rollouts 挖 `(obs, instruction, safer_action, weight)` supervision

### Phase 4
finetune integration
- anchor loss
- weighted safer-candidate regression

### Phase 5
evaluation
- standard LIBERO utility
- SafeLIBERO
- Support-Critical-Neighbor
- Chain-Reaction

---

## 14. 明确 deferred 的内容（不要让 code agent 提前做）

以下内容全部 deferred，除非主路径已经稳定：

- full V_prog / exact predicate hook
- contact impulse
- exact support-break modeling
- Safety Alternative Buffer
- merged dataset builder 复杂版本
- precursor attribution
- EditGain / LeadTime
- 4 个 procedural templates 全量版
- OOD 扰动大包
- 真机实验
- full candidate-distribution KL projection

---

## 15. 最后一句话（整个项目的真正主线）

这个项目的核心不是“再加一个安全层”，而是：

> **在不增加 test-time shield 的前提下，让 manipulation VLA 少犯那些本来可以用更安全局部动作避免的内生风险排序错误。**

首版只需要把这件事做扎实，不需要把 full CAPRA 的所有理想模块一次性实现完。
