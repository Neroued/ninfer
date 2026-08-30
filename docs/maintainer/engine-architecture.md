# NInfer Engine 架构

本文定义 NInfer 从产品请求到模型执行与结果发布的顶层控制面。它是请求生命周期、调度顺序和跨模块
提交关系的维护者权威。

本文只规定长期稳定的边界：

- 谁拥有请求、顺序、缓存策略和物理状态；
- 请求在哪些稳定状态之间转换；
- 模型状态、输出状态和资源状态何时可以对外发布；
- cancellation 和 failure 如何到达唯一终态。

资源选择和上下文缓存策略由
[资源调度与上下文缓存](resource-scheduling-and-context-cache.md)定义；KV 的页、replica、block table
和 consumer contract 由 [Paged KV Context Store](paged-kv-cache.md)定义。

---

## 1. 产品执行模型

Generation purpose 的 NInfer Engine 固定运行：

- 一张 GPU；
- 一个常驻模型实例；
- 启动时确定的 `max_concurrency=1..8`；
- 一个有界 FIFO 等待队列；
- 不抢占已经激活的请求；
- 每个 decode round 将全部 decode-ready 请求组成一个紧凑批次。

Text、Vision、prefix reuse、MTP、DFlash、CLI 和 HTTP serving 都通过公共 `ninfer::Engine` 路径。
MTP 和 DFlash 是 Program 内部的执行后端，不产生第二套请求调度或结果发布机制。

同一个公共 Engine 还提供启动时固定的 CausalScoring purpose。它只服务离线文本评分：
`CausalScoreCore` 串行调用 Program，窗口使用临时的空 State 与 Main KV，不创建请求、continuation、
checkpoint 或 cache replica，也不进入 Scheduler/ResourceManager。Generation 与 CausalScoring
不在运行期切换，评分专用 staging 只在 CausalScoring 启动时分配。

`max_concurrency` 限制同时激活的请求数，不把共享 KV 容量平均切分给 lane。请求只有在 Program
证明其完整执行资源已经得到保障后才会进入 Active；进入 Active 后，它不会因为另一个请求或
inactive cache 的保留而丢失完成能力。

本文不覆盖多 GPU placement、active request preemption、priority/QoS、跨 Engine context store
或大规模 continuous batching。这些工作负载需要重新定义 admission 与公平性合同，不能直接从当前
小并发模型外推。

---

## 2. 四个执行边界

请求路径只有四层：

```text
Gateway
  │ product/protocol input
  ▼
Frontend
  │ PreparedPrompt / OutputSession
  ▼
Engine
  │ request order / lifecycle / publication
  ▼
Program
    physical resources / model execution
```

### 2.1 Gateway

Gateway 是 CLI、HTTP server 或其他产品入口，拥有：

- 协议解析、连接和 streaming transport；
- message、tool、media 到公共输入的转换；
- URL、path 和 data acquisition；
- response schema、usage 和产品侧错误；
- Gateway 自身的并发与连接生命周期。

Gateway 不选择请求顺序、缓存来源、active lane 或 KV page。

### 2.2 Frontend

Frontend 拥有模型家族的输入与输出语义：

- tokenizer、chat template、Vision preprocessing 和 MRoPE prompt construction；
- owning `PreparedPrompt` 及其内容 identity；
- stop、thinking/content channel、detokenization、最终文本和模型私有结构化输出；
- 每个请求独占的 `OutputSession`。

Frontend 可以预览一次模型输出将产生的语义效果，但只有 Engine 完成提交后才能发布该效果。
Frontend 不拥有等待队列、cache catalog 或物理模型状态。

### 2.3 Engine

Engine 是请求控制平面，拥有：

- outstanding capacity、FIFO queue 和 request deadline；
- request record、active slots、cancellation 和 response event；
- Scheduler 与 ResourceManager；
- admission、prefill、decode、control、capture 和 terminal 的编排；
- 模型提交、输出提交和 response publication 的顺序；
- Engine-wide failure cleanup。

Engine 理解请求、预算、finish reason 和可发布输出，不解释 transformer layer、KV plane 或 allocator。

### 2.4 Program

Program 是 exact target package 的唯一物理执行入口，拥有：

- active sequence 与完整 continuation state；
- State/KV stores、allocator、replica、reference 和 reservation；
- prefill、ordinary decode、MTP/DFlash 和 forced control；
- provisional model state 及 accepted-prefix commit/rollback；
- resource feasibility、物理 transition 和 `ResourceResult`；
- workspace、CUDA Graph 和 target execution schedule；
- CausalScoring 窗口的临时 State/KV 与 `lm_head`/logprob staging。

Program 不维护 FIFO、SessionIndex、cache retention 价值或用户可见输出。

---

## 3. 唯一所有权

| 事实或决策 | 唯一所有者 |
|---|---|
| 协议、连接、transport | Gateway |
| prompt 与 output 语义 | Frontend |
| waiting queue、request record、response event | EngineCore |
| FIFO head、backfill、prefill/decode 顺序、round membership | Scheduler |
| logical lane、cache catalog、session binding、retention policy | ResourceManager |
| physical State/KV、reservation、placement、model state | Program |

其他组件可以读取 owner 发布的稳定 summary，但不能复制一份可独立修改的同类状态。

### 3.1 Scheduler：只决定“谁在何时运行”

Scheduler 维护：

- 当前 FIFO head；
- staged-prefill owner；
- admission、prefill 与 decode 的公平门；
- 每轮紧凑执行成员；
- blocked head 的 backfill protection。

Scheduler 不读取 checkpoint payload、victim list、resource vector 或 allocator 状态。

### 3.2 ResourceManager：只决定“保留什么逻辑上下文”

ResourceManager 维护：

- logical lane 状态；
- private/shared checkpoint catalog；
- SessionIndex 与 prefix candidate index；
- retention class、命中观测和逻辑 claim；
- admission、capture 与 finish 的缓存策略。

它把请求和缓存候选交给 Program 评估，并采用 Program 返回的完整物理结果。它不维护一份
Device/Host bytes、page refcount 或 allocator free space 的镜像。

### 3.3 Program：只决定“物理上能否执行以及如何执行”

Program 中的真实 stores 与 allocators 是物理事实的唯一权威。Program：

- 从完整候选终态计算占用、共享、回收和阶段峰值；
- seal 与当前 `resource_revision` 绑定的 opaque `ResourcePlan`；
- 执行唯一的物理 transition；
- 返回足以让 ResourceManager 更新逻辑 catalog 的完整结果。

Program 不根据 session、FIFO 位置或用户身份决定缓存价值。

### 3.4 Package 与 Engine lifetime

Engine 构造时读取 `.ninfer` identity，并从 closed registry 选择 exact compile-time package。Package 提供
同一组 Frontend、request-plan、Program 和 execution-result 语义；target identity、artifact binding、模型
view 与 execution leaves 保持 package-private。Qwen3.6 family 的共享 schedule 通过 compile-time Variant
实例化，worker hot path 不执行 runtime family selection。

权重、State/KV backing、block-table matrices、workspace 与 CUDA Graph resources 在 Engine 开始接受请求前
建立。运行期改变 ownership、mapping、frontier 与 replica placement，但不重建这些大块 Device allocations。

---

## 4. 请求与资源生命周期

### 4.1 Request 状态

请求沿以下稳定状态前进：

```text
Waiting
  -> Materializing
  -> Prefill
  -> DecodeReady | ControlReady
  -> TerminalPending
  -> Finished
```

- **Waiting**：只有 owning prompt 和输出状态，没有 Program sequence。
- **Materializing**：资源 transition 已取得逻辑 claim；source 在 commit 或 abort 前仍有效。
- **Prefill**：请求已经 Active，正在消费 prompt suffix。
- **DecodeReady**：可加入下一次紧凑 decode round。
- **ControlReady**：Frontend 要求提交规范的 forced-control suffix。
- **TerminalPending**：模型执行已经终止，但 active resources 尚未完成 retain 或 release。
- **Finished**：资源与输出均已提交，response 可以完成。

### 4.2 Logical lane 状态

ResourceManager 的 lane 状态是：

```text
Free -> Materializing -> Active -> TerminalPending -> Free
```

请求只有在 materialization 的物理结果和逻辑结果均成功采用后才能进入 Active。Terminal request
必须保留 active ownership，直到完整 checkpoint 被发布或全部 active resources 被释放。
Streaming request 在 admission 选择提交后、任何输出 delta 前发布一次 `GenerationStart`；其中的
prompt token 和 reused-prefix token 是已提交的资源选择事实，不等待 prefill 完成。

Control lane、StateImage slot、KV execution row 和 decode batch row 是不同身份：

- lane 是 Engine 的长期 active request 位置；
- State/KV execution resource 由 Program 分配；
- compact row 只是当前 GPU unit 的序号。

它们不得互相推导所有权。

### 4.3 Gateway 与 Engine capacity

Gateway 可以在 prepare 或 media acquisition 前取得自己的 request lifetime。Engine outstanding capacity
从非零输出请求成功 submit 开始，由两个独立事实共同释放：

```text
response_done       worker 已形成最终 result 或 error
consumer_released   wait 已结束，或 GenerationHandle 被放弃

release_capacity =
    response_done && consumer_released && !capacity_released
```

两者可以任意先后，但 capacity 只释放一次。请求句柄被放弃只设置 cancellation 和
`consumer_released`，不会从 consumer thread 调用 Program。

### 4.4 Continuation 与 session

Active continuation 是可写的模型状态；published checkpoint 是不可变的可复用状态。一个可复用
checkpoint 必须证明同一 frontier 上的完整 State、Main KV、selected backend KV 和继续执行所需
metadata。

Session key 只是查找提示，不拥有 continuation。每个请求进入 Engine 时取得单调的
`publication_order`；只有更新顺序更晚的完成结果才能替换 SessionIndex binding。

---

## 5. Worker 与调度

### 5.1 单一 mutation owner

只有 Engine worker 修改 request record、Scheduler、ResourceManager 和 Program。Ingress、consumer
和 transport thread 通过队列、cancellation flag 与 response event 交互，不直接改变模型状态。

一个 worker boundary 按语义顺序处理：

1. 接收等待请求的 timeout 或 cancellation；
2. 推进已经开始的 resource transition；
3. 完成可以结算的 TerminalPending 请求；
4. 冻结 active cancellation snapshot；
5. 在公平门允许时尝试一次 admission；
6. 选择 staged prefill、forced control 或紧凑 decode；
7. 运行至多一个模型 execution unit；
8. 提交模型、预算和 Frontend 输出状态；
9. 发布 response event 与观测。

具体循环拆分可以变化，但以下顺序不能变化：

- in-flight GPU unit 必须先到稳定边界，再修改其资源映射；
- Program commit 必须先于用户输出发布；
- resource result 必须先被采用，lane 才能改变可见状态；
- 一个 global resource topology transition 未结算时，不启动另一个 transition。

### 5.2 Admission 顺序

Scheduler 先确定唯一可尝试的 waiting request，ResourceManager 再为它选择缓存与资源终态。
资源条件不能反向改变 FIFO 所有权。

FIFO head 暂时受 active incumbents 阻塞时，Scheduler 记录 protected head 和必须结束的 donor set。
后续请求只有在 Program 证明以下条件时才能 backfill：

> borrower 持续占用其完整 active reservation 后，既定 donor set 结束仍足以让 head 以
> root 且释放全部 inactive cache 的方案进入。

这个证明不使用“borrower 预计先完成”的时间假设。改变全局资源拓扑的 transition 会推进
`resource_revision`，之后的 backfill 必须重新证明。

### 5.3 Prefill 与 decode

Scheduler 保证：

- 同时最多一个 staged-prefill request；
- 已有 decode work 不会被连续 prefill 饿死；
- decode round 包含所有且仅包含当前 decode-ready requests；
- batch 使用精确 `B`，不以 inactive lane padding 到 `max_concurrency`。

Program 接收紧凑的 `SequenceHandle[B]` 和每行预算。Prefix reuse 只减少 materialization 或 suffix
prefill，不创建另一条调度路径。

### 5.4 Admission invalidation

只有会改变 admission 结论的事实才重新触发检查：

- waiting queue 或 FIFO head 变化；
- lane 释放；
- staged-prefill gate 变化；
- resource transition 到达终态；
- Program 的全局资源 revision 变化。

普通 decode frontier 推进、输出发布和统计更新不扫描 cache catalog，也不重复运行 pressure planner。

---

## 6. 两类提交事务

Engine 跨模块编排两类互不替代的事务。

### 6.1 Resource transition

Resource transition 在以下边界改变全局资源或 ownership：

- waiting request materialization；
- active checkpoint capture；
- terminal retain 或 release；
- inactive checkpoint 的 placement 或删除。

需要 planning、allocation 或 transfer 的路径遵循：

```text
logical choice
  -> Program seals ResourcePlan
  -> RunningTransaction
  -> ResourceResult
  -> ResourceManager adopts result
```

`ResourcePlan` 与 Program 的 `resource_revision` 绑定。Start 前过期的 plan 可以无副作用地重新规划；
start 后不能更换 source、victim 或 stage 顺序。Abort 也必须返回完整终态，使所有 claim、pin 和已提交
的安全降级得到唯一解释。

资源候选、可行性公式、成本模型和有界搜索由
[资源调度与上下文缓存](resource-scheduling-and-context-cache.md)定义。

### 6.2 Model-unit transaction

Prefill finalization、decode 和 control execution 可以产生 move-only `PendingBatch`：

```text
frozen sequence membership
provisional tokens
per-row produced extent
Program-owned provisional state
```

Engine 对每行输出进行 Frontend preview，形成 accepted-prefix decision，再用一次
`Program::commit` 或 `Program::abort_pending` 消费整个 batch。Program 同时提交或回滚该 prefix
对应的 Main/backend KV、recurrent state、RNG 和 speculative state。

非取消行遵循：

```text
1 <= accepted_tokens <= produced_tokens
nonterminal -> accepted_tokens == produced_tokens
terminal    -> accepted_tokens may be a produced prefix
```

取消行使用零 accepted token 并进入 terminal。Engine 不允许逐行遗弃一个仍未消费的
`PendingBatch`。

### 6.3 输出发布顺序

一次模型输出的可见顺序是：

```text
Frontend preview
  -> Program commits accepted model state
  -> terminal resource result, when required
  -> generation budget and scheduler accounting
  -> OutputSession commits preview
  -> publish stream/aggregate event
```

因此 consumer 不会看到尚未提交的 token，也不会看到与 Program frontier 不一致的 continuation。
Forced control 使用同一提交顺序，但 token 由 Frontend 提供，不调用 sampler，也不推进 sampling RNG。

---

## 7. Terminal、cancellation 与 failure

### 7.1 Terminal

成功终止的 Active request 先进入 TerminalPending：

```text
choose retain or release
  -> Program publishes one complete checkpoint or releases the sequence
  -> ResourceManager adopts the terminal result
  -> optional SessionIndex update
  -> lane becomes Free
```

Checkpoint 的逻辑 publication slot 在 activation 时已经保留。若 retention 无法形成完整 continuation，
确定性终态是 release，而不是让请求停留在 TerminalPending。

### 7.2 Cancellation

Cancellation 在 Engine worker 的稳定边界生效：

| 观察到 cancellation 时的状态 | 结果 |
|---|---|
| Waiting | 不创建 Program state，直接结束请求 |
| Materializing | abort resource transition，并采用其完整结果 |
| Active | 当前 GPU unit 稳定后进入 TerminalPending 并 release |
| PendingBatch | 通过 cancelled row decision 提交或整体 abort |
| 已 commit、尚未 adopt | 先 adopt 已提交结果，再执行 terminal 路径 |

Cancellation 不修改 in-flight mapping，也不从未完成的 active state 发布 checkpoint。

### 7.3 Request-local rejection

在 Program mutation 前可以只拒绝当前请求：

- queue timeout、overload 或 waiting cancellation；
- 输入超过公开 context contract；
- prompt 或 generation request 无法表示；
- 即使采用 root source 并释放全部 inactive cache，单请求仍不可行。

### 7.4 Engine-wide failure

以下情况说明共享物理状态已无法安全解释，必须使整个 Engine 失败：

- GPU mutation 后既不能 commit 也不能形成稳定 abort；
- `PendingBatch` membership 或 disposition 不一致；
- handle owner/generation 不匹配；
- resource、checkpoint completeness 或 noexcept adoption invariant 被破坏。

Cleanup 顺序必须先终止 Program 中未决的 resource/model transaction，再释放 active state，最后清空
ResourceManager 与完成所有 request response。内部不变量错误不能降级成 cache miss、等待或重试。

---

## 8. 物理执行的顶层约束

以下约束属于 Engine 架构，但具体实现由 Program 和下层文档定义：

- Program 在启动时建立固定数量的 control/state/table resources；
- growing KV 由共享 paged pools 支持，active request 持有完整增长 reservation；
- 一个 GPU execution unit 内 State/KV mapping 保持稳定；
- CUDA Graph 按合法 exact-`B` topology 建立，request identity 和 page IDs 是稳定输入数据，不是 graph key；
- ordinary decode 不运行 catalog scan、pressure search 或后台 replica scan；
- workspace 是 Program 启动时统一规划的 backing，Vision、Text 和 speculative schedule 按互斥 lifetime
  使用其内部区域。

Serve warmup 使用同一个公共 Engine 执行路径，但其 request-level context cache 固定关闭。Warmup 可以建立
CUDA Graph、library 和 allocator 的运行时状态，结束后不得留下可供外部请求命中的 continuation 或占用
checkpoint catalog。

---

## 9. 核心不变量

1. 请求顺序只由 Scheduler 决定。
2. logical cache policy 只由 ResourceManager 决定。
3. physical occupancy、reservation、feasibility 和 model state 只由 Program 决定。
4. Active publication 前必须具备完整 continuation 和完成 reservation。
5. 一个 global resource topology transition 在任意时刻至多一个。
6. 一个 `PendingBatch` 必须由一次 Program commit 或 abort 完整消费。
7. Program、Frontend 和 budget 均提交后，模型输出才对 consumer 可见。
8. TerminalPending 结束前不得释放或复用该请求的 active ownership。
9. Cancellation 只能在线性化的 worker boundary 改变请求状态。
10. 每个 accepted request、resource transaction 和 model transaction 都有有限且唯一的终态。

---

## 10. 实现位置与相邻权威

| 职责 | 主要位置 |
|---|---|
| 公共 Engine facade | `include/ninfer/engine.h`, `src/runtime/engine/engine.cpp` |
| Engine worker 与 request lifecycle | `src/runtime/engine/engine_core.h`, `src/runtime/engine/request_record.h` |
| Scheduler | `src/runtime/engine/scheduler.h`, `admission_policy.*` |
| ResourceManager 与 materialization planner | `src/runtime/engine/resource_manager.h`, `materialization_planner.h` |
| package-neutral runtime contracts | `src/runtime/contract/types.h` |
| target Program | `src/targets/qwen3_6/impl/runtime/` |
| physical primitives | `src/core/` |
| semantic Ops | `src/ops/`, `include/ninfer/ops/` |
| HTTP Gateway | `src/serve/` |

这些路径用于定位当前 authority，不把文件拆分固化为外部接口。

相邻文档：

- [资源调度与上下文缓存](resource-scheduling-and-context-cache.md)：resource vector、checkpoint
  capability、pressure planning 和 physical transition；
- [Paged KV Context Store](paged-kv-cache.md)：typed pools、logical pages、replicas、block tables
  和 consumer address contract；
- [Op development](op-development.md)：Op 正确性与性能准入；
- [CLI](../cli.md)与 [HTTP serving](../serving.md)：外部行为。
