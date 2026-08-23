# NInfer 小规模并发推理架构

本文定义 NInfer 在单 GPU、单模型实例下支持少量并发请求的执行架构。典型
`max_concurrency` 为 2–8。

设计目标不是让多个请求轮流执行，而是让所有处于 decode 阶段的请求形成一次真正的 batched
model execution：一次 model traversal、一次 CUDA Graph replay 和一组 batched operators 同时为
多个请求产生结果。

本文只定义会影响模块边界、调度语义、资源所有权或执行正确性的决策。协议错误格式、allocator
实现、kernel 组织、target-specific profile 数值和测试计划分别属于对应的 serving、runtime、operator
和 qualification 文档。

---

## 1. Scope

### 1.1 Supported workload

- 单 GPU、单 resident model instance；
- 启动时固定 `max_concurrency=C`，典型 `C=2..8`；
- 运行时 `0..C` 个 admitted requests；
- Text 与 image/video prompt；
- ordinary decoding 与 engine-wide speculative decoding；
- streaming 与 non-streaming output；
- prefix reuse；
- 不同 prompt length、context length、generation limit 和 sampling configuration。

### 1.2 Non-goals

- request preemption、swap 或 pause/resume；
- 多请求 batched prefill 或 prefill/decode mixed forward；
- 多 GPU 或 distributed inference；
- priority、tenant QoS 或 deadline-aware GPU scheduling；
- 面向数十至数百请求的通用 continuous batching；
- serving 期间为新 shape 动态捕获 CUDA Graph。

### 1.3 Required sequence-state substrate

并发引擎要求下层提供 concurrent-capable sequence-state substrate，并以该 contract 已满足为前提。
该 substrate 必须提供：

- 不依赖 slot 或 per-request 物理连续区间的 shared, block-addressable growing-context storage；
- 按 sequence 分配的 fixed-size recurrent/backend state；
- 对 growing 和 fixed state 的统一 reserve、commit、truncate 和 release 语义；
- retained state 的 pin、ownership transfer 和 eviction 语义；
- 可由 attention 及其他 model operators 直接消费的 opaque per-sequence state view；
- 在一个 GPU execution unit 期间稳定的 state handles 和逻辑 valid frontier。

本文定义并发层如何预留、持有、组批和转移这些 state，不定义 substrate 的 page
geometry、物理 slab layout、allocator、block-table representation 或 page-aware operator implementation。
Growing KV substrate 的 active contract 由
[Paged KV Context Store](paged-kv-cache.md)定义；本文只消费其 allocation、reservation 和 execution-view
语义。

---

## 2. Core invariants

### 2.1 Maximal batched decode

在任一 round boundary，若有 `B>0` 个 decode-ready requests，下一次 decode round 必须包含全部
`B` 个请求。禁止把它们拆成多个 request-local forwards，也禁止为了复用单请求路径而依次执行。

### 2.2 Boundary-only membership

请求只在 GPU round 或 prefill chunk 完成后的 boundary 加入或退出 active execution。GPU work
in-flight 期间，slot binding、batch membership 和 sequence-state ownership 不变。

### 2.3 Fixed slots, shared state memory

`max_concurrency` 固定 control slots 的数量，不把 KV/context memory 平均分给 slots。KV cache、GDN
state 和 speculative-backend state 由 §1.3 的 Sequence-State Store 提供。Slot 只持有 allocation
handle 和 reservation，不拥有静态 context partition。

Engine configuration 必须能同时容纳 `C` 份 active fixed-size sequence state、`C` 行 execution
metadata 和一份 shared executor workspace；否则该 `max_concurrency` 无效。Retained state 只留在
当前空闲 lane 的 fixed backing 并占用其实际 mapped growing state；new admission 可以 claim 或驱逐它，
因此不能降低 `C` 个 active sequences 的 guarantee。Growing context 仍是 shared capacity，所以不承诺
任意 `C` 个长上下文都能同时 admission。

### 2.4 Admission guarantees completion capacity

NInfer 不支持 preemption，因此 request 只有在其 prompt、声明的最大生成长度和必要的临时增长都能
获得完整资源承诺后才可 admission。已经 admitted 的 request 不会因为后来请求到达而被截断或逐出。

### 2.5 One prefill owner

同一时刻最多有一个 admitted request 拥有 prefill/finalization path。Suffix prefill 以 bounded chunk 为
单位，在 decode rounds 之间运行。其他等待请求仍留在 host queue，不占 slot 或 model state。

### 2.6 Single GPU execution owner

一个 GPU executor 串行提交所有 prefill chunks 和 decode rounds，并且是 active slot、sequence state
和下一轮 batch 的唯一修改者。CPU preparation、request ingress 和 output I/O 可以并行，但不能直接
推进模型状态。

### 2.7 Bounded ingress and output

Engine 与 HTTP server 都把 generation request lifetime 数量限制为
`max_concurrency + max_pending_requests`。每个 request 还有有限输入大小、有限 effective output token
bound 和 absolute pending deadline；media preparation 使用固定 worker 数、有界 FIFO 和 live-byte
account。因此持续 ingress 只能填满有限 request records，不能建立无界等待队列。

Owning result 的总上界来自“有限 request 数 × 每请求有限输出”；result/event storage 属于各自 request，
并持续计入 request lifetime capacity，直到 response consumer 释放它。

### 2.8 Per-request semantic isolation

Sampling、RNG、stop conditions、generation limit、usage 和 output state 属于 request。它们不得依赖
request 当前位于哪个 slot 或 compact batch row，也不得影响同一 batch 的其他 requests。

### 2.9 Protected, resource-aware admission

Prepared request order 是 admission 的公平基线，但暂时无法取得完整资源承诺的 FIFO head 不得无条件
封锁后续所有请求。Scheduler 可以用后续 request 回填当前未承诺的 lane/KV capacity，但必须先为最老的
blocked head 建立唯一 protection epoch；later admission 不能破坏该 head 在 frozen incumbent releases 后的
逐 pool 资源可行性，也不能通过持续 ingress 无限延长 temporal borrowing。

Backfill 只改变 waiting request 的 admission order。它不取得 active request 已承诺但尚未 materialize 的
资源，不产生 partial admission，不抢占已经 admitted 的 request，也不建立第二套 decode priority。所有
backfilled decode-ready requests 仍进入同一个 maximal compact batch。

---

## 3. Overall architecture

```text
 clients / CLI / OpenAI / Anthropic
                  │
                  ▼
┌──────────────── Server Frontend ─────────────────────┐
│ validate · bounded CPU preparation · ordered pending │
│ queue · protected-head resource backfill             │
│ finite count/per-request bytes/deadline · responses  │
└───────────────────┬──────────────────────────────────┘
                    │ admission at a boundary
                    ▼
┌──────────────────── GPU Executor ────────────────────┐
│ Boundary Coordinator · Scheduler · Slot Table[C]     │
│                         │                            │
│          ┌──────────────┴──────────────┐             │
│          ▼                             ▼             │
│  PrefillChunk(one sequence)   RoundMembership[B]    │
│          │                             │             │
│          │                    Target Batch Assembler │
│          │                             │             │
│          └───────────┬─────────────────┘             │
│                      ▼                               │
│       shared Model Runtime + DecodeBatchFrame[C]     │
│       weights · workspace · graph families           │
│                      │                               │
│                      ▼                               │
│               per-row round results                  │
└───────────────┬───────────────────┬──────────────────┘
                │                   │ committed output
                ▼                   ▼
┌─ Sequence-State Store prerequisite ─┐  Server Frontend
│ Text KV                            │  / async output
│ GDN / recurrent state             │
│ speculative-backend state         │
│ state entitlements                │
│ optional retained prefixes        │
└──────────────────────────────────────┘
```

各组件的职责如下：

| 组件 | 职责 |
|---|---|
| Server Frontend | 请求校验、有界 CPU preparation/pending work、取消输入和响应 I/O |
| GPU Executor | admission、boundary processing、状态提交和全部 GPU submission |
| Slot Table | 保存 admitted requests 的稳定控制状态 |
| Scheduler | 在 boundary 执行 protected-head admission，并选择下一 `PrefillChunk` 或完整 active `DecodeRound` |
| Target Batch Assembler | 把 round membership 转成 target 所需的 typed controls 和 state selectors |
| Model Runtime | 持有唯一 resident model、共享 execution memory 和 graph assets，执行 whole-batch schedule |
| DecodeBatchFrame | 一份最大容量为 `C` 的地址稳定 round staging；每轮只使用 exact-`B` prefix |
| Sequence-State Store | 前置 storage substrate；拥有 per-sequence model state 和容量 |

这些是责任和所有权边界，不要求采用同名的 C++ class。

所有 product entrypoints 都向同一个 GPU Executor 提交 request；caller 或连接线程不直接驱动 model
loop，也不独占一份 Model Runtime。`C` 个并发 request 共享同一份 weights、workspace、round frame 和
graph families，而不是构造 `C` 个单请求 Program。

Blocking `Engine::generate` 可以保留为 product facade，但其语义是提交一个 owning request record 并等待该
request 完成，而不是在 caller thread 内运行完整 generation loop。一个 Engine-owned GPU worker 执行全局
boundary loop；并发 callers 等待各自的 result/output queue。GPU worker 只追加 committed output events，
不调用 network write 或可能阻塞的 `OutputSink`。

---

## 4. Request and slot model

### 4.1 Request lifecycle

```text
RECEIVED
   │ validation and bounded pending capacity
   ▼
WAITING
   │ bounded CPU preparation, then boundary admission:
   │ slot/lane + state entitlement
   ▼
PREFILL
   │ zero or more suffix chunks, then one finalization unit
   │ exact retained hit skips suffix work, not finalization
   ├── terminal ──────────────────────────────────► MODEL_FINISHED
   ▼
DECODE_READY
   │ participates in successive batched rounds
   ├────────────────────────────────────────────► DECODE_READY
   │ EOS / stop / limit
   ▼
MODEL_FINISHED
   │ response may continue draining
   ▼
RESPONSE_FINISHED
```

Cancellation 或 failure 可以从任意阶段终止请求。`WAITING` 同时包含有界 CPU preparation 和等待 GPU
admission；二者是 server-side phases，不需要扩展成更多 model-execution states。

`DECODE_READY` 是稳定状态：request 已拥有完整 committed model frontier、下一次 decode round
需要的唯一 current token/anchor，且 sampling、RNG 和已发布输出与该 frontier 一致。是否属于
当前 in-flight round 由该 round 记录，不再建模成一个长期 request state。

Intermediate prefill chunks 只推进 prompt state。Finalization unit 执行 target output selection并建立 decode
anchor；如果立即命中 terminal condition，请求不进入 decode batch。Retained current frontier 正好覆盖完整
prompt 时可以跳过 suffix prefill，但仍通过同一 finalization unit 从 retained tail state 产生下一 anchor，并在
MTP 下完成 exact-hit bridge/proposal。命中已保存 rewrite checkpoint 时则从该 checkpoint prefill 新 suffix。
单独的 KV match 不足以进入 `DECODE_READY`。

### 4.2 Slot

slot 是租给一个 admitted request 的稳定控制位置。它保存或引用：

- request identity 和 lifecycle state；
- sampling、RNG、stop 和 generation-limit state；
- per-request output session、usage、timing 和 speculative statistics；
- shared pool 中 `SequenceState` 的 handle；
- scheduler 所需的 prompt/decode progress；
- cancellation 和 terminal flags。

slot 不拥有：

- 固定比例的 context capacity；
- 永久 batch row；
- shared model workspace；
- model completion 后的 response buffers；
- model completion 后 retained `SequenceState` 的 logical ownership。

请求到达 `MODEL_FINISHED` 后，model result 移交 response path，slot 可以在当前或之后的 boundary
复用。Network completion 不属于 slot lifetime。slot index 也不是 external request identity；旧请求的
cancellation 或 completion 不能作用于之后的 occupant。

当前 runtime 让 control slot 与一条 physical execution lane 一一对应，但 slot、external request identity
和 compact batch row 仍是三种不同身份。固定大小的 device state 可以按 lane 预留，并在 lane 被新请求
占用时重建。Request 完成后，active slot 立即释放；若保留 prefix，`SequenceState` 可以继续留在该空闲
lane 的 backing 中，但不再存在 active request control，也不计入 active concurrency。

### 4.3 SequenceState

`SequenceState` 是 target 定义的一条可继续执行的 model state。一个 occupied slot 对它拥有唯一写权限，
它至少包含或引用：

- Main/backend KV allocations 及 committed frontiers；
- Linear Attention 和其他 fixed model-state allocation；
- target decode cursor，包括 current anchor 和 position/RoPE progress；
- continuation 所需的 hidden/checkpoint state；
- prefix identity 和 target-defined reusable checkpoints。

`SequenceState` 不包含 stop/output/transport state，不拥有 batch row、round activations、logits、shared
workspace 或 graph。当前 fixed-state backing 是 lane-affine 的：retained state 不搬到另一 lane，planner 在
所有 free lanes 中寻找可复用 continuation，选中后在同一 lane 建立新的 request control；需要 active
capacity 时可以先驱逐其他 free lanes 上的 retained state。新 request 的 sampling、RNG、stop 和 output
state 始终重新创建。

Qwen3.6 的 lane 是 Linear Attention state 的唯一 locator。`C=max_concurrency` 时，shared pool 固定使用
`[0,C)` 作为各 lane 的 current committed state，使用 `[C,2C)` 作为各 lane 的 rewrite-checkpoint
state；一份 slot 同时选择全部 GDN layers 的 convolution history 和 recurrent state。Decode round
不在 `SequenceState` 中维护随 speculative position 变化的 state selector。

### 4.4 Batch row

batch row 只存在于一个 decode round。每个 boundary，Target Batch Assembler 都重新建立 compact mapping：

```text
batch row -> slot -> sequence-state handle
```

例如：

```text
slots:          [0] [1] [2] [3]
decode-ready:    Y   N   Y   Y

logical rows:   [0] [1] [2]
row -> slot:     0   2   3
logical B:       3
```

empty slot 不进入 model batch，因此 slot reuse 或 hole 不需要不同的 execution model。

---

## 5. Request ingress and admission

### 5.1 Bounded ingress

在 expensive preparation 之前，Server Frontend 校验 request syntax、model capability、input size 和
有限 generation bound。通过校验的请求才能取得有界 CPU preparation 和 pending capacity。

host side 对以下资源设置有限上限：

- generation-request lifetime 数量，固定为 `C + max_pending_requests`；
- 每请求 body/media bytes、raw patches 和 Vision tokens；media item 没有独立固定上限；
- 同时占用 request records 的 CPU preparation work；
- cache-retained bytes 与所有 cache/request/runtime 引用共同计费的 live BF16 payload bytes；
- 由 `live capacity / 单请求最大 payload` 推导的 CPU preparation permits，避免多个请求各自
  部分占用后相互等待；permit 在 CPU preparation 结束时释放，不延长到 GPU request lifetime；
- request 从取得 lifetime capacity 到 admission 的最长时间；
- 每请求 effective output token bound。

request lifetime count 已满时，新 generation request 立即以 overload 拒绝。不存在额外的无界 overflow
queue；每请求输入上限与有限 outstanding count 共同约束 host input memory，而不是另设 aggregate byte
allocator。

CPU preparation 产生 owning、immutable prompt representation。一个通用 host worker-pool 实现提供固定
线程数、有界 FIFO、future/异常传播和 backpressure；Qwen Frontend 用它执行独立 media cache miss 的
decode、resize 和 BF16 pack。缓存 ready entry、single-flight state、PreparedPrompt 和 Vision runtime
分别持有同一 per-item payload 的 typed shared reference。每个 item 首次完成 Vision encode 后只释放其
payload reference；item-index slot 保留到 prompt lifetime 结束，使跨 Text-prefill chunks 的同一 item
继续复用已经生成的 Vision transient。最后一个引用销毁时精确归还 live bytes。
Prepared requests 按 preparation completion 顺序进入 Engine ordered admission queue。
Queue timeout 从请求首次取得 request lifetime capacity 时开始，因而包含 media acquisition、prompt
preparation 和 Engine admission waiting。

该顺序是公平基线，不是绝对的 head-of-line barrier。只有 §5.3–§5.6 定义的 protected-head backfill 可以
让 later request 先 admission；CPU preparation completion、prefix reuse 或请求内容本身不产生其他隐式
priority。

### 5.2 Authoritative admission feasibility

Scheduler 不从 token 数或显存字节重新推导请求能否进入。Target/resource layer 为每个 prepared request
给出已经按各 pool 物理粒度取整的 authoritative entitlement vector。逻辑上至少包含：

```text
E(request) = {
    active lane:                  1,
    Main Text page groups:       E_main,
    selected-backend page groups: E_backend,
    other request-time units:    target-defined, if any
}
```

各维只能与同类容量逐维比较；Main/backend pages 不能按字节相加或互借，allocation 尾页 slack 也不是可用
capacity。当前 fixed recurrent/backend backing 已按 `C` 份建立并随 physical lane 转移 ownership，因此由
`active lane` 这一维表达，不把固定 backing 的显存字节再次计入动态 entitlement。Graph、workspace、round
frame 和权重是 engine-fixed resources；owning result capacity 已在 ingress 时取得，也不属于本向量。

一个 candidate 只有同时满足以下条件才具有当前 admission feasibility：

1. 存在可用 control lane；
2. request 对固定 model、frontend 和 selected execution backend 合法；
3. active-priority retained eviction 后，每个 entitlement dimension 都能承诺 request 的完整生命周期需求；
4. 当前没有其他 prefill owner；admission 会立即建立该 request 的 suffix-prefill 或 exact-hit finalization
   ownership。

Selected backend 的差异只体现在 target 给出的 typed entitlement；Scheduler 不为 ordinary、MTP 或 DFlash
建立不同 queue policy。即使 startup sizing 保证 backend 不会早于 Main pool 成为正常 backpressure，
admission 仍消费完整 authoritative vector，backend 提前失败属于 sizing/accounting invariant violation。

Admission 是一次 atomic boundary transaction：prepared entry、optional retained-state claim、slot/lane 和
全部 fixed/growing state entitlement 要么同时取得并发布 admitted request，要么不改变 request-visible
ownership。Feasibility probe 和 protected-head shadow accounting 不分配 page、不 pin retained entry，也不
给予 queued request 部分 ownership；最终仍由同一次 authoritative transaction 复核并提交。

Request 必须先通过 exclusive feasibility：驱逐所有可驱逐 retained state、没有其他 active request 时，
它能够独占 Engine admission。不能独占容纳的 request 永久拒绝，不能成为 protected head；如果没有 active
request 而队首仍无法进入，则只能是 permanent rejection 或 resource-accounting failure，不能 idle 等待。

### 5.3 Ordered queue and protected head

每个 admission opportunity 先检查最老 prepared entry：

```text
oldest head is feasible now
    -> admit head

oldest head is permanently infeasible / cancelled / expired
    -> remove it, then reconsider the new oldest entry

oldest head is exclusively feasible but blocked by active lane/entitlement
    -> protect this head and consider qualified backfill
```

若 head 的完整 lane/entitlement 已经可行、只因现有 one-prefill-owner 不能 admission，它保持 ordinary
head waiting，不建立 protection epoch，也不扫描 later candidates；prefill owner 清除后的下一个合法
admission turn 先重试该 head。Protection 只在没有 prefill owner、head 确实需要现有 active request 释放
lane 或 entitlement 时建立，因此 frozen donor set 非空。

同一时刻只保护一个最老 head。Protection 绑定稳定的 request generation identity，不绑定可回收的 lane、
slot index 或 compact batch row。它只是一份 Scheduler shadow state，不为 waiting request 预占真实 pages、
lane 或 retained continuation。

Protection lifecycle 为：

```text
UNPROTECTED
    │ oldest head is transiently blocked at an admission boundary
    ▼
PROTECTED_OPEN
    │ protected resource opportunity has matured,
    │ but head is still blocked by a temporal borrower/prefill owner
    ▼
PROTECTED_DRAIN
```

`PROTECTED_OPEN` 允许 §5.5 定义的 backfill。`PROTECTED_DRAIN` 禁止所有 later/backfill admission，只推进
已经 admitted 的 bounded prefill/decode work，并在每个合法 admission turn 继续 first-retry protected head。
Head 成功 admission、cancellation、queue timeout 或 permanent rejection 清除当前 protection；下一最老
request 之后可以建立自己的 epoch，但不会继承旧 frontier、work credit 或 shadow ledger。多个 waiting
大请求不建立嵌套 reservations。

Queue deadline 在 protection 或被越过时不重置。Backfill 只保证正常 Engine progress 下不会因持续 later
ingress 产生无限饥饿，不承诺 protected request 一定早于其 configured queue timeout admission。

### 5.4 Frozen release frontier and persistent-safe backfill

Protection 建立时，令：

- `K` 为逐维 usable capacity；
- `H` 为 protected head；
- `A` 为此刻已经 admitted 的 frozen incumbent identities；
- `D` 为 `A` 中预计最早完成、且其释放足以让 `H` admission 的最小有序前缀。

Scheduler 按 §5.5 的 remaining-service projection 排列 `A`，逐个加入 donor，直到以下关系逐维成立：

```text
sum(E(a), a in A without D) + E(H) <= K
```

`D` 定义的是一组 incumbent completion/release events，不是对外承诺的 wall-clock ETA。它一经建立只能因
incumbent 提前 completion/cancellation 而缩短；later admitted backfill 永远不能成为 donor，也不能把
frontier 换成更晚的 active request。任何其他 incumbent 的提前释放如果已经让 `H` 可行，Scheduler 可以
早于 frozen frontier admission `H`。

设 `P` 为当前仍 active、被分类为 persistent-safe 的 backfill requests。它们必须始终满足累计 shadow
invariant：

```text
sum(E(a), a in surviving members of A without D)
+ sum(E(p), p in P)
+ E(H)
<= K
```

一个 candidate 只有同时 `fits-now` 且加入后仍满足该逐维关系，才能成为 persistent-safe backfill。必须
累计核算所有仍存活的 `P`；不能让多个 candidate 分别与同一份初始 surplus 比较。Persistent-safe request
完成后从 ledger 移除，其 surplus 可以被另一个 qualified request 使用；但任何 resource-release boundary
若同时是 admission turn，都必须先重试 `H`，再考虑复用 surplus；否则只记录 release/maturity，并在下一
合法 turn 先重试 `H`。

这一 invariant 保证：即使所有 `P` 在 donor frontier 到达时仍然 active，`H` 也不会因它们失去所需 lane、
Main pages 或 backend pages。它不保证 `H` 的 wall time 不受影响；backfill 的 bounded prefill 和更大的
decode batch 仍会改变 incumbent round latency。

Retained state 在 protection accounting 中是可驱逐 cache，而不是 queued ownership。`E(H)` 使用 cold
admission 的完整 entitlement；不为 `H` pin matching lane/checkpoint。Incumbent 或 backfill completion 后
产生的 retained state，在 active admission 需要时必须于同一 boundary 按 active-priority 语义驱逐，因此
可以把其完整 active entitlement 视为可释放。不可驱逐的 engine-fixed occupancy 属于 `K` 之外的 baseline，
不能伪装成 donor release。

### 5.5 Service projection and bounded temporal borrowing

总 entitlement 决定资源可行性，但不能单独代表 residence time。Scheduler 使用一份有限、单调、以统一
正整数 scheduling-work quanta 表示的 service projection 来识别相对短的 work：

```text
service work = known Text/Vision suffix-prefill and finalization work
             + effective remaining output reservation expressed as decode work
```

Output 部分使用声明的 finite effective output bound，不预测 prompt 内容、reasoning difficulty 或实际 EOS。
Prefill 部分使用已经 bounded 的 target scheduling-unit profiles；短 output 不能抵消任意长的 Text/Vision
prefill。当前 Qwen3.6 profile 以 externally scheduled prefill/finalization steps 的有限上界作为 prefill
quanta，并把每个 effective remaining output token 计为一个 decode quantum；prompt snapshot 和 Vision item
造成的已知 prefill split 在 planning 时计入。Selected speculative backend 可以影响 target 的统一 work
projection，但不会在 Scheduler 中产生 MTP/DFlash policy branches。

Projection 只服务以下两件事：选择 frozen incumbent donor order，以及约束 temporal borrowing。它不是
completion ETA、内存安全依据或公平性证明。Active request 可以下一轮提前 EOS，speculative acceptance 和
batch round latency也会变化；估计错误必须由 protection state 收口，而不是假定预测准确。

令 `W_frontier` 为 epoch 建立时、frozen donors 并行持续推进到 required release event 的 projected
remaining service distance。Protection 以它初始化一份正数、有限且不补充的 temporal credit：

```text
credit_initial = W_frontier
credit_after_admission = credit_before_admission - service_work(candidate)
```

所有可 admission request 的 service work 至少为一个 quantum；即使 exact retained hit 没有 suffix token，
仍有 finalization work。Credit 也使用相同离散单位，因此每次 temporal admission 至少扣除一个 unit，不存在
无限递减但永不耗尽的序列。Candidate 若不能满足 persistent-safe invariant，但当前完整 entitlement 能够
`fits-now`，只有同时满足以下条件才能临时借用 `H` 在 frontier 后需要的 critical capacity：

1. candidate 的完整 service work 不超过当前 projected remaining distance，预计在 frozen frontier 前完成；
2. 该 service work 不超过 remaining temporal credit；
3. admission 后从 credit 中扣除 candidate 的完整 work，request 提前完成也不返还；
4. protected opportunity 成熟后立即停止 temporal admission。

这类 request 称为 temporal backfill；其完整 active footprint 仍进入当前 authoritative resource accounting，
但不进入 persistent-safe ledger，因为 policy 预计它会在 frontier 前释放。Work credit 防止一个 32K 长
request 与一个 2K 短 request 被当作相同的一次 bypass，也限制一个 epoch 内后来请求能够借入的累计服务
工作。Credit 只减不增；new arrival 不能扩展 epoch。

Scheduler 对 bounded pending snapshot 按 prepared order 扫描，选择最老的 qualified candidate。Qualification
可能把较早但不安全/不够短的 entry 跳过，但不对整个 queue 永久执行 SJF、best-fit 或 knapsack。若所有
request 都声明相同的巨大 default output bound，Scheduler 没有可靠证据判断实际难度；此时 temporal
qualification 会自然保守，仍可使用可证明的 persistent-safe backfill，不能根据 prompt 内容猜测。

### 5.6 Admission boundary and progress rules

只有满足以下条件的 boundary 才是 admission turn：

```text
no prefill owner
and (no decode-ready request or the completed GPU unit was a DecodeRound)
```

其他 boundary 可以处理 completion、cancellation、timeout 和 protection bookkeeping，但不能 commit head
或 backfill admission。这个 gate 保证已有 decode-ready donors 在两次 admission 之间至少完成一次 progress
round；final prefill/finalization 结束后不能立即连续 admission 另一个 request。

一次 admission turn 最多成功 admission 一个 request。Cancellation、timeout 和 permanent-invalid entry 的
清理不算 admission；Scheduler 可以在同一 frozen pending snapshot 中继续寻找新的 head/candidate，但同一
资源状态下失败的 candidate 不在该 boundary 反复尝试。

每次 admission opportunity 的顺序固定为：

```text
1. resolve completed GPU unit
2. finish/cancel active requests and release-or-retain their state
3. remove visible pending cancellation, timeout and permanent failures
4. if this is an admission turn, retry the protected head or the oldest FIFO head
5. on the same turn, if an open protected head remains blocked,
   admit at most one qualified backfill
6. choose and launch one next GPU unit
```

若 exact current accounting 已让 protected head 可行，它总是先于 later request admission。若 frozen donors
已经释放，或当前非-temporal shadow resources 已足以容纳 `H`，但 `H` 仍被 temporal borrower 或现有
prefill owner 阻塞，protection 立即进入 `PROTECTED_DRAIN`。Drain 不抢占 borrower；它禁止 later/backfill
admission，但 protected head 仍在每个合法 turn 优先重试。Persistent-safe borrower 尚在 prefill 时也可能
短暂阻止 head admission，但 one-prefill-owner work 有限且此期间不能再接纳其他请求。

Frontier member completion/cancellation 只在其 state 于 boundary 真正释放后计为 release。该 boundary 若是
admission turn 就立即 head-first retry；否则更新 maturity，并在下一合法 turn 先重试。Active request 的
used/reserved 转换不改变其 entitlement；尚未 materialize 的 future reservation 不是 tail surplus。
Backfill completion retention 也不能覆盖 head-first active-priority eviction。

Protection 不改变 GPU progress policy：

- 两次成功的 backfill admissions 之间，每个仍存活 frozen donor 至少参加一次 maximal `DecodeRound`；
- 每个非 terminal row 的成功 decode/speculative round 至少 commit 一个 output token，否则直接 terminal；
- 每个 `PrefillChunk` 推进正数的有限 prompt work，或完成 finalization；
- 每个 request 有 finite output bound，每个 GPU unit 有 bounded execution extent。

因此 frozen donors 在有限 rounds 内 release。Frontier 前能发生的 temporal admissions 也有限；frontier
成熟后的 miss 进入 drain，已经 admitted 的 borrowers 最终 completion/cancellation 后释放资源。持续 ingress
不能把 donor frontier 向后滚动，也不能形成资源 deadlock。该保证是 eventual progress，不是零 wall-time
interference 或 admission ETA。

### 5.7 Waiting outcomes and sustained ingress

Admission 结果分为：

| 条件 | 结果 |
|---|---|
| request 非法或超过 semantic limit | 永久拒绝 |
| active-priority eviction 后独占也无法容纳 | 对当前 Engine configuration 拒绝 |
| request lifetime capacity 已满 | ingress 以 overload 拒绝 |
| 独占可容纳，但当前 slot/resource/prefill owner 不允许 admission | protected waiting 或 ordinary waiting |
| qualified later request 使用当前可用 backfill opportunity | 先于 blocked head admission |
| admission 前 queue timeout 到期 | 以 queue-timeout 结束 |
| Engine 正在停止或不可用 | 以 unavailable 拒绝 |

只有 admitted request 获得 GPU completion commitment。Queued request 只有 bounded waiting commitment，
没有 execution commitment或 admission ETA。它等待到 boundary admission、cancellation/disconnect、queue
timeout 或 Engine shutdown/failure 中最先发生的事件。

在 sustained ingress 下，有限 queue 填满后立即拒绝后续请求。Later arrival 可以在 open protection epoch
内通过同一 persistent-safe/temporal qualification backfill，但不能改变 protected identity、增加 temporal
credit、成为当前 frontier donor，或增加任何 admitted request 的 workload/resource commitment。

---

## 6. Shared context and state memory

本节只定义 concurrent engine 使用 Sequence-State Store 的 resource contract 和 ownership
transitions；存储 substrate 本身由 §1.3 预先提供。

### 6.1 State ownership

每个 admitted request 在选中的 free lane 上获得一份 sequence-state ownership。根据选定的 model mode，
该 state 包含：

- Text KV cache；
- GDN convolution/recurrent state；
- speculative-backend persistent state；
- position 和 model-continuation metadata。

Growing KV pages 来自 shared pools，不按 lane 静态切分，因此不同 request 可以占用差异很大的 context
memory。Fixed recurrent/backend backing 与 current lane 绑定；这是物理定位，不把 external request 或
compact row identity绑定到 lane。

### 6.2 Completion reservation

Admission 为每个 request 计算 target-specific entitlement。Main Text 的逻辑上界为：

```text
prompt_tokens + effective_output_tokens - 1
```

它按 page size 向上取整。DFlash Full backend 使用相同 entitlement；MTP backend 还覆盖最多 `K-1` 个
provisional leading positions，并受单序列 `max_context` 截断。Prefix reuse 只改变已经 materialized 的部分和
prefill work，不降低新 request 最终可达到的 entitlement。

这些 target facts 经各 pool 的 page geometry 取整后形成 §5.2 的 authoritative admission vector。同一向量
用于 actual reservation、protected frontier 和 persistent-safe shadow ledger；Scheduler 不维护第二套 token
或 byte 估算公式。

每一类 shared state resource 都必须始终满足：

```text
active used capacity
+ active reserved-but-not-yet-used capacity
+ retained used capacity
<= total usable capacity
```

Selected speculative backend 的物理 KV pool 虽与 Main Text pool 分离，但容量由同一个 target-specific
profile 联合规划。设 `S=max_context`、`P` 为 page size、`L=ceil(S/P)`、
`C=max_concurrency`、`M_min=max(L,C)`、`M_max=C*L`。Explicit policy 从用户 token capacity 得到
`M=ceil(K_main/P)`；Automatic policy 在权重加载后保留 headroom `R`，从完整 target physical layout 的
reservation curve 直接求出 `F-R` 可容纳的最大 `M`。CLI/server 使用 `R=1 GiB`，并在完整 startup 后
报告实际 free memory。Main 与 DFlash 各
使用 `M` 个 physical page groups，每条
allocation 的 logical capacity 为 `L`；MTP 使用
`M + C*ceil((K_draft-1)/P)` 个 physical groups，logical capacity 同样为 `L`，其中 `K_draft` 是
speculative draft window。额外 groups 只容纳多个
concurrent MTP rows 的 provisional lead，不扩大 request context。Startup 要求 `K_main>=S`、`M>=C`、
`M<=C*L`；`K_main>=S` 只适用于 Explicit。Capacity resolution 后全部 typed pools 与 Graph topology
固定，不在 request-time 扩容。Active-priority retained eviction 后，任何满足 advertised Main pool contract 的 active
entitlement set 都必须同时满足 backend reservation；backend 更早失败属于 startup sizing 或 accounting
invariant violation，不是正常 admission backpressure。

Prefill/decode 推进只把该 request 已有的 logical reservation 转换成 used capacity，不创建
新的资源要求。Physical blocks 是提前划归还是按需绑定，属于 Sequence-State Store。

Prefix reuse 在同一 lane 原子调整 retained entitlement 并转为 active entitlement，不重复计费。Retained
entry 在 request 结束时只保留已经 mapped 的 pages，不保留 future growth entitlement；eviction 释放整份
continuation 后再重试 active admission。

启动 prefill chunk 或 decode round 前，Scheduler 用 request 的 remaining reservation 和 logical
context/output limit 限制该 unit 最大可能的 state growth。没有合法增长空间的 request 直接 terminal，
不能启动可能越界的 unit。

这使 concurrency 与 context length 解耦：

- 一个 request 可以使用 pool 的大部分容量；
- 两个 request 可以使用完全不同的容量；
- active reservations 留下的容量不足时，新 request 等待；
- active request 不会在运行后期才发现声明的 output 无法容纳。

因此每个 request 都必须有有限 maximum output bound。Caller 未提供时，Engine 使用 configured
generation cap。

### 6.3 Fixed and growing state

KV cache 等 growing state 按 request context 计费。Fixed-size recurrent/backend state 按 admitted
sequence 分配。即使二者来自不同内部 pool，也必须在同一次 admission decision 中同时满足。

Growing KV state 由 [Paged KV Context Store](paged-kv-cache.md) 以 target-defined homogeneous
page-group pools 承载。一个 sequence 始终持有 Main Text allocation；Engine 选定 MTP 或 DFlash 时还
必须持有对应的 backend allocation。各 pool 具有独立 frontier 和 reservation；单个 request 的 physical
context 不要求连续。Concurrent engine 只操作 bundle handle、logical frontiers 和 reservation vector，
不参与 page 或物理地址管理。

GPU work in-flight 期间，per-request state handle 和 view 的含义必须稳定。Page size、slab layout、
block-table 和 allocator 的 contract 属于 Paged KV Context Store，不在本并发文档重复定义。

### 6.4 Prefix reuse

Retained prefix 是从已结束 request 中分离出来的、单一 owner 的 SequenceState。它留在原 physical lane，
但该 lane 的 control slot 对 scheduler 是 free。Retained state 只发布 target 已保存完整 continuation state
的 checkpoints。当前 Qwen3.6 retained state 可以发布 current resume frontier，以及一份有效时的 typed
rewrite checkpoint。后者按捕获时的用途标记为 `TurnClosure` 或 `ResponseReplay`；两种 kind 互斥复用同一
份物理 payload，并额外保存对应的 recurrent、hidden、speculative-backend 和 position state，不复制 KV
payload。

Qwen frontend 从有效 `preserve_thinking` 语义发布 desired checkpoint：`false` 选择最后一个真实 user
之后第一条 assistant opener 的末尾；`true` 选择本次完整 deterministic generation prologue 的末尾，
即当前 prompt frontier。Thinking generation 包含 `<think>\n`，non-thinking generation 包含完整 empty
thinking block。Boundary 先作为 byte offset 产生，再独立 tokenize 并验证为完整 prompt token prefix；
schema adapter 不推断或改写这些 target-private 语义。

Admission 在同一 lane 成功时消费 retained entry，并把 SequenceState ownership 转移给新 request：

- incoming prompt 在 current resume frontier 结束时，跳过 suffix token processing，经 finalization unit
  产生下一 anchor 后进入 `DECODE_READY`；
- incoming prompt 完整包含 current resume frontier 并有后续 suffix 时，从该 frontier prefill suffix；
- incoming prompt 正好结束在 matching rewrite checkpoint 时，恢复其完整 continuation state，经同一
  finalization unit 产生下一 anchor；
- incoming prompt 匹配已保存 rewrite checkpoint 并在其后有新 suffix 时，在 atomic admission transaction
  提交后把 growing allocations truncate 到 checkpoint frontier、恢复完整 checkpoint，再 prefill suffix；
- common prefix 结束在没有 checkpoint 的任意其他位置时视为 cache miss。

Rewrite-checkpoint restore 保留包含 checkpoint 的部分尾页，释放其后的完整 pages。KV page 或 token prefix match
本身不是 checkpoint；当前架构不支持 arbitrary longest-common-prefix reuse。

Checkpoint kind 不是 reuse compatibility bit。Planner 总是先按 token、position、media identity 和完整
continuation state 尝试 current frontier，再尝试已有 rewrite checkpoint；即使本次
`preserve_thinking` 选择了另一种 desired kind，匹配的旧快照仍可恢复，`prefix_reuse_path` 报告实际恢复的
kind。若 desired boundary 位于本次待 prefill suffix 中，则在跨过它时覆盖物理 slot；若它已经位于 selected
reuse frontier 之前且该位置没有快照，则本次保留合法 reuse 并延迟新 checkpoint，而不是为建立辅助快照
主动 full reset。只有 incoming prefix 不匹配任何完整 checkpoint 时才 full reset。

一个 retained entry 同时只能被一个 active request 消费。多个 active requests 不共享同一份可写
sequence state，也不使用 copy-on-write branching。Request-local RNG、sampling、stop、generation
limit 和 output state 始终由新 request 创建。

Retained state 占用实际 state-pool memory，但不占 active control slot，也不保留 future growth
reservation。Active admission 优先；cache occupancy 阻塞原本可行的 request 前，先驱逐 free lanes 上的
retained entries。Planner 不复制或迁移 retained physical state，而是在 free lanes 中选择最大合法 reuse。
只有在 slot/lane 和完整 entitlement 都已满足后才能 claim cache ownership。

Prefix lookup 只改变 uncached prompt work 和 prospective reuse plan，不自行授予 queue priority。它可以保守地
缩短 §5.5 的 service projection，但仍须通过相同 protected-head qualification；无论是否命中，最终 active
request 都进入相同 prefill/decode schedule 和 compact batch formation。

---

## 7. Scheduling model

### 7.1 GPU scheduling units

Scheduler 只提交两类 GPU work：

```text
PrefillChunk(request)
DecodeRound(all decode-ready requests)
```

完整 request 不是 scheduling unit。所有 GPU work 在一条 execution lane 上串行执行。

Model Runtime 拥有一份地址稳定的 shared workspace，由串行的 GPU units 复用，不按 request
复制。Prefill owner 可在 Vision/Text phases 及多个 chunks 之间持有一份 request-transient lease；
final prefill、cancellation 或 failure 后释放。

### 7.2 Boundary processing

当前 GPU unit 完成，或 GPU idle 时收到 control event，GPU Executor 执行一次 boundary：

```text
1. snapshot control events visible to this boundary
2. resolve the completed unit's per-request results
3. finish/cancel requests and release or retain their state
4. clean visible pending cancellation/timeout/permanent failures
5. if this boundary is an admission turn, apply one head-first
   protected admission/backfill opportunity
6. choose, prepare and launch one next GPU unit
```

boundary 开始后到达的 event 留到下一 boundary。这保证一次 membership update 有限，持续 arrival 不能
无限延迟下一次 launch。Admission turn 的 gate、frozen pending snapshot 和 protection state 由 §5.6 定义；
不满足 gate 的 boundary 只更新 control/resource state。清理多个无效 entries 不算多个 admissions，但一次
boundary 最多发布一个新 admitted request。

### 7.3 Decode/prefill policy

调度策略为：

```text
if the completed unit was a DecodeRound and a prefill owner exists:
    run one latency-bounded PrefillChunk
else if one or more requests are DECODE_READY:
    run one DecodeRound containing all of them
else if a prefill owner exists:
    run the next PrefillChunk
else:
    remain idle
```

因此 decode 和 prefill 同时持续 runnable 时：

```text
DecodeRound -> PrefillChunk -> DecodeRound -> PrefillChunk -> ...
```

没有 decode-ready request 时，prefill chunks 连续执行；没有 prefill owner 时，decode rounds 连续执行。

当没有 prefill owner 而 ordered pending queue 非空时，§5 选中的 head 或 backfill request 占用下一次
prefill/finalization opportunity：GPU idle 时可以立即 admission；已有 decode-ready rows 时，先完成一个
DecodeRound，再 admission selected request 并执行它的 first prefill/finalization unit。若该 unit 未完成，
它成为唯一 prefill owner并进入上述交替；若它完成，request 在下一 boundary 加入 decode batch。持续
ingress 因此不能在两个 donor progress rounds 之间连续 admission 多个 requests，也不能无限延迟 frozen
frontier 的 decode progress。

Prefill chunk profile 限制插入两个 decode rounds 之间的 GPU 时间。其具体 token/media extent 是经过
target 和 hardware qualification 的配置，不属于 scheduler semantic。Vision 和其他 prefill GPU phases
必须本身构成 bounded unit，或已被计入该 chunk 的 latency bound；不存在 scheduler 之外的
unbounded prefill work。

### 7.4 Joining and leaving decode

- request 在 final prefill/finalization 完成后加入 active decode set；exact retained hit 只省略 suffix work，
  不绕过现有 prefill owner；
- newly ready request 首次出现在下一个 boundary 构建的 batch；
- protected-head policy 只影响 admission；backfill 一旦 decode-ready，不得为了保护 waiting head 而从
  maximal batch 中排除；
- EOS、stop 或 generation limit 在 completed round commit 后移除 request；
- 每次 join/leave 后重新 compact batch rows；
- empty slot 永不产生 empty row。

### 7.5 Cancellation

Cancellation 不打断 in-flight GPU unit。Boundary 在 launch 前观察一次 cancellation，已取消的 request 不
进入下一 membership。Unit 返回后、任何 per-row output preview/commit 之前，再对 frozen membership 取得
一次 row-aligned cancellation snapshot；这份 snapshot 在整次 resolve 中不变。此时取消的 row 丢弃本 unit
尚未 commit 的 result，释放其 model state 和 slot，并按 serving contract 关闭 response output。
Provisional KV/state writes 随整条 `SequenceState` 一起释放，不需要 rollback，也不影响同一 round 的其他
rows。

因此 cancellation 最多等待一个 membership 已固定的 GPU unit，再加一次 boundary processing。

Waiting protected head 或 backfill candidate 的 cancellation 在 §7.2 control cleanup 中移除；protected head
离队会原子清空 epoch，下一最老 entry 不继承 shadow state。Frozen donor 的 cancellation 只有在其 active
state 真正释放后才缩短 frontier；若该 boundary 不是 admission turn，则在下一合法 turn 先触发 head retry。

---

## 8. Batched model execution

### 8.1 Runtime ownership

Resident Model Runtime 是 model-instance object，不是 request object。它在 Engine lifetime 内唯一拥有：

- immutable model weights 和 target schedule；
- shared Sequence-State Store；
- 一份 shared execution workspace；
- 一份最大容量为 `C` 的 `DecodeBatchFrame`；
- speculative backend 启用时，一份容量为 `C`、宽度为 `draft_window+1` 的 all-layer ReplaySSM record arena；
- startup-captured graph definitions 和 topology executables。

每个 request 的持久状态只存在于 slot control 和该 slot 当前拥有的 `SequenceState`。Model Runtime 不保存
“current request”，也不为每个 slot 复制 workspace、round buffers 或 graph assets。

一次 decode round 另外建立两个短生命周期对象：

| 对象 | 位置与 lifetime | 内容与所有权 |
|---|---|---|
| `RoundMembership` | host；从 batch build 持续到该 round commit 完成 | immutable `B` 和 `row -> slot` mapping；保持 slot/state alive |
| `DecodeBatchFrame` | host/device stable storage；由所有 rounds 串行复用 | typed input controls、state selectors、round activations 和 result staging；不拥有任何 sequence state |

`RoundMembership` 不上传 request identity。Device schedule 只看到 target/Op 所需的 typed selectors；完成结果
再由 host mapping 对应回原 slot。GPU work in-flight 期间，membership、相关 slot binding、state allocation 和
frame ingress 都不可修改。

### 8.2 DecodeBatchFrame

`DecodeBatchFrame` 在 Engine startup 按 `C` 规划一次。每个 exact-`B` schedule 只取得相同 backing 的
prefix views，不为 `B=1..C` 分别分配一套 round memory。`B` 之外的 stale rows 不会被 graph 读取，因此
boundary 不需要清零 inactive tail。

Ordinary decode 所需的 typed fields 至少包括：

```text
current_tokens[B]
cache_positions[1,B]
RoPE positions/deltas[per target contract, B]
Main KV table rows[B] + optional backend KV table rows[B]
lanes[B]
SamplingConfig[B] + logical sampling positions[B]
sampled_tokens[B]
```

`lanes[b]` 是 row `b` 的 stable execution lane。Qwen3.6 用它同时选择 Linear Attention current state 和
continuation-hidden destination；speculative Fold 也从 frozen membership 取得同一个 lane。KV table row
保持独立，因为它描述 paged allocation binding，不是 fixed-state ownership。

Hidden activations、mixer intermediates 和 logits 使用同一 shared frame/workspace 中的 exact-`B` views。
`DecodeBatchFrame` 是 runtime 对这些 typed regions 的逻辑集合，不是传给所有 Ops 的通用 descriptor ABI；
每个 Op 仍只接收自己的语义输入。Host/device control staging 可以一次传输整个固定容量 `C` 的小型
SoA frame；model schedule 只读取 exact-`B` prefixes，不为 inactive tail 提交 model work。

Batch assembly 对不同数据采用不同处理：

| 数据 | 组批方式 |
|---|---|
| token、position、sampling config 等小型 controls | 按 compact row 写入连续 batch ingress |
| KV payload 和 block tables | 保留在 shared paged pool，只写 per-row table-row selector |
| Linear Attention / backend fixed state | 保留在 shared state pool，以 stable lane 或 backend-defined row 定位 |
| request stop、output 和 external identity | 只保留在 host slot，不进入 model graph |
| activations、hidden、logits | 由一次 whole-batch schedule 在 shared execution memory 中产生 |

不得 gather/copy KV 或 recurrent state 来制造连续 batch，也不得为每行构造 device-pointer array。Target
需要跨 round 保留的最终 hidden 或其他 continuation image，必须在同一 batched schedule 内通过 typed
destination selectors 发布到各自的 `SequenceState`，不能在 boundary 对每个 request 分别发起 D2D copy。

### 8.3 Round preparation

在 boundary 选中 `DecodeRound` 后，Target Batch Assembler 按以下顺序准备：

```text
all DECODE_READY slots
        │ compact once
        ▼
immutable RoundMembership[B]
        │
        ├─ materialize each row's already-reserved one-round state growth
        ├─ fill current token, exact positions and typed state selectors
        ├─ fill per-row sampling inputs and continuation destinations
        └─ select one whole-batch execution profile
        ▼
publish exact-B ingress -> launch
```

普通 round 的每个 member 都有一个有效 column，因此不产生 inactive rows 或 valid-column mask。
Materialization 可以为不同 allocations 更新 block tables，但 pool/table-matrix base address 保持不变。所有
rows 必须在 launch 前完成准备；不允许先启动部分 batch，再为另一行补充 state。

Profile selection 使用整个 batch 的 bound，例如全部 rows 中最大的 visible-context frontier。Exact per-row
positions 和 context lengths 仍作为 device data。若一个合法 active set 不能由单个 profile 表示，则该 route
不满足并发 contract，不能把 rows 拆成 cohorts。

Host 可以对 `B<=C` 的 metadata 做轻量循环，但 GPU ingress 必须是 batch-level publication；不得为每行发起
独立的 scalar copy、transition kernel 或 model call。

### 8.4 Ordinary round state transaction

一个 `DECODE_READY` sequence 始终保存 target-defined committed cursor 和唯一 current decode anchor。以
Qwen3.6 的 ordinary transition 为例：

```text
before:
  target state valid for positions [0,p)
  current anchor token is at position p

execute:
  process the B anchors at their respective positions
  write each row's provisional KV/recurrent/continuation state
  sample one next anchor per row

commit for surviving row b:
  state frontier  := p[b] + 1
  current anchor  := sampled_tokens[b]
  anchor position := p[b] + 1
```

Op 只写被本轮许可的 physical state，不拥有 committed frontier。Graph 内也不把 compact-row position 或
selector 原地推进成“下一轮状态”；权威 cursor 由 `SequenceState` 在 boundary commit 时更新，下一轮再按新的
row mapping 生成 controls。这样 row 可以自由移动，而不需要 vectorized scalar helpers 或持久 row-local
state。

Ordinary round 对每个未取消 row 恰好 license 一个 token。EOS、stop 或 generation limit 可以让该 token
成为 terminal token，但不把同一 row 的 model state 与 output 截在不同 frontier。Cancellation 是唯一可以
丢弃整行 provisional result 的 ordinary boundary outcome；该 `SequenceState` 随即释放。

Qwen3.6 ordinary GDN 以 `initial_state_slots=lanes`、`snapshot_base_slots=lanes` 调用已有 width-1 Snapshot
leaf。该 leaf 在完整读取 row 的 initial checkpoint 后原地覆盖同一 current slot，因此不产生 speculative
trajectory，也不需要额外 state slot。

### 8.5 Whole-model execution

logical batch size 为 `B` 时，一次 replay 的 model schedule 为：

```text
current_tokens[B]
  -> one batched embedding/input stage
  -> one traversal of all model layers
  -> publish B continuation images
  -> one batched lm_head
  -> one batched sampler
  -> sampled_tokens[B]
```

Ordinary column layout 等价于 `[D,1,B]`，column-independent projection/MoE/FFN routes 一次消费 aggregate
`T=B`。Control code 可以遍历 lightweight row metadata，但不能为每个 request 分别调用一次 layer、model
schedule、lm_head 或 sampler。

Per-request KV/context、recurrent state、sampling state 和 output ownership 保持独立；batched execution
合并的是当前 activation columns 和 weight-consuming work，不把多个 sequences 拼成一条 sequence。

### 8.6 Operator contract

| 算子族 | Batched execution 要求 |
|---|---|
| Linear、projection、dense FFN | 一个 operator schedule 消费 aggregate row/token extent |
| MoE | 对 aggregate token set routing，再对整个集合执行 grouped experts |
| Full attention | 以 ragged batch 消费 per-row context length 和 substrate-provided KV view |
| GDN / linear attention | 对独立的 per-row recurrent-state handles 执行一次 batched transition |
| `lm_head` | 一次产生全部 active rows 或 verification positions 的 logits |
| Sampler | 消费 per-row sampling configuration 和 RNG state，不拆分 batch |

不同 context length、stop condition、sampling value 或 generation limit 不形成不同 model batch。需要
不同 model topology 的 request option 必须在 Engine startup 时固定，否则作为 unsupported 拒绝。

具体 compact layout、typed extents 和 selector ABI 由对应 `include/ninfer/ops/` semantic contracts 定义；
本架构不复制逐 Op 接口清单。

### 8.7 Result resolution and commit

Graph replay 后只回传 compact per-row result，例如 ordinary sampled token，不能回传 logits 或逐层状态。
GPU Executor 等待一次 whole-round completion，然后通过 frozen `RoundMembership` 对每行独立 resolve：

```text
row result
  -> locate its still-owned slot through RoundMembership
  -> apply cancellation / stop / output-limit policy
  -> commit the exact licensed prefix to SequenceState
  -> update request sampling, usage and owning output record
  -> mark DECODE_READY again or MODEL_FINISHED
```

对每行而言，model-state frontier、target cursor、sampler/penalty progress、usage 和 owning output record 是
一次逻辑 transaction。Output event 只有在这些 state 全部 commit 后才能对 response path 可见。一行结束、
取消或在 speculative mode 中接受较短 prefix，不改变其他行的 commit result。

Speculative backend 的 target GDN 使用 ReplaySSM 时，GPU graph 只读 lane 的 current state 并写
Program-owned raw records，不推进 committed GDN state。CPU output preview 得到每行最终提交长度后，
`resolve_pending_batch` 先用原始 `B` 行执行一次 all-layer Fold，再完成必要的 hidden/backend correction，
同步成功后才推进 host frontiers。取消行以 `commit_columns=0` 参与原始 row mapping，Fold 对该行严格
no-op。Executor 只能在这个 commit tail 成功后提交 output preview 和发布 output event。

全部 rows resolve 后，`RoundMembership` 销毁，frame 可以被下一 unit 覆盖。继续运行的 slots 在下一
boundary 重新 compact；没有任何 row identity 从当前 frame 继承到下一 frame。

### 8.8 B=1 and prefill

`B=1` 使用完全相同的 membership、frame、whole-model schedule 和 commit transaction，不保留独立的
request-local ordinary decode path。

Prefill 仍是单 sequence unit。它独占自己的 `SequenceState`，但复用 Model Runtime 和 shared workspace；
它不占有一个长期 `DecodeBatchFrame` row。Final prefill 建立完整 decode cursor 后，该 request 只在下一
boundary 通过正常 batch assembly 加入 ordinary decode。

---

## 9. CUDA Graph model

### 9.1 Exact-B graph definitions

因为 `C` 较小且固定，Engine 在 startup 为每个 logical batch size 捕获 exact definition：

```text
Definition[family, B=1, profile]
Definition[family, B=2, profile]
...
Definition[family, B=C, profile]
```

`family` 由 Engine startup mode 固定为 ordinary、MTP 或 DFlash；三个 mode 都使用 exact `B=1..C`，
DFlash 不保留单独的 `B=1` concurrency route。

`B=1` 是 first-class exact path。单请求不会运行永久 padding 到 `C` 行的 graph。
每个 definition 表示该 semantic family、exact `B` 和 whole-batch profile 的完整 decode round，绑定同一份
shared frame、workspace 和 state-pool bases。它读取 typed controls 和 selectors，不读取 request/slot
identity。

Captured definition 和 replayable executable 不是一一对应。Exact `B` 是 executable 的结构键；同一
`B` 内只有真实 CUDA node topology 不同才增加 executable：

```text
exact definitions[family,B,profile]
          │ exact B + target-declared profile topology class
          ▼
executable[family,B,topology class]
```

不同 context profiles 若在同一 exact `B` 下具有可更新的 node topology，共享一个 executable，并在 profile
变化时安装选中的 definition。不同 `B` 不执行 cross-B graph update；这避免 batch-dependent operator route、
kernel 参数和 launch shape 触发 `cudaGraphExecUpdate` 不兼容。资源数量因此是 exact `B` 的有限集合，而不是
`B × context profile` 的完整笛卡尔积。

Profile 在 capture 前按 configured context ceiling 截断，只有实际 reachable 的 topology classes 才实例化
executable。Backend-specific proposal shape 只有在真实改变 CUDA node topology 时才形成每个 exact `B`
下的 topology class，不生成 ordinary-tail 或额外 `B=1` compatibility graph。

Startup 对 graph family 的准备顺序固定为：

1. 对启用的 semantic family，以 `B=1` 的首个 reachable profile 执行一次 eager round，使 CUDA code 和
   library runtime 在 stream capture 前完成 lazy materialization；
2. 捕获全部 exact-`B`/profile definitions。Capture 只记录 CUDA work，不执行 model round，也不承担
   production 数值 qualification；
3. 每个 topology executable 由其首个 definition 实例化。该 topology 的其余 definitions 逐个通过
   `cudaGraphExecUpdate` 验证兼容性，并通过 `cudaGraphUpload` 完成 executable resource materialization；
4. 每个 executable 只 replay 一次首个 definition 作为 startup smoke。遍历其余 profiles 后，以
   update/upload 恢复首个 installed definition，不额外 replay。

因此 startup 不为每个 definition 执行 eager warm，也不把每个 profile 的 update 当成 real-model replay
qualification。完整算子与 route correctness 由独立测试和 real-artifact integration route 负责；production
startup 只验证 graph inventory、update compatibility、resource materialization 和每个 executable 的一次
可执行性。

### 9.2 Dynamic active set

active set 改变时，runtime 发布新的 frame ingress 并选择匹配的预捕获 definition：

```text
[A]       -> Definition[B=1]
[A,B]     -> Definition[B=2]
[B]       -> Definition[B=1]
[B,C,D]   -> Definition[B=3]
```

Runtime 先选择 exact-`B` executable；若它当前安装的是同一 `B` 的另一个 context-profile definition，先执行
whole-graph update，再 replay 一次。`B` 改变时直接选择另一个预实例化 executable。Active identity 改变但
`B/profile` 不变时只更新 typed data，不更新 executable。Graph 不按 request identity、slot subset 或 slot
bitmask 建 key，join、leave 和 slot reuse 不触发 capture。

Installed-definition state 只由 GPU Executor 修改；update 和 replay 位于同一串行 execution lane。并发 callers
不能直接 launch 或更新同一个 executable。

Paged-KV allocation、Linear Attention state ownership 和 retained-state reuse 只改变 block-table content 或
selector values。Graph 始终绑定 pool/table-matrix base，因此这些 ownership transitions 也不形成 graph
key。

### 9.3 Context and prefill profiles

Exact context length 是 typed runtime control value。若 exact target 对不同 context range 需要少量不同
kernel topology，则除 `B` 外可使用启动时固定并捕获的有限 whole-batch profiles。Profile
由 batch-level bound，例如 maximum row context-length bucket，选择一次；graph 仍以 exact per-row length 执行
ragged work。

Context profile 不能把 active set 分成 cohorts，也不能使用 request identity 或 active-slot combination 建
key。无法同时表示任意合法 per-row lengths 的 profile 不能作为 concurrent decode route；必须使用能
保持 whole-batch execution 的 route。

Prefill 是 single-request work，可以使用 target-specific fixed execution profile。它不改变 decode-batch
graph model，也不能在 serving 时触发会阻塞 active decode 的 graph capture。

### 9.4 Stable execution memory

所有 exact definitions 复用：

- 一份按 `C` 规划的 `DecodeBatchFrame`；
- 一份按所有 reachable phase/`B`/profile peak 取最大值的 workspace；
- 同一组 model/state pool bases；
- 一组地址稳定、按 `C` 规划的 pinned host ingress/result staging。

Graph-enabled exact-`B` definition 包含开头的一次 batch-level control upload 和结尾的一次 batch-level
result download。Staging object 可以按固定 `C` 大小传输；只有前 `B` 行具有本轮语义：

```text
fill pinned ingress rows [0,B)
        -> optional already-reserved KV page/table materialization
        -> one graph replay:
             one H2D typed-control frame
             whole-model batch schedule
             one D2H result frame
        -> one round completion wait
```

Cross-page materialization 发生在 replay 前的同一 execution lane，不改变 captured pool/table bases，也不形成
graph key。Graph-off mode 按相同顺序 eager 提交这些动作。

不得为每个 `B`、profile 或 captured definition 复制 logits、hidden、workspace 或 per-sequence state。
Model/control ingress、forward 和 result egress 不存在 per-row CUDA submission；跨 page 时的 table
publication 属于 state substrate materialization。Serving 期间不 capture、instantiate 或扩展 graph family。
Startup graph allowance 必须计入全部 reachable exact-`B` definitions，以及每个 exact `B`、每个实际
topology class 的一份 executable，不能沿用只覆盖 `B=1` definitions 的 reservation。

Capture/smoke 使用的 temporary Paged-KV allocation 每个有效 row 只 materialize 一个 private physical
page，并把该 page 重复发布到 temporary block-table row。准备一次 eager/smoke 时只清零 `[0,B)` 对应的
temporary pages、fixed-state lanes 和 typed controls；definition capture 本身只 reset shared workspace 的
host allocator。不得按 configured KV capacity 清零整个 physical pool，也不得为每个 definition 清零全部
`C` 行 state。

---

## 10. Speculative decoding

Speculative decoding 是 Engine-level decode mode：

```text
speculative_backend = off | MTP | DFlash
```

同一 Engine 的全部 requests 使用相同 backend 和 proposal window。Scheduler 仍然只提交一个
`DecodeRound(all decode-ready requests)`。

该 round 内部执行：

```text
all pending request anchors
  -> one batched proposal phase
  -> one batched target verification
  -> per-request accepted-prefix result
  -> per-request state commit
```

MTP autoregressively 推进 proposal positions，每个 position 都在全部 active requests 上 batch 执行。
DFlash 通过一次 batched block forward 产生 proposal block。二者的区别只存在于选定的 decode graph
内部，不改变 slot、admission、scheduling 或 active-batch formation。

MTP 与 DFlash 保留各自的 proposal frame、proposal state 和 graph topology；它们只共用 non-owning target
verify/accept view、per-row accepted-prefix result 和 target state transaction。并发层不建立隐藏 backend
差异的 virtual interface，也不为两个 backend 维护两套 membership 或 commit loop。

Engine capability 在 startup 时固定。MTP Engine 可以同时启用 Vision；MTP 的 shifted input 落在视觉列
时使用与 target 相同的 Vision-composed embedding，prefix-reuse bridge 与正常 prefill 遵守同一输入
语义。当前 DFlash companion checkpoint 是 text-only，因此 `DFlash + Vision` 在 Engine 构造时拒绝，
不形成 request-level fallback。

不同 request 可以接受不同数量的 proposals。Acceptance length 是 result metadata，不能据此拆分
verification、重放 model 或形成 acceptance cohort。Target model 始终是 output authority，只有 accepted
target/backend state 可以 commit。

Qwen3.6 的 MTP 与 DFlash 共用同一 target ReplaySSM transaction：target verify 按 compact row 把每层
convolution/key/value/gate records 写入固定 arena，physical record row 恒等于本轮 batch row；CPU 得到
最终 output prefix 后，一次 Fold 用 frozen `lanes[b]` 把 row `b` 提交到该 lane 的 current state。Rows
不得因取消或不同 acceptance length 被压缩、重排。Record 位于 CUDA Graph 内，Fold 位于 CPU 决策后的
eager commit tail；下一 GPU unit 必须等当前 records 被 Fold 消费后才能覆盖 arena。

Target execution 完成到 Fold 结束期间，请求处于 Pending：authoritative execution/ledger frontiers、ledger
内容和 prefix identity 仍停在 round base；licensed tokens 和 backend staging 只作为未发布候选存在。Fold、
Text/backend KV trim、continuation hidden、proposal continuation 和 host frontier 必须提交同一个最终前缀。
Continuing row 提交全部 licensed outputs；terminal row 可因 stop/EOS/output limit 提交严格前缀；取消行
提交零列并释放 sequence。

每行的 valid proposal extent 受 remaining output/context capacity 限制。Fixed proposal window 中未使用
的位置被 mask，不能更新 request state 或 output。

若一行尚未 terminal 但 valid proposal extent 为零，同一 DecodeRound 必须对该行执行 ordinary
target progress，不能保留一个不前进的 active row。EOS、stop、output limit 或 context limit 在 per-row
commit 前截断 effective committed extent，model state 和 output 只提交同一有效前缀。

---

## 11. Completion and response handling

Ingress 为 request 建立 owning record；有效 output bound 在 admission 时确定。每个 boundary 把 committed
tokens 追加到该 record，因此 ordinary 或 multi-token speculative round 都不依赖 network progress 才能
commit。

Model completion 时，GPU Executor 终结 owning result record，释放 request 的 slot 和 unused reservation，
并释放 sequence state 或按 §6.4 把带有 target-declared reusable checkpoints 的 state 转移给
retained-prefix cache。上述 GPU
resource 被复用后，response path 仍可继续读取 owning record。

Committed output events 写入 request-owned result/event storage，由等待该 request 的 caller thread 取出。
Network write、protocol serialization 和 `OutputSink` callback 不在 GPU Executor 上运行。Caller 消费缓慢
不会阻塞 GPU boundary loop；积压仍受该 request 的 finite output bound 限制。Request lifetime capacity 在
response consumer 释放 owning record 前继续计费，因此 model slot 已复用并不产生无界 completed-result
backlog。

因此 model completion 和 response completion 是两个独立 lifetime：

```text
model lifetime:     admission ───────────── model finished
response lifetime:  request received ───────────────── response finished
active slot/state:   admission ──────────── model finished
retained state:                                 optional ───── eviction/reuse
```

GPU execution 或 state-integrity failure 属于 Engine-wide failure：停止 admission，active 和 queued
requests 以 unavailable/failure 结束，重新创建 Engine 后才能恢复 serving。Scheduler 不猜测 failed
shared round 中某一行可以安全继续。

---

## 12. End-to-end examples

### 12.1 New request joins an existing decode

```text
time ───────────────────────────────────────────────────────────►

GPU: Decode[A]
       │ boundary: admit B as the prefill owner
       ▼
     Prefill[B, chunk 0]
       ▼
     Decode[A]
       ▼
     Prefill[B, final chunk]
       │ B becomes decode-ready
       ▼
     Decode[A,B] -> Decode[A,B] -> ...
```

B 不执行单独的 decode forward，而是在 final prefill 后加入 A 的下一 logical batch。

### 12.2 Two rows assembled and committed

假设 A、B 分别占用两个 slots，且持有完全不同的 context/state allocations：

```text
                         row 0 (A)        row 1 (B)
current token               101              202
cache position               41              900
Text KV table row             5                1
stable lane                   0                2
sampling state lane           0                2
```

Batch Assembler 只把上述 controls/selectors 写入 frame。KV pages 和 Linear Attention state 仍留在 shared
pools 的原位置。`Definition[B=2]` 对两列执行一次完整 model traversal，并返回：

```text
sampled_tokens = [303, 404]
```

Boundary 用 frozen mapping 把 303 交给 A、404 交给 B。若 A 继续而 B 的 404 是 terminal token，则两行都
先提交各自正确的 model/output frontier，随后 B 的 slot/state 被释放或 retained。下一轮重新组批为 A 的
`B=1` frame；A 的 KV row 和 stable lane 不因 compact batch row 变化而改变。

### 12.3 Active batch grows and shrinks

```text
ready requests          selected definition

[A]                     Definition[B=1]
[A,B]                   Definition[B=2]
[A,B,C]                 Definition[B=3]
[B,C]       A finished  Definition[B=2]
[B,C,D]     D joined    Definition[B=3]
```

request identity 和 occupied slot 都不影响 graph identity。

### 12.4 Shared 128K context with two slots

假设相关 shared context capacity 为 128K units：

```text
request A: used prompt 50K + reserved output 30K = 80K commitment
request B: prompt       32K + reserved output 16K = 48K commitment
                                                       -----
                                                       128K
```

A、B 可以同时 admission。随着 context 增长，reserved capacity 转换为 used capacity，因此二者的
`used + remaining reservation` 始终不超过 128K；可以到达边界，但不能越过边界。

若 B 需要 `32K prompt + 32K output = 64K`，A 的 commitment 之后只剩 48K。B 保持在 pending
queue 并成为 protected head，直到 A 释放容量、B 的 queue timeout 到期或其他 terminal event。NInfer
不会先 admission B，再在接近 128K 时决定截断哪个 active request。

若还有 later request C，它只有通过 §5 的完整向量核算后才能 backfill。以 Main capacity 单维说明，A 是
B 的 frozen donor；A release 后为 B 留出的容量之外仍有 64K shadow surplus。只要 lane/backend 等其他维
也成立，这给出 C 的 frontier-safe 上界；但 C 还必须满足当前 `fits-now`，此刻真实空闲只有 48K，因此
C 当前最多取得 48K。即使 C 在 A 完成时仍 active，release boundary 仍先 admission B。若 C 需要借用 B
的 critical capacity，则还必须满足 temporal work credit，并在 frontier miss 时触发 drain。

### 12.5 Persistent-safe and temporal backfill

以下例子只写 Main entitlement；实际决策同时逐维检查 lane 和 selected-backend pages。假设 capacity 为
174K、`C>=4`：

```text
active A: 64K, projected to finish first
active B: 64K
free:     46K
head H:  100K, exclusively feasible but blocked now
```

选择 `D={A}` 后，frontier accounting 为：

```text
B 64K + H 100K = 164K
safe surplus = 10K
```

一个 8K candidate P 可以成为 persistent-safe backfill：

```text
B 64K + P 8K + H 100K = 172K <= 174K
```

即使 P 在 A release 时仍运行，H 也保有完整 entitlement。另一个 20K candidate T 当前可以放入剩余
capacity，但不能加入 persistent-safe ledger；它只有在完整 prefill/decode service work 预计早于 A 的
release frontier，且 epoch temporal credit 足够时才能作为 temporal backfill。

```text
Decode[A,B]
  -> admit/fill P
  -> Decode[A,B,P]
  -> admit/fill T if temporally qualified
  -> Decode[A,B,P,T] ...
```

如果 A 提前 EOS 而 T 尚未结束，Scheduler 不把这个估计错误解释为新的 backfill opportunity：

```text
A releases
  -> retry H first
  -> H is still blocked by T
  -> PROTECTED_DRAIN
  -> no new admission; existing P/T and B continue
  -> T releases
  -> admit H
```

若 T 已经及时结束，或其他 active request 的提前完成使 exact accounting 足够，H 在对应 boundary 直接
admission。Later arrivals 从未成为 H 的 donor，也不能恢复已消费的 temporal credit。

---

## Related documents

- [Paged KV context storage](paged-kv-cache.md)
- [ReplaySSM GDN technical reference](replayssm-gdn.md)
- [Serving behavior](../serving.md)
- [Qwen3.6-27B model semantics](qwen3.6-27b-model.md)
- [Qwen3.6-35B-A3B model semantics](qwen3.6-35b-a3b-model.md)
