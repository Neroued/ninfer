# NInfer Engine 架构

本文是NInfer请求路径、执行所有权和生命周期的顶层维护者权威。它只描述当前产品模型及已经选定的架构，
不记录迁移历史，也不为未支持的部署方式预留抽象。

资源选择、物理结算和上下文缓存的窄层合同由
[资源调度与上下文缓存架构](resource-scheduling-and-context-cache.md)定义；两份文档共同描述当前实现，
不保留迁移期的旧所有权模型。

本文回答四个问题：

1. 请求经过哪些边界；
2. Scheduler、ResourceManager和Program分别决定什么；
3. admission、模型执行、terminal和输出如何提交；
4. 单GPU、小并发下如何保证正确性和稳定的热路径。

Paged KV的页几何、物理layout和consumer contract由
[Paged KV Context Store](paged-kv-cache.md)定义。本文只保留Engine层必须知道的部分。

---

## 1. 产品边界

### 1.1 当前工作负载

NInfer运行期固定为：

- 一张GPU、一个resident model instance；
- 启动时固定`max_concurrency=1..8`；
- 有界FIFO ingress，无request preemption；
- 每个round boundary把全部decode-ready请求组成一个compact batch；
- Text、Vision、MTP、prefix reuse、CLI和serving走同一公共`ninfer::Engine`路径；
- 35B-A3B target在text-only模式还可使用DFlash。

`max_concurrency`限制active request数，不把共享KV等分给lane。一个请求可使用共享pool的大部分容量，只要
Program对全部active reservations、inactive allocations和transaction peaks证明整体可行。

### 1.2 非目标

当前架构不支持：

- 大规模或抢占式continuous batching；
- priority/QoS、多租户公平或跨Engine调度；
- 多GPU placement和active swap；
- weight offload、跨Engine context storage或独立后台cache worker；
- runtime model discovery和字符串驱动的通用执行图；
- 未注册模型、其他GPU架构或假想storage tier的占位接口。

这些能力若进入产品范围，应重新定义调度与资源不变量，不能借用本文的局部机制推导。

---

## 2. 请求路径

NInfer只有四个请求边界：

```text
Gateway
  │ protocol/product request, response transport
  ▼
Frontend
  │ PreparedPrompt, OutputSession
  ▼
Engine
  │ request order, logical cache policy, output transaction
  ▼
Program
    physical state, resource transaction, model execution
```

公共`ninfer::Engine` facade同时暴露Frontend的prepare能力和Engine的submit/generate能力。CLI、HTTP server和
直接C++ caller都通过这条路径进入；它们不会绕过Engine调用target Program。

### 2.1 Gateway

Gateway包括HTTP serving、CLI及其他产品入口，负责：

- 协议解析、连接、streaming transport和API错误；
- message/tool/media到公共`PromptInput`的转换；
- URL/path/data acquisition；
- usage与response schema；
- Gateway自身的请求生命周期和容量限制。

Gateway不选择request顺序、checkpoint、physical lane或KV pages。

### 2.2 Frontend

Frontend拥有模型家族的输入与输出语义：

- tokenizer、chat template、Vision preprocessing与MRoPE prompt construction；
- owning `PreparedPrompt`及其内容identity；
- stop、thinking/content channel、detokenization与最终文本；
- `OutputSession`的preview/commit状态。

`PreparedPrompt`是完成输入语义转换后可由Program规划的owning value。`OutputSession`由一条请求独占，
可以preview但只有Engine提交后才产生可发布增量。

Frontend不拥有队列、cache catalog、physical state或调度policy。

### 2.3 Engine

Engine是请求控制平面，负责：

- bounded outstanding capacity、FIFO pending queue和deadline；
- `RequestRecord`、active request slots、cancellation和response events；
- Scheduler与ResourceManager；
- admission、prefill、decode、control、capture和terminal的boundary orchestration；
- OutputSession、GenerationBudget和Program commit的组合事务；
- failure cleanup和runtime observations。

Engine理解request、budget、finish reason与publication，不解释transformer layer、KV plane或CUDA Graph细节。

### 2.4 Program

Program是目标package提供的唯一物理执行入口，负责：

- State/KV stores、allocators、references和reservations；
- active sequences、immutable checkpoints和physical replicas；
- prefill、ordinary decode、MTP/DFlash、forced control；
- provisional state、accepted-prefix commit和rollback；
- resource planning与`ResourcePlan -> RunningTransaction -> ResourceResult`；
- workspace、Vision handoff、CUDA Graph和memory summary。

Program不维护FIFO、protected-head policy、retention价值或用户输出。

---

## 3. 所有权

### 3.1 唯一authority

| 状态或决策 | 唯一authority |
|---|---|
| protocol、connection、transport | Gateway |
| prompt、stop、thinking与output语义 | Frontend |
| pending queue、RequestRecord、response event | EngineCore |
| FIFO head、backfill、prefill/decode次序、round membership | Scheduler |
| logical lane、catalog、session、retention与candidate policy | ResourceManager |
| State/KV occupancy、reservation、placement、allocator、model execution | Program |

“唯一authority”允许其他组件读取稳定summary，但不允许它们复制并 independently mutate同一事实。

### 3.2 EngineCore

EngineCore持有：

- FIFO `pending_`；
- 固定上限的request slots；
- Scheduler和ResourceManager；
- 单一Engine worker；
- Program稳定引用；
- outstanding、response和stats状态。

EngineCore只做orchestration。它先从Scheduler取得哪条请求可尝试，再让ResourceManager形成逻辑choice，
最后把完整choice交给Program规划和执行。它不以raw lane/page/slot绕过任何owner。

### 3.3 Scheduler

Scheduler只拥有时间和顺序：

- FIFO head及其protection状态；
- staged-prefill owner；
- admission/prefill/decode公平门；
- maximal compact decode membership；
- persistent-safe backfill判定。

Scheduler不持有checkpoint handle、eviction list、physical resource vector或allocator facts。

### 3.4 ResourceManager

ResourceManager只拥有逻辑policy：

- `Free | Materializing | Active | TerminalPending` lane状态；
- private/shared checkpoint catalog与SessionIndex；
- continuation identity、retention class和bounded candidate indexes；
- source、placement intent、victim action和capture/retention choice；
- Program终态结果的逻辑adoption。

它不维护Device/Host占用账本、per-lane physical charge、page refcount镜像或allocator snapshot。逻辑引用可以
影响Program计算last reference，却不是物理bytes。

### 3.5 Program

Program中的真实State/KV containers及其allocator是唯一物理权威。所有`SequenceHandle`、
`ContinuationHandle`和transaction handle均属于产生它们的Program；owner/generation不匹配是内部错误。

Program可维护由真实stores同步更新的O(1) counters和可重建索引，但这些不是第二份独立账本。物理可行性、
逐stage peak和联合last-reference reclaim只能由Program判定。

---

## 4. Package与启动边界

### 4.1 Compile-time package

Engine读取`.ninfer` identity并从closed registry选择exact package。27B与35B-A3B是peer compile-time
Variants；worker中没有runtime family分派。

每个package向EngineCore提供同构语义类型：

```text
PreparedPrompt / OutputSession / PublishedOutput
RequestBasePlan / CandidateMatch / ResourcePlan
SequenceHandle / ContinuationHandle / PendingBatch
StartResult / ProgressResult / ResourceResult
Program
```

类型名可以在实现中细化，但边界必须保持：Engine不包含模型数学，Program不读取serving request或Scheduler
私有状态。

### 4.2 Startup

构造顺序为：

```text
validate EngineOptions
  -> read artifact identity
  -> select exact package
  -> build artifact and SequencePlan
  -> preflight KV capacity
  -> load immutable model
  -> resolve final capacity
  -> construct Frontend and Program
  -> construct EngineCore, Scheduler and ResourceManager
```

权重、State/KV backing、block-table matrices、workspace和graph storage都在启动期建立。运行期可以改变引用、
mapping和frontier，不重建这些大块device allocations。

公开cache配置的默认关系不是硬上限。Program必须按实际配置完成可表示性和启动分配验证；失败时构造失败，
不会在运行期缩容或改走备用算法。

`ninfer-serve`在开始接受HTTP请求前可以通过同一公共Engine执行路径运行内部warmup，但warmup不是产品请求，
其request-level context-cache participation固定为disabled。它不读取prefix candidate、不建立capture、也不发布
continuation；执行期间取得的active State/KV entitlement在terminal finish时全部释放。warmup可以留下已分配的
backing、CUDA Graph和kernel/library初始化状态，但首个外部请求看到的logical context cache必须与新建Engine一致。

---

## 5. 生命周期

### 5.1 Gateway与Engine capacity

HTTP Gateway可在prepare前取得独立`RequestLifetime`，覆盖media acquisition、Engine wait和response
processing。它不等同于Engine outstanding capacity。

非零输出请求在`Engine::submit`时取得outstanding slot。归还由两个latch决定：

```text
response_done       worker已经完成result/error
consumer_released   wait结束或GenerationHandle被放弃

release_capacity :=
    response_done && consumer_released && !capacity_released
```

二者可以任意先后，capacity只归还一次。`requested_output_tokens == 0`是prepare后的非执行旁路，不建立
RequestRecord。

### 5.2 Request与lane

请求控制状态为：

```text
WAITING
  -> MATERIALIZING
  -> PREFILL
  -> CONTROL_READY | DECODE_READY
  -> TERMINAL_PENDING
  -> MODEL_FINISHED
```

逻辑lane状态为：

```text
Free -> Materializing -> Active -> TerminalPending -> Free
```

| Lane状态 | Program binding | 允许动作 |
|---|---|---|
| Free | 无 | 尝试admission |
| Materializing | `RunningTransaction`；source仍有效 | progress、commit或abort |
| Active | `SequenceHandle`与完整completion reservation | prefill、decode、control、capture |
| TerminalPending | 仍持有`SequenceHandle`与reservation | finish、discard |

请求只有在materialization commit结果已被ResourceManager采用后才能进入Active。Active终止时先进入
TerminalPending；checkpoint retention成功或明确release后才回到Free。这样terminal路径不会提前释放完成保障。

### 5.3 Continuation与session

Active mutable continuation和published immutable checkpoint是不同对象。Checkpoint脱离原RequestRecord，
可以有Device、Host或Both replicas；它必须包含同一frontier的完整State和所有必需typed KV。

Session key只是lookup hint。每个request在进入Engine时取得单调`publication_order`；finish/capture只有在
order更大时才能更新SessionIndex。Capability generation验证handle资格，不表达新旧顺序。

### 5.4 两类事务

Engine只跨模块处理两类不同事务：

1. **Resource transition**：在admission、resume、capture、finish或显式inactive release时改变global available
   capacity、inactive placement或active/checkpoint ownership；需要分配、pressure或transfer时运行
   `ResourcePlan` transaction，已有active reservation覆盖的finish/discard可以原子同步完成；
2. **PendingBatch transaction**：一次模型unit产生provisional tokens，再按每行decision提交或回滚。

Planned resource transaction的外部形态只有：

```text
ResourcePlan -> RunningTransaction -> ResourceResult
```

PendingBatch只有`Program::commit`或`Program::abort_pending`可以消费。两者都不能靠析构触发隐式GPU工作。

---

## 6. Admission与资源规划

### 6.1 先选请求，再选资源

Scheduler先选择唯一可尝试的waiting request。ResourceManager只为该请求建立有界candidate shortlist：

- SessionIndex指向的private endpoint或rewrite；
- marker/shared-prefix index命中的stable prefixes；
- matching private anchors及有界anonymous candidates；
- 永久存在的root fallback。

Hash、session、marker或KV匹配只是索引。Program必须验证完整checkpoint state、typed KV coverage、frontier和
target facts。

### 6.2 完整choice

ResourceManager交给Program的规划单位同时固定：

```text
request
+ source checkpoint or root
+ destination lane
+ placement intent
+ complete victim/degradation set
+ capture/retention intent
```

Program对完整choice的joint post-state与每个有序stage peak进行验证，并在成功时seal一个opaque
`ResourcePlan`。ResourceManager看不到raw allocations，也不能把独立victim effects相加。

Canonical degradation chain是：

```text
preferred source, no pressure
  -> same source, preserving demotion/degradation
  -> cheaper valid source
  -> root, preserving demotion/degradation
  -> root, release all inactive cache
```

最后一项是admission完整性回退。若它仍不可行，原因必须是active reservations、输入本身或配置限制，而不是
cache retention policy。

### 6.3 Start、progress与terminal

```text
Scheduler grants request
  -> ResourceManager prepares logical adoption storage
  -> Program assesses and seals ResourcePlan
  -> Program start revalidates resource revision and capabilities
  -> zero or more progress units
  -> Program commit or abort returns one ResourceResult
  -> ResourceManager adopts logical terminal result
  -> Engine installs Active or adopts the stable result
```

Pre-start stale rejection无物理副作用，可以重新规划。Start成功后source持续有效到commit/abort，且不能静默
换candidate。Program commit后的logical adoption已经预分配、noexcept、无新的容量判断。

Abort保证target不会半Active，但已安全提交的victim demotion/deletion可以保留；`ResourceResult`必须完整报告
这些变化。ResourceManager不接收逐copy receipt，也不自行推导physical delta。

### 6.4 Completion reservation

Program在Active publication前锁定该sequence直到terminal的最大合法增长、State写入和selected backend需求。
`SequenceHandle`内部持有这份concrete reservation。Active在mapped与reserved之间转换不改变global
available capacity。

因此普通prefill/decode不会再次admission，也不会被其他cache choice挤出。Active truncate把页归还自己的
reservation，而不是直接送回global free pool。

### 6.5 Protected head与backfill

当FIFO head因当前active incumbents暂时不可行时，Scheduler记录：

- head identity与base request facts；
- 必须先terminal的donor set；
- 用于重新验证的Program resource revision。

Later request只有在Program证明以下persistent条件后才可backfill：

> Borrower持续持有完整active reservation时，在同一donor set结束后，head仍可取得
> root/release-all-inactive plan。

不使用预计borrower先完成的temporal credit。任一resource transaction终态改变revision后，启动下一个
backfill前重新验证。普通decode frontier推进不改变global resource revision。

---

## 7. Worker boundary与GPU调度

### 7.1 单一mutation owner

只有Engine worker修改RequestRecord模型状态、Scheduler、ResourceManager和Program。Memory introspection与
fail-all cleanup通过同一execution ownership串行。

每个boundary依次执行：

```text
expire/cancel waiting requests
  -> finish unique RunningTransaction when present
  -> resolve TerminalPending
  -> freeze active cancellation snapshot
  -> process boundary-cancelled active requests
  -> consume one invalidated admission check when fairness permits
  -> choose staged prefill or maximal compact decode
  -> run at most one GPU model unit
  -> commit Runtime, budget and output state
  -> publish events and observations
```

Materialization/capture transaction与terminal release不并发改变同一topology。Transaction进行时，既有active
requests只有在其mapping与reservation不受影响时才可执行；它们可以提交已经发出的model unit并进入
TerminalPending，但finish/discard和active cancellation release必须等待transaction结束。

新active若仍持有未结算State Fork，首个state-writing commit先解除source pin；在此之前Program拒绝启动下一
global resource transaction。该结算属于active entitlement内部的执行提交，不增加cleanup transaction种类。

### 7.2 Prefill与decode

Scheduler满足：

- 最多一个request处于staged prefill；
- prefill不能连续饿死已有decode；
- admission check与decode同时存在时使用明确公平门；
- decode round包含所有且仅包含当前`DECODE_READY` requests；
- inactive physical lanes不以padding rows加入batch。

Program接收compact `SequenceHandle[B]`和`RoundBudget[B]`。Lane是长期physical home，batch row只是当前
round ordinal。

### 7.3 Admission invalidation

Waiting queue变化、lane释放、prefill gate打开或resource transaction terminal会使admission check待处理。
普通decode、未完成prefill chunk、输出publication和stats不会触发candidate/catalog重扫。

`TemporarilyBlocked`后保持静默，直到上述事实之一变化。Accepted plan在同一worker boundary start；不把未pin
的choice跨boundary缓存。

### 7.4 Capture与placement

Placement只在具体Resume、Capture、Finish或admission pressure transition中改变。不存在独立周期性promotion/demotion
scan。

ActiveCapture只有在已提交模型frontier上创建checkpoint offer。若同一boundary同时出现terminal，或另一个resource
transaction仍open，capture offer被消费并跳过。Capture transaction未完成时，请求停留在记录的post-capture控制状态。

---

## 8. 模型输出事务

### 8.1 PendingBatch

Prefill finalization和decode都可返回move-only `PendingBatch`：

```text
rows[B]        frozen SequenceHandle membership
tokens         Program-owned ragged token view
row_counts[B]  licensed extent per row
row_stride     physical row stride
```

其中tokens只是provisional。MTP/DFlash可产生多个proposal，但Program在commit前必须能把所有persistent state
折叠到每行最终accepted prefix。

### 8.2 Preview、commit、publish

每个PendingBatch遵循：

```text
freeze cancellation snapshot
  -> OutputSession previews every row
  -> stage accepted token IDs
  -> Program::commit(PendingBatch, decisions)
  -> adopt row dispositions
  -> move successful terminal rows to TerminalPending
  -> finish/discard terminal rows
  -> commit GenerationBudget and Scheduler service state
  -> OutputSession::commit_preview
  -> publish per-request events
  -> complete responses
```

Program state、terminal resource result和Frontend state都提交后，consumer才看到输出。Preview失败时，
Engine回滚staged token IDs并以`abort_pending`消费整个frozen membership；不能逐row释放未决batch。

Aggregate模式不建立per-token event queue；Streaming模式才发布增量事件。Consumer mode在submit时固定。

### 8.3 Row decision

非取消row必须满足：

```text
1 <= accepted_tokens <= produced_tokens
nonterminal => accepted_tokens == produced_tokens
terminal    => accepted_tokens may be a produced prefix
```

取消snapshot row使用`accepted_tokens=0, terminal=true, cancelled=true`。Program返回：

| disposition | 含义 | Engine动作 |
|---|---|---|
| Active | committed continuation仍可执行 | 保持DecodeReady |
| Finishable | accepted state已提交 | 进入TerminalPending；无open resource transaction后finish |
| Cancelled | provisional state已回滚 | 无open resource transaction时release；否则下一boundary取消 |

Main/Backend KV frontier、Linear Attention/GDN state、RNG、anchor和speculative state在Program中作为一个
accepted frontier提交；Engine不分别更新。

### 8.4 Thinking cap与forced control

Thinking budget由Frontend按accepted model-origin tokens追踪。自然close、stop、总output/context limit和
cancellation优先；只有thinking仍打开且准确到达cap的nonterminal request进入`CONTROL_READY`。

Frontend在构造期编码规范control suffix。Control transaction先preview完整span，再由Program执行短
no-sample continuation，最后提交budget、Frontend和output。Forced tokens：

- 消耗generated-token、context和service-work；
- 不调用sampler；
- 不推进RNG或sampling occurrence counter；
- 提交后进入prefix identity；
- Program成功前不可见。

Sequence token ledger在startup按max context预留，control只做事务式suffix append，不复制完整历史。

---

## 9. Terminal、取消与错误

### 9.1 Finish与retention

成功终止的Active request先进入TerminalPending：

```text
ResourceManager selects retain or discard
  -> wait until no planned resource transaction is open
  -> Program atomically publishes complete checkpoint or releases sequence
  -> ResourceManager adopts the terminal result
  -> optional SessionIndex update by publication_order
  -> lane becomes Free
```

Catalog slot在active publication时已经预留，finish不临时搜索或分配逻辑storage。若retention因内部状态条件失败，
确定性fallback是release sequence；请求不能永久停在TerminalPending。

Checkpoint只有在State、Main/Backend KV、hidden/position和selected speculative state属于同一frontier时才
可发布。

### 9.2 Cancellation

Engine在GPU boundary观察cancellation；已经启动的unit先到稳定终态：

| 状态 | 动作 |
|---|---|
| Waiting | 无Program state，直接terminal output |
| Materializing | abort RunningTransaction，采用完整ResourceResult |
| Active/ControlReady | 无open resource transaction时进入TerminalPending并discard；否则冻结cancel intent |
| Frozen PendingBatch | 无open resource transaction时按cancelled decision提交；否则本unit按原snapshot提交，下一boundary取消 |
| 已commit尚未adopt | 先采用commit结果，再执行terminal路径 |

GenerationHandle abandonment只设置cancellation并完成`consumer_released`，不会从consumer线程调用Program。
每条请求的terminal preview恰好一次。

### 9.3 Request-local rejection

以下错误若发生在Program mutation前，只结束当前request：

- queue timeout或overload；
- waiting cancellation；
- represented input超过公开context contract；
- request planning拒绝输入；
- root/release-all-inactive证明该请求本身不可行。

### 9.4 Engine-wide failure

以下表示内部状态已不能安全继续：

- GPU unit或physical mutation开始后Program抛出且无法形成稳定abort result；
- PendingBatch layout/disposition不一致；
- capability owner/generation不匹配；
- checkpoint完整性、resource invariant或noexcept adoption被破坏；
- unresolved output/resource transaction无法线性消费。

Worker在execution ownership内：

```text
mark Engine failed and detach pending
  -> reset Scheduler
  -> Program aborts any resource transaction, then releases active lanes
  -> clear ResourceManager logical catalog/lanes
  -> complete active and waiting responses with error
```

并发执行路径发现错误时不得先局部释放active ownership；resource transaction的pins和destinations必须先被
Program终止。无open resource transaction时，执行路径可以直接做幂等的lane cleanup。不变量错误不得伪装为
cache miss或无限retry。

---

## 10. Physical execution与热路径

### 10.1 Fixed lanes、compact rows

Program在startup建立`max_concurrency`个control lanes。每轮映射为：

```text
compact row -> SequenceHandle -> physical lane -> block-table/state selectors
```

Active lanes可以稀疏，GPU launch仍使用精确`B`。Catalogued checkpoint不占active lane。

### 10.2 Paged KV

Growing Main/backend KV使用shared page pools和固定block-table matrices。Program维护page allocation、
reservation、mapping、valid/provisional frontier与COW。

一个GPU unit内mapping稳定。Unit前的Program boundary可在同一active reservation内materialize或truncate；
改变global capacity、inactive placement或checkpoint ownership的transition必须与其他resource transition串行；
需要分配、pressure或transfer时使用ResourceTransaction。Prefix reuse要求完整continuation，不能以KV page
match代替State验证。

### 10.3 Workspace与Vision

Program拥有一个startup-frozen workspace allocation。普通Text/MTP/DFlash/control/graph使用共同高水位；
Vision encode和固定handoff region复用同一backing，但生命周期保证handoff active时普通consumer不会覆盖它。

`MemorySummary.workspace`是唯一可相加容量；Vision layout/active extent只是其内部视图。

### 10.4 Exact-B CUDA Graph

Program为合法`B=1..max_concurrency`建立decode graph families。Graph key描述执行topology，不包含request
identity、page IDs或active lane集合。变化通过稳定buffers、row selectors、frontiers和block tables输入。

CUDA Graph executable的driver-owned内存没有Program-owned精确计数器。SequencePlan在startup
预留`cuda_graph_allowance_bytes`，Graph实例化或其他CUDA操作的真实分配失败直接终止启动；
不得用两次`cudaMemGetInfo`的process-global差值归属到单个Program，也不得由该差值
构造Program内存不变量。`available_after_startup_bytes`仅是完成启动后的全局可用显存快照。

### 10.5 Speculative backends

MTP与DFlash是Program内部closed schedules，不是第二套Engine。它们只改变unit内部provisional width和state
transition，不改变：

- round membership冻结；
- 每行output policy独立；
- persistent state一次commit；
- 只有accepted tokens推进budget；
- published continuation与accepted frontier一致。

### 10.6 普通decode开销

正常decode round不执行：

```text
catalog scan
pressure scan
physical snapshot synchronization
ResourceManager accounting update
background replica scan
```

Host observations由实际owner在已有boundary累计；统计不驱动额外结构遍历，也不根据总耗时差值反推阶段。

---

## 11. 端到端路径

普通非零输出请求的完整路径为：

```text
Gateway/product adapter
  -> Engine::prepare
  -> Frontend produces PreparedPrompt
  -> Engine::submit reserves outstanding capacity
  -> FIFO Waiting
  -> Scheduler selects request
  -> ResourceManager forms logical choice
  -> Program seals and runs resource transaction
  -> ResourceResult adoption
  -> Active Prefill
  -> zero or more prefill units
  -> PendingBatch preview/commit
  -> repeated maximal compact decode
  -> optional forced control
  -> TerminalPending
  -> serialized finish retain or discard transition
  -> optional checkpoint/session publication
  -> OutputSession commit and response_done
  -> consumer_released
  -> exactly-once outstanding capacity release
```

Prefix reuse只改变source、materialization和suffix prefill，不创建第二条请求生命周期。MTP/DFlash只改变Program
unit内部，不创建第二套调度或publication路径。

---

## 12. 架构检查表

一个Engine层变更只有在以下问题都有唯一答案时才完整：

1. 请求顺序是否仍只由Scheduler决定？
2. 逻辑cache policy是否仍只由ResourceManager决定？
3. 物理occupancy、reservation和reclaim是否只由Program真实stores决定？
4. Global resource topology mutation是否由Program串行，并在需要分配、pressure或transfer时使用ResourceTransaction？
5. Provisional model output是否在publication前完成Program、Frontend和terminal commit？
6. Active request的完成reservation是否始终保留到TerminalPending结束？
7. Cancel、stale plan、retention failure和Engine failure是否都有有限终态？
8. 普通decode是否仍不触发catalog、pressure或physical accounting工作？

任何需要跨层“补账”、猜测另一层状态或在失败后重放部分receipt的设计都违反这些边界。

---

## 13. 代码映射与相关权威

### 13.1 主要位置

| 职责 | 主要位置 |
|---|---|
| 公共Engine facade | `include/ninfer/engine.h`, `src/runtime/engine/engine.cpp` |
| RequestRecord与response latch | `src/runtime/engine/request_record.h`, `engine_core.h` |
| Scheduler | `src/runtime/engine/scheduler.h`, `admission_policy.*` |
| ResourceManager logical policy | `src/runtime/engine/resource_manager.h` |
| package-neutral contracts | `src/runtime/contract/types.h` |
| package registry/startup | `src/targets/registry.*`, `src/targets/<package>/` |
| Qwen3.6 Frontend | `src/targets/qwen3_6/impl/frontend/` |
| Qwen3.6 Program | `src/targets/qwen3_6/impl/runtime/` |
| physical primitives | `src/core/` |
| semantic CUDA Ops | `src/ops/`, `include/ninfer/ops/` |
| HTTP Gateway | `src/serve/` |

路径用于定位authority，不把当前文件拆分固化成外部API。

### 13.2 相关文档

- [资源调度与上下文缓存架构](resource-scheduling-and-context-cache.md)：资源公式、checkpoint、transaction、
  retention与session；
- [Paged KV Context Store](paged-kv-cache.md)：KV pool、page ownership、layout、frontier与consumer contract；
- [Op development](op-development.md)：Op正确性与性能准入；
- [Qwen3.6-27B model](qwen3.6-27b-model.md)和
  [Qwen3.6-35B-A3B model](qwen3.6-35b-a3b-model.md)：模型数学与持久状态；
- [Artifact container](artifact-container.md)、[storage layouts](storage-layouts.md)和
  [tensor formats](tensor-formats.md)：`.ninfer`格式；
- [CLI](../cli.md)、[HTTP serving](../serving.md)和[Performance](../performance.md)：外部行为与测量。
