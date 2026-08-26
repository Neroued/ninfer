# NInfer 资源调度与上下文缓存架构

**文档状态：** 当前实现合同

**目标模型：** 当前注册的 Qwen3.6/Qwen3.8 family variants

**目标硬件：** 单张 NVIDIA RTX 5090

**所属层级：** Engine / ResourceManager / Program

**目标场景：** 本地 Agent、小规模并发、长上下文、MTP/DFlash speculative decoding

本文定义 NInfer 的资源所有权、prefix reuse、Device/Host context storage、admission 与资源事务合同。
顶层请求生命周期和 worker boundary 见 [Engine 架构](engine-architecture.md)，物理 KV pool、page table
和 consumer contract 见 [Paged KV Context Store](paged-kv-cache.md)。

本文只定义长期语义。具体 allocator 数据结构、scratch 字节布局、索引实现和调优常数不属于架构合同。

---

## 1. 产品边界

当前产品只有一张 GPU、一个 resident model、一个 Engine worker 和启动固定的 \(1\ldots 8\) 个 active lanes。
请求不抢占；Scheduler 先决定哪个 waiting request 有资格尝试 admission，ResourceManager 不能因为另一个请求
具有更长或更热的 prefix 而替换它。

Prefix reuse 是降低重复 prefill 成本的资源策略，不是第二套请求调度器。它必须满足：

- cache miss、合法降级和合法逐出只影响性能，不改变模型语义；
- active request 一经发布便具有完成其最大合法上下文所需的资源保证；
- inactive cache 可以为新请求让路，active resource 不可被借走；
- State、Main KV、Backend KV 和 Host storage 分别受各自容量约束；
- 所有 target-specific identity、frontier 和 state 完整性判断都由 Program 完成。

---

## 2. 三层职责

### 2.1 Scheduler

Scheduler 只拥有请求顺序与时间公平性：

- FIFO head、protected-head 和 backfill；
- admission 尝试时机；
- staged prefill 与 compact decode membership；
- projected service work。

Scheduler 不持有 checkpoint、victim、physical page 或 allocator 信息。

### 2.2 ResourceManager

ResourceManager 只拥有逻辑 policy：

- `Free | Materializing | Active | TerminalPending` lane 状态；
- private/shared catalog 的逻辑可见性和预留 publication slot；
- SessionIndex、ContentPrefixIndex 和 retention observations；
- exact candidate shortlist；
- source disposition、logical degradation action 顺序和成本比较；
- Program 返回的 opaque handles 与稳定终态结果的采用。

ResourceManager 不保存物理占用账本，不保存 per-lane State/KV resource vector，不从 owner delta 推导容量，
也不镜像 Program 的 allocator 或 replica 状态。

### 2.3 Program

Program 是唯一物理 authority。它拥有：

- StateImage、KVAddressSpace 和 LogicalKVPage 的真实对象与引用；
- Device/Host State stores；
- typed Device KV pools 和 Host KV extent store；
- replica generation、content epoch、coverage 和 transfer state；
- active sequence binding 与完成 reservation；
- 物理 allocator、容量计数和阶段峰值检查；
- 同一时刻至多一个改变全局资源拓扑的 transaction；
- prefill、decode、speculative commit、capture 和 terminal state transition。

Program 的现有 stores、allocators 和引用关系本身就是物理事实。架构中的“资源图”是这些事实的概念模型，
不要求再维护一个与真实容器并列的 `PhysicalGraph` 对象。

整体调用方向为：

```text
Scheduler
    选择当前请求
        │
        ▼
ResourceManager
    枚举 exact candidates
    排序 logical degradation actions
        │
        ▼
Program
    从真实对象与 allocator 联合验证物理transition
    需要分配、pressure或transfer时seal ResourcePlan
    串行执行一个resource transition
        │
        ▼
ResourceManager
    只采用稳定的 logical ResourceResult
```

不存在从 Program 返回到 ResourceManager 的逐步物理账本同步环。

---

## 3. 核心资源模型

### 3.1 四个不能混合的概念

资源架构必须分别回答四个问题：

| 概念 | 问题 | 表示 |
|---|---|---|
| logical reference | 哪些 active sequence 或 checkpoint 需要该对象？ | Program-owned object references |
| protection/reservation | 哪些对象或未来容量当前不可回收？ | active sequence 与 open transaction 的内部记录 |
| physical occupancy | 哪些 Device/Host allocation 实际存在？ | Program stores 与 allocators |
| marginal reclaim | 删除一组 logical references 后真正释放什么？ | 对完整 hypothetical post-state 联合计算 |

Logical owner 不是物理字节的结算 owner。Shared page 从两个引用变为一个引用时，只改变 logical references；
它不发生物理“所有权转移”，也不产生等量 removed/added delta。

### 3.2 唯一占用公式

对每个资源维度 \(r\)，稳定或 transaction 内部状态 \(S\) 的物理占用为：

\[
Used_r(S)=
\sum_{x\in UniquePhysicalAllocations(S)} size_r(x)
+
\sum_{y\in ConcreteReservations(S)} size_r(y)
\]

其中：

- `UniquePhysicalAllocations` 包含 Device/Host State replicas、Device KV page replicas 和 Host KV extents；
- `ConcreteReservations` 包含 active future growth、COW/tail destination 和 transaction destination；
- 同一个容量单位恰好出现在一项中：已经materialize的部分计allocation，尚未materialize但被保护的部分计reservation；
- 同一 allocation 被多少 logical objects 引用都只计一次；
- lane、catalog slot、SessionIndex cell 不进入物理资源向量。

物理容量轴固定为：

```text
Device = state slots, typed KV page slots
Host   = state slots, KV bytes
```

对每个维度始终满足：

\[
Used_r(S)\le Capacity_r
\]

### 3.3 阶段峰值而不是最终净值

一个 `ResourcePlan` 是有序物理阶段：

\[
S_0\rightarrow S_1\rightarrow\cdots\rightarrow S_n
\]

它可执行当且仅当：

\[
\forall r,\qquad
\max_{0\le i\le n} Used_r(S_i)\le Capacity_r
\]

因此以下操作必须按实际顺序检查：

- copy destination 分配；
- D2H/H2D/D2D；
- replacement replica publication；
- old replica release；
- partial-tail COW；
- active growth reservation；
- Host extent allocation、split 和 coalescing。

最终 `Used(S_n)-Used(S_0)` 不能替代中间峰值。

### 3.4 联合可回收量

对完整 logical action 集合 \(V\)：

\[
Reclaim_r(S,V)=
Used_r(S)-Used_r(RemoveReferences(S,V))
\]

`RemoveReferences` 必须先在同一个 hypothetical state 中应用全部 source disposition、victim actions、
replica changes 和 destination references，再根据最后引用决定实际 release。

通常不成立：

\[
Reclaim(S,A\cup B)=Reclaim(S,A)+Reclaim(S,B)
\]

只有 Program 从不相交的实际 objects 证明可加时，结果才会偶然相等。ResourceManager 永远不依赖这种可加性。

典型边界为：

```text
删除 catalog alias
active sequence 仍引用同一 page

logical references changed
physical occupancy unchanged
active protection unchanged
reclaim = 0
```

### 3.5 一份资源 revision

Program 暴露单调 `resource_revision`，只用于使只读 plan 失效。下列稳定可观察事实改变时递增：

- owner/reference topology；
- Device/Host replica placement；
- global free/reserved capacity；
- 会改变后续 allocator feasibility 的几何状态。

Active 在已经取得的 reservation 内执行以下操作时不递增：

- reserved page 变为 mapped；
- truncate/rollback 后同一 page 变回 reserved；
- committed coverage 或 token frontier 前进；
- 不改变 global free set 的 state 内容更新。

一个 open transaction 存续期间不允许建立第二个 topology plan，因此内部阶段不需要向外发布 revision。
Transaction 到达稳定终态后，如果上述事实发生变化，只推进一次 revision。Capability generation 继续独立拒绝
slot reuse 后的 stale handle。

---

## 4. Continuation、Checkpoint 与恢复语义

### 4.1 State 决定能否恢复

对于当前混合 GDN/Full Attention targets：

- Full Attention KV 可以按 target 允许的 frontier 截断；
- recurrent/GDN state 不能从较晚状态无损回退；
- 只有某个 frontier 存在完整 target state，才可从该位置继续；
- 只有 KV 而没有匹配 state 不构成 prefix hit。

因此：

> Checkpoint 的 StateImage 证明该 frontier 可恢复；typed KV requirements 描述恢复该 frontier 还需要哪些分页前缀。

### 4.2 Continuation

`Continuation` 表示一条线性历史，而不是整体迁移的 bundle：

```text
PrivateContinuation
├── target identity / token ledger
├── Main KV address space
├── optional Backend KV address space
├── optional endpoint checkpoint
├── optional typed rewrite checkpoint
└── optional sparse long anchors
```

Endpoint、rewrite 和 long anchors 可以引用同一 KV address space 的不同前缀。Continuation 组织逻辑历史；
State 与 KV 的 Device/Host residency 仍独立。

Shared stable prefix 是不可变的独立 continuation source：

```text
SharedPrefix
├── exact checkpoint state
├── Main/Backend KV prefix
└── immutable shared full pages
```

### 4.3 Checkpoint

一个 checkpoint 包含：

```text
kind / scope
exact target identity
token frontier
complete StateImage handle
target-defined typed KV requirements
optional surviving fallback
```

当前逻辑种类为：

- `SessionEndpoint`；
- `TurnClosure`；
- `ResponseReplay`；
- `SharedStablePrefix`；
- `LongAnchor`。

`TurnClosure` 与 `ResponseReplay` 是 typed rewrite checkpoints。`TargetKVRequirement` 由 Program 定义，
Main 与 Backend 不要求具有相同数值 frontier。

Typed rewrite frontier 必须位于下一请求允许替换的 assistant suffix 之前，而不能位于该 suffix 内部。
`ResponseReplay` 位于 generation opener 之前；`TurnClosure` 位于当前 open turn 中第一个可能被重写的
assistant opener 之前。确定性的 opener/thinking prologue 属于低成本 rebuild suffix。这样立即后继请求无论回放
response、关闭 tool loop，还是直接追加 replacement user message，都保留同一个稳定 conversation prefix；每条
continuation 仍只需要一个 rolling rewrite StateImage。

Checkpoint 发布后逻辑内容不可变：

- identity 和 frontier 不变；
- StateImage 不再原地修改；
- required committed KV prefix 不被覆盖；
- Device/Host residency 可以变化。

### 4.4 有效与 Device-ready

Checkpoint \(c\) 有效，当且仅当：

1. StateImage 至少有一份完整有效 replica；
2. 每个 typed required KV page 至少有一份 content epoch 一致、coverage 足够的 replica。

\[
\forall s,\forall p\in RequiredPages_s(c):
ValidDevice(p,c)\lor ValidHost(p,c)
\]

只有 State 和全部 required KV 都在 Device，且 active future reservation 可以取得时，checkpoint 才是
Device-ready。有效 checkpoint 不等于可立即执行 checkpoint。

### 4.5 Exact identity

Exact identity 由 target Program 定义，可能包括：

- token IDs 和 token types；
- position/MRoPE axes 与 RoPE delta；
- Vision spans 和 media digest；
- template/runtime mode；
- checkpoint frontier。

SessionKey、request ID、raw string、caller marker 和 hash 都只是 shortlist 或 ownership hint。最终命中必须由
Program exact verification。Catalog 已绑定当前 Program/model instance，不增加 runtime target tag。

### 4.6 Request-level cache participation

每个请求在提交Engine前已经确定context-cache participation，只有两种语义：

- `ReadWrite`：可以枚举并精确验证prefix candidate，也可以在合法frontier capture并在finish发布continuation；
- `Disabled`：不读取candidate、不建立capture group、不发布continuation，finish必须释放该请求持有的active
  State/KV资源。

Serve启动配置决定外部请求采用哪一种语义；内部operational warmup固定采用`Disabled`。`Disabled`请求执行期间
仍取得正常的active completion guarantee，但它结束后不得留在private/shared catalog、SessionIndex或任何
Device/Host context replica中。因此warmup完成后不占用额外checkpoint容量`H`、Host容量`R`或catalog
容量，首个外部请求的cache选择只由外部请求历史决定。

---

## 5. State 物理语义

### 5.1 Device 与 Host pools

设 \(C\) 为最大 active concurrency，\(H\) 为额外 Device checkpoint state slots，\(R\) 为 Host state slots：

\[
Capacity_{DeviceState}=C+H
\]

\[
Capacity_{HostState}=R
\]

所有 Device state slots 同构，不固定划分 current/rewrite/cache，也不永久绑定 lane。每个 active sequence
需要独占可写 destination；多个 rows 可以共享 immutable source。

Host State 以完整 StateImage 为单位，不存在 partial state image。一个 logical StateImage 可以同时具有 Device
和 Host replica，两份 replica 分别计入各自容量。

### 5.2 Lane 与 state slot 解耦

Lane 表示 request control identity 和 Scheduler membership。State slot 表示 target continuation image：

```text
lane 0 -> state slot 4
lane 1 -> state slot 1
checkpoint A -> state slot 0
```

Program 内部 execution binding 为每个 row 提供：

```text
state_src_slot
state_dst_slot
```

普通执行 `src == dst`；Fork 时 `src != dst`。这些 selector 只属于 Program。

### 5.3 五个基本操作

#### InPlace

Active state 原地推进：

```text
read S0
write S0
```

这是普通 prefill/decode 热路径。

#### Move

Private checkpoint 被唯一 branch destructive consume：

```text
CHECKPOINT S0 -> ACTIVE S0
```

没有 D2D state copy。是否仍可 Move 必须在完整 victim/source disposition 后根据实际剩余引用判断。

#### Fork

Shared 或必须保留的 source：

```text
read immutable S0
write private S1
```

在 Program commit 确认 S1 已成为完整可继续执行的 target state 前，S0 保持 pin。Engine 不根据某次 kernel launch
或某个字段推进自行推断 Fork 完成。

#### Freeze

完整 committed active state 原子转成 immutable checkpoint。若请求继续执行，则下一 unit 从该 checkpoint Fork
到新 destination。优先使用 `Freeze + Fork`，不复制完整 state image。

#### Snapshot / Restore

完整 StateImage 在 Device 与 Host 之间 D2H/H2D。Private restore 可以直接成为 active；shared restore 可以建立
Device checkpoint replica 后反复 Fork。

---

## 6. KV 物理语义

### 6.1 Typed address spaces

当前 Variant 具有一个 Main KV pool，并在需要时具有独立 Backend KV pool。每个 pool 有独立：

- page namespace；
- logical frontier；
- physical capacity；
- Device/Host residency；
- materialization requirement。

一个 pool 的空闲容量不能补偿另一个 pool。

每个 continuation 在每个 pool 持有 Program-owned `KVAddressSpace`：

```text
KVAddressSpace
├── typed pool identity
├── committed frontier
└── ordered LogicalKVPageHandle[]
```

Catalog 只保存 opaque address-space/checkpoint handles 与 policy summary，不复制 page membership、replicas、
refcount 或 physical generation。

### 6.2 Page identity、content 与 replica

Logical page 至少具有以下可判定语义：

```text
object identity + capability generation
current content epoch
committed coverage
logical references / writer protection
optional Device replica
optional Host replica
transfer state
```

Replica 能覆盖 page 前 \(n\) 个 columns，当且仅当：

\[
replica.content\_epoch=page.content\_epoch
\quad\land\quad
replica.committed\_coverage\ge n
\]

正常 append 不改变已提交 prefix 的 content epoch，只推进 coverage。Speculative 或尚未提交的 bytes 不进入
canonical coverage，也不能被 checkpoint 引用。

### 6.3 Private append 与 shared COW

Private continuation 保持单 writer。Endpoint、rewrite 与 active frontier可以共享同一 address space 和 partial
tail，只要 writer 不覆盖任何 surviving checkpoint 的 protected prefix。

Shared full pages不可变。Shared frontier 位于 page 中间时，各 branch 必须：

```text
share immutable full pages
copy one private partial tail
append private suffix
```

是否需要 COW 由完整 post-state 中的实际 sharing/writer 关系决定，不能只依据某个 pre-plan refcount。

### 6.4 Host KV

Host KV 使用 packed logical-order extents，不保存临时 Device physical page IDs：

```text
Host extent
├── typed pool
├── logical begin/count
├── packed payload
└── allocation handle/generation
```

表示方式与传输方式分离：

```text
Host representation = logical-order packed
transfer execution  = coalesced physical runs
```

Host KV 使用有界 variable-size arena，短 continuation 只占实际 payload。可行性必须由真实 allocator 几何判断；
`free bytes >= request bytes` 不是充分条件。架构不规定 allocator 必须是红黑树、segment tree 或其他具体容器。

### 6.5 Partial residency

Checkpoint 可以同时具有：

```text
State: Device or Host
Main KV: prefix Device, suffix Host
Backend KV: independent placement
```

Pressure 只迁移或释放覆盖实际缺口所需的低价值 extents，不默认整体 park continuation。优先保留被更多
checkpoints 引用的早期 pages 和高 fan-out shared pages。

---

## 7. Active reservation 与 lane 生命周期

### 7.1 完成保证

Admission 为 selected candidate 取得：

```text
required Device checkpoint prefix
+ remaining prompt growth
+ maximum effective output growth
+ target-defined bounded provisional growth
+ required State/COW destinations
```

这些 reservation 由 Program 内部绑定到返回的 `SequenceHandle`。它们可以是具体 slots/pages，也可以是 allocator
内部等价的不可挪用 reservation，但必须从 global available capacity 中精确扣除一次。

另一个请求不能逐出或借用这些资源。Mapped 与 reserved-growth 之间转换不改变 total occupancy。

### 7.2 Lane 状态

逻辑 lane 状态只有：

```text
Free
Materializing
Active
TerminalPending
```

并满足：

\[
NonFreeLanes\le C
\]

- `Materializing`：已取得 lane 和 transaction destination reservation，但尚无 published active sequence；
- `Active`：RequestRecord 持有有效 `SequenceHandle`；
- `TerminalPending`：row 已退出 GPU membership，但仍持有 `SequenceHandle` 和完整 reservation，等待 terminal
  resource transaction；
- `Free`：无 sequence binding。

Lane 不进入物理资源向量。

### 7.3 Catalog publication capacity

任何可能发布新 private/shared descriptor 的 operation，在 Program 第一次 mutation 前必须由 ResourceManager
预留目标 catalog slot。Logical adoption 不允许在 Program commit 后分配或扩容。

Private Move 可以复用 source descriptor slot；root/Fork 需要预留 destination slot。Finish 保留 endpoint时复用
active private slot或预留替换位置；无法获得合法 publication capacity 时，optional capture跳过，terminal finish
释放而不是无限 blocked。

---

## 8. Candidate 与 pressure planning

### 8.1 Scheduler 先选择请求

Scheduler 先按 FIFO/protected-head/backfill 规则选择一个 request。ResourceManager 只为该 request 规划：

```text
session endpoint
session typed rewrite
explicit shared stable prefix
other matching shared prefixes
long anchors
root/full reset
```

ContentPrefixIndex 只生成 shortlist。相同 checkpoint 经多个 index/marker 命中时先按 capability identity 去重，
再由 Program exact verify。Root 永远存在且只出现一次。

### 8.2 完整 choice 是唯一规划单位

一个 plan 的输入必须同时固定：

```text
source checkpoint or root
source disposition: Move or Fork
logical degradation actions
replica placement actions
destination lane/state/KV reservations
optional catalog publication slot
```

Program 在同一个 hypothetical post-state 中联合决定：

- Move/Fork；
- last-reference release；
- State/KV Device/Host placement；
- partial-tail COW；
- Host/Device allocator feasibility；
- ordered physical stages；
- active completion reservation；
- terminal logical outcomes。

Program内部可以保存候选的阶段需求，但该值不越过Program边界。ResourceManager不能读取它，也不能把多个
owner的局部释放量相加后自行判定可行性。

候选在当前稳定状态中的初始缺口由Program逐维计算：

\[
Deficit_r(c)=\max\bigl(0, Used_r(S_0)+PeakAdditional_r(c)-Capacity_r\bigr)
\]

`PeakAdditional`已经包含候选自己的source disposition与阶段顺序。Pressure option只针对这个真实缺口生成；
不能用“所有维度都短缺”之类的哨兵替代，否则Host缺口会错误阻止Device到Host降级。

### 8.3 Canonical degradation sequence

Correctness不要求求解任意victim子集的全局最优组合。Program为每个owner返回有界、自包含的保留型
alternatives，例如：

```text
release redundant Device replica
move cold suffix/State to Host
drop one low-value anchor or endpoint suffix
```

每个alternative描述该owner的一份完整目标状态，不是供ResourceManager相加的delta。ResourceManager按以下稳定
policy key排序这些alternatives，并把完整owner eviction固定放在该owner最后：

```text
RetentionClass: Disposable < RecentPrivate < LiveSession < SharedStable
estimated future recovery loss
shared fan-out
last successful hit
stable owner/checkpoint ordinal
```

Protected source、active references和open transaction pins不进入序列。ResourceManager每次推进一个owner的下一
alternative；同一owner的新alternative替换其旧alternative，然后让Program从真实当前状态重新seal完整choice。
因此Program逐个联合评估：

1. no pressure；
2. canonical sequence产生的完整selected-action states；
3. maximal safe release：保留 selected source/protected objects，删除其余全部可释放 inactive cache。

即使单个action的standalone release为零，Program也必须从完整post-state重新计算，不能由ResourceManager跳过。
中间states用于提高保留价值，不承担admission correctness。

### 8.4 完备回退

每次 request planning 都额外检查：

```text
root + release all unprotected inactive cache
```

因此 cache policy 不会永久阻止一个本来能在空 Engine中运行的 request：

- request 的 isolated root completion requirement 已超过总/per-sequence容量：`PermanentlyInfeasible`；
- isolated root可行，但当前被active reservations、lane或open transaction阻塞：`TemporarilyBlocked`；
- 当前可行但需要transfer/pressure stages：`NeedsTransfer`；
- 当前已经Device-ready且可同步发布：`Ready`。

不对同一未变化的稳定状态重复轮询 `TemporarilyBlocked`；lane释放、resource revision变化、head protection变化或
open transaction结束后重新检查。

### 8.5 成本

每个可行 plan 的比较成本为：

\[
Cost(c,V)=
T_{state\ restore}
+T_{KV\ restore}
+T_{remaining\ prefill}
+T_{transition}
+Loss(V)
\]

对每个受影响 checkpoint \(i\)：

\[
Loss_i=
w_i\cdot
\max(0,
FutureRecoveryCost_i(after)-FutureRecoveryCost_i(before))
\]

`after` 必须来自联合 post-state：仍有效时使用其最佳 surviving replica path；被删除时使用 surviving fallback
加重建成本。\(w_i\) 来自 retention class、fan-out、hit/recency observations。权重和机器时间系数属于 policy
与离线校准，不参与资源正确性。

比较先使用预测 nanoseconds；精确相同时依次偏好：

```text
fewer destroyed SharedStable/LiveSession checkpoints
fewer transferred bytes/operations
more reused prompt tokens
stable candidate/action ordinal
```

不宣称 canonical chain 上的 minimum 是任意 victim subset 的全局 optimum。

---

## 9. Resource transition

### 9.1 两种执行形态

需要分配、pressure或异步transfer的transition使用三个概念：

```text
ResourcePlan            immutable, read-only, resource_revision-bound
RunningTransaction      Program-owned, at most one
ResourceResult          Committed or Aborted stable outcome
```

内部可以具有同步/异步 transfer stages，但不向 ResourceManager暴露逐阶段物理 receipt。

已经由active reservation完整覆盖、且不需要新分配或transfer的`Finish`、`Discard`和inactive release是
零阶段原子transition。它们在Program确认没有open transaction后同步返回稳定结果，不为“统一类型”制造空
`ResourcePlan`。两种形态共享相同的串行化、revision、完整结果与logical adoption规则。

### 9.2 Start

同一 worker 在任何 mutation 前完成：

1. 核对 plan 的 `resource_revision`；
2. 核对 source/victim/destination capabilities 与 generations；
3. 核对 ResourceManager 已预留的 logical adoption capacity；
4. 用当前真实 allocator重新确认 ordered stages 的容量和几何；
5. claim/pin source、victims 和依赖；
6. 取得 transaction destinations 与 active completion reservations。

任一步 stale 或不再可行时，Program 返回无副作用 rejection；ResourceManager撤销尚未发布的 logical claims并
fresh plan。成功 start 后 plan 被线性消费，不能换 candidate。

### 9.3 Progress

Transfer 可以同步完成，也可以跨 CUDA event：

- source、destination 和 release dependency始终被pin；
- copy完成前不发布 replacement replica；
- replacement发布后才可释放old replica；
- active sequences只使用自己已有的 reservations；
- 不启动第二个 topology transaction。

一个 long transaction 可以与既有active prefill/decode交错，只要两者的Program workspace/stream contract允许。
Active execution不得使用自身reservation之外的global available capacity；需要真正把reservation归还全局的
finish与capture等待ResourceTransaction。

### 9.4 Commit

Planned transaction commit是materialization/capture checkpoint的publication point，并把transaction带到稳定终态。它完成：

- 新 active/checkpoint binding；
- source disposition；
- committed victim demotion/eviction；
- transaction-only pin/reservation cleanup；
- 必要的 `resource_revision`推进；
- 一个稳定 `ResourceResult`。

`ResourceResult`只报告 ResourceManager需要采用的logical outcomes，例如：

```text
new SequenceHandle or checkpoint handle
source retained or consumed
which selected logical degradation actions committed
session/publication replacement result
terminal sequence released
```

它不携带供 ResourceManager重建物理占用的delta。Program可同时提供只读physical usage diagnostics，但该值不参与
ResourceManager correctness。

ResourceManager在Program commit前已预留所有adoption storage；commit后的logical adoption必须allocation-free且
noexcept。若Program outcome与预留logical alternatives不一致，进入Engine-wide failure，不能重新规划。

### 9.5 Abort

只有target active/checkpoint publication前可以request-local abort。安全完成的victim demotion/deletion stage
可以已经物理生效；Abort保证：

- selected source仍然有效；
- 不存在半发布active/checkpoint；
- transaction destinations、unused reservations和pins被释放；
- 返回一个稳定logical outcome。

已经完成并用于capacity preparation的合法 victim demotion/eviction不要求回滚；在transaction期间相应catalog
entries保持claimed/hidden，最终由一次 `ResourceResult`明确哪些变化已提交。由于没有ResourceManager物理镜像，
不存在逐progress同步或回放physical delta。

CUDA状态不可信、Program内部引用/allocator invariant失败或physical publication后无法adopt时进入Engine-wide
cleanup/failure，不能伪装成cache miss。

---

## 10. Cancellation、capture 与 terminal

### 10.1 Waiting 与 pre-start cancellation

Waiting request尚未取得physical resources。Start前观察到cancel时，撤销logical destination slot/claims，不调用
Program，不改变resource revision。

### 10.2 Materializing cancellation

Open Resume transaction在publication前cancel时执行Abort。若Program已经commit，则先采用active publication，
下一boundary再把它转为terminal discard；不能把已commit结果倒退成pre-start cancellation。

### 10.3 Active terminal

Active row在model/output commit确定结束后进入`TerminalPending`：

- 退出后续GPU membership；
- 继续持有 `SequenceHandle`及其全部reservation；
- 冻结`Finish`或`Discard`决定；
- 等待当前open transaction结束。

Worker优先结算最早的terminal pending row，然后才启动新的capture/admission。Finish可以保留新endpoint，也可以
release；若保留方案在一次fresh plan中不可行，必须选择release，不能永久阻塞lane。Discard不发布checkpoint。
Open transaction期间的active cancellation同样只记录意图；已发出的GPU unit可以提交到稳定active或
`TerminalPending`，但不能在transaction中途释放reservation。

### 10.4 Capture

Capture只从完整committed state发布。需要稳定frontier时，row在capture transaction结束前暂停继续推进。

Optional capture无法获得catalog/State/KV容量或另一个resource transaction已经open时直接skip。一个execution
result同时出现capture offer和terminal/cancel时，terminal优先，offer直接丢弃；不存在随后对stale frontier重试的
capture。

### 10.5 Fork 结算

State Fork 的source pin不是独立cache任务。首个真正写入destination的execution commit原子关闭Fork并解除pin；
这是已取得active entitlement内部的状态提交，不建立第三种resource transaction。任何active Fork尚未关闭时，
Program拒绝启动新的global resource transaction。因而不存在一边执行Fork settlement、一边按旧pin拓扑执行
materialization/capture plan的窗口，也不需要后台或pending cleanup协议。

---

## 11. Session 与 publication order

SessionKey只属于ResourceManager。Program不读取SessionKey，也不以它证明prefix identity。

每个允许更新SessionIndex的request在RequestRecord建立时取得单调`publication_order`。Session entry只需要：

```text
current binding handle/revision or none
last adopted publication_order
optional exact request that currently claimed the binding
```

规则为：

- 命名source只有incoming持有相同SessionKey且允许更新时才可destructive Move；否则Retain/Fork；
- 匿名content-matched private source可以Move；
- finish结果只有在自己的order新于当前last adopted order时才替换binding；
- 较旧request晚完成时，其checkpoint转成anonymous cache或按policy释放，不能复活旧binding；
- Move abort只恢复自己exact claim的former binding；若它已不再是current claim，则former object按anonymous policy处理；
- `update_session_index=false`不创建writer，也不改变现有binding。

一份publication order加capability generation足以处理乱序完成与slot reuse；不需要在Program与ResourceManager之间
复制SessionKey或建立多套writer/terminal epochs。

---

## 12. Protected head 与 backfill

当FIFO head的isolated root plan可行、但当前被active reservations阻塞时，Scheduler建立protected-head状态：

1. 按projected remaining service work和submission order排列当前active lanes；
2. 对 successive donor prefixes \(F\)，让Program联合模拟这些active protections解除、其terminal cache结果可被pressure；
3. 找到第一个能为head构造root/full-release plan的 \(F\)；
4. 保存head request generation、donor generations和当前resource revision。

Later request只有在Program对其完整post-state证明以下条件时才可backfill：

> Borrower持续持有其完整active reservation时，在同一donor set \(F\)结束后，head仍能取得root/full-release plan。

不使用“borrower预计先完成”的temporal credit。任何ResourceTransaction稳定终态改变resource revision后，在启动下一
backfill前重新验证protected-head状态。普通active frontier推进不使它失效。

该检查由Program对完整hypothetical state执行；Scheduler和ResourceManager不保存per-lane physical charge或
active unlock bucket表。

---

## 13. Retention 与 pressure

RetentionClass只有：

```text
SharedStable
LiveSession
RecentPrivate
Disposable
```

Active、selected source、transaction victim dependency和pending terminal是临时protected状态，不是第五种
retention class。

典型pressure顺序为：

### Device State

1. 释放已有Host replica的低价值Device duplicate；
2. 将cold checkpoint state下沉Host；
3. 删除被替换或Disposable checkpoint；
4. 删除低价值anchor/rewrite；
5. optional capture放弃。

### Device KV

1. 释放zero-reference pages；
2. 释放已有Host replica的低价值Device duplicate；
3. 下沉private endpoint在rewrite之后的suffix；
4. 下沉cold private pages；
5. 最后处理高fan-out shared pages。

### Host State/KV

1. 删除zero-reference或Device已有完整替代的duplicate；
2. 删除被替换/Disposable objects；
3. 删除低价值anchor/private suffix；
4. 删除接近surviving fallback的checkpoint；
5. 最后删除LiveSession或SharedStable。

若释放replica会使checkpoint失去最后一份完整State或required KV coverage，同一logical action必须删除该checkpoint，
或先发布满足要求的replacement。

不存在周期性background promotion/demotion scan。Placement改变只来自具体Resume/Capture/Finish或admission
pressure plan。

---

## 14. 关键边界

| 情况 | 唯一处理规则 |
|---|---|
| 删除一个shared alias | 只删reference；不是last reference则occupancy和reclaim均不变 |
| 多个victims共同形成last reference | 在联合post-state中只释放一次 |
| Move还是Fork | 应用完整source/victim actions后按surviving references决定 |
| Device copy-before-release | destination、copy、publication、old release逐stage检查peak |
| Host free bytes足够但碎片化 | 真实allocator exact preflight决定可行性 |
| Active跨页增长 | 从自身reservation映射，不改变global available capacity |
| Active truncate/rollback | page回到同一sequence reservation，不直接归还global pool |
| Active仍有未结算State Fork | 先完成首个state-writing commit；Program拒绝启动global resource transaction |
| Long transaction期间row完成 | 进入TerminalPending并保留SequenceHandle |
| Long transaction期间active被cancel | 冻结cancel intent；不得从commit/abort路径提前释放reservation |
| Resume publication前cancel | Abort并保留source；已完成victim changes由终态结果采用 |
| Program已commit后cancel | 先采用commit，再走terminal Discard |
| Capture与terminal同时出现 | terminal获胜，capture offer丢弃 |
| Active到达capture frontier但另一个transaction仍open | 消费并skip optional capture；不启动第二个transaction |
| Finish无法保留checkpoint | release sequence，lane必须有限步回到Free |
| 同Session请求乱序finish | 只允许较新的publication_order更新binding |
| 只有KV没有State | 不是hit，回退到更早checkpoint或root |
| Cache占满导致admission失败 | 检查root+release-all-inactive；不把cache policy错误解释成永久不可行 |

---

## 15. 概念接口

以下接口描述模块边界，不规定具体C++类型名：

```cpp
RequestBasePlan describe_request(...);
CandidateMatch inspect_checkpoint(...);
ResourceAssessment assess(...);
ResourcePlan seal(...);

StartResult start(ResourcePlan&&, PreparedLogicalAdoption&&);
ProgressResult progress(RunningTransaction&);
ResourceResult commit(RunningTransaction&&);
ResourceResult abort(RunningTransaction&&);

TerminalResult finish_or_discard(SequenceHandle&&);
```

`inspect_checkpoint`、`assess`和`seal`无副作用。`ResourcePlan`是不透明的Program-owned value，
不得向ResourceManager暴露raw allocation id、page id、slot id或allocator内部结构。

`start`先验证plan引用的resource revision、source和victim assumptions；失效必须在任何物理修改前返回。成功后，
Program独占`RunningTransaction`，ResourceManager只保留逻辑lane状态和预分配的adoption storage。

`progress`推进有界物理工作，不生成供ResourceManager重放的physical receipt。`commit`或`abort`只返回一个稳定的
`ResourceResult`，其中包含ResourceManager必须采用的逻辑终态：

- source/target checkpoint的保留或发布；
- 已提交victim degradation或deletion；
- lane下一状态；
- terminal/capture处理结果；
- 诊断所需的plan与成本摘要。

ResourceManager对终态的采用必须noexcept、无分配、无容量判断；相应catalog slot、victim result storage和lane transition
storage在`start`前已经准备完毕。

---

## 16. 正确性不变量

以下不变量同时约束实现、测试和后续文档：

1. Scheduler先选择请求；资源层只能选择该请求的source、placement与pressure actions。
2. Program中的State/KV stores、allocators、references和reservations是唯一物理权威。
3. ResourceManager不维护第二份physical ledger、snapshot mirror或per-lane charge。
4. Lane ownership、logical cache identity与physical placement彼此独立。
5. Occupancy按unique physical allocations和concrete reservations计数，不按owner/reference重复计数。
6. Reclaim对完整victim set的联合post-state计算，不相加独立owner delta。
7. Plan只有在每个有序stage的peak均满足容量和allocator约束时才可sealed。
8. Active sequence的未来增长reservation不可被其他请求撤销或借用。
9. Active在自身mapped/reserved之间转换不改变global available capacity，也不推进resource revision。
10. 每个active continuation恰有一个mutable state writer。
11. Published checkpoint State和受保护KV内容不可被active执行原地修改。
12. Shared full KV page只读；tail写入在共享前必须COW。
13. Checkpoint恢复要求完整State和所有必需typed KV coverage。
14. KV、session/hash命中或位置标记本身均不能构成有效checkpoint hit。
15. 只有已提交State transition对应的tokens才能发布为canonical continuation。
16. 被拒绝的speculative bytes不能扩展canonical KV coverage。
17. Resume source在transaction commit或abort之前保持有效。
18. 任何会删除最后可恢复副本的替换都必须先完成replacement publication。
19. 一个materialization只属于一个Program topology transaction。
20. Transaction进行时，active execution只能使用既有active reservations，不得依赖transaction临时资源。
21. 未结算State Fork与global resource transaction不能同时存在。
22. Pre-start stale rejection无物理副作用；start成功后不得静默改换candidate。
23. Program commit后，ResourceManager adoption已经预分配且不能失败。
24. Abort不会留下半active target，但可保留已经安全提交的victim degradation；这些变化必须出现在终态结果中。
25. TerminalPending持续持有`SequenceHandle`及其完整reservation。
26. Finish retention失败时必须释放sequence，并让lane有限步回到Free。
27. Session binding只接受更大的`publication_order`，capability generation只验证持有者资格。
28. Root加“释放全部inactive cache”的plan是admission完整性回退。
29. Engine-wide failure先abort open resource transaction，再释放active lanes；异常catch不得颠倒顺序。
30. 不变量破坏是内部错误，不得降级成普通cache miss或silent retry。

---

## 17. 性能原则与成本模型

普通decode round满足：

```text
state_src == state_dst
no catalog scan
no pressure scan
no physical snapshot synchronization
no ResourceManager accounting update
```

因此热路径只执行模型计算、已分配KV写入和必要的token publication。State copy只发生在Move/Freeze/Fork或显式snapshot；
KV transfer只搬运选定checkpoint缺失的pages。

Resource planning只发生在admission、resume、capture和finish边界。Program可为这些边界维护O(1) counters
和可重建索引，但它们必须由真实stores同步更新，不能成为第二权威。所有bounded storage按启动配置一次分配。

现有统一估算形式保留：

```text
transfer =
    max(batch_ns + copies * operation_ns,
        bytes * ns_per_byte)

prefill =
    chunks * chunk_ns
  + suffix_tokens * token_ns
  + attention_pairs * attention_pair_ns
  + vision_work
```

其中：

\[
attention\_pairs = B S + \frac{S(S+1)}{2}
\]

其中\(B\)是复用prefix tokens，\(S\)是待prefill suffix tokens。

模型参数是启动时不可变的实现配置。编译内置default profile是权威默认值；可选preset只有在目标、量化、硬件和相关运行
配置完全匹配时才能替换它。成本模型影响choice排序，不改变可行性判定。

---

## 18. 配置与有界性

公开配置继续使用`ContextCacheOptions`中的现有容量维度：

| 配置 | 默认值 |
|---|---:|
| Extra Device checkpoint State slots \(H\) | \(C\) |
| Host State slots \(R\) | 8 |
| Host KV budget | 8 GiB |
| Private continuations \(P\) | \(2C\) |
| Shared prefixes \(S\) | \(C\) |
| Long anchors per private continuation \(L\) | 2 |
| Cache markers per request \(M\) | 4 |

这些是默认值，不是新的上限。显式配置只受已有产品并发上限 \(1 \le C \le 8\)、整数/地址可表示性、启动时真实内存分配
和输入复杂度约束。实现不得把默认关系误写成`P <= 2C`、`S <= C`、`L <= 2`或`M <= 4`。

Device State总量是\(C+H\)：前\(C\)个是active guarantee，后\(H\)个是全局Device-checkpoint capacity，
不是每lane固定配对。Host State按image计费，Host KV按actual extent bytes计费。

缓存关闭时解析为root-only：\(H=R=0\)、Host KV bytes为0、\(S=L=0\)、\(P=C\)；成功请求不发布
continuation，\(M\)仍限制输入复杂度。Active reservation和正确执行所需资源仍然存在。

Program按真实配置构造所有固定上限storage。若配置无法安全表示或启动分配失败，构造直接失败；运行时不得偷偷缩小容量、
切换另一套调度算法或依赖动态扩容。

---

## 19. 可观测性

诊断数据从Program的物理权威和ResourceManager的逻辑权威分别导出；不得为了观测建立第二份可写physical ledger。
边界事件至少能够说明：

- 被选request、candidate、ready/degraded状态；
- sealed plan、各stage peak、pressure actions与成本摘要；
- State操作、KV transfer/COW和实际字节数；
- transaction start/commit/abort及resource revision变化；
- victim degradation、checkpoint publication/deletion；
- TerminalPending、capture、finish和Fork结算结果；
- 当前Program物理占用、concrete reservations与allocator失败原因；
- cache hit层级和避免的prefill工作量。

`RuntimeStats::terminal_pending_requests`在worker boundary计数已退出GPU membership、但仍持有完成reservation等待
当前resource transaction收口的rows；它只用于诊断，不参与调度。

普通decode不刷新结构性资源gauge。结构性统计只在resource revision或明确的统计采样边界读取；日志不得改变调度或分配时序。

---

## 20. 结论

整个架构归结为五条：

1. ResourceManager决定“什么值得保留、尝试哪个恢复方案”，Program决定“物理上能否做到以及如何原子完成”。
2. 容量只由unique allocations、concrete reservations和逐stage peak决定。
3. 共享资源的释放只由完整post-state中的last-reference事实决定。
4. 每次改变global available capacity、inactive placement或active/checkpoint ownership的操作都由Program串行；
   需要分配、pressure或transfer的操作走`ResourcePlan -> RunningTransaction -> ResourceResult`，已经由active
   reservation覆盖的terminal/release走零阶段原子transition。
5. Active completion、terminal release与session publication都有不可被cache policy破坏的确定性终点。

实现不需要owner vector结算、ResourceManager物理镜像、per-call execution receipt、permit网络、allocator digest或固定arena phase。
这些结构既不增加上述语义，也不能替代Program对真实post-state和逐stage峰值的验证。
