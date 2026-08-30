# NInfer 资源调度与上下文缓存

本文定义 NInfer 在有限 Device/Host 资源下选择、验证并提交上下文状态的规则。它是 admission
materialization、prefix reuse、cache retention 和 pressure planning 的维护者权威。

本文解决的问题是：

> Scheduler 已经选择一个等待请求后，应从哪个可恢复位置开始，保留或降级哪些 inactive context，
> 才能以最低预测代价得到一个物理可执行且不会破坏 active request 完成保证的终态？

顶层请求顺序和生命周期由 [Engine 架构](engine-architecture.md)定义。KV page、address space、
replica 和 consumer view 的物理合同由 [Paged KV Context Store](paged-kv-cache.md)定义。

---

## 1. 术语与边界

本文使用以下术语：

| 术语 | 含义 |
|---|---|
| owner | 一个 active sequence、private continuation 或 shared prefix |
| checkpoint | owner 内某个可恢复 frontier 的不可变完整状态 |
| candidate | incoming request 可以采用的一个 root 或 exact checkpoint source |
| planning target | candidate 与全部受影响 inactive owners 的完整逻辑终态 |
| reservation | 已从全局可用容量中扣除、为未来物化或事务目的保留的容量 |
| resource plan | Program 针对一个 target seal 的有序物理 transition |
| placement | 一个完整 StateImage 或一个 KV logical page 的 Device/Host replica 状态 |

本文后续简称 planning target 为 `target`；注册模型及其 package 写作 `model target`，两者不是同一概念。

Prefix reuse 是 admission 的一种来源选择。它减少重复 prefill，但不改变模型语义、请求顺序或生成结果。

当前产品条件为单 GPU、单 resident model、固定 `max_concurrency=1..8` 和非抢占 active requests。
由此得到两个基本规则：

1. Scheduler 先确定本次尝试的 request，资源层只优化该 request 的 materialization；
2. 一个 request 发布为 Active 后，其最大合法执行范围已经获得完整资源保证，inactive cache policy
   不能再借用这部分容量。

---

## 2. 决策所有权

### 2.1 Scheduler

Scheduler 选择：

- 当前 FIFO head 或已经证明安全的 backfill request；
- admission 尝试时机；
- prefill、control 和 decode 的执行顺序。

Scheduler 不读取 checkpoint placement、victim、page 或 allocator facts。

### 2.2 ResourceManager

ResourceManager 拥有逻辑缓存策略：

- private/shared catalog 与 SessionIndex；
- candidate shortlist 和 logical claim；
- retention class、命中历史与 publication order；
- 跨 candidates 的 target 搜索和成本比较；
- Program 终态结果的逻辑采用。

ResourceManager 可以知道“某个 checkpoint 可从 Device、Host 或 root 恢复”的 Program summary，
但不保存物理 occupancy、refcount、free page 或 Host arena 几何的副本。

### 2.3 Program

Program 是唯一物理 authority，拥有：

- State 与 KV 的真实 objects、references 和 replicas；
- Device pools、Host stores、allocators 和 reservations；
- continuation completeness 与 exact prefix identity；
- Move、Fork、COW、transfer 和 last-reference release；
- 完整 post-state、每个 transition stage 的物理可行性；
- 与当前资源状态绑定的 `ResourcePlan`。

调用关系为：

```text
Scheduler selects one request
        │
        ▼
ResourceManager enumerates candidates and compares targets
        │
        ▼
Program projects each complete target from real physical state
        │
        ▼
Program seals and executes one ResourcePlan
        │
        ▼
ResourceManager adopts one stable ResourceResult
```

逻辑策略与物理事实通过 candidate assessment、opaque target handle、`ResourcePlan` 和
`ResourceResult` 交互，不通过一份跨层共享的物理账本交互。

---

## 3. 资源模型

### 3.1 容量轴

Program 在启动时建立一个固定容量向量：

```text
PhysicalCapacity = {
    Device StateImage slots,
    Host StateImage slots,
    Device page groups for each typed KV pool,
    Host KV arena bytes and allocator geometry,
}
```

这些轴相互独立。一个资源轴的余量不能补偿另一个轴的缺口。Host KV 的总空闲字节也不能替代
allocator 对具体 extent 几何的可分配性判断。

Engine 另有固定的逻辑容量：

- active lanes；
- private continuation catalog slots；
- shared prefix catalog slots；
- 每个 continuation 的 checkpoint/anchor 上限。

Lane 和 catalog slot 不计入物理 bytes，但 target 只有同时满足逻辑容量和物理容量才可发布。

### 3.2 唯一物理占用

对任一物理资源维度 \(r\)，状态 \(S\) 的占用为：

\[
Used_r(S)=
\sum_{x\in UniqueAllocations(S)} size_r(x)
+
\sum_{y\in ConcreteReservations(S)} size_r(y)
\]

`UniqueAllocations` 包括：

- Device/Host StateImage replicas；
- Device KV page replicas；
- Host KV extents。

`ConcreteReservations` 包括：

- active request 尚未 materialize 的未来增长；
- partial-tail COW destination；
- resource transition 的 copy destination 和 replacement capacity。

同一个物理 allocation 被多个 checkpoints 或 address spaces 引用时只计一次。已经物化的容量计
allocation，尚未物化但被保护的容量计 reservation；同一容量单位不能同时计入两项。

每个稳定状态和 transition 中间状态都必须满足：

\[
Used_r(S)\le Capacity_r
\]

### 3.3 完整 post-state 与联合回收

资源回收由最终剩余引用决定。对一组 logical actions \(V\)：

\[
Reclaim_r(S,V)=
Used_r(S)-Used_r(PostState(S,V))
\]

`PostState` 必须同时应用：

- selected source 的 retain 或 consume；
- 所有 victim owner/checkpoint outcomes；
- 所有 surviving Device/Host placements；
- active destination references 和 reservations；
- 零引用 objects 的 release。

因此通常有：

\[
Reclaim(S,A\cup B)\ne Reclaim(S,A)+Reclaim(S,B)
\]

删除一个 alias 可能不释放任何 allocation；两个 owners 一起删除最后一批 references 时，同一 allocation
只释放一次。ResourceManager 不计算或相加 per-owner reclaim credit。

Move、Fork、partial-tail COW 和 source placement 也由完整 post-state 决定。一个 source 在初始状态中
存在多个引用，并不意味着最终一定 Fork；反之，保留任一 checkpoint 所需的引用后也不能把它当作 Move。

### 3.4 有序阶段峰值

一个物理 plan 是稳定状态之间的有序阶段：

\[
S_0\rightarrow S_1\rightarrow\cdots\rightarrow S_n
\]

它可执行当且仅当：

\[
\forall r,\qquad
\max_{0\le i\le n} Used_r(S_i)\le Capacity_r
\]

阶段顺序必须表达真实 dependency：

```text
reserve destination
  -> copy
  -> verify and publish replacement
  -> release old replica
```

State restore、KV H2D/D2H、D2D fork、partial-tail COW 和 Host extent replacement 都遵循这一规则。
最终净占用不能掩盖 copy-before-release 的瞬时峰值。

### 3.5 Resource revision

Program 暴露单调 `resource_revision`，使只读 assessment 和 sealed plan 与一个稳定物理状态绑定。
以下变化会推进 revision：

- owner/reference topology 改变；
- inactive Device/Host placement 改变；
- global free/reserved capacity 改变；
- 会改变后续 allocator feasibility 的几何状态改变。

Active request 在自己的 reservation 内进行 mapped/reserved 转换、推进 committed frontier 或更新
state 内容时，不改变全局可用容量，因此不推进 revision。

一个 open resource transition 的中间阶段不向外发布 revision。它到达稳定 commit/abort 状态后，
若全局物理事实发生变化，只推进一次。Handle 的 owner/generation validation 独立阻止 slot reuse
产生的 stale capability。

---

## 4. Continuation 与 checkpoint

### 4.1 完整恢复条件

当前 model targets 同时包含可分页 Full Attention KV 和不能从任意较晚状态无损回退的 recurrent state。
因此某个 frontier 可以复用，当且仅当 Program 能证明：

1. 存在该 frontier 的完整 StateImage；
2. Main KV 满足 target 定义的 typed coverage；
3. selected backend 的 KV 与 fixed state 满足同一 continuation；
4. token、position、Vision 和 mode identity 与 incoming prompt 精确一致。

只有 token match、KV bytes 或 page match 时，缺少的是完整 continuation，不是一次部分 cache hit。

### 4.2 Continuation 组织

Private continuation 表示一条可继续演化的会话历史：

```text
PrivateContinuation
├── exact target/token history
├── Main KV address space
├── optional backend KV address space
└── one or more immutable checkpoints
```

Shared prefix 是不可变的复用来源，可以被多个 private branches Fork。一个 continuation 内的多个
checkpoints 可以引用相同 address space 的不同 prefix，但每个 checkpoint 都有自己的完整 StateImage
identity。

### 4.3 Checkpoint 种类

| Kind | 语义 |
|---|---|
| `SessionEndpoint` | 已完成请求的最新可继续状态 |
| `TurnClosure` | 当前 turn 中可替换 assistant suffix 之前的稳定状态 |
| `ResponseReplay` | generation opener 之前、可重新生成 response 的稳定状态 |
| `LongAnchor` | retention policy 选择的较早长上下文恢复点 |
| `SharedStablePrefix` | 多条历史共同使用的不可变稳定前缀 |

`TurnClosure` 和 `ResponseReplay` 的 frontier 必须位于下一请求可能替换的 assistant suffix 之前。
这样对话系统追加新的 user message 或改写生成尾部时，仍能复用稳定历史，而不会把可变 suffix 错当成
prefix identity 的一部分。

Checkpoint 发布后：

- identity、frontier 和 required coverage 不变；
- StateImage 与受保护的 KV prefix 不再原地修改；
- Device/Host placement 可以由 resource transition 改变。

### 4.4 Valid 与 Device-ready

Checkpoint \(c\) 有效，当且仅当：

- StateImage 至少有一个完整 Device 或 Host replica；
- 每个 required KV logical page 至少有一个 content epoch 一致且 coverage 足够的 Device 或 Host replica。

\[
\forall p\in RequiredPages(c),\qquad ValidDevice(p,c)\lor ValidHost(p,c)
\]

有效 checkpoint 可能需要 H2D restore。只有全部 execution requirements 已在 Device，且 active future
reservation 可以取得时，它才是 Device-ready。

### 4.5 Exact identity

Program 的 exact verification 可以使用：

- token IDs 和 token types；
- position/MRoPE axes 与 RoPE delta；
- Vision spans 和 media digest；
- template/runtime mode；
- checkpoint frontier。

Session key、marker、hash 和 prefix index 只缩小 candidate 集合，不证明命中。

### 4.6 Request cache participation

每个请求在进入 Engine 前固定一种语义：

- `ReadWrite`：允许读取 exact candidates、capture checkpoints，并在 finish 时发布 continuation；
- `Disabled`：只允许 root materialization，结束时释放全部 active context resources。

`Disabled` 请求仍获得正常 active completion guarantee，但不会留下 private/shared catalog entry、
SessionIndex binding 或 Device/Host checkpoint replica。Serve warmup 使用这一语义。

---

## 5. State 与 KV 的 placement 语义

### 5.1 StateImage

StateImage 是完整迁移单位。设最大 active concurrency 为 \(C\)，额外 Device checkpoint slots 为 \(H\)，
则：

\[
Capacity_{DeviceState}=C+H
\]

前 \(C\) 份容量形成 active guarantee，全部 slots 仍由同一个 Program pool 管理，不与 lane 固定配对。
Host State 以完整 image 和独立 slot 容量计费。

State transition 的语义只有：

| 操作 | 条件与结果 |
|---|---|
| InPlace | 唯一 active writer 在自己的 destination 上推进 |
| Move | private source 被 consume，原 StateImage 成为 active destination |
| Fork | immutable 或仍需保留的 source 写入新的 private destination |
| Freeze | committed active StateImage 转为 immutable checkpoint |
| Snapshot/Restore | 完整 StateImage 在 Device 与 Host 之间复制 |

Program 根据完整 post-state 选择 Move 或 Fork。Fork source 在 destination 首次 state-writing commit 完成前
保持 pin；该结算属于 active model transaction，不能与新的 global resource transition 重叠。

### 5.2 Typed KV

每个 continuation 对 Main KV 及 selected backend KV 分别持有 typed address space。各 pool 的 frontier、
capacity 和 placement 独立。

资源规划只使用以下语义：

- required logical pages 与 committed coverage；
- 每页可用的 Device/Host replica；
- immutable references 和唯一 writer；
- partial-tail COW requirement；
- restore、demote 或 release 对 capacity 与 machine work 的影响。

具体 page geometry、Host packed extent、COW 和 block-table publication 由
[Paged KV Context Store](paged-kv-cache.md)定义。

Prefix Fork 后的 active address 可以按以下顺序同时包含共享与私有页：

```text
immutable full pages | unique mutable full pages | optional unique mutable tail
```

Active capture 对不可变整页只增加引用，对唯一可写整页原地 Freeze；非对齐尾页保留给 checkpoint，并只
复制一页作为新的 active writer。对齐 frontier 不发生 KV copy，后续增长再物化新的 writer page。
Program 的 capture assessment、reservation 和 commit 使用同一个 snapshot-shape 分析，因此资源信用、
transfer cost 与实际可提交的页所有权条件一致。

State 与 KV 可以独立 placement，例如 State 在 Device、Main KV 部分在 Device、其余 Main KV 在 Host，
backend KV 采用另一种 coverage。Checkpoint 只有在全部 required components 至少保留一份有效 replica
时才继续存在。

---

## 6. Active completion guarantee

### 6.1 Admission reservation

一个 candidate 发布为 Active 前，Program 为它保障：

```text
required Device checkpoint prefix
+ remaining prompt growth
+ maximum effective output growth
+ target-defined provisional/speculative growth
+ State and partial-tail COW destinations
+ selected backend requirements
```

这些资源绑定在返回的 `SequenceHandle` 中。物化进度只在 allocation 与 reserved-but-unmapped 之间转换，
不会把容量交给另一个 request。

Active truncate 或 speculative rollback 可以解除 mappings，但对应容量仍属于该 active reservation。
只有 terminal release 或明确缩减 active entitlement 的资源 transition 才能把容量归还全局。

### 6.2 Terminal 与 capture

TerminalPending request 继续持有 `SequenceHandle` 和完整 reservation，直到 Program 完成：

- **Finish**：发布一个完整 immutable checkpoint；或
- **Discard**：释放整个 active continuation。

若 Finish 无法形成合法保留终态，必须采用 Discard。Optional capture 可以通过同一 pressure graph 改变
inactive owners，但只有严格优于 private-only baseline 的完整终态才会提交；否则跳过 shared publication，
不阻塞 active request，也不改变同一 frontier 原本可执行的 private capture。

### 6.3 Persistent backfill proof

Scheduler 是否允许 backfill 由 [Engine 架构](engine-architecture.md#52-admission-顺序)决定。Program
提供的物理证明使用同一 feasibility oracle：

```text
hypothetical state =
    current stable state
    - protected donor active reservations after their terminal release
    + borrower full active reservation
    + head root materialization
    with all unprotected inactive cache releasable
```

只有该完整状态可行，borrower 才具有 persistent-safe backfill 资格。证明与 `resource_revision` 绑定，
不使用 borrower 或 donor 的预计完成时间。

---

## 7. Materialization planning problem

### 7.1 Candidate

ResourceManager 为已选 request 建立有界 candidate 集合：

- root；
- matching private endpoint；
- typed rewrite checkpoint；
- retained long anchor；
- shared stable prefix。

相同 capability 经不同 index 命中时先去重。Program 对每个 shortlist entry 执行 exact verification，
并产生一个 `AdmissionCandidate`。Candidate 固定 source、destination、prompt 和 request work，但不提前
决定 Move/Fork、victims 或最终 placements。

Root 始终存在。它不复用 continuation state，代表从 prompt 开始正常 prefill。

### 7.2 Shared publication candidate

Shared prefix 的三个状态不能混为一谈：

- **boundary** 是协议或 Engine 指出的逻辑内容边界；
- **candidate** 是 Frontend 已解析到 exact token frontier 的可选写入机会；
- **owner** 是 Program 已发布、具有完整 State/KV identity 与 placement 的 immutable checkpoint。

外部协议或 C++ `PromptInput` 每个请求最多提交四个显式 markers。Frontend 还可以生成最多三个 Engine
candidates：全部 tools 之后、连续 leading System/Developer 之后，以及 full prompt。相同 frontier 合并，
因此每个请求最多七个 prepared candidates。这个固定上限不是启动配置。

Candidate 保存 evidence flags。策略含义为：

| Evidence | 首次出现时的准入能力 |
|---|---|
| `ExplicitBoundary` | 可以参与 pressure，但仍须有严格正净收益 |
| `RequestedAutomatic` | 可以参与 pressure，但仍须有严格正净收益 |
| `DefaultAutomatic` | 只能使用不降低现有 owner 的空余终态 |
| `EngineStructural` | 只能使用不降低现有 owner 的空余终态 |
| `EngineObserved` | 至少两个独立 reuse domains 观测到相同 key 后才可创建 |

Marker、evidence 和 shortlist key 都不证明命中；Program 对 read、dedup 和 publication 重新验证完整
identity。候选创建 source 的顺序为：exact shared owner 直接 dedup；selected exact private reuse base 在
第一次模型 mutation 前做零-prefill promotion；更晚 frontier 由 active prefill capture。早于 selected
reuse base 且没有现成 exact source 的候选只保留为需求观察，不回滚状态或重新 prefill。

Shared candidate 必须在 prefill 前决定是否增加 split，而物理状态只能在 frontier 到达时确定。因此实现
分为两步：

1. admission 已选定 materialization source 后，ResourceManager 用完整逻辑 portfolio 的乐观收益上界和
   Program 返回的精确 split cost 选择 candidate subset；这一步不预留资源，也不承诺 publication；
2. capture offer 到达后，`SharedCapturePlanner` 从当前 `resource_revision` 搜索完整物理 targets，可以联合
   demote/evict 多个合法 inactive private/shared owners。shared catalog 已满时，logical replacement 与其他
   物理 victims 属于同一个 target 和同一个 transaction。

实际 planner 的 baseline 是同一 frontier 的 private-only policy。Shared publication 是可选投资：没有可行
且严格正收益的 target 时执行 private baseline 或 Skip。

### 7.3 Target

Planner 的搜索节点是完整 target，而不是一条孤立 eviction action。一个 target 同时确定：

```text
selected candidate
selected source retained or consumed-to-active
every eligible owner retained or evicted
every surviving checkpoint and required coverage
every surviving State/KV Device/Host placement
logical publication destination
```

Eligible victims 只包括未被 active reference、selected-source protection 或 open transaction pin 保护的
inactive owners。Target 应用后产生的零引用 objects 自动释放。

Program 从该完整终态重新推导：

- Move 或 Fork；
- State/KV replacement 与 partial-tail COW；
- last-reference releases；
- active completion reservation；
- Host allocator geometry；
- ordered stage peaks；
- immediate work 与未来 recovery work。

一个 candidate 的 `identity target` 只应用 incoming activation 与必需的 source disposition，保持其他
eligible inactive owners 的 checkpoint contents 和 placements 不变。它是该 candidate 压力降级图的根。

### 7.4 单调降级图

Pressure target 从 candidate 的 identity target 沿单调降级方向扩展：

- 移除冗余 Device 或 Host replica；
- 将 Device State/KV 降为 Host-only；
- 缩短不再保留的 checkpoint coverage；
- 删除 checkpoint；
- evict 完整 inactive owner。

Program 根据当前 residual deficit、checkpoint frontiers、page/extent boundaries 和共享引用生成语义
successors。搜索不枚举物理操作的排列；每个 target 由 Program 映射到唯一规范 stage order。

单个 owner action 暂时不产生 reclaim，也不能在联合 post-state 之前被删除，因为它可能与另一 action
共同形成 last-reference release。

---

## 8. 规划目标与算法

本节中的 `incumbent` 指当前已经找到的最佳可行完整 target。

### 8.1 可行性与价值分离

Program 先判断 target 是否结构合法、物理可行以及是否仍可继续降级。ResourceManager 只在物理可行且
logical publication 可采用的 targets 之间比较价值。

成本模型不参与容量判定。任何预测误差都只能改变“选择哪个可行方案”，不能让不可行方案通过。

### 8.2 机器成本

Program 将物理工作折叠为：

```text
Immediate =
    ordered transfer phases
  + remaining Text/Vision prefill
  + required State/KV copy work
```

一次同 phase、同 direction 的 transfer batch 使用：

\[
Transfer =
\max(batch\_ns+operations\cdot operation\_ns,\ bytes\cdot ns\_per\_byte)
\]

Text suffix 的 prefill work 使用：

\[
AttentionPairs = B\,S+\frac{S(S+1)}{2}
\]

其中 \(B\) 是已复用 prefix tokens，\(S\) 是剩余 suffix tokens。Vision item/patch work 使用同一
startup-resolved machine model 的独立分量。

### 8.3 Portfolio value 与 future loss

ResourceManager 保存最近 32 个成功 materialize 为 Active 的请求。每条 demand record 保存 reuse domain
以及该请求 exact-matched、selected 或可创建的 prefix keys；不保存 prompt、媒体内容、session 原文或 API
cache key。有 Engine session key 的请求使用其稳定 digest 作为 domain；无 session key 的每个请求形成独立
domain。

对 checkpoint \(p\) 和完整 portfolio state \(S\)：

\[
Saving(p,S)=\max(0,\ Rebuild(p)-Recovery(p,S))
\]

`Rebuild` 是从 root 到该 frontier 的 canonical prefill cost；`Recovery` 是 Program 根据实际 State/KV
placement 返回的最小 restore、copy 和必要 interval-prefill cost。每条 demand 只取所有 matching
checkpoints 中的最大 saving：

\[
EmpiricalValue(S)=\sum_q\max_{p\ matches\ q}Saving(p,S)
\]

因此 tools、instructions 和 full-prompt 等嵌套 prefixes 不会对同一个未来请求重复计价。再加上尚未到期
的显式 shared credit，得到公共价值：

\[
PublicValue(S)=EmpiricalValue(S)+SharedCredit(S)
\]

Private retention 表示每个 owner 还有一次未观测的后续复用。比较 baseline \(S_b\) 与 target \(S_t\) 时，
保持 checkpoint identity 不变，先计算每个 checkpoint 的 saving 损失，再在 owner 内取最大值：

\[
PrivateLoss_o(S_b,S_t)=w_o\max_{p\in o}
\max(0,Saving(p,S_b)-Saving(p,S_t))
\]

不同 owners 的 `PrivateLoss` 相加。这样不会把同一 continuation 的嵌套 checkpoints 全部当作未来请求，
也不会让一个仍然可用的末端 checkpoint 掩盖较早 TurnClosure 或 long anchor 的损失。权重为：

| Private retention | prior weight |
|---|---:|
| Disposable | 1 |
| RecentPrivate | 4 |
| LiveSession | 16 |

Shared owner 没有固定 retention multiplier。`ExplicitBoundary` 或 `RequestedAutomatic` 在 publication 时带来
一个 owner-scoped credit；它在第一次后续 exact match 时消费，或在 32 次成功 materialization 后到期。
同一 frontier 的多个 evidence 不叠加 credit。

对 mandatory materialization 的 candidate identity state \(I_c\) 和 pressure target \(T\)：

\[
FutureLoss_c(T)=\max(0,PublicValue(I_c)-PublicValue(T))
+\sum_o PrivateLoss_o(I_c,T),\qquad
J(c,T)=Now(c,T)+FutureLoss_c(T)
\]

对 optional shared capture，比较 private-only state \(S_b\) 与 complete shared target \(S_t\)：

\[
NetGain=PublicValue(S_t)-PublicValue(S_b)
-\sum_o PrivateLoss_o(S_b,S_t)-ImmediateDelta
\]

执行前 candidate subset 另减精确 `SplitCost`；capture offer 到达后 split 已经发生，不再重复计费。
`ImmediateDelta` 包括 shared target 相对 private baseline 的 transfer/copy cost，并乘以当前 capture owner
及被该 global transition 阻塞的其他请求数。只有 `NetGain>0` 才发布；相等、溢出或饱和无法证明严格
正收益时保持 baseline。

Selected source 的合法 `ConsumedToActive` 是 ownership transfer，不计作 eviction。其他随 pressure
删除或降级的 checkpoints 通过 public value 与 private transition loss 进入 future loss。

### 8.4 Lower bound

每个搜索节点具有：

\[
LB(c,T)=MinimumRequestCost(c,T)+FutureLoss_c(T)
\]

`MinimumRequestCost` 保留无法被后续降级消除的 restore/prefill work，只排除完整 victim set 可能消除的
private-source Fork/COW 或 optional pressure work。

Lower bound 用于两件事：

- 判断一个可行 identity target 是否已经支配所有可展开 alternatives；
- 当 queue 最小 lower bound 严格大于 incumbent 完整成本时证明当前模型内最优。

相等 lower bound 仍需进入确定性 tie-break，不能提前停止。

### 8.5 无压力路径

Planner 首先评估所有 exact candidate 的 identity target。若最佳可行 identity 的完整成本严格低于每个
expandable candidate 的 lower bound，则直接 seal：

```text
inspect exact candidates
  -> choose dominating feasible identity
  -> seal ResourcePlan
```

这条路径不建立 pressure target graph，也不扫描与本次选择无关的物理对象。它覆盖资源充足时的普通
cache hit 和 root admission。

### 8.6 Correctness incumbent

如果没有 identity target 同时满足物理与 logical publication 条件，Planner 在可选搜索预算之外先评估：

```text
root candidate
+ release all unprotected inactive cache
```

这是 maximal-release target。它不保留可牺牲的 cache value，但保证：

- 搜索预算耗尽不会把本可运行请求错误地判成 blocked；
- planner 始终拥有一个 correctness-first incumbent；
- active reservations、selected-source protection 和 transaction pins 仍不可被释放。

若 isolated root 本身超过 per-request 或总容量，结果为 permanently infeasible。若 isolated root 可行，
但当前 active reservations、lane 或 open transaction 仍阻塞 maximal-release target，结果为 temporarily
blocked。

### 8.7 有界 best-first search

有了 incumbent 后，Planner 在全部 candidate roots 上执行 best-first branch-and-bound：

1. 以 lower bound 最小的 target 为下一 expansion；
2. Program 原子地产生其 canonical successors；
3. 每个等价完整 target 只 assessment 一次；
4. 可行且逻辑可采用的 target 可以替换 incumbent；
5. 仍可扩展且 lower bound 不高于 incumbent 的 target 回到 queue；
6. queue 穷尽、证明 model-optimal，或达到 target/time/value budget 时停止；
7. 始终使用停止时的最佳完整 incumbent。

一次 expansion 的 children 要么完整加入，要么完整丢弃，避免 target budget 使结果依赖生成顺序。
Budget 只控制 cache retention 质量；它不截断 candidate identity 检查和 maximal-release correctness
assessment。

若本次实际 assessment 的 canonical targets 为 \(N\)，heap search 的管理成本为 \(O(N\log N)\)，总成本再加
Program 对这些 targets 的 physical projection work。设当前有 \(A\) 个 eligible owners，stored target
decisions 使用 \(O(NA)\) memory；physical oracle 与 successor scratch 还受启动时配置的 owner、checkpoint
和 logical-page descriptor 上限约束。\(N\) 由固定 target budget 封顶。
因此组合目标空间可以很大，但一次 admission 的规划时间和内存不会随理论组合数无界增长。

### 8.8 确定性选择

首要比较键是 \(J\)。成本相等时依次偏好：

1. 少影响已观测命中的 checkpoints，优先保留最近命中；
2. 少 owner eviction、checkpoint deletion、copy operation 和 transferred bytes；
3. 少剩余 Text/Vision prefill；
4. 多复用 prompt tokens；
5. 当前 session binding；
6. candidate 与 target 的稳定 ordinal。

`model_optimal` 只表示在当前 target graph、规范 stage order 和机器成本模型内已经证明最优，不表示真实
请求分布上的全局最优 TTFT。

### 8.9 Readiness

最终结果只有四类：

| 结果 | 含义 |
|---|---|
| `PermanentlyInfeasible` | isolated root completion requirement 超过当前 Engine 合同 |
| `TemporarilyBlocked` | 单请求可行，但当前 active/lane/transaction 状态阻塞 |
| `NeedsTransfer` | target 可行，需要异步 transfer 或 pressure stages |
| `Ready` | target 已 Device-ready，可在当前 boundary 同步发布 |

`TemporarilyBlocked` 只在 lane、FIFO protection、open transaction 或 resource revision 变化后重新检查。
普通 decode 不重复运行 planner。

---

## 9. Resource transition

### 9.1 三个对象

需要 allocation、pressure 或 transfer 的终态通过：

```text
ResourcePlan
  -> RunningTransaction
  -> ResourceResult
```

- `ResourcePlan`：Program seal 的 immutable、move-only、revision-bound transition；
- `RunningTransaction`：Program 当前执行中的唯一 global resource transition；
- `ResourceResult`：Committed 或 Aborted 的完整稳定结果。

Active reservation 已经覆盖且不需要新 allocation/transfer 的 finish、discard 或 inactive release 可以作为
零阶段原子 transition 完成，但仍遵守同一串行化、revision 和完整结果规则。

### 9.2 Start

Program 在第一次 mutation 前重新验证：

1. plan 的 `resource_revision`；
2. source、victim 和 destination capability generation；
3. ResourceManager 已保留的 logical publication capacity；
4. 当前 allocator 对每个 ordered stage 的容量和几何；
5. source/victim dependencies 与 active protections。

Stale 或不再可行的 plan 在 start 前无物理副作用。Start 成功后，source、victims 和 destinations 被
claim/pin，plan 被线性消费，不再更换 candidate 或 target。

### 9.3 Progress

Transaction 可以跨同步 copy 或 CUDA event 推进。每个 replacement 遵循：

```text
destination reserved
  -> copy complete
  -> epoch/coverage verified
  -> replacement published
  -> old replica released when unreferenced
```

同一时间至多一个 global resource transition。既有 active execution 只有在使用自身既有 reservation、
且不触碰 transaction mappings 时才能与 transfer 交错。

### 9.4 Commit

Commit 发布：

- new active sequence 或 checkpoint；
- selected source disposition；
- 已提交的 owner/checkpoint degradation；
- stable State/KV placements；
- transaction pin/reservation cleanup；
- 必要的 `resource_revision` 变化。

`ResourceResult` 包含 ResourceManager 采用逻辑终态所需的绝对结果，不包含供上层重建物理 occupancy 的
delta。ResourceManager 在 start 前已预留 adoption storage，因此 commit 后的 adoption 必须 allocation-free
且不能进行新的容量判断。

### 9.5 Abort

Publication 前的 cancellation 或执行失败可以得到 request-local abort，只要 Program 仍能形成稳定物理状态。
Abort 保证：

- selected source 仍有效；
- target 没有半 Active 或半 published checkpoint；
- unused destinations、reservations 和 pins 已释放；
- 已安全完成的 victim demotion/deletion 被明确记录在 `ResourceResult` 中。

已经发布 replacement 后完成的合法 pressure degradation 不需要逆向复制回原 placement。ResourceManager
采用 abort result 中的实际终态，而不是回放中间 receipts。

若 Program 已无法形成可解释的稳定状态，则进入 Engine-wide failure。

---

## 10. Retention、session 与 publication

### 10.1 Retention policy

Private retention class 通过 owner prior 影响 portfolio value，不提供固定 eviction 顺序。Shared owner 的价值
来自真实 demand 与尚未到期的 explicit credit。Planner 可以组合：

- 删除冗余 replica；
- 将 Device replica 降为 Host-only；
- 删除不再保留的 checkpoint 或 suffix coverage；
- evict 完整 inactive owner。

如果一个 placement change 会删除 checkpoint 的最后一份完整 State 或 required KV coverage，同一个
target 必须先建立 replacement，或同时删除该 checkpoint。

Placement 只在 admission、capture、finish 或显式 inactive release 的 resource boundary 改变。普通 decode
不运行周期性 promotion/demotion。

### 10.2 Session ordering

SessionKey 属于 ResourceManager，只提供 candidate lookup 与 binding。Program 不读取 SessionKey。

每个可更新 SessionIndex 的请求取得单调 `publication_order`：

- 新结果只有 order 更大时才能替换当前 binding；
- 较旧请求晚完成时成为 anonymous cache 或按 policy 释放；
- matching private source 可以在合法时 consume-to-active；
- shared source 始终保持 immutable；
- abort 只能恢复自己仍持有 exact claim 的 former binding。

Capability generation 判断 handle 是否仍有效，publication order 判断哪个完成结果更新 session；两者职责不同。

### 10.3 Logical publication capacity

ResourceManager 必须在 Program mutation 前预留 active/private/shared destination slot。Program commit 后的
logical adoption 不再扩容或搜索 slot。

Private Move 可以复用 source descriptor；root 或 Fork 需要 destination。Optional capture 没有 slot 时跳过；
terminal Finish 没有合法 publication capacity 时采用 Discard。

---

## 11. 配置语义与有界性

`ContextCacheOptions` 在 Engine 构造时解析为固定容量：

| 配置轴 | 架构含义 |
|---|---|
| `device_state_slots` | active guarantee 之外可保留的 Device checkpoint StateImages |
| `host_state_slots` | 完整 Host StateImages |
| `host_kv_capacity_bytes` | typed packed Host KV arena 总容量 |
| `max_private_continuations` | private owner/catalog 容量 |
| `max_shared_prefixes` | shared immutable owner/catalog 容量 |
| `max_long_anchors_per_continuation` | 每条 private history 的 retained long checkpoints 上限 |

所有 stores、catalogs、planner scratch 和 transaction adoption storage 都按解析后的上限在启动时建立。
每请求 external markers 固定最多四个、Frontend candidates 固定最多三个、demand window 固定为 32；这些
是产品语义与算法有界性，不是可调容量轴。
配置必须满足：

- private continuation capacity 至少覆盖全部 active requests；
- Device State 总容量为 `max_concurrency + device_state_slots`；
- Host State 与 Host KV 独立计费；
- logical counts、address-space counts 和 storage sizes 可表示；
- Program 可以兑现最小 active capacity 与 selected backend requirements。

Context cache disabled 时采用 root-only 语义：不读取或发布 inactive continuation，Device/Host checkpoint
容量为零，但 active execution 所需的 State/KV reservation 保持不变。

公开默认值和启动命令由 `EngineOptions`、[CLI](../cli.md)与 [Serving](../serving.md)维护，不在本架构
文档中复制。

---

## 12. 核心不变量

1. Scheduler 先选择 request；资源层不能以 cache value 改变请求顺序。
2. ResourceManager 是 logical cache policy 的唯一 authority，Program 是 physical state 的唯一 authority。
3. Physical occupancy 按 unique allocations 与 concrete reservations 计数，不按 logical references 重复计数。
4. Reclaim、Move/Fork 和 COW 必须从完整联合 post-state 推导。
5. 每个有序 transition stage 都必须满足容量与 allocator geometry。
6. Prefix hit 必须具有 exact identity、完整 StateImage 和全部 required typed KV coverage。
7. Published checkpoint immutable；每条 active continuation 至多一个 mutable writer。
8. Active completion reservation 在 terminal resource result 采用前不可被借用或回收。
9. 同一时刻至多一个 global resource transition，且 sealed plan 与 `resource_revision` 绑定。
10. Start 前 stale rejection 无物理副作用；start 后不能更换 candidate 或 target。
11. Program commit/abort 都返回完整稳定终态；ResourceManager adoption 已预分配且不能失败。
12. Root 加释放全部 unprotected inactive cache 是 bounded search 之外的 correctness fallback。
13. Planner 成本模型只排序可行 targets，不参与物理正确性。
14. Session binding 只接受更大的 publication order。
15. Retention 或 capture 失败必须退化为 skip/release，使 active lane 有有限终态。
16. Shared publication 必须严格优于 private-only baseline；单个 demand 对嵌套 prefixes 只能贡献一次最大
    saving。
17. Candidate selection 不是资源预留；实际 capture target 必须在 frontier 到达后按当前 revision 重新
    证明完整物理终态。

---

## 13. 实现位置

| 职责 | 主要位置 |
|---|---|
| logical catalog、claims、session policy | `src/runtime/engine/resource_manager.h` |
| cross-candidate bounded planner | `src/runtime/engine/materialization_planner.h` |
| shared capture bounded planner | `src/runtime/engine/shared_capture_planner.h` |
| shared/private portfolio value | `src/runtime/engine/context_portfolio_value.h` |
| machine cost model | `src/runtime/engine/context_cost.*` |
| common resource summaries | `src/runtime/contract/types.h` |
| target physical projection 与 transaction | `src/targets/qwen3_6/impl/runtime/program*.h` |
| State stores | `src/targets/qwen3_6/impl/runtime/state_image_store.h` |
| logical KV/address spaces | `src/targets/qwen3_6/impl/runtime/logical_kv_store.h` |
| Host KV extents | `src/targets/qwen3_6/impl/runtime/host_kv_extent_store.h` |
| public capacity options | `include/ninfer/types.h` |

路径用于定位当前实现，不改变本文定义的所有权边界。
