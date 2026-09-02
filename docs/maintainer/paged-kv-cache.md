# NInfer Paged KV Context Store

本文定义 NInfer growing KV 的物理存储与消费合同。它是 typed KV pools、logical pages、
Device/Host replicas、address spaces、reservations、block tables 和 GPU consumer views 的维护者权威。

本文只回答两个问题：

1. 一个已经选定的 logical context target 能否由当前 KV stores 兑现；
2. 兑现后的 KV 如何被模型 execution unit 直接消费。

请求顺序和生命周期由 [Engine 架构](engine-architecture.md)定义；candidate、retention、pressure target
和资源成本由 [资源调度与上下文缓存](resource-scheduling-and-context-cache.md)定义。KV Store 不选择
request、source 或 victim。

---

## 1. 物理模型

Growing KV 使用一组启动时固定的 homogeneous pools。每个 pool：

- 存储具有同一 frontier 和 lifetime 的全部 planes；
- 使用固定的 token page size、plane order 和 page-group count；
- 拥有独立的 physical page-ID namespace 与 free capacity；
- 为每条 active sequence 提供一个 logical address space；
- 由 consumer 通过 block table 直接寻址。

一个 request 的 KV 不要求物理连续，也不与 control lane 固定绑定。所有 active 与 inactive address spaces
共享 pool capacity；active request 通过 reservation 获得不会被其他 request 使用的完成容量。

Paged storage 覆盖按上下文增长的 KV。DFlash local cyclic state、Vision/query temporary K/V 和其他固定
state 具有不同 lifetime，由其各自的 StateImage 或 workspace contract 管理。

---

## 2. 三种独立粒度

KV 架构区分：

| 粒度 | 含义 | 当前合同 |
|---|---|---|
| allocation granularity | pool 一次取得或释放多少 token payload | 每个 growing pool 为 `P=64` |
| valid-frontier granularity | consumer 可以读取到哪个 logical position | 1 token |
| reusable-state granularity | 哪个 frontier 具有完整模型 continuation | target-defined checkpoint |

Page boundary 不是 Attention mask boundary，也不是 prefix hit boundary。一个 valid frontier 可以位于 page
内部任意 offset。

KV Store 可以在任意 token frontier 表示、truncate 或保护 prefix；这不证明模型可以从该位置恢复。
可复用 frontier 必须同时存在完整 StateImage 与 target-defined backend state，具体规则见
[Continuation 与 checkpoint](resource-scheduling-and-context-cache.md#4-continuation-与-checkpoint)。

---

## 3. Typed pool set 与容量

### 3.1 Pool set

Exact target 和 selected speculative backend 在启动时确定 pool set：

```text
ordinary:
    Main Text

MTP:
    Main Text
    MTP

DFlash:
    Main Text
    DFlash Full
```

MTP 与 DFlash 在一个 Engine 内互斥，因此当前最多有两个 growing pools。

| Pool | 内容 | 逻辑 frontier |
|---|---|---|
| Main Text | target full-attention K/V 与其 code/scale planes | target materialized KV frontier |
| MTP | MTP persistent K/V 与其 code/scale planes | MTP KV frontier |
| DFlash Full | DFlash persistent full-context K/V | DFlash context frontier |

Main Text 与 MTP 使用 Engine 选择的 BF16、INT8-G64、FP8-E4M3FN-row256、NVFP4-G16 或 K8V4
KV profile；DFlash Full 使用自己的 BF16 profile。`BFloat16` 名称下的物理 layout 为 BF16 K、FP16 V，
写入端将 BF16 V 一次转换为 FP16。K8V4 是封闭的非对称 profile，不是运行时 bit-width 组合：K 固定为
FP8-E4M3FN-row256，V 固定为 NVFP4-G16。

`PagedKVStorageLayout` 将选定的 closed profile 解析为 K/V data/scale plane schema；target planner 按
layer 展开该 schema 并确定 plane ordinal。Common pool implementation 仍只接收已展开的
`KVPageGeometry`、plane inventory 和 capacity，不解释 storage mode。

### 3.2 Main capacity

设：

- \(S\)：单条 sequence 的 `max_context`；
- \(C\)：`max_concurrency`；
- \(P=64\)：Main page size；
- \(L=\lceil S/P\rceil\)：单 address space 的 logical page capacity；
- \(M\)：Main pool 的 physical page-group count。

可用范围为：

\[
M_{min}=\max(L,C)
\]

\[
M_{max}=C\,L
\]

\(M_{min}\) 同时保证一条 request 可以达到 \(S\)，且每个 active lane 至少可取得一个 page。
\(M_{max}\) 是全部 active requests 同时达到 per-sequence ceiling 时的物理上界。

`kv_capacity` 的 explicit policy 解析为：

\[
M=\left\lceil K_{main}/P\right\rceil
\]

并要求 \(K_{main}\ge S\) 且 \(M\in[M_{min},M_{max}]\)。

### 3.3 Automatic capacity

Automatic policy 在权重加载后，以当前可用显存 \(F\) 和要求保留的 headroom \(R\) 解析一次 \(M\)。
Exact target 提供 affine `SequenceCapacityCurve`：

\[
B(M)=B_{min}+(M-M_{min})B_{step}
\]

其中 \(B(M)\) 是该 Main capacity 对应的完整 runtime Device reservation，包含：

- Main 与 selected backend typed pools；
- active/checkpoint State storage；
- block tables 与固定 persistent state；
- unified workspace；
- CUDA Graph allowance。

Automatic 选择：

\[
M=
\min\left(
M_{max},
M_{min}+
\left\lfloor
\frac{F-R-B_{min}}{B_{step}}
\right\rfloor
\right)
\]

并要求 \(F\ge R+B_{min}\)。当 \(M_{min}=M_{max}\) 时直接取该单点。Target 用同一生产 layout builder
生成 \(B_{min}\)、\(B_{step}\) 和最终 layout，并验证 capacity curve；resolver 不复制模型维度公式，也不靠
allocation probing 猜测容量。

最终公开的 Main KV capacity 为 \(M\,P\) token-equivalents。Page rounding 只增加 physical padding，
不会扩大单 sequence 的 logical ceiling \(S\)。

### 3.4 Backend capacity

Selected backend 的 logical page capacity仍为 \(L\)。Physical capacity从 Main \(M\) 推导：

```text
backend off:
    Main physical pages = M

MTP with draft window K:
    Main physical pages = M
    MTP physical pages  = M + C * ceil((K - 1) / P)

DFlash:
    Main physical pages        = M
    DFlash Full physical pages = M
```

MTP 的额外 pages 只覆盖每条 active row 在一个 speculative round 中相对 Main 的 provisional lead，
不扩大任一 address space 的 logical capacity。DFlash Full 没有这种 provisional lead。

各 pools 物理分离。一个 pool 的 free page 不能变成另一 pool 的 payload。Program 在启动时一次性建立
完整 typed capacity vector，运行期不扩容或重分 pool geometry。

### 3.5 Host capacity

Main 与 selected backend 的 Host replicas共用一个 startup-fixed pinned `HostKVArena`，但每个 allocation
携带自己的 typed page layout。Host capacity 按实际 packed bytes 和 allocator extent geometry计费；
它不扩大 Device active entitlement 或单 sequence context ceiling。

---

## 4. Page group 与物理 layout

### 4.1 Grouping invariant

只有同时满足以下条件的 planes 才进入同一个 pool：

1. 使用相同 logical cache ordinal；
2. 共享同一个 committed frontier；
3. 一起 reserve、materialize、truncate、retain 和 release；
4. 使用相同 page size；
5. 没有独立释放或独立容量复用的语义。

因此 Main layers 的 K、V、code 和 scale 可以共享一个 page-group ID；Main、MTP 与 DFlash Full
必须属于不同 pools。

### 4.2 Page group

一个 pool-local page-group ID \(g\) 同时选择该 logical block 在全部 grouped planes 中的 payload：

```text
page group g
├── layer/plane 0 slice g
├── layer/plane 1 slice g
├── ...
└── layer/plane n slice g
```

Planes 拥有 Engine-lifetime-stable backing，但不要求组成一个连续 blob。一个 page group 是 allocation、
reservation、reference 和 transfer 的最小 Device 单位。

Physical IDs 可以任意排列。Allocator 可以优先返回连续 IDs 改善 locality，但 correctness、admission
和 kernel launch topology 不依赖连续性。固定大小 page groups 不产生 variable-size external
fragmentation，也不需要 Device compaction。

### 4.3 Closed Device plane orders

所有 registered growing pools 使用 \(P=64\)，并选择两种 closed orders 之一。

Main Text 与 MTP 使用 page-major：

\[
[X,P,H,N_{physical}]
\]

DFlash Full 使用 head-major page run：

\[
[X,P,N_{physical},H]
\]

其中：

- \(X=D\) 表示 K/V 或 quantized code plane；
- INT8-G64 scale plane 使用 \(X=D/64\)；
- FP8-E4M3FN-row256 scale plane 使用 \(X=1\)；
- NVFP4-G16 code plane 使用 packed U8 \(X=D/2\)，scale plane 使用 U8 \(X=D/16\)；
- \(H\) 是 KV heads；
- \(N_{physical}\) 是该 pool 的 physical page count。

对 logical position \(p\)：

\[
b=\lfloor p/P\rfloor,\qquad o=p\bmod P,\qquad g=block\_table[b]
\]

Page-major 地址为：

\[
address=base+d\,nb_0+o\,nb_1+h\,nb_2+g\,nb_3
\]

Head-major 地址为：

\[
address=base+d\,nb_0+o\,nb_1+g\,nb_2+h\,nb_3
\]

Code 与 scale planes 使用同一个 \(g\)，但使用各自 Tensor 的 leading coordinate 和 strides。
Exact persistent codec 由 [`kv_cache_append.h`](../../include/ninfer/ops/kv_cache_append.h) 定义，
consumer arithmetic 由 [`softmax_attention.h`](../../include/ninfer/ops/softmax_attention.h) 定义；
allocator 只解释 plane bytes、order 和 page-group identity。

D256 Main/MTP profile 的单 token/head 物理 payload 为：

| profile | K code + scale | V code + scale | K+V |
|---|---:|---:|---:|
| BF16 | 512 B | 512 B | 1024 B |
| INT8-G64 | 256 B + 8 B | 256 B + 8 B | 528 B |
| FP8-E4M3FN-row256 | 256 B + 2 B | 256 B + 2 B | 516 B |
| NVFP4-G16 | 128 B + 16 B | 128 B + 16 B | 288 B |
| K8V4 | 256 B + 2 B | 128 B + 16 B | 402 B |

K/V 的 code 和 scale planes 具有各自的 dtype、leading extent 和 group size；它们仍共享 page-group
identity、frontier 和 lifetime。Capacity curve、Device/Host replica、continuation transfer 和 memory
summary 均从这一 typed plane inventory 计算，不能用 `2 * vector_bytes` 代替 K8V4 的非对称字节数。

### 4.4 Logical position domain

Block table 使用 autoregressive cache ordinal：

```text
logical block b covers positions [b*P, (b+1)*P)
```

RoPE、MRoPE、Vision axes 和 `rope_delta` 是 Attention input metadata，不改变 KV slot ownership。

### 4.5 Page payload

一个 pool 的 logical page-group payload 为：

\[
PageBytes=\sum_{plane} PlaneBytesPerToken\cdot P
\]

Startup physical bytes 由各 plane slab 的完整 span 与 alignment 得到。不同 pools 的 `PageBytes` 可以不同，
但一个 pool 内的所有 page groups 等价。

`P=64` 同时满足当前 32/64-key Attention tiles、128-token aligned prefill chunks、有限 block-table
metadata 和 bounded tail slack。Prefix hit granularity不参与 page-size 选择。改变 page size、grouping
或 closed plane order 都是架构变更。

---

## 5. Logical page 与 replicas

### 5.1 Logical page identity

Device page ID 只是当前 Device replica 的物理位置，不是 prefix identity。Program 为每个 pool 维护
generation-checked logical pages：

```text
LogicalKVPage
├── object generation
├── content epoch
├── committed columns [0, P]
├── protected columns
├── address-space references
├── active references and writer state
├── optional Device page-group lease
├── optional Host extent membership
└── transaction pins
```

一个 logical page 的 canonical content 由 `content epoch + committed columns` 标识。Speculative 或尚未
提交的 bytes 不扩展 committed coverage。

Replica 对 checkpoint 所需前 \(n\) 列有效，当且仅当：

\[
replica.epoch=page.epoch
\quad\land\quad
replica.coverage\ge n
\]

### 5.2 Device replica

Device replica 使用 `4` 的 consumer-native plane layout。`DeviceKVPageLease` 独占一个 pool-local
physical page group；generation 防止 release/reuse 后的 stale handle。

Pool 对同一容量单位区分：

```text
allocated page lease
reserved but not materialized page
globally available page
```

\[
allocated+reserved+available=capacity
\]

Materialize 把 reservation 转成 lease；dematerialize 把 lease 还回同一 reservation。两者不改变其他
requests 可用的 global capacity。

### 5.3 Host replica

Host replica 使用 logical-order packed `HostKVPageLayout`：

- 不保存 Device page ID 或 block-table holes；
- 每个 page 包含该 typed pool 的全部 grouped plane payload；
- variable-size extent只为实际 page count 付费；
- Main/backend layouts可以在同一 arena 中分配不同 stride 的 extents。

Host arena 是有界 variable-size allocator。Plan 必须针对完整 release/allocation recipe 验证 extent geometry；
`free_bytes` 只是占用摘要，不是可分配性的充分证明。

### 5.4 Replica transfer

D2H/H2D transfer 复制完整 page payload，可以把相邻 physical IDs 合并成更少的 transfer runs。
一个 replacement replica 的 publication 顺序为：

```text
reserve destination
  -> copy page payload
  -> verify epoch and committed coverage
  -> publish replica
  -> release source when no longer required
```

Copy 进行时 source 与 destination 都被 pin。Copy 完成前，destination 不进入 address space 或 execution
table；唯一有效 source 也不能先释放。

### 5.5 Descriptor lifetime

Logical descriptor 不构成第三份 payload。它可以在 Device-only、Host-only 或 Both placements 下继续存在。
只有当 references、replicas 和 transaction pins 均为零时，descriptor 才能回收并推进 generation。

---

## 6. KV address space

### 6.1 Owning membership

每个 continuation 在每个 enabled pool 中持有一个 `KVAddressSpace`：

```text
KVAddressSpace
├── ordered logical-block -> LogicalKVPage membership
├── committed frontier
├── checkpoint-protected frontier
├── active/inactive state
├── active growth reservation
└── optional execution-row lease
```

Address space 拥有 logical order；block table只是 active Device mapping 的执行镜像。Inactive checkpoint
不占 execution row，也不绑定原 control lane。

### 6.2 三个 extent

每个 address space 区分：

| Extent | 含义 |
|---|---|
| entitlement | active request 被保证可取得的最大 Device page count |
| membership | 已经属于该 address space 的 logical pages |
| committed frontier | consumer 可以读取的 canonical token prefix |

Membership 可以大于 committed frontier，例如预先 materialize 一个 prefill chunk 或 speculative window。
这些 bytes 只有在 target commit frontier 后才成为 canonical state。

不同 pools 的 membership 和 frontier 可以不同；Program model schedule 定义它们之间的语义关系，
KV Store 不自行推导。

### 6.3 Active bundle

一个 active sequence 持有：

```text
SequenceKVBundle
├── Main Text KVAddressSpace
└── selected backend KVAddressSpace, when enabled
```

Activation 必须为全部 enabled pools 原子取得：

- address-space descriptors；
- missing Device replicas；
- future growth reservations；
- partial-tail COW destinations；
- execution rows。

任一 pool 失败都不能发布部分 Active bundle。Active sequence 不能按 request 临时关闭 selected backend。

### 6.4 Execution rows

每个 pool 在启动时建立固定地址的 Device block-table matrix：

\[
block\_tables[N_{logical},C]
\]

其中 \(N_{logical}=L\)，\(C=max\_concurrency\)。每个 active address space lease 一行；每一项是 I32
pool-local physical page ID。

Activation 将 address-space membership 的 Device page IDs 批量发布到所租 row。Inactive address space
没有 row；同一个 continuation 下次 activation 可以取得另一行。Execution row 不拥有 logical page、
reservation 或 frontier。

---

## 7. 生命周期

### 7.1 Activation

从 root、private source 或 shared source 激活时：

```text
create or claim destination address spaces
  -> reserve every typed pool
  -> restore required Host-only pages
  -> Move or Fork memberships
  -> create private partial tails when required
  -> lease execution rows
  -> publish complete table mappings
  -> publish Active sequence
```

ResourceManager 选择 logical target，Program 根据真实 references 决定 Move/Fork/COW。Active publication
前，source 保持有效，任一失败都回到完整 inactive 终态。

### 7.2 Prefill 与 decode

在一个 GPU unit launch 前，target schedule 给出每个 pool 本次可能写到的最大 logical position。
KV Store 从该 active address space 的 reservation materialize 全部必需 pages，并发布 table slice。

Unit 成功后，Program 才推进 corresponding committed frontiers。部分 layers 已写但 unit 没有形成合法
commit 时，新 frontier 不可见。

Ordinary decode 通常只在跨越 page boundary 时 materialize 一个新 Main page。Tail page 尚有空间时无需
allocator 工作。

### 7.3 Truncate 与 rollback

每个 pool 根据自己的 canonical/provisional frontier计算保留 membership：

\[
NeededPages_s=
\left\lceil
\frac{\max(CommittedFrontier_s,ProvisionalFrontier_s)}{P}
\right\rceil
\]

Trailing mappings 可以解除；最后一个部分页保留。对 active address space，解除的 Device leases 回到
同一 active reservation，而不是全局 available capacity。Page 内 frontier 之后的 stale bytes 不进入
consumer valid domain。

### 7.4 Deactivation、retain 与 release

Active finish 后，address space先退出 execution：

- 释放 execution row；
- 解除 active writer/reference；
- 释放尚未物化的 growth entitlement；
- 根据 terminal target 保留为 immutable inactive address space，或释放全部 memberships。

Retain 只有在 Main、backend 与 StateImage 构成同一完整 continuation 时才能发布。Release 逐个解除
logical references，并只在最后引用消失时释放 physical replicas。

### 7.5 Stable execution unit

从 block-table publication 到 GPU unit 完成：

- selected rows 的 table entries 不变化；
- page payload 不搬迁或逐出；
- logical memberships 和 readable frontiers 对该 unit 冻结；
- execution row 不回收；
- allocator 与 transfer 不进入 kernel。

Mapping、frontier commit、truncate 和 row recycling 都发生在 GPU boundary。

---

## 8. Sharing、Move 与 COW

### 8.1 Private continuation

Private source 在最终 post-state 中没有 surviving immutable reference 时，可以 Move：

```text
inactive address space -> active address space
```

Move 不复制 KV payload，也不改变 logical page identity。Private tail 的唯一 writer 可以继续在未保护
suffix append，只要不会覆盖任一 surviving checkpoint 的 committed prefix。

### 8.2 Immutable Fork

Shared source或仍需保留的 private source使用 Fork：

- frontier 之前的完整 pages增加 immutable address-space reference；
- destination取得自己的 State writer和growth reservation；
- shared pages没有 writer；
- source在整个 activation commit/abort前保持有效。

### 8.3 Non-page-aligned frontier

设 \(P=64\)，checkpoint frontier \(F=1000\)：

```text
full immutable pages = 15
partial tail columns = 40
```

Private Move 可以从 position 1000 原地继续。Immutable Fork 共享前 15 个完整 pages，并把 tail 的 40 个
committed columns复制到一个 private destination page；后续 append只写 private tail。

Checkpoint frontier 不向下取整到 960。Page-size allocation与 token-level validity保持独立。

### 8.4 Writer invariant

Reference count 表示 sharing，不单独决定写权限。Program 同时验证：

- surviving checkpoint protected coverage；
- writer cardinality；
- active references；
- content epoch；
- source/destination pins。

任一 logical page 同时至多一个 writer。存在多个 address-space references 的 full page immutable；
需要写入 shared partial tail 时必须先 COW。

---

## 9. Speculative 与非 growing KV

### 9.1 Independent pool frontiers

MTP/DFlash runtime 使用同一个 KV Store 的 backend pool，不建立独立 allocator。

一次 speculative unit 中，Main 与 backend：

- 分别从各自 active reservation materialize；
- 可以具有不同 mapped/provisional frontiers；
- 分别提交 accepted frontier；
- 分别 trim rejected trailing mappings。

MTP draft 期间 backend mapped extent可以暂时领先 Main；DFlash Full 通常落后于 Main。Provisional lead
不构成 committed checkpoint coverage。Rejected bytes 可以留在 partial page 中，但后续读取前必须被新的
canonical write 覆盖。

### 9.2 Fixed 与 transient K/V

以下 storage 不进入 growing pools：

| Resource | Owner |
|---|---|
| DFlash local sliding-window K/V | fixed per-sequence StateImage |
| DFlash boundary-local snapshot | fixed checkpoint StateImage |
| Vision/query temporary K/V | Program workspace |

DFlash cyclic K/V 使用自己的 `CyclicKVCacheLayerView` 与 modulo/window 语义；它不持有 page ID、
block table 或 growing reservation。Exact target geometry由对应 model 文档定义。

---

## 10. Consumer contract

### 10.1 Single-sequence view

Single-sequence growing-cache Op 使用 non-owning `PagedKVLayerView`：

```text
PagedKVLayerView
├── k_pages / v_pages
├── optional k_scale_pages / v_scale_pages
├── block_table          I32 [Nlogical]
├── head_dim
├── num_kv_heads
└── storage
```

View 只包含一个 layer 的 plane tensors 与一行 block table。它不包含 allocator handle、request identity、
reservation、ownership 或 frontier。

### 10.2 Batched view

Batched Op 使用 `PagedKVBatchLayerView`：

```text
PagedKVBatchLayerView
├── shared layer planes
├── block_tables         I32 [Nlogical, C]
├── head_dim / num_kv_heads
├── storage
└── table_rows[B]        separate Op input
```

`table_rows[b]` 为 compact row \(b\) 选择对应 active address-space row。Compact batch order、Engine lane
和 table row可以彼此不同。Per-row context length、valid columns和positions由 Op 的其他 typed inputs
提供。

### 10.3 Address translation

当前 \(P=64\)：

```text
logical_block = position >> 6
page_offset   = position & 63
physical_page = block_table[logical_block]
```

随后使用 `4.3` 的 closed plane order计算元素地址。Batched consumer先用 `table_rows[b]` 选择 table row，
再执行同一 translation。

Wrapper 验证 Tensor dtype、geometry、closed strides、table shape和execution envelope。Caller 保证：

- 本次可能访问的 logical blocks 已 materialize；
- read domain 不超过该 sequence 的 frozen valid frontier；
- writable K/V code 与 scale 在 frontier publication 前全部完成；
- selected table rows 在 GPU unit 内稳定。

### 10.4 Direct paged execution

Growing-cache Ops 直接消费 paged views。Page translation在 page/tile 粒度计算并复用，inner loop不解释
request、pool kind或allocator state。Kernel correctness不能依赖相邻 logical pages映射到相邻 physical IDs。

Paging 不引入 gather-to-contiguous cache或与 context 长度成比例的 staging copy。Route-specific tile、
split、warp和shared-memory方案可以独立优化，只要保持同一 logical Attention、persistent codec与上述
address contract。Op 的数值与性能准入规则见 [Op development](op-development.md)。

---

## 11. CUDA Graph 与 table publication

Plane bases与 block-table matrix base在 Engine lifetime内稳定。跨 replay变化的是：

- table content；
- `table_rows` selectors；
- positions、context lengths和valid counts；
- model state selectors。

Page IDs、request identity和physical contiguity不进入 graph key。

在需要新 mappings 的 execution unit 前，Program：

```text
materialize all required pages
  -> update host-side membership
  -> publish one contiguous table slice
  -> launch/replay consumer on the ordered stream
```

同一 boundary新增多个 pages或重新激活完整 membership时使用批量 publication。Mapping update必须先于
consumer，且 replay in-flight期间不得改写同一 row。

---

## 12. 核心不变量

1. 每个 Device page-group lease 在所属 pool 中至多承载一个 logical page replica。
2. 同一 pool 只组合共享 frontier、lifetime、page size与allocation语义的planes。
3. 一个 pool 的 K/V/code/scale planes 对同一 logical block 使用同一个 page-group ID。
4. Device occupancy 按 allocated leases与reservations计数；logical aliases不重复计费。
5. Logical page identity、content epoch与physical page ID彼此独立。
6. Valid frontier精确到token；page boundary不改变Attention或checkpoint语义。
7. Published checkpoint所需coverage不可被writer覆盖；任一logical page至多一个writer。
8. Shared full pages immutable；non-aligned writable tail先建立private COW page。
9. Host/Device replacement在copy与epoch/coverage验证完成后才发布。
10. Active entitlement在terminal release前不进入global available capacity。
11. 一个GPU execution unit内membership、block tables、replicas与read frontier稳定。
12. Inactive address space不占execution row；execution row不拥有logical pages。
13. Main与backend pools分别reserve、materialize、commit和truncate。
14. Growing-cache consumer只通过paged view和block table访问KV，不取得allocator或ownership authority。
15. Kernel correctness不依赖physical page ID连续性，也不通过gather建立request-contiguous KV。
16. Checkpoint可复用性由完整target continuation证明，KV page存在本身不构成hit。

---

## 13. 实现位置

| 职责 | 主要位置 |
|---|---|
| Device page pools、reservations与execution tables | `src/core/paged_kv_cache.*` |
| closed K/V data/scale plane schema | `src/core/paged_kv_storage.h` |
| Host packed page layout与arena | `src/core/host_kv_arena.*` |
| logical pages、references与address spaces | `src/targets/qwen3_6/impl/runtime/logical_kv_store.h` |
| Host extent membership | `src/targets/qwen3_6/impl/runtime/host_kv_extent_store.h` |
| Program-level KV transition | `src/targets/qwen3_6/impl/runtime/program*.h` |
| target pool layout与capacity curve | `src/targets/qwen3_6/impl/runtime/layouts*.h` |
| public paged consumer views | `src/core/paged_kv_cache.h` |
| growing-cache Ops | `include/ninfer/ops/`, `src/ops/` |

Exact model state 和 backend mathematics 见
[Qwen3.6-27B model](qwen3.6-27b-model.md)与
[Qwen3.6-35B-A3B model](qwen3.6-35b-a3b-model.md)；persistent KV codec 和 causal consumer
numerical contract 由上表中的 growing-cache Ops 定义。路径用于定位当前实现，不把文件或类名本身提升为
外部接口。
