# NInfer Paged KV Context Store

本文定义 NInfer growing KV 的物理存储与消费合同。它是 typed KV pools、logical pages、
Device/Host replicas、address spaces、reservations、block tables 和 GPU consumer views 的维护者权威。

请求顺序和生命周期见 [Engine 架构](engine-architecture.md)；checkpoint、缓存保留、抢占和资源
回收策略见 [资源调度与上下文缓存](resource-scheduling-and-context-cache.md)。KV Store 兑现选定操作的
物理需求，并向模型 execution unit 提供稳定的直接访问视图。

集中 `ContextDirectory` 由 ModelSession 持有，负责 State、页、history、execution row、Host 范围、
引用和预留。RankProgram 持有固定 backing 与 `ContextStorage`，按值身份执行复制和 mapping 更新，
不维护第二套空闲页分配器。单 rank 使用同一合同，由 LocalExecutor 在进程内驱动。

---

## 1. 物理模型

Growing KV 使用一组启动时固定的 homogeneous pools。每个 pool：

- 存储具有同一 frontier 和 lifetime 的全部 planes；
- 使用固定的原 token page size、plane order、各 plane 行跨度和 page-group count；
- 拥有独立的 physical page-ID namespace 与 free capacity；
- 为每条 active sequence 提供一个 logical address space；
- 由 consumer 通过 block table 直接寻址。

一个 request 的 KV 不要求物理连续，也不与 control lane 固定绑定。所有 active 与 inactive address spaces
共享 pool capacity。普通 reservation 保障下一单元增量；暂停恢复的 reservation 覆盖重建至旧
frontier 及首个真实新单元，并跨 chunk 持有。

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

各 plane 的物理行密度由 `KVPlaneGeometry::tokens_per_row` 单独描述：一行可覆盖 1 或 4 个原 token。
它不改变 allocation、frontier 或 checkpoint 的原 token 域；有效行数按第 4 节从 coverage 推导。

KV Store 可以在任意 token frontier 表示、truncate 或保护 prefix；这不证明模型可以从该位置恢复。
可复用 frontier 必须同时存在完整 StateImage 与 target-defined backend state，具体规则见
[Continuation 与 checkpoint](resource-scheduling-and-context-cache.md#checkpoints)。

---

## 3. Typed pool set 与容量

### 3.1 Pool set

模型配置和 selected speculative backend 在启动时确定 pool set：

```text
ordinary:
    Main Text

MTP:
    Main Text
    MTP

DFlash / DFlash2:
    Main Text
    Draft Full, when the selected draft config contains full-attention layers
```

Qwen3_5 的 speculative backends 在一个 Engine 内互斥，因此该架构当前最多有两个 growing pools。公共资源合同使用启动固定的 pool ID，并不限制模型的 pool 数量。官方 DFlash2
配置的五层全部是 local attention，只需要 Main growing pool，其 draft context 使用 cyclic storage。

| Pool | 内容 | 逻辑 frontier |
|---|---|---|
| Main Text | target attention K/V、code/scale 及模型声明的压缩 index planes | target materialized KV frontier |
| MTP | MTP persistent K/V、code/scale 及模型声明的压缩 index planes | MTP KV frontier |
| Draft Full | selected draft 的 full-context K/V | draft context frontier |

Main Text 与 MTP 使用 Engine 选择的 BF16、INT8-G64、FP8-E4M3FN-row256、NVFP4-G16 或 K8V4
KV profile；Draft Full 使用自己的 BF16 profile。`BFloat16` 名称下的物理 layout 为 BF16 K、FP16 V，
写入端将 BF16 V 一次转换为 FP16。K8V4 是封闭的非对称 profile，不是运行时 bit-width 组合：K 固定为
FP8-E4M3FN-row256，V 固定为 NVFP4-G16。

`PagedKVStorageLayout` 将选定的 closed profile 解析为 K/V data/scale plane schema；target planner 按
layer 展开该 schema 并确定 plane ordinal。Common pool implementation 仍只接收已展开的
`KVPageGeometry`、plane inventory 和 capacity，不解释 storage mode。
Qwen3.5 与 Qwen4Exp 的主 K/V data/scale planes 均使用 `tokens_per_row=1`。
Qwen4Exp 在每个 QSA 层追加 BF16 index plane，使用 `tokens_per_row=4`，与该层主 KV
共享页组和原 token coverage；未完成块的 raw tail 保存在 StateImage。其数学与模型交接见
[Qwen4Exp 执行](qwen4_exp-execution.md)，通用行几何见第 4 节。

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

\(M_{min}\) 分别满足单请求独占时达到 \(S\)，以及 C 个请求各需一个最小页的几何下界。
\(M_{max}\) 是全部 active requests 同时达到 per-sequence ceiling 时的物理上界。

`kv_capacity` 的 explicit policy 解析为：

\[
M=\left\lceil K_{main}/P\right\rceil
\]

并要求 \(K_{main}\ge S\) 且 \(M\in[M_{min},M_{max}]\)。

### 3.3 Automatic capacity

Automatic policy 在权重加载、Graph 校准资源销毁后，以预算内剩余显存 \(F\)、规划余量 \(R\)
和 Graph 重建估计 \(G\) 解析 \(M\)。驱动保留开销已反映在 \(F\) 中，不再重复扣除。
模型 planner 根据已绑定参数和选定执行域提供 affine `SequenceCapacityCurve`：

\[
B(M)=B_{min}+(M-M_{min})B_{step}
\]

其中 \(B(M)\) 是该 Main capacity 对应的显式 runtime Device backing，包含：

- Main 与 selected backend typed pools；
- active/checkpoint State storage；
- block tables 与固定 persistent state；
- unified workspace；
- 启用 offload 时的 MoE backing。

Automatic 选择：

\[
M=
\min\left(
M_{max},
M_{min}+
\left\lfloor
\frac{F-R-G-B_{min}}{B_{step}}
\right\rfloor
\right)
\]

并要求 \(F\ge R+G+B_{min}\)。当 \(M_{min}=M_{max}\) 时直接取该单点。Target 用同一生产 layout builder
生成 \(B_{min}\)、\(B_{step}\) 和最终 layout，并验证 capacity curve；resolver 不复制模型维度公式，也不靠
allocation probing 猜测显式布局大小。MoE offload 的自动策略先保留一条完整上下文，再将剩余预算
用于 expert cache；Graph 变体改变时重新校准，见[引擎架构](engine-architecture.md#7-物理执行与-cuda-graph)。

最终公开的 Main KV capacity 为 \(M\,P\) token-equivalents。Page rounding 只增加 physical padding，
不会扩大单 sequence 的 logical ceiling \(S\)。

### 3.4 Backend capacity

Selected backend 的 logical page capacity仍为 \(L\)。Physical capacity从 Main \(M\) 推导：

```text
backend off:
    Main physical pages = M

MTP with D draft tokens (verification width D+1):
    Main physical pages = M
    MTP physical pages  = M

DFlash / DFlash2 with full-attention layers:
    Main physical pages        = M
    Draft Full physical pages = M
```

MTP 在当前轮生成 drafts。Main 的本轮峰值为 `F+D+1`，MTP 峰值为 `F+D`；两者各自的
reservation 覆盖其实际写入范围，不为跨轮预生成 drafts 额外分配 backend pages。Draft Full
按自己的 context frontier 保留；没有 full layer 的 draft 配置不分配此 pool。

各 pools 物理分离。一个 pool 的 free page 不能变成另一 pool 的 payload。启动规划一次性建立
完整 typed capacity vector，目录采用这些容量，rank 绑定对应存储；运行期不扩容或重分 pool geometry。

### 3.5 Host capacity

StateImage、Main KV 和 selected backend KV 共用 rank Resources 的启动固定 pinned backing。
目录中的 `HostAllocator` 集中分配其 offset/length，`HostDirectory` 管理各 KV layout 的 extent
成员；StateDirectory 直接持有同一 allocator 的 State allocation。物理视图由 ContextStorage 解析。
Host pool ID 指定所属 backing，各 pool 的空闲字节不能互相抵扣。Spark 的 Host context 容量为零。
Host capacity 按实际 packed bytes 和 extent geometry 计费，不扩大 Device capacity 或单 sequence context
ceiling。输入、请求 ledger 与其他 CPU 数据有各自的生命周期，不计入这个物理 context backing。

---

## 4. Page group 与物理 layout

### 4.1 Grouping invariant

只有同时满足以下条件的 planes 才进入同一个 pool：

1. 使用相同 logical cache ordinal；
2. 共享同一个 committed frontier；
3. 一起 reserve、materialize、truncate、retain 和 release；
4. 使用相同 page size；
5. 没有独立释放或独立容量复用的语义。

因此 Main layers 的 K、V、code、scale 和压缩 index 可以共享一个 page-group ID；Main、MTP 与 DFlash Full
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

### 4.3 Plane 行跨度与 Device order

Device pool 使用原 token 页跨度 \(P=64\)。每个 `KVPlaneGeometry` 独立给出 dtype、leading extent
\(X\)、head extent \(H\)、alignment 和 `tokens_per_row` \(s\)，物理行数为 \(R=P/s\)。
当前接纳以下组合：

| \(s\) | 每页行数 \(R\) | Device order |
|---:|---:|---|
| 1 | 64 | PageMajor 或 HeadMajor |
| 4 | 16 | PageMajor |

`kv_plane_rows_per_page()` 为 Device 与 Host 规划提供同一检查：\(s\) 只能为 1 或 4，
\(P\) 必须为正且能被 \(s\) 整除，\(R\) 必须能表示为正的 Tensor extent；四-token 行要求 PageMajor。
dtype、\(X\)、\(H\) 和 alignment 仍是各 plane 的存储事实，\(X\)、\(H\) 必须为正。
行跨度不表示 feature 轴的量化 group size；同一 token 的 K/V code 与 scale 各占一行。
行跨度参与 geometry 相等性和 Host layout 选择；Device backing 的实际 shape 必须与该 geometry 一致。

两种 closed orders 为：

\[
PageMajor:[X,R,H,N_{physical}],\qquad HeadMajor:[X,R,N_{physical},H]
\]

Main Text 与 MTP 的 K/V planes 使用 PageMajor；DFlash Full 使用 HeadMajor。
对这些现有 K/V profiles，\(s=1\)，其中：

- \(X=D\) 表示 K/V 或 quantized code plane；
- INT8-G64 scale plane 使用 \(X=D/64\)；
- FP8-E4M3FN-row256 scale plane 使用 \(X=1\)；
- NVFP4-G16 code plane 使用 packed U8 \(X=D/2\)，scale plane 使用 U8 \(X=D/16\)；
- \(H\) 是 KV heads；
- \(N_{physical}\) 是该 pool 的 physical page count。

对 logical position \(p\)：

\[
b=\lfloor p/P\rfloor,\qquad o=p\bmod P,\qquad r=\lfloor o/s\rfloor,\qquad g=block\_table[b]
\]

\(r\) 定位包含该原位置的物理行；该行是否完整且可读仍由第 4.4 节的 coverage 决定。

Page-major 地址为：

\[
address=base+d\,nb_0+r\,nb_1+h\,nb_2+g\,nb_3
\]

Head-major 地址为：

\[
address=base+d\,nb_0+r\,nb_1+g\,nb_2+h\,nb_3
\]

所有 grouped planes 使用同一个 \(g\)，但使用各自的行跨度、leading coordinate 和 Tensor strides。
Feature 轴连续，地址和容量计算使用能覆盖完整 Tensor span 的整数类型。
Exact persistent codec 由 [`kv_cache_append.h`](../../include/ninfer/ops/kv_cache_append.h) 定义，
consumer arithmetic 由 [`softmax_attention.h`](../../include/ninfer/ops/softmax_attention.h) 与
[`qsa.h`](../../include/ninfer/ops/qsa.h) 定义；
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

对于已提交前沿 \(F\)，逻辑页 \(b\) 内的有效原 token 数为：

\[
c=\operatorname{clamp}(F-bP,0,P)
\]

跨度为 \(s\) 的 plane 只有行 \([0,\lfloor c/s\rfloor)\) 完整有效；行 \(r\) 对应原位置
\([bP+rs,bP+(r+1)s)\)。例如 \(c=5,s=4\) 时只有第一行有效，第五个 token 不会提前发布第二行。
所有 planes 共用同一原 token coverage，不维护独立行前沿。Consumer 还须遵守各自的因果读域，
不能把物理行存在或 chunk 末端的 coverage 当作每个 query 的可见范围。

尚未形成完整行的模型状态由其 StateImage 合同保存；页组本身不能替代该状态。
Provisional 写入或复制无效后缀字节均不扩大 committed coverage。

### 4.5 Page payload

各 plane 的完整页 payload 与 page-group payload 为：

\[
PlanePageBytes=sizeof(dtype)\,X\,H\,(P/s)
\]

\[
PageBytes=\sum_{plane} PlanePageBytes
\]

Startup physical bytes 由各 plane slab 的完整 span 与 alignment 得到。不同 pools 的 `PageBytes` 可以不同，
但一个 pool 内的所有 page groups 等价。Device slab、Host packed offsets、传输字节和 capacity query
均使用同一 plane geometry；不能将四-token plane 按 \(P\) 行计费。

Host layout 将每个 plane 的起点按 256 B 对齐，末端按 256 B 对齐形成 `page_stride`。
该 stride 用于 Host allocation；payload 字节不包含 plane 间和页末 padding。

`P=64` 同时满足当前 32/64-key Attention tiles、128-token aligned prefill chunks、有限 block-table
metadata 和 bounded tail slack。Prefix hit granularity不参与 page-size 选择。改变 page size、grouping
或 closed plane order 都是架构变更。

---

## 5. Logical page 与 replicas

### 5.1 Logical page identity

Device page ID 只是当前 Device replica 的物理位置，不是 prefix identity。集中 PageDirectory 为每个
pool 维护 generation-checked logical pages。`PageId` 包含 pool、descriptor index 和 generation，
实例身份由执行 session 确定；跨执行边界不传目录 owner 指针：

```text
LogicalKVPage
├── object generation
├── content epoch
├── committed columns [0, P]
├── required/protected columns
├── history references
├── active references and writer state
├── optional Device page-group lease + valid columns
├── optional Host extent membership
└── transaction pins
```

一个 logical page 的 canonical content 由 `content epoch + committed columns` 标识。
Committed/protected columns 始终计原 token，跨度不同的 planes 由此推导有效行数。
Speculative 或尚未提交的 bytes 不扩展 committed coverage。

同一 logical page 在所有引用它的 history 中保持相同页序号。Fork 和 view 按原位置共享
前缀，COW 与增长创建新 logical page，截断只删除后缀；一个 history 内也不重复引用同一页。
因此，核对选中页的完整持有者时，可以在各检查点覆盖内按该页序号查询实际 handle。

Replica 对 checkpoint 所需前 \(n\) 列有效，当且仅当：

\[
replica.epoch=page.epoch
\quad\land\quad
replica.coverage\ge n
\]

### 5.2 Device replica

Device replica 使用 `4` 的 consumer-native plane layout。目录 `PageLease` 独占一个 pool-local physical page group；值型 `PageSlot` 包含 pool、index 和
generation。PageAllocator 校验分配身份，rank 按已接受的有序命令使用对应物理位置。

Pool 对同一容量单位区分：

```text
allocated page lease
reserved but not materialized page
globally available page
```

\[
allocated+reserved+available=capacity
\]

PageAllocator 优先延续前驱页后的完整连续范围，其次选择完整 free run，最后使用离散 runs。
Materialize 把 reservation 转成 lease，不进行第二次容量准入；dematerialize 把 lease 还回同一 reservation。普通 unit 结算
释放余额；受保护恢复则保留未来覆盖所需余额，直到真实新进展后归还。Checkpoint 别名和 history
共享引用不重复占物理页。

### 5.3 Host replica

Host replica 使用 logical-order packed `HostKVPageLayout`：

- 不保存 Device page ID 或 block-table holes；
- 每个 page 包含该 typed pool 的全部 grouped plane payload；
- variable-size extent只为实际 page count 付费；
- Main/backend layouts可以在同一 arena 中分配不同 stride 的 extents。

HostAllocator 在启动固定 backing 上管理有界 variable-size ranges。State 与不同 KV layouts 竞争同一容量，彼此没有固定
配额。申请必须满足 alignment 和连续 extent geometry；`free_bytes` 只是占用摘要，不是可分配性的
充分证明。共享 Host extent 按实际 allocation 计一次，pending destination 在 publication 前已经占用容量。

每个复制读者持有具体副本的 epoch 与 coverage。Host 与 Device 读者分别保护实际来源；移除最后
一个 history/checkpoint 引用不会回收仍被复制读取的 bytes。Host extent 可在其他成员退役时分裂，
被读的页范围保持原地址，新的成员 capability 不影响已提交 DMA。Host 副本替换持有独立目的
claim，发布前不能取得指向旧副本的新 Host 读者。

### 5.4 Replica transfer

Device COW 与 D2H/H2D transfer 复制页组的全部 plane payload，按各 plane 的实际行数、bytes 和
offset 搬运原 words。不同密度的 plane 不单独分配、保留或回收。传输可以把相邻 physical IDs
合并成更少的 transfer runs。
DevicePageStorage 按已绑定的 plane geometry 选择 2D copy：短 PageMajor run 可以合并等宽、等间距的 planes，
HeadMajor 可以选择 page 或 head 作为提交外层。只在减少调用数且 pitch 合法时使用合并路径；
Host layout 与 payload 不变。rank 准备完成后导出无地址 `KVTransferProfile`，包含 payload、
Device copy 次数和按 run 长度选择的 Host copy 次数；目录使用该实际计划报价。迁移在目的位置
确定后形成真实 Host/Device runs，物理执行核对计划与实际工作量；观测记录实际字节和 copy operations。来源选择时尚无
目的物理 ID，只按已知 Host runs 给出名义连续传输估计，不能将它当作实际提交次数。
一个 replacement replica 的 publication 顺序为：

```text
reserve destination
  -> copy page payload
  -> verify epoch and committed coverage
  -> publish replica
  -> release source when no longer required
```

Copy 进行时 source 与 destination 都被 pin。复制、mapping upload 和 consumer 使用有序依赖；
consumer 不能在所需 copy 与 table upload 完成前读取目的页。目录在迁移完成后发布 replica，
唯一有效 source 在最后消费者退役前保持可用。

### 5.5 Descriptor lifetime

Logical descriptor 不构成第三份 payload。它可以在 Device-only、Host-only 或 Both placements 下继续存在。
只有当 references、replicas 和 transaction pins 均为零时，descriptor 才能回收并推进 generation。

---

## 6. KV history 与前缀引用

`HistoryOwner` 是一个 pool 的唯一可写 owner，`HistoryPrefix` 持有同一实际历史的 `[0, F)`。
引用只拥有逻辑内容，不固定 Device residency。Main 与 backend 使用各自的 owner；它们的前沿
关系由模型决定。完整快照将各 pool 前缀与同次执行的 StateImage、metadata 和输入身份一起保存。

```text
HistoryDirectory
├── generation-checked history descriptors
├── shared prefix directory / private append path
├── committed frontier and provisional memberships
├── counted retained frontiers
├── unit / recovery page reservation
└── optional execution-row lease
```

每个 history 对其页面贡献一次 required coverage，并按最深存活前缀贡献 protection。
引用增减只更新改变的范围，不扫描 Session 的 checkpoint 表。释放可写 owner 后，目录缩到最深
存活前缀；最后一个引用消失后释放其页引用。目录节点的结构共享不决定内容保护或资源归属。

保存非对齐前缀不复制尾页。同一个 writer 可以继续追加，读取者只读自己保留的前沿。
回退将覆盖保留内容、或独立 writer 竞争同一 partial tail 时，先分支目录并建立私有 COW 页。
公共发布与私有保存使用同一前缀机制，不因保留用途改变物理布局。

每个 pool 在启动时建立固定地址的 Device block-table matrix：

\[
block\_tables[N_{logical},C]
\]

其中 `N_logical=L`，`C=max_concurrency`。活动 history 租用一行，条目为 pool-local I32
physical page ID。绑定发布完整 mapping，追加或 COW 只发布改变的 slice。Execution row 不拥有
logical pages 或 frontier；暂停后恢复可以租用另一行。

## 7. 绑定、写入与结算

### 7.1 绑定

`binding_demand` 根据来源前缀、目标位置与峰值位置计算缺失副本、COW 和增长页。
它接受本次授权退休的前缀列表，计算退休后的实际保护；查询本身不消费来源。
`HistoryBinding` 随后持有来源引用、目的 history、Device reservation、待恢复副本和 execution row；
ContextTransaction 持有绑定、State 目的、Host 预留及纯值动作：

```text
reserve complete binding
  -> enqueue restore, conflicting-tail COW and reserved-row mapping
  -> wait for this batch's completion event
  -> publish replicas and consume transferred source ownership
  -> adopt writer
```

已有只读前缀可以保留在被接管的 history 上。即使首个 unit 不追加 token，绑定仍取得 partial
尾页的 writer 身份，防止多个待运行请求同时把它报价为零 COW。MTP 等需要从较早位置开始的
情况由原生模型提供目标位置；消费来源前，原前沿与内容继续有效。

Session 使用目录统一检查 State 与所有 enabled pools。暂停恢复取得“重建至旧 frontier + 首个真实新单元”
的完整覆盖。任一 pool 不足则不执行授权退役。已经接纳的退役按请求合同生效；未采用目的的
取消必须结束实际 DMA 后再释放 reservation 与来源引用。
复制期间按 logical page 的实际 writer 阻塞冲突执行；来源前缀所在的 history 不能替代写入者身份。

### 7.2 显式写入区间

`write_pages(history, begin, end)` 计算增长与 COW 需求。`prepare_write(owner, begin, end, transaction)`
从实际写入起点处理保护、分支和映射，在已取得的 reservation 内物化页面；它不推进 committed
frontier。已准备的更长 speculative mapping 可以保留到结算，较短的写入请求不隐式裁剪它。

Session 先取得调用集合的完整执行和 typed 资源许可，再进入不可撤销的目录准备。prepare_write/trim
产生 COW 与 mapping 动作；此后取消改变输出采用意图，已经接受的动作仍执行至退役，准备失败
使执行实例失败并排空。绑定的独立目的预留在尚未提交时可以撤销。普通 unit 只预留当前峰值；恢复 reservation
跨 chunk 保留到完整恢复覆盖，其他请求不能使用其余额。算子与 Graph 继续只消费 typed views、
selectors 和固定地址的 block tables。

### 7.3 提交与回退

`prepare_commit` 校验实际接受前缀并整理未采用的 provisional memberships，保留退役租约；
此时 committed frontier 和 coverage 不变。Executor 接收物理提交后，事务 `adopt` 才推进各 pool 的
KV frontier，模型同步采用 State 与 metadata。Rejected suffix 的 bytes 可以残留，但不扩大有效范围。
不同密度的 plane 仍使用 `floor(coverage/tokens_per_row)`。

`trim` 删除超出采用前沿的 provisional memberships；若旧前缀仍需更长内容，则为 writer 分支，
保留旧目录。真正覆盖 partial tail 前由 `prepare_write` 执行必要 COW。普通结算归还未用页，
恢复结算保留到目标覆盖所需的余额；到达旧 frontier 本身不结束恢复保护。

逻辑 trim 与物理回收分别发生。被旧执行引用的页和 row 租约保留在 ContextTransaction，直到设备
消费者完成。提交准备记录恢复覆盖需求，退役归还旧页后再补足 reservation；已稳定状态的独立
结算使用 `settle_growth`。
这不会为旧页和下一轮预留暂时占用两份容量。当前执行槽在结算完成前不可复用。

### 7.4 释放与共享计费

暂停或结束释放 execution row、active references 与未用 reservation，完整快照继续持有其精确
前缀。删除 checkpoint 只撤销相应引用。`released_pages` 按本次实际删除的前缀集合计算最后
物理持有者，公共别名与共享完整页不重复计费；StateImage 也按实际内容引用数结算。

从 mapping publication 到 GPU unit 完成，selected rows、页成员和实际消费的 payload 保持稳定。
前缀引用和已选定 mapping 由协调侧集中更新；COW source leases、页表上传源和 row 的实际复用
遵守该 execution 的完成依赖。取消提交中的传输不提前归还空间；未录入完成 fence 的部分提交
失败时，由 rank 排空实际 stream 后再允许目录退役。

## 8. 部分页与副本 coverage

一个页组的 Device 与 Host 副本可以覆盖不同长度。例如 Device 已有前 48 列，Host 只保存前
40 列：Host 对 `[0,40)` 仍有效；若仍有引用需要 48 列，则不能丢弃 Device 或从该 Host 恢复 48 列。
最后需要较长范围的引用释放后，可直接使用较短 Host 副本保留剩余前缀。

写入准备只允许修改未被其他内容 owner 保留的范围。原位改写会缩小 Host 的有效 coverage，
随后 Device 提交扩展自己的有效范围。需要卸载更长前缀时，先申请新 Host extent 并完成复制，
再替换旧 membership；报价包含新旧副本暂时共存的 Host 空间。共享 extent 的其他页继续保留。

独立 writer 的 COW 复制全部 plane 原 words，前沿仍精确到原 token。例如 (P=64,F=1000)，
两个 writer 共享前 15 个完整页，各自写入最后 40 列所在的 partial page；只读保存本身无需复制。

---

## 9. Speculative 与非 growing KV

### 9.1 Independent pool frontiers

MTP 和带 full layer 的 DFlash backend 使用 ContextDirectory 的 backend pool，不建立私有分配器。

一次 speculative unit 中，Main 与 backend：

- 分别从各自 unit reservation materialize；
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
| DFlash/DFlash2 local sliding-window K/V | fixed per-sequence StateImage |
| DFlash/DFlash2 boundary-local snapshot | fixed checkpoint StateImage |
| Vision/query temporary K/V | RankProgram workspace |

DFlash/DFlash2 cyclic K/V 使用自己的 `CyclicKVCacheLayerView` 与 modulo/window 语义；它不持有 page ID、
block table 或 growing reservation。模型配置决定其 layer count、heads 和 window。

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
plane_row     = page_offset / tokens_per_row
```

随后使用 `4.3` 的 closed plane order计算元素地址，并检查该 plane row 在有效完整行范围内。
Batched consumer先用 `table_rows[b]` 选择 table row，再执行同一 translation。
K/V views 的现有 codec planes 使用 `tokens_per_row=1`。其他密度的 plane 由对应 Op 的 typed view
表达；storage geometry 本身不扩展 `PagedKVLayerView` 的数学合同。

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

在需要新 mappings 的 execution unit 前：

```text
Session materializes pages and prepares mapping actions
  -> Executor transfers owning value descriptors
  -> rank copies mapping IDs into its pinned execution-slot staging
  -> uploads the changed table slices
  -> launches/replays consumers on the ordered stream
```

同一 boundary新增多个 pages或重新激活完整 membership时使用批量 publication。Mapping update必须先于
consumer，且 replay in-flight期间不得改写同一 row。每次 H2D 的 Host 源也必须保持到传输完成，
不能靠 stream 顺序允许 CPU 提前覆写 pinned mapping 数据。Compute preparation 与独立上下文
迁移各有执行槽、mapping staging 和完成事件，其地址在启动后稳定。

---

## 12. 核心不变量

1. 每个 Device page-group lease 在所属 pool 中至多承载一个 logical page replica。
2. 同一 pool 只组合共享 frontier、lifetime、page size与allocation语义的planes。
3. 一个 pool 的全部 planes 对同一 logical block 使用同一个 page-group ID，行密度不产生第二份页表。
4. Device occupancy 按 allocated leases与reservations计数；logical aliases不重复计费。
5. Logical page identity、content epoch与physical page ID彼此独立。
6. Valid frontier精确到原 token；plane 有效完整行数为 floor(coverage/tokens_per_row)。
   Page boundary不改变Attention或checkpoint语义。
7. Published checkpoint所需coverage不可被writer覆盖；任一logical page至多一个writer。
8. 只读前缀不复制尾页；writer 冲突或覆盖受保护内容时必须先分支并 COW。
9. Host/Device replacement在copy与epoch/coverage验证完成后才发布。
10. 普通 unit 结算归还未用容量；恢复 reservation 跨 chunk 保留至真实新进展或终态。
11. 一个GPU execution unit内membership、block tables、replicas与read frontier稳定。
12. 无活动 writer 的 history 不占 execution row；execution row 不拥有 logical pages。
13. Main与backend pools分别reserve、materialize、commit和truncate。
14. Growing-cache consumer只通过paged view和block table访问KV，不取得allocator或ownership authority。
15. Kernel correctness不依赖physical page ID连续性，也不通过gather建立request-contiguous KV。
16. Checkpoint可复用性由完整target continuation证明，KV page存在本身不构成hit。

---

## 13. 实现位置

| 职责 | 主要位置 |
|---|---|
| 纯页几何、Host packed layout | `src/core/paged_kv_geometry.*` |
| Device layout 规划与 public paged views | `src/core/paged_kv_cache.*` |
| closed K/V data/scale plane schema | `src/core/paged_kv_storage.h` |
| 集中目录、事务及值动作 | `src/models/common/context/directory.*`, `contract.h` |
| 连续页、Host 范围分配 | `src/models/common/context/directory/allocation.*` |
| logical pages、replicas、coverage 与 references | `src/models/common/context/directory/pages.*` |
| history、前缀引用、写入与绑定 | `src/models/common/context/directory/history.*` |
| Host extent membership 与部分回收 | `src/models/common/context/directory/host.*` |
| rank 本地页、State、页表视图与 DMA | `src/models/common/context/storage.*` |
| 异步动作槽、staging、完成依赖与观测 | `src/models/common/context/transfer.*` |
| StateImage 几何与字段视图 | `src/core/state_image.*`；架构声明字段与类型化视图 |
| State 逻辑身份、slot 与副本预留 | `src/models/common/context/directory/state.*` |
| 完整快照与记录 lease | `src/models/common/context/snapshot.h`, `checkpoint_store.h` |
| 原生上下文、checkpoint 和恢复数学 | `src/models/qwen3_5/session/`, `src/models/qwen4_exp/session/` |
| model pool layout 与 capacity curve | 各架构 `program/planning/` 或 `program/planning.*` |
| growing-cache Ops | `include/ninfer/ops/`, `src/ops/` |

Exact model state 和 backend mathematics 见
[Qwen3.5 model](qwen3_5-model.md)与 [DFlash](dflash.md)；persistent KV codec 和 causal consumer
numerical contract 由上表中的 growing-cache Ops 定义。路径用于定位当前实现，不把文件或类名本身提升为
外部接口。
