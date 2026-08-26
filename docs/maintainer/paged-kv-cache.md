# NInfer Paged KV Context Store

本文定义NInfer在单GPU、单resident model、少量并发请求下的growing KV存储架构。它是
[NInfer Engine架构](engine-architecture.md)所依赖的physical KV substrate。

本文定义 KV pool layout、logical-page ownership、Device/Host replica、容量预留、逻辑 frontier、prefix
sharing 和生命周期，同时定义 Device block-table representation、single-sequence 与 batched paged KV
execution view、受影响 Op 的状态效果、kernel 寻址约束和性能准入条件。具体 allocator 算法和 CUDA
kernel 代码不由本文规定。

顶层所有权以Engine架构文档为准，continuation/checkpoint、placement policy和resource transaction以
[资源调度与上下文缓存架构](resource-scheduling-and-context-cache.md)为准。

ResourceManager只选择logical source、placement intent和victim actions；Program中的KV Store、allocator、
logical pages、Device/Host replicas、references和reservations是KV物理事实的唯一权威。KV Store不选择请求
顺序、candidate价值或eviction victim，也不向ResourceManager输出一套可重放的physical delta。

---

## 1. Requirements

- `max_concurrency=1..8` 的 active requests 共享各类 growing KV capacity；
- 单个 request 可以使用 main KV pool 的大部分容量，不按 slot 平均切分；
- 不同 request 的物理 KV 不要求连续；
- active、catalogued checkpoint 和 speculative provisional KV 使用同一套 typed physical accounting；
- private continuation 可以转移给 active request；shared immutable full pages 可以被多个 address spaces
  引用，非 page-aligned shared tail 在产生新 writer 前复制为 private page；
- logical page 可以只保留 Device replica、只保留 packed Host replica，或在 transfer boundary 同时具有两份
  epoch/coverage 一致的 replicas；
- active request 一旦 admission，其声明范围内的 prefill、decode 和 speculative temporary growth
  都有 completion capacity guarantee；
- 一个 GPU execution unit 期间，page mappings 和 logical valid frontiers 保持稳定；
- BF16、INT8-G64、FP8-E4M3FN-row256 以及 target 定义的其他固定 bytes-per-token layouts 使用同一
  管理语义；
- common allocator 不理解 GQA、MHA、MLA、MTP 或 DFlash 等模型语义；
- single-sequence prefill/cached consumers 和 batched ordinary/MTP/DFlash decode consumers 都直接消费
  paged KV，不要求任何 sequence 的 growing KV 物理连续；
- paged 寻址不能引入 gather-to-contiguous copy，并必须通过与冻结 baseline 相同 workload 的性能准入。

### 1.1 Non-goals

- active request preemption/swap、weight offload 或跨 GPU storage；
- shared writable page 或多 writer；共享只适用于 immutable committed prefix，分支写入使用 private
  destination；
- 没有完整 target checkpoint 证明的 arbitrary token-only longest-common-prefix reuse；
- 用一个 universal raw-byte allocator 在 serving 期间动态重分不同 KV layouts 的显存；
- 为 bounded cyclic KV 或 operator transient K/V 强行提供同一种 paging；
- 在 serving 期间改变 pool layout 或 page size；
- scheduler、active-batch formation 或其他非 KV model-execution policy；
- 借 KV storage 设计重组 Attention 源码目录、重命名 Attention family 或扩大支持的数学 domain。

---

## 2. Three independent granularities

KV 设计必须区分三个彼此独立的粒度：

| 粒度 | 含义 | 固定值或约束 |
|---|---|---|
| allocation granularity | 一个 homogeneous pool 一次取得或释放多少 token 的 payload | 当前各 growing pool 均为 `P=64` |
| valid-frontier granularity | 哪些逻辑 KV positions 可以被读取 | 1 token |
| reusable-state granularity | 哪些 prefix positions 拥有完整 continuation state | 由完整 sequence state checkpoint 决定 |

Page boundary 既不是 Attention 语义边界，也不是 prefix-cache hit boundary。一个合法 frontier 可以位于
page 内部任意 token offset。

当前 Qwen3.6 target 的 Linear Attention state 不能从 KV prefix 单独重建。Prefix lookup 只能使用保留了
完整 continuation state 的 target-declared checkpoint。可复用位置包括 private endpoint、typed rewrite、
被采纳的 sparse long anchor 和 shared stable-prefix checkpoint；具体 Move/Fork/Restore 语义见 §10。仅有
token 或 KV page match、但不存在完整 StateImage/backend continuation 的位置不是可复用 frontier。

Paged KV Store 本身可以按任意 logical frontier truncate；这不表示上层拥有在任意位置恢复整个模型的
能力。Reusable position 必须由 target 的完整 checkpoint 证明；增加新的 checkpoint 不需要改变 page
geometry。

---

## 3. KV pool set and capacity model

### 3.1 Homogeneous pools

一个 Engine 在 startup 时由 exact target、KV dtype 和 speculative backend 建立一个固定的 KV pool set：

```text
ordinary Engine:
    Main Text KV Pool

MTP Engine:
    Main Text KV Pool
    MTP KV Pool

DFlash Engine:
    Main Text KV Pool
    DFlash Full-Context KV Pool
```

MTP 与 DFlash 是 engine-wide mutually exclusive backends，因此一个 Engine 当前最多包含两个 growing
KV pools。DFlash local cyclic KV 属于 fixed state，不在这个 pool set 中。

这些名称属于 target/runtime 对 pool 的使用方式。Common KV Store 只看到一组 immutable pool layouts
和 opaque pool handles，不根据 feature name、attention type 或 runtime string 分派。

### 3.2 Capacity is a vector

Engine 的 startup configuration 给出：

```text
max_context                  单个 sequence 的逻辑上限 S
kv_capacity                  Explicit(K_main) 或 Automatic(R) 的 sizing policy
max_concurrency              active sequence 上限
selected speculative backend off、MTP 或 DFlash
context_cache.host_kv_capacity_bytes  packed Host KV arena容量，0表示无Host KV replica
```

`max_concurrency` 只确定 control lanes、block-table rows 和 active fixed-state guarantee，不把 KV
容量切成等份。`EngineOptions.max_context` 只约束任何单条 sequence 的 frontier；
`EngineOptions.kv_capacity` 解析为 Main pool 的 shared physical page-group 数。Backend capacity 不是独立
用户配置，而由 resolved Main capacity、selected backend、`max_concurrency` 和 exact target frontier
contract 唯一导出。

这里的`kv_capacity`只解析Device typed pools。Host KV capacity经启动配置直接进入一块pinned packed arena，
由Main和selected backend layouts共享，不参与`M`求解，也不能替代active Device entitlement。

Planner 将 Main contract 归一化为 physical page capacity：

```text
L     = ceil(S / P_main)
M_min = max(L, max_concurrency)
M_max = max_concurrency * L
M     = resolved physical page groups in [M_min,M_max]
Main physical page groups = M
per-allocation logical page capacity = L
```

Explicit policy 使用 `M=ceil(K_main/P_main)`。Automatic policy 带有必须保留的 device headroom `R`，在
权重加载并同步后查询剩余显存 `F`。CLI/server 的 `R` 固定为 1 GiB；Engine policy 可以显式指定该值。
Exact target 用同一个完整 physical candidate builder 得到：

```text
B(M)     = persistent + unified workspace + CUDA Graph allowance
B_min    = B(M_min)
B_step   = B(M_min+1) - B(M_min)
M_auto   = min(M_max, M_min + floor((F-R-B_min)/B_step))
```

Automatic 要求 `F>=R+B_min`；当 `M_min=M_max` 时直接取该单点。未达到 `M_max` 时，最终
`planned_slack=F-B(M)` 位于 `[R,R+B_step)`，从而不会把显存压到一个 page-group 以下的随机余量。
`B(M)` 包含 Main 与 selected backend 的 coupled growing pools，且
persistent 部分直接来自生产 `LayoutBuilder`，workspace 来自生产 schedule 的 capacity recipe；common
resolver 不维护模型维度或 bytes-per-token 公式，也不做 allocation probing。最终 plan 再由同一 builder
按 `M` 生成并核对 reservation curve。

`R` 是 capacity solver 刻意不消费的 sizing headroom。CUDA allocator、context 和 module 的物理占用不全
等同于 arena payload；因此 Instance 与 Graph 完整建立并同步后再次查询实际 free memory，并与 policy、
planned slack 一起报告。默认 1 GiB 同时吸收这部分差值，并为同一 GPU 上后续的小额占用留下实际余量。

Engine 对外同时报告 configured `max_context` 和 resolved Device `kv_capacity=M*P_main`。最后一个 physical page
的 rounding tail 只属于 storage padding，不能让 sequence frontier 超过 `S`。Admission 以 page-group
entitlement 计费；Device pool 必须同时容纳 unique mapped replicas、active future-growth reservations 和
transaction destinations，同一 shared logical page 的多个 address-space references 只计一份 physical
occupancy。Host KV 使用独立的 packed byte capacity，不扩大任何 active sequence 的 logical ceiling 或
Device execution guarantee。

内部使用 target-derived capacity vector：

```text
KVCapacity = {
    main_page_groups,
    optional_backend_page_groups,
}
```

Main per-allocation logical capacity 是 `L`，physical capacity 是 `M` page groups。Backend 的 logical
capacity 和 physical page-group count 必须分别描述：MTP 的额外 physical headroom 不能被解释为更大的
per-sequence logical context。Backend sizing 由 selected backend 的 fixed frontier relation 推导，不能从
main/backend bytes-per-token 比例推导，也不能独立配置。Explicit 要求 raw `K_main >= S`；两种 policy
都要求 `M >= max_concurrency` 且 `M <= max_concurrency*L`，分别保证一个 request 可达到 `S`、每个
active lane 至少可获得一页、且不接受 lanes 永远无法使用的 physical capacity。

### 3.3 Physical separation, coordinated capacity

物理分池不表示 Main 与 backend 可以互借容量，但两者的即时 mapped extent 也不要求保持大小关系。
对 request `r`，分别记 Main 与 backend 的 reservation entitlement 为 `E_main(r)`、
`E_backend(r)`，当前已取得物理 page 的范围为 `M_main(r)`、`M_backend(r)`。运行期必须分别满足：

```text
0 <= M_main(r)    <= E_main(r)
0 <= M_backend(r) <= E_backend(r)
```

不存在普遍成立的 `M_backend(r) <= M_main(r)`。两种 backend 的临时推进关系不同：

- MTP 的 autoregressive draft 会在 target verify 前写入 provisional backend KV；一个 round 内 Backend
  mapped extent 可以暂时领先 Main。target verify 只需按自身可见域 materialize Main，不能为了维持虚假的
  mapped-frontier 大小关系而额外占用 Main page；
- DFlash Full 只保存从 target-produced committed features 生成的 persistent context，proposal query K/V
  是 transient workspace，local K/V 是 fixed cyclic state，因此它通常落后于 Main。

已提交的稳定 KV frontier 仍满足 backend 不领先 Main；MTP 的领先只存在于 round 内已映射的 provisional
范围，不能被解释为已提交状态。

当前 registered targets 的 Main、MTP 和 DFlash Full pools 都使用 `P=64`。令
`C=max_concurrency`，MTP draft window 为 `K_draft`。Startup capacity profile 固定为：

```text
speculative_backend = off:
    Main physical=M, logical-per-allocation=L
    no backend growing pool

speculative_backend = MTP:
    Main physical=M, logical-per-allocation=L
    MTP physical=M + C*ceil((K_draft-1)/P), logical-per-allocation=L

speculative_backend = DFlash:
    Main physical=M, logical-per-allocation=L
    DFlash Full physical=M, logical-per-allocation=L
```

MTP 的额外 physical groups 覆盖最多 `C` 条 concurrent rows 各自相对 Main entitlement 多出的
`K_draft-1` provisional positions；它不增加 block-table width，也不允许任一 allocation 超过 `L` logical
pages。
DFlash Full 不存在这类 provisional lead，因此不需要额外 headroom。

两个 pools 不共享 physical pages；它们只是为相同数量的 logical 64-token groups 分别规划 typed
payload。由于 materialize 时机不同，一个 pool 有空闲 page 而另一个 pool 已全部 materialize 是正常
状态。例如 MTP draft 可能使 Backend 暂时多 materialize 一个 page，DFlash 则可能让 Main 多
materialize 若干 pages。任一 pool 都只需在自己的 entitlement 内取得页面，不能借用、重解释或动态
重分配另一个 pool 的 typed capacity。

因此，startup 必须在建立任一 pool 前完成全部 typed slabs、fixed state、workspace 和 driver allowance
的显存规划。Explicit 无法兑现时拒绝 Engine configuration，不静默降低 capacity；Automatic 根据权重
加载后的剩余显存只解析一次最大合法 `M`。最终 pool 建立后两种 policy 都不再扩容、重分配或重建
Graph。Checkpoint replica回收后，若一个request set仍满足Main entitlement contract，却无法取得必要
backend reservation，这是 sizing/accounting invariant violation；运行期按 Engine failure 处理，不能等待
碰巧释放 backend pages。

任一 pool 已 materialize 全部 entitlement 而另一个 pool 仍有 free pages，都不表示容量 sizing
失败。空闲 pages 不能解释为另一种 payload，属于 coordinated profile 的 typed slack，不是外部碎片。
Exact backend 在 startup 时已固定，Engine 应一次性规划各 typed slabs，而不是引入 variable-size raw
allocator、runtime repartition 或 compaction。

---

## 4. Pool layout and grouping rule

### 4.1 The grouping invariant

只有同时满足以下条件的 cache planes 才能组成一个 homogeneous page-group pool：

1. 使用同一 cache-ordinal domain；
2. 由同一个 logical valid frontier 管理；
3. 一起 reserve、materialize、truncate、retain 和 release；
4. 适合相同的 token page size `P`；
5. 不存在有意义的独立 ownership 或独立容量复用需求。

因此 grouping boundary 由 frontier 和 lifetime 决定，不由“属于同一个 request”或“属于同一个模型”
决定。

当前 pool grouping 是：

| Pool | Grouped planes | Pool frontier |
|---|---|---|
| Main Text | 所有 target full-attention layers 的 K、V 和 optional code/scale planes | target materialized KV frontier |
| MTP | MTP layer 的 K、V 和 optional code/scale planes | MTP KV frontier |
| DFlash Full | DFlash full-context layer 的 K、V planes | DFlash context frontier |

Text、MTP 和 DFlash 不组成同一个 physical page group，因为它们可以具有不同 valid/provisional
frontiers、不同 retention readiness 和不同 consumer geometry。

### 4.2 Allocator-visible layout

每个 pool layout 至少给出：

- page size `P`；
- plane 数量；
- 每个 plane 的 element type、logical page payload bytes、physical strides 和 alignment；
- pool page-group count；
- target-owned plane index mapping。

KV Store 只使用这些 storage facts。Head count、head dimension、GQA/MHA/MLA 关系、quantization 公式、
MTP/DFlash 身份以及 Attention 语义全部由 target layout 和 consuming Op 解释。Allocator 不包含
attention-type variant。

Consumer 对 K/V plane 使用统一的逻辑坐标 `K/V[d,h,p]`。Physical axis order 由 homogeneous pool
固定，不由 allocator 或单次 request 选择：

```text
Pool            K/V or code plane       INT8-G64 scale plane       FP8-row256 scale plane
Main Text/MTP   [D, P, Hkv, Nphysical]  [D/64,P,Hkv,Nphysical]     [1,P,Hkv,Nphysical]
DFlash Full     [D, P, Nphysical, Hkv]  not used                    not used
```

Main Text/MTP 使用 contiguous page-major order。对 element bytes `E` 和第一维 extent `X`（K/V/code
为 `D`，INT8 scale 为 `D/64`，FP8-row256 scale 为 `1`）：

```text
nb[0] = E
nb[1] = X * nb[0]
nb[2] = P * nb[1]
nb[3] = Hkv * nb[2]
```

DFlash Full 使用 contiguous head-major page-run order：

```text
nb[0] = E
nb[1] = X * nb[0]
nb[2] = P * nb[1]
nb[3] = Nphysical * nb[2]
```

Plane 之间可以有 slab alignment，但单个 plane 内不允许 padding、axis permutation 或 target-specific
第三种 physical order。对逻辑 position `p`：

```text
b = floor(p / P)
o = p mod P
g = block_table[b]

Main/MTP:    K[d,h,p] = k_pages[d,o,h,g]
             V[d,h,p] = v_pages[d,o,h,g]

DFlash Full: K[d,h,p] = k_pages[d,o,g,h]
             V[d,h,p] = v_pages[d,o,g,h]
```

INT8 和 FP8 code 使用同一公式；INT8 scale 把 `d` 换成 quant group `d/64`，FP8-row256 scale 的
第一维固定为 0。K、V、code 和 scale 不保存各自的 page pointer table，而是使用同一个 pool-local
page-group ID `g`。

INT8 与 FP8 K code 第一维仍使用相同 D256 physical coordinate 和地址公式，但其值是 causal
Attention producer/consumer 配对拥有的私有表示；allocator 不解释 rotation、logical decode 或 Q
preparation。V code 继续直接表示原坐标 V。

FP8 D256 profile固定为每个 `(position,kv_head)` 的完整 row各一个 FP16 scale，K/V 合计 payload 为
`256+2+256+2=516` bytes。对 represented BF16 source row `x[256]`：

```text
a = max_d abs(FP32(x[d]))

a == 0:
    scale_bits = FP16(+0)
    code[d]    = E4M3FN(+0)

a != 0:
    raw_scale  = a / 448
    scale_bits = FP16_RNE(clamp(raw_scale, 0x1p-24, 65504))
    s          = FP32(scale_bits)
    inv        = FP32(1 / s)
    code[d]    = E4M3FN_RNE_SATFINITE(FP32(x[d]) * inv)

decode[d] = FP32(E4M3FN(code[d])) * FP32(scale_bits)
```

V 的 `x` 是原坐标 source row。K 的 `x=R*k`，其中 `R=H256/16` 是 producer/consumer配对拥有的固定
正交变换；logical K 为 `R^T*decode(K)`，logical V 为 `decode(V)`。Standalone append与 fused
append写出相同 persistent representation。Hadamard、transient Q preparation以及具体 MMA staging不是
allocator metadata或新的公共 Tensor边界。

Common allocator 接收已经确定的 closed plane order、bytes、strides 和 alignment，不从中推导 head、codec
或 Attention 语义。Production wrapper 只接受其 route 对应的上述 closed stride formula。Kernel 在
page/tile 粒度计算 base，不在 hot loop 中解释 arbitrary layout，也不存在 runtime layout selector。

### 4.3 Closed physical orders

Main Text 与 MTP 使用 page-major：

```text
plane
├── physical page 0: head 0, head 1, ... head H-1
├── physical page 1: head 0, head 1, ... head H-1
└── ...
```

一个 page-group ID 在每个 Main/MTP plane 中选择一个 contiguous `P*Hkv` slice。

DFlash Full 使用 head-major page run：

```text
plane
├── head 0: physical page 0, page 1, ... page N-1
├── head 1: physical page 0, page 1, ... page N-1
└── ...
```

一个 DFlash page-group ID 在每个 head slab 中选择一个 `P`-position slice；这些 head slices 不要求彼此
物理相邻。该 order 让一个拥有单 KV head、连续遍历长 context 的 CTA 读取连续 page runs，同时保留完全
相同的 pool allocation、page ID 和 block-table 语义。

Allocator 可以返回任意 page IDs，consumer correctness 和 launch topology 不依赖 ID 连续性。以上两个
order 是 registered growing pools 的完整集合；Main/MTP Op 只接受 page-major，DFlash Full Op 只接受
head-major，不提供 arbitrary-stride 或 dual-layout execution path。

### 4.4 Logical position domain

Page table 使用 autoregressive cache ordinal，而不是 RoPE/MRoPE coordinate：

```text
logical block b covers cache positions [b*P, (b+1)*P)
```

Multimodal 三轴位置、`rope_delta` 和其他 position transformation 不改变 KV slot ownership。它们属于
Attention input metadata。

### 4.5 Fixed bytes per logical position

对一个 cache family 的 `layers=L`、KV heads `H`、head dimension `D`：

```text
BF16 bytes/token
    = 2(K,V) * L * H * D * 2

INT8-G64 bytes/token
    = 2(K,V) * L * H * D
    + 2(K,V) * L * H * (D/64) * sizeof(FP16 scale)

FP8-E4M3FN-row256 bytes/token
    = 2(K,V) * L * H * D
    + 2(K,V) * L * H * sizeof(FP16 scale)
```

一个 homogeneous pool 的 logical page-group payload 是其全部 grouped planes 的 bytes/token 之和乘以
该 pool 的 `P`。Startup physical pool bytes 则由 registered plane storage spans 之和再加 slab/head
alignment 得出；alignment 不改变 allocator 的 page-group ID 计数。不同 pools 的 page-group payload
无需相等。

FP8-row256是D256 causal Op可构造和消费的closed storage profile。Engine固定选择一种Main Text/MTP
storage profile，并以相同dtype和scale geometry实例化对应的homogeneous pools；DFlash Full仍使用
独立BF16 pool。

---

## 5. Physical page-group pool

### 5.1 One allocation unit within one pool

一个 Main Text page group 表示同一段 `P_main` positions 在全部 Main Text planes 中的 payload：

```text
Main Text page-group id = g

target layer 3 K plane   -> page[g]
target layer 3 V plane   -> page[g]
target layer 7 K plane   -> page[g]
target layer 7 V plane   -> page[g]
...
```

MTP 和 DFlash pools 各自拥有独立的 page-group ID namespace、free set 和 block tables。数值相同的
page ID 在不同 pool 中没有任何 ownership 关系。

### 5.2 Plane slabs

同一个 pool 内的 grouped planes 不要求组成一个连续 blob。每个 plane 拥有 Engine-lifetime-stable storage；
pool-local page-group ID `g` 按该 pool 的 closed physical order，在每个 plane 中选择对应的 logical page
group slices：

```text
Main/MTP plane:    [page 0: all heads] [page 1: all heads] ...
DFlash Full plane: [head 0: all pages] [head 1: all pages] ...

page-group g      -> plane 0 slices selected by g
                  -> plane 1 slices selected by g
                  -> ...
```

一个 pool 只有一个 page-group ID namespace、一个 free set 和一份 per-sequence block table。§7 中的
page-group payload 是 reservation/accounting unit 的总 bytes；不同 planes 之间不要求物理相邻。

Plane base 在 Engine lifetime 内保持不变。Consumer 使用 plane base、registered strides 和 `g` 计算地址；
不得把 block table 展开为 per-layer、per-head 或 per-plane device pointer arrays。Allocator 可以优先返回
连续 page IDs 改善 locality，但 kernel correctness 和 launch topology 不能依赖这种连续性。

### 5.3 Why K/V and layers remain grouped inside a pool

同一 pool 内的 K、V、code、scale 和 full-attention layers 具有相同 frontier 和 lifetime。将它们分别
分配不会产生可单独释放的有效资源，却会引入：

- 每层或每 plane 的独立 block table；
- 重复 reservation counters；
- layer-by-layer partial allocation failure；
- checkpoint ownership 的重复 bookkeeping。

因此 Main Text layers 应分组，但 Main Text 与 MTP/DFlash 不应分组。

### 5.4 Fragmentation and locality

每个 pool 内所有 page groups 等价，因此不产生 variable-size external fragmentation，也不需要
compaction。Allocation success 只依赖该 pool 的 free-page 总数，不依赖 largest contiguous hole。

例如 Main Text pool 有 128 Ki token-equivalents，`P=64`，共 2048 pages：

```text
A owns 48 Ki = 768 pages
B owns 48 Ki = 768 pages
free          = 512 pages
```

A release 后，768 个旧 pages 与尾部 512 个 pages 可能被 B 的物理区间隔开。新请求 C 需要
64 Ki，即 1024 pages：

```text
total free pages = 1280
needed pages     = 1024
```

连续 allocator 会因为最大 hole 只有 768 pages 而失败；paged allocator 可以把 C 的前 768 个 logical
pages 映射到 A 的旧 pages，再从尾部取得 256 pages，因此无需搬移 B。

Catalogued checkpoint持有的Device replicas不是free capacity。若checkpoint A阻塞active admission，
ResourceManager可以把“删除Device duplicate”“先下沉Host”或“删除logical checkpoint”放入完整choice；
只有Program对全部surviving references、copy-before-release stages和allocator状态验证并提交后，相应容量
才可用于destination。这是replica occupancy回收，不是compaction。

唯一的 payload slack 是每个 pool allocation 的最后一个未填满 page，最多 `P-1` positions。对于
`P=64`，8 个 active allocations 在一个 pool 中的最大尾页 slack 为：

```text
8 * 63 = 504 token-equivalents
504 / 131072 = 0.38% of a 128 Ki pool
```

Allocator 应优先给同一 sequence 分配连续 page IDs，使相邻 logical pages 在 plane slab 中尽可能连续；
但连续 run 不是 admission 条件。找不到连续 run 时使用任意 free pages，不能因此拒绝 request。

Device pool 的 free set 由按 physical ID 排序、互不相交且相邻即合并的 runs 表示；run metadata 在 pool
构造时预留到固定 capacity，release 路径不做 Host allocation。Materialize 先尝试从上一 logical page 的
后继位置取得完整连续 run，再选择第一个足够大的 run；没有单一 run 足够时按 physical ID 顺序跨 runs
消费。释放页与左右相邻 run 合并。操作量取决于涉及的 runs 和实际返回的 page handles，不再为一次
多页申请反复移动逐页 free-ID vector。

### 5.5 Logical pages and Device/Host replicas

Device page-group ID 只是当前 Device replica 在 typed slab 中的物理位置，不是 prefix identity。Program
为每个 typed pool维护 generation-checked logical page；一个 logical page包含 canonical content epoch、
committed columns、address-space references，以及 optional Device/Host replicas：

```text
LogicalKVPage
├── content epoch + committed coverage
├── address-space references and writer ownership
├── optional Device page-group lease
└── optional packed Host extent membership
```

Device replica使用 §4 定义的 consumer-native plane layout。Host replica存放同一 logical page的 packed
logical payload，不保存 Device page ID、holes 或 block-table row；Main 与 selected backend共享一块
startup-fixed `HostKVArena` byte capacity，但使用各自 layout descriptor分配 extent。D2H/H2D按完整
page coverage执行，并可合并连续 physical runs；copy完成且epoch/coverage核对后才发布新replica或释放
source。

Logical descriptor容量覆盖Device physical pages和Host arena按相应page stride能够容纳的pages；它不构成
第三份payload容量。零引用logical page只有在Device/Host replicas和全部transaction pins都解除后才能回收
descriptor。Replica、extent、reference和address-space membership均由Program维护；ResourceManager只采用
Program返回的逻辑终态。Active pages不搬迁。

---

## 6. Per-sequence allocation bundle

一个 active sequence 持有与 control-lane identity 无关的 typed address-space bundle：

```text
SequenceKVBundle
├── Main Text KVAddressSpace
└── selected Backend KVAddressSpace (MTP or DFlash Full, iff enabled)
```

每个 `KVAddressSpace` 独立包含：

```text
KVAddressSpace
├── ordered logical-block -> LogicalKVPage membership
├── committed frontier and optional checkpoint frontier
├── active-only Device growth reservation and execution-row lease
└── active/inactive ownership state
```

MTP 与 DFlash 互斥。Backend 关闭时 bundle 只有 Main；启用某个 backend 时，每个 admitted sequence
必须同时拥有 Main 与该 backend 的 address space，不能按 request 降级成缺少 backend state 的 bundle。
Published private/shared checkpoints 可以持有 inactive address spaces；只有 active address space 持有
execution row 和 future-growth entitlement。

### 6.1 Three extents per pool

每个 typed address space 必须区分：

| Extent | Ownership |
|---|---|
| reserved entitlement | active admission/resource transaction；保证未来可取得的 Device pages |
| mapped extent | KV Store；已经绑定 logical pages 的范围，replica 可以位于 Device 或 Host |
| valid frontier | target sequence state；当前可以被 consuming Op 读取的 prefix |

Mapped extent 可以暂时大于 valid frontier，用于 prefill chunk 或 speculative round 的 provisional writes。
Mapped 不等于 committed。

不同 pool allocations 可以具有不同 mapped extents 和 valid frontiers。Store 不计算它们之间的语义关系。

### 6.2 Active execution views

每个 enabled pool 在 startup 建立 `max_concurrency` 行固定地址的 Device block-table metadata，最大宽度为：

```text
ceil(max_pool_logical_extent[s] / P[s])
```

一个 pool 的 rows 组成固定地址矩阵：

```text
device_block_tables[s][execution_row][logical_block] -> pool-local Device page-group ID
```

每行 contiguous，元素为 I32 page ID。行宽由该 pool 的 maximum logical extent 决定；矩阵是 execution
metadata，不是 KV capacity，也不拥有 physical 或 logical pages。

`KVAddressSpace` 持有 authoritative ordered membership。Materialization 先确保 checkpoint 所需
logical pages 具有 Device replica，再为 address space lease 任意空闲 execution row 并一次性发布其
Device page IDs。Inactive private/shared address spaces 不占 table row，也不绑定原 control lane 或
StateImage slot；同一 address space 每次 activation 可以取得不同 row。KV Store 不从 row 推导 ownership。
未映射的 tail entries 不具有 consumer-visible 含义，可以在 debug/test 中使用 invalid sentinel，但
production kernel 不为每次读取增加 page-ID bounds branch。

Single-sequence prefill/cached Op 取得一个 table row 的 `PagedKVLayerView`。Batched decode Op 取得共享
plane bases、完整 table matrix 的 `PagedKVBatchLayerView`，再通过独立 `table_rows[B]` selector 为每个
compact batch row选择 allocation。Compact row、control lane 和 block-table row 因此不必相同，也不要求
KV payload gather。

### 6.3 Stable execution unit

在一次 prefill chunk 或 decode round 开始前，KV Store 为每个可能写入的 pool materialize 本 unit 的
最大 logical extent，并发布完整 mappings。GPU work in-flight 期间：

- 所有参与 pools 的 block tables 不增删、不改写；
- page payload 不搬迁；
- consuming operators 使用该 unit 冻结的 valid-frontier inputs；
- execution-row recycling、address-space membership mutation 和 replica transfer 禁止发生。

所有 mapping 更新、frontier commit 和 page release 都发生在 GPU boundary。

---

## 7. Fixed page size

Page size 是 homogeneous pool layout 的属性，不是整个 KV Store 的全局语义。当前全部 registered
growing pools 固定为：

```text
P_main          = 64
P_mtp           = 64
P_dflash_full   = 64
```

三个 pools 拥有独立的 storage、page ID namespace 和 capacity，但采用同一个 `P=64`。`P` 不是 request
option，也不在 serving 期间动态变化。

### 7.1 Design basis

一个 pool 的 `P` 由 storage 和 consumer locality 决定：

- consuming Attention 常用 key tile 覆盖多少连续 positions；
- prefill chunk 或 proposal block 如何跨 page；
- page-table lookup、address discontinuity 和 boundary handling 的频率；
- per-sequence block-table metadata；
- active/checkpoint address space 的尾页 slack；
- 单 plane page 是否足够大，能形成连续、合并良好的读写区间。

Prefix hit granularity不参与 `P` 的选择。合法 checkpoint frontier 可以位于 page 内任意 offset；只有完整
Linear Attention/backend state 是否存在才决定该 frontier 能否复用。

### 7.2 Main Text geometry

当前 target 的 main decode Attention 以 32/64-key 作为主要读取尺度，Text prefill chunk 是 128 的
倍数。固定 `P_main=64` 后：

- 一个 page 足以覆盖一个 aligned 64-key span；
- 128-token aligned prefill interval 只跨两个 pages；
- 128 Ki context 只需 2048 entries，native 256 Ki context 只需 4096 entries；
- 最大的 27B Main Text BF16 pool 每个 page-group 为 4 MiB，单 allocation 的尾页 slack 小于 4 MiB；
- block-table metadata 相对 KV payload 可忽略，同时 32-key tile 不跨 page、64-key tile 与 page 对齐。

### 7.3 Backend page sizes

MTP full Attention 的 context traversal 与相应 target full Attention 使用相同的 `P_mtp=64` 和
page-major order。DFlash proposal block 为 16 positions，`P_dflash_full=64` 正好容纳四个 proposal
blocks；其 Full pool 使用 §4.3 的 head-major page-run order。两者保留独立 pool ownership 和 capacity，
只共享 page size、allocation 与 block-table 语义。

### 7.4 Current pool payloads at `P=64`

下表只计算 KV payload，不含静态 slab alignment：

| Pool layout | KV storage | bytes/token | page-group payload | worst tail slack |
|---|---|---:|---:|---:|
| 27B Main Text | BF16 | 65536 | 4.0000 MiB | 3.9375 MiB |
| 27B Main Text | INT8-G64 | 33792 | 2.0625 MiB | 2.0303 MiB |
| 27B Main Text | FP8-row256 | 33024 | 2.0156 MiB | 1.9841 MiB |
| 35B-A3B Main Text | BF16 | 20480 | 1.2500 MiB | 1.2305 MiB |
| 35B-A3B Main Text | INT8-G64 | 10560 | 0.6445 MiB | 0.6345 MiB |
| 35B-A3B Main Text | FP8-row256 | 10320 | 0.6299 MiB | 0.6200 MiB |
| 27B MTP | BF16 | 4096 | 0.2500 MiB | 0.2461 MiB |
| 27B MTP | INT8-G64 | 2112 | 0.1289 MiB | 0.1269 MiB |
| 27B MTP | FP8-row256 | 2064 | 0.1260 MiB | 0.1240 MiB |
| 35B-A3B MTP | BF16 | 2048 | 0.1250 MiB | 0.1230 MiB |
| 35B-A3B MTP | INT8-G64 | 1056 | 0.0645 MiB | 0.0634 MiB |
| 35B-A3B MTP | FP8-row256 | 1032 | 0.0630 MiB | 0.0620 MiB |
| 35B-A3B DFlash Full | BF16 | 4096 | 0.2500 MiB | 0.2461 MiB |

27B Main Text BF16 的 4 MiB page group 分布在全部 full-attention planes。单层单个 K 或 V plane
每个 page ID 对应的 aggregate bytes 为 128 KiB；35B-A3B Main Text BF16 对应 64 KiB。Consumer 始终
看到一个 pool-specific typed plane view，而不是跨模型机制的 composite payload。

假设 page ID 为 32-bit，128 Ki context 的单个 block-table row 为 8 KiB。即使
`max_concurrency=8` 且每个 slot 同时持有 main 与一个 backend row，全部 metadata 也只有 128 KiB。

---

## 8. Reservation and admission

NInfer不抢占active request，因此每条sequence在Active publication前必须取得完整typed KV reservation：

```text
KVReservation(request) = {
    main_total_pages,
    optional_backend_total_pages,
}
```

它覆盖prompt尚未materialize的positions、最大合法output、backend provisional tail、pool-local rounding、
partial-tail COW和新suffix。Reservation是由真实pool capacity支持的concrete protection，不一定提前挑选
physical page IDs。

对每个Device pool `s`，任一稳定边界满足：

```text
used[s] =
    unique Device replicas[s]
  + reserved-but-unmapped active pages[s]
  + transaction destination reservations[s]

used[s] <= total Device page groups[s]
```

Host occupancy按unique packed extents的实际bytes计算。一个logical page无论被多少checkpoints或address
spaces引用，Device/Host replica都只占用一次；logical references本身不占payload bytes。

Admission不是一个per-request delta check，而是Program对完整choice的联合post-state检查。Choice包括source、
destination、全部victims和有序copy/release stages。每个stage都必须满足pool capacity；copy-before-release
不能用最终净值掩盖瞬时峰值。

Last-reference reclaim也只从完整post-state得到。删除一个alias通常不释放page；多个victims共同删除最后一批
references时，每个physical replica只释放一次。ResourceManager不得相加per-owner release credits。

Startup `KVCapacityProfile`仍须满足§3.3的active-set implication：当request在lane、fixed state和advertised
Main capacity上可admit时，selected backend reservation也必须可兑现。否则是startup sizing错误，不是运行期
等待条件。

Prefill/decode把active reservation在mapped与reserved-but-unmapped之间转换，不改变global available
capacity，也不推进resource revision。因此已admitted request不会因其他request或cache retention失去完成
capacity。Truncate/rollback释放的mapped page回到该sequence的reservation；即使physical ID回到allocator
free set，这部分容量也不能被另一plan消费。

Catalogued checkpoint没有future-growth reservation。Private endpoint可以Move；仍需保留的private source和
shared source使用Fork。Host-only pages必须先恢复Device replica。任何path都必须在Active publication前同时
兑现incoming suffix、output、tail COW、backend continuation和execution-row需求。

Selected speculative backend在Engine lifetime内固定。每条Active sequence一直保留相应address space与
reservation；某轮只运行ordinary target progress不构成capability降级。Finish要么发布Main、backend和其他
State一致的完整bundle，要么释放整个active bundle。

KV Store不自行选择victim。ResourceManager提供逻辑choice，Program seal并执行唯一ResourceTransaction；
ResourceManager最终只采用`ResourceResult`中的catalog/lane终态。

---

## 9. KV lifecycle

### 9.1 Admission

```text
WAITING
  -> select and pin optional checkpoint
  -> MATERIALIZING: reserve Device/Host destinations and active entitlements
  -> restore required replicas / Move or Fork typed address spaces
  -> install Device mappings in leased execution rows
  -> publish complete ActiveSequence
```

没有transfer或copy的fast path也遵循同一publication contract，只是在当前worker boundary同步完成。
这些步骤与完整StateImage、backend state和output capacity一起构成Engine/Program materialization
transaction；publish前的address space不能进入compact batch。

### 9.2 Prefill chunk

在 chunk launch 前，target 确定该 chunk 会写哪些 pools 以及各自最大 position。KV Store 分别
materialize 所需 pages。Chunk 成功后，target 才推进对应 pool frontiers。

Chunk 内部分 layers 已写、但 unit 未成功完成时，不发布新的 frontier；GPU/state-integrity failure 按
Engine-wide failure 处理。

### 9.3 Ordinary decode

Main Text decode 通常增加一个 materialized position。当前尾页尚有空间时不分配；跨越
`P_main` boundary 时，在 round launch 前从 main entitlement 中绑定下一个 page。

Backend pool 是否在同一 round 增长由 target decode mode 决定，不与 main page ID 绑定。

### 9.4 Truncate and rollback

每个 pool 根据自己的 valid/provisional frontier 独立计算需要保留的 page 数：

```text
needed_pages[s] = ceil(max(valid_frontier[s], provisional_frontier[s]) / P[s])
```

完全位于该范围之后的pool-local mappings可以解除。Active sequence仍持有原完整reservation，因此这些
page对应的容量回到该sequence的reserved-but-unmapped部分，而不是变成可供其他plan使用的global capacity。
最后一个部分使用的page保留；frontier之后的bytes可以是stale data，但不得进入execution view的有效范围。

### 9.5 Finish, retain, release

Request到达model completion后进入TerminalPending，并执行两种顶层结果之一：

```text
publish:
    Freeze a complete endpoint and/or selected typed checkpoint
    Program commits one ResourceResult with immutable checkpoint capabilities
    ResourceManager adopts the private/shared catalog update

release:
    every active KVAddressSpace releases its memberships
    all unused reservations are cancelled
```

启用 speculative backend 时，只有 Main、backend 和其他 continuation state 位于同一 exact frontier 的
完整 bundle 才能发布为 checkpoint。不完整状态释放整个 bundle，不能发布 target-only reusable
entry。

Finish choice、checkpoint slot、source/victim facts和每个stage peak在Program start前已经验证。若retention
不能完成，确定性fallback是释放active bundle；lane不能无限停在TerminalPending。

Device table rows和active StateImage slot随后可以复用；catalogued address spaces和replicas独立存在，不与
这些execution resources绑定。Network response lifetime不直接延长KV ownership，Responses endpoint只通过
opaque session key向下一请求提供lookup hint。Cancellation在Engine第一个观察到它的GPU boundary进入
TerminalPending；它不修改in-flight round mappings，也不从未完成active state发布新checkpoint。

---

## 10. Prefix reuse

### 10.1 Reusable checkpoints

Program只发布target已证明拥有完整continuation state的reusable checkpoints。当前Qwen3.6 catalog可以
包含private endpoint、typed `TurnClosure`/`ResponseReplay` rewrite checkpoint、被retention policy采纳的
sparse long anchor，以及由stable-prefix marker形成的shared checkpoint。

每个 checkpoint 都必须同时描述：

- Main Text address space 在该 frontier 所需的 logical pages 与 per-page committed coverage；
- selected backend 的 address space、frontier 和 continuation state（backend 关闭时无此项）；
- 完整 immutable StateImage，包括 Linear Attention、继续执行所需 hidden 和 Variant-owned backend fixed
  state；
- position/model-continuation metadata 与 checkpoint kind；
- 与上述状态一致的 prefix identity。

多个checkpoints可以引用同一KV address space的不同frontiers，也可以通过不同address spaces共享完整
immutable pages；每个checkpoint仍拥有独立StateImage identity。ResourceManager先用digest/index形成有界
候选集，Program再验证exact target prefix identity、typed frontier和replica requirements。SessionKey、marker、
hash、token match或page match都不能替代该验证。

Incoming prompt可以从所选checkpoint直接完成finalization或prefill suffix。没有完整checkpoint的位置即使
KV bytes仍存在也按cache miss处理；这不是page-size limitation，缺少的是相应Linear Attention/backend
continuation state。

### 10.2 Non-page-aligned frontier

例如 Main Text `P=64`、checkpoint frontier `F=1000`：

```text
mapped main pages = ceil(1000 / 64) = 16
last page valid offsets = [0, 40)
```

Private destructive Move 后可以从 logical position 1000 继续，无需把 frontier 向下取整到 960，也无需
复制或重算最后一页。若source必须保持immutable并产生第二个writer，则完整的15页保留共享引用，含40列
committed prefix的部分尾页复制到destination，后续写入只发生在private tail。Backend address space使用
自己的`P`和frontier独立描述，不要求与main尾页对齐。

### 10.3 Private Move, immutable Fork and page COW

Private continuation被选为source后从其他candidate set隐藏，但其checkpoint与replicas保持有效直到整个
ResourceTransaction commit或abort。若旧endpoint不需要保留，Program可在commit阶段把address spaces和
Device StateImage slot以Move语义转为active ownership。Move不复制KV payload，也不改变logical page
identity。

Shared checkpoint或仍需保留的private checkpoint使用Fork。对每个typed address space：

- checkpoint frontier之前的完整immutable pages增加address-space reference，不复制payload；
- 非page-aligned尾页为destination建立private Device page并复制exact committed columns；
- destination取得唯一writer和suffix growth reservation；source保持immutable且可被后续requests继续Fork；
- Main与selected backend、StateImage fork destination和execution rows作为同一transaction发布。

Logical page的reference count只表达physical sharing，不能单独判定可写性。Program同时核对writer cardinality、
surviving checkpoint的protected coverage、content epoch和source/transfer pins。正常private append可以在唯一
writer page的未保护suffix原地继续；destructive rewrite只有在不会覆盖任一surviving checkpoint所需内容时
才可原地切换coverage/epoch，否则建立private destination。任何时刻都不存在shared writable page或两个
writers。

---

## 11. Speculative decoding

MTP/DFlash runtime 不管理自己的显存 allocator。它们的 growing KV 仍从 common KV Store 中 target-defined
backend pool 获取，只是与 Main Text pool 物理分离。

一次 speculative round 的 KV 过程是：

```text
1. main/backend reservations 已在 admission 时完成
2. round boundary 分别 materialize 本轮各 pool 最大可能触及的位置
3. target writes provisional Main Text KV
4. backend writes provisional MTP 或 DFlash KV
5. acceptance 产生 target-owned committed frontiers
6. rejected tails 在各 pool 中分别变为 logically unreachable
7. 各 pool 解除不再需要的trailing mappings，容量回到同一active reservation
```

不同 requests 接受不同 proposal length，只改变各自 bundle frontiers，不形成新的 pool type。Rejected
bytes 可以留在部分尾页；后续 append 在它们重新变为 valid 前必须完整覆盖对应 code 和 scale planes。

KV Store 不理解 proposal、verify 或 acceptance，也不推导 Main Text、MTP 与 DFlash frontiers 之间的
关系。Target 把每个 pool 的最终 frontier 作为 transaction result 提交。

---

## 12. KV resources outside growing pools

并非所有 K/V-shaped state 都按 context 增长：

| Resource | Storage class | Reason |
|---|---|---|
| target full-attention KV | Main Text paged pool | 每个 target materialized token 增长 |
| MTP persistent KV | MTP paged pool | 独立 MTP frontier |
| DFlash full-context KV | DFlash Full paged pool | 独立 DFlash context frontier |
| DFlash local sliding-window KV | fixed per-sequence KV state | 容量固定为 window，不随总 context 增长 |
| DFlash boundary-local snapshot | fixed per-sequence KV state | 固定 checkpoint payload |
| Vision/temporary query K/V | shared execution workspace | 只在一个 operator/phase 内存活 |

### 12.1 Fixed cyclic physical contract

DFlash local sliding-window 和 boundary-local snapshot 使用同一个 closed cyclic layout，但拥有彼此独立的
storage。当前 registered target 对每个 local layer 固定为：

```text
capacity        = 4096
padded_capacity = 4096
K/V dtype       = BF16
K/V shape       = [128, 4096, 8] = [D, padded_capacity, Hkv]

nb[0] = sizeof(BF16)
nb[1] = D * nb[0]
nb[2] = padded_capacity * nb[1]

physical_slot(p) = p mod capacity
K[d,h,p] = k[d, physical_slot(p), h]
V[d,h,p] = v[d, physical_slot(p), h]
```

每个 active sequence 持有五层 local K/V 和五层等形状 boundary-local K/V。两组 plane bases 在其 fixed
state unit lifetime 内稳定；save/restore 复制完整五层 payload，不能通过 aliasing 让 live local state 与
checkpoint 共用可写 storage。

`CyclicKVCacheLayerView` 只携带 contiguous K/V tensors、capacity 和 geometry，不携带 block table、page
ID、allocator handle 或 mutable frontier。Caller 用绝对位置和 live interval 定义有效内容。Cyclic storage
不进入 growing pools，不参与 page reservation，不使用 `PagedKVLayerView`，也不因 paged migration 改变
modulo/window 语义。

这些 fixed cyclic payload 是完整 `StateImage` 的 Variant-owned组成部分，不进入 growing KV address space、
HostKVArena或paged reservation。Active sequence必须在Device `StateImage` slot中拥有它们；checkpoint按完整
StateImage单位保留Device或Host replica，不能只迁移其中的cyclic KV。统一Device state pool容量为
`max_concurrency + device_state_slots`，checkpoint不绑定原control lane或原slot，且cache occupancy不能破坏
`max_concurrency`路active state guarantee。

---

## 13. Correctness invariants

1. 每个Device page-group lease在所属pool内至多承载一个logical page replica；一个logical page可以有多个
   immutable references。
2. Pool grouping只包含共享frontier、lifetime和page geometry的planes。
3. Device/Host occupancy按unique replicas与concrete reservations计算，不按reference或owner重复计数。
4. Last-reference reclaim对完整victim set的联合post-state计算；同一replica至多释放一次。
5. Active reserved-but-unmapped capacity和transaction destination reservation不能被其他plan消费。
6. 每个resource transaction stage都满足pool capacity；final net delta不能替代peak检查。
7. Materialization对完整State/KV bundle原子发布；abort不留下半Active destination。
8. Source checkpoint在resource transaction commit或abort前保持有效。
9. 一个execution unit内所有pool mappings稳定。
10. Operator只能读取对应pool valid frontier内的positions。
11. Frontier publication前，完整K/V及必要code/scale planes已经写入。
12. Recycle不要求清零，但新content epoch不能从stale bytes或旧frontier推导valid state。
13. Page mapping使用cache ordinal，不使用RoPE/MRoPE coordinate。
14. Prefix hit必须由Program验证的完整continuation证明；token/page match本身不构成hit。
15. Active execution replicas在in-flight unit内不被transfer、eviction或其他request growth移动或释放。
16. Active合法最大per-pool growth始终由其完整reservation覆盖；truncate不把该保障借给其他request。
17. `KVCapacityProfile`按§3.2–§3.3从`max_context`、`kv_capacity`、`max_concurrency`和selected backend
    唯一导出；MTP headroom不扩大per-allocation logical capacity。
18. 一个address-space logical block在该pool全部grouped planes中使用同一Device page-group ID。
19. Execution table row只镜像Device mapping，不拥有logical page、payload、reservation或frontier；
    inactive checkpoint不占row。
20. Growing-cache Op只消费`PagedKVLayerView`或`PagedKVBatchLayerView`，不取得allocator、request identity
    或lifecycle authority。
21. K/V、code和scale的logical axes、closed physical order、strides与bounds符合§4.2。
22. DFlash local与boundary-local分别拥有§12.1的fixed cyclic payload；`p mod 4096`只选择各自physical slot。
23. Selected backend下的Vision prefill、prefix reuse与ordinary tail都保留完整backend continuation。
24. Shared full pages没有writer；Fork的partial tail使用private COW；任一logical page同时至多一个writer。
25. Host replica与logical page具有相同content epoch并覆盖必需committed columns；copy完成前不发布
    destination，也不释放唯一source。
26. Rejected speculative bytes不能扩展canonical coverage；回收的mapping容量仍属于同一active reservation。
27. Capability mismatch、reservation shortfall或checkpoint不完整是内部错误，不得伪装为cache miss。

---

## 14. Edge cases

| Case | Required behavior |
|---|---|
| context 在 page 中间结束 | 只推进 exact frontier，保留相应 pool 的部分尾页 |
| prefill chunk 跨多个 pages | launch 前分别 materialize 所需 pool pages，unit 内 mappings 冻结 |
| speculative tails 不同 | Main、MTP 或 DFlash 按各自 frontier 独立 trim |
| accepted length 为零 | target 决定各 pool progress；KV Store 不产生特殊状态 |
| checkpoint frontier 非 page-aligned | 精确保留 committed columns；private Move 原地继续，immutable Fork 复制 partial tail |
| prompt 正好结束在 private endpoint/rewrite checkpoint | 验证完整 checkpoint 后 Move 或 Fork，并从 checkpoint hidden 完成 finalization |
| 命中 checkpoint 且有 suffix | 恢复完整 StateImage 与 typed KV requirements，再 prefill suffix |
| 多个 requests 命中同一 shared prefix | 共享完整 immutable pages；每个 branch 建立 private state destination、partial tail 和 suffix |
| selected page 只有 Host replica | H2D 恢复并核对 epoch/coverage 后才安装 execution row |
| 只有更短 token prefix match，但该位置没有 checkpoint | cache miss |
| checkpoint occupancy阻塞admission | ResourceManager提出完整victim set；Program按联合post-state与stage peaks验证并提交 |
| 删除一个shared alias | 只删reference；非last reference时occupancy不变 |
| 多个victims共同删除last references | Program在联合post-state中只释放一次replica |
| checkpoint replica回收后，request set满足Main contract但backend reservation失败 | startup sizing或reservation invariant violation；不是正常等待 |
| request cancellation | Materializing abort并保留source；Active在in-flight unit后进入TerminalPending再释放 |
| active truncate/rollback | mapping可解除，但容量回到该sequence reservation，不增加global available capacity |
| admitted request materialize时无可兑现page | reservation invariant violation，作为Engine failure |
| 一个 request 使用 main pool 大部分容量 | 合法，只要其他 per-pool entitlements 仍满足 invariants |

---

## 15. Consumer boundary

Paged KV storage 同时服务 single-sequence prefill/cached routes 与 batched decode routes：

```text
shared typed page planes + fixed block-table matrix
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
one selected table row   table_rows[B] selectors
single-sequence Op       batched decode Op
```

Storage view 只解释 physical planes 与 page translation。Per-row context lengths、valid columns 和 compact
batch membership 由 consuming Op 的 typed inputs 表达；allocator、frontier commit 和 prefix ownership 均不
进入 view。

### 15.1 `PagedKVLayerView`

Single-sequence growing KV consumer 使用一个 non-owning typed view。Tensor shape 同时表达 logical
extent 和所属 pool 的 closed physical order。逻辑字段为：

```text
PagedKVLayerView
├── k_pages           Tensor (route-closed physical axes)
├── v_pages           Tensor (route-closed physical axes)
├── k_scale_pages     optional Tensor (same pool order)
├── v_scale_pages     optional Tensor (same pool order)
├── block_table       I32 Tensor [Nlogical]
├── head_dim          D
├── num_kv_heads      Hkv
├── dtype             BF16, I8, or FP8_E4M3FN
└── quant_group       0, 64, or 256
```

`P` 和 `Nphysical` 由 route 对 page tensors shape 的解释给出，logical capacity 为
`Nlogical*P`。当前所有注册
growing routes 只接受 `P=64`；kernel 将其作为 compile-time fact，而不是在 inner loop 中执行 runtime
division。Main/MTP causal routes 只接受 contiguous `[D,P,Hkv,Nphysical]` 及相同 order 的 scale
planes；DFlash Full append/context routes 只接受 contiguous `[D,P,Nphysical,Hkv]`。Op 不接受 permuted、
arbitrary-stride 或另一 pool profile 的 growing view。

同一 pool 的每个 layer view 引用不同 plane slabs，但引用同一个 active execution-row block table。View 不含：

- allocation、reservation 或 free-list handle；
- request identity、slot identity 或 batch row；
- mapped extent 或 valid/provisional frontier；
- prefix-cache ownership 或 backend mode；
- per-page K/V pointers。

这些事实分别属于 KV Store、Program target state 或 Engine round membership。Consumer wrapper 根据
Tensor shape 和 execution envelope 验证静态范围；caller 保证实际访问位置已 mapped 且位于该 Op 声明的
有效域。

### 15.2 `PagedKVBatchLayerView`

Batched growing KV consumer 使用同一组 typed plane tensors 和完整 block-table matrix：

```text
PagedKVBatchLayerView
├── k_pages / v_pages
├── optional k_scale_pages / v_scale_pages
├── block_tables       I32 Tensor [Nlogical,C]
├── head_dim / num_kv_heads
└── dtype / quant_group

Op inputs
└── table_rows         I32 Tensor [B]
```

`table_rows[b]` 为 compact row `b` 选择一个 active allocation 的 table row。View 不内嵌 `B`、membership、
context length 或 valid-column metadata；同一 exact-`B` invocation 可以选择任意互不冲突的 active rows。
Plane bases和table matrix base在Engine lifetime内稳定，row selector和table content是round data。

### 15.3 Device address contract

对当前 `P=64`，两类 route 共享 page translation：

```text
logical_block = position >> 6
page_offset   = position & 63
physical_page = block_table[logical_block]

Main/MTP element_address = plane_base
                         + d             * nb[0]
                         + page_offset   * nb[1]
                         + kv_head       * nb[2]
                         + physical_page * nb[3]

DFlash element_address  = plane_base
                         + d             * nb[0]
                         + page_offset   * nb[1]
                         + physical_page * nb[2]
                         + kv_head       * nb[3]
```

Batched consumer 先使用 `table_rows[b]` 选出 `block_tables[:,table_rows[b]]`，随后执行完全相同的
logical-block translation；batch axis 不改变 pool layout 或 page ID domain。

INT8 scale 使用 quant group `d/64` 作为第一维坐标；FP8-row256 scale 使用第一维坐标 0。两者都使用
scale Tensor 自己的 `nb`。一个 key tile 取得 `physical_page` 后，同一 pool 的 K、V、code 和 scale
都复用该 page-group ID。Exact strides 由 §4.2 对该 pool 唯一确定。

上述公式是 Op contract，不要求 production kernel 在每个 element 上执行四次通用整数乘法。Wrapper
验证 route-closed strides；CTA 在 page/head 粒度计算 base 并广播，inner loop 继续使用静态 row/vector
offsets。Production kernel 不检查 page ID 是否为 sentinel；mapping completeness 已由 execution-unit
materialization 保证。

### 15.4 Pool-to-consumer binding

Target/runtime 在调用 Op 前组合两类互不拥有的 view：

```text
pool.layer_planes(layer)
        +
selected allocation table row or table matrix
        =
PagedKVLayerView or PagedKVBatchLayerView
```

Main Text consumer 只能绑定 Main Text pool；MTP consumer 只能绑定 MTP pool；DFlash full-context
consumer 只能绑定 DFlash Full pool。Common Op 不接收 pool kind，也不根据 target、layer role 或 backend
string 选择 pool。

---

## 16. Op contracts

Growing-cache 参数只接受 paged view；不存在 parallel continuous overload，也没有名为
`paged_attention` 的第二套数学 Op。Paging 是 cache storage/addressing contract；Attention 的公式、
可见域和 per-sequence isolation 保持不变。Exact ABI 由对应 `include/ninfer/ops/` header 负责，本文只定义
storage/view boundary。

### 16.1 Affected entries

| Entry | Cache contract | State effect |
|---|---|---|
| `causal_softmax_attention` | writable `PagedKVBatchLayerView` + `table_rows[B]` | 为 `B` 条独立 sequences append valid K/V columns，并执行一次 ragged causal Attention |
| `causal_softmax_attention_cached` | read-only `PagedKVLayerView` | 只读已经 populated 的 paged cache |
| `kv_cache_append` | writable `PagedKVLayerView` | 写入全部 supplied rows，BF16 copy、INT8-G64 encode 或 FP8-row256 encode |
| paged `kv_cache_append_prefix` overload | writable `PagedKVBatchLayerView` + counts/table rows | 只写每行 device count 选择的 exact prefix |
| `context_softmax_attention` | read-only `PagedKVBatchLayerView` + table rows | batched 读取 DFlash Full pool；query K/V 仍是 transient Tensor |
| cyclic `kv_cache_append_prefix` overload | batched `CyclicKVCacheLayerView` + lane selectors | DFlash local fixed window，不属于 growing pool |
| `sliding_window_attention` | batched `CyclicKVCacheLayerView` + lane selectors | DFlash local fixed window，不属于 growing pool |

这些 entries 不构成 parallel storage-overload family。Workspace capacity query 由
现有execution envelope和token interval决定；paging不会成为一个workspace route，也不额外申请与context
长度成正比的transient buffer。

Growing `KVCacheLayerView` 和依赖连续 `[D,padded_context,Hkv]` 的实现不属于产品。固定 cyclic storage
拥有独立 layout/container，不复用 paged pool，也没有 continuous-growing backing abstraction。

### 16.2 Tensor domains

Main Text 和 MTP `causal_softmax_attention` 接受 request-major batched tensors：

```text
Q/Out          BF16 [D,Hq,W,B]
K/V            BF16 [D,Hkv,W,B]
positions      I32  [W,B]
valid_columns  optional I32 [B]
table_rows     I32  [B]
```

`B=1` 覆盖 single-sequence prompt/decode route；`B=2..8` 覆盖 batched decode/verify 的 `W=1..16`。
`valid_columns[b]` 只让每行的 prefix `[0,Vb)` 生效。`positions` 是 KV logical ordinal，不是
RoPE/MRoPE coordinate。对 row `b` 的 query position `p`，causal visible domain 仍是该 allocation 的
`[0,p]`；page boundary 不产生 mask boundary。

`causal_softmax_attention_cached` 和 `kv_cache_append` 保留 single-sequence `[D,H,T]` tensor
domain，用于 prompt 拆分路径，不构成第二套 growing-cache storage contract。

DFlash full-context entry 使用 `[D,H,W,B]` query block、per-row `context_lengths[B]`、
`valid_columns[B]` 和 `table_rows[B]`。每行只读取自己 allocation 的 context `[0,Lb)`，再加该行完整
transient query K/V segment；不同 rows 的 KV domain 永不互相可见。

### 16.3 Wrapper validation and caller promises

Wrapper 必须验证：

- K/V page tensors 的 logical geometry 一致、`P=64`，并精确匹配该 route 在 §4.2 的 closed
  contiguous order；
- physical page count、head geometry、dtype 和 optional scale planes 一致；
- single view 的 block table 是 contiguous I32 `[Nlogical]`；batch view 的 table matrix 是 contiguous
  I32 `[Nlogical,C]`，row selectors 是 contiguous I32 `[B]`；
- BF16 cache 不携带 scale planes；INT8-G64 的 scale leading extent/quant group 为 `D/64`/64，
  FP8-E4M3FN-row256 为 `1`/256，且两者 scale dtype 均为 FP16；
- causal `max_visible_keys <= Nlogical*P`；
- DFlash `max_context <= Nlogical*P`；
- input/output Tensor domain 与当前 entry 的已注册 geometry 一致。

Wrapper 不读取 device positions、context length、commit count 或 page IDs。Caller 保证：

- execution envelope 覆盖本次 exact device values；
- page tensors 由 pool binding 在对应 registered plane storage bounds 内建立；
- 本次最大可能访问的每个 logical block 已在 launch 前 materialize；
- read domain 不超过 target-owned valid frontier；
- writable rows 的全部 K/V code/scale bytes会在 frontier publish 前完成；
- Op in-flight 期间所有 selected block-table rows 不变化。

对 device-count prefix append，materialization 按 envelope 的最大可能 per-row write extent完成，而不是按
host 不可见的 exact counts 猜测。每行 rejected tail 不写入，Op 本身也不推进 frontier。

---

## 17. Kernel addressing design

### 17.1 Addressing primitive

Kernel 只接收 layer plane bases、一个 block-table row pointer 和现有 execution inputs。Page translation
应是 route-local inline device primitive；当前生产 route 对 `P=64`、其 closed physical order、head
geometry 和 codec 编译期专用。Page/head bases 在 CTA 或 tile 粒度计算并共享，不能把 stride formula
机械地留在每次 scalar/vector access 上。

不得实现一个在 inner loop 中解释 arbitrary page size、arbitrary layout 或 cache-kind variant 的通用
runtime accessor。可以共享 shift/mask、page-base calculation 和 INT8 codec 等窄 primitive，但 BF16、
INT8、causal prefill、causal small-T 和 DFlash context 保留各自可独立优化的 kernel body。

### 17.2 Causal small-T BF16

现有 small-T partial/reduce 分解保持不变：

```text
partial grid: (KV head, split)
reduce grid:  (Q head, D chunk, query token)
```

不增加 batch dimension。寻址约束为：

1. growing cache 寻址不包含基于 `padded_context` 的 flat cache index；
2. 对每个 logical key tile 取得一次 page ID，计算当前 KV head 的 page-local K/V bases；
3. page ID 在 CTA 内共享，不能由每个 vector lane 重复查询；
4. current K/V rows 先写入对应 page，并通过与历史 rows 相同的 cache representation 参与
   Attention；
5. 不让其他 split依赖本 split的 cache write；
6. partial accumulator、softmax statistics 和 reducer layout 不因 paging 改变。

当前 BF16 key tile 为32 keys，`P=64`恰好包含两个 tiles。Route planner必须避免保留会产生任意
`split_start` 的 `ceil(window/active_splits)` 分割。正常路径按完整32-key tiles分配；极短 context 若为
增加并行度拆分一个 tile，sub-page span 的 start/end 也必须落在 page-local有效范围内。任何 vectorized
global load 都不得跨两个 physical pages。

### 17.3 Causal small-T INT8-G64

INT8 code和FP16 scale使用同一 page ID。一个 key tile 的 translation 顺序为：

```text
logical tile -> physical page ID
             -> K/V code page bases
             -> K/V scale page bases
```

32/64-key implementation blocks都整除 `P=64`。Route可以继续针对 T 和 context选择不同 warp数、
key block和dynamic shared-memory profile，但 split span必须以 page-compatible tile units表达。现有24、
480等经验跨度不是语义；若它们造成跨页tile，必须在 paged route tuning 中重新选择，而不能为
保留旧 heuristic引入逐元素boundary branch。

Append/encode阶段应以至少 `(token, kv_head)` 为page-translation共享单位。四个64-d quant groups不能
各自从global memory重复加载同一个 block-table entry。V codec公式不变；K writer先形成一个完整
D256 row的paired causal profile私有固定rotation，再使用现有G64 codec。实现可以由一个warp拥有
整行，也可以在同一CTA内以FP32 fragment分解变换并让四个group warp共享page translation；不能在
中间增加低精度cast或跨CTA handoff。raw K code/scale bits不是独立qualification结果。

### 17.4 Causal FP8-E4M3FN-row256

FP8 code与单-row FP16 scale共享同一个page ID。Q/K在固定Hadamard坐标中rowwise量化，K code直接作为
native E4M3FN Tensor Core operand进入QK，accumulator保持FP32；不得建立global dequantized K。V的全部
finite E4M3FN code先在片上exact转换为FP16，再与represented FP16 scale做一次FP16 multiplication；P从
FP32 Softmax结果转换为FP16，PV使用FP16 operands与FP32 accumulator。Online Softmax、split
statistics、partial numerator、merge和normalize均保持FP32，只在公共output store转换为BF16。

Small-T CTA以 `(sequence,kv_head,split)` 组织并让一个KV head对应的全部Q heads复用一次K/V stream；
32/64-key blocks都不跨 `P=64` page。Prompt与small-T分别拥有独立FP8 kernel body，但必须消费同一个
persistent row codec；不得引入global K/V transcode、第二次完整cache pass或FP16/BF16 cache副本。

### 17.5 Causal prefill and cached prompt route

Prefill 保持现有两阶段效果：

```text
fill current K/V rows into cache
    -> Attention reads complete populated history
```

Fill kernel按每个 logical token选择 page。一个 execution unit可以从 page中间开始并跨任意数量pages；
不要求 prefill chunk、checkpoint frontier 或 positions[0] page-aligned。

Attention key loop继续从logical position 0遍历到当前query可见上界。当前 BF16、INT8与FP8 prefill的
key tile均为64，因而一个完整key tile正好对应一个physical page：

```text
logical block b -> block_table[b] -> one K page group and one V page group
```

每个 tile 的 page ID 在 K stage 和 V stage 之间复用。现有 Q staging、online softmax、MMA和output
epilogue不因 paging 改变。最后一个partial tile继续通过logical visibility mask清零越界rows，不能读取
未初始化page tail作为有效key。

Cached prompt route使用相同page-aware key traversal，但不执行fill。

### 17.6 Standalone append

Full append和device-count prefix append都直接写最终physical pages，不建立连续staging cache。

- BF16 copy：一个 `(token, kv_head)` work unit查询一次page并协作复制完整D；
- INT8 encode：同一work unit复用page ID写code与scale；
- FP8 encode：同一work unit形成完整D256 K rotation或V row、反馈represented FP16 row scale，并复用
  page ID写code与scale；
- sequential positions允许一个CTA处理多个tokens，但跨page时必须重新取得page ID；
- device-count prefix route先读取一次合法count，再只调度或mask `[0,count)`；
- paged与cyclic append编译为不同physical-address routes，不在每个store上保留runtime cache-kind branch。

### 17.7 DFlash full-context Attention

DFlash transient query K/V保持连续；只有persistent full context改为paged。现有32/64-key blocks均整除
`P=64`，每个context tile取得一次page ID。Split partial和reduce workspace语义保持不变。

`context_length=0` 的direct route不访问block table。DFlash local和boundary-local cache继续使用独立
cyclic storage，不能因full-context migration改变其modulo/window语义。

### 17.8 CUDA Graph behavior

Plane slab bases和每个execution row的block-table row pointer在Engine lifetime内稳定。跨replay变化的是table
content、positions、context length和commit count的device values，而不是kernel pointer arguments。

在需要新page的execution unit之前：

```text
reserve entitlement already exists
    -> batch bind all newly required physical page IDs
    -> update host allocation mapping
    -> publish the contiguous device table slice with one H2D copy
    -> launch/replay consumer on the same ordered stream
```

同一 boundary 新增多个 pages、activation 重新绑定完整 membership、prefix fork 和 active snapshot 都遵循
上述批量 publication。单页 decode boundary 是这个 API 的 `count=1` 特例；不得把长 prefill 恢复为逐页
4-byte H2D。任何 batch publication 失败都在 address frontier/page-count 发布前退还本批 descriptors 和
physical leases，保留旧 execution mapping authority。

Graph capture不以page IDs、physical contiguity或checkpoint owner为key。Mapping update必须先于consumer，
in-flight期间禁止改写同一row；Op和kernel内部不调用allocator，也不等待host page fault。

Startup graph construction为每个temporary row保留一个private page，并可将同一page ID重复写入该row的
全部logical entries，以覆盖任意reachable context envelope。只有eager code warm和每个executable的一次
smoke会真实访问这些pages；准备时只清零当前exact `B`对应的private pages。Definition capture和
update/upload validation不执行consumer，不能因此扫描或清零整个physical pool。

### 17.9 Prohibited implementations

以下实现即使功能正确也不接受：

- 每次Attention前把paged KV gather成request-contiguous buffer；
- 为每个layer、K/V或scale维护独立logical page table；
- block table存放per-plane 64-bit pointers；
- 每个scalar/vector lane独立执行同一个page-table lookup；
- 一个physical page启动一次独立CUDA kernel；
- 假设相邻logical pages拥有相邻physical IDs；
- page boundary回退到保留的continuous-cache kernel；
- 把 exact-`B` invocation 永久 padding 到 `max_concurrency` 并读取 inactive table rows。

---

## 18. Correctness qualification

Paging不得改变Attention数学结果、cache codec或状态效果。现有独立Attention oracle继续从逻辑
positions和represented cache values计算结果，不复制production page traversal。

### 18.1 Required mapping patterns

每个受影响Op至少使用以下三种physical mapping执行同一logical case：

| Pattern | Purpose |
|---|---|
| identity | `logical block b -> physical page b`，便于oracle和boundary诊断 |
| contiguous offset | page IDs连续但不从0开始，排除隐含base假设 |
| fragmented permutation | logical相邻pages映射到非相邻IDs，证明无连续性依赖 |

三者必须产生相同逻辑结果和相同目标rows的cache bits。

### 18.2 Required cases

- positions/context覆盖 `0, 1, 31, 32, 63, 64, 65, 127, 128` 等tile/page边界；
- prefill从page中间开始，跨一个和多个pages；
- private checkpoint frontier 位于 page 中间并由唯一 writer 在未保护 tail继续append；
- immutable checkpoint fork共享完整pages、复制non-aligned tail，并从exact frontier继续；
- Host-only required page恢复到新Device page ID后保持exact represented values；
- BF16 append bit-exact；
- INT8 V code和FP16 scale保持既有exact codec；K representation通过append后由causal Attention
  消费并直接对独立Attention oracle；
- FP8 standalone与fused append的row256 code/scale bit-identical；FP8 append-and-attend与cached-only
  直接对由represented logical K/V计算的同一个独立Attention oracle；
- cached-only route不修改任意cache plane；
- prefix append的count为0、page边界前后和full count；
- rejected/provisional stale bytes不进入valid read domain；
- DFlash full-context长度为0、page边界和最大registered范围；
- cyclic DFlash local behavior保持其固定窗口语义；
- graph replay之间更新同一个stable table row的content，下一replay读取新mapping；
- 同一 pool 中多个 allocations 绑定不同 table rows，在一次 batched invocation 中使用不同 context
  lengths、valid columns 和 row selectors 后仍互不串扰；
- page release/recycle后，新owner不能观察旧owner的stale bytes为valid state。

---

## 19. Performance qualification

### 19.1 Cost model

`P=64`时，一个page-table entry为4 bytes，而每个KV head读取一个完整page的K+V payload约为：

| Format | K+V payload per KV head per page |
|---|---:|
| D256 BF16 | 64 KiB |
| D256 INT8-G64 incl. scales | 33 KiB |
| D256 FP8-E4M3FN-row256 incl. scales | 32.25 KiB |
| D128 BF16 | 32 KiB |

因此page-table payload本身不是主要带宽成本。需要实测防止的是重复lookup、跨页vector load、TLB/cache
locality下降、split imbalance和原有`cp.async`/MMA overlap被破坏。性能 qualification 只评估和调整
route-local tile、split、lookup sharing 和 staging；storage layout 不是 route tuning variable。

### 19.2 Comparison boundary

Paged KV 的初始准入使用删除前单独保存的 continuous production binary 作为 reference。Reference 与
candidate 必须使用同一 RTX 5090、toolchain、build profile、artifact geometry、数学语义、workload、
dtype、context、cache condition 和 measurement procedure。Reference binary 和结果只是一次性比较证据，
不属于产品；production tree 和 benchmark 都没有 continuous runtime route、control kernel 或
compatibility switch。后续 paged route 变更以当前已准入的 paged production route 为 reference。

Candidate 可以按 §17 调整 tile、split、lookup sharing 和 staging。需要归因 page translation 成本时，
可以增加 topology 与 split policy 匹配的受控对照，但该对照不替代 production gate。短 kernel 使用
reference/candidate 成对交替测量，结合重复分布和绝对 latency；长 route 使用吞吐中位数。只有结果无法
解释或超过 gate 时才使用 profiler，不建立长期 benchmark research framework。

### 19.3 Representative coverage

性能证据覆盖实际 route，而不是所有参数的笛卡尔积：

- 两个 exact target 的 BF16/INT8-G64 causal decode `B=1,2,4,8`、`T=1` 和代表性 verify small-T；
- causal prefill/cached-only 的一个普通 chunk 和一个长 chunk，并包含 non-aligned prefix base；
- standalone append 的 small-T 与一个 prefill-sized chunk；
- DFlash full-context 的 `T=1` 和完整 proposal block，覆盖普通与长 context；
- identity mapping 与一个固定 fragmented permutation；
- production CUDA Graph replay，以及 ordinary、MTP、DFlash 的 real-artifact request paths。

§18 已覆盖 page/tile 边界的 correctness，不为每个 boundary 重复建立性能 case。只有现有 production
route 在某个 context seam 改变 kernel topology 时，才在 seam 两侧补测。

### 19.4 Admission gates

对 latency 指标定义：

```text
latency_regression = candidate_median / reference_median - 1
```

对 throughput 指标定义：

```text
throughput_regression = 1 - candidate_median / reference_median
```

下列 gate 对每个列出的 target、dtype、route 和 workload case 独立成立，不能跨 cases 求平均，也不能用
另一条 route 的加速抵消当前 route 的回退。阈值是最大可接受回退，不是预期回退；目标仍是与 reference
持平或更快。若 measurement noise 大到无法判断是否越过阈值，该结果无效，必须改善测量条件或增加
有效重复，不能按通过处理。

Storage/addressing route 只有同时满足以下条件才准入：

1. 两个 targets 的代表性 end-to-end decode throughput中位数相对同一 batch profile baseline 回退不超过
   1%；
2. context不小于2K的causal decode/cached Attention和DFlash full-context kernel中位数回退不超过3%；
3. prefill Attention吞吐中位数回退不超过3%；
4. standalone append不出现超过5%的可重复带宽回退；
5. CUDA Graph replay不产生额外host synchronization、per-page launch或mapping-dependent recapture；
6. §18 correctness matrix全部通过。

短context的单个极小kernel容易受timer noise影响，不能用一个相对百分比点否决或掩盖结果；它同时受
repeat distribution、绝对latency和end-to-end gate约束。任何稳定超过上述门槛的回退都必须先定位到
lookup、TLB、split、staging或reduction，再调整对应 route；不能通过 continuous fallback 绕过准入。

### 19.5 Current qualification record

初始 paged storage migration 已在 RTX 5090、CUDA 13.1、`sm_120a` 上完成准入。下列冻结
contiguous-KV reference 只记录当时的 `B=1` paging migration，不是当前并发吞吐数据：

- causal matrix 覆盖两个 registered geometries、BF16/INT8-G64、append/cached、`T=1/4/16`、
  `L=2K/8K`、identity/fragmented mapping 和 CUDA Graph cold-cache execution；对临界 case 使用
  reference/candidate 交替重复，未出现可重复的 3% 以上回退；
- causal prefill、standalone append 和 DFlash full-context 分别通过 3%、5% 和 3% gate；
- identity、contiguous-offset、fragmented、page-boundary、multi-address-space、checkpoint/recycle 和 graph
  replay correctness 均通过；
- 两个真实 artifacts 的 ordinary、MTP、DFlash 和 prefix-reuse execution 均通过。

代表性 product-route 结果如下；throughput 越高越好，round latency 越低越好：

| Route | Frozen reference | Paged result | Change |
|---|---:|---:|---:|
| 27B ordinary decode | 80.698 tok/s | 84.443 tok/s | +4.64% |
| 35B-A3B ordinary decode | 333.403 tok/s | 341.756 tok/s | +2.51% |
| 27B MTP round | 26.845 ms | 26.065 ms | -2.91% |
| 35B-A3B DFlash steady round | 10.312 ms | 10.005 ms | -2.97% |

---

## 20. Fixed decisions and tuning freedom

以下是实现必须遵守的 architecture contract：

- `EngineOptions.max_context=S` 是 per-sequence logical ceiling，`EngineOptions.kv_capacity` 是
  `Explicit(K_main)` 或 `Automatic(R)`；令 `L=ceil(S/64)`、`M_min=max(L,max_concurrency)`、
  `M_max=max_concurrency*L`，Explicit 取 `M=ceil(K_main/64)`，Automatic 根据完整 target physical
  reservation curve 与权重加载后的空闲显存扣除 headroom `R` 后，直接求得区间内最大的 `M`；
  CLI/server 的 `R` 为 1 GiB；Main 与 DFlash Full 的
  per-allocation logical capacity 均为 `L`、physical capacity 均为 `M` pages，MTP 的 logical capacity
  为 `L`、physical capacity 为 `M + max_concurrency*ceil((K_draft-1)/64)` pages，其中 `K_draft` 是
  speculative draft window；Explicit 不满足 `K_main>=S`、任一 policy 的 resolved
  `M` 不在 `[M_min,M_max]`，或 minimum/runtime reservation 无法容纳时即拒绝；
- growing KV 使用 homogeneous pools、generation-checked logical pages、pool-local I32 Device page-group
  IDs 和 address-space-owned ordered membership；
- logical page 可以具有 Device/Host replicas并被多个immutable address spaces引用；Device replica使用
  consumer-native plane layout，Host replica使用packed logical layout；shared Fork只共享完整pages并为
  non-aligned tail建立private copy，shared writable page不成立；
- 全部 registered growing pools 的 page size 为 `P=64`；
- Main Text/MTP 的 K/V 与 code planes 固定为 contiguous page-major `[D,P,Hkv,Nphysical]`，INT8-G64
  scale planes 固定为 `[D/64,P,Hkv,Nphysical]`，FP8-E4M3FN-row256 scale planes固定为
  `[1,P,Hkv,Nphysical]`；DFlash Full K/V 固定为 contiguous head-major page-run
  `[D,P,Nphysical,Hkv]`；
- exact strides 由 §4.2 对每个 homogeneous pool 唯一确定；request 和 runtime mode 不选择 order；
- K/V/code/scale 及同 pool layers 共享一个 page-group ID 和一份 per-sequence block table；
- device tables 使用 startup-fixed `max_concurrency` rows；single-sequence Ops 消费一行
  `PagedKVLayerView`，batched Ops 消费完整 `PagedKVBatchLayerView` 和独立 row selectors；
- no gather-to-contiguous、no per-plane/per-head pointer tables、no continuous fallback、no
  arbitrary-layout runtime dispatch；
- DFlash local 与 boundary-local 使用 §12.1 的 fixed contiguous cyclic layout，不进入 growing pools；
- ownership、replica location、reservation、frontier、request identity 和 slot identity 不进入 Op view。

以下内容属于 route-specific implementation profile，可以在不改变 storage contract 时测量和调整：

- BF16/INT8/FP8 route 的 warps per CTA、split count、keys per split 和 32/64-key route interval；
- block-table entry 在 register/shared memory 中的广播方式；
- append kernel 每个 CTA 处理的 tokens/heads；
- allocator 优先选择连续 free IDs 的 heuristic；
- route-private staging、workspace representation 和 reduction decomposition。

改变 `P`、pool grouping、page ID model 或 closed pool orders 都是 architecture contract 变更，不属于
kernel tuning。任何实现都不能在 kernel 内暗中建立第三套 storage contract。
