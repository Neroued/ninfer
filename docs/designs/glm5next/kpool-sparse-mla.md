# Kpool Sparse MLA 实现设计

性质：待落地的完整链路接口与状态交接设计。决策、实现状态和任务范围见
[设计索引](README.md)。模型数学由[模型参考](../../maintainer/glm5next-model.md#4-sparse-mla-与-kpool4-indexer)
维护，精度、codec 和表示边界由[数值政策](../../maintainer/glm5next-quantization.md#3-mlaindexer-cache-与计算政策)
维护；本文定义具体权重合并、page、views、写域、选择与接受前缀接口。

所有权沿用[Engine 架构](../../maintainer/engine-architecture.md)和
[Paged KV context store](../../maintainer/paged-kv-cache.md)，资格遵循
[Op development](../../maintainer/op-development.md)。本设计不扩展[基础清单](op-checklist.md)的实施范围。

## 1. 使用范围与几何

| 项目 | 合同 |
|---|---|
| 首个部署 | 双 Spark，MLA/Indexer 按请求 DP2；owner 计算完整 heads |
| 使用位置 | Main 11 个 MLA 层，MTP 1 个 MLA 层；各层独立 cache/selection |
| Input / Q bottleneck | hidden=4096，Q bottleneck=1536 |
| MLA | 64 heads，Q/K/V head dim=256，latent dim=512，NoPE |
| Indexer | 32 query heads，dim=128，kpool=4，Top512 完整 pools |
| 有效上下文 | 至多 1048576 个原 tokens |
| 本地批次 | B=1..8；T 为任意正的物理宽度，受可用 views/storage 约束 |

本 family 使用数值文档已定的 BF16／E4M3FN-G128-FP32-scale latent profiles、固定
BF16 Indexer 及相同计算精度。Decode、verify 与 prefill 使用同一语义；kernel 数量、
tile、流水和合法路线分派属于实现选择，不增加精度模式。

## 2. 权重、投影与执行链

矩阵使用逻辑 `W[N,K]` 记法；native Tensor 第零维连续，`M=T*B`、
`column=t+T*b`。x 是外部 attention input norm 之后的 BF16 `[4096,T,B]`；
本 family 不重复执行 block input norm，也不接收 mHC 的四流 residual。

### 2.1 固定权重布局

Prepare 接收一个 BF16 contiguous `W_A[2336,4096]`，行序为：

| 行区间 | 源角色 | 输出用途 |
|---|---|---|
| [0,1536) | q_a_proj | Q bottleneck，1536 维 RMSNorm |
| [1536,2048) | kv_a_proj_with_mqa | latent，512 维 RMSNorm |
| [2048,2176) | indexer.wk | 128 维 affine LayerNorm |
| [2176,2304) | index_kpool_compress_gate | raw pool gate |
| [2304,2336) | indexer.weights_proj | FP32 raw head coefficients |

最后 32 行直接保留 FP32 出口，不能随其他投影转换成 BF16。它们不包含 sigmoid、
softmax、query scale 或 `1/sqrt(32)`。

Q 投影使用 BF16 contiguous `W_B[20480,1536]`：前 16384 行为 MLA Q_b，后 4096 行
为 Indexer Q。普通调用通过 Linear/split 直接写两个独立连续输出；MTP 复用 selection
时只使用前 16384 行的只读 view，无运行时 packing 或重复驻留权重。

KV_b 保留因子表示，分别按 head 保存 U_K/U_V：每个 head 逻辑 `[256,512]`，
native BF16 `[512,256,64]`，latent 轴连续。源 KV_b 每个 head 的行序为
`[K256,V256]`，converter 按此映射保留 words。U_K/U_V 不与 Q_b/O 相乘生成更大权重。
最终 `W_O[4096,16384]` 为独立 BF16 Linear。

Norm gain/bias 和 pool APE 均保留源 BF16 words，使用直接 gamma/affine，不采用
Qwen4Exp 的 `1+weight`。APE native 布局为 `[128,4]`，计算时升 FP32。

### 2.2 链路

```text
x + W_A
    → prepare_record / prepare_update
        → q_r BF16 [1536,T,B]
        → head_coeff FP32 [32,T,B]
        → latent cache、完整 pooled keys、records 或目标 tail
    → W_B Linear/split(q_r)
        → query BF16 [256,64,T,B]
        → index_query BF16 [128,32,T,B]
    → kpool_indexer(index_query, head_coeff, index cache)
        → pool selection + selection_end
    → kpool_sparse_mla(query, U_K/U_V, latent cache, selection)
        → heads BF16 [256,64,T,B]
    → W_O Linear
        → block output BF16 [4096,T,B]
```

Prepare 是包含 projection、norm 与 cache/state 效果的完整融合 Op。它可复用 Linear
的私有计算机制，但不要求先调用公共 Linear 并物化 BF16 的 W_A 输出。
W_B 与 W_O 的纯投影继续属于 Linear family。

## 3. 64-token page 与四 token 连续加载

P 固定为 64 个原 tokens，每页 16 个 pools。单层 page-major planes：

| Profile / plane | Native Tensor | tokens_per_row |
|---|---|---:|
| BF16 latent | BF16 [512,64,1,Npage] | 1 |
| FP8 latent codes | E4M3FN [512,64,1,Npage] | 1 |
| FP8 latent scales | FP32 [4,64,1,Npage] | 1 |
| Indexer pooled keys，两个 profile 均使用 | BF16 [128,16,1,Npage] | 4 |

只实例化所选 latent profile 的 planes。所有 plane 起点 256 B 对齐，页内各行紧密
排列：latent row stride 分别为 1024/512 B，FP8 scale row stride 为 16 B，
index row stride 为 256 B。Page stride 覆盖完整页 payload、保持 256 B 对齐，可保留
显式外层页间距；不在相邻 token 的 latent 数据之间插入 scale 或 metadata。

同一个 pool 的全部层/planes 共用 page-group ID、原 token frontier 和 block table。
`block_tables` 为 Device I32 `[logical_pages,table_rows]`，页项连续、行跨度显式。
Physical IDs 可任意排列，不要求多个 logical pages 在物理上连续。

选中 logical pool g：

```text
logical_page  = g / 16
pool_in_page  = g % 16
physical_page = block_tables[logical_page, table_rows[b]]

index row     = pool_in_page
latent rows   = 4*pool_in_page + {0,1,2,3}
```

一次页表寻址定位四条连续 latent，不跨 KV page。四 token payload 为 BF16 4096 B，
或 FP8 codes 2048 B＋另一 plane 的 64 B scales；data 起点自然保持 256 B 对齐。
完整 pool 可整组搬运，未完成 tail 必须按有效 token 数 mask，不读无效第四条。
Selection 不展开成 2048 个零散 token IDs，也不要求先构造全局 gather buffer。

Main 的 11 层与 MTP 的 1 层属于不同 growing pools，各有自己的 page namespace、
block table 与 frontier；同一 pool 内的 latent/scales/index 一起 reserve、保护、
截短与释放。现有 `tokens_per_row=1/4` geometry 可承载这些 planes，但消费者使用
专门的 MLA/Indexer typed views，不伪装成普通 K/V pair。

## 4. 公共 views、metadata 与身份

### 4.1 Batch 与 cache views

```cpp
enum class KpoolMlaStorage {
    BFloat16,
    Fp8E4m3fnGroup128F32,
};

struct KpoolMlaBatchView {
    Tensor positions;      // Device I32 [T,B], 原 token ordinal。
    Tensor valid_columns;  // Device I32 [B], 0 <= n[b] <= T。
    Tensor table_rows;     // Device I32 [B]。
};

struct KpoolMlaLatentCacheView {
    Tensor data;           // 第 3 节的 latent data plane。
    Tensor scales;         // BF16 profile 为空；FP8 profile 为 FP32 [4,64,1,Npage]。
    Tensor block_tables;
    KpoolMlaStorage storage;
};

struct KpoolIndexCacheView {
    Tensor keys;           // BF16 [128,16,1,Npage]。
    Tensor block_tables;
};

struct KpoolMlaCacheView {
    KpoolMlaLatentCacheView latent;
    KpoolIndexCacheView index;
};

struct KpoolExecutionEnvelope {
    uint32_t max_visible_tokens; // 0..1048576，Host 资源上界。
};
```

组合 cache 的两份 tables 必须为相同 address/shape/strides，Npage 相同。MLA 与
Indexer 的单独只读入口只接收实际需要的 view，未使用的 plane 不被读取。

活跃行的 positions 连续且非负；padding 不消耗 ordinal。每个有效 `p+1<=1048576`，
并在对应 tables 与资源 envelope 覆盖内。Envelope=0 只允许所有行无效，不是
数学 mask 或第二份 sequence frontier。B=0 时调用者不提交该 Op。

State slot、table row、batch row、record row、selection row 是不同身份，显式映射。
Op 不接收 request ID、rank、world size 或 communicator；调用者保证这些视图属于
同一请求、同一层及正确的历史。

### 4.2 Tail 与 records

```cpp
struct KpoolTailStateView {
    Tensor raw_keys;        // BF16 [128,3,Slots]，本层。
    Tensor raw_gates;       // BF16 [128,3,Slots]，不含 APE。
    Tensor slots;          // Device I32 [B]。
};

struct KpoolRawRecordWriteView {
    Tensor raw_keys;        // BF16 [128,W,B]，本层。
    Tensor raw_gates;       // BF16 [128,W,B]。
    Tensor offsets;        // Device I32 [B]。
};

struct KpoolTailAllLayersView {
    Tensor raw_keys;        // BF16 [128,3,L,Slots]。
    Tensor raw_gates;       // BF16 [128,3,L,Slots]。
    Tensor slots;          // Device I32 [B]，各层共同选择 state image。
};

struct KpoolRawRecordReadView {
    Tensor raw_keys;        // BF16 [128,W,L*R]。
    Tensor raw_gates;       // BF16 [128,W,L*R]。
    Tensor base_positions;  // Device I32 [R]，F0，record column 0 的位置。
    Tensor valid_columns;   // Device I32 [R]，u，有效 record 前缀。
    int32_t layers;         // L。
    int32_t record_capacity;// R，1..8；本次 B<=R。
};
```

L 为调用包含的层数，支持单层 view 与全部 Main MLA 层；Main/MTP 的不同 frontier
不混入一次 tail 提取。跨层 records 的 outer index 固定为 `layer*R+b`，不能误用
活动 B 作 layer stride。Prepare 获得该层前 B 行的 view；物理 record 行从生产直到
接受结果退役保持身份不变。

W 为正的 record 宽度；写入要求 `0<=offset[b]` 且 `offset[b]+n[b]<=W`。
连续累积 records 时，当前起点为 `base_positions[b]+offset[b]`。有效长度及来源
身份由调用者管理，Op 不自行发布这些 metadata。Tail/record 只要求 feature 连续，
其他 strides 显式、正且不重叠；不能假设 layer、field 或 slot 紧邻。

### 4.3 Selection

```cpp
struct KpoolBlockSelection {
    Tensor pool_ids;        // Device I32 [512,Nselection]。
    Tensor pool_counts;     // Device I32 [Nselection]。
    Tensor selection_end;   // Device I32 [Nselection]。
};
```

Indexer 输出 `Nselection=T*B`，行号为 `t+T*b`。Sparse MLA 另接收 Device I32
`selection_rows[T,B]`，可引用不同宽度、仍有效的 selection 表，包括 MTP 首步结果。
Selection 是执行 frame 数据，不能替代其 cache 的 read lease 或 checkpoint 身份。

## 5. Prepare 的完整计算

### 5.1 参数与表示边界

```cpp
struct KpoolMlaPrepareParameters {
    Weight a_projection;   // BF16 contiguous W_A[2336,4096]，固定 A16。
    Tensor query_norm;     // BF16 [1536]。
    Tensor latent_norm;    // BF16 [512]。
    Tensor index_norm;     // BF16 [128]。
    Tensor index_bias;     // BF16 [128]。
    Tensor pool_ape;        // BF16 [128,4]。
};
```

五类投影结果按[准备数值边界](../../maintainer/glm5next-quantization.md#42-mlaindexer-输入准备)
产生 q_r、latent encoder 输入、raw K/gate 和 FP32 head_coeff；norm、pooling 和
records 共同消费已规定的表示，不在本接口引入额外舍入。

公开 q_r 为 BF16 `[1536,T,B]`，head_coeff 为 FP32 `[32,T,B]`，连续、16 B 对齐。
W_A 输出在 norm 前没有强制 BF16 落点。k_raw/g_raw 是 pooling、records 与 tail 的
共同表示：同一个 pool 的四个 tokens 即使全部处于本次调用，也必须使用这些 BF16
words，不能绕过边界而改用未舍入的 FP32 values。

### 5.2 Pooling 与写域

设活跃行起点 `F=positions[0,b]`、长度 n，before 是 F 处的 canonical tail。
每个新完成的 `g in [floor(F/4),floor((F+n)/4))`，对每个 channel d：

按模型的逐 channel 四 token weighted pooling 计算 pooled key，数值顺序遵循
数值文档。早于 F 的 raw words 从 before 取得，其余使用本次实际产生的 words。
APE 只在 pooling 时加入；LayerNorm 位于 pooling 之前。

Prepare 写 `[F,F+n)` 的 latent、上述完整 pool 范围，以及所选状态出口。
未完成 pool 没有 index cache 写入。允许提前准备整个窗口，但各 query 的可见集合
由第 6 节独立决定，不能使用窗口末端的 coverage 替代逐 query 的因果边界。

### 5.3 两种状态出口

```cpp
void kpool_mla_prepare_record(
    const Tensor& x, const KpoolMlaPrepareParameters& parameters,
    const KpoolMlaBatchView& batch, const KpoolTailStateView& before,
    KpoolMlaCacheView cache, KpoolRawRecordWriteView records,
    Tensor& q_r, Tensor& head_coeff, MemoryArena& workspace,
    CudaExecutionView execution);

void kpool_mla_prepare_update(
    const Tensor& x, const KpoolMlaPrepareParameters& parameters,
    const KpoolMlaBatchView& batch, const KpoolTailStateView& before,
    KpoolMlaCacheView cache, KpoolTailStateView after,
    Tensor& q_r, Tensor& head_coeff, MemoryArena& workspace,
    CudaExecutionView execution);
```

Record 出口逐 word 保存本次实际 k_raw/g_raw 到 `offset[b]+t`，保留 before 不变。
有效 raw words 是唯一记录来源，不用缩短 T 或其他 producer 重跑来恢复接受前缀。
Verify 保留这些 records 直到决定退役；它们不进入长期 checkpoint。

Update 出口按第 8 节直接生成 F+n 处的 after，无需持久保存整段 raw records。
普通 prefill/decode 使用 update；MTP provisional draft state 也可以更新自己的独立
目标。两种出口使用相同计算公式与表示边界，不因出口改变数值政策。

n=0 时 q_r/head_coeff 写精确正零，不写 cache、records 或 after，不读取 state、
positions 或 table selectors 的 payload。矩阵投影允许读取物理 padding x，因此
所有物理 x columns 必须是有效有限 BF16 数据；padding 不得影响有效输出。
无效 record 后缀和未寻址 cache 保持原样。

## 6. Indexer 与分组 selection

```cpp
void kpool_indexer(
    const Tensor& index_query, const Tensor& head_coeff,
    const KpoolIndexCacheView& cache, const KpoolMlaBatchView& batch,
    KpoolExecutionEnvelope envelope, MemoryArena& workspace,
    KpoolBlockSelection& out, CudaExecutionView execution);
```

输入 index_query 为连续 BF16 `[128,32,T,B]`，head_coeff 为连续 FP32 `[32,T,B]`。
对有效 query p，`E=p+1`、`C=floor(E/4)`，只评价 `g in [0,C)`：

```text
dot[h,g] = sum_d(FP32(index_query[d,h]) * FP32(index.K[d,g]))
S[g]     = sum_h(head_coeff[h] * ReLU(dot[h,g]))
```

Q/K 点积的操作数为 BF16，accumulator、head 乘法与归约为 FP32。
S 与模型第 4 节的数学分数只差公共正因子 1/64，省略它是本合同的一部分。
Coefficients 带符号，不能移到 ReLU 内；最终分数可以为负。非法候选直接排除，
不能补零并参与 TopK。Indexer scores 只参与选择，不进入 MLA logits。

固定选择 `min(512,C)` 个不同 pools；计算分数相同时优先较小 logical g。
输出顺序不规定，不要求按分数或 ID 排序。C<=512 直接全选，不读取 query、
coefficients、keys 或 tables payload；仍输出正确 count 与 E。此优化覆盖 E<=2051。

有效输出行 `m=t+T*b` 的 count 为上述选择数，end 为 E。只有前 count 个 IDs 有效，
之后内容不规定。无效行 count/end 写 I32 零，不读 positions、query、coefficients、
cache 或 table selectors；IDs 内容不规定。所有输入/cache 保持不变。

不做近似 TopK。实现可分区、归并或私有排序，但必须满足同一 Top512/同分规则。
只有未完成 pool 的 tail 被强制加入后续 attention；最新的完整 pool 不额外强制入选。

## 7. Sparse MLA 与冻结 tail 边界

```cpp
struct KpoolMlaFactors {
    Tensor key_up;         // BF16 [512,256,64]，逻辑 U_K[h][256,512]。
    Tensor value_up;       // BF16 [512,256,64]，逻辑 U_V[h][256,512]。
};

void kpool_sparse_mla(
    const Tensor& query, const KpoolMlaFactors& factors,
    const KpoolMlaLatentCacheView& cache, const KpoolMlaBatchView& batch,
    const KpoolBlockSelection& selected, const Tensor& selection_rows,
    KpoolExecutionEnvelope envelope, MemoryArena& workspace,
    Tensor& out, CudaExecutionView execution);
```

Query/out 为连续 BF16 `[256,64,T,B]`；selection_rows 为连续 Device I32 `[T,B]`。
有效 query 从指定 selection 行取得 E、count 与 IDs，要求：

- `0<=E<=p+1`，且 selection 属于同一请求、同一层和正确 cache 历史；
- `0<=count<=min(512,floor(E/4))`；
- 前 count 个 IDs 不同、非负且小于 floor(E/4)。

消费者可独立接收任意合法集合，不要求它恰为当前 query 的实际 TopK。
令 J 为所选每个 g 的四个 tokens，再并入 `[4*floor(E/4),E)`。
这两个部分不重叠，至多 2051 tokens。E 是 selection 的冻结边界，不能替换成当前
query 的 p+1。只有有效选中行和有效 tail 被读取。

对每个 head：

Query 经 U_K 映射、latent attention 和 U_V 映射后产生 BF16 heads，完整数学与
舍入顺序见[latent attention 数值政策](../../maintainer/glm5next-quantization.md#43-latent-域-attention)。

c 来自所选 cache，FP8 profile 遵守
[数值文档的 cache 消费规则](../../maintainer/glm5next-quantization.md#33-消费)。Scale 固定为 1/16，与训练的
256 维 QK 匹配，不因 latent=512 改变。没有 RoPE、attention bias、output gate、
额外 sink 或额外强制 self token。

QK、加权累加、因子投影与私有 q_latent/u 均遵循上述数值政策。公共输出为完整
U_V 映射后的 BF16 heads，最终 O projection 不在本 Op 中。

空 J 与无效 query 输出精确 BF16 正零，不读取 query/factors/cache payload；
无效 query 还不读取 positions、selection 或 table selectors。输入、selection、
metadata 和 cache 保持不变。Private tiling 不改变这一读域；不加载无效第四条 tail。

## 8. Canonical tail 与精确前缀提取

### 8.1 状态

每层在前沿 F 的 source slot，前 `F%4` 个 raw K/gate columns 为未完成 pool，
按时间顺序存放；其余 columns 为 BF16 正零。相位由 F 推导，不在每层保存另一份
计数。F=0 时两 planes 全零。APE 不存入 raw gate。

StateImage 持有 tail；完整 pools 留在 growing pool。Active image、不可变 checkpoint
及目标 image 的选择、复制和保护由现有目录负责。Tail 无法从 MLA latent 反推恢复。

### 8.2 提取与零接受

```cpp
void kpool_tail_extract(
    const KpoolTailAllLayersView& before, const KpoolRawRecordReadView& records,
    const Tensor& accepted_columns, KpoolTailAllLayersView after,
    CudaExecutionView execution);
```

accepted_columns 为 Device I32 `[B]`。每行要求 `0<=a<=u<=W`，before 对应 F0，
`F0+u<=1048576`。L、record_capacity 与 views 匹配；before/after 的 slot 数可以不同。
可用 L=1 的 view 提取一层，或一次覆盖 Main 全部 MLA 层；不由模型逐 token 回放。

对 a>0：

```text
F1 = F0 + a
r  = F1 % 4
s  = F1 - r

for j in [0,3):
    if j >= r:
        after.K[:,j] = BF16 +0
        after.G[:,j] = BF16 +0
    elif s+j < F0:
        old_column = s+j - (F0 - F0%4)
        after.K/G[:,j] = before.K/G[:,old_column]
    else:
        after.K/G[:,j] = records.K/G[:,s+j-F0]
```

这是精确 word 复制/零填充，无浮点运算；保留复制 words 的位模式。只读取所需
接受前缀，不能读取 rejected suffix 或扫描更早完成的 pools。

a=0 为严格 no-op：不读写任何 state/record payload，即使 source!=destination
也不复制。该行继续采用原 image，不能采用未写的目标。此处明确不同于现有 QSA
tail_extract 的异址零接受复制行为，与已定 KDA 提交的零接受规则一致。

允许同一行原位提取，结果使用原始 source words。目标区域互不重叠，不能覆盖
其他活动行的 source 或任何 records/metadata；只读 source 可共享。Alias 按实际
选中地址定义，不按 view 对象或 slot 编号推断。仅修改本 family 的 tail 字段，
不触碰 KDA、hidden 或其他 StateImage fields。无全局 scratch 或隐藏分配。

Prepare/update 对有效 n 采用同一 tail 公式，将本次实际 raw words 视作临时记录；
n=0 同样不写目标。它不需要向调用者暴露或保留整段 records。

### 8.3 Growing cache 的采用

F1=F0+a 处的有效范围：

```text
latent       : [0,F1)
pooled keys  : [0,floor(F1/4))
tail         : [4*floor(F1/4),F1)
```

例如 F0=63、本轮准备到 69、只接受到 65：pool [60,64) 有效；[64,68) 即使已物理
写入也不可见，新 tail 只保存位置 64 的 raw words。Rejected suffix 不扩大 coverage。
共享页的保护/COW、reservations、table 更新和 frontier 发布由调用者负责。
完整 checkpoint 同时持有正确的 growing prefix、tail 以及其他模型状态。

## 9. MTP 与 DP owner 交接

Main/MTP cache 和 tail 分离、owner 相同，不跨请求或跨层共享 selection。
一轮 MTP draft 首步正常生成 selection，后续 steps 复用 pool IDs、count 和 E；
不能仅冻结 pools 后按新位置重算 tail。

后续 steps 省去 Index Q projection、评分与 TopK，MLA Q 使用 W_B 前 16384 行。
Prepare 仍维护独立 provisional latent、pooled keys 与 tail，并按接口产生
head_coeff；复用步不消费该 coefficient。这样 grouped planes 的 provisional 内容
保持完整。Selection 及其映射存于 frame，cache read leases 保持到最后消费者退役；
新 selection 写入不能覆盖仍被引用的冻结行。

Canonical MTP 历史按真实 Main conditioning 更新。递归 draft 的 hidden 不因 token
被接受就等于真实 Main hidden；需要重建的接受部分使用正常 prepare/update 链路，
不能通过截短 draft cache 直接采用这些状态。Family 使用调用者提供的各自 ordinal、
状态与 cache views，不自行推导 Main/MTP 前沿偏移、输入 token 或 conditioning。

RankProgram 持有固定 backing、frames、workspace 与 Graph；ModelSession 选择
source/destination、接受长度、前沿、checkpoint 及分布式采用关系。Op 不分配页，
不选 owner，不执行通信。Owner 的完整 block output 由模型执行交给后续 TP/EP；
KDA 仍使用两 ranks 的独立 TP state 分片。

## 10. 内存、workspace 与 Graph

Activation/output 连续且至少 16 B 对齐，BF16/FP32 参数向量自然对齐；Device I32
metadata 连续并自然对齐。State/record feature 连续，允许正的显式外层 strides，
访问跨度必须可表示且不重叠；cache 页内布局严格遵守第 3 节。

除第 8 节明确允许的同一行 tail 原位更新，所有写区域与读取区域不重叠。
Read-only views 可共享 backing；输出、metadata、cache、records 和活跃 scratch
的实际访问区域不得发生非法 write/read overlap。Host 检查可见的 dtype、shape、
stride、alignment、capacity、device 和固定地址关系；Device counts、selectors、
可读页及选择后 alias 关系由调用者保证，不回读 Device metadata。

Prepare/Indexer/sparse MLA 分别提供 caller-owned workspace capacity query：

```cpp
size_t kpool_mla_prepare_record_workspace_capacity_bytes(
    KpoolMlaStorage storage, int32_t min_tokens, int32_t max_tokens,
    int32_t batch_size, int32_t multiprocessor_count);
size_t kpool_mla_prepare_update_workspace_capacity_bytes(
    KpoolMlaStorage storage, int32_t min_tokens, int32_t max_tokens,
    int32_t batch_size, int32_t multiprocessor_count);
size_t kpool_indexer_workspace_capacity_bytes(
    int32_t min_tokens, int32_t max_tokens, int32_t batch_size,
    KpoolExecutionEnvelope envelope, int32_t multiprocessor_count);
size_t kpool_sparse_mla_workspace_capacity_bytes(
    KpoolMlaStorage storage, int32_t min_tokens, int32_t max_tokens,
    int32_t batch_size, KpoolExecutionEnvelope envelope,
    int32_t multiprocessor_count);
```

Query 覆盖固定 profile/B/envelope 下整个正 T 区间，包括分派边界；以 256 B 对齐的
Op-local cursor 计内部峰值和对齐间隙，不含输出/cache/state/records backing，也不含
末端 padding。SM count 与 execution 相同。执行与 sizing 共用分派及分配规则。

Scratch 使用有界 query windows；不要求物化 [query,head,context] 的完整 Indexer
分数，不建立全局展开的 BF16 latent cache。独立的 records/selection 由 frame 保持
到其最后读者退役，不能作为已经可复用的临时 arena 空间。

同一固定 T/B、profile、views、envelope 与 device facts 的 Graph replay 从 Device
读取新的 counts、positions、slots、page mappings 与 selection mappings。无 Host
readback、隐藏分配或依赖 Device 值改变的 Host launch 拓扑。改变 Host envelope 时准备
符合资源上界的 capture/update；改写 Host 变量不会修改已经捕获的 Graph。
操作提交到 execution.stream()，同流顺序或显式依赖保证生产/消费可见性。Workspace、
records、state 和 page leases 的实际 device lifetime 由调用者保持。

## 11. 容量与性能口径

1M growing cache、请求 tail、TP2 KDA state 和执行资源的统一预算见
[并行资源设计](parallel.md#3-1m-上下文的持久状态容量)。Raw records、selection 和
workspace 按本设计的实际 T/B/容量及第 10 节 sizing 单独计费，不能只按 cache
payload 估算启动内存。

性能比较覆盖完整 prepare/indexer/sparse-MLA/tail 链路，区分 BF16 与 FP8 latent
存储，记录实际上下文、T/B、有效长度、selection 复用和 Graph 条件。FP8 收益需
计入 scale 读取、片上解码及分支投影，四 token 连续加载的收益需在完整消费者上
测量；局部 kernel 数据不能替代双机模型结果。

测量通过完整公开 Op，固定硬件/toolchain、输入、T/B、context、storage、selection
与运行条件，分别报告 prepare 两种出口、Indexer、sparse MLA、tail 提取及必要的组合。
覆盖四种相位、短上下文全选、长上下文扫描与 MTP 冻结选择；报告临时空间、分布、
最差变化与例外。对照和工作负载在评估前固定，不以微基准替代双机模型性能。

## 12. 验收合同与实现边界

| 对象 | 验收依据 |
|---|---|
| Projection、norm、pooling、MLA | 从 represented inputs 出发的独立 FP64 数学 oracle，融合按完整公式检查 |
| FP8 encoder | 给定 FP32 输入的独立精确 codec oracle，覆盖舍入、零、极小值及饱和 |
| Tail、raw record 保存、无效区域 | 逐 word/byte 精确检查，包括未触碰范围 |
| Indexer | 独立评分 oracle；明确同分 cases 精确匹配，近同分按 FP32 点积/head 归约误差界检查 |
| FP8 MLA 消费 | 独立解读实际 code/scale；存储误差与 attention 算术误差分开，不因 FP8 cache 降低计算政策 |

真实产生的 BF16 raw K/gate 是可观察状态边界；验证该生产计算的数学正确性后，
tail 提取必须精确复制同一物理窗口的实际 words。缩短 T 重跑、另一个 kernel 的
近似输出或可读生成结果不能替代这两个独立要求。

有限 conformance cases 覆盖：

- 两种 MLA storage、两个 prepare 出口、五个语义入口及实际生产分派边界；
- 四种 pool 相位、任意 chunk 起点、跨页/跨 chunk、混合长度、零长度行；
- E=2051/2052 的全选边界、负 coefficients、同分、不同 selection row 映射、
  最新完整 pool 可被排除与未完成 tail 必须加入；
- 1M 索引域、任意 physical page 排列，以及未来已写 pool 对早期 query 不可见；
- 零接受、短接受、全接受、原位/异位、共享只读源、非法跨行覆盖；
- 同一物理窗口 actual records、rejected suffix 改写不影响提交、未寻址区域与 guards；
- MTP 首步和冻结 E 复用、真实 conditioning 重建、Graph metadata 改变后的行为。

浮点归约不要求不同合法路线逐位相同；固定数值政策的数学验收不能按 T/B、storage
或私有路线选择性放宽。Exact codec、copy 与 no-op 仍逐位验证。本合同未声明任何
已测精度损失或性能收益。

实现归属固定为 `include/ninfer/ops/kpool_sparse_mla.h` 和对应
`src/ops/softmax_attention/kpool_sparse_mla/` family；纯 Q/O 投影留在 Linear。
Core 管理非 owning views、checked layout 和固定 storage，模型执行只经公开合同
调用。Header、Core views/layout、codec 和资格测试落地后分别接管对应合同；
模型接入完成后由实际执行代码及当前执行说明接管组合关系。其余部分继续保留，
全部接管后删除本文。实施状态与任务范围只在设计索引和基础清单维护；文档迁移
不授权执行 agent 重新选择 codec、布局或状态语义。
