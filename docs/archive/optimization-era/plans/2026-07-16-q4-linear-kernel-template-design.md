# Q4 RowSplit 纯 Linear 三模板族设计

> 状态：已实现并完成 production 原子切换；route、数值、资源、NCU、真实
> Text/Vision/MTP 和 matched end-to-end 验收均已完成。
>
> 范围：只讨论 RTX 5090 `sm_120a` 上的纯
> `Q4G64_F16S / row-split-k128-v1` Linear：
> `Y[Rows, Cols] = W[Rows, K] * X[K, Cols]`。本文不设计 Q5、Q6、W8、Dense、
> grouped projection、SwiGLU、MoE grouped 或任意运行时 epilogue。
>
> 上位边界：
> [Linear Kernel Architecture](2026-07-16-linear-kernel-architecture-refactor.md)。
> 当前算子和物理问题：
> [Linear 算子盘点](2026-07-16-linear-operator-inventory.md)。
> 测量、失败方案和资格限制：
> [Linear Kernel Architecture Experiment Log](2026-07-16-linear-kernel-architecture-experiment-log.md)。

## 1. 设计决定

Q4 纯 Linear 收敛为三个完整 kernel lifecycle：

```text
q4_rowsplit_gemv_kernel
q4_rowsplit_gemm_simt_kernel
q4_rowsplit_gemm_mma_kernel
```

它们分别承担：

1. 一个或多个 warp 合作计算一个输出行的 GEMV；
2. 一个 warp 同时计算一个输出行的多个输出列，并在列 tile 内复用一次权重流；
3. 将 Q4 权重解码为 BF16 tile 后使用 Tensor Core MMA。

这里的“三个 kernel”指三个模板族，而不是三个固定的 CUDA binary entry。每个模板族
可以有少量经过测量的编译期 schedule 实例。生产 planner 只能选择这些闭合
`Q4ScheduleId + Q4KernelVariant`，不能在运行时自由组合模板参数。

三个分析区域与三个 kernel 族不是一一强绑定：

- `Cols == 1` 可以选择 GEMV，也可以选择 SIMT `ColsPerTile=4`；
- 较小 `Cols` 可以选择 SIMT，也可以在实测 crossover 后选择 MMA；
- “T1 / Small-T / Prefill”只描述问题规模，不进入 kernel、policy 或文件名。

这一点是强制约束。最终资格实验已经证明：

- `[1024,5120]`、`[4096,5120]`、`[6144,5120]` 虽然 K 相同，C4/C8 的切换点分别是
  16、5、8；
- `[131072,5120] C1` 选择 R4/W1 direct GEMV，而三个较小 K5120 row view 的 C1
  选择 R1/W8 split-K GEMV；
- 两个 Vision shape 的 early-SIMT 路由不同，但都在 C320/324 之间由 MMA C64
  切换到 MMA C128；
- 因此不能用场景名、Rows 大小或一个全局 token 阈值决定 lifecycle。

## 2. 目标与非目标

### 2.1 目标

本设计必须做到：

1. 用三个中立命名的模板族覆盖当前主要 Q4 纯 Linear 计算几何；
2. 使相似的对齐 shape 默认只需测试已有 `Q4ScheduleId` 并增加 route；
3. 保留对 CTA ownership、列 tile、K stage、pipeline、cache、MMA tile 和边界策略的
   编译期控制；
4. 避免运行时格式分支、运行时 schedule 组合和热循环中的多态；
5. 保持 `linear(x, w, out, ws, stream)` 接口不变；
6. 使 Q4 的 capability、admission、route、fixed launch 和实验候选各有唯一所有者；
7. 在饱和区域达到相应的带宽或 Tensor Core roofline，在非饱和区域选择实测最低延迟
   policy；
8. 在代码切换前保存数值、性能、资源和 profiler 证据。

### 2.2 非目标

本文明确不做：

- 一个统一低比特框架；
- 把 Q4/Q5/Q6/W8 重新抽象为运行时格式；
- arbitrary aligned shape 的产品 fallback；
- caller 或 target 传入 policy/profile；
- application role 驱动的 kernel 命名；
- runtime epilogue、callback 或 function pointer；
- grouped、paired、SwiGLU 或 residual topology；
- 为所有模板参数生成笛卡尔积实例；
- 仅因为某个配置“理论上更通用”就接受稳定性能退化。

## 3. 数学、存储与能力合同

### 3.1 语义

输入和输出使用以下逻辑维度：

```text
W     : Q4 RowSplit [Rows, K]
X     : contiguous BF16 [K, Cols]
Y     : contiguous BF16 [Rows, Cols]

数学 oracle:
  Ref[row,col]
    = sum_k FP64(dequantize_to_fp32(W[row,k]))
            * FP64(BF16(X[k,col]))
```

`Rows` 对应当前代码的 `n`，`Cols` 对应当前代码的 `t`。实际 backend 数值边界是：

- GEMV/SIMT：FP32 dequant + FP32 FMA，最后 BF16 store；
- MMA：Q4 先反量化并舍入为 BF16 operand，再做 BF16 MMA/FP32 accumulate，最后 BF16
  store。

因此上式定义独立比较 oracle，不承诺三个 lifecycle 使用相同的中间舍入或逐 word
输出。

### 3.2 固定格式事实

以下事实属于本次 Q4 kernel 的定义，不是模板参数：

| 固定事实 | 值 |
|---|---|
| 数值格式 | signed Q4 |
| group K | 64 |
| code bytes/group | 32 |
| scale | FP16，一个 group 一个 scale |
| physical layout | RowSplit `row-split-k128-v1` |
| logical/padded K | `Kpad == K` |
| K alignment | `K % 128 == 0` |
| X/Y dtype | BF16 |
| SIMT accumulation | FP32 FMA |
| MMA operand | BF16 |
| MMA accumulation | FP32 |
| final store | BF16 plain store |
| auxiliary output | 无 |
| fused finalizer | 无 |

禁止为了源码统一引入以下模板轴：

```cpp
NumericFormat
PhysicalEncoding
ScaleType
InputType
OutputType
AccumulatorType
Epilogue
```

SIMT 和 MMA 使用不同的 Q4 decode atom：

- GEMV/SIMT：Q4 × FP16 scale 形成 FP32 权重值，然后执行 FP32 FMA；
- MMA：Q4 × FP16 scale 在 shared tile 中舍入为 BF16，然后执行 BF16 MMA。

二者不能被一个“通用 decode 公式”强制统一。

### 3.3 首期 capability envelope

新 Q4 backend 的快速能力合同为：

```text
format              = Q4G64_F16S
layout              = RowSplit
Kpad                = K
K % 128             = 0
Rows % 16           = 0
X/Y contiguous      = true
X/Y base alignment  >= 16 B
code/scale views    = validated RowSplit views
```

`Rows % 16 == 0` 覆盖所有当前主要 Q4 几何，包括 `Rows=4304`。不能提高为
`Rows % 32 == 0` 或 `Rows % 64 == 0`。

当前需要覆盖的底层计算几何是：

```text
[1024,5120]
[4096,5120]
[6144,5120]
[34816,5120]
[131072,5120]
[3456,1152]
[4304,1152]
```

未来 `[131072,2048]` 是 measurement-only 设计压力，不在本文中注册产品支持。

grouped input、LinearSwiGLU 和未来 MoE grouped 可以复用下层 Q4 atom 或 producer，但它们
不共享纯 Linear route catalog，也不通过纯 Linear capability 自动获得支持。

## 4. 模板参数边界

### 4.1 必须模板化的事实

如果一个事实会改变以下任一项，就应当是模板参数或闭合 policy type：

- CTA/warp 对输出的 ownership；
- 静态线程数；
- accumulator 数量与寄存器压力；
- shared memory 数组形状；
- pipeline stage 数和 `cp.async` wait 距离；
- reduction 与 barrier 拓扑；
- load/cache 指令；
- MMA tile、warp tile 和 fragment 数量；
- full/checked 热路径；
- `__launch_bounds__` 和编译器寄存器约束。

### 4.2 默认保持运行时的事实

以下值默认保持运行时：

```text
Rows
Cols
K
groups_per_row
K stage count
最后一个 K stage 的 active groups
grid dimensions
codes/scales/X/Y pointers
```

原因是这些值主要改变循环次数和地址，不应使注册一个相似 shape 自动产生新 CUDA
实例。

允许增加 exact-K 实验实例，但必须同时满足：

1. 运行时 K 的现有实例在该 K 上存在可重复的显著退化；
2. exact-K 实例仍复用同一个 lifecycle 模板；
3. route 明确选择该实例；
4. 新实例通过资源、SASS、数值和三进程冷启动 timing 检查。

exact `Rows` 和 exact `Cols` 不作为生产模板参数。

### 4.3 闭合 schedule，而不是自由参数包

模板定义可以表达较宽的合法空间，但生产只显式实例化少量命名 schedule：

```cpp
enum class Q4KernelVariant {
    None,
    Full,
    Predicated,
};

template<class Schedule>
__global__ void q4_rowsplit_gemv_kernel(...);

template<class Schedule, Q4KernelVariant Variant>
__global__ void q4_rowsplit_gemm_simt_kernel(...);

template<class Schedule, Q4KernelVariant Variant>
__global__ void q4_rowsplit_gemm_mma_kernel(...);
```

planner 只接触：

```cpp
enum class Q4ScheduleId;
enum class Q4KernelVariant;
```

它不接触模板参数，不构造 schedule，不选择 cache flag，也不把一组 knobs 传给 launcher。

## 5. 公共 Q4 私有基座

三个模板族只共享 Q4 私有的窄基座：

```text
Q4 RowSplit row/group address calculation
packed nibble decode constants
FP16 scale load
16B vector load/copy helpers
group-stage validity helpers
warp reduction helpers
launch-contract validation
```

建议内部类型：

```cpp
struct Q4RowSplitStorage {
    static constexpr int kGroupK = 64;
    static constexpr int kCodeBytesPerGroup = 32;
    static constexpr int kScaleBytesPerGroup = 2;
};

struct Q4SimtDecodeAtom;
struct Q4MmaDecodeAtom;
```

`Q4SimtDecodeAtom` 和 `Q4MmaDecodeAtom` 是不同的手写 atom。它们可以共享 signed nibble
常量，但不共享一条完整的 decode 指令路径。

公共基座不能吸收：

- K-loop lifetime；
- output traversal；
- CTA ownership；
- accumulator topology；
- pipeline stage；
- full/edge 选择；
- application role；
- fused finalizer。

## 6. 模板族一：Q4 RowSplit GEMV

### 6.1 身份与用途

kernel 名称：

```cpp
q4_rowsplit_gemv_kernel
```

它只描述：

```text
Cols == 1
一个或多个 warp 合作完成每个输出行的 K dot product
```

它不叫 head、GDN、attention、decode 或 T1-special。

### 6.2 Schedule

```cpp
enum class Q4ActivationAccess {
    Direct,
    CtaSharedFullK,
};

enum class Q4GemvLaneMapping {
    PackedByte2,
    PackedWord8,
};

enum class Q4GemvDecodeMode {
    ScalarInteger,
    Fp16Mantissa,
};

enum class Q4GemvCodeTransfer {
    SyncVector16,
    AsyncVector16,
};

enum class Q4GemvScaleAccess {
    Scalar16Shuffle,
    SharedPair32,
};

enum class Q4CacheOp {
    Ca,
    Cg,
};

template<
    int RowsPerCta,
    int WarpsPerRow,
    int GroupsPerWarpTile,
    int PipelineStages,
    Q4ActivationAccess ActivationAccess,
    Q4GemvLaneMapping LaneMapping,
    Q4GemvDecodeMode DecodeMode,
    Q4GemvCodeTransfer CodeTransfer,
    Q4GemvScaleAccess ScaleAccess,
    Q4CacheOp CodeCache,
    int StaticGroupsPerRow,
    int LaunchBoundsMinBlocks>
struct Q4RowSplitGemvSchedule;
```

派生事实：

```cpp
static constexpr int kCtaWarps = RowsPerCta * WarpsPerRow;
static constexpr int kThreads  = kCtaWarps * 32;
```

`Threads` 不是独立模板参数。

### 6.3 参数语义

#### `RowsPerCta`

一个 CTA 同时完成多少个输出行。它决定：

- CTA 数量和 wave 数；
- CTA warp 数；
- code/scale shared 分区；
- X 在 CTA 内的复用机会。

首期候选是 1、2、4、8，但只实例化实测 schedule。

#### `WarpsPerRow`

一个输出行由多少个 warp 分担：

```text
WarpsPerRow = 1  -> warp-per-row
WarpsPerRow > 1  -> split-K-per-row
```

它静态决定：

- 每个 warp 负责的 group 序列；
- partial accumulator 数；
- CTA reduction shared；
- `__syncthreads()`；
- 最终 store owner。

warp-per-row 和 split-K 是同一模板族的不同 partial specialization，不是 kernel 内运行时
分支。

#### `GroupsPerWarpTile`

一个 warp 每次加载和消费多少个 Q4 group。首期候选：

```text
8 groups  = 512 K values
16 groups = 1024 K values
```

它决定 code/scale shared 大小、循环展开和 load cadence。

#### `PipelineStages`

决定 code/scale pipeline 深度。首期候选 1 或 2。它必须是模板参数，因为会改变：

- shared memory；
- `cp.async` commit/wait；
- barrier 位置；
- occupancy。

#### `LaneMapping` 与 `DecodeMode`

这两个轴描述 lane 实际消费的 packed ownership，不能隐藏在一个看似通用的 decode
helper 后：

- `PackedByte2 + ScalarInteger`：每个 lane 对一个 group 读取一个 packed byte，产生两个
  Q4 值；warp 依次遍历 group；
- `PackedWord8 + Fp16Mantissa`：每个 lane 读取一个 32-bit packed word，产生八个 Q4 值；
  warp 在一个 phase 中覆盖四个 group。

两种映射改变 X load 宽度、scale 广播、寄存器 ILP 和 hot-loop SASS。E053 的首轮
split-K 回退证明它们不能被当作等价实现细节。合法组合由 `static_assert` 闭合，不能
运行时选择。

#### `CodeTransfer` 与 `ScaleAccess`

首期实现两种真实的数据供给拓扑：

- `SyncVector16 + Scalar16Shuffle`：lane 同步加载 16B code vector，活跃 lane 各加载一个
  16-bit scale，再通过 warp shuffle 广播；该路径使用单 stage；
- `AsyncVector16 + SharedPair32`：code 用 16B `cp.async`，相邻两个 FP16 scale 用 4B
  `cp.async` 放入 shared。

它们影响 shared memory、barrier、寄存器和指令序列，因此均为 schedule 轴。
`CodeCache` 只对实现允许的 transfer 组合生效。

#### `ActivationAccess`

两种真实存在的 X 路径：

- `Direct`：warp 从 global/L1 读取 X；
- `CtaSharedFullK`：CTA 将完整运行时 K 的 X 放入动态 shared 后由多个输出行复用。

二者必须编译为不同 load atom。launcher 可以依据已选 `Q4ScheduleId` 提供动态 shared
字节数，但不能运行时选择 access mode。

#### `CodeCache`

控制 Q4 code 的 cache operator。首期只实例化一个已测 winner；保留该参数是因为
`.ca/.cg` 会改变实际 load 指令。

#### `LaunchBoundsMinBlocks`

作为 schedule 字段进入 `__launch_bounds__`，用于约束编译器寄存器分配。它只是编译
约束，不等于实际驻留 block 数；动态 shared、寄存器和线程上限仍可能降低 residency。

#### `StaticGroupsPerRow`

默认值 0 表示 K/group 数完全运行时。只有 matched timing 和 SASS 证明运行时 ownership
阻止编译器生成已知 winner 时，才允许为一个广泛复用的 K 几何绑定正值：

```text
0   -> groups_per_row = K / 64
80  -> K=5120 的 80 个 group；Rows 仍为运行时
```

它不是 exact Rows 或应用 shape 特化。一个 `StaticGroupsPerRow=80` 的 split-K schedule
可服务所有合法的 K5120 Rows；launcher 必须验证运行时 K 与该值一致。新增其他静态值
仍需要独立性能证据，不能把常见 K 列表无条件实例化。

### 6.4 静态合法性

```text
RowsPerCta > 0
WarpsPerRow > 0
GroupsPerWarpTile > 0
GroupsPerWarpTile % 2 == 0
PipelineStages >= 1
LaunchBoundsMinBlocks >= 1
StaticGroupsPerRow >= 0
StaticGroupsPerRow == 0 || StaticGroupsPerRow % 2 == 0
RowsPerCta * WarpsPerRow <= 32
Threads <= 1024
static shared <= kernel static limit
LaneMapping/DecodeMode/CodeTransfer/ScaleAccess 为已实现的闭合组合
```

`CtaSharedFullK` 还要求 host capability 根据运行时 K 检查：

```text
static shared + K*sizeof(BF16) <= configured dynamic shared limit
```

### 6.5 K ownership

split-K 使用运行时均衡的连续 scale-pair 分区，而不是要求 group 数可整除 warp 数。
因为 `K % 128 == 0`，每行始终包含整数个相邻 group pair：

```cpp
const int pairs_per_row = groups_per_row / 2;
const int pair_begin =
    (pairs_per_row * warp_in_row) / WarpsPerRow;
const int pair_end =
    (pairs_per_row * (warp_in_row + 1)) / WarpsPerRow;
const int group_begin = pair_begin * 2;
const int group_end   = pair_end * 2;

for (int group0 = group_begin;
     group0 < group_end;
     group0 += GroupsPerWarpTile) {
    consume_group_tile(group0);
}
```

每个 warp 获得一个尽量均衡的连续 group 区间，区间计算位于 K hot loop 之外。这样既不
要求整除，也保证区间起点和终点保持 group-pair 对齐。`SharedPair32` 以 4B 加载 scale；
`Scalar16Shuffle` 可逐个加载 2B scale，但 ownership 边界仍按 pair 切分。静态要求
`GroupsPerWarpTile` 为正偶数。`StaticGroupsPerRow=0` 的实例可以合法覆盖：

```text
K=1152  -> 18 groups
K=2048  -> 32 groups
K=5120  -> 80 groups
```

最后一个不足 `GroupsPerWarpTile` 的 tile 仍只加载完整 Q4 group 的 code；
`SharedPair32` 不产生 2B `cp.async`，`Scalar16Shuffle` 使用普通 2B load。consumer
只访问 active group 对应的 X，不存在 inactive X load 或逐元素 K tail。

### 6.6 初始候选

首期至少实现：

```text
GemvR4W1G16:
  RowsPerCta=4, WarpsPerRow=1, GroupsPerWarpTile=16
  PackedWord8/Fp16Mantissa, AsyncVector16/SharedPair32
  StaticGroupsPerRow=0

GemvR1W8G16K5120:
  RowsPerCta=1, WarpsPerRow=8, GroupsPerWarpTile=16
  PackedByte2/ScalarInteger, SyncVector16/Scalar16Shuffle
  StaticGroupsPerRow=80
```

实验可以增加：

```text
GemvR2W4G16
GemvR8W1G16
```

但只有被 route 使用或正在进行资格比较的实例才进入 binary。

### 6.7 T1 不独占 GEMV

当 `Cols == 1` 时，resolver 必须同时允许：

```text
GEMV warp-per-row
GEMV split-K
SIMT ColsPerTile=4
```

资源更少、模板名更“专用”都不是选择依据。winner 由 exact shape 的冷缓存测量决定。

## 7. 模板族二：Q4 RowSplit SIMT GEMM

### 7.1 身份与 lifecycle

kernel 名称：

```cpp
q4_rowsplit_gemm_simt_kernel
```

每个 warp 拥有一个输出行和一个输出列 tile：

```text
CTA x dimension  -> Rows tile
CTA y dimension  -> Cols tile
one warp         -> one output row across ColsPerTile columns
```

同一个权重 row stage 在 `ColsPerTile` 个 accumulator 之间复用。

### 7.2 Schedule

```cpp
template<
    int RowsPerCta,
    int ColsPerTile,
    int GroupsPerStage,
    int PipelineStages,
    Q4CacheOp CodeCache,
    int LaunchBoundsMinBlocks>
struct Q4RowSplitSimtGemmSchedule;
```

派生事实：

```cpp
static constexpr int kCtaWarps = RowsPerCta;
static constexpr int kThreads  = RowsPerCta * 32;
static constexpr int kStageK   = GroupsPerStage * 64;
```

### 7.3 参数语义

#### `RowsPerCta`

一个 CTA 中的 warp 数和同时处理的输出行数。当前可靠起点为 8。

#### `ColsPerTile`

一个 warp 同时持有的输出列 accumulator 数。它是此 lifecycle 最重要的资源轴：

- 增大它可以在更多列之间复用一次权重流；
- 同时线性增加 accumulator、X loads 和寄存器压力；
- 当前 TT16 已经被证明资源过重。

模板定义可以支持 1..8，但首批只实例化 4 和 8。

#### `GroupsPerStage`

取代旧实现中硬编码的 1024-value slab：

```text
GroupsPerStage=8  -> K stage 512
GroupsPerStage=16 -> K stage 1024
```

首期使用 16；8 只作为 K1152 等几何的实验候选。

#### `PipelineStages`

首期使用 2。更深 pipeline 必须作为独立 schedule 测量，不能由 launcher 自动尝试。

#### `CodeCache` 与 `LaunchBoundsMinBlocks`

分别控制 load 指令和编译器资源约束。后者不声明实际 residency。

### 7.4 静态合法性

```text
RowsPerCta > 0
ColsPerTile > 0
ColsPerTile <= 8
GroupsPerStage > 0
GroupsPerStage % 2 == 0
PipelineStages >= 2
LaunchBoundsMinBlocks >= 1
RowsPerCta <= 32
Threads <= 1024
static shared <= kernel static limit
```

### 7.5 group-granular K edge

旧通用 Small-T kernel 只 pipeline 完整的 1024-value slab，剩余 K 使用 scalar pair
路径。Q4-only 新实现不保留该退化：

```text
K % 128 == 0
=> groups_per_row 是偶数
=> 最后一个 stage 可能不足 GroupsPerStage
=> code 仍按完整 group 的 32B 向量加载
=> scale 始终以相邻两个 FP16 scale 的 4B pair 加载
```

partial-stage atom 显式接收一个偶数 `active_groups`：

```text
code copy:
  仅 active group 发出合法 global vector load；
  inactive shared 区域由参与线程置零。

scale copy:
  active_groups / 2 个 4B scale-pair load；
  不存在 2B cp.async。

consume:
  只迭代 active group；
  只有 active group 才计算对应 X 地址并发出 X load；
  inactive group 绝不读取越界 X。
```

因此：

```text
K=1152  -> 16 groups full stage + 2 groups checked stage
K=2048  -> 2 x 16 groups full stage
K=5120  -> 5 x 16 groups full stage
```

不存在逐元素、奇数 K 或 padded-K 路径。

### 7.6 Kernel variant

首期边界类型：

```cpp
Q4KernelVariant::None
Q4KernelVariant::Full
Q4KernelVariant::Predicated
```

SIMT 使用 `Full` 或 `Predicated`；`None` 只用于没有 tile variant 的 GEMV。`Full` 要求：

```text
Rows % RowsPerCta == 0
Cols % ColsPerTile == 0
K stages 全满
```

`Predicated` 处理最后一个 Cols tile 和最后一个 group stage。当前主要 Rows 都是 8 的整数倍，
所以首期没有必要为 row-only、col-only 和 K-only 生成四种组合。

只有 profiling 证明统一 `Predicated` 对主流 full tile 有显著代价时，才细分 variant。

### 7.7 初始候选

```text
SimtR8C4G16 : RowsPerCta=8, ColsPerTile=4, GroupsPerStage=16, Stages=2
SimtR8C8G16 : RowsPerCta=8, ColsPerTile=8, GroupsPerStage=16, Stages=2
```

它们对应当前已验证的 TT4/TT8 lifecycle，但使用 Q4-only 存储和 group edge。

不自动生成 TT1/TT2/TT3/TT5/TT6/TT7。

## 8. 模板族三：Q4 RowSplit MMA GEMM

### 8.1 身份与 lifecycle

kernel 名称：

```cpp
q4_rowsplit_gemm_mma_kernel
```

每个 CTA 拥有一个：

```text
BlockRows x BlockCols
```

输出 tile。K 以 `BlockK` 递进：

1. X tile 进入 shared；
2. Q4 code/scale 进入 shared；
3. Q4 权重被解码为 BF16 A tile；
4. warp 使用 `mma.sync.m16n8k16` 形成 FP32 accumulator；
5. plain BF16 store。

### 8.2 Schedule

```cpp
enum class Q4FragmentPipeline {
    Serial,
    PingPong,
};

enum class Q4ScaleLoad {
    Scalar16,
    Pair32,
};

template<
    int BlockRows,
    int BlockCols,
    int BlockK,
    int WarpRows,
    int WarpCols,
    int PipelineStages,
    int LaunchBoundsMinBlocks,
    Q4FragmentPipeline FragmentPipeline,
    Q4CacheOp QuantCache,
    Q4CacheOp ActivationCache,
    Q4ScaleLoad ScaleLoadMode>
struct Q4RowSplitMmaGemmSchedule;
```

派生而不是自由指定：

```cpp
WarpGridRows = BlockRows / WarpRows;
WarpGridCols = BlockCols / WarpCols;
Warps        = WarpGridRows * WarpGridCols;
Threads      = Warps * 32;
MmaRows      = WarpRows / 16;
MmaCols      = WarpCols / 8;
GroupsPerK   = BlockK / 64;
SharedBytes  = constexpr estimate;
```

静态约束：

```text
BlockK == 64
BlockRows % WarpRows == 0
BlockCols % WarpCols == 0
WarpRows % 16 == 0
WarpCols % 8 == 0
PipelineStages >= 2
LaunchBoundsMinBlocks >= 1
Threads <= 1024
SharedBytes <= selected sm_120a shared limit
```

虽然 `BlockK` 保留在 schedule 类型中以忠实描述 tile，首期实现静态要求
`BlockK == 64`。支持多 group K tile 需要重新设计 shared raw-Q4 layout、scale lifetime
和 decode atom，不能只放宽一个 `static_assert`。

### 8.3 关键模板参数

#### `BlockRows / BlockCols / BlockK`

定义 CTA tile。首期固定：

```text
BlockRows = 64
BlockK    = 64
```

`BlockK=64` 恰好等于一个 Q4 group，使 scale lifetime 和 decode 最简单。

`BlockCols` 首期有 64 和 128：

- 64：减少 crossover 附近的空 token 工作和 partial wave；
- 128：提高稳定 prefill 的 X/A reuse 和 compute throughput。

#### `WarpRows / WarpCols`

定义 CTA 内的 warp tile 和 accumulator 数量。不能由 launcher 根据 N/T 动态修改。

#### `PipelineStages`

决定 X 与 raw Q4 producer 的 shared pipeline 深度。首期使用 2。

#### `FragmentPipeline`

serial/ping-pong 会改变 fragment live range 和寄存器数，必须编译期闭合。

#### `QuantCache / ActivationCache`

Q4 payload 和 X 有不同复用性质，因此保留两个 cache 轴。但首期只实例化已有实测组合，
不生成 `Ca/Cg × Ca/Cg` 全组合。

#### `ScaleLoadMode`

在 `K % 128 == 0` 下，每行 group 数为偶数，scale row stride 保持 4B 对齐，因此
`Pair32` 合法；`Scalar16` 与 `Pair32` 会改变 load 指令和 shared layout，属于模板参数。

### 8.4 Kernel variant

同样使用：

```cpp
Q4KernelVariant::Full
Q4KernelVariant::Predicated
```

MMA `Full` 要求：

```text
Rows % BlockRows == 0
Cols % BlockCols == 0
K % BlockK == 0
Kpad == K
```

`Predicated` 处理：

- `Rows=4304` 的最后 16 行；
- 任意最后一个 Cols tile。

在本文合同中 `K % 128 == 0` 且首期 `BlockK=64`，所以 MMA 没有 K edge。K1152 的
不足 `GroupsPerStage=16` 尾 stage 只属于 SIMT lifecycle，不能混入 MMA variant。

planner 在解析最终 plan 时计算 variant。fixed launcher 不得再次决定 full/predicated。

### 8.5 初始候选

```text
MmaR64C64K64
  BlockRows=64
  BlockCols=64
  BlockK=64
  WarpRows=32
  WarpCols=32
  PipelineStages=2
  LaunchBoundsMinBlocks=3
  FragmentPipeline=PingPong
  QuantCache=Ca
  ActivationCache=Ca
  ScaleLoadMode=Scalar16

MmaR64C128K64
  BlockRows=64
  BlockCols=128
  BlockK=64
  WarpRows=64
  WarpCols=32
  PipelineStages=2
  LaunchBoundsMinBlocks=1
  FragmentPipeline=Serial
  QuantCache=Cg
  ActivationCache=Cg
  ScaleLoadMode=Pair32
```

首期 schedule 与当前 BN64/BN128 winner 对齐，但 policy 名不再包含 short、prefill、
Vision 或其他应用场景。

## 9. ScheduleId、KernelVariant 与实例预算

### 9.1 ScheduleId

最终 production 闭合枚举：

```cpp
enum class Q4ScheduleId {
    GemvR4W1Direct,
    GemvR1W8Direct,
    SimtR8C4,
    SimtR8C8,
    MmaR64C64,
    MmaR64C128,
};
```

每个枚举值不可变地绑定一个完整 schedule。最终 production schedule 的完整事实是：

| ScheduleId | 完整 codegen schedule |
|---|---|
| `GemvR4W1Direct` | R4/W1/G16/S1，runtime groups，`Direct`，PackedWord8/Fp16Mantissa，code AsyncVector16 `Ca`，scale SharedPair32，launch-bounds min blocks 1 |
| `GemvR1W8Direct` | R1/W8/G16/S1，StaticGroupsPerRow=80，`Direct`，PackedByte2/ScalarInteger，code SyncVector16 `Ca`，scale Scalar16Shuffle，launch-bounds min blocks 1 |
| `SimtR8C4` | R8/C4/G16/S2，code `Ca`，launch-bounds min blocks 1 |
| `SimtR8C8` | R8/C8/G16/S2，code `Ca`，launch-bounds min blocks 1 |
| `MmaR64C64` | R64/C64/K64，WR32/WC32，S2，fragment PingPong，Q/X `Ca`，scale `Scalar16`，launch-bounds min blocks 3 |
| `MmaR64C128` | R64/C128/K64，WR64/WC32，S2，fragment Serial，Q/X `Cg`，scale `Pair32`，launch-bounds min blocks 1 |

资格阶段还实现并测量过 `GemvR4W1Shared`。它与 `GemvR4W1Direct` 共享除
`ActivationAccess=CtaSharedFullK` 外的全部 schedule 事实，但没有获得任何 current
product route；因此它是应在原子切换时删除的实验实例，不属于最终 production enum。

短枚举名只用于 C++ switch。benchmark CSV、NCU tag、资源记录和 experiment log 必须使用
可唯一复现的 canonical name，例如：

```text
q4.gemv.r4.w1.g16.s1.groups_runtime.x_direct.lane_q4x8.decode_fp16mantissa.code_async_vec16_ca.scale_shared_pair32.lb1
q4.gemv.r1.w8.g16.s1.groups_static80.x_direct.lane_q4x2.decode_scalar_integer.code_sync_vec16_ca.scale_scalar16_shuffle.lb1
q4.simt.r8.c4.g16.s2.code_ca.lb1
q4.mma.r64.c128.k64.wr64.wc32.s2.frag_serial.q_cg.x_cg.scale_pair32.lb1
```

如果其中任一 codegen 轴变化，就新增新的 `Q4ScheduleId` 或版本化 canonical identity，
不能在保留旧身份的情况下静默修改 schedule。名称只编码 kernel 事实，不编码应用。

实际 binary entry 由 schedule 与 variant 共同唯一标识：

```cpp
struct Q4Plan {
    Q4ScheduleId schedule;
    Q4KernelVariant variant;
};
```

合法组合：

```text
GEMV       -> None
SIMT/MMA   -> Full 或 Predicated
```

capability 必须判断完整的 `ScheduleId + KernelVariant`，不存在“schedule 要求 full，同时
plan 又携带 predicated”的双重事实。

### 9.2 实际 binary entry 数量

最终 production：

```text
GEMV      2 schedules x None                    = 2 entries
SIMT      2 column schedules x Full/Predicated = 4 entries
MMA       2 column schedules x Full/Predicated = 4 entries
```

共 10 个 CUDA entry。这个规模是有意的：

- 足够表达真实 lifecycle 和边界；
- 不按 shape 复制；
- 可以逐项检查资源和 SASS；
- 不形成模板笛卡尔积。

### 9.3 新增实例规则

新增 schedule 必须同时提供：

1. 无法由已有实例满足的明确 shape/Cols 问题；
2. candidate timing；
3. 数值 oracle；
4. registers/shared/local/stack；
5. 至少一个代表性 NCU 报告，或说明为何不需要；
6. route 使用方或明确的 measurement-only 生命周期；
7. 对 binary size 和编译时间的影响。

未被 route 使用、也不在当前实验中的实例应删除。

## 10. Capability、Admission 与 Route

### 10.1 Capability

capability 描述一个 `Q4ScheduleId + Q4KernelVariant` 的物理合法性。它拆成两层，避免
把 host pointer 事实伪装成纯 problem predicate。

第一层是几何/lifecycle predicate：

```cpp
bool q4_candidate_is_legal(
    Q4ScheduleId schedule,
    Q4KernelVariant variant,
    Q4ProblemSig problem);
```

它检查：

```text
schedule 的 Rows/K/Cols envelope
variant 是否与 lifecycle 匹配
Full 的全部整除条件
Predicated 的边界能力
None 只用于 GEMV 且 Cols==1
静态 K schedule 与运行时 K 一致
运行时动态 shared envelope
未知 ScheduleId 必须拒绝
```

第二层是 operand contract：

```text
Q4G64 RowSplit / FP16 scale / BF16 X,Y
Kpad、shape、stride 和 contiguous 条件
X/Y/qdata 16B 对齐、scale 4B 对齐
非空指针
```

最大 K 主要约束 `Q4ActivationAccess::CtaSharedFullK`：动态 X shared 字节数与运行时 K
相关，几何 predicate 必须证明该 K 落在 schedule 的已配置 shared envelope；launcher
再按所选 schedule 提供精确 dynamic shared 字节数。`Direct` 实例可以使用不同的 K
envelope。

Capability 回答：

```text
这个已编译 schedule 能否合法执行这个物理问题？
```

它不回答：

```text
这个 shape 是否已注册？
这个 schedule 是否是 winner？
```

### 10.2 Admission

Q4 纯 Linear 使用 exact physical support：

```cpp
struct Q4ProblemSig {
    int32_t rows;
    int32_t k;
    int32_t padded_k;
};

struct Q4SupportSpec {
    Q4ProblemSig problem;
    TokenSet admitted_cols;
};
```

首期 exact admission 与当前产品可达域是：

| `Rows,K,Kpad` | `admitted_cols` | 当前所有者/可达性 |
|---|---|---|
| `1024,5120,5120` | 每个整数 `1..16` | attention input 的 K view，较大 Cols 由 grouped Op 接管 |
| `4096,5120,5120` | 每个整数 `1..16` | GDN input 的 Q/K projection，较大 Cols 由 grouped Op 接管 |
| `6144,5120,5120` | 每个整数 `1..16` | attention input 的 Q view，较大 Cols 由 grouped Op 接管 |
| `34816,5120,5120` | 每个整数 `2..16` | LinearSwiGLU 的 materialized pure-Linear 子问题 |
| `131072,5120,5120` | `{1}` | optimized draft projection |
| `3456,1152,1152` | `4..131072` 且 `Cols % 4 == 0` | Vision QKV |
| `4304,1152,1152` | `4..131072` 且 `Cols % 4 == 0` | Vision FC1 |

这张表是本轮 production route 和“最小验证矩阵”的边界。以下问题只允许通过
measurement-only fixed candidate：

```text
[34816,5120] Cols1
[7168,5120] 旧父矩阵
[17408,5120] / [2048,5120] 等未注册 row views
任意满足 capability 但不在上表中的 aligned shape
future [131072,2048]
```

不满足 `Rows%16/K%128/Kpad==K` 的旧 arbitrary 测试既不是 production support，也不是
新模板的合法 candidate；它们应改为 public rejection 检查或删除。

生产不使用：

- `ShapeFamily`；
- `T1/SmallT/LargeT` enum；
- application role；
- `Rows >= X` 的 family rule；
- arbitrary aligned fallback。

对齐能力不等于产品 admission。任意数值测试通过内部 fixed candidate 入口，不扩大
`linear()` 的产品支持面。

### 10.3 Route

```cpp
struct Q4Route {
    TokenSet cols;
    Q4ScheduleId schedule;
};
```

route catalog 的职责：

```text
exact (Rows,K,Kpad) + Cols
  -> one qualified Q4ScheduleId
  -> one final KernelVariant
  -> one fixed launcher
```

同一个 shape 可以有多个不连续 interval；不能假设一个单调全局阈值。

### 10.4 最终 production route

以下 route 已用新 binary 完成结构 screen、三个独立进程确认、数值 oracle 和代表性
NCU 资格。区间端点都包含在内：

| exact problem | admitted Cols | schedule |
|---|---|---|
| `[1024,5120,5120]` | 1 | `GemvR1W8Direct` |
|  | 2--15 | `SimtR8C4` |
|  | 16 | `SimtR8C8` |
| `[4096,5120,5120]` | 1 | `GemvR1W8Direct` |
|  | 2--4 | `SimtR8C4` |
|  | 5--16 | `SimtR8C8` |
| `[6144,5120,5120]` | 1 | `GemvR1W8Direct` |
|  | 2--7 | `SimtR8C4` |
|  | 8--16 | `SimtR8C8` |
| `[34816,5120,5120]` | 2--4 | `SimtR8C4` |
|  | 5--16 | `SimtR8C8` |
| `[131072,5120,5120]` | 1 | `GemvR4W1Direct` |
| `[3456,1152,1152]`, `Cols%4==0` | 4--36 | `SimtR8C4` |
|  | 40--320 | `MmaR64C64` |
|  | 324--131072 | `MmaR64C128` |
| `[4304,1152,1152]`, `Cols%4==0` | 4 | `SimtR8C4` |
|  | 8 | `SimtR8C8` |
|  | 12 | `SimtR8C4` |
|  | 16--24 | `SimtR8C8` |
|  | 28--320 | `MmaR64C64` |
|  | 324--131072 | `MmaR64C128` |

Vision 范围只表示 4 的倍数，不允许区间内的其他整数。

variant 不构成第二层性能 dispatch。resolver 在 schedule 已确定后使用唯一规则：

```text
GEMV                         -> None
SIMT/MMA 且 Full 几何合法    -> Full
其余合法 SIMT/MMA            -> Predicated
```

即 resolver 不比较 Full/Predicated 的阈值，也不由 caller 指定 variant。资格实验保留
Full，因为它在代表性 full point 上相对 Predicated 有明确收益。

future `[131072,2048]` 仍不进入 admission。transfer screen 表明它可以复用现有族：

```text
Cols1       -> R4/W1 GEMV
Cols8       -> SIMT C8
Cols9..64   -> MMA C64
Cols65+     -> MMA C128
```

未来真正注册时仍需针对实际可达 Cols 完成数值、route 和端点确认，不能直接复制这张
measurement-only screen。

## 11. Planner 与 Fixed Launcher

### 11.1 唯一执行链

Q4 生产执行链：

```text
linear(original semantic inputs)
  -> existing tensor/storage validation
  -> validate Q4 fast capability contract
  -> normalize Q4ProblemSig
  -> resolve_q4_plan(problem, Cols)
  -> execute_q4_fixed(plan)
```

`linear()` 不接受 target/profile/policy。

本轮只原子迁移 Q4。`linear.cpp` 中的新分支必须位于旧
`classify_shape()`、`classify_regime()` 和 `resolve_plan()` 之前：

```cpp
const LinearFormat format = classify_format(w);

if (format == LinearFormat::Q4G64_RowSplit) {
    const Q4Plan plan = resolve_q4_plan(normalize_q4_problem(x, w));
    execute_q4_fixed(plan, x, w, out, stream);
    return;
}

// Q5/Q6/W8/Dense 暂时继续当前 planner。
```

Q4 进入该分支后不得再经过旧 `ShapeFamily`、`LinearRegime`、全局 threshold 或旧
`LinearPolicyId`。本设计不要求在同一工作中迁移 Q5/Q6/W8/Dense。

### 11.2 Fixed launcher

```cpp
void execute_q4_fixed(
    Q4Plan plan,
    const Tensor& x,
    const Weight& w,
    Tensor& y,
    cudaStream_t stream);
```

它使用 exhaustive switch：

```cpp
switch (plan.schedule) {
case Q4ScheduleId::GemvR4W1Direct:
    require(plan.variant == Q4KernelVariant::None);
    launch_gemv_r4_w1_direct(...);
    break;
...
}
```

每个 case 只能：

- 检查该 fixed schedule 的静态前提；
- 使用 plan 已经给出的 variant；
- 计算 grid、block 和动态 shared bytes；
- 启动一个确定的模板实例。

每个 case 不能：

- 再比较 Cols；
- 再选择 BN64/BN128；
- 再选择 C4/C8；
- 再选择 warp-row/split-K；
- fallback 到另一个 kernel。

### 11.3 Measurement-only fixed candidate

benchmark/test 需要内部入口：

```cpp
void launch_q4_candidate(
    Q4ScheduleId schedule,
    Q4KernelVariant variant,
    const Tensor& x,
    const Weight& w,
    Tensor& y,
    cudaStream_t stream);
```

该入口：

- 位于 `src/` 私有头文件；
- 不进入安装 API；
- 不被 target schedule 调用；
- 不修改 production route；
- 对不合法 candidate 直接报错；
- 在 benchmark CSV 中记录精确 ScheduleId、KernelVariant 和 canonical identity。

## 12. 文件与所有权

建议新增：

```text
src/ops/linear/q4/
  q4_rowsplit_storage.cuh
  q4_rowsplit_gemv.cuh
  q4_rowsplit_gemv.cu
  q4_rowsplit_gemm_simt.cuh
  q4_rowsplit_gemm_simt.cu
  q4_rowsplit_gemm_mma.cuh
  q4_rowsplit_gemm_mma.cu
  q4_rowsplit_plan.h
  q4_rowsplit_plan.cpp
  q4_rowsplit_launch.h
```

实现可以在首阶段复用现有通用 Small-T/MMA 内部 atom，但最终 Q4 的 production
ScheduleId、route 和 launcher 必须位于 `q4/` 下并使用中立命名。

不要在未验证前直接删除共享文件：

- 当前 generic Small-T 仍服务 Q5/Q6/W8；
- 当前 low-bit MMA 仍服务 Q5/Q6；
- grouped input 和 folded SwiGLU 直接依赖 MMA header 中的配置和 helper。

Q4 原子切换后：

- generic Small-T launcher 不再接受 Q4 production route；
- generic MMA launcher 不再接受纯 Q4 production route；
- Q4 application-named plain Linear policy 被删除；
- Q5/Q6/W8 行为保持不变；
- grouped/fused Q4 保持其各自 Op-owned lifecycle，直到单独审查。

旧实现必须按以下手术表处理：

| 旧内容 | 原子切换操作 |
|---|---|
| `MlpGateUp34816Q4RowsplitGemv` | 删除 policy 和普通 Linear launch |
| `AttnInQKV7168Q4RowsplitGemv` | 删除 policy；旧父矩阵不进入新 admission |
| `GdnInQK4096Q4RowsplitGemv` | 删除 policy |
| `LmHeadQ4RowsplitGemv` | 删除 policy |
| `linear_rowsplit_gemv_gdn_in_qk_4096.*` | 文件只包含普通 Q4，可整文件及 CMake 项删除 |
| `linear_rowsplit_gemv_attn_in_7168.*` | 只删除 Q4 kernel/launcher，保留 Q5 |
| `linear_rowsplit_gemv_lm_head.*` | 只删除 Q4 kernel/launcher，保留 Q6 |
| `linear_rowsplit_gemv_mlp_gate_up_34816.*` | 只删除普通 Q4 Linear；保留 fused SwiGLU kernel |
| `linear_rowsplit_gemm_smallt.*` | 保留 Q5/Q6/W8；删除旧 Q4 launcher case 和不再使用的 `Q4Smallt` |
| `linear_rowsplit_gemm_mma.*` | 保留 Q5/Q6；删除纯 Linear Q4 launcher case |
| `Q4Codec` | 保留，grouped/fused Q4 仍使用 |
| 旧 MMA `GemmCfg`、swizzle、async helper | 本轮保留，grouped/fused 仍直接依赖 |

不能因为纯 Linear 有了独立 Q4 backend，就顺手移动或删除 grouped/fused 仍在使用的旧
MMA 基础设施。

## 13. 数值合同

### 13.1 GEMV/SIMT

```text
activation input boundary : BF16
Q4 dequant                : FP32
multiply/accumulate       : FP32 FMA
final output              : BF16
```

GEMV warp-row、GEMV split-K 和 SIMT 可能因 FP32 求和结合顺序不同而产生不同 BF16
输出。bitwise equality 不是跨 schedule 合同。

### 13.2 MMA

```text
activation input boundary : BF16
Q4 dequant                : BF16 shared operand
multiply                  : BF16 Tensor Core
accumulate                : FP32
final output              : BF16
```

它与 GEMV/SIMT 不要求逐 word 相同。

### 13.3 Oracle

独立 oracle：

```text
BF16-rounded X
test-side RowSplit Q4 dequant to FP32
FP64 CPU W @ X
```

判断：

- GEMV/SIMT 使用 `Tolerance::linear_bf16()`；
- MMA 使用 `Tolerance::linear_tc()`，当前 normwise `rel_l2 <= 4e-3`；
- R4/W1 必须在真实 K5120 上覆盖五个完整 group tile；小 K partial tile 不能替代该
  production oracle；
- route seam 两侧都必须单独验证；
- K1152 checked group stage、Rows4304 edge、Cols edge 必须覆盖。

## 14. 性能与 Roofline 验收

“达到 roofline”按 lifecycle 分别解释，不能使用一个统一百分比。

### 14.1 GEMV

GEMV 的 useful bytes 可以由 code、scale、X 和 Y 合同计算，但实际 X cache 命中、
transaction 放大和 DRAM bytes 必须来自 NCU。cold copy ceiling只提供同场硬件上下文，
不能单独证明 kernel 达到物理 roofline。

验收：

1. 相对旧 winner 不稳定回退超过 1%；
2. 相对全部合法 candidate 选择最低冷缓存延迟；
3. 大 N 饱和代表通过 NCU 判断 DRAM、SM、issue 或 occupancy 限制；
4. physical DRAM bytes 与 RowSplit code/scale 读取相符；
5. 没有 local/stack spill。

小 N 或不足一个完整 wave 的问题以最低延迟和 wave 几何为准，不要求统一带宽百分比。

### 14.2 SIMT

SIMT 的权重逻辑流量至少为：

```text
(code bytes + scale bytes) * ceil(Cols / ColsPerTile)
```

但它并不总是带宽受限。历史 TT8 已出现高 SM SOL、低 DRAM SOL 的 compute/issue-limited
状态，因此不能给所有 SIMT 实例套用 copy-ceiling 目标。

验收：

1. 相对旧 winner 不稳定回退超过 1%；
2. 相对 C4/C8 和合法 GEMV/MMA candidate 选择最低冷缓存延迟；
3. 用 NCU 判断最终代表是 DRAM、SM、issue、latency 还是 occupancy bound；
4. 物理 bytes 只采用 NCU 结果，不从 useful bytes 推断；
5. 没有 local/stack spill。

### 14.3 MMA

当 arithmetic intensity 位于 ridge 右侧且 wave 充分时：

```text
useful FLOP   = 2 * Rows * K * Cols
executed FLOP = 2 * ceil(Rows/BM)*BM * K * ceil(Cols/BN)*BN
```

验收同时记录 useful/executed TFLOP/s，避免 edge tile 掩盖真实效率。

大而稳定的 full-wave 问题目标：

- 至少约 85～90% 同场 BF16 MMA probe；
- NCU SM SOL 与 tensor instruction accounting 一致；
- 没有 spill；
- 相对现有 winner 不回退超过 1%。

partial-wave 和 launch-floor 问题按最佳候选延迟判断，不要求虚假的峰值百分比。

### 14.4 Candidate 决策规则

- 三个独立进程；
- 每进程至少 8 次 warmup、40 次 cold sample；
- 每次 cold sample 前 256 MiB L2 flush；
- 使用三个进程 median 的 median；
- 小于 1% 的差异默认 practical tie；
- 1%～3% 的差异需要重复确认；
- route 优先选择更简单且资源稳定的 policy，除非复杂 policy 有稳定收益；
- 任何 route 变化都必须记录 exact ScheduleId、KernelVariant 和 canonical identity。

## 15. 最小实验矩阵

### 15.1 实验工具前置门

该门已完成。当前 `ninfer_linear_op_bench` 支持：

1. 当前源码 Release build；
2. `Rows/K/Cols` 数值输入；
3. fixed `Q4ScheduleId + Q4KernelVariant` candidate；
4. canonical schedule、variant、build type、CUDA 和 GPU CSV 字段；
5. GEMV/SIMT useful-byte 与 MMA executed-FLOP 的分离统计；
6. `--help`、smoke CSV 和 candidate legality 验证。

资格阶段曾保留独立 legacy measurement selector，用于完成新旧 matched 对照；原子
cutover 后这些 selector 与旧实现一起删除，当前工具只保留 production `auto` 和六个
fixed Q4 schedule。

资格原始结果位于 `profiles/bench/q4-linear-qualification-20260716/`，NCU 报告位于
`profiles/ncu/q4-linear-template-20260716/`；结论持续记录在 experiment log。

### 15.2 模板抽取等价矩阵

这一矩阵只回答：

```text
把现有 lifecycle 放进新模板后，是否无意改变了代码生成或性能？
```

最小 matched 点：

| lifecycle | matched geometry |
|---|---|
| warp-row GEMV | `[34816,5120]` Cols1，旧 application kernel vs 新 fixed candidate |
| split-K GEMV | `[4096,5120]` Cols1，旧 split-K8 vs 新 fixed candidate |
| SIMT C4 | `[6144,5120]` Cols4 |
| SIMT C8 | `[34816,5120]` Cols8 |
| MMA C64 | 一个合法的旧 BN64 exact point |
| MMA C128 | `[3456,1152]` 的一个合法 Full 或 Predicated point |

如果算法、group edge 或累加顺序未改变，要求：

- BF16 word-exact；
- registers/shared/local/stack 不恶化；
- hot-loop SASS 保持相同结构；
- 三进程冷缓存 timing 不稳定回退超过 1%。

如果主动改变了 K edge 或累加顺序，就不要求 word-exact，但必须通过独立 FP64 oracle，
并以 candidate timing/NCU 重新资格化。

### 15.3 Candidate 数值矩阵

数值矩阵不等同于 timing 全扫。最小 forced candidate 点：

| 问题 | Cols |
|---|---|
| `[1024,5120]` | 1、4、5、8、16 |
| `[4096,5120]` | 1、4、5、8、16 |
| `[6144,5120]` | 1、4、5、8、16 |
| `[34816,5120]` | 1、2、4、5、8、9、16 |
| `[131072,5120]` | 1 |
| `[3456,1152]` | 4、24、28、128、129 |
| `[4304,1152]` | 4、12、13、15、16、128、129 |
| future `[131072,2048]` | 1、8、9、64、65 |

只执行该点上物理合法、正在比较的 candidate。`Full` 只能在其静态整除条件满足时执行；
其他点只运行 `Predicated`。强制一个非法 `Full` 必须在 host 侧拒绝，不能依赖 device
mask。

现有测试中：

- `[70,130]` 不满足 capability，改为 public rejection 或删除；
- `[128,128]`、`[256,256]`、`[320,384]` 满足 capability 但未 admission，通过 fixed
  candidate 测试；
- `[7168,5120]`、`[17408,5120]`、`[2048,5120]` 等不进入 production admission；
- Q4 tolerance 不能再使用 `Cols > 16`，必须依据实际 ScheduleId：
  GEMV/SIMT 使用 `linear_bf16`，MMA 使用 `linear_tc`。

### 15.4 Candidate timing 矩阵

第一轮只做结构 screen：

```text
GEMV：
  [4096,5120]   Cols1 : split-K8 / split-K4 / SIMT C4
  [34816,5120]  Cols1 : warp-row direct/shared / SIMT C4
  [131072,5120] Cols1 : warp-row / SIMT C4

SIMT C4/C8：
  Cols4/5
  [34816,5120] Cols8/9
  [3456,1152] Cols24..28
  [4304,1152] Cols12..16

MMA C64/C128：
  已知第一 crossover 附近
  Cols63/64/65
  Cols127/128/129

transfer-only：
  [131072,2048] Cols1/8/9/64/65
```

screen 使用 5 warmup、20 cold sample。只有：

- winner 反转点；
- 差异小于 3%；
- full/checked 或 wave 变化点；

进入三进程 8 warmup/40 cold confirmation。

Vision 高 Cols 只需确认最终 N128 policy 的结构稳定性：

```text
[3456,1152] 32768
[4304,1152] 32768
以及产品上界 131072
```

如果 32768 到 131072 的相对效率和 wave 行为稳定，不重新运行每个 tile band。

### 15.5 KernelVariant 与资源检查

对每个最终保留 entry 记录：

- registers/thread；
- static/dynamic shared；
- local/stack；
- launch-bounds 参数；
- full/checked binary symbol；
- binary size 增量。

比较规则：

- `Full` 与 `Predicated` 只在两者都合法的 full point 比较；
- edge point 只运行 `Predicated`；
- 若 `Full` 在代表性 full point没有稳定收益，可以删除它以减少实例；
- 任何 spill 默认阻断该实例进入 production。

### 15.6 NCU

产品切换前只对最终保留的不同物理 topology 采集：

1. split-K GEMV；
2. warp-row GEMV，如果它进入 production route；
3. SIMT C4；
4. SIMT C8；
5. MMA C64；
6. MMA C128 full-wave。

Rows4304 edge 或 K1152 checked stage 只有在 timing/resource 无法解释回退时才额外采集。
NCU profiler duration只用于诊断，不替代无 profiler 的 CUDA event timing。

### 15.7 Cutover 后 auto-route 矩阵

最终未提交 cutover change set 必须检查：

1. 每个 exact admission 的首点、末点和每个 route seam；
2. `auto` 的 plan identity 等于预期 `Q4ScheduleId + Q4KernelVariant`；
3. `auto` 输出与该 fixed candidate BF16 word-exact；
4. `auto` timing 与 fixed winner 的差异不超过测量噪声；
5. Vision 上界 131072 至少完成 launch、同步、输出 finite 和资源行为 smoke；
6. capability-valid 但未 admission 的问题由 public `linear()` 拒绝；
7. capability-invalid 的问题在进入 resolver 前拒绝；
8. future `[131072,2048]` 仍只能由 measurement-only candidate 启动。

## 16. 实施与原子切换

### 16.1 阶段 A：设计与 dormant candidate

状态：已完成并提交。

1. 落地本文；
2. 新增 `q4/` 私有 ScheduleId、KernelVariant 和模板；
3. 新增 measurement-only fixed candidate；
4. production `linear()` 保持原行为；
5. build、数值和 candidate timing 通过。

该阶段允许新旧 Q4 kernel 同时存在，但产品 dispatch 仍只有旧路径。

### 16.2 阶段 B：资格测量

状态：已完成；最终 route 见第 10.4 节，完整结果见 experiment log。

1. 重新构建 benchmark，禁止使用陈旧 build binary；
2. 完成第 15 节矩阵；
3. 将每次实验追加到 experiment log；
4. 确定最终 support envelope、route intervals 和 KernelVariant；
5. 确定需要保留的 ScheduleId；
6. 删除未使用实验实例。

### 16.3 阶段 C：准备未提交的完整 cutover change set

状态：已完成。

在一个未提交的完整 worktree change set 中同时完成：

1. Q4 在 `linear()` 中进入新 exact resolver；
2. production route 全部改为新 Q4 ScheduleId/KernelVariant；
3. 删除旧 Q4 application-named plain Linear policies；
4. 删除旧 Q4 plain launch 分支；
5. 按第 12 节手术表删除被替代源码、局部实现和 CMake 项；
6. generic Q5/Q6/W8 dispatch 保持不变；
7. capability-valid 但未 admission 的 Q4 测试改用 internal fixed candidate；
8. capability-invalid 的旧 arbitrary Q4 测试改为 rejection 或删除；
9. exact Q4 plan test 覆盖 support、route seam 和 rejection。

该 change set 在完成下一阶段验收前不提交。不能产生“部分 shape 走新架构、部分 shape
仍依赖旧 Q4 fallback”的已提交产品中间状态。

### 16.4 阶段 D：验收后一次提交

状态：已完成；阶段 C 的完整 change set 作为一个原子 cutover 提交落地。

对阶段 C 的未提交完整 change set 执行：

1. build 和完整 Linear test；
2. Q4 exact numerical suites；
3. `ninfer_linear_op_bench` auto 与 fixed winner 对照；
4. 必要 NCU；
5. 当前 27B Text/Vision/MTP 相关集成；
6. `ninfer_bench` 端到端回归；
7. 更新 experiment log 和本文状态；
8. 复查旧 Q4 plain symbols 已消失；
9. 一次提交完整 cutover。

阶段 D 不再进行实现清理；任何遗漏都回到未提交 change set 修正并重新验收。

### 16.5 实际 cutover 结果

最终实现与本文边界一致：

- `q4_rowsplit_plan.{h,cpp}` 唯一拥有 7 个 support、21 条 route 和 variant 派生；
- `linear()` 只接收原语义输入，在 Q4 validation 后直接进入该 planner；
- production fixed launch 与 measurement-only candidate 共用闭合实例，但 production
  不调用或暴露 benchmark policy；
- 最终 binary 只保留 6 个 schedule、10 个 schedule/variant entry；
- 未获 route 的 R4/W1 shared 实例和全部 legacy benchmark selector 已删除；
- 旧 application-named plain Q4 GEMV、旧 Small-T Q4 case、旧 generic MMA Q4 case 和
  旧 Q4 policy 已删除；
- Q5/Q6/W8、grouped input、folded SwiGLU、Q5 residual 和共享 MMA header 基座保持原
  所有权；
- target 中从未可达的 `Phase::Decode` 分支及其 Q4/Q5 父矩阵执行假象一并删除，packed
  父对象仍作为 row-view storage 存在。

验收包括独立 fixed-candidate 数值 oracle、完整 host plan 闭包、34 个 public/fixed
BF16 word-exact route 点、完整 Linear 回归、真实 artifact Text/Vision/MTP 测试以及
matched old/new `ninfer_bench`。详细命令和结果见 experiment log 的 E056。

## 17. 添加相似 shape 的工作流

以后增加一个满足 Q4 capability 的相似 shape：

```text
确认 RowSplit/Q4/K128/对齐事实
  -> 在 measurement-only benchmark 中加入 exact Rows/K
  -> 枚举已有 Q4ScheduleId/KernelVariant
  -> 数值验证合法候选
  -> 测量 Cols、wave、full/edge 和 crossover
  -> 添加 exact support 与 route
```

默认不写新 kernel，不模板化 exact shape，也不使用“最近 shape”继承。

只有所有已有 `Q4ScheduleId` 都明显不够好时，才：

```text
在现有三个模板族之一增加一个命名 schedule
  -> 显式实例化
  -> 完成同样资格流程
```

如果问题需要不同的：

- producer lifetime；
- warp ownership；
- synchronization；
- accumulator topology；
- output mapping；

它就不是现有 lifecycle 的新参数，应重新审查是否属于新的 semantic Op 或第四个 kernel
族。

## 18. 复杂度控制红线

必须长期保持：

1. 模板参数只描述真实 codegen 差异；
2. production 只显式实例化 route 使用的 schedule；
3. exact Rows/K/Cols 默认运行时化；
4. `Threads` 等派生量不成为独立参数；
5. 使用 enum/policy type，避免不可读 bool 集合；
6. Q4 格式、dtype、scale 和 plain store 不模板化；
7. 三个 lifecycle 不合并成 producer/finalizer DSL；
8. target 和调用者看不到 ScheduleId/KernelVariant；
9. launcher 不执行二次 dispatch；
10. 新抽象不能接受超过 1% 的稳定回退；
11. 新实例必须有 route 或当前实验用途；
12. experiment binary 与源码必须可复现，不能引用陈旧 build 结果。

## 19. 完成定义

本工作完成需要同时满足：

- 三个 Q4 模板族存在且使用中立命名；
- 当前主要 Q4 纯 Linear 问题均通过 exact admission；
- Q4 production resolver 没有 ShapeFamily、场景名或全局 threshold；
- production Q4 不依赖 generic quantized fallback；
- 新相似 shape 可以只通过已有 ScheduleId/KernelVariant + route 扩展；
- 数值 oracle 覆盖三个 lifecycle 和所有 route seam；
- 饱和 GEMV/MMA 有对应物理 roofline 证据，SIMT 有明确的 NCU 限制来源；
- partial-wave 问题有 best-candidate 延迟证据；
- 没有 local/stack spill；
- 当前 27B Text/Vision/MTP 产品路径通过；
- experiment log 记录全部命令、结果、失败候选和资格限制；
- 被替代的 application-named Q4 plain kernels 已删除；
- Q5/Q6/W8 和 fused/grouped 行为未被本次 Q4 纯 Linear 切换意外改变。
