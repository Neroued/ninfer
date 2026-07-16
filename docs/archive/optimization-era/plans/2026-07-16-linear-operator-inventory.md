# Linear 算子清单（迁移前历史快照）

> 状态：已归档。本文记录逐算子迁移开始时的源码快照，不描述当前实现。

本文统计当前产品基线中已经存在的 Linear 算子、精确执行问题、执行策略和 CUDA kernel。
它与
[`2026-07-16-linear-kernel-architecture-refactor.md`](../../../plans/2026-07-16-linear-kernel-architecture-refactor.md)
分工不同：架构文档描述未来应当采用的所有权和调度边界；本文只回答当前有什么、哪些路径
真实可达、现有证据是否充分，以及后续应逐项决定保留、重写、扩面、泛化还是删除。

## 1. 统计层级与术语

本文严格区分四个层级：

1. **语义算子（semantic Op）**：target 可以直接调用的完整数学合同，例如
   `linear_add`；
2. **精确执行问题**：由权重格式、逻辑 `N/K`、补齐后的 `K`、T 取值域和输出拓扑共同
   确定的执行域；
3. **执行策略**：针对一个精确执行问题和 T 范围选定的一套完整执行方式；
4. **内核实现变体**：某个执行策略使用的具体 CUDA 实现或编译期配置；只有明确指模板
   参数实例时才称为“内核特化实例”。

这四层不能混为一谈。

- 一个 wrapper 即使内部调用两次 `linear()`，仍然可能是一个独立语义算子，但不是一个
  kernel。
- 一个 `LinearPolicyId` 也不一定唯一确定 kernel，因为当前内核启动层仍可能继续按
  shape、T、full/edge 等条件二次选择。

本文统一使用以下译法：

- grouped：共享输入的多投影成组执行，简称“成组执行”；不使用容易与 G32/G64 量化组
  混淆的“分组执行”；
- materialized：基础 Linear 先显式写出中间张量，再执行后续算子；
- fused：融合执行；正文会进一步区分融合语义算子、融合内核和融合写回；
- route：路由项或一次调用的实际执行路径；
- mainloop：主循环；
- finalizer：最终写回器，即 reduction 完成后的最终变换与写回逻辑；
- exact problem：精确执行问题。

## 2. 总体统计

当前源码包含：

- 7 个 target 可调用的 Linear-family 语义算子；
- base `linear()` 接受 6 种权重格式：
  连续 BF16/FP32，以及 RowSplit Q4/Q5/Q6/W8；
- 10 个仍服务 Q5/Q6/W8/dense 的 legacy base `LinearPolicyId`，以及独立的 Q4
  `Q4ScheduleId + Q4KernelVariant` 精确计划；
- 12 个 `ShapeFamily` 值，其中包含 `Generic`；
- 3 个全局 `LinearRegime`；
- 24 个产品真实可达的基础投影问题，其中包含融合或成组执行算子内部可能执行的投影；
- 2 个精确 `LinearAdd` 问题；
- 1 个 `LinearSwiGLU` 问题；
- 1 个产品可达的 `LinearPair` 问题；
- 2 个共享输入多投影问题；
- 1 个独立 dense GDN control 问题；
- 一批为死分支编译的专用 kernel；
- 主要由测试和测量使用的任意 shape dense/Q5/Q6/W8 通用执行面，以及不进入 production
  admission 的 Q4 fixed-candidate 测量面。

27B Text 拓扑包含 16 个 Full Attention 层、48 个 GDN 层和 64 个 MLP。一次完整
`run_layers()` 会执行 352 次 Linear-family 语义调用，对应 496 次数学上的 `W @ X`
投影：

| 区域 | 语义调用数 | 数学投影数 |
|---|---:|---:|
| 16 个 Full Attention mixer | 32 | 80 |
| 48 个 GDN mixer | 192 | 288 |
| 64 个 MLP | 128 | 128 |
| **合计** | **352** | **496** |

最终 target head 还会增加 1 次语义调用和 1 次投影。每次 Vision encode 包含 111 次普通
`linear()`。MTP 的总调用数较少，但其 base Linear、LinearPair 和 head 路由对 T 非常
敏感。

对于一次 Text verification：

- `T=1` 时，当前各 wrapper 总共会进入 public base `linear()` 209 次；
- `T=2..6` 时，`LinearAdd` 和 `LinearSwiGLU` 的显式中间张量路径会把数量提高到
  401 次。

因此，T1 和 Small-T 的调度开销、launch 数量和 kernel 选择是产品一级问题，不只是
microkernel 的局部细节。

## 3. 当前语义算子

### 3.1 完整清单

| ID | 语义算子 | 精确语义合同 | 当前执行形式 | 产品状态 |
|---|---|---|---|---|
| S1 | `linear` | 一个 BF16 `X[K,T]`、一个 dense 或 RowSplit `W[N,K]`，输出 BF16 `Y[N,T]` | 按 weight format 分发；Q4/Q5/Q6 使用独立 exact plan，W8/dense 暂留 compatibility 路径 | Text、MTP、Vision 全部可达 |
| S2 | `linear_pair` | 同一输入上的两个同 shape W8 投影 | `T>16` W8 dual MMA；其他情况调用两次 public `linear()` | W8 `[1024,5120] x2` 可达 |
| S3 | `attn_input_proj` | 从 `[5120,T]` 输入计算固定的 Q/gate/K/V 四个投影 | `T<=16` 调用四次 public `linear()`；`T>16` 执行两次成组 MMA | 16 个 Full Attention 层可达 |
| S4 | `gdn_input_proj` | 固定 Q4 Q/K 投影和 Q5 V 投影，并拼接为一个输出 | `T<=16` 两次 public `linear()` 加两次 D2D copy；`T>16` 一次 mixed-format 成组 MMA | 48 个 GDN 层可达 |
| S5 | `linear_add` | 固定 Q5 投影后原地更新 BF16 residual | T1 精确融合；T2..24 显式中间张量加 residual；T25+ residual MMA | 两个问题，共调用 128 次/层遍历 |
| S6 | `linear_swiglu` | 固定 Q4 gate/up 投影后执行 SwiGLU | T1 精确融合；T2..16 显式中间张量加 `silu_mul`；T17+ 将 SwiGLU 折入 MMA 写回 | 一个问题，共调用 64 次/层遍历 |
| S7 | `gdn_gating_proj` | 两个 dense BF16 `[48,5120]` 投影，再生成 FP32 GDN gate | 精确 T1；Small-T partial/reduce；较大 T 使用 cooperative 或 direct dense MMA | 48 个 GDN 层可达 |

当前源码所有权分布为：

- `src/ops/linear/linear.cpp`：S1、S2；
- `src/ops/wrapper/input_proj.cpp`：S3、S4；
- `src/ops/wrapper/linear_add.cpp`：S5；
- `src/ops/wrapper/linear_swiglu.cpp`：S6；
- `src/ops/wrapper/gdn_gating_proj.cpp`：S7。

### 3.2 `linear`、`attn_input_proj` 和 `gdn_input_proj` 的区别

三者底层都包含矩阵投影，但语义和输出拓扑不同。

| 项目 | `linear` | `attn_input_proj` | `gdn_input_proj` |
|---|---|---|---|
| 数学语义 | 一个 `Y=BF16(WX)` | 同一个 X 上的四个独立投影：Q、gate、K、V | 同一个 X 上的两个独立投影：QK、V；各自 BF16 舍入后按行拼接 |
| 权重数量 | 1 | 4 | 2 |
| 输出数量 | 1 个连续矩阵 | 4 个彼此独立的连续矩阵 | 1 个连续 `[10240,T]` 矩阵，内部有两个固定行区间 |
| 固定 shape | 接口允许多种 N/K | X `[5120,T]`；Q/gate `[6144,T]`；K/V `[1024,T]` | X `[5120,T]`；QK `[4096,T]`；V `[6144,T]` |
| 权重格式 | BF16/FP32/Q4/Q5/Q6/W8 | Q、K 为 Q4；gate、V 为 Q5 | QK 为 Q4；V 为 Q5 |
| Small-T 当前实现 | 一次 base Linear | 四次 base Linear，直接写四个输出 | 两次 base Linear写临时矩阵，再执行两次二维复制完成拼接 |
| Large-T 当前实现 | 一次低比特或 W8 MMA | 两次成组 MMA：一次 Q4 Q/K，一次 Q5 gate/V | 一次 mixed Q4/Q5 成组 MMA，直接写最终拼接输出 |
| 当前 workspace | 实际为零 | wrapper 自身不分配 | `T<=16` 需要 `(4096+6144)*T*2 = 20480*T` 字节 |

#### 纯 `linear`

`linear` 只表达一件事：

```text
Y[N,T] = BF16(W[N,K] @ X[K,T])
```

它不知道输出在模型中是 Q、K、V、gate 还是普通 MLP 投影，也不知道调用者之后是否会
拼接、加 residual 或进行 attention。它负责一个投影的输入检查、精确执行问题识别和
kernel 选择。

#### `attn_input_proj`

`attn_input_proj` 表达的是一个固定的四投影语义：

```text
q    = linear(x, q_weight)
gate = linear(x, gate_weight)
k    = linear(x, k_weight)
v    = linear(x, v_weight)
```

四个输出都是独立连续矩阵，并且每个投影都保留 `linear()` 的 BF16 输出舍入边界。

当前实现中：

- `T<=16` 时顺序调用四次 public `linear()`；
- `T>16` 时不再执行四个独立 kernel，而是：
  - 用一次成组 MMA 同时处理 Q4 的 Q/K jobs；
  - 用一次成组 MMA 同时处理 Q5 的 gate/V jobs。

性能动机不是改变数学语义，而是把共享同一 X 的多个投影视为一个联合执行问题：

- 减少 kernel launch 数；
- 让多个不同 N 的小投影共同填充 CTA wave；
- 在一个成组 grid 中统一管理 job mapping 和输出所有权。

当前成组 kernel 仍然是一 CTA 对应一个 projection tile，每个 CTA 独立 staging 自己使用的
X tile；“成组”本身不等于已经消除了不同投影之间的 X 重读。是否进一步共享 X staging
属于以后需要单独测量的 kernel 设计问题。

#### `gdn_input_proj`

`gdn_input_proj` 表达的是固定的双投影加拼接：

```text
qk  = linear(x, qk_weight)   # [4096,T]
v   = linear(x, v_weight)    # [6144,T]
qkv = concat_rows(qk, v)     # [10240,T]
```

QK 和 V 仍是两个独立投影，并分别保留 BF16 舍入边界；拼接后的行布局属于这个 Op 的
语义合同。

当前实现中：

- `T<=16` 时：
  - 分配 `[4096,T]` 和 `[6144,T]` 两个连续临时矩阵；
  - 调用两次 public `linear()`；
  - 用两次 `cudaMemcpy2DAsync` 将结果拼入 `[10240,T]`；
- `T>16` 时：
  - 使用一个 mixed Q4/Q5 成组 MMA；
  - kernel 通过 output leading dimension 和 row offset 直接写入最终 qkv；
  - 不再需要临时矩阵和 D2D copy。

#### 为什么它们应当是独立语义算子

`attn_input_proj` 和 `gdn_input_proj` 不是 target caller 对多次 `linear()` 的简单代码缩写。

它们各自拥有闭合合同：

- 固定的 weight 数量、格式和 shape；
- 固定的输出数量或拼接布局；
- 明确的 BF16 舍入边界；
- 完整的 alias 和 workspace 约束；
- 自己的成组执行/显式中间张量候选及 crossover。

联合执行的胜者不等于各个 base Linear 最优策略的简单相加。是否使用成组 kernel
取决于 job 数、N 分布、CTA wave、X staging、activation 重读、workspace 和 launch
数量，因此应由各自 Op 内部独立规划，不能由 target schedule 手写阈值。

在未来架构中，即使 Small-T 的最佳策略仍是显式中间张量组合，planner 也应一次性固定
内部各个 base 执行策略；executor 不应再次调用 public auto-dispatch `linear()`，避免二次调度和
base 路由变化静默改变成组执行算子。

#### 它们不是通用 epilogue

epilogue 或最终写回器发生在一个 contraction 完成 reduction 之后，例如：

- 普通 Store；
- residual add；
- 对 accumulator 做终端激活后写回。

`attn_input_proj` 和 `gdn_input_proj` 则会在进入 K-loop 之前决定：

- 要执行几个 contraction；
- 使用哪些 weight；
- CTA 拥有哪些 jobs；
- X 的 shared-memory 生命周期；
- 输出写到哪些矩阵或行区间。

因此它们属于多问题主循环/输出拓扑策略，而不是 base Linear 的可插拔
epilogue。可以复用 codec、MMA atom、staging 和 typed job-map helper，但不应向 public
`linear()` 增加运行时成组 job descriptor 或 epilogue 参数。

## 4. 当前 base `linear()` 调度

当前 planner key 为：

```text
(LinearFormat, ShapeFamily, LinearRegime)
```

全局 `LinearRegime` 分类为：

```text
T <= 1       -> T1
2 <= T <=16 -> SmallT
T > 16       -> LargeT
```

这个粗粒度 plan 并没有唯一决定最终 kernel。多个内核启动层仍会继续检查 shape 或 T。

### 4.1 Dense BF16 和 FP32

| T/问题范围 | 当前执行策略 | 内核启动层内部选择 |
|---|---|---|
| `T=1`，任意合法 N/K | `GenericDenseGemv` | 每个输出行一个 256-thread reduction block |
| `T>1` 且 `N<=128 && T<=16` | `GenericDenseGemm` | 每个 `(row,column)` 一个 GEMV-like block |
| 其他 `T>1` | `GenericDenseGemm` | 16x16 标量输出 tile，每个 thread 遍历全部 K |

这些 kernel 是正确性/参考实现，不是 tensor-core 生产 GEMM，也不能证明任意 dense
shape 已获得高性能支持。

### 4.2 Q4 RowSplit 纯 Linear

Q4 已从旧 `ShapeFamily + LinearRegime + generic fallback` 链原子切换到
`src/ops/linear/q4/` 私有 backend：

```text
linear(original operands)
  -> Q4 metadata/alignment validation
  -> exact admission
  -> support-local route
  -> one Q4ScheduleId + Q4KernelVariant
  -> one fixed launch
```

当前只登记七个物理问题：

| `Rows,K,Kpad` | admitted Cols |
|---|---|
| `1024,5120,5120` | 每个整数 1..16 |
| `4096,5120,5120` | 每个整数 1..16 |
| `6144,5120,5120` | 每个整数 1..16 |
| `34816,5120,5120` | 每个整数 2..16 |
| `131072,5120,5120` | 仅 1 |
| `3456,1152,1152` | 4..131072 且 Cols 为 4 的倍数 |
| `4304,1152,1152` | 4..131072 且 Cols 为 4 的倍数 |

生产只包含六个物理 schedule：

```text
GemvR1W8Direct
GemvR4W1Direct
SimtR8C4
SimtR8C8
MmaR64C64
MmaR64C128
```

它们来自三个中立模板族：`q4_rowsplit_gemv_kernel`、
`q4_rowsplit_gemm_simt_kernel` 和 `q4_rowsplit_gemm_mma_kernel`。exact route、variant
规则、模板轴和资格证据见
[Q4 Linear 模板设计](2026-07-16-q4-linear-kernel-template-design.md)。

Q4 不再进入旧通用 Small-T/MMA launcher，也没有 arbitrary aligned fallback。格式和
对齐满足只表示某个 fixed candidate 可能物理合法，不表示 public `linear()` 已登记该
shape。

### 4.3 Q5/Q6 RowSplit

Q5 和 Q6 已从旧 planner 与共享 kernel 模板中拆出。二者分别拥有 storage/decode、
schedule capability、exact admission、route table 和 fixed launcher。

Q5 当前 pure Linear 路由为：

| 精确问题 | token 集合 | schedule |
|---|---|---|
| `[1024,5120,5120]` | T1..4 / T5..16 | SIMT C4 / SIMT C8 |
| `[6144,5120,5120]` | T1 / T2..6 / T7..24 / T25..64 / T65+ | GEMV / split-4 / SIMT C8 / MMA C64 / MMA C128 |
| `[5120,6144,6144]` | T2..6 / T7..24 | split-2 / SIMT C8 |
| `[5120,17408,17408]` | T2..6 / T7..24 | split-2 / SIMT C8 |
| `[1152,1152,1152]`，`T%4==0` | T4 / T8..56 / T60+ | SIMT C4 / SIMT C8 / MMA C128 |
| `[1152,4304,4352]`，`T%4==0` | T4 / T8..84 / T88+ | SIMT C4 / SIMT C8 / MMA C128 |

Q6 当前支持 `[248320,5120,5120] T1..6` 和
`[1152,1536,1536] T4..131072, T%4==0`。前者使用独立 Q6 SIMT C4；后者使用
Q6 SIMT C4、MMA C64 和 MMA C128 的 measured exact intervals。Q6 不引用 Q5 的
codec、kernel body、schedule 类型或 route table。

### 4.4 W8G32 RowSplit

| 问题范围 | 当前执行策略 | 内核启动层内部选择 |
|---|---|---|
| `T<=16` | `RowsplitLowbitGemmSmallt` | 通用 W8 TT4 或 TT8 |
| `T>16`、`K%8==0`、`Kpad%256==0` | `RowsplitW8G32GemmMma` | `N<=1024` small-M 或通用配置，再选完整/边界 tile |
| 其他 T | `RowsplitLowbitGemmSmallt` | 通用 Small-T 兜底路径 |

因此，当前仍不能完整确定 kernel 的只剩 W8 compatibility plan；Q4/Q5/Q6 plan 已经
解析到 fixed schedule 和完整/边界 variant。

## 5. 已有 kernel 家族

本节按算法家族统计，不把每个模板实例都当作独立 kernel。

### 5.1 通用或维度驱动的 kernel

| 家族 | 源码 | 实际支持面 | 泛化价值 |
|---|---|---|---|
| Dense reference GEMV/GEMM | `linear_generic_dense.cu/.cuh` | BF16/FP32，任意合法 N/K/T | 仅适合作为正确性 oracle 和兜底候选 |
| Q4 RowSplit 三模板族 | `linear/q4/q4_rowsplit_*` | exact-admitted Q4；GEMV、SIMT C4/C8、MMA C64/C128 | 当前 Q4 pure Linear 生产 backend |
| W8 RowSplit Small-T | `linear_rowsplit_gemm_smallt.cu/.cuh` | W8，任意 N/K，TT4/TT8，支持标量 K tail | W8 尚未迁移前的 compatibility 路径 |
| Q5 RowSplit 后端 | `linear/q5/q5_rowsplit_*` | 精确 Q5 支持面；GEMV、SIMT、split-2/split-4、MMA C64/C128 | Q5 独立生产 backend |
| Q6 RowSplit 后端 | `linear/q6/q6_rowsplit_*` | 精确 Q6 支持面；SIMT C4、MMA C64/C128 | Q6 独立生产 backend |
| W8 MMA | `linear_rowsplit_w8g32_gemm_mma.cu/.cuh` | W8，运行时 N/K/T，small-M/通用配置，完整/边界 tile | 独立 W8 生命周期 |

### 5.2 精确 T1 或拓扑专用 kernel

| 家族 | 精确问题 | 说明 |
|---|---|---|
| Q4 MLP gate/up fused GEMV | `[34816,5120]` | 只保留 `linear_swiglu` T1 融合写回；plain Q4 kernel 已删除 |
| Q5 GEMV/residual core | 多组固定 N/K | Q5 后端内部普通写回与 residual finalizer |
| Dense GDN gating T1 | 两个 BF16 `[48,5120]` 权重 | 融合两个投影和 FP32 gate 变换 |

### 5.3 多输出和融合 tensor-core kernel

| 家族 | 语义使用者 | 输出拓扑 |
|---|---|---|
| W8 paired K/V MMA | `linear_pair` | 一个输入、两个同 shape W8 权重、两个输出 |
| Q4/Q5 成组 input MMA | `attn_input_proj`、`gdn_input_proj` | 2 或 4 个 jobs，独立或拼接输出 |
| Q4 folded gate/up SwiGLU MMA | `linear_swiglu` | 成对 gate/up 行和 SwiGLU 最终写回 |
| Q5 residual low-bit MMA | `linear_add` | 低比特 MMA 主循环加 residual 最终写回 |
| Dense GDN cooperative MMA | `gdn_gating_proj` | 两个 dense 投影；必要时使用 split-K workspace；FP32 输出 |

### 5.4 当前源码已经暴露的合理复用边界

当前代码已经证明以下窄层复用有价值：

- Q4 私有 storage/decode atom 与三个独立 lifecycle 模板；
- Q5/Q6/W8 的通用 RowSplit Small-T 拓扑；
- grouped/fused Q4 与 Q5/Q6 仍使用的旧 MMA header 基座；
- 一个支持普通写回、residual、dual-output 的参数化 Q5 T1 core；
- 2/4 projection jobs 的成组 job 描述；
- 完整 tile 与边界 tile 实现变体；
- residual 和 SwiGLU 最终写回变体。

当前代码也证明以下内容不能未经实验直接合并：

- W8 的 group size 和 tensor-core 生命周期不同；
- 成组 input、paired output、residual、SwiGLU 的联合执行经济不同；
- 多个精确 T1 kernel 的 warp ownership 和 K splitting 不同；
- dense GDN control Op 输出 FP32，并包含明确的 BF16 投影舍入边界。

这些只是后续审查输入，不代表应当用一个万能 kernel 模板替换所有实现。

## 6. 产品真实可达的基础投影问题

以下 24 行是当前基础投影清单。即使某个投影只在融合/成组语义算子内出现，只要
显式中间张量策略可能执行它，就仍列为基础投影问题。dense GDN control 不经过生产 base
`linear()`，因此单独列在第 7 节。

### 6.1 Text 及 Text 融合内部问题

| ID | 格式与精确 `W[N,K,Kpad]` | 可达语义用途 | 数量 |
|---|---|---|---:|
| B01 | Q4 `[6144,5120,5120]` | `attn_input_proj` 的 Full Attention Q view | 16 |
| B02 | Q5 `[6144,5120,5120]` | Full gate、GDN V、普通 GDN Z | 112 个权重槽 |
| B03 | Q4 `[1024,5120,5120]` | Full Attention K view | 16 |
| B04 | Q5 `[1024,5120,5120]` | Full Attention V view | 16 |
| B05 | Q4 `[4096,5120,5120]` | GDN 拼接 Q/K 投影 | 48 |
| B06 | Q5 `[5120,6144,6144]` | Full/GDN output `linear_add` | 64 |
| B07 | Q4 `[34816,5120,5120]` | 全部 Text `linear_swiglu` gate/up | 64 |
| B08 | Q5 `[5120,17408,17408]` | 全部 Text MLP-down `linear_add` | 64 |
| B09 | Q6 `[248320,5120,5120]` | full target head | 1 |

可达 T 由阶段决定：

- ordinary verification 为 T1；
- MTP target verification 可达 `T=2..6`；
- 一个 Text prefill chunk 最大为 1024 列，尾 chunk 可以是任意正数；
- 成组/融合语义算子决定基础投影是否真的显式写出中间张量。

### 6.2 MTP 问题

| ID | 格式与精确 `W[N,K,Kpad]` | 可达语义用途 |
|---|---|---|
| B10 | W8 `[5120,10240,10240]` | MTP stem input projection |
| B11 | W8 `[14336,5120,5120]` | MTP Q/K/gate/V 父权重投影 |
| B12 | W8 `[1024,5120,5120]` | MTP prompt K/V `linear_pair`，两个权重 |
| B13 | W8 `[6144,5120,5120]` | 最终 prompt Q 和 gate row view |
| B14 | W8 `[5120,6144,6144]` | MTP attention output |
| B15 | W8 `[34816,5120,5120]` | MTP gate/up projection |
| B16 | W8 `[5120,17408,17408]` | MTP MLP down projection |
| B17 | Q4 `[131072,5120,5120]` | optimized proposal head |

W8 `[14336,5120]` 父权重整体投影及其 `[6144]`、`[1024]` 有效 row view 会在不同
MTP schedule 中真实执行。未来精确 catalog 必须同时覆盖父权重整体投影和有效投影视图；
只登记其中一种是不完整的。

### 6.3 Vision 问题

P 表示 raw patch 列数，V 表示合并后的 Vision 列数。产品 frontend 限制
`P<=131072`、`V<=32768`，并满足 `P=4V`。

| ID | 格式与精确 `W[N,K,Kpad]` | 每次 media encode 调用数 | T |
|---|---|---:|---|
| B18 | Q6 `[1152,1536,1536]` | 1 | P |
| B19 | Q4 `[3456,1152,1152]` | 27 | P |
| B20 | Q5 `[1152,1152,1152]` | 27 | P |
| B21 | Q4 `[4304,1152,1152]` | 27 | P |
| B22 | Q5 `[1152,4304,4352]` | 27 | P |
| B23 | W8 `[4608,4608,4608]` | 1 | V |
| B24 | W8 `[5120,4608,4608]` | 1 | V |

B22 是当前唯一一个逻辑 K 与 Kpad 不同的登记问题。其 padding 和边界 tile 成本必须单独测量，
不能从对齐的 Text shape 推断。

## 7. 独立复合算子执行问题

| ID | 语义算子 | 精确执行问题 | 当前路由面 |
|---|---|---|---|
| C01 | `linear_add` | Q5 B06 | 融合 T1；显式中间张量 T2..16；residual MMA T17+ |
| C02 | `linear_add` | Q5 B08 | 融合 T1；显式中间张量 T2..16；residual MMA T17+ |
| C03 | `linear_swiglu` | Q4 B07 | 融合 T1；显式中间张量 T2..16；T17+ 将 SwiGLU 折入 MMA 写回 |
| C04 | `linear_pair` | 两个 W8 B12 权重 | T1..16 两次 base；T17+ paired MMA |
| C05 | `attn_input_proj` | B01+B02+B03+B04 | T1..16 四次 base；T17+ 两次成组 MMA |
| C06 | `gdn_input_proj` | B05+B02 | T1..16 两次 base 加 copy；T17+ 一次成组 MMA |
| C07 | `gdn_gating_proj` | 两个 dense BF16 `[48,5120]` 权重，加 FP32 向量和输出 | 精确 T1；Small-T 两个 kernel；较大 T 使用 split-K dense MMA |

当前工作区行为同样属于各语义算子自身：

- base `linear()`、`linear_pair`、`attn_input_proj` 当前没有实际 scratch；
- `linear_add` 在 `T=2..24` 显式写出 `[5120,T]` 中间张量；
- `linear_swiglu` 在 `T=2..16` 显式写出 `[34816,T]` 中间张量；
- `gdn_input_proj` 在 `T<=16` 显式写出 `[4096,T]` 和 `[6144,T]` 中间张量；
- `gdn_gating_proj` 在部分 T 范围使用 split-K FP32 workspace。

## 8. 未登记问题与仍不可达的执行路径

target 的 `run_layers()` 只会收到 `Phase::Verify` 或 `Phase::Prefill`。原有两个
`Phase::Decode` 死分支及枚举值已删除；它们不再制造父权重支持假象。

Q4 原子切换同时删除了以下 plain 执行路径：

- Q4 父权重 `[7168,5120]` T1 attention GEMV；
- Q4 `[4096,5120]` application-named GEMV；
- Q4 runtime-N head GEMV；
- Q4 `[34816,5120]` plain T1 GEMV；
- 旧通用 Small-T/MMA 中的 pure Q4 case。

Q5 切换已经删除父权重 `[7168,5120]` T1 GEMV、Q5 dual
`[6144,5120]` V/Z T1 GEMV，以及旧 application-named Q5 GEMV launcher。

以下 view 会被绑定，但不会传给 Linear-family Op：

- GDN Q/K `[2048,5120]` row view；
- Text MLP gate/up `[17408,5120]` row view；
- MTP MLP gate/up `[17408,5120]` row view。

packed 父权重仍然是真实 artifact storage。删除死执行问题不等于修改 artifact
layout，也不等于删除其他语义仍需要的 row view。

以下源码可见执行策略可以通过仓库内部 API 调用，但不会被 current-product route 选中：

- Q4 measurement-only fixed candidate 可以运行 capability-valid、未 admission 的问题，
  但 public `linear()` 会拒绝；
- 单输出 W8 MMA small-M 配置：唯一可达的 W8 `N=1024` 问题是 B12，而其 `T>16`
  路径会被 paired W8 kernel 截获；
- compatibility planner 的 `Generic` 仍可接受部分尚未迁移的 dense/W8 shape。

这些实现应当作为复用、测量或参考候选审查，不能自动视为产品登记需求。

## 9. 盘点暴露的源码不一致

当前源码存在多处失真的描述，不能用于指导架构决策：

- `linear_rowsplit_gemm_smallt.cu/.cuh` 声称不存在 W8 MMA，但 planner 和专用内核启动层
  明确会把 Large-T W8 路由到 MMA；
- 尚未迁移的 Q5/Q6/W8/dense planner 只命名粗粒度执行策略，内核启动层仍会继续做
  行为选择；Q4 已不再有这层二次 dispatch；
- 融合语义 wrapper 各自重复相同的全局 `T=1/16` 阈值；
- benchmark 的 “all target” matrix 已修正 Q4 exact supports，但仍不等于全部 24 个
  Linear-family 基础问题；
- GDN gating benchmark 的 split-K 模型与当前实现不一致。

这些不是现在需要立即处理的独立清理任务。它们说明源码注释、planner 名称和 benchmark
标签都不能替代本文的精确执行问题清单。

## 10. 当前验证与测量覆盖

### 10.1 当前源码测试

| 范围 | 当前证据 | 主要缺口 |
|---|---|---|
| Base dense Linear | BF16/FP32 权重的 FP64 CPU oracle；覆盖多种对齐/非对齐 N/K/T | 性能实现有意保持参考级别 |
| Base quantized Linear | Q4 fixed-candidate oracle、exact plan test、34 个 public-auto/fixed BF16 word 对照；Q5/Q6/W8 CPU dequant oracle | 完整规模 high-T Vision 使用 fixed/public 等价与真实 Engine smoke，不做超大 FP64 oracle |
| `linear_pair` | 小型 generic shape 的 W8 正确性 | 缺 B12 精确几何和全部路由边界 |
| 共享输入多投影算子 | T4 materialized 与 T17/128/129 grouped 路径同 fixed base 输出比较 | 不是完全独立的数学 oracle |
| `linear_add` | T17/128/129 与 Linear 加 residual 比较 | 未直接保留 T1 和显式中间张量 Small-T 语义边界测试 |
| `linear_swiglu` | T4 materialized 与 T17/128/129 fused 路径同 fixed Linear + `silu_mul` 比较 | T1 由真实 Engine 与既有 fused kernel 路径覆盖 |
| `gdn_gating_proj` | T1、2、6、7、8、128、512、1024、2048、4096、4097 的独立 CPU oracle；T1024 graph capture | 缺保留的 NCU 和同一轮性能 baseline |

### 10.2 当前 benchmark 二进制

| 二进制 | 有效覆盖 | 清单问题 |
|---|---|---|
| `ninfer_linear_bench` | dense 与四个 exact Q4 decode problem 的快速信号 | 不是 route seam 或 end-to-end 证明 |
| `ninfer_linear_op_bench` | Q4 auto/fixed exact identity、任意 measurement-only Rows/K、W8 pair、cold/warm 与 roofline probe | “all target” 仍是便捷矩阵，不替代本文 24 个基础问题 |
| `ninfer_gdn_gating_proj_bench` | 独立 dense 融合测量 | split-K 模型已不匹配当前实现阈值 |

因此，当前 benchmark 源码不能自动定义“所有 Linear 算子”。

### 10.3 历史实验资料

保留的实验报告和 profiler 原始文件包含以下有价值的物理证据：

- 24 个基础投影问题和 prototype 的 83 个路由项清单；
- geometry-local Small-T/MMA crossover；
- Q5 direct kernel；
- W8 single/paired 路由；
- `LinearAdd` 和 `LinearSwiGLU` 的非单调行为；
- Vision high-T 和 partial-wave 行为；
- 代表性 SASS、资源、NSYS 和 NCU 捕获。

这些证据产生于 target/profile 所有权仍在探索的阶段，而该所有权设计已经被否决。
测量结果仍可作为候选证据，但 prototype 路由表和实现不能视为当前源码行为，也不能未经
重新资格验证直接复制。

保留的原始 campaign 包含超过一千个 benchmark 文件以及数十个 NCU/NSYS 报告。数量不等于
资格证明；只有 matched、release-build、精确问题的证据才能进入最终算子决策。

当前 Q4 纯 Linear 已有 Op-owned exact catalog、编译期闭包证明、完整 route seam host
测试和 public/fixed GPU dispatch 测试。Q5/Q6/W8/dense 以及各融合/成组语义算子的旧
planner、二次 dispatch 和显式中间张量策略仍需按照新的所有权与 token 支持范围逐项重写；
不能把历史 prototype 的 83 路由表直接视为当前生产事实。

## 11. 逐算子审查记录模板

每个语义算子都使用同一份简洁记录：

1. 精确可达执行问题和 T 范围；
2. 数学语义以及 BF16/FP32 舍入边界；
3. 现有执行策略和全部内核启动层二次分派；
4. 独立数值证据；
5. cold/warm microbenchmark 与 roofline 模型；
6. 端到端调用次数和热点占比；
7. 可复用内核基座与 shape-hard-coded 逻辑的边界；
8. 决策：保留、调优、重写、泛化、拆分或删除；
9. 决策后的精确支持面和路由表；
10. workspace 与 CUDA Graph 后果。

每次算子审查都应同步更新实验日志，记录：

- 完整命令；
- 硬件与工具环境；
- 原始结果路径；
- 数值结果；
- 性能结果；
- 被否决的候选；
- 最终结论。

## 12. 推荐审查顺序

这是审查顺序，不是实现顺序，也不是迁移顺序。

1. **Base RowSplit Linear 基础**
   - Q4/Q5/Q6/W8 T1 和 Small-T；
   - Q4/Q5/Q6 Large-T MMA；
   - W8 Large-T MMA；
   - 精确 head kernel。
2. **LinearAdd**
   - 两个高频 Text 问题；
   - 融合、显式中间张量、residual-MMA 执行策略。
3. **LinearSwiGLU**
   - 一个高频 Text 问题；
   - paired-row 拓扑和非单调融合经济。
4. **AttnInputProj 与 GdnInputProj**
   - 共享输入的多投影成组拓扑；
   - Small-T launch 数和 Large-T 成组 kernel。
5. **LinearPair**
   - 产品可达的 W8 MTP K/V 问题；
   - 删除或单独论证不可达 Q5 路径。
6. **Vision base 问题**
   - 大动态 P/V 范围；
   - partial wave；
   - B22 padded-K 成本。
7. **GdnGatingProj**
   - 独立 dense 生命周期；
   - split-K/workspace 执行策略。
8. **不可达与 generic 支持面**
   - 删除不可达父权重执行策略；
   - 只保留有明确用途的参考/测量泛化。

合理的泛化边界应在每次算子审查中决定，而不是预先假设。codec、主循环拓扑、固定几何、
输出拓扑和最终写回器是不同维度；共享其中一个维度，不代表必须共享完整语义执行策略。
