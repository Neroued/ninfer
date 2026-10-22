# mHC 计算与 Op 设计

本文定义 GLM5Next mHC 的计算组合、公共接口、缓冲布局和实现方案。当前为设计，
尚未实现或测量性能。模型数学以[模型语义](../../maintainer/glm5next-model.md#2-main-与-mhc-residual)
为准，计算精度与舍入以[数值约定](../../maintainer/glm5next-quantization.md#5-mhc-数值边界)为准。
本文将这些要求落实为可独立调用的 Op，不另设模型专用的数学路径。

Main 的每层 attention、FFN 各有一个 mHC site；当前 45 层共 90 个，各有独立权重。
MTP 使用普通 residual，不调用本 family。mHC 与 Qwen4Exp gated residual 的系数生成、
归一化和残差混合不同，保留独立入口；底层矩阵计算、归约和访存机制可以共享。

## 1. 权重与 tensor 布局

`D` 为 hidden，`S` 为 residual stream 数，`N=S*(S+2)` 为系数投影行数。
首个实现几何为 `D=4096, S=4, N=24`。参数由模型配置和权重绑定提供，支持范围由
Op 判断；模型和 Engine 不复制 Op 的几何限制。

Native Tensor 第零维连续。所有计算逐 token 独立，公共接口只接收列数 `M`。
模型的 `[T,B]` 可直接展平为 `M=T*B, m=t+T*b`，不需要重排或复制。

| 对象 | Native 布局 | 语义 |
|---|---|---|
| `residual / updated_residual` | BF16 `[D,S,M]` | hidden 连续，随后为 stream |
| `embedding / block_input / block_output / hidden` | BF16 `[D,M]` | 完整分支 hidden |
| `post` | FP32 `[S,M]` | 已激活的写入系数 q |
| `mix` | FP32 `[S_new,S_old,M]` | `mix[j,i,m]=C_math[i,j,m]` |
| `projection`，数学记为 F | BF16 `Weight`，逻辑 `[N,S*D]`、native `[S*D,N]` | `QuantLayout::Contiguous`，无 bias、无 padding |
| `base / scale` | BF16 `[N]` / `[3]` | pre、post、mix 的偏置及三组标量 scale |
| `gamma / gamma_final` | BF16 `[D]` | 直接 gain |

以元素为单位的偏移为：

```text
residual(d,i,m) = d + D*(i + S*m)
block(d,m)      = d + D*m
post(j,m)       = j + S*m
mix(j,i,m)      = j + S*i + S*S*m
F(a,k)         = k + S*D*a, k=i*D+d
```

F 的行序为 `[pre:S | post:S | mix:S*S]`，mix 行号 `2*S+S*i+j`，即旧流 i 在外、
新流 j 连续。权重行序与输出 mix 的连续轴一致。Converter 保留源 BF16 words，
按此顺序导入；不同 site 的输入不同，不合并其 F，也不在执行时重新打包权重。

## 2. 完整计算与表示边界

以下省略 token 轴，所有 BF16 输入先按其实际表示值参与计算。平方和、点积累加、
小矩阵归一化及后处理采用 FP32。当前语义参数如下，属于模型数学，不用于调优：

| 参数 | 当前值 |
|---|---:|
| `coefficient_epsilon / branch_epsilon / final_epsilon` | `1e-5` |
| `pre_epsilon / sinkhorn_epsilon` | `1e-6` |
| `sinkhorn_iterations` | 20 |
| `post_multiplier` | 2 |

### 2.1 Pre：系数与分支输入

先对完整 S×D residual 求无 affine 的 RMS 系数，再生成 N 行 logits：

```text
r = rsqrt(sum_{i,d} X[i,d]^2 / (S*D) + coefficient_epsilon)
z[a] = r * sum_{i,d} F[a,i*D+d] * X[i,d]

p[i]   = sigmoid(scale[0]*z[i]         + base[i])         + pre_epsilon
q[j]   = post_multiplier * sigmoid(scale[1]*z[S+j] + base[S+j])
A[i,j] = scale[2]*z[2*S+S*i+j]         + base[2*S+S*i+j]
```

F 和 X 是 BF16，可使用 BF16 乘积、FP32 累加的 Tensor Core；r 在点积之后以 FP32
应用。不物化 BF16 normalized X，不将 logits 经普通 Linear 的 BF16 出口再读取。

对 mixing logits 执行固定的 Sinkhorn 序列，令 `e=sinkhorn_epsilon`：

```text
C = softmax_row(A) + e
C[i,j] /= sum_i C[i,j] + e

repeat sinkhorn_iterations - 1 times:
    C[i,j] /= sum_j C[i,j] + e
    C[i,j] /= sum_i C[i,j] + e
```

Softmax 减去每行最大值；每次归一化的分母使用该步开始时的完整行或列。
当前共 20 次列归一化、19 次显式行归一化，不能改为 20 轮完整行列归一化或早停。
有限迭代和 epsilon 使 C 仅近似双随机，后续不能假定其行列和严格等于 1。

分支输入从原始 residual 汇合，再执行该分支的带权 RMSNorm：

```text
v[d] = sum_i p[i] * X[i,d]
u[d] = RN_BF16(gamma[d] * v[d]
               * rsqrt(sum_d v[d]^2 / D + branch_epsilon))
```

p 不跨流归一化，v 不除以 S；v 保持 FP32，norm 前没有 BF16 舍入。这里的 D 维
分支 norm 与前面的 S×D 维系数 norm 是两个归约。`gamma` 为直接增益，不加 1。
输出 u 已完成分支输入 norm，后续 KDA、MLA、dense MLP 或 MoE 不重复执行该 norm。

p 只在 pre 内部使用。跨越分支执行保存的是 FP32 q 和 C。

### 2.2 Post：四流混合与注入

分支产生完整 BF16 输出 y 后：

```text
X_next[j,d] = RN_BF16(sum_i C[i,j]*X[i,d] + q[j]*y[d])
```

C 的方向为 `C[old,new]`，计算为 `C^T X + q*y`。混合与注入以 FP32 相加后只做
一次 BF16 舍入。q/C 来自该分支执行前的 X，直接应用，不重新 sigmoid 或 Sinkhorn。

`post_pre` 将此 post 与下一 site 的 pre 合并。下一 site 的全部计算都消费实际
BF16 X_next：

```text
FP32 mixed residual → RN_BF16 → X_next
                                 ├→ next F projection / coefficient RMS
                                 └→ next collapse / branch RMSNorm
```

转换可在寄存器或 shared memory 完成，但不能一面写 BF16 residual，一面用未舍入
的 FP32 值生成下一组系数。`RN_BF16` 为 round-to-nearest、ties-to-even；归约顺序
和 kernel 分解不固定，不要求独立入口与融合入口逐 bit 相同。

### 2.3 Main 的入口与出口

入口在 embedding／媒体特征替换完成后，将每个 BF16 word 精确复制到 S 条流，
包括 signed zero，再执行第一个 site 的 pre。

最后一个 site 先完成 post 的 BF16 residual 边界，再计算：

```text
mean[d] = sum_j X_next[j,d] / S
hidden[d] = RN_BF16(gamma_final[d] * mean[d]
                    * rsqrt(sum_d mean[d]^2 / D + final_epsilon))
```

mean 保持 FP32；最终 norm 前不插入 BF16 舍入。最后的 S 条 BF16 流可以只在片上
形成，无需写回完整 residual。不能利用近似双随机性跳过混合，也不能跨越该 BF16
边界代数合并求均值。hidden 供 Main head 和 MTP conditioning 共同使用。

## 3. 公共接口

声明统一放在 `include/ninfer/ops/mhc.h`。以下结构只借用已绑定的 views，不拥有
资源；输出结构中的 Tensor 描述符按值传入，写入其指向的设备数据。

```cpp
namespace ninfer::ops {

struct MhcGeometry {
    std::int32_t hidden;
    std::int32_t streams;
};

struct MhcPreParameters {
    float coefficient_epsilon;
    float branch_epsilon;
    float pre_epsilon;
    float sinkhorn_epsilon;
    float post_multiplier;
    std::int32_t sinkhorn_iterations;
};

struct MhcPreWeights {
    Weight projection; // BF16 Contiguous [N,S*D], logical output rows first.
    Tensor base;       // BF16 [N].
    Tensor scale;      // BF16 [3].
    Tensor gamma;      // BF16 [D], direct gain.
};

struct MhcCoefficients {
    Tensor post; // FP32 [S,M].
    Tensor mix;  // FP32 [S_new,S_old,M].
};

struct MhcPreOutputs {
    Tensor block_input; // BF16 [D,M], branch norm already applied.
    MhcCoefficients coefficients;
};

void mhc_expand_pre(
    const Tensor& embedding, const MhcPreWeights& weights,
    const MhcPreParameters& parameters, Tensor& residual,
    MhcPreOutputs outputs, MemoryArena& workspace,
    CudaExecutionView execution);

void mhc_pre(
    const Tensor& residual, const MhcPreWeights& weights,
    const MhcPreParameters& parameters, MhcPreOutputs outputs,
    MemoryArena& workspace, CudaExecutionView execution);

void mhc_post(
    const Tensor& residual, const Tensor& block_output,
    const MhcCoefficients& coefficients, Tensor& updated_residual,
    CudaExecutionView execution);

void mhc_post_pre(
    const Tensor& residual, const Tensor& block_output,
    const MhcCoefficients& coefficients,
    const MhcPreWeights& next_weights,
    const MhcPreParameters& next_parameters, Tensor& updated_residual,
    MhcPreOutputs next_outputs, MemoryArena& workspace,
    CudaExecutionView execution);

void mhc_post_mean_norm(
    const Tensor& residual, const Tensor& block_output,
    const MhcCoefficients& coefficients, const Tensor& gamma_final,
    float final_epsilon, Tensor& hidden, MemoryArena& workspace,
    CudaExecutionView execution);

[[nodiscard]] std::size_t mhc_expand_pre_workspace_capacity_bytes(
    const MhcGeometry& geometry, std::int32_t sm_count,
    std::int32_t min_columns, std::int32_t max_columns);

[[nodiscard]] std::size_t mhc_pre_workspace_capacity_bytes(
    const MhcGeometry& geometry, std::int32_t sm_count,
    std::int32_t min_columns, std::int32_t max_columns);

[[nodiscard]] std::size_t mhc_post_pre_workspace_capacity_bytes(
    const MhcGeometry& geometry, std::int32_t sm_count,
    std::int32_t min_columns, std::int32_t max_columns);

[[nodiscard]] std::size_t mhc_post_mean_norm_workspace_capacity_bytes(
    const MhcGeometry& geometry, std::int32_t sm_count,
    std::int32_t min_columns, std::int32_t max_columns);

} // namespace ninfer::ops
```

### 3.1 各入口的完整效果

| 入口 | 读取 | 完整输出 |
|---|---|---|
| `mhc_expand_pre` | embedding、当前权重与参数 | 精确复制的 residual、归一化 block input、q/C |
| `mhc_pre` | residual、当前权重与参数 | 归一化 block input、q/C |
| `mhc_post` | residual、完整 block output、对应 q/C | BF16 updated residual |
| `mhc_post_pre` | 旧 residual、完整 block output、旧 q/C、下一 site 权重与参数 | BF16 updated residual、下一 block input、下一 q/C |
| `mhc_post_mean_norm` | 旧 residual、最后的 block output、对应 q/C、final gamma/epsilon | BF16 hidden |

运行时 D/S/M 从 residual view 得到；expand 从目标 residual view 得到，并核对
embedding 和权重形状。`MhcGeometry` 只用于无地址容量规划，不在执行接口中再次
传递同一组形状。Weight 的格式固定为 BF16，不增加 activation policy 选项。

数值参数由模型语义显式绑定，没有 checkpoint 专用默认值。epsilon 和 multiplier
为有限正数，iterations 为正整数。参数的数值不改变 scratch 布局；容量查询覆盖
同几何下合法参数的需求。实际支持的几何、类型、布局与参数由本 family 入口验证。

### 3.2 写域与 alias

- M 为正，没有独立 batch 上限或 prefill/decode 模式。Op 不接收长度、位置、rank、
  KV 页或 state slot。所有物理列完整计算、完整写入；padding 列提供有效有限输入，
  有效列选择由后续消费者处理。
- 所有 Tensor 和 F 连续、基址至少 16-byte 对齐；全部 operands 与 arena 位于
  execution 的 CUDA device。Device 数据值是调用前提，不为验证 gate 或 C 的值
  发起 Host readback。
- `mhc_post`、`mhc_post_pre` 的 updated residual 可与输入 residual 完全同 view，
  或完全不重叠；禁止部分 overlap。每个 hidden 位置必须读完全部旧流后再覆盖。
- 除这一原位更新，输出与输入、权重、其他输出及当前 scratch 不重叠。新旧 q/C
  必须分开；原位修改的 residual 不能兼作其他只读输入。
- 只读输入在形状和语义允许时可以共享 backing，所有未声明修改的输入保持不变。
  末尾入口只输出 hidden，不修改输入 residual，也不要求物化最终四流 residual。

### 3.3 Workspace 与 Graph

`mhc_post` 是固定零 scratch 入口，不传 arena、不提供容量查询。其余入口可使用
caller-owned `MemoryArena`，由匹配查询返回整个 Op 所需容量；合法的零 scratch
路线返回零。Arena 与 operands 可以来自同一 backing，但使用区域不能重叠。

查询只读取 geometry、物理 SM 数和闭区间 `[min_columns,max_columns]`，不依赖
权重地址、stream 或真实设备分配。返回值是覆盖区间内所有合法 M 和 alias 形式的
最小高水位容量，包含所有私有阶段及内部对齐，不能只查区间上界。容量从 256-byte
对齐的 Op-local cursor 计量，排除公共输出和末尾额外 padding。

`sm_count` 为正且与执行设备的物理 SM 数一致，区间满足
`0 < min_columns <= max_columns`；不支持的 geometry 或无效查询参数由 Op 报错。
精确单次查询使用相同的区间端点。

查询与执行共享同一套选择及分配规则。调用在 arena scope 内借用 scratch，返回时
恢复 cursor；设备使用期通过提交 stream 的顺序保证，Host scope 结束不等于 GPU
执行结束。跨 stream 的复用同步由资源所有者负责。

所有入口提交到 `execution.stream()`，不隐藏设备分配或 Host 同步。在固定几何、
参数及绑定地址下可 capture/replay；重播读取当前输入、权重和系数内容。
kernel 数、tile、partial 布局和路线阈值由实现决定，不成为模型参数。

## 4. 实现组织与两阶段计算

### 4.1 文件与共享边界

```text
include/ninfer/ops/mhc.h        公共数学、views、别名与执行资源合同
src/ops/mhc/                   入口、校验、容量、私有计算与 launch
src/ops/mhc/dgx_spark/         Spark 的实例选择与调优策略
tests/ops/mhc/                 公共入口的独立数学验证
bench/ops/mhc/                 完整入口的测量
```

这是职责划分，不要求每项单独成文件。按复杂度组织 pre 与 post 的实现；相邻融合
共享本 family 的投影／系数／混合计算体，不逐层调用公共 Op 来拼装融合。
使用 `ops/common` 已有的数学、归约和访存基础，BF16 矩阵计算参考 Linear 的模板。
只有实际适配共同计算的机制才抽取共享，mHC 的行序与 Sinkhorn 留在本 family。
Spark 的分派不进入 RTX 5090 的调优目录，公共数学保持一致。

### 4.2 第一阶段：更新／展开、投影 partial 与平方和

Pre 相关入口采用两阶段作为首版实现。第一阶段沿 hidden D 分 tile，每个 tile
负责该 hidden 范围的全部 S 条流，以及其 token tile：

| 入口 | 第一阶段取得当前 X 的方式 |
|---|---|
| `mhc_pre` | 读取已有 residual |
| `mhc_expand_pre` | 读取 embedding，精确复制并写出 S 条 BF16 流 |
| `mhc_post_pre` | 读取旧 S 条流、y 和旧 q/C，计算并写出 BF16 X_next |

每个 tile 使用这些 BF16 值计算 N 行投影的 FP32 partial，以及一份平方和 partial。
`post_pre` 在转换成 BF16 后直接复用片上值参与下一投影，不再发射独立 post kernel。
`expand_pre` 同理，不需要先发射完整展开 kernel。

按 hidden 分块、在同一个 tile 处理所有流，使 post 的每份旧 X 和 y 可被复用，并
允许原位更新。每个 residual 元素只有一个 writer；首版一个 tile 同时计算全部
24 行系数，不能让多个输出行 CTA 重复执行原位 post、相互读取已经覆盖的 residual。

若 D 被拆为 P 个 tile，partial 的逻辑 payload 为 FP32 `[N+1,P,M]`，当前为
`100*P*M` bytes。实际 padding、排列和 P 的选择属于私有实现，以容量查询为准。
S 条流在 K 轴上各有一个 hidden 子区间，partial 必须累加全部子区间，不能误当作
只覆盖某一段连续 K。

### 4.3 第二阶段：系数、Sinkhorn、collapse 与 norm

第一阶段结束后，第二阶段逐 token 或 token tile 完成：

1. 归约 N 行投影 partial 与平方和，计算 r、z、p、q 和 mixing logits。
2. 在片上完成全部 4×4 Sinkhorn 步骤，保存 FP32 q/C。
3. 读取当前 BF16 residual，计算 FP32 collapse v、其平方和及 branch RMSNorm。
4. 输出完整 BF16 block input。

4×4 矩阵可由一个 warp 内的 16 个 lanes 承载，行列归约使用 shuffle；下一轮必须
读到上一步完整更新后的值。广播 p 给负责 hidden 的 threads，v 可暂存在寄存器或
shared memory 后归约 norm。无需全局物化 p、v 或 normalized residual。

这种分解增加一次 residual 读取，但在小 M 时能通过 hidden 分块获得足够并行度，
同时把 gate、固定迭代和 norm 留在一个收尾 kernel。大 M 可在第一阶段 tile 多个
tokens，用 BF16 Tensor Core 复用 F；窄 N=24 的 padding 和尾部由私有 kernel 处理。

先按完整两阶段入口调优。若进一步融合能减少流量且不损失并行度，可在同一公共
接口内加入路线；保留 BF16 边界、alias 和容量规则，无需修改模型调用。

### 4.4 独立 post 与末尾收束

独立 post 沿 hidden 和 M 向量化。一个计算单元先读取所需的全部旧流，再计算并写
全部新流，复用 q/C 和 y，直接支持 exact in-place。

`post_mean_norm` 在片上计算 S 个新的 BF16 residual 值，立即转回 FP32 参与 mean。
再归约 mean 的平方和、应用 final gamma，最终只写 `[D,M]` hidden。首版可按 token
使用一个 CTA；若小 M 的并行度需要分块，则通过本入口的私有 scratch 和归约实现，
不新增公开 partial 接口或 BF16 mean 落点。

## 5. 模型层序与资源所有权

```text
embedding / 已替换的媒体特征
  → mhc_expand_pre(attention 0)
  → attention 0
  → mhc_post_pre(attention 0 → FFN 0)
  → FFN 0
  → mhc_post_pre(FFN 0 → attention 1)
  → ...
  → 最后一个 FFN
  → mhc_post_mean_norm
  → Main hidden → LM head / MTP conditioning
```

90 个 site 对应一次 expand-pre、89 次相邻 post-pre、一次末尾收束。独立 pre/post
保留完整计算边界，用于需要独立组合的调用；融合入口不要求先执行独立入口。

| 所有者 | 内容 |
|---|---|
| ModelShard／权重绑定 | 不可变 F、base、scale、branch/final gamma；语义参数来自模型配置 |
| RankProgram frame | residual、block input/output、两组 q/C、scratch 和 Graph 的执行资源 |
| 模型计算组织 | site 顺序、next weights、分支 norm 的归属、完整分支输出的通信交接 |
| mHC Op | 数学、布局校验、支持范围、路线选择、借用资源上的计算与写入 |

两组 q/C 交替使用：旧组留给当前 post，另一组接收下一 pre 的结果。Residual 可
原位更新，也可双缓冲；都按执行层序复用，不为 90 个 site 常驻保存全部激活。
每物理 token 一份 residual 为 32 KiB，双组 q/C 共 160 B；完整容量见
[并行资源预算](parallel.md#4-mhc-执行资源)。

mHC 没有跨 token 递推状态，不进入 KV pool、StateImage、prefix checkpoint 或
ReplaySSM。投机执行只需要管理下游 KDA/MLA 状态和最终 Main hidden；无需保存
用于回滚的 mHC 历史 residual。

当前双 Spark 计划在两 rank 上复制 mHC。KDA/FFN 的局部结果归约、MLA owner 输出
汇集完成后，再向 mHC 传入完整 BF16 y；具体通信与行序见
[并行分工](parallel.md#1-首版并行分工)。mHC 不接收 communicator，不判断 TP/DP/EP。

## 6. 实施与验证

### 6.1 落地顺序

1. 实现入口合同、地址无关容量查询、独立 pre/post 和数学参考。先得到正确的
   系数、Sinkhorn、collapse/norm 与残差更新。
2. 复用计算体实现 expand-pre、两阶段 post-pre 和末尾 post-mean-norm，完成
   BF16 边界与原位更新验证；不以串接公共入口代替融合实现。
3. 按 M 调优完整入口，形成 Spark 的实例与分派，再由模型按第 5 节接入。

### 6.2 数学与合同验证

使用独立 FP64 参考解读实际 BF16 输入，按完整公式计算；显式加入合同要求的
BF16 residual 边界，其他中间值不为迎合某条实现增加舍入。精度要求按公共输出
语义制定，不随私有路线或内部采用的更高精度变化。

重点覆盖以下能区分数学或表示错误的样本，不展开无目的的组合穷举：

| 样本／调用 | 检查目的 |
|---|---|
| 非对称 C、四流幅值不同 | mixing 方向、全宽系数 RMS、collapse 使用原始 X |
| 非平凡 p/gamma、零输入、饱和 logits | gain、epsilon 位置、稳定 softmax、p 不归一化或除以 S |
| 不平衡 mixing logits | 固定迭代次数与行列顺序，不依赖精确双随机性 |
| BF16 舍入边界附近的 residual，以及有明显舍入差异的 collapse/mean | 下一 pre 使用实际 BF16 X_next，同时不增加其他 BF16 中间边界 |
| 多列与列重排、exact in-place 与 disjoint | 列独立性、完整写域、原位更新的旧流依赖 |
| 入口复制与末尾收束 | BF16 words 精确复制；最后一次混合、注入、均值和 norm 完整执行 |
| Graph replay 时修改绑定输入内容 | 读取当前内容，无隐藏分配、Host readback 或长期 scratch 借用 |

融合入口独立对完整数学验证；此外，从其实际返回的 BF16 updated residual 计算
下一 pre 参考，确认系数与 block input 的来源。不能只做融合／拆分结果对比。
未修改输入、复制 words 和界外区域按精确规则检查。M 取小列数、代表性内部值和
实际分派边界；不引入不存在的 batch 支持限制。

### 6.3 性能目标与计量

覆盖小 M 的 decode／verify 和大 M 的 prefill，测量点至少包括
`1,2,4,8,16,32,64,128,256,512,1024,2048,4096`，必要时补真实分派边界。
分别报告完整 pre、post、post-pre、expand-pre 和末尾入口的延迟与 workspace。

当前几何下，F 为 0.75 MiB，系数投影每 token 为 786432 FLOPs；激活的逻辑流量为：

| 入口／方案 | 每 token 的主要激活流量 |
|---|---:|
| 独立 post：旧 X 读、y 读、新 X 写 | 72 KiB |
| 两阶段 post-pre：上述流量＋新 X 再读＋u 写 | 112 KiB |
| post-pre 的外部输入输出最低流量，不含内部再读 | 80 KiB |

这些计数不含权重、小参数、q/C、partials、padding 和重复事务，不能当作实测 DRAM
流量。Roofline 分别核算带宽与 BF16/F32 计算下界，并纳入实际路线的 scratch
读写；以完整入口的延迟、有效带宽、计算利用率及 roofline 展示结果。

小 M 要观察并行度、launch 和 Sinkhorn 收尾，大 M 要观察 F 的跨 token 复用、
residual 及 partial 流量。单 site 的 F 较小，重复同地址可能命中 L2；热缓存结果
与模拟跨 site 权重轮换的结果分开解释，不把 L2 服务的逻辑字节除以时间宣称为
显存带宽。融合与拆分对照使用相同数学、表示、输入和测量条件。

性能结论以完整公共 Op 为单位。单独 GEMM 或 Sinkhorn 的数据用于定位，不代替
完整入口；双机端到端收益需要模型接入后另行测量。
