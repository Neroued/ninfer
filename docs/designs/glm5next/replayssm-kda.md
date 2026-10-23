# KDA 输入准备、递推与 ReplaySSM 设计

本文定义 GLM5Next 的 KDA 计算链、状态布局和目标 C++ 接口。输入准备参考现有
GDN projection/conv，状态使用现有 `LinearAttentionState` 和 `StateImage`，接受前缀
采用 ReplaySSM。新增接口尚待实现，现有 KDA 递推、Linear 和输出 norm 按下文复用。

模型数学与数值政策分别以[模型定义](../../maintainer/glm5next-model.md#3-kda)、
[量化方案](../../maintainer/glm5next-quantization.md#41-kda-与-ffn)为准。
状态接口参考 [GDN 输入准备](../../../include/ninfer/ops/gdn_input_proj.h)、
[GDN 递推](../../../include/ninfer/ops/gated_delta_net.h)和
[GDN ReplaySSM](../../../include/ninfer/ops/gdn_replay.h)。

## 1. 计算流程与职责

首个实例为双 Spark：Main 的 34 个 KDA 层，每 rank 持有 32 个完整 heads，
`Dk=Dv=128`，Q/K/V 各 4096 channels，卷积通道总数 `C=12288`。
这些数值是模型实例和已准备 Op profile 的参数；状态容器不按模型名称判断几何。

输入 x 为分支 input norm 后的 BF16 `[4096,T,B]`。整个本地 KDA 分支为：

```text
x ── Q8 QKV projection + FP32 four-tap convolution + SiLU ── q, k, v
  └─ BF16 control split ── beta_raw, f_a, g_a
                           f_a ── BF16 f_b projection ── forget_raw
                           g_a ── BF16 g_b projection ── output_gate_raw

q, k, v, forget_raw, beta_raw, A_log, dt_bias, source recurrent state
  ── KDA recurrence ── o
  ── sigmoid_gated_rmsnorm(o, gamma, output_gate_raw)
  ── Q8 O projection ── local output [4096,T,B]
```

控制投影使用已有纯 Linear/split：`[288,4096]` 输出 `[beta32;f_a128;g_a128]`，
f_b/g_b 各 `[4096,128]`，消费各自的低秩输入。Q/K/V 无复制 view 为
`[128,32,T,B]`；forget/output gate 同样 view 为 `[128,32,T,B]`。
Output norm 复用 `sigmoid_gated_rmsnorm()`，gamma 为共享的 BF16 `[128]`、
epsilon 为 `1e-5`；O 使用 Q8 `[4096,4096]`。输出不在本地加入 residual。

| 所有者 | 职责 |
|---|---|
| Core | StateImage 字段、records 的布局计算、绑定和非 owning views |
| 输入准备 Op | QKV projection/conv/SiLU、最终 conv history 或 conv records |
| KDA recurrence Op | Q/K normalization、内部 gate、递推/readout、transition records |
| KDA fold Op | 从 source 重放指定前缀，写 destination 的 recurrent/conv state |
| RankProgram | frame、workspace、records、prepared fold plan、Op 顺序与 Graph |
| ModelSession | source/destination 的使用权限、最终采用长度、frontier 和 checkpoint |

Op 的状态写入是物理效果；完整模型状态的采用由 Session 在执行依赖满足后完成。

## 2. 持久状态与递推输入

### 2.1 持久状态

复用 [LinearAttentionState](../../../src/core/linear_attention_state.h)：

| 字段 | 单层 native Tensor | 含义 |
|---|---|---|
| recurrent | FP32 `[128,128,Hv]` field | 逻辑 `S[value,key]`，key 轴连续 |
| conv | BF16 `[C,3]` field | 最近三列卷积前投影，最旧到最新，channels 为 `[Q;K;V]` |

每条 State record 的字段内连续；字段与层偏移由 StateFieldLayout 给出。All-layer view 的 conv 和
recurrent layer strides 独立，按现有 StateImage field 布局读取。
初始 image 的这两个字段置零，初始化和存储寿命由现有 context/state 机制负责。

Checkpoint 保存已采用的 recurrent/conv 字段。Records 属于本轮执行 frame；本轮
完成 fold 后，持久 state 已代表最新采用前沿，不保留跨轮待 fold 的已提交 records。

### 2.2 递推操作数

以下为 batched 视图；单序列入口省略 B，单列 update 使用 T=1。

| 参数 | 类型与形状 | 含义 |
|---|---|---|
| q / k | BF16 `[128,Hq,T,B]` | convolution/SiLU 后、L2 normalization 前 |
| v | BF16 `[128,Hv,T,B]` | convolution/SiLU 后的 value |
| g | BF16 `[128,Hv,T,B]` | raw forget logits，即模型公式中的 f |
| beta | BF16 `[Hv,T,B]` | raw beta logits |
| A_log | FP32 `[Hv]` | 每 head 的 decay 参数 |
| dt_bias | FP32 `[128,Hv]` | 每 head、每 key channel 的 bias |
| lower_bound | FP32 scalar | 有限且在 `[-5,0]`，本模型为 -5 |
| scale | FP32 scalar | `1/sqrt(128)` |
| out | BF16 `[128,Hv,T,B]` | 更新 state 后的 scaled query readout |

沿用现有 KDA 的 `Hq>=1`、`Hv>=Hq`、`Hv%Hq==0`；value head h 使用
`qh=floor(h/(Hv/Hq))`。当前实例 Hq=Hv=32。

Q/K 在 KDA 内部按 epsilon=`1e-6` 做 L2 normalization。Gate 同样在内部形成：

```text
log_decay[c,h,t,b] = lower_bound * sigmoid(exp(A_log[h]) * (g[c,h,t,b] + dt_bias[c,h]))
beta_value[h,t,b] = sigmoid(beta[h,t,b])
```

State 按逐 key channel 的 decay、prediction、delta、rank-one update 顺序更新，
再产生 readout。运行和发布的 state 都是 FP32。Record 保存上述实际 FP32 gate
words；fold 从 log_decay 重新计算 alpha 的 exp，使用与 update/record 相同的转移。

这里没有把 gate 生成移到外部投影，也没有给 recurrent 输入增加一个 gate 量化边界。
输出 norm 和 O projection 不属于 state transition。

## 3. 几何、行身份与 record 布局

### 3.1 几何与有效范围

- L：本地 KDA 层数，当前为 34，按 Main 中 KDA 层的出现顺序编号。
- `C=2*Hq*Dk+Hv*Dv`，当前为 12288；计算入口的 Dk/Dv 为 128。
- R：frame 的 record 行容量，1..8；B：本次活动行数，`1<=B<=R`。
- T：物理窗口宽度，任意正整数，受 Tensor 和已分配容量约束。
- n[b]：本次有效前缀长度，`1<=n[b]<=T`；空 `valid_columns` 表示各行均为 T。
- m[b]：最终采用的输入列数，`0<=m[b]<=n[b]`。

第零维连续，token/batch 展平为 `column=t+T*b`。Record row、State record 和
request 身份分别表达；一轮 record 到 fold 完成期间，physical row 顺序保持不变。
实际 T/B 是本轮产生 records 时的形状，不能把 backing 直接重解释成更短的窗口。
持久 state 通过 StateFieldView/ConstStateFieldView 的 Device I64 `[B]` byte offsets 选择，
相对于本 rank 的 Context backing。字段布局不含容量轴，临时 record rows 仍保持 I32 索引语义。
存储与借用合同见 [Context 存储](../../maintainer/paged-kv-cache.md)。

### 3.2 五个 planes

| Plane | 完整分配形状 | 单层 B 行视图 | 内容 |
|---|---|---|---|
| conv | BF16 `[C,T,L*R]` | `[C,T,B]` | 每步最新的卷积前 Q/K/V history words |
| key | BF16 `[Dk,Hq,T,L*R]` | `[Dk,Hq,T,B]` | normalization 前的实际 key words |
| value | BF16 `[Dv,Hv,T,L*R]` | `[Dv,Hv,T,B]` | 实际 value words |
| log_decay | FP32 `[Dk,Hv,T,L*R]` | `[Dk,Hv,T,B]` | 内部逐 channel log-decay |
| beta | FP32 `[Hv,T,L*R]` | `[Hv,T,B]` | 内部 sigmoid 后的 beta |

外层索引统一为 `outer=layer*R+b`，各 plane 连续、起点 256 B 对齐、互不重叠。
`layer(l,B)` 选择 `[l*R,l*R+B)`，stride 中使用容量 R，不使用活动 B。
Log-decay 与 beta 分开保存，beta 不沿 128 个 channels 重复。

### 3.3 Core 接口

新增 `core/kda_replay_records.h`，沿用 GDN records 的布局和绑定方式：

```cpp
namespace ninfer {

struct KdaReplayRecordSpec {
    std::int32_t layers = 0;
    std::int32_t record_capacity = 0;
    std::int32_t width = 0;
    std::int32_t conv_channels = 0;
    std::int32_t qk_heads = 0;
    std::int32_t value_heads = 0;
    std::int32_t key_dim = 0;
    std::int32_t value_dim = 0;
};

struct KdaReplayRecordLayout {
    KdaReplayRecordSpec spec;
    TensorRegion conv, key, value, log_decay, beta;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

[[nodiscard]] KdaReplayRecordLayout plan_kda_replay_records(
    LayoutBuilder& builder, const KdaReplayRecordSpec& spec);

struct KdaReplayRecordLayer {
    Tensor conv, key, value, log_decay, beta;
};

struct KdaReplayRecords {
    Tensor conv, key, value, log_decay, beta;
    KdaReplayRecordSpec spec;

    KdaReplayRecords() = default;
    KdaReplayRecords(MemorySpan backing, const KdaReplayRecordLayout& layout);

    [[nodiscard]] KdaReplayRecordLayer layer(std::int32_t layer,
                                            std::int32_t rows) const;
};

} // namespace ninfer
```

Planner 使用 `LayoutBuilder` 计算五个 TensorRegion，检查物理形状、字节数与 extent
溢出；不申请内存。`payload_bytes()` 是五个 plane 的有效载荷之和，包含 alignment
的 frame 容量使用 builder 的最终结果。绑定对象借用 backing，不保存有效长度、
source offset、frontier 或采用策略。计算支持域由消费这些 views 的 Op 验证。

## 4. 输入 projection/conv

### 4.1 权重与计算

首个 profile 是 Q8-G32/FP16、A16Only 的单个 parent `[12288,4096]`，行序为
`[Q4096;K4096;V4096]`。x 为 BF16 `[4096,T,B]`，三个输出各为 BF16 `[4096,T,B]`。

Convolution weight 为 FP32 native `[C,4]`，channel 连续，四个 taps 顺序为
`t-3,t-2,t-1,t`。Artifact 逻辑 shape 为 `[4,C]`。各源 `[8192,1,4]` 保值转置为
`destination[tap,channel]=source[channel,0,tap]`；加载时按本地 heads 组合 Q/K/V
channels，执行时直接使用该布局。

令 `p=concat(Wq*x,Wk*x,Wv*x)`，将源 history 的三列与当前窗口 p 按时间衔接，
逐 channel 做四 tap causal convolution，再做 SiLU。Q 的 history 同样保留，
即使 recurrent replay 不需要 query。

Projection、conv、SiLU 是完整计算。当前窗口的私有 projected values 不增加强制
BF16 舍入；Q/K/V 输出和对外发布的 history/conv_record 为 BF16。
因此普通 Linear 的 BF16 输出不是此融合 Op 的内部必经缓冲。

### 4.2 两种出口

声明位于 `include/ninfer/ops/kda_input_proj.h`，使用已有
`SingleProjectionWeight` / `LinearProfile` 表达单 parent 及 activation permission：

```cpp
namespace ninfer::ops {

struct KdaConvOutputs {
    Tensor query, key, value;
};

struct KdaConvSourceState {
    ConstStateFieldView history; // Read-only BF16 [C,3], with Device I64 record offsets.
};

struct KdaConvUpdateState {
    ConstStateFieldView source; // BF16 [C,3].
    StateFieldView destination; // BF16 [C,3], with independently selected record offsets.
};

void kda_input_proj_conv_update(
    const Tensor& x, const SingleProjectionWeight& qkv,
    const Tensor& conv_weight, KdaConvUpdateState state,
    const Tensor& valid_columns, KdaConvOutputs outputs,
    MemoryArena& workspace, CudaExecutionView execution);

void kda_input_proj_conv_record(
    const Tensor& x, const SingleProjectionWeight& qkv,
    const Tensor& conv_weight, KdaConvSourceState state,
    const Tensor& valid_columns, KdaConvOutputs outputs,
    Tensor& conv_record, MemoryArena& workspace,
    CudaExecutionView execution);

[[nodiscard]] std::size_t kda_input_proj_conv_update_workspace_capacity_bytes(
    const LinearProfile& qkv, std::int32_t batch,
    std::int32_t min_tokens, std::int32_t max_tokens);

[[nodiscard]] std::size_t kda_input_proj_conv_record_workspace_capacity_bytes(
    const LinearProfile& qkv, std::int32_t batch,
    std::int32_t min_tokens, std::int32_t max_tokens);

} // namespace ninfer::ops
```

两个入口均支持 B=1..8、T 为正；`valid_columns` 为空或为 Device I32 `[B]`。
输出行为如下：

| 出口 | history 效果 | records | Q/K/V |
|---|---|---|---|
| update | 读取 source，向 destination 写 n[b] 步之后的最终三列 | 不要求 records | 有效前缀完整写入，无效后缀为 BF16 +0 |
| record | source history 始终只读 | 写 BF16 `[C,T,B]` 的有效前缀，无效后缀不变 | 同上 |

`conv_record[:,t,b]` 是该次物理执行在位置 t 会发布的最新 BF16 history 列。
Update 的最终 history 等于 `tail_3(source_history || conv_record[0:n[b]])`，其中
conv_record 指同次计算对应的 words，不要求 update 实际物化它。
在相同源、权重、输入、物理 T/B 下，update/record 的有效 Q/K/V 和对应 history
words 按同一计算实现保持逐 bit 一致。

Update 的 source/destination selectors 可为相同 Tensor，单行允许原位更新。
各行 destinations 互异，不能覆盖其他行的 source。实现须在覆盖原位 history 前
完成所有依赖旧三列的读取；长窗口并行计算也遵循此规则。
不同只读 rows 可以使用同一 source。Record 不写任何 state record。

x 的全部物理列需可读且为有效有限输入，投影可以计算 padding 列；只有 n[b] 的
前缀参加卷积/history 语义。两个入口都不对外暴露每个位置的完整 history snapshots。

### 4.3 存储与容量

History 字段内 `[C,3]` 连续，记录偏移由 I64 metadata 给出；字段起点至少 16 B 对齐。
x、Q/K/V、conv_record、conv weight 连续且至少 16 B 对齐，selectors 至少 4 B 对齐。
QKV 使用原生 Q8 codes/scales 对齐要求。History 的实际数据块与所有非 state
operands 和当前 scratch disjoint。输出和 records 彼此 disjoint，与输入、权重、
metadata 和 scratch disjoint；只读 metadata 可以共享 backing。

Capacity query 只使用无地址 profile、batch 和闭区间 `[min_tokens,max_tokens]`，
覆盖完整 projection/conv 及所有合法有效长度。调用使用 caller arena，结束后恢复
cursor；先前的 arena allocations 可以保存 operands。合法无 scratch 路线返回零。

## 5. 递推执行接口

### 5.1 单序列完整更新

保留 `include/ninfer/ops/kimi_delta_attention.h` 的现有单序列接口，用于完整窗口
执行和长 prefill。其显式 source/destination 形式为：

```cpp
namespace ninfer::ops {

[[nodiscard]] std::size_t kimi_delta_attention_workspace_capacity_bytes(
    std::int32_t qk_heads, std::int32_t value_heads,
    std::int32_t min_tokens, std::int32_t max_tokens);

void kimi_delta_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta,
    const Tensor& A_log, const Tensor& dt_bias,
    float lower_bound, float scale, MemoryArena& workspace,
    const Tensor& state_in, Tensor& state_out,
    Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops
```

输入使用第 2 节去掉 B 的视图，state 为连续 FP32 `[128,128,Hv]`，写出全部 T 步
outputs 和末端 state。`state_in/state_out` 可以完全同 view 或 disjoint。现有同一
state 参数的重载保持同样语义。长 prefill 的 recurrent/chunked 选择和 Spark 调优
由此入口内部负责；数学参考始终是完整 KDA 转移。

### 5.2 单列 state-pool update

扩展现有 batch update，对齐 GDN 的独立 source/destination selectors：

```cpp
namespace ninfer::ops {

void kimi_delta_attention_batch_update(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta,
    const Tensor& A_log, const Tensor& dt_bias,
    float lower_bound, float scale, ConstStateFieldView source,
    StateFieldView destination,
    Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops
```

T=1、B=1..8；source/destination views 的 record offsets 为 Device I64 `[B]`。每行从 source 做一次 recurrent 更新，
写 destination 和 out。允许原位，各行 destinations 互异且不能覆盖其他行 source；
只读 source 可以重复。只写 recurrent 字段，conv 由输入准备负责。该入口无全局
workspace。实施时替换旧单 selector 签名及其 callers。

### 5.3 Recurrent record

```cpp
namespace ninfer::ops {

struct KdaRecurrenceRecords {
    Tensor key;       // BF16 [128,Hq,T,B], before normalization.
    Tensor value;     // BF16 [128,Hv,T,B].
    Tensor log_decay; // FP32 [128,Hv,T,B].
    Tensor beta;      // FP32 [Hv,T,B], after sigmoid.
};

void kimi_delta_attention_replay_record(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta,
    const Tensor& A_log, const Tensor& dt_bias,
    float lower_bound, float scale,
    ConstStateFieldView initial_state, const Tensor& valid_columns,
    KdaRecurrenceRecords records,
    Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops
```

B=1..8、T 为正；initial_state 的 offsets 为 Device I64 `[B]`，非空 valid_columns 为 I32 `[B]`。
每行从指定只读 source state 开始，按 n[b] 个有效 inputs 产生 outputs 和四个
transition record planes。它不写任何 state record，不保存完整 recurrent trajectory。
有效 records 完整写入，无效后缀不变；out 的无效后缀为 BF16 +0。

Key/value records 保留实际输入 words；log_decay/beta records 保留内部实际 gate
words。每个元素由唯一写者发布，state tiles 不竞争重复写同一 record。
该入口无全局 workspace，内部轨迹与实现 scratch 不成为持久资源。

`k` 与 `records.key`、`v` 与 `records.value` 分别允许完全同 view，其他 record
重叠不允许。同 view 时复用既有 words，不重复拷贝。这样输入准备可直接将 K/V
写入对应 frame record plane，再以 reshape 后的 view 供递推消费。
该复用下，输入准备仍按 Q/K/V 合同将无效输出置零；record 入口保持后缀不变，
指保持它被调用时的内容。

### 5.4 共同存储要求

State record 内的 `[128,128,Hv]` 字段连续，字段偏移显式，起点至少 16 B 对齐。
Q/K/V/g/out 连续且至少 16 B 对齐；beta、A_log、dt_bias 自然对齐，record offsets 至少 8 B 对齐。
Record vectors 至少 16 B 对齐，beta record 至少 4 B 对齐；Core planner 提供更强的
256 B plane 对齐。

除声明的 state 原位和 K/V record 同 view 外，写入范围与所有其他 operands disjoint。
只读 operands 可在数学与形状允许时共享 backing。Operands、metadata、workspace
均属于 execution device。Device offset/length 的合法值由调用者保证，Op 检查公开
metadata 和地址域；不为验证 device 值增加 Host readback。

## 6. Prepared all-layer fold

声明位于 `include/ninfer/ops/kda_replay.h`，与当前 GDN 同时提供 Host 和 Device 行描述：

```cpp
namespace ninfer::ops {

struct KdaReplayFoldRow {
    std::int64_t source_offset;
    std::int64_t destination_offset;
    std::int32_t commit_columns;
};

class KdaReplayFoldPlan {
public:
    KdaReplayFoldPlan(const KdaReplayRecords& records,
                      LinearAttentionStateAllLayersView states);

    void execute(std::span<const KdaReplayFoldRow> rows,
                 CudaExecutionView execution) const;

    void execute(const Tensor& source_offsets, const Tensor& destination_offsets,
                 const Tensor& commit_columns, CudaExecutionView execution) const;
};

} // namespace ninfer::ops
```

构造时验证固定 records/state 的几何、dtype、device、strides、alignment、地址域和
不重叠。Plan 借用稳定 backing，不分配设备存储。State 的 layers、heads、conv
channels/width 与 records 对应；conv_width=3、Dk=Dv=128。

`rows[b]` 对应 physical record row b，Host rows.size() 或 Device metadata 长度为
B，`1<=B<=R`。Selectors 是 arena-relative byte offsets，m 在 `[0,T]`；调用者保证 `m[b]<=n[b]`，
且 source 内容仍是产生该行 records 的原始版本。

All-layer state 的字段、层和记录起点遵循 Core 的 256 B 对齐。Device offsets 为连续、至少
8 B 对齐的 I64 `[B]`，commit_columns 为至少 4 B 对齐的 I32 `[B]`；metadata 与所选字段及
records disjoint。Host 形式检查行数、offset、m 和行间写入关系；Device
形式检查 Tensor metadata/地址域，值与行间关系由调用者保证，不做 Host readback。

### 6.1 正提交与零提交

对每个 m>0 的 row、每个 layer：

```text
S = source.recurrent
for t in [0,m):
    k = normalize(record.key[t])
    alpha = exp(record.log_decay[t])
    S = KDA_transition(S, k, record.value[t], alpha, record.beta[t])

destination.recurrent = S
destination.conv = tail_3(source.conv || record.conv[0:m])
```

Fold 不读取 query、raw gate logits、权重、A_log/dt_bias，不运行 projection、
convolution、SiLU、sigmoid 或 readout。Prediction/delta 依赖当前重放 state，
按同一 transition 重算。

Conv 更新是 words 提取：m=1 取旧末两列加 record[0]；m=2 取旧末列加 record[0:2]；
m>=3 取 record[m-3:m]。原位时保留所需旧列之后再写。

m=0 是严格 no-op：不读 state/record payload，不写两个字段，即使 source 与
目的不同也不复制。调用方继续采用原 image。Selectors 仍遵守合法范围与行间关系。
允许单行原位 fold；destinations 互异，不能覆盖其他活动行的 source。

一次调用覆盖本 rank 的全部 KDA layers，只写正提交行的目标字段。Records 只读，
拒绝后缀不读取；无需全局 workspace，也不要求 Host 按 layer 循环提交。

### 6.2 精确关系

Ordinary batch update、recurrent record 和 fold 共享同一有限精度 transition。
从相同 source，使用同次 record 调用实际消费的 represented inputs：

- record 的有效输出与逐步 ordinary batch update 的输出逐 bit 一致；
- fold(m) 的 recurrent words 与上述轨迹第 m 步逐 bit 一致；
- conv words 与同次物理输入准备在第 m 步对应的 history 一致。

共享范围包含 key normalization、gate/exp、prediction reduction、乘加和 FP32
state 边界。优化可改变实现，但这三个入口须作为一致的数值组合维护。
输入投影不在接受长度 m 上重跑；更短 T/B 的另一次输入准备不是精确前缀参考。

长 prefill 的 chunked 路径按独立数学精度验收，不与 recurrent 路径要求逐 bit
相同；它实际发布的 FP32 末端 state 可以作为下一轮 source。

## 7. 执行组合与 Graph

| 调用场景 | 输入准备 | 核心 | 最终状态 |
|---|---|---|---|
| 完整 prefill 窗口 | `kda_input_proj_conv_update` | 现有单序列 `kimi_delta_attention`，内部选择 recurrent/chunked | 两项直接写目标 image |
| Ordinary 单列 | `kda_input_proj_conv_update` | source/destination `batch_update` | 两项直接写目标 image |
| Verify | `kda_input_proj_conv_record` | `replay_record` | 接受长度确定后 all-layer fold |

Prefill 单序列输入使用实际有效 T；长窗口的 checkpoint 边界由调用方拆分或显式
捕获。普通单列每个活动 row 都有一个有效 input。纯控制投影、输出 norm 和 O
在三条路径中使用同样的数学组合。

Verify 的 source image 保持只读，frame 保存 physical row 到 source image/version
的对应关系。最终 m 是采用的 verify inputs 数，包含已执行的 anchor/接受前缀，
由模型的接受结果与发布边界给出。Fold 完成、模型其他字段也就绪之后，Session 才
采用目标 image；records 在 fold 和所有消费者退役后复用。

Graph 使用规则：

- 输入准备和 record 读取当前 Device selectors/valid lengths；固定地址下可更新内容。
- Device fold 的两个 I64 offsets 与一个 I32 commit metadata 每次 replay 读取当前值。全零 m 在 kernel
  内跳过，不通过 Host 读取值决定是否放入 Graph。
- Host fold 将 row descriptors 按值提交；capture 后改变 Host vector 不改变图中值。
  全零 Host rows 可直接返回。需要每轮改变 descriptors 的图使用 Device 形式。
- Plan、records 和 state backing 在引用它们的提交及 Graph 生命周期内保持有效。
  T、R、几何和布局固定于相应执行 variant；同一 backing 上的下一轮写入等待旧消费者。

TP2 每 rank 独立处理本地 32 heads 和 34 层，source/destination record offsets 可以不同，
逻辑 batch 和采用长度一致。模型负责完整输出的归约和状态采用。GLM 的 MTP 层是
MLA，不增加 MTP KDA records；这套 ReplaySSM 服务 Main 的 speculative verify。

## 8. 实现组织与复用

| 位置 | 改动 |
|---|---|
| `src/core/kda_replay_records.h/.cpp` | 五个 record planes 的布局、绑定、layer views |
| `include/ninfer/ops/kda_input_proj.h` | 输入 update/record 及容量查询 |
| `include/ninfer/ops/kimi_delta_attention.h` | source/destination batch update、新增 recurrent record；保留完整窗口入口 |
| `include/ninfer/ops/kda_replay.h` | Host/Device all-layer fold plan |
| `src/ops/linear_attention/kimi_delta_attention/input_conv/` | Q8 projection、FP32 conv、SiLU、history/record 输出 |
| `src/ops/linear_attention/kimi_delta_attention/replay/` | record/fold、prepared validation、Spark 分派 |
| 现有 KDA recurrent/chunked 实现 | 提取共同 transition，接 source/destination 与 record 输出；保持原有数学 |

新增内容留在 `linear_attention/kimi_delta_attention`。Projection 模板复用 Linear，
基础数值/内存操作复用 `ops/common`；输入融合拥有自己的调优策略。StateImage、
linear-attention state、layout builder 和 Graph 基础直接复用。KDA 的向量 decay
record 使用自己的明确字段，不改变 GDN scalar gate plane 的含义。

现有 `sigmoid_gated_rmsnorm()`、control Linear/split 和 O Linear 不再增加 KDA
专用副本。容量、state/record 视图绑定与 compute 支持检查各由所属模块负责。

## 9. 实施与验证

实施顺序为 records 容器和输入 update/record、递推 source/destination 与 record、
Host/Device all-layer fold，随后完成生产链调优。容器和算子的构建注册、现有 KDA
callers、tests/bench 随相应接口一起更新。

验证按以下几类完成，复用现有 KDA、GDN 和 core layout 的测试组织：

1. Projection/conv/SiLU、gate、递推及 state 对独立数学参考；FP32 conv 权重与 tap
   顺序、直接 gamma、逐 channel decay、padding 输出分别按合同检查。
2. Records 的实际 words、同物理 T/B 的 update/record 一致性；K/V 同 view 与独立
   backing 两种合法形式；固定 R 下 B<R 的跨层寻址。
3. Fold 与逐步 recurrent state/output 的精确关系，覆盖 m=0/1/2/3/完整前缀、混合
   接受长度、非连续 record offsets、原位和独立目的、显式 state strides。
4. 固定地址的 Graph 重放改变 inputs、selectors、valid lengths 和 Device commit
   lengths，检查未使用 record 后缀及非目标 state 字段保持不变。

数值要求按公开数学确定，不随私有 kernel 改精度阈值。精确 replay 检查和完整 KDA
数学检查分别保留。

性能使用完整公共 Op：input update/record、batch update、record、all-layer fold
以及长窗口 core 分别测量。Decode/verify 看不同 B/T/采用长度；长 prefill 独立看
chunked 的计算与 workspace 流量。State 带宽按实际 state 读写计，record 写入和
fold 重放另计；容量推导见[并行资源](parallel.md#5-kda-records-容量与成本)。
