# MTP 支持预先准备工作第二部分：底层算子支持

> GQA API note (2026-07-04): the public `gqa_attention_prefill` /
> `gqa_attention_decode` split described in older sections has been superseded by the unified
> `gqa_attention(..., positions, ...)` contract. Keep the old names below as historical analysis of
> the pre-refactor implementation, not as live API.

本文承接 [Part 1](2026-07-02-mtp-foundation-part1-design.md) 的 q5090 v3
MTP_DRAFT layout，只讨论为了执行 MTP head 和后续 speculative target
verification 需要补齐的 L1/operator 能力。

本文不修改 q5090 权重格式，不修改 Python converter/ref model，不接入
Engine、C++ model card、scheduler 或 acceptance/rejection sampler。本文把
GDN/SSM/conv 的状态提交问题拆清楚：算子层只定义输入输出和 small-T 能力，
accepted-prefix commit/rollback 仍是 Engine/runtime 的状态生命周期问题。

Part 1 固定 MTP_DRAFT 的五个 W8 logical linears：

```text
mtp.fc
mtp.attn_in.w8
mtp.o_proj
mtp.mlp.gateup.w8
mtp.mlp.down
```

本文关心的是这些权重进入计算图后，L1 层需要能处理哪些 qtype、shape、
view、caller-owned cache/state 边界，以及为了一次处理多个 draft/verify
token，需要给现有 GEMV/GQA/GDN 算子补哪些 small-T 形态。

## 1. Scope

### 1.1 In Scope

1. `W8G128_F16S` row-split linear 的最小正确性支持和性能扩展方向。
2. MTP 五个 W8 shape family 的精确维度、payload 规模和 T regime。
3. MTP 支持所需算子的分类：从头实现并调优、低成本契约/实现改造、
   T=1 已有但 small-T 需要深调优。
4. MTP FC 输入 concat/pack 的计算公式和 shape。
5. `mtp.attn_in.w8` 输出行区间到 Q/K/Gate/V 的 split 约定，以及 T>1
   时的 contiguous 限制。
6. MTP full-attention/GQA append-batch 的 caller-owned-buffer L1 contract。
7. target speculative verify 为一次处理多个 token 需要的 small-T operator
   形态。
8. GDN/SSM/conv 在 target verify 中的算子边界和状态输入输出建议。
9. 可复用的已有 elementwise/reduction ops，以及后续性能 fused op 候选。
10. 数值测试、状态测试、benchmark 和 profiling 的准备清单。

### 1.2 Out Of Scope

1. 不修改 q5090 v3 layout、block/segment/fusion group 约定；这些属于 Part 1。
2. 不修改 Python converter、verifier 或 ref model；这些属于 Part 1。
3. 不修改 Engine、runtime scheduler、CUDA graph、KV cache allocation ownership、
   proposal/verification state machine 或 target hidden-state handoff。
4. 不接 C++ model card 或 MTP forward。
5. 不在 L1 kernel 内实现 acceptance/rejection 逻辑。

本文所有 attention/cache 描述都是算子输入输出边界：`cache` 是 caller-owned
KV buffer/slice，算子不得分配、扩容、推进或解释 runtime state。GDN/conv/SSM
同理，算子可以有 state input/output contract，但不能决定接受多少 draft token。

## 2. Current Operator Baseline

### 2.1 Linear

当前公共 API 是：

```cpp
void linear(const Tensor& x, const Weight& w, Tensor& out,
            WorkspaceArena& ws, cudaStream_t stream);
```

shape contract：

```text
x   [K,T] BF16
w   [N,K] Weight
out [N,T] BF16
ne[0] is the fastest dimension
```

源码状态：

- `include/qus/kernels/linear.h` 定义了上面的公共 API。
- `src/kernels/linear/linear.cpp` 当前只 validation/dispatch dense
  `BF16_CTRL/FP32_CTRL` 和 row-split `Q4G64_F16S/Q5G64_F16S/Q6G64_F16S`。
- `src/kernels/linear/plan/linear_plan.{h,cpp}` 当前没有
  `LinearFormat::W8G128_RowSplit`，也没有 `5120x10240`、
  `14336x5120` 等 MTP shape family。
- `src/kernels/linear/codec/linear_codec.cuh` 只有 Q4/Q5/Q6 codec，并且
  `kGroupK=64`。
- `tests/kernels/test_linear.cpp` 当前有 `unsupported_qtype_validation()`，
  明确把 `W8G128_F16S` 当作 unsupported 行为。
- q5090 格式层已经认识 W8：`include/qus/core/tensor.h` 有
  `QType::W8G128_F16S = 3`，`src/core/weight_store.cpp` 对 W8 使用
  128 code bytes/group 和 0 high bytes。

现有 linear regime：

| regime | current behavior |
|---|---|
| `T == 1` | Q4/Q5/Q6 generic GEMV 或 hot-shape GEMV |
| `1 < T <= 16` | `linear_rowsplit_gemm_multistep_launch()` |
| `T > 16` | `linear_rowsplit_gemm_mma_launch()`，K 不满足对齐时回退 small-T |

这些机制可以作为 W8 的设计模板，但不能直接复用，因为 W8 是 K128 group、
int8 code plane、无 high plane。

### 2.2 GQA Attention

当前 GQA API：

```cpp
gqa_attention_prefill(q [256,24,T],
                      k [256, 4,T],
                      v [256, 4,T],
                      cache_offset,
                      out [256,24,T])

gqa_attention_decode(q [256,24,1],
                     k [256, 4,1],
                     v [256, 4,1],
                     pos [1],
                     out [256,24,1])
```

源码状态：

- `src/kernels/wrapper/gqa_attention.cpp` 固定 `head_dim=256`、`q_heads=24`、
  `kv_heads=4`、`scale=0.0625`。
- prefill wrapper 接受 `T > 0` 和 `cache_offset`，并校验
  `[cache_offset, cache_offset+T)` 不越界。
- `src/kernels/launcher/gqa_attention_prefill.cu` 先 launch fill kernel 写 K/V，
  再 launch causal attention kernel。
- `src/kernels/kernel/gqa_attention_prefill.cuh` 的 causal mask 本质是
  `key <= cache_offset + token_index`。
- decode wrapper 严格要求 `T == 1`，使用固定 `kGqaDecodeSplits=192` 的
  split-KV partial/reduce scratch。

结论：现有 prefill 已经具备 append-contiguous correctness 语义，但不是
small-T 最优路径；现有 decode 是 T=1 性能路径，不能一次输出多个 token。

### 2.3 GDN / SSM / Conv

MTP head 本身不使用 GDN。GDN/SSM/conv 的问题来自 target speculative
verification：如果主模型一次验证多个 draft token，target GDN state 必须只提交
accepted prefix。

源码状态：

- `include/qus/core/state_store.h` 的 `GdnState` 每个 GDN 层持有：
  - `conv [10240,3]` BF16；
  - `ssm [128,128,48]` FP32。
- `causal_conv1d_decode()` 明确要求 `T == 1`。
- `causal_conv1d_prefill()` 支持 `[C,T]`，但会把最后 3 个输入直接写回
  `conv_state`。
- `gated_delta_rule_recurrent()` 支持任意正 `T`，kernel 内循环 `t=0..T-1`
  并在末尾写回 `ssm_state`。
- `gated_delta_rule_chunked()` 对 64-token full chunks 用 chunk kernel，小于
  64 的 tail 走 recurrent。
- `gdn_in_ab_gated_prefill()` 支持 `x [5120,T] -> g/beta [48,T]`。
- `gdn_in_vz_decode()` 只有 T=1 fused decode API；T>1 correctness 可以先走
  通用 `linear()` 分别计算 V/Z。

结论：GDN 数学算子基本能算 small-T，但公共 decode fused API 多数固定 T=1，
并且现有 in-place state API 会无条件提交整段 T。

## 3. Two Different Multi-Token Problems

MTP 支持里有两个容易混淆的 multi-token 问题。

### 3.1 MTP Draft Forward

MTP head 是 draft model。第 `i+1` 个 draft token 的输入 token id 依赖第 `i`
个 draft logits 的 argmax，因此连续生成多个未知 draft token 不能简单用一次
MTP forward 全部完成。

MTP 算子仍然需要支持 `T > 1`，原因是：

- MTP KV prefill 可以处理一段已知 token/hidden 序列；
- ref/model 验证和 benchmark 需要批量输入；
- 未来可以对已知 draft prefix 或 target hidden batch 做 small-T 计算；
- GQA append-batch primitive 可以复用给 target verify 的 small-T path；
- W8 small-T 只属于 MTP head 的 known-prefix/ref/benchmark/proposal 实验，
  target verifier 主路径仍是 TEXT_CORE 的 Q4/Q5/Q6 linears、GQA 和 GDN。

MTP draft forward 本身不读写 target KV，不读写 target `GdnState`，只维护独立的
MTP full-attention KV namespace。

### 3.2 Target Multi-Token Verification

target verifier 一次验证多个 draft token 时，输入通常是：

```text
[last_sampled_token, draft_1, ..., draft_k]
```

如果 `draft_count` 是 1..5，那么 target verify 的 operator T 通常是 2..6。
如果 runtime 设计成 last sampled token 已经单独提交，则 small-T 至少需要覆盖
1..5。底层算子建议统一覆盖：

```text
T_verify in [1,6]
```

这里的“单次验证多个 token”不意味着整个 64 层 target model 变成一个 CUDA
launch。对 L1 来说，它意味着每个相关 operator 要能在一个调用中消费
`T=1..6` 的 token 维，避免退化成每 token 重放整条 decode graph。

## 4. Operator Classification

本节把后续所有算子需求先按工程性质分类。这里的分类不是执行阶段编号，而是
评估实现成本和调优风险的依据：

- Class A: 需要从头补齐 kernel family，并以性能为第一等目标调优。
- Class B: 可以复用或小改已有实现，重点是 API/shape/state 契约，调优代价不高。
- Class C: 已有 T=1 decode 或已有 correctness path，但要让 `T=2..6`
  成为高性能 small-T path，需要专门的 kernel/epilogue/fusion 调优。

本文里的“single-call small-T”默认指一次 L1 API 调用消费多个 token。它可以由
多个 CUDA kernel launch 组成，例如 fill + attention 或 partial + reduce。
“strict one-launch”特指一个 CUDA kernel launch 完成 append、attention 和 reduce；
这只有在明确写出时才成立。二者不能混用，否则会把 correctness path 和深度性能
kernel 混成一个任务。

### 4.1 Class A: New Kernel Families That Need Tuning

这些能力目前没有等价实现。即使可以借鉴现有 Q4/Q5/Q6 或 prefill 代码结构，
也不能只靠 wrapper 改造完成。

| operator family | why Class A | minimum shape/dtype contract | tuning risk |
|---|---|---|---|
| W8G128 row-split codec | `linear` 计划层和 codec 层没有 W8 format；现有 codec 全是 K64 Q4/Q5/Q6 | `W8G128_F16S`, row-split, K group 128, FP16 scale, no high plane | 所有 W8 kernel 的基础；scale boundary 是 K128，不能沿用 K64 数据路径 |
| W8 GEMV for MTP decode | `tests/kernels/test_linear.cpp` 仍把 W8 当 unsupported；MTP decode `T=1` 必走 W8 | `[5120,10240]`, `[14336,5120]`, `[5120,6144]`, `[34816,5120]`, `[5120,17408]`, `T=1` | 大矩阵 decode 延迟敏感；需要按 shape 决定 rows/block、K loop、scale/code staging |
| W8 small-T GEMM/multistep | 现有 multistep 只实例化 Q4/Q5/Q6 codec；W8 需要 K128 weight reuse | 同上，`T=2..6` first-class，建议同时覆盖 `T<=16` | MTP known-prefix/ref/benchmark 的核心路径；要避免对已知 MTP 输入按 token 重放 W8 GEMV |
| W8 LargeT MMA | 现有 MMA path 的 dequant tile 和 codec 以 K64 low-bit 权重为前提 | 同上，`T>16` MTP prefill/benchmark | BK 应优先对齐 128，避免一个 W8 scale group 被拆进不同 MMA stage |
| strict one-launch GQA append verify | 现有 prefill 是 fill + attention 两 launch；decode 是 partial + reduce 两 launch | `q [256,24,T]`, `k/v [256,4,T]`, cache BF16, `T=1..6` | 若要求单 CUDA kernel 内完成长上下文 split/reduce，就没有低成本改造路径 |

Class A 的边界是“数学主算子本身不存在”。`mtp.attn_in.w8` fused split、W8
residual epilogue、W8 gateup+SiLU 虽然也会产生新 kernel，但它们建立在 W8
linear 已经存在的基础上，首轮应按 Class C 的性能融合处理，而不是阻塞
Class A correctness。strict one-launch GQA append verify 是例外：它不是
correctness 必需项，但如果产品目标明确要求一个 CUDA launch 完成验证 attention，
它应按 Class A 风险评估，因为跨 split softmax reduce 没有现成全局同步路径。

### 4.2 Class B: Low-Cost Contract Or Wrapper Adaptations

这些能力已经有数学实现，或者只是数据搬运。首要工作是把 shape、contiguity、
state input/output 约定写清楚，让上层可以正确组合；不应先投入深度性能调优。

| operator / boundary | current evidence | required adaptation | why low-cost |
|---|---|---|---|
| `mtp_pack_fc_input` | 没有专用 op，但公式只是两个 `[5120,T]` 拼成 `[10240,T]` | 新增 BF16 2D copy/pack kernel，或先由 caller 分两段 copy | 无乘加；性能风险主要是少量内存带宽和 launch 开销 |
| W8 metadata/planning/test plumbing | q5090 core already recognizes `W8G128_F16S`; L1 dispatch rejects it today | add validation, format enum, shape families, fixture/test/bench qtype parsing | plumbing is mechanical; the actual W8 math kernels remain Class A |
| `mtp_split_attn_in` compact split-copy | `attn_in_out [14336,T]` row slice 在 `T>1` 不 compact | copy 成 q/k/gate/v 四个 compact tensor | 纯数据重排；正确性优先，后续再考虑 fused W8 output |
| `gqa_attention_prefill` as append correctness | prefill 已有 `cache_offset` 和 batch causal mask | 以 append-contiguous API 暴露或临时复用 prefill wrapper | 语义已经成立；问题是 launch/tiling 不是 small-T 最优 |
| KV cache batch fill/append contract | prefill fill kernel 已按 `cache_offset + token` 写入 cache | bounds/overflow validation 保持 caller-owned cache，不推进 `kv.pos` | 物理写 suffix 低风险；accepted/rejected 逻辑提交属于 runtime |
| `gdn_pack_conv_input` / `gdn_split_conv_output` | GDN prefill 当前在 model 层用 `cudaMemcpy2DAsync` pack/extract | pack `q [2048,T]`, `k [2048,T]`, `v [6144,T]` -> `[10240,T]`; split conv output back to compact q/k/v | 纯 BF16 数据搬运；必须明确 compactness contract，避免把 model-local helper 当隐式 L1 能力 |
| `causal_conv1d_sequence` correctness | prefill 支持 `[C,T]`，decode 支持 T=1 | 给 prefill 增加 `state_in/state_out` 变体；当前 in-place API 可作为同指针 wrapper | 数学已覆盖 T>1；只需避免 in-place state 过早提交 |
| `gated_delta_rule_recurrent_state` correctness | recurrent wrapper 支持任意 `T>0`，kernel 内循环 T | 拆开 `state_in` 和 `state_out` 指针；当前 in-place API 可作为同指针 wrapper | 小 T 不需要 chunked path；接受/回滚由 Engine 管 |
| `gdn_in_ab_gated_prefill` | 已有 `[5120,T] -> g,beta [48,T]` prefill path | 在 target verify path 直接调用 T=1..6 | 输出很小，已有 prefill test/bench 基础 |
| generic Q4/Q5/Q6 `linear()` for target small-T | plan 层已有 `T<=16` multistep 和 `T>16` MMA | target verify correctness 先用 generic linear | 不是 MTP W8 新 qtype；性能融合另列 Class C |
| elementwise/reduction ops | `rmsnorm`, `rope`, `l2norm`, `gdn_gating`, `sigmoid_gate_mul`, `silu_and_mul`, `residual_add`, `argmax` 已有 T 维测试 | 只要求输入 compact contiguous；必要时补 T=2..6 shape coverage | kernel 本身 shape-generic，主要风险是上游传入 strided view |
| `lm_head` + `argmax` verify output | Q6 lm_head 走已有 `linear()`，`argmax` 支持 `[vocab,T]` | 确认 `T_verify` 输出列和 ids 输出契约 | 第一阶段无需 fused sampler |

Class B 的原则是先让 MTP/verify 的 operator graph 可以“一次调用处理 T 个 token”，
但不承诺这是最终最快路径。这里最重要的边界是：算子可以接受 `state_in` 并产生
`state_out`，但不能知道 accepted prefix，也不能推进 runtime 的 KV/GDN state。

### 4.3 Class C: Existing T=1 Or Correctness Path, But Small-T Needs Deep Tuning

这些算子是性能风险最大的部分。它们通常有很快的 T=1 decode kernel，或者有能算
T>1 的 prefill/generic path；问题是 `T=2..6` 会落在两者之间，不能简单认为任何
一边都足够快。

| operator family | existing path | high-performance small-T target | tuning risk |
|---|---|---|---|
| GQA append-batch attention | decode 是 T=1 split-KV；prefill 是 T>1 两 launch causal attention | `gqa_attention_append_contiguous_small_t`，一次处理 `T_new=1..6`，每个 token 独立 reduce partials | 必须同时写入新 K/V、允许 token i 看见 batch 内前缀、禁止看未来 token；cache window 每列不同 |
| `gdn_in_vz_batch` | `gdn_in_vz_decode()` 是 T=1 dual Q5 GEMV；T>1 可用两个 generic linear | small-T dual-output Q5 path，`[5120,T] -> v/z [6144,T]` | 两个权重、两个输出、T 小且 N 大；generic path launch/weight traffic 可能偏重 |
| Q5 `linear_residual_add` small-T | decode fused op 只覆盖 Q5 T=1 out/down；T>1 可 generic linear + residual | small-T residual epilogue or fused multistep | target verify 每层多次用；多 launch residual 可能显著累积 |
| Q4 text `mlp_gate_up_silu` small-T | decode fused gateup+SiLU 只覆盖 T=1；T>1 可 generic gateup + `silu_and_mul` | small-T gate/up projection with activation epilogue | gateup 输出 `[34816,T]` 很大，写回再读会放大带宽 |
| causal conv tuned small-T | decode T=1 一 launch；prefill T>1 输出+state 两 launch | T=2..6 sequence kernel with explicit state_out, preferably one output/state pass | correctness 低风险，但 state copy、launch 数和 C=10240 带宽仍需 profiling |
| GDN recurrent tuned small-T | recurrent 已支持 T>1；chunked 针对 T>=64 | T=2..6 recurrent state-in/out variant with measured occupancy/register pressure | 数学可复用，但 SSM state `[128,128,48]` 读写大，不能假设状态管理开销免费 |
| fused MTP `attn_in` output | W8 linear + split-copy correctness path | W8 `mtp.attn_in.w8` 直接写 q/k/gate/v compact outputs | 消除 `[14336,T]` 中间写回和 split-copy，但会复杂化 W8 epilogue |
| fused MTP MLP/residual epilogues | W8 linear + elementwise correctness path | W8 gateup+SiLU、W8 o/down residual epilogue | 属于性能阶段；不应挡住 W8 基础 kernel correctness |

Class C 的判断标准是：不是“能不能算”，而是“是否能让 speculative verify 的
small-T path 真正比逐 token decode replay 更便宜”。没有 nsys/ncu 数据前，
这些不应该被写成第一阶段 correctness blocker。

Class C 里的 GQA append-batch small-T 推荐先按两 launch split-KV 设计：
partial kernel 产生带 token 轴的 partials，reduce kernel 独立 reduce 每个
`(token_i, q_head, d)`。这个目标仍是 high-performance small-T，但不承诺 strict
one-launch。strict one-launch 版本只有在 profiling 证明 launch/reduce overhead
超过 split-KV 并行度损失时才值得单独立项。

### 4.4 Priority Matrix

| priority | class | items | reason |
|---|---|---|---|
| P0 | A | W8 codec + W8 GEMV + W8 small-T/LargeT correctness path for MTP | 没有它们，MTP head 不能执行 |
| P0 | B | FC pack、attn split-copy、compactness validation | 没有它们，T>1 输出会被错误 view 消费 |
| P0 | B | GQA prefill-as-append correctness contract | MTP KV prefill 和 target verify 先要有正确 append 语义 |
| P1 | B | GDN/conv/recurrent state-in/out contract | 让 target verify 可以评估 suffix 而不污染 committed state |
| P1 | C | GQA append-batch small-T two-launch kernel | speculative verify 性能的最大独立 attention 风险 |
| P1 | C | target Q4/Q5 small-T fused paths | 避免每层落到 generic linear + elementwise 多 launch |
| P2 | C | MTP W8 fused epilogues and direct split output | 在 W8 基础性能达标后减少内存流量 |
| P2 | C | GDN tuned small-T/fusions | 先度量 GDN state/conv 开销，再决定是否深调优 |
| P3 | A | strict one-launch GQA append verify | 只有明确要求单 CUDA launch 或 profiling 证明必要时才做 |

## 5. Required: W8G128 Row-Split Linear

Classification:

- Class A for W8 codec, W8 T=1 GEMV, W8 small-T multistep/GEMM and W8 LargeT MMA.
- Class C for W8 fused epilogues or direct compact multi-output variants after the
  base W8 linear path exists.

MTP 的五个 linear 全部是：

```text
qtype:       W8G128_F16S
layout:      ROW_SPLIT
group_size:  128
scale_dtype: FP16
high plane:  none
```

W8 codec：

```text
K group size: 128
code plane:   int8 codes, 128 bytes per row per group
scale plane:  fp16 scale, 2 bytes per row per group
dequant:      w = float(int8_code) * fp16_scale
code range:   [-127, 127]
```

Runtime descriptor validation should check:

```text
w.qtype             == W8G128_F16S
w.layout            == RowSplit
w.group             == 128
w.group_size        == 128
w.q5090_scale_dtype == FP16
w.qhigh             == nullptr
w.high_plane_bytes  == 0
w.padded_shape[0]   == w.shape[0]
w.padded_shape[1]   == align_up(w.shape[1], 128)
```

Payload size formula:

```text
groups       = padded_K / 128
code_bytes   = N * groups * 128
scale_offset = align_up(code_bytes, 256)
scale_bytes  = N * groups * 2
payload_min  = scale_offset + scale_bytes
```

### 5.1 MTP W8 Shape Families

These are the exact W8 matrix families needed by the Part 1 target layout:

| logical op | shape `[N,K]` | groups | payload bytes | common regimes |
|---|---:|---:|---:|---|
| `mtp.fc` | `[5120,10240]` | 80 | 53,248,000 | MTP prefill, MTP decode |
| `mtp.attn_in.w8` | `[14336,5120]` | 40 | 74,547,200 | MTP prefill/decode |
| `mtp.o_proj` | `[5120,6144]` | 48 | 31,948,800 | MTP prefill/decode |
| `mtp.mlp.gateup.w8` | `[34816,5120]` | 40 | 181,043,200 | MTP prefill/decode |
| `mtp.mlp.down` | `[5120,17408]` | 136 | 90,521,600 | MTP prefill/decode |

Total W8 payload is about 431,308,800 bytes, or 411.3 MiB, excluding small BF16
norm tensors and alignment outside each payload.

Existing TEXT_CORE shape families are useful references but not sufficient:

- `mtp.fc [5120,10240]` is new.
- `mtp.attn_in.w8 [14336,5120]` differs from TEXT `attn_in.q4/q5 [7168,5120]`.
- `mtp.mlp.gateup.w8 [34816,5120]` has the same dimensions as TEXT gateup but a
  different qtype.
- `mtp.o_proj [5120,6144]` and `mtp.mlp.down [5120,17408]` match existing dimensions
  but need W8 kernels, not Q5 kernels.

### 5.2 Linear Dispatch Changes

Minimum L1 changes:

```text
LinearFormat:
  W8G128_RowSplit

ShapeFamily:
  MtpFc5120x10240
  MtpAttnIn14336x5120
  existing Out5120x6144 reused with W8 format or named W8 variant
  existing MlpGateUp34816x5120 reused with W8 format or named W8 variant
  existing MlpDown5120x17408 reused with W8 format or named W8 variant

LinearPolicyId:
  GenericW8RowsplitGemv
  W8RowsplitGemmMultistep
  W8RowsplitGemmMma
  optional tuned W8 shape policies
```

Recommended regimes:

| regime | required behavior |
|---|---|
| `T == 1` | W8 GEMV, decode correctness and first performance target |
| `2 <= T <= 6` | W8 small-T single-call GEMM/multi-step GEMV for MTP known-prefix/ref/benchmark coverage |
| `7 <= T <= 16` | can share small-T path until profiling says otherwise |
| `T > 16` | W8 tensor-core GEMM or generic correct GEMM for MTP/prompt prefill |

The existing Q4/Q5/Q6 threshold is 16. W8 should start with the same split for
correctness and then use `bench/linear_op_bench.cu` data to tune it.

### 5.3 W8 Kernel Notes

Correctness-first W8 GEMV can mirror the generic low-bit GEMV structure, but with a
new codec:

```text
W8Codec:
  kGroupK = 128
  no high plane
  load_group: int8[128] * fp16 scale -> float[128]
  load_quad: each warp lane owns 4 int8 values for K128
```

For small-T, a warp-per-row multi-step kernel should dequantize each K128 group once
per row and reuse the weights across a token tile, for example `kTt=8`. This is the
important MTP-side path for known-prefix/ref/benchmark `T=2..6`; it is not the
target verifier's TEXT_CORE path.

For LargeT, W8 MMA should use a BK that respects the K128 scale boundary. A BK=128
variant avoids splitting one W8 scale group across MMA stages. This differs from the
current Q4/Q5/Q6 MMA path, which is built around K64.

## 6. Required: MTP FC Input Pack / Concat

Classification: Class B. This is a pure BF16 packing operator, not a math kernel.

MTP first combines normalized embedding and normalized hidden state:

```text
e[:, t] = RMSNorm_mtp_embedding(Embed[token_t])
h[:, t] = RMSNorm_mtp_hidden(hidden_t)

u[:, t] = concat(e[:, t], h[:, t])    # [10240,T]
z[:, t] = mtp.fc(u[:, t])             # [5120,T]
```

The concat operator is a deterministic BF16 pack along the row dimension:

```text
for t in 0..T-1:
  for i in 0..5119:
    u[i, t]        = e[i, t]
    u[5120 + i, t] = h[i, t]
```

Required op contract:

```text
mtp_pack_fc_input(embedding_norm: [5120,T],
                  hidden_norm:    [5120,T],
                  out:            [10240,T])
```

This op performs no arithmetic. A simple 2D copy kernel is enough for correctness.
Later, this can be fused with RMSNorm output staging or directly with the W8
`mtp.fc` input load path.

## 7. Required: Attention Input Split

Classification:

- Class B for explicit compact split-copy from `[14336,T]` into q/k/gate/v.
- Class C for a W8 `mtp.attn_in.w8` kernel that writes q/k/gate/v compact outputs
  directly.

`mtp.attn_in.w8` emits:

```text
attn_in_out [14336,T]

q    rows [0,6144)      -> [256,24,T]
k    rows [6144,7168)   -> [256, 4,T]
gate rows [7168,13312)  -> [256,24,T]
v    rows [13312,14336) -> [256, 4,T]
```

Converter-side q/gate de-interleave in Part 1 is what makes the split a pure row
range split at runtime.

Important implementation detail: for contiguous `[N,T]` tensors where `ne[0]` is
fastest, a row slice of `[14336,T]` is contiguous only when `T == 1` or when the
slice covers all rows. For `T > 1`, `attn_in_out[0:6144, :]` has a column stride of
14336 elements, while downstream operators currently expect their own compact
contiguous layout.

Therefore the correctness-first L1 contract should be one of:

```text
Option A: explicit split-copy
  mtp_split_attn_in(attn_in_out [14336,T],
                    q    [256,24,T],
                    k    [256, 4,T],
                    gate [256,24,T],
                    v    [256, 4,T])

Option B: fused W8 output
  W8 mtp_attn_in writes four compact output tensors directly.

Option C: stride-aware downstream ops
  rmsnorm/rope/gqa/sigmoid accept strided row views.
```

Recommended order:

1. Use Option A for correctness and simple tests.
2. Add Option B for decode/small-T performance.
3. Avoid Option C unless there is a clear performance reason, because it broadens
   every downstream op contract.

Zero-copy row views are valid only for `T == 1` or for consumers that explicitly
accept the original stride.

## 8. Required: GQA Append-Batch Attention

Classification:

- Class B for reusing or wrapping `gqa_attention_prefill()` as a correctness-first
  append-contiguous API.
- Class C for decode-derived `gqa_attention_append_small_t()` with token-axis
  partial/reduce buffers.
- Class A only for a strict one-launch long-context append+attention+reduce kernel.

The required L1 boundary is a caller-owned-buffer contiguous append primitive:

```text
gqa_attention_append_contiguous(q:            [256,24,T_new],
                                k:            [256, 4,T_new],
                                v:            [256, 4,T_new],
                                cache_offset: uint32,
                                cache:        caller-owned KV slice,
                                out:          [256,24,T_new])
```

Semantics:

```text
for i in 0..T_new-1:
  absolute_pos = cache_offset + i
  write k[:, :, i], v[:, :, i] into cache at absolute_pos
  out[:, :, i] = attention(q[:, :, i],
                           cache positions 0..absolute_pos)
```

Inside the same append batch, token `i` may attend to earlier new tokens `0..i`.
It must not attend to future new tokens `i+1..T_new-1`.

Shape constants:

```text
head_dim = 256
q_heads  = 24
kv_heads = 4
group    = 6 query heads per KV head
scale    = 1/sqrt(256) = 0.0625
```

KV cache layout should remain the existing layout:

```text
cache_k/cache_v [256,padded_context,4]
index(kv_head,d,pos) = d + 256 * (pos + padded_context * kv_head)
```

### 8.1 Correctness Path

Current `gqa_attention_prefill()` already implements this append-contiguous semantic:

1. fill K/V for `[cache_offset, cache_offset+T)`;
2. compute causal attention with `key <= cache_offset + token_index`;
3. output `[256,24,T]`.

It can be reused first for MTP KV prefill and target small-T verification. It is an
API call with two CUDA launches, not a decode-optimized single launch.

### 8.2 Small-T Optimized Path

For `T_new=1..6`, a decode-derived split-K append kernel is the right performance
target:

```text
partial_acc [256,24,T_new,splits]
partial_m   [24,T_new,splits]
partial_l   [24,T_new,splits]
out         [256,24,T_new]
```

Implementation choices:

1. prewrite the whole K/V batch to cache, then run token-aware decode attention;
2. or write K/V inside the partial kernel with guards;
3. window for token `i` is `cache_offset + i + 1`;
4. reduce partials independently for every `(token_i, q_head, d)`.

The existing decode path remains useful as the `T_new == 1` hot path. The append path
should be a phase-neutral primitive: MTP proposal passes MTP KV, target verification
passes target KV. The kernel must not advance `KVCache::pos`.

## 9. GDN / Conv / SSM Requirements For Target Verify

Classification:

- Class B for explicit `state_in/state_out` contracts on causal conv and recurrent
  GDN, plus direct reuse of GDN prefill/generic T>1 paths.
- Class C for tuned small-T GDN kernels and decode-only fusion replacements.
- Class A only for a future fused multi-stage GDN pipeline, and only after profiling
  proves the current sequence of kernels dominates verify latency.

MTP head itself does not need any GDN, SSM or conv operator. All GDN requirements in
this section are for target multi-token verification.

Target verify state semantics:

```text
verify input = [last_sampled_token, draft_1, ..., draft_k]
accepted draft count = a
commit input length = 1 + a
```

If a rejection happens, rejected suffix state must not remain in target KV, conv or
SSM. KV has a position axis and stale suffix can be logically ignored by rewinding
`kv.pos`. GDN `conv_state` and `ssm_state` are fixed rolling states without a position
axis, so they must be restored or recomputed to the accepted prefix.

### 9.1 Current Operator Capability

| op | current decode | current T>1 path | gap for verify |
|---|---|---|---|
| GDN Q/K projection | fused `in_qk_q4` via `linear()` T=1 | `linear()` T>1 works after generic linear path | performance small-T plan |
| GDN V/Z projection | `gdn_in_vz_decode()` T=1 | generic `linear()` for V and Z | optional `gdn_in_vz_batch()` |
| GDN conv pack/split | T=1 uses compact slices; T>1 currently uses model-local copy/extract helpers | define pack/split contract or caller-owned 2D copies | compactness/data-movement boundary |
| causal conv | `causal_conv1d_decode()` T=1 | `causal_conv1d_prefill()` T>1 | state is in-place and commits whole T |
| GDN A/B/g/beta | `gdn_in_ab_gated_decode()` T=1 | `gdn_in_ab_gated_prefill()` T>1 | performance only |
| l2norm | shape-generic | reusable | none |
| gated delta rule | recurrent T=1 | recurrent supports any T, chunked for T>=64 | public API is in-place state |
| GDN gated RMSNorm | shape-generic | reusable | none |
| GDN out proj | `linear_residual_add()` T=1 Q5 | generic `linear()` + `residual_add()` | optional fused small-T |

### 9.2 State-In/State-Out Operator Contract

For quick runtime experiments, Engine can snapshot `GdnState`, run verify in-place,
then restore and replay accepted prefix if needed. That is an Engine strategy, not an
L1 kernel policy.

For this operator preparation part, the L1 contract should expose explicit state
input/output variants. They are pure sequence transforms: `state_in` is read-only,
`state_out` receives the state after all `T` tokens, and accepted-prefix decisions
remain outside the kernel.

```text
causal_conv1d_sequence(x:         [10240,T] BF16 compact,
                       weight:    [10240,4] BF16 compact,
                       state_in:  [10240,3] BF16 compact readonly,
                       out:       [10240,T] BF16 compact,
                       state_out: [10240,3] BF16 compact)
```

```text
gated_delta_rule_recurrent_state(q:         [128,16,T] BF16 compact,
                                 k:         [128,16,T] BF16 compact,
                                 v:         [128,48,T] BF16 compact,
                                 g:         [48,T] FP32 compact,
                                 beta:      [48,T] FP32 compact,
                                 state_in:  [128,128,48] FP32 compact readonly,
                                 out:       [128,48,T] BF16 compact,
                                 state_out: [128,128,48] FP32 compact)
```

The current in-place APIs can remain wrappers that pass the same tensor as input and
output. The new contract makes it possible for target verify to evaluate a suffix
against a shadow state without forcing kernel code to understand acceptance.

### 9.3 GDN Performance Direction

Small-T verify should not permanently rely on large prefill-style overhead. Performance
work should focus on:

1. `gdn_in_vz_batch()` for `[5120,T] -> v/z [6144,T]` with T=1..6.
2. GDN conv pack/split fused with adjacent ops only if copy/extract is visible in
   profiling.
3. conv sequence kernel tuned for T=2..6, preferably one launch for output and final
   state.
4. recurrent GDN kernel tuned for T=2..6, using explicit `state_in/state_out`.
5. optional fusion around GDN V/Z, conv output split, l2norm and state update only
   after profiling shows it matters.

Acceptance remains outside these kernels.

## 10. Reusable Ops And Validation Limits

| op | use in MTP / verify | current status |
|---|---|---|
| `embed_gather` | shared embedding for MTP and target verify ids | supports `[T] -> [5120,T]` |
| `rmsnorm` | all hidden/head norms | reusable for compact contiguous tensors |
| `rope` | q/k RoPE for full attention | supports positions `[T]` and `[256,heads,T]` |
| `l2norm` | GDN q/k normalization | supports compact `[128,16,T]` |
| `gdn_gating` | standalone GDN A/B to g/beta when fused `gdn_in_ab_gated_*` is not used | supports `[48,T]` |
| `sigmoid_gate_mul` | gated attention output | reusable for compact contiguous tensors |
| `silu_and_mul` | MLP activation | shape-generic, output contiguous |
| `residual_add` | attention/MLP residual | reusable for compact contiguous tensors |
| `linear` Q6 lm_head | logits `[248320,T]` | T>1 supported by generic low-bit path |
| `argmax` | draft/target greedy ids | supports `[vocab,T] -> [T]` |

The main validation issue is not dtype, but compactness. Many wrappers require
contiguous tensors. Any row slice out of a larger `[N,T]` output must be copied or
produced directly in compact layout before passing to these ops when `T > 1`.

## 11. End-To-End MTP Operator Shape Flow

For `T` known MTP input tokens:

```text
ids                         [T]
embedding                   [5120,T]
embedding_norm              [5120,T]
hidden_in                   [5120,T]
hidden_norm                 [5120,T]
fc_input                    [10240,T]
mtp_fc_out                  [5120,T]

layer_input_norm            [5120,T]
attn_in_out                 [14336,T]
q                           [256,24,T]
k                           [256, 4,T]
gate                        [256,24,T]
v                           [256, 4,T]
q_norm                      [256,24,T]
k_norm                      [256, 4,T]
q_rope                      [256,24,T]
k_rope                      [256, 4,T]
attn_out                    [256,24,T]
gated_attn_out              [256,24,T]
attn_flat                   [6144,T]
o_proj_out                  [5120,T]
residual_after_attention    [5120,T]

post_attention_norm         [5120,T]
mlp_gateup                  [34816,T]
mlp_gate                    [17408,T]
mlp_up                      [17408,T]
mlp_act                     [17408,T]
mlp_down                    [5120,T]
residual_after_mlp          [5120,T]

mtp_final_norm              [5120,T]
lm_head_logits              [248320,T_logits]
draft_tokens                [T_logits]
```

`T_logits == T` for batched MTP prefill/ref/benchmark runs. Autoregressive MTP
proposal may materialize only the last logits column, in which case `T_logits == 1`.

For autoregressive MTP proposal, practical decode still runs one draft step at a
time because each next input id depends on previous logits. The bottom operators
should nevertheless support `T > 1` for MTP prefill and test/benchmark coverage.

## 12. Target Verify Small-T Operator Flow

For target verifier with `T_verify` input tokens:

```text
verify_ids                  [T_verify]
positions                   [T_verify]
embedding                   [5120,T_verify]
```

Each full-attention layer needs:

```text
input_norm                  [5120,T_verify]
q/gate/k/v projections      [6144,T_verify], [6144,T_verify], [1024,T_verify], [1024,T_verify]
q/k norm                    [256,24,T_verify], [256,4,T_verify]
rope                        [256,24,T_verify], [256,4,T_verify]
gqa append                  [256,24,T_verify]
gate + o_proj + residual    [5120,T_verify]
```

Each GDN layer needs:

```text
input_norm                  [5120,T_verify]
q/k/v/z/a/b projections     [2048,T_verify], [2048,T_verify], [6144,T_verify], [6144,T_verify], [48,T_verify], [48,T_verify]
conv input                  [10240,T_verify]
conv output                 [10240,T_verify]
g, beta                     [48,T_verify]
q/k l2norm                  [128,16,T_verify]
gated delta rule            [128,48,T_verify]
GDN norm + z gate           [128,48,T_verify]
out proj + residual         [5120,T_verify]
```

Each MLP tail needs:

```text
post_norm                   [5120,T_verify]
gateup                      [34816,T_verify]
activation                  [17408,T_verify]
down + residual             [5120,T_verify]
```

Final verifier output:

```text
final_norm                  [5120,T_verify]
lm_head_logits              [248320,T_verify]
target_tokens               [T_verify]
```

Sampler/acceptance logic consumes these target tokens and decides committed prefix.
That logic must not be embedded into L1 kernels.

## 13. Existing Fused Ops Not Required For Correctness

These TEXT_CORE decode fusions do not block MTP or target-verify operator
correctness if generic W8 and generic T>1 paths exist:

| current fused op | why not directly reusable | first replacement | classification |
|---|---|---|---|
| Q4/Q5 text `attn_in` GEMV | wrong qtype and wrong fused shape | W8 `mtp.attn_in.w8` linear | A for W8 base kernel, C for direct compact output |
| Q5 `linear_residual_add` | only supports Q5 T=1 `o_proj` / `mlp_down` shapes | W8 or Q5 linear then `residual_add` | B for split ops, C for small-T epilogue |
| Q4 `mlp_gate_up_silu_decode` | only supports Q4 T=1 gateup | Q4 text gateup or W8 MTP gateup, then `silu_and_mul` | B for split ops, C for fused small-T |
| `gdn_in_vz_decode` | T=1 only | generic linears, later `gdn_in_vz_batch` | B for correctness fallback, C for dual-output batch |
| `causal_conv1d_decode` | T=1 only | `causal_conv1d_prefill` or sequence state API | B for state contract, C for tuned small-T |
| `gqa_attention_decode` | T=1 only | append-contiguous small-T GQA | B for prefill wrapper, C for decode-derived small-T, A for strict one-launch |

Performance follow-up can add fused variants:

```text
w8_linear_residual_add for [5120,6144] and [5120,17408]
w8_mlp_gateup_silu_decode or small-T for [34816,5120]
w8_attn_in_split for [14336,5120]
q5 target linear_residual_add small-T
gdn_in_vz_batch for [6144,5120] x 2
gqa_append_contiguous_small_t for T=1..6
```

## 14. Phased Support Plan

### 14.1 Phase 0: Correctness-First Operator Enablement

Goal: make the MTP and target verify math executable without optimizing launch count.

Required:

1. Add W8 metadata validation to `linear()`.
2. Add `W8G128_RowSplit` codec and generic GEMV/GEMM path.
3. Register all five MTP W8 shapes.
4. Add C++ test pack/decode support for W8G128.
5. Update linear/q5090 test packers and `linear_op_bench` qtype parsing so W8 can
   be tested and measured.
6. Replace the current W8 unsupported test with numerical W8 linear tests.
7. Add `mtp_pack_fc_input`.
8. Add compact split-copy for `mtp.attn_in.w8` output.
9. Reuse `gqa_attention_prefill()` as append-contiguous correctness path.
10. Define GDN conv pack/split as either L1 copy ops or a caller-owned
    `cudaMemcpy2DAsync` contract.
11. For target verify prototypes, reuse T>1 generic/prefill GDN paths and keep
    state commit/rollback outside L1.

Verification focus:

```text
qus_linear_test:
  W8 [small synthetic shapes] T=1,2,5,17
  W8 MTP representative shapes with real K alignment

qus_gqa_attention_test:
  append T=1,2,3,5,6
  cache_offset=0,17,128,2882
  cache append bit equality and future-token mask

qus_causal_conv1d_test / qus_gated_delta_rule_test:
  existing T>1 equivalence remains green

GDN pack/split checks:
  pack q/k/v into [10240,T] and split conv output back to compact q/k/v
  T=1,2,6 produce the same tensors as the current model-local copy/extract helpers
```

### 14.2 Phase 1: Small-T Verify Operator Path

Goal: avoid per-token replay for target verification with `T_verify=1..6`.

Required:

1. Explicit `gqa_attention_append_contiguous()` API, implemented by existing prefill
   or a new small-T kernel.
2. `causal_conv1d_sequence()` with `state_in/state_out`.
3. `gated_delta_rule_recurrent_state()` with `state_in/state_out`.
4. GDN conv pack/split coverage for `T_verify=1..6`.
5. T>1 alternatives for decode-only fused ops:
   - `gdn_in_vz_batch()` or generic linears;
   - Q5 small-T `linear_residual_add` or generic linear plus residual;
   - MLP gateup small-T plus `silu_and_mul`.
6. `lm_head [248320,T_verify]` and `argmax [T_verify]` path for verifier output.

Verification focus:

```text
small-T operator-level parity harness:
  T=1 append/sequence operator path equals the existing T=1 decode operator output
  T=2..6 append/sequence operator path equals sequential per-token operator replay
  covered outputs include attention out, GDN sequence out, final logits where harnessed

state behavior:
  conv/SSM state after T sequence equals sequential T single steps
  explicit state_out variant does not mutate state_in
```

### 14.3 Phase 2: Fused Kernels And Profiling

Goal: reduce launch count and memory traffic after correctness is stable.

Candidates:

1. W8 tuned T=1 GEMV for MTP decode.
2. W8 tuned T=2..6 multi-step GEMM.
3. W8 LargeT MMA with K128 scale staging.
4. Fused `mtp_attn_in` W8 linear that writes compact q/k/gate/v outputs.
5. Fused W8 gateup + SiLU for MTP MLP.
6. W8 residual epilogues for `o_proj` and `mlp.down`.
7. GQA small-T append kernel with combined K/V write and attention partials.
8. GDN small-T fusions only after nsys/ncu shows GDN state or conv overhead is material.

Runtime CUDA graph buckets for fixed `T_verify` values are not L1 work. The only
operator requirement is stable shape/API contracts that a future runtime bucket can
capture.

Benchmark focus:

```text
linear_op_bench:
  --qtype W8
  --shape MtpFc5120x10240
  --shape MtpAttnIn14336x5120
  --shape MtpGateUp34816x5120
  --t-sweep 1,2,3,5,6,16,64

gqa_attention_bench:
  append-contiguous T=1,2,3,5,6
  cache positions around 2K, 8K, 32K

gdn benches:
  conv sequence T=1,2,3,5,6
  gated delta recurrent T=1,2,3,5,6
  state_in/state_out copy overhead
```

### 14.4 Phase 3: Strict One-Launch Experiments

Goal: evaluate kernels that intentionally collapse work across boundaries only after
the Phase 1/2 path has profiling evidence.

Candidates:

1. strict one-launch GQA append verify that combines new K/V write, long-cache
   attention and per-token reduce.
2. fused GDN small-T pipeline across V/Z projection, conv, A/B, recurrent state and
   output staging.
3. MTP `attn_in.w8` direct split output plus q/k norm or gate epilogue, if the
   split-copy path remains visible in nsys.

These are not correctness blockers. Each candidate needs an explicit baseline:
sequential decode replay, Phase 1 small-T API, and Phase 2 fused-but-not-strict-one-launch
variant where applicable.

## 15. Acceptance Criteria For This Preparation Part

This operator preparation part is complete when:

1. L1 `linear()` accepts `W8G128_F16S` row-split weights and covers all five MTP
   W8 shape families.
2. L1 has W8 numerical tests for T=1, small T and at least one larger T regime.
3. L1 exposes a simple BF16 `mtp_pack_fc_input` or equivalent concat/pack op.
4. L1 handles `mtp.attn_in.w8` output split correctly for `T > 1`, either by
   compact split-copy or fused compact output.
5. L1 exposes or clearly reuses a caller-owned-buffer GQA append-batch attention
   primitive for small `T_new`, with explicit q/k/v/cache-position/out shapes and
   no runtime/GDN state ownership.
6. Target verify small-T operator requirements are documented and covered by tests
   before runtime relies on them, including GDN conv pack/split and state-in/state-out
   sequence contracts.
7. GDN/conv/SSM kernels do not own acceptance; state-in/state-out APIs are pure
   sequence transforms.
8. Existing reusable ops remain unchanged unless their validation is too narrow for
   the listed compact MTP/verify shapes.
9. Any W8 or GDN fused decode replacements are treated as performance follow-up,
   not correctness blockers.
