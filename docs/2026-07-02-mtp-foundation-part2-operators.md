# MTP 支持预先准备工作第二部分：底层算子支持

本文承接 [Part 1](2026-07-02-mtp-foundation-part1-design.md) 的 q5090 v3 MTP_DRAFT layout，
只讨论为了执行该 layout 需要补齐的 L1/operator 能力。本文不修改 q5090 权重格式，不修改
Python converter/ref model，不接入 Engine、C++ model card、scheduler 或 speculative
accept/reject 流程。

Part 1 已经固定 MTP_DRAFT 的五个 W8 logical linears：

```text
mtp.fc
mtp.attn_in.w8
mtp.o_proj
mtp.mlp.gateup.w8
mtp.mlp.down
```

本文关心的是这些权重进入计算图后，L1 层需要能处理哪些 qtype、shape、view 和 caller-owned
state 边界。

## 1. Scope

### 1.1 In Scope

1. `W8G128_F16S` row-split linear 的最小正确性支持。
2. MTP 五个 W8 shape family 的精确维度。
3. MTP FC 输入 concat/pack 的计算公式和 shape。
4. `mtp.attn_in.w8` 输出行区间到 Q/K/Gate/V view 的约定。
5. MTP full-attention/GQA append-batch 的 caller-owned-buffer L1 contract。
6. 可复用的已有 elementwise/reduction ops，以及后续性能 fused op 候选。

### 1.2 Out Of Scope

1. 不修改 q5090 v3 layout、block/segment/fusion group 约定；这些属于 Part 1。
2. 不修改 Python converter、verifier 或 ref model；这些属于 Part 1。
3. 不修改 Engine、runtime scheduler、CUDA graph、KV cache allocation/ownership、GDN/MTP
   runtime state lifecycle、proposal/verification state machine 或 target hidden-state handoff。
4. 不接 C++ model card 或 MTP forward。
5. 不定义 target KV/GDN rejected-suffix commit-or-rollback 策略。

本文所有 attention/cache 描述都是算子输入输出边界：`cache` 是 caller-owned KV buffer/slice，
算子不得分配、扩容、推进或解释 runtime state。

## 2. Required: W8G128 Row-Split Linear

Current `linear()` accepts Q4/Q5/Q6 and dense. MTP requires W8. The operator must support:

```text
qtype:       W8G128_F16S
layout:      ROW_SPLIT
group_size:  128
scale_dtype: FP16
high plane:  none
```

W8 codec:

```text
K group size: 128
code plane:   int8 codes, 128 bytes per row per group
scale plane:  fp16 scale, 2 bytes per row per group
dequant:      w = float(int8_code) * fp16_scale
code range:   [-127, 127]
```

Minimum dispatch support:

| regime | required behavior |
|---|---|
| `T == 1` | W8 GEMV |
| `1 < T <= small threshold` | W8 small-T GEMM or multi-step GEMV |
| `T > threshold` | W8 tensor-core GEMM or generic correct GEMM |

Correctness can start with generic W8 GEMV/GEMM. Performance work should add specialized plans for
the five MTP W8 shapes listed below.

## 3. Required W8 Shape Families

These are the exact W8 matrix families needed by the Part 1 target layout:

| logical op | shape `[N,K]` | common regimes |
|---|---:|---|
| `mtp.fc` | `[5120,10240]` | MTP prefill `T=prompt`, MTP decode `T=1` |
| `mtp.attn_in.w8` | `[14336,5120]` | MTP prefill/decode |
| `mtp.o_proj` | `[5120,6144]` | MTP prefill/decode |
| `mtp.mlp.gateup.w8` | `[34816,5120]` | MTP prefill/decode |
| `mtp.mlp.down` | `[5120,17408]` | MTP prefill/decode |

The existing TEXT_CORE shape families are useful references but not sufficient, because:

- `mtp.fc [5120,10240]` is new;
- `mtp.attn_in.w8 [14336,5120]` differs from TEXT `attn_in.q4/q5 [7168,5120]`;
- `mtp.mlp.gateup.w8 [34816,5120]` has the same dimensions as TEXT gateup but a different qtype;
- `mtp.o_proj [5120,6144]` and `mtp.mlp.down [5120,17408]` match existing dimensions but need W8
  kernels, not Q5 kernels.

## 4. Required: MTP FC Input Pack / Concat

MTP first combines normalized embedding and normalized hidden state:

```text
e[:, t] = RMSNorm_mtp_embedding(Embed[token_t])
h[:, t] = RMSNorm_mtp_hidden(hidden_t)

u[:, t] = concat(e[:, t], h[:, t])    # [10240,T]
z[:, t] = mtp.fc(u[:, t])             # [5120,T]
```

The concat operator is a deterministic BF16 pack along the fastest dimension:

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

This op performs no arithmetic. A simple 2D copy kernel is enough for correctness. Later, this can be
fused with the two RMSNorm outputs or with the W8 `mtp.fc` input staging, but the first bottom-layer
contract should be explicit and easy to verify.

## 5. Required: Attention Input Split Views

No arithmetic kernel is required if `mtp.attn_in.w8` emits the Part 1 row order. L1 must make the output
layout stable enough for later model/runtime integration to create zero-copy views:

```text
attn_in_out [14336,T]

q    [256,24,T] from rows [0,6144)
k    [256, 4,T] from rows [6144,7168)
gate [256,24,T] from rows [7168,13312)
v    [256, 4,T] from rows [13312,14336)
```

This is why converter-side q/gate de-interleave is required in Part 1. It turns q/gate split into a
zero-copy view operation for later integration.

## 6. Required: GQA Append-Batch Attention

Current kernels cover:

- prefill contiguous append via `cache_offset`, writing positions `[cache_offset, cache_offset+T)` and
  attending over old cache plus the current causal chunk;
- decode for `T=1` with scalar position.

For MTP operator preparation, the required L1 boundary is a caller-owned-buffer contiguous append
primitive. If the existing prefill path already serves this contract for the needed small `T_new`, it can
be reused. A new kernel is needed only for a small-T optimized append path or for future vector-position
semantics.

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

Inside the same append batch, token `i` may attend to earlier new tokens `0..i`. It must not attend to
future new tokens `i+1..T_new-1`.

This is a stateless L1 operator boundary. `cache` is an explicit caller-owned KV buffer or slice. The op
may write only K/V positions `[cache_offset, cache_offset + T_new)` and `out`; it must not allocate,
resize, advance, retain, or interpret KV/GDN/MTP runtime state. For MTP proposal, the caller passes the
separate MTP KV namespace. For target verification, the caller passes the target KV namespace. The target
KV/GDN commit-or-rollback policy for rejected draft suffixes is an engine-state problem outside this
operator preparation part.

For MTP bottom-op preparation, the important shape constraint is not new head geometry. It is still:

```text
head_dim = 256
q_heads  = 24
kv_heads = 4
group    = 6 query heads per KV head
scale    = 1/sqrt(256) = 0.0625
```

Implementation can be derived from the decode kernel by adding a small `T_new` dimension:

1. prewrite the whole new K/V batch to cache positions `cache_offset..cache_offset+T_new-1`, or write
   inside the partial kernel with per-token guards;
2. run decode-style split-KV attention per `(token_i, kv_head, q_subgroup, split)`;
3. set each token's attention window to `cache_offset + i + 1`;
4. reduce partials into `out[:, :, i]`.

This differs from decode because it emits multiple output tokens in one call. It may be implemented by
specializing the existing prefill append path for small `T_new`, or by extending the decode path with a
token dimension. A separate future scattered-position op would need a distinct contract, for example
`positions [T_new]` with monotonic positions and explicit causal-window rules; that is not required for
this operator preparation part.

## 7. Existing Ops That Can Be Reused

| op | use in MTP | status |
|---|---|---|
| `embed_gather` | gather `Embed[token]` for MTP input token ids | reusable |
| `rmsnorm` | pre-fc norms, MTP layer norms, q/k norms, final MTP norm | reusable |
| `rope` | MTP full-attention q/k RoPE | reusable because dimensions match |
| `sigmoid_gate_mul` | `attention_out *= sigmoid(gate)` | reusable |
| `silu_and_mul` | MTP MLP activation | reusable |
| `residual_add` | attention and MLP residuals | reusable |
| `argmax` | greedy draft token selection | reusable, including `[vocab,T]` if logits buffer is sized |
| `lm_head` Q6 linear | MTP logits through shared top-level lm_head | reusable |

## 8. Existing Fused Ops Not Required For Correctness

These TEXT_CORE decode fusions do not block MTP correctness if W8 generic linear exists:

| current fused op | why not directly reusable | first MTP replacement |
|---|---|---|
| Q4/Q5 text `attn_in` GEMV | wrong qtype and wrong fused shape | W8 `mtp.attn_in.w8` linear |
| Q5 `linear_residual_add` | only supports Q5 `o_proj` / `mlp_down` shapes | W8 linear then `residual_add` |
| Q4 `mlp_gate_up_silu_decode` | only supports Q4 gateup | W8 gateup linear then `silu_and_mul` |

Performance follow-up can add W8 fused variants:

```text
w8_linear_residual_add for [5120,6144] and [5120,17408]
w8_mlp_gateup_silu_decode for [34816,5120]
w8_attn_in_decode for [14336,5120]
```

These are not necessary for the first bottom-layer enablement.

## 9. End-To-End MTP Operator Shape Flow

For `T` MTP input tokens:

`hidden_in` is target final hidden states for the first MTP proposal after target forward. For additional
autoregressive draft steps, `hidden_in` is the previous MTP final hidden state. It is not target GDN state
and is never written back into target model state.

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
attn_out = GQA(q_rope,k_rope,v) [256,24,T]
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

For MTP proposal, `T_logits` is often 1 or a small number. The bottom operators should not assume
`T_logits == 1`, because both MTP prefill and target verification benefit from small multi-token paths.

## 10. Acceptance Criteria For This Preparation Part

This operator preparation part is complete when:

1. L1 `linear()` accepts `W8G128_F16S` row-split weights and covers all five MTP W8 shape families.
2. L1 exposes a simple BF16 `mtp_pack_fc_input` or equivalent concat/pack op.
3. L1 exposes stable zero-copy view boundaries for `mtp.attn_in.w8` output row ranges.
4. L1 exposes or reuses a caller-owned-buffer GQA append-batch attention primitive for small `T_new`,
   with explicit q/k/v/cache-position/out shapes and no runtime/GDN state ownership.
5. Existing reusable ops remain unchanged unless their validation is too narrow for the listed MTP
   shapes.
6. Any W8 fused decode replacements are treated as performance follow-up, not correctness blockers.
