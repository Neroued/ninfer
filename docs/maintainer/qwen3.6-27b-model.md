# Qwen3.6-27B Model Reference

This reference records the exact Text, MTP, Vision, multimodal-position, numeric, and persistent-state
semantics used by the registered target. Fixed dimensions are encoded in the target's
`impl/config.h`; artifact assignment and binding are defined in
[`qwen3.6-27b-artifact.md`](qwen3.6-27b-artifact.md).

## 1. Model identity

The target checkpoint is called Qwen3.6-27B in this project and uses the `qwen3_5` /
`qwen3_5_text` architecture names in Hugging Face implementations. It is a dense multimodal model
with three runtime components:

- a 64-layer hybrid Text decoder;
- a one-layer MTP draft model;
- a 27-layer Vision transformer and patch merger.

The implementation is fixed to this checkpoint shape. A different layer count, hidden size, head
layout, or Vision tower is a different model implementation rather than a runtime configuration.

## 2. Global dimensions

### 2.1 Text decoder

| Field | Value |
|---|---:|
| hidden size | 5120 |
| decoder layers | 64 |
| intermediate size | 17408 |
| output/embedding matrix rows | 248320 |
| tokenizer-addressable token IDs | 248077 (`0..248076`) |
| full-attention interval | 4 |
| full-attention layers | 16 |
| Gated-DeltaNet layers | 48 |
| RMSNorm epsilon | `1e-6` |
| RoPE theta | `1e7` |
| checkpoint position capacity | 262144 |
| MTP layers | 1 |

Layer `i` is full attention exactly when `(i + 1) % 4 == 0`; all other layers are GDN. The model
has a dense SwiGLU MLP in every layer and does not contain MoE experts.

### 2.2 Full attention

| Field | Value |
|---|---:|
| query heads | 24 |
| KV heads | 4 |
| head dimension | 256 |
| Q width | 6144 |
| K/V width | 1024 each |
| rotated dimensions per head | 64 |
| Q heads per KV head | 6 |
| attention scale | `1/sqrt(256) = 0.0625` |

Each query projection also produces a 256-dimensional per-head output gate. Q and K use
zero-centered `(1+w)` RMSNorm before partial NeoX RoPE. The attention result is multiplied by
`sigmoid(gate)` before the output projection.

### 2.3 Gated-DeltaNet

| Field | Value |
|---|---:|
| Q/K heads | 16 |
| Q/K head dimension | 128 |
| V heads | 48 |
| V head dimension | 128 |
| Q width | 2048 |
| K width | 2048 |
| V width | 6144 |
| Z output-gate width | 6144 |
| A/B gate width | 48 each |
| causal convolution width | 4 |
| recurrent state dtype | FP32 |
| delta-rule scale | `1/sqrt(128)` |

Every group of three V heads shares one Q/K head. GDN state size is fixed per layer and does not grow
with context length.

### 2.4 Vision

| Field | Value |
|---|---:|
| transformer depth | 27 |
| hidden size | 1152 |
| intermediate size | 4304 |
| attention heads | 16 |
| head dimension | 72 |
| spatial patch | 16 × 16 |
| temporal patch | 2 frames |
| flattened patch width | `3 × 2 × 16 × 16 = 1536` |
| spatial merge | 2 × 2 patches |
| merger input width | 4608 |
| merger output width | 5120 |
| position table | 48 × 48 |
| Vision RoPE theta | 10000 |

### 2.5 Runtime extents

`T` denotes the Text/MTP token extent supplied to an Op. It is any positive tensor extent that
fits the applicable storage or explicit state capacity. Decode (`T=1`), verification-sized calls,
and prefill chunks are workload points and private implementation routes, not different Op domains.
The configured prefill chunk controls target workspace and request decomposition; its default 1024
does not cap an Op's `T`.

Vision uses different axes. `P` is the aggregate raw-patch count and must be a positive multiple of
4 because of the 2x2 spatial merge; `V=P/4` is the aggregate merged-token count. The registered 27B
processor/implementation envelope is `4<=P<=131072` and `1<=V<=32768`, with the frontend's media,
attention-pair, and prompt budgets imposing any additional request-specific restriction. These
Vision columns are not Text token `T` and are not expanded beyond that envelope.

## 3. Shared decoder layer skeleton

Both Text layer types use a pre-norm residual structure:

```text
h = RMSNorm(x, input_norm, unit_offset=true)
x = x + mixer(h)
h = RMSNorm(x, post_attention_norm, unit_offset=true)
x = x + down_proj(SiLU(gate_proj(h)) * up_proj(h))
```

The registered `.ninfer` target stores some projection groups as fused logical views, and its
private kernels may fuse projection, activation, or residual work. Those are storage/execution
choices; the mathematical result remains the schedule above.

All ordinary layer norms and Q/K norms use zero-centered weights: the effective scale is `(1+w)`.
The GDN internal gated norm is a plain ones-initialized RMSNorm and does not use the unit offset.

## 4. Full-attention layer

For normalized input `h[5120,T]`:

```text
q, gate = q_gate_projection(h)        # [24,256,T] each
k       = k_projection(h)             # [4,256,T]
v       = v_projection(h)             # [4,256,T]

q = RMSNorm(q, q_norm, unit_offset=true)
k = RMSNorm(k, k_norm, unit_offset=true)
q, k = partial_mrope(q, k, rotary_dims=64)

a = causal_gqa(q, k, v, scale=1/sqrt(256), kv_cache)
a = a * sigmoid(gate)
x = x + o_projection(a)
```

Prefill appends all K/V columns and evaluates causal attention for the chunk. Decode appends one
column and attends over the resident prefix. KV storage may be BF16 or INT8-G64. The exact runtime
cache codec and the common ideal attention oracle are defined by the repository-internal
[`gqa_attention.h`](../../include/ninfer/ops/gqa_attention.h) contract. Both cache formats and their
optimized compute profiles are judged by that one oracle construction rather than by
implementation-mirroring references.

Text-only positions use the same scalar position for temporal, height, and width MRoPE sections.
Multimodal prefill supplies distinct three-axis positions. Only 64 of each 256-dimensional head are
rotated, divided across the model's interleaved MRoPE sections `[11,11,10]`.

For the registered RTX 5090 implementation, `attn_input_proj` consumes the physical
`query_key [7168,5120]` Q4 parent and `gate_value [7168,5120]` Q5 parent directly. At
`T=1..16`, one Q4 projection writes Q/K through a split epilogue and one Q5 projection writes
Gate/V through a split epilogue, so a full-attention layer issues exactly two input-projection
kernels. The `T>=17` route likewise evaluates the two homogeneous parents with grouped MMA
launches. This is an implementation profile; Q, Gate, K, and V remain the four logical Op outputs.

## 5. Gated-DeltaNet layer

The normalized input produces Q, K, V, Z, and per-V-head A/B controls:

```text
q = in_q(h)       # [16,128,T]
k = in_k(h)       # [16,128,T]
v = in_v(h)       # [48,128,T]
z = in_z(h)       # [48,128,T]
a = in_a(h)       # [48,T]
b = in_b(h)       # [48,T]
```

Q/K/V are concatenated into `[10240,T]` and passed through the depthwise causal width-4 convolution
and SiLU. Q and K are then L2-normalized per head. The decay and update controls `g` and `beta` are
observable FP32 values with the logical formula:

```text
g    = -exp(A_log) * softplus(a + dt_bias)
beta = sigmoid(b)
```

For `T=1..16`, the registered `gdn_input_proj` implementation keeps QK and V as two independent
projections but gives each launcher a row slice of the final `[10240,T]` tensor with leading
dimension 10240. It therefore allocates no concat workspace and performs no input-projection D2D
copy. The grouped `T>=17` route is unchanged. Public `linear` output remains contiguous-only; the
pitched row slices are private to this Op implementation.

Verify `T=1..6` invokes the shared `gdn_input_proj_conv_snapshot` contract: the target leaf combines
the Q4/Q5 projections with causal convolution, SiLU, direct q/k/v placement, and publication of
BF16 convolution snapshots to slots `0..T-1`. The eliminated qkv intermediate is not a semantic
cast boundary; each exact route uses its directly oracle-qualified private precision and staging.
The separate Z projection remains in the target output-gate leaf.

For V head `j`, let `q` and `k` come from Q/K head `j // 3`. With recurrent state
`S[128,128]`, one token performs:

```text
k  = k / ||k||
q  = q / ||q||
S  = exp(g) * S
Sk = S @ k
u  = beta * (v - Sk)
S  = S + u outer k
o  = (S @ q) * (1/sqrt(128))
```

The CUDA recurrence uses an algebraically equivalent ordering appropriate to the kernel. Prefill
uses chunked parallel state passing for large T and recurrent/small-T paths where appropriate;
decode and MTP verification use recurrent/snapshot-capable paths.

The per-head output is normalized and gated before projection:

```text
on = gated_rmsnorm(o, gdn_norm, z)    # RMSNorm(o) * SiLU(z)
x  = x + out_projection(on)
```

Each GDN layer owns:

- three previous convolution columns for normal continuation;
- an FP32 recurrent matrix for every V head;
- additional snapshot slots when MTP or reusable turn-boundary state requires them.

## 6. Text prefill and decode

### Prefill

1. gather quantized token embeddings into `[5120,T]`;
2. replace placeholder columns with Vision merger output when input is multimodal;
3. run the 64 decoder layers in aligned chunks while carrying KV/GDN state;
4. retain the hidden columns required by MTP preparation;
5. apply final `(1+w)` RMSNorm to the last required column;
6. evaluate the full `lm_head[248320,5120]`;
7. select the first generated token with greedy or configured sampling.

Chunking limits workspace. It does not reset positions or state between chunks.

### Ordinary decode

1. embed the current token;
2. run all 64 layers for one position;
3. apply final norm and full `lm_head`;
4. sample the next token;
5. commit updated KV/GDN state and position.

The eager and CUDA Graph record/replay paths execute the same model schedule.

## 7. MTP draft model

The checkpoint contains one MTP decoder layer. It is a small draft model conditioned on both the
target hidden state and the token that follows it; it is not another `lm_head` attached directly to
the main decoder.

The MTP stem is:

```text
e = RMSNorm(embed(token), pre_fc_norm_embedding, unit_offset=true)
h = RMSNorm(target_hidden, pre_fc_norm_hidden, unit_offset=true)
x = fc(concat(e, h))
attention_residual = RMSNorm(x, input_norm, unit_offset=true)
```

The single MTP layer is a full-attention layer with the same 24 Q heads, 4 KV heads, 256 head
dimension, Q/K norm, partial MRoPE, gated output, and 17408-wide SwiGLU MLP as a Text full-attention
layer. It has its own one-layer KV cache. A final MTP norm produces the draft hidden state.

MTP reuses the Text embedding and full `lm_head` semantics; the checkpoint does not contain a tied
MTP embedding/head pair. At proposal sites the runtime may instead use the optional
`[131072,5120]` Q4 shortlisted head and remap its row index to a real vocabulary id. Target
verification always uses the full `lm_head`.

## 8. Speculative round semantics

For `k` configured draft tokens, the runtime prepares a candidate window, runs target verification,
and accepts only the prefix licensed by the target distribution.

In greedy mode, each draft token is accepted while it equals the target argmax at that position. In
sampling mode, proposal and target probabilities use the same processed distributions and the
accept/reject correction preserves the target distribution. A bad draft therefore reduces
acceptance and throughput; it must not change the distribution of emitted target tokens.

Target verification advances speculative KV/GDN state into multiple slots. After the accepted
length is known, the runtime commits the matching slot, rewinds logical positions for rejected
candidates, and continues from the accepted target token. Near context capacity, the Engine falls
back to the one-token target path when a complete safe round does not fit.

## 9. Vision preprocessing

The native processor accepts structured text/image/video message parts. For each media item it:

1. consumes media bytes already acquired by the CLI or serving layer;
2. decodes the image or samples video frames;
3. chooses dimensions aligned to the 32-pixel merge factor;
4. bicubic-resizes and normalizes RGB values;
5. packs channel-major `2 × 16 × 16` temporal-spatial patches into FP32 rows of width 1536;
6. expands the chat-template placeholders and records the token spans;
7. constructs Vision grids, timestamps, token types, and three-axis text positions;
8. computes `rope_delta` for subsequent Text decode positions.

Images repeat a frame to form the temporal pair. Videos are sampled at the configured rate and
packed in temporal pairs. Media and compute budgets reject oversized work before Vision execution.

## 10. Vision tower

Patch rows are converted to BF16 and projected into `[1152,P]`. The runtime adds a bilinearly
interpolated learned 48×48 position table.

Each of the 27 transformer blocks performs:

```text
h = LayerNorm(x)
q, k, v = qkv_projection(h) + bias
q, k = vision_rope(q, k, 2D patch positions)
a = segmented_flash_attention(q, k, v, cu_seqlens)
x = x + projection(a) + bias

h = LayerNorm(x)
h = GELU_tanh(fc1(h) + bias)
x = x + fc2(h) + bias
```

Attention is segmented by image or video frame grid using cumulative sequence lengths; unrelated
media/frame segments do not attend to each other.

The merger applies LayerNorm, groups each spatial 2×2 patch block into width 4608, then computes:

```text
visual = fc2(GELU_exact(fc1(merged) + bias)) + bias    # [5120,V]
```

The result replaces the matching placeholder embedding columns before Text prefill. Vision state is
not retained for autoregressive decode.

## 11. Multimodal positions

The family prepared prompt's `positions` value is axis-major `[3,T]` in temporal, height, width
order.

- ordinary text tokens advance all three axes together;
- image/video placeholder tokens receive positions derived from the merged Vision grid;
- following text resumes after the maximum multimodal position;
- `rope_delta = next_multimodal_position - token_count` converts later scalar decode indices into
  the correct MRoPE position.

The Text RoPE kernel consumes these positions during multimodal prefill and uses the saved
`rope_delta` during decode. This is why Vision can disappear after prefill while Text positions
remain consistent.

## 12. Precision and oracle boundaries

- activations are BF16 at public model/operator boundaries;
- ordinary and Q/K norm oracles evaluate their reductions in FP32/FP64 and compare the declared
  BF16 outputs; production reduction and staging are route-private choices;
- GDN `g`, `beta`, and recurrent state are FP32;
- the ideal GQA oracle evaluates dot products, stable softmax, and value reduction in FP64 from
  BF16 Q and logical cache values; the BF16 Op output is promoted to FP64 for comparison;
- low-bit weight storage changes representation, not the intended dequantized matrix;
- INT8-G64 KV stores FP16 scales and signed codes, and its ideal logical K/V values are their FP32
  decode;
- the target's INT8 attention path intentionally quantizes Q to Q8-G64 for production computation;
  this native compute profile does not replace BF16 Q in the common ideal oracle, and its delta is
  accepted through the separate named INT8 tolerance rather than the BF16 tolerance;
- the full target `lm_head` is used for prefill, verification, and ordinary decode regardless of
  draft-head mode.

These points define public representations and mathematical oracles, not a mandatory kernel
operation order. Private accumulator precision, Tensor Core operand staging, intermediate
materialization, workspace dtype, and reduction association are selected by each implementation
route and accepted against the corresponding oracle tolerance.

GQA numerical qualification covers both registered geometries, supported prompt and small-T
regimes, the maintained conformance matrix, and target-representative activation ranges. Its
`Tolerance::attention_bf16()` and `Tolerance::attention_int8()` acceptance envelopes are explicit
named standards in `tests/ops/op_check.h`; they are not claimed as pointwise bounds for every
arbitrary or adversarial BF16 tensor. A1 append-and-attend and A3 cached-only attention are each
checked directly against the common ideal oracle. Equality between those different numerical paths
is not a contract or acceptance test.

## 13. State inventory

| State | Shape basis | Lifetime |
|---|---|---|
| Text GQA KV | 16 layers × context × 4 heads × 256 | active sequence |
| MTP KV | 1 layer × context × 4 heads × 256 | active sequence when MTP enabled |
| GDN conv | 48 layers × 10240 × 3, plus slots | active sequence |
| GDN recurrent | 48 layers × 48 heads × 128 × 128 FP32, plus slots | active sequence |
| Text step buffers | token, positions, logits, verify/draft/sampling tensors | Program lifetime |
| Vision workspace | patch/control/intermediate/merger tensors | one multimodal prefill |
| Text workspace | chunk/block temporaries | arena scope |

KV memory grows with configured context. GDN recurrent state does not, although MTP snapshot count
changes its fixed allocation.

## 14. Implementation map

| Model concern | Source |
|---|---|
| exact dimensions/layer counts and family hybrid-layer mapping | `src/targets/qwen3_6_27b/impl/config.h`, `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/hybrid_topology.h` |
| immutable Text/MTP/Vision bindings | `src/targets/qwen3_6_27b/impl/load/` |
| split attention projection, staged GDN projection/control, dense post-mixer leaves, leaf workspace, and graph frontier ranges | `src/targets/qwen3_6_27b/impl/variant.h`, `impl/variant.cpp` |
| Text/MTP/Vision execution, planning, Program lifecycle, workspace composition, prefix/state transactions, and graph mechanics | `src/targets/qwen3_6/impl/runtime/` |
| tokenizer, template, multimodal processing, output decoder | `src/targets/qwen3_6/impl/frontend/` |
| mathematical and explicit local-state Op contracts/implementations | `include/ninfer/ops/`, `src/ops/` |
| GQA physical cache container and checked per-layer views | `src/core/kv_cache.*` |
| GDN layout/views/reset/copy and Text/MTP/GDN composition | `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/decoder_state.h`, `src/targets/qwen3_6/impl/state/decoder_state.cpp` |
| generated-round buffer schema, MTP alignment, and Vision control | `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/`, `src/targets/qwen3_6/impl/state/round_state.cpp`, `src/targets/qwen3_6/impl/vision/control.cpp` |
| `.ninfer` tensor assignment and binding | [`qwen3.6-27b-artifact.md`](qwen3.6-27b-artifact.md), `tools/reference/qwen3_6_27b/bindings.py` |
| native `.ninfer` converter and verifier | `tools/convert/qwen3_6_27b` |
| artifact-native Python Text/Vision/MTP reference | `tools/reference/qwen3_6_27b` |

The Python reference is an independent executable implementation for model/artifact inspection and
diagnosis; it is not the per-Op mathematical oracle, does not prescribe private C++ kernel
precision, and does not define cross-runtime generated-token equality. Each Op is checked against
its own naive FP32/FP64 or exact oracle. The C++ target
implements the complete Text/Vision/MTP product over `.ninfer` through the closed Engine
architecture.
