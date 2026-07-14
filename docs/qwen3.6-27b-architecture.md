# Qwen3.6-27B Architecture Reference

> Status: current model and runtime-semantics reference.
>
> The fixed dimensions are encoded in `include/ninfer/model/config.h`. This document describes the
> Text decoder, MTP model, Vision tower, multimodal positions, numerics, and persistent state. It does
> not define either artifact byte layout or CUDA implementation policy. The current C++ Engine's
> q5090 assignment is defined by
> [`q5090_packed_file_format_v4.md`](q5090_packed_file_format_v4.md); the implemented native
> `.ninfer` assignment and binding contract are defined by
> [`qwen3.6-27b-ninfer-artifact.md`](qwen3.6-27b-ninfer-artifact.md).

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
| vocabulary | 248320 |
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

## 3. Shared decoder layer skeleton

Both Text layer types use a pre-norm residual structure:

```text
h = RMSNorm(x, input_norm, unit_offset=true)
x = x + mixer(h)
h = RMSNorm(x, post_attention_norm, unit_offset=true)
x = x + down_proj(SiLU(gate_proj(h)) * up_proj(h))
```

The q5090 artifact may store projection groups in fused blocks and L1 may fuse projection,
activation, or residual work. Those are storage/execution choices; the mathematical result remains
the schedule above.

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
column and attends over the resident prefix. KV storage may be BF16 or per-group INT8; quantization
changes cache representation, not attention semantics.

Text-only positions use the same scalar position for temporal, height, and width MRoPE sections.
Multimodal prefill supplies distinct three-axis positions. Only 64 of each 256-dimensional head are
rotated, divided across the model's interleaved MRoPE sections `[11,11,10]`.

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
and SiLU. Q and K are then L2-normalized per head. The decay and update gates are computed in FP32:

```text
g    = -exp(A_log) * softplus(a + dt_bias)
beta = sigmoid(b)
```

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

1. reads a local path or allowed remote source;
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

`ProcessedInput.positions` is axis-major `[3,T]` in temporal, height, width order.

- ordinary text tokens advance all three axes together;
- image/video placeholder tokens receive positions derived from the merged Vision grid;
- following text resumes after the maximum multimodal position;
- `rope_delta = next_multimodal_position - token_count` converts later scalar decode indices into
  the correct MRoPE position.

The Text RoPE kernel consumes these positions during multimodal prefill and uses the saved
`rope_delta` during decode. This is why Vision can disappear after prefill while Text positions
remain consistent.

## 12. Precision invariants

- activations are BF16 at public model/operator boundaries;
- ordinary and Q/K norms use FP32 accumulation and BF16 output;
- GDN `g`, `beta`, and recurrent state are FP32;
- attention softmax and reductions use the accumulation policy defined by their numerical tests;
- low-bit weight storage changes representation, not the intended dequantized matrix;
- INT8 KV stores per-group codes/scales but preserves the BF16 attention contract within documented
  tolerance;
- the full target `lm_head` is used for prefill, verification, and ordinary decode regardless of
  draft-head mode.

## 13. State inventory

| State | Shape basis | Lifetime |
|---|---|---|
| Text GQA KV | 16 layers × context × 4 heads × 256 | active sequence |
| MTP KV | 1 layer × context × 4 heads × 256 | active sequence when MTP enabled |
| GDN conv | 48 layers × 10240 × 3, plus slots | active sequence |
| GDN recurrent | 48 layers × 48 heads × 128 × 128 FP32, plus slots | active sequence |
| Text step buffers | token, positions, logits, verify/draft/sampling tensors | Engine lifetime |
| Vision workspace | patch/control/intermediate/merger tensors | one multimodal prefill |
| Text workspace | chunk/block temporaries | arena scope |

KV memory grows with configured context. GDN recurrent state does not, although MTP snapshot count
changes its fixed allocation.

## 14. Implementation map

| Model concern | Source |
|---|---|
| dimensions and layer mapping | `include/ninfer/model/config.h` |
| Text/MTP weights and schedule | `include/ninfer/model/model.h`, `src/model/qwen3_6_27b.cpp` |
| Vision weights and schedule | `include/ninfer/model/vision.h`, `src/model/qwen3_6_vision.cpp` |
| multimodal processor | `include/ninfer/model/processor.h`, `src/model/processor.cpp` |
| multimodal position/control helpers | `src/model/position.*`, `src/model/vision_ops.*` |
| MTP accept/commit helpers | `src/model/mtp_ops.h`, `src/kernels/kernel/mtp_round.cuh` |
| GQA cache | `include/ninfer/core/kv_cache.h` |
| GDN state | `include/ninfer/core/state_store.h` |
| current C++ `.qus` tensor assignment | [`q5090_packed_file_format_v4.md`](q5090_packed_file_format_v4.md) |
| native `.ninfer` tensor assignment and binding | [`qwen3.6-27b-ninfer-artifact.md`](qwen3.6-27b-ninfer-artifact.md), `tools/reference/qwen3_6_27b_rtx5090/bindings.py` |
| native `.ninfer` converter and verifier | `tools/convert/qwen3_6_27b_rtx5090` |
| complete Python Text/Vision/MTP reference | `tools/reference/qwen3_6_27b_rtx5090` |

The Python reference is the complete executable oracle for the native artifact route. The C++
sources above remain the currently delivered `.qus` Engine implementation; they have not yet been
reorganized into the accepted multi-target Engine architecture or switched to `.ninfer` loading.
