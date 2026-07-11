# L1 Operator Catalog — qwen3.6-ultraspeed

> Status: implemented API contract. Synchronized 2026-07-11.
> Scope: the public L1 surface in `include/qus/kernels/` consumed by the Qwen3.6 model card.

## 1. Contract

The public API names mathematical or model semantics. Backend details stay below the wrapper:

- qtype and layout dispatch are internal;
- GEMV/GEMM, small-T/large-T, decode/prefill, and recurrent/chunked selection are internal;
- fused operations have separate public names when their inputs, outputs, or mathematical meaning
  differ from the unfused primitive;
- fixed MTP schedule helpers and prompt-only GQA decomposition helpers are model-private;
- signatures order arguments as `(read-only inputs, scalar parameters, state/workspace, outputs,
  stream)` and always place `cudaStream_t` last.

The project has no compatibility layer. Renamed or replaced project-owned APIs are deleted.

## 2. Public primitive operators

### `linear`

```cpp
void linear(const Tensor& x, const Weight& w, Tensor& out,
            WorkspaceArena& ws, cudaStream_t stream);
```

Computes `out[:,t] = W @ x[:,t]`. The wrapper dispatches by weight format, registered shape, and
token-count regime. GEMV/GEMM and quantized/dense backend names never appear in the public API.

### `embedding`

```cpp
void embedding(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream);
```

Embedding lookup. Dense BF16 copy and Q6 row-split dequant-gather are internal qtype variants.

### Normalization and position

```cpp
void rmsnorm(const Tensor& x, const Tensor& weight, float eps,
             bool unit_offset, Tensor& out, cudaStream_t stream);
void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                float eps, Tensor& out, cudaStream_t stream);
void l2norm(const Tensor& x, float eps, Tensor& out, cudaStream_t stream);

void rope(const Tensor& positions, int rotary_dim, float theta,
          Tensor& x, cudaStream_t stream);
void rope(const Tensor& positions, int rotary_dim, float theta,
          Tensor& q, Tensor& k, cudaStream_t stream);
```

`layer_norm` is the affine Vision normalization primitive. It uses FP32 mean/variance and affine
math and emits BF16. `rope` dispatches the fixed Qwen3.6 Text 1-D, Text three-axis MRoPE, and Vision
two-axis layouts from the position rank and tensor shape. Vision Q/K may be strided views into the
packed QKV projection.

The unary `rope` overload infers Q/K specialization from the head-count shape. The two-tensor
overload preserves the combined Q+K launch.

### Attention

```cpp
void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                   const Tensor& positions, float scale, KVCache& kv, int layer,
                   WorkspaceArena& ws, Tensor& out, cudaStream_t stream);
void vision_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                      const Tensor& cu_seqlens, WorkspaceArena& ws,
                      Tensor& out, cudaStream_t stream);
```

Appends K/V and computes causal grouped-query attention. T-regime and BF16/I8 KV-cache dispatch are
internal. Prompt-only cache append and cached-attention decomposition are not public operators.

`vision_attention` is separate because it is packed non-causal MHA with 16 equal Q/K/V heads,
head dimension 72, independent `cu_seqlens` segments, and no KV cache. Its workspace does not grow
with the square of segment length.

### GDN recurrence

```cpp
void causal_conv1d_silu(const Tensor& x, const Tensor& weight,
                        Tensor& conv_state, Tensor& out, cudaStream_t stream);
void causal_conv1d_silu(const Tensor& x, const Tensor& weight,
                        const Tensor& conv_state_in, Tensor& conv_state_out,
                        Tensor& out, cudaStream_t stream);

void gdn_gating(const Tensor& a, const Tensor& b,
                const Tensor& A_log, const Tensor& dt_bias,
                Tensor& g, Tensor& beta, cudaStream_t stream);

void gated_delta_rule(const Tensor& q, const Tensor& k, const Tensor& v,
                      const Tensor& g, const Tensor& beta, float scale,
                      WorkspaceArena& ws, Tensor& ssm_state, Tensor& out,
                      cudaStream_t stream);
void gated_delta_rule(const Tensor& q, const Tensor& k, const Tensor& v,
                      const Tensor& g, const Tensor& beta, float scale,
                      WorkspaceArena& ws, const Tensor& ssm_state_in,
                      Tensor& ssm_state_out, Tensor& out, cudaStream_t stream);
```

`causal_conv1d_silu` explicitly names the fused SiLU semantics. Its in-place form uses the decode
launcher at T=1 and the prefill launcher otherwise. The distinct-state form preserves prefix-append
prefill semantics.

`gated_delta_rule` selects recurrent or fixed chunk-64 execution internally. Chunk size is a kernel
policy, not a public parameter. The distinct-state form preserves prefix-append prefill state
publication.

### Elementwise and selection

```cpp
void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);
void sigmoid_mul(const Tensor& gate, Tensor& x, cudaStream_t stream);
void residual_add(const Tensor& y, Tensor& x, cudaStream_t stream);
void add_bias(const Tensor& bias, Tensor& x, cudaStream_t stream);
void gelu(Tensor& x, GeluMode mode, cudaStream_t stream);
void scatter(const Tensor& src, const Tensor& indices, Tensor& dst, cudaStream_t stream);
void vision_pos_embed_add(const Tensor& table, const Tensor& indices,
                          const Tensor& weights, Tensor& x, cudaStream_t stream);
void argmax(const Tensor& logits, Tensor& out, cudaStream_t stream);
void sample(const Tensor& logits, Tensor& out, const SamplingConfig* config,
            const std::int32_t* pos_base, std::int32_t purpose, cudaStream_t stream);
```

`vision_pos_embed_add` explicitly names its fused residual semantics: four-corner interpolated
positions are rounded to BF16 before the BF16 residual add. `scatter` overwrites destination
columns and is used to inject merger output into image/video placeholder positions.

## 3. Public fused operators

These entries remain separate because fusion changes the operation's input/output contract or
mathematical semantics.

| Operator | Semantics | Internal dispatch |
|---|---|---|
| `linear_pair` | Two projections sharing one input | W8 K/V large-T fusion; Q5 V/Z T=1 fusion; otherwise two `linear` calls |
| `linear_add` | `residual += W @ x` | Q5 shape and T-specific fused kernels, otherwise `linear` + `residual_add` |
| `linear_swiglu` | Gate/up projection followed by SwiGLU | T=1, large-T, and unfused fallback |
| `attn_input_proj` | Full-attention Q/Gate/K/V projections | grouped large-T kernel, ordinary linears for small T |
| `gdn_input_proj` | GDN Q/K/V projections into contiguous QKV | grouped large-T kernel, ordinary linears plus packing for small T |
| `gdn_gating_proj` | A/B projections followed by GDN gate preparation | dense shape/T-specific fused implementation |
| `gated_rmsnorm` | plain-weight RMSNorm followed by `SiLU(z)` gate | shared RMSNorm launcher with gated specialization |

## 4. Snapshot-state operators

Snapshot storage changes state semantics and therefore remains explicit:

```cpp
void causal_conv1d_silu_snapshot(const Tensor& x, const Tensor& weight,
                                 Tensor& conv_states, const Tensor& initial_slot,
                                 Tensor& out, cudaStream_t stream);
void gated_delta_rule_snapshot(const Tensor& q, const Tensor& k, const Tensor& v,
                               const Tensor& g, const Tensor& beta, float scale,
                               WorkspaceArena& ws, Tensor& ssm_states,
                               const Tensor& initial_slot, Tensor& out,
                               cudaStream_t stream);
```

## 5. Model-private CUDA helpers

The following declarations live under `src/model/`, not `include/qus/kernels/`:

- `gqa_prompt_ops.h`: prompt-only KV append and cached attention for the MTP schedule;
- `mtp_ops.h`: MTP packing, verification-input preparation, accept/commit, remapping, counters,
  and GDN snapshot-slot updates.
- `vision_ops.h`: processor-grid control metadata and the fixed F32-patch to BF16 upload cast.

They remain CUDA wrappers but are not part of the reusable L1 operator contract.

## 6. Dispatch ownership

| Public operator | Wrapper dispatch keys |
|---|---|
| `linear` | qtype, layout, `(N,K)`, T regime |
| `linear_pair` | qtype, paired shapes, T regime |
| `embedding` | table qtype/layout |
| `rope` | one/two tensor overload and head-count shape |
| `gqa_attention` | T regime and KV-cache dtype |
| `causal_conv1d_silu` | state form and T |
| `gated_delta_rule` | state form and T; chunk size fixed internally |
| fused projection ops | registered weights, shapes, and T regime |

Launcher and kernel symbols may use hardware- and policy-specific names. Those names are private and
are intentionally not mirrored into the API.
