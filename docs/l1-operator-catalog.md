# L1 Operator Catalog — qwen3.6-ultraspeed

> Status: implemented API contract. Date: 2026-06-26; status synchronized 2026-06-27.
> Scope: the **complete, finalized list of L1 operators** the text-only v1 model card calls, each
> with its generic (algorithm-based) name and its public **api signature**. This is the naming +
> declaration contract; the 13 public headers now exist in `include/qus/kernels/` and the current
> implementations live under `src/kernels/`.
>
> This catalog *inherits* its structure and seam from two approved docs and does not re-derive them:
> - **[`l1-kernel-layering.md`](l1-kernel-layering.md)** — the api→wrapper→launcher→kernel split,
>   per-op headers, and the rule that validation+dispatch live in the wrapper.
> - **[`weight-handle-design.md`](weight-handle-design.md)** — the single tagged `Weight` handle
>   (dense = `BF16_CTRL`/`FP32_CTRL`); the precision seam ops take `const Weight&` and `switch(qtype)`.
>
> It **supersedes the operator names** in [`qwen3.6-27b-architecture.md`](qwen3.6-27b-architecture.md)
> §10 and is the consumer-side contract for [`l2-model-card-design.md`](l2-model-card-design.md) §4.

---

## 1. Purpose & decisions

The model card (L2) issues a fixed schedule of kernel calls. This catalog fixes the
**operator surface**: what ops exist, what they're called, and what their signatures are. The
implemented card and kernels are expected to keep matching this API/naming contract.

**Naming guiding rule:** name by the **algorithm/operation**, not by the model that uses it. Drop
brand prefixes; keep field-standard algorithm names. (`gemma_rmsnorm` is a brand name for a
parameterized standard op → it becomes `rmsnorm` + a flag; `gated_delta_rule`, `rope`, `gqa` are
algorithm names → kept.)

### Decisions log (this session)

| # | Decision | Choice |
|---|---|---|
| 1 | Naming principle | Algorithm-named; no model/brand prefixes; variations are parameters |
| 2 | Variant expression | Generic **runtime-parameterized** signatures; the impl/dispatch monomorphizes internally |
| 3 | Granularity | Declare **unfused primitives** now (M2 baseline); fused ops are added later as new named ops (M4) |
| 4 | Signature order | `(read-only inputs…, params…, out/in-place…, stream)` — from committed `silu_and_mul.h` |
| 5 | Precision seam | `linear` + `embed_gather` take `const Weight&`; everything else is dense `const Tensor&` |
| 6 | `gemma_rmsnorm` | Resolved → one parameterized `rmsnorm` (the `(1+w)` and the gate are params) |

### The `gemma_rmsnorm` resolution (research-backed)

The norm **is** the unit-offset form `x/rms(x)·(1+w)` — confirmed authoritatively: vLLM aliases
`GemmaRMSNorm as Qwen3NextRMSNorm` and applies `weight.float()+1.0`
(`vllm/.../layers/layernorm.py:129`, `models/qwen3_next.py:358`), and llama.cpp folds `+1` into every
`*norm.weight` except `linear_attn.norm.weight` (`convert_hf_to_gguf.py:4802`). But mathematically it
is just standard RMSNorm with the weight offset by 1 — exactly "a special parameter variation." So we
name it `rmsnorm` and make `(1+w)` a `bool unit_offset` parameter. Because our packer stores norm
weights **verbatim** (`design.md` §10, `l2-model-card-design.md` §7), the runtime kernel applies the
offset itself — it cannot be a packer trick.

---

## 2. Conventions (inherited + the deltas this catalog fixes)

Inherited verbatim from the two source docs; restated here only where this catalog pins a detail:

- **Layering** (`l1-kernel-layering.md`): every op below = one public `include/qus/kernels/<op>.h`
  plus a `wrapper/<op>.cpp` (validate + dispatch) + `launcher/<op>[_<variant>].{h,cu}` +
  `kernel/<op>[_<variant>].cuh`. There is **no** `dispatch.h` object — dispatch is the wrapper's
  routing. Phase-split ops declare `<op>_prefill` / `<op>_decode` in one header.
- **Seam type** (`weight-handle-design.md`): `linear` and `embed_gather` take `const Weight&` and
  `switch(w.qtype)`; the `BF16_CTRL`/`FP32_CTRL` arm projects via `as_dense(w)` and reuses the dense
  kernel. All other ops take `const Tensor&` weights (always-dense params: `*_norm`, `conv1d`,
  `A_log`, `dt_bias`).
- **Signature order (delta):** `(read-only inputs…, scalar params…, out-or-in-place tensors…,
  cudaStream_t stream)`. The out/in-place tensor(s) are second-to-last; `stream` is always last.
  (Matches committed `silu_and_mul.h`. Note: `weight-handle-design.md` §5.2 sketches
  `linear(out, x, w, stream)` with out *first* — that illustration should be reordered to
  `linear(x, w, out, stream)`; see §9.)
- **Variant axes:** (a) **phase** — only where the *algorithm* differs (attention, gdn rule, conv);
  expressed as `_prefill`/`_decode` entries. (b) **qtype** — the seam ops, resolved in the wrapper.
  (c) **dims** — e.g. `linear` routes decode `T==1` (GEMV) vs prefill `T>1` (GEMM) from the input
  shape; **no `Phase` param is needed on `linear`**.
- **Numerics (intrinsic, not parameters):** bf16 in/out activations; fp32 accumulate in
  norms/softmax/reductions/recurrence. Kernels own **all** model-semantic transforms — `(1+w)`,
  `-exp(A_log)`, softplus, the `1/√256` and `1/√128` scales, partial-RoPE rotation.
- **Phase ownership:** phase-split public ops encode phase in the entry name (`_prefill`,
  `_decode`, `_recurrent`, `_chunked`). The model card owns its local `Phase` enum for schedule
  helpers; no extra L1 `phase.h` public header is required by the current implementation.

---

## 3. The catalog

13 operators. Each entry: the api header, the signature(s), the variant axis, and notes. Types:
`Tensor`/`Weight`/`KVCache`/`WorkspaceArena` from L0; `cudaStream_t` last on every op.

> Shapes are written in **`ne` order** (`ne[0]` = fastest-varying, per `tensor.h`): a row-major
> `[T tokens, d]` activation is `[d, T]`. So a hidden state is `[5120, T]`, q is `[256, 24, T]`.

### 3.1 `linear` — the projection/GEMV workhorse (seam)

```cpp
// include/qus/kernels/linear.h
namespace qus::kernels {
void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
}
```

- **Used by:** every projection — `q/k/v/gate/o_proj`, `gate/up/down_proj`, GDN `in_q/in_k/in_v/in_z/in_a/in_b`, `out_proj`, and `lm_head`.
- **Variant axis:** `qtype` × dims. Wrapper `switch(w.qtype)` → `Q4G64`/`Q5G64`/`Q6G64`/`W8G128` quant launchers, `BF16_CTRL`/`FP32_CTRL` → dense launcher fed `as_dense(w)`; and routes `x` rows `T==1` (GEMV, decode) vs `T>1` (GEMM, prefill).
- **Notes:** W·A16 = dequant+matmul fused in the kernel; fp32 accumulate. `x`/`out` bf16. This replaces `design.md` §6 / arch §10's `w4a16_gemm`/`w4a16_gemv` (those are the `Q4G64` variant). Dense `in_a`/`in_b` (`BF16_CTRL`) flow through the same verb — no separate dense entry at the card.

### 3.2 `embed_gather` — embedding lookup (seam)

```cpp
// include/qus/kernels/embed_gather.h
void embed_gather(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream);
```

- **Variant axis:** `qtype`. `Q6G64` → dequant-gather one row per id; `BF16_CTRL` → plain row copy via `as_dense(table)`.
- **Notes:** `ids` `i32 [T]`; `out` `bf16 [5120, T]`. The Qwen3.6 `embed_tokens` table is `Q6G64` (quantized).

### 3.3 `rmsnorm` — all four RMSNorm uses, parameterized (dense)

```cpp
// include/qus/kernels/rmsnorm.h
void rmsnorm(const Tensor& x, const Tensor& weight, float eps,
             bool unit_offset,        // true => x/rms(x)·(1+w); false => ·w
             const Tensor* z,         // nullptr => no gate; else out *= SiLU(z) (norm-before-gate)
             Tensor& out, cudaStream_t stream);
```

- **Normalizes over the fastest dim `ne[0]`**, so the caller selects the axis by the view shape:
  - input / post-attn / final norm: `x = [D=5120, T]`, `unit_offset=true`, `z=nullptr`.
  - q-norm / k-norm: `x = [256, H, T]` (norm over `head_dim=256`), `unit_offset=true`.
  - GDN gated norm: `x = [128, 48, T]` (norm over `dv=128`), `unit_offset=false`, `z=&z`.
- **Notes:** fp32 variance; collapses vLLM's `RMSNorm` + `GemmaRMSNorm` + `RMSNormGated` into one op. Gate activation is SiLU (the model's only use); promote to an `enum Activation` only if a second use appears. Replaces arch §10 `gemma_rmsnorm` / `gemma_rmsnorm_headwise` / `gated_rmsnorm`.

### 3.4 Elementwise / activation (dense)

```cpp
// include/qus/kernels/silu_and_mul.h   (already committed)
void silu_and_mul    (const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream); // SwiGLU
// include/qus/kernels/sigmoid_gate_mul.h
void sigmoid_gate_mul(const Tensor& gate, Tensor& x, cudaStream_t stream);   // x *= σ(gate), in place
// include/qus/kernels/residual_add.h
void residual_add    (const Tensor& y,    Tensor& x, cudaStream_t stream);   // x += y, in place
```

- `silu_and_mul`: MLP SwiGLU, `out = SiLU(gate) ⊙ up`.
- `sigmoid_gate_mul`: full-attn output gate, `attn ⊙ σ(gate)`, in place on `x`.
- `residual_add`: pre-norm residual, in place on `x`. (Fused add+norm is a deferred M4 op, §8.)

### 3.5 `rope` — partial rotary position embedding (dense, in place)

```cpp
// include/qus/kernels/rope.h
void rope(const Tensor& positions, int rotary_dim, float theta,
          Tensor& q, Tensor& k, cudaStream_t stream);
```

- **In place** on `q`/`k`. Rotates the first `rotary_dim=64` of `head_dim=256` (NeoX split), passes the rest through. `positions` `i32 [T]` (decode: the single device `pos`). `theta = 1e7`.
- **Notes:** v1 = plain 1-D partial RoPE; the MRoPE `[11,11,10]` 3-axis split is added (as an optional param) only when vision lands. Replaces arch §10 `rope_partial_mrope`. Applied **after** q/k-norm.

### 3.6 `gqa_attention` — full attention (phase-split)

```cpp
// include/qus/kernels/gqa_attention.h
void gqa_attention_prefill(const Tensor& q, const Tensor& k, const Tensor& v, float scale,
                           KVCache& kv, int layer, Tensor& out, cudaStream_t stream);
void gqa_attention_decode (const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                           float scale, KVCache& kv, int layer, Tensor& out, cudaStream_t stream);
```

- **Shapes:** `q [256,24,T]`, `k/v [256,4,T]` (GQA 24 query / 4 kv heads), `out [256,24,T]`, `scale = 1/√256`.
- **`prefill`:** writes `k,v` for all `T` into `kv[layer]`; causal (flash-style) attention.
- **`decode`:** appends `k,v` at `pos` into `kv[layer]`; attends the `[0..pos]` window. `pos` is the device scalar (graph-safe).
- **Notes:** the **KV append is folded into attention** (no separate `kv_cache_append` op); `kv` is in-out. Replaces arch §10 `flash_attn_causal_gqa` / `attn_decode_gqa`.

### 3.7 GDN linear attention (Gated DeltaNet)

```cpp
// include/qus/kernels/causal_conv1d.h   (phase-split)
void causal_conv1d_prefill(const Tensor& x, const Tensor& weight,
                           Tensor& conv_state, Tensor& out, cudaStream_t stream);
void causal_conv1d_decode (const Tensor& x, const Tensor& weight,
                           Tensor& conv_state, Tensor& out, cudaStream_t stream);

// include/qus/kernels/gdn_gating.h       (GDN = Gated DeltaNet)
void gdn_gating(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                Tensor& g, Tensor& beta, cudaStream_t stream);

// include/qus/kernels/l2norm.h
void l2norm(const Tensor& x, float eps, Tensor& out, cudaStream_t stream);

// include/qus/kernels/gated_delta_rule.h (phase-split)
void gated_delta_rule_recurrent(const Tensor& q, const Tensor& k, const Tensor& v,
                                const Tensor& g, const Tensor& beta, float scale,
                                WorkspaceArena& ws, Tensor& ssm_state, Tensor& out, cudaStream_t stream);
void gated_delta_rule_chunked  (const Tensor& q, const Tensor& k, const Tensor& v,
                                const Tensor& g, const Tensor& beta, float scale, int chunk_size,
                                WorkspaceArena& ws, Tensor& ssm_state, Tensor& out, cudaStream_t stream);
```

- **`causal_conv1d`:** depthwise, causal, kernel-4, over the `[q;k;v]` = 10240 channels; **applies SiLU** (the model's fused convention — documented; parameterize to an `Activation` only if reused without it). `weight` is dense bf16; `conv_state` in-out (decode keeps a rolling width-3 state). Replaces arch §10 `causal_conv1d_prefill/decode`.
- **`gdn_gating`:** `g = -exp(A_log)·softplus(a+dt_bias)`, `beta = σ(b)`, all fp32 (`a,b [48,T]`, `A_log/dt_bias [48]`). The `exp`/softplus are runtime (weights stored raw). Replaces arch §10 `gdn_gating`.
- **`l2norm`:** per-head L2 normalize over `ne[0]=dk=128` (applied to `q`,`k`). Replaces arch §10 `l2norm_headwise`.
- **`gated_delta_rule`:** the recurrence. `recurrent` (decode, `T=1`, §7.1) updates `ssm_state` in place; both public wrappers use `ws` for bf16↔fp32 cast scratch, and `chunked` (prefill, §7.2, chunk 64, WY/UT) also uses it for stage scratch. `scale = 1/√128` (applied to `q`). GVA mapping (v-head `h` uses k/q head `h//3`) is internal. Replaces arch §10 `gated_delta_recurrent/chunked`.

### 3.8 `argmax` — greedy next-token (sampling)

```cpp
// include/qus/kernels/argmax.h
void argmax(const Tensor& logits, Tensor& out, cudaStream_t stream);
```

- `logits [vocab=248320, T]` → `out i32 [1]` (the id for the scored position). Greedy v1; a full sampler is deferred (§8). Replaces arch §10 `argmax_vocab`.

---

## 4. Coverage map — card schedule → ops

Confirms the catalog is complete for `l2-model-card-design.md` §4. (`P`=prefill, `D`=decode.)

| Card step | Op(s) | Phase |
|---|---|---|
| embed | `embed_gather` | P,D |
| input / post / final norm | `rmsnorm(unit_offset=true)` | P,D |
| q/k/v/gate/o, gate/up/down, in_*, out_proj, lm_head | `linear` | P,D |
| q-norm / k-norm | `rmsnorm(unit_offset=true)` (head view) | P,D |
| RoPE | `rope` | P,D |
| full attention (+KV) | `gqa_attention_prefill` / `gqa_attention_decode` | P / D |
| attn output gate | `sigmoid_gate_mul` | P,D |
| residual adds | `residual_add` | P,D |
| SwiGLU MLP | `silu_and_mul` (+ `linear`) | P,D |
| GDN conv+SiLU | `causal_conv1d_prefill` / `_decode` | P / D |
| GDN gates | `gdn_gating` | P,D |
| GDN q/k L2-norm | `l2norm` | P,D |
| GDN recurrence | `gated_delta_rule_chunked` / `_recurrent` | P / D |
| GDN gated out-norm | `rmsnorm(unit_offset=false, z=…)` | P,D |
| argmax | `argmax` | P,D |

---

## 5. State ownership (no separate state ops)

Per `l0-infrastructure-design.md`, KV/conv/ssm state live in L0 (`KVCache`, `GdnState`). The catalog
folds the writes into the consuming op rather than exposing standalone state kernels (arch §10's
`kv_cache_append` / `gdn_state_rw` are **not** separate ops):

- **KV append** → inside `gqa_attention_*` (`kv` is in-out).
- **conv state** → inside `causal_conv1d_*` (`conv_state` in-out).
- **ssm state** → inside `gated_delta_rule_*` (`ssm_state` in-out).

This keeps the write fused with the read it feeds and minimizes the op count on the captured decode
path.

---

## 6. Variant & dispatch summary

| Op | Header | Variants | Dispatch key (in wrapper) |
|---|---|---|---|
| `linear` | `linear.h` | 6 qtype × {gemv,gemm} | `w.qtype`, `x` rows `T` |
| `embed_gather` | `embed_gather.h` | 2 qtype | `table.qtype` |
| `rmsnorm` | `rmsnorm.h` | flags only | `unit_offset`, `z!=null` (in-kernel) |
| `silu_and_mul` | `silu_and_mul.h` | single | — |
| `sigmoid_gate_mul` | `sigmoid_gate_mul.h` | single | — |
| `residual_add` | `residual_add.h` | single | — |
| `rope` | `rope.h` | single (v1) | — |
| `gqa_attention` | `gqa_attention.h` | prefill/decode | phase (call site) |
| `causal_conv1d` | `causal_conv1d.h` | prefill/decode | phase (call site) |
| `gdn_gating` | `gdn_gating.h` | single | — |
| `l2norm` | `l2norm.h` | single | — |
| `gated_delta_rule` | `gated_delta_rule.h` | recurrent/chunked | phase (call site) |
| `argmax` | `argmax.h` | single | — |

---

## 7. Utilities (runtime-owned, not core compute ops)

Two tiny device helpers the card/runtime needs for graph-safe position handling; listed for
completeness, kept out of the compute catalog:

- **`positions` fill** — write `[0..T-1]` for prefill (an `iota` into the positions buffer).
- **`advance_pos`** — device `pos += 1` each decode step (a 1-thread kernel), so `pos` stays on
  device (`l2-model-card-design.md` §6). Prefill sets `pos = T`.

These may live as trivial helpers in the runtime rather than as L1 ops; revisit if they need
specialization.

---

## 8. Deferred (not declared now)

Per Decision #3 (primitives first), these are **not** in the v1 catalog and are added as new named
ops when their milestone arrives:

- **Fused ops (M4)** — e.g. `fused_add_rmsnorm`, `dequant_gemv` (already implied by `linear`),
  `qk_norm_rope_gate`, `conv_silu_split`, `gated_rmsnorm_out_proj`. Each is a normal
  api/wrapper/launcher/kernel set spanning what were several primitives.
- **Sampler (post-v1)** — temperature / top-k / top-p (a `sample` op beside `argmax`).
- **MTP / vision ops** — out of v1 text scope; the `ModuleKind`/`source_kind` space already reserves ids.
- **Low-precision prefill** — fp8/fp4 GEMM as a `qtype`/dtype variant axis on `linear`.

---

## 9. Reconciliation with existing docs — status

1. **`weight-handle-design.md` §5.2 — DONE.** The `linear` / `embed_gather` sketches were reordered
   to `(inputs…, out, stream)` to match the committed convention.
2. **`l2-model-card-design.md` §3.2/§5 — already in place.** The L2 doc binds the seam as
   `const Weight*` and uses the `linear` / `embed_gather` verbs; only a historical
   "`Weight` replaces `LinearW`" note remains.
3. **Legacy family-header stubs — already removed.** The L1 restructure deleted
   `gemm/norm/attention/gdn/rope/activation/sampling/dispatch.h`. There is no `dispatch.h`
   (dispatch lives in each wrapper). The per-op headers in §10 — including `rope.h` — now exist.

---

## 10. File manifest (implemented public api headers)

```
include/qus/kernels/
  linear.h              # seam (Weight)
  embed_gather.h        # seam (Weight)
  rmsnorm.h
  silu_and_mul.h        # exists
  sigmoid_gate_mul.h
  residual_add.h
  rope.h
  gqa_attention.h       # _prefill / _decode
  causal_conv1d.h       # _prefill / _decode
  gdn_gating.h
  l2norm.h
  gated_delta_rule.h    # _recurrent / _chunked
  argmax.h
```

Each spawns its `src/kernels/wrapper/<op>.cpp` + `launcher/<op>[_<variant>].{h,cu}` +
`kernel/<op>[_<variant>].cuh` per `l1-kernel-layering.md` §3. These 13 public headers are present in
the current tree; the legacy family headers were already removed (§9).

---

## 11. Sources

- `l1-kernel-layering.md` (structure), `weight-handle-design.md` (the `Weight` seam),
  `l2-model-card-design.md` §4 (the consumer schedule), `design.md` §6/§9/§11, `qwen3.6-27b-architecture.md`
  §5–§10 (op math; names superseded here).
- Upstream naming/semantics research: vLLM `layers/layernorm.py` (`GemmaRMSNorm`),
  `models/qwen3_next.py`, `layers/fla/ops/{chunk,fused_recurrent,l2norm}.py`; llama.cpp
  `convert_hf_to_gguf.py` (`Qwen3_5TextModel`, the `+1` norm fold), `src/models/delta-net-base.cpp`.
