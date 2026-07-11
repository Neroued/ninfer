# L2 Model Card Design — qwen3.6-ultraspeed

> Status: implemented design/reference. Date: 2026-06-26; §4 schedule updated 2026-06-27 to the
> implemented L1 op signatures; status synchronized after M2 implementation.
> The L2 model card (`config.h`/`model.h`/`qwen3_6_27b.cpp`) and the `Engine`
> (`engine.h`/`engine.cpp`) now exist. This document remains the design/reference for that
> implementation and for future graph/fusion work. See
> [`archive/pre-optimization/plans/l2-model-card-m2.md`](archive/pre-optimization/plans/l2-model-card-m2.md)
> for retained M2 history.
> Scope: the **L2 model card** — the hand-written static forward schedule for Qwen3.6-27B that
> issues L1 kernel calls in order, plus the prefill/decode drivers and KV/state lifecycle. It
> sits above L1 kernels and L0 infra. See [`design.md`](design.md) §5/§6/§7 (system architecture
> & data flow), [`l0-infrastructure-design.md`](l0-infrastructure-design.md) (the layer beneath),
> and [`qwen3.6-27b-architecture.md`](qwen3.6-27b-architecture.md) §6/§9 (computation order). The
> precision/quant seam — the `Weight` handle that replaces `LinearW` — is specified in
> [`weight-handle-design.md`](weight-handle-design.md).
> Current implementation files: `include/qus/model/{config,model}.h`,
> `src/model/qwen3_6_27b.cpp`, `include/qus/runtime/engine.h`, and `src/runtime/engine.cpp`.

---

## 1. Purpose, principle & decisions

The card is the one layer that is deliberately **固化 (frozen)**: it *is* the static compute
graph. No dynamic graph is built at runtime — the schedule is plain C++.

**The tension.** The card is several levels above the actual operators. Too much abstraction
(virtual layers, a runtime op-graph) hides the structure the optimizer needs and fights the
M4–M5 endgame (fusion → CUDA-graph/megakernel for decode). Too little (everything hardcoded and
tangled) makes the few changes we *do* want — precision/quant swaps — expensive.

**Dominant principle (resolved):** optimize for **minimum indirection and maximum optimizer
visibility**. The structure is **hardcoded and specialized** for Qwen3.6-27B; the *only*
deliberately-soft seam is **precision/quantization**. A structurally different model later =
a **new card** reusing L0/L1, not generic infra in this one.

### Decisions log

| # | Decision | Choice |
|---|---|---|
| 1 | Flexibility scope | Quant/precision only is soft; structure hardcoded; min indirection |
| 2 | Engine ⟂ card boundary | Thin `Qwen3_6_27B` object owns bindings + schedule (incl. the on-device argmax kernel); `Engine` owns resources + outer loop + next-token readback + bench |
| 3 | Linear/quant seam | One tagged `Weight` handle (dense = `BF16_CTRL`/`FP32_CTRL` qtype) + one generic `linear()` verb; precision is *data*; L1 dispatches per `qtype` (see [`weight-handle-design.md`](weight-handle-design.md)) |
| 4 | Prefill vs decode | Two drivers `prefill()` / `decode_step()` over shared helpers taking a runtime `Phase` |
| 5 | Graph-readiness | Honor the 3 cheap invariants now; defer capture/fusion/megakernel/overlay to M4–M5 |
| 6 | No virtual dispatch | No `Layer::forward` vtables, no runtime op-graph; straight-line C++ over `constexpr` dims |
| 7 | Impl selection | Stays in the L1 dispatcher (by dims+phase); the card expresses *what* runs, not *which impl* |

---

## 2. Layering & ownership

| Actor | Layer | Owns | Does NOT own |
|---|---|---|---|
| `Engine` | runtime/driver | `DeviceContext`, the 3 `DeviceArena`s, `WeightStore`, `KVCache`, `GdnState`, persistent `StepState`, the `Qwen3_6_27B` object; the outer decode loop, next-token readback (D2H→host) + EOS, bench | the schedule logic (incl. the argmax *kernel*, which runs on-device inside `decode_step`) |
| `Qwen3_6_27B` | L2 card | the **resolved weight bindings** (constexpr-sized per-layer structs) + the **schedule** (`prefill`/`decode_step` + block helpers) | physical memory — holds **references** to the Engine's resources, no per-step heap state |
| L0 | infra | bytes + storage mechanics | anything model-specific |

**The bind step (the key seam).** `WeightStore` knows tensors by `(ModuleKind::TextCore,
SourceKind, layer)`. Mapping that by-id storage to named per-layer fields (`q_proj`,
`gdn.in_q`, `conv1d`, …) is model-specific, so it runs **once, in the `Qwen3_6_27B`
constructor**, producing a fixed array of pointer-structs. After bind, the hot path never
touches `WeightStore` or does a name/id lookup.

**Lifecycle.** `Engine::load(path)` → `WeightStore.load` (validate + upload) → size
`KVCache`/`GdnState`/`StepState` from `(max_context, dims)` → construct `Qwen3_6_27B` (binds).
Then `Engine::generate(ids)` drives `prefill` once and `decode_step` in a loop, reading the
next token at the **boundary** (host) for EOS/emit.

---

## 3. Core data structures

All dims are `constexpr`, so loop bounds / head counts / the 3:1 schedule fold at compile time.

### 3.1 `ModelConfig` — frozen truth + schedule math

```cpp
struct ModelConfig {
  int hidden=5120, n_layers=64, intermediate=17408, vocab=248320;
  int gdn_conv_k=4, gdn_k_heads=16, gdn_k_dim=128, gdn_v_heads=48, gdn_v_dim=128;
  int n_q=24, n_kv=4, head_dim=256, rotary_dim=64;          // partial rope 0.25 -> 64
  int full_interval=4; float rms_eps=1e-6f, rope_theta=1e7f;
  static constexpr bool is_full(int l){ return (l+1)%full_interval==0; }
  static constexpr int  n_full(){ return n_layers/full_interval; }    // 16
  static constexpr int  n_gdn (){ return n_layers - n_full(); }       // 48
  static constexpr int  full_idx(int l){ return (l+1)/full_interval - 1; }  // l=63 -> 15
  static constexpr int  gdn_idx (int l){ return l - (l+1)/full_interval; }  // l=62 -> 47
};
inline constexpr ModelConfig kCfg{};
```

`full_idx` / `gdn_idx` map an absolute layer to its slot in the 16-entry `KVCache` / 48-entry
`GdnState`.

### 3.2 The `Weight` handle — the one soft seam (precision is data)

A seam weight is one tagged `Weight` (the renamed `QuantWeight`); **dense is just
`qtype = BF16_CTRL`/`FP32_CTRL`**, so there is no dense-vs-quant union. Seam params bind as
`const Weight*` and the one generic `linear()` verb dispatches on `qtype` (§5). Full contract:
[`weight-handle-design.md`](weight-handle-design.md).

```cpp
struct MlpW { const Weight *gate, *up, *down; }; // seam params are Weight; precision is the qtype
```

### 3.3 Per-layer bindings — flat structs mirroring `SourceKind`

```cpp
struct FullLayerW {                 // 16
  const Tensor* input_norm;                                    // InputLayernorm (dense)
  const Weight *q_proj, *gate_proj, *k_proj, *v_proj, *o_proj; // AttnQ/Gate/K/V/O (seam)
  const Tensor *q_norm, *k_norm;                               // AttnQNorm/AttnKNorm (dense)
  const Tensor* post_attn_norm; MlpW mlp;                      // PostAttnLayernorm + MLP
};
struct GdnLayerW {                  // 48
  const Tensor* input_norm;
  const Weight *in_q, *in_k, *in_v, *in_z, *in_a, *in_b;       // GdnInProj{Q,K,V,Z,A,B} (in_a/in_b = BF16_CTRL)
  const Tensor* conv1d;                                        // GdnConv1d (depthwise k=4 over [q;k;v])
  const Tensor *a_log, *dt_bias;                               // GdnALog / GdnDtBias (fp32 control)
  const Tensor* gdn_norm;                                      // GdnNorm (gated rmsnorm)
  const Weight* out_proj;                                      // GdnOutProj (seam)
  const Tensor* post_attn_norm; MlpW mlp;
};
```

Two **flat structs** rather than a shared base — at "min indirection" the small duplication
(`input_norm`/`post_attn_norm`/`mlp`) reads better than an inheritance hop, and the loop already
dispatches by type. `MlpW` is the one shared sub-struct (the MLP is identical in both). Seam params
are `const Weight*`; always-dense params (`*_norm`/`conv1d`/`a_log`/`dt_bias`) stay `const Tensor*`
(§3.2). The §4 sketches write `w.q_proj` for the bound `*w.q_proj` for brevity.

### 3.4 What the packer fixes (verified against `tools/q5090_convert/tensor_plan.py`)

The card binds these as-is; **no in-card slicing** (the splits happen offline):

| Logical | Stored as | qtype | Note |
|---|---|---|---|
| `embed_tokens` | row-grouped | **Q6G64** | quantized table; gather dequantizes one row → `embed_` is a `Weight*` |
| `lm_head` | tiled | **Q6G64** | quantized GEMV; precision lever via the `Weight` handle |
| attn `q_proj` | `q_proj.q` per-head `view[24,512,5120][:, :256]` | Q4 | per-head interleaved `[q\|gate]`; gate is a **separate** tensor (NOT contiguous rows 0:6144) |
| attn `gate_proj` | `q_proj.gate` per-head `view[24,512,5120][:, 256:]` | Q5 | applied after attention as `⊙ σ(gate)` |
| attn `k_proj`/`v_proj`/`o_proj` | tiled | Q4 / Q5 / Q5 | |
| GDN `in_q`/`in_k`/`in_v` | row-slices of `in_proj_qkv` (0:2048 / 2048:4096 / 4096:10240) | Q4 / Q4 / Q5 | |
| GDN `in_z` / `out_proj` | tiled | Q5 / Q5 | |
| GDN `in_a` / `in_b` | contiguous | **bf16 dense** | tiny `[48,5120]` projections → dense `Weight` (`BF16_CTRL`) |
| `conv1d.weight` | contiguous | **bf16 dense** | depthwise k=4 |
| all `*norm.weight` | contiguous | **bf16** | stored **verbatim** (see §7) |
| `A_log` / `dt_bias` | contiguous | **fp32** | stored **verbatim** |
| MLP `gate`/`up`/`down` | tiled | Q4 / Q4 / Q5 | |

The mixed qtypes are exactly what the `Weight` handle + per-`qtype` dispatch exists for.

### 3.5 `StepState` — persistent device buffers (the graph-invariant carrier)

```cpp
struct StepState {        // allocated once in a persistent region (Engine-owned), stable addresses
  Tensor token;   // [1]      I32  device  — autoregressive cursor: embed reads it; argmax writes it
  Tensor pos;     // [1]      I32  device  — current position; read by rope/kv-append/attn; +1 in-step
  Tensor logits;  // [vocab]  device       — lm_head output for the scored position
};
```

`token` is dual-used within a step (embed reads at the start, argmax overwrites at the end) —
safe because embed strictly precedes argmax, and across steps the value flows through the buffer.
That on-device feedback is what lets decode replay as one captured graph with **no host-side
feedback**. Prefill uploads the prompt through a separate staging/workspace buffer and deposits
its first result into `token`.

`pos` is the **single logical position**, but tracked in two mirrored places: the device
`io_.pos` (what kernels read — graph-friendly) and the host `KVCache.pos` (eager bookkeeping +
slot sizing). v1 keeps them in lock-step (`prefill` sets both to `T`; `decode_step` increments
both). At M5 the device copy is authoritative *inside* the captured region and the host counter
is reconciled at the boundary.

---

## 4. The schedule

One dispatch per layer; the MLP tail is shared. Shorthand: `s = ctx_.stream`; `scratch(shape)` and
`scratch_f32(shape)` = `work_` bump-arena allocs (BF16 / FP32); `eps = kCfg.rms_eps`;
`kAttnScale = 1/√256`, `kGdnScale = 1/√128`; bound weights are dereferenced (`*w.q_proj`). Calls use
the **real L1 signatures** (`l1-operator-catalog.md` §3): out-param `(inputs…, out, stream)`; the
phase-split ops pick the `_prefill` / `_decode`(`_recurrent`) entry by `ph`. `rmsnorm` / `l2norm` /
`causal_conv1d` / `gated_delta_rule` write a **distinct** out (not in place); `rope` /
`sigmoid_mul` / `residual_add` are in place.

```cpp
void attn_mix(const FullLayerW& w, Tensor& x, int fidx, Phase ph) {
  Tensor h = scratch({5120, T});
  rmsnorm(x, *w.input_norm, eps, /*unit_offset=*/true, nullptr, h, s);          // input layernorm
  Tensor q = scratch({256,24,T}), gate = scratch({256,24,T});
  Tensor k = scratch({256,4,T}),  v    = scratch({256,4,T});
  linear(h, *w.q_proj, q, s);   linear(h, *w.gate_proj, gate, s);
  linear(h, *w.k_proj, k, s);   linear(h, *w.v_proj, v, s);
  Tensor qn = scratch({256,24,T}), kn = scratch({256,4,T});
  rmsnorm(q, *w.q_norm, eps, true, nullptr, qn, s);                             // q/k-norm, per head dh=256
  rmsnorm(k, *w.k_norm, eps, true, nullptr, kn, s);
  rope(io_.pos, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);                   // partial NeoX, in place
  Tensor a = scratch({256,24,T});
  gqa_attention(qn, kn, v, io_.pos, kAttnScale, kv_, fidx, work_, a, s);
  sigmoid_mul(gate, a, s);                                                      // a *= σ(gate)
  Tensor o = scratch({5120, T});
  linear(a, *w.o_proj, o, s);
  residual_add(o, x, s);                                                        // x += o
}

void gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
  Tensor h = scratch({5120, T});
  rmsnorm(x, *w.input_norm, eps, true, nullptr, h, s);
  Tensor qkv = scratch({10240, T});                                            // q|k|v contiguous
  linear(h, *w.in_q, qkv.slice(0, 0,    2048), s);                             // [2048,T]
  linear(h, *w.in_k, qkv.slice(0, 2048, 2048), s);
  linear(h, *w.in_v, qkv.slice(0, 4096, 6144), s);
  Tensor a = scratch({48,T}), b = scratch({48,T});
  linear(h, *w.in_a, a, s);  linear(h, *w.in_b, b, s);                         // bf16 dense
  Tensor qkv_c = scratch({10240, T});
  causal_conv1d_silu(qkv, *w.conv1d, st_.conv[gidx], qkv_c, s);                 // T dispatch inside
  Tensor g = scratch_f32({48,T}), beta = scratch_f32({48,T});
  gdn_gating(a, b, *w.a_log, *w.dt_bias, g, beta, s);                          // fp32 g/beta
  Tensor qn = scratch({128,16,T}), kn = scratch({128,16,T});
  l2norm(qkv_c.slice(0, 0,    2048).view({128,16,T}), 1e-6f, qn, s);          // per-head dk=128
  l2norm(qkv_c.slice(0, 2048, 2048).view({128,16,T}), 1e-6f, kn, s);
  Tensor vv = qkv_c.slice(0, 4096, 6144).view({128,48,T});
  Tensor o = scratch({128,48,T});
  gated_delta_rule(qn,kn,vv, g,beta, kGdnScale, work_, st_.ssm[gidx], o, s);    // T dispatch inside
  Tensor z = scratch({128,48,T}); linear(h, *w.in_z, z.view({6144,T}), s);     // gate z
  Tensor on = scratch({128,48,T});
  rmsnorm(o, *w.gdn_norm, eps, /*unit_offset=*/false, &z, on, s);             // gated norm: plain w · SiLU(z)
  Tensor out = scratch({5120, T});
  linear(on.view({6144,T}), *w.out_proj, out, s);
  residual_add(out, x, s);
}

void mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph) {
  Tensor h = scratch({5120, T});
  rmsnorm(x, *post_norm, eps, true, nullptr, h, s);
  Tensor g = scratch({17408, T}), u = scratch({17408, T});
  linear(h, *m.gate, g, s);  linear(h, *m.up, u, s);
  Tensor a = scratch({17408, T}); silu_mul(g, u, a, s);                         // SwiGLU
  Tensor d = scratch({5120, T});  linear(a, *m.down, d, s);
  residual_add(d, x, s);
}

void run_layers(Tensor& x, Phase ph) {
  for (int l = 0; l < 64; ++l) {
    if (ModelConfig::is_full(l)) attn_mix(full_[full_idx(l)], x, full_idx(l), ph);
    else                         gdn_mix (gdn_ [gdn_idx (l)], x, gdn_idx (l), ph);
    mlp_tail(layer_post_norm(l), layer_mlp(l), x, ph);   // post_attn_norm + MlpW from the active struct
  }
}

void prefill(span<const int> ids) {                       // T = ids.size()
  Tensor x = scratch({5120, T}); embedding(upload(ids), *embed_, x, s);         // Q6 dequant gather
  run_layers(x, Phase::Prefill);
  Tensor xf = scratch({5120, T}); rmsnorm(x, *final_norm_, eps, true, nullptr, xf, s);
  linear(xf.slice(1, T-1, 1), lm_head_, io_.logits, s);   // last position only -> [vocab,1]
  argmax(io_.logits, io_.token, s);                        // first token -> device cursor
  kv_.pos = T; set_pos(io_.pos, T); work_.reset();         // host + device position in lock-step
}

void decode_step() {                                       // identical kernel sequence every call
  Tensor x = scratch({5120, 1}); embedding(io_.token, *embed_, x, s);           // reads device cursor
  run_layers(x, Phase::Decode);
  Tensor xf = scratch({5120, 1}); rmsnorm(x, *final_norm_, eps, true, nullptr, xf, s);
  linear(xf, lm_head_, io_.logits, s);
  argmax(io_.logits, io_.token, s);                        // next token -> same cursor
  advance_pos(io_.pos); kv_.advance(); work_.reset();      // pos += 1 (device authoritative; kv appended @pos inside)
}
```

Notes (aligned to the implemented L1 ops):
- `linear(x[K,T], const Weight& w[N,K], out[N,T], s)`; `embed_`/`lm_head_` are `Weight` (Q6); every
  projection goes through this one verb (§5).
- `gated_delta_rule` takes **separate** q/k/v views (`[128,16,T]`,`[128,16,T]`,`[128,48,T]`) sliced
  from the conv'd `qkv`; it updates `st_.ssm[gidx]` in place; `_recurrent` and `_chunked` take
  `work_` for boundary cast scratch, and `_chunked` also takes `chunk_size=64`. `causal_conv1d`
  updates `st_.conv[gidx]` in place and applies SiLU.
- **`GdnState.ssm[gidx]` layout = `[dk=128, dv=128, Hv=48]` fp32, AR-transposed**, with the **grouped**
  GVA head map `h_qk = h_v // (H_v/H_qk) = h_v // 3` (confirmed via vLLM; handled inside
  `gated_delta_rule` — the card only slices q/k/v). Construct `GdnState` with this layout.
- Scales: attention `1/√256`, GDN q-scale `1/√128`; norm convention `(1+w)` except the GDN gated norm
  (`unit_offset=false` + `z`); MRoPE reduces to plain partial RoPE for text-only v1.

---

## 5. The precision/quant seam in action

The card always calls one verb: `linear(out_or_ws, x, const Weight&)`. Dispatch is a single
`switch (w.qtype)` in the L1 wrapper: `Q4G64`/`Q5G64`/`Q6G64`/`W8G32` → the matching monomorphized
quant GEMV; `BF16_CTRL`/`FP32_CTRL` → the dense GEMV (the wrapper projects the handle via
`as_dense(w)` and reuses the bf16/fp32 kernel — no new op). Precision is therefore **data**:
repacking a tensor (e.g. `lm_head` Q6→bf16, or any projection's bit-width) only flips the bound
handle's `qtype`, and the **same schedule runs**. Each binding's `qtype` is fixed at load, so the
wrapper resolves the tag **once** (before any CUDA-graph capture) and the per-step kernel sequence
stays identical and fully specialized. Full contract:
[`weight-handle-design.md`](weight-handle-design.md).

**Resolved lever:** `lm_head` is Q6G64 today; the bf16↔W-bit decode-bandwidth-vs-quality
experiment (`design.md` §9/§15) is a repack-and-rerun, never a schedule edit.

---

## 6. CUDA-graph readiness — cheap invariants now, capture later

v1 is the eager, correctness-first driver (`design.md` §11). The card honors only the invariants
that cost ~nothing and prevent a painful M5 refactor:

1. **Fixed per-step decode sequence** — `decode_step` emits the same 64-layer kernel sequence
   every call. The 3:1 split is `constexpr`, so `is_full(l)` is *not* data-dependent.
2. **Per-step varying data lives in device buffers** — `pos`/`positions`/`token` are read/written
   on-device; **never** passed as host scalars (which would bake step-0 values into the graph).
3. **Stable addresses** — `work_` is a bump-reset arena with identical alloc order each step →
   stable offsets; `io_` buffers are persistent. The only host↔device touch is the next-token
   readback, done by the Engine **at the step boundary**, outside the captured region.

**Anti-patterns to avoid** (these are what make graphs hard later): host-scalar `pos`; reading a
value back mid-step and branching on it; scratch whose address depends on history; launch dims
that scale with `pos`. **Caveat:** GDN recurrent decode is naturally fixed-size (context-independent
state) and graph-perfect; **GQA full-attention decode** is the one kernel whose work grows with
`pos` — its graph-friendly forms (fixed-launch persistent/grid-stride reading `pos` from device,
or bucketed/piecewise graphs) are an **L1** concern, kept open by the card passing `pos` by device
pointer.

**Deferred to M4–M5:** actual `cudaGraph` capture/replay, kernel fusion, megakernel, and the
liveness-based workspace overlay (`l0-infrastructure-design.md` §9).

---

## 7. Kernel ↔ packer contract (pinned; authoritative source = the Python packer)

The packed file stores **everything verbatim** — `tools/q5090_convert/layouts.py::encode_contiguous`
is a pure dtype cast with no math, and `_prepare_source` only does `row_slice`/`reshape`. There is
**no** offline folding of `+1`, log-decay `exp`, or any model-semantic transform. Therefore the
**runtime kernels own all model-semantic math**, and the card merely names the verbs:

- RMSNorm `(1+w)` for input/post/final and q/k norms; **plain `w`** for the GDN gated norm.
- `g = -exp(A_log)·softplus(a + dt_bias)` (A_log/dt_bias raw, fp32, softplus guarded ~20).
- `β = σ(b)`; q/k L2-norm then q scaled by `1/√128`; attention `1/√256`; partial RoPE.

This reconciles a documentation conflict: `design.md` §10 (packer folds nothing) is correct;
`architecture.md` §5's note that the packer adds `+1` was **wrong** and has been corrected.

---

## 8. Parity / debug seam

Per-layer parity (`design.md` §12) needs activation dumps that never touch the hot path or a
captured graph. A single compile-time switch, zero-cost in release:

```cpp
template <class Tap = NullTap> struct Qwen3_6_27B { ... };  // NullTap::enabled == false
// at fixed tap points (after embed, each mixer, each MLP, final norm, logits):
if constexpr (Tap::enabled) tap_(TapId::AfterMixer, l, x);  // D2H copy + tagged dump
```

Release (`NullTap`) compiles the taps out entirely, leaving the decode body pristine for capture.
The parity build (`FileTap`) is used only for debug/parity runs and never during capture.

---

## 9. What the card does NOT own

Per the frozen/flexible split (`design.md` §4.2–4.3): *impl selection* (chunked vs recurrent,
flash vs single-query, per-`qtype` GEMV) → L1 dispatcher; *fusion / CUDA-graph / megakernel* →
M4–M5; *liveness workspace overlay* → deferred. Adding a structurally different model later = a
**new L2 card** reusing L0/L1, never edits to this one.

---

## 10. File layout

```
include/qus/model/
  config.h        # ModelConfig (constexpr) + is_full/full_idx/gdn_idx + kCfg
  model.h         # Qwen3_6_27B: bindings (FullLayerW/GdnLayerW/MlpW over Weight/Tensor), drivers, StepState ref
src/model/
  qwen3_6_27b.cpp # bind() (WeightStore -> per-layer structs) + attn_mix/gdn_mix/mlp_tail + drivers
```

`Engine` (in `src/runtime/`) owns the resources + `StepState` + outer loop and constructs the card.

---

## 11. Open questions / deferred

- **GQA decode graph form** (fixed-launch vs bucketed) — L1 decision; card passes `pos` by device
  pointer to keep both open (§6).
- **`lm_head` precision** — mechanism resolved (the `Weight` handle); value (Q6 vs bf16 vs W4) decided by
  measurement (§5, `design.md` §9).
- **MTP / vision** — future cards/extensions; the `ModuleKind`/`SourceKind` space already reserves
  ids. Out of scope here.
- **`StepState.logits` dtype** — follows `lm_head` output precision (bf16 vs fp32); argmax adapts.

---

## 12. Sources

- `design.md` §4–§7, §9–§12 (architecture, data flow, numerics, validation, ladder).
- `l0-infrastructure-design.md` (the layer beneath; arena/KV/state/tensor contracts).
- `qwen3.6-27b-architecture.md` §5–§9 (numerics conventions, ordered component math, schedule).
- `tools/q5090_convert/{tensor_plan,layouts,convert}.py` (authoritative packer: splits, qtypes,
  verbatim control tensors).
- `include/qus/core/{tensor,weight_store,kv_cache,state_store,arena,device}.h` (L0 APIs the card uses).
