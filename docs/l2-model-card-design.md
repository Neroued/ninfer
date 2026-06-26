# L2 Model Card Design — qwen3.6-ultraspeed

> Status: design (approved in brainstorm). Date: 2026-06-26.
> Scope: the **L2 model card** — the hand-written static forward schedule for Qwen3.6-27B that
> issues L1 kernel calls in order, plus the prefill/decode drivers and KV/state lifecycle. It
> sits above L1 kernels and L0 infra. See [`design.md`](design.md) §5/§6/§7 (system architecture
> & data flow), [`l0-infrastructure-design.md`](l0-infrastructure-design.md) (the layer beneath),
> and [`qwen3.6-27b-architecture.md`](qwen3.6-27b-architecture.md) §6/§9 (computation order). The
> precision/quant seam — the `Weight` handle that replaces `LinearW` — is specified in
> [`weight-handle-design.md`](weight-handle-design.md).
> Implements the stubs `include/qus/model/{config,model}.h` and `src/model/qwen3_6_27b.cpp`.

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
| attn `q_proj` | `q_proj.q` rows 0:6144 | Q4 | gate is a **separate** tensor |
| attn `gate_proj` | `q_proj.gate` rows 6144:12288 | Q5 | applied after attention as `⊙ σ(gate)` |
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

One dispatch per layer; the MLP tail is shared. `work_`, `kv_`, `st_`, `io_` are references the
card holds to Engine-owned resources. (Sketches are illustrative; exact kernel signatures are L1.)

```cpp
void attn_mix(const FullLayerW& w, Tensor& x, int fidx, Phase ph) {
  Tensor h    = gemma_rmsnorm(work_, x, w.input_norm);            // [T,5120], (1+w)
  Tensor q    = linear(work_, h, w.q_proj);    // [T,6144]=24*256
  Tensor gate = linear(work_, h, w.gate_proj); // [T,6144]
  Tensor k    = linear(work_, h, w.k_proj);    // [T,1024]=4*256
  Tensor v    = linear(work_, h, w.v_proj);    // [T,1024]
  qk_norm(q, w.q_norm); qk_norm(k, w.k_norm);                     // per-head dh=256, (1+w)
  rope_partial(q, k, io_.pos);                                    // first 64 dims; reads device pos
  Tensor a    = attention(work_, q, k, v, kv_, fidx, io_.pos, ph);// GQA 24/4; append@pos (decode)
  sigmoid_gate_mul(a, gate);                                      // attn ⊙ σ(gate)
  residual_add(x, linear(work_, a, w.o_proj));                   // o_proj + residual
}

void gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
  Tensor h   = gemma_rmsnorm(work_, x, w.input_norm);
  Tensor qkv = work_.alloc(BF16,{T,10240});                      // contiguous q|k|v
  linear_into(qkv.slice(0,2048),    h, w.in_q);                  // q  16*128
  linear_into(qkv.slice(2048,2048), h, w.in_k);                  // k  16*128
  linear_into(qkv.slice(4096,6144), h, w.in_v);                  // v  48*128
  Tensor z = linear(work_, h, w.in_z);                           // [T,6144]
  Tensor b = linear(work_, h, w.in_b), a = linear(work_, h, w.in_a); // [T,48] bf16 dense
  causal_conv1d_silu(qkv, w.conv1d, st_.conv[gidx], ph);         // depthwise k=4 over 10240
  Tensor g, beta;
  gdn_gates(g, beta, a, b, w.a_log, w.dt_bias);                  // fp32: g=-exp(Alog)·softplus(a+dt); β=σ(b)
  l2norm(q_of(qkv)); l2norm(k_of(qkv));                          // per-head dk=128
  Tensor o = gated_delta(work_, qkv, g, beta, st_.ssm[gidx], ph);// recurrent(D) / chunked C=64 (P)
  gated_rmsnorm(o, z, w.gdn_norm);                               // plain w · SiLU(z), dv=128
  residual_add(x, linear(work_, flatten(o), w.out_proj));       // [T,6144]→[T,5120] + residual
}

void mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph) {
  Tensor h = gemma_rmsnorm(work_, x, post_norm);
  Tensor g = linear(work_, h, m.gate), u = linear(work_, h, m.up);
  residual_add(x, linear(work_, silu_and_mul(work_, g, u), m.down)); // SwiGLU
}

void run_layers(Tensor& x, Phase ph) {
  for (int l = 0; l < 64; ++l) {
    if (ModelConfig::is_full(l)) attn_mix(full_[full_idx(l)], x, full_idx(l), ph);
    else                         gdn_mix (gdn_ [gdn_idx (l)], x, gdn_idx (l), ph);
    const auto& [pn, mlp] = layer_tail(l);     // post_attn_norm + MlpW from the active struct
    mlp_tail(pn, mlp, x, ph);
  }
}

void prefill(span<const int> ids) {                   // T = ids.size()
  Tensor x = embed_gather(work_, embed_, upload(ids)); // [T,D], Q6 dequant gather
  run_layers(x, Phase::Prefill);
  Tensor xf = gemma_rmsnorm(work_, x, final_norm_);
  linear_into(io_.logits, xf.row(T-1), lm_head_);     // last position only
  argmax(io_.logits, io_.token);                      // first token -> device cursor (on-device)
  kv_.pos = T;  set_pos(io_.pos, T);  work_.reset();  // host + device position in lock-step
}

void decode_step() {                                   // identical kernel sequence every call
  Tensor x = embed_gather(work_, embed_, io_.token);  // reads device cursor
  run_layers(x, Phase::Decode);
  Tensor xf = gemma_rmsnorm(work_, x, final_norm_);
  linear_into(io_.logits, xf, lm_head_);
  argmax(io_.logits, io_.token);                      // next token -> same device cursor (on-device)
  advance_pos(io_.pos); kv_.advance();  work_.reset();// pos += 1 (device authoritative; kv appended @pos inside)
}
```

Notes carried from the architecture reference: GVA mapping (v-head `h` uses k/q head `h//3`),
conv is over the `[q;k;v]`=10240 channels only (not z/a/b), q/k are L2-normed then the kernel
scales q by `1/√128`, attention scale is `1/√256`, MRoPE reduces to plain partial RoPE for
text-only v1.

---

## 5. The precision/quant seam in action

The card always calls one verb: `linear(out_or_ws, x, const Weight&)`. Dispatch is a single
`switch (w.qtype)` in the L1 wrapper: `Q4G64`/`Q5G64`/`Q6G64`/`W8G128` → the matching monomorphized
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
The parity build (`FileTap`) is used only during M2 and never during capture.

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
