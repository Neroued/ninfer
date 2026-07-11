# The Weight Handle — the precision/quant seam — qwen3.6-ultraspeed

> Status: design (approved in brainstorm). Date: 2026-06-26.
> Scope: the **weight-handle contract** — one tagged `Weight` type that represents a loaded
> parameter at *any* precision (Q4/Q5/Q6/W8 **or** dense bf16/fp32), and the operator-API rule that
> lets one verb (`linear`, `embedding`, the `lm_head` GEMV) consume it without separate
> dense-vs-quant signatures. This sits at the **L0↔L1↔L2 seam**: it refines the L0 weight types
> (`include/qus/core/tensor.h`), fixes the L1 dispatch axis (see
> [`l1-kernel-layering.md`](l1-kernel-layering.md) §3/§5), and **replaces the `LinearW` tagged union**
> in the L2 card (see [`l2-model-card-design.md`](l2-model-card-design.md) §3.2/§5). Builds on
> [`design.md`](design.md) §4.3 (precision is the one flexible axis) and §9 (W4A16 + the `lm_head`
> bandwidth lever).

---

## 1. Problem, research & decision

### 1.1 The friction

The card binds two kinds of weight — dense (`Tensor`) and quantized (`QuantWeight`) — and v0 glued
them with a tagged union:

```cpp
struct LinearW { const QuantWeight* q = nullptr; const Tensor* d = nullptr; }; // exactly one set
```

Every operator that can take *either* precision must then accept `LinearW`, and its body must branch
`q` vs `d`. This:

1. **leaks a two-type dichotomy into operator signatures** — the very thing we want to keep out of
   the L1 API surface;
2. **mis-names the handle** when the weight is not a linear, e.g.
   `embedding(out, ids, const LinearW& table)` — an embedding table is not a "Linear";
3. **re-introduces a split the type system was already designed to erase**: `QType` already reserves
   `BF16_CTRL`/`FP32_CTRL`, `QuantLayout::Contiguous` already exists, and `QuantWeight` already
   carries `shape`/strides/`payload`. The dense case was *meant* to be just another tag.

### 1.2 How the references handle it

- **llama.cpp / ggml — one tagged type, data dispatch.** There is no separate "quantized" type:
  every weight and activation is one `ggml_tensor` carrying `enum ggml_type type`. Dense F16/F32 are
  the degenerate end of that enum (`blck_size == 1`, `is_quantized == false`). All operators take the
  uniform `ggml_tensor*`; quant behavior lives behind a traits table indexed by `type`
  (`to_float`, `vec_dot`, `blck_size`, …) and ops dispatch internally on `src0->type`
  (`ggml_get_rows`, `ggml_mul_mat`). Scales are inline in each per-type block struct; there is no
  per-tensor quant-metadata sidecar.
- **vLLM — polymorphic strategy object.** Each layer holds a `quant_method` that owns *both* weight
  creation and compute; `forward()` is a uniform `self.quant_method.apply(self, x, bias)` with no
  per-scheme branching. Powerful for orchestration, but it is **runtime virtual dispatch per op** —
  the opposite of this project's "no virtual dispatch" rule and hostile to the
  decode-as-one-captured-graph endgame. We keep only its good idea — resolve the method *once* — which
  we already do at bind time.

### 1.3 Decision — Option A: one tagged `Weight`, dense is a qtype

Adopt the llama.cpp model, scoped to weights: a single tagged `Weight` handle in which **dense is
just `qtype = BF16_CTRL`/`FP32_CTRL`**. Precision-polymorphic verbs take `const Weight&` and
`switch(qtype)`; the dense arm projects the handle to a `Tensor` view and reuses the existing dense
GEMV kernel. `LinearW` is deleted. `Tensor` is reserved for activations and always-dense params.

| # | Decision | Choice |
|---|---|---|
| 1 | Weight representation | One tagged `Weight` (rename of `QuantWeight`); dense = `BF16_CTRL`/`FP32_CTRL` qtype |
| 2 | Seam operator signature | `const Weight&`; dispatch `switch(qtype)` in the wrapper; no dense-vs-quant overloads |
| 3 | Dense arm | `as_dense(const Weight&) -> Tensor` projection; reuse the bf16/fp32 GEMV kernel — no new kernel |
| 4 | `Tensor` role | activations + always-dense params (`*_norm`/`conv1d`/`a_log`/`dt_bias`) only |
| 5 | `LinearW` | removed; seam bindings become `const Weight*` (or `Weight` by value) |
| 6 | vs vLLM strategy object | rejected — conflicts with no-vtable + CUDA-graph/megakernel (`design.md` §11) |
| 7 | Out of scope | the q5090 packer, the file format, `Tensor`, and all dense kernels — unchanged |

---

## 2. Mental model — `Tensor` vs `Weight`

Distinguish by **role**, not by precision:

| | `Tensor` | `Weight` |
|---|---|---|
| Represents | "what flows" — live runtime data | "what's baked in" — a loaded parameter |
| Mutability | read **and** written by kernels | read-only; never written |
| Lifetime | per-step (`work_` arena / `io_` buffers) | resident; loaded once |
| Fields | `{data, dtype, ne[4], nb[4]}` | payload + `qtype` + scales + group + layout + shape |
| Precision | bf16 / fp32 / i32 dense | Q4 / Q5 / Q6 / W8 **or** dense (`BF16_CTRL`/`FP32_CTRL`) |

**A `Weight` is a superset of a `Tensor`.** For the dense case it already holds a pointer + shape +
(contiguous) strides + dtype, so it projects to a `Tensor` view in one line (`as_dense`). That
projection is the bridge: **one uniform binding/signature (`Weight`), while the dense kernels keep
taking `Tensor`.**

**Boundary rule — where each type appears:**

- **Activations** (embeddings, hidden states, q/k/v, logits) → always `Tensor`.
- **Params on the precision-polymorphic seam** (`q/k/v/o_proj`, `gate/up/down`, `in_q…in_z`,
  `out_proj`, `embed`, `lm_head`) → `Weight` — *including* the bf16 ones (`in_a`/`in_b`), so the seam
  ops see exactly one type.
- **Always-dense params** consumed only by dedicated kernels (`*_norm`, `conv1d`, `a_log`,
  `dt_bias`) → stay `Tensor`. They never touch the seam, so there is no dichotomy to unify.

Analogy: `Weight` plays the role of `ggml_tensor`-as-a-parameter; `Tensor` plays the role of the F32
activation buffers that ops produce.

---

## 3. The `Weight` type

`Weight` is the current `QuantWeight` (`include/qus/core/tensor.h`), renamed to drop the
"quant-only" misnomer; **no field changes are required** — it already carries both worlds. There are
no external consumers yet, so this is a **clean rename with no transitional alias**: every
`QuantWeight` reference (the L0 store + its tests) is updated in one pass.

```cpp
struct Weight {                       // (= today's QuantWeight)
    const void* payload     = nullptr;
    QType       qtype       = QType::Q4G64_F16S;  // Q4/Q5/Q6/W8 OR BF16_CTRL/FP32_CTRL (dense)
    QuantLayout layout      = QuantLayout::W4A16KernelPackedV1; // or Contiguous for dense
    const void* qdata       = nullptr;            // dense: the bf16/fp32 data; quant: packed blocks
    const void* scales      = nullptr;            // dense: nullptr
    std::int32_t n = 0, k = 0, group = 0;         // GEMV dims; group==0 for dense
    std::int32_t shape[4] = {1,1,1,1};
    std::uint32_t ndim = 0;
    /* … module/source/scale-stride metadata unchanged … */
};
```

Population — same struct, the tag is the only structural difference:

```cpp
// Quantized seam weight (q_proj, Q4 tiled) — exactly what make_quant_descriptor already builds:
Weight{ .qtype = Q4G64_F16S, .layout = W4A16KernelPackedV1,
        .qdata = <packed nibbles>, .scales = <fp16 scales>, .n = 6144, .k = 5120, .group = 64 };

// Dense seam weight (in_a, [48,5120] bf16) — the BF16_CTRL case:
Weight{ .qtype = BF16_CTRL, .layout = Contiguous,
        .qdata = <bf16 data>, .scales = nullptr, .n = 48, .k = 5120, .group = 0 };
```

---

## 4. The bridge — `as_dense` / `weight_from_dense`

These two inline helpers (proposed home: `include/qus/core/weight.h`, or alongside `Tensor`) are the
entire glue. `as_dense` mirrors the loader's existing contiguous-view builder (`make_tensor_view` in
`src/core/weight_store.cpp`), so a dense `Weight` costs **no new kernel** — it feeds the same dense
GEMV that already takes a `Tensor`:

```cpp
// Dense Weight -> Tensor view. Valid only for BF16_CTRL / FP32_CTRL (layout == Contiguous).
inline Tensor as_dense(const Weight& w) {
    const DType dt = (w.qtype == QType::FP32_CTRL) ? DType::FP32 : DType::BF16;
    void* p        = const_cast<void*>(w.qdata);
    switch (w.ndim) {
        case 1:  return Tensor(p, dt, {w.shape[0]});
        case 2:  return Tensor(p, dt, {w.shape[0], w.shape[1]});
        case 3:  return Tensor(p, dt, {w.shape[0], w.shape[1], w.shape[2]});
        default: return Tensor(p, dt, {w.shape[0], w.shape[1], w.shape[2], w.shape[3]});
    }
}

// Dense contiguous seam param (e.g. in_a/in_b) -> Weight{BF16_CTRL}. Inverse of as_dense.
// Seam linears are 2-D weight matrices [n, k]; the existing convention is ne[0]=n, ne[1]=k.
inline Weight weight_from_dense(const Tensor& t) {
    Weight w{};
    w.qtype  = (t.dtype == DType::FP32) ? QType::FP32_CTRL : QType::BF16_CTRL;
    w.layout = QuantLayout::Contiguous;
    w.qdata  = t.data;  w.payload = t.data;  w.scales = nullptr;
    w.n = t.ne[0];  w.k = t.ne[1];  w.group = 0;
    w.ndim = 2;  w.shape[0] = t.ne[0];  w.shape[1] = t.ne[1];
    return w;
}
```

Note: `as_dense` reads from a `Weight`; the `n = shape[0]`, `k = shape[1]`, `ne[i] = shape[i]`
convention matches `make_quant_descriptor` and `make_tensor_view` (`src/core/weight_store.cpp`)
exactly, so the projection round-trips.

---

## 5. Operator-API contract (the precision seam)

### 5.1 The polymorphic surface is tiny

Only a handful of call sites are precision-polymorphic: **`linear`** (used for every projection
*and* the `lm_head` GEMV) and **`embedding`**. Everything else (`rmsnorm`, `conv1d`, `gdn_gates`,
attention, rope, residual, `argmax`, …) is dense-only and keeps `const Tensor&`. The "must consider
both `Tensor` and `Weight`" worry applies to **two** op signatures, not to all of L1.

### 5.2 Dispatch lives in the wrapper — `qtype` is the variant axis

Per [`l1-kernel-layering.md`](l1-kernel-layering.md) §3/§5, the **api** header declares the verb over
`const Weight&`; the **wrapper** validates and dispatches on `qtype` to a per-qtype **launcher
variant** — i.e. `qtype` *is* the `_<variant>` axis. Dense is one variant
(`linear_dense`) that wraps `as_dense`.

```cpp
// include/qus/kernels/linear.h  (api) — one verb, precision-agnostic
namespace qus::kernels {
void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
}
```

```cpp
// src/kernels/wrapper/linear.cpp  (validate + dispatch on qtype)
void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t s) {
    // validate out/x dtype + shape against w.n / w.k (host-side, fail fast) …
    switch (w.qtype) {
        case QType::Q4G64_F16S:  detail::linear_q4_launch(x, w, out, s); break;
        case QType::Q5G64_F16S:  detail::linear_q5_launch(x, w, out, s); break;
        case QType::Q6G64_F16S:  detail::linear_q6_launch(x, w, out, s); break;
        case QType::W8G32_F16S: detail::linear_w8_launch(x, w, out, s); break;
        case QType::BF16_CTRL:
        case QType::FP32_CTRL: {
            const Tensor wd = as_dense(w);
            detail::linear_dense_launch(x, wd, out, s);                  // reuse the dense GEMV
        } break;
        default: throw std::invalid_argument("linear: unsupported weight qtype");
    }
}
```

```cpp
// include/qus/kernels/embedding.h  (api) — table is a Weight; out/ids are Tensor
void embedding(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream);
// wrapper: Q6G64_F16S -> dequant-gather one row per id; BF16_CTRL -> plain row copy via as_dense(table)
```

This also **reconciles the `w4a16_gemv` name** from `design.md` §6: that is simply the `Q4G64`
*variant* of `linear`. The L2-facing verb is precision-agnostic (`linear`); the per-precision kernels
are its variants, selected by `qtype`.

---

## 6. Binding changes (L2)

`LinearW` and its `q`/`d` arms are deleted. `MlpW` and the per-layer structs hold `const Weight*` on
the seam and `const Tensor*` for dense-only params:

```cpp
struct MlpW { const Weight *gate, *up, *down; };

struct FullLayerW {                 // 16
  const Tensor* input_norm;                                       // dense-only -> Tensor
  const Weight *q_proj, *gate_proj, *k_proj, *v_proj, *o_proj;    // seam -> Weight
  const Tensor *q_norm, *k_norm;                                  // dense-only -> Tensor
  const Tensor* post_attn_norm; MlpW mlp;
};

struct GdnLayerW {                  // 48
  const Tensor* input_norm;
  const Weight *in_q, *in_k, *in_v, *in_z, *in_a, *in_b;          // seam -> Weight (in_a/in_b = BF16_CTRL)
  const Tensor* conv1d;                                           // dense-only -> Tensor
  const Tensor *a_log, *dt_bias;                                  // dense-only (fp32) -> Tensor
  const Tensor* gdn_norm;
  const Weight* out_proj;                                         // seam -> Weight
  const Tensor* post_attn_norm; MlpW mlp;
};
```

`bind()` has three cases:

```cpp
// (1) quant seam param: direct — QuantRecord already holds a Weight
w.q_proj = store.qweight(TextCore, AttnQ, layer);                 // const Weight*

// (2) dense seam param (in_a/in_b): store holds it as a contiguous Tensor record;
//     synthesize a stable Weight{BF16_CTRL} owned by the card. seam_dense_ is reserve()d to its
//     final size (96) BEFORE binding, so emplace_back never reallocates and dangles a stored pointer.
w.in_a = &seam_dense_.emplace_back(weight_from_dense(*store.tensor(TextCore, GdnInProjA, layer)));

// (3) dense-only param: unchanged
w.input_norm = store.tensor(TextCore, InputLayernorm, layer);     // const Tensor*
```

**Ownership.** The two synthesized dense seam weights per GDN layer (`in_a`, `in_b`) need stable
addresses; the card owns a small `std::vector<Weight> seam_dense_` for them, **`reserve(96)`d before
binding** so `emplace_back` cannot reallocate and invalidate an already-stored pointer. Simpler
still: make the seam fields `Weight` **by value** (a ~100-byte POD; 64 layers × ~10 ≈ 64 KB total,
and more cache-local on the hot path) and the ownership question disappears entirely. Either is fine;
pointers keep a single source of truth for the quant ones, by-value removes the side-store.

---

## 7. Graph-readiness (unchanged invariants)

Each binding's `qtype` is fixed at load, so the wrapper `switch(qtype)` always selects the **same**
launcher variant for a given call site. The card resolves the tag **once** before any `cudaGraph`
capture; the per-step kernel sequence is therefore identical and fully specialized — no vtable, no
runtime op-graph. This is exactly `l2-model-card-design.md` Decisions #3 (precision is data) and #6
(no virtual dispatch), and structurally identical to ggml dispatching on `src0->type`.

---

## 8. Scope & migration

**Changes:**

- `include/qus/core/tensor.h` — rename `QuantWeight` → `Weight` (clean rename, no alias); no field
  changes.
- `include/qus/core/weight.h` (new, small) — `as_dense` / `weight_from_dense` inline helpers.
- `include/qus/core/weight_store.h` / `src/core/weight_store.cpp` — `qweight(...)` accessor returns
  `const Weight*` (mechanical rename; `QuantRecord`→holds `Weight`).
- L1 — `linear` and `embedding` api over `const Weight&`; wrappers dispatch on `qtype`; per-qtype
  launcher/kernel variants, with `linear_dense` wrapping `as_dense`.
- L2 `model.h` / `qwen3_6_27b.cpp` — delete `LinearW`; seam fields → `const Weight*` (or `Weight`);
  `bind()` synthesizes `in_a`/`in_b` via `weight_from_dense`.

**Unchanged (explicitly):** the q5090 packer and on-disk file format (still store contiguous dense +
quant blocks exactly as today — see `q5090_packed_file_format_v1.md`), the `Tensor` type, and **all
dense kernels** (they keep taking `Tensor`).

**Optional clean-up (deferred, not required for v1).** Let `WeightStore` store *every* parameter as a
`Weight` record (the contiguous branch already maps cleanly through `make_quant_descriptor`, which
sets `scales = nullptr`). Then the bind seam is uniform `store.weight(...)`, `weight_from_dense`
disappears, and dense-only params are projected once at bind via `as_dense` into a card-owned
`Tensor` side-store. This trades a larger L0 change (collapsing `TensorRecord`/`QuantRecord` into one
`WeightRecord`, plus test updates) for a fully uniform store API; defer until after correctness (M2).

---

## 9. Sources

- This session's source research: llama.cpp `ggml_tensor` / `ggml_type` / `type_traits[]` dispatch
  (`ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml-cpu`/`ggml-cuda` `get_rows`/`mul_mat`); vLLM
  `QuantizeMethodBase` / `LinearMethodBase` strategy
  (`vllm/model_executor/layers/quantization/base_config.py`, `.../linear.py`).
- `design.md` §4.3 (precision is the one flexible axis), §6 (operator list), §9 (W4A16, `lm_head`
  bandwidth lever), §11 (optimization ladder).
- `l1-kernel-layering.md` §3 (variant axis), §5 (wrapper validation + dispatch).
- `l2-model-card-design.md` §3.2–§3.4, §5 (the `LinearW` seam this supersedes).
- `include/qus/core/tensor.h` (`QuantWeight`/`QType`/`QuantLayout`), `src/core/weight_store.cpp`
  (`make_quant_descriptor`, `make_tensor_view`, `dtype_for_contiguous`).
