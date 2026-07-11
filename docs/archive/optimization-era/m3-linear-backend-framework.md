# M3 Linear Backend Framework Design — q5090 GEMV/GEMM

> Status: **framework design**.
> Date: 2026-06-28.
> Revision: incorporates design-review decisions — dispatch via `constexpr` metadata table + `switch`
> (no `LaunchFn` table), `Generic` as the bootstrap path and permanent fallback, codec group/tile
> granularity, regime-as-function, registry 1:1 live semantics, mandatory Tier C1 K-split, and the
> Q5/Q6 unpack risk.
> Scope: M3 linear backend architecture for q5090 weight-only linear layers.
> This document defines framework boundaries and policy direction. It is **not** a Codex task list,
> does **not** define an implementation schedule, and does **not** freeze final kernel tile sizes.
>
> Related context: `docs/design.md`, `docs/l1-operator-catalog.md`,
> `docs/l1-kernel-layering.md`, `docs/weight-handle-design.md`,
> `docs/q5090_packed_file_format_v1.md`,
> `docs/qwen3_6_27b_q5090_final_quant_format_v1.md`, and the current
> correctness-baseline `linear` implementation under `src/kernels/`.

---

## 1. Purpose

`linear` is the main performance seam of qwen3.6-ultraspeed. It covers almost every large projection
in the Qwen3.6-27B text decoder: attention projections, Gated DeltaNet projections, MLP projections,
dense control projections, and `lm_head`.

The current correctness implementation proves the public API, shape convention, and q5090 payload
semantics. M3 needs a backend framework that can evolve this baseline into high-performance CUDA
kernels while preserving the project boundary:

```text
fixed model + fixed GPU + fixed q5090 ABI + static execution graph + hand-written kernels
```

The public L1 API remains unchanged:

```cpp
namespace qus::kernels {
void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
}
```

Internally, `linear` becomes plan-driven. The framework has two kernel families:

- **`GemvBackend`** for `T == 1`.
- **`GemmBackend`** for `T > 1`.

These names describe the mathematical/kernel regime, not high-level runtime phases. Decode normally
uses GEMV. Prefill normally uses GEMM. The framework should use that distinction without turning
`linear` into a general-purpose quantized GEMM runtime.

---

## 2. Design principle: keep the core dispatch small

The framework must be extensible, but extensibility must not become a multiplication of runtime axes.
M3 should only promote a concept into the core plan key if it changes one of the following in v1:

1. q5090 payload interpretation;
2. required CUDA kernel family or launch policy;
3. public `linear` output semantics.

Everything else belongs in one of these lower-cost places:

- plan metadata;
- benchmark tags;
- debug/parity paths;
- kernel-internal policy details;
- post-M3 extension hooks;
- a separate L2/private operator.

This gives the M3 framework a deliberate complexity budget.

| Concept | M3 core dispatch axis? | Reason |
|---|---:|---|
| qtype + layout | Yes | Defines how the weight payload is decoded. |
| shape family | Yes | Defines tuned launch-policy family and specialization tier. |
| T regime | Yes | Separates `T == 1`, small `T > 1`, and larger GEMM. |
| backend kind | Derived | `T1` maps to GEMV; `SmallT`/`LargeT` map to GEMM. |
| source kind / layer role | No | Useful for attribution, but usually should not select kernels. |
| epilogue kind | No in M3 | `linear()` has one store-output contract in v1. |
| math mode | No in M3 | Distinct math/dequant/MMA modes require distinct policy IDs and benchmark identities, but not a user-visible runtime axis. |
| workspace policy | No in M3 | M3 performance `linear()` plans should be externally workspace-free. |
| derived layout | No in initial M3 | Requires explicit design revision and q5090 ABI/manifest update. |
| W8G32 / MTP / vision | No in TEXT-core M3 | Supported at the format seam, not a TEXT-core performance obligation. |

The goal is not to model every future possibility. The goal is to define the smallest stable
framework that can host the M3 performance work.

---

## 3. Project constraints relevant to `linear`

This design is specialized to the v1 target:

- single model: Qwen3.6-27B text decoder;
- single user, single sequence, batch = 1;
- single RTX 5090, target `sm_120`;
- static C++ model card, no dynamic compute graph;
- CUDA-only kernels, no runtime dependency on cuBLAS, CUTLASS, CUB, Thrust, or a general GEMM
  library;
- BF16 activations;
- TEXT core large weights use q5090 weight-only quantization;
- runtime loads one fixed q5090 file and consumes device-resident weights directly;
- current correctness baseline remains available as reference/fallback infrastructure.

The relevant TEXT-core shapes are finite and known. The framework should exploit that fact without
creating one kernel per layer or per tensor.

---

## 4. Non-goals

This document does **not**:

- define implementation tasks;
- choose final tile sizes, warp layouts, launch bounds, stage counts, or persistent policies;
- replace the current benchmark / I/O / memory observability gate;
- change the public L1 `linear` API;
- change the q5090 canonical file ABI;
- require runtime hidden repacking of weights;
- design MTP or vision linear kernels in detail;
- define sampler, top-k, or argmax APIs;
- promise one universal kernel for all qtypes, shapes, and `T` values;
- assert that Tensor Core is optimal for every `T > 1` case.

The document fixes framework boundaries. Kernel tuning remains profiling-driven.

---

## 5. Core decisions

| # | Decision | M3 choice |
|---|---|---|
| 1 | Public API | Keep `linear(x, w, out, stream)` unchanged. |
| 2 | Output semantics | M3 `linear()` always writes `out[N,T]` as BF16. |
| 3 | Internal dispatch | Wrapper validates and selects a `LinearPlan` from a static registry. |
| 4 | Dispatch axes | Use only `LinearFormat`, `ShapeFamily`, and `LinearRegime`. |
| 5 | GEMV/GEMM split | `T == 1` is owned by `GemvBackend`; `T > 1` is owned by `GemmBackend`. |
| 6 | Small-T handling | `GemmBackend` may use GEMV-like or weight-streaming scheduling for small `T`. |
| 7 | Large-T GEMM direction | For Tier A/B quantized TEXT-core shapes in `LargeT`, staged Tensor Core GEMM is the primary performance target. |
| 8 | GEMV direction | Bandwidth-first, packed-weight streaming, no default full-tile dequant-to-SMEM. |
| 9 | q5090 seam | Use a narrow `WeightCodec` / `FormatTraits` seam for address, unpack, and dequant primitives. |
| 10 | Shape specialization | Use named shape families and selective hot-shape template instantiation. |
| 11 | q5090 layout | Keep canonical `TILE_N64_K64` for TEXT-core M3. No hidden runtime repack. |
| 12 | Workspace | M3 performance `linear()` plans are externally `NoWorkspace`; per-block registers/shared memory are normal kernel resources. |
| 13 | `lm_head` | Dedicated GEMV store-output plan in M3; argmax/top-k are future separate operators. |
| 14 | Reference | Keep reference backend permanently for correctness and unsupported fallback. |

---

## 6. Public API and wrapper contract

The L1 public API remains:

```cpp
void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
```

Shape convention remains:

```text
x   : [K, T], BF16
w   : [N, K], Weight
out : [N, T], BF16
ne[0] is the fastest-varying dimension
```

M3 `linear()` has exactly one output behavior:

```text
compute W @ x and store BF16 output to out[N,T]
```

The wrapper owns:

1. dtype validation;
2. shape validation;
3. contiguity validation;
4. q5090 qtype/layout metadata validation;
5. empty-input short-circuit;
6. construction of a minimal `LinearPlanKey`;
7. dispatch to the selected plan launcher.

The wrapper must not include device kernel headers. It calls private launcher declarations under
`src/kernels/launcher/`, preserving the existing L1 layering rules.

### 6.1 Validation and fast path

The correctness/debug path may validate every call.

For performance runs, the model card or engine may pre-resolve a `LinearPlanHandle` per callsite and
`T` regime. This avoids repeated heavy host-side dispatch in decode and improves CUDA Graph
friendliness later. Prebinding is an optimization of host overhead; it does not change the public API
or the plan key.

Plan resolution and launch must be:

```text
allocation-free
synchronization-free
deterministic
graph-capture friendly
```

M3 performance launchers must not perform hidden allocation, hidden repacking, or device-wide
synchronization.

---

## 7. Architecture overview

Desired call chain:

```text
L2 model card / Engine
  └── kernels::linear(x, w, out, stream)
        └── wrapper/linear.cpp
              ├── validate Tensor / Weight / shape / layout
              ├── classify LinearFormat
              ├── classify ShapeFamily
              ├── classify LinearRegime
              ├── lookup LinearPlan
              └── call plan launcher
                    ├── GemvBackend<ShapeTag, Codec, Policy>
                    ├── GemmBackend<ShapeTag, Codec, Policy>
                    └── ReferenceLinearBackend
```

The public operator remains generic. Specialization starts only after validation and plan selection.
This keeps qtype/layout complexity behind L1 and keeps L2 model code clean.

---

## 8. `LinearPlanKey` and `LinearPlanRegistry`

The plan registry is the central control point of the framework. It replaces ad hoc growth such as:

```text
switch(qtype) + switch(layout) + if (T == 1) + shape-specific special cases
```

with an explicit mapping:

```text
LinearPlanKey -> LinearPlan
```

### 8.1 Minimal M3 plan key

The M3 key has only three dispatch axes:

```cpp
enum class LinearFormat {
    Q4G64_N64K64,
    Q5G64_N64K64,
    Q6G64_N64K64,
    DenseBF16,
    DenseFP32,
    GenericUnsupported,
};

enum class ShapeFamily {
    DenseCtrl48x5120,
    AttnKV1024x5120,
    GdnQK2048x5120,
    Proj6144x5120,
    Out5120x6144,
    MlpGateUp17408x5120,
    MlpDown5120x17408,
    LmHead248320x5120,
    Generic,
};

enum class LinearRegime {
    T1,
    SmallT,
    LargeT,
};

struct LinearPlanKey {
    LinearFormat format;
    ShapeFamily shape;
    LinearRegime regime;
};
```

Important choices:

- `LinearFormat` combines qtype and layout because performance kernels depend on the pair, not on
  qtype alone.
- `N` and `K` are normally represented by `ShapeFamily`; only `Generic` paths need runtime dimensions.
- `source_kind` / layer role is not part of dispatch in M3.
- epilogue kind is not part of dispatch in M3.
- math mode is not part of dispatch in M3.
- workspace policy is not part of dispatch in M3.

If a future case genuinely needs different kernels for the same shape/format/regime, the first
preferred solution is to split the `ShapeFamily` or add a post-M3 operator seam, not to add global
runtime axes prematurely.

### 8.1.1 Generic is the bootstrap path and the permanent fallback

`ShapeFamily::Generic` / `LinearFormat::GenericUnsupported` and the reference backend are not
speculative runtime flexibility — they are the **first thing implemented**. M3 starts with every
`LinearPlanKey` resolving to a correct, runtime-dimension-driven plan backed by the reference
backend (§16), so the whole registry, wrapper, and benchmark contract run end-to-end before a single
tuned kernel exists. Performance work then **replaces entries incrementally**: one
`(format, shape, regime)` at a time is repointed from the generic/reference plan to a specialized
plan, each flip gated by parity against the generic result plus a §18 benchmark line.

`Generic` therefore stays permanently as (a) the bootstrap path, (b) the migration fallback for
not-yet-tuned combinations, and (c) the route for any shape outside the named set (e.g. bench-only
shapes). A miss must resolve to the generic/reference plan **explicitly and be logged as such**
(§16); it must never silently route to an unrelated tuned kernel.

### 8.2 Conceptual plan

A plan is a **metadata descriptor** for one resolved key. It records the chosen backend, policy
identity, and benchmark/audit fields — it does **not** carry a runtime function pointer:

```cpp
enum class LinearBackendKind {
    Gemv,
    Gemm,
    Reference,
};

struct LinearPlan {
    LinearBackendKind backend;
    LinearPolicyId policy;   // distinct per dequant / accumulator / MMA mode
    const char* plan_id;     // stable identity for logs and benchmarks
    bool uses_tensor_cores;  // derived metadata, for reports only
};
```

Dispatch is an explicit `switch` over `LinearPlanKey` into the selected templated launcher; the
registry is a `constexpr` metadata table keyed by the same `LinearPlanKey`. The table is the
auditable source of truth for `plan_id` / policy / benchmark identity; the `switch` is the actual
launch. We deliberately avoid a `LaunchFn` function-pointer table: for the finite M3 plan set a
`switch` keeps the launch directly inlinable and easy to read, while the `constexpr` table still
gives the auditable, benchmarkable identity a bare `switch` could not. (`is_reference` is derivable
from `backend == Reference`, so it is not a stored field.)

The live registry is **1 `LinearPlanKey` → 1 `LinearPlan`**. Different dequantization, accumulator,
or MMA-consumption modes are distinct `LinearPolicyId` / `plan_id` values; only one is bound live per
key, but the others remain available as **candidate policies** the benchmark harness can instantiate
for the same key. Policy / math mode is never a user-visible runtime dispatch axis, but it must be
visible to benchmark and parity reports.

### 8.3 T regime classification

Top-level ownership:

```text
T == 1  -> LinearRegime::T1     -> GemvBackend
T > 1   -> LinearRegime::SmallT -> GemmBackend
T > 1   -> LinearRegime::LargeT -> GemmBackend
```

`SmallT` thresholds are tuning data. They may be shape-family-specific and format-specific. A single
global threshold such as `T <= 16` should not be assumed before profiling.

Regime selection is therefore a function `classify_regime(format, shape, T)` backed by a tunable
threshold table, not a constant bucket. This is the only place `T` enters dispatch. As a
consequence, the `LinearPlanHandle` prebinding in §6.1 is only fully static for `T1` (decode, where
`T` is always 1); for `T > 1` the regime depends on the runtime prompt length, so prefill resolves
its regime per call. That is acceptable: decode is the latency-critical, graph-captured path, and it
is exactly the case prebinding makes static.

`SmallT` is still owned by `GemmBackend`, but its internal schedule may look like bundled GEMV,
weight-streaming multi-column matvec, or lightweight GEMM. This keeps backend ownership clear while
avoiding a false commitment that every `T > 1` call should use a heavy Tensor Core pipeline.

`LargeT` names the kernel regime, not a public runtime phase. It typically corresponds to prefill
workloads, but the plan key intentionally avoids embedding the word `prefill` in the regime enum.

### 8.4 Plan IDs

Every plan, including reference fallback, must have a stable plan ID for logs and benchmarks.
Example style:

```text
linear.gemv.q4.n64k64.mlp_gate_up_17408x5120.t1.store_bf16.v1
linear.gemm.q5.n64k64.mlp_down_5120x17408.small_t.bundle_v1
linear.gemm.q4.n64k64.proj_6144x5120.large_t.tc_stage_v1
linear.ref.q6.n64k64.lm_head_248320x5120.t1.store_bf16.v1
```

The ID may include implementation details such as `tc_stage`, `simt_ref`, `bundle`, accumulator
style, or dequant staging mode, but these are metadata and identity strings, not additional runtime
dispatch dimensions.

---

## 9. Shape-family specialization policy

Dimension specialization does **not** mean one hand-written kernel per matrix. It means:

```text
finite Qwen3.6 shape set
  -> named ShapeFamily
  -> static plan registry
  -> selective CUDA template instantiation for hot families
```

Layer index is not a specialization axis.

### 9.1 Shape families

```cpp
enum class ShapeFamily {
    DenseCtrl48x5120,
    AttnKV1024x5120,
    GdnQK2048x5120,
    Proj6144x5120,
    Out5120x6144,
    MlpGateUp17408x5120,
    MlpDown5120x17408,
    LmHead248320x5120,
    Generic,
};
```

`MlpGateUp17408x5120` deliberately covers both MLP gate and MLP up projections. They share shape and
should normally share a plan unless measurement proves otherwise.

### 9.2 Specialization tiers

| Tier | Shapes | M3 policy |
|---|---|---|
| A | `LmHead248320x5120`, `MlpGateUp17408x5120`, `MlpDown5120x17408` | Dedicated plan families, explicit benchmark lines, likely explicit template instantiations. |
| B | `Proj6144x5120`, `Out5120x6144` | Explicit plans using shared tuned kernel families where possible. |
| C1 | `AttnKV1024x5120`, `GdnQK2048x5120` | Shared tuned q5090 policies before any M3 performance claim. Smaller per call than Tier A/B, but recurrent across layers. |
| C2 | `DenseCtrl48x5120` | Separate dense-control policy. May remain simple in M3 or later fuse into GDN control logic. |

Tier C1 is not cold enough to justify permanent reference performance paths. `[1024,5120]` and
`[2048,5120]` are smaller per call, but they recur across layers. They should have at least one
shared tuned policy before any M3 performance claim.

Tier C2 is different in nature. Dense control projections are small and dense, not q5090 large
projections. They should not inherit q5090 Tensor Core staging assumptions by default.

### 9.3 What should be compile-time specialized

Good candidates for compile-time specialization:

- `LinearFormat` / codec;
- group size and q5090 tile layout constants;
- hot shape tags;
- major K-loop structure;
- backend policy family;
- Tensor Core staging policy for `LargeT` Tier A/B shapes.

Usually not worth compile-time specialization in M3:

- exact layer index;
- source role when shape/format/regime are identical;
- exact `T` value inside a regime bucket;
- benchmark-only math-mode labels.

---

## 10. `WeightCodec` / `FormatTraits`

`WeightCodec` is the qtype/layout seam. It owns interpretation of q5090 storage and exposes small
primitives to backend policies.

A codec is selected by `LinearFormat`, which combines qtype and layout.

Initial format families:

| LinearFormat | qtype | layout | M3 role |
|---|---|---|---|
| `Q4G64_N64K64` | `Q4G64_F16S` | `TILE_N64_K64` | TEXT-core Q4 large linears. |
| `Q5G64_N64K64` | `Q5G64_F16S` | `TILE_N64_K64` | TEXT-core Q5 sensitive linears. |
| `Q6G64_N64K64` | `Q6G64_F16S` | `TILE_N64_K64` | `lm_head` and other Q6 linears. |
| `DenseBF16` | `BF16_CTRL` | `CONTIGUOUS` | Dense BF16 control weights. |
| `DenseFP32` | `FP32_CTRL` | `CONTIGUOUS` | Dense FP32 control weights. |

W8G32 is the sole W8 format used by MTP and the vision merger; its row-split backend enters through
the same format/shape dispatch seam. These are not TEXT-core M3 performance obligations.

### 10.1 Codec responsibilities

A codec should provide compile-time traits and device helpers equivalent to:

```cpp
template <typename Codec>
struct WeightCodecTraits {
    static constexpr int kBits;
    static constexpr int kGroupK;
    static constexpr int kTileN;
    static constexpr int kTileK;
    static constexpr int kBytesPerRowPerGroup;
    static constexpr int kBytesPerTile;
};
```

Device-side codec logic owns:

- q5090 tile addressing;
- scale addressing;
- packed-byte addressing;
- vectorized packed loads;
- signed low-bit unpack primitives;
- FP16 scale application primitives;
- dense BF16/FP32 load primitives for control weights.

These primitives operate at **group/tile granularity, not per code**. The canonical codec call loads
one 64-code K-group (a `kGroupK` run for one row) with vectorized packed loads and unpacks it into a
register array of 64 dequantized values with the FP16 scale already applied. Per-code helpers such as
the current baseline `load_q4_weight` in `kernel/linear_q4.cuh` are correctness scaffolding only:
they cannot vectorize the packed load and must not define the seam. Group/tile granularity is the
balance point — it is vectorizable, yet it still hands every scheduling/staging decision to the
backend, so the codec never owns SMEM, CTA shape, K split, or reduction.

### 10.2 Codec does not own scheduling

The codec does **not** decide:

- GEMV vs GEMM ownership;
- CTA shape;
- warp mapping;
- K split;
- persistent scheduling;
- Tensor Core policy;
- register vs shared-memory staging strategy;
- MMA fragment layout.

Those decisions belong to `GemvBackend` or `GemmBackend` policies.

The important boundary is:

```text
WeightCodec / FormatTraits:
  address + load + unpack + dequant primitives

BackendPolicy:
  scheduling + staging + reduction + MMA consumption
```

This prevents the codec from becoming a god class while still avoiding duplicated qtype logic across
GEMV and GEMM kernels.

### 10.3 Dense control path

Dense BF16/FP32 control linears should use a separate dense-control policy family. They should not
inherit assumptions from q5090 Tensor Core GEMM policies.

`FP32_CTRL` in particular may be better served by a small SIMT dense kernel or by later fusion with
GDN control logic. M3 only needs a correct and measurable dense-control path, not a generalized dense
GEMM subsystem.

---

## 11. `GemvBackend` — `T == 1`

`GemvBackend` handles:

```text
out[N,1] = W[N,K] @ x[K,1]
```

The primary performance model is weight streaming. For quantized TEXT-core linears, each generated
token reads large portions of the resident weight set once. The backend should optimize for effective
weight bandwidth, packed-load efficiency, unpack throughput, and low-overhead reductions.

### 11.1 GEMV design principles

- Optimize for global/L2 bandwidth and packed-weight load efficiency.
- Keep `x[K]` reuse cheap; `x` is small relative to weight traffic.
- Avoid full-tile dequantization to shared memory by default; at `T == 1`, decoded weight reuse is
  often too small to justify the extra staging traffic and SMEM footprint.
- Keep qtype unpack and scale application close to consumption.
- Prefer vectorized packed loads and warp/CTA reductions over scalar per-weight loops.
- Use split-K only if it pays for its reduction cost without external workspace.
- Treat `lm_head` as a dedicated shape family.

### 11.2 GEMV policy space

The framework should be able to host policies such as:

- CTA per N stripe;
- CTA per `N64` q5090 tile stripe;
- warp-per-row or warp-group-per-row variants;
- vectorized code-byte unpack;
- K partitioning inside one CTA where useful;
- persistent CTA variants for selected shapes if they remain externally workspace-free;
- `lm_head`-specific row-striping and in-kernel reduction strategies for the store-output contract.

These are kernel policy choices, not public API choices. The exact policies must be selected by
measurement.

One constraint is not free to defer. The small-`N` Tier C1 shapes (`AttnKV1024x5120`,
`GdnQK2048x5120`) do not produce enough row-parallel work to fill the target GPU (`N=1024` yields on
the order of 10^3 warps against the ~170 SMs of the RTX 5090), and the workspace-free rule (§17)
rules out the usual multi-CTA split-K with a separate reduction kernel. For these shapes,
**CTA-internal K-splitting with an in-SMEM reduction is effectively mandatory, not optional**.
"Achieves acceptable occupancy on small-`N` shapes with no external workspace" is a
definition-of-done item for Tier C1, not a tuning afterthought.

### 11.3 `lm_head` under M3 `linear()`

`LmHead248320x5120` is a first-class GEMV special plan.

For M3 `linear()`:

```text
lm_head T == 1 -> dedicated GEMV StoreOutputBF16 plan
```

`linear()` still writes logits to `out[248320,1]`. It does not return a token and does not silently
skip the full output write.

For ordinary prefill, L2 should not default to full-vocabulary logits for every prompt position. If
only the next-token logits are needed, L2 should pass the last hidden-state column as `[5120,1]` and
use the `T1` `lm_head` GEMV plan.

A full-logits `lm_head` GEMM for `T > 1` may exist for parity, logprobs, or debug workflows, but it
should not be the default ordinary-inference route.

The M3 `lm_head` linear plan optimizes the contractual `StoreOutputBF16` path. It is not necessarily
the final greedy-decode output path once a future `lm_head_select` seam exists.

---

## 12. `GemmBackend` — `T > 1`

`GemmBackend` handles:

```text
out[N,T] = W[N,K] @ x[K,T]
```

For `LargeT` on Tier A/B quantized TEXT-core shapes, staged Tensor Core GEMM is the primary
performance target. This is stronger than a vague future option: large prefill GEMM should be
structured around Tensor Core mainloops unless measurement proves that a specific shape/format/regime
cannot benefit.

This does not imply that every `T > 1` path must use Tensor Cores. `SmallT`, dense-control weights,
reference/fallback paths, and debug prototypes remain measurement-driven.

### 12.1 Meaning of Tensor Core target for q5090

Tensor Core use does **not** require q5090 Q4/Q5/Q6 codes to map directly to native low-bit MMA.
The expected large-T path is conceptually:

```text
packed q5090 weight tile
  -> codec load/unpack/scale primitives
  -> backend-specific staging into MMA-consumable representation
  -> Tensor Core MMA with BF16 activation fragments
  -> accumulator conversion
  -> store BF16 out[N,T]
```

Whether dequantized weights are staged as BF16, FP16, or another MMA-consumable representation is a
kernel policy and benchmarking question. It should be recorded in `plan_id` and benchmark metadata,
but it is not an M3 runtime dispatch axis.

**Primary risk for Q5/Q6.** Unlike Q4 nibble unpack, Q5 (5-bit) and Q6 (6-bit) codes are LSB-first
bit-packed across byte boundaries (`q5090_packed_file_format_v1.md` §7.1). The unpack/dequant ALU
cost can dominate both DRAM and MMA, so the Tensor Core mainloop may be unpack-bound before it is
MMA-bound. The "Tensor Core is the primary `LargeT` target" direction is therefore conditional for
Q5/Q6: an unpack-throughput microbench must show that decode does not starve the MMA pipeline before
a TC mainloop is committed for those formats. Q4 does not carry this risk to the same degree.

### 12.2 GEMM regime policy

`GemmBackend` should support two broad regimes:

```text
SmallT:
  T > 1 but too small to assume a heavy Tensor Core pipeline wins.
  Scheduling may resemble multi-column GEMV or lightweight weight-streaming GEMM.

LargeT:
  T large enough for activation/weight-tile reuse and Tensor Core staging to be the primary target
  for Tier A/B quantized TEXT-core shapes.
```

The threshold between `SmallT` and `LargeT` is measurement data, not a framework constant. It may
vary by `ShapeFamily` and `LinearFormat`.

### 12.3 GEMM design principles

- Keep `T > 1` ownership in `GemmBackend` for clean framework structure.
- Do not assume all `T > 1` cases should use the same mainloop.
- For `LargeT` Tier A/B quantized shapes, target Tensor Core MMA after q5090 dequantization.
- Overlap unpack/dequant with MMA work where possible.
- Treat canonical `TILE_N64_K64` as the initial M3 input ABI.
- Use shared memory where it improves reuse or enables efficient MMA staging.
- Preserve reference fallback for unsupported formats and parity tests.
- Keep all M3 performance plans externally workspace-free.

### 12.4 SIMT GEMM role

SIMT GEMM remains useful for:

- reference correctness;
- early prototypes;
- small dense-control paths;
- unsupported fallback;
- measurement comparisons.

It should not be presented as the intended long-term `LargeT` path for Tier A/B quantized TEXT-core
GEMM unless profiling proves that q5090 staging overhead defeats Tensor Core policies for a specific
shape/regime.

---

## 13. Output semantics and future epilogues

M3 `linear()` has one output behavior:

```text
StoreOutputBF16
```

There is no M3 `LinearEpilogueKind` dispatch axis. In particular, these are **not** part of M3
`linear()`:

```text
FusedArgmax
FusedTopK
Sampler epilogue
Optional token output instead of out[N,T]
```

This avoids violating the public contract. A caller that invokes `linear(x,w,out)` is entitled to a
fully written `out[N,T]` tensor.

### 13.1 Future `lm_head_select` seam

Future greedy or sampler-oriented output reduction should be a separate operator or L2-private seam,
for example:

```text
lm_head_select(x_last, lm_head_weight, sampler_state, token/topk output, optional logits output)
```

That future operator may use fused argmax/top-k, multi-kernel reduction, or workspace. It should not
be hidden under `linear()` unless the public API changes to express that behavior.

---

## 14. q5090 layout policy

### 14.1 Canonical layout remains stable in M3

For v1 TEXT core, q5090 `TILE_N64_K64` remains the canonical storage ABI for Q4/Q5/Q6 large linear
weights.

Runtime must not silently repack q5090 weights into a second device-resident performance layout.
Hidden runtime repack violates the project boundary:

```text
offline quantize + relayout/pack -> one fixed file -> runtime loads and runs
```

This boundary matters for reproducibility, memory accounting, startup behavior, and benchmark
interpretability.

### 14.2 Derived layouts are outside the initial M3 core registry

A derived layout is not part of the initial M3 core registry. Introducing one requires an explicit
design revision and q5090 ABI/manifest update.

Approved evolution path:

```text
converter emits explicit derived payload/layout
q5090 metadata records its existence
WeightStore exposes it as a distinct layout/view
PlanRegistry selects it when available and explicitly enabled
canonical TILE_N64_K64 fallback remains available
```

A derived layout may reorder or duplicate q5090-equivalent codes/scales. It must not silently change
the dequantized mathematical value. If the converter changes signed integer q5090 quantization into
a different numerical format such as FP4/FP6/MXFP/NVFP, that is a new qtype or quantization scheme,
not merely a layout variant.

Derived layouts require separate memory-budget review. On a 32GB target, duplicating large weights
for performance cannot be a default assumption.

---

## 15. Shape-family plan table

Initial TEXT-core shape families:

| Shape family | Shape `[N,K]` | Main uses | Tier | GEMV route | GEMM route |
|---|---:|---|---|---|---|
| `DenseCtrl48x5120` | `[48,5120]` | GDN `in_a`, `in_b` | C2 | Dense-control GEMV | Dense-control GEMM/SIMT as needed |
| `AttnKV1024x5120` | `[1024,5120]` | full-attn K/V | C1 | Shared tuned q5090 GEMV | Shared SmallT/LargeT GEMM |
| `GdnQK2048x5120` | `[2048,5120]` | GDN Q/K | C1 | Shared tuned q5090 GEMV | Shared SmallT/LargeT GEMM |
| `Proj6144x5120` | `[6144,5120]` | full-attn Q/gate, GDN V/Z | B | Explicit shared tuned GEMV | Tensor Core target for LargeT |
| `Out5120x6144` | `[5120,6144]` | attention/GDN output projection | B | Explicit shared tuned GEMV | Tensor Core target for LargeT |
| `MlpGateUp17408x5120` | `[17408,5120]` | MLP gate/up | A | Dedicated GEMV family | Dedicated Tensor Core target for LargeT |
| `MlpDown5120x17408` | `[5120,17408]` | MLP down | A | Dedicated GEMV family | Dedicated Tensor Core target for LargeT |
| `LmHead248320x5120` | `[248320,5120]` | final logits | A | Dedicated GEMV StoreOutputBF16 | Full-logits GEMM only for explicit parity/logprobs/debug use |

This table is a starting registry design. Profiling may split or merge families, but family-level
identity should remain stable for reports.

---

## 16. Reference and fallback backend

The correctness baseline should become a permanent reference/fallback backend.

Reference backend responsibilities:

- preserve exact q5090 decode semantics;
- support correctness tests for every qtype/layout;
- support unsupported shape/format/regime combinations;
- provide performance comparison baselines;
- keep tuned kernels honest.

Reference fallback is not a performance strategy for recurring TEXT-core linears. It is a correctness
and coverage mechanism.

Fallback must be explicit. A shape/format/regime miss should not silently route to an unintended
tuned kernel. Logs and benchmarks must identify when reference fallback was used.

---

## 17. Workspace policy

M3 performance `linear()` plans are externally workspace-free.

A plan is not eligible for the M3 performance registry if it requires:

- hidden global scratch allocation;
- engine-global temporary buffers not declared at a higher level;
- multi-kernel partial-sum reduction;
- persistent cross-CTA state that depends on external scratch;
- hidden synchronization.

This rule does **not** forbid normal per-kernel resources:

```text
registers
per-block shared memory
dynamic shared memory declared at launch
in-kernel staging buffers
```

The purpose is to ban hidden external workspace, hidden allocations, and multi-kernel reductions
behind the current `linear()` contract, not to ban shared-memory Tensor Core staging inside a single
kernel.

This rule intentionally excludes some possible optimizations from M3 `linear()`. That is acceptable:
it keeps the public API simple, preserves graph-capture friendliness, and makes benchmarks easier to
interpret.

If a future `lm_head_select` or sampler path needs workspace, it should declare that in its own
operator contract rather than adding workspace semantics to M3 `linear()`.

---

## 18. Benchmark and profiling contract

M3 linear optimization must be plan-based, not only operator-based.

Minimum benchmark identity:

```text
plan_id
backend kind: GEMV / GEMM / Reference
LinearFormat
ShapeFamily
LinearRegime
N, K, T
qtype and layout as decoded from LinearFormat
source role / callsite metadata if available, for attribution only
canonical vs reference/fallback path
policy metadata: Tensor Core use, dequant staging mode, accumulator mode, etc.
```

Do not add benchmark-only fields to `LinearPlanKey` unless they truly change M3 runtime dispatch.

### 18.1 Minimum metrics

For every relevant plan:

- microseconds per call;
- effective qweight bytes read, including scale bytes;
- effective weight bandwidth;
- input/output traffic estimate;
- packed load efficiency;
- register count and spills;
- shared memory per block;
- external workspace bytes, expected to be zero for M3 performance plans;
- auxiliary kernel count, expected to be zero for M3 performance plans;
- DRAM throughput;
- L2 hit behavior;
- numerical error vs reference.

For GEMV-specific analysis:

- token-level contribution by shape family;
- unpack instruction pressure;
- integer pipe pressure where measurable;
- reduction overhead;
- achieved qweight bandwidth relative to estimated resident-weight traffic.

For GEMM-specific analysis:

- Tensor Core utilization for `LargeT` Tensor Core target plans;
- unpack/dequant overhead relative to MMA work;
- shared-memory read/write traffic for staged tiles;
- SMEM bank conflict indicators;
- mainloop stall reasons;
- TTFT/prefill wall-clock contribution, not only isolated kernel time.

A kernel should not be labeled memory-bound or compute-bound from theory alone. That label requires
profiling evidence for the specific plan, shape, format, and `T` regime.

### 18.2 Minimum benchmark set before freezing policies

Before declaring a policy family successful, measure at least:

```text
T = 1
T = 2, 4, 8, 16
T = 32, 64, 128
representative long-prefill T values
```

The exact long-prefill values should match engine benchmark scenarios. Small-T measurements are
mandatory because `T > 1` is not automatically a large Tensor Core GEMM problem.

---

## 19. Integration with existing file layout

The public file remains unchanged:

```text
include/qus/kernels/linear.h
```

`linear` is the one L1 operator that does **not** use the flat `wrapper/ launcher/ kernel/` layout.
Because the M3 backend is large, its whole private implementation lives in a dedicated
`src/kernels/linear/` subtree, organized by responsibility. This exception is recorded in
`l1-kernel-layering.md` §3.1; every other operator stays flat.

```text
src/kernels/linear/
  linear.cpp                            # wrapper (host/gcc): validate -> classify -> switch dispatch
  plan/
    linear_plan.h                       # device-free: LinearFormat / ShapeFamily / LinearRegime,
                                        #   LinearPlanKey / LinearPlan, classify_*, constexpr registry + switch
  codec/
    linear_codec.cuh                    # WeightCodecTraits + device decode primitives (group/tile)
  reference/
    linear_generic.h                    # detail launch prototypes
    linear_generic_gemv.{cu,cuh}        # Generic GEMV (T==1), codec-driven, correctness
    linear_generic_gemm.{cu,cuh}        # Generic GEMM (T>1), codec-driven, correctness
  gemv/                                 # tuned GEMV plans (added per performance phase)
  gemm/                                 # tuned GEMM plans (added per performance phase)
```

The old flat `src/kernels/{wrapper,launcher,kernel}/linear*` files are removed (no backward
compatibility). The build glob is recursive, so the subtree needs no CMake change; private headers
still resolve as `kernels/linear/...`.

Layer responsibilities are unchanged, only relocated:

```text
api      : public declaration only            (include/qus/kernels/linear.h)
wrapper  : validation + classify + dispatch   (linear/linear.cpp, host)
plan     : keys, registry, switch, codec seam (linear/plan, linear/codec)
backend  : templated device compute           (linear/reference, later linear/gemv, linear/gemm)
```

No separate public dispatcher object is required.

---

## 20. MTP and vision boundary

MTP and vision are out of v1 TEXT-core M3 scope.

The framework should not block them:

- the format seam includes `W8G32_F16S + ROW_SPLIT`;
- the registry can later add shape families for MTP or vision;
- future modules may introduce new operators or derived layouts;
- TEXT-core M3 benchmarks should not carry W8/MTP/vision performance obligations.

This keeps v1 focused while preserving a clean precision/layout seam for later work.

---

## 21. Post-M3 and explicit-revision extension hooks

The following are intentionally excluded from the initial M3 core dispatch but remain plausible future
work.

### 21.1 `lm_head_select`

A separate operator for greedy/top-k/sampler-oriented output reduction. It may avoid full logits
writeback and may use workspace or multi-stage reductions. It should not be hidden behind
`linear()`.

### 21.2 Derived q5090 layouts

Explicit offline artifacts for GEMM-friendly weight staging if canonical `TILE_N64_K64` proves
insufficient after profiling. This requires explicit design revision and q5090 ABI/manifest update.

### 21.3 Workspace-enabled reductions

Split-K, top-k, or large-output reductions may need scratch. These should enter through an operator
contract that declares workspace, not through the current `linear()` API.

### 21.4 Additional formats

MTP, vision, or future hardware-native formats can be added through explicit `LinearFormat` values
and shape families when they enter scope.

### 21.5 Fused multi-projection GEMV

Keeping every projection under per-call `linear()` means decode launches qkv (`2048+2048+6144`) and
gate/up (`17408×2`) as separate kernels and reloads `x` for each. A fused multi-projection GEMV (one
launch over the concatenated `N`, shared `x` load) is a real decode occupancy / input-reuse lever,
especially for the small-`N` projections. It is intentionally **out of M3 `linear()` scope**: it is a
different operator (M4 / L2 fusion), because the q5090 file already stores these as separate payloads
(`q5090_packed_file_format_v1.md` §10) and `linear()` is contractually one projection per call. It is
recorded here so the boundary is a conscious decision, not an omission.

---

## 22. Open design questions for M3 profiling

The following remain measurement questions:

1. Best GEMV CTA/warp mapping for each Tier A/B shape.
2. Whether Tier C1 shared GEMV should be row-parallel, N64-stripe-based, or hybrid.
3. Exact `SmallT` threshold for each major shape family and format.
4. Whether small `T > 1` should use bundled GEMV-like scheduling or early Tensor Core staging.
5. Best canonical `TILE_N64_K64` staging strategy for Tensor Core `LargeT` candidates.
6. Whether Q5 requires distinct unpack/dequant policies because of bit-packing cost.
7. Whether `lm_head` store-output GEMV should use a distinct row-stripe/reduction policy from other
   GEMV shapes.
8. Whether a derived GEMM layout is justified by end-to-end TTFT/prefill improvement, not just a
   microbenchmark win.
9. Which Tier B shapes deserve explicit template instantiation after first profiling.
10. Whether persistent GEMV policies improve or hurt occupancy and scheduling on the target GPU.

These are not framework blockers. They should be resolved by benchmark data.

---

## 23. Summary

The M3 `linear` framework is:

```text
one public linear API with fixed StoreOutputBF16 semantics
  + static LinearPlanRegistry
  + three core dispatch axes:
      LinearFormat = qtype + layout
      ShapeFamily
      LinearRegime = T1 / SmallT / LargeT
  + narrow WeightCodec / FormatTraits seam for address + unpack + dequant primitives
  + separate GemvBackend and GemmBackend kernel families
  + bandwidth-first GEMV for T1
  + SmallT GEMM policies that may use GEMV-like scheduling
  + staged Tensor Core GEMM as the primary LargeT target for Tier A/B quantized TEXT-core shapes
  + shape-family specialization without per-layer kernels
  + permanent reference/fallback backend
  + externally workspace-free M3 performance plans
  + canonical q5090 TEXT-core layout in the initial M3 registry
```

M3 should not include:

```text
no epilogue polymorphism under linear()
no fused argmax/top-k hidden behind linear()
no source_kind in the core plan key
no math mode in the core plan key
no external workspace policy in the core linear API
no derived-layout policy in the initial M3 registry
no W8/MTP/vision performance obligation for TEXT-core M3
```

This design keeps the model card clean, protects the q5090 ABI, and gives M3 enough structure for
deep hand-tuned CUDA work without turning qwen3.6-ultraspeed into a general-purpose quantized GEMM
runtime.
