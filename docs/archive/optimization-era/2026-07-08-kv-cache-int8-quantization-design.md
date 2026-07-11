# KV Cache 8-bit Quantization — Design

Date: 2026-07-08
Status: design proposal (implementation not started)
Scope: L0 `KVCache` representation, L1 `gqa_attention` kernels, L2 model card wiring, CLI/serve surface.

---

## 0. TL;DR

- The single remaining blocker for full single-card long context (256K) is KV cache size. Full
  attention KV is **64 KiB/token** in bf16; at 131072 tokens that is 8 GiB and the card is already
  nearly full. 256K in bf16 does not fit at all next to the 16.3 GiB weight file.
- We add a second, first-class KV representation alongside bf16: **signed int8, symmetric,
  per-token, group-wise (group = 64), with one fp16 scale per group**. bf16 KV stays as a
  supported option (the accuracy/short-context default); int8 KV is the long-context option.
- This is a **runtime-only memory format**. The q5090 weight file ABI is untouched.
- The chosen format was selected after comparing int8 (sym/asym), fp8 (e4m3/e5m2) and several
  granularities against **this model's** KV statistics. The decisive facts: Qwen3.6 applies
  per-head RMSNorm to K (`k_norm`) *before* it is cached, RoPE touches only the first 64 of 256
  head dims, and V is a bias-free projection. Under those conditions group-wise int8 is the
  lowest-error, lowest-complexity, most kernel-friendly option, and the format matches the
  project's existing "signed int8 + per-group fp16 scale" weight convention (`W8G128`/`W8G32`).
- Kernel strategy: **quantization is confined to the global-memory representation and the
  global↔shared-memory staging boundary. The tensor-core QK / softmax / PV core stays bf16 and
  byte-identical.** The attention kernels are templated on a small `KvCodec` policy; only the
  K/V *append* (quantize) and *stage* (dequant) paths are specialized.

Result (bandwidth-roofline estimates, not measured):

| Context | KV mode | KV bytes | Fits 31.8 GiB usable? | Decode roofline* |
|---|---|---|---|---|
| 128K | bf16 | 8.5 GiB | yes, barely (~27.9 GiB total) | ~69 tok/s |
| 128K | int8 | 4.4 GiB | comfortable (~23.8 GiB) | ~82 tok/s |
| 256K | bf16 | 17.0 GiB | **no** (~36 GiB total) | — |
| 256K | int8 | 8.8 GiB | yes (~28.1 GiB) | ~68 tok/s |

\* decode is KV-bandwidth-bound at long context; roofline uses ~16.3 GiB weight traffic/step +
full KV read/step at 1.79 TB/s. Numbers are planning anchors, to be replaced by the e2e bench.

---

## 1. Goal & non-goals

### Goal
- Halve full-attention KV footprint so a single RTX 5090 can hold **256K** context next to the
  27B W4 weights, and improve **128K decode throughput** (decode reads the whole KV every step, so
  halving KV bytes lifts the KV-bandwidth-bound ceiling).
- Keep **bf16 KV** as a supported runtime mode (max accuracy, short/medium context, parity oracle).
- Add int8 KV support to the GQA prefill and decode kernels via templating, with a numerically
  faithful dequant so the tensor-core compute path is unchanged.

### Non-goals (this change)
- No 4-bit KV, no per-channel-across-sequence K quantization, no FP8 tensor-core attention math.
  These are discussed in §11 as explicitly deferred, with the reasons they are not chosen now.
- No change to the q5090 packed weight file format (KV is never serialized).
- No change to GDN/linear-attention state (those 48 layers keep no KV; they use conv+SSM state).
- No batching / multi-sequence work.

---

## 2. Motivation: the memory & bandwidth budget

### 2.1 What actually consumes VRAM today

Model constants (`include/qus/model/config.h`): `n_layers = 64`, full attention every 4th layer →
`n_full = 16`; `n_kv = 4`, `head_dim = 256`. KV is stored only for the 16 full-attention layers
(plus 1 MTP layer when `--mtp-draft-tokens > 0`).

Per full layer, per token, bf16:

```
K:  n_kv(4) * head_dim(256) * 2 B = 2048 B
V:  n_kv(4) * head_dim(256) * 2 B = 2048 B
K+V per layer per token          = 4096 B  (4 KiB)
16 full layers                   = 64 KiB / token   (68 KiB with the MTP layer)
```

| Region | 128K | 256K | Notes |
|---|---|---|---|
| W4 weights (`...v4.qus`) | 16.3 GiB | 16.3 GiB | 17,480,324,608 B; loaded once |
| Full-attn KV (16L) bf16 | 8.0 GiB | 16.0 GiB | 64 KiB/token |
| + MTP KV (1L) bf16 | +0.5 GiB | +1.0 GiB | only with `--mtp-draft-tokens>0` |
| GDN conv+SSM state (fp32) | ~0.77 GiB | ~0.77 GiB | context-independent |
| Workspace + IO/logits | ~1.5–2 GiB | ~1.5–2 GiB | reused per step |
| CUDA context/driver | ~0.5–1 GiB | ~0.5–1 GiB | |
| **Total (bf16 KV)** | **~27.9 GiB** | **~36 GiB** | 256K bf16 does not fit |

The card reports 32607 MiB total (~31.8 GiB usable). The `--max-context 131072` server the
operator runs today is exactly the "almost full" case in the table.

### 2.2 Why int8 unblocks 256K

int8 code + one fp16 scale per group of `g` elements costs `1 + 2/g` bytes/element:

| Scheme | Bytes/elem | vs bf16 | 16L KV/token |
|---|---|---|---|
| bf16 | 2.000 | 1.00× | 64 KiB |
| int8, g=256 (per-token) | 1.008 | 0.504× | 32.25 KiB |
| **int8, g=64 (chosen)** | **1.031** | **0.516×** | **33.0 KiB** |
| int8, g=32 | 1.063 | 0.531× | 34.0 KiB |

At g=64: 128K KV → **4.4 GiB**, 256K KV → **8.8 GiB** (incl. MTP layer). 256K total ≈ **28.1 GiB**,
fits with ~3.7 GiB headroom.

### 2.3 It is also a decode-speed win

Split-KV decode reads the **entire** KV history every step. KV bytes read/step equals KV storage
size. At 128K, bf16 KV = 8 GiB/step on top of ~16.3 GiB weight traffic; int8 halves the KV term.
Roofline at 1.79 TB/s:

| Context | weight + KV traffic/step | est. tok/s |
|---|---|---|
| 128K bf16 | 16.3 + 8.0 GiB | ~69 |
| 128K int8 | 16.3 + 4.1 GiB | ~82 (+19%) |
| 256K int8 | 16.3 + 8.3 GiB | ~68 |

So int8 KV is not only a capacity fix; it directly raises the long-context decode ceiling. (These
are roofline anchors; the M2.8 e2e bench is the source of truth for any claim.)

---

## 3. Choosing the 8-bit format

This is the crux of the design. We evaluate along three axes against this model's KV statistics.

### 3.1 What we are actually quantizing (model-specific facts)

From the model card forward (`src/model/qwen3_6_27b.cpp`), the tensors that land in the cache are:

- **K**: produced by `qkv` linear → reshaped per head → **`rmsnorm` (`k_norm`) per 256-dim head
  vector** → **RoPE on the first `rotary_dim = 64` dims** → written to cache. So cached K is
  *post-QK-norm and post-RoPE*.
- **V**: `qkv` linear slice → written to cache directly. No norm, no RoPE, no bias.

Two consequences dominate the choice:

1. **K is RMS-normalized before caching.** `k_norm` forces every cached K head vector to have
   (approximately) a fixed L2 magnitude times a learned per-channel gain. This is the crucial
   difference from raw-K models (e.g. the LLaMA setups that motivated KIVI's per-channel K
   scheme): per-token K dynamic range is already tamed, so **per-token int8 for K is viable
   here**, which it often is not elsewhere.
2. **Residual per-channel disparity still exists inside a head vector.** RMSNorm bounds the L2
   norm, not the max channel; the learned `k_norm` gain multiplies channels unevenly, and RoPE
   concentrates/rotates energy in the first 64 dims. A single scale over all 256 dims would let a
   few large channels inflate the scale and crush the rest → this is exactly what group-wise
   quantization fixes.

V is the classic "easy" tensor: attention output `O = Σ_j softmax_j · V_j` averages V quant error
over the whole (long) context, so V tolerates coarse quantization well.

### 3.2 Axis 1 — numeric type

| Option | For | Against (for this use) |
|---|---|---|
| **int8 symmetric** (scale only) | Uniform grid → lowest MSE for bounded, ~zero-mean data with a good per-group scale; dequant = 1 FMA into the existing bf16 path; matches project `W8*` convention; trivially reproducible for parity | Wastes ~1 code if a group is strongly one-signed (rare after norm) |
| int8 asymmetric (scale + zero-point) | Recovers skewed groups | 2nd param/group + a subtract in the hot dequant; skew is small here (bias-free projections, zero-mean-ish groups) — cost > benefit as default |
| FP8 e4m3 | HW-native cvt on sm_120; wide dynamic range for outliers | For a *bounded, per-group-scaled* range it spends mantissa on range we've already removed → higher error than int8 at 8 bits; its real payoff (FP8 tensor-core matmul) is unused because we dequant to bf16; ~3.5 effective mantissa bits |
| FP8 e5m2 | Even wider range | Only ~2.5 mantissa bits → clearly worse precision than int8 for KV |

The empirical consensus (KIVI, KVQuant, and every production `q8_0`-style KV path) is that
**well-scaled int8 KV is effectively lossless for attention** (sub-0.1% perplexity movement). FP8
KV is also "fine" but generally a touch worse at equal bits and buys us nothing here because we do
not run FP8 tensor cores on the KV. **Decision: int8, symmetric.**

### 3.3 Axis 2 — granularity

| Granularity | Append-friendly? | K accuracy | Cost |
|---|---|---|---|
| per-tensor | no (needs global stats) | poor | — |
| per-channel across sequence (KIVI-K) | **no** — scale depends on the whole growing sequence; needs a fp16 residual window + periodic requant | best for raw-K outliers | high complexity, awkward for a streaming append-only cache and for this tiled kernel |
| per-token (g=256) | **yes** — each new token quantized independently at append | good *because K is pre-normed* | 1 scale/token |
| **per-token group-wise (g=64)** | **yes** | good; isolates the RoPE'd/high-gain channels into their own group | 4 scales/token (0.4 KiB/token overhead across 16L) |

A growing KV cache is fundamentally append-only, which rules out per-channel-across-sequence
without a residual+requant machine we don't want. Given K is already normed, **per-token
group-wise** captures nearly all of what per-channel would, while staying append-native.

Group size: `g` must divide the vectorized load width so a single load stays inside one group.
The kernels move 16-byte chunks (8 bf16, or 16 int8) per thread, and `8|64`, `16|64`, so g=64
keeps one scale per load. g=64 also mirrors the weight group size (mental-model consistency) and
costs only 4 fp16 scales per 256-dim vector. g=32 is a drop-in fallback (still divides 16) if
parity ever demands finer K granularity.

**Decision: per-token, group-wise, group = 64 (compile-time `kKvQuantGroup`, K and V may differ).**

### 3.4 Decision

> **int8, signed, symmetric, per-token, group-wise (group = 64), one fp16 scale per group.**
> bf16 KV remains a supported mode.

Rejected, with reasons: FP8 (no matmul benefit, worse precision at 8 bits), asymmetric (skew too
small to justify the extra param + subtract as default), per-channel-across-sequence K (not
append-native; the pre-cache `k_norm` already removes most of its motivation).

---

## 4. The format specification (int8 KV)

Per full-attention layer, K and V are stored independently, each as two planes.

**Code plane** — signed int8, logical shape identical to today's cache
`[head_dim = 256, padded_context, n_kv = 4]`, 1 byte/elem. Slot layout unchanged: the 256 head
dims of one `(kv_head, position)` are contiguous.

```
code_index(kv_head, d, pos) = d + 256 * (pos + padded_context * kv_head)
```

**Scale plane** — fp16, shape `[G, padded_context, n_kv = 4]` where `G = head_dim / g = 256/64 = 4`.

```
scale_index(kv_head, group, pos) = group + G * (pos + padded_context * kv_head)
```

Both planes share the `(pos, kv_head)` addressing, so a kernel that knows the code offset derives
the scale offset with the same position math (just `/g` on `d` and a narrower stride).

**Quantize (append), per group of 64 dims of one head vector:**

```
absmax_g = max_{d in group} |x_d|                # x is the bf16 K or V head vector
scale_g  = absmax_g / 127         (fp16)         # scale_g = 0 → store 0, code = 0
code_d   = clamp(round(x_d / scale_g), -127, 127)  (int8)
```

**Dequantize (stage):** `x_d ≈ float(code_d) * float(scale_g)` → bf16.

Overhead: 4 fp16 scales per 256 int8 codes = 8 B / 256 B = 3.1%. Per-layer per-token K+V =
`2 * (256 + 8) = 528 B` vs 4096 B bf16 → 0.516×.

Notes:
- Symmetric uses 127 (not 128) as the positive-clip magnitude so `code = -128` never occurs; this
  keeps `|code| ≤ 127` and dequant symmetric.
- The uninitialized/padded cache tail is never read (the causal mask + `max_query_abs` predication
  in the kernels already drop it), so quantized garbage in unwritten slots is harmless, exactly as
  today's bf16 tail is.

---

## 5. Runtime data structures

### 5.1 `DType` (L0)

Add two element types used only by the quantized cache planes:

```
enum class DType : uint8_t { BF16, FP32, I32, U8, I64, I8, FP16 };
// dtype_size: I8 -> 1, FP16 -> 2
```

`I8` is a distinct signed type (vs the existing `U8`) so kernel/wrapper validation is explicit;
`FP16` is needed for the scale plane (the project already uses fp16 scales for weights, just via
raw bytes — here it becomes a first-class tensor dtype).

### 5.2 `KVCache` (L0, `include/qus/core/kv_cache.h`, `src/core/kv_cache.cpp`)

Extend the cache to carry scale planes when quantized. The existing `DType dtype` field selects
the mode.

```
struct KVCache {
    std::vector<Tensor> k, v;              // BF16 mode: bf16 data. INT8 mode: I8 code planes.
    std::vector<Tensor> k_scale, v_scale;  // empty in BF16 mode; FP16 [G, padded_context, n_kv].
    ...
    DType dtype = DType::BF16;             // BF16 or I8
    int   quant_group = 0;                 // 0 for bf16; 64 for int8
};
```

- Constructor allocates code planes at 1 B/elem and (when `dtype == I8`) the scale planes.
- `preflight_cache_arena` / `default_cache_bytes_for` size math (in `engine.cpp`) switch on
  `dtype`: replace `dtype_size(BF16)` with `dtype_size(I8)` for the code planes and add the scale
  bytes (`G * padded_context * n_kv * sizeof(fp16)` per K/V per layer).
- `slot_at` / `slot` / `append_slot` become mode-aware: in int8 mode they return the code sub-slot
  and the parallel scale sub-slot (`KVHeadSlot` gains optional `k_scale`/`v_scale` views). These
  helpers are used by tests and CPU-side bookkeeping; the actual GPU append happens in the kernels.
- `advance` / `rewind` / `reset` are index-only and unchanged (prefix-cache reuse, MTP rewind, GDN
  snapshot logic all keep working — they never touch element bytes).

### 5.3 Engine wiring (`src/runtime/engine.cpp`)

- `EngineOptions` gains `DType kv_dtype = DType::BF16;` and `int kv_quant_group = 64;`.
- `Engine::load` passes `kv_dtype` to both the main and MTP `KVCache` constructors.
- The cache-arena budget helper accounts for the int8+scale sizes so the preflight stays exact.

---

## 6. Kernel design

### 6.1 Guiding principle

The QK GEMM, online softmax, and PV GEMM in both `gqa_attention_prefill.cuh` and
`gqa_attention_decode.cuh` operate entirely on **bf16 tiles already in shared memory**. We keep
that core byte-identical. Quantization is inserted at exactly two seams:

1. **Append/quantize (write):** bf16 new-token K/V → int8 code + fp16 scale in the global cache.
2. **Stage/dequant (read):** int8 code + fp16 scale in global → bf16 tile in shared memory.

Everything between those seams is unchanged, which bounds the accuracy risk to "int8 round-trip on
the KV inputs" and bounds the perf risk to the staging routine.

### 6.2 Templating strategy

Introduce a compile-time `KvCodec` policy and template the kernels on it:

```
struct KvBf16 {                     // identity: current behavior
    static constexpr bool kQuantized = false;
    const __nv_bfloat16 *k, *v;
    // stage_tile(...) = today's cp.async of bf16 -> swizzled smem
    // append(...)     = today's int4 copy
};

struct KvInt8G64 {
    static constexpr bool kQuantized = true;
    static constexpr int  kGroup = 64;
    const int8_t *k_code, *v_code;
    const __half *k_scale, *v_scale;
    // stage_tile(...) = cp.async int8+scale -> dequant -> swizzled bf16 smem
    // append(...)     = per-group absmax -> quantize -> write code+scale
};
```

- `gqa_attention_prefill_kernel<Codec>` and
  `gqa_attention_small_t_tc_partial_kernel<TokenTile, WarpsPerCta, Codec>` are templated; the MMA
  body is shared source, only `Codec::stage_tile` / `Codec::append` differ.
- Instantiate exactly two codecs (bf16, int8-g64). This doubles the small-T instantiation set
  (T=1..6 × warp configs) and the prefill kernel — acceptable binary-size cost.
- Launchers dispatch on `kv.dtype`; wrappers validate the mode (see §6.6).

### 6.3 Append / quantize path

**Prefill** (`gqa_attention_prefill_fill_kernel`): today one thread copies an 8-bf16 chunk of a
`(kv_head, token)` head vector into the cache via `int4`. The int8 variant instead:
- cooperatively loads the 256-dim bf16 head vector for a `(kv_head, token)`,
- computes per-64-group `absmax` (warp/shared reduction over the group's threads),
- writes 256 int8 codes + 4 fp16 scales.
This is a small, self-contained kernel; the reduction is cheap (prefill T is at most the chunk).

**Decode**: the current small-T partial kernel *fuses* the new-K/V write into the attention kernel
and also reads the new tokens straight from `k_new`/`v_new` (the `from_new` path) to avoid a RAW
sync. For int8 we do **not** thread quantization through that hot, hand-tuned kernel. Instead:
- Add a tiny **`kv_quantize_append` pre-pass kernel** (T ≤ 6 tokens × 4 kv_heads) that quantizes
  the new K/V into the cache before attention runs.
- The int8 decode partial kernel then reads K/V **exclusively from the (quantized) cache**,
  dropping the `from_new` special-casing entirely (a genuine simplification). The diagonal/current
  tokens are read back through the same int8 round-trip as history, which is the correct, consistent
  behavior for a quantized cache.
- The **bf16 decode kernel keeps its current fused, `from_new` path unchanged** — no regression to
  the tuned bf16 hot path. The pre-pass exists only for int8. This asymmetry is deliberate: protect
  the existing bf16 decode performance, isolate all new code in the int8 specialization.

In the CUDA-graph decode/round capture, the int8 pre-pass is one extra small node before each
full-attention layer's attention node.

### 6.4 Stage / dequant path

Both kernels currently stage a `[Bc, D]` K or V tile into a swizzled bf16 smem buffer via
`cp.async` (pure copy). For int8:

- **Prefill (compute-bound):** keep `cp.async` to preserve K/V load↔MMA overlap. `cp.async` the
  int8 codes (half the bytes) + the tile's fp16 scales into a compact int8/scale smem staging
  area, then a warp pass converts code×scale → bf16 into the existing swizzled tile. Extra smem:
  one `[Bc,D]` int8 tile per K and V = `2 * 64 * 256 = 32 KiB` on top of the current 96 KiB dynamic
  smem → 128 KiB, within sm_120's opt-in max (~227 KiB) via `cudaFuncAttributeMaxDynamicSharedMemorySize`.
- **Decode (bandwidth-bound):** the simplest correct path is a plain vectorized global load
  (`LDG.128` = 16 int8 = 16 dims/thread) + register dequant straight into the swizzled bf16 smem
  tile. Losing `cp.async` overlap barely matters when the load itself is halved and the kernel is
  memory-bound; the split-KV scratch (`partial_acc/m/l`) is unaffected. A `cp.async`-into-staging
  variant is a later tuning option if profiling wants it.

Scale locality: a thread's 16-dim chunk sits in exactly one 64-group, so it loads a single fp16
scale and applies one multiply across its 16 dequantized values.

### 6.5 Shared-memory budget check

| Kernel | bf16 smem | int8 add'l staging | total |
|---|---|---|---|
| prefill (dynamic) | 96 KiB (Q+K+V) | +32 KiB (int8 K+V) | 128 KiB (opt-in) |
| decode small-T (static) | ~32–40 KiB | 0 (register dequant) | unchanged |

Prefill must raise its dynamic-smem opt-in from 96 → 128 KiB for the int8 instantiation. Occupancy
impact is limited (prefill already runs 1 block/SM by `__launch_bounds__(128,1)`).

### 6.6 Wrapper / launcher / dispatch

- `src/kernels/wrapper/gqa_attention.cpp`: the current hard check
  `kv.dtype != DType::BF16 → throw` becomes `kv.dtype ∈ {BF16, I8}`; in I8 mode validate the code
  planes are `I8 [256, padded_context, 4]` and the scale planes are `FP16 [G, padded_context, 4]`,
  contiguous and non-null. q/k/v/out stay bf16 (only the *cache* is quantized).
- Launchers select the `KvCodec` instantiation from `kv.dtype` and, for int8 decode, enqueue the
  `kv_quantize_append` pre-pass before the partial kernel.
- No change to the public `gqa_attention(...)` signature — the mode travels inside `KVCache`.

---

## 7. Config & CLI surface

Add one option, threaded ServeOptions/CLI → `EngineOptions.kv_dtype`:

- `--kv-dtype {bf16|int8}` (default `bf16`), plumbed in `src/serve/serve_options.cpp` +
  `src/text/cli.cpp`, mapped in `src/main.cpp` and `src/serve/generation_service.cpp`.
- Optional `--kv-quant-group {32|64}` (default 64) for tuning; may be omitted from the first cut
  and hardcoded to 64.
- Startup log line already prints memory stats; extend it to report KV mode + measured KV bytes so
  the operator can see the budget. (`EngineMemoryStats` can gain a `kv_dtype`/`kv_bytes` field.)

The operator's current `--max-context 131072` run becomes `--max-context 262144 --kv-dtype int8`.

---

## 8. Verification plan

Performance is the primary acceptance gate for this change; correctness is a hard precondition.
Both use the existing bench/test harnesses. No `compute-sanitizer` step is required for this work.

### 8.1 Performance benchmarks (primary gate)

Two harnesses, both already parameterized by token count and context length. Both must gain an
**int8 KV mode** and the microbench must fix its byte model (today it hardcodes bf16 via
`sizeof(std::uint16_t)`; int8 KV traffic is `1 B/elem code + fp16 scale/group`, so the achieved-
bandwidth and copy-ceiling math must switch on KV dtype). Every point is measured for **bf16 and
int8** so the report is a direct A/B.

**(a) Op microbench — `bench/gqa_attention_bench.cu` (`qus_gqa_attention_bench`).**
It already exposes `--tokens` (T) and `--context` (history length), and reports latency + achieved
KV bandwidth against a copy ceiling. Required sweeps:

- **Prefill T=1024 across context** (the `--append-prompt-baseline`/`--prefill` paths):

  ```
  qus_gqa_attention_bench --append-prompt-baseline --tokens 1024 \
      --context 0,8192,32768,65536,131072,262144
  ```
  Report per context: kernel latency, achieved TFLOP/s (and %-of-peak the bench already computes),
  KV-read bytes and achieved GB/s, for bf16 vs int8. Expectation: prefill is compute-bound, so
  int8 should be within a few % of bf16 at the same context; the int8 K/V staging (extra smem +
  dequant pass, §6.4/§6.5) must not regress it materially. Watch the largest contexts where KV
  staging bytes grow.

- **Decode T=1/2/3/4 across context** (the split-KV small-T path, `--append-small-t`):

  ```
  for T in 1 2 3 4; do
    qus_gqa_attention_bench --append-small-t --tokens $T \
        --context 4096,8192,16384,32768,65536,131072,262144
  done
  ```
  Report per (T, context): kernel latency, KV-read bytes, achieved GB/s vs the `--copy-ceiling`
  reference, for bf16 vs int8. Expectation: decode is KV-bandwidth-bound, so halving KV bytes
  should move latency toward ~0.5× at long context; the win should grow with context and be
  largely flat across T (T only scales the query side, not the dominant KV read). The context grid
  spans the split-tier boundaries (4096/8198/16390) so tier transitions are visible.

**(b) End-to-end real-weight bench — `bench/qus_bench.cpp` (`qus_bench`).**
Confirms the op wins survive the full model and the memory budget holds. Add `--kv-dtype`.

- **Prefill (pp) at length 1024** and **decode (tg) at context offsets** via the combined path,
  bf16 vs int8:
  ```
  qus_bench --weights out/qwen3_6_27b.q5090_w4g64_mixed_v4_1.qus --kv-dtype int8 \
      -pg 8192,64;32768,64;65536,64;131072,64;262144,64 -r 5
  ```
  `-pg P,G` prefills `P`, then times `G` decode steps at context offset `P` — this is how we get
  "decode tok/s as a function of context length". Report `prefill t/s` and both decode tok/s views
  per context, bf16 vs int8 (256K bf16 is expected to be unloadable — record that as the capacity
  result). Repeat with `--mtp-draft-tokens 3` so the decode-round path (verify T = 4) is measured
  under int8 KV, matching the T=4 op sweep.
- **Memory**: `EngineMemoryStats` at load confirms 256K int8 loads with KV ≈ 8.8 GiB and total
  < 31.8 GiB; the same probe shows 256K bf16 failing to fit. These replace the §2 roofline anchors
  with measured numbers.

Deliverable: a results doc under `profiles/` (matching the existing profile-report convention)
with the bf16-vs-int8 tables for prefill-T=1024-vs-context and decode-T{1..4}-vs-context, plus the
e2e `qus_bench` A/B and the memory result.

### 8.2 Correctness (precondition)

Per the testing policy, KV quant touches **numerical CUDA-kernel correctness** (whitelist #1) and
**binary/format contracts** (whitelist #2):

1. **Op-level parity** (`tests/kernels/test_gqa_attention.cpp`, the existing oracle): add an int8
   variant that runs the same shapes as the bf16 test and compares int8-KV attention output
   against an **fp32 reference that applies the identical int8 round-trip to the reference K/V**
   (so the test measures kernel fidelity to the format, not the format's model-level loss). Assert
   tight rel error vs that oracle. Include long-context tiles (multi-key-block) and the decode
   split-KV path, plus the diagonal/current-token read-back.
2. **Format round-trip** (small): CPU quantize→dequantize of known vectors incl. an all-zero group
   (scale 0 → code 0) and a single-outlier group, asserting clamp/round.
3. **Model-level accuracy gate**: greedy-match / judge run (the existing `out/…_judge` harness) at a
   fixed prompt set comparing bf16-KV vs int8-KV decode. Acceptance: negligible divergence (target:
   identical greedy tokens for a meaningful prefix, or a pre-agreed small divergence threshold). If
   K fails, drop K to g=32 (V stays g=64) and re-measure before considering asym.

No source-structure/string tests. bf16 KV parity tests remain the primary correctness oracle.

---

## 9. Accuracy risk & mitigations

| Risk | Why bounded here | Mitigation |
|---|---|---|
| K logit sensitivity (QK is exponentiated) | K is pre-`k_norm`'d → tame per-token range; group=64 isolates RoPE'd/high-gain channels | g=32 for K if parity fails; asym only if a skew is actually measured |
| V error | Averaged over the softmax sum → very forgiving | g=64 is already conservative for V |
| RoPE outlier channels (first 64 dims) | Fall entirely inside group 0 (g=64) or groups 0–1 (g=32) → their own scale | group-wise already handles it |
| Padded/unwritten tail garbage | Causal mask + `max_query_abs` predication never read it | unchanged from bf16 |
| Determinism/parity | int8 symmetric round-trip is exactly reproducible | fixed round-to-nearest, clamp ±127 |

---

## 10. Rollout / task breakdown

Execution mode: **direct, sequential single-developer** implementation (no subagents). Land the
steps in order; each step builds the affected targets and runs its check before the next begins.

| # | Step | Owns / touches | Done when |
|---|---|---|---|
| T1 | `DType::I8`/`FP16` + `KVCache` int8 planes + engine budget | `dtype.*`, `kv_cache.*`, `engine.cpp` budget, `test_kv_cache.cpp` | int8 cache allocates; sizes match hand math; `test_kv_cache` extended and green |
| T2 | int8 append/quantize kernels (prefill fill + decode pre-pass) + CPU round-trip test | `gqa_attention_prefill.cuh` fill, new `kv_quantize_append`, launchers | codes/scales match the CPU round-trip oracle |
| T3 | int8 stage/dequant + kernel templating (`KvCodec`) for prefill & decode | `gqa_attention_{prefill,decode}.cuh`, launchers, wrapper validation | int8 attention matches the round-trip oracle in `test_gqa_attention`; bf16 path byte-unchanged |
| T4 | CLI/serve `--kv-dtype`, engine option, mem-stats reporting | `serve_options.*`, `cli.cpp`, `main.cpp`, `generation_service.cpp`, `engine.h` stats | `--kv-dtype int8` runs; stats show KV mode+bytes |
| T5 | int8 KV mode + byte model in the benches | `gqa_attention_bench.cu` (byte model), `qus_bench.cpp` (`--kv-dtype`) | both benches accept int8 and account bytes correctly |
| T6 | **Performance sweeps + report** (§8.1) and the accuracy gate (§8.2) | `profiles/` report only | prefill-T=1024 and decode-T{1..4} bf16-vs-int8 tables captured; 256K int8 loads < 31.8 GiB; judge within threshold |

Dependencies: T1 → T2/T3 (shared `KVCache` shape); T2 → T3 (decode reads what append writes);
T4 independent after T1; T5 after T3/T4; T6 last (needs T5's bench modes). Correctness checks in
T2/T3 gate the performance work in T6 — do not tune against a numerically wrong kernel.

Review: T2/T3 change CUDA kernels and numerics; review the int8 index math, the raised prefill
smem opt-in, and the append pre-pass lifetime under repeated prefill/decode/rewind by reading the
diffs and confirming the op-parity test (§8.2) covers those paths.

---

## 11. Deferred (and why not now)

- **4-bit KV**: doubles the win again (256K → ~4.4 GiB) but int4 KV needs finer groups / asym /
  possibly a fp16 residual window to hold accuracy, and its dequant is heavier. Land int8 first as
  the proven, low-risk step; revisit int4 only if 256K headroom or bandwidth demands it.
- **Per-channel-across-sequence K (KIVI-style)**: highest K fidelity, but not append-native
  (scale spans the growing sequence) and needs a residual+requant machine. The pre-cache `k_norm`
  removes most of its motivation for this model; excluded on complexity grounds.
- **FP8 tensor-core attention**: would keep KV in FP8 *and* run FP8 MMA (skip dequant-to-bf16).
  Bigger kernel rewrite, separate accuracy study, and orthogonal to the capacity goal. Not part of
  this change.

---

## 12. Files touched (summary)

- L0: `include/qus/core/dtype.h`, `src/core/dtype.cpp`, `include/qus/core/kv_cache.h`,
  `src/core/kv_cache.cpp`.
- L1: `src/kernels/kernel/gqa_attention_prefill.cuh`, `src/kernels/kernel/gqa_attention_decode.cuh`,
  `src/kernels/launcher/gqa_attention_{prefill,decode}.cu`, `src/kernels/launcher/gqa_attention.h`,
  `src/kernels/wrapper/gqa_attention.cpp`.
- L2/runtime: `src/runtime/engine.cpp`, `include/qus/runtime/engine.h`.
- Surface: `src/serve/serve_options.cpp`, `include/qus/serve/serve_options.h`, `src/text/cli.cpp`,
  `src/main.cpp`, `src/serve/generation_service.cpp`.
- Tests/bench: `tests/kernels/test_gqa_attention.cpp`, `tests/test_kv_cache.cpp`,
  `bench/gqa_attention_bench.cu` (int8 mode + byte model), `bench/qus_bench.cpp` (`--kv-dtype`).
- Docs/reports: `docs/design.md` §8/§9 (record int8 KV as a supported mode); a new bf16-vs-int8
  performance report under `profiles/` (prefill-T=1024-vs-context, decode-T{1..4}-vs-context, e2e).

The q5090 weight file format is **not** touched.
