# GQA int8 decode kernel redesign — int8-native tensor core, occupancy-first

Date: 2026-07-08
Status: design + implementation plan (implementation starting)
Scope: L1 `gqa_attention_small_t_tc_partial_i8_kernel`
(`src/kernels/kernel/gqa_attention_decode_i8.cuh`), the shared scaffolding it uses
(`gqa_attention_decode.cuh`), and the int8 decode parity oracle in
`tests/kernels/test_gqa_attention.cpp`. The bf16 decode kernel and the prefill
kernels are **not** touched.

---

## 0. TL;DR

The current int8 decode kernel is the bf16 kernel with an int8→bf16 dequant bolted
onto the load path; it then runs the *identical* bf16 tensor-core core. That design
is **slower than bf16** because:

1. it gets **zero tensor-core compute benefit** (still `m16n8k16` bf16 MMA, still
   64 QK + 64 PV MMAs per key block);
2. it pays **extra instructions** (per-group scale staging + a 4-sub-tile convert
   pipeline + int8→float→`cvt.bf16x2` dequant for *both* K and V) with ~9
   `__syncthreads` per key block vs the bf16 kernel's 2; and
3. at decode occupancy the kernel runs at **~1 warp/scheduler**, where — as the
   sister prefill kernel's NCU audit proved
   (`docs/2026-07-08-gqa-prefill-fa-parity-audit.md` §2) — *runtime tracks issued
   instructions almost linearly*. The added instructions cost more wall time than
   the ~2× DRAM bytes int8 saves, and the kernel never reaches the bandwidth
   roofline where int8 is supposed to win.

The redesign keeps **KV int8 all the way through the tensor cores** and uses int8's
half-size smem to **raise occupancy from 2 → 3–4 blocks/SM** (1 → 3–4
warps/scheduler), which is what actually converts "half the bytes" into "half the
time."

Two phases:

- **Phase 1 (this doc's implementation target):** native `m16n8k32.s8` QK (Q
  quantized on-chip to int8, K read int8 from cache, per-group int32→float
  rescale), **bf16 PV** (V dequanted), smem reuse for occupancy, single-wave
  cp.async pipeline, FFMA-form softmax. This alone should flip int8 from
  slower→faster.
- **Phase 2 (follow-up):** full int8 PV via per-group P-folding to drop the bf16 V
  tile and push occupancy to ~4 blocks/SM. Deferred; noted in §7.

---

## 1. Goal & non-goals

### Goal
- Make int8 decode/verify (T ∈ [1,4], code covers 1..6) **faster than bf16** at the
  decode shapes, approaching the int8 KV-bandwidth roofline (~2× below bf16 at long
  context).
- Preserve numerical fidelity to the int8 KV format: the kernel must match an fp64
  oracle that applies the *same* int8 round-trip the kernel uses (whitelist #1
  numerical-correctness test).
- Keep the bf16 decode kernel and both prefill kernels byte-unchanged.

### Non-goals
- No change to the int8 KV cache format (per-token, group-wise g=64, one fp16
  scale/group). The kernel consumes the fixed format.
- No FP8 path, no int4, no per-channel-across-sequence K.
- No change to the split-KV grid schedule (`kGqaDecodeSplits = 192`), the reducer
  kernel, the launcher signature, or the public `gqa_attention` API.
- Phase-2 full-int8 PV is out of scope for the first landing.

---

## 2. Why the current kernel loses (evidence)

- **Same MMA as bf16:** `gqa_attention_decode_i8.cuh` calls
  `gqa_small_t_tc_mma_m16n8k16_bf16` for both QK and PV; the only MMA helper in the
  shared header is bf16 (`gqa_attention_decode.cuh:116-123`). int8's 2× TC and
  halved contraction (k=32 vs k=16) are unused.
- **Dequant + barrier tax:** the convert pipeline (`gqa_attention_decode_i8.cuh`
  ~277-323) dequants both K and V (32×256 conversions/key block) and issues ~9
  `__syncthreads` per key block; the bf16 kernel issues 2.
- **Occupancy is the ceiling:** `__launch_bounds__(128, 2)` +
  `qkv_s`(32 KB)+`p_s`(4 KB)+dynamic(8.7 KB) ≈ 44 KB/block → 2 blocks/SM; at T=1
  (Wc=2, 64 threads) that is 128 threads/SM = **1 warp/scheduler**. The prefill
  audit establishes that in this regime wall time ≈ instruction count, so the
  dequant instructions dominate the bandwidth savings.

Roofline sanity: decode reads the whole KV every step; K+V int8 ≈
`window·4·528 B`/layer, ~2× below bf16. The current kernel being *slower* means it
is instruction/latency-bound, not bandwidth-bound — so the fix must cut
instructions **and** raise occupancy, exactly what int8-native tiles enable.

---

## 3. Hardware & workload envelope

- RTX 5090, `sm_120a`, CUDA 13.1, ~1.79 TB/s DRAM, 170 SMs, ~100 KB smem/SM,
  48 KB static / opt-in dynamic beyond.
- `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`: A = 16×32 s8 → 4×b32/thread,
  B = 8×32 s8 → 2×b32/thread, C/D = 4×s32/thread. **Same register footprint** as
  the current `m16n8k16` bf16 fragments, so the warp/lane scaffolding
  (`gid/lid/a_*`/`b_*`) transfers.
- Workload: T ∈ [1,6], rows = T·6 ∈ [6,36 capped to Br]; D=256; 4 KV heads ×
  group 6. Split-KV grid `(4, ≤192)` already provides key-parallelism across CTAs;
  the missing ingredient is intra-SM latency hiding = occupancy.

---

## 4. Architecture (Phase 1)

### 4.1 Read all keys from the quantized cache (drop `from_new`)

The int8 MMA needs both operands int8. The fused append already quantizes the new
tokens' K/V into the cache; the kernel then reads **every** key (history *and* the
current/diagonal tokens) back from the int8 cache. `__syncthreads()` after the
append gives the block the ordering guarantee (same-block global write→read).
Within a launch, each absolute position belongs to exactly one split's range, so
there is no cross-CTA readback race. This drops the `from_new` special-casing (a
genuine simplification) and makes the diagonal token go through the same int8
round-trip as history — the correct behavior for a quantized cache.

### 4.2 QK: native `m16n8k32.s8`

- **Q on-chip quant.** During Q staging, quantize each (row, 64-group) of Q to
  symmetric int8: `qs[r,g] = absmax_g / 127` (fp32), `Qi8 = round(Q/qs)`, clamp
  ±127. Q is read once; the reduction is a warp reduction over the group lanes.
- **K stays int8**: cp.async int8 K tile → smem → int8 fragments (no dequant).
- **Per-group score:**
  ```
  S[r,key] = Σ_{g=0..3} qs[r,g]·ks[key,g]·( Σ_{d∈g} Qi8[r,d]·Ki8[key,d] )
  ```
  Each 64-group = two `m16n8k32.s8` MMAs sharing one int32 accumulator; read out
  the int32, FMA `qs·ks·int32` into the fp32 score, zero for the next group.
  Per key block: **32 s8 MMAs** (vs 64 bf16) + a light per-group rescale.

Numerically this is *closer* to the fp64 oracle than today's path: int8×int8 is
exact in int32; rounding only enters at the per-group float rescale, versus
bf16-rounding every partial product.

### 4.3 PV: bf16 (Phase 1)

V is quantized per (key, group); PV contracts over keys, so the per-key V scale
cannot be factored out of an int8 accumulation. Phase 1 keeps PV in bf16:
cp.async int8 V → dequant `code·scale → bf16` into the bf16 V tile → existing bf16
PV MMA with bf16 P. V is still **read from DRAM as int8** (bandwidth win kept;
dequant is on-chip). Only V is dequanted now (K is not), halving today's dequant.

### 4.4 Occupancy via smem reuse

Q is staged to smem, `ldmatrix`'d into registers, then that region is dead — reuse
it for the K/V tiles (the bf16 kernel already reuses `qkv_s` this way).

| region | bytes |
|---|---|
| Q int8 stage (reused for K/V after ldmatrix) | Br·D·1 (≤16 KB) |
| K int8 tile | Bc·D·1 = 8 KB |
| V int8 stage + V bf16 tile | 8 KB + 16 KB |
| P bf16 | Wc·16·Bc·2 = 4 KB |
| K/V fp16 scales (staged) | ~1 KB |

Peak ≈ max(Q-stage, K+V) + P + scales ≈ **~28 KB** vs ~44 KB today →
`__launch_bounds__(128, 3)` (3 blocks/SM). Use Wc=4 (128 threads) across T for a
consistent 3×128 = 384 threads/SM = ~3 warps/scheduler, vs 1 today. This is the
change that makes the halved bytes show up as halved time.

### 4.5 Pipeline & instruction hygiene

- **One tile, one wave:** single cp.async of the full int8 K (and V) tile, one
  `wait<0>`, one `__syncthreads`. K needs no convert; V-dequant is one fused pass.
  Double-buffer across key blocks at *tile* granularity (not sub-tile within a
  block); target ≤3 `__syncthreads`/key block.
- **FFMA-form softmax** (pre-scaled max, `__fmaf_rn(score, scale_l2, -m_scaled)`)
  and **deferred row-sum allreduce** (prefill audit §10 items 2 & 4).
- **cp.async store map = 8 threads / 64B page** and **precomputed base+immediate
  ldmatrix addresses** (prefill audit §5.3–5.4).

---

## 5. Numerics & the parity oracle

The int8 decode oracle in `tests/kernels/test_gqa_attention.cpp`
(`one_int8_decode_case`) currently: keeps Q in bf16, dequant-rounds K to bf16, and
reads the new/diagonal tokens **raw bf16** (via `cpu_gqa_prefill`). The redesigned
kernel changes all three, so the oracle must be updated to model the kernel's
actual arithmetic:

1. **Quantize Q** per (row, 64-group) with the same fp16-scale round-trip the
   kernel uses; use the exact dequant `qs·Qi8` (fp, not bf16) in the QK dot.
2. **All keys from the quantized cache:** use `expected_k` (history + new codes)
   for every position; K dequant for QK is exact `ks·Ki8` (no bf16 rounding, since
   the s8 MMA does not round K to bf16).
3. **V stays bf16-dequant** (`dequantize_cache_bf16(expected_v)`) for the bf16 PV,
   for every position including the diagonal.

Because QK is now int8×int8 accumulated in int32, algebraically
`Σ_g qs_g·ks_g·(Σ_{d∈g} Qi8·Ki8) = Σ_d (qs·Qi8)(ks·Ki8)`, so the oracle can
dequantize Q and K to fp and take the fp64 dot; the composite `attention_bf16`
tolerance absorbs the fp32/bf16 accumulation difference. A new
`cpu_gqa_decode_int8` reference (mirroring `cpu_gqa_prefill`'s per-token causal
alignment but reading all keys from the quantized cache with quantized Q) replaces
the `cpu_gqa_prefill` call in `one_int8_decode_case`. The `one_int8_prefill_case`
oracle is untouched (prefill still uses bf16 dequant QK).

Accuracy gate: int8-Q is a new model-level variable. Per-token/group int8 Q is
standard in production int8 attention; it must still pass the greedy/judge harness.
Fallback if it fails: bf16 QK (keep §4.4 occupancy + §4.5 pipeline, lose only the
2× QK TC).

---

## 6. Task breakdown (execution mode: direct, sequential, single-agent)

This is one tightly-coupled CUDA kernel plus its parity oracle, debugged
iteratively against a single numerical test; it is not decomposable by independent
file ownership, so it is **not** subagent-driven. Land in order; each step builds
and runs its check before the next.

| # | Step | Owns / touches | Done when |
|---|---|---|---|
| T1 | Shared s8 helpers | `gqa_attention_decode.cuh`: `mma_m16n8k32_s8`, int8 fragment load helper(s), int8 swizzle | compiles; a standalone smoke MMA matches a hand dot in the parity test's first int8 decode case |
| T2 | Kernel rewrite | `gqa_attention_decode_i8.cuh`: Q int8 quant, s8 QK + per-group rescale, bf16 PV, smem reuse, single-wave pipeline, FFMA softmax; `__launch_bounds__(128,3)`, Wc=4 | builds; `gqa_attention_small_t_tc_partial_i8_kernel` replaced |
| T3 | Launcher smem | `src/kernels/launcher/gqa_attention_decode.cu`: dynamic smem size for the new layout | launch config matches kernel smem |
| T4 | Oracle update | `tests/kernels/test_gqa_attention.cpp`: `cpu_gqa_decode_int8` + rewire `one_int8_decode_case` | int8 decode cases pass at `attention_bf16` tolerance |
| T5 | Verify | build + `ctest -R gqa` + `compute-sanitizer` + bench A/B | see §8 |

Dependencies: T1 → T2 → T3 → T4 → T5 (T4 can be drafted alongside T2).

---

## 7. Deferred — Phase 2 full int8 PV

Absorb the per-key V scale into P, per group: for group g,
`Pg[r,key] = P[r,key]·vs[key,g]`, quantize `Pg` per-row → `Pgi8, ps[r,g]`, then
`O[r,d∈g] += ps[r,g]·(Σ_key Pgi8[r,key]·Vi8[key,d])` via `m16n8k32.s8`. Removes the
bf16 V tile (V stays int8, 8 KB) → ~21 KB peak → ~4 blocks/SM, at the cost of a
per-block P-quant (4 groups). Do it only if Phase-1 profiling shows V-dequant is
still the ceiling.

---

## 8. Verification

- **Correctness (precondition):** `ctest --test-dir build -R gqa_attention` green,
  including all `one_int8_decode_case` shapes (T=1..6, base=0/17/100/2048/2882) at
  `Tolerance::attention_bf16()`; `compute-sanitizer --tool memcheck` clean on the
  int8 decode cases (new smem-reuse lifetime + s8 fragment loads).
- **Performance (primary gate):** A/B via the existing bench —
  ```
  for T in 1 2 3 4; do
    build/bench/qus_gqa_attention_bench --append-small-t --tokens $T \
      --context 4096,8192,16384,32768,65536,131072 --kv-dtype int8
    build/bench/qus_gqa_attention_bench --append-small-t --tokens $T \
      --context 4096,8192,16384,32768,65536,131072 --kv-dtype bf16
  done
  build/bench/qus_gqa_attention_bench --copy-ceiling --tokens 1 --context 131072 --kv-dtype int8
  ```
  Expect int8 to move from slower-than-bf16 to ≥ parity at short context and
  ~1.5–2× faster at long context, trending toward the `--copy-ceiling` int8
  reference. `--profile-once --cold-cache` + `ncu` should show warps/scheduler up
  and instruction count / dequant down.
- **Model accuracy:** greedy/judge A/B (bf16-KV vs int8-KV) within the pre-agreed
  threshold, to gate the int8-Q change.

---

## 9. Files touched

- `src/kernels/kernel/gqa_attention_decode.cuh` (shared s8 helpers).
- `src/kernels/kernel/gqa_attention_decode_i8.cuh` (kernel rewrite).
- `src/kernels/launcher/gqa_attention_decode.cu` (dynamic smem size).
- `tests/kernels/test_gqa_attention.cpp` (int8 decode oracle).

The bf16 decode kernel, prefill kernels, KV format, launcher signature, reducer,
and public API are unchanged.
