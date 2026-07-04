# GQA decode / small-T bandwidth redesign - design spec (2026-07-05)

Status: approved design, pending implementation plan.
Owner op: `qus::kernels::gqa_attention` (unified GQA attention with folded KV append).
Depends on: `feat(gqa): unify attention api and add small-t route` (`8890da6`) and
`docs/bench/gqa-unified-smallt-2026-07-04.md`.

## 1. Problem and goal

The unified `gqa_attention` refactor routes `T` by host-known token count:

- `T=1` -> decode-derived **scalar** split-K partial kernel
  (`gqa_attention_small_t_partial_kernel<6,1>`), synchronous global loads.
- `T=2..6` -> **tensor-core** split-K partial kernel
  (`gqa_attention_small_t_tc_partial_kernel`), `cp.async` staged K/V + MMA.
- `T>6` -> prompt-scale flash kernel.

Benchmarking on RTX 5090 (measured live, `./build/bench/qus_gqa_attention_bench`) shows the T=1 path
collapses at long context while the T=2..6 path is healthy, reading the *same* bytes:

| shape | route | median | useful-KV GB/s | % of 1792 GB/s peak |
|---|---|---:|---:|---:|
| T=1, ctx=32768 | scalar decode | 1002 us | 134 | 7.7% |
| T=2, ctx=32768 | tensor-core | 110 us | 1213 | 68% |
| T=6, ctx=32768 | tensor-core | 116 us | 1152 | 64% |
| T=6, ctx=32768 | prompt baseline (replaced) | 3111 us | 43 | 2.4% |

Both T=1 and T=2 move 128 MiB of KV; T=1 is ~9x slower. Round-robin-16 layers (defeats L2) confirms
T=1 stays ~134 GB/s, so this is a genuine DRAM-path collapse, not an L2 artifact.

Root cause: memory pipeline, not math. The scalar path issues synchronous per-key
`gqa_cached_pair_lane_value` loads with a full warp reduction on the load->reduce->FMA critical path,
so each warp keeps only ~1-2 loads in flight and cannot generate the memory-level parallelism DRAM
needs. The tensor-core path uses `cp.async` bulk tile loads (many outstanding requests) and saturates
~68%. The scalar T=1 read path is byte-identical to the pre-refactor decode kernel (`0e79ee3`), so
this is a **pre-existing** inefficiency the refactor exposed, not a regression.

T=1 decode is the dominant attention op during generation (every full-attention layer, every step,
plus the MTP AR draft step). At >=32k context ~1 ms per attention op is a severe hot-path bottleneck.

### Goal

Bring T=1 (and, if it wins, all of T=1..6) to a `cp.async`-fed, memory-bound kernel that approaches
the useful-KV DRAM roofline, without regressing correctness, graph-compatibility, the T=2..6 win, or
the public API.

### Non-goals

- No change to the public `gqa_attention` API (already unified on device `positions`).
- No change to KV cache layout (`[256, padded_context, 4]` BF16) or the workspace/partial layout.
- No change to the routing contract: the route may key only on host-known `T` and static capacity,
  never on device-resident context length.
- No new model models, generic runtime flexibility, or speculative genericity (per `AGENTS.md`).

## 2. Performance targets

Peak DRAM reference: `kDramPeakGBs = 1792.0` (bench constant). The acceptance ceiling is a measured
device copy-kernel baseline (`out=in` streaming the same K+V bytes), per
`docs/l1-op-test-standard.md` 2.3. Call the measured copy ceiling `C_copy` GB/s (to be recorded in
Phase 2, Task T1).

Numeric targets, all at the real GQA shape (head_dim=256, 24 q-heads, 4 kv-heads):

1. **T=1 primary (must-hit).** ctx=32768 useful-KV `>= 1150 GB/s` (median `<= ~115 us`), i.e.
   `>= 8.5x` faster than today's 1002 us. This only requires matching the already-proven tensor-core
   T=2 route.
2. **Family roofline (stretch / real gate).** For `T in [1,6]` at `ctx >= 16384`, either
   `ncu dram__throughput.avg.pct_of_peak_sustained_elapsed >= 85%` **or** useful-KV `>= 0.85 * C_copy`,
   whichever the reviewer accepts as the honest ceiling. If the kernel stops being DRAM-bound before
   the gate, document the new limiter with an `ncu` `.ncu-rep` and keep tuning until useful-KV is
   saturated or a separate architecture round is justified.
3. **Redundancy gate.** `total_dram / useful_kv <= 1.10` for all `T` and all contexts (single-KV-pass;
   each key/value slice read once).
4. **Scratch gate.** partial/reduce scratch bytes `<=` useful-KV bytes at every measured point.
5. **No regression.** T=2..6 at ctx=32768 no worse than the current tensor-core route within noise
   (T=2 ~110 us, T=6 ~116 us). T=1 short context (`pos <= 2048`) no worse than today (~50 us).
6. **End-to-end (the payoff).** A single-token decode step at ctx `>= 32768` must show the
   full-attention-layer attention wall-clock drop by roughly the same factor as the op bench (target:
   from ~1002 us/op toward ~115 us/op). Report before/after.

## 3. Phase 1 - spike: measure the ceiling (throwaway)

Purpose: cheaply establish the empirical "`cp.async` alone" ceiling for T=1 and confirm the memory
pipeline is the lever, before investing in the new kernel. Does **not** ship.

Steps:

1. Relax the tensor-core partial guard `tokens >= 2` -> `tokens >= 1`, and in
   `gqa_attention_small_t_launch` route `case 1` to `launch_tc_partial<1, WarpsPerCta>` (pick warps so
   `row_count = 1*6 = 6 <= Br`; `WarpsPerCta = 4` keeps `Br=64`). Padding rows (6..63) are already
   masked by the existing `row < row_count` guards.
2. Keep the scalar kernel in place; do not delete anything.
3. Measure: `qus_gqa_attention_test` T=1 cases pass; `compute-sanitizer memcheck` clean; bench T=1
   over ctx `{2048, 8192, 16384, 32768}` recording useful-KV GB/s and redundancy.

Decision gate: record the T=1 tensor-core ceiling (expected ~1000-1200 GB/s vs 134). If `cp.async`
does not move T=1 substantially, stop and re-analyze before Phase 2. After recording, revert the
spike (the shipped design is Phase 2).

## 4. Phase 2 - unified `cp.async` CUDA-core token-tile kernel (deliverable)

A single memory-bound partial kernel for `T=1..6` that reads each K/V slice once via `cp.async` and
reuses it across all 6 q-heads in the group and all `T` query tokens, with low register pressure so
occupancy (hence MLP) is high enough to saturate DRAM. This realizes the plan's original
"candidate B" (CUDA-core token-tile), which was never built or measured.

### 4.1 Kernel identity

New kernel: `gqa_attention_small_t_stream_partial_kernel<int QHeadsPerCta, int TokenTile>` in
`src/kernels/kernel/gqa_attention_decode.cuh`.

- `QHeadsPerCta == 6` (one kv-head group per CTA; `q_subgroups = 1`).
- `TokenTile == T` (single token tile per launch; `1 <= T <= 6`). Instantiated at `TokenTile in
  {1,2,3,4,5,6}` and dispatched by the launcher `switch (q.ne[2])`.

Reused unchanged: the reducer `gqa_attention_small_t_reduce_output_kernel<DChunk>`, the partial-buffer
layout (`partial_acc [256,24,T,192]`, `partial_m/l [24,T,192]`; T=1 aliases decode), the adaptive
split policy `gqa_small_t_active_splits`, index helpers, and neutral-partial writers.

### 4.2 Launch geometry

- Grid: `dim3(kGqaKVHeads /*=4*/, kGqaDecodeSplits /*=192*/, 1)`. Only `split < active_split_count`
  do work; the rest early-return (same pattern as today).
- Block: `32 * QHeadsPerCta = 192` threads (6 warps). Warp `w` owns query head
  `q_head = kv_head*6 + w`. Lane holds dims `d = lane + 32*k`, `k in [0,8)` (8 dims/lane, head_dim=256).
- Split count: `T=1` uses all 192 splits (aliases decode). `T>1` uses `gqa_small_t_active_splits`
  (`ceil(window/480|512)`, clamped `[4,192]`) exactly as today; partial and reducer must agree
  (both derive it from `pos[T-1]`).

### 4.3 Shared-memory staging and `cp.async` pipeline

Stage K and V key tiles into a shared-memory ring using the existing helpers in
`src/kernels/kernel/gdn_common.cuh`: `async_copy_global_to_shared<16>`, `async_copy_commit`,
`async_copy_wait<N>`.

- Tile width `Bc` keys x 256 dims; ring depth `S` stages. Per-stage bytes
  `= Bc * 256 * 2 (K) + Bc * 256 * 2 (V)`. Total dynamic smem `= S * Bc * 1024` bytes.
  - Default start: `Bc = 32`, `S = 2` -> 64 KiB. Tune (Task T3) over `Bc in {16,32,64}`,
    `S in {2,3,4}` under `ncu`, maximizing `dram__throughput` subject to correctness and occupancy.
    Use `cudaFuncAttributeMaxDynamicSharedMemorySize` if a chosen config exceeds the 48 KiB static
    default; keep enough occupancy for MLP (aim >= 2 resident CTAs/SM if the sweep allows).
- Coalesced load: 192 threads copy the `[Bc x 256]` tile as `int4` (8 bf16) chunks; addresses are
  16-byte aligned because `d` steps by 8 and head_dim rows are 512-byte aligned.
- Software pipeline (multistage):
  - Prologue: issue async loads for stages `0 .. S-2`, `async_copy_commit()` per stage.
  - Steady state over key-blocks `kb = 0 .. key_blocks-1`:
    `async_copy_wait<S-1>()` (wait for the oldest in-flight stage) -> `__syncthreads()` ->
    compute on stage `kb % S` -> if `kb + S - 1 < key_blocks` issue async load for the next stage and
    `async_copy_commit()`.
  - Out-of-range key lanes within the last partial tile store zeros to smem (masked out in compute).

### 4.4 New-row source, cache write, causal mask (identical semantics to today)

- Cache write (append): only the split that **owns** each new position writes it, and only warp 0 /
  lane group 0, using `int4` stores; positions come from `pos[token]`. Reuse the current guard
  (`split_start <= p_tok < split_end`). This preserves graph-safe append and the future-token
  isolation property.
- New-row source: a key at absolute position `key` with `new_token = key - pos[0] in [0,T)` and
  `key >= pos[0]` is read from the `k_new/v_new` inputs (avoids the RAW hazard across CTAs); otherwise
  from cache. Same rule the tensor-core kernel uses.
- Causal mask: for query token `tt` (absolute `pos[tt]`), mask keys with `key > pos[tt]`; for new
  rows additionally `new_token > tt` (redundant under contiguous positions, kept as a guard). T>1
  contiguity of `positions` is the caller's device contract.

### 4.5 Inner compute (dims-per-lane, single-KV-pass)

Per warp (one q-head), maintain online-softmax state for all `T` tokens simultaneously so each staged
key is applied to every token in one pass:

```text
registers (per lane):
  q_d[T][8]     # this warp's q-head, T tokens, 8 dims/lane   (default in regs; may stage in smem)
  acc[T][8]     # split-local weighted-V accumulator          (mandatory, single-pass)
  m[T], l[T]    # running max / denom                         (mandatory)
for each staged key `key` in [split_start, split_end):
  load k_d[8], v_d[8] from the smem K/V tile (this key's row)
  for tt in 0..T-1:
    if masked(key, tt): continue
    dot   = warp_reduce_sum( sum_k q_d[tt][k] * k_d[k] )      # 256-dim reduction across 32 lanes
    score = dot * scale
    online-softmax update of m[tt], l[tt], acc[tt][*] with v_d
epilogue: write partial_m/l (lane 0) and partial_acc[q_head,d,tt,split] (int4 stores)
```

- The per-key warp reduction stays (inherent to the dims-per-lane PV layout at head_dim=256; the
  tensor-core route only hides it inside MMA). It is cheap (~5 `shfl`) and overlaps the next tile's
  async load; at memory-bound occupancy it should hide. If `ncu` shows it on the critical path,
  batch-reduce multiple keys before the softmax merge (still dims-per-lane).
- Register budget: T=1 ~40 regs/thread (very high occupancy -> should exceed the tensor-core route's
  68%); T=6 ~130-145 regs (better than the tensor-core kernel's 254). If T=6 occupancy caps below the
  gate, stage `q_d` in smem instead of registers (frees ~48 regs) - a Task T3 tuning lever.

Note on the rejected `lane-per-key` layout: assigning distinct keys to lanes removes the per-key
reduction but forces a full 256-dim O accumulator per lane (256 regs), which is infeasible at
head_dim=256. Do not pursue it; dims-per-lane is the design.

### 4.6 Routing and tensor-core retirement

- `gqa_attention_uses_small_t` stays `T in [1,6]`. The internal per-`T` route now targets the stream
  kernel for the `T` values where it wins.
- After the (T x context) sweep (Task T3/T4): if the stream kernel meets or beats the tensor-core
  kernel for **every** measured `(T, context)` cell, delete `gqa_attention_small_t_tc_partial_kernel`,
  its launcher path, and the TC helpers (no back-compat, per `AGENTS.md`). If the tensor-core kernel
  still wins some cell, keep a measured per-`T` split and record it in the evidence doc. Either way the
  route table must be backed by `ncu` evidence, not assumption.

## 5. Verification and gates

Per `docs/l1-op-test-standard.md` and `AGENTS.md` testing policy (no low-value tests):

- Correctness: `qus_gqa_attention_test` (fp64 oracle: prefill T=1..6 x bases, decode pos sweep incl.
  32768, stress, future-token isolation, graph relaunch, validation) passes with `attention_bf16`
  tolerance. No new tests unless a new real risk surfaces.
- Memory safety: `compute-sanitizer --tool memcheck` and `--tool racecheck` both clean (the `cp.async`
  pipeline + shared-memory reuse are the risk; racecheck is mandatory).
- Perf: bench T x context sweep (`--append-small-t` and `--decode`), reporting useful-KV GB/s,
  redundancy, scratch, splits. `ncu --set roofline` `.ncu-rep` saved under `profiles/` for the T=1 and
  T=6 partial kernels at ctx=32768; record `dram__throughput.avg.pct_of_peak_sustained_elapsed` and
  the copy-kernel ceiling `C_copy`.
- End-to-end: measure a full-attention-layer decode-step attention wall-clock at ctx >= 32768 before
  and after (model runner or a targeted harness), confirming the op-level speedup shows up in the hot
  path.

## 6. Execution mode (direct, sequential)

Execution is direct and sequential: a single author works the tasks below in order, in one session,
verifying each before moving on. (`AGENTS.md` requires naming the mode when a plan opts out of its
subagent default; the reason here is that the work is one coherent CUDA kernel effort on a few
tightly-coupled files with a strict profile->change->re-profile loop, best held in one author's
context.)

The tasks below are ordered work units, done one at a time and verified before moving on. The shared
files (`gqa_attention_decode.cuh`, `gqa_attention_decode.cu`, `gqa_attention_bench.cu`, the evidence
doc) are edited sequentially, never concurrently.

- **T0 - spike (Phase 1).** Touches: `gqa_attention_decode.cu` (launcher), `gqa_attention_decode.cuh`
  (guard), `docs/bench/...`. DoD: T=1 tensor-core ceiling recorded over the ctx sweep; test + memcheck
  pass; spike reverted. Reading list: this spec S3; launcher; tensor-core kernel guards.
- **T1 - copy-kernel ceiling.** Touches: `gqa_attention_bench.cu`. Adds an `out=in` streaming baseline
  for the 128 MiB K+V volume and prints `C_copy`. DoD: `C_copy` recorded (hot and cold). Reading list:
  bench byte model, l1-op-test-standard 2.3.
- **T2 - stream kernel.** Touches: `gqa_attention_decode.cuh` (new kernel), `gqa_attention_decode.cu`
  (dispatch T=1..6 to the stream kernel; tensor-core kept temporarily). DoD: correctness test +
  memcheck + racecheck clean; bench shows T=1 ctx=32768 >= 1150 GB/s. Reading list: this spec S4;
  current scalar + tensor-core kernels; `gdn_common.cuh` async helpers.
- **T3 - tune.** Touches: kernel tunable constants (`Bc`, `S`, q_d-in-smem) + bench/evidence doc.
  `ncu`-driven per S2.4 loop (profile -> one change -> re-profile). DoD: gate met (S2.2) or limiter
  documented with `.ncu-rep`. Reading list: l1-op-test-standard 2.2-2.4.
- **T4 - route + retire.** Touches: `gqa_attention_decode.cu` (route), `gqa_attention_decode.cuh`
  (delete tensor-core kernel if dominated), `docs/bench/...` (route table + evidence). DoD: per-`T`
  route backed by (T x ctx) `ncu` evidence; no-regression bars met.
- **T5 - end-to-end + review.** Touches: e2e wall-clock harness/report. DoD: hot-path speedup
  confirmed, then a strict review (CUDA kernel, numerics, GPU memory lifetime, route table) per
  `AGENTS.md` risk scaling, using the oracle test, `compute-sanitizer`, and `ncu` evidence as the
  review inputs.

Sequencing: T0 -> T1 -> T2 -> T3 -> T4 -> T5, done in order. T0 and T1 are mutually independent (either
first), but neither is worked concurrently with the others.

## 7. Risks

- `cp.async` pipeline correctness (stage indexing, wait depth, last-tile masking) -> gated by
  racecheck + oracle tests.
- Occupancy vs smem: aggressive `Bc/S` can drop below 2 CTAs/SM and lose MLP -> `ncu` sweep in T3.
- T=6 register pressure -> `q_d`-in-smem fallback.
- Retiring the tensor-core kernel prematurely -> only after a full per-`(T,ctx)` comparison.

## 8. Open choices (resolved by measurement, not up front)

- `Bc` (16/32/64), pipeline depth `S` (2/3/4), and `q_d` in registers vs smem.
- Whether `T>1` split-count policy should change for the stream kernel (currently reuse
  `gqa_small_t_active_splits`).
- Whether the tensor-core kernel survives for any `(T, context)` cell.
