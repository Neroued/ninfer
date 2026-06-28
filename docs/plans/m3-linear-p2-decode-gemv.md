# M3 Linear — Phase 2: Optimized Low-bit Decode GEMV (Codex-self-contained)

> Self-contained plan; does not depend on any IDE "skill loader" EXCEPT the two **codex-registered
> profiling skills** named below, which the executor invokes directly. Read this whole file plus:
> the Phase-1 plan [`docs/plans/m3-linear-p1-framework-and-generic.md`](m3-linear-p1-framework-and-generic.md)
> (for the subagent workflow + prompt templates — reused verbatim here),
> [`docs/m3-linear-backend-framework.md`](../m3-linear-backend-framework.md) (§8 registry, §11 GEMV,
> §12.1 Q5/Q6 unpack risk, §16 reference, §17 workspace-free, §18 benchmark contract),
> [`docs/q5090_packed_file_format_v1.md`](../q5090_packed_file_format_v1.md) §6–§7 (bit packing),
> and [`AGENTS.md`](../../AGENTS.md).

**Goal:** Replace the generic correctness GEMV with a tuned, codec-templated **low-bit decode GEMV**
(`T==1`) that serves **all** Q4/Q5/Q6 projections used in decode, driven by profiling evidence. This
targets the measured decode hot path: `lowbit_gemv<Q5Codec>` = 79.4% and `<Q4Codec>` = 18.2% of
decode GPU kernel time (97.6% combined).

**Architecture:** Two independent levers on the existing shared seam — (1) a **branchless, vectorized
unpack** inside `Q5Codec`/`Q6Codec` (the dominant Q5 cost; Q5 is unpack-bound, not bandwidth-bound:
per layer Q5 bytes ≈ Q4 bytes yet Q5 costs 4.4× more per call), and (2) an **optimized GEMV kernel**
(warp-cooperative, vectorized loads, fp32 accumulate, shuffle reduction) wired as a new tuned plan in
the registry. The generic kernel stays as the reference backend; the existing fp64 `qus_linear_test`
oracle is the correctness gate. Every perf claim is gated by the two profiling skills below.

**Tech Stack:** C++20, CUDA 13.x (sm_120, RTX 5090), CMake ≥ 3.28. Build dir `build/`. Branch `master`.

---

## Profiling skills (codex-registered — invoke directly)

This plan **requires** these two skills; do not hand-roll nsys/ncu commands when a skill applies:

- **`nsys-inference-analysis`** — capture/analyze the full **decode** timeline and rank kernels by
  summed GPU duration. Used to (a) reproduce the baseline ranking and (b) prove the GEMV share
  dropped end-to-end after the change, and surface the next hotspot. It is the source of truth for the
  decode-window kernel ranking.
- **`ncu-kernel-profile`** — kernel-level roofline / SpeedOfLight / Occupancy / MemoryWorkloadAnalysis
  / warp-stall metrics on a **named** kernel. Used to (a) confirm the baseline is unpack/issue-bound
  vs DRAM-bound and (b) gate each optimized kernel against the roofline.

Skill usage rules for this project:
- ncu targets the **isolated** `./build/bench/qus_linear_bench` (real shapes, stateless GEMV — safe to
  replay). nsys targets the **full decode run** that produced the baseline table.
- For ncu, name the kernel with `--kernel-name regex:'lowbit_gemv'` (add the codec tag when needed),
  start `--set basic --launch-skip 0 --launch-count 1` to confirm the regex matches, then escalate to
  `--set roofline` / `--section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis`.
- Run the ncu preflight once (`~/.codex/skills/ncu-kernel-profile/scripts/preflight.sh`); if
  `ERR_NVGPUCTRPERM`, follow the skill's fix.
- Commit the skill artifacts (`.nsys-rep` summary tables and `.ncu.txt`/`.ncu.csv`) under `profiles/`
  and the human summary under `docs/bench/`.

---

## Non-goals (Phase 2)

- No GEMM / `T>1` / prefill work (separate later phase). Only `LinearRegime::T1` low-bit GEMV.
- No dense GEMV work (`linear_generic_dense_gemv_kernel`, 0.08% of decode, `[48,5120]`) — stays on the
  dense policy; revisit only if it ever shows up in a profile.
- No q5090 ABI / layout change. Optimize unpack on the **canonical** `TILE_N64_K64`. A derived Q5
  layout (framework §21.2) is the **escalation path only if** a branchless unpack on the canonical
  layout still leaves Q5 unpack-bound per ncu — and it is a separate, ABI-revising phase, out of scope
  here.
- No new correctness tests. Reuse the `qus_linear_test` fp64 oracle; never modify it or its
  tolerances (`AGENTS.md` testing policy).
- No public API / `linear.h` change; no `CMakeLists` edits (globs are recursive; bench/test targets
  exist).
- No Tensor Core (that is GEMM/LargeT territory).

---

## Execution mode (subagent-driven)

Same controller loop and the three subagent prompt templates as the Phase-1 plan
(`docs/plans/m3-linear-p1-framework-and-generic.md`, "Execution mode" + "Subagent prompt templates")
— reuse them verbatim, with these deltas:
- The implementer additionally runs the relevant profiling skill (`ncu-kernel-profile` and/or
  `nsys-inference-analysis`) as named in the task and pastes the key metrics into its report.
- Tasks are sequential **T1 → T2 → T3 → T4** (T2 and T3 touch different files — codec vs new
  `gemv/` — but T3's kernel consumes T2's unpack, so keep order).
- Hard rules: do not modify `tests/kernels/test_linear.cpp` / the frozen test framework / tolerances;
  keep the generic kernels as the reference backend (do not delete); one commit per task on `master`.

## Verification commands

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build -R qus_linear_test --output-on-failure     # fp64 oracle: PASS (every task that touches device code)
compute-sanitizer ./build/tests/qus_linear_test                   # clean
cmake --build build -j --target qus_linear_bench                  # ncu target
# ncu / nsys are invoked via the two skills (see each task)
```

---

## Shared contracts

All code lives in `namespace qus::kernels::detail`. File layout follows the `linear/` subtree
(layering §3.1): new tuned kernel under `src/kernels/linear/gemv/`.

### C1. Branchless vectorized unpack (rewrite `Q5Codec`/`Q6Codec::load_group` in `codec/linear_codec.cuh`)

Keep the **signature and bit-exact result** of the current `load_group`; only replace the per-bit loop
with vectorized 32-bit loads + funnel-shift field extraction. The 5-bit codes are LSB-first packed
across byte boundaries (q5090 §7.1). Packed base is 4-byte aligned (tile is 16-aligned, `+128` is
16-aligned, `rit*40` is 8-aligned) — use `uint32_t` loads (not `uint4`).

```cpp
// Q5: 64 codes × 5 bits = 320 bits = 40 bytes = 10x uint32 per (row, group).
__device__ static void load_group(const std::uint8_t* payload, std::int32_t row,
                                  std::int32_t group, std::int32_t kg, float out[kGroupK]) {
    const std::int32_t tile = row / 64;
    const std::int32_t rit  = row - tile * 64;
    const std::int64_t off  = (static_cast<std::int64_t>(tile) * kg + group) * kTileBytes;
    const std::uint16_t sb  = static_cast<std::uint16_t>(payload[off + rit * 2]) |
                              static_cast<std::uint16_t>(
                                  static_cast<std::uint16_t>(payload[off + rit * 2 + 1]) << 8);
    const float scale = __half2float(__ushort_as_half(sb));
    const std::uint8_t* packed =
        payload + off + 64 * 2 + static_cast<std::int64_t>(rit) * kBytesPerRowPerGroup;

    std::uint32_t w[11];                                  // 10 words + 1 zero guard for funnelshift
    #pragma unroll
    for (int j = 0; j < 10; ++j) {
        w[j] = static_cast<std::uint32_t>(packed[j * 4 + 0]) |
               (static_cast<std::uint32_t>(packed[j * 4 + 1]) << 8) |
               (static_cast<std::uint32_t>(packed[j * 4 + 2]) << 16) |
               (static_cast<std::uint32_t>(packed[j * 4 + 3]) << 24);
    }
    w[10] = 0u;
    #pragma unroll
    for (int lane = 0; lane < kGroupK; ++lane) {
        const int bitpos = lane * kBits;                 // kBits == 5
        const int wi     = bitpos >> 5;
        const int sh     = bitpos & 31;
        const std::uint32_t bits = __funnelshift_r(w[wi], w[wi + 1], sh);
        const int s = static_cast<int>(bits << 27) >> 27;  // mask 5 bits + sign-extend (bit 4)
        out[lane] = static_cast<float>(s) * scale;
    }
}
```

`Q6Codec` is analogous: `kBits == 6`, 48 bytes = 12 `uint32` words `w[0..11]`; use a 13-element array
with `w[12] = 0` as the funnel-shift guard; extract each code with `static_cast<int>(bits << 26) >> 26`
(mask 6 bits, sign bit 5). `Q4Codec` is already a cheap nibble extract; keep
its math but you MAY load its 32 bytes as `uint32 w[8]` and extract nibbles to help vectorization. The
funnel-shift result must be bit-identical to the current per-bit loop — the fp64 oracle proves it.

> If the executor prefers `__byte_perm`/PTX `bfe` over `__funnelshift_r`, that is allowed as long as
> the decoded codes are bit-identical and ncu shows lower issue/stall pressure.

### C2. Tuned GEMV kernel — `src/kernels/linear/gemv/linear_lowbit_gemv.cuh`

Required properties (the *what*, fixed); the exact tiling is profiling-tuned in T4 (the *how*):

- **Codec-templated** `template <class Codec>` so one kernel serves Q4/Q5/Q6 (the launcher switches on
  `LinearFormat`).
- **Warp-cooperative per output row** (starting design): one warp computes one row's dot product; the
  32 lanes split the row's K64 groups (`group = warp_lane, warp_lane+32, …`), each lane calls
  `Codec::load_group` for its groups, multiplies the 64 decoded weights by `x[group*64 + ·]`
  (`__bfloat162float`), accumulates in **fp32**, then a **warp-shuffle reduction** produces the row
  result; lane 0 writes `out[row]` as bf16. This gives enough CTAs for both dominant shapes:
  `mlp.down [5120,17408]` (N=5120 → 5120 warps, long-K split across lanes for occupancy) and
  `mlp.gate/up [17408,5120]` (N=17408 → ample warps).
- **K-tail correct**: skip `kk = group*64 + lane >= k` (logical `k`, not `padded_k`); `kg =
  padded_k/64`. **N-tail correct**: only warps with `row < n` write. Must match the generic result for
  arbitrary `n,k` (the oracle uses non-64-multiple shapes like `[70,130]`).
- **Externally workspace-free** (framework §17): registers + shuffles + optional dynamic SMEM only.
  No global scratch, no second kernel, no atomics across CTAs.
- **Register pressure**: prefer to decode-and-accumulate per group (or in sub-chunks) rather than
  materializing all 64 weights in a `float[64]` per lane; T4's ncu pass must check for register spills
  and adjust. (`Codec::load_group` returning `float[64]` is the simple v1; inline decode+fma if it
  spills.)
- `x` may stay in global (L2 keeps the small `[K]` vector resident across rows); SMEM-staging `x` is an
  optional T4 knob, not required for v1.

The launcher `src/kernels/linear/gemv/linear_lowbit_gemv.{h,cu}` mirrors
`reference/linear_generic_lowbit_gemv.cu`: compute grid/block, `payload = w.payload ? w.payload :
w.qdata`, `padded_k = w.padded_shape[1]`, `switch (fmt)` to instantiate `Q4Codec/Q5Codec/Q6Codec`,
`CUDA_CHECK`. Declare `linear_tuned_lowbit_gemv_launch(const Tensor& x, const Weight& w, Tensor& out,
LinearFormat fmt, cudaStream_t stream)` in a new `src/kernels/linear/gemv/linear_lowbit_gemv.h`.

### C3. Registry wiring (`plan/linear_plan.h`, `plan/linear_plan.cpp`, `linear.cpp`)

Add one policy id and flip the low-bit T1 keys to it; keep generic for parity/fallback.

```cpp
// plan/linear_plan.h  — extend the enum (append; do not renumber)
enum class LinearPolicyId {
    GenericLowbitGemv,
    GenericLowbitGemm,
    GenericDenseGemv,
    GenericDenseGemm,
    TunedLowbitGemv,     // NEW: Phase-2 optimized low-bit decode GEMV (Q4/Q5/Q6, T1)
};
```

```cpp
// plan/linear_plan.cpp  — resolve_plan: low-bit T1 now selects the tuned policy
LinearPlan resolve_plan(LinearPlanKey key) {
    const bool dense = (key.format == LinearFormat::DenseBF16 || key.format == LinearFormat::DenseFP32);
    const bool gemv  = (key.regime == LinearRegime::T1);
    LinearPolicyId policy;
    if (dense) {
        policy = gemv ? LinearPolicyId::GenericDenseGemv : LinearPolicyId::GenericDenseGemm;
    } else {
        policy = gemv ? LinearPolicyId::TunedLowbitGemv   // was GenericLowbitGemv
                      : LinearPolicyId::GenericLowbitGemm;
    }
    return LinearPlan{ gemv && !dense ? LinearBackendKind::Gemv : LinearBackendKind::Reference,
                       policy, policy_name(policy), /*uses_tensor_cores=*/false };
}
// policy_name: add case TunedLowbitGemv -> "linear.gemv.lowbit.tuned.v1"
```

```cpp
// linear.cpp  — add the dispatch case (include "kernels/linear/gemv/linear_lowbit_gemv.h")
case detail::LinearPolicyId::TunedLowbitGemv:
    detail::linear_tuned_lowbit_gemv_launch(x, w, out, fmt, stream);
    break;
```

The generic low-bit GEMV remains compiled and reachable as the reference backend (framework §16); it
is simply no longer the live T1 plan.

---

## Tasks

### Task 1 — Decode GEMV bench coverage + baseline profiling (uses BOTH skills)

**Owns:** `bench/linear_bench.cu` (extend); commits baseline artifacts under `profiles/` + a short
`docs/bench/m3-p2-gemv-baseline.md`.

**Reading list:** this plan's "Profiling skills" + C-section; `bench/linear_bench.cu` (existing q4/q6
helpers + `make_tile_payload`); `bench/qus_bench_common.h`; the two skill files
(`~/.codex/skills/ncu-kernel-profile/SKILL.md`, `~/.codex/skills/nsys-inference-analysis/SKILL.md`).

**Spec:**
- Extend `linear_bench.cu` so every decode low-bit GEMV shape is benchmarkable in isolation: add a Q5
  payload helper (`make_tile_payload(n,k,2688)` + a `q5_weight`) and `--decode` runs for the real
  shapes — Q4: `mlp.gate/up [17408,5120]`, `gdn q/k [2048,5120]`, `attn q [6144,5120]`; Q5:
  `mlp.down [5120,17408]`, `v/z [6144,5120]`, `out [5120,6144]`, `attn gate [6144,5120]`; Q6:
  `lm_head [248320,5120]` (exists). Keep the existing `--q4/--q6` flags; add `--q5`.
- Use **`nsys-inference-analysis`** on the decode run that produced the baseline table to capture and
  commit the baseline kernel ranking (confirm `lowbit_gemv<Q5Codec>` ≈79%, `<Q4Codec>` ≈18%) plus the
  decode capture span.
- Use **`ncu-kernel-profile`** (`--set roofline` + `--section SpeedOfLight --section Occupancy
  --section MemoryWorkloadAnalysis`) on the **current generic** kernel via `qus_linear_bench` at the
  dominant Q5 shape (`mlp.down [5120,17408]`) and Q4 shape (`mlp.gate/up [17408,5120]`). Record DRAM%,
  SM/issue utilization, achieved occupancy, and top stall reasons. **Expected finding** to confirm:
  Q5 is issue/ALU-bound (low DRAM%, high integer/issue, stalls on the unpack), Q4 closer to bandwidth.
  These numbers are the **gates** for T2–T4.

**DoD / verify:** `qus_linear_bench` builds and runs all shapes; baseline nsys ranking + ncu roofline
committed; `docs/bench/m3-p2-gemv-baseline.md` states the per-shape DRAM%/occupancy/bound-type.

**Commit:** `perf(linear): extend decode GEMV bench and record baseline profiles`

---

### Task 2 — Branchless vectorized Q5/Q6 (and Q4) codec unpack

**Owns:** `src/kernels/linear/codec/linear_codec.cuh`.

**Reading list:** C1; current `codec/linear_codec.cuh`; `docs/q5090_packed_file_format_v1.md` §7.1;
`reference/linear_generic_lowbit.cuh` (caller); the T1 baseline numbers.

**Spec:** Rewrite `Q5Codec::load_group` and `Q6Codec::load_group` per C1 (vectorized `uint32` loads +
`__funnelshift_r` extraction, bit-identical). Optionally vectorize `Q4Codec` loads. Do not change
signatures, traits, or the tile math. This is the single biggest Q5 lever and benefits both the
generic and (later) tuned kernels.

**DoD / verify:**
- `ctest -R qus_linear_test` PASS + `compute-sanitizer` clean (bit-exact decode — the fp64 oracle is
  the proof).
- **`ncu-kernel-profile`** on the still-generic `lowbit_gemv<Q5Codec>` (via `qus_linear_bench`,
  `mlp.down`/`[6144,5120]`): issue/integer pressure and unpack-related stalls drop materially vs the
  T1 baseline; report the delta. (Bandwidth may still be low because the generic skeleton is scalar —
  that is fixed in T3.)

**Commit:** `perf(linear): branchless vectorized q5/q6 codec unpack`

---

### Task 3 — Tuned low-bit decode GEMV kernel + registry wiring

**Owns / creates:**
- `src/kernels/linear/gemv/linear_lowbit_gemv.cuh` (C2 kernel)
- `src/kernels/linear/gemv/linear_lowbit_gemv.h` (launch prototype)
- `src/kernels/linear/gemv/linear_lowbit_gemv.cu` (launcher)
**Owns / edits:** `plan/linear_plan.h`, `plan/linear_plan.cpp`, `linear.cpp` (C3 wiring).

**Reading list:** C2 + C3; `reference/linear_generic_lowbit_gemv.cu` (launcher pattern to mirror);
`reference/linear_generic_lowbit.cuh` (semantics to match); `linear.cpp` dispatch switch;
`plan/linear_plan.cpp` `resolve_plan`/`policy_name`; framework §11.1–§11.2, §17.

**Spec:** Implement the warp-cooperative codec-templated kernel (C2), its launcher, and the C3 registry
wiring so low-bit `T==1` routes to `TunedLowbitGemv` for Q4/Q5/Q6. Keep the generic kernel as
reference. Externally workspace-free.

**DoD / verify:**
- `ctest -R qus_linear_test` PASS (T1 path now exercises the tuned kernel for all low-bit shapes incl.
  the odd `[70,130]`/`[128,128]` tail shapes) + `compute-sanitizer` clean.
- **`ncu-kernel-profile`** (`--set roofline`, SoL/Occupancy/Memory) on `linear_tuned_lowbit_gemv`
  via `qus_linear_bench` at `mlp.down [5120,17408]`, `mlp.gate/up [17408,5120]`, `[6144,5120]`: report
  DRAM%, occupancy, stalls, and the speedup vs the T1 baseline durations. Gate: a clear improvement
  toward the memory roofline on Q5 (now that unpack is cheap) and no regression on Q4.

**Commit:** `perf(linear): tuned low-bit decode GEMV kernel and registry route`

---

### Task 4 — Per-shape tuning + end-to-end validation (ncu sweep + nsys)

**Owns:** `src/kernels/linear/gemv/*` (tuning); optional `plan/linear_kernel_policy.h` (only if needed);
commits `profiles/` artifacts + `docs/bench/m3-p2-gemv-after.md`.

**Reading list:** Task 3 output + its ncu report; framework §18 (benchmark contract), §11.2 (K-split for
occupancy); both skill files.

**Spec:**
- Use **`ncu-kernel-profile`** to sweep a small set of kernel-shape knobs on the dominant shapes —
  warps/CTA, threads, K-split factor, vector width, optional `x`-in-SMEM — especially checking whether
  `mlp.down` (long-K, moderate-N → occupancy/K-split sensitive) and `mlp.gate/up` (large-N) want
  different configs. Introduce a constexpr `LinearKernelPolicy` (the `GemmExtraOptions` analog) and
  bind a per-format/per-shape winner **only if** measurement shows divergence; otherwise keep one
  config (YAGNI — do not add the knob speculatively).
- Use **`nsys-inference-analysis`** on the same decode run as the T1 baseline; confirm the
  `lowbit_gemv` share of decode dropped substantially and capture the **new ranking + next hotspot**.
  Commit a before/after summary in `docs/bench/m3-p2-gemv-after.md` (per-shape ncu DRAM%/occupancy +
  the nsys decode delta).

**DoD / verify:** `qus_linear_test` PASS + sanitizer clean; ncu roofline recorded for the dominant
shapes; nsys shows the decode GEMV time materially reduced vs baseline; after-report committed.

**Commit:** `perf(linear): tune decode GEMV per shape and verify end-to-end`

---

## Definition of done (phase)

- All low-bit `T==1` GEMV (Q4/Q5/Q6) route to the tuned kernel; generic remains the reference backend.
- Q5/Q6 unpack is branchless/vectorized (bit-exact, oracle-proven); the tuned kernel is
  warp-cooperative, fp32-accumulate, externally workspace-free, tail-correct.
- `qus_linear_test` PASS + `compute-sanitizer` clean; no public API / ABI / CMake / test-framework
  changes.
- Evidence committed: baseline (T1) and after (T4) nsys decode rankings + ncu roofline for the
  dominant Q5/Q4 shapes, showing the GEMV decode share materially reduced.

## Review phase (risk-scaled per AGENTS.md)

CUDA kernels + numerical decode → every task uses the two-reviewer pattern (spec-compliance, then
code-quality). Spec review must check decode bit-exactness reasoning (C1) and tail correctness (C2)
against the generic kernel + q5090 §7.1; the merge gate is the fp64 oracle + `compute-sanitizer` +
the ncu/nsys evidence from the named skills.

## Self-review notes (author)

- ROI-ordered: T2 attacks the 79% Q5 unpack first (biggest lever, helps even the generic kernel),
  T3 adds the bandwidth/occupancy skeleton, T4 tunes + proves end-to-end. Covers **all** low-bit
  decode GEMV via one codec-templated kernel.
- Both required skills are wired to concrete gates: `nsys-inference-analysis` for the decode ranking
  (T1 baseline, T4 after), `ncu-kernel-profile` for per-kernel roofline (T1–T4). ncu→isolated bench
  (stateless GEMV), nsys→full decode run.
- Profiling-first and evidence-committed per the project's perf discipline; no new tests (oracle
  reused), no ABI change; derived-layout escalation explicitly deferred to a separate phase.
- One coordination point: T3 edits `plan/*` + `linear.cpp` to add `TunedLowbitGemv`; it is the only
  task that touches the registry, so no shared-file contention.
