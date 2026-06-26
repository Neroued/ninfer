# L1 Tier-2 GDN Port — Implementation Plan (Codex-self-contained)

> Self-contained for Codex. The full design + decisions are in
> [`docs/l1-gdn-port-design.md`](../l1-gdn-port-design.md) — **read it fully first**. This plan reuses
> the execution machinery (subagent workflow, prompt templates, ncu procedure) already written in
> [`docs/plans/l1-tier1-simple-ops.md`](l1-tier1-simple-ops.md) — read its "How to execute", "Subagent
> prompt templates", and "ncu procedure" sections; they apply here unchanged.

**Goal:** Port the chunked (prefill) + AR (decode) Gated-DeltaNet kernels from `~/chunked_gdn` into the
`gated_delta_rule_recurrent` / `gated_delta_rule_chunked` ops, localized to qus conventions and the
frozen test/bench framework.

**Architecture:** Boundary-cast bf16↔fp32 (design §3.1); **grouped** head mapping `qk_head = h_v / G`,
`cta_h_v = identity` (design §3.2, confirmed via vLLM); `B=1`, `kda=false`, `chunk_size=64`; chunked
tail routed through AR. Four-layer L1 layout per `l1-kernel-layering.md`; correctness and performance
are separate tasks.

**Tech Stack:** C++20, CUDA 13.1 (sm_120), CMake ≥ 3.28, gcc 13.3, build dir `build/`.

---

## Hard rules (read before any task)
- **Frozen framework is read-only:** do not modify `tests/kernels/op_check.h` / `op_tester.h` /
  `bench/qus_bench_common.h` tolerance *logic*. You MAY *add* new named presets (this plan adds two);
  you may not weaken existing ones. If a test fails, fix the kernel.
- **No math approximation** to gain speed (design §0); the chunked TF32 path is fine (it is the source's
  proven math) but do not introduce new lossy shortcuts.
- **FORMAT (required):** before every commit, run `clang-format` (the repo `.clang-format`) on every
  new/changed `*.h` / `*.cuh` / `*.cpp` / `*.cu` file, and verify it is clean:
  ```bash
  clang-format -i <files...>
  clang-format --dry-run --Werror <files...>     # must exit 0 (no reformatting needed)
  ```
  The ported `~/chunked_gdn` code does NOT match house style as-is; reformatting it is part of the
  localization. A task is not done until format is clean.
- Work on `master`, one commit per task, with the commit message given in the task.

## Decisions carried from the design (do not re-derive)
- dtype: keep kernels fp32 internally; cast bf16 `q/k/v` → fp32 scratch (from `WorkspaceArena`), run,
  cast fp32 `attn_out` → bf16 `out`. `g/beta/ssm_state` are fp32 → pass through.
- head mapping: GROUPED. In the ported `head_map`: `qk_head(h_v) = h_v / G` (use
  `init_fastdiv_values(G)`, fastdivide — NOT the source's fastmodulo by `H_qk`); `cta_h_v(cta_h) = cta_h`.
- shapes (`ne`-order): `q,k=[128,16,T]`, `v=[128,48,T]` bf16; `g,beta=[48,T]` fp32; `ssm_state=
  [128,128,48]` fp32 (AR-transposed, in-out); `out=[128,48,T]` bf16; `scale=1/√128`. Layouts already
  match the source byte-for-byte (design §2).
- `chunk_size==64` only (assert). KDA dropped.

## New test presets (add to `op_check.h`, then it is frozen again)
```cpp
static constexpr Tolerance gdn_output_bf16() { return {1e-3, 1.0e-2, 2e-3, 5.0, 8e-3}; }  // bf16 attn out
static constexpr Tolerance gdn_state_fp32()  { return {5e-4, 5.0e-3, 2e-2, 5.0, 5e-3}; }  // fp32 state (chunked_gdn cross_codepath)
```
Adding these for a new op class is allowed (design §5); existing presets are untouched.

## ncu / perf note (different from Tier-1)
GDN's chunked path is matmul-heavy (TF32), **not** pure DRAM-bound — do NOT apply the 85%-DRAM gate.
The perf gate is **no regression vs the source**: the ported kernel's median µs must be within a small
margin of `~/chunked_gdn`'s `bench_chunked` / `bench_ar` at the same shape, plus a recorded `ncu`
profile (procedure: `docs/plans/l1-tier1-simple-ops.md` "ncu procedure" + `~/.cursor/skills/profile-cuda/`).

---

## Tasks

### Task gdn-1 — head_map + device helpers (foundational)
- **Reading list:** design §3.2/§3.3; `~/chunked_gdn/include/gdn_common.h`, `~/chunked_gdn/include/cuda_utils.cuh`; `l1-kernel-layering.md`.
- **Files:** `src/kernels/kernel/gdn_common.cuh` (ported `head_map` with GROUPED `qk_head`/identity
  `cta_h_v` + the needed `cuda_utils` device helpers: fastdiv/fastdivide, warp-shuffle, TF32 MMA
  helpers), namespaced `qus::kernels`. Host-only unit test `tests/kernels/test_gdn_common.cpp`
  (register in `tests/CMakeLists.txt`) asserting `qk_head(h_v)==h_v/G` for `H_qk=16,H_v=48` and
  `cta_h_v` identity.
- **DoD:** test PASS; `clang-format` clean; commit `feat(kernels): GDN head_map + device helpers (grouped mapping)`.

### Task gdn-2 — gated_delta_rule_recurrent (correctness)
- **Reading list:** design §1–§5; `~/chunked_gdn/ar/ar_gdn.{cuh,cu}`, `reference/cpu_ref.h`,
  `tests/test_utils.h`; `l1-operator-catalog.md` §3.7; `op_check.h`/`op_tester.h`;
  `src/kernels/*/silu_and_mul.*` (4-layer template).
- **Files:** `include/qus/kernels/gated_delta_rule.h` (both api decls); `src/kernels/wrapper/gated_delta_rule.cpp`
  (validate + recurrent dispatch + bf16↔fp32 cast scratch from `ws`); `src/kernels/launcher/gated_delta_rule.h`
  + `gated_delta_rule_recurrent.cu` (ported AR launch, grouped head_map, `scale` param); `src/kernels/kernel/
  gated_delta_rule_recurrent.cuh` (ported AR kernel) + `gdn_cast.cuh` (bf16↔fp32). Test:
  `tests/kernels/test_gated_delta_rule.cpp` + `tests/kernels/gdn_ref.h` (port `cpu_ref::gdn_forward` as an
  **fp64** golden writing `double` out+state). Add the two presets to `op_check.h`.
- **CPU ref / inputs:** AR recurrence in fp64 (architecture §7.1); honest seeded inputs ported from
  `test_utils::make_inputs` (q/k L2-normed, `g∈[-4,0]`, `beta∈[0.05,0.95]`, `state∈[-0.1,0.1]`) + the
  `g∈[-1,-0.05]` stress; `kda=false`; round q/k/v to bf16 before the golden.
- **Shapes:** decode `T=1` (the real case) + `T∈{2,7,64}` for the recurrence; `S=128,H_qk=16,H_v=48`.
- **DoD:** recurrent kernel vs fp64 golden passes (`gdn_output_bf16` for `out`, `gdn_state_fp32` for the
  updated state); `compute-sanitizer` clean; `clang-format` clean; commit
  `feat(kernels): gated_delta_rule_recurrent (GDN decode, correctness)`.

### Task gdn-3 — gated_delta_rule_recurrent (performance)
- **Reading list:** design §6; `bench/qus_bench_common.h`; Tier-1 plan ncu procedure; `~/chunked_gdn/bench/bench_ar.cpp`.
- **Files:** `bench/gated_delta_rule_bench.cu` (+ `qus_add_bench`), decode shape `[S128,Hqk16,Hv48,L1]`.
- **DoD:** profile with ncu; median µs within a small margin of `~/chunked_gdn` `bench_ar` at the same
  shape (no regression); record the `.ncu-rep`; recurrent test still PASS; `clang-format` clean; commit
  `perf(kernels): gated_delta_rule_recurrent`.

### Task gdn-4 — gated_delta_rule_chunked (correctness)
- **Reading list:** design §1–§5, §3.4 (tail split); `~/chunked_gdn/chunked/{chunked_gdn.cuh,stage_*.cu,chunked_internals.cuh}`,
  `reference/cpu_chunked_ref.h`; the recurrent files from gdn-2.
- **Files:** `src/kernels/launcher/gated_delta_rule_chunked.cu` (ported 3-stage pipeline: prepare_wy_wu →
  state_passing → chunk_output; workspace bump-allocated from `ws` via the ported `workspace_bytes`; bf16↔fp32
  casts; **tail split**: full 64-chunks via chunked, then `T%64` tail via the gdn-2 AR kernel using the
  chunked end-state); `src/kernels/kernel/{gdn_prepare_wy_wu,gdn_state_passing,gdn_chunk_output}.cuh`
  (ported stages, grouped head_map); chunked dispatch in `gated_delta_rule.cpp`. Extend
  `tests/kernels/gdn_ref.h` with `cpu_chunked_ref::gdn_forward_chunked` (fp64) for the cross-check.
- **Tests (extend test_gated_delta_rule.cpp):** chunked vs fp64 golden; chunked vs the recurrent kernel
  (cross-codepath); **chain-equivalence** (chunked over `T` == AR stepped `T` times); shapes
  `T∈{64,128,256,4096}` (multiples of 64) + a non-multiple `T=200` (tail path). `compute-sanitizer` clean.
- **DoD:** all chunked tests pass (`gdn_output_bf16`/`gdn_state_fp32`); sanitizer clean; `clang-format`
  clean; commit `feat(kernels): gated_delta_rule_chunked (GDN prefill, correctness)`.

### Task gdn-5 — gated_delta_rule_chunked (performance)
- **Reading list:** design §6; Tier-1 ncu procedure; `~/chunked_gdn/bench/bench_chunked.cpp`.
- **Files:** extend `bench/gated_delta_rule_bench.cu` with the prefill chunked shape `[…,L4096]` (+ `--sweep`).
- **DoD:** profile with ncu; median µs within a small margin of `~/chunked_gdn` `bench_chunked` at the
  same shape (no regression); record the `.ncu-rep`; chunked test still PASS; `clang-format` clean; commit
  `perf(kernels): gated_delta_rule_chunked`.

---

## Done criteria
`gated_delta_rule_recurrent` and `_chunked` pass the frozen-framework tests (vs the fp64 golden + the
cross-codepath + chain-equivalence checks), `compute-sanitizer` clean, all code `clang-format`-clean, and
both bench paths show no regression vs `~/chunked_gdn` with recorded ncu profiles. Head mapping is grouped;
final correctness is confirmed at M2 per-layer parity vs HF/vLLM.

## Self-review notes (author)
- Tasks follow design §7; correctness (gdn-2/4) and performance (gdn-3/5) are separate; gdn-1 lands the
  shared (grouped) head_map + helpers first.
- Every task DoD includes the `clang-format` step (per the user) — the ported source must be reformatted
  to house style.
- New presets are additive; the frozen framework logic is untouched. Perf uses no-regression-vs-source,
  not the 85%-DRAM gate (GDN chunked is compute-bound).
