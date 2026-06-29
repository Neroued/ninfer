# q5090 v2 — Step 2: CPP Runtime on v2 (correct inference, then per-kernel tuning)

> Phase 2 of [docs/q5090_v2_roadmap.md](../q5090_v2_roadmap.md). Spec:
> [q5090_packed_file_format_v2.md](../q5090_packed_file_format_v2.md);
> assignment: [qwen3_6_27b_q5090_v2_tensor_plan.md](../qwen3_6_27b_q5090_v2_tensor_plan.md);
> gates: [q5090_v2_verification_contract.md](../q5090_v2_verification_contract.md). Prereq: Phase 1
> (v2 file + Python ref) is green.

## Goal

Make the cpp runtime **load the v2 layout, infer correctly, then run the decode-critical kernels at
high memory throughput**:

1. **Load** the v2 layout (parser + `Weight`/segment binding).
2. **Adapt the kernels** — GEMV (decode `T=1`) and GEMM (prefill `T>1`) — to `ROW_SPLIT`, correct first.
3. **Prove correct inference** end to end (parity vs the Phase-1 Python ref and HF).
4. **Per-kernel tuning** of the `ROW_SPLIT` GEMV/GEMM, profile-driven, to approach the weight-bandwidth
   roofline the v2 layout was designed for.

Stages C→D have a hard gate: **no tuning before correctness is green.**

## Non-goals / hard constraints

- **Framework unchanged.** Keep the linear backend structure (`LinearFormat` / `ShapeFamily` /
  `LinearRegime` / `LinearBackendKind` / `LinearPlan` registry, the `linear()` wrapper dispatch, the
  `WeightStore` API, the model-card per-projection call sites). Stage D adds **tuned plans into the
  existing registry** (as the framework intends); it does not restructure the framework.
- **No fused-projection group GEMV.** Per-segment GEMV is used throughout. Fusing a group into one
  large-`N` GEMV (framework §21.5) is a **bigger lever that changes call structure**; it is explicitly
  **out of scope** here and noted as the next phase.
- **No backward compatibility.** Remove all cpp TILE/v1 layout code; no fallback.
- **Do not delete the v1 weight file** (`out/…mixed_v1.qus`) — Phase 3 cutover. cpp now loads **v2**.
- **Correctness is invariant under tuning.** The fp64 oracle and greedy parity must re-pass after every
  Stage-D change; performance claims must be ncu/nsys-backed (contract §6).
- No MTP/Vision runtime, no CUDA Graph, no KV/attention changes.

## Execution mode

Subagent-driven, **sequential** (shared core files). One implementer subagent per task; after each code
task, a spec-compliance reviewer then a code-quality reviewer; fix Critical/Important before
proceeding. Stage D uses an implementer + a profiler subagent per round (like the attention perf plan).
Final independent review after T8 (correctness) and again after T11 (performance).

Performance agents must read and follow before profiling:
- `/home/neroued/.cursor/skills/profile-cuda/SKILL.md`
- `/home/neroued/.codex/skills/ncu-kernel-profile/SKILL.md`, `nsys-inference-analysis/SKILL.md`

## Scope and ownership (files)

**Stage A — Load:** `include/qus/core/tensor.h` (`QuantLayout`+`Weight`), `weight_store_parser.{h,cpp}`,
`weight_store.{h,cpp}`, `include/qus/core/weight.h`, `tools/parity/block_dump.cpp`.

**Stage B — Kernels (correctness):** `src/kernels/linear/codec/linear_codec.cuh`,
`src/kernels/linear/reference/linear_generic_lowbit.cuh`, `linear_generic_lowbit_gemv.cu`,
`linear_generic_lowbit_gemm.cu`, `linear_generic.h`, `linear_generic_dense.{cuh,cu}`,
`src/kernels/linear/plan/linear_plan.{h,cpp}`, `src/kernels/linear/linear.cpp`,
`src/kernels/wrapper/embed_gather.cpp`, **remove** `src/kernels/linear/gemv/linear_lowbit_gemv.{h,cuh,cu}`,
`tests/kernels/q5090_pack.h`, `tests/kernels/test_linear.cpp`.

**Stage C — Integration:** `src/model/qwen3_6_27b.cpp` (verify), `src/runtime/engine.cpp` (verify),
`tests/test_model_bind.cpp`, `tests/test_model_blocks.cpp`, `tests/CMakeLists.txt`.

**Stage D — Tuning:** new tuned kernels under `src/kernels/linear/gemv/` (e.g.
`linear_rowsplit_gemv.{h,cuh,cu}`) and, if needed, `src/kernels/linear/gemm/`; `linear_plan.{h,cpp}`
(register tuned plans); `bench/` linear bench (if a per-op bench is needed); profiles under
`profiles/ncu-linear-v2/` and `profiles/nsys/` (local artifacts only).

**Coordination points:** `tensor.h` (T1, consumed by all) → `weight_store*` (T1/T2) → linear backend
(T3–T5) → model/tests (T6) → `linear_plan.{h,cpp}` again in Stage D (register tuned plans). Edit shared
files only in the owning task.

## Reference paths

- v2 weights: `out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus`
- HF bf16 oracle: `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16`
- Phase-1 Python ref + dumps + greedy snapshot: `tools/parity/*`, `out/*dump*.v2.json`

---

# Stage A — Load the v2 layout

## Task 1 — v2 parser & `QuantLayout`

**Files:** `tensor.h`, `weight_store_parser.{h,cpp}`.
**Reading:** binary spec §1–§9; current `weight_store_parser.{h,cpp}`, `tensor.h`.
**Requirements:** parse the v2 header (magic `Q5090MIXEDV2`, version 2, segment/fusion offsets+counts),
`ModuleRecord`, `TensorEntry` (block: `segment_*`, `fusion_*`, `code_plane_bytes`, `scale_plane_bytes`),
and the new `SegmentRecord` + `FusionGroupRecord` tables into `ParsedQ5090File`. `QuantLayout` gains
`RowSplit`, drops TILE/row-grouped. Validate contract **L0** structural items.
**DoD:** parser test parses the Phase-1 v2 file; L0 passes; no TILE enum remains in the parser.

## Task 2 — `WeightStore` segment binding + cpp dump

**Files:** `weight_store.{h,cpp}`, `weight.h`, `tools/parity/block_dump.cpp`.
**Reading:** tensor-plan doc; contract L0/§5; current `weight_store.cpp`, `block_dump.cpp`.
**Requirements:** per segment, build a `Weight` with `payload = code_plane + row_begin·G·bpr`,
`scales = scale_plane + row_begin·G·2`, `n = row_count`, `k = K`, `layout = RowSplit`; dense `Tensor`
from `CONTIGUOUS` blocks. `qweight(module, source_kind, source_layer)` keys unchanged. Emit a cpp dump
matching the converter/Python schema.
**DoD:** every projection resolves to a segment view; `compare_dumps` shows cpp == converter == Python
(offsets, CRCs, sampled dequant).

# Stage B — Adapt the kernels (correctness, untuned)

## Task 3 — `ROW_SPLIT` codec + generic decode GEMV (`T=1`)

**Files:** `linear_codec.cuh`, `linear_generic_lowbit.cuh`, `linear_generic_lowbit_gemv.cu`,
`linear_generic.h`, `q5090_pack.h`, `test_linear.cpp`.
**Reading:** binary spec §9; current codec + generic GEMV; `q5090_pack.h`.
**Requirements:** codec addresses `(row,group)` → code `code_ptr+(row·G+group)·bpr`, scale
`scale_ptr+(row·G+group)·2` (row is segment-relative). Reuse §9.1 unpack. Implement a **correct,
untuned** generic GEMV over `ROW_SPLIT`. Update the test packer to `ROW_SPLIT`.
**DoD:** `qus_linear_test` fp64-oracle parity for Q4/Q5/Q6 GEMV (incl. non-64 tails); `compute-sanitizer
--tool memcheck` clean.

## Task 4 — generic prefill GEMM (`T>1`)

**Files:** `linear_generic_lowbit_gemm.cu`, `linear_generic_lowbit.cuh` (shared), `test_linear.cpp`.
**Reading:** binary spec §9; current generic GEMM.
**Requirements:** correct, untuned `ROW_SPLIT` GEMM (`T>1`) on the same codec; SMEM-staged per-row
segment loads (decode-first layout, prefill compute-bound — correctness only here).
**DoD:** `qus_linear_test` fp64-oracle parity for GEMM (`T>1`) across qtypes/shapes.

## Task 5 — Dispatch rename, remove TILE, embed gather

**Files:** `linear_plan.{h,cpp}`, `linear.cpp`, **remove** `linear/gemv/linear_lowbit_gemv.*`,
`embed_gather.cpp`, `CMakeLists`.
**Reading:** framework doc §5–§14; current `linear_plan.*`, `linear.cpp`, `embed_gather.cpp`.
**Requirements:** `LinearFormat` `Q*G64_N64K64` → `Q*G64_RowSplit`; `classify_format` maps v2 weights;
the registry resolves all quantized keys to the **generic** `ROW_SPLIT` GEMV/GEMM (tuned plans come in
Stage D). Wrapper validates `layout == RowSplit`. Delete the tuned-TILE GEMV + its build target.
`embed_gather` reads one row from the Q6 `ROW_SPLIT` planes. Framework types/flow otherwise unchanged.
**DoD:** `qus` builds; `rg -n "TileN64K64|RowGrouped|N64K64|linear_lowbit_gemv|Q5090MIXEDV1" src/ include/`
is empty.

# Stage C — Integrate & prove correct inference

## Task 6 — Model integration & block parity

**Files:** `qwen3_6_27b.cpp` (verify), `engine.cpp` (verify), `test_model_bind.cpp`,
`test_model_blocks.cpp`, `tests/CMakeLists.txt`.
**Requirements:** confirm `bind()` resolves every projection from segment weights with no model-card
structural change; engine loads the v2 file and sizes the weight arena from its payload. Adapt the bind
/ block tests to v2.
**DoD:** `qus_model_bind_test` and `qus_model_blocks_test` pass (contract L3 per-op block parity).

## Task 7 — End-to-end correctness + sanitizer

**Requirements:** run `qus` on the v2 file; greedy decode matches the Phase-1 snapshot (contract
**L4-greedy**, cpp vs Python/snapshot) and per-op taps match the Python ref (**L3-impl**, cos ≥ 0.9999);
`compute-sanitizer` clean over a short decode on the load + linear path.
**DoD:** greedy matches the snapshot exactly; memcheck clean. (Throughput irrelevant here.)

## Task 8 — Remove residual cpp v1 layout code (audit)

**Requirements:** no `TileN64K64`/`RowGrouped`/v1 magic/tuned-TILE references in cpp. v1 *weight file*
and v1 *docs* untouched (Phase 3).
**DoD:** `rg -n "TileN64K64|RowGrouped|TILE_N64|Q5090MIXEDV1" src/ include/ tests/ tools/` empty;
`cmake --build build -j && ctest --test-dir build` green.

> **CORRECTNESS GATE.** Stage D does not begin until T1–T8 are green and the post-T8 review passes.

# Stage D — Per-kernel tuning (performance)

Decode is **weight-bandwidth bound** (~15.4 GB of weights streamed per token); the v2 `ROW_SPLIT` layout
was designed for GEMV-friendly streaming (row-contiguous, coalesced, vectorizable, `N`-parallel). The v1
tile kernels were layout-capped at ~64% weighted DRAM; the goal here is to let the tuned `ROW_SPLIT`
kernels approach the cold-cache DRAM ceiling per shape.

**Methodology (every round):**
- One identified limiter per round; before/after `ncu` artifacts under `profiles/ncu-linear-v2/`.
- Cold-cache (L2-flushed) `dram__throughput.avg.pct_of_peak_sustained_elapsed` is the gate; hot-cache
  numbers are diagnostic only. Re-calibrate the achievable ceiling for `ROW_SPLIT`.
- **Correctness re-verified every commit**: `qus_linear_test` (fp64 oracle) + the Task-7 greedy parity
  must stay green. A tuning change that breaks either is reverted.
- Register tuned plans in the existing `LinearPlan` registry, dispatched by `ShapeFamily`; the generic
  `ROW_SPLIT` plan remains the correctness fallback.

## Task 9 — Baseline profile & roofline

**Files:** none (profile only); artifacts under `profiles/ncu-linear-v2/`.
**Requirements:** ncu cold-cache the generic `ROW_SPLIT` GEMV on the dominant shapes
(`MlpGateUp17408x5120` Q4, `MlpDown5120x17408` Q5, `Out5120x6144`/`Proj6144x5120` Q5,
`LmHead248320x5120` Q6); report duration, achieved occupancy, DRAM/L2 throughput, sectors, top
limiter; compute the per-shape weight-bandwidth roofline. Identify the first limiter to attack per shape.
**DoD:** a metric-backed baseline + per-shape limiter list; no source edits.

## Task 10 — Tuned `ROW_SPLIT` decode GEMV (per shape, profile-driven)

**Files:** new `src/kernels/linear/gemv/linear_rowsplit_gemv.{h,cuh,cu}`, `linear_plan.{h,cpp}`,
optional `bench/` linear per-op bench.
**Requirements:** implement a tuned `ROW_SPLIT` decode GEMV and tune it per shape family, one limiter
per round (candidate levers: warp-per-row vs warps-per-row; vectorized 16-byte code loads; scale-plane
prefetch; loop unroll; `N`-parallel CTA count for occupancy; minimizing the `Q5/Q6` unpack cost). Wire
hot shapes to the tuned plan via the registry. Re-verify correctness each round.
**DoD:** for each dominant shape, cold-cache DRAM throughput materially exceeds the Task-9 baseline and
approaches the calibrated ceiling, or `ncu` shows a non-bandwidth limiter requiring a larger change
(documented). Correctness green throughout; before/after `ncu` artifacts saved.

## Task 11 — End-to-end nsys + optional prefill GEMM tuning

**Files:** none for nsys (artifacts under `profiles/nsys/`); prefill GEMM tuning only if pursued
(`src/kernels/linear/gemm/…`, `linear_plan.{h,cpp}`).
**Requirements:** nsys a long decode run on the v2 file; confirm the tuned GEMV moves real decode time
(lower-bit GEMV share drops, tok/s rises) and report the new top decode bottleneck. Prefill GEMM tuning
is **optional and lower priority** (prefill is compute-bound, not the decode bottleneck); pursue only if
TTFT is a concern.
**DoD:** e2e nsys shows the per-op gains move the real decode workload; the remaining top bottleneck is
named with profile evidence.

> **Beyond per-kernel tuning (next phase, not here):** the largest remaining decode lever is the
> **fused-projection group GEMV** (one large-`N` GEMV per fusion block, framework §21.5), which the v2
> layout already stores for. It changes the L1/L2 call structure and is planned separately.

## Definition of done

**Correctness (Stages A–C):** cpp loads v2; every projection binds as a `ROW_SPLIT` segment;
cpp/converter/Python dumps identical; generic GEMV/GEMM pass the fp64 oracle; model block parity passes;
`qus` greedy matches the Phase-1 snapshot (L4) and per-op taps match the Python ref (L3-impl);
`compute-sanitizer` clean; framework unchanged; no TILE/v1 layout code remains; v1 weight file untouched.

**Performance (Stage D):** per dominant shape, the tuned `ROW_SPLIT` GEMV beats the generic baseline and
approaches the cold-cache ceiling (or the limiter is documented); e2e nsys confirms the decode gain;
correctness gates remained green throughout; tuned plans are registered in the existing framework.

## Review phase

Risk: CUDA kernels, numerical behavior, q5090 ABI, GPU memory, benchmark evidence. Reviews:

**After T8 (correctness):**
1. **Numerical-correctness reviewer** — codec addressing, generic GEMV/GEMM math, fp64 oracle, L3-impl/L4.
2. **CUDA memory/lifetime reviewer** — segment sub-view bounds, weight-load/arena lifetime,
   `embed_gather` bounds; `compute-sanitizer` evidence.
3. **Format/ABI reviewer** — parser vs spec, segment binding, dump cross-check, `source_kind` rule.
4. **Scope reviewer** — framework unchanged, no perf kernels yet, no fused-group GEMV, v1 file untouched,
   cpp v1 layout code removed.

**After T11 (performance):**
5. **Performance-evidence reviewer** — claims are cold-cache `ncu`/`nsys`-backed (not hot-cache), one
   limiter per round with before/after artifacts, and correctness (fp64 oracle + greedy) stayed green
   across every tuning commit.
