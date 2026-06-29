# q5090 v2 — Step 2: CPP Runtime on v2 (correct inference, then per-kernel tuning)

> Phase 2 of [docs/q5090_v2_roadmap.md](../q5090_v2_roadmap.md). Spec:
> [q5090_packed_file_format_v2.md](../q5090_packed_file_format_v2.md);
> assignment: [qwen3_6_27b_q5090_v2_tensor_plan.md](../qwen3_6_27b_q5090_v2_tensor_plan.md);
> gates: [q5090_v2_verification_contract.md](../q5090_v2_verification_contract.md). Prereq: Phase 1
> (v2 file + Python ref) is green.

## Goal

Make the cpp runtime **load the v2 layout, infer correctly, then run the decode-critical kernels at
high memory throughput**:

1. Load the v2 layout (parser + `Weight`/segment binding) and adapt the GEMV (`T=1`) / GEMM (`T>1`)
   kernels to `ROW_SPLIT`, **correct first**, proving end-to-end inference (reproduce the recorded
   quantized greedy snapshot exactly; HF is diagnostic only).
2. Then one profile-driven tuning loop per decode-critical `ROW_SPLIT` GEMV, toward the weight-bandwidth
   roofline the v2 layout was designed for (prefill GEMM is compute-bound and deferred).

Hard gate: **no tuning before correctness is green.**

## Non-goals / hard constraints

- **Framework unchanged.** Keep the linear backend structure (`LinearFormat` / `ShapeFamily` /
  `LinearRegime` / `LinearBackendKind` / `LinearPlan` registry, the `linear()` wrapper dispatch, the
  `WeightStore` API, the model-card per-projection call sites). Tuning adds **tuned plans into the
  existing registry**; it does not restructure the framework.
- **No fused-projection group GEMV.** Per-segment GEMV throughout. Fusing a group into one large-`N`
  GEMV (framework §21.5) is a bigger lever that changes call structure — **out of scope**, next phase.
- **No backward compatibility (AGENTS.md).** Delete all cpp TILE/v1 layout code outright — the
  `QuantLayout::{TileN64K64, W4A16KernelPackedV1, TileN64K128, RowGroupedG64}` enumerators, the v1 magic
  `Q5090MIXEDV1`, the tuned-TILE GEMV, the inline RowGrouped embed-gather path. No shims, no fallback.
- **Do not delete the v1 weight file** (`out/…mixed_v1.qus`) — Phase 3 cutover. cpp now loads **v2**.
- **Correctness is invariant under tuning.** The fp64 oracle and greedy parity must re-pass after every
  tuning change; performance claims must be ncu/nsys-backed (contract §6).
- No MTP/Vision runtime, no CUDA Graph, no KV/attention changes.

## Execution mode

This is a **single relayout refactor**, not a feature build. It pivots on one enum (`QuantLayout` in
`tensor.h`): the moment its TILE/RowGrouped values are removed, every `qus_core` translation unit that
references them (`weight_store_parser`, `weight_store`, `linear`, `linear_plan`, `embed_gather`, plus
the kernels and every affected test/fixture/bench) stops compiling. **Nothing builds, links, or tests
until the whole correctness refactor lands.** Splitting it into per-file/per-stage tasks only creates a
chain of non-building intermediate states whose per-task DoD, build, test, and review are impossible or
meaningless. So:

- **Task 1 (correctness) is one atomic task.** A single implementer subagent does the entire v1→v2 cpp
  relayout in one worktree, with **no intermediate verification, commit, or review**. It is verified
  **once** against the full gate set and reviewed **once** (consolidated) at the end. The ordered
  implementation sequence inside Task 1 is guidance, not a set of verification boundaries.
- **Phase 2 (performance) is one iterative loop per kernel.** Each decode-critical GEMV ShapeFamily
  gets its own profile→optimize→re-profile loop task (Task 2.1–2.7), each genuinely builds, tests, and
  profiles on its own (correctness stays green throughout), so per-round profiling is meaningful. They
  run only after the Task-1 correctness gate passes.

Subagent-driven. Two rules govern Phase 2 and are non-negotiable:

- **Strongest model, repeated iteration.** Every tuning-loop subagent is dispatched on the **strongest
  reasoning/coding model the harness offers** and is re-dispatched/resumed to keep iterating
  (profile → optimize → re-profile) until its gate is met. Tuning loops must **not** be downgraded to a
  faster/weaker model, and must **not** be run one-shot.
- **The coordinator must NOT prescribe optimization direction.** The coordinator hands a tuning subagent
  only the *target* (ShapeFamily + qtype variants), the *correctness invariants*, the *cold-cache
  bandwidth gate + how to measure*, and the *iteration protocol*. It must **not** suggest limiters,
  levers, kernel layouts, or hypotheses. The subagent discovers the limiter from its own `ncu` evidence,
  forms its own hypothesis, implements, re-profiles, and reports back what it found and changed.

Performance agents must read and follow before profiling:
`/home/neroued/.cursor/skills/profile-cuda/SKILL.md`,
`/home/neroued/.codex/skills/ncu-kernel-profile/SKILL.md`, `nsys-inference-analysis/SKILL.md`.

**Prerequisite (coordinator, before Task 1).** Phase 1 rewrote the shared `tools/q5090_convert/*`
modules to v2 (layout enum now `{ROW_SPLIT=0, CONTIGUOUS=1}`; v1 enums deleted). The cpp
fixture-dependent tests regenerate their fixture at runtime via `tests/fixtures/make_q5090_fixture.py`,
which still references the deleted v1 enums — so several cpp tests are **already red** before Phase 2.
Run `cmake --build build -j && ctest --test-dir build` once and record the baseline (already-red:
`qus_q5090_parser_test`, `qus_model_bind_test`, `qus_model_blocks_test`, `qus_weight_store_test`) so the
final green is attributable; do not treat a pre-existing red as introduced by Task 1.

**Python env.** Run all Python steps (fixture generation, `compare_dumps`, any parity/HF diagnostic)
with the `vllm-bench` conda env: `/home/neroued/miniconda3/envs/vllm-bench/bin/python` (CUDA PyTorch).

## Reference paths

- v2 weights: `out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus`; preserved v1: `out/…mixed_v1.qus` (never write).
- HF bf16 oracle (diagnostic only): `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16`.
- Phase-1 dumps: `out/conv_dump.v2.json` (converter), `out/ref_dump.v2.json` (Python ref).
- Greedy snapshot (G-SNAPSHOT authority): `profiles/e2e/m3-output-gate.json`; canonical prompt
  `bench/fixtures/prompts/cn_short.ids` (case `cn_short`), manifest
  `bench/fixtures/prompts/m2.8-v1.manifest.json`.

---

## Task 1 — v2 cpp correctness refactor (atomic; one subagent, one worktree, one gate)

Implement the entire v1→v2 cpp relayout as one change. The tree is expected to be non-building until the
sequence is complete; do not attempt intermediate builds/tests/reviews.

**Files (the whole correctness surface):**
- Core/types: `include/qus/core/tensor.h` (`QuantLayout`+`Weight`), `include/qus/core/weight.h`.
- Parse/load: `include/qus/core/weight_store_parser.h`, `src/core/weight_store_parser.cpp`,
  `src/core/weight_store.{h,cpp}`.
- Kernels: `src/kernels/linear/codec/linear_codec.cuh`,
  `src/kernels/linear/reference/linear_generic_lowbit.cuh`, `linear_generic_lowbit_gemv.cu`,
  `linear_generic_lowbit_gemm.cu`, `linear_generic.h`, `src/kernels/linear/plan/linear_plan.{h,cpp}`,
  `src/kernels/linear/linear.cpp`; **remove** `src/kernels/linear/gemv/linear_lowbit_gemv.{h,cuh,cu}`.
- Embed gather (full path): `src/kernels/wrapper/embed_gather.cpp`,
  `src/kernels/kernel/embed_gather.cuh`, `src/kernels/launcher/embed_gather.{h,cu}`,
  `include/qus/kernels/embed_gather.h`.
- Model/engine (verify, minimal/no change): `src/model/qwen3_6_27b.cpp`, `src/runtime/engine.cpp`.
- Dumps: `tools/parity/block_dump.cpp`, `tools/parity/layer_dump.cpp`.
- Fixtures/tests/benches: `tests/fixtures/make_q5090_fixture.py`, `tests/kernels/q5090_pack.h`,
  `tests/test_q5090_parser.cpp`, `tests/test_weight_store.cpp`, `tests/test_weight_store_real.cpp`,
  `tests/test_q5090_pack_golden.cpp`, `tests/kernels/test_linear.cpp`,
  `tests/kernels/test_embed_gather.cpp`, `tests/test_model_bind.cpp`, `tests/test_model_blocks.cpp`,
  `bench/linear_bench.cu`, `bench/embed_gather_bench.cu`, `tests/CMakeLists.txt`/`bench/CMakeLists.txt`
  as needed.

**Reading list:** binary spec §1–§13 (header/records, ROW_SPLIT §9.2, fusion §10, consumption §11);
verification contract (G-STRUCT/G-DUMP/G-KERNEL/G-SNAPSHOT, HF diagnostic-only); tensor-plan doc;
framework doc §5–§14; the current versions of every file above; the v2 converter writer in
`tools/q5090_convert/{format,convert,layouts,qtypes}.py` (the fixture should mirror it).

**Implementation sequence (ordered; not verification boundaries):**
1. `tensor.h`: `QuantLayout = { RowSplit = 0, Contiguous = 1 }`; repoint the default `Weight.layout`
   (currently `W4A16KernelPackedV1`) to `RowSplit`. (Everything below now must land before the tree
   builds again.)
2. **Parser:** parse the v2 header (magic `Q5090MIXEDV2`, version 2, segment/fusion offsets+counts,
   `format_minor`), `ModuleRecord`, `TensorEntry` block fields (`segment_count`, `segment_begin`,
   `fusion_group_id`, `fusion_index`, `code_plane_bytes`, `scale_plane_bytes`), and the new
   `SegmentRecord` + `FusionGroupRecord` tables into `ParsedQ5090File` (new vectors + struct fields).
   Validate G-STRUCT structurals (offsets in range/ordered, payloads 256-aligned/non-overlapping,
   per-block `crc32`, segments partition `[0,N)`, fusion adjacency + `source_kind` rule, plane-byte
   formulas).
3. **Weight store:** build the `Weight` table from **segments**, not per-tensor: per segment,
   `qdata = code_plane + row_begin·G·bpr`, `scales = scale_plane + row_begin·G·2`, `n = row_count`,
   `k = K`, `layout = RowSplit`, identity `(source_kind, source_layer)` + canonical name; dense `Tensor`
   from `CONTIGUOUS` blocks. `qweight(name)` and `qweight(module, source_kind, source_layer)` keys
   unchanged (a fused block yields one `Weight` per segment).
4. **Codec + generic kernels:** codec takes **two** plane pointers, segment-relative `(row,group)` →
   code `code_ptr+(row·G+group)·bpr`, scale `scale_ptr+(row·G+group)·2` (reuse §9.1 unpack). Implement
   correct, untuned generic GEMV (`T=1`) and GEMM (`T>1`) over `ROW_SPLIT` on that codec; the launches
   pass `w.qdata` (code plane) + `w.scales` (scale plane).
5. **Dispatch + TILE removal:** `LinearFormat` `Q*G64_N64K64` → `Q*G64_RowSplit`; `classify_format`
   maps v2; replace `require_tile_lowbit_metadata` with a row-split validator (§9.2 plane sizes,
   unpadded `N`, `K_pad`); route all quantized keys (incl. `T1`) to the **generic** GEMV/GEMM. Delete
   `LinearPolicyId::TunedLowbitGemv`, `linear_tuned_lowbit_gemv_launch`, its dispatch case, and the
   removed-header include (a tuned plan returns in Task 2).
6. **Embed gather on `ROW_SPLIT`:** wrapper validates `layout == RowSplit`; kernel + launcher read one
   row from the **two** Q6 planes (code-plane row run + separate scale plane), not the inline 50-byte
   RowGrouped rows.
7. **Fixtures/tests/benches:** `make_q5090_fixture.py` emits a small **v2** file (ROW_SPLIT standalone
   + a multi-segment/fused block + CONTIGUOUS control + ≥1 fusion group), reusing the converter writer;
   `q5090_pack.h` test packer → ROW_SPLIT; update all listed tests to v2 (repoint
   `test_weight_store_real.cpp` from the v1 file to `…mixed_v2.qus`); update the two benches; emit cpp
   dumps from `block_dump`/`layer_dump` in the converter/Python schema.
8. **Model/engine:** confirm `bind()` resolves every projection from segment weights with no model-card
   structural change and the engine loads v2 (`Engine::default_weight_bytes` is already file-size
   derived — verify, do not change).

**Definition of done (run ONCE, all must pass):**
- `cmake --build build -j` clean (all targets incl. benches/tools).
- **G-STRUCT:** `qus_q5090_parser_test` green; structural + plan-conformance checks pass.
- **G-DUMP:** `qus_weight_store_test` + `qus_weight_store_real_file_test` green; cpp `block_dump`
  vs `out/conv_dump.v2.json` and `out/ref_dump.v2.json` via `compare_dumps` (vllm-bench python)
  bit-exact on metadata + sampled `(scale16, q_i)`.
- **G-KERNEL:** `qus_linear_test` fp64 oracle for Q4/Q5/Q6 **GEMV and GEMM** (incl. non-64 tails);
  `qus_embed_gather_test` green; `compute-sanitizer --tool memcheck` clean on both.
- **Model:** `qus_model_bind_test` + `qus_model_blocks_test` green (in-tree/recorded reference, bf16
  tol; no HF gate, no per-op cpp-vs-Python comparison).
- **G-SNAPSHOT:** the cpp engine reproduces the snapshot exactly for its length via `qus_e2e_bench`:

```bash
cmake --build build --target qus_e2e_bench -j
./build/bench/qus_e2e_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
  --output-json out/e2e.v2.json \
  --fixture-manifest bench/fixtures/prompts/m2.8-v1.manifest.json \
  --case cn_short:bench/fixtures/prompts/cn_short.ids:96 \
  --warmup-repeats 1 --repeats 1 --max-ctx 8192 --device 0 \
  --stop-token-id 248046 --stop-token-id 248044
# out/e2e.v2.json cn_short generated_token_ids must equal
# profiles/e2e/m3-output-gate.json cn_short generated_token_ids, exactly, for the
# snapshot length (tools/bench/compare_e2e_reports.py, or assert the prefix). HF is diagnostic only.
```
  `compute-sanitizer --tool memcheck` clean over a short decode on the load+linear path.
- **Audit + full suite:**
```bash
rg -n "TileN64K64|TileN64K128|RowGroupedG64|W4A16KernelPackedV1|TILE_N64|RowGrouped|N64K64|linear_lowbit_gemv|TunedLowbitGemv|Q5090MIXEDV1" \
   src/ include/ tests/ tools/ bench/   # empty
cmake --build build -j && ctest --test-dir build   # green (baseline reds cleared; v1 file untouched)
```

**Review (ONCE, after the gate passes).** Single consolidated independent review (the change touches
CUDA kernels, q5090 ABI, GPU memory, and numerics, so all four lenses apply — but run them together,
not per-step):
1. Numerical-correctness — codec addressing, generic GEMV/GEMM math, Q6 embed-gather dequant, the
   CUDA↔cpp fp64 oracle (G-KERNEL), the snapshot match (G-SNAPSHOT); every HF comparison diagnostic only.
2. CUDA memory/lifetime — segment sub-view bounds, weight-load/arena lifetime, embed-gather plane
   bounds; `compute-sanitizer` evidence.
3. Format/ABI — parser vs spec, segment binding, dump cross-check, `source_kind` rule, and the v2
   fixture/test migration.
4. Scope — framework unchanged, no perf kernels yet, no fused-group GEMV, v1 weight file untouched, all
   v1/TILE layout code removed across src/include/tests/tools/bench.

> **CORRECTNESS GATE.** Phase 2 does not begin until Task 1 is green and its review passes.

---

## Phase 2 — Per-kernel tuning (one iterative loop task per decode-critical GEMV)

Decode is **weight-bandwidth bound** (~15.4 GB of weights streamed per token); the v2 `ROW_SPLIT` layout
was designed to lift the v1 tile kernels' ~64% weighted-DRAM cap. Each decode-critical GEMV is tuned
**on its own** in a dedicated profile-driven loop toward its cold-cache DRAM ceiling — *how* to get
there is for each loop's own profiling to determine, not for this plan to prescribe.

> **Correctness is invariant under tuning.** Every Phase-2 loop keeps `qus_linear_test` (fp64 oracle) and
> the Task-1 G-SNAPSHOT green; a change that breaks either is reverted, not patched forward.

### Task 2.0 — Tuning harness + roofline table (measurement only; no kernel changes)

Build the shared rig the loops measure with. **No optimization analysis here** — just the rig and the
raw numbers (limiter discovery belongs to each loop).
- **Files:** `bench/linear_op_bench.cu` (cold-cache, L2-flushed per-op GEMV bench: takes ShapeFamily +
  qtype, runs the generic `ROW_SPLIT` plan, emits duration + achieved GB/s), `bench/CMakeLists.txt`;
  artifacts under `profiles/ncu-linear-v2/`.
- **DoD:** the bench builds and runs each shape/format below; record a baseline table of `{shape, qtype,
  cold-cache duration, achieved GB/s, achieved DRAM %}` and the **computed weight-bandwidth roofline**
  per shape (bytes streamed ÷ measured cold-cache STREAM ceiling). No limiter labels, no levers.

### Shared loop protocol (identical for Tasks 2.1–2.7; coordinator states only the target, never the how)

Each loop is run by one subagent on the **strongest available model**, iterating until its gate:
1. **Profile.** `ncu` cold-cache (L2-flushed) the current kernel for this target via the Task-2.0 bench.
2. **Discover.** From the `ncu` evidence *the subagent itself* identifies the single dominant limiter and
   forms one hypothesis. The coordinator/plan provides no limiter, lever, or layout suggestion.
3. **Change.** Implement that one change in this loop's own kernel file; register/refresh its tuned plan.
4. **Re-profile + verify.** Re-run `ncu` (before/after artifacts saved); re-run `qus_linear_test` (fp64
   oracle) for this format and confirm G-SNAPSHOT still holds. Revert if correctness regresses.
5. **Iterate.** Repeat 1–4 (one limiter per round) until the gate is met or `ncu` evidence documents a
   non-bandwidth structural wall — which the subagent **reports back** (it does not guess a redesign).

- **Gate metric:** cold-cache `dram__throughput.avg.pct_of_peak_sustained_elapsed` approaching the
  Task-2.0 roofline for this shape; hot-cache is diagnostic only.
- **File ownership (one kernel per loop, no overlap):** each loop owns
  `src/kernels/linear/gemv/linear_rowsplit_gemv_<tag>.{cuh,cu}` and its own ncu artifacts. Shared
  read-only: `linear_codec.cuh`, the generic fallback kernel, the Task-2.0 bench. **Coordination point:**
  `src/kernels/linear/plan/linear_plan.{h,cpp}` (each loop adds its `LinearPolicyId` + `resolve_plan`
  entry) — run the loops **sequentially by priority** so registry edits and GPU profiling do not
  collide. The generic `ROW_SPLIT` plan stays the correctness fallback for every shape.
- **Per-loop DoD:** cold-cache DRAM throughput for the target shape/format(s) materially beats the
  Task-2.0 baseline and approaches its roofline, **or** `ncu` documents a non-bandwidth limiter; fp64
  oracle + G-SNAPSHOT green; tuned plan registered; before/after `ncu` artifacts committed.
- **Reading list:** the profiling skills above; binary spec §9/§17; the Task-1 generic GEMV/codec; the
  Task-2.0 baseline table.

### Per-kernel loop tasks (each = one ShapeFamily's decode GEMV; priority = scheduling order)

| Task | ShapeFamily (`N×K`) | qtype variant(s) | tier / priority | owned kernel file `…_<tag>` |
|---|---|---|---|---|
| 2.1 | `MlpGateUp17408x5120` | Q4 | A · P1 | `_mlp_gate_up` |
| 2.2 | `MlpDown5120x17408` | Q5 | A · P2 | `_mlp_down` |
| 2.3 | `LmHead248320x5120` | Q6 | A · P3 | `_lm_head` |
| 2.4 | `Proj6144x5120` | Q5 + Q4 | B · P4 | `_proj_6144` |
| 2.5 | `Out5120x6144` | Q5 | B · P5 | `_out_6144` |
| 2.6 | `GdnQK2048x5120` | Q4 | C1 · P6 | `_gdn_qk_2048` |
| 2.7 | `AttnKV1024x5120` | Q5 + Q4 | C1 · P7 | `_attn_kv_1024` |

- A loop that serves two qtypes (2.4, 2.7) tunes both format instantiations of that shape in the same
  loop (same kernel template, separate registry entries).
- **Tier A (2.1–2.3) is mandatory.** Tier B (2.4–2.5) is strongly recommended. Tier C1 (2.6–2.7) only if
  the Task-2.8 nsys shows these shapes still hold a material share of decode time.
- `DenseCtrl48x5120` (dense control) and Q6 `embed_gather` (gather, negligible per-token bytes) are
  **not** tuning targets. Prefill GEMM (`T>1`, compute-bound) is **out of scope** unless TTFT becomes a
  goal — then it is its own later loop, not folded in here.

### Task 2.8 — End-to-end nsys integration (after the loops)

nsys a long decode on the v2 weights; confirm the tuned GEMVs moved real decode time (lower-bit GEMV
share drops, tok/s rises) and name the new top decode bottleneck. **DoD:** nsys report shows measurable
decode-time reduction attributable to the tuned kernels; fp64 oracle + G-SNAPSHOT still green.

### Phase-2 review (ONCE, after the loops + nsys)

Performance-evidence review: every claim is cold-cache `ncu`/`nsys`-backed (not hot-cache); each loop
shows one-limiter-per-round before/after artifacts and a subagent-discovered limiter (not a
coordinator-prescribed one); correctness (fp64 oracle + greedy) stayed green across every tuning commit;
tuned plans live in the existing registry with the generic plan still the fallback.

> **Beyond per-kernel tuning (next phase, not here):** the largest remaining decode lever is the
> **fused-projection group GEMV** (one large-`N` GEMV per fusion block, framework §21.5), which the v2
> layout already stores for. It changes the L1/L2 call structure and is planned separately.
