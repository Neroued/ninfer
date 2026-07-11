# q5090 v2 — Decode GEMV Fusion + Per-kernel Tuning (kernel-level decode optimization)

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` for the
> subagent tasks (Task 1 + the Phase-2 tuning loops) and `superpowers:using-git-worktrees` for the
> parallel tuning worktrees. Steps use checkbox (`- [ ]`) syntax. Task 0 is executed directly by the
> coordinator (no subagent).
>
> Successor to [docs/plans/2026-06-29-q5090-v2-step2-cpp-runtime.md](2026-06-29-q5090-v2-step2-cpp-runtime.md),
> whose closing note names "fused-projection group GEMV" as the next lever. Spec:
> [q5090_packed_file_format_v2.md](../q5090_packed_file_format_v2.md) §6/§7/§10/§11. Evidence:
> [docs/bench/q5090-v2-long-decode-nsys-report.md](../bench/q5090-v2-long-decode-nsys-report.md),
> [docs/bench/q5090-v2-rowsplit-gemv-ncu-report.md](../bench/q5090-v2-rowsplit-gemv-ncu-report.md).

**Goal:** Cut q5090 v2 single-stream **decode** cost at the kernel/operator layer by (1) realizing the
v2 layout's designed-in **shared-input projection fusion** so decode issues one GEMV per fused block
instead of one per projection, eliminating the GDN q/k/v pack/unpack `cudaMemcpy2DAsync` traffic; then
(2) tuning each decode-critical GEMV — the **new fused kernels** and the two unfusable underfilled ones
(`mlp_down` Q5, `out` Q5) — to the weight-bandwidth roofline.

**Architecture:** The v2 fusion groups (`ATTN_IN`, `GDN_IN`, `MLP_GATEUP`) are already stored as single
contiguous `[N_block,K]` `ROW_SPLIT` matrices; the runtime currently fragments each into per-segment
GEMVs. One atomic refactor exposes block-level `Weight`s from the weight store, adds **correct (not yet
optimal)** fused decode (`T=1`) GEMV kernels, rewires the decode path of the model schedule to call one
fused `linear()` per block into a combined output buffer (downstream ops consume sub-views; GDN writes
straight into the conv `qkv` buffer), and deletes the superseded per-segment decode kernels. Once those
kernel APIs are stable, a fleet of **parallel per-kernel tuning loops** (one git worktree each) drives
each kernel to its roofline using only the per-op bench (never full-model inference, so no GPU
contention between loops).

**Tech Stack:** C++20 / CUDA (RTX 5090, sm_120), CMake (`file(GLOB_RECURSE CONFIGURE_DEPENDS)`),
`qus_core` static lib, `ctest`, `qus_linear_op_bench` (cold-cache per-op), `qus_e2e_bench` (greedy
snapshot, integration only), `compute-sanitizer`, `ncu`/`nsys`, `git worktree`.

---

## Non-goals / hard constraints

- **No CUDA Graphs.** Launch *cost* (graphs) is out of scope this round; we only reduce launch *count*
  (fusion) and improve per-kernel occupancy/bandwidth. (User decision.)
- **No bytes/token change, no requantization, no layout change.** v2 stores the same codes/scales; this
  plan never edits the `.qus` file, the converter, the parser's byte contract, or any qtype. The fused
  block is the matrix v2 already stores (§10); fusion is a pure dispatch-granularity change.
- **Prefill is untouched.** Fusion targets the decode path (`T==1`) only. Prefill (`T>1`, compute-bound
  GEMM via `GenericLowbitGemm`) keeps its current per-projection call sites and separate buffers. The
  fused rowsplit GEMVs are `T=1` specializations.
- **No backward compatibility (AGENTS.md).** Once decode no longer dispatches a per-segment shape,
  delete its kernel, plan, `ShapeFamily`, `LinearPolicyId`, dispatch case, and bench rows outright. No
  shims, no fallback, no dead specialization "just in case." The generic `ROW_SPLIT` GEMV stays the
  correctness fallback for unspecialized shapes.
- **Correctness invariants.**
  - Linear kernel correctness is judged by the fp64/tolerance oracle in `qus_linear_test`. The old
    per-segment reduction order is not a compatibility contract.
  - Phase-2 tuning may reorder reductions when that improves the kernel. The numeric gate is the fp64
    oracle; token identity is not an optimization correctness gate.
- **No MTP/Vision, no attention-algorithm change, no KV/state-shape change.**

## Execution mode

Three execution tiers, matched to task type:

1. **Task 0 — coordinator executes directly (no subagent).** A one-line signature change; dispatching a
   subagent would cost more than doing it.
2. **Task 1 — one atomic subagent, one worktree, one gate, one review.** The fusion is a
   refactor/migration whose intermediate states (half-rewired schedule, deleted-but-still-referenced
   kernel) have no meaningful build/test/review boundary. So it is **not** split into sub-tasks: a
   single implementer subagent does the whole correctness refactor in one worktree with **no
   intermediate verification, commit, or review**, then it is verified **once** against the full gate
   and reviewed **once**. The implementation sequence inside Task 1 is guidance, not verification
   boundaries. Task 1 delivers *correct, structurally-faster* kernels (fewer launches, larger-`N`
   occupancy, no memcpy) — **not** bandwidth-optimal ones.
3. **Phase 2 — parallel per-kernel tuning loops, one git worktree each.** Optimization tasks have a
   sharp target and a sharp boundary (one kernel file, one roofline), so they split cleanly and run
   **in parallel**. Each loop owns exactly one kernel `.cu/.cuh` and its own ncu artifacts, edits **no**
   shared file (the registry policy IDs are fixed in Task 1; tuning changes only the launcher body), and
   measures with **only** the per-op bench + `ncu` — **never** `qus_e2e_bench`. Because no loop loads
   the 15.4 GB model or KV cache, concurrent loops do not contend for GPU memory; `ncu` replay
   serializes on the SM but that only slows wall-clock, it does not corrupt results. After all loops
   finish, the coordinator merges each single-kernel-file diff and runs the integration gate once.

- **Coordination points (shared files — edited only inside Task 1; never by two parties at once):**
  `include/qus/kernels/linear.h` + `src/kernels/linear/linear.cpp` (Task 0 signature, Task 1 dispatch),
  `src/kernels/linear/plan/linear_plan.{h,cpp}` (Task 1 registry), `bench/linear_op_bench.cu` (Task 1
  shape tables), `include/qus/model/model.h` + `src/model/qwen3_6_27b.cpp` (Task 1 schedule). CMake
  needs **no** edits to add/remove kernels — `src/CMakeLists.txt` globs `kernels/*.cu` with
  `CONFIGURE_DEPENDS`; rebuild reconfigures.
- **Phase-2 tuning-loop rule (retained from the prior plan):** each loop runs on the **strongest
  available model**, iterates profile→optimize→re-profile until its gate, and **discovers its own
  limiter** from `ncu` (the report's "partial-wave underfill" is the starting hypothesis, not a
  prescription).

Performance subagents must read first: `/home/neroued/.cursor/skills/profile-cuda/SKILL.md`,
`/home/neroued/.codex/skills/ncu-kernel-profile/SKILL.md`,
`/home/neroued/.codex/skills/nsys-inference-analysis/SKILL.md`.

## Reference paths

- v2 weights: `out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus`. Git: `master` (worktrees branch off the
  post-Task-1 commit).
- Greedy snapshot authority: `profiles/e2e/m3-output-gate.json`; prompt `bench/fixtures/prompts/cn_short.ids`
  (case `cn_short`), manifest `bench/fixtures/prompts/m2.8-v1.manifest.json`.
- Tokenizer (for nsys end-to-end re-profile): `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16`.
- nsys baseline this plan must beat: decode `60.610 s`, `46.25 tok/s`, rowsplit GEMV `87.34%` of decode
  kernel time, `822816` decode `cudaMemcpy2DAsync` (D2D), `~1413` launches/step.
- Python (any parity/HF diagnostic): `/home/neroued/miniconda3/envs/vllm-bench/bin/python`.

## Model facts this plan depends on (verified in tree + spec §10)

- Decode schedule issues one `kernels::linear()` per projection (`src/model/qwen3_6_27b.cpp`
  `attn_mix` 308–311 q/gate/k/v share `h`; `gdn_mix` 343–345 in_q/in_k/in_v share `h`; `mlp_tail`
  413–414 gate/up share `h`).
- Fused blocks are physically one contiguous `[N_block,K]` matrix; `make_quant_descriptor`
  (`src/core/weight_store.cpp` 193–243) builds each projection `Weight` as a row offset
  (`qdata = base + row_begin·code_row_bytes`, `n = segment.row_count`) into shared planes.
- Block composition / row order (spec §10, lines 374–378, 395–397, 498–506):
  - `ATTN_IN` (group 1, 16 full layers): Q4 `[7168,5120]` = `ATTN_Q[0,6144)·ATTN_K[6144,7168)`;
    Q5 `[7168,5120]` = `ATTN_GATE[0,6144)·ATTN_V[6144,7168)`.
  - `GDN_IN` (group 2, 48 GDN layers): Q4 `[4096,5120]` = `GDN_IN_PROJ_Q[0,2048)·GDN_IN_PROJ_K[2048,4096)`;
    Q5 `[6144,5120]` = `GDN_IN_PROJ_V` (1 segment). `in_z` is **excluded** (standalone, computed late);
    `in_a`/`in_b` `[48,5120]` dense gates are **not** members.
  - `MLP_GATEUP` (group 3, all 64): Q4 `[34816,5120]` = `MLP_GATE[0,17408)·MLP_UP[17408,34816)`.
- `config.h`: `hidden=5120`, `intermediate=17408`, `key_dim=2048`, `value_dim=6144`, `conv_dim=10240`
  (`= 2·key_dim + value_dim`, GDN `qkv` buffer = `[q(2048)|k(2048)|v(6144)]`), `q_size=6144`,
  `kv_size=1024`, `head_dim=256`, `n_q=24`, `n_kv=4`. (`q_proj_out=12288` is the **HF-side** combined
  q+gate width, not a v2 block — ignore for fusion.)
- GEMV kernels are warp-per-row, `grid = ceil(N/4)`, `kN` a compile-time constant asserted at launch
  (`linear_rowsplit_gemv_mlp_gate_up.cu` 30–33, 79–83). Larger `N` ⇒ larger grid ⇒ closer to a full
  wave — this is the structural fusion win that fixes the ncu `<1 wave/SM` underfill on small-`N`
  members, available immediately from a correct clone (before any tuning).
- `gdn_mix` does 6 `cudaMemcpy2DAsync`/GDN layer (`copy_bf16_block` ×3 pack 348–350,
  `extract_bf16_block` ×3 unpack 372–374) = 288/step = the `822816` D2D memcpys.
- `out` shape `[5120,6144]` Q5 serves **both** `attn.o_proj` (`a.view({q_size,T})→[5120,T]`) and
  `gdn.out_proj` (`on.view({value_dim,T})→[5120,T]`); one tuned kernel covers both.
- Workspace convention is explicit `WorkspaceArena& ws` for scratch-needing ops (`gqa_attention_decode`,
  `gated_delta_rule_*`), consumed via `ArenaScope` RAII + `ws.alloc(...)`
  (`src/kernels/wrapper/gqa_attention.cpp` 20–29, 157–165).
- `Tensor` supports `slice(dim, start, count)` and `view({...})`; at `T==1` a `slice(0, …)` row-range of
  `[N,1]` is contiguous, so `.view({head_dim,heads,1})` is valid. `linear()` takes `Tensor& out`
  (lvalue) — bind sub-views to **named** locals before passing them as the output.

---

## Task 0 — `linear()` gains an explicit `WorkspaceArena&` (coordinator, direct; no subagent)

Thread the house workspace convention into the one `linear()` verb so split-K loops can declare scratch
explicitly. No behavior change. The coordinator does this directly and commits before dispatching Task 1.

**Files:**
- Modify: `include/qus/kernels/linear.h`, `src/kernels/linear/linear.cpp` (signature only; ignore `ws`
  in every current plan).
- Modify call sites: `src/model/qwen3_6_27b.cpp` (pass `work_` at every `kernels::linear(...)`),
  `bench/linear_op_bench.cu`, `bench/linear_bench.cu`, `tests/kernels/test_linear.cpp` (each creates one
  small `qus::WorkspaceArena ws(64ULL << 20);` and passes it).

**New signature:**

```cpp
// include/qus/kernels/linear.h
void linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws, cudaStream_t stream);
```

**DoD:**
- [ ] `cmake --build build -j` clean (lib + benches + tests).
- [ ] `ctest --test-dir build` shows **no new** failures vs the pre-task baseline (record it first).
- [ ] `rg -n "kernels::linear\(" src bench tests` — every call passes a `WorkspaceArena&`.
- [ ] Commit (this is the base commit Phase-2 worktrees branch from, after Task 1).

---

## Task 1 — Fusion correctness refactor (atomic; one subagent, one worktree, one gate, one review)

Implement the entire decode fusion + GDN direct-write + superseded-kernel deletion as **one change**.
The tree may be non-building partway through; do not attempt intermediate builds/tests/reviews. Deliver
**correct** fused kernels (clone the existing warp-per-row template with the larger `kN`) — optimality
is Phase 2.

**Files (the whole fusion surface):**
- Weight store: `include/qus/core/weight_store.h`, `src/core/weight_store.cpp`.
- Kernels (create, correct/untuned): `src/kernels/linear/gemv/linear_rowsplit_gemv_mlp_gate_up_34816.{cuh,cu}`
  (Q4, `kN=34816`), `linear_rowsplit_gemv_attn_in_7168.{cuh,cu}` (Q4 **and** Q5, `kN=7168`),
  `linear_rowsplit_gemv_gdn_in_qk_4096.{cuh,cu}` (Q4, `kN=4096`).
- Registry/dispatch: `src/kernels/linear/plan/linear_plan.{h,cpp}`, `src/kernels/linear/linear.cpp`.
- Launcher `ws`-normalization (so Phase-2 tuning never touches a signature/dispatch): the surviving
  **existing** specialized rowsplit launchers `linear_rowsplit_gemv_mlp_down.{cuh,cu}`,
  `linear_rowsplit_gemv_out_6144.{cuh,cu}`, `linear_rowsplit_gemv_proj_6144.{cuh,cu}` (Q5),
  `linear_rowsplit_gemv_lm_head.{cuh,cu}` (Q6) gain a `WorkspaceArena& ws` parameter (body unchanged,
  `ws` initially unused), matching the new fused launchers.
- Schedule: `include/qus/model/model.h`, `src/model/qwen3_6_27b.cpp`.
- Delete (superseded decode kernels): `linear_rowsplit_gemv_attn_kv_1024.{cuh,cu}`,
  `linear_rowsplit_gemv_gdn_qk_2048.{cuh,cu}`, `linear_rowsplit_gemv_mlp_gate_up.{cuh,cu}` (kN=17408,
  superseded by 34816); delete the **Q4** kernel+launch from `linear_rowsplit_gemv_proj_6144.{cuh,cu}`
  (keep Q5 — still used by `in_v`/`in_z`).
- Bench: `bench/linear_op_bench.cu` (add fused shapes, remove deleted shapes).
- Tests: `tests/test_weight_store.cpp`, `tests/test_weight_store_real.cpp`, `tests/kernels/test_linear.cpp`.

**Reading list:** spec §6/§7/§10 (fusion records, group ids, block→segment tables, `source_kind` rule);
`src/model/qwen3_6_27b.cpp` (`bind()` 237–291, `attn_mix` 293–331, `gdn_mix` 333–401, `mlp_tail`
403–422); `src/core/weight_store.cpp` `make_quant_descriptor` (193–243) + `is_quant_layout` loop
(331–339); `include/qus/core/weight_store_parser.h` (`ParsedQ5090Tensor.fusion_group_id/fusion_index/
segment_count/shape`); `linear_rowsplit_gemv_mlp_gate_up.cu` (template to clone) +
`linear_rowsplit_gemv_proj_6144.cu` (Q4+Q5-in-one-file pattern); `linear_plan.cpp`
(`classify_shape`/`resolve_plan`/`policy_name`); `linear.cpp` dispatch; `silu_and_mul.h`,
`sigmoid_gate_mul.h`, `causal_conv1d.h`, `l2norm.h` (confirm `T==1` view inputs).

**Implementation sequence (ordered guidance; not verification boundaries):**

1. **Weight store — expose block-level fused `Weight`s.** Add to `weight_store.h`:

```cpp
struct FusedBlockRecord {
    ModuleKind    module       = ModuleKind::TextCore;
    std::uint16_t group_id     = 0;
    std::uint16_t fusion_index = 0;
    std::uint32_t source_layer = kQ5090NoLayer;
    Weight        weight;
};
// public:
const Weight* qfused(ModuleKind module, std::uint16_t group_id, std::uint16_t fusion_index,
                     std::uint32_t source_layer) const noexcept;
// private:
std::vector<FusedBlockRecord> fused_;
```
   In `load()`, inside the `is_quant_layout(tensor.layout)` branch (after the per-segment records),
   emit one block-level record per grouped block via a synthetic full-block segment:

```cpp
if (tensor.fusion_group_id != 0) {
    ParsedQ5090Segment block_seg{};
    block_seg.source_kind  = 0;                  // OTHER: block identity is the group (spec §4)
    block_seg.source_layer = tensor.source_layer;
    block_seg.row_begin    = 0;
    block_seg.row_count    = tensor.shape[0];     // full block N
    next_fused.push_back(FusedBlockRecord{
        tensor.module_kind, static_cast<std::uint16_t>(tensor.fusion_group_id),
        static_cast<std::uint16_t>(tensor.fusion_index), tensor.source_layer,
        make_quant_descriptor(tensor, block_seg, payload)});
}
```
   Commit `next_fused` into `fused_` at the end of `load()`; clear it in `clear()`. `qfused` is the
   linear scan mirroring `qweight(module, source_kind, source_layer)`. Assert (fail loud) `fusion_index
   < group block_count` and that the block's `source_layer` matches the `FusionGroupRecord`; if a fused
   block's `TensorEntry.source_layer` is not the layer, key off the `FusionGroupRecord` instead (decide
   from the real file).

2. **Correct (untuned) fused decode GEMV kernels.** Clone the existing warp-per-row template with the
   larger compile-time `kN` (and, for `attn_in_7168`, a Q4 and a Q5 kernel in one file like
   `proj_6144`). Launch signature `(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
   cudaStream_t)`; assert `w.n==kN && w.k==5120`; write `out[row]` for `row∈[0,kN)` (the block's
   segment-concatenated order). These are correct and already structurally faster (larger grid); they
   are not yet bandwidth-optimal.

3. **Registry + dispatch (and freeze the tuning API).** `linear_plan.h`: add
   `ShapeFamily::{MlpGateUp34816x5120, AttnInQKV7168x5120, GdnInQK4096x5120}` and
   `LinearPolicyId::{MlpGateUp34816Q4RowsplitGemv, AttnInQKV7168Q4RowsplitGemv,
   AttnInQKV7168Q5RowsplitGemv, GdnInQK4096Q4RowsplitGemv}`. `linear_plan.cpp`: add `{34816,5120}`,
   `{7168,5120}`, `{4096,5120}` to `classify_shape`; add `resolve_plan` T1 entries `(Q4,34816)`,
   `(Q4,7168)`, `(Q5,7168)`, `(Q4,4096)`; add `policy_name`/`shape_name` strings. `linear.cpp`: include
   the three new `.cuh`; add the four new dispatch cases; and **forward `ws` to every specialized
   rowsplit launcher** — change each `..._launch(x, w, out, stream)` call (new and surviving:
   `mlp_down`, `out_6144`, `proj_6144` Q5, `lm_head` Q6) to `..._launch(x, w, out, ws, stream)`. After
   this, the policy IDs, the `resolve_plan` mapping, and every rowsplit launch signature are **frozen**;
   a Phase-2 loop only rewrites a launcher *body* (and can pull split-K scratch from `ws` via
   `ArenaScope`) without touching `linear_plan.*`, `linear.cpp`, or the schedule.

4. **Schedule rewiring (decode `T==1` branch; keep the existing body as the prefill `else`).**
   `bind()` (group ids per §7: `ATTN_IN=1, GDN_IN=2, MLP_GATEUP=3`; existing per-segment bindings stay
   for prefill):

```cpp
out.qkv_q4   = require_weight_fused(weights_, /*ATTN_IN*/1, 0, source_layer, "attn.qkv.q4");   // [7168,5120] Q4
out.gatev_q5 = require_weight_fused(weights_, /*ATTN_IN*/1, 1, source_layer, "attn.gatev.q5"); // [7168,5120] Q5
out.in_qk_q4 = require_weight_fused(weights_, /*GDN_IN*/2, 0, source_layer, "gdn.in_qk.q4");    // [4096,5120] Q4
m.gate_up    = require_weight_fused(weights_, /*MLP_GATEUP*/3, 0, source_layer, "mlp.gate_up"); // [34816,5120] Q4
```
   New `model.h` fields: `MlpW.gate_up`, `FullLayerW.{qkv_q4, gatev_q5}`, `GdnLayerW.in_qk_q4`.
   `require_weight_fused` = `qfused` + null-check (mirror `require_weight`).

   `mlp_tail` decode branch (drop the leading `(void)ph;`):

```cpp
if (ph == Phase::Decode) {
    Tensor gu = work_.alloc(DType::BF16, {2 * kCfg.intermediate, 1});      // 34816
    kernels::linear(h, *m.gate_up, gu, work_, s);
    Tensor g = gu.slice(0, 0, kCfg.intermediate);                          // [17408,1]
    Tensor u = gu.slice(0, kCfg.intermediate, kCfg.intermediate);
    Tensor a = work_.alloc(DType::BF16, {kCfg.intermediate, 1});
    kernels::silu_and_mul(g, u, a, s);
    Tensor d = work_.alloc(DType::BF16, {kCfg.hidden, 1});
    kernels::linear(a, *m.down, d, work_, s);
    kernels::residual_add(d, x, s);
    return;
}
// ... existing per-projection prefill body unchanged ...
```

   `attn_mix` decode branch (one Q4 q+k GEMV + one Q5 gate+v GEMV into combined `[7168,1]` buffers;
   `q/k/gate/v` are views; `qn/kn/a` allocated as today):

```cpp
if (ph == Phase::Decode) {
    Tensor qk    = work_.alloc(DType::BF16, {kCfg.q_size + kCfg.kv_size, 1});  // 7168
    Tensor gatev = work_.alloc(DType::BF16, {kCfg.q_size + kCfg.kv_size, 1});
    kernels::linear(h, *w.qkv_q4,   qk,    work_, s);
    kernels::linear(h, *w.gatev_q5, gatev, work_, s);
    Tensor q    = qk.slice(0, 0, kCfg.q_size).view({kCfg.head_dim, kCfg.n_q, 1});
    Tensor k    = qk.slice(0, kCfg.q_size, kCfg.kv_size).view({kCfg.head_dim, kCfg.n_kv, 1});
    Tensor gate = gatev.slice(0, 0, kCfg.q_size).view({kCfg.head_dim, kCfg.n_q, 1});
    Tensor v    = gatev.slice(0, kCfg.q_size, kCfg.kv_size).view({kCfg.head_dim, kCfg.n_kv, 1});
    // q_norm/k_norm rmsnorm, rope, gqa_attention_decode, sigmoid_gate_mul(gate, a), o_proj, residual
    // — identical to today, on these views.
    ...
    return;
}
```

   `gdn_mix` decode branch (fused q+k and `in_v` write straight into `qkv`; `qc/kc/vc` become views of
   `qkv_c` — **no** `copy_bf16_block`/`extract_bf16_block`):

```cpp
if (ph == Phase::Decode) {
    Tensor qkv    = work_.alloc(DType::BF16, {kCfg.conv_dim, 1});            // 10240
    Tensor qk_out = qkv.slice(0, 0, 2 * kCfg.key_dim);                       // [4096,1] view, named lvalue
    Tensor v_out  = qkv.slice(0, 2 * kCfg.key_dim, kCfg.value_dim);          // [6144,1] view, named lvalue
    kernels::linear(h, *w.in_qk_q4, qk_out, work_, s);                       // q|k -> qkv[0,4096)
    kernels::linear(h, *w.in_v,     v_out,  work_, s);                       // v   -> qkv[4096,10240)
    Tensor a = work_.alloc(DType::BF16, {kCfg.gdn_v_heads, 1});
    Tensor b = work_.alloc(DType::BF16, {kCfg.gdn_v_heads, 1});
    kernels::linear(h, *w.in_a, a, work_, s);
    kernels::linear(h, *w.in_b, b, work_, s);
    Tensor qkv_c = work_.alloc(DType::BF16, {kCfg.conv_dim, 1});
    kernels::causal_conv1d_decode(qkv, *w.conv1d, conv_state, qkv_c, s);
    // gdn_gating(a,b,a_log,dt_bias -> g,beta) as today
    Tensor qc = qkv_c.slice(0, 0, kCfg.key_dim);
    Tensor kc = qkv_c.slice(0, kCfg.key_dim, kCfg.key_dim);
    Tensor vc = qkv_c.slice(0, 2 * kCfg.key_dim, kCfg.value_dim);
    // l2norm(qc.view(...)), l2norm(kc.view(...)), recurrent, in_z (standalone, late), gdn_norm,
    // out_proj, residual — identical to today, on these views.
    ...
    return;
}
```
   `in_z` stays standalone (`linear(h, *w.in_z, z_flat, work_, s)`, computed late, §10).

5. **Delete superseded decode kernels + registry entries.** Remove the kernels listed in **Files**;
   remove `ShapeFamily::{AttnKV1024x5120, GdnQK2048x5120, MlpGateUp17408x5120}`,
   `LinearPolicyId::{AttnKV1024Q4/Q5RowsplitGemv, GdnQK2048Q4RowsplitGemv, Proj6144Q4RowsplitGemv,
   MlpGateUpQ4RowsplitGemv}`, their `classify_shape`/`resolve_plan`/name entries, and the `linear.cpp`
   includes+cases. `Proj6144x5120` Q5, `Out5120x6144`, `MlpDown5120x17408`, `LmHead`, `DenseCtrl48x5120`
   stay.

6. **Bench.** `linear_op_bench.cu`: add `{"MlpGateUp34816x5120",34816,5120}`,
   `{"AttnInQKV7168x5120",7168,5120}`, `{"GdnInQK4096x5120",4096,5120}` to `kShapes` + matching
   `kTask2Targets` (Q4 34816; Q4+Q5 7168; Q4 4096); remove the deleted shape/target rows.

**Definition of done (run ONCE, all must pass):**
- [ ] `cmake --build build -j` clean (lib + benches + tools + tests).
- [ ] **Weight store:** `tests/test_weight_store_real.cpp` asserts
      `qfused(TextCore,3,0,0)→n==34816,Q4`, `qfused(TextCore,1,0,3)→n==7168,Q4`,
      `qfused(TextCore,1,1,3)→n==7168,Q5`, `qfused(TextCore,2,0,0)→n==4096,Q4`, each with
      `qdata/scales` equal to its first-segment `qweight`; no fused record for standalone blocks
      (`o_proj`, `mlp.down`). `tests/test_weight_store.cpp` (fixture): `qfused.n` == Σ segment
      `row_count`. `ctest -R weight_store` green.
- [ ] **Kernel numerics:** `tests/kernels/test_linear.cpp` fp64 oracle passes for `(Q4 34816×5120,
      Q4 7168×5120, Q5 7168×5120, Q4 4096×5120)`. `compute-sanitizer --tool memcheck`
      clean on `qus_linear_test`.
- [ ] **Model:** `ctest -R model` green (`qus_model_bind_test`, `qus_model_blocks_test`).
- [ ] **G-SNAPSHOT smoke:** `qus_e2e_bench` on v2 runs the `cn_short` fixture without crashing:

```bash
cmake --build build --target qus_e2e_bench -j
./build/bench/qus_e2e_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
  --output-json out/e2e.v2.fusion.json \
  --fixture-manifest bench/fixtures/prompts/m2.8-v1.manifest.json \
  --case cn_short:bench/fixtures/prompts/cn_short.ids:96 \
  --warmup-repeats 1 --repeats 1 --max-ctx 8192 --device 0 \
  --stop-token-id 248046 --stop-token-id 248044
# Inspect generated_token_ids if needed; token identity is not a correctness gate.
```
- [ ] `compute-sanitizer --tool memcheck` clean over a short decode (slice/view bounds, `qkv` direct
      write, `qkv_c` views).
- [ ] **Structural win (nsys decode re-profile, same invocation as the baseline report):** GDN
      `cudaMemcpy2DAsync` count `822816 → 0`; rowsplit GEMV launches/step drop ~30% (gate/up 128→64,
      gdn q/k 96→48, attn q/k/v 4→2 per full layer). (No roofline gate yet — that is Phase 2.)
- [ ] **Audit + full suite:**
```bash
rg -n "AttnKV1024|GdnQK2048|MlpGateUp17408|Proj6144Q4|attn_kv_1024|gdn_qk_2048|mlp_gate_up\b" \
   src include bench tests   # only proj_6144 Q5 (no Q4); no 1024/2048/17408 leftovers
rg -n "rowsplit_gemv_.*_launch\(" src/kernels/linear/linear.cpp   # every call ends in ", ws, stream)"
cmake --build build -j && ctest --test-dir build   # green (baseline reds cleared)
```
- [ ] **Tuning API frozen:** every surviving specialized rowsplit launcher signature is
      `(const Tensor&, const Weight&, Tensor&, WorkspaceArena&, cudaStream_t)`; the dispatch forwards
      `ws` to all of them. (This is the precondition that lets Phase-2 loops touch only a kernel body.)
- [ ] Commit. **This commit is the base for all Phase-2 worktrees.**

**Review (ONCE, after the gate — independent; touches CUDA kernels, GPU memory, ABI, numerics):**
1. **Numerical** — fused GEMV math passes the fp64/tolerance oracle; GDN direct-write keeps tensor
   shapes and bounds correct.
2. **CUDA memory/lifetime** — combined-buffer `slice/view` bounds at `T==1`, `qkv`/`qkv_c` offsets,
   arena `mark/rewind` scoping; `compute-sanitizer` evidence.
3. **Format/ABI** — `qfused` keying vs spec §6/§7/§10 (`fusion_index` Q4=0/Q5=1, group ids), block
   `source_kind=OTHER` (§4).
4. **Scope** — prefill path untouched; all superseded decode kernels/plans/shapes removed; no CUDA
   graphs; `.qus`/parser byte contract untouched.

> **GATE — what Phase 1 hands to every tuning loop (so a loop edits *only* its kernel file):**
> Phase 2 does not begin until Task 1 is green and reviewed. After Task 1, a tuning loop can rely on all
> of the following being already in place — it never touches any of them:
> - **Signature** — every specialized rowsplit launcher already takes `(x, w, out, ws, stream)` (new
>   fused **and** existing `mlp_down`/`out_6144`/`proj_6144` Q5/`lm_head` Q6); split-K scratch comes from
>   `ws` via `ArenaScope`, so no signature change is ever needed.
> - **Dispatch** — `linear.cpp` already forwards `ws` to that launcher; `linear_plan.*` policy IDs and
>   `resolve_plan`/`classify_shape` already route the target shape to it. No registry/dispatch edit.
> - **Schedule** — `qwen3_6_27b.cpp` already calls the kernel (via fused block `Weight`s + combined
>   buffers) and passes `work_` as `ws`. The schedule is agnostic to a kernel's internal strategy.
> - **Weights** — `qfused`/`qweight` already expose the exact `Weight` (planes, `n`, `k`) the kernel reads.
> - **Measurement + correctness** — `qus_linear_op_bench` (with `ws`) already measures the target shape,
>   and `qus_linear_test` (fp64/tolerance oracle, with `ws`) already covers it. No bench/test plumbing
>   change.
>
> A loop therefore rewrites only its launcher *body* (and may add a private `__global__` reduction kernel
> inside its own `.cu`). The frozen API is exactly what makes the loops independent and parallel-safe.

---

## Phase 2 — Parallel per-kernel tuning loops (one git worktree each)

Decode is weight-bandwidth bound (~15.4 GB streamed/token). Each decode-critical GEMV is tuned **on its
own** toward its cold-cache DRAM ceiling — the new fused kernels (correct but untuned from Task 1) and
the two unfusable underfilled standalone kernels. *How* to reach the roofline is each loop's own
profiling to determine.

### Two non-negotiable rules

1. **The coordinator must NOT prescribe optimization direction.** When dispatching a loop, the
   coordinator hands the subagent only: the *target* (ShapeFamily + qtype variant(s) + owned kernel
   file), the *correctness invariants* (fp64/tolerance oracle only), the
   *gate metric + how to measure it* (cold-cache DRAM% via `qus_linear_op_bench` + `ncu`), the
   *recorded baseline*, and the *iteration protocol*. It must **not** suggest a limiter, a lever, a
   block/grid shape, split-K, a layout, or any hypothesis. The subagent discovers the dominant limiter
   from its **own** `ncu` evidence, forms its own hypothesis, implements, re-profiles, and reports back
   what it found and changed. (If the coordinator catches itself about to write "try split-K / increase
   occupancy / vectorize" into a dispatch prompt — stop; that is a rule violation.)
2. **Every change is evidence-backed: baseline before, comparison after.** No loop may claim a win
   without a **before** record (the post-Task-1 kernel) and an **after** record (the tuned kernel)
   measured on the **same rig** (cold-cache `qus_linear_op_bench` + `ncu`, same flags), plus the
   per-round `ncu` before/after for the one limiter changed that round. This comparison is the
   **evidence the coordinator uses to accept the merge** — a loop with no measured improvement and no
   documented non-bandwidth wall is **not merged**.

### Parallelization (worktrees)

Loops are independent because (a) each owns a disjoint kernel `.cu/.cuh`, (b) the registry/launch APIs
are frozen after Task 1 so **no loop edits a shared file**, and (c) each loop runs **only** the per-op
bench + `ncu` + the fp64 oracle for its shape — **never** `qus_e2e_bench`, so no loop loads the model
or KV cache and concurrent loops do not contend for GPU memory.

Coordinator, per loop `<tag>`:

```bash
git worktree add ../qus-tune-<tag> <task1-commit> -b tune/<tag>
# dispatch the tuning subagent into ../qus-tune-<tag> with its own build dir
```
Each subagent builds in its own worktree, iterates, and leaves a **single-kernel-file** diff +
`profiles/ncu-linear-v2/<tag>/` artifacts. When all loops finish, the coordinator validates each loop's
evidence (below), merges each branch (disjoint files ⇒ no conflicts), and runs the **Integration gate**
once.

### Coordinator: baseline snapshot + tuning status ledger (before dispatching any loop)

1. **Record the canonical baseline** once, off the Task-1 commit, on the real GPU:

```bash
./build/bench/qus_linear_op_bench --all-targets --csv-out profiles/ncu-linear-v2/baseline/op_bench.csv
# + per-target cold-cache ncu into profiles/ncu-linear-v2/baseline/<tag>.ncu-rep
```
   This is the authoritative before-number every loop's "after" is compared against (same machine
   state, apples-to-apples). Each loop also re-measures at its first profiling round, but the ledger's
   baseline column is this snapshot.

2. **Maintain a status ledger** at `profiles/ncu-linear-v2/tuning_status.md` (a table the coordinator
   owns and updates — never a subagent). It is the single source of truth that prevents parallel loops
   from colliding or losing track of rounds:

```markdown
| tag | shape | qtype | branch | status | rounds | base cold_us | base DRAM% | best cold_us | best DRAM% | Δ vs base | limiter (subagent-found) | evidence dir | merged |
|-----|-------|-------|--------|--------|--------|--------------|------------|--------------|------------|-----------|--------------------------|--------------|--------|
| mlp_gate_up_34816 | MlpGateUp34816x5120 | Q4 | tune/mlp_gate_up_34816 | pending | 0 | — | — | — | — | — | — | profiles/ncu-linear-v2/mlp_gate_up_34816/ | no |
```
   - **status** ∈ `pending → baselined → in-progress → gate-met | non-bw-wall | blocked → merged`.
   - The coordinator updates a row when it: records the baseline (`baselined`), dispatches/resumes a
     loop (`in-progress`, bump `rounds`), receives a subagent report (record `best *`, `Δ`, the
     subagent-found `limiter`), validates evidence and merges (`merged`), or hits a blocker (`blocked`).
   - **Parallel-safety:** before dispatching, confirm the row is `pending`/`baselined` and its branch
     isn't already checked out; never run two loops against the same `tag`/kernel file; the ledger makes
     a half-finished or stalled loop visible instead of silently lost.

### Shared loop protocol (identical for every loop; strongest model; iterate to gate)

0. **Record this loop's baseline (round 0, before any change).** Cold-cache (L2-flushed)
   `./build/bench/qus_linear_op_bench --shape <ShapeFamily> --qtype <Q4|Q5>` + `ncu`, saved to
   `profiles/ncu-linear-v2/<tag>/round0_baseline.*`. Must reproduce the coordinator's canonical baseline
   for this shape (same kernel, off Task-1 commit).
1. **Profile + discover.** From this loop's **own** `ncu` evidence, identify the single dominant limiter
   and form one hypothesis. (The coordinator gave no limiter/lever — discovery is the subagent's job.)
2. **Change** only this loop's kernel file. Do **not** edit `linear_plan.*` or `linear.cpp` (policy
   frozen). If split-K is used, scratch comes from the Task-0 `ws` param via `ArenaScope`.
3. **Re-profile + verify:** re-run `ncu` (save before/after as `round<N>_<limiter>.*` under the loop's
   evidence dir); run the fp64/tolerance oracle for this shape (`ctest -R linear` filtered, or the
   op-level oracle). Revert anything that fails the oracle.
4. **Iterate** (one limiter/round) until the gate or a documented non-bandwidth wall.
5. **Report.** Produce a before→after comparison table (`round0` vs best: cold_us, achieved GB/s,
   DRAM%, % of roofline) + a one-line-per-round log of `{limiter found → change → result}`, saved to
   `profiles/ncu-linear-v2/<tag>/summary.md`. This summary + the round artifacts are the merge evidence
   handed to the coordinator.

- **Gate:** cold-cache `dram__throughput.avg.pct_of_peak_sustained_elapsed` materially up and
  approaching the `qus_linear_op_bench` roofline for the shape, **or** `ncu` documents a non-bandwidth
  limiter; the shape's fp64 oracle green. Hot-cache is diagnostic only.

### Tuning targets (each = one worktree; priority = merge order)

| Loop | kind | ShapeFamily (`N×K`) | qtype | owned kernel file `…_<tag>` |
|---|---|---|---|---|
| T1 | fused (new) | `MlpGateUp34816x5120` | Q4 | `_mlp_gate_up_34816` |
| T2 | fused (new) | `AttnInQKV7168x5120` | Q4 + Q5 | `_attn_in_7168` |
| T3 | fused (new) | `GdnInQK4096x5120` | Q4 | `_gdn_in_qk_4096` |
| T4 | split-K | `MlpDown5120x17408` | Q5 | `_mlp_down` |
| T5 | split-K | `Out5120x6144` | Q5 | `_out_6144` |
| T6 (optional) | re-tune | `Proj6144x5120` / `LmHead248320x5120` | Q5 / Q6 | `_proj_6144` / `_lm_head` |

- T2 tunes both qtypes in one loop (they share the file). T5's `out` kernel serves both `attn.o_proj`
  and `gdn.out_proj`.
- T6 only if the post-Task-1 nsys shows `in_v`/`in_z` (Proj6144 Q5) or `lm_head` Q6 still holding a
  material decode share; otherwise skip.

**Per-loop DoD (the evidence bundle the subagent returns):**
- `round0_baseline` + best-round `ncu`/op-bench artifacts and `summary.md` (before→after table +
  per-round limiter→change→result log) committed under `profiles/ncu-linear-v2/<tag>/`.
- Cold-cache DRAM% for the target **materially beats its round-0 baseline and approaches the roofline**,
  **or** `ncu` documents a non-bandwidth wall (with the evidence).
- The limiter(s) addressed were **subagent-discovered from `ncu`** (the summary names them); no
  coordinator-supplied direction.
- The shape's fp64 oracle is green; only the owned kernel file changed (`git diff --stat` on the branch
  shows one `.cu`/`.cuh` pair).

### Coordinator: accept-and-merge (per loop, evidence-gated)

For each finished loop, **before** merging its branch: open `summary.md`, confirm (a) before→after shows
a real cold-cache improvement (or a documented non-bw wall), (b) the limiter was subagent-discovered,
(c) the fp64 oracle passed, (d) `git diff --stat` touches only the owned kernel file. Record `best *`,
`Δ`, the limiter, and `merged=yes` in the ledger, then merge. A loop failing (a)–(d) is **not merged** —
it is re-dispatched (resume the same subagent with its own prior evidence) or marked `blocked`.

### Integration gate (coordinator, after merging all loop branches — run ONCE, sequentially)

- [ ] `cmake --build build -j` clean; `ctest --test-dir build` fully green.
- [ ] **Numerics:** fp64/tolerance oracle green for every tuned shape.
- [ ] **G-SNAPSHOT smoke:** re-run the Task-1 `qus_e2e_bench` command and confirm the model run
      completes. Greedy token identity is not a Phase-2 correctness gate.
- [ ] `compute-sanitizer --tool memcheck` clean over a short decode (covers every new split-K scratch
      path).
- [ ] **nsys decode re-profile:** measurable decode-time / tok-s improvement attributable to the tuned
      kernels; name the new top decode bottleneck.

### Phase 2 review (ONCE, after integration — independent; performance evidence + numerics)

Every speedup claim is cold-cache `ncu`/`nsys`-backed (not hot-cache) with a **round-0 baseline →
best-round comparison** in each loop's `summary.md`; each loop shows one-limiter-per-round before/after
artifacts and a **subagent-discovered** limiter (verify no coordinator dispatch prescribed a
direction); the fp64/tolerance oracle stayed green on every loop; the ledger (`tuning_status.md`)
accounts for every target's status/rounds/Δ and matches what was merged; integration smoke completed;
split-K scratch is declared through `ws`/`ArenaScope` and
`compute-sanitizer` is clean; the frozen registry/API was not edited by any loop.

---

## Self-review (against the approved design + the user's task-shape feedback)

- **Task shape:** Task 0 = coordinator-direct (no subagent) ✓; fusion = one atomic refactor task,
  verified+reviewed once, no fragile intermediate boundaries ✓; the new fused kernels are written
  *correct* in Task 1 and *tuned* in Phase 2 ✓; optimization = independent per-kernel loops,
  parallelized across git worktrees, never running full-model inference ✓.
- **Spec coverage:** block-level `qfused` ✓; fused kernels + registry ✓; schedule fusion + GDN
  direct-write ✓; superseded deletion ✓; tuning of fused kernels + split-K (`mlp_down`, `out`) ✓; no
  CUDA graphs ✓; `linear()` gains `ws`, otherwise API-unchanged ✓.
- **Type consistency:** `qfused(module, group_id, fusion_index, source_layer)`, `MlpW.gate_up`,
  `FullLayerW.{qkv_q4,gatev_q5}`, `GdnLayerW.in_qk_q4`, the new `ShapeFamily`/`LinearPolicyId` names,
  and the frozen launch signatures are used identically across Task 1 and Phase 2.
- **Decode-only fusion:** every fused call site is inside a `ph == Phase::Decode` branch; prefill
  per-segment bindings retained.
- **Parallel-safety:** loops touch disjoint kernel files, edit no shared registry/schedule file, and
  run only the per-op bench + `ncu` + per-shape fp64 oracle (no model load) ⇒ no GPU-memory contention;
  e2e/nsys run once at the sequential integration gate.
- **Evidence + control (user feedback):** coordinator records a canonical baseline before any loop;
  each loop returns a round-0→best comparison as the merge gate; the coordinator prescribes **no**
  optimization direction (subagent discovers the limiter from `ncu`); a `tuning_status.md` ledger
  tracks per-kernel status/rounds/Δ/merged to keep parallel loops from colliding or stalling silently.
