# MTP GDN Indirect State Refactor Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute this
> plan as one atomic refactor. Steps use checkbox (`- [ ]`) syntax for tracking. Do not use
> subagent-driven development for implementation.

**Goal:** Replace MTP GDN commit-copy state management with device-scalar accepted-slot reads for
both conv and SSM state, while preserving the logical `num_layers * (1 + num_draft_tokens)` slot
model.

**Architecture:** Each GDN layer owns one contiguous conv tensor and one contiguous SSM tensor with
`Slots = 1 + k` slots. Target GDN verify and MTP-enabled fallback decode read their initial state
from `io_.gdn_initial_slot`, then write produced states sequentially from slot 0. The engine
carries the device selector across rounds instead of copying accepted slots into slot 0.

**Tech Stack:** C++17 runtime/model code, CUDA BF16/FP32 kernels, existing `Tensor`,
`DeviceArena`, `WorkspaceArena`, MTP round helpers, and kernel tests.

---

## Execution Mode

Implementation mode: **single-agent atomic refactor**.

Reason: this change intentionally crosses public kernel APIs, runtime round sequencing, model-card
GDN call sites, tests, build registrations, and active MTP docs. The intermediate tree will be
unusable after the API signature changes and before all call sites/tests/docs are updated, so
subagent-driven implementation and task-by-task review are intentionally not used.

Final review mode: after the atomic refactor builds and the targeted verification set has run,
perform one comprehensive final review checklist. Do not perform intermediate implementation
reviews.

## Goal And Non-Goals

Goals:

- Keep logical GDN state capacity as `n_gdn * (1 + k)` slots.
- Ensure each layer has one physical conv tensor and one physical SSM tensor.
- Use one shared device selector for conv and SSM initial state.
- Make target verify read from slot `io_.gdn_initial_slot[0]` and write snapshots to `0..k`.
- Make MTP-enabled fallback T=1 decode read from the selected slot, write the resulting state to
  slot 0, then reset the selector to 0.
- Remove project-owned `gdn_commit` copy code and tests.
- Merge the BF16 GDN recurrent normal/snapshot kernels into one kernel controlled by
  `template <bool Spec>`.
- Keep GDN math, output layout, and BF16/FP32 rounding unchanged except for the intended state
  address changes.

Non-goals:

- No migration path or compatibility alias for `gdn_commit`.
- No generic runtime abstraction for future models.
- No round-level CUDA graph work.
- No tests that only enforce source layout or coverage count.
- No changes to MTP draft algorithm, GQA attention math, q5090 format, or sampling policy.

## New State Contract

Per GDN layer:

```text
conv_states: [conv_dim, 3, Slots] BF16
ssm_states:  [128,128,48,Slots] FP32
Slots = 1 + mtp_draft_tokens
```

Persistent round scalar:

```text
io_.gdn_initial_slot: I32 [1]
```

MTP round semantics:

```text
round start:
    initial_slot = io_.gdn_initial_slot[0]

target verify:
    conv initial state = conv_states[..., initial_slot]
    ssm initial state  = ssm_states[..., initial_slot]
    after-token-j state is written to slot j, j in [0,k]

accept:
    accepted = a
    io_.gdn_initial_slot = a

next round:
    target verify reads slot a
```

Fallback decode semantics when MTP is enabled:

```text
decode_step_one:
    target T=1 GDN reads conv/ssm from io_.gdn_initial_slot
    target T=1 GDN writes after-token state to slot 0
    engine resets io_.gdn_initial_slot = 0 after the decode kernel sequence
```

This precludes the stale-state bug where fallback decode would read slot 0 even though the current
committed GDN state lives in a non-zero accepted slot.

Prefill and MTP-disabled decode semantics:

```text
prefill reset:
    clear slot 0 and set io_.gdn_initial_slot = 0

prefill chunked GDN:
    read/write slot 0 only

MTP-disabled T=1 decode:
    read/write slot 0 only
```

Slot 0 is no longer the cross-round committed invariant while MTP round mode is active. The
committed GDN state is selected by `io_.gdn_initial_slot`.

## Files And Ownership

Atomic implementation may touch all files below. Do not split into independently reviewed commits
unless the full tree remains buildable after each commit.

Core/runtime/model:

- `include/qus/core/state_store.h`
- `src/core/state_store.cpp`
- `include/qus/model/model.h`
- `src/model/qwen3_6_27b.cpp`
- `include/qus/runtime/engine.h`
- `src/runtime/engine.cpp`
- `bench/target_verify_bench.cpp`
- `tests/test_model_blocks.cpp`
- `tests/test_model_bind.cpp`

MTP scalar helpers:

- `include/qus/kernels/mtp_round.h`
- `src/kernels/wrapper/mtp_round.cpp`
- `src/kernels/launcher/mtp_round.h`
- `src/kernels/launcher/mtp_round.cu`
- `src/kernels/kernel/mtp_round.cuh`
- `tests/kernels/test_mtp_round.cpp`

Conv state kernels:

- `include/qus/kernels/causal_conv1d.h`
- `src/kernels/wrapper/causal_conv1d.cpp`
- `src/kernels/launcher/causal_conv1d.h`
- `src/kernels/launcher/causal_conv1d.cu`
- `src/kernels/kernel/causal_conv1d.cuh`
- `tests/kernels/test_causal_conv1d.cpp`

SSM/GDN recurrent kernels:

- `include/qus/kernels/gated_delta_rule.h`
- `src/kernels/wrapper/gated_delta_rule.cpp`
- `src/kernels/launcher/gated_delta_rule.h`
- `src/kernels/launcher/gated_delta_rule_recurrent.cu`
- `src/kernels/kernel/gated_delta_rule_recurrent.cuh`
- `src/kernels/launcher/gated_delta_rule_chunked.cu`
- `tests/kernels/test_gated_delta_rule.cpp`
- `bench/gated_delta_rule_bench.cu`

Delete old commit-copy path:

- `include/qus/kernels/gdn_commit.h`
- `src/kernels/wrapper/gdn_commit.cpp`
- `src/kernels/launcher/gdn_commit.h`
- `src/kernels/launcher/gdn_commit.cu`
- `src/kernels/kernel/gdn_commit.cuh`
- `tests/kernels/test_gdn_commit.cpp`
- `src/CMakeLists.txt`
- `tests/CMakeLists.txt`

Docs:

- `docs/2026-07-03-mtp-spec-decode-overview.md`
- `docs/2026-07-03-mtp-state-management.md`
- `docs/2026-07-03-mtp-round-algorithm.md`
- `docs/2026-07-03-mtp-implementation-requirements.md`
- `docs/2026-07-03-mtp-roadmap.md`

## Atomic Implementation Task

### Task 1: Implement Indirect GDN State End To End

This is one atomic task. Complete all implementation steps before expecting the tree to build.

**Reading list:**

- `AGENTS.md`
- `include/qus/core/state_store.h`
- `src/core/state_store.cpp`
- `include/qus/model/model.h`
- `src/model/qwen3_6_27b.cpp`
- `src/runtime/engine.cpp`
- `src/kernels/kernel/causal_conv1d.cuh`
- `src/kernels/kernel/gated_delta_rule_recurrent.cuh`
- `src/kernels/kernel/mtp_round.cuh`
- `tests/kernels/test_causal_conv1d.cpp`
- `tests/kernels/test_gated_delta_rule.cpp`
- `tests/kernels/test_mtp_round.cpp`
- `tests/test_engine_mtp_e2e.cpp`
- active MTP docs listed above

**Step 1: Add selector state and keep state-store physical layout explicit**

- [ ] Add `Tensor gdn_initial_slot;` to `model::StepState` next to the other round scalars.
- [ ] Update every `StepState{...}` aggregate initializer, not only `Engine::load`.
      Required current known sites:
      - `src/runtime/engine.cpp`
      - `bench/target_verify_bench.cpp`
      - `tests/test_model_blocks.cpp`
      - `tests/test_model_bind.cpp`
- [ ] Allocate `gdn_initial_slot` as `DType::I32, {1}`.
- [ ] In direct model owners that construct `StepState` outside `Engine`, explicitly initialize
      `gdn_initial_slot` to zero before calling model forward/verify paths:
      - `bench/target_verify_bench.cpp` helper/reset path;
      - `tests/test_model_blocks.cpp` helper/reset path;
      - `tests/test_model_bind.cpp` helper/setup path.
- [ ] Update `default_cache_bytes_for()` manual I/O byte accounting and alignment slack for the
      added scalar allocation.
- [ ] Keep `GdnState` per-layer physical tensors as one contiguous conv tensor and one contiguous
      SSM tensor. Add small explicit helpers only if useful:

```cpp
std::int64_t conv_slot_stride_elements() const noexcept; // conv_dim * conv_width
std::int64_t ssm_slot_stride_elements() const noexcept;  // key_head_dim * value_head_dim * value_heads
```

Do not introduce per-slot allocations or pointer arrays.

**Step 2: Add device scalar helper kernels**

- [ ] Add MTP helper wrappers for scalar reset/copy:

```cpp
void mtp_reset_gdn_initial_slot(Tensor& gdn_initial_slot, cudaStream_t stream);
void mtp_set_gdn_initial_slot_from_accepted(const Tensor& accepted, Tensor& gdn_initial_slot,
                                           cudaStream_t stream);
```

- [ ] Validate both tensors as contiguous non-null `I32 [1]`.
- [ ] Implement a tiny kernel or reuse an existing one-line scalar kernel pattern in
      `src/kernels/kernel/mtp_round.cuh`.
- [ ] Use these helpers instead of host copies. Accepted count must stay device-driven.

**Step 3: Change conv snapshot API and kernel addressing**

- [ ] Replace the snapshot public signature with:

```cpp
void causal_conv1d_sequence_snapshot(const Tensor& x, const Tensor& weight,
                                     Tensor& conv_states, const Tensor& initial_slot,
                                     Tensor& out, cudaStream_t stream);
```

- [ ] Validate:
      - `x` BF16 `[C,T]`, `T > 0`;
      - `weight` BF16 `[C,4]`;
      - `conv_states` BF16 `[C,3,Slots]` with `Slots >= T`;
      - `initial_slot` I32 `[1]`;
      - `out` BF16 `[C,T]`;
      - all tensors contiguous and non-null.
- [ ] Pass `slots = conv_states.ne[2]` and `slot_stride = C * 3` to the launcher/kernel.
- [ ] Kernel load rule:

```cpp
const int slot = initial_slot[0];
const int safe_slot = (slot >= 0 && slot < slots) ? slot : 0;
const __nv_bfloat16* init = conv_states + safe_slot * slot_stride;
```

- [ ] Kernel write rule stays sequential:

```cpp
__nv_bfloat16* snapshot = conv_states + t * slot_stride;
```

- [ ] Preserve existing conv math and BF16 rounding.

**Step 4: Merge BF16 SSM recurrent kernels with `template <bool Spec>`**

- [ ] Replace the separate BF16 normal and snapshot kernel bodies with one templated body:

```cpp
template <int HeadDim, bool Spec>
__global__ void gated_delta_rule_recurrent_bf16_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const float* g,
    const float* beta,
    float* ssm_state_or_states,
    const std::int32_t* initial_slot,
    __nv_bfloat16* out,
    std::int64_t T,
    head_map heads,
    float scale,
    std::int64_t state_slot_stride,
    std::int32_t slots);
```

- [ ] `Spec=false` reads/writes a single `[128,128,48]` state exactly as the current recurrent path.
- [ ] `Spec=true` reads initial state from `safe_slot * state_slot_stride` and writes snapshots to
      `t * state_slot_stride`.
- [ ] Use `HeadDim == 128` for the recurrent head dimension template parameter. Use `slots` or
      `slot_count` for `ssm_states.ne[3]`; do not reuse `S` for both concepts.
- [ ] Compute `state_slot_stride = ssm_states.ne[0] * ssm_states.ne[1] * ssm_states.ne[2]` in
      elements, not bytes.
- [ ] Use this public snapshot signature:

```cpp
void gated_delta_rule_recurrent_snapshot(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& g, const Tensor& beta, float scale,
                                         WorkspaceArena& ws, Tensor& ssm_states,
                                         const Tensor& initial_slot, Tensor& out,
                                         cudaStream_t stream);
```

- [ ] Validate snapshot inputs:
      - `q [128,16,T]`, `k [128,16,T]`, `v [128,48,T]` BF16;
      - `g [48,T]`, `beta [48,T]` FP32;
      - `ssm_states [128,128,48,Slots]` FP32 with `Slots >= T`;
      - `initial_slot I32 [1]`;
      - `out [128,48,T]` BF16.
- [ ] Keep `gated_delta_rule_chunked` prefill behavior by passing a slot-0 view to the
      `Spec=false` path.
- [ ] Remove dead FP32 recurrent launcher/kernel declarations if nothing uses them.

**Step 5: Thread selector through target GDN model paths**

- [ ] In `Qwen3_6_27B::gdn_mix(Phase::Verify)`, pass `io_.gdn_initial_slot` to both snapshot
      kernels.
- [ ] In MTP-enabled `Phase::Decode` fallback, keep the existing decode projection/gating/output
      code path and replace only conv/SSM state addressing:
      - conv reads from `state_.conv[gidx]` slot `io_.gdn_initial_slot[0]` and writes after-token
        state to slot 0;
      - SSM reads from `state_.ssm[gidx]` slot `io_.gdn_initial_slot[0]` and writes after-token
        state to slot 0.
      Do not route fallback through the non-decode/prefill GDN branch because that would change
      linear/gating/output kernels instead of only state addressing.
- [ ] In MTP-disabled `Phase::Decode`, keep the existing fast slot-0 decode path.
- [ ] Keep `Phase::Prefill` slot-0-only.
- [ ] Ensure conv and SSM always use the same selector in every selected-state path.

**Step 6: Update engine sequencing**

- [ ] In `Engine::load`, zero `io_.gdn_initial_slot` with the other round scalars.
- [ ] In `Engine::prefill`, reset `io_.gdn_initial_slot` to zero after `state_->reset(ctx_->stream)`
      and before/with the target prefill work.
- [ ] In `Engine::decode_round`, after `mtp_accept_tokens`, update
      `io_.gdn_initial_slot = io_.accepted` on device.
- [ ] Remove the `commit_gdn_snapshots()` call from `decode_round`.
- [ ] In `Engine::decode_step_one`, when MTP is enabled, let target decode read via the selected
      slot and write slot 0, then reset `io_.gdn_initial_slot` to zero.
- [ ] Keep host `kv_.pos` / `mtp_kv_->pos` mirroring behavior unchanged except where existing round
      logic already updates it.

**Step 7: Delete old commit-copy code**

- [ ] Remove `#include "qus/kernels/gdn_commit.h"` from runtime code.
- [ ] Delete `Engine::commit_gdn_snapshots()` from header and source.
- [ ] Delete the old gdn_commit API, wrapper, launcher, kernel, and test files.
- [ ] Remove gdn_commit source and test registrations from CMake.
- [ ] Confirm no code/build/test references remain:

```bash
rg -n "gdn_commit|commit_gdn_snapshots" src include tests bench CMakeLists.txt
```

Expected: no matches.

**Step 8: Add and rewrite high-value tests**

- [ ] Extend `tests/kernels/test_mtp_round.cpp` to verify:
      - `mtp_set_gdn_initial_slot_from_accepted` copies values `0..k`;
      - `mtp_reset_gdn_initial_slot` writes zero.
- [ ] Extend `tests/kernels/test_causal_conv1d.cpp`:
      - selected initial slot 0 matches current snapshot chain behavior;
      - selected initial slots `1..Slots-1` match repeated `causal_conv1d_decode` initialized from that
        selected slot;
      - snapshots are written to slots `0..T-1`, not to `initial_slot..initial_slot+T-1`.
- [ ] Extend `tests/kernels/test_gated_delta_rule.cpp`:
      - selected initial slots `0..Slots-1`;
      - bitwise equality against repeated T=1 recurrent calls initialized from the selected slot;
      - alias case `initial_slot == 0` proves the kernel loads state before overwriting slot 0.
- [ ] Replace `tests/kernels/test_gdn_commit.cpp` by deletion only. Do not preserve old behavior.
- [ ] Extend `tests/test_engine_mtp_e2e.cpp` with a scenario that forces:
      - a round immediately before fallback leaves `gdn_initial_slot` non-zero, not merely cumulative
        `accepted_tokens > 0`;
      - the next step takes the capacity fallback `decode_step_one`;
      - fallback decode observes selected-slot state use rather than stale slot 0.
      Prefer a deterministic engine-visible debug/stat counter for this specific test if one exists
      or is added for testability; otherwise the final review must trace the selected-state decode
      branch and the test must still assert fallback occurs after a non-zero accepted round. The
      test may use the existing real-weight skip behavior.

**Step 9: Update active docs**

- [ ] Update `docs/2026-07-03-mtp-spec-decode-overview.md`.
- [ ] Update `docs/2026-07-03-mtp-state-management.md`.
- [ ] Update `docs/2026-07-03-mtp-round-algorithm.md`.
- [ ] Update `docs/2026-07-03-mtp-implementation-requirements.md`.
- [ ] Update `docs/2026-07-03-mtp-roadmap.md`.
- [ ] Required doc semantic changes:
      - replace "GDN commit-copy" current behavior with `gdn_initial_slot` indirect read;
      - state that conv and SSM share the same selector;
      - state that verify writes snapshots to slots `0..k`;
      - state that fallback T=1 decode reads selected slot, writes slot 0, then resets selector;
      - remove or mark obsolete the earlier rejection of vLLM-style indirect reads;
      - remove active-current claims that slot 0 is committed between MTP rounds.
- [ ] Confirm active docs no longer describe commit-copy as current behavior:

```bash
rg -n "commit-copy|gdn_commit|slot 0 = committed|slot\\[0\\].*committed|round.*slot\\[0\\].*committed|state_in = slot\\[0\\]|slot\\[a\\].*slot\\[0\\]|indirect-read.*future|indirect read.*后续|间接.*后续" \
  docs/2026-07-03-mtp-spec-decode-overview.md \
  docs/2026-07-03-mtp-state-management.md \
  docs/2026-07-03-mtp-round-algorithm.md \
  docs/2026-07-03-mtp-implementation-requirements.md \
  docs/2026-07-03-mtp-roadmap.md
```

Expected: no active-current semantic references. Historical notes are allowed only if explicitly
marked obsolete.

**Step 10: Build and targeted verification**

Run:

```bash
cmake --build build --target \
  qus_mtp_round_test \
  qus_causal_conv1d_test \
  qus_gated_delta_rule_test \
  qus_model_bind_test \
  qus_model_blocks_test \
  qus_engine_mtp_e2e_test \
  qus_gated_delta_rule_bench \
  qus_target_verify_bench -j

./build/tests/qus_mtp_round_test
./build/tests/qus_causal_conv1d_test
./build/tests/qus_gated_delta_rule_test
./build/tests/qus_model_bind_test
./build/tests/qus_model_blocks_test
./build/tests/qus_engine_mtp_e2e_test batched
./build/tests/qus_engine_mtp_e2e_test capacity_fallback
./build/tests/qus_engine_mtp_e2e_test stop_truncation
```

Run sanitizer for the two state-addressing tests:

```bash
compute-sanitizer --tool memcheck ./build/tests/qus_causal_conv1d_test
compute-sanitizer --tool memcheck ./build/tests/qus_gated_delta_rule_test
```

If real weights are absent, `qus_engine_mtp_e2e_test` may print its existing SKIP message. Record
the exact skip message in the completion summary.

**Definition of done for Task 1:**

- The full tree builds.
- Targeted tests pass or real-weight E2E skips for the documented fixture reason.
- Sanitizer passes for conv and SSM state tests.
- No `gdn_commit` code/build/test references remain.
- Active docs describe indirect GDN state selection, not commit-copy.
- MTP-enabled fallback decode cannot read stale slot 0 after a non-zero accepted round.

## Final Review Phase

Only after Task 1 is complete, perform one comprehensive final review covering all items below.

Review area 1: runtime and round sequencing.

- Check `Engine::decode_round`, `Engine::decode_step_one`, prefill reset, cache budget, and all
  `StepState` aggregate initializers.
- Verify there is no host branch on accepted count for GDN state selection.
- Verify fallback transition after `accepted > 0` is sound.

Review area 2: CUDA kernels and state addressing.

- Check conv and SSM selected-slot load math, slot stride units, bounds handling, and slot 0 alias
  behavior.
- Check the BF16 recurrent kernel really uses `template <bool Spec>` instead of duplicate normal
  and snapshot bodies.
- Check math and output indexing did not change.

Review area 3: tests, docs, and cleanup.

- Check tests protect real numerical/state/lifetime risks and do not preserve commit-copy behavior.
- Check active docs and roadmap do not present old semantics as current.
- Check CMake and source tree no longer reference deleted gdn_commit code.

All Critical and Important review findings must be fixed before completion. Minor findings may be
tracked if they do not affect correctness, safety, or requested semantics. This is a final review
only; do not insert intermediate review gates into the implementation.

## Verification Summary To Report

Final handoff must include:

- build targets run;
- test commands run and pass/skip status;
- sanitizer pass/skip status;
- `rg` cleanup results for `gdn_commit|commit_gdn_snapshots`;
- summary of final review findings and fixes.

## Testing Policy Alignment

Allowed tests in this plan:

- Selected-slot conv snapshot equivalence protects real GDN state correctness.
- Selected-slot SSM recurrent equivalence protects numerical correctness and alias safety.
- MTP scalar helper tests protect a real device-scalar contract.
- MTP E2E fallback-after-accept protects observable runtime state sequencing.
- `compute-sanitizer` protects GPU memory/lifetime risks.

Forbidden tests:

- No source string scans as tests.
- No tests for deleted `gdn_commit`.
- No tests for trivial field existence or default constructors.
- No tests that only preserve old aliases or compatibility paths.

## Risk Register

- Slot 0 alias hazard: `initial_slot == 0` must load initial state before writing slot 0.
- Fallback stale state: MTP-enabled fallback must read selected slot before resetting selector.
- StepState aggregate drift: all positional aggregate initializers must add the new tensor.
- Cache budget drift: `default_cache_bytes_for()` must include the new scalar allocation and
  alignment slack.
- Unit mismatch: slot stride values passed to kernels must be elements, not bytes, unless the
  pointer arithmetic explicitly uses bytes.
- Conv/SSM divergence: both state types must use the same selector in every selected path.
- Doc drift: top-level overview, state-management, round-algorithm, requirements, and roadmap all
  currently contain old commit-copy semantics and must be updated together.

## Suggested Single Commit

Because the tree may be unusable mid-refactor, prefer one final commit after verification:

```text
refactor(mtp): select gdn state by accepted slot
```

Do not commit until the final verification and review phase are complete.
