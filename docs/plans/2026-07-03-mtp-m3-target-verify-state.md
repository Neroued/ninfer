# MTP M3 Target Verify And State Foundation Implementation Plan

**Goal:** Land MTP roadmap milestone M3 by adding target KV rewind, GDN snapshot slots and
snapshot/commit operators, a correctness-first target Verify phase, full-column prefill hidden
output, fixed StepState buffers/scalars, verify-aware memory budgeting, and a measured verify cost
checkpoint.

**Requirements:** `docs/2026-07-03-mtp-roadmap.md` M3 and coordination table;
`docs/2026-07-03-mtp-state-management.md` §2, §4, §5, and §6;
`docs/2026-07-03-mtp-round-algorithm.md` §2.1 and §2.2;
`docs/2026-07-03-mtp-spec-decode-overview.md` C2, C3, C5, §7.2, and §8; and
`docs/2026-07-02-mtp-foundation-part2-operators.md` §9 and §12. The objective file in
`/home/neroued/.codex/attachments/5826a101-b0e7-4b61-8c88-dfd8473cb595/goal-objective.md` is part
of the controlling spec for this plan.

**Execution Mode:** Main agent direct implementation. Subagent-driven implementation is
intentionally not used because the user explicitly required the main agent to implement M3 directly,
and the highest-risk changes cross shared runtime/model/kernels state where one coordinator should
own integration order. Subagents are used only for two reviews: one plan audit before implementation
and one code review after self-test. Every review subagent must use `gpt-5.5`, the current strongest
available model reported by the multi-agent tool, with no weaker, cheaper, legacy, or
compatibility-oriented model overrides.

**Architecture:** Treat `EngineOptions::mtp_draft_tokens` as the load-time constant `k`. Target GDN
state owns `S = k + 1` slots per layer, with slot 0 preserving the committed-state invariant between
rounds; `k == 0` uses exactly one slot and existing prefill/decode code reads and writes only slot 0.
Verify is added as a target-only model-card phase that consumes caller-provided ids and positions,
uses full-attention prefill-as-append, uses GDN snapshot recurrent operators, keeps all fused
T=1-only decode kernels out of the Verify path, and writes all `k + 1` final hidden/logits/argmax
columns into StepState buffers. Accept/commit decisions remain outside M3 except for exposing the
device-scalar-driven `gdn_commit` primitive.

## Non-Goals

- No accept/reject decision kernel, round state machine, sampled output assembly, strict-sequential
  mode, host/device round synchronization, or generation path change; those belong to M4.
- No MTP forward, MTP KV scheduling, shifted-pass input assembly, AR-step loop, or MTP binding
  changes except preserving the M2 surfaces already in the tree.
- No W8 work and no small-T performance kernels. Verify uses TEXT_CORE Q4/Q5/Q6 generic/prefill
  paths, with tuned GQA/GDN/fused epilogues left to M5.
- No host branch or kernel shape whose execution depends on accepted count `a`. M3 may use host
  `cache_offset` for eager `gqa_attention_prefill`, as allowed by state-management §5.3, but the
  data layout and exposed primitives must be device-scalar compatible for M4/M5.
- No compatibility aliases, deprecated flags, dual old/new schemas, or tests whose only purpose is
  to preserve old behavior.

## Final M3 Contracts

`KVCache`:
- Add `void rewind(std::uint32_t position)`.
- `rewind(position)` validates `position <= pos`, then sets `pos = position`.
- Existing `slot()` read guard `position >= pos` remains the stale-suffix visibility rule.
- `append_slot()` and `advance()` continue using the current logical end.

`GdnState`:
- Add `snapshot_slots`, set to `max(1, k + 1)` at engine load.
- Allocate each `conv[layer]` as BF16 `[conv_dim, conv_width, snapshot_slots]`.
- Allocate each `ssm[layer]` as FP32 `[key_head_dim, value_head_dim, value_heads, snapshot_slots]`.
- Add slot view helpers for slot-local `[conv_dim, conv_width]` and
  `[key_head_dim, value_head_dim, value_heads]` tensors.
- `reset()` only clears slot 0 for every layer. Slots `1..k` are always write-before-read.
- Existing prefill/decode paths pass slot 0 views into existing in-place APIs.

L1 kernel APIs:
- Keep `causal_conv1d_prefill`, `causal_conv1d_decode`,
  `gated_delta_rule_recurrent`, and `gated_delta_rule_chunked` unchanged.
- Add `causal_conv1d_sequence_snapshot(x, weight, states, out, stream)` where `states` is
  `[C,3,S]` and `T <= S`.
- Add `gated_delta_rule_recurrent_snapshot(q, k, v, g, beta, scale, ws, states, out, stream)`
  where `states` is `[128,128,48,S]` and `T <= S`.
- Add `gdn_commit(conv_state, ssm_state, accepted, stream)` where `accepted` is an I32 device scalar
  selecting slot `a`; the kernel short-circuits when `a == 0`.

`StepState`:
- Preserve existing active fields `token`, `pos`, and `logits`, but allocate `logits` as
  `[vocab, k + 1]` when MTP is enabled and use `logits[:,0]` for legacy single-column prefill/decode.
- Add `verify_hidden [hidden,k+1]`, `prefill_hidden [hidden,prefill_chunk]`,
  `target_tokens [k+1]`, `drafts [max(1,k)]`, `sampled_out [k+1]`, `num_sampled [1]`,
  `verify_ids [k+1]`, `shifted_ids [k+1]`, `positions [k+1]`, `window_base [1]`, `a [1]`,
  `ar_pos [1]`, `mtp_ar_hidden [hidden,1]`, and i64 stats counters. `max(1,k)` is required because
  `Tensor` forbids zero-sized dimensions.
- `pos` remains the existing committed-length device scalar for k=0 decode. `window_base`, `a`,
  `ar_pos`, and counters are allocated in M3 and consumed by M4.

Model-card Verify entry:
- Add `Phase::Verify`.
- Add a target verify method that accepts caller-owned `verify_ids [T]`, `positions [T]`, and host
  `cache_offset`, with `T == k + 1` for real MTP and `T == 1` for k=0 checks.
- The method embeds `verify_ids`, runs `run_layers(..., Phase::Verify, cache_offset)`, computes final
  RMSNorm for every column into `io_.verify_hidden`, computes `lm_head` into all columns of
  `io_.logits`, and writes `argmax` into `io_.target_tokens`.
- The method does not update `kv_.pos`, does not decide acceptance, and does not mutate `io_.token`
  except through callers that explicitly use the single-column decode path.

## Scope And Ownership

- `include/qus/core/kv_cache.h` and `src/core/kv_cache.cpp`: add logical rewind.
- `include/qus/core/state_store.h` and `src/core/state_store.cpp`: add GDN slot allocation, slot
  helpers, and slot-0-only reset.
- `include/qus/kernels/causal_conv1d.h`,
  `src/kernels/wrapper/causal_conv1d.cpp`, `src/kernels/launcher/causal_conv1d.{h,cu}`, and
  `src/kernels/kernel/causal_conv1d.cuh`: add the sequence snapshot conv operator.
- `include/qus/kernels/gated_delta_rule.h`,
  `src/kernels/wrapper/gated_delta_rule.cpp`, `src/kernels/launcher/gated_delta_rule.{h,cu}`, and
  `src/kernels/kernel/gated_delta_rule_recurrent.cuh`: add the recurrent snapshot operator.
- `include/qus/kernels/gdn_commit.h`, `src/kernels/wrapper/gdn_commit.cpp`,
  `src/kernels/launcher/gdn_commit.{h,cu}`, and `src/kernels/kernel/gdn_commit.cuh`: add the
  slot commit primitive.
- `include/qus/model/model.h` and `src/model/qwen3_6_27b.cpp`: add `Phase::Verify`, target verify
  entry, GDN slot views, Verify generic fallbacks, all-column final hidden/logits, and prefill
  full-column hidden.
- `include/qus/runtime/engine.h` and `src/runtime/engine.cpp`: allocate new StepState buffers,
  construct `GdnState` with `k + 1` slots, and update workspace/cache formulas.
- `tests/test_kv_cache.cpp`, `tests/test_state_store.cpp`,
  `tests/kernels/test_causal_conv1d.cpp`, `tests/kernels/test_gated_delta_rule.cpp`,
  `tests/kernels/test_gdn_commit.cpp`, `tests/test_model_blocks.cpp`,
  `tests/test_engine_memory_stats.cpp`, and `tests/CMakeLists.txt`: add only
  behavior/numerical/schema tests that protect the M3 risks.
- `bench/target_verify_bench.cpp`, `bench/CMakeLists.txt`, `docs/bench/`, and
  `docs/2026-07-03-mtp-spec-decode-overview.md`: add the cost checkpoint tooling, real-artifact
  parity mode, report, and §7.2 epsilon update after measurement.

## Coordination Points

- `src/model/qwen3_6_27b.cpp` / `include/qus/model/model.h`: preserve M2 MTP APIs and bind paths.
  M3 edits target `run_layers`, target prefill, target verify, and `StepState`.
- `src/runtime/engine.cpp`: merge M2 MTP KV/workspace terms with M3 verify/GDN snapshot terms. The
  final formula must account for both when `mtp_draft_tokens > 0`.
- CMake/test registration is the only expected mechanical conflict. New source files under `src/`
  are picked up by globbing; new tests and benches require explicit registration.
- `Tensor` cannot represent zero-length shapes. Any k-only buffer uses `max(1,k)` while consumers
  still treat logical draft length as `k`.

## Task Breakdown

### Task 1: Plan Review

**Reading List:**
- This plan.
- `docs/2026-07-03-mtp-roadmap.md` M3 and coordination table.
- `docs/2026-07-03-mtp-state-management.md` §4, §5, and §6.
- `docs/2026-07-03-mtp-round-algorithm.md` §2.1 and §2.2.
- `docs/2026-07-03-mtp-spec-decode-overview.md` C2, C3, C5, and §8.
- `docs/2026-07-02-mtp-foundation-part2-operators.md` §9 and §12.

**Implementation:**
- Dispatch one review subagent using `model=gpt-5.5`.
- Require the reviewer to audit the plan against snapshot slot semantics, write timing, no
  accepted-count host branching, M3/M4/M5 boundaries, StepState layout, memory budgeting, test
  policy, sanitizer coverage, and benchmark evidence requirements.
- Revise this file for every valid Critical or Important finding before implementation starts.

**Definition Of Done:**
- Plan review has no unresolved Critical or Important findings.
- Any accepted review finding is reflected in this plan before code changes.

**Verification Commands:**
- No build command is required for this task.
- Record the review summary and plan revisions in the implementation notes and final evidence.

### Task 2: L0 KV Rewind, GDN Slots, And StepState Allocation

**Reading List:**
- `include/qus/core/kv_cache.h`, `src/core/kv_cache.cpp`, and `tests/test_kv_cache.cpp`.
- `include/qus/core/state_store.h`, `src/core/state_store.cpp`, and `tests/test_state_store.cpp`.
- `include/qus/model/model.h`, `include/qus/runtime/engine.h`, and `src/runtime/engine.cpp`.
- `docs/2026-07-03-mtp-state-management.md` §2, §4, and §6.

**Implementation:**
- Add `KVCache::rewind(std::uint32_t position)` with `position <= pos` validation.
- Extend `GdnState` construction with a `snapshot_slots` argument defaulting to 1 for direct tests
  and non-MTP callers.
- Allocate GDN slot shapes as `[conv_dim, conv_width, snapshot_slots]` and
  `[key_head_dim, value_head_dim, value_heads, snapshot_slots]`.
- Add slot view helpers and update model code to pass slot 0 views into existing prefill/decode
  operators.
- Make `reset()` clear only slot 0.
- Extend `StepState` fields in `model.h`.
- In `Engine::load`, allocate `StepState` buffers using `window_cols = mtp_draft_tokens + 1` and
  `draft_cols = max(1, mtp_draft_tokens)`.
- Allocate `mtp_ar_hidden [hidden,1]` even though M4 consumes it, because state-management §6 makes
  the MTP AR hidden buffer part of the M3 StepState base layout.
- Keep k=0 allocation shapes equivalent for active decode outputs: `logits [vocab,1]`, one token,
  one position, and one GDN state slot.

**Tests:**
- Extend `qus_kv_cache_test` with a rewind-and-rewrite scenario: write positions 0 and 1, rewind to
  1, assert reads at position 1 throw, rewrite position 1 with new bytes, advance, and assert the new
  bytes are visible while position 0 remains unchanged.
- Extend `qus_state_store_test` with slots=1 and slots=3 construction, slot shape checks, slot helper
  shape checks, non-alias checks, and reset coverage proving slot 0 is cleared while slot 1 retains a
  sentinel.
- Extend `qus_engine_memory_stats_test` after Task 6 formulas land so MTP-enabled cache capacity
  increases by snapshot slots and StepState buffers.

**Definition Of Done:**
- Existing decode/prefill code compiles using slot 0 views.
- k=0 state layout has one slot and no behavior change in active decode outputs.
- Rewind rejects forward moves and exposes stale suffix as unreadable through existing `slot()`.

**Verification Commands:**
```bash
cmake --build build --target qus_kv_cache_test qus_state_store_test -j
./build/tests/qus_kv_cache_test
./build/tests/qus_state_store_test
```

**Commit Subject:**
```text
feat(core): add target state rewind and gdn slots
```

### Task 3: Snapshot And Commit Operators

**Reading List:**
- `include/qus/kernels/causal_conv1d.h`, `src/kernels/wrapper/causal_conv1d.cpp`,
  `src/kernels/launcher/causal_conv1d.cu`, and `src/kernels/kernel/causal_conv1d.cuh`.
- `include/qus/kernels/gated_delta_rule.h`, `src/kernels/wrapper/gated_delta_rule.cpp`,
  `src/kernels/launcher/gated_delta_rule_recurrent.cu`, and
  `src/kernels/kernel/gated_delta_rule_recurrent.cuh`.
- `tests/kernels/test_causal_conv1d.cpp` and `tests/kernels/test_gated_delta_rule.cpp`.
- `docs/2026-07-03-mtp-state-management.md` §5.1, §5.2, §5.4, and §5.5.

**Implementation:**
- Add `causal_conv1d_sequence_snapshot` validation for BF16 compact `x [C,T]`, `weight [C,4]`,
  `states [C,3,S]`, and `out [C,T]`, with `T > 0` and `T <= S`.
- Implement the snapshot conv kernel as a per-channel sequence loop that loads slot 0 into local
  registers before any write, computes output with the same accumulation order as
  `causal_conv1d_decode`, shifts local state, and writes after-token states to slots `0..T-1`.
- Add `gated_delta_rule_recurrent_snapshot` validation for BF16 compact `q/k/v`, FP32 `g/beta`,
  FP32 `states [128,128,48,S]`, BF16 `out`, and `T <= S`.
- Implement the recurrent snapshot kernel by reusing the existing recurrent token loop and storing
  each after-token `s_tile` into slot `t` after the update. The initial state is read from slot 0
  before any slot write.
- Add `gdn_commit` wrapper/launcher/kernel. A device I32 scalar selects slot `a`; `a < 0` or
  `a >= S` performs no writes in device code, and `a == 0` returns without copying. The wrapper must
  not copy `accepted` to host or branch host-side on its value.
- Register `qus_gdn_commit_test` in `tests/CMakeLists.txt`.

**Tests:**
- Extend `qus_causal_conv1d_test` with T=1..6, multiple seeds, and stress values. Compare every
  snapshot slot byte-for-byte against repeated `causal_conv1d_decode` single-token calls using the
  same inputs and initial state. Also compare output columns to the decode chain.
- Extend `qus_gated_delta_rule_test` with T=1..6, multiple seeds, and stress values. Compare every
  snapshot slot byte-for-byte against repeated `gated_delta_rule_recurrent` T=1 calls and compare
  output columns under the existing op-test tolerance.
- Add `qus_gdn_commit_test` covering slot 0 no-op, slot 1/slot k copy for both conv and ssm states,
  source-slot preservation after commit, and invalid `a` no-write behavior for negative and
  out-of-range scalar values.

**Definition Of Done:**
- Snapshot slots match repeated T=1 calls bit-for-bit for conv state and SSM state.
- Existing in-place prefill/decode recurrent and conv APIs are unchanged.
- Commit-copy is device-scalar driven and safe for all slots.

**Verification Commands:**
```bash
cmake --build build --target qus_causal_conv1d_test qus_gated_delta_rule_test qus_gdn_commit_test -j
./build/tests/qus_causal_conv1d_test
./build/tests/qus_gated_delta_rule_test
./build/tests/qus_gdn_commit_test
compute-sanitizer --tool memcheck ./build/tests/qus_causal_conv1d_test
compute-sanitizer --tool memcheck ./build/tests/qus_gated_delta_rule_test
compute-sanitizer --tool memcheck ./build/tests/qus_gdn_commit_test
```

**Commit Subject:**
```text
feat(kernels): add gdn snapshot state operators
```

### Task 4: Target Verify Phase And Prefill Full-Column Hidden

**Reading List:**
- `include/qus/model/model.h`.
- `src/model/qwen3_6_27b.cpp`.
- `include/qus/kernels/gqa_attention.h`, `src/kernels/wrapper/gqa_attention.cpp`, and
  `src/kernels/launcher/gqa_attention_prefill.cu`.
- `docs/2026-07-03-mtp-round-algorithm.md` §2.1.
- `docs/2026-07-02-mtp-foundation-part2-operators.md` §12 and §13.

**Implementation:**
- Add `Phase::Verify`.
- Add a target verify entry point that validates `ids [T]`, `positions [T]`, `T == io_.target_tokens.ne[0]`,
  and `cache_offset + T <= kv_.max_context`, then writes all final outputs to StepState buffers.
- Update `attn_mix` so `Phase::Verify` uses generic T>1 projections and
  `gqa_attention_prefill(..., cache_offset, ...)`.
- Update `gdn_mix` so `Phase::Verify` uses generic linears for Q/K/V/Z, `gdn_in_ab_gated_prefill`
  for G/Beta, `causal_conv1d_sequence_snapshot`, `gated_delta_rule_recurrent_snapshot`, generic
  out projection, and `residual_add`. `Phase::Verify` must never call `gated_delta_rule_chunked`.
- Update `mlp_tail` so `Phase::Verify` uses generic `linear`, `silu_and_mul`, generic down
  projection, and `residual_add`, never T=1-only decode fused kernels.
- Update prefill so each chunk computes final RMSNorm for all columns into
  `io_.prefill_hidden[:,0:len]`; logits are still computed only from the final column.
- Update decode and prefill logits paths to pass `io_.logits[:,0]` into single-column `linear` and
  `argmax` when `io_.logits` has multiple columns.
- Preserve tap behavior by tapping the final prefill column for `AfterFinalNorm` and the single
  logits column for existing prefill/decode taps.

**Tests:**
- Extend `qus_model_blocks_test` with a focused Verify smoke path using the model-block fixture:
  `Phase::Verify` must run the generic small-T fallback path, GDN snapshot slots `0..k` must be
  written, and the model path must not accidentally use in-place or chunked GDN state.
- Use `qus_target_verify_bench --parity` as the real-artifact parity path. It runs T=1..6 Verify
  windows and sequential T=1
  decode replay from identical prefilled state. It should compare hidden/logits under the existing
  house tolerance and report argmax mismatches only when the top1/top2 gap satisfies the near-tie
  rule from overview §8.2.
- Keep the real-artifact parity command outside always-on CTest if it depends on the large `out/`
  artifact; it is still part of the M3 gate evidence.

**Definition Of Done:**
- Verify `T=k+1` writes target KV at `cache_offset..cache_offset+k` without advancing host `kv_.pos`.
- Verify GDN snapshot operators populate slots `0..k` and leave slot 0 as after-token-0 until
  `gdn_commit` runs.
- Prefill full-column hidden is available for every chunk and k=0 decode behavior remains stable.

**Verification Commands:**
```bash
cmake --build build --target qus_model_blocks_test qus_target_verify_bench -j
./build/tests/qus_model_blocks_test
compute-sanitizer --tool memcheck ./build/tests/qus_model_blocks_test
./build/bench/qus_target_verify_bench --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus --warmup 1 --reps 3 --parity
```

**Commit Subject:**
```text
feat(model): add target verify phase
```

### Task 5: Memory Budget Formulas And Stats Coverage

**Reading List:**
- `src/runtime/engine.cpp`.
- `include/qus/runtime/engine.h`.
- `tests/test_engine_memory_stats.cpp`.
- `docs/2026-07-03-mtp-state-management.md` §4.5 and §6.
- `docs/2026-07-03-mtp-spec-decode-overview.md` §7.1.

**Implementation:**
- Replace the current `default_cache_bytes_for(max_ctx, include_mtp_kv)` helper with a helper that
  takes `draft_tokens` and `prefill_chunk`.
- Include target KV, optional MTP KV, GDN conv/ssm slots, StepState logits/hidden/id/scalar/stat
  buffers, `mtp_ar_hidden [hidden,1]`, alignment slack, and existing margin in cache budgeting.
- Replace or extend `default_work_bytes_for(prefill_chunk, include_mtp_work)` with a helper that
  takes `draft_tokens`.
- Include explicit verify peak stages for `T_v = k + 1`, including attention, GDN snapshot generic
  path, MLP gateup `[34816,T_v]`, final RMSNorm/lm_head temporaries, and MTP shifted/AR work when
  MTP is enabled. If prefill or existing MTP work dominates the peak, keep the formula explicit and
  let the max select the peak.
- Ensure `Engine::load` uses the draft-token-aware helpers whenever user-specified arena sizes are
  zero.
- Update `qus_engine_memory_stats_test` to assert:
  - k=0 default capacities still match the k=0 helpers;
  - k=1 MTP capacity exceeds k=0;
  - k=5/max_ctx=8192 budget includes the documented +720 MiB SSM snapshot increment and the
    StepState buffers.

**Definition Of Done:**
- MTP-enabled loads have enough cache arena for `k + 1` GDN slots and all M3 StepState buffers,
  including `mtp_ar_hidden`.
- k=5/max_ctx=8192 budget numbers are recorded for final evidence and the bench report.
- Public k=0 `Engine::default_work_bytes(prefill_chunk)` remains meaningful for non-MTP callers.

**Verification Commands:**
```bash
cmake --build build --target qus_engine_memory_stats_test -j
./build/tests/qus_engine_memory_stats_test
```

**Commit Subject:**
```text
feat(runtime): budget target verify state
```

### Task 6: Cost Checkpoint Bench And Documentation

**Reading List:**
- `bench/qus_bench.cpp`, `bench/qus_bench_support.{h,cpp}`, and `bench/CMakeLists.txt`.
- `docs/bench/mtp-m1-w8g32-linear-baseline.md` for report style.
- `docs/2026-07-03-mtp-spec-decode-overview.md` §7.2.
- `docs/2026-07-03-mtp-roadmap.md` M3 checkpoint language.

**Implementation:**
- Add a dedicated `qus_target_verify_bench` executable that loads a weights file, warms up, pre-fills
  a canonical prompt, and measures:
  - T=1 eager decode replay cost as baseline `C_t`;
  - target Verify cost for T=2,3,4,5,6 using the generic M3 path.
- Report per-T median or mean latency, epsilon `verify_latency / decode_latency - 1`, GPU name,
  CUDA version, artifact path, max context, prefill seed length, repetitions, and git commit.
- Run the benchmark against `out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus` or the current
  equivalent real artifact in `out/`.
- Add `docs/bench/2026-07-03-mtp-m3-target-verify-cost.md` with the measured table and method.
- Update `docs/2026-07-03-mtp-spec-decode-overview.md` §7.2 with the measured epsilon range and a
  short implication for M5 priority.

**Definition Of Done:**
- The report contains actual T=2..6 epsilon values, not estimates.
- The overview speed model points to the report and states whether small-T verify tuning is a high,
  medium, or low M5 priority based on the data.

**Verification Commands:**
```bash
cmake --build build --target qus_target_verify_bench -j
./build/bench/qus_target_verify_bench --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus --warmup 1 --reps 3 --parity
```

**Commit Subject:**
```text
bench(mtp): record target verify cost
```

### Task 7: Full M3 Gate, Code Review, Fixes, And Commits

**Reading List:**
- This plan.
- All changed source, test, bench, and documentation files.
- `docs/2026-07-03-mtp-state-management.md` §4 and §5.
- `docs/2026-07-03-mtp-round-algorithm.md` §2.1 and §2.2.
- `docs/2026-07-03-mtp-spec-decode-overview.md` §8.

**Implementation:**
- Run the smallest verification set after each task and the full M3 gate after Task 6.
- Dispatch one code-review subagent using `model=gpt-5.5` after local self-test. The reviewer must
  focus on slot addressing, alias safety, snapshot write timing, Verify mask/window boundaries, k=0
  zero-impact behavior, memory-budget derivation, no host copy/branch on the `gdn_commit` accepted
  scalar, and M3/M4/M5 boundary discipline.
- Fix every valid Critical or Important code-review finding and rerun affected verification.
- Split final commits along the task subjects above, keeping plan and benchmark docs in the relevant
  commits.

**Definition Of Done:**
- Full build passes.
- All existing tests pass.
- Snapshot operator tests pass and compute-sanitizer is clean.
- KV rewind/rewrite test passes.
- Target Verify real-artifact parity passes for T=1..6, with any argmax mismatch justified by
  near-tie evidence.
- `qus_engine_memory_stats_test` passes and k=5/max_ctx=8192 budget numbers are captured.
- Cost checkpoint report is committed under `docs/bench/`.
- Code review has no unresolved Critical or Important findings.
- Conventional commits are present for the completed M3 work.

**Verification Commands:**
```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck ./build/tests/qus_causal_conv1d_test
compute-sanitizer --tool memcheck ./build/tests/qus_gated_delta_rule_test
compute-sanitizer --tool memcheck ./build/tests/qus_gdn_commit_test
compute-sanitizer --tool memcheck ./build/tests/qus_model_blocks_test
./build/bench/qus_target_verify_bench --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus --warmup 1 --reps 3 --parity
git status --short
```

## Final Evidence To Report

- Change summary by layer: L0 state, L1 kernels, L2 model Verify, runtime memory, tests, bench docs.
- Gate evidence: build, CTest, sanitizer, real-artifact Verify parity, memory stats, and benchmark
  command outputs.
- Epsilon checkpoint numbers and the resulting M5 tuning priority.
- M4 handoff notes: StepState field meanings, `target_verify` inputs/outputs, `gdn_commit` timing,
  KV rewind timing, and remaining accept/round assembly responsibilities.
