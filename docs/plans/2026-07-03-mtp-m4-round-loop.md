# MTP M4 Round Loop And Acceptance Implementation Plan

**Goal:** Land MTP roadmap milestone M4 by wiring the eager speculative round loop into the engine:
prefill MTP KV fill, verify -> accept/commit -> propose, strict-sequential validation mode, multi-token
delivery, MTP statistics/reporting, and round-level dump support.

**Requirements:** `docs/2026-07-03-mtp-roadmap.md` M4; `docs/2026-07-03-mtp-round-algorithm.md`
all sections, especially §1, §2, §3, §4, and §5; `docs/2026-07-03-mtp-spec-decode-overview.md`
§4, §5, and §8; `docs/2026-07-03-mtp-state-management.md` §2.2, §6, and §7;
`docs/2026-07-03-mtp-implementation-requirements.md` R-RT-1..5/7, R-L1-5/6, R-TOOLS-1, and the
R-TOOLS-2 schema portion; the objective file
`/home/neroued/.codex/attachments/46fed0ef-a94b-4ba1-8efc-b80488d0733b/goal-objective.md`; and
repository `AGENTS.md`.

**Execution Mode:** Main agent direct implementation. Subagent-driven implementation is intentionally
not used because the objective explicitly requires the main agent to implement M4 directly. Subagents
are required only for two strict reviews: one plan audit before implementation and one code review
after self-test. Every review subagent must use the current strongest available model exposed by the
multi-agent tool; weaker, cheaper, legacy, or compatibility-oriented models are not allowed.

**Architecture:** Keep M4 correctness-first and eager. The engine owns the round state machine and
host-visible stop/length decisions. Small L1 kernels assemble round inputs, compute acceptance,
update device scalars/counters, assemble shifted MTP inputs, and gather the accepted MTP row without
host branching on `a`. The model card keeps target Verify and MTP forward math localized, adding only
the row-gather MTP logits variant and prefill-time MTP integration needed by the engine. The current
attention wrappers still use host `cache_offset` for prefill-shaped windows and host `KVCache::pos`
validation for decode-shaped AR steps; the M4 eager path synchronizes host cursor mirrors at fixed
round boundaries and leaves device-scalar attention-window variants to M5 as explicitly required.

## Preflight Status

Current evidence before this plan:

- M3 state foundation focused tests are green:
  `ctest --test-dir build -R 'qus_(kv_cache|state_store|model_blocks|engine_memory_stats|gdn_commit)_test' --output-on-failure`
  passed 5/5.
- `qus_mtp_pack_test` is green through CTest.
- Source has M2 C++ MTP entry points: `Qwen3_6_27B::mtp_forward_batch` and
  `Qwen3_6_27B::mtp_forward_ar_step`.
- `Engine::load` allocates an MTP KV cache and MTP StepState buffers when `mtp_draft_tokens > 0`.
- The objective preflight also asks for C++ `mtp_forward` vs ref-model parity evidence. The committed
  M2 plan records an owner waiver for cross-implementation MTP-forward parity and says no
  `qus_mtp_forward_test` parity harness should remain. The current source tree indeed has no
  `tests/test_mtp_forward.cpp`; only a stale built binary exists under `build/tests/`. A local
  Task 0 attempt to restore strict hidden/logit parity reproduced the expected math-path mismatch:
  draft tokens matched exactly, but BF16 hidden/logit tensors failed the frozen composite tolerance.

This plan treats the mismatch as Task 0 and resolves it by honoring the source-controlled M2 waiver:
do not restore or keep a standalone cross-implementation MTP-forward parity harness. M4 correctness
is instead proven by the M4 gates: strict-sequential equality with MTP-off, batched-vs-strict
near-tie analysis, real-weight `forward_mtp_verified` token/stat acceptance, sanitizer, and schema
checks.

## Non-Goals

- No round-level CUDA graph capture.
- No device-scalar attention window-base variant; eager M4 may pass the host round-start mirror to
  existing prefill-shaped attention. M5 owns graph-ready attention variants.
- No small-T target Verify tuning, fused MTP kernels, launch-count reduction, or kernel performance
  optimization.
- No change to the verification/acceptance algorithm in `round-algorithm.md`.
- No probabilistic sampler integration.
- No backward-compatible aliases, deprecated CLI flags, dual schemas, or legacy report fields.
- No source-structure tests. Tests must protect observable numerical, schema, CLI, E2E, or GPU
  lifetime risks allowed by `AGENTS.md`.

## Final Contracts

### Engine Options And Runtime API

- `EngineOptions::mtp_draft_tokens` remains the load-time `k`, valid in `[0,5]`.
- Add `EngineOptions::mtp_strict_sequential`; default false.
- Add `EngineOptions::mtp_round_dump_dir`; empty disables round dumps.
- `mtp_draft_tokens == 0` keeps the existing prefill/decode/generate behavior and memory surface
  unchanged except for inactive fields already allocated by M3.
- `Engine::decode_step()` remains a one-token API for existing callers. When MTP is enabled, add an
  internal pending-token buffer. A call to `decode_step()` launches a full MTP round only when the
  pending buffer is empty, then drains one sampled token per public call. This lets existing
  token-counting benchmark loops measure MTP rounds without a parallel benchmark-only API.
- Add an internal `decode_round()` returning a host vector of sampled tokens for `generate` and text
  runner so user-facing generation can stream one whole sampled batch at a time.
- `Engine::generate()` and `TextGenerationRunner` consume one prefill token plus repeated sampled
  batches, apply stop-token truncation, and use overshoot-and-truncate for `max_new_tokens`.

### StepState Counters

Increase `model::kStepStatsCounters` from 8 to 9 and use this fixed order:

| Index | Name | Meaning |
|---:|---|---|
| 0 | `draft_tokens` | `k * rounds` for batched MTP rounds |
| 1 | `accepted_tokens` | sum of accepted draft count `a` |
| 2 | `rounds` | count of completed batched MTP rounds |
| 3 | `fallback_steps` | count of ordinary T=1 fallback decode steps |
| 4 | `accepted_per_pos_0` | accepted draft count at draft position 0 |
| 5 | `accepted_per_pos_1` | accepted draft count at draft position 1 |
| 6 | `accepted_per_pos_2` | accepted draft count at draft position 2 |
| 7 | `accepted_per_pos_3` | accepted draft count at draft position 3 |
| 8 | `accepted_per_pos_4` | accepted draft count at draft position 4 |

The public report expands this into `accepted_per_pos[]` with length `k`. For `k < 5`, unused
per-position counters stay zero and are omitted from the public array. The implementation must not
publish an aggregated tail as if it were exact per-position data.

### Round Flow

Batched MTP round:

1. Device assembly kernel copies committed `L` to `window_base`, fills
   `verify_ids = [io_.token, drafts...]`, and fills `positions = window_base + j`.
2. `target_verify(verify_ids, positions, host_window_base)` writes target KV `L..L+k`, GDN snapshot
   slots, `verify_hidden`, `logits`, and `target_tokens`.
3. Accept kernel computes `a`, `t*`, `sampled_out`, `num_sampled`, `io_.token = t*`, updated device
   `L`, and stats.
4. Eager cursor boundary: copy only the updated committed length and accepted count needed to set
   host-owned `KVCache::pos` mirrors. Do not make stop or max-length decisions here. Rewind target KV
   and MTP KV host mirrors to `L+a+1`.
5. Commit phase runs `gdn_commit` for every GDN layer using device `accepted`. This is after all
   Verify snapshots and before any later Verify can read GDN slot 0.
6. Shifted assembly kernel fills `shifted_ids[j] = verify_ids[j+1]`, overwrites row `a` with `t*`,
   and leaves junk rows unchanged.
7. MTP shifted pass runs fixed `T = k+1` at `host_window_base` and writes MTP KV `L..L+k`.
8. Because `mtp_forward_batch` sets `mtp_kv_->pos = host_window_base + k + 1`, rewind the MTP KV host
   mirror back to `L+a+1` immediately after the shifted pass and before AR steps. This preserves the
   M2 cursor contract while still computing junk rows.
9. Device row-gather logits variant computes `drafts[0]` from `mtp_hidden[:, a]` and writes
   `mtp_ar_hidden`.
10. Run `k-1` fixed host-loop AR steps with device `ar_pos` scalar incremented by a small kernel
    after each step. The loop count depends only on `k`.
11. Host reads `sampled_out[0..num_sampled-1]` and updated `L`, then handles stop, max-new
    truncation, and final host cursor mirrors.

Strict-sequential debug mode:

- Enabled only when `k > 0`.
- Replaces batched Verify with up to `k+1` ordinary T=1 decode replays on the target path.
- GDN uses in-place slot 0, not snapshots.
- Add an explicit model-card helper `target_decode_strict_step_capture(input_token, position,
  output_column)` that uses the exact existing T=1 target decode kernels, writes the final hidden
  column into `io_.verify_hidden[:, output_column]`, writes logits/argmax for that column, advances
  device `io_.pos` exactly once, and leaves host KV advancement to the engine exactly once.
- Stop at the first draft mismatch and output exactly the accepted prefix plus the target token.
- Propose still rebuilds drafts from the committed target/MTP state. If matching MTP-off equality
  requires using the same propose path as batched mode after the strict target replay, implement that
  directly rather than adding a shortcut that hides MTP state bugs.

### Prefill MTP Integration

When `k > 0`, every target prefill chunk immediately runs the MTP shifted pass:

- Non-last chunks use host-known next prompt token for the final shifted id and skip logits/AR steps.
- Last chunk uses device `io_.token` as the final shifted id, computes `drafts[0]`, and runs `k-1`
  AR steps with `ar_pos = prompt_len`.
- `Engine::prefill()` still resets target KV, MTP KV, and GDN state once at the start only.
- Round loop must never call `Engine::prefill()`.

## Scope And Ownership

- `include/qus/kernels/mtp_round.h`, `src/kernels/wrapper/mtp_round.cpp`,
  `src/kernels/launcher/mtp_round.{h,cu}`, `src/kernels/kernel/mtp_round.cuh`: round assembly,
  accept/stat, shifted assembly, row gather, and scalar increment/set kernels.
- `include/qus/model/model.h`, `src/model/qwen3_6_27b.cpp`, `src/model/position.{h,cu}`: MTP
  prefill integration, row-gather MTP logits entry, reusable scalar helpers, and state dump hooks.
- `include/qus/runtime/engine.h`, `src/runtime/engine.cpp`: options, MTP round loop, strict mode,
  host mirror synchronization, fallback accounting, generation batching, memory stats, and dump
  orchestration.
- `include/qus/text/cli.h`, `src/text/cli.cpp`, `src/main.cpp`, `src/text/text_runner.cpp`: CLI flags
  and multi-token streaming/detokenization.
- `bench/qus_bench_support.{h,cpp}`, `bench/qus_bench.cpp`, `bench/README.md`,
  `tests/test_qus_bench_support.cpp`: MTP benchmark options, schema version bump, `mtp` report
  fields, table/CSV/JSON rendering, and schema contract tests.
- `tools/parity/ref_model.py` only if dump comparison needs a small C++ dump consumer contract
  update; do not change the oracle algorithm.
- `tests/`: add only allowed high-value tests: L1 round kernels, CLI/report schema, strict E2E
  canonical fixture behavior, and GPU lifetime/sanitizer scenarios.
- `docs/bench/` and `docs/`: add M4 evidence report, dump-tool notes, and M5 handoff notes.

## Coordination Points

- `src/model/qwen3_6_27b.cpp` is shared by prefill, target Verify, and MTP forward. Keep changes
  localized to public MTP helpers and the existing prefill loop.
- `src/runtime/engine.cpp` owns host/device position synchronization. Do not duplicate round policy
  in `TextGenerationRunner` or `qus_bench`.
- CMake registration is a coordination point for new kernel tests and parity/debug tools.
- `StepState::stats` length and bench schema must change together. M4 increases the counter count to
  nine so exact `accepted_per_pos[0..4]` is available for k=5.
- Existing CUDA graph decode is for k=0 T=1 decode. Disable graph use for MTP eager rounds unless a
  later M5 plan explicitly captures the fixed DAG.

## Task Breakdown

### Task 0: Preflight Evidence Reconciliation

**Reading List:**
- Objective file preflight section.
- `docs/plans/2026-07-03-mtp-m2-cpp-forward.md` verification and Testing Policy notes.
- `docs/2026-07-03-mtp-roadmap.md` M2 and M4 rows.
- `tools/parity/ref_model.py` `mtp_forward()` and `forward_mtp_verified()`.
- Current source for `Qwen3_6_27B::mtp_forward_batch` and `mtp_forward_ar_step`.

**Implementation:**
- Do not restore or keep a standalone Python-vs-C++ MTP-forward hidden/logit parity harness. The M2
  plan explicitly waived that requirement and made absence of `qus_mtp_forward_test` part of M2 DoD.
- Record the failed strict parity attempt as evidence that the historical math-path mismatch still
  applies. The exact draft sequence matched, so the failure does not indicate a demonstrated token
  contract break.
- Reconfirm the M2-approved preflight instead:
  - MTP pack/split behavior test is green;
  - model binding and real-file engine load cover MTP weights and MTP KV allocation;
  - engine memory stats cover MTP cache/workspace accounting;
  - `Qwen3_6_27B::mtp_forward_batch` and `mtp_forward_ar_step` remain present and compile.
- Keep M4 correctness acceptance on the round-level gates, where the algorithm is observable:
  strict-sequential equality with MTP-off, batched-vs-strict near-tie analysis, real-weight
  `forward_mtp_verified` token/stat comparison, and sanitizer.

**Definition Of Done:**
- No source-controlled standalone MTP-forward parity harness exists.
- The failed strict hidden/logit parity attempt and the M2 waiver decision are documented.
- M2-approved focused tests and source checks are green enough to proceed into M4 implementation.

**Verification Commands:**
```bash
ctest --test-dir build --output-on-failure -R 'qus_mtp_pack_test|qus_model_bind_test|qus_engine_memory_stats_test|qus_engine_real_file_test'
rg 'mtp_forward_batch|mtp_forward_ar_step|mtp_kv_' include/qus/model/model.h src/model/qwen3_6_27b.cpp src/runtime/engine.cpp
test ! -e tests/test_mtp_forward.cpp
```

### Task 1: Plan Review

**Reading List:**
- This plan.
- All normative docs listed in Requirements.
- `src/runtime/engine.cpp`, `src/model/qwen3_6_27b.cpp`, `src/model/position.{h,cu}`,
  `bench/qus_bench_support.{h,cpp}`, and `tests/test_qus_bench_support.cpp`.

**Implementation:**
- Dispatch one review subagent using the current strongest available model.
- Require the reviewer to audit: Task 0 preflight handling, full round-algorithm coverage, `L` and
  `window_base` timing, GDN commit timing, target/MTP KV cursor synchronization, strict-sequential
  equality design, prefill shifted-pass integration, stop/overshoot behavior, bench schema, dump
  tool scope, and `AGENTS.md` test-policy compliance.
- Revise this plan for every valid Critical or Important finding before code implementation.

**Definition Of Done:**
- Plan review has no unresolved Critical or Important findings.
- Any accepted finding is reflected in this plan.

**Verification Commands:**
- No build command. Record reviewer summary and plan revisions in implementation notes.

### Task 2: L1 Round Assembly, Accept, And Gather Kernels

**Reading List:**
- `docs/2026-07-03-mtp-round-algorithm.md` §2.1, §2.2, §2.3, and §4.
- Existing wrapper/launcher/kernel pattern in `include/qus/kernels/mtp_pack.h`,
  `src/kernels/wrapper/mtp_pack.cpp`, `src/kernels/launcher/mtp_pack.{h,cu}`, and
  `tests/kernels/test_mtp_pack.cpp`.
- `include/qus/model/model.h` `StepState`.

**Implementation:**
- Add public wrappers:
  - `mtp_prepare_verify(token, drafts, L, verify_ids, positions, window_base, stream)`.
  - `mtp_accept(target_tokens, drafts, L, token, sampled_out, num_sampled, accepted, stats, k, stream)`.
  - `mtp_prepare_shifted(verify_ids, token, accepted, shifted_ids, stream)`.
  - `mtp_gather_hidden_row(mtp_hidden, accepted, out_hidden, stream)`.
  - `mtp_increment_scalar(scalar, stream)` and `mtp_set_scalar(scalar, value, stream)`.
- Accept kernel must implement exactly:
  `a = longest prefix where target_tokens[i] == drafts[i]`, `t* = target_tokens[a]`,
  `sampled_out[0..a-1] = drafts[0..a-1]`, `sampled_out[a] = t*`, `num_sampled = a + 1`,
  `io_.token = t*`, `L += a + 1`, and stats updates.
- Update stats with exact `accepted_per_pos[0..k-1]` using the nine-counter layout in this plan, and
  update memory formulas/tests in the same task.
- Add behavior tests for k=1..5 covering all-reject, partial accept, all-accept/bonus, sampled
  output, `L`, `io_.token`, `num_sampled`, shifted overwrite, row gather, and stats.

**Definition Of Done:**
- Round kernels are device-scalar driven, fixed-shape for a given `k`, and have no host copy of `a`.
- Tests cover the exact acceptance and shifted-index rules from `round-algorithm.md`.

**Verification Commands:**
```bash
cmake --build build --target qus_mtp_round_test -j
./build/tests/qus_mtp_round_test
compute-sanitizer --tool memcheck ./build/tests/qus_mtp_round_test
```

### Task 3: Model Card MTP Row-Gather Entry And Prefill Integration

**Reading List:**
- `docs/2026-07-03-mtp-round-algorithm.md` §1, §2.3, and §2.4.
- `src/model/qwen3_6_27b.cpp` `prefill_impl`, `mtp_forward_batch`, `mtp_forward_ar_step`, and
  `target_verify`.
- `src/model/position.{h,cu}`.

**Implementation:**
- Add `Qwen3_6_27B::mtp_forward_batch_gather(...)` so the M4 caller can compute logits/draft from a
  device row index `accepted` while preserving the existing host `logits_column` test API.
- The device-row variant must gather `mtp_hidden[:, a]` into `io_.mtp_ar_hidden`, run lm_head on
  that single column, and write `io_.drafts[0]`.
- In `prefill_impl`, after each target chunk writes `io_.prefill_hidden`, run one MTP shifted pass
  when MTP is enabled:
  - non-last chunk: build shifted ids with host next token, run MTP batch, skip logits;
  - last chunk: use device `io_.token` for the final shifted id, compute `drafts[0]`, copy last MTP
    hidden to `mtp_ar_hidden`, and run `k-1` AR steps with `ar_pos = prompt_len`.
- Preserve `k == 0` behavior and existing tap output.
- Keep `Engine::prefill()` reset semantics: reset target KV, MTP KV, and GDN once before calling
  model prefill; no round-loop call to `prefill()`.

**Definition Of Done:**
- MTP KV is filled for every prefill chunk when `k > 0`.
- Last prefill chunk produces `d1..dk`.
- Non-MTP prefill outputs and memory behavior are unchanged.

**Verification Commands:**
```bash
cmake --build build --target qus_model_blocks_test qus_engine_memory_stats_test -j
ctest --test-dir build --output-on-failure -R 'qus_model_blocks_test|qus_engine_memory_stats_test'
```

### Task 4: Engine Round Loop, Cursor Sync, Fallback, And Strict Mode

**Reading List:**
- `src/runtime/engine.cpp` `prefill`, `decode_step`, and `generate`.
- `include/qus/runtime/engine.h`.
- `docs/2026-07-03-mtp-round-algorithm.md` §2, §3, and §5.
- `docs/2026-07-03-mtp-spec-decode-overview.md` §4, C1-C5, and §8.2.

**Implementation:**
- Add private helpers:
  - `bool mtp_enabled() const`.
  - `bool can_run_mtp_round(std::uint32_t host_L) const` implementing `host_L + 2k <= max_ctx`.
  - `std::vector<int> decode_round()`.
  - `std::vector<int> decode_mtp_batched_round()`.
  - `std::vector<int> decode_mtp_strict_round()`.
  - `int decode_fallback_step()` that increments `fallback_steps`.
  - `void sync_committed_length_for_cursors()` that reads device `L` and `accepted` only for
    host-owned cursor rewinds after accept and after the shifted MTP pass.
  - `void sync_host_positions_after_round()` that reads final device `L` after sampled output read
    and synchronizes host mirrors for stop/length/capacity decisions.
- Add a private `std::vector<int> pending_sampled_` that `decode_step()` drains when MTP is enabled.
  Clear it at `load()` and `prefill()`.
- Fix the existing `decode_step` / `decode_step_record` asymmetry by making model-card decode own
  device `io_.pos` and engine own host `kv_.advance()` exactly once per committed token.
- Batched round order must be: prepare verify -> target verify -> accept kernel -> fixed-boundary
  cursor sync and target/MTP KV rewind -> GDN commit -> shifted assembly -> MTP shifted pass ->
  post-shifted MTP KV rewind -> row-gather logits -> AR steps -> host sampled output read.
- Strict mode must use `target_decode_strict_step_capture` for T=1 target decode replay and produce
  the same output as MTP-off for canonical prompts. It must stop replay at first mismatch, must not
  use GDN snapshots, must capture final hidden columns for MTP propose, and must not recursively call
  the MTP-enabled public `decode_step()`.
- Capacity fallback must run ordinary T=1 decode until stop or `max_new_tokens`; each fallback step
  increments stats.
- Disable CUDA graph for MTP rounds in v1 eager mode even when `use_cuda_graph` is true.

**Definition Of Done:**
- `mtp_draft_tokens == 0` path remains identical.
- MTP round never calls `Engine::prefill()`.
- Capacity fallback triggers only when `L + 2k > max_ctx`.
- Device `L` is authoritative; host mirrors are synchronized after fixed round/fallback boundaries.
- Existing `decode_step()` loops, including `qus_bench`, execute MTP rounds when MTP is enabled by
  draining `pending_sampled_`.

**Verification Commands:**
```bash
cmake --build build --target qus_core qus_engine_memory_stats_test -j
ctest --test-dir build --output-on-failure -R 'qus_engine_memory_stats_test'
```

### Task 5: Generate, Text Runner, CLI, And Stop/Overshoot Semantics

**Reading List:**
- `src/text/text_runner.cpp`.
- `include/qus/text/cli.h`, `src/text/cli.cpp`, `src/main.cpp`.
- `tests/test_qwen_text_cli.cpp`, `tests/test_qwen_text_runner.cpp`.
- `docs/2026-07-03-mtp-round-algorithm.md` §2.5.

**Implementation:**
- Add CLI flags:
  - `--mtp-draft-tokens N`, valid `0..5`, default 0.
  - `--mtp-strict-sequential`, default false.
  - `--mtp-round-dump-dir DIR`, optional debug output.
- Wire CLI flags into `EngineOptions`.
- Change `Engine::generate` to append sampled batches, stop at stop token, and truncate any
  overshoot beyond `max_new_tokens` without trying to roll back state after generation is ending.
- Change `TextGenerationRunner` stream callback to feed each accepted token through
  `TokenStreamDecoder` in order and apply the same truncation rules.
- Update tests for CLI parsing and host-only runner behavior where possible. Do not add source-shape
  tests.

**Definition Of Done:**
- Text and token-id generation surfaces both handle multi-token sampled batches.
- Stop tokens truncate within a round; max-new truncates overshoot.
- CLI rejects invalid MTP draft counts and exposes strict mode.

**Verification Commands:**
```bash
cmake --build build --target qus_qwen_text_cli_test qus_qwen_text_runner_test -j
ctest --test-dir build --output-on-failure -R 'qus_qwen_text_cli_test|qus_qwen_text_runner_test'
```

### Task 6: Bench Schema, Runtime Stats, And Memory Stats

**Reading List:**
- `bench/qus_bench_support.{h,cpp}`.
- `bench/qus_bench.cpp`.
- `tests/test_qus_bench_support.cpp`.
- `bench/README.md`.
- `docs/2026-07-03-mtp-round-algorithm.md` §4.
- `docs/2026-07-03-mtp-implementation-requirements.md` §6.

**Implementation:**
- Add `MtpStats` to runtime and bench surfaces:
  `k`, `rounds`, `fallback_steps`, `draft_tokens`, `accepted_tokens`, `acceptance_rate`,
  `acceptance_length`, and `accepted_per_pos[]`.
- Add `--mtp-draft-tokens` and `--mtp-strict-sequential` to `qus_bench`.
- Wire `BenchOptions::mtp_draft_tokens` and `BenchOptions::mtp_strict_sequential` into
  `EngineOptions`.
- Keep `qus_bench` decode loops token-count based, but make them measure MTP through the
  `Engine::decode_step()` pending-buffer path. Clear pending sampled tokens on every prefill so
  repetitions remain independent.
- Set bench `decode_path` to include `mtp_eager` or `mtp_strict` when enabled.
- Bump `qus_bench` schema version and add an `mtp` object to every JSON report. For k=0, emit
  `enabled=false`, `k=0`, zero counts, null rates, and an empty `accepted_per_pos` array. For k>0,
  emit exact counts and formulas. Update README/tests consistently without preserving the old schema.
- Add table/CSV columns that expose the main acceptance metrics without making the table noisy.
- Update memory stats if the public struct needs MTP stats or exact k=5 budget reporting.

**Definition Of Done:**
- `qus_bench_support_test` validates the new schema version and all MTP fields.
- Reported acceptance metrics use the exact `round-algorithm.md` formulas.
- k=0 reports are deterministic and schema-valid.

**Verification Commands:**
```bash
cmake --build build --target qus_bench qus_bench_support_test -j
./build/tests/qus_bench_support_test
```

### Task 7: Round-Level Dump And Parity Notes

**Reading List:**
- `src/model/qwen3_6_27b.cpp` `FileTap`.
- `tests/test_runtime_file_tap.cpp`.
- `docs/2026-07-03-mtp-state-management.md` §7.
- `tools/parity/ref_model.py` state/debug support.

**Implementation:**
- Add a round dump entry that writes GDN slot 0 and MTP KV state at round boundaries when
  `EngineOptions::mtp_round_dump_dir` is set.
- Reuse `FileTap`-style f32/binary writing patterns, but name files with round index and state type
  so dumps are stable and inspectable.
- Document dump usage and expected comparison against ref-model committed state.
- Add a behavior test that validates dump naming, element count, dtype marker, and at least one
  deterministic sentinel value for GDN slot 0 and MTP KV content using tiny tensors or existing
  FileTap-like helpers. Do not assert private source structure and do not merely check file
  existence.

**Definition Of Done:**
- Round dump can capture GDN slot 0 and MTP KV state after a round.
- Dump docs explain how to compare against `forward_mtp_verified` committed state.

**Verification Commands:**
```bash
cmake --build build --target qus_runtime_file_tap_test -j
ctest --test-dir build --output-on-failure -R 'qus_runtime_file_tap_test'
```

### Task 8: E2E Correctness, Real-Weight Validation, Sanitizer, Review, And Commits

**Reading List:**
- `docs/2026-07-03-mtp-spec-decode-overview.md` §8.
- `docs/2026-07-03-mtp-foundation-part1-verification.md`.
- All changed source/test/bench/docs files.

**Implementation:**
- Add allowed E2E canonical fixture coverage in `tests/test_mtp_e2e.cpp` for strict-sequential
  MTP-on vs MTP-off equality at k=1..5.
- Add batched-vs-strict comparison with near-tie diagnostics for divergence.
- Run real-weight validation against `tools/parity/ref_model.py forward_mtp_verified` using the
  prompt and acceptance table from `docs/2026-07-03-mtp-foundation-part1-verification.md`.
- Run `compute-sanitizer` scenarios for partial accept, all accept, all reject, capacity fallback,
  and stop truncation.
- Capture k=5/max_ctx=8192 real-load memory stats.
- Run `qus_bench` to capture eager MTP baseline tok/s for M5.
- Dispatch one code-review subagent using the current strongest available model. Fix valid Critical
  and Important findings and rerun affected verification.
- Commit in conventional chunks after verification:
  - `docs(plan): add mtp m4 round loop plan`
  - `feat(kernels): add mtp round assembly kernels`
  - `feat(model): integrate mtp prefill proposals`
  - `feat(runtime): add mtp eager round loop`
  - `feat(text): wire mtp generation controls`
  - `bench(mtp): report acceptance metrics`
  - `test(mtp): add round correctness gates`

**Definition Of Done:**
- Full build passes.
- All existing tests pass.
- k=0 output and memory behavior are proven unchanged by focused checks.
- Strict-sequential MTP-on equals MTP-off for canonical fixtures at k=1..5.
- Batched MTP only diverges from strict at documented near-tie points.
- Real-weight strict output and acceptance stats match ref-model evidence.
- `compute-sanitizer` is clean for required MTP scenarios.
- k=5/max_ctx=8192 memory budget fits with documented margin.
- `qus_bench` schema contains exact MTP stats and tests pass.
- Round dump docs/evidence are committed.
- Code review has no unresolved Critical or Important findings.
- Conventional commits are present and `git status --short` is clean.

**Verification Commands:**
```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck ./build/tests/qus_mtp_round_test
compute-sanitizer --tool memcheck ./build/tests/qus_mtp_e2e_test --scenario partial_accept
compute-sanitizer --tool memcheck ./build/tests/qus_mtp_e2e_test --scenario all_accept
compute-sanitizer --tool memcheck ./build/tests/qus_mtp_e2e_test --scenario all_reject
compute-sanitizer --tool memcheck ./build/tests/qus_mtp_e2e_test --scenario capacity_fallback
compute-sanitizer --tool memcheck ./build/tests/qus_mtp_e2e_test --scenario stop_truncation
/home/neroued/miniconda3/envs/py311/bin/python -m tools.parity.ref_model \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus --mtp-draft-tokens 5
./build/bench/qus_bench --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus \
  -pg 512,128 --mtp-draft-tokens 5 --no-cuda-graph -o json \
  --output-file out/mtp_m4_qus_bench_k5.json
git status --short
```

## Review Phase

### Plan Review

The plan-review subagent must audit:

- preflight mismatch handling and whether Task 0 satisfies the objective without violating the M2
  waiver;
- every round-algorithm section, including prefill chunk MTP fill, four-stage decode order,
  capacity guard, forbidden `prefill()` in round loop, strict-sequential mode, stats formulas, and
  correctness invariants;
- `window_base` vs `L` timing and host/device mirror synchronization;
- GDN snapshot/commit timing;
- MTP KV junk-row overwrite safety and AR cursor sequencing;
- stop-token and max-new overshoot boundaries;
- k=0 zero-impact behavior;
- tests against `AGENTS.md` whitelist and forbidden-test rules.

### Code Review

The code-review subagent must audit:

- accept kernel exactness, tie behavior, and stats counters;
- row-gather MTP logits and absence of host `a`-dependent kernel shapes;
- prefill shifted id construction for non-last and last chunks;
- host cursor sync and decode_step/decode_step_record symmetry;
- strict-sequential equality with MTP-off;
- capacity fallback and stop/overshoot logic;
- bench schema formulas and exact `accepted_per_pos[]`;
- dump state naming and content;
- sanitizer and real-weight evidence.

Valid Critical and Important findings are fixed before final verification. If a finding is rejected,
record the technical reason and command or code evidence.

## Verification Evidence

This section is append-only after implementation begins. Each entry must include command, exit
status, and key output lines.

### 2026-07-03 Task 0 Preflight Attempt

Command:
```bash
cmake --build build --target qus_mtp_forward_dump -j2
```

Exit status: 0.

Key output:
```text
[3/3] Linking CXX executable tests/qus_mtp_forward_dump
```

Command:
```bash
/home/neroued/miniconda3/envs/vllm-bench/bin/python tools/parity/mtp_forward_fixture.py \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus \
  --out-dir /tmp/qus_mtp_m4_task0_ref --ar-steps 2
```

Exit status: 0.

Command:
```bash
build/tests/qus_mtp_forward_dump \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus \
  --input-dir /tmp/qus_mtp_m4_task0_ref \
  --out-dir /tmp/qus_mtp_m4_task0_cpp --steps 2
```

Exit status: 0.

Key output:
```text
OK mtp_forward_dump T=4 steps=2
```

Command:
```bash
python - <<'PY'
import numpy as np
for name in ["ref", "cpp"]:
    p = f"/tmp/qus_mtp_m4_task0_{name}/drafts.i32"
    print(name, np.fromfile(p, dtype="<i4").tolist())
PY
```

Exit status: 0.

Key output:
```text
ref [9, 198, 1]
cpp [9, 198, 1]
```

Command:
```bash
python tools/parity/compare_mtp_forward_dump.py \
  --ref-dir /tmp/qus_mtp_m4_task0_ref \
  --dump-dir /tmp/qus_mtp_m4_task0_cpp
```

Exit status: 1.

Key output:
```text
FAIL: shifted_hidden.bf16: tolerance failed; violating_frac=0.225488, worst_ratio=20.2415 at 170, rel_l2=0.00759284
FAIL: shifted_logits.bf16: tolerance failed; violating_frac=0.103701, worst_ratio=20.9562 at 42297, rel_l2=0.0054171
FAIL: ar_hidden.bf16: tolerance failed; violating_frac=0.2875, worst_ratio=27.0371 at 4834, rel_l2=0.00854963
FAIL: ar_logits.bf16: tolerance failed; violating_frac=0.199813, worst_ratio=33.4811 at 255787, rel_l2=0.00769358
shifted_hidden.bf16: max_abs=0.1875 rel_l2=0.00759284 violating=4618/20480 worst_ratio=20.2415
shifted_logits.bf16: max_abs=0.0625 rel_l2=0.0054171 violating=25751/248320 worst_ratio=20.9562
ar_hidden.bf16: max_abs=0.25 rel_l2=0.00854963 violating=2944/10240 worst_ratio=27.0371
ar_logits.bf16: max_abs=0.125 rel_l2=0.00769358 violating=99235/496640 worst_ratio=33.4811
```

Stop condition: Task 0 did not meet the exact MTP-forward preflight criteria. Do not begin M4
runtime implementation until the objective is amended or the Python/C++ hidden/logit parity gap is
resolved.

### 2026-07-03 Task 0 Resolution

The strict hidden/logit parity criteria above were removed from Task 0 after root-cause review. They
reintroduced the exact standalone Python-vs-C++ MTP-forward parity harness that the source-controlled
M2 plan forbids:

```text
Do not add or keep a cross-implementation MTP forward parity harness. The project owner explicitly
waived that requirement because Python and C++ use different math paths.
```

The failed attempt still provides useful diagnostic evidence: C++ and RefModel produced the same
draft sequence `[9, 198, 1]`, while the hidden/logit tensor tolerances failed at BF16 composite
thresholds consistent with the waived math-path mismatch. The temporary harness files and CMake
target were removed. M4 implementation proceeds only against the round-level correctness gates:
strict-sequential equals MTP-off, batched-vs-strict near-tie analysis, real-weight
`forward_mtp_verified` token/stat comparison, sanitizer, and report-schema checks.

Command:
```bash
ctest --test-dir build --output-on-failure \
  -R 'qus_mtp_pack_test|qus_model_bind_test|qus_engine_memory_stats_test|qus_engine_real_file_test' &&
rg 'mtp_forward_batch|mtp_forward_ar_step|mtp_kv_' \
  include/qus/model/model.h src/model/qwen3_6_27b.cpp src/runtime/engine.cpp &&
test ! -e tests/test_mtp_forward.cpp
```

Exit status: 0.

Key output:
```text
100% tests passed, 0 tests failed out of 4
include/qus/model/model.h:    void mtp_forward_batch(...)
include/qus/model/model.h:    void mtp_forward_ar_step(...)
src/runtime/engine.cpp:        mtp_kv_.emplace(...)
src/model/qwen3_6_27b.cpp:void Qwen3_6_27B::mtp_forward_batch(...)
src/model/qwen3_6_27b.cpp:void Qwen3_6_27B::mtp_forward_ar_step(...)
```

### 2026-07-03 L1 Round Helper Kernels

Command:
```bash
cmake --build build --target qus_mtp_round_test -j2 &&
ctest --test-dir build -R qus_mtp_round_test --output-on-failure
```

Exit status: 0.

Key output:
```text
100% tests passed, 0 tests failed out of 1
```

Command:
```bash
compute-sanitizer --tool memcheck ./build/tests/qus_mtp_round_test
```

Exit status: 0.

Key output:
```text
OK mtp_round correctness
========= ERROR SUMMARY: 0 errors
```

### 2026-07-03 Model Helper Surface

Command:
```bash
cmake --build build --target qus_core qus_mtp_round_test qus_model_blocks_test qus_model_bind_test -j2 &&
ctest --test-dir build -R 'qus_mtp_round_test|qus_model_blocks_test|qus_model_bind_test' --output-on-failure
```

Exit status: 0.

Key output:
```text
100% tests passed, 0 tests failed out of 3
```

### 2026-07-03 CLI And Bench Schema Surface

Command:
```bash
cmake --build build --target qus_qwen_text_cli_test qus_bench_support_test -j2 &&
ctest --test-dir build -R 'qus_qwen_text_cli_test|qus_bench_support_test' --output-on-failure
```

Exit status: 0.

Key output:
```text
100% tests passed, 0 tests failed out of 2
```

### 2026-07-03 Engine MTP Smoke And Stats

Command:
```bash
cmake --build build --target qus_bench qus_engine_memory_stats_test \
  qus_engine_real_file_test qus_bench_support_test qus_mtp_round_test -j2 &&
ctest --test-dir build \
  -R 'qus_engine_memory_stats_test|qus_engine_real_file_test|qus_bench_support_test|qus_mtp_round_test' \
  --output-on-failure &&
./build/bench/qus_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus \
  -n 2 --mtp-draft-tokens 1 --warmup 0 -r 1 --no-cuda-graph \
  -o json --output-file /tmp/qus_mtp_smoke.json
```

Exit status: 0.

Key output:
```text
100% tests passed, 0 tests failed out of 4
wrote /tmp/qus_mtp_smoke.json
```

Smoke report MTP section:
```json
{
  "enabled": true,
  "k": 1,
  "draft_tokens": 1,
  "accepted_tokens": 0,
  "acceptance_rate": 0,
  "acceptance_length": 1,
  "rounds": 1,
  "fallback_steps": 1,
  "accepted_per_pos": [0]
}
```

Command:
```bash
cmake --build build --target qus qus_bench qus_mtp_round_test qus_model_bind_test \
  qus_model_blocks_test qus_engine_memory_stats_test qus_engine_real_file_test \
  qus_qwen_text_cli_test qus_bench_support_test -j2 &&
ctest --test-dir build \
  -R 'qus_mtp_round_test|qus_model_bind_test|qus_model_blocks_test|qus_engine_memory_stats_test|qus_engine_real_file_test|qus_qwen_text_cli_test|qus_bench_support_test' \
  --output-on-failure
```

Exit status: 0.

Key output:
```text
100% tests passed, 0 tests failed out of 7
```

Command:
```bash
./build/bench/qus_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus \
  -n 4 --max-ctx 16 --mtp-draft-tokens 5 --warmup 0 -r 1 --no-cuda-graph \
  -o json --output-file /tmp/qus_mtp_k5_smoke.json
```

Exit status: 0.

Smoke report MTP section:
```json
{
  "enabled": true,
  "k": 5,
  "draft_tokens": 15,
  "accepted_tokens": 1,
  "acceptance_rate": 0.06666666667,
  "acceptance_length": 1.333333333,
  "rounds": 3,
  "fallback_steps": 0,
  "accepted_per_pos": [1, 0, 0, 0, 0]
}
```

### 2026-07-04 Strict Sequential And Round Dump Surface

Command:
```bash
cmake --build build --target qus_engine_real_file_test -j2 &&
ctest --test-dir build -R qus_engine_real_file_test --output-on-failure
```

Exit status: 0.

Key output:
```text
1/1 Test #11: qus_engine_real_file_test ........   Passed  102.11 sec
100% tests passed, 0 tests failed out of 1
```

Coverage: the real-file test now checks prompt `{1}` for three generated tokens. Strict-sequential
MTP output matches MTP-off for `k=1..5`; batched MTP output matches strict-sequential for `k=1..5`;
all MTP runs record rounds and zero fallback steps for this capacity-safe prompt.

Command:
```bash
cmake --build build --target qus qus_bench qus_mtp_round_test qus_model_bind_test \
  qus_model_blocks_test qus_engine_memory_stats_test qus_engine_real_file_test \
  qus_qwen_text_cli_test qus_bench_support_test -j2 &&
ctest --test-dir build \
  -R 'qus_mtp_round_test|qus_model_bind_test|qus_model_blocks_test|qus_engine_memory_stats_test|qus_engine_real_file_test|qus_qwen_text_cli_test|qus_bench_support_test' \
  --output-on-failure
```

Exit status: 0.

Key output:
```text
100% tests passed, 0 tests failed out of 7
```

Command:
```bash
./build/bench/qus_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus \
  -n 3 --max-ctx 32 --mtp-draft-tokens 2 --mtp-strict-sequential \
  --warmup 0 -r 1 --no-cuda-graph -o json \
  --output-file /tmp/qus_mtp_strict_smoke.json
```

Exit status: 0.

Strict smoke report MTP section:
```json
{
  "enabled": true,
  "k": 2,
  "draft_tokens": 6,
  "accepted_tokens": 1,
  "acceptance_rate": 0.1666666667,
  "acceptance_length": 1.333333333,
  "rounds": 3,
  "fallback_steps": 0,
  "accepted_per_pos": [1, 0]
}
```

Command:
```bash
cmake --build build --target qus_runtime_file_tap_test qus_qwen_text_cli_test \
  qus_engine_real_file_test -j2 &&
ctest --test-dir build \
  -R 'qus_runtime_file_tap_test|qus_qwen_text_cli_test|qus_engine_real_file_test' \
  --output-on-failure
```

Exit status: 0.

Key output:
```text
100% tests passed, 0 tests failed out of 3
```

Coverage: `qus_runtime_file_tap_test` now validates the round dump writer on tiny `GdnState` and
`KVCache` tensors, including exact filenames, f32 payloads, manifest dtype marker, committed length,
and MTP KV filename. The text CLI test validates `--mtp-round-dump-dir DIR`.

### 2026-07-04 L1 Sanitizer Refresh

Command:
```bash
cmake --build build --target qus_mtp_round_test -j2 &&
ctest --test-dir build -R qus_mtp_round_test --output-on-failure &&
compute-sanitizer --tool memcheck ./build/tests/qus_mtp_round_test
```

Exit status: 0.

Key output:
```text
1/1 Test #38: qus_mtp_round_test ...............   Passed    0.24 sec
OK mtp_round correctness
========= ERROR SUMMARY: 0 errors
```

Coverage: the L1 round helper test now includes partial accept, all reject, all accept with bonus,
shifted overwrite for partial/all-accept, device-row gather, scalar increment, fallback counter, and
validation errors.

### 2026-07-04 Real-Weight E2E, Memory, And Foundation Bench

Command:
```bash
./build/tests/qus_engine_real_file_test
```

Exit status: 0.

Key output:
```text
ENGINE_REAL mtp=0 loaded_payload=16378329088 weight_capacity=17842987520 weight_used=16378329088 tensor_count=1164 quant_count=634
ENGINE_REAL mtp=1 loaded_payload=16829596672 weight_capacity=17842987520 weight_used=16829596672 tensor_count=1164 quant_count=634
ENGINE_REAL mtp=5 max_ctx=8192 weight_capacity=17842987520 cache_capacity=1574773740 workspace_capacity=385875968 total_capacity=19803637228
```

Coverage: real q5090 `k=5/max_ctx=8192` combined arena capacity is `19,803,637,228` bytes
(~18.45 GiB), below the 32 GiB device budget. The same test covers strict k=1..5 equality with
MTP-off, batched k=1..5 equality with strict for prompt `{1}`, in-round stop truncation, and decode
capacity fallback.

Command:
```bash
/home/neroued/miniconda3/envs/py311/bin/python - <<'PY'
from tools.parity.ref_model import DEFAULT_STOP_TOKEN_IDS, RefModel, parse_stop_token_ids
weights = 'out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus'
prompt_ids = [int(x) for x in open('/tmp/mtp_foundation_prompt.ids').read().split()]
decode = 64
model = RefModel(weights, device='cuda', resident='auto')
stop = parse_stop_token_ids(DEFAULT_STOP_TOKEN_IDS)
base = model.forward(prompt_ids, decode, stop_token_ids=stop)
tokens, stats = model.forward_mtp_verified(prompt_ids, decode, draft_count=5, stop_token_ids=stop)
print(f'PROMPT_TOKENS {len(prompt_ids)}')
print(f'DECODE {decode}')
print(f'MATCH {tokens == base}')
print(f'DRAFT_TOKENS {stats.draft_tokens}')
print(f'ACCEPTED_TOKENS {stats.accepted_tokens}')
print(f'ACCEPTANCE_RATE {stats.acceptance_rate:.6f}')
print(f'ACCEPTANCE_LENGTH {stats.acceptance_length:.6f}')
print(f'DRAFTED_PER_POS {stats.draft_tokens_per_pos}')
print(f'ACCEPTED_PER_POS {stats.accepted_tokens_per_pos}')
PY
```

Exit status: 0.

Key output:
```text
PROMPT_TOKENS 89
DECODE 64
MATCH True
DRAFT_TOKENS 60
ACCEPTED_TOKENS 50
ACCEPTANCE_RATE 0.833333
ACCEPTANCE_LENGTH 4.846154
DRAFTED_PER_POS [13, 12, 12, 12, 11]
ACCEPTED_PER_POS [13, 12, 10, 8, 7]
```

Command:
```bash
./build/bench/qus_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus \
  --corpus /tmp/mtp_foundation_prompt.ids -pg 89,64 --max-ctx 256 \
  --mtp-draft-tokens 5 --warmup 0 -r 1 --no-cuda-graph -o json \
  --output-file /tmp/qus_mtp_foundation_k5.json
```

Exit status: 0.

C++ k=5 MTP section:
```json
{
  "enabled": true,
  "k": 5,
  "draft_tokens": 65,
  "accepted_tokens": 54,
  "acceptance_rate": 0.8307692308,
  "acceptance_length": 5.153846154,
  "rounds": 13,
  "fallback_steps": 0,
  "accepted_per_pos": [13, 13, 11, 9, 8]
}
```

C++ timing:
```text
k=0 pp89+tg64 decode_tok_s=71.2920686
k=5 pp89+tg64 decode_tok_s=55.18569109
```

Interpretation: C++ acceptance is the same order as the Python oracle (`0.830769` vs `0.833333`).
The slight count difference is expected from C++ fixed-k overshoot-and-truncate near the generation
limit, while the Python oracle clips the final draft count to the remaining decode budget. Eager M4
k=5 is slower than k=0 on this prompt; M5 owns small-T verify/launch-count performance work.

### 2026-07-04 Full Build And Test

Command:
```bash
cmake --build build -j2
```

Exit status: 0.

Key output:
```text
[98/98] Linking CXX executable bench/qus_target_verify_bench
```

Command:
```bash
ctest --test-dir build --output-on-failure
```

Exit status: 0.

Key output:
```text
100% tests passed, 0 tests failed out of 38
Total Test time (real) = 223.67 sec
```

### 2026-07-04 Code Review Resolution

Reviewer: strongest-model subagent `Faraday`.

Accepted findings and fixes:

- Prefill MTP capacity guard: final prefill now skips MTP draft proposal setup when
  `prompt_len + k - 1 > mtp_kv.max_context`; decode then uses the existing T=1 fallback path.
- `Engine::generate` now clears pending sampled tokens at generation boundaries, so max-new
  overshoot cannot leak through a later `decode_step()`.
- `Engine::load` clears pending sampled tokens on reload.
- `qus_bench` now reports `decode_path = mtp_eager` or `mtp_strict` whenever MTP is enabled, even if
  `use_cuda_graph` is true; table output includes an `mtp acc` column.
- Round dump manifests now mark committed MTP KV range, active proposal range, dumped padded range,
  and `gdn_state = slot0_committed`.
- Added real-weight E2E scenario target for strict, batched, capacity fallback, and stop truncation
  sanitizer runs.

Deferred minor finding:

- Last-chunk MTP prefill still copies `t0` through host when building `mtp_ids`. This is an
  implementation deviation from the preferred device insertion, but it happens at prefill setup
  only, does not change the round state machine, and is covered by strict/batched real-weight
  verification. It is not a blocker for M4 correctness.

Command:
```bash
cmake --build build --target qus_core qus_bench qus_bench_support_test \
  qus_runtime_file_tap_test qus_engine_real_file_test -j2 &&
ctest --test-dir build \
  -R 'qus_bench_support_test|qus_runtime_file_tap_test|qus_engine_real_file_test' \
  --output-on-failure
```

Exit status: 0.

Key output:
```text
1/3 Test #11: qus_engine_real_file_test ........   Passed  167.69 sec
2/3 Test #20: qus_bench_support_test ...........   Passed    0.01 sec
3/3 Test #21: qus_runtime_file_tap_test ........   Passed    0.22 sec
100% tests passed, 0 tests failed out of 3
```

Command:
```bash
./build/bench/qus_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus \
  -n 1 --max-ctx 16 --mtp-draft-tokens 1 --warmup 0 -r 1 \
  -o json --output-file /tmp/qus_mtp_decode_path_smoke.json &&
jq '.config.decode_path, (.tests[0].mtp)' /tmp/qus_mtp_decode_path_smoke.json
```

Exit status: 0.

Key output:
```text
"mtp_eager"
"rounds": 1
"fallback_steps": 0
```

Command:
```bash
ctest --test-dir build -R qus_engine_mtp_e2e_strict_test --output-on-failure &&
./build/tests/qus_engine_mtp_e2e_strict_test batched &&
./build/tests/qus_engine_mtp_e2e_strict_test capacity_fallback &&
./build/tests/qus_engine_mtp_e2e_strict_test stop_truncation
```

Exit status: 0.

Key output:
```text
1/1 Test #12: qus_engine_mtp_e2e_strict_test ...   Passed   27.42 sec
OK MTP E2E scenario batched
OK MTP E2E scenario capacity_fallback
OK MTP E2E scenario stop_truncation
```

Command:
```bash
compute-sanitizer --tool memcheck ./build/tests/qus_engine_mtp_e2e_strict_test capacity_fallback
compute-sanitizer --tool memcheck ./build/tests/qus_engine_mtp_e2e_strict_test stop_truncation
compute-sanitizer --tool memcheck ./build/tests/qus_engine_mtp_e2e_strict_test strict
compute-sanitizer --tool memcheck ./build/tests/qus_engine_mtp_e2e_strict_test batched
```

Exit status: 0 for all four commands.

Key output:
```text
OK MTP E2E scenario capacity_fallback
========= ERROR SUMMARY: 0 errors
OK MTP E2E scenario stop_truncation
========= ERROR SUMMARY: 0 errors
OK MTP E2E scenario strict
========= ERROR SUMMARY: 0 errors
OK MTP E2E scenario batched
========= ERROR SUMMARY: 0 errors
```

Command:
```bash
git diff --check &&
cmake --build build -j2 &&
ctest --test-dir build --output-on-failure
```

Exit status: 0.

Key output:
```text
[108/108] Linking CXX executable bench/qus_target_verify_bench
100% tests passed, 0 tests failed out of 39
Total Test time (real) = 264.05 sec
```
