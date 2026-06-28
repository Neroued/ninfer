# Test Suite Governance Cleanup Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the existing test suite and active compatibility surfaces into alignment with
`AGENTS.md`.

**Architecture:** Cleanup proceeds in safe layers. Delete tests that protect no real risk, remove
project-owned compatibility entry points, and keep invalid-but-risk-covering structure tests until
valid behavior or sanitizer-backed replacements exist.

**Tech Stack:** C++20/CUDA CTest targets, Python pytest tools, CMake test registration.

---

## Audit Classification

### KEEP

- `tests/test_arena.cpp`
- `tests/test_device.cpp`
- `tests/test_engine_memory_stats.cpp`
- `tests/test_kv_cache.cpp`
- `tests/test_model_bind.cpp`
- `tests/test_model_blocks.cpp`
- `tests/test_model_config.cpp`
- `tests/test_q5090_pack_golden.cpp`
- `tests/test_q5090_parser.cpp`
- `tests/test_runtime_file_tap.cpp`
- `tests/test_state_store.cpp`
- `tests/test_tensor.cpp`
- `tests/test_weight.cpp`
- `tests/test_weight_store.cpp`
- `tests/test_weight_store_real.cpp`
- `tests/fixtures/make_q5090_fixture.py`
- `tests/kernels/op_check.h`
- `tests/kernels/op_tester.h`
- `tests/kernels/gdn_ref.h`
- `tests/kernels/q5090_pack.h`
- `tests/kernels/test_argmax.cpp`
- `tests/kernels/test_causal_conv1d.cpp`
- `tests/kernels/test_embed_gather.cpp`
- `tests/kernels/test_gated_delta_rule.cpp`
- `tests/kernels/test_gdn_gating.cpp`
- `tests/kernels/test_gqa_attention.cpp`
- `tests/kernels/test_l2norm.cpp`
- `tests/kernels/test_linear.cpp`
- `tests/kernels/test_residual_add.cpp`
- `tests/kernels/test_rmsnorm.cpp`
- `tests/kernels/test_rope.cpp`
- `tests/kernels/test_sigmoid_gate_mul.cpp`
- `tests/kernels/test_silu_and_mul.cpp`
- `tests/test_bench_report_tools.py`
- `tests/test_bench_tokenizer_tools.py`
- `tests/test_e2e_bench_support.cpp`
- `tools/q5090_convert/tests/test_packing.py`
- `tools/q5090_convert/tests/test_tensor_plan.py`

### DELETE Or MERGE Without Replacement

- `tests/test_dtype.cpp`
- `tests/test_engine_load.cpp`
- `tests/test_graph_readiness_structure.cpp`
- `tests/test_hardening_cleanup_structure.cpp`
- `tests/kernels/test_gdn_common.cpp`
- `tests/kernels/test_linear_dense_structure.cpp`

### REWRITE Or MERGE Completed

- `tests/test_runtime_file_tap.cpp`
- `tests/test_weight_store_real.cpp`
- `tests/kernels/test_embed_gather.cpp`
- `tests/test_e2e_bench_support.cpp`
- `tests/test_decode_e2e_report_redaction.py`
- `tests/test_weight.cpp`
- `tests/test_bench_report_tools.py`
- `tests/test_bench_tokenizer_tools.py`

## Task 1: First Safe Cleanup Batch

**Files:**
- Delete: `tests/test_dtype.cpp`
- Delete: `tests/test_engine_load.cpp`
- Delete: `tests/test_hardening_cleanup_structure.cpp`
- Delete: `tests/kernels/test_gdn_common.cpp`
- Delete: `tests/kernels/test_linear_dense_structure.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `bench/e2e_bench_support.cpp`
- Modify: `tests/test_e2e_bench_support.cpp`
- Modify: `src/model/qwen3_6_27b.cpp`
- Modify: `tools/parity/greedy_match.py`
- Modify: `tests/test_runtime_file_tap.cpp`

- [x] Remove CMake registrations for deleted tests.
- [x] Delete the five invalid standalone tests listed above.
- [x] Remove the `--eos-token-id` compatibility alias from e2e bench argument parsing and usage.
- [x] Keep only `--stop-token-id` behavior in `tests/test_e2e_bench_support.cpp`.
- [x] Remove FileTap's old `layer_%02d.f32` output.
- [x] Update `tools/parity/greedy_match.py` to consume `layer_%02d_mlp.f32`.
- [x] Remove compatibility-only assertions from `tests/test_runtime_file_tap.cpp`.

Verification:

```bash
cmake --build build -j --target qus_e2e_bench_support_test qus_runtime_file_tap_test qus_engine_memory_stats_test qus_linear_test
ctest --test-dir build --output-on-failure -R '^(qus_e2e_bench_support_test|qus_runtime_file_tap_test|qus_engine_memory_stats_test|qus_linear_test)$'
pytest -q tools/q5090_convert/tests tests/test_bench_report_tools.py tests/test_bench_tokenizer_tools.py
rg -n 'eos-token-id|deprecated|legacy|compat' include src bench tools tests --glob '!third_party/**'
```

Expected:

- Build succeeds.
- Listed CTest targets pass or GPU-dependent targets skip cleanly where no CUDA device is available.
- Pytest passes.
- Compatibility scan has no project-owned compatibility entry points. A test that verifies a removed
  flag is rejected may still mention the old flag text.

Actual verification:

```bash
cmake -S . -B build
cmake --build build -j --target qus_e2e_bench_support_test qus_runtime_file_tap_test qus_engine_memory_stats_test qus_linear_test
ctest --test-dir build --output-on-failure -R '^(qus_e2e_bench_support_test|qus_runtime_file_tap_test|qus_engine_memory_stats_test|qus_linear_test)$'
ctest --test-dir build --output-on-failure
pytest -q tests tools/q5090_convert/tests
```

Result: all commands exited 0 before later graph-readiness deletion; final verification is recorded
below.

## Completed Rewrite And Pruning Tasks

### Task 2: Remove Graph-Readiness Private Helper Test

- [x] Deleted `tests/test_graph_readiness_structure.cpp` after review showed the rewrite still
  locked `src/model/detail` private helpers instead of observable behavior.
- [x] Dropped source-shape checks for exact model call sites, one-thread launch spelling, host scalar
  cache absence, and GDR workspace allocation internals.
- [x] Reused existing `test_model_bind.cpp` coverage for canonical conv1d binding.

### Task 3: Replace Runtime Tap Source Scan

- [x] Replaced `tests/test_runtime_tap_structure.cpp` source scans with
  `tests/test_runtime_file_tap.cpp` FileTap output file creation/content checks and custom tap
  compile-path coverage.
- [x] Removed compatibility-only assertions and internal template/trampoline/order checks.
- [x] Removed EOS validation after runtime stop behavior moved to `stop_token_ids`.
- [x] Switched FileTap output checks to a unique temp directory.
- [x] Dropped engine-generate EOS, layer-dump invocation, and tap/workspace lifetime claims that
  were not behaviorally covered by this target.

### Task 4: Rewrite E2E Bench Support Tests

- [x] Kept `.ids` parsing, stop-token list behavior, fixture manifest loading, max-context
  rejection, and raw/error JSON report behavior.
- [x] Replaced raw/error report substring checks with structured `nlohmann::json` assertions.
- [x] Preserved old `--eos-token-id` only as a rejected unknown argument.

### Task 5: Rewrite Real q5090 Artifact Test

- [x] Kept large-artifact integration intent and optional artifact skip behavior.
- [x] Replaced manifest string search in `tests/test_weight_store_real.cpp` with typed
  `nlohmann::json` validation.

### Task 6: Extend Embed Gather Shape Coverage

- [x] Extended `tests/kernels/test_embed_gather.cpp` with dense and Q6 real hidden-size coverage
  for `d=5120, T=5`.
- [x] Preserved the existing dense/Q6 oracle behavior and sentinel-filled output checks.

### Task 7: Prune Python Tool Test Duplicates

- [x] Moved the unique CLI default-redaction assertion into
  `tests/test_bench_tokenizer_tools.py`.
- [x] Deleted duplicate `tests/test_decode_e2e_report_redaction.py`.
- [x] Removed a private-helper assertion in `tests/test_bench_report_tools.py` and trimmed prose
  locking/call-count checks without removing schema or artifact coverage.

### Task 8: L0 Test Pruning

- [x] Removed standalone `Weight` descriptor field smoke checks from `tests/test_tensor.cpp`.
- [x] Trimmed non-consumer-visible pointer/null field checks from `tests/test_weight.cpp`.
- [x] Kept dense `Weight` to `Tensor` bridge behavior that protects shape/type/data risk.

Actual verification:

```bash
cmake --build build -j --target qus_runtime_file_tap_test qus_e2e_bench_support_test qus_weight_store_real_file_test qus_embed_gather_test qus_weight_test qus_tensor_test
ctest --test-dir build --output-on-failure -R '^(qus_runtime_file_tap_test|qus_e2e_bench_support_test|qus_weight_store_real_file_test|qus_embed_gather_test|qus_weight_test|qus_tensor_test)$'
pytest -q tests/test_bench_report_tools.py tests/test_bench_tokenizer_tools.py tools/q5090_convert/tests
```

Result: covered by final verification below.

### Task 9: Runtime Stop-Token Cleanup

- [x] Replaced `EngineOptions::eos_token_id` with `EngineOptions::stop_token_ids`.
- [x] Replaced `src/main.cpp` `--eos-token` with repeatable `--stop-token-id`.
- [x] Kept report/tool rejection of stale `eos_token_id` fields as schema validation, not as active
  compatibility behavior.

Verification: covered by final verification below.

## Final Verification

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
pytest -q tests tools/q5090_convert/tests
git diff --check
rg -n 'eos-token|eos_token_id|deprecated|legacy|compat' include src bench tools tests --glob '!third_party/**'
rg -n 'read_file\(|expect_present\(|expect_absent\(|source-string|source string|_compare_case_identity|assert_called|call_count' tests --glob '!__pycache__/**'
```

Result: all commands exited 0; full CTest reported 29/29 passing; pytest reported 68 passed and
39 subtests passed. The compatibility scan has no active runtime/CLI compatibility surfaces; the
remaining `eos_token_id` matches are schema/tool rejection checks for stale reports or removed
flags. The source-structure scan has no source-string/private-helper tests; remaining matches are
binary fixture reads and memory-stat behavior helpers.
