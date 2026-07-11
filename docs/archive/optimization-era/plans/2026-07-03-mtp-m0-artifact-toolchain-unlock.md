# MTP M0 Artifact And Toolchain Unlock Implementation Plan

**Goal:** Land MTP roadmap milestone M0 by making the C++ loader and local toolchain accept the v3+MTP q5090 artifact with W8G32 MTP weights, without adding W8 execution or MTP runtime semantics.

**Requirements:** R-FMT-1, R-FMT-2, and R-FMT-3 from `docs/2026-07-03-mtp-implementation-requirements.md` §1. The milestone boundary and gate are the M0 section of `docs/2026-07-03-mtp-roadmap.md`.

**Execution Mode:** Main agent direct implementation. Subagent-driven implementation is intentionally not used because M0 is small, format changes are tightly coupled across parser/descriptors/fixtures, and the user explicitly requested main-agent implementation. Subagents are used only for two independent reviews: one plan audit before implementation and one code review after self-test. Every subagent must use the current strongest available model, `gpt-5.5`, with `reasoning_effort: "xhigh"` and `service_tier: "priority"`.

**Non-Goals:**
- Do not add W8 support to `kernels::linear()` or any W8 CUDA kernel.
- Do not change `tests/kernels/test_linear.cpp` behavior that rejects W8 qtypes.
- Do not add MTP model binding, `mtp_forward`, model-card schedule changes, round loops, GDN/KV work, CLI flags, or runtime MTP semantics.
- Do not preserve old behavior that treats qtype tag 6 as invalid.

## Scope And Ownership

- `include/qus/core/tensor.h`: owns the C++ qtype ABI enum. Add `W8G32_F16S = 6`.
- `src/core/weight_store_parser.cpp`: owns structural q5090 validation. Accept tag 6 and validate W8G32 as a quantized row-split qtype with group 32, base bytes/group 32, high bytes/group 0, FP16 scales, and v3 payload math.
- `src/core/weight_store.cpp`: owns runtime `Weight` descriptors. Mirror W8G32 plane math so segment and fused-block descriptors point at the correct base and scale rows.
- `include/qus/runtime/engine.h` and `src/runtime/engine.cpp`: own public engine load options and default arena sizing. Add `EngineOptions::mtp_draft_tokens` with default 0; validate nonnegative values; pass `load_mtp = mtp_draft_tokens > 0`; include the fixed MTP payload budget of `451,267,584` bytes in default weight arena sizing; validate MTP module expectations when MTP is enabled.
- `tests/fixtures/make_q5090_fixture.py`: owns compact fixture generation. Replace the stale two-block MTP fixture with the v3 MTP module shape: 12 blocks, 16 segments, 2 fusion groups, five W8G32 dense/fused linears, seven BF16 controls.
- `tests/kernels/q5090_pack.h`: owns the C++ test-only row-split packer. Add W8G32 quant spec and pack/decode handling matching Python `packing.py`.
- `tests/test_q5090_parser.cpp`, `tests/test_weight_store.cpp`, `tests/test_q5090_pack_golden.cpp`, and focused engine/load tests: own allowed format-contract coverage for legal W8G32 parsing, malformed W8G32 payload rejection, C++/Python golden pack bytes, and MTP fixture parse/load.
- `tests/test_weight_store_real.cpp`: owns optional real-artifact inventory/load coverage. Update it to the `_mtp_w8g32` artifact and W8G32 manifest/module constants.
- `bench/linear_op_bench.cu`: owns qtype parsing for the per-op bench. Accept W8G32 names and compute group 32 plane sizes/descriptors, while existing `kernels::linear()` rejection remains the runtime outcome until M1.
- Optional supporting tools that compile against qtype switches, such as `tools/parity/q5090_structural_dump.h`, may be updated only to keep qtype naming/structural dumps correct for tag 6.

## Coordination Points

- `QType` enum and every qtype switch must land together so the tree compiles.
- `tests/fixtures/make_q5090_fixture.py` changes affect parser, weight-store, model-bind, model-blocks, and engine-memory tests. Counts in C++ tests must be updated to the new fixture contract.
- Engine default weight bytes and MTP expectations must remain load-only. They must not require model-card MTP binding or change `generate()` behavior when `mtp_draft_tokens == 0`.
- Real artifact paths for gate verification are:
  - `out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus`
  - `out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus.manifest.json`

## Task Breakdown

### Task 1: W8G32 ABI And Parser Contract

**Reading List:**
- `docs/q5090_packed_file_format_v3.md` §7, §9.1, §9.2, §13, §14.
- `tools/q5090_convert/qtypes.py`.
- `tools/q5090_convert/packing.py`.
- `include/qus/core/tensor.h`.
- `src/core/weight_store_parser.cpp`.

**Implementation:**
- Add `QType::W8G32_F16S = 6`.
- Update `qtype_from_tag`, `is_quant_qtype`, `quant_group_size`, `nibble_bytes_per_group`, and `high_bytes_per_group`.
- Ensure malformed W8G32 row-split metadata is rejected through the existing structural checks:
  - group size other than 32;
  - nonzero high plane;
  - wrong nibble or scale plane byte count;
  - wrong payload size from `scale_rel + scale_plane_bytes`.

**Definition Of Done:**
- A valid fixture W8G32 row-split block parses as tag 6 with group size 32, high plane 0, and expected v3 plane sizes.
- Mutating the same fixture to malformed W8G32 metadata throws from `parse_q5090_file`.

**Verification:**
- `ctest --test-dir build --output-on-failure -R 'qus_q5090_parser_test'`

### Task 2: WeightStore Descriptors And MTP Load Expectations

**Reading List:**
- `docs/q5090_packed_file_format_v3.md` §9.3, §10.1, §14.
- `src/core/weight_store.cpp`.
- `include/qus/core/weight_store.h`.
- `include/qus/runtime/engine.h`.
- `src/runtime/engine.cpp`.
- `tests/test_weight_store.cpp`.
- `tests/test_engine_memory_stats.cpp`.

**Implementation:**
- Mirror W8G32 row-split plane constants in `weight_store.cpp` descriptor math.
- Add `EngineOptions::mtp_draft_tokens` as `int`, default `0`.
- Reject negative `mtp_draft_tokens` in the `Engine` constructor.
- In `Engine::load`, set `LoadOptions::load_mtp = options_.mtp_draft_tokens > 0`.
- This is the user-requested M0 load-only subset of `mtp_draft_tokens`: do not add CLI wiring, do not enforce M4's `[1,5]` runtime range, and do not attach any execution semantics beyond `load_mtp = (k > 0)`.
- Keep runtime behavior unchanged after load: no CLI option, no MTP execution path, and no changes to `generate()`, `prefill()`, or `decode_step()`.
- Add a direct MTP expectation validation when MTP is requested:
  - MTP module exists with 12 blocks.
  - MTP contributes 16 segments.
  - MTP contributes 2 fusion groups.
  - Exactly five MTP row-split blocks are W8G32: `mtp.fc.weight`, `mtp.layers.0.attn_in.w8`, `mtp.layers.0.self_attn.o_proj.weight`, `mtp.layers.0.mlp.gateup.w8`, `mtp.layers.0.mlp.down_proj.weight`.
  - Exactly seven MTP contiguous controls are BF16: `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`, `mtp.layers.0.input_layernorm.weight`, `mtp.layers.0.self_attn.q_norm.weight`, `mtp.layers.0.self_attn.k_norm.weight`, `mtp.layers.0.post_attention_layernorm.weight`, `mtp.norm.weight`.
  - MTP fusion groups are `ATTN_IN` and `MLP_GATEUP`, both one W8G32 block, layer 0.
- Include `451,267,584` bytes in `default_weight_bytes()` so the default weight arena covers the MTP payload and alignment overhead even when MTP is enabled.

**Definition Of Done:**
- `WeightStore` can load text-only metadata with unloaded MTP descriptors.
- `Engine::load` with `mtp_draft_tokens == 0` still loads text only.
- `Engine::load` with `mtp_draft_tokens > 0` loads MTP payloads and validates expectations without binding MTP into the model card.
- Memory stats show higher loaded payload and weight arena usage when MTP is enabled.

**Verification:**
- `ctest --test-dir build --output-on-failure -R 'qus_weight_store_test|qus_engine_memory_stats_test'`

### Task 3: Fixture And Packer Toolchain

**Reading List:**
- `tests/fixtures/make_q5090_fixture.py`.
- `tests/kernels/q5090_pack.h`.
- `tests/test_q5090_pack_golden.cpp`.
- `tools/q5090_convert/layouts.py`.
- `tools/q5090_convert/packing.py`.
- `tools/q5090_convert/tensor_plan.py`.

**Implementation:**
- Replace stale default-fixture MTP blocks with the canonical v3 MTP module.
- Preserve compact fixture sizes where possible by using small shapes for non-real fixture profiles, but preserve the exact block names, qtypes, source kinds, segment layout, module ordering, and fusion records required by v3.
- Ensure `model-bind` and `model-blocks` fixture profiles include MTP blocks so load-and-parse coverage sees W8G32 in all relevant generated fixtures.
- Add W8G32 to the C++ test packer:
  - bits 8;
  - group size 32;
  - qmax 127;
  - qmin -127;
  - base plane is raw signed int8 bytes;
  - high plane is empty.
- Extend the Python golden test to include W8G32 and assert the expected Python encoder metadata for group 32.

**Definition Of Done:**
- Fixture parse sees MTP `12/16/2` counts.
- C++ W8G32 pack bytes exactly match Python `encode_tensor`.
- Existing low-bit pack golden cases remain covered.

**Verification:**
- `ctest --test-dir build --output-on-failure -R 'qus_q5090_pack_golden_test|qus_q5090_parser_test|qus_model_bind_test|qus_model_blocks_test'`

### Task 4: Bench Qtype Parsing

**Reading List:**
- `bench/linear_op_bench.cu`.
- `src/kernels/linear/linear.cpp`.
- `tests/kernels/test_linear.cpp`.

**Implementation:**
- Add W8G32 to `linear_op_bench.cu` qtype names and parsing (`W8G32`, `w8g32`, `w8g32_f16s`).
- Add group-size-aware plane math so W8G32 uses group 32, base bytes/group 32, high bytes/group 0, and `Weight::group_size/group = 32`.
- Do not add W8 to `kTask2Targets`; M1 owns W8 shape scanning and kernels.
- Do not change `kernels::linear()` support. The bench may accept qtype parsing while the kernel rejects W8 at launch until M1.

**Definition Of Done:**
- `qus_linear_op_bench --shape MlpGateUp34816x5120 --qtype W8G32 --repeat 1 --warmup 0 --copy-repeat 1 --t-sweep 1 --stream-ceiling-gbs 1` accepts the qtype and reaches the current W8 rejection path.
- `tests/kernels/test_linear.cpp` still rejects W8 qtypes.

**Verification:**
- `cmake --build build --target qus_linear_op_bench qus_linear_test`
- `ctest --test-dir build --output-on-failure -R 'qus_linear_test'`
- `./build/bench/qus_linear_op_bench --shape MlpGateUp34816x5120 --qtype W8G32 --repeat 1 --warmup 0 --copy-repeat 1 --t-sweep 1 --stream-ceiling-gbs 1`
  - Expected for M0: argument parsing succeeds, W8G32 descriptor construction uses group 32, and the process exits nonzero at the current `linear: unsupported weight qtype` path.

### Task 5: Full M0 Gate And Evidence

**Reading List:**
- `docs/2026-07-03-mtp-roadmap.md` M0 gate.
- `bench/README.md`.
- `tools/parity/greedy_match.py`.
- `bench/qus_bench_support.cpp`.

**Implementation:**
- Add a verification evidence section to this plan after commands are run, with exact commands and observed results.
- Keep commits conventional and split by meaningful behavior once verification evidence is recorded.

**Definition Of Done:**
- All M0 gate items have fresh command evidence or an explicit environmental skip with reason.
- Git status is clean after commits.

**Verification Commands:**
- `cmake --build build`
- `ctest --test-dir build --output-on-failure`
- Focused new/changed tests:
  - `ctest --test-dir build --output-on-failure -R 'qus_q5090_parser_test|qus_q5090_pack_golden_test|qus_weight_store_test|qus_engine_memory_stats_test|qus_linear_test|qus_weight_store_real_file_test'`
- Real artifact load:
  - `ctest --test-dir build --output-on-failure -R 'qus_weight_store_real_file_test'`
  - a dedicated engine load smoke with `mtp_draft_tokens=0` and `mtp_draft_tokens=1`;
  - record `memory_stats()` weight arena capacity/used and `q5090_loaded_payload_bytes`.
- Pure text regression:
  - `python3 tools/parity/greedy_match.py --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus`
- Bench short run:
  - `./build/bench/qus_bench --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus -pg 128,8 -r 1 --warmup 0 --max-ctx 256 --prefill-chunk 128`

## Review Phase

### Plan Review

Before implementation, dispatch one `gpt-5.5` review subagent with `reasoning_effort: "xhigh"` and `service_tier: "priority"`. The reviewer must audit this plan against:
- `docs/2026-07-03-mtp-roadmap.md` M0 boundary;
- `docs/2026-07-03-mtp-implementation-requirements.md` R-FMT-1/2/3;
- `docs/q5090_packed_file_format_v3.md` W8G32 and MTP assignment;
- `tools/q5090_convert/qtypes.py` and `tools/q5090_convert/packing.py`.

Revise this plan before implementation for any missing required behavior, scope creep, or weak verification.

### Code Review

After implementation and local self-test, dispatch one `gpt-5.5` review subagent with `reasoning_effort: "xhigh"` and `service_tier: "priority"`. The reviewer must focus on:
- parser boundaries and malformed input rejection;
- C++/Python W8G32 pack byte equality;
- MTP expectations and fixture counts;
- arena budget math and overflow handling;
- avoiding M1/M2/M3 scope creep.

Fix all Critical and Important findings before running the final gate.

## Testing Policy Notes

- New/updated parser and packer tests are allowed under AGENTS.md hard whitelist 2 because they protect binary/file-format contracts and malformed q5090 rejection.
- Fixture parse/load coverage is allowed under whitelist 4 because it validates observable q5090 load behavior on canonical compact fixtures.
- No tests will assert source structure, implementation call order, deprecated behavior, or W8 execution support.

## Verification Evidence

This section is the append-only post-implementation evidence log. Each entry must include the exact command, exit status, and key output lines from the fresh run.

### 2026-07-03 Final M0 Gate

- Plan review subagent: `gpt-5.5`, `reasoning_effort=xhigh`, `service_tier=priority`.
  - Result: plan revised before implementation to include real `_mtp_w8g32` artifact coverage, clarify M0 load-only `mtp_draft_tokens`, and add W8G32 `linear_op_bench` verification.
- Code review subagent: `gpt-5.5`, `reasoning_effort=xhigh`, `service_tier=priority`.
  - Finding: MTP expectation validation needed canonical block/segment/fusion layout checks, not just aggregate counts.
  - Fix: `WeightStore::require_mtp_module_expectations()` now validates canonical names, source kinds/layers, W8G32/BF16 types, segment row order, fused block layout, and real-artifact MTP dimensions when the real payload size is present; `qus_weight_store_test` includes a parser-accepted swapped `ATTN_K`/`ATTN_GATE` malformed fixture that expectation validation rejects.

- Command: `cmake --build build`
  - Exit: 0.
  - Evidence: build completed all targets, ending with `Linking CXX executable bench/qus_bench`.

- Command: `ctest --test-dir build --output-on-failure`
  - Exit: 0.
  - Evidence: `100% tests passed, 0 tests failed out of 35`; total test time `95.20 sec`.

- Command: `ctest --test-dir build --output-on-failure -R 'qus_weight_store_test|qus_weight_store_real_file_test|qus_engine_memory_stats_test|qus_engine_real_file_test'`
  - Exit: 0.
  - Evidence: `100% tests passed, 0 tests failed out of 4`; real file tests passed for `qus_weight_store_real_file_test` and `qus_engine_real_file_test`.

- Command: `./build/tests/qus_engine_real_file_test`
  - Exit: 0.
  - Evidence:
    - `ENGINE_REAL mtp=0 loaded_payload=16378329088 weight_capacity=17842987520 weight_used=16378329088 tensor_count=1164 quant_count=634`
    - `ENGINE_REAL mtp=1 loaded_payload=16829596672 weight_capacity=17842987520 weight_used=16829596672 tensor_count=1164 quant_count=634`

- Command: `./build/bench/qus_linear_op_bench --shape MlpGateUp34816x5120 --qtype W8G32 --repeat 1 --warmup 0 --copy-repeat 1 --t-sweep 1 --stream-ceiling-gbs 1`
  - Exit: 2, expected for M0.
  - Evidence: argument parsing accepted `W8G32`; execution reached current boundary error `qus_linear_op_bench: linear: unsupported weight qtype`.

- Command: `/home/neroued/miniconda3/envs/py311/bin/python <inline fixture regeneration for cn_short>`
  - Exit: 0.
  - Evidence: regenerated `/tmp/qus_cn_short.ids` from the `tools/parity/ref_model.py` default Chinese prompt and local Qwen tokenizer because `bench/fixtures/prompts/cn_short.ids` is not present in this checkout; token count `31`, `.ids` SHA256 `bbda603ae1ad452a0b03e24aa068cf4aa99fa00eeb0eb7339487932282d26630`, matching `profiles/e2e/m3-output-gate.json`.

- Command: `/home/neroued/miniconda3/envs/py311/bin/python tools/parity/greedy_match.py --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus --fixture /tmp/qus_cn_short.ids --case cn_short`
  - Exit: 0.
  - Evidence: `PASS snapshot token match length=26`; generated ids exactly matched `profiles/e2e/m3-output-gate.json` case `cn_short` repeat 0.

- Command: `./build/bench/qus_bench --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus -pg 128,8 -r 1 --warmup 0 --max-ctx 256 --prefill-chunk 128`
  - Exit: 0.
  - Evidence: loaded `17123284480` byte `_mtp_w8g32` artifact and completed `pp128+tg8` with `prefill t/s 1065.17`, `decode t/s 78.48`, `work peak 22.6 MiB`.

### Plan Review Evidence

- Review subagent: `gpt-5.5`, `reasoning_effort: xhigh`, `service_tier: priority`.
- Result: plan revised before implementation to add real-artifact coverage for
  `tests/test_weight_store_real.cpp`, clarify that `EngineOptions::mtp_draft_tokens` is M0
  load-only behavior with no CLI/runtime semantics, and add explicit `qus_linear_op_bench --qtype
  W8G32` verification.

### Code Review Evidence

- Review subagent: `gpt-5.5`, `reasoning_effort: xhigh`, `service_tier: priority`.
- Important finding: `WeightStore::require_mtp_module_expectations()` initially validated MTP mainly
  by counts and source-kind lookups, leaving canonical MTP block/segment layout malformed inputs
  underchecked.
- Resolution: tightened MTP expectation validation to require canonical MTP block names, W8G32/BF16
  qtypes, dimension relationships, fusion records, and ATTN/MLP segment names/source kinds/row
  ranges. Added a `tests/test_weight_store.cpp` malformed fixture case that swaps MTP ATTN_K and
  ATTN_GATE segment source kinds; parser accepts the structurally valid file, and MTP expectation
  validation rejects it.

### Build And Tests

- Command: `cmake --build build`
  - Exit status: `0`.
  - Evidence: completed the full build; final lines included linking `bench/qus_linear_op_bench` and
    `bench/qus_bench`.
- Command: `ctest --test-dir build --output-on-failure`
  - Exit status: `0`.
  - Evidence: `100% tests passed, 0 tests failed out of 35`; total test time `95.20 sec`.
  - Coverage included the new W8G32 parser malformed-input checks, C++/Python pack golden, MTP
    fixture parse/load, linear W8 rejection, real weight-store file coverage, and real Engine load
    coverage.

### Real Artifact Load And Arena Evidence

- Command: `./build/tests/qus_engine_real_file_test`
  - Exit status: `0`.
  - Evidence:
    - `ENGINE_REAL mtp=0 loaded_payload=16378329088 weight_capacity=17842987520 weight_used=16378329088 tensor_count=1164 quant_count=634`
    - `ENGINE_REAL mtp=1 loaded_payload=16829596672 weight_capacity=17842987520 weight_used=16829596672 tensor_count=1164 quant_count=634`
  - Interpretation: enabling `mtp_draft_tokens=1` loads an additional `451,267,584` bytes and the
    default weight arena capacity includes the MTP payload budget.

### W8G32 Bench Qtype Boundary

- Command: `./build/bench/qus_linear_op_bench --shape MlpGateUp34816x5120 --qtype W8G32 --repeat 1 --warmup 0 --copy-repeat 1 --t-sweep 1 --stream-ceiling-gbs 1`
  - Exit status: `2`.
  - Evidence: output starts with `qus_linear_op_bench: linear: unsupported weight qtype`; usage text
    includes `--qtype Q4|Q5|Q6|W8G32`.
  - Interpretation: M0 qtype parsing and descriptor construction accepts `W8G32`, then reaches the
    existing `linear()` unsupported-W8 boundary as intended.

### Short Real-Weight Bench

- Command: `./build/bench/qus_bench --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus -pg 128,8 -r 1 --warmup 0 --max-ctx 256 --prefill-chunk 128`
  - Exit status: `0`.
  - Evidence: loaded `out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus (17123284480 bytes)` and
    completed `pp128+tg8` with prefill `1065.17 t/s`, decode `78.48 t/s`, workspace peak `22.6 MiB`.

### Greedy Snapshot Gate

- Command attempted: `python3 tools/parity/greedy_match.py --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus`
  - Exit status: `1`.
  - Evidence: system `python3` reported CPU-only PyTorch and requested using a CUDA Python
    environment.
- Command attempted: `/home/neroued/miniconda3/envs/py311/bin/python tools/parity/greedy_match.py --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus --case cn_short`
  - Exit status: `1`.
  - Evidence: the command read the default `bench/fixtures/bench_corpus.ids` prompt while comparing
    the `cn_short` snapshot, then failed with CUDA OOM from a 65,536-token prompt.
- Command attempted: `/home/neroued/miniconda3/envs/py311/bin/python tools/parity/greedy_match.py --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus --fixture bench/fixtures/prompts/cn_short.ids --case cn_short`
  - Exit status: `1`.
  - Evidence: `FileNotFoundError: bench/fixtures/prompts/cn_short.ids`; the current checkout only has
    `bench/fixtures/bench_corpus.ids` under `bench/fixtures`.
- User decision: the user explicitly approved skipping this gate for M0 completion in this run.
