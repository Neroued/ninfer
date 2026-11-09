# Tests

The retained tests protect current `.ninfer`, numerical operator, model, runtime-transaction,
benchmark-report, and external protocol behavior. Repository verification principles are defined in
[`../AGENTS.md`](../AGENTS.md); Op contract and CUDA implementation guidance is in
[`../docs/maintainer/op-development.md`](../docs/maintainer/op-development.md).

## Organization

- `artifact/` — v3 framing, directory/binding records, codecs, sharding, selected-object
  materialization, load-time parent slices and two-device placement, and Python-writer/C++-reader
  interoperability;
- `convert/` — source interpretation, Qwen logical mapping, recipe overrides/sharing, optional
  components, resources, proposals and numerical conversion methods;
- `models/qwen3_5/` — config/binding, frontend, state/context stores, workspace, MTP alignment and
  opt-in real Engine integration;
- `ops/` — semantic Op qualification with independent mathematical or state-transition oracles;
  Linear and fused Linear suites are separated by their supported weight/activation paths;
- root C++ tests — core storage, runtime admission/resource policy, public API, serving protocols,
  logging, benchmark reports and causal-scoring evaluation;
- `test_serve_corpus.py` — agreement between the serving request-log schema and its measurement
  consumer.

Tests are grouped by observable risk, not by mirroring every source file or class.
`CMakeLists.txt` includes explicit registrations from `cmake/`, `artifact/`, `models/qwen3_5/`
and `ops/`. Registration helpers live in `cmake/NinferTests.cmake`; included manifests keep
executables and CTest working directories under `build/tests/`.
`ops/op_tester.h` and `ops/op_check.h` own only reusable device/guard and comparison mechanics.
Concrete numerical criteria remain named by the semantic Op suite; there are no cross-Op tolerance
presets.

`ops/quantized_weight.h` is the common packed-weight fixture for Q4/Q5/Q6/Q8, FP8 and NVFP4 Op tests. It
owns deterministic payload generation, device `Weight` views, row views, and independent logical
weight decoding.

## Build and run

Select a Python environment with the dependencies for the tests first. The maintained environment
uses Python 3.11; CMake finds Python 3 without restricting its minor version.
`Python3_EXECUTABLE` selects the interpreter used by interop and frontend tests explicitly.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Alternatively, `cmake --preset dev` enables products, tests and benchmarks together.
After building, `ctest --preset dev` runs the same CTest suite. See
[Build system](../docs/maintainer/build-system.md) for local interpreter presets.

The chat-template reference test uses Python Jinja2.

Run a focused target for a localized change:

```bash
cmake --build build --parallel --target ninfer_sampling_test
ctest --test-dir build -R ninfer_sampling_test --output-on-failure
```

Enable uniform floating-point error records when establishing or reviewing an Op criterion:

```bash
NINFER_OP_REPORT_STATS=1 \
  ctest --test-dir build -V -R '^ninfer_(rmsnorm|softmax_attention)_test$'
```

Every participating comparison emits one `OP_ERROR_STATS` record containing the stable case label,
actual error, active limit, and error-to-limit ratio. The switch changes reporting only; the same
statistics still drive the normal verdict. Passing tests remain quiet without it.

`ninfer_softmax_attention_test --causal-only` runs both D256 geometries and all five KV types;
`--kv-dtype bf16|int8|fp8|nvfp4|k8v4` selects the same complete causal suite for one type.
The suite covers prefill, decode/spec widths, batched prefixes, cache effects, and Graph replay and
updates with changing live lengths. Numerical cases include small, unit-RMS and RMS1.8 Q/K inputs.
The FP64 oracle retains internal Q quantization error; the INT8/FP8 compute budgets account for
that accepted approximation. The default invocation also runs plain and packed attention.

Linear tests are independently runnable by weight and activation-compute profile:

```bash
cmake --build build --parallel --target \
  ninfer_linear_q4_a16_test ninfer_linear_q5_a16_test \
  ninfer_linear_q6_a16_test ninfer_linear_q8_a16_test
ctest --test-dir build -R '^ninfer_linear_(q4|q5|q6|q8)_a16_test$' --output-on-failure
```

All Linear files use `ops/linear/linear_test_common.{h,cpp}` and the same
`ops/quantized_weight.h` fixture as the fused projection tests. The fixture produces the complete
packed GPU payload and exact-decodes the logical float rows used by the one
`cpu_linear_gemm_fp64()` reference. The reference performs naive double accumulation and never
reproduces a production route's activation quantization, staging, reduction tree, or BF16 output
rounding. Each activation compute path selects one centrally defined comparison tolerance for its
whole suite; private kernel, schedule, launcher, and T selection do not change it. Individual test
files call public `linear()` and contain no private selector, launcher, schedule, or kernel
assertions.

The Linear, LinearAdd and LinearSwiGLU common `.cpp` implementations each compile once into a
test support library. Both those libraries and the Op test executables receive the oracle's
`-fno-fast-math` and `-ffp-contract=off` options on GNU/Clang C++ compilers.

Run the native Python suites with the project Python environment:

```bash
python3 -m pytest \
  tests/artifact tests/convert \
  tests/test_serve_corpus.py
```

Agentic measurement and long-context probe tests (no GPU):

```bash
python3.11 -m unittest discover -s tests -p test_agentic_measurement.py
python3.11 -m unittest discover -s tests -p test_long_context.py
```

The Python suites exercise conversion and encoded output, without running model inference.
The maintained environment uses Python 3.11 with the dependencies for those suites. C++ binding and Engine tests
cover consumption of their resulting representation.

The real loading test accepts an explicit artifact path and optional component selection:

```bash
./build/tests/ninfer_qwen3_5_loading_real_test \
  --artifact out/qwen3_6_27b.ninfer --vision --speculative mtp --proposal optimized
```

Add `--host-only` to check semantic binding without uploading weights. This does not construct a
Program or establish native Op support.

The C++ prefix/MTP integration test is separately opt-in because it loads the full artifact and
runs the real engine. It and the other one-device 27B tests below need an artifact whose weights
fit one board (on 16 GB boards the 3-bit GSQ `q3_g128_fp16` artifact; the NVFP4 artifacts need
`--tp 2`); with a larger one they skip, as described under the tensor-parallel tests:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_prefix_real_test --output-on-failure
```

The causal-scoring integration test uses the same artifact variable and checks a full 1,024-column
score tile, overlapping target suffixes, and repeated-window State/KV isolation:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_score_real_test --output-on-failure
```

`ninfer_qwen3_5_score_tp2_real_test` scores the same artifact on two devices (INT8 KV), through
the half-chunk prefill overlap of the Text forward.

The two-device (tp 2) tests need two CUDA devices, use devices 0 and 1, and return 77 (skipped)
with fewer. Without an artifact:

- The nine op suites `ninfer_{allreduce,linear_split,output_head_split,attention_headlocal,
  attn_input_proj_split,gdn_projections_split,gdn_headsplit,linear_swiglu_split,linear_add_split}_test`
  qualify each column- or row-parallel form at the shard shapes, and the all-reduce and row
  gather against FP64 and exact oracles.
- `ninfer_artifact_sharded_materialization_tp2_test` uploads Replicated, Rows, Columns,
  PrimaryOnly and SingleDevice parents to both devices and compares every device's bytes with
  host-applied slices; the plan checks (`ninfer_artifact_slices_test`, `ninfer_qwen3_5_shard_map_test`, the
  one-device `ninfer_artifact_sharded_materialization_test`) run without a second device.
- `ninfer_qwen3_5_text_context_tp2_test` compares a synthetic two-layer model's tp 2 prefill and
  decode logits with tp 1 and checks the rank-0 logits gather byte for byte.

```bash
ctest --test-dir build -R '_(split|headlocal|headsplit|allreduce)_test|sharded_materialization|text_context_tp2_test' \
  --output-on-failure
```

With the artifact, split across the two devices: `ninfer_qwen3_5_sharded_load_real_test` (and its
`sharded_load_mtp_real` variant with the MTP head) checks the per-device placement and bytes of the
loaded model; `ninfer_qwen3_5_text_context_tp2_real_test` prefills and decodes one prompt through
the tp 2 `TextContext`; the Engine test serves single, concurrent, prefix-reuse and one-shot flood
requests. The MTP test compares MTP (K=3) answers with the same tp 2 model without speculation,
checks the draft acceptance and runs two-lane MTP rounds; its prefix-reuse legs resume a retained
~9k-token conversation through the MTP bridge on both ranks, with 4 lanes and 1 lane, and must
give the answer of the same prompt prefilled cold and of its exact (zero-suffix) repeat. Its
`optimized_real` variant selects the optimized proposal head. The DFlash2 test (K=4, optimized
proposal head) requires answers identical to the tp 2 model without speculation on three short
prompts, a nonzero acceptance, two-lane DFlash2 rounds and prefix reuse across two turns. The
Vision test (`ninfer_qwen3_5_engine_vision_tp2_real_test`, artifact with the Vision tower) rejects
a `vision_device` outside `devices`, requires a text answer identical with and without Vision,
names the color of synthetic red and blue images with the tower on device 0, and requires the same
token ids with the tower on device 1, where the embeddings are copied the other way; its
`vision_tp2_mtp_real` and `vision_tp2_dflash2_real` variants serve the images with MTP and DFlash2
and resume a second turn of the image conversation (with the QUASAR-QAT artifact, which has no
DFlash2 component, the two DFlash2 variants fail with `missing component dflash2` instead of being
skipped):

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R 'ninfer_qwen3_5_(engine_((mtp|dflash2|vision)_)?tp2|sharded_load|text_context_tp2_real)' \
  --output-on-failure
```

Without `NINFER_TEST_ARTIFACT`, CTest marks these real Engine tests as skipped. Every test that loads
a model artifact, and every other test that can hold more than 1 GiB of a board, holds the CTest
resource lock `ninfer_gpu_memory` (`tests/CMakeLists.txt`), so `ctest -j` runs them one at a time
beside the smaller GPU tests. The one-device tests (`prefix_real`, `score_real`,
`vision_workspace`, `dflash2_real`, `dflash_prefill_real`) plan the load first and skip, with the
planned weight bytes in their output, when those weights plus 1 GiB of runtime exceed the board's
total memory: the 27B NVFP4 artifacts (17-18 GB of weights on one device with MTP, DFlash2 or
Vision) need tp 2 on 16 GB boards, so run these tests with an artifact that fits one board.
`NINFER_PREFIX_REAL_SCENARIO` selects a focused prefix scenario such as `vision`, `pressure-resume`,
`concurrent` or `forced-token-kv-row`; the default is `all`. These integration checks use behavior
and state accounting rather than another numerical path's generated tokens as a golden.

The `attention` scenario checks the selected KV type, chunked prefill, concurrent Graph decode
across a resource tier, prefix continuation, and workspace bounds:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_27b.ninfer \
NINFER_PREFIX_REAL_SCENARIO=attention NINFER_TEST_KV_DTYPE=fp8 \
NINFER_TEST_SPECULATIVE=mtp NINFER_TEST_BATCH=2 \
  ./build/tests/ninfer_qwen3_5_prefix_real_test
```

KV choices are `bf16`, `int8`, `fp8`, `nvfp4`, and `k8v4`; backend choices are `none`, `mtp`
and `dflash2`, requiring an artifact with the selected component. Batch defaults to 2;
`NINFER_TEST_DRAFT_TOKENS` overrides the default MTP3 or DFlash2 K=7 block. The DFlash2-specific
integration executable also accepts all five KV names as its fifth positional argument and rejects
unknown names.

The capability-evaluation coordinator has its own environment and unittest entry point:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'
```

Run the serving contract manually after starting a resident server in another terminal:

```bash
./build/apps/ninfer-serve out/qwen3_6_27b.ninfer \
  --host 127.0.0.1 --port 18080
```

```bash
python3 -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

This smoke check is intentionally not a CTest: it needs the real artifact, a supported GPU, and a
server process that remains alive while the client exercises OpenAI Responses/Chat, Anthropic,
state, streaming, and multimodal requests.

The thinking-preservation fixture starts and stops its own server, submits a fixed two-step tool
history, compares stripped and preserved closed-turn prompt lengths, and verifies compatible
prefix reuse, speculative execution, frontier bounds and Responses inheritance:

```bash
python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_6_27b.ninfer --backend mtp

python3 tools/smoke/serve_thinking_preservation.py \
  --artifact out/qwen3_8_27b.ninfer --backend dflash2
```

The shared messages are in
[`fixtures/serve/qwen3_6_thinking_preservation.json`](fixtures/serve/qwen3_6_thinking_preservation.json).

## What belongs here

A permanent test should protect one current risk, such as:

- exact artifact bytes, geometry, object binding, or conversion transform;
- a numerical operator contract with an independent oracle;
- model Frontend or Program frontier, prefix, MTP, or multimodal behavior;
- generated-token commit/stop/cancel consistency;
- public benchmark or OpenAI/Anthropic observable behavior;
- a reproduced supported bug.

Performance-only assertions belong in benchmarks and profiler review. Source scans,
implementation-shape assertions, trivial getters/configuration, retired command surfaces, and
broad additions without a concrete regression risk do not belong in the permanent suite.

## DFlash2 Engine integration

The DFlash2 prefill regression checks the drafter's local KV contents after a StateImage fork and
a conflicting decode binding, including shortened chunks and oversized local appends. It uses native
Program storage and the production prefill route:

```bash
cmake --build build -j --target ninfer_qwen3_5_dflash_prefill_real_test
NINFER_TEST_ARTIFACT=out/qwen3_8_27b_nvfp4.ninfer \
  build/tests/ninfer_qwen3_5_dflash_prefill_real_test
```

The Engine test uses an artifact containing DFlash2 and checks output budgets, speculative activity,
forced thinking-control append, penalty-enabled sampling, compact batches with unequal budgets,
same-route same-seed replay, retained/fresh prefix behavior and absence of a full backend KV pool.
The page-boundary fixture (`speculative_page_boundary.h`) starts decode at token 63, verifies across
the page boundary, stops after one target column at token 64, and checks the exact retained frontier
and subsequent generation with and without reuse.
The KV Store test checks exact mapping and reservation accounting for the same transition.
K>=7 also exercises a stop inside a licensed block; K=15 additionally checks oversized prefill,
local ring wrap, and the logical context-capacity tail. Optional Vision runs image/video capture
and prefix restore. Zero extra Device StateImage slots exercise Host snapshot/restore.

```bash
cmake --build build -j --target ninfer_qwen3_5_dflash2_real_test
NINFER_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/tests/ninfer_qwen3_5_dflash2_real_test 15 1 1 8
NINFER_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/tests/ninfer_qwen3_5_dflash2_real_test 7 1 0 2 bf16 1 0
NINFER_TEST_ARTIFACT=out/qwen3_8_27b_nvfp4.ninfer \
  build/tests/ninfer_qwen3_5_dflash2_real_test 2 0 0 2 int8
```

Arguments are K, Graph enabled, optimized head enabled, maximum B, target KV (`bf16` or `int8`),
Vision enabled, and extra Device StateImage slots. Defaults are `15 1 1 8 bf16 0 3`. The individual
Op suites remain the numerical/state-transition oracle;
the fixed Engine fixture does not define bit parity across arbitrary floating-point routes.

### Grouped MoE metadata

`./build/tests/ninfer_moe_grouped_scan_test` compares exact offsets and job lists
with a CPU prefix/enumeration oracle, including empty, skewed and tile-boundary counts.
`--bench` emits 21 CUDA-event samples of 32 captured production launches per case
(after warmup); these measure metadata only. Run the same fixture on baseline and candidate.
The existing `ninfer_moe_grouped_test` remains the full mathematical qualification.

`ninfer_moe_grouped_test --bench` measures the complete public grouped Op on the same
synthetic pool used for qualification: 64 resident experts in randomized slots, dispersed
and skewed routes, token counts 1–16,384 including production widths 256, 4,096 and 16,384
and chunk boundaries. Counts below 256 are diagnostic: production uses the slots route there.
It emits 21 event samples of a single captured call after warmup, excluding fixture upload
and CPU work.
Run the normal oracle test separately; benchmark mode checks guards but is not an oracle.
