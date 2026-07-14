# Tests

The retained tests protect current `.ninfer`, numerical operator, target, runtime-transaction,
benchmark-report, and external protocol behavior. Repository verification principles are defined in
[`../AGENTS.md`](../AGENTS.md); CUDA guidance is in
[`../docs/kernel-development.md`](../docs/kernel-development.md).

## Organization

- `artifact/` — Python container, registered layout, quantization, resource, converter-inventory,
  and source-verifier behavior;
- `kernels/` — independent numerical checks at real supported shapes, plus the shared row-split
  packing helper;
- `targets/qwen3_6_27b/` — registered artifact recipe/bindings/reference behavior, target Frontend,
  multimodal/MTP behavior, and the opt-in real-Engine prefix test;
- `test_ninfer_artifact_reader.cpp` — C++ framing/directory/geometry and Python-written fixture
  agreement;
- `test_output_resolution.cpp` and `test_runtime_pending_round.cpp` — generated-token transaction
  and pending-handle semantics;
- `test_openai_schema.cpp`, `test_anthropic_schema.cpp`, and `test_tool_call_parser.cpp` — current
  protocol translation and tool-call behavior;
- `test_ninfer_bench_support.cpp` — product benchmark CLI, timing boundary, and schema-v8 reports;
- device/tensor/arena tests — reusable lower-component behavior; KV/GDN tests exercise the exact
  target-owned state implementations.

Tests are grouped by observable risk, not by mirroring every source file or class.

## Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run a focused target for a localized change:

```bash
cmake --build build -j --target ninfer_sampling_test
ctest --test-dir build -R ninfer_sampling_test --output-on-failure
```

Run the native Python suites with the project Python environment, for example:

```bash
python -m pytest tests/artifact tests/targets/qwen3_6_27b
```

The real prefix/MTP integration test is opt-in because it loads the full artifact:

```bash
NINFER_QWEN3_6_27B_WEIGHTS=$PWD/out/qwen3_6_27b_rtx5090.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_27b_prefix_real_test --output-on-failure
```

Without that variable the test reports a skip and exits successfully.

## What belongs here

A permanent test should protect one current risk, such as:

- exact registered artifact bytes, geometry, object binding, or conversion transform;
- a numerical operator contract with an independent oracle;
- target Frontend, Program frontier, prefix, MTP, or multimodal behavior;
- generated-token commit/stop/cancel consistency;
- public benchmark or OpenAI/Anthropic observable behavior;
- a reproduced supported bug.

Performance-only assertions belong in benchmarks and profiler review. Source scans,
implementation-shape assertions, trivial getters/configuration, retired command surfaces, and
broad additions without a concrete regression risk do not belong in the permanent suite.
