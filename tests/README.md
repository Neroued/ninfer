# Tests

The retained tests protect current `.ninfer`, numerical operator, target, runtime-transaction,
benchmark-report, and external protocol behavior. Repository verification principles are defined in
[`../AGENTS.md`](../AGENTS.md); Op contract and CUDA implementation guidance is in
[`../docs/op-development.md`](../docs/op-development.md).

## Organization

- `artifact/` — Python container, registered layout, quantization, and resource behavior;
- `ops/` — independent numerical and state-transition checks at real supported shapes, plus the
  shared row-split packing helper;
- `targets/qwen3_6/` — shared tokenizer/template, multimodal preprocessing, MRoPE, prepared-prompt,
  stop/output decoding, hybrid topology, decoder/GDN and round-state layouts/views, shifted-MTP
  alignment, and Vision-control behavior;
- `targets/qwen3_6_27b/` — registered inventory, converter recipe, source verifier, artifact
  bindings, reference behavior, target Program/multimodal/MTP behavior, and the opt-in real-Engine
  prefix test;
- `targets/qwen3_6_35b_a3b/` — future-target converter contracts plus the artifact-native MoE
  oracle, typed binding, selected-expert row access, and 256K INT8 reference-memory calculation;
- `test_ninfer_artifact_reader.cpp` — C++ framing, directory, encoded-size, payload-span, and
  geometry behavior against a self-contained C++ fixture;
- `test_generation_controller.cpp` — accepted-prefix, cancellation, publication ordering, and
  request-abort behavior in the common generated-token loop;
- `test_openai_schema.cpp`, `test_anthropic_schema.cpp`, and `test_tool_call_parser.cpp` — current
  protocol translation and tool-call behavior;
- `test_ninfer_bench_support.cpp` — product benchmark CLI, timing boundary, and schema-v8 reports;
- `test_bench_matrix.py` — schema-v8 report consumption by the Python matrix summarizer;
- device/tensor/arena tests — reusable lower-component behavior; KV tests cover the core physical
  container, family runtime tests cover dimension-driven GDN storage/view mechanics, and Op tests
  cover mathematical state transitions at their own boundary.

Tests are grouped by observable risk, not by mirroring every source file or class.

## Build and run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DPython3_EXECUTABLE=/home/neroued/miniconda3/envs/py311/bin/python
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run a focused target for a localized change:

```bash
cmake --build build -j --target ninfer_sampling_test
ctest --test-dir build -R ninfer_sampling_test --output-on-failure
```

Run the native Python suites with the project Python environment:

```bash
/home/neroued/miniconda3/envs/py311/bin/python -m pytest \
  tests/artifact tests/targets/qwen3_6_27b tests/targets/qwen3_6_35b_a3b \
  tests/test_bench_matrix.py
```

The Python binding tests use `NINFER_QWEN3_6_27B_ARTIFACT` when set, otherwise they look for
`out/qwen3_6_27b_rtx5090.ninfer`. They report a pytest skip when neither path provides the real
artifact. The 35B-A3B reference binding test follows the same rule with
`NINFER_QWEN3_6_35B_A3B_ARTIFACT` and `out/qwen3_6_35b_a3b_rtx5090.ninfer`. The remaining Python
target tests still run without either artifact.

The C++ prefix/MTP integration test is separately opt-in because it loads the full artifact and
runs the real engine:

```bash
NINFER_QWEN3_6_27B_WEIGHTS=$PWD/out/qwen3_6_27b_rtx5090.ninfer \
  ctest --test-dir build -R ninfer_qwen3_6_27b_prefix_real_test --output-on-failure
```

Without that variable CTest marks the C++ integration test as skipped.

The capability-evaluation coordinator has its own environment and unittest entry point:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'
```

Run the serving contract manually after starting a resident server in another terminal:

```bash
./build/apps/ninfer-serve out/qwen3_6_27b_rtx5090.ninfer \
  --host 127.0.0.1 --port 18080 --model-id qwen3.6-27b
```

```bash
/home/neroued/miniconda3/envs/py311/bin/python -m tools.smoke.serve_contract \
  --base-url http://127.0.0.1:18080 --model qwen3.6-27b
```

This smoke check is intentionally not a CTest: it needs the real artifact, a supported GPU, and a
server process that remains alive while the client exercises OpenAI, Anthropic, streaming, and
multimodal requests.

## What belongs here

A permanent test should protect one current risk, such as:

- exact registered artifact bytes, geometry, object binding, or conversion transform;
- a numerical operator contract with an independent oracle;
- family Frontend or target Program frontier, prefix, MTP, or multimodal behavior;
- generated-token commit/stop/cancel consistency;
- public benchmark or OpenAI/Anthropic observable behavior;
- a reproduced supported bug.

Performance-only assertions belong in benchmarks and profiler review. Source scans,
implementation-shape assertions, trivial getters/configuration, retired command surfaces, and
broad additions without a concrete regression risk do not belong in the permanent suite.
