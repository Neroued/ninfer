# Tests

The test suite protects numerical, artifact, runtime-lifetime, end-to-end, and external schema
contracts for the fixed Qwen3.6-27B engine. Repository testing policy and the allowed test categories
are defined in [`../AGENTS.md`](../AGENTS.md); kernel verification guidance is in
[`../docs/kernel-development.md`](../docs/kernel-development.md).

## Organization

- `kernels/` — numerical operator tests at real model shapes;
- `fixtures/` — canonical q5090 and text fixtures;
- `test_q5090_*`, `test_weight_store*` — binary format, parser, loader, and real-artifact contracts;
- `test_model_*`, `test_engine_*` — model binding, memory formulas, Text/MTP/Vision observable flows,
  context fallback, and prefix reuse;
- `test_qwen_text_*`, `test_processor.cpp` — tokenizer, template, CLI, runner, and multimodal
  preprocessing behavior;
- `test_openai_schema.cpp`, `test_anthropic_schema.cpp`, `test_tool_call_parser.cpp` — external HTTP
  schema and tool-call contracts;
- `test_qus_bench_support.cpp` — machine-readable benchmark report behavior;
- `qus_block_dump` and `qus_layer_dump` — diagnostic parity executables, not unit-test oracles.

## Run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Build or run a narrower target when validating a localized change, for example:

```bash
cmake --build build -j --target qus_sampling_test
ctest --test-dir build -R qus_sampling_test --output-on-failure
```

Some real-file and end-to-end tests require the repository's generated q5090 fixture or local
artifact prerequisites. Their executable output states the missing prerequisite when they cannot
run.

## What does not belong here

- performance-only assertions belong in benchmarks/profiler review, not unit tests;
- source string scans and implementation-shape assertions are forbidden;
- retired flags, aliases, formats, and command surfaces are not compatibility-tested;
- broad test additions without a concrete observable regression risk are not accepted.
