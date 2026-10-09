# Tests

The retained tests protect current `.ninfer`, numerical operator, model, runtime-transaction,
benchmark-report, and external protocol behavior. Op qualification and CUDA implementation rules
are defined in [Op development](../docs/maintainer/op-development.md); product execution and
ownership contracts are defined in [Engine architecture](../docs/maintainer/engine-architecture.md).

## Organization

- `artifact/` — v3 framing, directory/binding records, codecs, sharding, selected-object
  materialization and Python-writer/C++-reader interoperability;
- `convert/` — source interpretation, Qwen logical mapping, recipe overrides/sharing, optional
  components, resources, proposals and numerical conversion methods;
- `models/qwen3_5/` — config/binding, frontend, state/context stores, workspace, MTP alignment and
  opt-in real Engine integration;
- `ops/` — semantic Op qualification with independent mathematical or state-transition oracles;
  Linear and fused Linear suites are separated by their supported weight/activation paths;
- `runtime/` — scheduler fairness, cache resource policy, capacity and context-cost calculations;
- `bench/ttft/` — HTTP measurements, campaign orchestration and workload construction;
- root C++ tests — core storage, public API, serving protocols, logging, benchmark reports and
  causal-scoring evaluation;
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

## Decision engine

The finite-schema compiler and probability oracle run without CUDA or model weights:

```bash
c++ -std=c++20 -O2 -Iinclude -Isrc -Ithird_party \
  src/decision/decision.cpp tests/test_decision.cpp -o /tmp/ninfer-decision-test
/tmp/ninfer-decision-test
```

The CMake target is `ninfer_decision_test`. It checks typed domains, validation, decimal and
integer boundaries, token-prefix merging, independent exact probability calculations,
multi-context cohorts, multi-round greedy traversal, cache opt-out and cancellation. These tests do not qualify GPU
execution or establish model accuracy/performance.

`ninfer_qwen3_5_decision_real_test` is an opt-in artifact execution comparison. Twelve rows cover
two contexts, partial-page KV forks, zero/1/15/16/17/33-token suffixes, multiple tiles, full batches,
cache hits/opt-out, mid-transaction cancellation and unchanged later ordinary/speculative output.
After releasing the generation Engine, it compares each row with the existing causal-prefill
scorer's full-vocabulary log probabilities, renormalized over the same candidates. The declared
maximum absolute conditional-probability error is 0.02 for differing BF16 execution shapes;
the test prints the worst error. This is an independent scheduling/execution reference, not a new
mathematical kernel oracle, model accuracy benchmark or speed measurement. It skips with status 77
without an artifact.
Run it on the supported Linux/CUDA build, with the live model server stopped to free the GPU:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel --target ninfer_decision_test ninfer_qwen3_5_decision_real_test
NINFER_TEST_ARTIFACT=/absolute/path/model.ninfer \
  ctest --test-dir build -R '^ninfer_(decision_test|qwen3_5_decision_real_test)$' --output-on-failure
```

Optional `NINFER_DECISION_TEST_SPEC=mtp|dflash|dflash2` exercises the selected supported artifact
component with three draft tokens (default `none`). `NINFER_DECISION_TEST_PROPOSAL_HEAD=1` also
selects the optimized proposal head. Those options must match components present in the artifact.
For HTTP/Docker use, see [decision serving](../docs/serving.md#parallel-decisions-jev).

### Codacus fixtures and Docker qualification

[`fixtures/decision/codacus.json`](fixtures/decision/codacus.json) contains all eight playground
presets from [thecodacus/decision-playground at 843f72e](https://github.com/thecodacus/decision-playground/blob/843f72e61f5ebaa1c2225c4902849bd42af44c05/src/playground/presets.ts):
arena, routing, eight-ticket routing, ticket triage, smart home, warehouse rover, robot arm,
and the 30-field RTS squad. Instructions, schema values/descriptions and context text are unchanged;
the playground's own separator/trim rule converts its context box to a `contexts` array.
The source revision and SHA-256 are recorded in the fixture. No upstream code executes at test time.

These are exact **demo inputs**, not a claimed copy of a Codacus golden-answer test suite.
`ninfer_decision_fixtures_test` exercises each in auto/tree/greedy mode using synthetic byte
tokenization and logits: 24 host contract cases, not model accuracy checks. The existing
`ninfer_decision_test` remains the independent finite-probability oracle. The Python unit test
checks that the HTTP smoke validator rejects malformed results; it does not contact a server.

From the repository root on the Linux/WSL Docker host:

```bash
docker build --target decision-tests -t ninfer:jev-tests .

# Compiler/math, protocol adapters, fixtures and smoke validation. No GPU or model needed.
docker run --rm ninfer:jev-tests ctest --test-dir /build \
  -R '^ninfer_(decision_test|decision_fixtures_test|decision_smoke_test|systemone_test)$' --output-on-failure

# Stop the live model server first. Set the exact artifact file, not a directory/glob.
ARTIFACT=/absolute/path/to/your-model.ninfer
docker run --rm --gpus all \
  --mount "type=bind,src=$ARTIFACT,dst=/models/test.ninfer,readonly" \
  -e NINFER_TEST_ARTIFACT=/models/test.ninfer ninfer:jev-tests
```

The second run must report all five tests passed, **not a skipped real-artifact test**.
Its numerical comparison must stay within the fixed 0.02 conditional-probability tolerance;
do not relax the threshold to manufacture a pass. The test image builds focused test targets
and includes Python only as a test/client interpreter. It is not the production serving image;
ordinary `docker build -t ninfer:jev .` still ends at the unchanged runtime stage.

The real-artifact check first loads generation with decisions disabled, then enabled, using
identical serving options. Minimum/actual reservations, the KV capacity increment, KV payload,
persistent/workspace capacities and graph allowance must match exactly, as must greedy chat
output. It also verifies that chat releases the temporary decision overlay before reuse.
Decision batch bounds are independent of the configured chat lane count. With speculative
decoding disabled it additionally compares KV payload with the causal reference engine.

Start the server using the serving instructions, then replay real requests from another terminal:

```bash
python3 tools/smoke/decision.py --list
python3 tools/smoke/decision.py --base-url http://127.0.0.1:8080 --all
python3 tools/smoke/decision.py --preset arena --check-arena
python3 tools/smoke/decision.py --preset routing-bulk --repeat 3
```

No Python packages are required. Alternatively, on the Linux/WSL Docker host use the built
test image as the client: `docker run --rm --network host ninfer:jev-tests python3
/src/tools/smoke/decision.py --all`. This client needs no GPU access and loads no model.
Use `NINFER_API_KEY` if the server requires authentication; `--model` is optional and takes the
served alias, not an artifact filename. `--mode tree` forces exhaustive finite-tree evaluation;
`--no-cache` exercises the uncached route. JSON responses go to stdout; summaries/errors to stderr.

`PASS contract` means typed domains, complete result shapes, finite probabilities, numeric
intervals and token accounting are valid. It does **not** certify semantic correctness.
`--check-arena` additionally checks five NInfer-authored expectations derived from the unchanged
arena rules: move right, turn right by 33 degrees, do not fire, stand. These are not upstream
golden labels. A wrong answer fails this optional quality check even when the engine contract
passes. Printed timings are observations, not comparisons with Codacus's hardware or Jev.

### System One adapter

`ninfer_systemone_test` qualifies TypeSafe-shaped parsing and formatting without CUDA,
a model, or a network connection:

```bash
c++ -std=c++20 -O2 -Iinclude -Isrc -Ithird_party \
  src/decision/decision.cpp src/serve/systemone.cpp tests/test_systemone.cpp \
  -o /tmp/ninfer-systemone-test
/tmp/ninfer-systemone-test
```

It checks published confidence examples against fixed numerical expectations, full Choice
distributions, Noul's probability-of-true semantics, unrounded Score means, structured/null
entries, limits and invalid distributions. A synthetic tokenizer/scorer exercises the real
compiler to verify question/ID isolation, shared batch submission, state/question prefix reuse,
multi-token divergence reuse, cache opt-out and unchanged ordinary decision response shape.
This is a host protocol oracle, not a claim of model accuracy or GPU performance.
The Docker test stage includes this target; live usage is documented in
[System One compatibility](../docs/serving.md#system-one-compatibility). The existing
`tools/smoke/decision.py` client still targets `/v1/decision`, not `/v1/systemone`.

### Research and application evaluation

The attached research's [Jev in the Wild paper](https://arxiv.org/abs/2609.30216) is an ecosystem
study, not a specification for reproducing Jev's proprietary model or an executable test suite.
For realistic workloads, TypeSafe publishes [workflow evaluations](https://evals.typesafe.ai/)
for customer service, invoice processing, security incidents and agent-trace observability.
Their reference labels are model consensus, not human-verified ground truth.

A downloadable evaluation source is [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions),
used by [Type-Safe Is Not Error-Free](https://arxiv.org/abs/2609.26758). To evaluate it here,
map each question to a field, preserve its criteria in the field description, use the state as
context, and score held-out decisions against the supplied labels. This is an API adaptation,
not a drop-in reproduction of Jev's distribution/calibration metrics. No dataset or additional
checkpoint is downloaded by the tests. Also measure sensitivity to option names/order; a valid
enum result alone does not prove that a model followed the option definitions.

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
that accepted approximation. The default invocation also runs packed and context attention.
Large causal reference calculations use at most eight CPU workers while preserving each output's
FP64 accumulation order; small cases remain serial. Cache fixtures populate the reachable KV
prefix while reserving the full execution-envelope page table. The runner reports elapsed time
per KV type.

Linear tests are independently runnable by weight and activation-compute profile:

```bash
cmake --build build --parallel --target \
  ninfer_linear_q4_a16_test ninfer_linear_q5_a16_test \
  ninfer_linear_q6_a16_test ninfer_linear_q8_a16_test
ctest --test-dir build -R '^ninfer_linear_(q4|q5|q6|q8)_a16_test$' --output-on-failure
```

For a change confined to one Q8 geometry, use the same public conformance cases with
`./build/tests/ninfer_linear_q8_a16_test --shape N K`. The default CTest invocation still covers
all registered Q8 geometries.

For a change confined to a newly supported BF16 geometry, use
`./build/tests/ninfer_linear_bf16_a16_test --shape N K`. These cases use the common full-K FP64
oracle with exact BF16 weights, complete output checks for small N, route boundaries, changed-input
Graph replay, both public overloads, workspace domains, and preservation/guard checks. The small
control projections also cover terminal-K pulses and paired cancellation.

All Linear files use `ops/linear/linear_test_common.{h,cpp}` and the same
`ops/quantized_weight.h` fixture as the fused projection tests. The fixture produces the complete
packed GPU payload and exact-decodes the logical float rows used by the one
`cpu_linear_gemm_fp64()` reference. The reference performs naive double accumulation and never
reproduces a production route's activation quantization, staging, reduction tree, or BF16 output
rounding. Each activation compute path selects one centrally defined comparison tolerance for its
whole suite; private kernel, schedule, launcher, and T selection do not change it. Individual test
files call public `linear()` and contain no private selector, launcher, schedule, or kernel
assertions.

The Q8 suite includes `[2560,6144]`, `[6144,2560]`, `[10240,2560]`, `[12288,2560]`, and
`[16384,2560]`: full-output FP64 comparisons at T=1/4/8,
sampled-output checks at larger extents including 512/1024 and 129/1025, production boundaries,
changed-input Graph replay, both public overloads, permissive policies, input/weight preservation,
output guards, and valid/invalid workspace intervals.

The Linear, LinearAdd and LinearSwiGLU common `.cpp` implementations each compile once into a
test support library. Both those libraries and the Op test executables receive the oracle's
`-fno-fast-math` and `-ffp-contract=off` options on GNU/Clang C++ compilers.

Run the native Python suites with the project Python environment:

```bash
python3 -m pytest \
  tests/artifact tests/convert tests/bench/ttft \
  tests/test_serve_corpus.py
```

The Python suites exercise conversion, encoded output and measurement tools without model inference.
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
runs the real engine:

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

Run the 35B-A3B MoE route independently:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_35b_a3b.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_moe_real_test --output-on-failure
```

Without `NINFER_TEST_ARTIFACT`, CTest marks these real Engine tests as skipped. Real-artifact targets
carry the `real` label and `RUN_SERIAL` so CTest runs them alone. Run directly invoked GPU
integration tests serially. `NINFER_PREFIX_REAL_SCENARIO` selects a focused prefix scenario such as
`vision`, `late-instructions` or `concurrent`; the default is `all`. These integration checks
use behavior and state accounting rather than another numerical path's generated tokens as a golden.

The `attention` scenario checks the selected KV type, chunked prefill, concurrent Graph decode
across a resource tier, prefix continuation, and workspace bounds:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_27b.ninfer \
NINFER_PREFIX_REAL_SCENARIO=attention NINFER_TEST_KV_DTYPE=fp8 \
NINFER_TEST_SPECULATIVE=mtp NINFER_TEST_BATCH=2 \
  ./build/tests/ninfer_qwen3_5_prefix_real_test
```

KV choices are `bf16`, `int8`, `fp8`, `nvfp4`, and `k8v4`; backend choices are `none`, `mtp`,
`dflash`, and `dflash2`, requiring an artifact with the selected component. Batch defaults to 2;
`NINFER_TEST_DRAFT_TOKENS` overrides the default MTP3 or DFlash7 block. The DFlash2-specific
integration executable also accepts all five KV names as its fifth positional argument and rejects
unknown names.

Continuation and pressure recovery have dedicated entries:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build -R ninfer_qwen3_5_agent_continuation_real_test --output-on-failure

NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
NINFER_TEST_BACKEND=dflash2 \
  ctest --test-dir build -R ninfer_qwen3_5_preemption_real_test --output-on-failure

NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ./build/tests/ninfer_qwen3_5_native_transactions_test dflash2
```

The agent entry checks multi-turn continuation and branching. The preemption entry exercises
Snapshot/Replay recovery and cancellation while paused or replaying; `NINFER_PREEMPTION_REAL_SCENARIO`
selects `all`, `snapshot`, `replay`, `cancel-paused` or `cancel-replay`. Native transaction tests cover
physical state/KV ownership, binding, capture, reclamation and abort; their positional backend is
`none`, `mtp`, `dflash` or `dflash2`. Each backend requires an artifact containing that component.
Public-HTTP latency and output gaps are measured separately by the
[TTFT campaign](../tools/bench/ttft/README.md).

`ninfer_qwen3_5_tools_real_test [none|mtp|dflash|dflash2] [graph|eager|basic|snapshot|replay|cancel] [concurrency]`
uses `NINFER_TEST_ARTIFACT` for strict tools, thinking, raw continuation, mixed batches, and
call/result prefix reuse, and required-tool → JSON continuation with committed constraint observations.
`basic` checks default constraints with open/complex schemas, continuation,
streaming and mixed strict/basic/free rows. Snapshot/Replay modes force resource pressure and
validate the completed argument value after recovery; `cancel` interrupts the paused request.
`NINFER_TEST_TOOL_REPORT` appends schema/output/timing JSONL.
`python3 tests/models/qwen3_5/test_tool_schema.py` checks the native Qwen grammar and decoder against
`jsonschema` (dependencies in `tests/text/requirements.txt`), including string pattern/Unicode-length
intersections. The JSON Schema oracle also covers finite-value filtering, reference/union siblings,
closed-object and positional-array intersections, draft-07 tuples, recursive conjunctions,
numeric endpoints and JSON publication rounding. HTTP parsing tests check schema-number precision
before protocol adapters serialize the schema.

`ninfer_qwen3_5_grammar_real_test [none|mtp|dflash|dflash2] [graph|eager] [concurrency] [vision]` uses
`NINFER_TEST_DRAFT_TOKENS` to override the default draft count of three and
`NINFER_TEST_ARTIFACT` to check GBNF/JSON/schema/choice/regex content, sampling, thinking, continuation, prefix reuse
and mixed batches. `ninfer_regex_choice_test` checks literal-set prefix masks and regex edge cases;
`python3 tests/text/test_regex_choice.py` compares regex membership with independent fullmatch semantics.
Set `NINFER_TEST_CONSTRAINT=grammar` or `json_schema` on the preemption test to
check matcher continuity through Snapshot/Replay and cancellation. `ninfer_grammar_test` and
`ninfer_json_schema_test` cover CPU language semantics. `ninfer_json_schema_oracle_test` compares
supported schemas with the independent Python `jsonschema` validator; install its dependency with
`python3 -m pip install -r tests/text/requirements.txt` in the selected test environment.
The sampling and speculative Op tests qualify masks against independent mathematical oracles.

The capability-evaluation coordinator has its own environment and unittest entry point:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m unittest discover \
  -s eval/tests -p 'test_*.py'
```

Run the serving contract manually after starting a resident server in another terminal:

```bash
./build/apps/ninfer-serve out/qwen3_6_27b.ninfer \
  --host 127.0.0.1 --port 18080 --vision
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
  --artifact out/qwen3_6_35b_a3b.ninfer --backend dflash
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

The DFlash prefill regression checks actual KV contents after a StateImage fork and a conflicting
decode binding, including shortened chunks and oversized local/full KV appends. It uses native
Program storage and the production prefill route; select the draft component stored in the artifact:

```bash
cmake --build build -j --target ninfer_qwen3_5_dflash_prefill_real_test
NINFER_TEST_ARTIFACT=out/qwen3_8_27b_nvfp4.ninfer \
  build/tests/ninfer_qwen3_5_dflash_prefill_real_test dflash2
NINFER_TEST_ARTIFACT=out/qwen3_6_35b_a3b.ninfer \
  build/tests/ninfer_qwen3_5_dflash_prefill_real_test dflash
```

The Engine test uses an artifact containing DFlash2 and checks output budgets, speculative activity,
forced thinking-control append, penalty-enabled sampling, compact batches with unequal budgets,
same-route same-seed replay, retained/fresh prefix behavior and absence of a full backend KV pool.
A shared DFlash/DFlash2 fixture starts decode at token 63, verifies across the page boundary, stops
after one target column at token 64, and checks the exact retained frontier and subsequent generation
with and without reuse.
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

Arguments are K, Graph enabled, optimized head enabled, maximum B, target KV (`bf16`, `int8`,
`fp8`, `nvfp4`, or `k8v4`), Vision enabled, and extra Device StateImage slots. Defaults are
`15 1 1 8 bf16 0 3`. Run GPU integration tests serially. The individual Op suites remain the
numerical/state-transition oracle; the fixed Engine fixture does not define bit parity across
arbitrary floating-point routes.
