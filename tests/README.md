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
- `models/common/` — shared Qwen frontend/tools, visual scatter, logical context stores and MoE;
- `models/qwen3_5/` — native config/binding, state layout, workspace, MTP alignment and
  opt-in real Engine integration;
- `ops/` — semantic Op qualification with independent mathematical or state-transition oracles;
  Linear and fused Linear suites are separated by their supported weight/activation paths;
- `runtime/` — scheduler fairness, cache resource policy, capacity and context-cost calculations;
- `cpu/` — topology/ISA, bounded task admission, bound collectives, publication and shutdown;
- `io/` — bounded direct reads, partial submission, completion, wakeup and borrowed-buffer lifetime;
- `models/qwen4_exp/` — native bank binding, state/page-group layouts and transfers, plus PLE
  disk binding, exact hash/NF4 reconstruction, incremental windows,
  cancellation and Host/CUDA buffer lifetimes;
- `bench/ttft/` — HTTP measurements, campaign orchestration and workload construction;
- root C++ tests — core storage, public API, serving protocols, logging, benchmark reports and
  causal-scoring evaluation;
- `test_serve_corpus.py` — agreement between the serving request-log schema and its measurement
  consumer.

Tests are grouped by observable risk, not by mirroring every source file or class.
`CMakeLists.txt` includes explicit registrations from `cmake/`, `artifact/`, the model directories
and `ops/`. Registration helpers live in `cmake/NinferTests.cmake`; included manifests keep
executables and CTest working directories under `<build-dir>/tests/`.
`ops/op_tester.h` and `ops/op_check.h` own reusable device/guard and comparison mechanics.
`ops/linear_common.h` shares extent checks, explicit sampling recipes, payload preservation and
0/1/77 entry handling; each projection family keeps its own fixtures and complete oracle.
Concrete numerical criteria remain named by the semantic Op suite; there are no cross-Op tolerance
presets.

Family-specific case declarations, independent oracles and numerical criteria live alongside the
Op suites. `ops/quantized_weight.h` owns packed-weight fixtures and independent logical decoding;
`ops/direct_bf16_weight.h` owns direct BF16 fixtures. GDN record/fold tests preserve their specified
snapshot reference separately from projection and recurrence mathematical qualification.

For segmented projection, run the original 2560-input suite and the migrated two/multi-output suites:
`ctest --test-dir build/rtx5090-dev -R '^ninfer_linear_split(_two_output|_multi_output)?_test$' --output-on-failure`.
`ninfer_qsa_sparse_gqa_test --kv-cache bf16|fp8|int8|nvfp4|k8v4` selects one cache profile;
the default checks all five. QSA tail, prepare, indexer and sparse-GQA partial-initialization cases
also check their permitted read footprints under Compute Sanitizer initcheck.

## Build and run

Select a Python environment with the dependencies for the tests first. The maintained environment
uses Python 3.11; CMake finds Python 3 without restricting its minor version.
`Python3_EXECUTABLE` selects the interpreter used by interop and frontend tests explicitly.

```bash
cmake --preset rtx5090-dev \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build build/rtx5090-dev -j
ctest --test-dir build/rtx5090-dev --output-on-failure
```

`rtx5090-dev` enables products, tests and benchmarks together. `rtx5090-ops` and
`dgx-spark-ops` instead register Core, Artifact and Op tests. Use the selected build directory
in the commands below; `ctest --preset <preset-name>` runs its registered suite. See
[Build system](../docs/maintainer/build-system.md) for local interpreter presets.

The chat-template reference test uses Python Jinja2.

PLE reading foundations run with
`ctest --test-dir build/rtx5090-dev -R 'ninfer_(nf4|read_queue|artifact_backing|ple_(disk|upload))' --output-on-failure`.
The NF4 oracle uses exact Python fractions. The PLE fixture includes a row split across three files.
`ninfer_ple_upload_test` checks exact Host-to-Device E and slot retirement using a direct Host
producer, without a table or disk service. Disk preparation tests separately cover their composition.
These tests use small synthetic files to check addressing, decoded values and queue lifecycle.
Real-weight I/O throughput measurements belong to separate, explicitly selected benchmarks.
The Host preparation test runs without visible CUDA devices. The GPU test delays upload/consumption,
injects event-record failures and checks graph consumption against independently reconstructed E.

CPU topology/task-pool tests link the C++ `ninfer_cpu` target. The x86 feature and collective
executor tests link `ninfer_cpu_executor` and are registered only for RTX 5090. None needs a GPU:
`ctest --test-dir build/rtx5090-dev -R '^ninfer_cpu_.*_test$' --output-on-failure`.
CPU/GPU MoE composition has its own `ninfer_moe_mixed_test`; pure GPU MoE remains in
`ninfer_moe_test`. Shared fixtures and mathematical oracles remain common across targets.
`ninfer_cpu_tensor_test` covers executor-based Tensor/arena computation;
`ninfer_memory_test` covers portable CPU storage and views. Frontend and HTTP transport tests
cover the task-pool consumers, including restricted media affinity and HTTP queue saturation.

Run a focused target for a localized change:

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_sampling_test
ctest --test-dir build/rtx5090-dev -R ninfer_sampling_test --output-on-failure
```

Enable uniform floating-point error records when establishing or reviewing an Op criterion:

```bash
NINFER_OP_REPORT_STATS=1 \
  ctest --test-dir build/rtx5090-dev -V -R '^ninfer_(rmsnorm|softmax_attention)_test$'
```

Every participating comparison emits one `OP_ERROR_STATS` record containing the stable case label,
actual error, active limit, and error-to-limit ratio. The switch changes reporting only; the same
statistics still drive the normal verdict. Passing tests remain quiet without it.

`ninfer_softmax_attention_test --causal-only` runs both D256 geometries and all five KV types;
`--kv-dtype bf16|int8|fp8|nvfp4|k8v4` selects the same complete causal suite for one type.
The default invocation also runs packed and context attention. Large causal oracle calculations
use at most eight CPU workers while preserving each output's FP64 accumulation order; small cases
remain serial. The runner reports elapsed time per KV type.

Run the RMSNorm and GDN control families together:

```bash
cmake --build build/rtx5090-dev -j --target \
  ninfer_rmsnorm_test ninfer_gated_rmsnorm_test \
  ninfer_gdn_control_test ninfer_gdn_control_proj_test ninfer_gdn_norm_control_proj_test
ctest --test-dir build/rtx5090-dev --output-on-failure \
  -R '^ninfer_(rmsnorm|gated_rmsnorm|gdn_control|gdn_control_proj|gdn_norm_control_proj)_test$'
```

Numerical criteria live in [test_gated.cpp](ops/rmsnorm/test_gated.cpp) and
[GDN criteria.h](ops/gdn_control/criteria.h). `NINFER_OP_REPORT_STATS=1` with verbose CTest output
reports the same suite-owned criteria used by the normal verdict.

Run the HC residual family:

```bash
cmake --build build/rtx5090-dev -j --target \
  ninfer_hc_expand_norm_test ninfer_hc_grouped_rmsnorm_test \
  ninfer_hc_combine_test ninfer_hc_combine_norm_test ninfer_hc_mix_test
ctest --test-dir build/rtx5090-dev --output-on-failure \
  -R '^ninfer_hc_(expand_norm|grouped_rmsnorm|combine|combine_norm|mix)_test$'
```

HC suites qualify the public [gated residual forms](../include/ninfer/ops/gated_residual.h),
including Mix, using [family-owned criteria](ops/gated_residual/criteria.h). Exact copying and
mathematical outputs retain their respective references and the public BF16 residual boundary.

PLE GPU Ops run with `ctest --test-dir build/rtx5090-dev -R '^ninfer_ple_(gate|conv|state_extract)_test$'`.
Gate uses an independent FP64 oracle for U and norm(actual BF16 U), including strided projections,
record windows, masks, Graph replay and the Conv/StateExtract handoff. These tests perform no file I/O.

Linear tests are independently runnable by weight and activation-compute profile:

```bash
cmake --build build/rtx5090-dev --parallel --target \
  ninfer_linear_q4_a16_test ninfer_linear_q5_a16_test \
  ninfer_linear_q6_a16_test ninfer_linear_q8_a16_test
ctest --test-dir build/rtx5090-dev -R '^ninfer_linear_(q4|q5|q6|q8)_a16_test$' --output-on-failure
```

For a change confined to one Q8 geometry, use the same public conformance cases with
`./build/rtx5090-dev/tests/ninfer_linear_q8_a16_test --shape N K`. The default CTest invocation still covers
all registered Q8 geometries.

For a change confined to one BF16 geometry, use
`./build/rtx5090-dev/tests/ninfer_linear_bf16_a16_test --shape N K`. The filter covers all registered BF16
geometries, including the original projections and selector matrix. The default runs all cases.

Linear (including both BF16 fixture recipes), LinearAdd (including FP8/NVFP4) and LinearSwiGLU
common implementations each compile once into a test support library. Their libraries and CXX Op test executables receive `-fno-fast-math` and `-ffp-contract=off`
on GNU/Clang. The remaining organization, entry-point and oracle-build work is tracked in the
[Linear test reorganization requirements](../docs/maintainer/linear-test-reorganization-plan.md).

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
./build/rtx5090-dev/tests/ninfer_qwen3_5_loading_real_test \
  --artifact out/qwen3_6_27b.ninfer --vision --speculative mtp --proposal optimized
```

Add `--host-only` to check semantic binding without uploading weights. This does not construct a
Program or establish native Op support.

The C++ prefix/MTP integration test is separately opt-in because it loads the full artifact and
runs the real engine:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build/rtx5090-dev -R ninfer_qwen3_5_prefix_real_test --output-on-failure
```

The causal-scoring integration test uses the same artifact variable and checks a full 1,024-column
score tile, overlapping target suffixes, and repeated-window State/KV isolation:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build/rtx5090-dev -R ninfer_qwen3_5_score_real_test --output-on-failure
```

`ninfer_moe_test` qualifies the integer routing, complete MoE and routed SwiGLU profiles.
`ninfer_moe_block_scaled_test` qualifies NVFP4 A4 and block-FP8 A8/G128 experts with BF16
control/shared and small-token A16 routes. It uses independent FP64 formulas and encoded-weight
decoders, covering per-expert divisors, K128 scales, zero groups, affine/indexed banks, masks,
changed Graph contents, private route boundaries and workspace guards. Optional `nvfp4` or `fp8`
arguments select one format; CTest runs both.
`ninfer_sum_contributions_test` covers FP32 partial summation with optional BF16 residual.
Their independent FP64 oracles start from represented inputs and decoded logical weights. The MoE
suite covers the resident codec profiles, FP32/BF16 outputs, residual modes, expert masks, strided/indexed
banks, empty GPU assignments, Graph relocation and workspace guards. The complete MoE retains
its existing numerical samples and criteria.

`ninfer_moe_cpu_groupwise_int_test` qualifies Q4+Q5/Q6 CPU routed SwiGLU against decoded-input
FP64 mathematics,
using an FP32 forward-error budget for dot products, SiLU, multiplication and weighted reduction.
`ninfer_moe_cpu_groupwise_int_avx2_test` and `ninfer_moe_cpu_groupwise_int_fma_test` use the same
public entry and oracle with
restricted test-only dispatch. All three run with CUDA devices hidden. They cover mask/None
semantics, arbitrary token tails through T=72, affine and strided parent banks, workspace and alias
checks, signed code domains, scale extremes and tiny-BF16 cancellation around the BF16-dot guard.
`ninfer_moe_cpu_nvfp4_test` and its `_avx2`/`_fma` variants qualify native NVFP4 banks against
independent FP64 mathematics, including scale swizzles, per-expert divisors, masked metadata,
column and chunk tails, full-precision products and tiny inputs. These CPU tests do not load
model files or use GPU memory.
`ninfer_moe_test` also qualifies complementary CPU/GPU subsets plus GPU shared/residual composition.

`ninfer_moe_execution_test` qualifies the shared single-layer execution and the Program-owned
multi-layer fixed-cache and streaming-window schedule against the same independent complete-MoE oracle. It uses registered native Host banks and the real ExpertCache,
covering Write/AddResidual outputs, absent CPU-only replicas, explicit region replacement,
All/None/partial partitions, bank IDs independent of traversal order, variable frame reuse through
T=1024, delayed bridge progress, failed acquisition and asynchronous teardown. Eager and segmented
Graph handoffs use the same oracle; Graph cases also cover expert relocation and streaming-window
transitions through the real cache schedule. The compact Graph calibration storage also runs
All/None/partial contributions against the same oracle while reusing one bank window.

```bash
ctest --test-dir build/rtx5090-dev -R '^ninfer_moe(_block_scaled|_cpu_(groupwise_int|nvfp4)(_avx2|_fma)?)?_test$' --output-on-failure
ctest --test-dir build/rtx5090-dev -R '^ninfer_moe_execution_test$' --output-on-failure
```

Run the 35B-A3B MoE route independently:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_35b_a3b.ninfer \
  ctest --test-dir build/rtx5090-dev -R ninfer_qwen3_5_moe_real_test --output-on-failure
```

`ninfer_qwen3_5_moe_offload_real_test` uses the same artifact with a 16 GiB Device budget and
segmented CUDA Graph execution (`NINFER_TEST_EAGER=1` selects eager). It covers MTP, prefix reuse, Vision, per-layer expert priorities,
eight resident rows, and stable small-T cache reuse. The DFlash and
preemption real tests also accept `NINFER_TEST_EAGER=1`; combine it with
`NINFER_VRAM_BUDGET_MIB` to exercise eager offload, or set only the budget for Graph offload.
CausalScoring always uses eager execution.

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
  ./build/rtx5090-dev/tests/ninfer_qwen3_5_prefix_real_test
```

KV choices are `bf16`, `int8`, `fp8`, `nvfp4`, and `k8v4`; backend choices are `none`, `mtp`,
`dflash`, and `dflash2`, requiring an artifact with the selected component. Batch defaults to 2;
`NINFER_TEST_DRAFT_TOKENS` overrides the default MTP3 or DFlash7 block. The DFlash2-specific
integration executable also accepts all five KV names as its fifth positional argument and rejects
unknown names.

Continuation and pressure recovery have dedicated entries:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build/rtx5090-dev -R ninfer_qwen3_5_agent_continuation_real_test --output-on-failure

NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
NINFER_TEST_BACKEND=dflash2 \
  ctest --test-dir build/rtx5090-dev -R ninfer_qwen3_5_preemption_real_test --output-on-failure

NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ./build/rtx5090-dev/tests/ninfer_qwen3_5_native_transactions_test dflash2
```

The agent entry checks multi-turn continuation and branching. The preemption entry exercises
Snapshot/Replay recovery and cancellation while paused or replaying; `NINFER_PREEMPTION_REAL_SCENARIO`
selects `all`, `snapshot`, `replay`, `cancel-paused` or `cancel-replay`.
`NINFER_TEST_PROPOSAL_HEAD=full` selects the full head for Qwen4Exp MTP; the default is `optimized`.
On Spark, use the replay/cancellation cases with Device context storage. Native transaction tests cover
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
`python3 tests/models/common/qwen/test_tool_schema.py` checks the native Qwen grammar and decoder against
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
./build/rtx5090-dev/apps/ninfer-serve out/qwen3_6_27b.ninfer \
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

Structural refactoring preserves existing cases, seeds, input distributions, oracles and numerical
criteria. Treat changes to test data or acceptance as explicit verification changes, identifying
the regression risk they protect and their execution cost.

Test each Op's workspace query against its own execution and interval contract. Shared allocation
helpers do not promise equal capacities across different Ops.

Stateful integration fixtures establish the prerequisites of the path they test. Isolate unrelated
cache history when qualifying restoration; share history deliberately when testing retention or
resource pressure. Enabling prefix reuse does not guarantee a cache hit after arbitrary preceding
work: availability and selection follow the [context-cache policy](../docs/maintainer/resource-scheduling-and-context-cache.md).

Performance-only assertions belong in benchmarks and profiler review. Source scans,
implementation-shape assertions, trivial getters/configuration, retired command surfaces, and
broad additions without a concrete regression risk do not belong in the permanent suite.

`ninfer_context_store_test` checks StateImage transitions and replicas. `ninfer_kv_history_test`
checks prefix retention, conflicting-writer COW, reservations, cancellation and Host extent lifetime.
It compares page bytes against an independent CPU layout oracle and restores a complete mixed
StateImage/KV snapshot at frontier 7, where `tokens_per_row=4` leaves one valid index row and a
three-entry raw tail. Native transaction tests exercise the corresponding model-owned source,
capture, retirement, recovery and consecutive decode/terminal paths.

## DFlash2 Engine integration

The DFlash prefill regression checks actual KV contents after a StateImage fork and a conflicting
decode binding, including shortened chunks and oversized local/full KV appends. It uses native
Program storage and the production prefill route; select the draft component stored in the artifact:

```bash
cmake --build build/rtx5090-dev -j --target ninfer_qwen3_5_dflash_prefill_real_test
NINFER_TEST_ARTIFACT=out/qwen3_8_27b_nvfp4.ninfer \
  build/rtx5090-dev/tests/ninfer_qwen3_5_dflash_prefill_real_test dflash2
NINFER_TEST_ARTIFACT=out/qwen3_6_35b_a3b.ninfer \
  build/rtx5090-dev/tests/ninfer_qwen3_5_dflash_prefill_real_test dflash
```

The Engine test requires a DFlash2 companion artifact and covers speculative transactions,
page-boundary commit/continuation and retained/fresh prefix behavior. K>=7 exercises a stop inside
a licensed block; K=15 adds oversized prefill, local-ring wrap and the context-capacity tail.
Optional Vision enables image/video and prefix-restore cases. Zero extra Device StateImage slots
exercise Host snapshot/restore.

```bash
cmake --build build/rtx5090-dev -j --target ninfer_qwen3_5_dflash2_real_test
NINFER_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/rtx5090-dev/tests/ninfer_qwen3_5_dflash2_real_test 15 1 1 8
NINFER_TEST_ARTIFACT=out/qwen3_8_27b.ninfer \
  build/rtx5090-dev/tests/ninfer_qwen3_5_dflash2_real_test 7 1 0 2 bf16 1 0
NINFER_TEST_ARTIFACT=out/qwen3_8_27b_nvfp4.ninfer \
  build/rtx5090-dev/tests/ninfer_qwen3_5_dflash2_real_test 2 0 0 2 int8
```

Arguments are K, Graph enabled, optimized head enabled, maximum B, target KV (`bf16`, `int8`,
`fp8`, `nvfp4`, or `k8v4`), Vision enabled, and extra Device StateImage slots. Defaults are
`15 1 1 8 bf16 0 3`. Run GPU integration tests serially. The individual Op suites remain the
numerical/state-transition oracle; the fixed Engine fixture does not define bit parity across
arbitrary floating-point routes.

## Expert sources and replicas

`ninfer_weight_bank_test` checks local matrix/plane addressing for INT, NVFP4 and block-FP8.
`ninfer_expert_cache_test` checks encoded bytes, variable entry sizes, bounded upload batches, required-set
changes, immutable fixed regions, window reconstruction, interrupted restoration, partial uploads, delayed consumers, Graph replay, independent prefetch readiness and failed
submission cleanup. The byte oracle is the immutable source representation, not another cache.
`ninfer_artifact_cpu_materialization_test` runs with CUDA devices hidden and verifies aligned Host
backing and retained resource lifetimes. The materialization suite also checks dual-destination I/O,
registered/ordinary Host regions and registered backing moves. Registered transfer views and pitched
copies are also exercised by `ninfer_arena_test`.

The opt-in loader test accepts `--host-experts` to put Main experts in registered RAM while retaining MTP and
fixed tensors on GPU. This checks storage and binding, not mixed MoE execution:

```bash
build/rtx5090-dev/tests/ninfer_qwen3_5_loading_real_test --artifact out/qwen3_6_35b_a3b.ninfer \
  --vision --speculative mtp --host-experts
```

Qwen4Exp loading/resources can be checked independently of the public Engine route:

```bash
./build/rtx5090-dev/tests/ninfer_qwen4_exp_loading_real_test out/qwen3_8_flash_next.ninfer --host-experts
./build/rtx5090-dev/tests/ninfer_qwen4_exp_loading_real_test out/qwen3_8_flash_next.ninfer --host-experts --materialize
```

The first command plans component combinations, KV formats and concurrency without loading weight
payloads. `--materialize` additionally loads one full-component model, checks stored words and
parameter sharing, constructs its execution resources, and reads/uploads a PLE window. It needs
about 64 GiB of available Host memory plus system headroom with `--host-experts`. For resident
loading on Spark, use its build directory and omit `--host-experts`. These are loading/resource checks;
Text/Vision/MTP generation, eager or CUDA Graph decode, and text causal scoring use the public Engine,
with current support described in the [execution reference](../docs/maintainer/qwen4_exp-execution.md)
and commands in the [CLI guide](../docs/cli.md).
