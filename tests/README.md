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
- `distributed/` — typed native work, bounded transport, and opt-in two-rank GPU execution;
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
Tests link the same Core, Ops and CPU shared libraries as products and benchmarks, stored in
`<build-dir>/lib/`; relative RPATH supplies their build-tree lookup. Test support and restricted
CPU ISA qualification libraries remain private static targets. See
[Build system](../docs/maintainer/build-system.md#shared-library-boundary) for linkage ownership.

CUDA and I/O failure cases interpose selected symbols only in their owning test executables.
`ninfer_test_interpose()` exports the replacements; `symbol_interpose.h` resolves the real shared
definitions with `RTLD_NEXT`. Keep failure state and assertions local to the relevant suite.

Op tests follow semantic families under `ops/`. Small related Ops share an executable;
large computations own dedicated runners and case catalogues. A case is a reproducible input
recipe or a complete state/Graph sequence. A source file, semantic suite, executable and CTest
group need not have the same boundary.

| Family | Organization |
|---|---|
| Elementwise, indexing, reduction, normalization, RoPE, embedding, sequence | One family executable; each semantic Op owns its cases and oracle. |
| Sampling | Sampling, candidate selection and acceptance; input preparation and exact indexing live in sequence/indexing. |
| Linear | Projection, Split, Add, SwiGLU, Pair and TopK; typed shape and policy catalogues. |
| Linear attention | KDA, causal convolution and GDN. GDN owns control, input-conv, recurrence and replay. |
| Softmax attention | Causal cache fixtures, mathematical oracle, runner and cases are separate. Packed/context, sliding-window and QSA keep their own semantics. |
| KV cache | Append codecs/paged storage, cyclic prefixes and context materialization. |
| Gated residual, dynamic grouped convolution, PLE | Related forms share an entry and substantial family support. PLE GPU tests perform no disk I/O. |
| MoE | GPU route/routed/complete computations, CPU arithmetic routes and mixed contributions. |

`ops/suite.*` owns metadata selection, result handling and a bounded CPU oracle executor.
`ops/op_tester.h` and `ops/op_check.h` own uploads, guards and comparison mechanics;
`ops/float_codec.h` owns shared test-side floating-point codecs. `ops/quantized_weight.h` and
`ops/direct_bf16_weight.h` own independent represented-weight fixtures. Linear-specific geometry,
sampling recipes and fixture reuse remain under `ops/linear/` and `ops/linear_common.h`.
Numerical criteria and complete mathematical/state references stay with their semantic suites.

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

Run all Op qualification groups or select a family:

```bash
ctest --test-dir build/rtx5090-dev -L ops --output-on-failure
ctest --test-dir build/rtx5090-dev -R '^ninfer_qsa_.*_test$' --output-on-failure
./build/rtx5090-dev/tests/ninfer_normalization_test --suite rmsnorm --list
./build/rtx5090-dev/tests/ninfer_qsa_sparse_gqa_test --format nvfp4 --execution graph
./build/rtx5090-dev/tests/ninfer_linear_projection_test --format q8 --shape 6144 5120
```

Every semantic Op executable under `ops/` accepts `--help`, `--list`, `--case ID`,
`--suite NAME`, `--format FORMAT`, `--policy a16|a8|a4` and `--execution eager|graph`.
Filters intersect declared metadata; an absent format/policy does not match that filter.
The first component of an ID names its suite. A case containing both eager and captured calls is
classified as Graph. `--case` takes an exact ID from `--list`; quote IDs containing spaces.
The six Linear executables additionally accept `--shape N K` and `--t T` (T=0 denotes metadata cases).
Filters select existing cases, without generating new shapes or inputs.

Listing and selection precede CUDA initialization, weight generation and fixture allocation.
Unknown options, duplicate IDs and empty selections fail. Exit 77 means the selected tests cannot
run on the available device/CPU ISA. An unavailable group is reported separately from failures.
CTest partitions Linear, dense attention and QSA sparse GQA by format; it does not
also run an aggregate entry. CPU MoE retains separate native, AVX2 and FMA qualification
executables with a shared resource lock.

Inputs, seeds and sample coordinates are independent of selection order. Grouped runners reuse
immutable fixtures, while each stateful case resets its own initial state. Graph cases include
all declared input/metadata updates. ReplaySSM record/fold compares the specified snapshot
or recurrent execution bit for bit; those computations have separate mathematical tests.
Linear and fused projection suites select their A4 criterion for `AllowA4` and A8 criterion for
`AllowA8`, including when an implementation chooses higher precision for a small input.

GDN recurrence uses one output criterion and one final-state criterion across running and
state-pool updates, normalization choices, targets, token extents and continuation partitions.
Its independent FP64 oracle starts from represented inputs and retains the complete history.
Replay record/fold uses public one-token state-pool updates to produce its exact reference
snapshots; their bitwise consistency remains separate from recurrence's mathematical accuracy.

Large independent output rows may use fixed samples of the complete mathematical formula.
QSA sparse GQA checks all live queries up to 64 per batch row; larger rows use fixed boundary,
interior and tail queries. GPU extents remain unchanged. QSA checks every output cell for a write
and finite value; invalid outputs, untouched state, preserved inputs and guards retain full checks.
Long recurrence histories are not shortened or sampled away. Attention cache generation/decoding
and independent oracle rows share at most eight CPU workers; nested work runs serially and each
dot product retains its accumulation order. Shared oracle support and CXX suite targets use
`-fno-fast-math -ffp-contract=off` on GNU/Clang.

Enable numerical error records when reviewing a criterion:

```bash
NINFER_OP_REPORT_STATS=1 \
  ./build/rtx5090-dev/tests/ninfer_gated_residual_test --suite combine-norm
```

Each participating comparison emits `OP_ERROR_STATS` with the case label, actual error, limit
and error-to-limit ratio. The switch changes reporting only. Passing tests otherwise print the
selected-case result; detailed failure diagnostics remain owned by each oracle.

QSA Indexer and sparse GQA qualify both metadata changes during Graph replay and envelope
changes through Graph executable updates, including transitions through all-select/inert calls.
Their numerical criteria remain independent of target, extent and private arithmetic.
QSA tail, prepare, indexer and sparse GQA retain partial-initialization cases for Compute Sanitizer
initcheck. Use `--list` / `--case` to select the read-footprint recipe rather than running a whole
format sweep under instrumentation.

Both execution targets share mathematical cases and criteria. Cases derived from device size query
the actual SM count after selection, so listing does not require a CUDA device. Run the Spark build
with `build/dgx-spark-dev` in these commands when qualifying that target.

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
RankProgram or establish native Op support.

The C++ prefix/MTP integration test is separately opt-in because it loads the full artifact and
runs the real engine:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_6_27b.ninfer \
  ctest --test-dir build/rtx5090-dev -R ninfer_qwen3_5_prefix_real_test --output-on-failure
```

The causal-scoring integration test uses the same artifact variable and checks a full 1,024-column
score tile, overlapping target suffixes, and finite, nonpositive results across repeated windows.
Floating-point execution does not promise bitwise-identical scores across calls:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer \
  ctest --test-dir build/rtx5090-dev -R ninfer_qwen3_5_score_real_test --output-on-failure
```

Qwen4Exp's native scoring test checks fresh-root state independently of the resulting logprobs.
It runs two short scoring jobs in one Session, verifies that the first job writes nonzero state,
then checks zero StateImage contents and initial PLE raw history before the second job computes.
The executed device batch must start at position zero with no preceding KV history visible.
It loads all weights on Device:

```bash
NINFER_QWEN4_EXP_TEST_ARTIFACT=out/qwen3_8_flash_next.ninfer \
  ./build/dgx-spark-dev/tests/ninfer_qwen4_exp_score_state_real_test
```

`ninfer_moe_test` qualifies the integer routing, complete MoE and routed SwiGLU profiles.
`ninfer_moe_block_scaled_test` qualifies NVFP4 A4 and block-FP8 A8/G128 experts with BF16
control/shared and small-token A16 routes. It uses independent FP64 formulas and encoded-weight
decoders, covering per-expert divisors, K128 scales, zero groups, affine/indexed banks, masks,
changed Graph contents, private route boundaries and workspace guards. `--format nvfp4` or
`--format fp8` selects one format; CTest runs both.
`ninfer_elementwise_test --suite sum_contributions` covers FP32 partial summation with optional
BF16 residual.
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
`ninfer_moe_mixed_test` qualifies complementary CPU/GPU subsets plus GPU shared/residual composition.

`ninfer_moe_execution_test` qualifies the shared single-layer execution and the RankProgram-owned
multi-layer fixed-cache and streaming-window schedule against the same independent complete-MoE oracle. It uses registered native Host banks and the real ExpertCache,
covering Write/AddResidual outputs, absent CPU-only replicas, explicit region replacement,
All/None/partial partitions, bank IDs independent of traversal order, variable frame reuse through
T=1024, delayed bridge progress, failed acquisition and asynchronous teardown. Eager and segmented
Graph handoffs use the same oracle; Graph cases also cover expert relocation and streaming-window
transitions through the real cache schedule. The compact Graph calibration storage also runs
All/None/partial contributions against the same oracle while reusing one bank window.

```bash
ctest --test-dir build/rtx5090-dev -R '^ninfer_moe(_mixed|_block_scaled|_cpu_(groupwise_int|nvfp4)(_avx2|_fma)?)?_test$' --output-on-failure
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

The MTP history test compares committed KV with an independent Main-conditioning oracle:

```bash
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer ./build/rtx5090-dev/tests/ninfer_qwen3_5_mtp_history_real_test
NINFER_TEST_ARTIFACT=$PWD/out/qwen3_8_27b_nvfp4.ninfer NINFER_TEST_GRAPH=1 ./build/rtx5090-dev/tests/ninfer_qwen3_5_mtp_history_real_test
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

`ninfer_tool_streaming_test` checks incremental tool arguments without loading weights: long HTML
strings and JSON containers, byte-split Unicode/framing, completion and interruption at every
boundary, continuation initialization, request-local format errors, stable protocol IDs, and
Chat/Responses/Anthropic argument equivalence. It also checks Anthropic eager versus parameter-buffered
publication, terminal-reason and item-status consistency, and exclusion of incomplete calls from
executable history. Unknown parameter types
wait for parameter completion before inference; known JSON encodings do not repair malformed values.
`ninfer_qwen_tool_constraints_test` additionally checks open-name uniqueness before sampling,
whole-token boundary crossings, speculative rollback, independent requests, and continuation seeds.

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
CTest supplies each Python oracle's native probe path. For direct invocation, set
`NINFER_TOOL_PROBE`, `NINFER_REGEX_PROBE`, or `NINFER_SCHEMA_PROBE` to the corresponding test binary
under the selected build tree; the scripts have no implicit build-directory choice.
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
  --server-bin build/rtx5090-dev/apps/ninfer-serve \
  --artifact out/qwen3_6_27b.ninfer --backend mtp

python3 tools/smoke/serve_thinking_preservation.py \
  --server-bin build/rtx5090-dev/apps/ninfer-serve \
  --artifact out/qwen3_6_35b_a3b.ninfer --backend dflash
```

The shared messages are in
[`fixtures/serve/qwen3_6_thinking_preservation.json`](fixtures/serve/qwen3_6_thinking_preservation.json).

## What belongs here

A permanent test should protect one current risk, such as:

- exact artifact bytes, geometry, object binding, or conversion transform;
- a numerical operator contract with an independent oracle;
- model Frontend, Session frontier/prefix policy, or RankProgram MTP/multimodal execution;
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

`ninfer_recovery_test` compares the shared branch-loss definition with an independent exhaustive
oracle, including aliases, invalid/foreign generations, forced shortlist collisions, and unavailable
fallbacks. It also checks physical union probes and rollback against complete holder enumeration.
`ninfer_resource_manager_test` compares incremental Host selection with full rescoring and bounds
deterministic score/holder work for 16–128 independent candidates. These operation counts check
algorithmic regressions; they do not establish inference throughput or latency.
`ninfer_kv_history_test` also exercises both actual Qwen checkpoint record types, separate backend
frontiers, external snapshots, reader pins, excluded aliases, and Device/Host release quantities.

## DFlash2 Engine integration

The DFlash prefill regression checks actual KV contents after a StateImage fork and a conflicting
decode binding, including shortened chunks and oversized local/full KV appends. It uses native
RankProgram storage and the production prefill route; select the draft component stored in the artifact:

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
NINFER_QWEN4_EXP_TEST_ARTIFACT=out/qwen3_8_flash_next.ninfer \
  ./build/rtx5090-dev/tests/ninfer_qwen4_exp_reclaim_real_test
```

The first command plans component combinations, KV formats and concurrency without loading weight
payloads. `--materialize` additionally loads one full-component model, checks stored words and
parameter sharing, constructs its execution resources, and reads/uploads a PLE window. It needs
about 64 GiB of available Host memory plus system headroom with `--host-experts`. For resident
loading on Spark, use its build directory and omit `--host-experts`. These are loading/resource checks;
Text/Vision/MTP generation, eager or CUDA Graph decode, and text causal scoring use the public Engine,
with current support described in the [execution reference](../docs/maintainer/qwen4_exp-execution.md)
and commands in the [CLI guide](../docs/cli.md).

The reclaim test loads Main experts on Host and MTP/fixed weights on Device, then constructs
native checkpoint contents without generating text. It checks complete holder quotes, actual-page
batch coverage, typed shortage clipping, stale holders/readers/residency, Main/MTP/State migrations,
and quoted Host release against final store occupancy. It requires the same Host memory headroom.

## Distributed execution

Configure with `-DNINFER_ENABLE_DISTRIBUTED=ON` and system NCCL development libraries.
`ninfer_distributed_protocol_test` is an ordinary CTest entry; it checks native Work values,
decision/commit requirements, input ownership, memory reports and transport backpressure.

The two-GPU test is explicit. Create a local TOML file using the commented deployment section in
`ninfer.example.toml`; set each node's `worker` to its absolute path to
`tests/ninfer_distributed_execution_test`. Both nodes need the matching binary and `lib/` directory.
From the configured coordinator:

```bash
./build/dgx-spark-dev/tests/ninfer_distributed_execution_test --config distributed.local.toml
```

The test launches local/SSH workers and uses the production Executor, transport and NCCL backend.
It checks SUM, unsigned MAX and exact PLE words, Gather/Scatter/Broadcast/P2P, Graph replay,
decision input, delayed-rank commit, owner subsets, cancellation, retirement, resource queries,
peak reset, remote launch failure and failure during a collective. No model payload is loaded.
The test's Program supplies an independent small mathematical/state reference; it is not a model
registration or an alternative inference entry.
