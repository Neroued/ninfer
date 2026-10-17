# Benchmarks

`ninfer_bench` measures the complete public `ninfer::Engine` route against a `.ninfer` artifact.
The `bench/ops/` `ninfer_<op>_bench` executables measure central public Op contracts while leaving
implementation selection behind those contracts. Model benchmarks measure Program/model
composition. Correctness lives in the affected test suites; development rules are in
[`../docs/maintainer/op-development.md`](../docs/maintainer/op-development.md).

The [Serve TTFT campaign](../tools/bench/ttft/README.md) measures public-HTTP latency, streaming gaps,
prefix reuse and preemption. Its frozen request corpus is documented under
[`fixtures/ttft/`](fixtures/ttft/README.md). That client does not call the benchmark executables or
Engine directly.

## Benchmark inputs

Synthetic numerical inputs use reproducible pseudorandom values with distinct operand seeds,
finite values, and format-correct codes/scales. Floating ranges reflect the operand's role;
control inputs (masks, positions, routing patterns), padding, and output/workspace initialization
retain their defined semantics. Model benchmarks use actual artifact weights and text fixtures.
The shared generators live in `common/fixture_data.cuh` and `ops/quantized_weight.cuh`.

Initialization and restoration are outside timed regions. In-place Ops restore their initial
operands before each sample, warmup, and profile call; their timed graphs contain one Op rather
than repeatedly transforming the same data. Cold measurements restore first, then evict L2 by
reading a preinitialized nonrepeating buffer; eviction must not leave a large dirty writeback
workload. These changes establish a new input/timing baseline: older constant or periodic
fixtures and repeated in-place chains are not directly comparable. Compare implementations using
the same fixture, seed, cache policy, and timing mode.

## Build

`CMakeLists.txt` includes explicit registrations from `ops/`, `inference/`, `context_cost/`
and `models/`. All executables remain under `<build-dir>/bench/`; the Op registration helper
lives in `cmake/NinferBenchmarks.cmake`.

`ninfer_cpu_execution_bench` measures synchronous collective dispatch and eight team barriers.
For example: `./build/rtx5090-dev/bench/ninfer_cpu_execution_bench --workers 8 --samples 500 --idle-us 1000`.
Workers select physical cores from the allowed topology; `--caller-cpu ID` optionally binds the
submitter. The benchmark prints actual affinity and ISA facts and P50/P95/max call latency.
Process CPU/wall cost is averaged over a separate run lasting at least 250 ms at the same cadence,
including each post-call idle interval; `cpu_calls` is that run's call count. This avoids interpreting
sub-millisecond OS accounting samples as steady CPU cost. Setup and warmup are excluded.
These are scheduling costs, not CPU arithmetic throughput or model performance.

`ninfer_ple_disk_prepare_bench` measures Qwen4Exp disk preparation and its independent upload slots
using an explicit NF4 artifact and a raw little-endian U32 token corpus:

```bash
build/rtx5090-dev/bench/ninfer_ple_disk_prepare_bench /explicit/model.ninfer /explicit/corpus.u32 8 16 6 64 \
  --read-budget-mib 64
```

Arguments after the paths are B, T, sample count and raw-row cache MiB (zero disables it).
CSV records IDs-ready to HostReady/upload-complete time, actual PLE reads and row reuse. Each process
starts with an empty application cache; successive windows consume consecutive corpus segments.
Lane b starts at corpus index `100002+b*100003`, matching the retained SSD experiment's trace.
The first sample is retained. The read budget bounds requested PLE data bytes; startup metadata and
optional projection-weight loading are separate. No table is generated or rewritten.
`--incremental-us N` models candidate arrival intervals and also reports time remaining after the
last append. `--overlap N` runs N independent real merged K/V projections, then consumes E; CUDA
events measure the work interval and the exposed wait before that consumer. This measures component
overlap, not complete model scheduling or generation speed. `--dump PATH` emits the first input,
Host/Device E and optional projection output for independent qualification, outside timing.

```bash
cmake --preset rtx5090-dev
cmake --build build/rtx5090-dev -j --target ninfer_bench
```

The `rtx5090-dev` configure preset also enables all benchmarks, alongside products and tests; see
[Build system](../docs/maintainer/build-system.md) for presets and dependencies. Tests use a
Python interpreter; a benchmark-only configuration does not require one. The `rtx5090-ops`
and `dgx-spark-ops` presets omit Engine/model benchmarks. CPU MoE/executor benchmarks are x86-only.
Examples below use `rtx5090-dev`; substitute the selected build directory. A multi-Op benchmark
may expose entries that its target does not implement; those requests report the Op error.

## Hardware references

[`common/roofline.h`](common/roofline.h) owns the compile-target reference table. Benchmarks report
the actual GPU separately; they do not choose a reference from its name or estimate nominal
bandwidth from runtime memory-clock attributes. Logical GB/s and TFLOP/s describe useful work,
not measured DRAM traffic or executed Tensor Core instructions.

| Reference | RTX 5090 | DGX Spark |
|---|---:|---:|
| Nominal memory bandwidth, GB/s | 1792 | 273 |
| Measured sustained pure-read bandwidth, GB/s | 1674.5 | 257.42 |
| Dense BF16 / FP32 accumulation, TFLOP/s | 209.5 | 125.89 |
| Dense FP8 E4M3 / FP16 accumulation, TFLOP/s | 838 | 250.80 |
| Dense FP8 E4M3 / FP32 accumulation, TFLOP/s | 419 | 249.75 |
| Dense MXFP8 E4M3 / FP32 accumulation, TFLOP/s | 838 | 249.52 |
| Dense NVFP4 / FP32 accumulation, TFLOP/s | 1676 | 499.02 |

The 5090 sustained-read reference comes from the [memory probe](../tools/hbm_bandwidth_probe.cu)
using a 4 GiB `uint4` read workload. The plain BF16/FP8 and NVFP4 peaks follow the dense rates in
[NVIDIA's RTX Blackwell whitepaper, Appendix A](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
The MXFP8 reference describes the full-rate block-scaled FP32 accumulation path. Spark's nominal
bandwidth follows the [NVIDIA hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).

Spark's sustained-read and Tensor Core references use the same measured calibration on
`192.168.1.51`, recorded on 2026-10-09 in `/home/neroued/SPARK.md`; its source is
`/home/neroued/gb10_bench.cu` on that machine. The fixed launch uses 48 SMs, four 256-thread CTAs
per SM, default GPU Boost, a two-second warmup, and the median of seven samples. Memory reads
stream a 1 GiB nonconstant buffer for 16 passes. Tensor tests keep operands in registers and use
eight independent accumulator chains; each warp MMA counts `2*M*N*K` FLOPs. Alternating signs
keep FP16 accumulators finite, and the benchmark checks the computed results. These are measured
MMA throughput references, not official nominal specifications or complete GEMM throughput.
In this calibration, ordinary FP8 FP32 accumulation has the same throughput as FP16 accumulation;
the RTX 5090 half-rate relationship does not apply to GB10. All Spark Tensor references, including
NVFP4, use that measured basis. Unestablished references remain absent: derived percentages print
`-` in console output and an empty CSV field.

Use the profile matching the arithmetic actually selected by the Op, not merely its weight type or
activation permission. Shape/T classification belongs to the affected benchmark, outside the
hardware table, and must follow changes to production dispatch. SIMT routes do not report Tensor
Core utilization. Hardware-reference changes do not change Op dispatch.

`ninfer_memory_bandwidth_bench` measures sustained read, write and copy rates using the selected
target's nominal bandwidth. `--peak-gbps` explicitly overrides that comparison value. Build/run
instructions are in [Tools](../tools/README.md#memory-bandwidth-probe).

## Product benchmark

The benchmark slices exact token counts from `bench/fixtures/bench_corpus.ids`, calls
`Engine::prepare_tokens()`, then calls `Engine::generate()` once for each repetition. It does not
have a private prefill/decode loop and does not call model implementation interfaces.

The matrix contains three independently measured test kinds:

- `pp{P}` prepares `P` tokens and requests one output token. This is the smallest request that runs
  the model; `prefill t/s` is `P / GenerationTimings.prefill_seconds`.
- `tg{G}` prepares a one-token seed outside the reported phase and requests `G+1` output tokens.
  The begin-round token belongs to prefill, leaving exactly `G` tokens in the reported decode
  phase.
- `pp{P}+tg{G}` uses the same `G+1` convention after a `P`-token prefill and reports both phase
  rates from the same generation call.

Unconstrained requests use raw output and disable model-default stops, keeping the requested
token count exact. All requests disable prefix reuse. When CUDA Graph is enabled and the matrix
contains decode work, a public generation batch with the selected grammar/concurrency settings
primes the decode graphs before warmups and measured repetitions.

## CLI

```text
ninfer_bench --weights <artifact.ninfer>
          [--corpus <ids-path>]
          [-p, --n-prompt <list>]
          [-n, --n-gen <list>]
          [-pg, --prompt-gen <P,G;P,G...>]
          [-r, --repetitions <n>] [--warmup <n>]
          [--concurrency <1..8>]
          [--grammar-file <path> | --json-schema-file <path> | --json-object | --regex <pattern> | --choice <text> ...]
          [--mixed-constraints]
          [--max-ctx <tokens>] [--prefill-chunk <tokens>]
          [--kv-dtype <bf16|int8|fp8|nvfp4|k8v4>]
          [--spec <mtp|dflash|dflash2> --draft-tokens <n>] [--lm-head-draft]
          [--device <id>] [--no-cuda-graph] [--profile-measured]
          [-o, --output <table|json|csv>] [--output-file <path>]
```

With no `-p`, `-n`, or `-pg`, the matrix is `pp512` and `tg128`.
`--prefill-chunk` accepts a positive multiple of 128. When omitted, Engine uses 1024 for resident
models or 4096 for CPU expert offload.

Example:

```bash
./build/rtx5090-dev/bench/ninfer_bench \
  --weights out/qwen3_6_27b.ninfer \
  -p 512,2048 -n 128 -pg '2048,128' -r 5 --warmup 1
```

Select a backend with `--spec mtp|dflash|dflash2 --draft-tokens K` (MTP K=1..5, DFlash/DFlash2
K=1..15); `--lm-head-draft` selects the optimized proposal head. CUDA Graph decode is
enabled by default.

`--profile-measured` is a benchmark-only profiler boundary. It requires exactly one selected test
and `-r 1`, synchronizes after warmup, and brackets only the measured repetition with
`cudaProfilerStart/Stop`. Use it with an Nsight Systems `cudaProfilerApi` capture range so artifact
load, graph construction, and warmup do not enter topology counts.

For a DFlash2 companion artifact:

```bash
./build/rtx5090-dev/bench/ninfer_bench --weights out/qwen3_8_27b_nvfp4.ninfer \
  -pg '2048,128' --spec dflash2 --draft-tokens 7 --lm-head-draft \
  --max-ctx 4096 --kv-dtype bf16 --warmup 1 -r 3
```

`--concurrency` submits that many prepared requests per repetition. `--grammar-file`,
`--json-schema-file`, `--json-object`, `--regex` and repeated `--choice` select mutually exclusive output constraints and enable
normal text output and default EOS; a constrained request may finish before the output limit.
`--mixed-constraints` applies the selected constraint to alternate requests and requires concurrency
of at least two. Warmups use the same mixture and compile constraints before measured repetitions.
`batch out t/s` measures all actual output tokens over the repetition's submit-to-completion wall
time. Per-request phase rates remain separate from this aggregate throughput.

The benchmark disables context retention because every repetition is an independent root request.
Reports record `speculative_backend`, `draft_tokens`, and the proposal head independently;
JSON and CSV identify DFlash2 explicitly. MTP alone reserves its extra lookahead KV margin.

## Context-cost calibration

`ninfer_context_cost_bench` measures the static coefficients used to compare context-cache
materialization alternatives. It is an offline tool, not a startup benchmark or runtime autotuner.
It has two independent suites:

- `transfer` measures one machine-level D2H/H2D/D2D roofline with the production paged-KV transfer
  primitives and representative PageMajor/HeadMajor geometries. Contiguous D2D batches independently
  vary bytes and operation count to represent StateImage components and identify device bandwidth.
  It uses synthetic non-compressible buffers and never inspects an artifact or loads a model.
- `prefill` loads one artifact through the public Engine and measures its Text/Vision recomputation
  cost under the canonical BF16/no-spec configuration. Its result is keyed only by hardware class
  and the actual Text/Vision `prefill_signature`.

The runtime models are:

```text
transfer = max(batch_ns + copy_operations * operation_ns,
               payload_bytes * ns_per_byte)
prefill  = chunks * chunk_ns
         + suffix_tokens * token_ns
         + attention_pairs * attention_pair_ns
         + vision_items * vision_item_ns
         + vision_patches * vision_patch_ns
```

`payload_bytes` is the actual copied payload, excluding Host arena padding. `copy_operations` is the
number of physical `cudaMemcpy*` calls implied by the page geometry and contiguous runs. Resource
kind, KV dtype, and speculative backend do not select different coefficients; any differences they
create are represented by those two physical quantities. For a suffix `S` after prefix `B`,
`attention_pairs = B*S + S*(S+1)/2`. `chunks` is the sum of
`ceil(segment_tokens/prefill_chunk)` across the actual prefill schedule's capture/rewrite segments;
with no such boundary it is simply `ceil(S/prefill_chunk)`.

Build the tool, then measure machine transfer without a model:

```bash
cmake --build build/rtx5090-dev -j --target ninfer_context_cost_bench
./build/rtx5090-dev/bench/ninfer_context_cost_bench \
  --suite transfer \
  --json profiles/bench/context_cost_transfer.json \
  --preset-out profiles/bench/context_cost_presets.json
```

Measure prefill separately when the GPU has room for the artifact:

```bash
./build/rtx5090-dev/bench/ninfer_context_cost_bench \
  --suite prefill \
  --artifact out/qwen3_6_27b.ninfer \
  --corpus bench/fixtures/bench_corpus.ids \
  --json profiles/bench/qwen3_6_27b_prefill_cost.json \
  --preset-out profiles/bench/context_cost_presets.json
```

`--suite all` performs transfer first, releases its fixtures, then loads the artifact for prefill.
The JSON report retains every repetition, median/MAD, floating and quantized coefficients, and
training/held-out predictions. A fit is accepted only when both training and held-out p95 relative
error are at most 35% for transfer and 15% for prefill, and materially different points in either
partition are never predicted in reverse order. Exact predicted ties are reported separately and
are allowed: they enter the runtime's deterministic semantic tie-break. Rejection writes the
report, exits with status 3, and leaves the preset unchanged.

`--preset-out` is valid for every suite. It atomically replaces only the component measured by that
suite: a transfer run updates the hardware node's three directions, while a prefill run updates one
prefill-signature entry and preserves machine transfer plus other entries. The resulting file is parsed by
the runtime's strict loader before publication.

[`context_cost_defaults.cpp`](../src/runtime/engine/context_cache/context_cost_defaults.cpp) is the sole table of
defaults compiled into the binary. JSON is not a build input: `--preset-out` produces a runtime
registry that can be selected immediately, without recompilation, through
`EngineOptions.context_cost.preset_path` or `ninfer-serve --context-cost-presets`. Resolution always
starts with generic numerical coefficients,
then independently applies matching compiled transfer/prefill values, then independently applies
matching external values. A malformed explicit file is an error; a missing hardware or prefill-signature
entry retains the preceding numerical layer. After real-scenario acceptance, a maintainer may
promote quantized values into the C++ table. The detailed `--json` report is diagnostic provenance
and is not a runtime input.

## Linear Op benchmark

`ninfer_linear_bench` measures only the public pure `linear()` contract. It supports Q4, Q5, Q6,
Q8, registered BF16 weights, the registered NVFP4 problems, and the registered FP8 problems.
Select an exact extent with `--t`, a regular range with `--sweep`, or explicit extents with
`--tokens T1,T2,...`; these options are mutually exclusive.
Existing formats use `--policy a16`; NVFP4 additionally supports `--policy a4`, and
FP8 supports `--policy a8`. Each permission lets the production resolver select the qualified
route for the exact geometry and T. LinearAdd, LinearSwiGLU,
LinearPair, Attention/GDN projections, and sparse MoE remain separate semantic Ops and are not
benchmark modes here.

This is a long-lived public benchmark: every timed and profiled point calls `ninfer::ops::linear`
and lets production dispatch choose the implementation. Candidate crossover work uses a
task-local temporary sweep, puts the winner or boundary in production dispatch, and deletes losing
candidates and temporary controls afterward; private launchers and route forcing do not belong in
this retained benchmark. An Op-scoped Linear qualification uses this benchmark directly and does
not load an artifact or invoke the target, Program, Engine, or round benchmarks documented
elsewhere in this file.

Build the benchmark and measure one exact production point:

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_linear_bench
./build/rtx5090-dev/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 4096 --k 5120 --t 8
./build/rtx5090-dev/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a4 --n 14336 --k 5120 --t 1024
./build/rtx5090-dev/bench/ninfer_linear_bench \
  --qtype fp8 --policy a8 --n 14336 --k 5120 --t 1
./build/rtx5090-dev/bench/ninfer_linear_bench \
  --qtype fp8 --policy a8 --n 16384 --k 5120 --t 1024
```

Linear tuning measures every valid T through 128 and the two bulk anchors, 512 and 1024.
[Linear tuning and performance reports](../docs/maintainer/linear-tuning.md) defines workload
priorities, dispatch tradeoffs, and the final curve and metric tables. Respect each Op's extent
alignment; these measurement targets do not limit supported T.

A continuous small-T sweep reuses one packed weight and one maximum-T activation/output
allocation:

```bash
./build/rtx5090-dev/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 4096 --k 5120 \
  --sweep 1:128:1 --csv-out profiles/bench/q4_4096x5120_t1_128.csv
./build/rtx5090-dev/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 3456 --k 1152 \
  --sweep 4:512:4 --csv-out profiles/bench/q4_vision_qkv.csv
./build/rtx5090-dev/bench/ninfer_linear_bench \
  --qtype q8 --policy a16 --n 248320 --k 5120 \
  --sweep 48:65:1 --warmup 5 --repeat 30
```

The registered suites run representative public Linear shapes for one or both exact products.
They are compact performance surveys, not copies of the production selector or numerical test
matrix:

```bash
./build/rtx5090-dev/bench/ninfer_linear_bench --suite qwen3_6_27b
./build/rtx5090-dev/bench/ninfer_linear_bench --suite qwen3_6_35b_a3b
./build/rtx5090-dev/bench/ninfer_linear_bench --suite all
```

When a point group includes `T=1`, `T1_lin_x` reports the calculated ratio
`T * median(T=1) / median(T)`. It is a scaling reference from one public `T=1` benchmark point,
measured with the configured warmup and repetitions; the benchmark never issues T sequential
`T=1` launches as a substitute workload.

For NCU, `--profile` performs setup, warmup, and the L2 flush before enabling the profiler, then
captures exactly one public Linear call. That call may emit one or more production-selected kernel
launches:

```bash
ncu --profile-from-start off --set full \
  ./build/rtx5090-dev/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 4096 --k 5120 --t 8 --profile
```

`--execution eager` (default) times the complete public call with CUDA events;
`--execution graph` captures that same call and times one replay. Both include every kernel the
Op emits. CSV records `execution` and, for Graph measurements, `graph_nodes`; the terminal header
also records execution mode and CUDA runtime. `--profile` follows the selected execution mode.

For short Ops, `--execution graph --graph-calls N` captures N serial calls (1..64, default 1).
Latency and calculated work rates are normalized per call; CSV `graph_nodes` counts all nodes in
the bundle and `graph_calls` records N. The cache flush occurs only before the bundle, so later
calls reuse cached inputs. These rows are labeled `cold-before-graph-bundle` and omit DRAM/read
percentages and the DRAM memory floor. Multiple calls are for timing and cannot be combined with
`--profile`, which always captures one complete public call.

Every ordinary sample is cold-cache: a 256 MiB L2 eviction read completes before the timed
interval. [Linear measurement rules](../docs/maintainer/linear-tuning.md#5-logical-metrics) define
byte/FLOP accounting, hardware references, timing conditions and suite expansion. Physical traffic
and instruction utilization require profiler evidence.

## LinearTopK Op benchmark

`ninfer_linear_topk_bench` measures the complete public projection-plus-stable-top-16 Op for the
Q8 and row-FP8 full heads and the mapped Q4 optimized head. Its independent matrix-column axis is
`U`; the default sweep covers every `U=1..120` needed by variable-width DFlash2 proposals. A fixed
positive `--columns U` also exercises larger matrices; `--columns U,...` selects a representative
set without a full sweep. Codes and stored scales vary across the physical head, including signs,
so ranking is not timed only on identical weight rows. Each sample replays the complete Op in a
CUDA Graph after a 256 MiB L2 eviction read; projection, partial-key reduction, workspace counter
initialization, and final ids/scores publication are included. Defaults are 8 warmups and 60 samples.
Reported logical bandwidth counts the encoded head once, useful FLOPs count candidate-eligible rows, and
workspace bytes and Graph nodes describe the actual public call. These are Op measurements.

```bash
cmake --build build/rtx5090-dev -j --target ninfer_linear_topk_bench
./build/rtx5090-dev/bench/ninfer_linear_topk_bench
./build/rtx5090-dev/bench/ninfer_linear_topk_bench --profile q8-full --columns 7
./build/rtx5090-dev/bench/ninfer_linear_topk_bench --columns 1,7,16,24,64,120 --repeat 25
./build/rtx5090-dev/bench/ninfer_linear_topk_bench --profile fp8-full --columns 120
./build/rtx5090-dev/bench/ninfer_linear_topk_bench --profile q4-optimized --columns 129 --repeat 60
```

## CandidateSelectorPath Op benchmark

`ninfer_candidate_selector_bench` measures the complete public conditional selector for
`K=1..15`, `B=1..8`, 16 candidates and rank 256, with full BF16 codebooks `[256,248320]`.
The default sweep covers all K/B pairs in greedy, stochastic and mixed modes (B=1 omits the
redundant mixed case). Each cold-cache CUDA Graph sample follows a 256 MiB L2 eviction read.
It reports the selected route's required caller workspace and actual Graph node count. The timed
interval includes all device work of the complete public Op; fixture allocation and setup are outside it.

```bash
cmake --build build/rtx5090-dev -j --target ninfer_candidate_selector_bench
./build/rtx5090-dev/bench/ninfer_candidate_selector_bench --warmup 8 --repeat 60
./build/rtx5090-dev/bench/ninfer_candidate_selector_bench --steps 15 --batch 8 --repeat 60
```

## Embedding Op benchmark

`ninfer_embedding_bench` measures the public quantized embedding profiles: Q6 `[248320,5120]`,
Q8 `[248320,5120]`, Q8 `[248320,2048]`, and row-scaled FP8 `[248320,5120]`. The default sweep is
T=1..128; `--tokens` selects other matrix extents. This is an Op column domain, independent of a
particular speculative algorithm's draft count.

The fixture initializes nonuniform valid code planes and positive stored scales at the complete
registered table shape. `--id-pattern normal` uses tokenizer-addressable IDs. `masked` generates
an anchor followed by masks at the explicit `--block-width` period: 248070 for D=5120 and 248077
for existing Q8/D=2048. Use T=B*W to measure complete proposal blocks. CSV reports unique IDs,
logical per-token bytes, execution/cache mode, Graph nodes/calls, scratch, and median/min/p95.

Each default interval contains one public Op. Cold mode evicts 256 MiB of L2 before timing;
`--graph-calls 32 --execution graph --cache warm` measures repeated public calls in one Graph and
normalizes per call. A cold bundle flushes only before its first call. `--profile` uses one public
call of one format and one extent for kernel profiling.

```bash
cmake --build build/rtx5090-dev -j --target ninfer_embedding_bench
./build/rtx5090-dev/bench/ninfer_embedding_bench --format q8-d5120 --execution graph --cache cold
./build/rtx5090-dev/bench/ninfer_embedding_bench --format fp8-d5120 \
  --tokens 8,16,24,32,40,48,56,64 --id-pattern masked --block-width 8
./build/rtx5090-dev/bench/ninfer_embedding_bench --format fp8-d5120 \
  --tokens 16,32,48,64,80,96,112,128 --id-pattern masked --block-width 16 \
  --cache warm --graph-calls 32 --csv-out /tmp/embedding.csv
./build/rtx5090-dev/bench/ninfer_embedding_bench --format fp8-d5120 --tokens 128 --profile
```

## Batched feature scatter benchmark

`ninfer_scatter_bf16_batch_bench` measures five public `scatter_bf16_batch` calls. Each copies
one `[D,W,B]` source into a different D-row slice of the same `[5*D,W,8]` parent pool; default
D=5120 matches DFlash2 feature capture. The calls use nonuniform exact BF16 bit patterns,
permuted lane IDs and `--counts full|one|ragged|zero`. The measurement is the sum of five
consecutive captures, without intervening model layers; it does not establish Engine latency.

CSV reports live columns, bytes, zero scratch, Graph nodes/calls and median/min/p95 latency.
`--graph-calls 32 --cache warm` repeats the entire five-call capture in one Graph and normalizes
time per capture, so a bundle has 160 kernel nodes. Cold mode flushes 256 MiB before the timed
interval, not between the five calls or individual repetitions within a bundle.

```bash
cmake --build build/rtx5090-dev -j --target ninfer_scatter_bf16_batch_bench
./build/rtx5090-dev/bench/ninfer_scatter_bf16_batch_bench --widths 1,2,8,9,16 --batches 1,8 --counts full
./build/rtx5090-dev/bench/ninfer_scatter_bf16_batch_bench --widths 2,8,16 --batches 1,8 \
  --counts ragged --cache warm --graph-calls 32 --csv-out /tmp/feature-scatter.csv
./build/rtx5090-dev/bench/ninfer_scatter_bf16_batch_bench --widths 16 --batches 8 --profile
```

## RMSNorm Op benchmark

`ninfer_rmsnorm_bench` measures one public RMSNorm call selected by
`--op rmsnorm|offset|silu-gated|sigmoid-gated`. Plain and offset forms select direct or unit-offset
gain; both gated forms use direct gain. `--dim D --heads H --tokens T[,T...] --batch B` describes
BF16 `[D,H,T,B]`, with a separate D-element reduction for each of the `H*T*B` rows. Inputs, gains
and gates are nonuniform represented BF16 values. Defaults are `rmsnorm`, D=128, H=48, T=1, B=1.
`--sweep START:END[:STEP]` replaces the token list; `--tokens` and `--sweep` are mutually exclusive.

The RMSNorm, GDN control and HC benchmarks share `bench/ops/op_measurement.cuh`. Each sample times
a complete public Op or an explicitly named composition comparison. Graph GPU timing brackets the
captured Op body with external CUDA event nodes; eager GPU timing includes host launch gaps.
Host submission is measured separately.
Fixture preparation and cold-cache eviction precede the timed interval. CSV reports semantic
geometry, execution/cache mode, GPU and host median/min/p95 microseconds, `workspace_bytes`,
`workspace_peak`, and `kernels` (actual captured kernel nodes, excluding event nodes).
RMSNorm uses zero workspace. These measurements establish Op latency, not Engine latency.
Its CSV includes `logical_bytes`, `logical_gb_s` and `bandwidth_reference_us`: input, optional gate
and output traffic plus one weight-vector read. The bandwidth reference uses the target's nominal
memory bandwidth; repeated reads and cache/writeback behavior are not DRAM traffic measurements.

All three benchmarks accept `--execution graph|eager|both`, `--cache cold|warm|both`, `--warmup N`,
`--repeat N`, `--flush-mib N` and `--csv-out PATH`. Defaults are Graph, both cache modes,
10 warmups, 100 samples and a 256 MiB cold-cache flush; CSV goes to stdout unless a path is supplied.

```bash
cmake --build build/rtx5090-dev -j --target ninfer_rmsnorm_bench
./build/rtx5090-dev/bench/ninfer_rmsnorm_bench \
  --op rmsnorm --dim 5120 --heads 1 --tokens 1,8,16,32,64,96,128,2048
./build/rtx5090-dev/bench/ninfer_rmsnorm_bench \
  --op offset --dim 256 --heads 24 --tokens 1,8,16,32,64,96,128 \
  --execution graph --cache warm --csv-out /tmp/rmsnorm.csv
./build/rtx5090-dev/bench/ninfer_rmsnorm_bench \
  --op silu-gated --dim 128 --heads 48 --tokens 1,8,16,128
./build/rtx5090-dev/bench/ninfer_rmsnorm_bench \
  --op sigmoid-gated --dim 128 --heads 48 --batch 8 --tokens 1,7,16 \
  --execution both --cache both
```

## GDN control Op benchmark

`ninfer_gdn_control_bench` selects `--op control|projection|norm-projection` for public
`gdn_control`, `gdn_control_proj` and `gdn_norm_control_proj`. Direct control consumes BF16 a/b;
projection includes AB projection and both nonlinear controls; norm-projection also includes
offset input RMSNorm and its explicit BF16 normalized output. Controls are FP32. All numerical
inputs are nonuniform represented values.

`--heads H --input-rows K --weights parent|pair` supplies numerical geometry and weight storage.
Parent storage is a contiguous BF16 `[2H,K]` matrix, with A before B; pair storage uses two BF16
`[H,K]` matrices. Supported combinations are:

| Op | H | K | Weights |
|---|---:|---:|---|
| `control` | 48 | — | none; explicit `--input-rows` and `--weights` are rejected |
| `projection` | 48 | 2560 | `parent` |
| `projection`, `norm-projection` | 48 | 5120 | `parent` or `pair` |
| `projection`, `norm-projection` | 32 | 2048 | `parent` |

Defaults are `projection`, H=48, K=2560, `parent`, tokens=1 and batch=1. Norm-projection requires
an explicit supported K. `--tokens T[,T...]` or `--sweep START:END[:STEP]` selects T, with
`M=T*B`, column=`t+T*b`, and `--batch B` selecting B. Timing flags and CSV statistics follow the
[RMSNorm benchmark](#rmsnorm-op-benchmark); GDN also reports `weights`. The workspace columns
compare the public point capacity query with measured arena peak; direct control has no scratch.
Only operands needed by the selected Op are allocated. Production owns all dispatch.

```bash
cmake --build build/rtx5090-dev -j --target ninfer_gdn_control_bench
./build/rtx5090-dev/bench/ninfer_gdn_control_bench \
  --op projection --heads 48 --input-rows 2560 --weights parent --sweep 1:128 \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/rtx5090-dev/bench/ninfer_gdn_control_bench \
  --op norm-projection --heads 48 --input-rows 5120 --weights pair \
  --tokens 1,2,4,8,9,16,32,64,128 --execution graph --cache warm \
  --csv-out /tmp/gdn-norm-warm.csv
./build/rtx5090-dev/bench/ninfer_gdn_control_bench \
  --op norm-projection --heads 32 --input-rows 2048 --tokens 16,17,128
./build/rtx5090-dev/bench/ninfer_gdn_control_bench \
  --op control --heads 48 --batch 8 --tokens 1,7,16 --execution both --cache warm
```

## HC gated residual Op benchmark

`ninfer_gated_residual_bench` measures the five public HC residual forms with
`--op expand-norm|grouped-rmsnorm|combine|combine-norm|add-norm`. Geometry is fixed at H=2560
and four streams. Input/output residual and normalized tensors are BF16 `[2560,4,M]`,
embedding/block tensors are BF16 `[2560,M]`, and activated injection is FP32 `[4,M]`.
`add-norm` selects the unit-weight overload without an injection operand.

`block-mix` and `final-mix` measure the complete public `hc_mix` overloads, including both BF16
projections, nonlinearities, four-stream mean and Block injection. They consume normalized BF16
`[2560,4,M]`, native compact BF16 Down/Up parents and caller-owned workspace. The fixture seeds
nonuniform inputs/weights, allocates only the selected form’s operands, and queries/allocates scratch
before timing. No norm or residual combine is included; internal routes remain production-owned.
These are new capabilities without a preexisting production Mix baseline.

`combine-then-norm` times public `hc_combine` followed by `hc_grouped_rmsnorm` as one interval.
`add-then-norm` uses the same composition with an all-one FP32 injection prepared before timing.
These are explicit two-Op decomposition comparisons for a new capability, with no preexisting
HC production implementation as a performance baseline. Production owns the fused entry's internal
choice of one or multiple kernels; the benchmark reports actual kernel count.

Defaults are `combine-norm`, tokens=1, batch=1 and `--alias disjoint`. `--alias inplace` is
available for `combine`, `combine-norm`, `add-norm` and the staged comparisons. Original inputs are
restored on the measurement stream before each interval, then cold mode flushes the cache. `--tokens T[,T...]`
or mutually exclusive `--sweep START:END[:STEP]`, together with `--batch B`, gives M=T*B.
Timing flags, defaults and CSV statistics follow the [RMSNorm benchmark](#rmsnorm-op-benchmark);
HC adds `alias` and `condition_us`. Residual forms have zero workspace; Mix reports its public
point capacity and actual arena peak, excluding public buffers. `--condition-us N`
defaults to 200: register-only GPU work precedes restoration and optional eviction, outside the
timed Op body. This gives both cache modes an activity prelude without warming operand data or
locking clocks. It also allows eager submissions to queue behind the prelude; use
`--condition-us 0` to measure without that setup, and keep the two conditions separate.
Fixtures allocate only selected operands and an in-place backup when needed. The measurements cover
Op latency and the named compositions.

Residual/norm rows also report `logical_bytes`, `logical_gb_s` and `bandwidth_reference_us`.
The logical count includes each public input read and output write, with shared weights counted
once per call; the named two-Op compositions include the intermediate residual reread.
The reference uses the target's nominal memory bandwidth. Private repeated loads and internal
staging are excluded, and cache retention/writeback can make the logical rate exceed DRAM bandwidth.
Mix leaves these fields empty because its contractions require separate work accounting.

```bash
cmake --build build/rtx5090-dev -j --target ninfer_gated_residual_bench
./build/rtx5090-dev/bench/ninfer_gated_residual_bench \
  --op expand-norm --tokens 1,8,128,1024,2048 --execution both --cache both
./build/rtx5090-dev/bench/ninfer_gated_residual_bench \
  --op combine-norm --alias inplace --tokens 1,8,128,2048,4096 \
  --execution graph --cache cold --csv-out /tmp/hc-combine-norm.csv
./build/rtx5090-dev/bench/ninfer_gated_residual_bench \
  --op combine-then-norm --alias inplace --tokens 1,8,128,2048,4096 \
  --execution graph --cache cold --csv-out /tmp/hc-staged.csv
./build/rtx5090-dev/bench/ninfer_gated_residual_bench \
  --op add-norm --tokens 1,7,16 --batch 8 --execution both --cache warm
./build/rtx5090-dev/bench/ninfer_gated_residual_bench \
  --op combine-norm --tokens 1,128 --execution eager --cache warm --condition-us 0
./build/rtx5090-dev/bench/ninfer_gated_residual_bench \
  --op block-mix --sweep 1:128 --execution graph --cache both \
  --warmup 10 --repeat 61 --csv-out /tmp/hc-block-mix.csv
./build/rtx5090-dev/bench/ninfer_gated_residual_bench \
  --op final-mix --tokens 1,4,8,64,128,512,1024 --execution both --cache both
./build/rtx5090-dev/bench/ninfer_gated_residual_bench \
  --op block-mix --tokens 1,7,16 --batch 8 --execution both --cache warm
```

## Dynamic grouped-convolution prepare Op benchmark

`ninfer_dynamic_grouped_conv_prepare_bench` measures the complete BF16
`rmsnorm_dynamic_grouped_conv_prepare` contract at every registered `B=1..8`. Each timed call
includes RMSNorm, the `[1280,5120]` coefficient projection, the input-side two-tap grouped
convolution, and both public output writes. Every sample follows a 256 MiB L2 eviction; the report
contains complete-Op latency, useful coefficient-projection TFLOP/s, and exact workspace capacity.

```bash
cmake --build build/rtx5090-dev -j --target ninfer_dynamic_grouped_conv_prepare_bench
./build/rtx5090-dev/bench/ninfer_dynamic_grouped_conv_prepare_bench --warmup 10 --repeat 80
```

## GDN benchmarks

The three GDN executables share `ops/op_measurement.cuh`: CSV fields, option parsing, fixture
uploads, and timing. Each graph contains one selected Op (or an explicitly named analysis stage),
with external CUDA events bracketing its body. `kernels` counts kernel nodes only. Eager GPU timing
includes host launch gaps; host submission time is reported separately. Mutable state is restored
before each invocation, and cold-cache samples evict 256 MiB from L2 before the timer. These are
Op measurements, not model throughput measurements.

Common flags are `--execution graph|eager|both`, `--cache cold|warm|both`, `--warmup N`,
`--repeat N`, `--flush-mib N`, and `--csv-out PATH`. CSV goes to stdout when no path is supplied.
`--tokens` accepts an exact extent or comma-separated list; `--sweep START:END[:STEP]` accepts an
inclusive interval. Capacity and actual workspace high-water are reported separately.

`ninfer_gdn_input_proj_bench` selects `--op snapshot|record`. `--format` and the actual
`--n`/`--k` identify the stored matrix; `--policy a16|allow-a8|allow-a4` reports the caller's
activation permission. Supported geometries are Q8 `[12288,2048]` and `[16384,2560]`, and
Q4/Q5, FP8, or NVFP4 `[16384,5120]`. Q4/Q5 uses the two stored parents `[4096,5120]` and
`[12288,5120]`. Convolution uses physical T and B, with optional device
valid-prefix extents supplied by `--valid-columns`.

```bash
./build/rtx5090-dev/bench/ninfer_gdn_input_proj_bench \
  --op snapshot --format nvfp4 --n 16384 --k 5120 --policy allow-a4 --sweep 1:17
./build/rtx5090-dev/bench/ninfer_gdn_input_proj_bench \
  --op record --format fp8 --n 16384 --k 5120 --policy allow-a8 \
  --tokens 6 --batch 3 --valid-columns 6,3,1 --csv-out .agent-tmp/gdn/record.csv
```

`ninfer_gated_delta_net_bench` selects `--mode running|batch-update|chunked-raw|chunked-normalized|recurrent`.
Running and batch-update call public Ops. The other modes force a production mechanism for
crossover analysis; `chunked-raw` normalizes raw Q/K inside prepare, while `chunked-normalized`
supplies pre-normalized BF16 inputs and skips that normalization. `--breakdown` additionally
times the private prepare and recurrence stages. Separate cache conditioning means these isolated
medians are not an additive breakdown of pipeline latency. `value_tile` identifies the selected
chunked stripe for interpreting packet reuse. `--qk-norm composed` measures two public L2Norm
calls followed by a batch update at B=1; its scratch footprint includes the two materialized
BF16 Q/K tensors.

```bash
./build/rtx5090-dev/bench/ninfer_gated_delta_net_bench \
  --mode running --qk-heads 16 --value-heads 48 --tokens 1,15,16,17,1024 --cache cold
./build/rtx5090-dev/bench/ninfer_gated_delta_net_bench \
  --mode batch-update --value-heads 32 --batch 8 --qk-norm fused
./build/rtx5090-dev/bench/ninfer_gated_delta_net_bench \
  --mode chunked-raw --value-heads 48 --tokens 1024 --breakdown
```

`ninfer_gdn_replay_bench` selects `--op fold|record`. Fold measures all specified layers;
record measures one recurrence layer. Specify actual `--layers`, `--value-heads`, and
`--conv-channels`; Q/K heads are 16 and head dimension is 128. Fold geometries are
`48/48/10240`, `30/32/8192`, and `36/48/10240`. Both Ops accept T=2..16 and B=1..8.
`--commits dense|one|mixed|N[,N...]` selects fold lengths, including zero;
`--valid dense|mixed|N[,N...]` selects positive record prefixes. Host fold rows are captured
at graph construction; eager timing includes their submission on every call.

Both recurrence and Replay CSVs report the same traffic fields:

- `logical_read_bytes`, `logical_write_bytes`, and `logical_bytes` count unique boundary payloads
  by direction. Full calls exclude internal workspace; isolated stages include their workspace
  inputs or outputs. Reads and writes to the same address count separately.
- `state_read_bytes` and `state_write_bytes` are the recurrent-state subset of logical traffic.
  With `S=4*128*128*Hv` bytes per row/layer, running reads/writes S once, independent of T.
  Batch update reads/writes B*S, record reads B*S without writing state, and fold reads/writes
  L*A*S, where A is the number of nonzero commit rows. Neither T nor chunk count multiplies S.
- `record_read_bytes` and `record_write_bytes` are the raw recurrence/convolution-record subset.
  Record counts valid input/record prefixes and the full output including zeroed suffixes.
  Fold reads only committed K/V/gates and the final required convolution-history columns;
  zero-commit rows contribute no state or record bytes. Convolution history also contributes
  to logical traffic, separately from the recurrent-state subset.
- `workspace_read_request_bytes` and `workspace_write_request_bytes` count intermediate payload
  accesses, including repeated packet reads. For C=ceil(T/16), current chunked prepare writes
  `C*(8192*Hq+2304*Hv)` bytes; recurrence reads `C*Hv*(128/value_tile)*(8192+2304)` bytes.
  Full packets include tail padding. Capacity/alignment alone is not an access count.
- `modeled_request_bytes` includes repeated source Q/K reads and intermediate traffic for the
  chunked implementation. It is absent for recurrent/Replay and composed calls, which do not
  maintain a complete request model; unique logical bytes are never substituted for it.
- `active_rows` counts rows with work; `valid_tokens` sums their valid/committed lengths, without
  multiplying by layers. Physical T/B and L remain separate columns.

`logical_GBps` and `modeled_request_GBps` divide their respective bytes by the complete measured
body's median time. Neither is measured DRAM bandwidth. `memory_roofline_GBps` is the target's
nominal memory bandwidth; `streaming_memory_reference_us` is logical bytes divided by that rate.
It assumes streaming reads and writes and is not a lower bound for cache-resident execution.
Cold samples evict before the timed body; they do not force its output writes back from L2 to
DRAM before the stop event. Logical bandwidth can therefore exceed nominal DRAM bandwidth even
in a cold row, particularly for an in-place state update that fits in L2.
The calibrated read-only bandwidth is not used as a mixed-read/write reference. Cache reuse
within a full chunked pipeline can also serve workspace requests without a DRAM round trip.

```bash
./build/rtx5090-dev/bench/ninfer_gdn_replay_bench \
  --op fold --layers 36 --value-heads 48 --conv-channels 10240 \
  --tokens 16 --batch 8 --commits mixed --execution both
./build/rtx5090-dev/bench/ninfer_gdn_replay_bench \
  --op record --value-heads 48 --tokens 2,8,16 --batch 8 --valid mixed
```

## PLE gate, convolution and state extraction

`ninfer_ple_bench` measures `--op gate|conv-add|conv-add-norm|state-extract` at H=2560,
four streams, dilation=3 and nine history columns. It uses the shared Op measurement harness
described above, with zero global workspace. `--valid` and `--accepted` accept `dense`, `mixed`,
or comma-separated per-row counts; zero counts are supported. `--offset` selects the Gate/Conv
record window. `--inplace` updates the residual or selected history slots and restores the input
before each measurement. CSV includes the actual counts, record offset and alias mode.
Gate consumes slices of one packed BF16 `[12800,T,B]` K/V parent and writes U plus N records.
It has no in-place form. Its unique activation traffic is 85 KiB per active token, plus U zero
writes for inactive tokens; the three norm vectors total 60 KiB and are reused across tokens.
This logical-byte count is not a measurement of DRAM traffic or cache reuse.
CSV reports `logical_bytes`, `logical_gb_s` and `bandwidth_reference_us`, using the target's
memory bandwidth reference. Conv counts unique active inputs, required history columns, weights
and both outputs; inactive columns count only zero writes. Extract counts nine columns read and
written per row, except zero-acceptance rows whose destination already aliases their source.
Device count/index reads are included; repeated reads and private staging traffic are excluded.

```bash
./build/rtx5090-dev/bench/ninfer_ple_bench --op gate --tokens 1,16,256,1024,2048,4096 --cache both
./build/rtx5090-dev/bench/ninfer_ple_bench --op gate --tokens 5,16 --batch 8 --valid mixed --offset 5
./build/rtx5090-dev/bench/ninfer_ple_bench --op conv-add-norm --tokens 1,16,1024,2048,4096 --cache cold
./build/rtx5090-dev/bench/ninfer_ple_bench --op conv-add --tokens 16 --batch 8 --valid mixed --offset 5
./build/rtx5090-dev/bench/ninfer_ple_bench --op state-extract --tokens 16 --batch 8 \
  --accepted 0,1,2,3,8,9,15,16 --inplace --execution both
```

## Softmax Attention Op benchmarks

The four Softmax Attention executables share one harness rule: fixture setup, public workspace
capacity queries, L2 conditioning, graph capture, and synchronization stay outside the timed
interval. An eager interval contains one public Op call; a captured graph contains that same one
public call. `--profile` likewise brackets one complete public call and requires one exact semantic
point. There are no private headers, launchers, route labels, candidate controls, tile sizes, split
counts, or kernel-name filters in these benchmarks.

`ninfer_causal_softmax_attention_bench` measures the two public causal-cache entries:
append-and-attend and cached-only. It covers the registered D256 H24/KV4 and H16/KV2 geometries
with BF16, INT8-G64, FP8-E4M3FN-row256, NVFP4-G16, and K8V4 KV storage. Production dispatch
receives the caller-visible execution envelope and owns prefill, decode/spec and work-partition
choices. `all` emits every storage mode as an independent row.

Append-and-attend accepts `--batch 1,2,4,8`; each ordinary `--context L` point gives every row the
same context and all `W` columns are valid. One exact mixed profile uses `--row-contexts`,
`--valid-columns`, and `--table-rows`, each with exactly `B` entries. Cached-only remains B=1.
The timed call consumes the whole batch once; metadata copies and graph capture remain outside the
interval. Uniform full-width profiles use the dense public contract; exact partial profiles use
device-resident valid extents. Q/K/V contain nonuniform finite values; initial cache rows are encoded
by the public KV append Op before timing. Reported useful bytes/FLOPs sum only valid row work.
`graph_nodes` counts the complete capture. `workspace_bytes` is the per-Op public capacity query
and `workspace_peak_bytes` is the observed peak; the arena is empty after each call. For short warm
Ops, `--execution graph --cache warm --graph-calls 32` captures 32 consecutive calls to reduce CPU
submission gaps. Reported times are normalized per Op, and the CSV records `graph_calls`. Repeated
append calls overwrite the same positions with identical values; this is a warm Op measurement,
not 32 speculative rounds. Cold measurements require one call per graph.

`--envelope-max N` uses a fixed `[1,N]` execution envelope while actual contexts vary; N must
cover every visible row. The default uses the exact visible length. CSV rows record both bounds,
so broad Graph-envelope measurements can be distinguished from exact-length measurements.

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_causal_softmax_attention_bench
./build/rtx5090-dev/bench/ninfer_causal_softmax_attention_bench \
  --entry append --geometry all --kv-dtype all --batch 1,2,4,8 \
  --tokens 1,2,4,8,16 --context 8192,32768,131072 --mapping fragmented \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/rtx5090-dev/bench/ninfer_causal_softmax_attention_bench \
  --entry append --geometry all --kv-dtype all --batch 1 \
  --tokens 1024 --context 8192,32768,131072 --mapping fragmented \
  --execution eager --cache cold --warmup 10 --repeat 61
./build/rtx5090-dev/bench/ninfer_causal_softmax_attention_bench \
  --entry append --geometry d256-h16-kv2 --kv-dtype int8 \
  --batch 3 --tokens 6 --row-contexts 127,2047,63 \
  --valid-columns 6,3,0 --table-rows 2,0,1 \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/rtx5090-dev/bench/ninfer_causal_softmax_attention_bench \
  --entry cached --geometry d256-h16-kv2 --kv-dtype int8 \
  --tokens 16 --context 8192 --execution graph --cache cold --profile
```

The report exposes separate QK/PV logical FLOPs, their full-public-Op-equivalent TFLOP/s,
`key_vector_bytes`/`value_vector_bytes`, and `physical_cache_bytes`. Persistent K+V bytes per D256
vector are 516 for FP8, 288 for NVFP4, and 402 for K8V4 (258-byte K plus 144-byte V).
`unique_kv_bytes` counts each visible persistent KV vector once; `unique_kv_gbps` divides it by
complete Op latency. Payload rates exclude repeated reads and do not measure DRAM bandwidth.
Logical FLOPs do not model private operand conversion, padding, or additional quantization work;
the benchmark therefore does not infer Tensor Core utilization from a storage-format label.

Use ordinary benchmark latency `t`, not profiler replay duration, for effective roofline fractions:
`unique_kv_bytes / (t * peak_bandwidth)` and
`(qk_flops / peak_qk_ops + pv_flops / peak_pv_ops) / t`, with consistent units and explicitly stated
hardware peaks. Current INT8 QK uses native INT8, FP8/K8V4 QK uses unit-scale MXFP8
(838 TFLOP/s at the RTX 5090 reference boost clock), and BF16/NVFP4 QK
and all PV use 16-bit Tensor Core throughput references. In particular, NVFP4 KV storage does not
imply native FP4 QK. These useful-work fractions differ from profiler pipeline activity and actual
DRAM traffic; use targeted profiling to explain the remaining gap.

`ninfer_context_softmax_attention_bench` measures the public read-only context-plus-query contract
at Q32/KV8/D128 with BF16 context storage. `T` is a complete non-causal query block and `L` is its
external context length.

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_context_softmax_attention_bench
./build/rtx5090-dev/bench/ninfer_context_softmax_attention_bench \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 0,2048,8192,32768,131072,196608,262144 \
  --execution graph --cache cold --warmup 10 --repeat 61
```

`ninfer_sliding_window_attention_bench` measures the public Q32/KV8/D128 symmetric sliding-window
contract over a 2048- or 4096-slot cyclic BF16-K/FP16-V cache and a complete non-causal query
block with nonzero represented K/V values. It reports Graph node count, workspace, and exact batch
shape. `--envelope-max N` holds the
Graph resource envelope fixed while the actual context varies. `--graph-calls 32 --execution graph`
measures 32 public calls in one Graph and reports latency per call; this removes host submission
gaps from short warm-cache measurements. For a cold bundle, L2 is flushed once before the bundle,
so only its first call starts cold. These rows are labeled `cold-before-graph-bundle`; warm and
bundled rows omit the DRAM roofline fraction. Profiling uses one public call (`--graph-calls 1`).

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_sliding_window_attention_bench
./build/rtx5090-dev/bench/ninfer_sliding_window_attention_bench \
  --window 2048 --tokens 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --batches 1,2,3,4,5,6,7,8 \
  --context 64,96,97,128,2047,2048,262144 \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/rtx5090-dev/bench/ninfer_sliding_window_attention_bench \
  --window 4096 \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --batches 1 \
  --context 0,32,64,96,128,4095,4096,8192,262144 \
  --execution graph --cache cold --warmup 10 --repeat 61
```

`ninfer_packed_softmax_attention_bench` measures both public dense Attention overloads: a uniform
plain-segment entry and a packed entry driven by cumulative segment lengths. Equal-length inputs
can compare both public entries; nonuniform inputs select only `packed`.

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_packed_softmax_attention_bench
./build/rtx5090-dev/bench/ninfer_packed_softmax_attention_bench \
  --entry both --segments 16 --length 256 \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/rtx5090-dev/bench/ninfer_packed_softmax_attention_bench \
  --entry packed --segment-lengths 128,256,384,512 \
  --execution graph --cache cold --warmup 10 --repeat 61
```

The bandwidth and FLOP fields are semantic useful-work models. They do not infer private scratch
traffic, launch decomposition, or selected implementation from kernel names; physical traffic and
instruction utilization require a profiler capture of the complete public call.

## KV cache append Op benchmark

`ninfer_kv_cache_append_bench` unifies the two public append contracts without combining them in
one timed body. `--mode full` calls full D256 KV publication for KV4/KV2 and BF16, INT8-G64,
FP8-E4M3FN-row256, NVFP4-G16, or K8V4 caches. Its report keeps key/value vector bytes separate for
asymmetric storage profiles. `--mode prefix` calls device-count prefix publication for BF16
D128/KV8 paged caches or cyclic caches with `--cyclic-capacity 2048|4096`. `T` is the physical input
width and `C` is the actual device commit count. `--max-count N` selects the host envelope upper
bound (default T), independently of C. Each default measurement contains one public call;
`--graph-calls 32 --execution graph --cache warm` captures 32 calls and reports latency per call,
with the total Graph node count. A cold bundle flushes L2 only before its first call. Profiling
uses one public call.

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_kv_cache_append_bench
./build/rtx5090-dev/bench/ninfer_kv_cache_append_bench \
  --mode full --full-geometry all --kv-dtype all --tokens 1,2,4,8,16 \
  --context 128 --execution graph --cache cold --warmup 10 --repeat 61
./build/rtx5090-dev/bench/ninfer_kv_cache_append_bench \
  --mode prefix --tokens 1,2,4,8,16 --counts 0,1,2,4,8,16 \
  --layout all --execution graph --cache cold --warmup 10 --repeat 61
```

Prefix useful traffic is 8192 bytes per committed token, multiplied by the batch size. `C=0`
with a positive envelope still exercises device count handling and reports zero useful bytes.
`--max-count 0 --counts 0` produces no kernels and reports zero GPU time.

## Ragged-prefix preparation Op benchmark

`ninfer_prepare_ragged_prefix_bench` measures the complete public gather/zero-fill operation,
including positions and counts. D=25600 is the DFlash2 feature vector; D=16384 covers existing
DFlash. `--counts full|one|ragged|zero` controls actual per-request prefixes. Zero prefixes still
write the full output tail and metadata. Each ordinary call has one kernel and zero scratch.

```bash
cmake --build build/rtx5090-dev -j --target ninfer_prepare_ragged_prefix_bench
./build/rtx5090-dev/bench/ninfer_prepare_ragged_prefix_bench \
  --rows 25600 --widths 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --batches 1,2,3,4,5,6,7,8 --counts ragged --execution graph --cache cold
./build/rtx5090-dev/bench/ninfer_prepare_ragged_prefix_bench \
  --widths 1,8,16 --batches 1,8 --counts full --cache warm --graph-calls 32
```

CSV includes actual live columns, total Graph nodes, call count, scratch, logical useful bytes,
and per-call median/min/p95. Cold mode flushes 256 MiB before the timed interval; a bundle flushes
only before its first call. `--profile` captures one selected public call and requires one shape.

## Masked-block preparation Op benchmark

`ninfer_prepare_masked_block_bench` measures the public exact I32 anchor/mask transform for every
registered `B=2..16`. Eager and graph modes call the same public contract; cold mode performs the
256 MiB L2 eviction before the timed interval.

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_prepare_masked_block_bench
./build/rtx5090-dev/bench/ninfer_prepare_masked_block_bench \
  --block-sizes 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --execution graph --cache both --warmup 20 --repeat 101
./build/rtx5090-dev/bench/ninfer_prepare_masked_block_bench \
  --block-sizes 16 --execution graph --cache cold --profile
```

Useful traffic is `(2+2B)*4` bytes: two device scalar reads and two complete I32 output writes.

## Q8 LinearSwiGLU Op benchmark

`ninfer_q8_linear_swiglu_bench` measures the public Q8 LinearSwiGLU profiles:
`--problem companion` is `[12288,2048] -> [6144,T]`; `--problem dflash2` is
`[34816,5120] -> [17408,T]`. Every timed call uses production dispatch and writes only the fused
output. Workspace is queried for the requested interval before timing. Private candidate forcing
and the former composed control are removed from this retained tool.

`--execution eager` (default) times the complete public call; `--execution graph` captures that
call and times one replay. Every sample follows a 256 MiB L2 flush. CSV records execution mode,
workspace, and Graph node count; the header records GPU and CUDA runtime. Reported work rates use
the logical weight/input/output bytes and the two projections' mathematical FLOPs. `--profile`
warms up and flushes before enabling CUDA profiling for exactly one complete Op call.

```bash
cmake --build build/rtx5090-dev -j --target ninfer_q8_linear_swiglu_bench
./build/rtx5090-dev/bench/ninfer_q8_linear_swiglu_bench \
  --problem dflash2 --t-sweep 1,8,16,32,40,41,51,52,63,64,65,80,81,88,89,96,97,128,1024 \
  --execution graph --warmup 8 --repeat 60 \
  --csv-out profiles/bench/q8_linear_swiglu.csv
./build/rtx5090-dev/bench/ninfer_q8_linear_swiglu_bench \
  --problem dflash2 --profile --t-sweep 128
```

## Q4 LinearSwiGLU Op benchmark

`ninfer_q4_linear_swiglu_bench` measures the public Q4 `[34816,5120] -> [17408,T]` profile and
queries workspace capacity for the requested aggregate interval.

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_q4_linear_swiglu_bench
./build/rtx5090-dev/bench/ninfer_q4_linear_swiglu_bench \
  --t-sweep 1,2,4,8,16,24,32,40,48 --warmup 10 --repeat 50
```

## NVFP4 LinearSwiGLU Op benchmark

`ninfer_nvfp4_linear_swiglu_bench` measures the public NVFP4
`[34816,5120] -> [17408,T]` profile. Each sample measures the complete production-dispatched call,
including activation quantization and caller-owned workspace when required. The fused epilogue
consumes FP32 gate/up accumulators and writes only the final BF16 output.

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_nvfp4_linear_swiglu_bench
./build/rtx5090-dev/bench/ninfer_nvfp4_linear_swiglu_bench \
  --policy a4 --t-sweep 5,48,49,56,64,65,96,97,128,256,1024 \
  --warmup 10 --repeat 50
```

## FP8 LinearSwiGLU Op benchmark

`ninfer_fp8_linear_swiglu_bench` measures the public row-scaled FP8 `[34816,5120] ->
[17408,T]` profile. `--policy a8` measures the production resolver, including caller-owned
activation workspace and the fused SwiGLU output; `--policy a16` measures the public A16 form.
The Tensor Core percentage uses the RTX 5090 dense MXFP8/FP32-accumulate reference of 838 TFLOP/s
only for extents that the production resolver sends to A8.

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_fp8_linear_swiglu_bench
./build/rtx5090-dev/bench/ninfer_fp8_linear_swiglu_bench \
  --policy a8 \
  --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24,32,40,48,1024 \
  --warmup 5 --repeat 30
```

## Q5 LinearAdd Op benchmark

`ninfer_q5_linear_add_bench` measures the public Q5 `[5120,6144]` and `[5120,17408]`
LinearAdd profiles. Every cold-cache sample is one production-dispatched public call, with the
workspace capacity queried for the requested interval.

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_q5_linear_add_bench
./build/rtx5090-dev/bench/ninfer_q5_linear_add_bench \
  --k 6144 --t-sweep 1,2,4,8,16,24,32,48,49,56,64,192,193 --warmup 10 --repeat 50
./build/rtx5090-dev/bench/ninfer_q5_linear_add_bench \
  --k 17408 --t-sweep 1,2,4,8,16,24,32,48,49,56,64,192,193 --warmup 10 --repeat 50
```

## BF16 LinearAdd Op benchmark

`ninfer_bf16_linear_add_bench` measures the contiguous BF16 `[5120,6144]` projection with its
in-place BF16 residual epilogue. Each production sample measures one complete public call after
restoring the residual outside the timed region and flushing L2.
Effective bandwidth counts the weight once, the activation once, and the residual read plus write.
`READ_%` and `TC_%` use the selected target's sustained-read and BF16 references. T=1..4 use SIMT
and omit the Tensor Core percentage and combined compute/memory roofline; the memory lower bound
is still reported.

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_bf16_linear_add_bench
./build/rtx5090-dev/bench/ninfer_bf16_linear_add_bench \
  --sweep 1:128:1 --warmup 10 --repeat 50 \
  --csv-out profiles/bench/bf16_linear_add_t1_128.csv
./build/rtx5090-dev/bench/ninfer_bf16_linear_add_bench \
  --t-sweep 1024,1536,2048 --warmup 10 --repeat 50
./build/rtx5090-dev/bench/ninfer_bf16_linear_add_bench \
  --t-sweep 1024 --profile
```

## Q8 LinearAdd Op benchmark

`ninfer_q8_linear_add_bench` measures the Q8 `[2048,4096]` and `[2048,6144]` projections with their
BF16 residual epilogue. Each sample calls the public Op, which updates the residual in place and
uses no workspace. Every cold-cache sample follows a 256 MiB L2 flush.

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_q8_linear_add_bench
./build/rtx5090-dev/bench/ninfer_q8_linear_add_bench \
  --k 4096 \
  --t-sweep 1,2,4,8,16,32,48,64,96,128 --warmup 10 --repeat 50
./build/rtx5090-dev/bench/ninfer_q8_linear_add_bench \
  --k 6144 \
  --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,33,64,65,96,128,256,640,641,896,960,1024,1280,2048 \
  --warmup 10 --repeat 50 --csv-out profiles/bench/q8_linear_add.csv
./build/rtx5090-dev/bench/ninfer_q8_linear_add_bench \
  --t-sweep "$(seq -s, 1 2048)" \
  --warmup 3 --repeat 15 --csv-out profiles/bench/q8_linear_add_all_t.csv
./build/rtx5090-dev/bench/ninfer_q8_linear_add_bench \
  --profile --t-sweep 1024
```

## FP8 LinearAdd Op benchmark

`ninfer_fp8_linear_add_bench` measures the two public row-scaled FP8 LinearAdd registrations,
including activation quantization, caller-owned workspace, contraction, residual read, and final
in-place BF16 write. `--policy a8` follows the independent production resolver of the selected
semantic Op: `[5120,6144]` uses A16 below `T=22`, while `[5120,17408]` uses A16 below `T=25`; larger
extents use FP8/FP32-accumulate Tensor Core contraction. `TC_%` is reported only when that A8 route
actually executes, against the RTX 5090 unit-scale MXFP8/FP32-accumulate 838 TFLOP/s reference.

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_fp8_linear_add_bench
./build/rtx5090-dev/bench/ninfer_fp8_linear_add_bench \
  --k 6144 --policy a8 --t-sweep 1,2,4,8,16,20,21,22,32,48,1024 \
  --warmup 5 --repeat 30
./build/rtx5090-dev/bench/ninfer_fp8_linear_add_bench \
  --k 17408 --policy a8 --t-sweep 1,2,4,8,16,24,25,32,48,1024 \
  --warmup 5 --repeat 30
```

## LinearSplit benchmarks

`ninfer_linear_split_multi_output_bench` measures the three/four-output forms of public
`linear_split()`: Q4/Q5 `[7168+7168,5120]`; Q8 `[9216,2048]`, `[6144,2048]` and
`[6144,5120]`; and BF16, NVFP4 and FP8 `[14336,5120]`.
Fixture packing and public workspace capacity queries happen
before timing. Every sample and profiler range contains exactly one public Op call, so production
owns format-specific dispatch and launch decomposition.
`--execution eager` (default) times the public call; `--execution graph` captures that call and
times one replay. Profiling follows the same selection. CSV and terminal rows identify execution
mode and captured node count; the terminal header records the GPU and CUDA runtime. Cold samples
flush 256 MiB before the complete Op; warm samples reuse the same buffers.

Q4/Q5 and FP8 target fixtures initialize nonuniform valid signed codes, stored scales and BF16
activations. `workspace_bytes` is the exact query for the displayed T, `workspace_peak_bytes` is
the arena peak observed during the public call/capture, and `workspace_capacity_bytes` is the
allocation reserved for the entire requested sweep. Logical GB/s counts semantic tensor/weight
bytes; it is not a measured DRAM utilization percentage.

```bash
cmake --build build/rtx5090-dev -j --target ninfer_linear_split_multi_output_bench
./build/rtx5090-dev/bench/ninfer_linear_split_multi_output_bench \
  --format all --tokens 1,2,4,8,12,16,32,64,128,256,512,1024 \
  --cache cold --warmup 10 --repeat 50 \
  --csv-out profiles/bench/linear_split.csv
./build/rtx5090-dev/bench/ninfer_linear_split_multi_output_bench \
  --format nvfp4 --nvfp4-policy a4 --tokens 1024 \
  --cache cold --warmup 10 --profile
./build/rtx5090-dev/bench/ninfer_linear_split_multi_output_bench \
  --format fp8 --fp8-policy a8 --tokens 1,5,32,64,65,96,128 \
  --cache cold --warmup 10 --repeat 50
./build/rtx5090-dev/bench/ninfer_linear_split_multi_output_bench \
  --format q8-dflash2-qkv --tokens 8,16,32,53,54,64,65,96,128,1024 \
  --execution graph --cache cold --warmup 10 --repeat 50
```

`ninfer_linear_split_two_output_bench` covers Q8 `[12288,2048]` and `[16384,2560]`,
FP8/NVFP4 `[16384,5120]`, and paired Q4/Q5 `[4096+12288,5120]`. It uses the common
`op_measurement.cuh` options described above; `--n` is the total logical row count.

```bash
./build/rtx5090-dev/bench/ninfer_linear_split_two_output_bench \
  --format q8 --n 16384 --k 2560 --tokens 1,16,1024,2048,4096 --cache cold
```

Stateful projection/convolution remains in `ninfer_gdn_input_proj_bench`.

## QSA Op benchmark

`ninfer_qsa_bench --op input-proj` (the default) measures public A16 `linear_split()` on the three
QSA projection geometries alongside the corresponding contiguous-output `linear()` calls.
`--projection qg|kv-index|qkgv-index|all` selects Q8 `[12288,2560]`,
BF16 `[1664,2560]`, or BF16 `[13952,2560]`. `--entry split|linear|both` selects the public entries;
`both` reuses identical weights, inputs, and total output storage and alternates entry order by round.
The split entry writes each contiguous segment directly and uses zero global workspace.

Timing uses `op_measurement.cuh`: Graph body and eager latency, cold/warm cache, Host submission,
and actual Graph kernel count. Cold measurements evict 256 MiB before timing. CSV rows include the
projection and round; fixture creation is outside timing.

```bash
cmake --build build/rtx5090-dev -j --target ninfer_qsa_bench
./build/rtx5090-dev/bench/ninfer_qsa_bench --projection all --entry both --sweep 1:128 \
  --rounds 3 --warmup 10 --repeat 100 --execution graph --cache both \
  --csv-out out/bench/qsa_small.csv
./build/rtx5090-dev/bench/ninfer_qsa_bench --tokens 256,512,1024,2048,4096 \
  --rounds 3 --warmup 10 --repeat 100 --execution both --cache both \
  --csv-out out/bench/qsa_prefill.csv
```

`--op tail-extract` measures the complete public `qsa_tail_extract()` call. `--capacity` selects
record capacities R; `--batch`, `--base-positions`, `--valid`, and `--accepted` specify the actual
Device metadata. A single position/count broadcasts to B rows; a list must contain B values.
`--accepted dense` retains the valid prefix, while `mixed` varies retention across rows.
`--layout contiguous|layered|unaligned` selects compact buffers, twelve-layer slab strides, or
naturally aligned BF16 views requiring scalar word copies. `--inplace` uses the source slots as
destinations and restores both source planes outside each timed call. Extraction CSV rows use
`tokens` for R and include base positions, layout, in-place mode and round. There is zero workspace.

```bash
./build/rtx5090-dev/bench/ninfer_qsa_bench --op tail-extract --capacity 16,4096 --batch 8 \
  --base-positions 6 --accepted mixed --layout layered --inplace \
  --rounds 3 --warmup 10 --repeat 100 --execution both --cache both \
  --csv-out out/bench/qsa_tail.csv
```

`--op prepare` measures the complete `qsa_prepare()` call. `--kv-cache bf16|int8|fp8|nvfp4|k8v4`
selects the main-cache profile (default `bf16`). `--tokens` selects T, `--batch` selects B,
and `--axes 1|3` selects current RoPE coordinates. `--base-positions` and
`--valid` set Device metadata; cache pages use a nonsequential mapping. `--layout` controls the
tail/record views. `--inplace` aliases both Q/IQ outputs and restores their original projection
words outside every timed call. CSV rows include positions, axes, alias mode, layout and round.

```bash
./build/rtx5090-dev/bench/ninfer_qsa_bench --op prepare --sweep 1:16 --batch 8 --axes 3 \
  --base-positions 63 --inplace --rounds 3 --cache both \
  --csv-out out/bench/qsa_prepare_small.csv
./build/rtx5090-dev/bench/ninfer_qsa_bench --op prepare --tokens 128,1024,2048,4096 \
  --kv-cache nvfp4 --axes 3 --rounds 3 --execution both --cache both \
  --csv-out out/bench/qsa_prepare_prefill.csv
```

`--op indexer` measures the complete public `qsa_indexer()` call, including paged scoring and
Top512 selection. `--contexts L[,L...]` gives each row's visible original-token count at its last
live query; one value broadcasts to B rows. `--valid` sets live query prefixes. The fixed
`--context-capacity` envelope defaults to 262144 and is independent of actual context lengths.
`--layout` selects contiguous, interleaved-page, or naturally aligned BF16 keys. Page mappings are
nonsequential. CSV includes actual contexts, envelope, layout, caller workspace capacity/peak,
and Graph kernel count. The index plane is always BF16, independent of main KV encoding.

```bash
./build/rtx5090-dev/bench/ninfer_qsa_bench --op indexer --tokens 1,8,64,1024,2048,4096 \
  --contexts 262144 --context-capacity 262144 --rounds 3 --execution graph --cache both \
  --csv-out out/bench/qsa_indexer.csv
./build/rtx5090-dev/bench/ninfer_qsa_bench --op indexer --tokens 8 --batch 8 \
  --contexts 262144,131072,32768,8192,2052,2051,4,0 --valid 8,8,8,8,8,8,4,0 \
  --rounds 3 --execution graph --cache both --csv-out out/bench/qsa_indexer_mixed.csv
```

`--op sparse-gqa` measures the complete public attention and sigmoid gate, including any split
reduction and inverse value-basis transform. `--kv-cache bf16|int8|fp8|nvfp4|k8v4` selects cache
storage; activations remain BF16/FP16. All profiles start from the same represented BF16 K/V
source values, initialized through public `kv_cache_append` outside timing. `--contexts` and
`--valid` have the same meaning as Indexer. Its
`--context-capacity` defaults to the batch's maximum actual context; an explicit wider value
measures that capture envelope. `--selection random|shared|contiguous` selects dispersed independent
sets, a shared shuffled block ranking filtered by each query's causal boundary, or recent adjacent
blocks. `--selected-blocks` defaults to 512 and permits 0..512; incomplete tails are always included.
Physical pages are shuffled, and `--layout` exercises the same three plane layouts.

CSV records the envelope, workspace, node count, separate useful QK/PV FLOPs, summed per-query
selected K/V bytes including scales, union K/V bytes across queries, and allocated physical cache
bytes. Payload rates are not measured DRAM bandwidth; use actual DRAM/L2 traffic to interpret the
roofline.

```bash
./build/rtx5090-dev/bench/ninfer_qsa_bench --op sparse-gqa --tokens 1,8,128,1024,2048,4096 \
  --contexts 262144 --kv-cache nvfp4 --execution graph --cache both \
  --csv-out out/bench/qsa_sparse.csv
./build/rtx5090-dev/bench/ninfer_qsa_bench --op sparse-gqa --tokens 9 --batch 8 \
  --contexts 64,2048,8192,32768,131072,262144,0,65536 --valid 9,9,9,5,9,9,0,9 \
  --context-capacity 262144 --execution graph --cache both
```

## MoE benchmark

`ninfer_moe_bench` calls the public MoE interfaces over the 35B Q4+Q5/Q6 and Q8+Q8 profiles.
`--op complete` measures router, routed/shared SwiGLU and BF16 residual addition; `route` measures
routing only. `routed` and `routed-shared` consume routes prepared outside timing and write FP32.
The latter includes the shared branch. Their `--subset all|mask-ones|mask-zeros|none` selects the
logical expert assignments; the fixture's mask assigns odd IDs to the 1 side.

```bash
./build/rtx5090-dev/bench/ninfer_moe_bench --codec q4-q5 --tokens 1 --execution graph --cache both
./build/rtx5090-dev/bench/ninfer_moe_bench --codec all --sweep 1:64:1 --distribution trace-like \
  --execution graph --cache both --warmup 10 --repeat 80 --csv-out /tmp/moe.csv
./build/rtx5090-dev/bench/ninfer_moe_bench --op routed-shared --codec q4-q5 --tokens 32 \
  --subset mask-ones --execution graph --cache cold
```

Graph replay is the default and authoritative timing mode. External event nodes bracket only the
public Op's device work. Input preparation, residual reset, L2 eviction, Graph setup/prime, Host
submission and synchronization are outside that interval. `eager` measures the same public call.
Cold samples flush 256 MiB; warm samples reuse the working set. `trace-like` provides repeated but
changing experts; `independent` and `same` cover low and full overlap. Production code alone chooses
the implementation.

CSV rows identify the Op, subset, selected assignment count, distribution, cache state and execution
mode. They report median/min/p95 latency, workspace, useful encoded-weight bandwidth and logical
FLOP/s. `unique_weight_gbps` counts each selected expert once, plus the control/shared matrices
actually used by that Op; it is a useful-traffic lower bound, not measured DRAM traffic.
`memory_lower_bound_us` divides these bytes by theoretical device bandwidth. For a measured
roofline, substitute sustained bandwidth and include private reloads, staging traffic and padded
computation when interpreting the gap. Logical FLOP/s counts only the requested matrix products.
The complete call uses empty external read-ahead hints; next-projection cache warming and its
downstream effect are covered by Engine measurements.

`ninfer_moe_block_scaled_bench` measures H2560/I640/E512/top-10 through the same public
interfaces. `--format nvfp4|fp8` selects NVFP4 or block-FP8 weights (default: `nvfp4`). `--operation routed|routed-shared|complete|route` selects the semantic scope; outputs
are BF16 Write. It uses the graph-body timing and 256 MiB cold-cache eviction described above.
`--tokens` is a comma-separated list of total columns (`T*B`); `independent`, `same`, and `mixed`
select uniform distinct top-10 routes, one shared top-10 set, or six shared experts plus four
independently sampled tail experts. All 512 experts have distinct physical backing. Input/router
construction makes the complete entry produce the same selected routes as the routed entry.

```bash
./build/rtx5090-dev/bench/ninfer_moe_block_scaled_bench --format nvfp4 --operation routed \
  --tokens 1,8,16,32,64,128,256,512,1024,2048,4096 --distribution all --cache both
./build/rtx5090-dev/bench/ninfer_moe_block_scaled_bench --format fp8 --operation complete \
  --tokens 1024,2048,4096 --repeat 50 > /tmp/moe-fp8.csv
```

CSV reports median/P95/min, complete workspace, and useful matrix FLOP/s. For routed/complete
calls, `unique_weight_GBps` counts unique routed expert payloads and divisors only; for `route`,
it counts the control matrix. It excludes reloads, activations and output traffic, so it is not
measured DRAM bandwidth. Use profiler traffic when constructing a kernel or complete-Op roofline.

`ninfer_moe_cpu_bench` measures the complete synchronous CPU `routed_swiglu` call, including
validation, grouping, decoding, arithmetic and worker publication/join. Setup and CPU/GPU transfers
are excluded. `--format q4-q5|q4-q6|nvfp4` selects the native format and geometry. Native Host
banks stay alive for the whole run. `repeat` reuses a fixed subset;
`stream` rotates selected experts across prefaulted banks (by default at least four times reported
LLC in actually selected weights). `same`, `skew` and `independent` vary reuse within each call.
Full contributions route across the complete expert domain; partial CPU contributions use one
masked half. NVFP4 supports zero through ten selected CPU experts per token.

```bash
./build/rtx5090-dev/bench/ninfer_moe_cpu_bench --format q4-q5 --workers 15 --caller-cpu 31 \
  --tokens 1,8,72 --active 1,8 --cache stream --distribution same --samples 240 --read-reference
./build/rtx5090-dev/bench/ninfer_moe_cpu_bench --format nvfp4 --workers 16 --caller-cpu 31 \
  --tokens 1,8,72,256 --active 1,4,10 --cache stream --distribution independent --samples 51 --read-reference
```

Output includes min/median/p95/max wall time, useful encoded-weight GiB/s, logical GFLOP/s,
workspace and reuse footprint. Process CPU cost uses a separate interval of at least 250 ms at
the same cadence; `--gap-us` is excluded from call latency and included in that accounting.
`--keep-active` keeps the worker team polling between calls, including those gaps; the default
allows workers to park after their bounded spin interval.
`encoded_read_bytes` models the selected four/two/one-column tiles and eight-token chunks; it
includes repeated weight reads, but is not a hardware traffic counter. `--read-reference` times
the same bank/column grouping and row ownership with decode and arithmetic removed. Its median
is an access-pattern reference, not a strict roofline or an end-to-end offload result.
The tile-read model and matching access-pattern reference apply only to the integer formats;
their CSV fields are `NA` for NVFP4, whose scales use the native swizzle.
That option also reports sustained sequential read GB/s over the complete native-bank corpus,
using the same worker team. Unique encoded bytes divided by this bandwidth give a compulsory-read
memory roofline reference for a streaming call; repeated tile reads may hit cache and must not all
be charged as DRAM bytes. No compute-peak or Engine-throughput claim follows from either reference.

## Q8 context K/V LinearPair benchmark

`ninfer_linear_pair_bench` measures one public `linear_pair()` call over exact adjacent Q8
`[1024,K]` K/V row views of a `[6144,K]` QKV parent, for the registered `K=2048` and `K=5120`
geometries. It uses cold-cache eager execution or CUDA Graph replay, accepts an arbitrary token
list or sweep, and reports route-neutral effective bandwidth, logical FLOP/s, and calculated T=1
linear extrapolation. `--flush-mib` must be positive; the header records the eviction size.

```bash
cmake --build build/rtx5090-dev -j --target ninfer_linear_pair_bench
./build/rtx5090-dev/bench/ninfer_linear_pair_bench \
  --k 2048 --sweep 1:128:1 --execution graph --warmup 5 --repeat 30
./build/rtx5090-dev/bench/ninfer_linear_pair_bench \
  --k 5120 --tokens 8,16,24,32,40,48,56,64 --execution graph --warmup 5 --repeat 30
./build/rtx5090-dev/bench/ninfer_linear_pair_bench \
  --k 5120 --tokens 128,1024,2048 --execution eager --warmup 5 --repeat 30
```

## Five-layer context K/V materialization benchmark

`ninfer_context_kv_materialize_bench` measures the complete capacity-2048 DFlash2 context-cache
state transition over `W=1..16,B=1..8` and single-request prefill `W=1..2048`. It calls the public
Op with five independent `[6144,5120]` QKV-parent K/V row views and caller-owned workspace.
CSV rows report physical geometry, count envelope, whole-Op latency, Graph nodes, and the public
workspace bound and actual arena peak. Every timed sample follows a 256 MiB cache flush; setup and input transfers are
outside the measurement. Zero-count rows verify an empty Graph and report latency as `nan`
(not applicable). `--counts full|one|ragged|zero` selects the device prefixes and their host
envelope. Widths above 16 are measured only for B=1. These are Op measurements, not Engine results.

```bash
cmake --build build/rtx5090-dev -j --target ninfer_context_kv_materialize_bench
./build/rtx5090-dev/bench/ninfer_context_kv_materialize_bench \
  --widths 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --batches 1,2,3,4,5,6,7,8 --counts full --execution graph --warmup 8 --repeat 60
./build/rtx5090-dev/bench/ninfer_context_kv_materialize_bench \
  --widths 8,16,128,2048 --batches 1,8 --counts one --warmup 8 --repeat 60
./build/rtx5090-dev/bench/ninfer_context_kv_materialize_bench \
  --widths 17,85,86,128,512,1024,2048 --batches 1 --warmup 8 --repeat 60
```

## MTP exact-transform benchmark

`ninfer_mtp_pack_bench` calls the public bit-exact pack or attention-split Op once per point:

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_mtp_pack_bench
./build/rtx5090-dev/bench/ninfer_mtp_pack_bench --op pack --d 5120 --tokens 1,2,3,4,5,6,48
./build/rtx5090-dev/bench/ninfer_mtp_pack_bench --op split --tokens 1,2,3,4,5,6,48
```

## Model MTP round benchmark

`ninfer_qwen3_5_mtp_round_bench` measures the native MTP proposal and verification round. It
constructs the same v3 ModelInstance used by Engine, prepares a fixed token seed, and drives the
production Program's decode/commit schedule. It reports round latency and licensed-token counts:

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_qwen3_5_mtp_round_bench
./build/rtx5090-dev/bench/ninfer_qwen3_5_mtp_round_bench \
  --artifact out/qwen3_6_27b.ninfer
```

## 35B complete DFlash round benchmark

`ninfer_qwen3_5_dflash_round_bench` drives the production Program through consecutive
steady DFlash rounds. A measured round includes the previous confirmed feature-to-context append,
the six-layer proposal, target verify/accept, and host publication. It reports GPU and wall latency,
real acceptance, per-position acceptance, mean licensed length, and published tokens/s:

```bash
cmake --build build/rtx5090-dev --parallel \
  --target ninfer_qwen3_5_dflash_round_bench
./build/rtx5090-dev/bench/ninfer_qwen3_5_dflash_round_bench \
  --artifact out/qwen3_6_35b_a3b.ninfer \
  --context 4096 --draft-tokens 15 --proposal-head optimized
```

Run separate invocations for `K=1..15`, `full|optimized`, representative contexts, and
`--no-cuda-graph`. This complete-round benchmark measures the DFlash model's actual greedy
acceptance. Transaction correctness is covered by the state and Engine tests.

## Token-decision Op benchmarks

The G1 benchmark calls public `argmax` for the Qwen3.6-35B full physical vocabulary with 248077
valid rows through `C=128`, and for the 131072-row shortlist through `C=120`. With no arguments it
covers every B=1 full-vocabulary width, both T1 routes, and both aggregate maxima:

```bash
cmake --build build/rtx5090-dev --parallel --target ninfer_argmax_bench ninfer_sampling_select_bench
./build/rtx5090-dev/bench/ninfer_argmax_bench
./build/rtx5090-dev/bench/ninfer_argmax_bench --shape full --cols 128
./build/rtx5090-dev/bench/ninfer_argmax_bench --shape shortlist --cols 120
```

The G2/G3/G4 benchmark uses physical rows 248320 and valid token domain 248077. G2 covers optional
occurrence counts and batched sampling at `B=1,2,4,8`; G3 covers one-hot MTP windows `K=1..5`.
With no arguments it runs the G2/G3 greedy/stochastic matrix. G4 covers DFlash2 sparse-q acceptance
with `K=1..15`, 16 proposal candidates, `P=0..K`, and `B=1..8`. `--drafts` defaults to the
checkpoint recommendation of seven; the default extent is the selected K.

G4 reports the raw-greedy/general route, Graph node count, caller workspace, and whole-Op timings.
A 256 MiB flush precedes each timed Graph; setup and input transfers are outside the measurement.
`--graph-calls N` bundles N calls and reports time divided by N, so later calls reuse cache.
`--mixed` cycles raw greedy, penalty-bearing greedy, and stochastic requests; `--general` selects
the conservative general envelope even for greedy inputs. `--reject-at J` changes one proposal to
exercise rejection; `--extent 0` checks the bonus-only case. These DFlash2 measurements do not
measure Engine inference.

```bash
./build/rtx5090-dev/bench/ninfer_sampling_select_bench --matrix
./build/rtx5090-dev/bench/ninfer_sampling_select_bench --sample --batch 8 --mode stochastic --top-k 20
./build/rtx5090-dev/bench/ninfer_sampling_select_bench --mtp --mode stochastic --mtp-k 5 --top-k 20
./build/rtx5090-dev/bench/ninfer_sampling_select_bench \
  --dflash2 --drafts 15 --batch 8 --mode greedy --graph-calls 32 --warmup 8 --repeat 60
./build/rtx5090-dev/bench/ninfer_sampling_select_bench \
  --dflash2 --drafts 15 --batch 8 --mode stochastic --top-k 20 --warmup 8 --repeat 60
./build/rtx5090-dev/bench/ninfer_sampling_select_bench \
  --dflash2 --drafts 7 --batch 8 --mode stochastic --mixed --extent 3
./build/rtx5090-dev/bench/ninfer_sampling_select_bench \
  --dflash2 --drafts 15 --batch 8 --mode stochastic --no-counts --reject-at 7
```

## 35B dFlash causal Attention qualification

The public causal benchmark covers exact verify widths `W=1..16`, all five KV types, and the
append-and-attend and already-cached entries for the D256 H16/KV2 geometry. Batched target
qualification uses the append entry:

```bash
./build/rtx5090-dev/bench/ninfer_causal_softmax_attention_bench \
  --entry append --geometry d256-h16-kv2 --kv-dtype bf16 \
  --batch 1,2,4,8 \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 128,1024,8192 --execution graph --cache cold
./build/rtx5090-dev/bench/ninfer_causal_softmax_attention_bench \
  --entry cached --geometry d256-h16-kv2 --kv-dtype int8 \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 128,1024,8192 --execution graph --cache cold
```

The former `attention_layer` executable composed multiple implementation-level stages and exposed
private route controls, so it is not retained as a public Op benchmark. Complete mixer and target
effects are measured through the public Engine benchmark or the target round benchmarks.

## Pointwise Op benchmarks

The pointwise benchmarks call their public Ops. Default invocation runs the registered small,
established, maximum-video, and maximum-image shapes:

```bash
cmake --build build/rtx5090-dev --parallel --target \
  ninfer_residual_add_bench ninfer_sigmoid_mul_bench \
  ninfer_gelu_bench ninfer_add_bias_bench

./build/rtx5090-dev/bench/ninfer_residual_add_bench [--patches P]
./build/rtx5090-dev/bench/ninfer_sigmoid_mul_bench \
  [--tokens T[,T...]]
./build/rtx5090-dev/bench/ninfer_position_bench \
  [--tokens T[,T...]] [--cold-graph] [--warmup N] [--repeat N]
./build/rtx5090-dev/bench/ninfer_gelu_bench [--mode tanh|exact --columns C]
./build/rtx5090-dev/bench/ninfer_add_bias_bench [--d D --columns C]
```

Aligned registered shapes use 16-byte BF16 packs in the cache-sized regime. GELU and AddBias
select their BF16x2 streaming routes for larger Vision items; odd or unaligned repository-internal
test shapes exercise the scalar fallbacks.

## Causal conv1d SiLU Op benchmark

`ninfer_causal_conv1d_silu_bench` times one public entry per invocation. `--tokens` takes a list, so
a route decision comes from one process rather than a series of them. `--cache` defaults to warm;
pass `--cache cold` for the state route decisions are made in.
`--execution graph|eager` defaults to Graph body timing; both cache and state-alias forms use the
same measurement harness. `--warmup` and `--repeat` default to 10 and 100. `--inplace` selects the
split entry's exact state alias, restoring its initial history outside every timed call.

```bash
./build/rtx5090-dev/bench/ninfer_causal_conv1d_silu_bench --split --cache cold --channels 8192 --tokens 1,2,7,15,16,17,24,32,33,64,65
./build/rtx5090-dev/bench/ninfer_causal_conv1d_silu_bench --split --cache cold --channels 10240 --tokens 1024,4096,8192
```

`--split` selects the split-output entry. Its partition follows the channel extent, because those
are the row profiles the entry admits: `--channels 8192` gives (2048, 2048, 4096) and
`--channels 10240` gives (2048, 2048, 6144).

## Reports

Table, JSON, and CSV reports identify the architecture, model instance, artifact, Engine configuration,
load summary, memory capacity, KV payload, workspace peak, phase throughput, and speculative
statistics. JSON schema version 24 records the public value objects directly:

- `config`: constraint type, source/file, literal choices and mixed-request selection, alongside execution settings;
- `load`: architecture, public name, actual formats, prefill signature, load/upload time,
  file/H2D/staging bytes and Device/Host object counts;
- `memory`: weights/sequence/unified-workspace arenas, the optional non-additive Vision layout,
  planned context, KV storage, CUDA Graph rebuild estimate and preparation increment, KV payload, Device budget and optional MoE
  Host/cache/bridge resources; MoE `uploaded_bytes` here is the startup snapshot;
- `repetition_wall_seconds`: each concurrent repetition's submit-to-completion interval;
- each request sample's `timings`: prepare, Vision, prefill, decode, and total seconds;
- each request sample's `constraint`: committed completion/branch, compilation cache access, CPU
preparation/mask/matcher seconds, evaluated positions and mask upload bytes; `null` without a constraint;
- `constraint_draft_wait_exposed_seconds`: the sample's exposure to draft-ready wait; shared batch
exposure must not be summed across requests, and CPU constraint work must not be added to wall time;
- each request sample's `speculative`: window, rounds, drafted/accepted tokens, fallbacks, and per-position
acceptance.

Each test reports `workspace_peak_bytes` from the planned phase markers, including CUDA Graph
replay, and `workspace_allocator_peak_bytes` from host-side arena allocation activity. These are
intentionally separate: replay reuses captured addresses without advancing the host allocator.
`expert_upload_bytes` counts expert payload submitted during measured repetitions, excluding
startup and warmup. It includes all requests in those repetitions.

`reps` contains request samples in repetition-major order, with explicit `repetition` and `row`.
`output_tok_s` divides each repetition's actual outputs by its wall time.
`decode_output_tok_s` counts actual outputs after the first token, allowing early grammar EOS.
`decode_engine_tok_s` uses the Program's speculative round statistics, so it also describes work performed by a final partially
committed speculative round. Reports also contain the command and machine information needed to
interpret a local measurement.

Raw reports and profiler captures remain local under `profiles/bench`, `profiles/ncu`, and
`profiles/nsys`.

## Expert replica transport

`ninfer_expert_cache_bench` uses real encoded bank parents through the production loader and cache.
Initialization is outside the transfer samples. It reports five cold-replica samples with warm RAM
sources, min/median/max wall latency, Host planning/submission, upload-stream timeline,
data/index bytes and copy API calls. These calls are not a hardware DMA transaction count.
All-hit Host timing has 101 samples with a real GPU code-plane reader between acquire and release;
its launch and device time are excluded from the Host timing. The H2D roofline reads the same
registered source backing in contiguous chunks, with 128 MiB warmup and at least 1 GiB measured.
Registered Host bytes and registration time are reported separately from steady-state transport.
The scattered samples use a fixed shuffled expert set. These measurements do not measure MoE computation.
The overlap case uses four synthetic GPU reads of the current cached bank while loading the next.

```bash
build/rtx5090-dev/bench/ninfer_expert_cache_bench --artifact out/qwen3_8_flash_next.ninfer \
  --bank text/layers/0/moe/experts --experts 512
build/rtx5090-dev/bench/ninfer_expert_cache_bench --artifact out/qwen3_6_35b_a3b.ninfer \
  --bank text/layers/0/moe/experts --experts 256
```

Layer 34 of the 35B artifact selects its Q6 down bank. `--batch-mib N` changes the maximum payload
per upload batch; default is 96 MiB, with three in-flight slots. Slots contain index snapshots and
descriptors, not weight payloads. `--resident` checks direct GPU plane access without dynamic replicas,
for example Flash Next `--bank mtp/layers/0/moe/experts --experts 512 --resident`.

## Single-layer mixed MoE execution

`ninfer_qwen3_5_moe_execution_bench` materializes one selected 35B layer's native expert parents,
control and shared weights. Its shared MoE execution entry includes route, actual ExpertCache use,
D2H, CPU/GPU contributions, H2D and residual merge. This is a layer microbenchmark, not Engine
generation. `--gpu-experts N` assigns logical IDs `[0,N)` by default; `--gpu-selection shuffled`
selects the first N IDs of a deterministic shuffled permutation. The remaining routed assignments
use CPU.

`--execution eager|graph` selects submission for the production parallel schedule (default eager).
Graph mode captures the GPU front for each fixed corpus input and performs the CPU handoff and
conditional completion outside the graph. Capture/upload are excluded from timing; the serial
comparison remains eager. The header records the selected execution mode.

```bash
build/rtx5090-dev/bench/ninfer_qwen3_5_moe_execution_bench \
  --artifact out/qwen3_6_35b_a3b.ninfer --layer 0 --tokens 128 --gpu-experts 128 \
  --workers 15 --caller-cpu 31 --samples 256 --distribution independent --schedule both
```

Layer 34 selects Q6 down. The 32-input synthetic BF16 Gaussian corpus uses either independent
columns or identical columns within each batch (`--distribution same`). Initial expert upload
is reported separately; measured calls reuse the fixed GPU assignment and must not upload expert data. Source,
cache, frame/scratch capacities, actual assignment counts and unique encoded weight bytes are
reported. Corpus bytes and per-call unique bytes are logical traffic, not measured DRAM traffic.
Bridge H2D bytes are averaged over the corpus; batches with no CPU assignments transfer no CPU
contribution in either schedule. The report includes the number of batches with CPU work.

Timing is Host wall time through final compute-stream completion. Warmup covers at least 100 ms
and the complete corpus. `--schedule parallel` measures the production entry; `serial` is a
standalone combination of the same Ops/transfers/team with GPU experts following CPU work. It
does not include the model entry's validation/frame binding, so its difference is a scheduling
reference, not an isolated measurement of overlap. `both` alternates order, reversing each
input's order on successive corpus passes, and reports min/median/p95/max and paired differences.
`--gap-us N` adds an excluded inter-call gap. Select a CPU affinity allowed on the current machine.
`--keep-active` keeps workers polling throughout warmup and measurement; without it, the team
uses its ordinary bounded spin and park policy. This layer benchmark does not apply the Program's
adaptive waiting policy.

For a timeline, capture this executable with `nsys --trace=cuda,nvtx`; the model marks CPU input
wait and CPU computation separately. `--profile` brackets measured calls with CUDA profiler API
calls when capture by that API is appropriate. Profiled durations do not replace ordinary wall
measurements.
