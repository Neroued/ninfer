# Benchmarks

`ninfer_bench` measures the complete public `ninfer::Engine` route against a `.ninfer` artifact.
The `bench/ops/` `ninfer_<op>_bench` executables measure central public Op contracts while leaving
implementation selection behind those contracts. Target benchmarks measure Program/model
composition. Correctness lives in the affected test suites; development rules are in
[`../docs/maintainer/op-development.md`](../docs/maintainer/op-development.md).

The frozen request corpus for the separate black-box Serve TTFT tool is documented under
[`fixtures/ttft/`](fixtures/ttft/README.md). That client does not call the benchmark executables or
Engine directly.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build --parallel --target ninfer_bench
```

## Product benchmark

The benchmark slices exact token counts from `bench/fixtures/bench_corpus.ids`, calls
`Engine::prepare_tokens()`, then calls `Engine::generate()` once for each repetition. It does not
have a private prefill/decode loop and does not call target implementation interfaces.

The matrix contains three independently measured test kinds:

- `pp{P}` prepares `P` tokens and requests one output token. This is the smallest request that runs
  the model; `prefill t/s` is `P / GenerationTimings.prefill_seconds`.
- `tg{G}` prepares a one-token seed outside the reported phase and requests `G+1` output tokens.
  The begin-round token belongs to prefill, leaving exactly `G` tokens in the reported decode
  phase.
- `pp{P}+tg{G}` uses the same `G+1` convention after a `P`-token prefill and reports both phase
  rates from the same generation call.

All benchmark requests use raw output, disable model-default stops, and disable prefix reuse. This
keeps the requested token count exact without adding another generation path. When CUDA Graph is
enabled and the matrix contains decode work, one ordinary public generation request primes the
decode graph before warmups and measured repetitions.

## CLI

```text
ninfer_bench --weights <artifact.ninfer>
          [--corpus <ids-path>]
          [-p, --n-prompt <list>]
          [-n, --n-gen <list>]
          [-pg, --prompt-gen <P,G;P,G...>]
          [-r, --repetitions <n>] [--warmup <n>]
          [--max-ctx <tokens>] [--prefill-chunk <tokens>]
          [--kv-dtype <bf16|int8|fp8|nvfp4|k8v4>]
          [--spec <mtp|dflash|dflash2> --draft-tokens <n>] [--lm-head-draft]
          [--device <id>] [--no-cuda-graph] [--profile-measured]
          [-o, --output <table|json|csv>] [--output-file <path>]
```

With no `-p`, `-n`, or `-pg`, the matrix is `pp512` and `tg128`.

Example:

```bash
./build/bench/ninfer_bench \
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
./build/bench/ninfer_bench --weights out/qwen3_8_27b_nvfp4.ninfer \
  -pg '2048,128' --spec dflash2 --draft-tokens 7 --lm-head-draft \
  --max-ctx 4096 --kv-dtype bf16 --warmup 1 -r 3
```

The benchmark disables context retention because every repetition is an independent root request.
Schema v14 records `speculative_backend`, `draft_tokens`, and the proposal head independently;
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
  and artifact `model_id/weights_id`.

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
cmake --build build -j --target ninfer_context_cost_bench
./build/bench/ninfer_context_cost_bench \
  --suite transfer \
  --json profiles/bench/context_cost_transfer.json \
  --preset-out profiles/bench/context_cost_presets.json
```

Measure prefill separately when the GPU has room for the artifact:

```bash
./build/bench/ninfer_context_cost_bench \
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
artifact entry and preserves machine transfer plus other artifacts. The resulting file is parsed by
the runtime's strict loader before publication.

[`context_cost_defaults.cpp`](../src/runtime/engine/context_cost_defaults.cpp) is the sole table of
defaults compiled into the binary. JSON is not a build input: `--preset-out` produces a runtime
registry that can be selected immediately, without recompilation, through
`EngineOptions.context_cost.preset_path` or `ninfer-serve --context-cost-presets`. Resolution always
starts with generic numerical coefficients,
then independently applies matching compiled transfer/prefill values, then independently applies
matching external values. A malformed explicit file is an error; a missing hardware or artifact
entry retains the preceding numerical layer. After real-scenario acceptance, a maintainer may
promote quantized values into the C++ table. The detailed `--json` report is diagnostic provenance
and is not a runtime input.

## Linear Op benchmark

`ninfer_linear_bench` measures only the public pure `linear()` contract. It supports Q4, Q5, Q6,
W8, registered BF16 weights, the registered NVFP4 problems, and the registered FP8 problems.
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
cmake --build build --parallel --target ninfer_linear_bench
./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 4096 --k 5120 --t 8
./build/bench/ninfer_linear_bench \
  --qtype nvfp4 --policy a4 --n 14336 --k 5120 --t 1024
./build/bench/ninfer_linear_bench \
  --qtype fp8 --policy a8 --n 14336 --k 5120 --t 1
./build/bench/ninfer_linear_bench \
  --qtype fp8 --policy a8 --n 16384 --k 5120 --t 1024
```

A continuous small-T sweep reuses one packed weight and one maximum-T activation/output
allocation:

```bash
./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 4096 --k 5120 \
  --sweep 1:32:1 --csv-out profiles/bench/q4_4096x5120_t1_32.csv
./build/bench/ninfer_linear_bench \
  --qtype q4 --policy a16 --n 3456 --k 1152 \
  --sweep 4:512:4 --csv-out profiles/bench/q4_vision_qkv.csv
./build/bench/ninfer_linear_bench \
  --qtype w8 --policy a16 --n 248320 --k 5120 \
  --sweep 48:65:1 --warmup 5 --repeat 30
```

The registered suites run representative public Linear shapes for one or both exact products.
They are compact performance surveys, not copies of the production selector or numerical test
matrix:

```bash
./build/bench/ninfer_linear_bench --suite qwen3_6_27b
./build/bench/ninfer_linear_bench --suite qwen3_6_35b_a3b
./build/bench/ninfer_linear_bench --suite all
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
  ./build/bench/ninfer_linear_bench \
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

Every ordinary sample is cold-cache: a 256 MiB L2 eviction write completes before the timed
interval. Reported effective bandwidth uses the encoded weight planes once, one BF16 activation
read, and one BF16 output write. Reported FLOPs are the mathematical `2*N*K*T`; neither metric
copies route-private tile, replay, padding, split, schedule, host-launcher, or kernel-instance
behavior. The fixed RTX 5090 memory reference is `1792 GB/s` DRAM bandwidth. Because `AllowA4` is
a permission rather than an execution-profile label, the long-lived benchmark does not infer or
report private activation compute or Tensor Core utilization. `READ_%` additionally compares the same one-read model
bytes with the measured `1674.5 GB/s` pure-read ceiling from `tools/hbm_bandwidth_probe.cu`; it is
the practical utilization measure for read-dominated points. Physical traffic and instruction
utilization still require NCU.

## LinearTopK Op benchmark

`ninfer_linear_topk_bench` measures the complete public projection-plus-stable-top-16 Op for the
W8 and row-FP8 full heads and the mapped Q4 optimized head. Its independent matrix-column axis is
`U`; the default sweep covers every `U=1..120` needed by variable-width DFlash2 proposals. A fixed
positive `--columns U` also exercises larger matrices; `--columns U,...` selects a representative
set without a full sweep. Codes and stored scales vary across the physical head, including signs,
so ranking is not timed only on identical weight rows. Each sample replays the complete Op in a
CUDA Graph after a 256 MiB L2 eviction write; projection, partial-key reduction, workspace counter
initialization, and final ids/scores publication are included. Defaults are 8 warmups and 60 samples.
Reported logical bandwidth counts the encoded head once, useful FLOPs count candidate-eligible rows, and
workspace bytes and Graph nodes describe the actual public call. These are Op measurements.

```bash
cmake --build build -j --target ninfer_linear_topk_bench
./build/bench/ninfer_linear_topk_bench
./build/bench/ninfer_linear_topk_bench --profile w8-full --columns 7
./build/bench/ninfer_linear_topk_bench --columns 1,7,16,24,64,120 --repeat 25
./build/bench/ninfer_linear_topk_bench --profile fp8-full --columns 120
./build/bench/ninfer_linear_topk_bench --profile q4-optimized --columns 129 --repeat 60
```

## CandidateSelectorPath Op benchmark

`ninfer_candidate_selector_bench` measures the complete public conditional selector for
`K=1..15`, `B=1..8`, 16 candidates and rank 256, with full BF16 codebooks `[256,248320]`.
The default sweep covers all K/B pairs in greedy, stochastic and mixed modes (B=1 omits the
redundant mixed case). Each cold-cache CUDA Graph sample follows a 256 MiB L2 eviction write.
It reports the selected route's required caller workspace and actual Graph node count. The timed
interval includes all device work of the complete public Op; fixture allocation and setup are outside it.

```bash
cmake --build build -j --target ninfer_candidate_selector_bench
./build/bench/ninfer_candidate_selector_bench --warmup 8 --repeat 60
./build/bench/ninfer_candidate_selector_bench --steps 15 --batch 8 --repeat 60
```

## Embedding Op benchmark

`ninfer_embedding_bench` measures the public quantized embedding profiles: Q6 `[248320,5120]`,
W8 `[248320,5120]`, W8 `[248320,2048]`, and row-scaled FP8 `[248320,5120]`. The default sweep is
T=1..128; `--tokens` selects other matrix extents. This is an Op column domain, independent of a
particular speculative algorithm's draft count.

The fixture initializes nonuniform valid code planes and positive stored scales at the complete
registered table shape. `--id-pattern normal` uses tokenizer-addressable IDs. `masked` generates
an anchor followed by masks at the explicit `--block-width` period: 248070 for D=5120 and 248077
for existing W8/D=2048. Use T=B*W to measure complete proposal blocks. CSV reports unique IDs,
logical per-token bytes, execution/cache mode, Graph nodes/calls, scratch, and median/min/p95.

Each default interval contains one public Op. Cold mode evicts 256 MiB of L2 before timing;
`--graph-calls 32 --execution graph --cache warm` measures repeated public calls in one Graph and
normalizes per call. A cold bundle flushes only before its first call. `--profile` uses one public
call of one format and one extent for kernel profiling.

```bash
cmake --build build -j --target ninfer_embedding_bench
./build/bench/ninfer_embedding_bench --format w8-d5120 --execution graph --cache cold
./build/bench/ninfer_embedding_bench --format fp8-d5120 \
  --tokens 8,16,24,32,40,48,56,64 --id-pattern masked --block-width 8
./build/bench/ninfer_embedding_bench --format fp8-d5120 \
  --tokens 16,32,48,64,80,96,112,128 --id-pattern masked --block-width 16 \
  --cache warm --graph-calls 32 --csv-out /tmp/embedding.csv
./build/bench/ninfer_embedding_bench --format fp8-d5120 --tokens 128 --profile
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
cmake --build build -j --target ninfer_scatter_bf16_batch_bench
./build/bench/ninfer_scatter_bf16_batch_bench --widths 1,2,8,9,16 --batches 1,8 --counts full
./build/bench/ninfer_scatter_bf16_batch_bench --widths 2,8,16 --batches 1,8 \
  --counts ragged --cache warm --graph-calls 32 --csv-out /tmp/feature-scatter.csv
./build/bench/ninfer_scatter_bf16_batch_bench --widths 16 --batches 8 --profile
```

## RMSNorm Op benchmark

`ninfer_rmsnorm_bench` measures public RMSNorm, with `--kind dflash2_hidden` for plain D=5120,
`hidden27` for offset D=5120, and `target_q27` / `target_k27` for offset D=256 with 24 / 4 heads.
The existing DFlash, 35B target and GatedRMSNorm profiles remain selectable in `--help`.
`--tokens` supplies aggregate matrix columns, independently of the speculative block width.
Inputs and gains are nonuniform represented BF16 values. CSV reports actual geometry, Graph
nodes/calls, zero scratch, logical bytes and median/min/p95 public-call latency.

Cold mode flushes 256 MiB before each measured interval. Warm Graph bundles avoid host launch
gaps and normalize time per public call; a cold bundle flushes only before its first call.
`--profile` brackets one public call with CUDA profiler start/stop.

```bash
cmake --build build -j --target ninfer_rmsnorm_bench
./build/bench/ninfer_rmsnorm_bench --kind dflash2_hidden --tokens 1,8,16,32,64,96,128,2048
./build/bench/ninfer_rmsnorm_bench --kind target_q27 --tokens 1,8,16,32,64,96,128 \
  --cache warm --graph-calls 32 --csv-out /tmp/rmsnorm.csv
./build/bench/ninfer_rmsnorm_bench --kind hidden27 --tokens 128 --profile
```

## GDN control-projection Op benchmark

`ninfer_gdn_gating_proj_bench` measures the complete public `gdn_norm_gating_proj` (`--op norm`,
default) or `gdn_gating_proj` (`--op control`). `--geometry 27b` uses D5120/H48 and defaults to the
Qwen3.8 contiguous BF16 `[96,5120]` parent; `--weights split` measures the two-weight form.
`--geometry 35b` uses the D2048/H32 parent. All numerical inputs are nonuniform represented values.

The benchmark reports eager/Graph latency, graph call/node counts, and the public point workspace
query against its measured peak. Cold runs flush 256 MiB before each call. For short warm Ops,
`--graph-calls 16` captures consecutive calls and normalizes times per Op; it requires Graph/warm.
`--profile` brackets one selected workload with CUDA profiler APIs. Production owns all dispatch.

```bash
cmake --build build -j --target ninfer_gdn_gating_proj_bench
./build/bench/ninfer_gdn_gating_proj_bench \
  --geometry 27b --op norm --weights parent --tokens 1,2,4,8,9,16,32,64,128 \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/bench/ninfer_gdn_gating_proj_bench \
  --geometry 27b --op norm --tokens 1,2,4,8,9,16,32,64,128 \
  --execution graph --cache warm --graph-calls 16 --csv-out /tmp/gdn-norm-warm.csv
./build/bench/ninfer_gdn_gating_proj_bench \
  --geometry 35b --op norm --tokens 16 --execution graph --cache cold --profile
```

## Dynamic grouped-convolution prepare Op benchmark

`ninfer_dynamic_grouped_conv_prepare_bench` measures the complete BF16
`rmsnorm_dynamic_grouped_conv_prepare` contract at every registered `B=1..8`. Each timed call
includes RMSNorm, the `[1280,5120]` coefficient projection, the input-side two-tap grouped
convolution, and both public output writes. Every sample follows a 256 MiB L2 eviction; the report
contains complete-Op latency, useful coefficient-projection TFLOP/s, and exact workspace capacity.

```bash
cmake --build build -j --target ninfer_dynamic_grouped_conv_prepare_bench
./build/bench/ninfer_dynamic_grouped_conv_prepare_bench --warmup 10 --repeat 80
```

## Gated DeltaNet Op benchmark

`ninfer_gated_delta_net_bench` measures the BF16 Gated DeltaNet contract with state/head dimension
128, production Q/K normalization, and any positive, divisible `value_heads >= qk_heads` mapping.
Running/chunked modes use batch 1; snapshot mode accepts exact `B=1..8` and optional mixed valid
prefixes. Every measurement is a CUDA Graph replay preceded by a 256 MiB L2 flush outside the timed
interval.

`--running` measures the public running-state entry across recurrent-only, complete 64-token
chunks, and chunked-plus-recurrent-tail routes. `--snapshot` measures the snapshot entry over the
production `W=1..16` batch range; `--qk-norm composed` retains the B=1 two-L2Norm comparison.
`--chunked-only` measures the complete pre-normalized BF16 pipeline through the public Op. Adding
`--breakdown` reports isolated `prepare_wy_wu`, `state_passing`, and `output` stage timings. These
three intrinsic algorithm stages are the benchmark's sole private-launcher exception; the complete
pipeline and every other mode remain public-contract calls.
`stage_share_pct` partitions the sum of isolated-stage medians. Each isolated stage receives its own
cold-L2 flush, so `relative_to_e2e_pct` is informative but is not an additive partition of the
pipeline latency.

`logical_bytes` and `logical_gbps` count each contract-visible tensor and state transfer once.
`traffic_bytes` and `traffic_gbps` instead sum one full tensor extent for every kernel input and
output in the selected implementation. This includes repeated consumption by different kernels,
the composed or public chunked Q/K-normalization intermediates, and every producer/consumer access
to chunked `g_cumsum`, W, U, `v_new`, and `h_chunk`. `intermediate_traffic_bytes` isolates those
normalization and chunked-workspace accesses. These deterministic byte counts describe
implementation-level tensor traffic; physical DRAM/L2 sectors and cache reuse still require NCU.

```bash
cmake --build build --parallel --target ninfer_gated_delta_net_bench
./build/bench/ninfer_gated_delta_net_bench \
  --running --value-heads 32 --sweep --warmup 20 --repeat 100 --csv
./build/bench/ninfer_gated_delta_net_bench \
  --snapshot --value-heads 32 --qk-norm fused --warmup 20 --repeat 100 --csv
./build/bench/ninfer_gated_delta_net_bench \
  --chunked-only --value-heads 32 --tokens 1024 --breakdown \
  --warmup 20 --repeat 100
```

## GDN input-projection Op benchmark

`ninfer_gdn_input_proj_bench` measures all registered public `gdn_input_proj` forms: the 27B
Q4/Q5 two-parent projection, the 35B W8 single-parent projection, and the 27B NVFP4 or row-scaled
FP8 single-parent projection under its admitted policies. Every timed and profiled point is exactly
one public Op call. Single-parent workspace is queried and allocated through the public capacity
entry before timing; the benchmark has no private launchers, route controls, or candidate mode.

Cold cache is the primary model-layer condition. Reported logical traffic counts encoded weights
once, BF16 input once, and QKV/Z outputs once. FLOPs describe the complete registered projection.
The benchmark reports caller policy rather than inferring a private activation-compute route.

```bash
cmake --build build --parallel --target ninfer_gdn_input_proj_bench
./build/bench/ninfer_gdn_input_proj_bench \
  --format all --tokens 1,2,4,8,12,16,32,64,128,256,512,1024 \
  --cache cold --warmup 5 --repeat 30 \
  --csv-out profiles/bench/gdn_input_proj.csv
./build/bench/ninfer_gdn_input_proj_bench \
  --format nvfp4 --nvfp4-policy a4 --tokens 1024 --cache cold --profile
./build/bench/ninfer_gdn_input_proj_bench \
  --format fp8 --fp8-policy a8 --tokens 1,2,3,4,5,6,7,8 --cache cold
```

## GDN input projection/convolution Snapshot/Record Op benchmark

`ninfer_gdn_input_proj_conv_snapshot_bench` measures the public Qwen3.6/Qwen3.8 Q4/Q5, NVFP4,
row-scaled FP8, and W8 `gdn_input_proj_conv_snapshot` / `gdn_input_proj_conv_record` forms for exact
`B=1..8`. The timed body is exactly one complete public Op call; the benchmark does not include
private launchers, candidate selection, duplicated compositions, or route labels. Its default
`T=1..6` sweep is the production MTP verification interval; Record begins at `T=2`.
`--form snapshot|record|both` selects the semantic form. NVFP4 accepts public `a16`/`a4`, while FP8
accepts `a16`/`a8`; the reported profile names caller policy, not a private resolved route.

CUDA Graph replay is the default execution mode. The graph contains external timing event nodes
around the complete Op body, while L2 eviction stays outside the timed interval. Cold-cache results
model successive model layers with distinct weights and are the authoritative comparison; warm
results make launch and cache effects visible. The initial state occupies a slot disjoint from all
published snapshot slots, so repeated replay does not introduce a benchmark-only state reset.

```bash
cmake --build build --parallel --target ninfer_gdn_input_proj_conv_snapshot_bench
./build/bench/ninfer_gdn_input_proj_conv_snapshot_bench \
  --format q4q5 --sweep 1:6 --execution graph --cache both \
  --warmup 10 --repeat 100 \
  --csv-out profiles/bench/gdn_input_proj_conv_snapshot.csv
./build/bench/ninfer_gdn_input_proj_conv_snapshot_bench \
  --format q4q5 --form record --tokens 8 --batch 2 \
  --execution graph --cache cold --warmup 10 --repeat 100
./build/bench/ninfer_gdn_input_proj_conv_snapshot_bench \
  --format nvfp4 --nvfp4-policy a4 --sweep 1:17 \
  --execution graph --cache cold --warmup 10 --repeat 100
./build/bench/ninfer_gdn_input_proj_conv_snapshot_bench \
  --format w8 --tokens 16 --batch 8 \
  --execution graph --cache cold --warmup 10 --repeat 100
./build/bench/ninfer_gdn_input_proj_conv_snapshot_bench \
  --format nvfp4 --tokens 6 --batch 3 --valid-columns 6,3,1 \
  --execution graph --cache cold --warmup 10 --repeat 100
./build/bench/ninfer_gdn_input_proj_conv_snapshot_bench \
  --format fp8 --fp8-policy a8 --form both --batch 1 --sweep 1:16 \
  --execution both --cache both --warmup 5 --repeat 30
```

`--execution eager|both` is available only to attribute launch behavior; it calls the same public
Op with the same operands and workspace.

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
receives the caller-visible execution envelope and owns all decode, prompt, Small-T, and split-KV
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

```bash
cmake --build build --parallel --target ninfer_causal_softmax_attention_bench
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry both --geometry all --kv-dtype all --batch 1 \
  --tokens 1,2,4,6,8,12,16 --context 0,128,2048,8192 \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry append --geometry d256-h16-kv2 --kv-dtype int8 \
  --batch 3 --tokens 6 --row-contexts 127,2047,63 \
  --valid-columns 6,3,0 --table-rows 2,0,1 \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry cached --geometry d256-h16-kv2 --kv-dtype int8 \
  --tokens 16 --context 8192 --execution graph --cache cold --profile
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry append --geometry all --kv-dtype fp8 --batch 1 \
  --tokens 1 --context 16384 --mapping fragmented \
  --execution graph --cache cold --warmup 100 --repeat 201
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry append --geometry all --kv-dtype fp8 --batch 1 \
  --tokens 1024 --context 16384 --mapping fragmented \
  --execution eager --cache cold --warmup 10 --repeat 61
```

The report exposes separate QK/PV logical FLOPs, their full-public-Op-equivalent TFLOP/s,
`key_vector_bytes`/`value_vector_bytes`, and `physical_cache_bytes`. Persistent K+V bytes per D256
vector are 516 for FP8, 288 for NVFP4, and 402 for K8V4 (258-byte K plus 144-byte V).
`unique_kv_bytes` counts each visible persistent KV vector once; `unique_kv_gbps` divides it by
complete Op latency. Payload rates exclude repeated reads and do not measure DRAM bandwidth.
Logical FLOPs do not model private operand conversion, padding, or additional quantization work;
the benchmark therefore does not infer Tensor Core utilization from a storage-format label.

`ninfer_context_softmax_attention_bench` measures the public read-only context-plus-query contract
at Q32/KV8/D128 with BF16 context storage. `T` is a complete non-causal query block and `L` is its
external context length.

```bash
cmake --build build --parallel --target ninfer_context_softmax_attention_bench
./build/bench/ninfer_context_softmax_attention_bench \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 0,2048,8192,32768,131072,196608,262144 \
  --execution graph --cache cold --warmup 10 --repeat 61
```

`ninfer_sliding_window_attention_bench` measures the public Q32/KV8/D128 symmetric sliding-window
contract over a 2048- or 4096-slot cyclic BF16-K/FP16-V cache and a complete non-causal query
block with nonzero represented K/V values. It reports the production route, key tile, split capacity,
merge warp count, Graph node count, workspace, and exact batch shape. `--envelope-max N` holds the
Graph resource envelope fixed while the actual context varies. `--graph-calls 32 --execution graph`
measures 32 public calls in one Graph and reports latency per call; this removes host submission
gaps from short warm-cache measurements. For a cold bundle, L2 is flushed once before the bundle,
so only its first call starts cold. Profiling uses one public call (`--graph-calls 1`).

```bash
cmake --build build --parallel --target ninfer_sliding_window_attention_bench
./build/bench/ninfer_sliding_window_attention_bench \
  --window 2048 --tokens 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 --batches 1,2,3,4,5,6,7,8 \
  --context 64,96,97,128,2047,2048,262144 \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/bench/ninfer_sliding_window_attention_bench \
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
cmake --build build --parallel --target ninfer_packed_softmax_attention_bench
./build/bench/ninfer_packed_softmax_attention_bench \
  --entry both --segments 16 --length 256 \
  --execution graph --cache cold --warmup 10 --repeat 61
./build/bench/ninfer_packed_softmax_attention_bench \
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
cmake --build build --parallel --target ninfer_kv_cache_append_bench
./build/bench/ninfer_kv_cache_append_bench \
  --mode full --full-geometry all --kv-dtype all --tokens 1,2,4,8,16 \
  --context 128 --execution graph --cache cold --warmup 10 --repeat 61
./build/bench/ninfer_kv_cache_append_bench \
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
cmake --build build -j --target ninfer_prepare_ragged_prefix_bench
./build/bench/ninfer_prepare_ragged_prefix_bench \
  --rows 25600 --widths 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --batches 1,2,3,4,5,6,7,8 --counts ragged --execution graph --cache cold
./build/bench/ninfer_prepare_ragged_prefix_bench \
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
cmake --build build --parallel --target ninfer_prepare_masked_block_bench
./build/bench/ninfer_prepare_masked_block_bench \
  --block-sizes 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --execution graph --cache both --warmup 20 --repeat 101
./build/bench/ninfer_prepare_masked_block_bench \
  --block-sizes 16 --execution graph --cache cold --profile
```

Useful traffic is `(2+2B)*4` bytes: two device scalar reads and two complete I32 output writes.

## W8 LinearSwiGLU Op benchmark

`ninfer_w8_linear_swiglu_bench` measures the public W8 LinearSwiGLU profiles:
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
cmake --build build -j --target ninfer_w8_linear_swiglu_bench
./build/bench/ninfer_w8_linear_swiglu_bench \
  --problem dflash2 --t-sweep 1,8,16,32,40,41,51,52,63,64,65,80,81,88,89,96,97,128,1024 \
  --execution graph --warmup 8 --repeat 60 \
  --csv-out profiles/bench/w8_linear_swiglu.csv
./build/bench/ninfer_w8_linear_swiglu_bench \
  --problem dflash2 --profile --t-sweep 128
```

## Q4 LinearSwiGLU Op benchmark

`ninfer_q4_linear_swiglu_bench` measures the public Q4 `[34816,5120] -> [17408,T]` profile and
queries workspace capacity for the requested aggregate interval.

```bash
cmake --build build --parallel --target ninfer_q4_linear_swiglu_bench
./build/bench/ninfer_q4_linear_swiglu_bench \
  --t-sweep 1,2,4,8,16,24,32,40,48 --warmup 10 --repeat 50
```

## NVFP4 LinearSwiGLU Op benchmark

`ninfer_nvfp4_linear_swiglu_bench` measures the public NVFP4
`[34816,5120] -> [17408,T]` profile. The A4 sweep includes both fused route seams and the
larger materialized/TMA paths.

```bash
cmake --build build --parallel --target ninfer_nvfp4_linear_swiglu_bench
./build/bench/ninfer_nvfp4_linear_swiglu_bench \
  --policy a4 --t-sweep 5,48,49,56,64,65,96,97,128,256,1024 \
  --warmup 10 --repeat 50
```

## FP8 LinearSwiGLU Op benchmark

`ninfer_fp8_linear_swiglu_bench` measures the public row-scaled FP8 `[34816,5120] ->
[17408,T]` profile. `--policy a8` measures the production resolver, including caller-owned
activation workspace and the fused SwiGLU output; `--policy a16` measures the public A16 form.
The Tensor Core percentage uses the RTX 5090 dense FP8/FP32-accumulate reference of 419 TFLOP/s
only for extents that the production resolver sends to A8.

```bash
cmake --build build --parallel --target ninfer_fp8_linear_swiglu_bench
./build/bench/ninfer_fp8_linear_swiglu_bench \
  --policy a8 \
  --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,24,32,40,48,1024 \
  --warmup 5 --repeat 30
```

## Q5 LinearAdd Op benchmark

`ninfer_q5_linear_add_bench` measures the public Q5 `[5120,6144]` and `[5120,17408]`
LinearAdd profiles. Every cold-cache sample is one production-dispatched public call, with the
workspace capacity queried for the requested interval.

```bash
cmake --build build --parallel --target ninfer_q5_linear_add_bench
./build/bench/ninfer_q5_linear_add_bench \
  --k 6144 --t-sweep 1,2,4,8,16,24,32,48,49,56,64,192,193 --warmup 10 --repeat 50
./build/bench/ninfer_q5_linear_add_bench \
  --k 17408 --t-sweep 1,2,4,8,16,24,32,48,49,56,64,192,193 --warmup 10 --repeat 50
```

## BF16 LinearAdd Op benchmark

`ninfer_bf16_linear_add_bench` measures the contiguous BF16 `[5120,6144]` projection with its
in-place BF16 residual epilogue. Production uses decode at `T=1`, exact-small-T at `T=2..4`,
aggregate MMA through `T=48`, and the large-T MMA afterward.
Every sample is cold-cache. Effective bandwidth counts the weight once, the activation once, and
the residual read plus write; its `READ_%` and `TC_%` use the benchmark's explicit RTX 5090 BF16
references.

```bash
cmake --build build --parallel --target ninfer_bf16_linear_add_bench
./build/bench/ninfer_bf16_linear_add_bench \
  --sweep 1:48:1 --route production --warmup 10 --repeat 50 \
  --csv-out profiles/bench/bf16_linear_add_t1_48.csv
./build/bench/ninfer_bf16_linear_add_bench \
  --t-sweep 1024,1536,2048 --route production --warmup 10 --repeat 50
./build/bench/ninfer_bf16_linear_add_bench \
  --t-sweep 1024 --route production --profile
```

## W8 LinearAdd Op benchmark

`ninfer_w8_linear_add_bench` measures the W8 `[2048,4096]` and `[2048,6144]` projections with their
BF16 residual epilogue. Production updates the residual in place and uses no workspace. Use
`--production-only` for public evidence; every cold-cache sample follows a 256 MiB L2 flush.

```bash
cmake --build build --parallel --target ninfer_w8_linear_add_bench
./build/bench/ninfer_w8_linear_add_bench \
  --k 4096 --production-only \
  --t-sweep 1,2,4,8,16,32,48,64,96,128 --warmup 10 --repeat 50
./build/bench/ninfer_w8_linear_add_bench \
  --k 6144 --production-only \
  --t-sweep 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,33,64,65,96,128,256,640,641,896,960,1024,1280,2048 \
  --warmup 10 --repeat 50 --csv-out profiles/bench/w8_linear_add.csv
./build/bench/ninfer_w8_linear_add_bench \
  --production-only --t-sweep "$(seq -s, 1 2048)" \
  --warmup 3 --repeat 15 --csv-out profiles/bench/w8_linear_add_all_t.csv
./build/bench/ninfer_w8_linear_add_bench \
  --profile --production-only --t-sweep 1024
```

## FP8 LinearAdd Op benchmark

`ninfer_fp8_linear_add_bench` measures the two public row-scaled FP8 LinearAdd registrations,
including activation quantization, caller-owned workspace, contraction, residual read, and final
in-place BF16 write. `--policy a8` follows the independent production resolver of the selected
semantic Op: `[5120,6144]` uses A16 below `T=22`, while `[5120,17408]` uses A16 below `T=25`; larger
extents use FP8/FP32-accumulate Tensor Core contraction. `TC_%` is reported only when that A8 route
actually executes, against the RTX 5090 419 TFLOP/s reference.

```bash
cmake --build build --parallel --target ninfer_fp8_linear_add_bench
./build/bench/ninfer_fp8_linear_add_bench \
  --k 6144 --policy a8 --t-sweep 1,2,4,8,16,20,21,22,32,48,1024 \
  --warmup 5 --repeat 30
./build/bench/ninfer_fp8_linear_add_bench \
  --k 17408 --policy a8 --t-sweep 1,2,4,8,16,24,25,32,48,1024 \
  --warmup 5 --repeat 30
```

## Attention input-projection Op benchmark

`ninfer_attn_input_proj_bench` measures every registered public `attn_input_proj()` weight/shape
contract: the 27B two-parent Q4/Q5 projection; the 35B W8 Q/K/gate/V and companion Q/K/V
projections; the DFlash2 W8 `[6144,5120]` Q/K/V projection; and the 27B BF16, NVFP4, and FP8
single-parent Q/K/gate/V projections. Fixture packing and public workspace capacity queries happen
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
cmake --build build -j --target ninfer_attn_input_proj_bench
./build/bench/ninfer_attn_input_proj_bench \
  --format all --tokens 1,2,4,8,12,16,32,64,128,256,512,1024 \
  --cache cold --warmup 10 --repeat 50 \
  --csv-out profiles/bench/attn_input_proj.csv
./build/bench/ninfer_attn_input_proj_bench \
  --format nvfp4 --nvfp4-policy a4 --tokens 1024 \
  --cache cold --warmup 10 --profile
./build/bench/ninfer_attn_input_proj_bench \
  --format fp8 --fp8-policy a8 --tokens 1,5,32,64,65,96,128 \
  --cache cold --warmup 10 --repeat 50
./build/bench/ninfer_attn_input_proj_bench \
  --format w8-dflash2-qkv --tokens 8,16,32,53,54,64,65,96,128,1024 \
  --execution graph --cache cold --warmup 10 --repeat 50
```

The stateful GDN projection/convolution/snapshot contract remains in its own public Op benchmark;
it is not a mode of Attention input projection. End-to-end target measurement uses `ninfer_bench`.

## 35B sparse-MoE dFlash benchmark

`ninfer_sparse_moe_bench` measures the complete routed-plus-shared post-mixer Op for the 35B Text
Q4+Q5/Q6 profiles and the MTP W8+W8 profile. It is a long-lived public Op benchmark: fixture setup
uses `sparse_moe_workspace_capacity_bytes()`, and every eager or captured measurement calls only
`ninfer::ops::sparse_moe()`. Production dispatch exclusively owns decode, Small-T, prefill,
workspace views, launch decomposition, and schedule selection.

CUDA Graph replay is the default and authoritative execution mode. The benchmark eagerly
materializes the public call, captures it with stable tensor and workspace addresses, instantiates
the graph, and primes one replay before configured warmup. Timing-enabled external event nodes
surround the captured public call, so `graph_replay` reports the complete device-side SparseMoe
body while excluding fixture reset, L2 eviction, graph capture/instantiation/prime, host launch,
and host synchronization. `eager` uses the same public-call lambda and is only a comparison mode.

```bash
cmake --build build --parallel --target ninfer_sparse_moe_bench
./build/bench/ninfer_sparse_moe_bench \
  --codec q4-q5 --tokens 1 --execution graph --cache both \
  --distribution trace-like --warmup 20 --repeat 200
./build/bench/ninfer_sparse_moe_bench \
  --codec q4-q5 --sweep 1:128:1 --execution graph --cache cold \
  --distribution trace-like --warmup 5 --repeat 50 \
  --csv-out profiles/bench/sparse_moe_public_graph.csv
```

Each cold sample flushes 256 MiB before the timed interval. `trace-like` is the primary
single-sequence verification distribution; `independent` and `same` bound zero and complete expert
overlap. `--execution eager|graph|both` and `--cache cold|warm|both` change only the harness around
the same public call. The CSV `timed_scope` is `full_sparse_moe_device_body`; it contains no private
route, candidate, grid, block, or launch-count fields.

`unique_weight_gbps` counts the router, shared weights, and each selected expert's encoded weights
once, so it is a distribution-aware useful-traffic lower bound rather than measured DRAM traffic.
Its peak percentage uses the device's theoretical DDR bandwidth. `logical_tflops` counts the
router plus eight routed and one shared gate/up and down matrix products.

## W8 context K/V LinearPair benchmark

`ninfer_linear_pair_bench` measures one public `linear_pair()` call over exact adjacent W8
`[1024,K]` K/V row views of a `[6144,K]` QKV parent, for the registered `K=2048` and `K=5120`
geometries. It uses cold-cache eager execution or CUDA Graph replay, accepts an arbitrary token
list or sweep, and reports route-neutral effective bandwidth, logical FLOP/s, and calculated T=1
linear extrapolation.

```bash
cmake --build build -j --target ninfer_linear_pair_bench
./build/bench/ninfer_linear_pair_bench \
  --k 2048 --sweep 1:128:1 --execution graph --warmup 5 --repeat 30
./build/bench/ninfer_linear_pair_bench \
  --k 5120 --tokens 8,16,24,32,40,48,56,64 --execution graph --warmup 5 --repeat 30
./build/bench/ninfer_linear_pair_bench \
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
cmake --build build -j --target ninfer_context_kv_materialize_bench
./build/bench/ninfer_context_kv_materialize_bench \
  --widths 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --batches 1,2,3,4,5,6,7,8 --counts full --execution graph --warmup 8 --repeat 60
./build/bench/ninfer_context_kv_materialize_bench \
  --widths 8,16,128,2048 --batches 1,8 --counts one --warmup 8 --repeat 60
./build/bench/ninfer_context_kv_materialize_bench \
  --widths 17,85,86,128,512,1024,2048 --batches 1 --warmup 8 --repeat 60
```

## MTP exact-transform benchmark

`ninfer_mtp_pack_bench` calls the public bit-exact pack or attention-split Op once per point:

```bash
cmake --build build --parallel --target ninfer_mtp_pack_bench
./build/bench/ninfer_mtp_pack_bench --op pack --d 5120 --tokens 1,2,3,4,5,6,48
./build/bench/ninfer_mtp_pack_bench --op split --tokens 1,2,3,4,5,6,48
```

## Target MTP round benchmark

`ninfer_qwen3_6_27b_mtp_round_bench` measures the registered target's native proposal and
verification round without introducing a second generation controller. It loads the same `.ninfer`
artifact through the target-private package facade, prepares a real prompt with that target's
Frontend, and reports draft/accept statistics for the target-owned MTP schedule:

```bash
cmake --build build --parallel --target ninfer_qwen3_6_27b_mtp_round_bench
./build/bench/ninfer_qwen3_6_27b_mtp_round_bench \
  --artifact out/qwen3_6_27b.ninfer
```

## 35B complete DFlash round benchmark

`ninfer_qwen3_6_35b_a3b_dflash_round_bench` drives the production Program through consecutive
steady DFlash rounds. A measured round includes the previous confirmed feature-to-context append,
the six-layer proposal, target verify/accept, and host publication. It reports GPU and wall latency,
real acceptance, per-position acceptance, mean licensed length, and published tokens/s:

```bash
cmake --build build --parallel \
  --target ninfer_qwen3_6_35b_a3b_dflash_round_bench
./build/bench/ninfer_qwen3_6_35b_a3b_dflash_round_bench \
  --artifact out/qwen3_6_35b_a3b.ninfer \
  --context 4096 --draft-tokens 15 --proposal-head optimized
```

Run separate invocations for `K=1..15`, `full|optimized`, representative contexts, and
`--no-cuda-graph`. The target-side benchmark above supplies forced `A=0..K` transaction evidence;
this complete-round benchmark intentionally preserves the DFlash model's real greedy acceptance.

## Token-decision Op benchmarks

The G1 benchmark calls public `argmax` for the Qwen3.6-35B full physical vocabulary with 248077
valid rows through `C=128`, and for the 131072-row shortlist through `C=120`. With no arguments it
covers every B=1 full-vocabulary width, both T1 routes, and both aggregate maxima:

```bash
cmake --build build --parallel --target ninfer_argmax_bench ninfer_sampling_select_bench
./build/bench/ninfer_argmax_bench
./build/bench/ninfer_argmax_bench --shape full --cols 128
./build/bench/ninfer_argmax_bench --shape shortlist --cols 120
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
./build/bench/ninfer_sampling_select_bench --matrix
./build/bench/ninfer_sampling_select_bench --sample --batch 8 --mode stochastic --top-k 20
./build/bench/ninfer_sampling_select_bench --mtp --mode stochastic --mtp-k 5 --top-k 20
./build/bench/ninfer_sampling_select_bench \
  --dflash2 --drafts 15 --batch 8 --mode greedy --graph-calls 32 --warmup 8 --repeat 60
./build/bench/ninfer_sampling_select_bench \
  --dflash2 --drafts 15 --batch 8 --mode stochastic --top-k 20 --warmup 8 --repeat 60
./build/bench/ninfer_sampling_select_bench \
  --dflash2 --drafts 7 --batch 8 --mode stochastic --mixed --extent 3
./build/bench/ninfer_sampling_select_bench \
  --dflash2 --drafts 15 --batch 8 --mode stochastic --no-counts --reject-at 7
```

## 35B dFlash causal Attention qualification

The public causal benchmark covers exact verify widths `W=1..16`, both KV codecs, and the
append-and-attend and already-cached entries for the D256 H16/KV2 geometry. Batched target
qualification uses the append entry:

```bash
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry append --geometry d256-h16-kv2 --kv-dtype bf16 \
  --batch 1,2,4,8 \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 128,1024,8192 --execution graph --cache cold
./build/bench/ninfer_causal_softmax_attention_bench \
  --entry cached --geometry d256-h16-kv2 --kv-dtype int8 \
  --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
  --context 128,1024,8192 --execution graph --cache cold
```

The former `attention_layer` executable composed multiple implementation-level stages and exposed
private route controls, so it is not retained as a public Op benchmark. Complete mixer and target
effects are measured through the public Engine benchmark or the target round benchmarks.

## Pointwise Op benchmarks

The Section 5 benchmarks cover the complete Qwen3.6-35B pointwise matrix. Default invocation runs
all registered small, established, maximum-video, and maximum-image shapes. `--control` preserves
the selected kernel topology and payload while replacing the mathematical operation with minimal
bitwise work:

```bash
cmake --build build --parallel --target \
  ninfer_residual_add_bench ninfer_sigmoid_mul_bench \
  ninfer_gelu_bench ninfer_add_bias_bench

./build/bench/ninfer_residual_add_bench [--patches P] [--control]
./build/bench/ninfer_sigmoid_mul_bench \
  [--tokens T[,T...]] [--control | --candidate-block B]
./build/bench/ninfer_position_bench \
  [--tokens T[,T...]] [--candidate-block B] [--cold-graph] [--warmup N] [--repeat N]
./build/bench/ninfer_gelu_bench [--mode tanh|exact --columns C] [--control]
./build/bench/ninfer_add_bias_bench [--d D --columns C] [--control]
```

Aligned registered shapes use 16-byte BF16 packs in the cache-sized regime. GELU and AddBias
select their BF16x2 streaming routes for larger Vision items; odd or unaligned repository-internal
test shapes exercise the scalar fallbacks.

## Causal conv1d SiLU Op benchmark

`ninfer_causal_conv1d_silu_bench` times one public entry per invocation. `--tokens` takes a list, so
a route decision comes from one process rather than a series of them. `--cache` defaults to warm;
pass `--cache cold` for the state route decisions are made in.

```bash
./build/bench/ninfer_causal_conv1d_silu_bench --split --cache cold --channels 8192 --tokens 1,2,7,15,16,17,24,32,33,64,65
./build/bench/ninfer_causal_conv1d_silu_bench --split --cache cold --channels 10240 --tokens 1024,4096,8192
```

`--split` selects the split-output entry. Its partition follows the channel extent, because those
are the row profiles the entry admits: `--channels 8192` gives (2048, 2048, 4096) and
`--channels 10240` gives (2048, 2048, 6144).

`--legacy-stage` times the stage the split entry replaced - one packed convolution into a `[C,T]`
plane followed by three `extract_bf16_columns` - so the two can be compared under one set of
timing conditions. It is a decision benchmark and is meant to be removed once that decision is
closed.

```bash
./build/bench/ninfer_causal_conv1d_silu_bench --legacy-stage --split --cache cold --channels 8192 --tokens 1024,8192
```

## Reports

Table, JSON, and CSV reports all identify the selected target, artifact, Engine configuration,
load summary, memory capacity, KV payload, workspace peak, phase throughput, and speculative
statistics. JSON schema version 13 records the public value objects directly:

- `load`: target, `weights_id`, load/upload time, file/H2D/staging bytes, tensor count, and resource
  count;
- `memory`: weights/sequence/unified-workspace arenas, the optional non-additive Vision layout,
  planned context, KV storage, CUDA Graph allowance, and KV payload;
- each repetition's `timings`: prepare, Vision, prefill, decode, and total seconds;
- each repetition's `speculative`: window, rounds, drafted/accepted tokens, fallbacks, and per-position
acceptance.

Each test reports `workspace_peak_bytes` from the planned phase markers, including CUDA Graph
replay, and `workspace_allocator_peak_bytes` from host-side arena allocation activity. These are
intentionally separate: replay reuses captured addresses without advancing the host allocator.

`decode_output_tok_s` counts the requested `G` decode outputs. `decode_engine_tok_s` uses the
Program's speculative round statistics, so it also describes work performed by a final partially
committed speculative round. Reports also contain the command and machine information needed to
interpret a local measurement.

Raw reports and profiler captures remain local under `profiles/bench`, `profiles/ncu`, and
`profiles/nsys`.
