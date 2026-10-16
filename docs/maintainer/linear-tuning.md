# Linear tuning and performance reports

This document owns the tuning extents, workload priorities, dispatch tradeoffs, logical metrics,
measurement conditions, suites, and final report format for Linear and related token projections.
[Op development](op-development.md) owns the shared implementation boundaries, candidate workflow,
and numerical qualification rules. [bench/README.md](../../bench/README.md#linear-op-benchmark)
provides executable commands; the measurement and suite rules below apply to pure `linear()`.

## 1. Workload and tuning extent

For Linear and related token projections, the latency-sensitive **hot interval** is `1 <= T <= 128`,
where `T` is the token-column extent of the actual Op call. This covers a target verification block
of 16 columns (15 drafts plus one anchor) across eight concurrent requests. For other Ops, derive
the measured axes and extents from their actual public inputs and product workload; preserve any
batch, sequence, or state dimensions required by their semantics.

The large-T optimization anchors for Linear and related token projections are `T=512` and
`T=1024`; report both separately. These tuning targets do not restrict supported extents. Other
valid extents, including `129..511` and values above 1024, use the selected broad routes and receive
supporting measurements where a transition or a specific performance question warrants them.

## 2. Small-T priorities

For Linear hot-interval tuning, use the following ordered priorities rather than a single weighted
score. Apply them to valid extents of the actual Op and the workload it serves.

| Priority | Token extents | Selection objective |
|---|---|---|
| 1 | `T=1` | Pursue the best repeatable measured latency independently; gains elsewhere do not compensate for a material regression here. A dedicated implementation is appropriate. |
| 2 | `T=4,8` | Optimize and report each core speculative point separately. Prefer a shared implementation when performance is equivalent. |
| 3 | Other concurrency-derived points below | Cover them efficiently with a small set of capacities, tiles, and broad intervals; add a useful schedule when these points materially lag. |
| 4 | Remaining valid integers through 128 | Maintain efficient interval coverage; modest local costs can justify fewer instances and branches. |

For active concurrency `B=1..8`, ordinary decode and proposal calls that process one column per
request contribute `T=B`. Target verification with three drafts contributes `T=4B`; seven drafts
contribute `T=8B`; fifteen drafts contribute `T=16B`. Their union gives the priority points:

- Ordinary decode / per-request proposal columns: `1,2,3,4,5,6,7,8`.
- MTP with three drafts: `4,8,12,16,20,24,28,32`.
- DFlash/DFlash2 with seven drafts: `8,16,24,32,40,48,56,64`.
- DFlash/DFlash2 with fifteen drafts: `16,32,48,64,80,96,112,128`.

Use the highest applicable priority for an overlapping point. Proposal and fused/stateful Ops use
their own actual extents; verification width does not describe every call in a speculative round.
For workloads such as Vision, choose priority points from their actual use rather than applying
the speculative workload priorities solely because they share a Linear geometry.

## 3. Dispatch and interval quality

Reusable templates live under [linear/kernels/](../../src/ops/linear/kernels/). Ordinary projection
owns [linear/projection/](../../src/ops/linear/projection/); segmented projection owns
[linear/split/](../../src/ops/linear/split/). Each Op keeps its dispatch, selected schedules and
shape policies private, even where tuning values currently agree.
Schedules describe computation organization, operands provide explicit inputs, launchers validate
constraints and launch kernels, and Output/Epilogue define output mapping and fused computation.
Each semantic Op selects its own production instances. A template's availability neither registers
a public shape nor implies that production dispatch uses it.

CTA-local sliced-K reduces partial sums within a block. Cross-CTA split-K requires caller-owned
workspace and a final reduction; apply the epilogue once, after the complete reduction.
New configuration/epilogue combinations receive development qualification; permanent tests cover
public behavior and selected production routes.

Measure every valid integer in the hot interval, respecting the Op's alignment contract. A
temporary private-launcher sweep can establish candidate crossovers and the pointwise performance
envelope. Dense measurement does not require dense dispatch or matching the fastest candidate at
every point. Prefer capacity or interval implementations beyond the core points; an isolated small
gain at an ordinary point does not justify another route. Full/tail specializations are useful
when their repeatable benefit justifies them. Review compiled instance count separately from host
branch count: generating many exact-T instances through a compact template still carries cost.
Each retained specialization or boundary should identify the priority point or useful interval it
protects and the measured cost of merging it. Set neither a universal route-count limit nor a
universal percentage threshold; record absolute latency, relative changes, measurement uncertainty,
and the implementation complexity relevant to the choice.

During candidate selection, review the full pointwise curve, core points individually, affected
priority points and intervals,
and the largest adjacent-extent increase and route/schedule seams. Per-priority normalized averages
may summarize results but do not replace those comparisons. Reasonable tile-count and CTA-wave
steps, and modest capacity-tail overhead, are acceptable within the hot interval. Prioritize broad
inefficient intervals and poorly served priority points; interval smoothness is not a completion
requirement. Explain material steps and retained tradeoffs using the available evidence, with
targeted candidate comparisons when they can change the selection. Never omit, interpolate over,
or replace an observed point with an invented value.

Beyond the hot interval, optimize both `T=512` and `T=1024` for Linear and related projections;
for other Ops, select a small number of anchors from their actual bulk workload. Compare each
anchor separately so an average cannot hide a regression at the other. Optimize throughput against
the roofline of the execution resource used by the selected route. Use sparse supporting points
around relevant transitions and as few broad routes as the evidence permits; a reasonable transition discontinuity is acceptable here. A permissive public
policy does not prove that a particular accelerator route ran, so roofline evidence must identify
and measure the implementation that production dispatch actually selects. These are completion
requirements for the large-extent region, not a mandatory position in the development order.

## 4. Report format

This section applies when a retained absolute-performance report is requested; routine changes do
not require a new report or plot. Task summaries follow [AGENTS.md](../../AGENTS.md#verification-and-reporting),
including comparative results, regressions, and limitations.

A retained Linear performance report describes the final implementation's absolute performance.
Use the [Q4 6144×5120 report](examples/q4-linear.md) as a worked example, with this structure:

1. State the format/layout, N/K, input/output types, activation policy, GPU/toolchain, timing
   boundary, cache conditions, warmup/repetitions, and latency statistic.
2. Plot the final latency at every valid T through 128 on linear axes, marking priority points
   and emphasizing 1/4/8. Show 512 and 1024 separately. Preserve measured steps and retain the
   figure as SVG alongside the report document.
3. Tabulate priority points and both bulk anchors: latency, logical GB/s, bandwidth utilization,
   useful TFLOP/s, and Tensor Core utilization. Use the existing benchmark's logical byte/FLOP
   accounting and explicitly name the hardware peak denominators. T=1 emphasizes bandwidth;
   512/1024 emphasize Tensor Core efficiency. Use the dense peak for the actual MMA input and
   accumulator precision; show a dash when the selected path does not use Tensor Cores. Label a
   sustained-read reference separately from the nominal bandwidth reference.
4. Briefly explain the final curve's material steps and known limitations, summarize numerical
   qualification, and provide reproduction commands and links to the implementation.

These utilization figures use logical workload divided by elapsed time and the stated peak,
rather than hardware counters. Retained reports omit old-implementation comparisons, speedup
ratios, aggregate improvement scores, and candidate-search history. Candidate measurements remain
working evidence for dispatch decisions; the report presents the resulting implementation.

## 5. Logical metrics

Pure Linear counts one encoded weight read, one BF16 activation read and one BF16 output write.
Weight bytes sum the consumed code and scale planes, excluding gaps between aligned planes.
For row-split Q4/Q5/Q6/Q8, let `K_pad = align_up(K,128)` and
`groups = N*K_pad/group_size`:

| Stored format | Group size | Weight bytes |
|---|---:|---:|
| Q4 G64 FP16 scales | 64 | `groups * (32 code + 2 scale)` |
| Q5 G64 FP16 scales | 64 | `groups * (32 low + 8 high + 2 scale)` |
| Q6 G64 FP16 scales | 64 | `groups * (32 low + 16 high + 2 scale)` |
| Q8 G32 FP16 scales | 32 | `groups * (32 code + 2 scale)` |
| NVFP4 | 16 | `N*K/16 * (8 code + 1 E4M3 scale)` |
| BF16 | — | `2*N*K` |
| FP8 E4M3 row BF16 scales | one row | `N*(K+2)` |

The current NVFP4 fixture uses its admitted aligned N/K directly; its scalar divisor is metadata,
outside the code/scale traffic model. For elapsed time `seconds` and `median_us`:

```text
model_bytes       = weight_bytes + 2*K*T + 2*N*T
effective_GB/s    = model_bytes / seconds / 1e9
useful_flops      = 2*N*K*T
useful_TFLOP/s    = useful_flops / seconds / 1e12
DRAM_%           = effective_GB/s / memory_gbs * 100
READ_%           = effective_GB/s / sustained_read_gbs * 100
memory_floor_us  = model_bytes / (memory_gbs * 1e9) * 1e6
memory_floor_pct = memory_floor_us / median_us * 100
```

These are logical workload rates. Physical cache transactions, weight rereads, workspace traffic,
split/reduction work and cache hits require profiler evidence. Dequantization, padding and repeated
private work do not increase useful FLOPs. The benchmark does not infer executed FLOPs or weight
replay from a schedule.

The compiled target selects references from
[`bench/common/roofline.h`](../../bench/common/roofline.h); their values and provenance are described
in [Benchmarks](../../bench/README.md#hardware-references). `READ_%` is useful for read-dominated
points; a read/write copy ceiling does not replace that reference. Tensor Core percentages use the
dense peak for the actual input and accumulator profile only where the exact geometry/policy/extent
establishes that profile. An activation permission alone does not identify the executed arithmetic.
Unknown references or arithmetic leave the associated fields empty in CSV and `-` in console output.
If the compute reference is absent, report the memory lower bound without claiming a complete
roofline bound or classifying the point as memory-limited.

Console and CSV identify labels, format, policy, N/K/T, median/min/p95, logical rates and applicable
peak percentages. CSV retains byte breakdown, execution, Graph nodes/calls, warmup, repetitions
and flush size; the console header identifies the actual GPU, CUDA runtime and cache mode.
A sweep reports adjacent-point `delta_%`; if the group contains
T=1, `T1_lin_x = T*median(T=1)/median(T)` is a scaling reference, not timing of T sequential calls.
Private candidate, tile, launcher and inferred physical-traffic fields do not belong in this report.

## 6. Pure Linear timing

Exact points, sweeps and suites use the same fixture and public `ops::linear()` call, including all
activation quantization, contraction and output work. Workspace comes from the public capacity
query before timing. Allocation, initialization, first-use setup and warmup are outside samples.

The default is eager CUDA-event timing, three warmups and twenty samples. Each ordinary sample
evicts L2 with a 256 MiB read before the start event, times one complete call and reports median,
min and p95. Graph mode captures that call and uses the existing graph-launch elapsed timer;
it does not adopt the distinct Graph-body event timer used by other Op benchmarks.

`--graph-calls N` bundles 1..64 public calls and normalizes time and work rates per call. Eviction
occurs once before the bundle, so later calls reuse cached inputs. N>1 is labeled
`cold-before-graph-bundle`, omits DRAM/read percentages and the memory floor, and cannot be combined
with `--profile`. Profiling admits one exact point and brackets one complete public call after
setup, warmup and eviction; it may contain several production kernels. Ordinary measurements do
not add an automatic second warm-cache pass.

## 7. Curated suites and extension

`qwen3_6_27b` and `qwen3_6_35b_a3b` are explicit convenience suites. Their current geometry/role
entries live in [linear_bench.cu](../../bench/ops/linear_bench.cu), and all use `A16Only`.
`Continuous` samples T=`1,16,128,1024`; `VisionStep4` samples T=`4,128,1024`.
The `all` suite unions identical `(qtype,policy,N,K,T)` points, retaining multiple role labels
without repeating their timing. No-argument invocation prints usage instead of starting a suite.

Points share one weight and maximum-T activation/output allocation per `(qtype,policy,N,K)` group.
Suites use the exact-point execution path and do not reproduce production admission or dispatch.
Fused projections and sparse MoE remain in their owning Op benchmarks even when their parent
weights are present in an artifact. NVFP4 A16/A4 development points remain explicit numeric
geometries: `[14336,5120]`, `[16384,5120]`, `[34816,5120]`, `[5120,6144]`, `[5120,17408]`.

A newly supported production shape is immediately measurable through numeric N/K/T arguments;
add it to a suite only for a representative workload role. Entries contain semantic geometry,
format, policy, label and sampling class, without private launch or route metadata. A new route
does not require a benchmark branch. A new format/profile requires a real public execution path,
its persistent-format fixture, an explicit byte formula and independent numerical qualification.
Candidate selection remains task-local work under [Op development](op-development.md#7-performance-evidence).
