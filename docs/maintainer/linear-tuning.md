# Linear tuning and performance reports

This document owns the tuning extents, workload priorities, dispatch tradeoffs, and final report
format for Linear and related token projections. [Op development](op-development.md) owns the
shared implementation boundaries, candidate workflow, and numerical qualification rules.
[Linear benchmark](linear-benchmark.md) defines the logical metrics;
[bench/README.md](../../bench/README.md#linear-op-benchmark) provides executable commands.

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

A retained Linear performance report describes the final implementation's absolute performance.
Use the [Q4 6144×5120 report](examples/q4-linear.md) as a worked example, with this structure:

Reports for shapes registered later live beside it; the
[Q6 34816×5120 report](examples/q6-linear.md) records the fused gate/up projection at Q6.

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
