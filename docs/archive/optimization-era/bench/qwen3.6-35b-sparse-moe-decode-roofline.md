# Qwen3.6-35B SparseMoe Decode Qualification

> Status: retained operator-level evidence for the repository-internal `SparseMoe` decode route.
> This does not register or qualify the complete 35B target.

## Scope and environment

The qualified domain is ordinary decode only: `T=1`, hidden size 2048, 256 routed experts,
top-8 selection, one shared expert, intermediate size 512, and the `AddResidual` epilogue. The
three admitted routed codec pairs are Q4+Q5, Q4+Q6, and W8+W8; shared gate/up and down weights are
W8 in every case.

Measurements were collected on 2026-07-17 with an NVIDIA GeForce RTX 5090, driver 591.86,
CUDA 13.1, and the Release build. The benchmark allocates the complete 256-expert address spans,
selects experts `{0,32,64,96,128,160,224,255}`, flushes 256 MiB before each pipeline-cold sample,
and keeps each timed stage's producer handoff hot. The retained matrix command was:

```bash
./build/bench/ninfer_sparse_moe_bench --matrix --warmup 10 --repeat 100 \
  --csv-out profiles/bench/sparse-moe-decode-20260717/qualified-matrix.csv
```

Warm measurements are diagnostic. In particular, the cold flush also raises the GPU clock, so a
warm number is not treated as an isolated cache-speedup claim.

## Correctness gate

One C++ oracle evaluates the complete logical formula naively in `double` from represented BF16
input/residual values and test-only exact row-split weight decode. It selects lower expert ids at
an exact top-8 boundary tie and rounds only the observable final destination to BF16. It neither
calls production CUDA decode atoms nor observes D1-D4 workspace values.

Q4+Q5, Q4+Q6, and W8+W8 all produced zero BF16 output difference on the retained fixture. The
cases include expert ids 0 and 255, a tie between ids 254 and 255 whose selected bank changes the
result, nonzero residual input, exact workspace capacity, essential invalid-domain rejection, and
two CUDA Graph replays. The production kernels retain FP32 scores, route scales, and nine
slot-major SwiGLU activations until the final BF16 `AddResidual` store; these are private natural
execution choices rather than oracle rounding boundaries.

## Accepted schedules

Pipeline-cold medians are microseconds. Payload controls read the same selected code/high/scale
planes and use the winner's launch topology; alternative schedule controls execute the complete
stage.

| Stage | Profile | Accepted schedule | Cold | Payload/control | Rejected schedule |
|---|---|---|---:|---:|---:|
| D1 | BF16 | 257 row CTAs, 8 warps | 5.216 | payload 5.344 | 4 warps 5.344 |
| D2 | all | one-warp register top-8 | 6.176 | launch/candidate envelope | serial 14.368 |
| D3 | Q4 routed + W8 shared | 512 CTAs, nine path warps | 18.432 | payload 12.288 | balanced 8-warp 20.480 |
| D3 | W8 routed + W8 shared | 512 CTAs, nine path warps; 32 lanes per W8 dot | 22.528 | payload 16.384 | balanced 8-warp 26.624 |
| D4 | Q5 routed + W8 shared | 2048 CTAs, nine path warps, one row | 12.288 | payload 12.288 | balanced R4 18.432 |
| D4 | Q6 routed + W8 shared | 2048 CTAs, nine path warps, one row | 14.336 | payload 14.336 | balanced R4 20.480 |
| D4 | W8 routed + W8 shared | 2048 CTAs, nine path warps, one row | 16.416 | payload 16.384 | balanced R4 22.528 |

The complete four-launch route measured 40.960 us for Q4+Q5, 43.008 us for Q4+Q6, and
49.120 us for W8+W8. These are operator-level results with synthetic registered-format weights;
they do not establish a 35B end-to-end latency claim.

### D1 and D2

The 8-warp D1 route is on the same-grid payload floor and avoids the second per-thread K vector
needed by the 4-warp neighbor. The one-warp D2 route is 2.33x faster than the serial control while
preserving score-descending/lower-id ordering and selected-only stable normalization.

A threadblock-draining D1+D2 experiment reduced the Q4 front from about 11.3 us to 9.2 us and the
Q4+Q5 complete route from about 40.9 us to 38.9 us. It was rejected because its completion counter
must be zero before the first launch, while public workspace is caller-owned transient storage
with no initial state. An extra memset would restore the contract by adding the launch/API work
the fusion was intended to remove. W8+W8 showed no complete-route gain. The counter, fused kernel,
and benchmark route were therefore removed; production has exactly four launches.

### D3

The documented nine-warp mapping beat the balanced shared-work split for both routed codecs. A
10-warp split-shared candidate tied Q4 and regressed W8, while a two-group ILP/prefetch experiment
regressed Q4 from about 18.4 us to 24.5 us. Neither remains in the implementation.

For Q4, Nsight Compute measured 39 registers/thread, 4.10 KiB static shared memory, no local spill,
51.06% achieved occupancy, 819.24 GB/s DRAM throughput, and 45.96% Memory SOL. Scheduler evidence
attributed 86.7% of warp cycles per issue to L1TEX scoreboard waits. The payload floor, failed ILP
candidate, and winning nine-warp envelope explain the remaining decode/load-use cost without
justifying a deeper pipeline.

The initial W8 pair mapping used only 16 lanes and took about 32.8 us. Giving each of 32 lanes one
W8 code and one activation reduced D3 to 22.5 us and the complete W8+W8 route by about 4.2 us.
Nsight Compute measured 60.67% DRAM throughput, 38.42% Compute SOL, 39 registers/thread, 4.10 KiB
static shared memory, 52.58% achieved occupancy, and zero local-load/store sectors. Applying that
mapping to D4 regressed its producer-aware envelope, so D4 retains its natural paired-value load.

### D4

One output row per nine-warp CTA is the payload floor for Q5 and Q6 and is within measurement noise
of the W8 producer-aware payload control. R2 did not improve Q6 or W8 and regressed Q5; the balanced
R4 schedule reduced grid parallelism and lost for every codec. The rejected R2 code was removed,
while R4 remains only as the matched schedule control.

For Q5, Nsight Compute measured 40 registers/thread, no local spill, 70.89% achieved occupancy,
573.37 GB/s DRAM throughput, and 43.72% Memory/Compute SOL. The exact Q5/Q6/W8 K=512 codec loops,
producer-hot FP32 activation read, route/shared merge, residual read, and sole BF16 store are all
inside the qualified kernel.

## Qualification result

The accepted decode plan is one closed four-launch route:

1. 8-warp row-CTA D1;
2. one-warp register D2;
3. nine-warp, one-intermediate-coordinate D3; and
4. nine-warp, one-output-row D4 with direct `AddResidual`.

It directly addresses only the selected eight routed banks, keeps the shared expert in the same D3
and D4 launches, has no selected-weight gather or per-expert launch, and uses caller-owned
graph-stable workspace. Small-T, prefill, target binding, and end-to-end 35B inference remain
outside this qualification.
