# Qwen3.6 small-T GDN fusion

This is the active implementation design for reducing launch and intermediate-transfer latency in
the Qwen3.6 GDN Verify path. The semantic Op and family schedule are shared by the 27B and 35B-A3B
peer Variants; their distinct packed projection layouts select separate exact implementations under
shared Op ownership. Measurements use RTX 5090 (`sm_120a`, CUDA 13.1). Prefill, activation
quantization, and changes to MTP commit/rollback policy are outside this work.

The document should be removed after the fused route is complete; stable Op and model contracts
belong in the existing active maintainer references.

## 1. Scope and baseline

The measured layer begins at the hidden RMSNorm and ends after the W8 residual output projection.
It includes the three q/k/v device-to-device extracts and excludes the following Sparse MoE tail.
The exact 35B geometry is:

| Value | Extent |
|---|---:|
| hidden | 2048 |
| q/k head dimension | 128 |
| q/k heads | 16 |
| value heads | 32 |
| q rows / k rows / value rows | 2048 / 2048 / 4096 |
| convolution rows | 8192 |
| input-projection rows | 12288 |
| snapshot slots | 7 |

Build and run the committed whole-layer benchmark with:

```bash
cmake --build build --parallel --target ninfer_gdn_layer_bench
./build/bench/ninfer_gdn_layer_bench --route composed
./build/bench/ninfer_gdn_layer_bench --route fused
```

The benchmark uses one CUDA Graph per T, five warmups, forty measured replays, graph-bracketing
CUDA events, and a 256 MiB L2 flush before every sample. The unfused baseline is:

| T | Graph nodes | Median (us) | Min (us) | P95 (us) |
|---:|---:|---:|---:|---:|
| 1 | 12 | 48.384 | 47.936 | 50.432 |
| 2 | 12 | 48.384 | 48.032 | 50.432 |
| 3 | 12 | 50.432 | 49.952 | 50.464 |
| 4 | 12 | 52.320 | 50.176 | 52.512 |
| 5 | 12 | 54.528 | 53.568 | 54.560 |
| 6 | 12 | 56.320 | 54.144 | 56.576 |

A CUDA Graph node trace at T=1, 4, and 6 attributes the median GPU duration as follows:

```bash
nsys profile --force-overwrite=true \
  --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  -o /tmp/ninfer_gdn_layer_t4_nodes \
  ./build/bench/ninfer_gdn_layer_bench --route composed --t-sweep 4 --warmup 5 --repeat 40
```

Substitute T=1 or T=6 for the endpoint traces. Nsight instrumentation raises the whole-graph
latency, so these values guide attribution rather than replace the uninstrumented benchmark
baseline.

| Current segment | T=1 (us) | T=4 (us) | T=6 (us) |
|---|---:|---:|---:|
| W8 input projection + convolution + three q/k/v extracts | 24.51 | 24.72 | 24.00 |
| hidden RMSNorm + BF16 control projection | 6.45 | 6.73 | 6.95 |
| two L2Norms + recurrent rule + gated RMSNorm | 7.54 | 10.36 | 11.83 |
| W8 output projection + residual | 8.52 | 11.18 | 13.43 |

The current production composition is:

```text
x --RMSNorm--> h --W8 input--> qkv,z --conv--> qkv_c --3x extract--> q,k,v
h --BF16 control----------------------------------------------------> g,beta
q,k --2x L2Norm--> qn,kn
qn,kn,v,g,beta --recurrent snapshot--> o --gated RMSNorm(z)--> on
on --W8 output + residual-------------------------------------------------> x'
```

The large W8 projections are necessary work. The actionable overhead lies in the eight short
normalization, convolution, transfer, and state/control nodes around them.

## 2. Implementation order

### 2.1 Fuse W8 input projection, causal convolution, and q/k/v placement

This is the first and highest-value change. Replace the current five nodes with one fused Op
implementation:

1. The W8 projection CTA completes the projected values for its owned output rows and active
   tokens.
2. For the first 8192 rows, feed the projection accumulator directly into the width-four depthwise
   convolution and SiLU epilogue using the implementation's natural qualified precision.
3. Route the result directly to contiguous q, k, or v storage according to the output row. The
   final 4096 projection rows continue to write z directly.
4. Read the device `initial_slot` at replay time and publish each token's convolution snapshot to
   slots `0..T-1` exactly once.

For T=2..6, the split-K kernel's final warp can stage its 16-row by T result tile in shared memory;
one lane per row then performs the short sequential convolution. The T=1 decode kernel can apply
the same epilogue directly in the row-owning warp.

The former materialized qkv tensor is not an observable output of the fused Op and therefore does
not impose a BF16 cast. The implementation should retain projection results in registers/shared
memory at the precision that gives the best qualified accuracy and latency. Only the fused Op's
BF16 q/k/v/z outputs and BF16 convolution snapshots are explicit cast boundaries.

The 27B physical leaf applies the same semantic fusion to its Q4 q/k and Q5 value projection
kernels. The fastest T=1..3 and T=5..6 paths run convolution and snapshot publication from the
projection epilogues. T=4 instead materializes one private BF16 projection tile and consumes it in
one fused conv/SiLU/split/snapshot kernel; this avoids the repeatable register-lifetime/serial
epilogue cliff at that extent. Its separate z projection remains in the existing output-gate leaf
because that is an actual checkpoint weight/schema difference, not a second GDN schedule.

Stretch target: graph nodes `12 -> 8` and at least 3 us lower median latency at every T.

Status: complete. The 35B whole-layer result (same cold-L2 CUDA Graph methodology as the baseline)
is:

| T | Composed nodes | Composed median (us) | Fused nodes | Fused median (us) | Saved (us) |
|---:|---:|---:|---:|---:|---:|
| 1 | 12 | 48.384 | 8 | 46.336 | 2.048 |
| 2 | 12 | 48.384 | 8 | 46.080 | 2.304 |
| 3 | 12 | 50.432 | 8 | 46.336 | 4.096 |
| 4 | 12 | 52.320 | 8 | 48.384 | 3.936 |
| 5 | 12 | 54.528 | 8 | 50.464 | 4.064 |
| 6 | 12 | 56.320 | 8 | 52.480 | 3.840 |

The 27B actual-shape Op benchmark uses the same 256 MiB cold-L2 graph-replay protocol with five
warmups and eighty measurements:

| T | Composed median (us) | Fused median (us) | Saved (us) |
|---:|---:|---:|---:|
| 1 | 39.872 | 34.048 | 5.824 |
| 2 | 52.096 | 42.240 | 9.856 |
| 3 | 56.096 | 48.384 | 7.712 |
| 4 | 52.032 | 46.048 | 5.984 |
| 5 | 68.480 | 54.528 | 13.952 |
| 6 | 73.536 | 58.496 | 15.040 |

Both exact leaves pass direct independent FP64 projection/conv/SiLU oracle checks at every T,
including produced BF16 q/k/v(/z), BF16 state slots `0..T-1`, and exact preservation of untouched
slots. Real 27B and 35B MTP3 runs also pass with CUDA Graphs enabled. The 35B T=1/2 result improves
but misses the stretch target; a staged W8 projection plus one fused post kernel was measured and
rejected because it regressed both median and P95 relative to the selected accumulator epilogue.

### 2.2 Fuse hidden RMSNorm into the control projection

The 35B BF16 control projection already uses a 32-CTA cooperative split-16 kernel. Extend it with a
leading phase:

1. The first T CTAs execute the existing block-256 RMSNorm reduction over hidden size 2048 and
   write BF16 h.
2. Synchronize the cooperative grid.
3. Run the existing split-K BF16 MMA and FP32 g/beta epilogue unchanged.

The fused Op produces h, g, and beta. The fused W8 input/conv Op then consumes h. This avoids
duplicating the hidden norm and preserves the shared BF16 h consumed by both projections.

Expected result: graph nodes `8 -> 7` and at least 1 us additional median improvement without a
regression at any T.

### 2.3 Fuse q/k normalization into the recurrent snapshot kernel

Do not independently normalize q/k in every recurrent CTA. Each q/k head is shared by two value
heads and eight state-row partitions, so that design repeats the reduction sixteen times.

Instead, make the recurrent kernel cooperative:

1. A small leading set of warps computes one FP32 inverse norm for every `(qk_head, token)` pair
   and stores the factors in a small workspace.
2. Synchronize the grid.
3. Run the existing 256 logical recurrent CTAs. On q/k load, apply the normalization factor and
   pass the naturally represented normalized values directly into the recurrent math.
4. Preserve the existing FP32 SSM state and slots `0..T-1` snapshot writes.

This removes both `qn` and `kn` materializations. Their former BF16 storage does not constrain the
fused implementation; only the fused Op's declared outputs and persistent state formats do.

Expected result: graph nodes `7 -> 5` and at least 1.5 us additional median improvement. The
recurrent kernel itself must not regress enough to consume the two removed launches.

### 2.4 Append gated RMSNorm to the recurrent cooperative kernel

After recurrent output and snapshot publication:

1. Stage the recurrent result in the fastest oracle-qualified private format.
2. Synchronize the grid.
3. Assign the `(value_head, token)` rows to a subset of warps and execute the existing D=128 gated
   RMSNorm using z, writing on.

An ordinary grid-wide handoff still requires a private global staging buffer between recurrent
state tiles and full value-head reductions. BF16 and FP32 staging are implementation candidates:
choose from direct oracle qualification and whole-kernel latency rather than preserving the former
standalone Op's storage dtype.

Expected result: graph nodes `5 -> 4`. Accept this step only if it improves the median by at least
0.8 us without worsening any T or materially widening P95. Otherwise the five-kernel endpoint is
the preferred result.

## 3. Intended final topology

```text
[hidden RMSNorm + BF16 control] -------> h, g, beta
                  |
                  v
[W8 input + conv + direct q/k/v] ------> q, k, v, z + conv snapshots
                  |
                  v
[q/k norm + recurrent + gated RMS] ---> on + FP32 SSM snapshots
                  |
                  v
[W8 output + residual] ----------------> x'
```

The practical target is four kernels. The acceptable fallback is five kernels with gated RMSNorm
left separate.

Do not fuse these boundaries without new evidence:

- Hidden RMSNorm into W8 input projection: output-row CTAs cannot cheaply share the hidden-wide
  reduction and would duplicate it.
- BF16 control projection into W8 input projection: the weight formats, MMA schedules, and useful
  grid geometries differ; a heterogeneous kernel risks slowing the dominant W8 work.
- Gated RMSNorm into W8 output projection: every output tile would repeat normalization and SiLU
  work, replacing one short launch with large duplicated computation.
- The complete layer into one kernel: repeated grid synchronization, combined register/shared
  memory pressure, and loss of independently tuned W8 schedules outweigh one additional removed
  launch.

## 4. Correctness and state acceptance

No fused route may quantize an activation. The represented inputs, packed/BF16 weights, FP32
control values, FP32 SSM state, BF16 persistent convolution state, and declared BF16 outputs remain
the same logical domains.

Each milestone must be checked for every T in `1..6` against the independent mathematical oracle,
covering both produced values and state transitions:

- final residual output;
- convolution snapshots in slots `0..T-1` from every legal device `initial_slot`;
- FP32 SSM snapshots in slots `0..T-1`;
- q/k normalization, control g/beta, and gated output at their declared output criteria;
- MTP accept, partial accept, and reject paths without premature state publication.

Pairwise parity with the unfused implementation is useful diagnostic evidence but is not the
oracle. Projection, normalization, convolution, and recurrent intermediates eliminated by fusion
have no inherited storage dtype or rounding requirement. Their accumulation order and precision
should be selected for the strongest directly oracle-qualified performance.

## 5. Performance acceptance

After every milestone, run both whole-layer benchmark routes and compare all six T values from the
same build and machine state. Reject a candidate if any T has a repeatable median regression above
0.5 us, if P95 materially widens, or if the claimed node reduction is not visible in a CUDA Graph
node trace.

The final four-kernel target is:

| T | Baseline median (us) | Target range (us) |
|---:|---:|---:|
| 1 | 50.43 | 41-43 |
| 4 | 52.48 | 44-46 |
| 6 | 56.54 | 48-50 |

All intermediate T values must improve as well. An 8-10 us per-layer reduction corresponds to
approximately 0.24-0.30 ms across the 30 GDN layers in one 35B verification pass; end-to-end
claims still require direct inference measurement after the Op-level target is met.
