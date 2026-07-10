# Dense GDN `in_a/in_b` Roofline Kernel Design

Date: 2026-07-11
Target: Qwen3.6-27B text GDN prefill on one RTX 5090
Shape: two BF16 projections `[48,5120] x [5120,T]`, followed by GDN gating

## Goal

Replace the under-filled WMMA prefill kernel with an SM120-oriented tensor-core
kernel that reaches the hardware roofline for the contraction while preserving
the existing fused operator contract:

```text
a = BF16(Wa @ x)
b = BF16(Wb @ x)
g = -exp(A_log) * softplus(a + dt_bias)
beta = sigmoid(b)
```

Decode (`T=1`) and the existing small-T split-K route (`T=2..8`) are out of
scope. This phase changes only the Large-T dense GDN kernel.

## Baseline and Root Cause

At `T=1024`, the current kernel launches `(16,3)` CTAs with four warps per CTA:

- 48 CTAs for 170 SMs, or 0.03 full waves;
- 2.99% tensor-pipe activity and 9.32% memory SOL;
- 8.33% achieved occupancy;
- 87.98 us median in the standalone benchmark;
- 8.68 ms over 48 GDN layers, 2.47% of pp1024 GPU kernel time.

The arithmetic is only about 1.007 GFLOP per layer at `T=1024`. At the measured
roughly 210 TFLOP/s dense BF16 MMA ceiling, the contraction roof is about 4.8 us.
The existing kernel is therefore launch-geometry and latency limited, not
bandwidth limited.

## Architecture

### CTA tile

The production CTA owns:

```text
head rows:  16 paired rows (16 from Wa and the matching 16 from Wb)
tokens:     128
K tile:      64
warps:        8, one 16-token tile per warp at T=1024 and unsplit T
stages:       2
```

The longer-context cooperative split routes compile a 16-warp specialization,
while `T=9..1024` dispatches the measured W8/S8 winner. W8 gives every warp four
independent A/B accumulator chains and makes the 192-CTA grid's thread count
match the 49,152 output elements exactly during the cooperative epilogue.

Every warp keeps both A and B FP32 accumulators. The activation fragment is
loaded once and feeds two `mma.sync.m16n8k16.bf16` instructions, one for each
projection. This retains the useful paired-projection reuse from the current
kernel.

### Shared-memory pipeline

Each stage contains:

- `x[128,64]` BF16: 16 KiB;
- `Wa[16,64]` BF16: 2 KiB;
- `Wb[16,64]` BF16: 2 KiB.

Two stages consume 40 KiB. Cooperative 16-byte `cp.async.cg` copies stage the
three operands. The existing XOR 64-column swizzle and `ldmatrix` helpers are
reused so the tensor fragments are bank-conflict free.

### Deterministic split-K

The unsplit output raster has only 24 CTAs at `T=1024`. K is therefore split
into contiguous 64-wide tiles:

| T range | split-K | main CTAs at the upper bound |
| ---: | ---: | ---: |
| `9..1024` | 8 | 192 |
| `1025..2048` | 4 | 192 |
| `2049..4096` | 2 | 192 |
| `>4096` | 1 | at least 192 |

For split-K routes, the main kernel writes FP32 partials as `[split,T,96]`,
executes a cooperative grid barrier, then uses the entire resident grid to
reduce splits in ascending order, round the completed A/B projections to BF16,
and apply both gating epilogues. There is no second kernel launch. Maximum
workspace is 3 MiB of live storage at T=1024 (6 MiB of write-plus-read traffic).

Split-K uses a deterministic reduction and never uses FP32 atomics. The FP32
association differs from the unsplit WMMA kernel, but BF16 rounding still occurs
only after the complete dot product. Correctness is judged against the existing
mathematical oracle and `gdn_output_bf16` tolerance.

`T=9..1024` deliberately keeps one canonical S8 K-association. Changing split-K
changes the FP32 grouping before the final BF16 round; the earlier short-T
S40/S20/S10 dispatch could therefore make 128-token chunked prefill and a
single larger prefill cross an argmax boundary. Canonical S8 restores exact
128/512 chunk token-stream parity while retaining the T=1024 roofline path.

For `split-K=1`, an eight-warp specialization performs the BF16 rounding and
gating epilogue directly and does not allocate workspace or execute a grid
barrier. This avoids the long-context regression seen with the 16-warp split
specialization.

## Dispatch and Tuning

The implementation was tuned in this order:

1. choose split-K from `{4,5,8,10,16}` at T=1024;
2. verify the choice at T=512 and T=2048;
3. use one S8 association through T=1024 to preserve chunk-size parity;
4. retain a single production dispatch with no tuning environment variables.

The T=1024 standalone medians for the split sweep were S4 20.41 us, S5
21.30 us, S8 20.67 us, S10 21.04 us, and S16 21.12 us before the cooperative
epilogue. The final cooperative W8/S8 configuration is faster than the W16/S8
candidate and reaches 16.80 us in the final long sweep.

## Measured Result

Final standalone sweep, 30 warmups, 300 repetitions, and at least 1.5 seconds
per shape:

| T | final median us | original median us | speedup |
| ---: | ---: | ---: | ---: |
| 17 | 10.78 | 156.12 | 14.48x |
| 32 | 10.68 | 83.34 | 7.80x |
| 64 | 10.67 | 87.44 | 8.20x |
| 128 | 10.74 | 87.38 | 8.14x |
| 256 | 10.80 | 88.52 | 8.20x |
| 512 | 10.82 | 88.42 | 8.17x |
| 1024 | 16.98 | 87.98 | 5.18x |
| 2048 | 27.27 | 89.21 | 3.27x |
| 4096 | 48.16 | 127.52 | 2.65x |
| 8192 | 77.93 | 205.20 | 2.63x |

At T=1024, Nsys measures the single production kernel at 13.60 us median. The
contraction contains 1.0066 GFLOP, so the fused kernel delivers about 74.0
useful TFLOP/s while also materializing and deterministically reducing 3 MiB of
FP32 partials and evaluating 98,304 transcendental epilogue outputs.

The full-model pp1024 capture reduces the dense GDN aggregate from 8.683 ms to
0.680 ms over 48 launches, a 12.77x reduction. Its kernel share falls from
2.47% to 0.24% of GPU kernel time. The standalone and integrated ratios differ
because the old full-model capture exposed much larger tail latency than the
isolated benchmark.

The NCU production report records 71 registers/thread, 40 KiB dynamic shared
memory, no local/shared spills, and 192 cooperative CTAs. Its whole-kernel
tensor-pipe percentage is not a pure GEMM roofline metric: it includes the
grid-wide barrier, deterministic FP32 reduction, BF16 rounding, and SFU-heavy
gating epilogue. The reported 30.65% active value must therefore not be
presented as contraction-only Tensor Core efficiency. The relevant fixed-shape
roofline result is the measured single-launch plateau: BK=128, BM=48, BN=256,
W8/W16, and S4/S5/S8/S10/S16 alternatives did not beat 13.60 us GPU time or
16.80 us operator time.

## Acceptance Gates

- report whole-kernel Tensor metrics without mislabeling barrier/epilogue time
  as contraction time;
- no local/shared spilling;
- standalone T=1024 operator at least 3x faster than 87.98 us;
- exact shape sweep covers `T={17,32,64,128,256,512,1024,2048,4096,8192}`;
- numerical tests cover an unsplit route and representative split-K routes;
- MTP E2E preserves exact 128/512 prefill-chunk token-stream parity;
- Compute Sanitizer memcheck reports zero errors;
- pp1024 Nsys confirms the cooperative single launch and a lower aggregate
  dense GDN time;
- this phase is committed before grouped-input work begins.
