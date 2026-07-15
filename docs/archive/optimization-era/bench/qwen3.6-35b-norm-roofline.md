# Qwen3.6-35B Norm Qualification

Date: 2026-07-15

Target: Qwen3.6-35B-A3B N1-N4 standalone normalization domains on RTX 5090:

- offset RMSNorm `[2048,T]`;
- per-head offset RMSNorm Q `[256,16,T]` and K `[256,2,T]`;
- per-head L2Norm Q/K `[128,16,T]`;
- gated plain RMSNorm O/Z `[128,32,T]`.

The measured matrix is `T=1..6` for decode and verification and `T=1024` for the canonical
prefill chunk. This report qualifies the standalone Ops; it does not claim a fused projection or
positional-transform path.

## Implementation

The public RMSNorm, gated RMSNorm, and L2Norm contracts remain dimension-driven. Their fast paths
use two width classes rather than model-shape kernels:

- aligned BF16 `D in {64,128,192,256}` uses one warp per row and 16 independent row warps per
  block;
- aligned BF16 RMS widths `512<=D<=3072, D%512==0` use one 256-thread CTA per row;
- aligned BF16 RMS widths `3072<D<=8192, D%1024==0` use one 512-thread CTA per row;
- other widths or unaligned tensors retain the functional generic route and are not qualified by
  this report.

All fast paths use BF16x2 loads/stores, retain x in registers across the reduction, accumulate in
FP32, and round the output once to BF16. Offset, plain, and gated RMS epilogues are compile-time
instances of the same row-reduction kernels, so no element loop contains a semantic branch or a
nullable-gate check. L2Norm uses the same warp-row geometry but retains its separate
`rsqrt(sum(x^2)+eps)` semantics.

The large-row candidate matrix compared 256 and 512 threads at `D=2048`. NCU measured 13.63 us
for the 256-thread route and 13.86 us for the 512-thread route at T=1024; the final dispatch keeps
256 threads. The general `D=5120` route was also retained: T=1024 measures 10.59 us versus 10.14 us
for its same-payload control and T=1 is at the same launch floor.

## Correctness

```bash
cmake --build build -j --target \
  ninfer_rmsnorm_test ninfer_l2norm_test \
  ninfer_rmsnorm_bench ninfer_l2norm_bench
ctest --test-dir build \
  -R '^ninfer_(rmsnorm|l2norm)_test$' --output-on-failure
```

Both tests pass their FP64-oracle comparison from BF16-rounded inputs. The RMS test includes exact
35B hidden `[2048,6]`, Q `[256,16,6]`, K `[256,2,6]`, and gated `[128,32,6]` cases plus the current
27B `D=5120` route. The L2 test includes exact `[128,16,T]` cases and the complete small-width fast
class `D=64,128,192,256`.

## Benchmark and fixed-resource controls

The benchmark defaults to the full T matrix. A single production/control pair is selected with:

```bash
./build/bench/ninfer_rmsnorm_bench --kind hidden35 --tokens 1024
./build/bench/ninfer_rmsnorm_bench --kind hidden35 --tokens 1024 --control
./build/bench/ninfer_rmsnorm_bench --kind q35 --tokens 1024
./build/bench/ninfer_rmsnorm_bench --kind q35 --tokens 1024 --control
./build/bench/ninfer_rmsnorm_bench --kind k35 --tokens 1024
./build/bench/ninfer_rmsnorm_bench --kind gated35 --tokens 1024
./build/bench/ninfer_l2norm_bench --tokens 1024
./build/bench/ninfer_l2norm_bench --tokens 1024 --control
```

The controls launch the same grid and block geometry and read/write the same external BF16
payloads, but replace normalization with a minimal epilogue. They establish the applicable fixed
work floor when a launch is too small to saturate device DRAM. CUDA-event medians are microseconds:

| Domain | T=1 op/control | T=6 op/control | T=1024 op/control | Interpretation |
|---|---:|---:|---:|---|
| hidden RMS D=2048 | 10.35 / 10.13 | 10.38 / 9.90 | 9.27 / 9.38 | one-CTA-per-row route is at the fixed-work floor |
| Q RMS D=256, H=16 | 9.86 / 9.93 | 9.89 / 9.86 | 9.77 / 9.55 | T=1024 reaches 1718.0 GB/s, 95.9% of the 1792 GB/s traffic roofline |
| K RMS D=256, H=2 | 10.28 / 9.98 | 9.92 / 9.88 | 9.96 / 9.81 | quarter-wave prefill and Small-T are control-limited |
| gated RMS D=128, H=32 | 10.30 / 9.89 | 10.35 / 9.88 | 10.05 / 10.11 | production remains within 4.8% of the same-payload control |
| L2 D=128, H=16 | 9.73 / 10.44 | 9.90 / 9.81 | 9.88 / 9.36 | production remains within 5.6% of the same-payload control |

The absolute Small-T GB/s values are intentionally not compared to full-device DRAM bandwidth:
those launches contain only 2 to 192 row warps. The production/control latency comparison is the
relevant fixed-resource roofline.

## NCU

Basic and detailed captures use a narrow demangled kernel regex and the first launch after the 20
benchmark warmups:

```bash
ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/norm/rms_hidden35_t1024_detailed \
  --set detailed --kernel-name regex:'rmsnorm_cta_bf16x2_kernel' \
  --launch-skip 20 --launch-count 1 \
  ./build/bench/ninfer_rmsnorm_bench --kind hidden35 --tokens 1024

ncu --force-overwrite \
  -o profiles/ncu/qwen3_6_35b_a3b/norm/rms_q35_t1024_detailed \
  --set detailed --kernel-name regex:'rmsnorm_warp_bf16x2_kernel' \
  --launch-skip 20 --launch-count 1 \
  ./build/bench/ninfer_rmsnorm_bench --kind q35 --tokens 1024
```

Equivalent captures are retained for K, gated RMSNorm, L2Norm, and every payload control. NCU
multi-pass duration is diagnostic rather than the acceptance timing above.

| Domain | NCU duration | DRAM SOL | Compute SOL | Occupancy | Registers/thread | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|
| hidden RMS | 9.28 us | 26.65% | 13.35% | 97.18% | 30 | 1.00 |
| Q RMS | 12.26 us | 52.46% | 18.52% | 84.35% | 31 | 2.01 |
| K RMS | 5.15 us | 13.73% | 5.52% | 33.16% | 31 | 0.25 |
| gated RMS | 19.01 us | 68.92% | 31.71% | 87.87% | 35 | 4.02 |
| L2 | 6.46 us | 37.53% | 20.92% | 72.54% | 20 | 2.01 |

No route reports local or shared-memory spilling. Scheduler/WarpState captures identify L1TEX
scoreboard waits as the dominant stall: 79.0% of issue cycles for hidden RMS and 86.5% for Q RMS.
That is consistent with these short load/reduce/store kernels and with production timing tracking
the same-payload controls; it does not identify another standalone-kernel optimization that merits
a new shape specialization.

## Result

N1, N2, N3, and N4 are qualified for their complete 35B standalone domains. Future fusion may
remove launches or intermediate activation traffic, but it is not required for standalone Norm
support and should not replace these general Op routes.
