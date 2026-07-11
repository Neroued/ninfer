# Q5090 V3 GDN Dense Prefill Fused Kernel Report

> Date: 2026-07-02
> Scope: Qwen3.6-27B GDN `in_a`/`in_b` dense prefill path on one RTX 5090.
> Current code commit: `2bd3b66 perf(gdn): fuse dense prefill gates`.
> Baseline comparison: `profiles/nsys-prefill-length-sweep-2026-07-02/`,
> captured before this fused dense change.

## Summary

The old GDN dense prefill path was far from optimal: it used the generic scalar
`linear_generic_dense_gemm_kernel` for two BF16 dense projections per GDN layer,
then launched `gdn_gating`. That path did not use tensor cores.

The new path replaces:

1. `linear(h, in_a) -> a [48,T]`
2. `linear(h, in_b) -> b [48,T]`
3. `gdn_gating(a,b,A_log,dt_bias) -> g,beta`

with one fused BF16 WMMA kernel:

`gdn_in_ab_gated_prefill(h, in_a, in_b, A_log, dt_bias) -> g,beta`

The nsys dense+gating subpath improved by `3.66x` to `7.77x` over 1k-8k and now
falls below `1%` of GPU kernel time at 4k/8k. Low-bit rowsplit GEMM is again the
clear prefill bottleneck.

## Shape

Per GDN layer, the logical GEMM is:

| item | shape |
| --- | --- |
| input `h` | `[5120,T]` BF16 |
| `in_a` | `[48,5120]` BF16 |
| `in_b` | `[48,5120]` BF16 |
| logical fused weight | `[96,5120]` BF16 |
| outputs | `g,beta [48,T]` FP32 |

There are 48 GDN layers, so the old path launched 96 dense GEMMs plus 48 gating
kernels per prefill. The new path launches 48 fused kernels.

## Per-Op Fused Sweep

Command:

```bash
./build/bench/qus_gdn_in_ab_prefill_bench \
  -p 128,256,512,1024,2048,4096,8192,16384 \
  --warmup 20 --repeat 80 --min-ms 500
```

| T | median us | useful TFLOP/s | TC peak pct |
| ---: | ---: | ---: | ---: |
| 128 | 87.424 | 1.439 | 0.65% |
| 256 | 87.445 | 2.878 | 1.31% |
| 512 | 87.733 | 5.737 | 2.61% |
| 1024 | 87.664 | 11.483 | 5.22% |
| 2048 | 87.947 | 22.892 | 10.41% |
| 4096 | 122.944 | 32.751 | 14.89% |
| 8192 | 185.803 | 43.342 | 19.70% |
| 16384 | 452.032 | 35.631 | 16.20% |

## Nsys Delta

Old row is `dense_gemm + gdn_gating`; new row is the fused kernel.

| pp | old ms | new ms | speedup | old share | new share | launches |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1024 | 28.472 | 7.789 | 3.66x | 7.19% | 2.32% | 144 -> 48 |
| 2048 | 41.944 | 8.548 | 4.91x | 5.47% | 1.29% | 144 -> 48 |
| 4096 | 72.416 | 11.109 | 6.52x | 4.81% | 0.85% | 144 -> 48 |
| 8192 | 138.000 | 17.751 | 7.77x | 4.56% | 0.67% | 144 -> 48 |

Current top kernels remain low-bit GEMM:

| pp | `q4` share | `q5` share | fused dense share |
| ---: | ---: | ---: | ---: |
| 1024 | 45.20% | 44.96% | 2.32% |
| 2048 | 45.99% | 43.59% | 1.29% |
| 4096 | 46.41% | 42.99% | 0.85% |
| 8192 | 45.94% | 41.62% | 0.67% |

## NCU Hardware Evidence

NCU was collected on the fused kernel with SpeedOfLight, Occupancy,
ComputeWorkloadAnalysis, and tensor-pipe counters.

| T | variant | grid | SM compute | memory | achieved occ | tensor active / elapsed | HMMA inst |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | NFrags=1 | `(16,3,1)` | 2.98% | 9.11% | 8.33% | 11.8% | 245760 |
| 2048 | NFrags=1 | `(32,3,1)` | 6.17% | 18.87% | 8.33% | 24.6% | 491520 |
| 4096 | NFrags=1 | `(64,3,1)` | 10.68% | 32.86% | 9.43% | 44.0% | 983040 |
| 8192 | NFrags=2 | `(64,3,1)` | 12.24% | 33.29% | 9.41% | 48.9% | 1966080 |

NCU reports the grid is too small to fill the GPU at every sampled length. The
root cause is the skinny `M=96` problem: even at 4k/8k the fused kernel has only
192 CTAs, which is barely enough to place one CTA per SM on RTX 5090 and leaves
achieved occupancy around 9%.

## Conclusion

The old dense path was far from optimal and has been fixed for the model path.
The new fused path uses tensor cores, preserves the BF16-rounded two-step
semantics, removes intermediate `a/b` tensors, and eliminates the dense prefill
bucket as a meaningful end-to-end bottleneck.

The standalone fused dense kernel is still not close to GQA's 80%+ full-device
TC utilization. That is expected for this shape: `M=96` gives too little CTA
parallelism. Pushing this specific kernel much higher would require changing the
algorithm shape, for example split-K partial accumulation plus a reduction/gate
kernel, or prepacking `in_a/in_b` into a true `[96,5120]` GEMM handled by a
library-grade grouped/CUTLASS-style kernel. Given the current nsys share, that
work is lower leverage than further low-bit rowsplit GEMM work.

## Verification

```bash
cmake --build build --target qus_gdn_in_ab_prefill_test -j$(nproc)
cmake --build build --target qus_gdn_in_ab_prefill_bench -j$(nproc)
cmake --build build --target qus_bench -j$(nproc)
./build/tests/qus_gdn_in_ab_prefill_test
compute-sanitizer --tool memcheck --error-exitcode 99 ./build/tests/qus_gdn_in_ab_prefill_test
./build/bench/qus_bench --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus -p 128 -r 1 --warmup 0 -o json --output-file /tmp/qus_prefill_fused_smoke.json
```

Artifacts:

- `profiles/gdn-dense-prefill-fused-2026-07-02/gdn_in_ab_prefill_fused_sweep.csv`
- `profiles/gdn-dense-prefill-fused-2026-07-02/gdn_in_ab_prefill_fused_ncu_summary.csv`
- `profiles/nsys-gdn-dense-prefill-fused-2026-07-02/gdn_dense_fused_nsys_kernel_breakdown.csv`
- `profiles/nsys-gdn-dense-prefill-fused-2026-07-02/gdn_dense_fused_vs_previous.csv`
