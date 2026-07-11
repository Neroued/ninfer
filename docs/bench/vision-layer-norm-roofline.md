# Vision LayerNorm Roofline Optimization

Date: 2026-07-11

Target: affine LayerNorm `[1152,4096]`, FP32 Welford statistics and BF16 output, on RTX 5090.

## Change

The baseline assigned one 256-thread CTA to each row and reduced three 256-entry shared arrays
through eight block-wide synchronization rounds. The final implementation assigns one warp to each
row and four rows to a 128-thread block:

- BF16x2 input, weight, bias, and output access;
- per-lane Welford accumulation over 36 values;
- one warp-shuffle Welford reduction;
- no shared memory and no block barrier;
- the generic scalar implementation remains for non-1152 or unaligned shapes.

Intermediate 128-thread one-row and 64-thread one-row variants measured 11.62 us and 12.18 us;
both were slower than the final four-row warp design.

## Timing

| Version | Median | Effective bandwidth |
| --- | ---: | ---: |
| shared-memory baseline | 16.72 us | 1129.4 GB/s |
| one-row shuffle | 11.62 us | 1624.9 GB/s |
| four-row warp final | 11.05 us | 1708.8 GB/s |

The final kernel improves median latency by 33.9% and reaches 95.4% of the fixed 1792 GB/s
effective bandwidth roofline.

## NCU

```bash
ncu --force-overwrite \
  -o profiles/vision-ops/layer-norm/baseline.ncu-rep \
  --set detailed --kernel-name regex:'layer_norm_kernel' \
  --launch-skip 20 --launch-count 1 build/bench/qus_layer_norm_bench

ncu --force-overwrite \
  -o profiles/vision-ops/layer-norm/warp_optimized.ncu-rep \
  --set detailed --kernel-name regex:'layer_norm_d1152_warp_kernel' \
  --launch-skip 20 --launch-count 1 build/bench/qus_layer_norm_bench
```

| Metric | Baseline | Final |
| --- | ---: | ---: |
| NCU duration | 29.82 us | 19.65 us |
| grid | `4096×256` | `1024×128` |
| registers/thread | 36 | 36 |
| static shared/block | 3.07 KB | 0 B |
| waves/SM | 4.02 | 0.50 |
| achieved occupancy | 91.53% | 51.25% |
| branch efficiency | 97.41% | 99.87% |
| local/shared spilling | 0 | 0 |

The lower final occupancy is intentional: only 4096 independent row warps exist, or about 24
warps per SM. Splitting a row across extra warps raises reported occupancy but requires cross-warp
synchronization and is slower in direct measurements. The final design is work-efficient and its
timing harness reaches 95.4% of the traffic roofline despite NCU correctly reporting a short,
sub-wave-limited kernel.

## Correctness

```bash
ctest --test-dir build -R '^qus_layer_norm_test$' --output-on-failure
```

The `[1152,1]`, `[1152,256]`, and `[1152,4096]` cases pass the FP64-oracle BF16 reduction gate.
