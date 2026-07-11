# Vision Add-Bias Roofline Optimization

Date: 2026-07-11

Target: `add_bias [3456,4096]` on NVIDIA GeForce RTX 5090.

## Change

The correctness kernel used one scalar element per thread and evaluated `i % d` for every element,
launching 55,296 blocks. The optimized path dispatches a two-dimensional channel-tile/token grid,
uses BF16x2 loads/stores, and processes four pairs per thread. Odd or unaligned shapes retain the
scalar fallback.

## Timing

| Version | Median | Effective bandwidth |
| --- | ---: | ---: |
| scalar baseline | 41.54 us | 1363.1 GB/s |
| BF16x2 tiled | 36.41 us | 1555.2 GB/s |

The median improves by 12.4%; the benchmark reaches 86.8% of the fixed 1792 GB/s DRAM roofline.

## NCU

Preflight:

```bash
~/.codex/skills/ncu-kernel-profile/scripts/preflight.sh
```

Baseline and optimized captures use one matched launch after 20 warmups:

```bash
ncu --force-overwrite \
  -o profiles/vision-ops/add-bias/baseline_detailed.ncu-rep \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --kernel-name regex:'add_bias_kernel' --launch-skip 20 --launch-count 1 \
  build/bench/qus_add_bias_bench

ncu --force-overwrite \
  -o profiles/vision-ops/add-bias/optimized_detailed.ncu-rep \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --kernel-name regex:'add_bias_bf16x2_kernel' --launch-skip 20 --launch-count 1 \
  build/bench/qus_add_bias_bench
```

| Metric | Baseline | Optimized |
| --- | ---: | ---: |
| matched grid | `55296×256` | `(2,4096)×256` |
| NCU duration | 67.30 us | 46.62 us |
| max memory throughput | 44.87% | 87.47% |
| DRAM throughput | 44.87% | 62.60% |
| L2 throughput | 21.95% | 87.47% |
| achieved occupancy | 83.71% | 76.20% |
| local/shared spilling | 0 | 0 |

NCU identifies memory as the limiting pipeline and reports greater than 80% utilization of the
available memory performance. The lower DRAM percentage is expected for the repeated-buffer
microbenchmark because 79.15% of L2 accesses hit; the independent timing harness still measures
86.8% of the hardware DRAM roofline as effective traffic.

## Correctness

```bash
ctest --test-dir build -R '^qus_vision_elementwise_test$' --output-on-failure
```

The real Vision dimensions 1152, 3456, and 5120 pass the FP64-oracle BF16 elementwise gate.
