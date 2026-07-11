# Vision GELU Roofline Optimization

Date: 2026-07-11

Target shapes: tanh-approximate and exact GELU over `[4304,4096]` on RTX 5090.

## Change

Both correctness kernels launched one thread per BF16 value. The optimized path processes four
BF16x2 pairs per thread, reducing the grid from 68,864 to 8,608 blocks and amortizing address and
launch scheduling overhead. The scalar path remains for unaligned buffers; an odd final value is
handled explicitly. The tanh and exact formulas remain distinct and retain their original FP32
operation order before BF16 rounding.

## Timing

| Mode | Baseline median | Optimized median | Effective roofline |
| --- | ---: | ---: | ---: |
| tanh approximate | 43.35 us | 43.82 us | 89.8% |
| exact erf | 45.66 us | 43.87 us | 89.7% |

The tanh timing difference is within run-to-run noise; exact improves by 3.9%. Both modes are at
approximately 90% of the fixed bandwidth roofline in the repeated-buffer timing harness.

## NCU

```bash
ncu --force-overwrite \
  -o profiles/vision-ops/gelu/tanh_optimized.ncu-rep \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --kernel-name regex:'gelu_bf16x2_kernel' --launch-skip 20 --launch-count 1 \
  build/bench/qus_gelu_bench

ncu --force-overwrite \
  -o profiles/vision-ops/gelu/exact_optimized.ncu-rep \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --kernel-name regex:'gelu_bf16x2_kernel' --launch-skip 20 --launch-count 1 \
  build/bench/qus_gelu_bench --exact
```

| Metric | tanh baseline | tanh optimized | exact baseline | exact optimized |
| --- | ---: | ---: | ---: | ---: |
| NCU duration | 73.66 us | 57.95 us | 68.99 us | 58.37 us |
| max memory throughput | 50.09% | 90.04% | 32.21% | 89.17% |
| DRAM throughput | 50.09% | 62.16% | 32.21% | 66.24% |
| compute throughput | 57.51% | 21.24% | 65.63% | 25.13% |
| achieved occupancy | 82.30% | 82.05% | 82.40% | 82.13% |
| local/shared spilling | 0 | 0 | 0 | 0 |

The optimized kernels are memory-pipeline limited. NCU reports more than 89% of the available
memory performance for both required GELU definitions, so further transcendental approximation is
not justified and would unnecessarily change the numerical contract.

## Correctness

```bash
ctest --test-dir build -R '^qus_vision_elementwise_test$' --output-on-failure
```

Both formulas pass the FP64-oracle BF16 elementwise gate at the real 4304-channel Vision MLP shape.
