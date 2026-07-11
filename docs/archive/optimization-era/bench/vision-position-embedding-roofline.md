# Vision Position-Embedding Optimization

Date: 2026-07-11

Target: fused four-corner position interpolation and BF16 residual add at `[1152,4096]`.

## Change

The baseline flattened all `P×D` elements. Every thread divided/modulo by 1152 and independently
reloaded the same four indices and weights, so each patch's control data was fetched 1152 times.

The optimized fixed-shape path assigns one block to one patch:

- four indices and weights are loaded once into 32 bytes of shared memory;
- 128 threads process the 576 BF16 channel pairs;
- each corner table row is read coalesced along channels;
- interpolation is still rounded to BF16 before the residual add;
- the generic scalar fallback remains available.

Block sizes 256, 128, and 64 measured 11.16 us, 10.97 us, and 11.06 us respectively; 128 is the
selected policy.

## Timing

| Version | Median |
| --- | ---: |
| flattened scalar baseline | 22.45 us |
| patch-tiled BF16x2 | 10.97 us |

Median latency improves by 51.1%. A simple byte-count roofline is not reported because the four
position rows are intentionally reused through L2 and the logical interpolation traffic is larger
than physical DRAM traffic.

## NCU

```bash
ncu --force-overwrite \
  -o profiles/vision-ops/pos-embed/baseline.ncu-rep \
  --set detailed --kernel-name regex:'vision_pos_embed_add_kernel' \
  --launch-skip 20 --launch-count 1 build/bench/qus_vision_pos_embed_bench

ncu --force-overwrite \
  -o profiles/vision-ops/pos-embed/optimized_b128.ncu-rep \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --kernel-name regex:'vision_pos_embed_add_d1152_kernel' \
  --launch-skip 20 --launch-count 1 build/bench/qus_vision_pos_embed_bench
```

| Metric | Baseline | Optimized |
| --- | ---: | ---: |
| NCU duration | 35.81 us | 18.50 us |
| grid | `18432×256` | `4096×128` |
| DRAM throughput | 36.60% | 64.52% |
| L2 throughput | 29.34% | 55.46% |
| memory throughput | 654 GB/s | 1.15 TB/s |
| L2 hit rate | 73.23% | 73.36% |
| achieved occupancy | 89.98% | 86.71% |
| local/shared spilling | 0 | 0 |

The final kernel is memory-bound. Its remaining gap to raw DRAM peak is explained by the required
four-row gather and high L2 reuse; the work-efficient latency has already fallen by roughly half.

## Correctness

```bash
ctest --test-dir build -R '^qus_vision_pos_embed_test$' --output-on-failure
```

The real 1152-channel path passes the FP64 oracle, including the mandatory intermediate BF16
rounding point.
