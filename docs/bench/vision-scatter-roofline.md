# Vision Scatter Optimization

Date: 2026-07-11

Target: scatter Vision embeddings `[5120,2040]` into prompt embeddings `[5120,4096]` with
one destination-token index per Vision column.

## Change

The baseline flattened all 10,444,800 BF16 elements. Every element performed 64-bit division and
modulo to recover its source column and channel, and each column index was independently loaded
5120 times.

The optimized path maps one block to one source column:

- thread 0 loads the destination index once and broadcasts it through shared memory;
- 256 threads copy 2560 aligned BF16 pairs along the contiguous channel dimension;
- invalid destination indices skip the complete column;
- an odd-width scalar path uses the same column ownership without division or modulo.

Measured 128-, 256-, and 512-thread variants produced 11.40 us, 11.16 us, and 11.15 us medians.
The 256-thread policy is selected because it has equivalent median latency and the best measured
tail latency.

## Timing

| Version | Median | Improvement |
| --- | ---: | ---: |
| flattened scalar baseline | 37.80 us | — |
| column-tiled BF16x2 | 11.16 us | 70.5% |

The benchmark repeatedly scatters the same 41.8 MB source and destination ranges, so its logical
GB/s benefits from cache residency and exceeds physical DRAM bandwidth. The NCU cold/replayed
hardware counters are used for the bandwidth assessment instead.

## NCU

```bash
ncu --force-overwrite \
  -o profiles/vision-ops/scatter/baseline.ncu-rep \
  --set detailed --kernel-name regex:'scatter_kernel' \
  --launch-skip 20 --launch-count 1 build/bench/qus_scatter_bench

ncu --force-overwrite \
  -o profiles/vision-ops/scatter/optimized_b256.ncu-rep \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --kernel-name regex:'scatter_bf16x2_kernel' \
  --launch-skip 20 --launch-count 1 build/bench/qus_scatter_bench
```

| Metric | Baseline | Optimized |
| --- | ---: | ---: |
| NCU duration | 72.54 us | 24.67 us |
| grid | `40800×256` | `2040×256` |
| compute throughput | 45.97% | 9.33% |
| DRAM throughput | 30.47% | 73.72% |
| memory throughput | 545 GB/s | 1.32 TB/s |
| achieved occupancy | 73.73% | 91.28% |
| local/shared spilling | 0 | 0 |

The optimized kernel is bandwidth-bound. Its required random destination-column order prevents a
single monotonic output stream, while source reads and every individual destination column remain
fully contiguous. At 1.32 TB/s it reaches 73.7% of the hardware DRAM roofline and removes the
address-arithmetic bottleneck.

## Correctness

```bash
ctest --test-dir build -R '^qus_vision_elementwise_test$' --output-on-failure
```

The real 5120-channel scatter path passes the BF16 oracle across three randomized inputs while
preserving every destination column not named by the scatter indices.
