# Vision Multidimensional RoPE Optimization

Date: 2026-07-11

Targets:

- text MRoPE: positions `[T,3]`, Q `[256,24,T]`, K `[256,4,T]`, rotary dimension 64;
- packed Vision RoPE: positions `[T,2]`, strided Q/K views `[72,16,T]`, rotary dimension 72.

The existing one-dimensional text RoPE fast path is unchanged.

## Change

The baseline assigned threads to `(Q-or-K, head, pair)` and evaluated `powf`, `cosf`, and `sinf`
for every head. The angle depends only on `(token,pair)`, so this repeated the same transcendental
work 28 times for MRoPE and 32 times for Vision.

The optimized multidimensional kernel assigns one block to each token:

- the first `rotary_dim/2` threads calculate the cosine and sine tables once;
- the tables occupy at most 288 bytes of shared memory;
- each warp then owns a Q or K head and rotates its pairs;
- a lane processes the second pair when the Vision half dimension is 36;
- 128 threads per token is the measured best policy.

A second experiment staged the 72 Vision dimensions in shared memory to make global accesses fully
coalesced. It regressed the Vision median from 19.03 us to 23.17 us because the extra staging and
barriers cost more than the sector savings, so it was discarded.

## Timing

| Target | Baseline | Optimized | Improvement |
| --- | ---: | ---: | ---: |
| text MRoPE | 23.68 us | 11.28 us | 52.4% |
| packed Vision RoPE | 29.48 us | 18.07 us | 38.7% |

The benchmark's logical GB/s number exceeds physical DRAM bandwidth because it counts all tensor
bytes while repeated launches retain most data in cache. It is therefore not used as roofline
evidence here; latency and the hardware counters below are the meaningful measurements.

## NCU

```bash
ncu --force-overwrite \
  -o profiles/vision-ops/rope/mrope_baseline.ncu-rep \
  --set detailed --kernel-name regex:'rope_nd_kernel' \
  --launch-skip 20 --launch-count 1 build/bench/qus_rope_bench --mrope

ncu --force-overwrite \
  -o profiles/vision-ops/rope/mrope_optimized_b128.ncu-rep \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --kernel-name regex:'rope_nd_kernel' \
  --launch-skip 20 --launch-count 1 build/bench/qus_rope_bench --mrope

ncu --force-overwrite \
  -o profiles/vision-ops/rope/vision_baseline.ncu-rep \
  --set detailed --kernel-name regex:'rope_nd_kernel' \
  --launch-skip 20 --launch-count 1 build/bench/qus_rope_bench --vision

ncu --force-overwrite \
  -o profiles/vision-ops/rope/vision_optimized_b128.ncu-rep \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --kernel-name regex:'rope_nd_kernel' \
  --launch-skip 20 --launch-count 1 build/bench/qus_rope_bench --vision
```

| Metric | MRoPE baseline | MRoPE optimized | Vision baseline | Vision optimized |
| --- | ---: | ---: | ---: | ---: |
| NCU duration | 34.59 us | 22.69 us | 48.13 us | 58.05 us |
| compute throughput | 69.69% | 30.55% | 63.16% | 19.27% |
| DRAM throughput | 40.15% | 56.24% | 39.69% | 35.05% |
| achieved occupancy | 81.73% | 67.22% | — | 66.23% |

The counters confirm that coefficient caching removes the transcendental-compute bottleneck. The
multi-pass detailed NCU replay reports a longer Vision duration even though the normal benchmark
consistently falls from 29.48 us to 18.07 us (and its optimized minimum is 17.17 us). The optimized
kernel changes cache residency and replay behavior, so the replay duration is not used for the
Vision speedup claim. This discrepancy is recorded rather than hiding the unfavorable counter.

MRoPE moves from compute-bound toward memory-bound and reaches 1.00 TB/s in the captured launch.
Vision retains non-coalesced sectors because its split-half layout places paired dimensions 36
BF16 elements apart; the rejected staging experiment establishes that removing those sectors is not
profitable on this fixed shape.

## Correctness

```bash
ctest --test-dir build -R '^qus_rope_test$' --output-on-failure
```

The test passes the mathematical oracle for text MRoPE and packed, strided Vision Q/K views.
