# Vision Attention Flash Kernel

Date: 2026-07-11

Target: packed, non-causal Qwen3.6 Vision MHA with 16 heads, head dimension 72, independent
`cu_seqlens` segments, BF16 Q/K/V/output, and no quadratic workspace.

## Baseline

The correctness-first kernel launched one 128-thread CTA per `(query,head)`. It scanned keys
serially and performed a shared-memory tree reduction for every QK dot product. Each key required
ten block barriers, Q was reloaded for every dot product, and all QK/PV arithmetic used scalar
FP32 FMA.

| Segment length | Baseline median |
| --- | ---: |
| 256 | 576.29 us |
| 1024 | 11910.75 us |

A first warp-per-query rewrite removed block barriers and cached Q. It reached 100.73 us and
1318.85 us respectively, but NCU still showed a scalar FMA/SFU bottleneck and only about 3 TFLOP/s
at L=256. It was an intermediate experiment, not retained code.

## Final Design

The final kernel is a fixed-shape FlashAttention implementation:

- one CTA owns 64 queries from one segment and one head;
- four warps each own 16 query rows;
- keys and values advance in 64-row tiles;
- Q, K, and V are staged with 16-byte `cp.async` copies into XOR-swizzled shared memory;
- `mma.sync.m16n8k16` computes both QK and probability-times-V with FP32 accumulators;
- online max/sum state and output accumulators stay in registers across key tiles;
- probabilities are rounded to BF16 only at the tensor-core PV boundary;
- no attention-score matrix is written to global memory.

The 72-element head is padded in shared memory to 128 elements so the 8-way XOR swizzle remains
in-bounds. QK executes five contraction steps, covering 80 dimensions; the last eight are zero.
PV emits exactly nine 8-column tiles, or 72 dimensions.

For one segment, tile coordinates are derived directly from the launch grid and there is no
workspace or setup kernel. Packed multi-segment inputs build compact 16-byte descriptors in
`O(ceil(P/64)+S)` workspace; descriptors preserve arbitrary, non-64-aligned boundaries.

## Timing

| Segment length | Baseline | Final | Speedup | Mathematical TFLOP/s | Issued-MMA TFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 | 576.29 us | 11.01 us | 52.3x | 27.4 | 29.0 |
| 1024 | 11910.75 us | 41.87 us | 284.5x | 115.4 | 121.8 |
| 4096 | — | 528.16 us | — | 146.4 | 154.5 |

Mathematical work counts `4 L² D H` FLOPs. Issued-MMA work counts the QK padding explicitly as
`2 L² H (80 + 72)`. The benchmark's GB/s line is only the compulsory Q/K/V/output byte floor;
attention is compute-bound, so TFLOP/s is the relevant readout.

## NCU

```bash
ncu --force-overwrite \
  -o profiles/vision-ops/attention/baseline_l256.ncu-rep \
  --set detailed --kernel-name regex:'vision_attention_reference_kernel' \
  --launch-skip 2 --launch-count 1 build/bench/qus_vision_attention_bench

ncu --force-overwrite \
  -o profiles/vision-ops/attention/final_l256.ncu-rep \
  --set detailed --kernel-name regex:'vision_attention_flash_kernel' \
  --launch-skip 2 --launch-count 1 build/bench/qus_vision_attention_bench

ncu --force-overwrite \
  -o profiles/vision-ops/attention/final_l1024.ncu-rep \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --section InstructionStats --kernel-name regex:'vision_attention_flash_kernel' \
  --launch-skip 2 --launch-count 1 build/bench/qus_vision_attention_bench --large

ncu --force-overwrite \
  -o profiles/vision-ops/attention/final_l4096.ncu-rep \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --section InstructionStats --kernel-name regex:'vision_attention_flash_kernel' \
  --launch-skip 2 --launch-count 1 build/bench/qus_vision_attention_bench --xlarge
```

| Metric | Final L=256 | Final L=1024 | Final L=4096 |
| --- | ---: | ---: | ---: |
| CTA count | 64 | 256 | 1024 |
| NCU duration | 13.70 us | 59.58 us | 730.37 us |
| compute throughput | 13.30% | 50.52% | 67.65% |
| L2 throughput | 14.86% | 41.52% | 49.96% |
| achieved occupancy | 8.29% | 12.67% | 15.89% |
| theoretical occupancy | 16.67% | 16.67% | 16.67% |
| local/shared spilling | 0 | 0 | 0 |

NCU detailed replay inflates absolute duration, especially for the long kernel; normal benchmark
timings above are used for latency and TFLOP/s. The counters establish the limiting regimes:

- L=256 launches only 64 CTAs for 170 SMs and is dominated by fixed launch/underfill cost;
- L=1024 averages 1.5 CTAs per SM and reaches roughly half of compute roofline;
- L=4096 fills the available two-CTA resource limit, reaches 95.3% of its theoretical occupancy,
  and is balanced between tensor compute and L2/shared-memory feed.

Bc=32 was tested to raise theoretical occupancy from 16.7% to 25%. It regressed the NCU L=1024
duration from 59.55 us to 63.52 us and the normal L=4096 median from 527.20 us to 543.62 us because
the extra key-loop barriers outweighed residency. Both row-major and tail-swizzled 80-wide shared
layouts also regressed L=1024 by 7–9%. These variants were discarded. Split-K was not added: it
would add linear partial-output workspace and a reduction launch to improve only underfilled short
segments, while the real long-segment kernel already approaches the fixed resource roofline.

## Correctness and Memory Safety

```bash
ctest --test-dir build -R '^qus_vision_attention_test$' --output-on-failure

compute-sanitizer --tool memcheck --error-exitcode=99 \
  build/tests/qus_vision_attention_test
```

The oracle covers contiguous and packed QKV, mixed segment boundaries, a multi-tile packed case,
and L=256. Relative L2 error remains approximately 0.21–0.23%, within the established BF16
attention tolerance. Compute Sanitizer reports `ERROR SUMMARY: 0 errors`.
