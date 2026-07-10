# Attention codec-grouped input projection

Date: 2026-07-11  
GPU: NVIDIA GeForce RTX 5090 (SM 12.0)  
Primary shape: `T=1024`, `K=5120`, Q4 `N=6144+1024`, Q5 `N=6144+1024`  
Weight artifact: `out/qwen3_6_27b.q5090_w4g64_mixed_v4_1.qus`

## Result

Replace the single runtime-mixed Q4/Q5 Attention input-projection launch with two
codec-specialized grouped launches:

1. Q4 launch: Q `[6144,5120]` plus K `[1024,5120]`;
2. Q5 launch: Gate `[6144,5120]` plus V `[1024,5120]`.

The production mean is **799.258 us** per Attention layer at `T=1024`, compared with
**834.426 us** for the previous mixed launch: **4.21% faster**. The new path also matches the
independently tuned two-GEMM baseline of **800.448 us** within measurement noise and is 0.15%
faster by the 16-call Nsys mean.

## Why codec grouping wins

The previous single launch used `WN=16`, eight warps, and a runtime-uniform `job.q5` branch. That
kept Q4 and Q5 tiles in one scheduler pool, but the compiled kernel had to carry both unpack paths
and could not use the higher-payload `WN=32` warp tile without excessive mixed-codec register
pressure.

The new template has three explicit modes: `Mixed`, `Q4`, and `Q5`. Attention instantiates only the
compile-time Q4 and Q5 modes with:

- CTA tile `BM=64`, `BN=128`, `BK=64`;
- warp tile `WM=64`, `WN=32`;
- four warps / 128 threads;
- two `cp.async.cg` stages;
- paired 4-byte scale loads;
- no runtime codec branch in code/high-plane staging or dequantization;
- two jobs per launch, selected uniformly by `blockIdx.x`.

Each codec launch has grid `(112,8,1)`: 96 large-projection tiles plus 16 KV-projection tiles.
The Q4 specialization removes the unused Q5 high plane from shared memory. GDN continues to
instantiate the original `Mixed`, four-job-capable, `WN=16` kernel and is not dispatched through
the Attention configuration.

This design deliberately trades one additional graph node for materially higher tensor-core
payload. It still reduces four logical projections to two launches, while keeping each launch
large enough to fill the GPU.

## T=1024 timing

Dedicated full-model Nsys capture, 16 Attention layers:

| Path | Calls | Mean | Median | Registers | Static shared |
|---|---:|---:|---:|---:|---:|
| old mixed Q4/Q5 launch | 16 | 834.426 us | 826.000 us | 119 | 46,592 B |
| new Q4 Q+K launch | 16 | 389.949 us | 392.045 us | 160 | 45,568 B |
| new Q5 Gate+V launch | 16 | 409.309 us | 409.837 us | 166 | 46,592 B |
| new codec-grouped sum | 32 | **799.258 us** | **801.882 us** | - | - |

Fresh standalone warm medians for the independently tuned `[7168,5120]` kernels are:

| Baseline | Warm median |
|---|---:|
| Q4 `[7168,5120]` | 389.632 us |
| Q5 `[7168,5120]` | 410.816 us |
| sum | **800.448 us** |

The Nsys production mean is the acceptance timing because it measures the actual separate-output
job mapping in the model. Standalone CUDA-event timing establishes the target rather than replacing
the in-model measurement.

## NCU roofline evidence

| Metric | Q4 Q+K | Q5 Gate+V |
|---|---:|---:|
| tensor-pipe active / elapsed peak | **84.20%** | **81.96%** |
| Compute (SM) throughput | **84.20%** | **81.96%** |
| DRAM throughput SOL | 3.12% | 3.50% |
| achieved occupancy | 16.26% | 16.38% |
| registers/thread | 160 | 166 |
| static shared memory | 45,568 B | 46,592 B |
| local memory/thread | 0 B | 0 B |

The low occupancy is intentional: each `WN=32` warp performs twice the MMA payload of the old
`WN=16` warp. The resulting 82-84% tensor-pipe utilization shows that additional occupancy is not
the limiter. The kernels are compute/dequantization bound and far from the DRAM roofline.

## Length check

The same codec-specialized path was checked at `T=128`, `512`, and `1024`. Dedicated Nsys medians
for Q4+Q5 are 235.263 us at `T=128` and 520.381 us at `T=512`; the corresponding independent warm
baselines are 251.200 us and 531.040 us. Thus the new path does not buy the `T=1024` result by
regressing the representative shorter prefill lengths. `T<=16` retains the existing ordinary
linear dispatch.

## Correctness and isolation

- Focused numerical tests at `T=17`, `128`, and `129` report `max_abs=0` for Q, Gate, K, and V.
- `compute-sanitizer --tool memcheck` on the focused prefill-fusion suite reports
  `ERROR SUMMARY: 0 errors`.
- The same full-model capture shows GDN unchanged at grid `(160,8,1)`, block 256, 119
  registers/thread, and 46,592 bytes static shared memory.

## Artifacts

- Production Nsys:
  `profiles/dense-gdn-roofline-20260711/pp1024-attn-codec-grouped-v1.nsys-rep`
- Production JSON:
  `profiles/dense-gdn-roofline-20260711/pp1024-attn-codec-grouped-v1.json`
- Length sweep Nsys:
  `profiles/dense-gdn-roofline-20260711/attn-codec-grouped-sweep-v1.nsys-rep`
- Q4 NCU:
  `profiles/dense-gdn-roofline-20260711/attn-codec-grouped-q4-v1.ncu-rep`
- Q5 NCU:
  `profiles/dense-gdn-roofline-20260711/attn-codec-grouped-q5-v1.ncu-rep`
- Standalone baselines:
  `profiles/dense-gdn-roofline-20260711/attn-grouped-q4-independent-t1024.csv` and
  `profiles/dense-gdn-roofline-20260711/attn-grouped-q5-independent-t1024.csv`
