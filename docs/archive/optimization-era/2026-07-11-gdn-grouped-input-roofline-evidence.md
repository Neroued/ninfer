# GDN grouped input projection: baseline and roofline evidence

Date: 2026-07-11  
GPU: NVIDIA GeForce RTX 5090 (SM 12.0)  
Shape: `T=1024`, `K=5120`, Q4 `N=4096`, Q5 `N=6144`  
Weight artifact: `out/qwen3_6_27b.q5090_w4g64_mixed_v4_1.qus`

## Decision

Keep the existing production GDN grouped-input kernel unchanged. It already exceeds the required
two-launch baseline and reaches the measured tensor-core roofline target for this mixed-codec
operation. Three more aggressive architectures were implemented and measured; all three were
strict regressions and were removed rather than retained as dead dispatch or tuning branches.

This is an evidence freeze for the GDN grouped-input stage. Attention grouped input remains a
separate optimization stage and is not changed here.

## Production architecture

The production kernel is
`linear_rowsplit_grouped_input_gemm_mma_kernel<GemmCfg<64,128,64,64,16,2,...>>`:

- one launch combines the Q4 `qk` and Q5 `v` projection grids;
- each CTA owns one homogeneous codec/output tile, so no warp mixes Q4 and Q5 decode work;
- CTA tile is `BM=64`, `BN=128`, `BK=64`, with `WM=64`, `WN=16`;
- eight warps / 256 threads cover the eight `WN=16` slices;
- two `cp.async` stages pipeline activation, code, scale, and Q5 high-bit traffic;
- weights are dequantized to a swizzled BF16 shared-memory tile and consumed through
  `ldmatrix` plus BF16 tensor-core MMA;
- static shared memory is 46,592 bytes, register use is 119 registers/thread, and measured
  achieved occupancy is 32.38%.

For the real GDN shape the launch grid is `(160, 8, 1)`: 64 Q4 row tiles plus 96 Q5 row tiles,
over eight token tiles.

## Acceptance baseline

The comparison baseline is two independently tuned production GEMMs executed separately at the
same shape. Warm CUDA-event medians are:

| Path | Time |
|---|---:|
| independent Q4 `[4096,5120]` | 260.672 us |
| independent Q5 `[6144,5120]` | 413.504 us |
| independent sum | 674.176 us |
| grouped production kernel, Nsys median | **560.469 us** |

The grouped kernel is **16.86% faster** than the independent baseline
(`1.203x` speedup). It therefore clears the GDN grouped-input baseline without requiring a new
production code path.

The final full-model capture contains 48 GDN calls: total 27.654 ms, mean 576.128 us, median
560.469 us, minimum 530.326 us, and maximum 866.703 us. The whole `pp1024` prefill takes
343.614 ms / 2,980.09 tok/s in the same run.

## NCU roofline evidence

The production `WN=16` report records:

| Metric | Result |
|---|---:|
| tensor-pipe active / peak | **75.78%** |
| Compute (SM) throughput | **75.78%** |
| memory throughput SOL | 43.24% |
| DRAM throughput SOL | 10.81% |
| achieved occupancy | 32.38% |
| registers/thread | 119 |
| local-memory spill | 0 bytes/thread |

This is a compute/tensor-pipe-bound mixed dequantization GEMM, not a DRAM-roofline kernel. The
remaining gap is dominated by required Q4/Q5 unpack/scale instructions and CTA synchronization
between dequantization and MMA stages. The acceptance roofline for this exact fused mixed-codec
operation is therefore the measured 75%+ tensor-pipe regime, which the production kernel reaches.
NCU replay duration is not compared with Nsys production timing because replay changes clocks and
launch conditions.

## Rejected architectures

| Experiment | Result | Reason for rejection |
|---|---:|---|
| paired Q4/Q5 CTA, 16 warps, grid `(96,8)` | 852.449 us mean | **48.0% slower**; codec warp groups serialize at CTA barriers and the Q5 tail leaves half the CTA idle |
| homogeneous `BM=128`, 16 warps, grid `(80,8)` | 745.257 us mean | **29.4% slower**; one large CTA loses residency and latency hiding |
| compile-time two-job dispatch | 563.545 us median | **0.52% slower**; registers increase from 119 to 120 with no useful inner-loop reduction |

The experiments show that launch-count and activation-reuse reductions alone are not sufficient.
Preserving two resident 256-thread CTAs and keeping codec work homogeneous are more important for
this shape.

## Verification and artifacts

- Build: `cmake --build build --target qus_linear_test qus_bench -j$(nproc)`
- Numerical oracle: `QUS_LINEAR_TEST_PREFILL_FUSIONS_ONLY=1 ./build/tests/qus_linear_test`
  reports exact equality (`max_abs=0`) for GDN QKV and all related prefill fusions.
- Memory safety: the same focused test under `compute-sanitizer --tool memcheck` reports
  `ERROR SUMMARY: 0 errors`.
- Final production Nsys:
  `profiles/dense-gdn-roofline-20260711/pp1024-gdn-baseline-final.nsys-rep`
- Final production JSON:
  `profiles/dense-gdn-roofline-20260711/pp1024-gdn-baseline-final.json`
- NCU:
  `profiles/q456-prefill-fusions-20260710/final/ncu/grouped_input_wn16.ncu-rep`
- Independent GEMM CSVs:
  `profiles/dense-gdn-roofline-20260711/gdn-qk-q4-t1024.csv` and
  `profiles/dense-gdn-roofline-20260711/gdn-v-q5-t1024.csv`

## Gate to the next stage

GDN grouped input is complete when the production implementation is numerically exact,
sanitizer-clean, faster than the independent baseline, and in the 75%+ tensor-pipe regime. All four
conditions are satisfied. Attention grouped input can now be optimized and committed independently.
