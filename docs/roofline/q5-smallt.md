# Q5 Rowsplit Small-T GEMM Roofline Notes

Date: 2026-07-05

Target kernel regex:

```bash
regex:linear_rowsplit_gemm_smallt_kernel.*Q5Smallt
```

Measurement command shape:

```bash
ncu --force-overwrite --target-processes all --kernel-name-base demangled \
  --set basic --launch-skip 1 --launch-count 1 \
  --kernel-name 'regex:linear_rowsplit_gemm_smallt_kernel.*Q5Smallt' \
  -o profiles/q5-smallt/<attempt>/<shape> \
  ./build/bench/qus_linear_op_bench --shape <shape> --qtype Q5 --t-sweep 4 \
  --warmup 1 --repeat 1 --flush-mib 1 --copy-mib 1 --copy-repeat 1 \
  --stream-ceiling-gbs 1792
```

SpeedOfLight (SOL) below is `max(Compute (SM) Throughput, Memory Throughput)`
from NCU basic.

## Baseline

Profiles:

- `profiles/q5-smallt/baseline/AttnInQKV7168x5120.ncu-rep`
- `profiles/q5-smallt/baseline/MlpDown5120x17408.ncu-rep`
- `profiles/q5-smallt/baseline/Proj6144x5120.ncu-rep`
- `profiles/q5-smallt/baseline/Out5120x6144.ncu-rep`
- `profiles/q5-smallt/baseline/AttnInQKV7168x5120-warpstats.ncu-rep`

| shape | kernel | duration us | SOL % | SM % | memory % | regs | static smem KiB | waves/SM | achieved occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AttnInQKV7168x5120 | `<Q5Smallt,4,8,2>` | 40.03 | 44.60 | 44.60 | 34.27 | 58 | 10.75 | 1.32 | 56.93 |
| MlpDown5120x17408 | `<Q5Smallt,4,8,2>` | 73.25 | 57.28 | 57.28 | 45.29 | 58 | 10.75 | 0.94 | 62.08 |
| Proj6144x5120 | `<Q5Smallt,4,8,2>` | 35.23 | 43.54 | 43.54 | 33.72 | 58 | 10.75 | 1.13 | 60.00 |
| Out5120x6144 | `<Q5Smallt,4,8,2>` | 29.47 | 51.59 | 51.59 | 40.01 | 58 | 10.75 | 0.94 | 61.81 |

Warp-state evidence from Attn baseline:

- Issued warp per scheduler: 0.56.
- No eligible cycles: 43.86%.
- L1TEX scoreboard stall: 6.4 cycles, 52.0% of warp issue interval.
- Theoretical occupancy: 66.67%, register-limited at 58 regs/thread.

Conclusion: the scalar small-T kernel is latency/eligibility limited, not close
to DRAM saturation. Register pressure and about-one-wave launches limit the
amount of latency hiding available.

## Attempts

| attempt | change | key NCU result | conclusion |
| --- | --- | --- | --- |
| 1 | Sequentialize `float2` x conversions in the consume loop. | Attn: 46.59 us, SOL 38.27%, regs 58. | Rejected. Register count did not drop and duration regressed. |
| 2 | Stage each T=4 activation slab once per block in shared memory. | Attn improved to 37.28 us / 51.98% SOL, but MlpDown regressed to 76.32 us and Out to 32.29 us. | Rejected. L1TEX stall dropped, but per-slab block sync and extra shared memory hurt representative shapes. |
| 3 | Attempt 2 plus `__launch_bounds__(...,5)`. | Attn: 38.43 us, regs 48, theoretical occ 83.33%; MlpDown: 82.40 us. | Rejected. Higher occupancy did not translate to throughput; ILP loss/reg cap cost dominated. |
| 4 | Add explicit L1 prefetches before dequant. | Attn: 129.41 us, regs 72. | Rejected. Prefetch instructions inflated live state and made the kernel much slower. |
| 5 | Launch-bounds cap on the original global-load consume path. | Attn: 66.75 us, regs 48, achieved occ 77.65%. | Rejected. Occupancy alone was worse than baseline. |
| 6 | Use 16 row-warps per block globally. | Attn: 36.42 us; MlpDown: 72.70 us; Proj: 35.26 us; Out: 31.17 us. | Rejected as a global default because Out materially regressed. |
| 7 | Use 4 row-warps per block globally. | Out: 30.02 us, SOL 51.09%. | Rejected. Out still did not beat the 8-row baseline. |
| 8 | First tensor-core Q5 T=4 body inside `<Q5Smallt,4,8,2>`. Variants used 8 warp-tiles/block, 1 warp-tile/block, then an explicit specialization to remove unused scalar shared memory. | Attn: 589.63 us / ~3.7% SOL for the first grid, 306.85 us / 8.30% SOL for the 1-warp-tile grid, 305.44 us / 8.35% SOL after specialization. | Rejected. The tensor-core math could not amortize per-K64 dequant/staging and only one compute warp was effectively active. |
| 9 | Scalar dual-row warp: one warp accumulates two output rows to reuse each x vector. | Attn: 36.99 us, SOL 41.44%, regs 78, static smem 21.50 KiB. | Rejected. Runtime was only marginally better on Attn while SOL, occupancy, register pressure, and shared memory all moved the wrong way. |
| 10 | Split-K by alternating 1024-value slabs, 2 warps per row, applied globally to full-aligned Q5 T=4. | Attn: 37.25 us / 48.03% SOL; MlpDown: 75.65 us / 54.03%; Proj: 32.13 us / 46.97%; Out: 31.65 us / 47.77%. | Rejected globally. It helped Attn/Proj but materially regressed MlpDown/Out. |
| 11 | Attempt 10 only for Attn/Proj; use 4-row fallback for MlpDown/Out. | MlpDown fallback: 74.88 us / 56.03% SOL; Out fallback: 30.05 us / 49.94% SOL. | Rejected. The 4-row fallback still regressed MlpDown versus the original 8-row path. |
| 12 | Split-K by alternating slabs with 4 warps per row. | Attn: 41.09 us, SOL 60.20%, regs 70, static smem 10.88 KiB. | Rejected. Higher memory SOL came from extra waves and imbalance, but elapsed time regressed. |
| 13 | Cooperative chunk split: 4 warps per row, one warp stages the Q5 slab once, each warp consumes one 256-wide chunk, then block-reduces four fp32 partials. | Attn: 32.38 us / 65.54% SOL, regs 48, static smem 2.82 KiB; Proj: 28.58 us / 64.32% SOL. | Accepted as the new architecture for the K=5120 Q5 T=4 shapes. |
| 14 | Keep chunk split and original 8-row fallback in the same exact specialization using a noinline fallback wrapper. | Attn: 62.85 us, SOL 32.77%, regs 156, static smem 13.57 KiB. | Rejected. The fallback wrapper inflated register use and occupancy collapsed. |
| 15 | Route both one-wave shapes, MlpDown and Out, to the existing chunk4 kernel after fresh basic/detailed NCU confirmed the original Out path had grid size 640 and only 0.94 waves/SM. | MlpDown chunk4: 76.86 us / 62.14% SOL, regs 48, static smem 2.82 KiB, waves/SM 3.01; Out chunk4: 28.77 us / 61.08% SOL, regs 48, static smem 2.82 KiB, waves/SM 3.01. | Accepted only for Out. MlpDown gained occupancy/SOL but regressed duration versus the original path. |
| 16 | Two-warp cooperative chunk split: 2 warps per row, 4 rows/block, one warp stages the slab and each warp consumes 512 K-values. | MlpDown: 98.27 us / 56.59% SOL, regs 56, static smem 5.50 KiB, waves/SM 1.88; Out: 35.58 us / 45.42% SOL. | Rejected. The half split added synchronization and register pressure without enough parallelism to beat either the original or chunk4 paths. |
| 17 | Keep chunk4 but have only the staging warp issue `cp.async` commit/wait groups; non-staging warps rely on the following block barrier. | Attn: 31.71 us / 65.37% SOL; Proj: 28.61 us / 63.16% SOL; Out: 28.58 us / 60.50% SOL. | Rejected. The Out gain was marginal and Proj slowed, so the original chunk4 pipeline structure was restored. |
| 18 | Route Out5120x6144 to chunk4, keep MlpDown on the original 8-row kernel, and leave chunk4 internals unchanged. | Final6: Out 28.74 us / 61.24% SOL; Attn 31.39 us / 67.15% SOL; MlpDown 70.59 us / 59.14% SOL; Proj 28.29 us / 63.70% SOL. | Accepted. Out is no longer low-wave limited on the original path, while MlpDown avoids the chunk-split duration regression. |
| 19 | Detailed MlpDown diagnosis with SchedulerStats/WarpStateStats. | Detailed basic: 72.35 us / 57.42% SOL, grid 640, waves/SM 0.94. Warp stats: issued warp/scheduler 0.61, no eligible 38.83%, L1TEX scoreboard 6.1 cycles, 49.3% of issue interval. | Diagnostic. The remaining original path is both low-wave and L1TEX-scoreboard limited; previous x-staging/chunk-split variants raised occupancy at too much synchronization cost. |
| 20 | Replace chunk4 full-block barriers with row-local named `bar.sync` barriers. | Attn: 33.89 us / 66.69% SOL, regs 51, barrier-limited to 8 blocks; Proj: 30.05 us / 65.09%; Out: 29.73 us / 62.53%. | Rejected. Narrower barrier scope increased register pressure and made barriers the occupancy limiter. |
| 21 | Route only MlpDown to the original kernel with 16 row-warps/block. | MlpDown: 74.53 us / 55.80% SOL, static smem 21.50 KiB, waves/SM 0.94. | Rejected. Same wave count, higher shared memory, and slower than the 8-row path. |
| 22 | Route only MlpDown to the original kernel with three cp.async stages. | MlpDown: 91.07 us / 46.16% SOL, regs 62, static smem 16.13 KiB. | Rejected. Deeper prefetching increased live state/shared memory and did not hide the scoreboard stalls. |
| 23 | Load aligned x vectors with `__ldg`. | MlpDown: 82.11 us / 50.61% SOL, regs 60; Attn: 31.62 us / 66.86%; Proj: 28.67 us / 63.83%; Out: 28.67 us / 61.66%. | Rejected. The read-only load path materially regressed MlpDown and did not consistently improve chunk4 shapes. |
| 24 | Remove chunk4 row-tail predicates for the exact even-N routed shapes. | Attn: 32.32 us / 65.28% SOL; Proj: 28.00 us / 64.36%; Out: 29.38 us / 59.51%. | Rejected. Branch removal did not reduce registers and regressed Attn/Out. |
| 25 | Direct one-row chunk4 body with one row of shared memory and no row-local indexing. | Final7: Attn 32.35 us / 64.46% SOL; Proj 29.73 us / 60.63%; Out 30.24 us / 57.41%; regs 44, static smem 1.41 KiB. | Rejected. Lower register count hurt scheduling/codegen and lost the cont11 speedup. |
| 26 | One-row chunk4 launch: 4 warps/block, one output row per block, normal block barriers over only the cooperating row-warps, retaining the measured-fast parameterized body shape. | Final8: Attn 31.36 us / 68.45% SOL; Proj 27.26 us / 67.39%; Out 27.42 us / 64.27%; all chunk4 shapes use regs 48 and static smem 1.41 KiB. | Accepted. This halves chunk4 shared memory and barrier participants while preserving occupancy and improving all chunk4-routed representative shapes versus final6. |

## Final Candidate

Code change: keep the original 8-row small-T kernel for Q5 generally, but route
the three full-aligned Q5 T=4 shapes that benefit from chunk parallelism
(`7168x5120`, `6144x5120`, and `5120x6144`) to
`linear_rowsplit_gemm_smallt_kernel_chunk4<Q5Smallt,4,4,2>`. The chunk kernel
uses 4 warps/block as four cooperative warps for one output row: one warp stages
the Q5 slab, four warps consume the four 256-wide chunks in parallel, and the
block reduces four fp32 partials before storing bf16 output. MlpDown stays on
the original `<Q5Smallt,4,8,2>` kernel because the chunk-split and launch-shape
variants regressed its duration.

Profiles:

- `profiles/q5-smallt/final8/AttnInQKV7168x5120.ncu-rep`
- `profiles/q5-smallt/final8/MlpDown5120x17408.ncu-rep`
- `profiles/q5-smallt/final8/Proj6144x5120.ncu-rep`
- `profiles/q5-smallt/final8/Out5120x6144.ncu-rep`

| shape | kernel | duration us | SOL % | SM % | memory % | regs | static smem KiB | waves/SM | achieved occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AttnInQKV7168x5120 | `kernel_chunk4<Q5Smallt,4,4,2>` | 31.36 | 68.45 | 68.45 | 43.88 | 48 | 1.41 | 4.22 | 75.49 |
| MlpDown5120x17408 | `<Q5Smallt,4,8,2>` | 71.36 | 58.94 | 58.94 | 46.60 | 58 | 10.75 | 0.94 | 61.92 |
| Proj6144x5120 | `kernel_chunk4<Q5Smallt,4,4,2>` | 27.26 | 67.39 | 67.39 | 43.21 | 48 | 1.41 | 3.61 | 74.65 |
| Out5120x6144 | `kernel_chunk4<Q5Smallt,4,4,2>` | 27.42 | 64.27 | 64.27 | 42.84 | 48 | 1.41 | 3.01 | 74.41 |

Correctness:

```bash
cmake --build build --target qus_linear_op_bench qus_linear_test -j
./build/tests/qus_linear_test
# OK linear correctness
```

The correctness suite now includes Q5 T=4 checks for the representative
`6144x5120`, `7168x5120`, `5120x6144`, and `5120x17408` shapes.

## Conclusion

The final candidate preserves the original 8-row kernel for MlpDown, where
chunk splitting did not hold up, and uses the one-row cooperative chunk split
for the K=5120 shapes plus Out5120x6144. Versus final6, the one-row chunk route
raises Attn SOL from 67.15% to 68.45%, Proj from 63.70% to 67.39%, and Out from
61.24% to 64.27%, with lower static shared memory per chunk4 block. Numerical
correctness is unchanged under the existing linear correctness suite.

The primary 85% SOL target was not reached. The best final representative SOL is
68.45% on Attn. The chunk split proves that the original warp-per-row body was
under-parallelized for several small-T shapes, but MlpDown remains limited by the
original low-wave, issue-eligibility behavior and the chunk-split variants raise
occupancy at too much per-slab synchronization cost. The failed tensor-core,
split-by-slab, chunk2, pipeline-cleanup, read-only-load, and deeper-pipeline
experiments indicate that reaching 85% SOL likely requires a larger architecture
change, such as a better-balanced split-K or tensor-core path that can feed all
warps without duplicating dequant/staging work.
