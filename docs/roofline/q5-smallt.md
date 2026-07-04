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
| 27 | Current-state detailed diagnosis after final8. | MlpDown original detailed: 73.79 us / 56.81% SOL, memory 45.08%, regs 58, static smem 10.75 KiB, waves/SM 0.94, achieved occupancy 61.25%, branch efficiency 74.03%. Attn chunk4 detailed: 31.23 us / 67.27% SOL, memory 43.86%, regs 48, static smem 1.41 KiB, waves/SM 4.22, achieved occupancy 75.91%, branch efficiency 91.38%. Attn SchedulerStats: issued warp/scheduler 0.75, no eligible 25.26%, eligible warps/scheduler 3.42. | Diagnostic. MlpDown remains the worse low-wave path; chunk4 is already much healthier, and attempts that only raise occupancy need to preserve the 48-register ILP schedule. |
| 28 | Route MlpDown to the current one-row chunk4 kernel. | MlpDown: 78.05 us / 61.87% SOL, memory 47.34%, regs 48, static smem 1.41 KiB, waves/SM 3.01, achieved occupancy 75.65%. | Rejected. More waves and higher SOL did not offset the synchronization/reduction cost over 17 slabs, so duration materially regressed versus the original path. |
| 29 | Route MlpDown through the original kernel with `kTt=2`, keeping 8 row-warps/block. | MlpDown: 105.89 us / 56.15% SOL, memory 31.40%, regs 48, static smem 10.75 KiB, waves/SM 1.51. | Rejected. Halving per-warp accumulator state raised occupancy but streamed the Q5 weights twice and lost too much bandwidth. |
| 30 | Add chunk4-only `__launch_bounds__(128, 12)` to force 12 resident 128-thread blocks/SM. | Attn: 34.11 us / 62.31% SOL, memory 41.85%, regs 40, theoretical occupancy 100%, achieved occupancy 89.85%, waves/SM 3.51. | Rejected. The register cap improved occupancy but removed the ILP needed by this body. |
| 31 | Add chunk4-only `__launch_bounds__(128, 11)` as a less aggressive occupancy cap. | Attn: 34.59 us / 62.56% SOL, memory 41.62%, regs 40, theoretical occupancy 100%, achieved occupancy 89.70%, waves/SM 3.51. | Rejected. Ptxas still capped to 40 registers and reproduced the launch-bounds regression. |
| 32 | Route MlpDown through the original kernel with one cp.async stage and 8 row-warps/block. | Repeat sample: 69.15 us / 56.87% SOL, memory 48.09%, regs 58, static smem 5.38 KiB, waves/SM 0.94. | Rejected. Duration improved versus recent two-stage samples, but the primary SOL metric moved down and the change did not address the low-wave limiter. |
| 33 | Route MlpDown through the original kernel with one cp.async stage and 4 row-warps/block. | MlpDown: 69.34 us / 56.11% SOL, memory 47.93%, regs 58, static smem 2.69 KiB, waves/SM 0.94. | Rejected. Smaller blocks did not increase wave count and further reduced SOL. |
| 34 | Replace MlpDown's staged cp.async small-T path with an exact-shape direct global-load Q5 T=4 body. One warp still owns one row and keeps the same fp32 FMA order, but codes/high/scales are loaded directly from global memory with per-group scale broadcast. | Final9 rerun: MlpDown 62.11 us / 59.97% SOL, memory 53.55%, regs 54, static smem 0 B, waves/SM 0.94, achieved occupancy 62.16%. | Accepted. For the long-K MlpDown shape, removing shared staging and warp sync cuts duration materially while preserving/slightly improving SOL versus final8's original path. |
| 35 | Route AttnInQKV7168x5120 to the existing one-warp direct Q5 T=4 body. | Attn: 37.60 us / 41.50% SOL, memory 36.43%, regs 54, static smem 0 B, waves/SM 1.32, achieved occupancy 57.20%. | Rejected. Removing shared staging without keeping the four-way K split lost too much row parallelism on the short-K shape. |
| 36 | Load the Q5 high-bit plane from staged shared memory as a 32-bit word, then shift each lane's byte, to reduce byte-load shared wavefronts. | Attn: 32.74 us / 66.26% SOL, memory 42.14%, regs 52, static smem 1.41 KiB, waves/SM 4.68, achieved occupancy 67.99%. | Rejected. The shared access pattern changed, but extra integer/live state raised registers and lowered occupancy enough to regress duration. |
| 37 | New direct split4 Q5 T=4 body for Attn: four warps/block, one row/block, each warp loads its 256-K slice directly from global memory and only synchronizes for the final four-part row reduction. | Attn: 28.58 us / 65.27% SOL, memory 47.95%, regs 40, static smem 64 B, waves/SM 3.51, achieved occupancy 88.43%. | Continued. Removing per-slab staging barriers was a large duration win, but the 100% occupancy version had worse partial-wave geometry and lower SOL than chunk4. |
| 38 | Add `__launch_bounds__(128, 10)` to the direct split4 body. | Attn: 27.58 us / 66.20% SOL, memory 49.71%, regs 48, static smem 64 B, waves/SM 4.22, achieved occupancy 74.10%. | Accepted for the split4 path. The occupancy cap restored the 10-block/SM wave geometry and improved duration versus uncapped split4. |
| 39 | Route all three short-K representatives to direct split4 without x preloading. | Attn: 27.84 us / 65.72% SOL; Proj: 23.94 us / 64.20%; Out: 23.97 us / 62.81%; all use regs 48 and 64 B static smem. | Continued. All short-K durations improved materially versus final9 chunk4, but SOL dropped because the direct path is L1TEX-scoreboard limited rather than shared/barrier limited. |
| 40 | Detailed diagnosis of direct split4 Attn. | Detailed Attn: 27.10 us / 66.53% SOL, memory 50.57%, regs 48, 64 B static smem. SchedulerStats/WarpStateStats: issued warp/scheduler 0.72, no eligible 28.36%, L1TEX scoreboard 5.3 cycles, 43.4% of issue interval. | Diagnostic. The new path is faster by removing per-slab sync, but global-load latency remains the dominant stall. |
| 41 | Preload all four T-column x vectors before the split4 FMA chains. | Final10 short-K samples: Attn 27.23 us / 67.24% SOL, memory 50.30%, regs 44; Proj 24.35 us / 65.37%, memory 48.22%; Out 23.58 us / 64.82%, memory 49.84%; all use 64 B static smem. | Accepted for short-K shapes. X preloading improves Attn/Out and keeps Proj materially faster than final9, while preserving the same fp32 accumulation order. |
| 42 | Apply the same x-preload pattern to the one-warp MlpDown direct body. | MlpDown: 62.69 us / 60.35% SOL, memory 53.07%, regs 54, static smem 0 B, waves/SM 0.94. | Rejected. The small SOL gain did not pay for itself in duration. |
| 43 | Route MlpDown to the direct split4 body. | MlpDown: 74.62 us / 65.41% SOL, memory 65.41%, regs 44, static smem 64 B, waves/SM 3.01, achieved occupancy 75.62%. | Rejected. This raises the primary SOL metric but materially regresses duration, violating the representative-shape guardrail. |
| 44 | Launch the one-warp MlpDown direct body with 4 rows/block instead of 8. | MlpDown: 62.88 us / 59.36% SOL, memory 52.86%, regs 54, waves/SM 0.84. | Rejected. Smaller blocks reduced wave count and regressed both duration and SOL. |
| 45 | Group two rows per direct split4 block to recover some within-block x reuse while retaining four K-slice warps per row. | Attn: 27.71 us / 66.48% SOL, memory 49.45%, regs 44, static smem 128 B, waves/SM 4.22, achieved occupancy 74.58%. | Rejected. The larger 256-thread block did not improve L1TEX behavior enough to beat the one-row split4 body. |
| 46 | Stage the four T-column x vectors once per chunk for a two-row direct split4 block, so both rows reuse shared x instead of issuing duplicate global x loads. | Attn: 62.14 us / 78.57% SOL, memory 78.57%, DRAM 22.01%, SM 32.13%, regs 44, static smem 8.32 KiB, waves/SM 4.22. | Rejected. The primary SOL number rose, but shared-memory traffic and two slab barriers per iteration more than doubled duration. |
| 47 | In direct split4, reduce high-bit global load instructions by having one lane per four load a 32-bit high word and broadcast it with `__shfl_sync`. | Attn: 28.06 us / 67.79% SOL, memory 48.77%, regs 44, static smem 64 B, waves/SM 4.22, achieved occupancy 74.05%. | Rejected. Fewer high-bit loads did not offset the shuffle/shift overhead and duration regressed versus final10. |

## Final Candidate

Code change: keep the generic 8-row small-T kernel for Q5 generally, but route
the three full-aligned short-K Q5 T=4 shapes (`7168x5120`, `6144x5120`, and
`5120x6144`) to
`linear_rowsplit_gemm_smallt_kernel_direct_split4_q5_t4<Q5Smallt>`. The split4
kernel uses four warps/block for one output row: each warp loads one 256-K slice
of each 1024-value slab directly from global memory, accumulates the four T
columns in fp32, then the block performs a single final four-part row reduction.
The old staged chunk4 path was removed because it is no longer routed.

MlpDown (`5120x17408`) now routes to
`linear_rowsplit_gemm_smallt_kernel_direct_q5_t4<Q5Smallt,8>`. That exact-shape
body keeps one warp per output row and the same fp32 accumulation order as the
original small-T path, but loads Q5 codes/high bits/scales directly from global
memory and broadcasts one scale per 64-value group instead of staging each
1024-value slab through shared memory. The direct path is only used for the
full-slab Q5 T=4 MlpDown shape.

Profiles:

- `profiles/q5-smallt/final10/AttnInQKV7168x5120.ncu-rep`
- `profiles/q5-smallt/final10/MlpDown5120x17408.ncu-rep`
- `profiles/q5-smallt/final10/Proj6144x5120.ncu-rep`
- `profiles/q5-smallt/final10/Out5120x6144.ncu-rep`

| shape | kernel | duration us | SOL % | SM % | memory % | regs | static smem KiB | waves/SM | achieved occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AttnInQKV7168x5120 | `direct_split4_q5_t4<Q5Smallt>` | 27.23 | 67.24 | 67.24 | 50.30 | 44 | 0.06 | 4.22 | 74.05 |
| MlpDown5120x17408 | `direct_q5_t4<Q5Smallt,8>` | 62.40 | 60.72 | 60.72 | 53.28 | 54 | 0.00 | 0.94 | 62.17 |
| Proj6144x5120 | `direct_split4_q5_t4<Q5Smallt>` | 24.35 | 65.37 | 65.37 | 48.22 | 44 | 0.06 | 3.61 | 73.36 |
| Out5120x6144 | `direct_split4_q5_t4<Q5Smallt>` | 23.58 | 64.82 | 64.82 | 49.84 | 44 | 0.06 | 3.01 | 72.53 |

Correctness:

```bash
cmake --build build --target qus_linear_op_bench qus_linear_test -j
./build/tests/qus_linear_test
# OK linear correctness
compute-sanitizer ./build/tests/qus_linear_test
# ERROR SUMMARY: 0 errors
```

The correctness suite now includes Q5 T=4 checks for the representative
`6144x5120`, `7168x5120`, `5120x6144`, and `5120x17408` shapes.

## Conclusion

The final candidate replaces the staged one-row chunk4 short-K path with a
direct global-load split4 body, while retaining the direct one-warp MlpDown
body. Versus final9, the three short-K durations improve materially: Attn
31.49 -> 27.23 us, Proj 27.14 -> 24.35 us, and Out 27.33 -> 23.58 us. MlpDown
uses the same direct path as final9; the final10 sample is 62.40 us versus the
final9 rerun at 62.11 us, within observed NCU run-to-run spread. Numerical
correctness is unchanged under the existing linear correctness suite and
`compute-sanitizer` reports no errors.

The primary 85% SOL target was not reached. The best final representative SOL is
67.24% on Attn. The direct split4 path proves that, for the short-K shapes,
per-slab shared staging and block barriers were more expensive than direct Q5
global loads, even though the resulting kernel is still L1TEX-scoreboard
limited. The direct MlpDown path proves that long-K staged cp.async overhead can
dominate even when wave count remains low, but split4 MlpDown regresses duration
despite higher SOL. The failed tensor-core, split-by-slab, staged chunk,
pipeline-cleanup, read-only-load, deeper-pipeline, launch-bounds, x-preload
MlpDown, and rows/block experiments indicate that reaching 85% SOL likely
requires a larger architecture change, such as a better-balanced split-K or
tensor-core path that can feed all warps without duplicating dequant/staging
work.
