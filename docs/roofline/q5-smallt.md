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
| 48 | Specialize direct split4 for the exact short-K shapes: template the full slab count and K stride, then fully unroll the 5- or 6-slab loop. | Final11: Attn 25.98 us / 68.75% SOL, memory 52.71%, regs 48; Proj 22.91 us / 66.51%, memory 51.22%; Out 22.85 us / 64.56%, memory 51.46%; MlpDown unchanged direct path 62.05 us / 60.22%. | Accepted. Exact shape constants improve instruction scheduling and memory throughput for all short-K durations without materially regressing MlpDown. |
| 49 | Detailed final11 diagnosis for exact direct split4 Attn. | Attn detailed: 25.28 us / 69.15% SOL, memory 54.26%, L1/TEX 48.92%, L2 34.16%, regs 48, waves/SM 4.22, achieved occupancy 73.66%. SchedulerStats: issued warp/scheduler 0.75, no eligible 25.37%, eligible warps/scheduler 3.13. WarpStateStats: L1TEX scoreboard 3.9 cycles, 32.68% of the issue interval. InstructionStats: 4,587,520 fused and 1,748,992 non-fused FP32 instructions. PM sampling did not produce source attribution because the kernel is too short for the minimum sampling interval. | Diagnostic. The current short-K split4 limiter is still L1TEX scoreboard latency plus FP32 instruction issue; sparse output stores are visible in the memory tables but are small byte volume. |
| 50 | Accumulate each 8-value lane chunk as unscaled dot products for the four T columns, then apply the Q5 scale once per T accumulator to reduce non-fused FP32 scale multiplies. | Attn: 26.21 us / 67.48% SOL, memory 52.27%, regs 48, static smem 64 B, waves/SM 4.22, achieved occupancy 74.74%. | Rejected. The rewrite did not reduce register allocation and regressed both duration and SOL versus final11. |
| 51 | Stage the exact shape's 16 Q5 scale halfwords per slab into shared memory once per block, replacing sparse per-warp scale global loads and `__shfl_sync` broadcasts. | Attn: 25.44 us / 65.64% SOL, memory 53.84%, regs 48, static smem 224 B, waves/SM 4.22, achieved occupancy 74.13%. | Rejected. Duration improved slightly, but the primary SOL metric regressed materially versus final11, so the change moves away from the utilization target. |
| 52 | Detailed final11 diagnosis for MlpDown's one-warp direct path. | MlpDown detailed: 62.78 us / 59.60% SOL, memory 53.00%, L1/TEX 36.20%, L2 33.50%, regs 54, waves/SM 0.94, achieved occupancy 62.84%. SchedulerStats: issued warp/scheduler 0.62, no eligible 37.80%, eligible warps/scheduler 1.73. WarpStateStats: L1TEX scoreboard 6.8 cycles, 56.20% of the issue interval. | Diagnostic. One warp per row leaves too few independent row-warps for the 5120-row long-K shape, so L1TEX latency dominates even though split4 had already shown too much row-split overhead. |
| 53 | Route MlpDown to a direct split2 exact-shape body: two warps per row, each warp covers two 256-K chunks per slab, then a two-part row reduction writes the four T outputs. | Final12: MlpDown 59.04 us / 65.88% SOL, memory 58.71%, regs 64, static smem 32 B, waves/SM 1.88, achieved occupancy 58.70%. Short-K final12 samples stayed within run-to-run spread: Attn 25.31 us / 69.11%, Proj 22.91 us / 65.17%, Out 23.20 us / 64.79%. | Accepted. Split2 gives MlpDown enough independent work to cut L1TEX latency impact without the duration regression seen in split4. |
| 54 | Force the MlpDown split2 body to 24 resident 64-thread blocks/SM with `__launch_bounds__(64,24)`. | MlpDown: 77.63 us / 58.45% SOL, memory 58.45%, regs 40, waves/SM 1.25, achieved occupancy 82.37%. | Rejected. Higher occupancy and lower register count destroyed the ILP schedule and materially regressed both duration and SOL. |
| 55 | Route Attn to the split2 geometry to reduce tail-wave count versus split4. | Attn: 24.77 us / 66.34% SOL, memory 55.33%, regs 64, static smem 32 B, waves/SM 2.64, achieved occupancy 56.89%. | Rejected. Duration improved, but SOL regressed versus split4, so the change moves away from the utilization target. |
| 56 | In MlpDown split2, stage all exact-shape Q5 scale halfwords into shared memory once per block, replacing per-chunk scale global loads and shuffles. | MlpDown: 75.10 us / 63.06% SOL, memory 63.06%, regs 64, static smem 576 B, waves/SM 1.88, achieved occupancy 59.10%. | Rejected. The shared preload and barrier turned the kernel L2/shared-traffic limited and materially regressed duration versus final12. |
| 57 | Add a two-row direct split4 short-K probe where each chunk warp loads the four T-column x vectors once, then reuses those registers for two adjacent output rows. | Attn: 34.37 us / 62.12% SOL, memory 62.12%, SM 42.63%, regs 48, static smem 128 B, waves/SM 2.11, achieved occupancy 74.81%. | Rejected. Register reuse cut the grid in half and made L2 the top limiter, but the lost wave geometry and doubled per-block row work materially regressed both duration and SOL versus final12. |
| 58 | Relax direct split4 `__launch_bounds__` from 10 resident blocks/SM to 8 and 9 to see whether more registers improve instruction scheduling. | Attn lb8: 27.20 us / 62.87% SOL, regs 64, waves/SM 5.27, achieved occupancy 60.07%. Attn lb9: 25.57 us / 66.37% SOL, regs 56, waves/SM 4.68, achieved occupancy 67.74%. | Rejected. Both variants allowed more registers but lost enough latency hiding and SOL that the accepted 10-block, 48-register schedule remains better. |

## Final Candidate

Code change: keep the generic 8-row small-T kernel for Q5 generally, but route
the three full-aligned short-K Q5 T=4 shapes (`7168x5120`, `6144x5120`, and
`5120x6144`) to exact-shape instantiations of
`linear_rowsplit_gemm_smallt_kernel_direct_split4_q5_t4<Q5Smallt,kFullSlabs,kStride>`.
The split4 kernel uses four warps/block for one output row: each warp loads one
256-K slice of each 1024-value slab directly from global memory, accumulates the
four T columns in fp32, then the block performs a single final four-part row
reduction. The old staged chunk4 path was removed because it is no longer
routed.

MlpDown (`5120x17408`) now routes to
`linear_rowsplit_gemm_smallt_kernel_direct_split2_q5_t4<Q5Smallt,17,17408>`.
That exact-shape body uses two warps for one output row: each warp loads half of
each 1024-value slab directly from global memory, accumulates the four T columns
in fp32, then the block performs a final two-part row reduction. The old
one-warp direct MlpDown path was removed because it is no longer routed.

Profiles:

- `profiles/q5-smallt/final12/AttnInQKV7168x5120.ncu-rep`
- `profiles/q5-smallt/final12/MlpDown5120x17408.ncu-rep`
- `profiles/q5-smallt/final12/Proj6144x5120.ncu-rep`
- `profiles/q5-smallt/final12/Out5120x6144.ncu-rep`

| shape | kernel | duration us | SOL % | SM % | memory % | regs | static smem KiB | waves/SM | achieved occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AttnInQKV7168x5120 | `direct_split4_q5_t4<Q5Smallt,5,5120>` | 25.31 | 69.11 | 69.11 | 54.18 | 48 | 0.06 | 4.22 | 74.17 |
| MlpDown5120x17408 | `direct_split2_q5_t4<Q5Smallt,17,17408>` | 59.04 | 65.88 | 65.88 | 58.71 | 64 | 0.03 | 1.88 | 58.70 |
| Proj6144x5120 | `direct_split4_q5_t4<Q5Smallt,5,5120>` | 22.91 | 65.17 | 65.17 | 51.32 | 48 | 0.06 | 3.61 | 73.34 |
| Out5120x6144 | `direct_split4_q5_t4<Q5Smallt,6,6144>` | 23.20 | 64.79 | 64.79 | 50.74 | 48 | 0.06 | 3.01 | 73.50 |

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
direct global-load split4 body, and replaces the one-warp direct MlpDown path
with a direct split2 body. Versus final9, the three short-K durations improve
materially: Attn 31.49 -> 25.31 us, Proj 27.14 -> 22.91 us, and Out 27.33 ->
23.20 us. MlpDown improves from the final11 one-warp direct sample at 62.05 us
to 59.04 us while raising SOL from 60.22% to 65.88%. Numerical
correctness is unchanged under the existing linear correctness suite and
`compute-sanitizer` reports no errors.

The primary 85% SOL target was not reached. The best final representative SOL is
69.11% on Attn. The direct split4 path proves that, for the short-K shapes,
per-slab shared staging and block barriers were more expensive than direct Q5
global loads, even though the resulting kernel is still L1TEX-scoreboard
limited. The split2 MlpDown path proves that the long-K shape benefits from more
row-local K parallelism than one warp can provide, but split4 MlpDown still
regresses duration and aggressive register caps remove too much ILP. The failed
tensor-core, split-by-slab, staged chunk, pipeline-cleanup, read-only-load,
deeper-pipeline, launch-bounds, x-preload MlpDown, rows/block, shared x-staging,
grouped high-bit load, shared-scale, and split2 short-K experiments indicate
that reaching 85% SOL likely requires a larger architecture change, such as a
better-balanced split-K or tensor-core path that can feed all warps without
duplicating dequant/staging work.

## Generalized Direct Split T=2..6

Date: 2026-07-05

Goal: generalize the accepted Q5 direct split bodies from T=4-only to the MTP
decode set T in {2,3,4,5,6}. The old `_t4` direct kernels were replaced by
templated exact-T direct kernels. Q4/Q6/W8 remain on the generic SmallT path:
there is no measured exercised Qwen3.6 call site in this task that justifies
extra instantiations.

Bench command:

```bash
./build/bench/qus_linear_op_bench --shape <shape> --qtype Q5 --t-sweep 2,3,4,5,6 \
  --warmup 1 --repeat 200 --flush-mib 256 --stream-ceiling-gbs 1792
```

NCU command shape:

```bash
ncu --force-overwrite --target-processes all --kernel-name-base demangled \
  --set basic --launch-skip 1 --launch-count 1 \
  --kernel-name 'regex:linear_rowsplit_gemm_smallt.*Q5Smallt' \
  -o profiles/q5-smallt/<attempt>/<shape>_T<t> \
  ./build/bench/qus_linear_op_bench --shape <shape> --qtype Q5 --t-sweep <t> \
  --warmup 1 --repeat 1 --flush-mib 1 --stream-ceiling-gbs 1792
```

Profile sets:

- Generic measurement-only bypass: `profiles/q5-smallt/generic-sweep/*.ncu-rep`
- Final generalized route: `profiles/q5-smallt/generalized-final/*.ncu-rep`
- Extracted CSV summaries: `profiles/q5-smallt/generic-sweep/summary.csv`,
  `profiles/q5-smallt/generalized-final/summary.csv`

Fresh current T=4 gap before generalization, using a measurement-only local
bypass for the generic T=4 path:

| shape | direct T=4 us | generic T=4 us | direct speedup |
| --- | ---: | ---: | ---: |
| AttnInQKV7168x5120 | 25.856 | 38.144 | 1.48x |
| MlpDown5120x17408 | 52.512 | 67.616 | 1.29x |
| Proj6144x5120 | 21.792 | 34.080 | 1.56x |
| Out5120x6144 | 23.520 | 31.904 | 1.36x |

Generic path headroom from the same pre-change sweep:

| shape | T=2 us | T=3 us | T=4 generic us | T=5 us | T=6 us |
| --- | ---: | ---: | ---: | ---: | ---: |
| AttnInQKV7168x5120 | 36.096 | 36.832 | 38.144 | 60.672 | 62.752 |
| MlpDown5120x17408 | 64.768 | 66.816 | 67.616 | 152.832 | 161.056 |
| Proj6144x5120 | 32.000 | 33.888 | 34.080 | 52.512 | 56.064 |
| Out5120x6144 | 29.952 | 30.464 | 31.904 | 56.928 | 62.752 |

Split-factor probes:

| probe | result | decision |
| --- | --- | --- |
| Short-K split2 instead of split4 | Attn improved for T=3..6 and tied T=2. Out improved most at T=5/6. Proj tied at T=2/3/6 but split4 stayed faster at T=4/5. | Route Attn and Out to split2; keep Proj on split4. |
| MlpDown split4 instead of split2 | Slower at every measured T: 46.368/54.304/57.472/66.624/72.800 us for T=2..6. | Keep MlpDown on split2. |

Final route:

- AttnInQKV7168x5120: `direct_split2_q5<Q5Smallt,kTt,5,5120>`
- MlpDown5120x17408: `direct_split2_q5<Q5Smallt,kTt,17,17408>`
- Proj6144x5120: `direct_split4_q5<Q5Smallt,kTt,5,5120>`
- Out5120x6144: `direct_split2_q5<Q5Smallt,kTt,6,6144>`

Final benchmark results versus the generic headroom:

| shape | T | generic us | final us | speedup | final kernel | NCU SOL % | regs | smem B | waves/SM | occ % |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| AttnInQKV7168x5120 | 2 | 36.096 | 21.760 | 1.66x | split2 | 62.87 | 64 | 16 | 2.64 | 57.84 |
| AttnInQKV7168x5120 | 3 | 36.832 | 22.912 | 1.61x | split2 | 58.03 | 64 | 24 | 2.64 | 57.54 |
| AttnInQKV7168x5120 | 4 | 38.144 | 25.792 | 1.48x | split2 | 65.93 | 64 | 32 | 2.64 | 57.10 |
| AttnInQKV7168x5120 | 5 | 60.672 | 29.952 | 2.03x | split2 | 65.65 | 64 | 40 | 2.64 | 56.34 |
| AttnInQKV7168x5120 | 6 | 62.752 | 31.680 | 1.98x | split2 | 67.71 | 64 | 48 | 2.64 | 57.06 |
| MlpDown5120x17408 | 2 | 64.768 | 44.320 | 1.46x | split2 | 69.62 | 64 | 16 | 1.88 | 60.47 |
| MlpDown5120x17408 | 3 | 66.816 | 46.336 | 1.44x | split2 | 67.15 | 64 | 24 | 1.88 | 60.10 |
| MlpDown5120x17408 | 4 | 67.616 | 55.328 | 1.22x | split2 | 64.98 | 64 | 32 | 1.88 | 57.60 |
| MlpDown5120x17408 | 5 | 152.832 | 62.720 | 2.44x | split2 | 62.61 | 64 | 40 | 1.88 | 58.33 |
| MlpDown5120x17408 | 6 | 161.056 | 67.584 | 2.38x | split2 | 66.14 | 64 | 48 | 1.88 | 57.18 |
| Proj6144x5120 | 2 | 32.000 | 19.712 | 1.62x | split4 | 58.49 | 48 | 32 | 3.61 | 74.29 |
| Proj6144x5120 | 3 | 33.888 | 21.728 | 1.56x | split4 | 54.65 | 48 | 48 | 3.61 | 73.53 |
| Proj6144x5120 | 4 | 34.080 | 23.712 | 1.44x | split4 | 59.77 | 48 | 64 | 3.61 | 73.31 |
| Proj6144x5120 | 5 | 52.512 | 27.552 | 1.91x | split4 | 63.35 | 48 | 80 | 3.61 | 73.76 |
| Proj6144x5120 | 6 | 56.064 | 29.824 | 1.88x | split4 | 63.93 | 48 | 96 | 3.61 | 74.16 |
| Out5120x6144 | 2 | 29.952 | 19.616 | 1.53x | split2 | 60.25 | 64 | 16 | 1.88 | 59.11 |
| Out5120x6144 | 3 | 30.464 | 19.744 | 1.54x | split2 | 55.63 | 64 | 24 | 1.88 | 58.24 |
| Out5120x6144 | 4 | 31.904 | 23.040 | 1.38x | split2 | 58.47 | 64 | 32 | 1.88 | 55.72 |
| Out5120x6144 | 5 | 56.928 | 25.888 | 2.20x | split2 | 60.58 | 64 | 40 | 1.88 | 54.16 |
| Out5120x6144 | 6 | 62.752 | 27.936 | 2.25x | split2 | 63.66 | 64 | 48 | 1.88 | 53.73 |

Generic NCU reference:

| shape | T | generic kernel | NCU duration us | SOL % | regs | waves/SM | occ % |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| AttnInQKV7168x5120 | 2 | `<Q5Smallt,4,8,2>` | 37.47 | 46.74 | 58 | 1.32 | 58.06 |
| AttnInQKV7168x5120 | 3 | `<Q5Smallt,4,8,2>` | 35.01 | 50.57 | 58 | 1.32 | 57.28 |
| AttnInQKV7168x5120 | 4 | `<Q5Smallt,4,8,2>` | 39.97 | 44.93 | 58 | 1.32 | 57.03 |
| AttnInQKV7168x5120 | 5 | `<Q5Smallt,8,8,2>` | 72.45 | 67.08 | 71 | 1.76 | 42.73 |
| AttnInQKV7168x5120 | 6 | `<Q5Smallt,8,8,2>` | 75.42 | 64.45 | 71 | 1.76 | 42.97 |
| MlpDown5120x17408 | 2 | `<Q5Smallt,4,8,2>` | 67.23 | 61.46 | 58 | 0.94 | 62.09 |
| MlpDown5120x17408 | 3 | `<Q5Smallt,4,8,2>` | 69.60 | 60.50 | 58 | 0.94 | 62.15 |
| MlpDown5120x17408 | 4 | `<Q5Smallt,4,8,2>` | 72.83 | 57.38 | 58 | 0.94 | 61.73 |
| MlpDown5120x17408 | 5 | `<Q5Smallt,8,8,2>` | 189.89 | 61.75 | 71 | 1.25 | 41.77 |
| MlpDown5120x17408 | 6 | `<Q5Smallt,8,8,2>` | 193.41 | 61.16 | 71 | 1.25 | 42.22 |
| Proj6144x5120 | 2 | `<Q5Smallt,4,8,2>` | 32.64 | 46.36 | 58 | 1.13 | 60.97 |
| Proj6144x5120 | 3 | `<Q5Smallt,4,8,2>` | 33.18 | 45.85 | 58 | 1.13 | 60.55 |
| Proj6144x5120 | 4 | `<Q5Smallt,4,8,2>` | 35.39 | 43.28 | 58 | 1.13 | 59.94 |
| Proj6144x5120 | 5 | `<Q5Smallt,8,8,2>` | 60.83 | 68.56 | 71 | 1.51 | 41.27 |
| Proj6144x5120 | 6 | `<Q5Smallt,8,8,2>` | 62.11 | 67.10 | 71 | 1.51 | 41.00 |
| Out5120x6144 | 2 | `<Q5Smallt,4,8,2>` | 28.90 | 50.99 | 58 | 0.94 | 61.54 |
| Out5120x6144 | 3 | `<Q5Smallt,4,8,2>` | 28.93 | 52.22 | 58 | 0.94 | 61.81 |
| Out5120x6144 | 4 | `<Q5Smallt,4,8,2>` | 28.99 | 51.84 | 58 | 0.94 | 61.56 |
| Out5120x6144 | 5 | `<Q5Smallt,8,8,2>` | 70.82 | 59.50 | 71 | 1.25 | 40.82 |
| Out5120x6144 | 6 | `<Q5Smallt,8,8,2>` | 63.84 | 65.00 | 71 | 1.25 | 41.21 |

Correctness and memory verification:

```bash
cmake --build build --target qus_linear_op_bench qus_linear_test -j
./build/tests/qus_linear_test
# OK linear correctness
compute-sanitizer ./build/tests/qus_linear_test
# ERROR SUMMARY: 0 errors
```

Conclusion: every measured non-T=4 decode size now uses a direct Q5 split
kernel on the registered shapes and improves over the generic path. The biggest
remaining tradeoff is T=4 code shape: the generalized looped body preserves the
direct-path advantage over generic T=4, but it does not keep every old
hand-unrolled T=4 microbenchmark result. The split-factor retune keeps the best
measured route per representative shape under the new exact-T implementation.
