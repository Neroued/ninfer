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

## Final Candidate

Code change: keep the original 8-row small-T kernel for Q5 generally, but route
the two full-aligned `K == 5120` Q5 T=4 shapes that benefit from split-K
parallelism to `linear_rowsplit_gemm_smallt_kernel_chunk4<Q5Smallt,4,8,2>`.
The chunk kernel uses 8 warps/block as four cooperative warps per output row:
one warp stages the Q5 slab, four warps consume the four 256-wide chunks in
parallel, and the block reduces four fp32 partials before storing bf16 output.
MlpDown and Out stay on the original `<Q5Smallt,4,8,2>` kernel.

Profiles:

- `profiles/q5-smallt/final5/AttnInQKV7168x5120.ncu-rep`
- `profiles/q5-smallt/final5/MlpDown5120x17408.ncu-rep`
- `profiles/q5-smallt/final5/Proj6144x5120.ncu-rep`
- `profiles/q5-smallt/final5/Out5120x6144.ncu-rep`

| shape | kernel | duration us | SOL % | SM % | memory % | regs | static smem KiB | waves/SM | achieved occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AttnInQKV7168x5120 | `kernel_chunk4<Q5Smallt,4,8,2>` | 32.16 | 66.61 | 66.61 | 42.71 | 48 | 2.82 | 4.22 | 75.85 |
| MlpDown5120x17408 | `<Q5Smallt,4,8,2>` | 72.51 | 58.21 | 58.21 | 45.95 | 58 | 10.75 | 0.94 | 62.07 |
| Proj6144x5120 | `kernel_chunk4<Q5Smallt,4,8,2>` | 27.97 | 65.69 | 65.69 | 42.14 | 48 | 2.82 | 3.61 | 74.94 |
| Out5120x6144 | `<Q5Smallt,4,8,2>` | 29.63 | 50.60 | 50.60 | 39.13 | 58 | 10.75 | 0.94 | 61.32 |

Correctness:

```bash
cmake --build build --target qus_linear_op_bench qus_linear_test -j
./build/tests/qus_linear_test
# OK linear correctness
```

## Conclusion

The final candidate preserves the original 8-row kernel for the shapes where
split-K did not hold up, and improves the K=5120 Q5 T=4 shapes with a
cooperative chunk split that raises occupancy and reduces register/shared-memory
pressure for those shapes. Numerical correctness is unchanged under the existing
linear correctness suite.

The primary 85% SOL target was not reached. The best final representative SOL is
66.61% on Attn and 65.69% on Proj. The chunk split proves that the original
warp-per-row body was under-parallelized for the K=5120 shapes, but MlpDown and
Out remain limited by the original low-wave, issue-eligibility behavior. The
failed tensor-core and split-by-slab experiments indicate that reaching 85% SOL
likely requires a larger architecture change, such as a better-balanced split-K
or tensor-core path that can feed all warps without duplicating dequant/staging
work.
