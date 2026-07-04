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

## Final Candidate

Code change: keep the original 8-row small-T kernel for Q5 generally, but route
`T == 4 && N == 7168 && K == 5120` to a 16-row block instantiation. This is the
only measured block-shape change that repeatedly improved a representative shape
without requiring a different algorithm.

Profiles:

- `profiles/q5-smallt/final2/AttnInQKV7168x5120.ncu-rep`
- `profiles/q5-smallt/final2/MlpDown5120x17408.ncu-rep`
- `profiles/q5-smallt/final2/Proj6144x5120.ncu-rep`
- `profiles/q5-smallt/final2/Out5120x6144-rerun.ncu-rep`

| shape | kernel | duration us | SOL % | SM % | memory % | regs | static smem KiB | waves/SM | achieved occ % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AttnInQKV7168x5120 | `<Q5Smallt,4,16,2>` | 37.98 | 46.93 | 46.93 | 36.04 | 58 | 21.50 | 1.32 | 59.59 |
| MlpDown5120x17408 | `<Q5Smallt,4,8,2>` | 71.74 | 58.21 | 58.21 | 46.10 | 58 | 10.75 | 0.94 | 62.37 |
| Proj6144x5120 | `<Q5Smallt,4,8,2>` | 35.30 | 42.89 | 42.89 | 32.97 | 58 | 10.75 | 1.13 | 59.86 |
| Out5120x6144 | `<Q5Smallt,4,8,2>` | 29.70 | 50.49 | 50.49 | 39.66 | 58 | 10.75 | 0.94 | 61.64 |

Correctness:

```bash
./build/tests/qus_linear_test
# OK linear correctness
```

## Conclusion

The final candidate preserves the original 8-row kernel for the shapes where
block-shape changes did not hold up, and improves the Attn Q5 T=4 path by using
16 row-warps per block. Numerical correctness is unchanged.

The primary 85% SOL target was not reached. The best final representative SOL is
58.21% on MlpDown, and the Attn specialization reaches 46.93%. NCU evidence says
the scalar rowsplit small-T kernel remains issue-eligibility limited. The failed
shared-x, prefetch, and launch-bounds experiments indicate that reaching 85% SOL
likely requires a different small-T architecture, such as a tensor-core or
split-K design with an explicit reduction path, rather than another local tuning
of the current warp-per-row scalar kernel.
