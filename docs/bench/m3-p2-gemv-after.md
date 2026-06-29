# M3 Phase 2 Decode GEMV After Report

Date: 2026-06-29.

## Phase 2.5 update — tile-centric coalescing redesign

The Phase 2 warp-per-row kernel was pipe-bound on scattered loads. Phase 2.5
replaced it with a tile-centric kernel for Q5/Q6: each CTA owns a 32-row stripe
of one N64 tile, `KSPLIT` warps split the K groups, and per group a warp
cooperatively loads the contiguous 32-row code block into shared memory with
fully coalesced word loads, then `lane == row` decodes its row (branchless) and
accumulates; partials reduce across warps in SMEM. `KSPLIT = 32` (`16` for very
large N) maximizes warps/CTA, which was the occupancy lever. Q4 keeps its direct
byte-per-lane kernel: its layout is already coalesced, so SMEM staging only added
overhead (measured Q4 `mlp.gate/up` regressed ~86 ms -> ~100 ms under the tile
kernel), so dispatch is per format.

Cold-cache results (ncu `dram__throughput.avg.pct_of_peak`, L2-flushed; ceiling
`C = 82.76%`):

| shape | qtype | before | after | note |
| --- | --- | ---: | ---: | --- |
| `mlp.down [5120,17408]` | Q5 | 47.4% | 54.3% | occupancy 53%->66%, 70->64 us; canonical-layout occupancy ceiling |
| `lm_head [248320,5120]` | Q6 | ~30% | ~87% | huge-N, now coalesced; at/above C |
| `out [5120,6144]` | Q5 | ~41% | ~45% | |
| `v/z [6144,5120]` | Q5 | ~33% | ~37% | |
| `mlp.gate/up [17408,5120]` | Q4 | ~63% | ~63% | unchanged (direct kernel kept) |

End-to-end decode GEMV summed time (same `cn_short:16` workload):

| | before (Phase 2) | after (Phase 2.5) | delta |
| --- | ---: | ---: | ---: |
| Q5 tile GEMV | 103.3 ms | ~95 ms | -8% |
| Q6 lm_head GEMV | 30.0 ms | ~10 ms | -67% |
| Q4 direct GEMV | 85.6 ms | ~88 ms | ~flat |
| total | 218.9 ms | ~191 ms | **-13%** |

Verification: `qus_linear_test` (fp64 oracle) PASS and `compute-sanitizer` clean
across all qtypes/shapes incl. non-64-multiple tails; externally workspace-free;
public API and q5090 ABI unchanged.

Remaining limit: `mlp.down` is occupancy-capped at ~66% (N=5120 -> 160 CTAs at 32
rows/CTA; a `ROWS=16` experiment gave more CTAs but did not raise occupancy and
wasted decode lanes, so it was reverted). Pushing `down` past ~55% on the
canonical layout is bounded; further gain needs a derived GEMM-friendly Q5 layout
(framework section 21.2), which is a separate ABI-revising phase.

## Phase 2 history (superseded by 2.5 above)

Status: blocked at the Task 4 end-to-end nsys gate. The per-shape Task 4 loop
has documented stop conditions, but the full decode trace still shows low-bit
GEMV as the dominant decode kernel cost.

## Workload

Same workload as the baseline:

```bash
nsys profile --force-overwrite=true \
  -o profiles/m3-p2-gemv-after/decode_cn16.nsys-rep \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  ./build/bench/qus_e2e_bench \
    --weights out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus \
    --output-json profiles/m3-p2-gemv-after/decode_cn16.json \
    --fixture-manifest bench/fixtures/prompts/m2.8-v1.manifest.json \
    --case cn_short:bench/fixtures/prompts/cn_short.ids:16 \
    --warmup-repeats 0 --repeats 1 --max-ctx 8192 --device 0 --quiet
```

There are no NVTX phase ranges in the trace, matching the baseline. The decode
window is inferred the same way as the baseline: first kernel after
`set_pos_kernel` through the final `advance_pos_kernel`.

Artifacts:

- `profiles/m3-p2-gemv-after/decode_cn16.nsys-rep`
- `profiles/m3-p2-gemv-after/decode_cn16.sqlite`
- `profiles/m3-p2-gemv-after/decode_cn16.nsys-summary.md`
- `profiles/m3-p2-gemv-after/decode_cn16_cuda_gpu_kern_sum.txt`
- `profiles/m3-p2-gemv-after/decode_cn16_cuda_gpu_mem_time_sum.txt`
- `profiles/m3-p2-gemv-after/decode_cn16_cuda_api_sum.txt`
- `profiles/m3-p2-gemv-after/decode_cn16_nvtx_sum.txt`
- `profiles/m3-p2-gemv-after/decode_cn16_decode_window.txt`
- `profiles/m3-p2-gemv-after/decode_cn16_decode_only_kernel_sum_start=20436055296_end=20797746440.txt`

## Decode Window Result

| metric | baseline | after |
| --- | ---: | ---: |
| decode window span | 4997.098 ms | 361.691 ms |
| decode kernel summed time | 4746.800 ms | 273.825 ms |
| low-bit GEMV summed time | 4694.028 ms | 224.640 ms |
| low-bit GEMV share | 98.9% | 82.0% |

After decode-window top kernels:

| kernel | instances | total ms | decode kernel time |
| --- | ---: | ---: | ---: |
| `linear_tuned_lowbit_gemv_kernel<Q5Codec>` | 3840 | 106.390 | 38.9% |
| `linear_tuned_lowbit_gemv_kernel<Q4Codec>` | 3840 | 88.396 | 32.3% |
| `linear_tuned_lowbit_gemv_kernel<Q6Codec>` | 15 | 29.853 | 10.9% |
| `gqa_attention_decode_kernel` | 240 | 25.916 | 9.5% |
| `rmsnorm_d5120_kernel` | 1935 | 5.827 | 2.1% |

The plan's end-to-end gate does not hold. Low-bit GEMV remains the dominant
decode cost and its combined share is still above the required `<50%`.

## Per-shape Stop Summary

Calibrated ceiling from the baseline report: `C = 82.76%` sustained DRAM.

| shape | best documented state | stop reason |
| --- | --- | --- |
| Q5 `mlp.down [5120,17408]` | 53.27% DRAM, 71.10 us, no spills | Diminishing returns: accepted round 5 had flat duration and a follow-up 64-thread CTA experiment regressed. |
| Q4 `mlp.gate/up [17408,5120]` | 63.74% DRAM, 45.31 us, no spills | Diminishing returns: rounds 4 and 5 each produced <5% duration improvement, with residual Q4 scale-sector/L1TEX scoreboard behavior. |
| Q5 `v/z [6144,5120]` | 42.56% DRAM, 27.58 us, no spills | Short-K/short-wave limiter; a Q5-only two-warps-per-row experiment regressed duration and DRAM. |
| Q5 `out [5120,6144]` | 41.14% DRAM, 28.54 us, no spills | Short-K/short-wave limiter; the same Q5-only two-warps-per-row experiment regressed duration and DRAM. |
| Q6 `lm_head [248320,5120]` | Tuned route active; after nsys shows 29.853 ms summed decode time | The plan's Task 4 shape order did not assign a separate Q6 tuning round; after nsys shows it is now the third low-bit GEMV decode cost. |

## Blocker

The Task 4 loop reached documented per-shape stop conditions, but not the
roofline target on the dominant Q5/Q4 shapes, and the full decode trace failed
the required structural outcome. Continuing within this plan's accepted
one-variable iteration loop is blocked: the required nsys gate is false after
all planned shapes have stopped.
