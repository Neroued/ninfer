# M3 Phase 2 Decode GEMV Baseline

Date: 2026-06-29. GPU: NVIDIA GeForce RTX 5090. CUDA runtime/driver from the
nsys report: 13.1/13.1. NCU preflight passed on WSL2 with `ncu` 2025.4.1.

## Bench Coverage

Command:

```bash
cmake -S . -B build
cmake --build build -j --target qus_linear_bench
./build/bench/qus_linear_bench --decode
```

Artifact: `profiles/m3-p2-gemv-baseline/linear_decode_bench.txt`.

| qtype | role | shape [N,K] | median us | bench GB/s | bench % of 1790 GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Q4 | mlp.gate/up | [17408,5120] | 170.08 | 278.7 | 15.6 |
| Q4 | gdn q/k | [2048,5120] | 169.57 | 32.9 | 1.8 |
| Q4 | attn q | [6144,5120] | 168.82 | 99.1 | 5.5 |
| Q5 | mlp.down | [5120,17408] | 1730.53 | 33.8 | 1.9 |
| Q5 | v/z | [6144,5120] | 457.06 | 45.2 | 2.5 |
| Q5 | out | [5120,6144] | 551.78 | 37.5 | 2.1 |
| Q5 | attn gate | [6144,5120] | 457.82 | 45.1 | 2.5 |
| Q6 | lm_head | [248320,5120] | 3599.84 | 276.1 | 15.4 |

The bench GB/s column is the harness estimate only; NCU is the evidence for
DRAM throughput and bound type.

## Nsys Decode Ranking

Command:

```bash
nsys profile --force-overwrite=true \
  -o profiles/m3-p2-gemv-baseline/decode_cn16.nsys-rep \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  ./build/bench/qus_e2e_bench \
    --weights out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus \
    --output-json profiles/m3-p2-gemv-baseline/decode_cn16.json \
    --fixture-manifest bench/fixtures/prompts/m2.8-v1.manifest.json \
    --case cn_short:bench/fixtures/prompts/cn_short.ids:16 \
    --warmup-repeats 0 --repeats 1 --max-ctx 8192 --device 0 --quiet
```

Artifacts:

- `profiles/m3-p2-gemv-baseline/decode_cn16.nsys-rep`
- `profiles/m3-p2-gemv-baseline/decode_cn16.sqlite`
- `profiles/m3-p2-gemv-baseline/decode_cn16.nsys-summary.md`
- `profiles/m3-p2-gemv-baseline/decode_cn16_cuda_gpu_kern_sum.txt`
- `profiles/m3-p2-gemv-baseline/decode_cn16_decode_window.txt`
- `profiles/m3-p2-gemv-baseline/decode_cn16_decode_only_kernel_sum_start=20709306128_end=25706404280.txt`

The full capture span was 25,813.929 ms. The trace has no NVTX ranges, so the
decode window is inferred from kernel ordering: first kernel after the prefill
`set_pos_kernel` through final `advance_pos_kernel`. That window is
4,997.098 ms.

| decode kernel | instances | total ms | decode kernel time |
| --- | ---: | ---: | ---: |
| `linear_generic_lowbit_gemv_kernel<Q5Codec>` | 3840 | 3774.242 | 79.5% |
| `linear_generic_lowbit_gemv_kernel<Q4Codec>` | 3840 | 858.300 | 18.1% |
| `linear_generic_lowbit_gemv_kernel<Q6Codec>` | 15 | 61.485 | 1.3% |
| `gqa_attention_decode_kernel` | 240 | 28.747 | 0.6% |

This confirms the expected baseline ranking: generic Q5 GEMV dominates decode,
with generic Q4 GEMV second.

## NCU Baseline

Required roofline command shape:

```bash
ncu --force-overwrite -o profiles/m3-p2-gemv-baseline/<tag> \
  --set roofline \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats \
  --kernel-name regex:'lowbit_gemv' --launch-skip 0 --launch-count 1 \
  ./build/bench/qus_linear_bench --decode --q5
```

The Q4 capture used the same command with `--q4`. Basic regex-match captures,
instruction summaries, and custom instruction-mix captures are also committed
beside the roofline reports.

| shape | report | DRAM % | memory SOL % | compute SOL % | issue slots busy | achieved occ. | regs/thread | top stalls | bound type |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Q5 mlp.down [5120,17408] | `ncu_q5_mlp_down_roofline` | 2.77 | 5.37 | 5.81 | 4.82% | 8.32% | 40 | L1TEX scoreboard 42.1%, fixed-latency dependency 35.3% | unpack/issue and grid-underfilled; not DRAM-bound |
| Q4 mlp.gate/up [17408,5120] | `ncu_q4_mlp_gate_up_roofline` | 10.48 | 13.43 | 8.26 | 8.19% | 8.41% | 29 | L1TEX scoreboard 64.4% | closer to bandwidth than Q5, but still generic-grid underfilled |

Instruction mix from custom metric captures:

| shape | integer inst | memory inst | FP add/mul/ffma inst | note |
| --- | ---: | ---: | ---: | --- |
| Q5 mlp.down [5120,17408] | 3,672,453,120 | 608,624,640 | 178,257,920 | Q5 executes about 6.7x Q4 integer instructions on the dominant shape. |
| Q4 mlp.gate/up [17408,5120] | 551,746,560 | 136,670,208 | 178,257,920 | Same FP32 dot-product instruction count, much lower unpack instruction count. |

## Ceiling C

Ceiling command used `silu_and_mul` at the large prefill shape:

```bash
ncu --force-overwrite -o profiles/m3-p2-gemv-baseline/ncu_silu_and_mul_prefill_roofline \
  --set roofline \
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
  --section SchedulerStats --section WarpStateStats \
  --kernel-name regex:'silu_and_mul_kernel' --launch-skip 0 --launch-count 1 \
  ./build/bench/qus_silu_and_mul_bench --prefill
```

`residual_add` was not used for `C`: the event-timed bench reports 262.3% of the
nominal 1790 GB/s roofline, which indicates cache-dominated reuse in this
harness. `silu_and_mul` reports 92.5% in the bench loop and NCU measures a real
DRAM sustained throughput of 82.76%.

Calibrated ceiling `C = 82.76%` sustained DRAM throughput.

| kernel | shape | DRAM % | memory throughput | compute SOL % | achieved occ. | regs/thread | bound type |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `silu_and_mul_kernel` | [17408,4096] | 82.76 | 1.48 TB/s | 20.90 | 71.51% | 42 | memory-bound streaming reference |

The current generic decode GEMV kernels are far below `C`. Q5 is dominated by
unpack instruction pressure and low issue/occupancy; Q4 reaches higher DRAM
throughput but is still limited by the generic one-thread-per-row launch shape.
