# Q5090 v3 prefill kernel sweep after rowsplit GEMM tuning

Date: 2026-07-02

Scope: rerun the historical prefill kernel-level sweep after the current rowsplit GEMM tuning
state. The run keeps the historical `1024, 2048, 4096, 8192` kernel bench/ncu lengths, and also
reruns the `qus_bench` host-visible prefill timing matrix used by the previous nsys length report.

Current capture commit: `c32227714cc2dc16b5b13f4f6e36877ec38570f9`, with `worktree_dirty=false` in
the `qus_bench` artifacts. The source optimization commit included in this build is
`e9551eb perf(gemm): stage q6 scales as pairs`.

Historical comparisons:

- `docs/bench/q5090-v3-prefill-kernel-level-ncu-sweep.md`
- `docs/bench/q5090-v3-prefill-length-nsys-post-gqa-opt.md`

## Summary

The rowsplit GEMM tuning moved the broad prefill throughput curve up by roughly 10-22% compared with
the post-GQA historical report:

- `pp4096` host prefill mean improved from `1538.345 ms` to `1366.292 ms` (`1.13x`).
- `pp8192` improved from `3055.059 ms` to `2748.453 ms` (`1.11x`).
- `pp16384` improved from `6216.728 ms` to `5640.026 ms` (`1.10x`).
- Q4 MLP gate+up-equivalent bench improved by `1.14x-1.22x` across 1k-8k.
- Q5 MLP down bench improved by `1.15x-1.16x` across 1k-8k.
- GQA prefill is unchanged within noise, which is expected because this rerun is measuring rowsplit
  GEMM work after the GQA optimization had already landed.
- Q6 lm_head now reaches `92-93%` ncu tensor-pipe SOL, while its useful TC accounting is
  `79-85%` because decode, address generation, scale handling, and epilogue work are included in the
  kernel but not in useful MMA FLOP accounting.

## Artifacts

Directory:

`profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/`

Important files:

| Artifact | Path |
| --- | --- |
| Linear op sweep | `profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/linear_op_t_sweep_1k_8k.csv` |
| GQA prefill sweep | `profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/gqa_prefill_t_sweep_1k_8k.csv` |
| GQA prefill JSON | `profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/gqa_prefill_t_sweep_1k_8k.json` |
| ncu summary | `profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/ncu_kernel_sweep_summary.csv` |
| Q4 ncu reports | `profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/ncu_q4_mlp_gateup_T*.ncu-rep` |
| Q5 ncu reports | `profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/ncu_q5_mlp_down_T*.ncu-rep` |
| Q6 ncu reports | `profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/ncu_q6_lm_head_T*.ncu-rep` |
| GQA ncu reports | `profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/ncu_gqa_prefill_T*.ncu-rep` |
| 128-4096 host timing | `profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/qus_bench_prefill_sweep_r3_warmup1.json` |
| 8192/16384 host timing | `profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/qus_bench_prefill_sweep_8192_16384_r3_warmup1.json` |

## Commands

Build and ncu preflight:

```bash
~/.codex/skills/ncu-kernel-profile/scripts/preflight.sh
cmake --build build -j --target qus_linear_op_bench qus_gqa_attention_bench qus_bench
```

Kernel bench sweep:

```bash
./build/bench/qus_linear_op_bench \
  --all-targets --t-sweep 1024,2048,4096,8192 \
  --warmup 2 --repeat 5 --copy-repeat 4 \
  --csv-out profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/linear_op_t_sweep_1k_8k.csv

./build/bench/qus_gqa_attention_bench \
  --prefill --tokens 1024,2048,4096,8192 \
  --warmup 2 --repeat 5 \
  --csv-out profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/gqa_prefill_t_sweep_1k_8k.csv \
  --json-out profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/gqa_prefill_t_sweep_1k_8k.json
```

Host-visible prefill timing:

```bash
./build/bench/qus_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
  -p 128,256,512,1024,2048,4096 \
  -r 3 --warmup 1 -o json \
  --output-file profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/qus_bench_prefill_sweep_r3_warmup1.json

./build/bench/qus_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus \
  -p 8192,16384 \
  -r 3 --warmup 1 -o json \
  --output-file profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/qus_bench_prefill_sweep_8192_16384_r3_warmup1.json
```

ncu used one single launch per length and per kernel family:

```bash
ncu --force-overwrite \
  --section SpeedOfLight --section Occupancy --section ComputeWorkloadAnalysis \
  --kernel-name regex:linear_rowsplit_gemm_mma_kernel \
  --launch-skip 0 --launch-count 1 \
  -o profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/<tag> \
  ./build/bench/qus_linear_op_bench --shape <shape> --qtype <Q4|Q5|Q6> \
    --t-sweep <T> --warmup 0 --repeat 1 --copy-repeat 1

ncu --force-overwrite \
  --section SpeedOfLight --section Occupancy --section ComputeWorkloadAnalysis \
  --kernel-name regex:gqa_attention_prefill_kernel \
  --launch-skip 0 --launch-count 1 \
  -o profiles/ncu-prefill-kernel-sweep-2026-07-02-q6-scale-pair/<tag> \
  ./build/bench/qus_gqa_attention_bench --prefill --tokens <T> --warmup 0 --repeat 1
```

ncu replay duration is not used as the latency source. Use bench medians for kernel timing.

## Host prefill timing

| pp | old mean ms | new mean ms | speedup | old tok/s | new tok/s | work peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 137.243 | 113.907 | 1.20x | 932.7 | 1123.7 | 22.6 |
| 256 | 167.681 | 136.937 | 1.22x | 1526.8 | 1869.5 | 45.2 |
| 512 | 237.138 | 199.515 | 1.19x | 2159.2 | 2566.2 | 90.4 |
| 1024 | 408.090 | 355.825 | 1.15x | 2509.3 | 2877.8 | 180.8 |
| 2048 | 780.992 | 686.802 | 1.14x | 2622.3 | 2981.9 | 361.5 |
| 4096 | 1538.345 | 1366.292 | 1.13x | 2662.6 | 2997.9 | 723.0 |
| 8192 | 3055.059 | 2748.453 | 1.11x | 2681.5 | 2980.6 | 1446.1 |
| 16384 | 6216.728 | 5640.026 | 1.10x | 2635.5 | 2905.0 | 2892.1 |

Readout: the gain is largest at short/mid lengths and stays visible through 16k. The long-context
curve is now roughly `2.9k-3.0k tok/s` from 2k through 16k in this r3 host timing run.

## Kernel bench useful efficiency

Historical rows are from `q5090-v3-prefill-kernel-level-ncu-sweep.md`. Current rows are from the new
bench CSVs.

| kernel bench | T | old ms | new ms | speedup | old useful TC | new useful TC | old TFLOP/s | new TFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4 MLP gate+up equivalent | 1024 | 2.238 | 1.889 | 1.19x | 73.9% | 87.8% | 163.1 | 193.3 |
| Q4 MLP gate+up equivalent | 2048 | 4.388 | 3.701 | 1.19x | 75.4% | 89.7% | 166.4 | 197.3 |
| Q4 MLP gate+up equivalent | 4096 | 8.964 | 7.321 | 1.22x | 73.9% | 90.6% | 162.9 | 199.5 |
| Q4 MLP gate+up equivalent | 8192 | 17.476 | 15.380 | 1.14x | 75.8% | 86.3% | 167.1 | 189.9 |
| Q5 MLP down | 1024 | 1.234 | 1.064 | 1.16x | 67.0% | 78.0% | 147.9 | 171.5 |
| Q5 MLP down | 2048 | 2.411 | 2.093 | 1.15x | 68.7% | 79.3% | 151.4 | 174.4 |
| Q5 MLP down | 4096 | 4.681 | 4.027 | 1.16x | 70.7% | 82.4% | 156.0 | 181.3 |
| Q5 MLP down | 8192 | 9.020 | 7.789 | 1.16x | 73.4% | 85.2% | 161.9 | 187.5 |
| GQA prefill | 1024 | 0.154 | 0.154 | 1.00x | 40.0% | 39.9% | 83.8 | 83.7 |
| GQA prefill | 2048 | 0.437 | 0.439 | 0.99x | 56.4% | 56.0% | 118.1 | 117.3 |
| GQA prefill | 4096 | 1.415 | 1.417 | 1.00x | 69.6% | 69.5% | 145.8 | 145.5 |
| GQA prefill | 8192 | 4.888 | 4.875 | 1.00x | 80.5% | 80.8% | 168.7 | 169.2 |

Additional current Q6 rowsplit bench:

| kernel bench | T | ms | useful TC | TFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| Q6 lm_head | 1024 | 13.976 | 84.7% | 186.3 |
| Q6 lm_head | 2048 | 29.226 | 81.0% | 178.2 |
| Q6 lm_head | 4096 | 59.482 | 79.6% | 175.1 |
| Q6 lm_head | 8192 | 119.565 | 79.2% | 174.2 |

## ncu hardware utilization

This is hardware SOL, not useful FLOP accounting. `tensor pipe` is extracted from the ncu
Compute Workload Analysis high-pipe rule. Duration comes from ncu replay and is included only for
artifact inspection, not timing comparison.

| kernel | metric | 1k | 2k | 4k | 8k |
| --- | --- | ---: | ---: | ---: | ---: |
| Q4 MLP gate+up equivalent | tensor pipe | 87.4% | 84.9% | 89.8% | 89.1% |
| Q4 MLP gate+up equivalent | SM | 87.4% | 84.9% | 89.8% | 89.1% |
| Q4 MLP gate+up equivalent | memory | 51.8% | 50.4% | 53.2% | 52.8% |
| Q4 MLP gate+up equivalent | DRAM | 9.6% | 12.6% | 10.5% | 12.0% |
| Q4 MLP gate+up equivalent | achieved occupancy | 32.8% | 33.0% | 33.1% | 33.2% |
| Q5 MLP down | tensor pipe | 78.1% | 72.9% | 81.4% | 84.3% |
| Q5 MLP down | SM | 78.1% | 72.9% | 81.4% | 84.3% |
| Q5 MLP down | memory | 50.8% | 48.8% | 52.8% | 54.6% |
| Q5 MLP down | DRAM | 4.6% | 5.9% | 2.7% | 2.4% |
| Q5 MLP down | achieved occupancy | 31.3% | 31.5% | 32.0% | 33.0% |
| Q6 lm_head | tensor pipe | 92.3% | 93.0% | 92.9% | 93.1% |
| Q6 lm_head | SM | 92.3% | 93.0% | 92.9% | 93.1% |
| Q6 lm_head | memory | 54.2% | 55.0% | 54.9% | 55.0% |
| Q6 lm_head | DRAM | 28.3% | 28.4% | 28.5% | 28.5% |
| Q6 lm_head | achieved occupancy | 16.6% | 16.6% | 16.6% | 16.6% |
| GQA prefill | tensor pipe | 36.5% | 50.5% | 62.3% | 71.3% |
| GQA prefill | SM | 36.5% | 50.5% | 62.3% | 71.3% |
| GQA prefill | memory | 22.3% | 27.3% | 31.7% | 35.0% |
| GQA prefill | DRAM | 4.9% | 3.8% | 3.0% | 1.8% |
| GQA prefill | achieved occupancy | 14.1% | 15.1% | 15.8% | 16.2% |

Compared with the historical ncu report:

- Q4 tensor-pipe SOL moved from about `68-71%` to `85-90%`.
- Q5 moved from about `62-66%` to `73-84%`.
- GQA remains in the same band, ending near `71%` ncu tensor-pipe SOL at 8k and `80.8%` useful TC in
  the bench model.
- Q6 lm_head has the highest ncu tensor-pipe SOL in this sweep (`92-93%`), but low achieved occupancy
  (`16.6%`) and higher DRAM share (`28%`) reflect the larger Q6 payload and the two-CTA resource
  shape.

## Readout

The post-GQA bottleneck is still rowsplit low-bit GEMM, but it is now much closer to the hardware
limit. The most visible remaining gap is not GQA; GQA is flat relative to the previous report and is
already in its intended long-context efficiency band. The broad e2e improvement comes from the
rowsplit GEMM side: Q4/Q5 representative kernels gained enough tensor-pipe utilization to explain
the `qus_bench` host timing uplift.

For Q6, the scale-pair path is now near saturated at the hardware-counter level. Further Q6 work
should focus on reducing non-MMA instruction overhead and epilogue/decode cost, not on trying to make
ncu tensor-pipe SOL higher. For Q4/Q5, the current ncu values show much less headroom than the
historical report, especially at 4k/8k.
