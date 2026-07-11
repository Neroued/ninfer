# Q5090 v3 prefill kernel-level bench/ncu sweep

Date: 2026-07-02

Scope: after the GQA prefill optimization, inspect the kernel-level hotspots for prompt lengths
`1024, 2048, 4096, 8192`. The end-to-end share is taken from the existing nsys sweep; this report
adds per-kernel microbench and ncu hardware counters.

## Artifacts

Directory: `profiles/ncu-prefill-kernel-sweep-2026-07-02/`

- `linear_op_t_sweep_1k_8k.csv`: `qus_linear_op_bench` sweep for rowsplit linear shapes.
- `gqa_prefill_t_sweep_1k_8k.csv`: `qus_gqa_attention_bench` prefill sweep.
- `ncu_kernel_sweep_summary.csv`: extracted ncu SOL/occupancy summary.
- `ncu_q4_mlp_gateup_T*.ncu-rep`: representative Q4 rowsplit GEMM ncu reports for the fused
  gate+up-equivalent shape.
- `ncu_q5_mlp_down_T*.ncu-rep`: representative Q5 rowsplit GEMM ncu reports.
- `ncu_gqa_prefill_T*.ncu-rep`: GQA prefill ncu reports.
- `ncu_dense_gemm_e2e_T*.ncu-rep`: first real e2e dense GEMM launch ncu reports.
- `nsys_top_kernel_reference_1k_8k.csv`: filtered reference from the existing e2e nsys sweep.

## Commands

Build and preflight:

```bash
~/.codex/skills/ncu-kernel-profile/scripts/preflight.sh
cmake --build build -j --target qus_linear_op_bench qus_gqa_attention_bench qus_bench
```

Bench sweeps:

```bash
./build/bench/qus_linear_op_bench \
  --all-targets --t-sweep 1024,2048,4096,8192 \
  --warmup 2 --repeat 5 --copy-repeat 4 \
  --csv-out profiles/ncu-prefill-kernel-sweep-2026-07-02/linear_op_t_sweep_1k_8k.csv

./build/bench/qus_gqa_attention_bench \
  --prefill --tokens 1024,2048,4096,8192 \
  --warmup 2 --repeat 5 \
  --csv-out profiles/ncu-prefill-kernel-sweep-2026-07-02/gqa_prefill_t_sweep_1k_8k.csv \
  --json-out profiles/ncu-prefill-kernel-sweep-2026-07-02/gqa_prefill_t_sweep_1k_8k.json
```

ncu template:

```bash
ncu --force-overwrite \
  --section SpeedOfLight --section Occupancy --section ComputeWorkloadAnalysis \
  --kernel-name regex:<kernel-base-name> \
  --launch-skip 0 --launch-count 1 \
  -o profiles/ncu-prefill-kernel-sweep-2026-07-02/<tag> \
  <bench command>
```

Note: ncu exposes the rowsplit template instances under the base name
`linear_rowsplit_gemm_mma_kernel`; Q4/Q5 are isolated by running only one shape/qtype per ncu
command. ncu replay duration is not used as the timing source; use the bench medians for timing.
The Q4 MLP point uses the fused `MlpGateUp34816x5120` bench shape as a combined gate+up equivalent:
actual prefill calls `m.gate` and `m.up` separately, but the pair has the same total output rows.

## E2E hotspot reference

These are kernel totals from the existing post-GQA nsys sweep. They are included only to identify
which kernel families need ncu evidence.

| kernel | 1k | 2k | 4k | 8k |
|---|---:|---:|---:|---:|
| rowsplit_gemm_mma_q4 | 172.9 ms / 43.7% | 337.2 ms / 44.0% | 662.8 ms / 44.0% | 1316.7 ms / 43.5% |
| rowsplit_gemm_mma_q5 | 169.7 ms / 42.9% | 330.8 ms / 43.1% | 642.8 ms / 42.7% | 1258.3 ms / 41.6% |
| dense_gemm | 28.4 ms / 7.2% | 41.9 ms / 5.5% | 72.3 ms / 4.8% | 137.8 ms / 4.6% |
| gqa_attention_prefill | 2.3 ms / 0.6% | 7.2 ms / 0.9% | 22.4 ms / 1.5% | 83.3 ms / 2.8% |
| gdn_state_passing | 5.3 ms / 1.3% | 11.4 ms / 1.5% | 26.0 ms / 1.7% | 53.4 ms / 1.8% |

Readout: rowsplit Q4+Q5 GEMM is the absolute hotspot at every requested length: 86.5%, 87.1%,
86.6%, and 85.1% of kernel time from 1k to 8k. GQA grows with `T^2`, but after the optimization it is
still only 2.8% at 8k.

## Bench useful efficiency

`linear_op_bench` reports useful GEMM TFLOP/s against its measured local dense bf16 MMA ceiling
(`220.584 TFLOP/s` in this run). `gqa_attention_bench` reports useful causal attention TFLOP/s
against the official RTX 5090 dense bf16/FP32-accumulate peak (`209.5 TFLOP/s`).

| kernel bench | metric | 1k | 2k | 4k | 8k |
|---|---|---:|---:|---:|---:|
| Q4 MLP gate+up equivalent | median | 2.238 ms | 4.388 ms | 8.964 ms | 17.476 ms |
| Q4 MLP gate+up equivalent | useful TC% | 73.9% | 75.4% | 73.9% | 75.8% |
| Q4 MLP gate+up equivalent | TFLOP/s | 163.1 | 166.4 | 162.9 | 167.1 |
| Q5 MLP down | median | 1.234 ms | 2.411 ms | 4.681 ms | 9.020 ms |
| Q5 MLP down | useful TC% | 67.0% | 68.7% | 70.7% | 73.4% |
| Q5 MLP down | TFLOP/s | 147.9 | 151.4 | 156.0 | 161.9 |
| GQA prefill | median | 0.154 ms | 0.437 ms | 1.415 ms | 4.888 ms |
| GQA prefill | useful TC% | 40.0% | 56.4% | 69.6% | 80.5% |
| GQA prefill | TFLOP/s | 83.8 | 118.1 | 145.8 | 168.7 |

Shape notes:

- Q4 representative: `MlpGateUp34816x5120`, a fused gate+up-equivalent bench shape. In the real
  prefill path, `mlp_tail()` launches `m.gate` and `m.up` separately, so this is the largest Q4 MLP
  work package, not a single exact e2e launch.
- Q5 representative: `MlpDown5120x17408`, the largest Q5 prefill rowsplit GEMM and an exact prefill
  single-launch shape.
- Other linear shapes in `linear_op_t_sweep_1k_8k.csv` show the same trend: rowsplit GEMM efficiency
  improves or stays flat as `T` grows, and remains compute/tensor-pipe oriented rather than DRAM
  oriented.

## ncu hardware utilization

This table reports ncu hardware counters, not useful FLOP accounting. `tensor pipe` is extracted from
Compute Workload Analysis when ncu reports Tensor as the highest-utilized pipeline.

| kernel | metric | 1k | 2k | 4k | 8k |
|---|---|---:|---:|---:|---:|
| Q4 MLP gate+up equivalent | tensor pipe | 70.4% | 68.3% | 71.5% | 71.0% |
| Q4 MLP gate+up equivalent | SM | 70.4% | 68.3% | 71.5% | 71.0% |
| Q4 MLP gate+up equivalent | memory | 42.2% | 41.0% | 42.9% | 42.7% |
| Q4 MLP gate+up equivalent | DRAM | 7.8% | 9.9% | 7.8% | 9.1% |
| Q4 MLP gate+up equivalent | achieved occupancy | 32.8% | 33.0% | 33.2% | 33.2% |
| Q5 MLP down | tensor pipe | 62.3% | 63.7% | 65.2% | 66.1% |
| Q5 MLP down | SM | 62.3% | 63.7% | 65.2% | 66.1% |
| Q5 MLP down | memory | 40.5% | 41.4% | 42.5% | 43.9% |
| Q5 MLP down | DRAM | 3.5% | 2.6% | 2.1% | 3.0% |
| Q5 MLP down | achieved occupancy | 30.6% | 31.5% | 32.3% | 32.8% |
| GQA prefill | tensor pipe | 38.7% | 50.3% | 64.4% | 72.4% |
| GQA prefill | SM | 38.7% | 50.3% | 64.4% | 72.4% |
| GQA prefill | memory | 23.7% | 27.4% | 32.6% | 35.5% |
| GQA prefill | DRAM | 4.6% | 3.7% | 2.6% | 1.8% |
| GQA prefill | achieved occupancy | 14.1% | 15.0% | 15.8% | 16.2% |
| dense_gemm e2e | tensor pipe | n/a | n/a | n/a | n/a |
| dense_gemm e2e | SM | 11.2% | 15.1% | 17.9% | 18.6% |
| dense_gemm e2e | memory | 50.5% | 67.8% | 80.7% | 83.4% |
| dense_gemm e2e | DRAM | 1.6% | 2.0% | 2.5% | 2.8% |
| dense_gemm e2e | achieved occupancy | 19.5% | 32.9% | 52.0% | 75.9% |

Readout:

- The Q4 MLP gate+up-equivalent rowsplit GEMM is already near a stable ~70-71% ncu tensor-pipe SOL
  with only ~33% achieved occupancy. It is tensor/SM bound, not DRAM-bound.
- Q5 rowsplit GEMM is lower, rising from 62% to 66% tensor-pipe SOL across the sweep. This looks like
  the largest remaining high-value target because it consumes almost the same e2e time as Q4 but has
  lower hardware utilization.
- GQA is now behaving as intended: tensor-pipe utilization rises with context and reaches 72.4% ncu
  tensor-pipe SOL at 8k. Its useful bench efficiency reaches 80.5% at 8k; the gap is expected because
  ncu counts the whole kernel pipeline, including softmax/control/epilogue.
- `dense_gemm` is not a tensor-core GEMM. The source is the correctness-only scalar FMA path under
  `src/kernels/linear/reference/linear_generic_dense.cuh`; ncu reports no Tensor highest-pipe value.
  It is L1/TEX/memory dominated and remains a secondary hotspot behind rowsplit GEMM.

## Conclusion

The current kernel-level bottleneck is decisively the rowsplit low-bit GEMM path, not GQA. Across
1k-8k, Q4+Q5 rowsplit GEMM accounts for ~85-87% of kernel time. GQA prefill is now a small e2e share
even at 8k, while its own long-context useful TC efficiency is already in the target range.

The next optimization target should be `linear_rowsplit_gemm_mma_kernel`, especially Q5
`MlpDown5120x17408` and then the other Q5 projection shapes. Q4 is also large, but ncu shows it is
already around ~71% tensor-pipe SOL; Q5 is consistently lower and has more visible headroom.
