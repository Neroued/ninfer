# MTP M1 W8G32 Linear Baseline

Date: 2026-07-03. Scope: M1 W8G32 row-split `linear()` baseline for the five MTP dense/fused
families. This is a correctness-era baseline only; M1 has no throughput gate.

## Command

```bash
cmake --build build --target qus_linear_op_bench -j
./build/bench/qus_linear_op_bench \
  --all-targets --t-sweep 1,2,3,5,6,16,64 \
  --warmup 2 --repeat 5 --copy-repeat 4 \
  --csv-out profiles/mtp-m1-w8g32-linear/linear_op_w8g32_t_sweep.csv
```

Artifact:

- `profiles/mtp-m1-w8g32-linear/linear_op_w8g32_t_sweep.csv`

Measured local ceilings from the run:

- stream-copy ceiling: `1504.616 GB/s`
- dense bf16 tensor-core probe: `220.722 TFLOP/s`

The CSV keeps the existing `linear_op_bench` logical-minimum byte columns. For W8G32 T>1 in M1, the
multi-step fallback re-streams weight tiles across T tiles, so `achieved_gbs`, `DRAM_%`, and
`roofline_us` are convenience readouts against logical minimum traffic, not measured or modeled
implementation traffic. The baseline table below uses timing and useful TFLOP/s only.

## W8G32 Rows

`cold_us` is the cold-cache median across 5 repeats. W8G32 LargeT currently uses the M1 generic
multi-step GEMM fallback, not the Q4/Q5/Q6 tensor-core MMA path.

| shape | payload MiB | T1 us | T2 us | T3 us | T5 us | T6 us | T16 us | T64 us | T64 TFLOP/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `MtpFc5120x10240` | 53.1 | 714.016 | 244.992 | 250.816 | 261.408 | 265.312 | 422.528 | 1334.560 | 5.03 |
| `MtpAttnIn14336x5120` | 74.4 | 361.280 | 300.320 | 306.464 | 314.176 | 318.464 | 539.936 | 1759.936 | 5.34 |
| `MtpOProj5120x6144` | 31.9 | 420.640 | 152.576 | 156.800 | 164.832 | 167.200 | 256.928 | 795.936 | 5.06 |
| `MtpMlpGateUp34816x5120` | 180.6 | 552.224 | 554.240 | 560.416 | 572.704 | 582.912 | 1107.232 | 4158.432 | 5.49 |
| `MtpMlpDown5120x17408` | 90.3 | 1213.728 | 421.152 | 433.152 | 453.920 | 468.224 | 738.624 | 2449.696 | 4.66 |

## Readout

- T1 is the generic W8 GEMV path needed for future MTP decode correctness.
- T2/T3/T5/T6/T16 and T64 all execute successfully through the public `linear()` API.
- The T64 rows are intentionally slow compared with existing Q4/Q5/Q6 MMA rows because W8G32 M1
  uses the correctness-first generic multi-step path. W8 MMA or fused epilogues are M5 work.
