# GQA unified small-T evidence - 2026-07-04

This records the measured route selected by the unified `gqa_attention(...)` refactor.

## Route

| T | route | split policy |
|---|---|---|
| 1 | decode-derived scalar split-K partial + reducer | 192 splits |
| 2..5 | split-K tensor-core small-T partial + reducer | device-side `ceil(window / 480)`, clamped to `[4, 192]` |
| 6 | split-K tensor-core small-T partial + reducer | device-side `ceil(window / 512)`, clamped to `[4, 192]` |
| >6 | prompt-scale policy | current prompt kernel |

The public API does not expose these routes.

## Bench Results

Command shape: `./build/bench/qus_gqa_attention_bench --append-small-t --tokens T --context 32768`.

| T | median us | useful-KV GB/s | modeled redundancy | splits |
|---|---:|---:|---:|---:|
| 2 | 109.93 | 1221.1 | 1.03 | 69 |
| 3 | 110.07 | 1219.5 | 1.04 | 69 |
| 4 | 111.78 | 1200.9 | 1.05 | 69 |
| 5 | 114.04 | 1177.1 | 1.07 | 69 |
| 6 | 116.49 | 1152.4 | 1.08 | 65 |

Cold-cache checks:

| T | command | median us | useful-KV GB/s | modeled redundancy |
|---|---|---:|---:|---:|
| 1 | `--append-small-t --tokens 1 --context 32768 --profile-once --cold-cache` | 1034.24 | 129.8 | 1.04 |
| 2 | `--append-small-t --tokens 2 --context 32768 --profile-once --cold-cache` | 113.92 | 1178.2 | 1.03 |
| 6 | `--append-small-t --tokens 6 --context 32768 --profile-once --cold-cache` | 119.81 | 1120.5 | 1.08 |

Prompt-scale baseline for the replaced verify path:

| T | context | route | median us |
|---|---:|---|---:|
| 6 | 32768 | forced prompt baseline | 6082.82 |
| 6 | 32768 | selected small-T route | 116.39 |

## NCU Artifacts

Saved reports:

- `profiles/ncu-gqa-smallt/t1_ctx32768_scalar_metrics.ncu-rep`
- `profiles/ncu-gqa-smallt/t6_ctx32768_tc_target512_metrics.ncu-rep`

Key T=6 partial-kernel metrics from `t6_ctx32768_tc_target512_metrics`:

| metric | value |
|---|---:|
| kernel duration | 141.50 us |
| DRAM throughput | 64.94% |
| SM throughput | 36.20% |
| registers/thread | 254 |
| static shared memory/block | 36.86 KiB |
| active warps vs peak | 12.93% |
| DRAM read bytes | 135.02 MB |
| DRAM write bytes | 29.45 MB |

## Limiter

The tensor-core small-T route removes the old prompt-scale underfilled long-context path and keeps
modeled redundancy under 1.10 for T=2..6, but it does not meet the 85% useful-KV roofline gate.
NCU shows the remaining limiter is not pure DRAM bandwidth: the T=6 partial kernel is constrained by
register/shared-memory occupancy and the full-width tensor-core PV accumulator (`254` registers per
thread, `36.86 KiB` static shared memory, `12.93%` active warps vs peak). Further improvement likely
requires a separate small-T architecture round that changes the output-accumulator layout or reducer
contract rather than more split-count tuning.
