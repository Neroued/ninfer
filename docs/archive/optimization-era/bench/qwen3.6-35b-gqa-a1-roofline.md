# Qwen3.6-35B A1 GQA Qualification

Date: 2026-07-15

Target: append-and-attend causal grouped-query attention with head dimension 256, 16 query heads,
2 KV heads, group ratio 8, BF16 Q/new K/new V/output, and either BF16 or INT8-G64 KV cache. The
qualified domains are split-KV `T=1..6`, prompt chunk `T<=1024`, and visible context
`L<=262144`.

Environment: NVIDIA GeForce RTX 5090, driver 591.86, CUDA 13.1, `sm_120a`, Nsight Compute
2025.4.1, and a Release build. Work started from source revision `19b4a33`. Timings below are CUDA
event medians after benchmark warmup. NCU uses one selected launch and application replay for the
stateful small-T kernel.

This report qualifies A1 only. Standalone KV append A2 is qualified in its
[separate report](qwen3.6-35b-gqa-a2-roofline.md). Cached-only A3 has a known small-T dispatch gap
recorded at the end of this report.

## Final implementation

- Head mapping is a compile-time geometry: 27B remains `24Q/4KV/group=6`, while 35B instantiates
  `16Q/2KV/group=8`. Inner kernel indexing contains no runtime generic-head branch.
- Long-context 35B decode uses 170 splits, giving `2*170=340` partial CTAs, the same grid width as
  the retained 27B `4*85=340` route.
- BF16 and INT8-G64 prompt, small-T, append, reduction, and cache indexing kernels share the
  operation implementation but compile independently for each registered geometry.
- INT8 T=5 uses three query row tiles for group 8. Its 24/12/6-warp launch tiers were measured
  separately; the 12-warp route is retained through a 4096-token window.
- Small-T workspace is transient: 1.35 MiB at T=1 and 8.09 MiB at T=6. Prompt attention uses no
  split workspace.

## Correctness and 27B preservation

```bash
cmake --build build-release --target \
  ninfer_gqa_attention_test ninfer_gqa_attention_bench -j8
build-release/tests/ninfer_gqa_attention_test
cmake --build build-release --target ninfer_qwen3_6_27b_rtx5090 -j8
```

The operation test runs the existing independent attention/cache oracle for both exact geometries.
It covers BF16 and INT8-G64 cache routes, `T=1..6`, prompt tails and nonzero history, standalone
append/cached parity, and graph relaunch position changes. It completes with
`OK gqa_attention correctness`.

The complete 27B benchmark matrix was captured before and after the geometry refactor. At
`L=261120`, BF16 T=1/T=6 changed from 676.16/699.17 us to 674.94/699.87 us, while INT8 changed
from 357.18/383.52 us to 358.03/384.88 us. The 1024-token BF16/INT8 prompt values changed from
42.622/41.592 ms to 42.725/41.649 ms. These are measurement-noise differences; the original
specialized kernel instances and launch policy are preserved.

## Canonical benchmark matrix

```bash
build-release/bench/ninfer_gqa_attention_bench \
  --append-prompt-baseline --geometry 35b --tokens 1024 \
  --context 0,128,512,2048,8192,32768,131072,261120 --kv-dtype bf16
build-release/bench/ninfer_gqa_attention_bench \
  --append-prompt-baseline --geometry 35b --tokens 1024 \
  --context 0,128,512,2048,8192,32768,131072,261120 --kv-dtype int8
build-release/bench/ninfer_gqa_attention_bench \
  --append-small-t --geometry 35b --tokens 1,2,3,4,5,6 \
  --context 0,128,512,2048,8192,32768,131072,261120 --kv-dtype bf16
build-release/bench/ninfer_gqa_attention_bench \
  --append-small-t --geometry 35b --tokens 1,2,3,4,5,6 \
  --context 0,128,512,2048,8192,32768,131072,261120 --kv-dtype int8
```

Prompt chunk T=1024, median milliseconds:

| Context | BF16 | INT8-G64 |
|---:|---:|---:|
| 0 | 0.092 | 0.101 |
| 128 | 0.109 | 0.115 |
| 512 | 0.151 | 0.156 |
| 2,048 | 0.317 | 0.317 |
| 8,192 | 0.974 | 0.948 |
| 32,768 | 3.591 | 3.456 |
| 131,072 | 14.874 | 13.844 |
| 261,120 | 28.861 | 28.173 |

BF16 small-T median microseconds:

| Context | T1 | T2 | T3 | T4 | T5 | T6 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 19.05 | 18.80 | 20.17 | 20.58 | 20.69 | 20.93 |
| 128 | 19.24 | 20.99 | 20.69 | 20.81 | 20.74 | 20.67 |
| 512 | 19.99 | 21.38 | 19.88 | 19.39 | 19.77 | 20.53 |
| 2,048 | 21.94 | 25.56 | 23.75 | 23.88 | 24.44 | 26.10 |
| 8,192 | 34.04 | 40.86 | 39.09 | 39.25 | 41.09 | 44.42 |
| 32,768 | 53.65 | 63.30 | 63.98 | 65.51 | 65.93 | 69.31 |
| 131,072 | 209.69 | 218.29 | 221.58 | 227.25 | 228.29 | 233.62 |
| 261,120 | 368.08 | 380.34 | 382.54 | 388.51 | 389.07 | 394.54 |

INT8-G64 small-T median microseconds:

| Context | T1 | T2 | T3 | T4 | T5 | T6 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17.71 | 18.79 | 20.21 | 19.19 | 20.51 | 20.67 |
| 128 | 18.29 | 19.44 | 20.95 | 20.52 | 20.72 | 20.47 |
| 512 | 19.58 | 19.86 | 19.67 | 19.95 | 19.54 | 20.46 |
| 2,048 | 20.62 | 21.75 | 23.70 | 24.32 | 25.92 | 26.24 |
| 8,192 | 31.70 | 32.07 | 35.95 | 37.90 | 44.27 | 35.70 |
| 32,768 | 43.16 | 44.10 | 48.66 | 50.74 | 64.78 | 68.86 |
| 131,072 | 127.00 | 128.85 | 133.70 | 139.06 | 152.55 | 157.04 |
| 261,120 | 211.41 | 213.50 | 217.95 | 223.38 | 238.12 | 243.39 |

## Roofline qualification

At `L=261120`, the same benchmark's read/write copy control reports 1493.9 GB/s for the BF16
payload size and 1490.4 GB/s for INT8-G64. The A1 byte model counts useful KV traffic, split
scratch, and output traffic consistently with that control:

| Route | Median | Modeled traffic | Fraction of copy roofline |
|---|---:|---:|---:|
| BF16 T=1 | 368.08 us | 1460.8 GB/s | 97.8% |
| BF16 T=6 | 394.54 us | 1399.7 GB/s | 93.7% |
| INT8-G64 T=1 | 211.41 us | 1318.1 GB/s | 88.4% |
| INT8-G64 T=6 | 243.39 us | 1204.7 GB/s | 80.8% |

The 1024-token prompt at `L=261120` sustains 152.09 useful TFLOP/s for BF16 and 155.80 useful
TFLOP/s for INT8-G64, respectively 72.60% and 74.37% of the RTX 5090 dense BF16/FP32-accumulate
peak used by the existing benchmark. This matches the retained 27B kernel family rather than
falling with the smaller head grid.

Representative NCU basic captures use these kernel regexes:

```bash
ncu --set basic --replay-mode application --launch-count 1 \
  --kernel-name regex:'gqa_attention_small_t_tc_partial_bf16_kernel' ...
ncu --set basic --replay-mode application --launch-count 1 \
  --kernel-name regex:'gqa_attention_decode_i8_tiled_kernel' ...
```

| Route | Grid x block | NCU duration | DRAM SOL | Compute SOL | Achieved occupancy | Registers/thread |
|---|---:|---:|---:|---:|---:|---:|
| BF16 T=1 | `340 x 64` | 425.79 us | 73.19% | 23.85% | 8.30% | 254 |
| BF16 T=6 | `340 x 128` | 439.36 us | 69.36% | 46.70% | 16.26% | 253 |
| INT8-G64 T=1 | `340 x 256` | 228.67 us | 68.98% | 22.24% | 33.08% | 128 |
| INT8-G64 T=6 | `340 x 192` | 264.86 us | 61.24% | 35.83% | 24.74% | 168 |

NCU clock control and replay make its durations unsuitable as steady-state latency, but every
long-context route is classified as memory-heavier than compute. Separate traffic captures report
534.84 MB DRAM read for BF16 T=1 and 275.81 MB for INT8 T=1, matching the expected physical cache
payloads. The event timings and measured copy control therefore establish the applicable A1
bandwidth roofline; prompt timing establishes the compute-oriented route.

Retained reports:

- `profiles/ncu/qwen3_6_35b_gqa_prefill_t1024_c261120_bf16__basic.ncu-rep`
- `profiles/ncu/qwen3_6_35b_gqa_prefill_t1024_c261120_int8__basic.ncu-rep`
- `profiles/ncu/qwen3_6_35b_gqa_decode_t1_c261120_bf16__basic.ncu-rep`
- `profiles/ncu/qwen3_6_35b_gqa_decode_t1_c261120_int8__basic.ncu-rep`
- `profiles/ncu/qwen3_6_35b_gqa_decode_t6_c261120_bf16__basic.ncu-rep`
- `profiles/ncu/qwen3_6_35b_gqa_decode_t6_c261120_int8__basic.ncu-rep`

## Remaining unqualified split contract

A3 prompt T=1024 is efficient, but its current `T<=6` route uses the prompt attention launcher
instead of split-KV. At `L=261120`, BF16 cached-only T=1/T=6 measures 15.094/15.405 ms and INT8
measures 13.228/13.510 ms. The qualified fused A1 route takes 0.368/0.395 ms and 0.211/0.243 ms
respectively. A3 therefore remains Adapt existing pending a high-performance small-T cached-only
design.
