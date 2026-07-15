# Qwen3.6-35B A3 Cached-Only GQA Qualification

Date: 2026-07-15

Target: cached-only causal grouped-query attention with head dimension 256, 16 query heads, two KV
heads, group ratio 8, BF16 Q/output, sequential absolute I32 positions, and either BF16 or INT8-G64
populated cache. The qualified performance domain is split-KV `T=1..6` and visible context
`L<=262144`; T=1 is the current model-required shape.

Environment: NVIDIA GeForce RTX 5090, driver 591.86, CUDA 13.1, `sm_120a`, Nsight Compute
2025.4.1, and a Release build.

## Implementation and effect

A3 uses the same geometry-specialized split-KV producer/reducer machinery as qualified A1. A
compile-time cache-effect type removes new K/V pointers, quantization, cache writes, and append
synchronization from the concrete A3 kernel. It retains Q staging, BF16 or INT8 cache decode, QK/PV
MMA, online softmax, partial storage, and reduction. Producer and reducer call the same dtype-aware
active-split function.

The public Op receives one physical layer view, caller workspace, device positions, and an explicit
host execution envelope. Device positions define the causal mask. The envelope only selects a
launch capacity and concrete INT8 function valid over a finite interval. A3 mutates no cache plane
and owns no Program frontier.

## Correctness

```bash
cmake --build build-release -j4 --target \
  ninfer_gqa_attention_test ninfer_gqa_attention_bench
build-release/tests/ninfer_gqa_attention_test
```

The independent operation test covers both 24Q/4KV and 16Q/2KV geometries, BF16 and INT8-G64 cache
formats, A2-then-A3 parity against fused A1, exact cache-code/scale preservation, prompt/small-T
overlap, and CUDA Graph replay at different device positions inside one envelope. It completes with
`OK gqa_attention correctness`.

## Canonical benchmark

```bash
build-release/bench/ninfer_gqa_attention_bench \
  --cached-small-t --geometry 35b --tokens 1,2,3,4,5,6 \
  --context 0,128,512,2048,8192,32768,131072,261120 --kv-dtype bf16
build-release/bench/ninfer_gqa_attention_bench \
  --cached-small-t --geometry 35b --tokens 1,2,3,4,5,6 \
  --context 0,128,512,2048,8192,32768,131072,261120 --kv-dtype int8
```

At the longest tested context, Release CUDA-event medians and the benchmark's complete modeled
traffic are:

| Route | Median | Modeled bandwidth | Same-size cold copy ceiling | Fraction |
|---|---:|---:|---:|---:|
| BF16 T=1 | 367.92 us | 1461.4 GB/s | 1485.6 GB/s | 98.4% |
| BF16 T=6 | 392.14 us | 1408.3 GB/s | 1485.6 GB/s | 94.8% |
| INT8-G64 T=1 | 213.50 us | 1305.1 GB/s | 1480.7 GB/s | 88.1% |
| INT8-G64 T=6 | 244.78 us | 1197.9 GB/s | 1480.7 GB/s | 80.9% |

The complete T/context logs are retained at:

- `profiles/bench/qwen3_6_35b_gqa_a3/bf16.txt`;
- `profiles/bench/qwen3_6_35b_gqa_a3/int8.txt`.

## NCU qualification

One long-context T=1 producer from each cache format was collected with application replay and one
matching launch:

```bash
ncu --force-overwrite --set basic --replay-mode application \
  --kernel-name regex:'gqa_attention_small_t_tc_partial_bf16_kernel' \
  --launch-skip 0 --launch-count 1 -o profiles/ncu/qwen3_6_35b_gqa_a3_t1_c261120_bf16__basic.ncu-rep \
  build-release/bench/ninfer_gqa_attention_bench --cached-small-t --geometry 35b \
  --kv-dtype bf16 --tokens 1 --context 261120 --profile-once
ncu --force-overwrite --set basic --replay-mode application \
  --kernel-name regex:'gqa_attention_decode_i8_tiled_kernel' \
  --launch-skip 0 --launch-count 1 -o profiles/ncu/qwen3_6_35b_gqa_a3_t1_c261120_int8__basic.ncu-rep \
  build-release/bench/ninfer_gqa_attention_bench --cached-small-t --geometry 35b \
  --kv-dtype int8 --tokens 1 --context 261120 --profile-once
```

| Route | Grid x block | NCU duration | DRAM SOL | Compute SOL | Achieved occupancy | Registers/thread |
|---|---:|---:|---:|---:|---:|---:|
| BF16 T=1 | `340 x 64` | 436.00 us | 73.69% | 23.21% | 8.30% | 254 |
| INT8-G64 T=1 | `340 x 256` | 240.10 us | 72.40% | 21.26% | 33.06% | 128 |

NCU classifies both producers as DRAM bottlenecked; the basic set does not collect warp-stall
breakdown. Event timing against the same-size copy control remains the steady-state roofline
evidence. The reports and exported CSV/text views are retained under `profiles/ncu/` with the tags
shown above.

This evidence qualifies A3 for the exact 35B small-T domain. It does not claim prompt-scale A3
performance beyond T=6 or another head geometry/device.
