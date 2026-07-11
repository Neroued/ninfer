# INT8 GQA short-context optimization report — 2026-07-10

## Result

On one RTX 5090, the final `T=1..6` sweep over context lengths
`{0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072}`
has no sampled INT8 latency regression greater than 3% versus BF16 for `context <= 4096`.
The worst canonical cell is `T=2, context=1`: 18.57 us versus 18.23 us, or 1.9% slower.

The primary `T=6` targets all pass:

| Context | Target | Final INT8 | BF16 | INT8 speedup |
|---:|---:|---:|---:|---:|
| 256 | <= 22.0 us | 20.30 us | 21.37 us | 1.053x |
| 512 | <= 22.0 us | 20.35 us | 21.44 us | 1.054x |
| 1024 | <= 22.0 us | 20.48 us | 21.55 us | 1.052x |
| 2048 | <= 22.0 us | 20.58 us | 22.01 us | 1.069x |
| 4096 | <= 26.5 us | 24.05 us | 28.05 us | 1.166x |
| 8192 | <= 32.5 us | 31.62 us | 35.87 us | 1.134x |

At long context the previous INT8 advantage is preserved. Relative to the pre-change INT8
baseline, `T=6` changes by +1.3% at 32768, -0.8% at 65536, and +0.4% at 131072; all are within
the 3% preservation gate.

## Measurement method and bandwidth meaning

Operator command shape:

```bash
build/bench/qus_gqa_attention_bench \
  --append-small-t --tokens "$T" --context "$C" --kv-dtype int8
build/bench/qus_gqa_attention_bench \
  --append-small-t --tokens "$T" --context "$C" --kv-dtype bf16
```

Each cell is the benchmark median over 100 timed launches after warmup. The sweep used the real
Qwen3.6-27B GQA shape: head dimension 256, 24 query heads, and 4 KV heads. Raw results are in
`profiles/bench/gqa-i8-short-opt-20260710/final_sweep.csv` and `final_tiny.csv` in the measurement
workspace.

`useful KV GB/s` is the physical useful cache payload divided by elapsed time. One INT8 cache
vector is 264 bytes (256 code bytes plus eight FP16 scales), while BF16 is 512 bytes. Therefore an
INT8 result can have lower physical GB/s and still be faster because it moves 1.939x fewer useful
bytes. The `BF16-equivalent` column below multiplies INT8 physical bandwidth by `512/264`; it is a
logical throughput aid, not measured DRAM traffic.

## T=6 latency and effective bandwidth

| Context | INT8 us | INT8 physical GB/s | INT8 BF16-equivalent GB/s | BF16 us | BF16 physical GB/s | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20.56 | 0.6 | 1.2 | 20.52 | 1.2 | 0.998x |
| 32 | 20.49 | 3.9 | 7.6 | 21.09 | 7.4 | 1.029x |
| 64 | 20.69 | 7.1 | 13.8 | 21.12 | 13.6 | 1.021x |
| 128 | 20.80 | 13.6 | 26.4 | 20.60 | 26.6 | 0.990x |
| 256 | 20.30 | 27.3 | 52.9 | 21.37 | 50.2 | 1.053x |
| 512 | 20.35 | 53.8 | 104.3 | 21.44 | 99.0 | 1.054x |
| 1024 | 20.48 | 106.2 | 205.9 | 21.55 | 195.8 | 1.052x |
| 2048 | 20.58 | 210.8 | 408.8 | 22.01 | 382.3 | 1.069x |
| 4096 | 24.05 | 360.2 | 698.6 | 28.05 | 599.0 | 1.166x |
| 8192 | 31.62 | 547.5 | 1061.8 | 35.87 | 936.0 | 1.134x |
| 16384 | 48.48 | 714.0 | 1384.7 | 52.21 | 1285.8 | 1.077x |
| 32768 | 70.24 | 985.5 | 1911.3 | 118.18 | 1135.9 | 1.683x |
| 65536 | 128.18 | 1080.0 | 2094.5 | 206.12 | 1302.4 | 1.608x |
| 131072 | 214.24 | 1292.2 | 2506.1 | 370.11 | 1450.6 | 1.728x |

## Cross-T comparison

The table reports `BF16 latency / INT8 latency`; values above 1 favor INT8.

| T | C=0 | C=128 | C=256 | C=1024 | C=4096 | C=8192 | C=32768 | C=131072 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.068x | 1.174x | 1.131x | 1.087x | 1.203x | 1.292x | 2.768x | 1.844x |
| 2 | 1.000x | 1.126x | 1.113x | 1.031x | 1.253x | 1.534x | 2.753x | 1.890x |
| 3 | 1.074x | 1.044x | 1.014x | 0.995x | 1.165x | 1.324x | 2.306x | 1.852x |
| 4 | 1.089x | 1.013x | 0.998x | 1.009x | 1.182x | 1.317x | 2.288x | 1.844x |
| 5 | 1.079x | 1.086x | 1.060x | 0.995x | 1.104x | 1.220x | 2.284x | 1.839x |
| 6 | 0.998x | 0.990x | 1.053x | 1.052x | 1.166x | 1.134x | 1.683x | 1.728x |

## What changed

- `T=4` uses a 16-warp CTA through context 1024.
- `T=5` uses a 16-warp CTA through context 1024, with a 32-warp CTA and one 32-key tile per split
  for windows 129 through 512.
- `T=6` uses 12 warps through context 2048. The narrow 129 through 160 window uses 24 warps and one
  32-key tile per split to avoid a nearly empty second tile.
- `T=6` uses a 64-key tile from 2K through 8K. Its arena is dynamic shared memory so the larger tile
  does not increase static shared memory for the long-context specialization.
- Near 8K, `T=6` is capped at 42 splits: 4 KV heads times 42 equals 168 CTAs, which fits in one wave
  on the 170-SM RTX 5090. A 43rd split caused a measurable second-wave latency cliff.

Nsight Compute supports the mechanism rather than just the end result:

| Shape | Kernel duration | Block | Waves/SM | Achieved occupancy | Registers/thread |
|---|---:|---:|---:|---:|---:|
| T6/C1024 before | 28.64 us | 192 threads | 0.20 | 12.37% | 168 |
| T6/C1024 after | 19.01 us | 384 threads | 0.40 | 24.63% | 168 |
| T5/C256 after | 11.84 us | 1024 threads | 0.21 | 64.86% | 64 |

The T6/C1024 partial kernel is 33.6% faster under NCU. This confirms that short context was
under-parallelized: the grid has far fewer CTAs than SMs, so increasing useful work per CTA raises
active warps without changing the arithmetic. At long context the kernel becomes cache/DRAM
traffic dominated, which is why INT8 compression already produced a clear advantage there.

## End-to-end `qus_bench` check

The current v4 real-weight artifact does not contain MTP weights, so this check covers ordinary
T=1 decode, not the new T=5/6 verification geometries. The available older MTP artifact was rejected
by the current loader with `q5090 bad magic`; no result from it is reported.

Using v4 weights, CUDA graph decode, one warmup, and two measured repetitions:

| Test | INT8 decode tok/s | BF16 decode tok/s | INT8 advantage | INT8 prefill tok/s | BF16 prefill tok/s |
|---|---:|---:|---:|---:|---:|
| pp256+tg16 | 82.79 | 82.28 | 0.6% | 1943.9 | 1955.9 |
| pp4096+tg16 | 77.93 | 72.97 | 6.8% | 2847.6 | 2948.3 |

Artifacts: `profiles/bench/gqa-i8-short-opt-20260710/qus_bench_int8.json` and
`qus_bench_bf16.json`.

## Verification

- `ctest --test-dir build -R gqa_attention --output-on-failure`: passed.
- Added numerical-oracle cases for T5/C256, T6/C128, and T6/C8192 so every new launch geometry and
  the dynamic-shared-memory path is exercised.
- `compute-sanitizer --tool memcheck`: 0 errors.
- `compute-sanitizer --tool racecheck`: 0 hazards, 0 errors, 0 warnings.
- Final NCU reports are under `profiles/ncu/gqa-short-i8/`.

Limitations: results are from one RTX 5090 and one software stack; the operator sweep is hot-cache
CUDA-event timing, while `qus_bench` is host-visible full-model timing. Short-context differences
below roughly 1% should be treated as parity, not as a durable win.
