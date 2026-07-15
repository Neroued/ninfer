# Qwen3.6-27B GQA Frontier-Tier CUDA Graph Qualification

Date: 2026-07-15

Environment: NVIDIA GeForce RTX 5090, driver 591.86, CUDA 13.1, `sm_120a`, the 17,502,555,648-byte
`out/qwen3_6_27b_rtx5090.ninfer` artifact, and the public `ninfer_bench` Engine route.

## Retained policy

The target owns independent finite tables over execution frontier E:

- ordinary variants cover `[0,C-1]` with preferred ends
  `127,511,2047,4095,8197,16389,32767`, followed by one final interval;
- MTP variants cover `[0,C-2K]`; their ends preserve final-AR split transitions and the concrete
  INT8 T=K+1 function transitions for K=3,4,5;
- ranges are sorted, non-overlapping, gap-free, and have no eager fallback inside their legal
  domain;
- after the producer grid reaches its fixed split cap, the remaining long domain is one interval.

An ordinary interval `[a,b]` captures GQA envelope `[a+1,b+1]`. An MTP interval uses separate
target-verify, batch, and AR envelopes derived from `[a,b]` and K. Every variant is warmed, captured,
and first-replayed from a restored representative device frontier and recurrent/control state.
Physical cache planes are initialized once to a valid zero representation before capture; this does
not publish a reusable prefix.

## Graph memory

`cudaMemGetInfo` before preparation and after all retained executables measured:

| Capacity/configuration | Executables | Observed preparation consumption |
|---|---:|---:|
| 4096, K=0 | 4 ordinary | 22.00 MiB |
| 4096, K=5 | 8 ordinary/aligned + 7 MTP | 76.00 MiB |
| 262144, K=0 | 8 ordinary | 45.25 MiB |
| 262144, K=5 | 16 ordinary/aligned + 12 MTP | peak 512.32 MiB |

Repeated cold 256K K=5 construction showed lower driver consumption on another run, so planning
uses the observed peak rather than the minimum. The target allowance model reserves 6 MiB per
ordinary or short-window MTP executable and 82 MiB per MTP executable whose final visible window is
above 4096. The 256K K=5 allowance is 548 MiB, 35.68 MiB above the observed peak. The existing
post-capture check remains authoritative for actual construction.

The real 256K K=5 public Engine construction succeeds in 32GB and reports 16.28 GiB weights,
9.78 GiB sequence storage, 160.6 MiB workspace, and 8.77 GiB physical INT8 KV payload.

## Replay boundaries and product performance

Short ordinary boundaries `127/128`, `511/512`, and `2047/2048` were exercised in one graph-enabled
Engine. Decode stayed at 64.40-66.38 output tokens/s; the matching eager route measured
51.11-54.10 tokens/s.

For a 256K-capacity Engine, the transition into the final ordinary interval was exercised at prompt
frontiers 32767/32768:

| Route | E=32767 | E=32768 |
|---|---:|---:|
| CUDA Graph | 61.06 t/s | 60.04 t/s |
| eager | 49.93 t/s | 51.06 t/s |

The K=5 MTP table was replayed on both sides of its short boundaries at 118/119, 122/123, 154/155,
502/503, 2038/2039, and 2048/2049. The final 256K MTP transition at 32758/32759 produced
146.38/151.13 output tokens/s with full draft acceptance. No range used cursor-derived capture or
runtime recapture.

Representative commands:

```bash
build/bench/ninfer_bench --weights out/qwen3_6_27b_rtx5090.ninfer \
  -pg '32767,2;32768,2' -r 1 --warmup 0 --max-ctx 262144 \
  --kv-dtype int8 --mtp-draft-tokens 0
build/bench/ninfer_bench --weights out/qwen3_6_27b_rtx5090.ninfer \
  -pg '32758,6;32759,6' -r 1 --warmup 0 --max-ctx 262144 \
  --kv-dtype int8 --mtp-draft-tokens 5
```

These measurements qualify the 27B graph-integrated launch selection and memory plan. The future
35B target must retain the same interface semantics but choose and measure its own target-private
variant tables and allowance.
