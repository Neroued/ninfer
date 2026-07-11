# GQA decode / MTP-verify vs FlashAttention — comparison and split-KV fix

Date: 2026-07-08
Kernels under study: `qus::kernels` small-T split-KV path
(`src/kernels/launcher/gqa_attention_decode.cu`,
`src/kernels/kernel/gqa_attention_decode.cuh`).
Reference: FlashAttention `flash_attn_with_kvcache` and `flash_attn_func`
(`flash-attn` 2.8.3.post1), conda env `vllm-bench`.
Device: RTX 5090 (CC 12.0).

This document extends the prefill parity audit
(`docs/2026-07-08-gqa-prefill-fa-parity-audit.md`) to the decode (T=1) and
MTP-verify (T=2/3/4) regimes, records the measured FA gap, and documents the
position-aware split-KV fix landed from it.

## 0. Ground rule

Every performance number below is **[MEASURED]** — a median from a CUDA-event
bench run in this session (ours via `build/bench/qus_gqa_attention_bench`, FA via
`tools/bench/flash_attn_gqa_bench.py`). Reproduction commands are in §6. Code
facts are **[SOURCE]** with file:line.

## 1. Shapes and method

- Decode: q `[1, 1, 24, 256]`, KV cache `[1, pos+1, 4, 256]`, append one token at
  index `pos`, causal, `scale = 0.0625`.
- MTP-verify: q `[1, T, 24, 256]` (T = 1..4), append T tokens at index `context`,
  attend causally over `context + T` keys.
- Both paths (ours) run the same small-T split-KV kernel plus a reduce kernel:
  `gqa_attention_small_t_tc_partial_kernel` → `gqa_attention_small_t_reduce_output_kernel`
  (`src/kernels/launcher/gqa_attention_decode.cu`).
- Timing: hot cache (KV re-read each iteration), `--warmup 20 --repeat 100
  --min-time-ms 300..500`. FA uses `num_splits=0` (its own heuristic).

FA comparators added to `tools/bench/flash_attn_gqa_bench.py`:

- `--decode` → `flash_attn_with_kvcache` with `seqlen_q = 1`.
- `--verify --verify-tokens T --verify-context C` → `flash_attn_with_kvcache`
  with `seqlen_q = T`.
- `--tokens T --context C --attention-only` → `flash_attn_func` (prefill path),
  used as an alternate FA small-T path.

## 2. Decode (T=1) — original state [MEASURED]

| pos | ours (before) | FA | useful-KV BW ours / FA |
|-----|---------------|-----|------------------------|
| 2048  | 35.44 us | 19.61 us | 236.8 / 427.9 GB/s |
| 2882  | 35.41 us | 26.12 us | 333.5 / 452.0 GB/s |
| 8192  | 45.79 us | 35.26 us | 732.8 / 951.7 GB/s |
| 32768 | 118.32 us | 106.00 us | 1134.4 / 1266.2 GB/s |

FA was faster at every context (1.11×–1.81×). Our T=1 sat at a ~35 us floor for
both pos=2048 and pos=2882 — latency/overhead bound, not bandwidth bound.

## 3. MTP-verify (T=2/3/4) — original state [MEASURED]

Ours (`--append-small-t`) vs FA (`flash_attn_with_kvcache`), median us:

| context | T | ours | FA | winner |
|---------|---|------|-----|--------|
| 2048  | 1 | 35.48 | 19.67 | FA 1.80× |
| 2048  | 2 | 19.86 | 35.30 | ours 1.78× |
| 2048  | 3 | 19.94 | 35.63 | ours 1.79× |
| 2048  | 4 | 19.91 | 35.49 | ours 1.78× |
| 8192  | 1 | 45.48 | 35.94 | FA 1.27× |
| 8192  | 2 | 32.89 | 87.95 | ours 2.67× |
| 8192  | 3 | 33.48 | 88.13 | ours 2.63× |
| 8192  | 4 | 33.48 | 88.07 | ours 2.63× |
| 32768 | 1 | 118.04 | 106.23 | FA 1.11× |
| 32768 | 2 | 109.72 | 303.28 | ours 2.76× |
| 32768 | 3 | 110.71 | 304.13 | ours 2.75× |
| 32768 | 4 | 112.77 | 303.94 | ours 2.70× |

FA's alternate small-T path (`flash_attn_func`) is equally slow: T=2/C=8192 =
93 us, T=2/C=32768 = 316 us, T=4/C=8192 = 91 us, T=4/C=32768 = 317 us. So FA has
no fast small-T-verify path over a long context.

Two structural facts explain the table:

- FA's `flash_attn_with_kvcache` only enables split-KV for `seqlen_q == 1`. At
  T≥2 it drops splitting and streams the whole KV cache from few CTAs
  (useful-KV BW collapses ~1263→442 GB/s at C=32768). We keep adaptive split-KV
  for T=2..4, so we win 1.8×–2.8×.
- Our own T=1 was *slower than our own T=2* at the same context (2048: 35.5 vs
  19.9 us) — the anomaly that motivated the fix below.

## 4. Root cause of the T=1 anomaly [SOURCE]

Both T=1 and T≥2 use the same small-T kernel; only the **split count** differed.

- Host `gqa_small_t_split_upper_bound` special-cased `tokens <= 1` → fixed
  `kGqaDecodeSplits = 192` (`include/qus/kernels/gqa_attention.h:12`); T≥2 got an
  adaptive tiered count.
- Device `gqa_small_t_active_splits` had the same `tokens <= 1` special case,
  disabling the in-kernel position-aware early-exit for T=1.

Over-splitting T=1 to 192 inflated the partial scratch (write 192 partials, reduce
sums 192 per output element). At C=2048 the bench byte model shows scratch
4.68 MiB vs 8.00 MiB useful KV — ~37% of traffic is pure split overhead.

Deeper defect: the host split count was keyed on `kv.max_context` (the allocation
ceiling), not the live context. The bench sizes `max_context ≈ actual context`,
so it *looked* adaptive; in production the `KVCache` is allocated with
`options_.max_ctx = --max-context` (e.g. 131072, `engine.cpp:483`,
`main.cpp:169`), so `gqa_small_t_split_upper_bound` returns 192
for **all** small T (the final `min(·,192)` cap). I.e. production over-split T=1
*and* T≥2 at short/mid context; the bench's T≥2 adaptiveness was a bench artifact.
The measured T=1=35 us (192 splits at actual ctx 2048) is the true production cost.

## 5. Fix: size split-KV by the live attention window

Drive the split count off the actual window `kv.pos + tokens` instead of
`max_context`. `kv.pos` is the host-known pre-round base position (engine advances
it only afterward in `read_round_output`, `engine.cpp:640-641`), and decode/verify
positions run `[base, base+T)`, so `window = kv.pos + tokens` is exact. The kernel
still receives the real `max_context` for cache-bounds checks only.

Changes:

- `src/kernels/launcher/gqa_attention_decode.cu`: `gqa_small_t_split_upper_bound`
  now takes `window` (no `tokens` special case); `gqa_attention_small_t_launch`
  computes `window = kv.pos + tokens` and uses it for both the partial grid and
  the reduce `partial_splits`. `max_context` is still passed to the kernel.
- `src/kernels/kernel/gqa_attention_decode.cuh`: `gqa_small_t_active_splits` drops
  the `tokens <= 1` special case, so T=1 and T≥2 share the same position-aware
  early-exit.
- `bench/gqa_attention_bench.cu`: set `kv.pos = context` on the small-T path and
  drop the matching `tokens <= 1` special case in the bench's `small_t_active_splits`
  model so reported `splits`/scratch match the kernel.

The split count no longer depends on `max_context` at all, so the bench (with
`kv.pos` set) now reflects production behavior directly.

## 6. Results after fix [MEASURED]

Decode (T=1), ours before → after vs FA:

| pos | before | after | FA T=1 | after verdict |
|-----|--------|-------|--------|---------------|
| 2048  | 35.44 us | 19.93 us | 19.61 | parity |
| 2882  | 35.41 us | 20.08 us | 26.12 | ours 1.30× |
| 8192  | 45.79 us | 27.20 us | 35.26 | ours 1.30× |
| 32768 | 118.32 us | 105.59 us | 106.00 | parity |

Small-T (ours, after fix; T=1 now fastest at every context):

| context | T=1 | T=2 | T=3 | T=4 |
|---------|-----|-----|-----|-----|
| 2048  | 20.03 | 20.21 | 20.28 | 20.09 |
| 8192  | 27.32 | 32.88 | 33.40 | 33.46 |
| 32768 | 105.43 | 109.62 | 110.86 | 112.49 |

The bench now reports the true split count for T=1 (C=2048: `splits=33`, scratch
0.80 MiB, redundancy 1.10 — was 192 / 4.68 MiB / 1.59). Combined with the
unchanged T≥2 win, ours now matches or beats FA across the entire decode +
MTP-verify space (T=1..4). T≥2 numbers are unchanged by the fix.

Reproduction:

```bash
# ours
./build/bench/qus_gqa_attention_bench --decode
for C in 2048 8192 32768; do for T in 1 2 3 4; do \
  ./build/bench/qus_gqa_attention_bench --append-small-t --tokens $T --context $C \
    --warmup 20 --repeat 100 --min-time-ms 300; done; done

# FA (conda env vllm-bench)
python tools/bench/flash_attn_gqa_bench.py --decode --decode-pos 2048,2882,8192,32768 \
  --warmup 20 --repeat 100 --min-time-ms 500
python tools/bench/flash_attn_gqa_bench.py --verify --verify-tokens 1,2,3,4 \
  --verify-context 2048,8192,32768 --warmup 20 --repeat 100 --min-time-ms 300
```

## 7. Verification

- Build: `ninja qus_gqa_attention_test qus_gqa_attention_bench` — clean.
- Correctness: `ctest -R qus_gqa_attention_test` — passed (covers small-T prefill
  T=1..6, offsets 0/1/17/128, T=65/offset=384 partial-tile, decode positions to
  8191).
- Memory safety: `compute-sanitizer --tool memcheck` on the test — 0 errors.

Output correctness is independent of the split count (the partial kernel uses the
device `positions` and the reduce handles any active count; the workspace is sized
for the 85 cap), so the fix is a pure scheduling change.
