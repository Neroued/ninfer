# GQA decode bandwidth implementation evidence - 2026-07-05

> Historical bf16 baseline. The int8 producer/consumer kernel, 85-split cap,
> and current measurements are documented in
> `docs/2026-07-08-gqa-decode-int8-kernel-redesign.md`.

Design source: `docs/2026-07-05-gqa-decode-bandwidth-design.md`.

## Route selected

The shipped small-T route is measured, not assumed:

| T | partial route | reason |
|---:|---|---|
| 1 | `gqa_attention_small_t_tc_partial_kernel<1, 2>` | best measured long-context T=1 route |
| 2..6 | `gqa_attention_small_t_tc_partial_kernel<T, 4>` | original TC route still beats the stream candidate |

The old scalar T=1 partial launcher and kernel were removed. The CUDA-core stream candidate was
implemented and profiled during this pass, then retired from shipped code because TC won every
measured cell. The rejected hypothesis is preserved here and in `profiles/gqa-stream/`.

## Phase 1 spike

Temporary spike: route T=1 to TC/4-warps, then revert before deliverable work.

Command shape:

```bash
cmake --build build -j --target qus_gqa_attention_test qus_gqa_attention_bench
./build/tests/qus_gqa_attention_test
compute-sanitizer --tool memcheck ./build/tests/qus_gqa_attention_test
for c in 2048 8192 16384 32768; do
  ./build/bench/qus_gqa_attention_bench --append-small-t --tokens 1 --context "$c"
done
```

Result:

| context | median us | useful-KV GB/s |
|---:|---:|---:|
| 2048 | 36.47 | 230.1 |
| 8192 | 46.67 | 719.1 |
| 16384 | 56.67 | 1184.3 |
| 32768 | 124.46 | 1078.5 |

`qus_gqa_attention_test` passed and memcheck reported `ERROR SUMMARY: 0 errors`.

## Copy ceiling

Added `qus_gqa_attention_bench --copy-ceiling --tokens T --context N`.

Command:

```bash
./build/bench/qus_gqa_attention_bench --copy-ceiling --tokens 1 --context 32768
```

Result:

| mode | median us | C_copy GB/s | payload-rate GB/s |
|---|---:|---:|---:|
| hot | 182.12 | 1474.0 | 737.0 |
| cold | 179.46 | 1495.9 | 747.9 |

`C_copy` counts total copy traffic, read plus write, while `payload-rate` uses the read-only useful
KV payload size. Compare total-DRAM-style throughput against `C_copy`; compare useful-KV numbers
against other attention candidates.

## Stream candidate

Candidate B was a CUDA-core `cp.async` stream kernel with K/V staging, one kv-head group per CTA,
single-KV-pass semantics, int4 cache append, int4 partial stores, and the existing small-T
reducer/workspace layout. It was built only to test the design hypothesis and was deleted from the
code after the measurements below.

Measured candidates at T=1, context 32768:

| candidate | median us | useful-KV GB/s | note |
|---|---:|---:|---|
| stream `Bc=32,S=2` | 218.55 | 614.1 | 1 CTA/SM, shared-memory limited |
| stream `Bc=16,S=2` | 130.96 | 1024.9 | best stream candidate |
| stream `Bc=16,S=3` | 158.94 | 844.5 | deeper pipe lost to lower occupancy |
| stream `Bc=16,S=2` + shared pair load | 189.69 | 707.6 | rejected |
| TC `T=1,WarpsPerCta=1` | 124.52 | 1077.9 | slower than 2 warps |
| TC `T=1,WarpsPerCta=2` | 116.94 | 1147.8 | selected |
| TC `T=1,WarpsPerCta=3` | 124.78 | 1075.6 | rejected |

Stream profile artifacts:

- `profiles/gqa-stream/t1_ctx32768_stream.ncu-rep`
- `profiles/gqa-stream/t1_ctx32768_stream_sched.ncu-rep`
- `profiles/gqa-stream/t1_ctx32768_stream_b16_sched.ncu-rep`
- `profiles/gqa-stream/t1_ctx32768_stream_b16_roofline.ncu-rep`

Key limiter for the best stream candidate: `gpu__dram_throughput` was only `59.38%`, while
`sm__throughput` was `69.93%` and `l1tex__throughput` was `75.69%`; scheduler stats showed shared
memory short-scoreboard pressure. The stream candidate is not DRAM-roofline-bound on this shape, so
this pass treats candidate B as falsified rather than leaving it as an alternate implementation.

## Final route sweep

Command:

```bash
for t in 1 2 3 4 5 6; do
  ./build/bench/qus_gqa_attention_bench --append-small-t --tokens "$t" --context 32768
done
```

Result:

| T | route | median us | useful-KV GB/s | redundancy |
|---:|---|---:|---:|---:|
| 1 | TC/2-warps | 117.92 | 1138.2 | 1.04 |
| 2 | TC/4-warps | 109.25 | 1228.6 | 1.03 |
| 3 | TC/4-warps | 110.43 | 1215.6 | 1.04 |
| 4 | TC/4-warps | 111.83 | 1200.3 | 1.05 |
| 5 | TC/4-warps | 113.82 | 1179.3 | 1.07 |
| 6 | TC/4-warps | 116.33 | 1154.0 | 1.08 |

Repeated T=1 runs after route selection were stable around `117.0 us / 1147 GB/s`; a post-cleanup
sanity run measured `118.22 us / 1135.3 GB/s`. This is a large improvement over the old scalar path
but is slightly below the design's strict `>=1150 GB/s` must-hit in some runs.

Final ncu artifacts:

- `profiles/gqa-stream/t1_ctx32768_tc2_final.ncu-rep`: `gpu__dram_throughput = 62.26%`,
  `gpu__time_duration = 136.77 us`.
- `profiles/gqa-stream/t6_ctx32768_tc4_final.ncu-rep`: `gpu__dram_throughput = 63.17%`,
  `gpu__time_duration = 142.40 us`.

These do not meet the stretch `85%` DRAM SOL gate. The selected route is the best measured route in
this implementation pass and preserves T=2..6 performance.

## Verification

Passed:

```bash
cmake --build build -j --target qus_gqa_attention_test qus_gqa_attention_bench
./build/tests/qus_gqa_attention_test
compute-sanitizer --tool memcheck ./build/tests/qus_gqa_attention_test
compute-sanitizer --tool racecheck ./build/tests/qus_gqa_attention_test
```

Sanitizers:

- memcheck: `ERROR SUMMARY: 0 errors`
- racecheck: `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`

Post-change long-context e2e artifact:

```bash
./build/bench/qus_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus \
  -pg 32768,1 --max-ctx 32769 --warmup 0 --repetitions 1 \
  --output json --output-file profiles/gqa-stream/e2e_pg32768_g1_final.json
```

Result from the JSON report: `pp32768+tg1` decode time was `0.033931406 s`
(`29.47122203 tok/s`). This is a post-change full-model measurement, not a before/after paired
attention-layer wall-clock trace.
