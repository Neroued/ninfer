# Qwen3.6-27B serving performance

[Performance index](../performance.md) · [Measurement and publication rules](methodology.md)

On this page: [run records](#scope-and-run-records), [context profile](#no-speculation-context-profile),
[single-request decode](#single-request-speculative-decode), [decode saturation](#decode-saturation),
[comparisons](#comparisons-and-limitations), [reproduction](#reproduction-and-reports).

## Scope and run records

All runs use the [common RTX 5090 serving profile](methodology.md#common-serving-profile).
The tables below specify differences; all runs use stochastic sampling. Dates identify historical
campaign labels. Run IDs are local to this page and link to the reproduction command, artifact
path, and local report directory.

| Run | Campaign date | Weights ID | Measurement | Context ceiling | C | CUDA driver API | KV capacity (tokens) |
|---|---|---|---|---:|---|---|---|
| [G0](#g0) | 2026-07-20 | `groupwise-int` | MTP0 | 262,144 | 1 | 13.1 | Not separately recorded |
| [N0](#n0) | 2026-07-31 | `nvfp4` | MTP0 | 262,144 | 1 | 13.3 | Not separately recorded |
| [G3](#g3) | 2026-07-24 | `groupwise-int` | MTP3 serial corpus | 262,144 | 1 | 13.1 | Not separately recorded |
| [N3](#n3) | 2026-08-11 | `nvfp4` | MTP3 corpus C=1 | 131,072 | 1 | 13.3 | auto → 131,072 |
| [GS](#gs) | 2026-08-11 | `groupwise-int` | MTP3 saturation | 16,384 | 1, 2, 4, 8 | 13.3 | auto → C × 16,384 |
| [NS](#ns) | 2026-08-11 | `nvfp4` | MTP3 saturation | 16,384 | 1, 2, 4, 8 | 13.3 | auto → C × 16,384 |

| Run | Tested Git revision |
|---|---|
| [G0](#g0) | `0795169393cab0f2c16246d4bac20dee735dc2a4` |
| [N0](#n0) | `b3d4d0f50b868711c62432bbd68e746217a2f49a` |
| [G3](#g3) | `5ea3242a206cdb0c4c1beaeb9d8a3048e6248423` |
| [N3](#n3) | `f4f21cc36bd1a83cbc046f668719d591dc9c1e2e` |
| [GS](#gs) | `26da9df7c1b3d3c04ea7bbd730271aa01d00742a` |
| [NS](#ns) | `26da9df7c1b3d3c04ea7bbd730271aa01d00742a` |

Both registered weight profiles have MTP0 and MTP3 phase results and C=1, 2, 4, 8 saturation
results. N3 phase statistics come from the C=1 point of a corpus-makespan run; full makespan results
are not published here. EvalScope accuracy is recorded in the [groupwise-int model card](../../model-cards/Qwen3.6-27B-NInfer/README.md#evaluation)
and [NVFP4 model card](../../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md#evaluation).

## No-speculation context profile

MTP0 uses the [single-request method](methodology.md#single-request-phases), five samples per context.

### groupwise-int

Run [G0](#g0).

| Prompt tokens | Samples | Prefill phase (tok/s) | Server TTFT (ms) | Decode phase (tok/s) |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 3,218.1 ± 4.3 | 2,392.4 ± 3.0 | 77.6 ± 0.1 |
| 64,512 | 5 | 2,655.9 ± 2.9 | 24,335.7 ± 25.2 | 70.7 ± 0.1 |
| 130,048 | 5 | 2,185.3 ± 0.3 | 59,590.3 ± 8.9 | 64.5 ± 0.1 |
| 260,096 | 5 | 1,614.8 ± 0.6 | 161,221.8 ± 62.5 | 54.8 ± 0.1 |

### nvfp4

Run [N0](#n0).

| Prompt tokens | Samples | Prefill phase (tok/s) | Server TTFT (ms) | Decode phase (tok/s) |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 11,191.5 ± 70.2 | 692.5 ± 4.3 | 86.4 ± 0.5 |
| 64,512 | 5 | 6,298.5 ± 97.6 | 10,288.6 ± 159.3 | 78.0 ± 1.2 |
| 130,048 | 5 | 4,204.7 ± 14.1 | 31,012.5 ± 104.6 | 71.2 ± 0.2 |
| 260,096 | 5 | 2,510.6 ± 16.8 | 103,761.1 ± 698.8 | 59.9 ± 0.3 |

## Single-request speculative decode

Five samples per reasoning fixture; category rows pool three fixtures × five seeds. Values are
mean ± sample standard deviation of per-request phase rates and speculative ratios. Output budgets
are 65,536 tokens for reasoning and 4,096 for other scenarios.

### groupwise-int

Run [G3](#g3), MTP3.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 10,686.2 ± 553.8 | 175.4 ± 1.0 | 77.9% ± 0.9% | 3.34 ± 0.03 |
| `long_decode_aime26_15` | 5 | 61,604.2 ± 5,677.9 | 161.9 ± 2.8 | 73.4% ± 1.7% | 3.20 ± 0.05 |
| `long_decode_aime26_30` | 5 | 47,339.8 ± 9,162.2 | 172.2 ± 0.9 | 78.8% ± 0.8% | 3.36 ± 0.02 |

#### Cross-scenario decode

| Category | Samples | Decode phase (tok/s) | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 167.0 ± 5.4 | 72.3% ± 3.5% | 3.17 ± 0.11 |
| Story | 15 | 112.6 ± 9.4 | 37.8% ± 5.9% | 2.13 ± 0.18 |
| Translation | 15 | 161.5 ± 11.3 | 68.3% ± 7.2% | 3.05 ± 0.22 |
| Structured | 15 | 193.0 ± 18.8 | 88.7% ± 11.7% | 3.66 ± 0.35 |

### nvfp4

Run [N3](#n3), MTP3.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 12,053.4 ± 820.9 | 231.0 ± 3.0 | 80.2% ± 1.2% | 3.41 ± 0.04 |
| `long_decode_aime26_15` | 5 | 63,109.0 ± 5,426.9 | 213.1 ± 4.2 | 76.3% ± 2.0% | 3.29 ± 0.06 |
| `long_decode_aime26_30` | 5 | 57,166.4 ± 9,204.9 | 223.3 ± 1.8 | 81.1% ± 1.5% | 3.43 ± 0.04 |

#### Cross-scenario decode

| Category | Samples | Decode phase (tok/s) | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 220.3 ± 8.2 | 74.2% ± 4.0% | 3.23 ± 0.12 |
| Story | 15 | 148.8 ± 11.6 | 39.2% ± 5.7% | 2.18 ± 0.17 |
| Translation | 15 | 213.6 ± 12.2 | 70.5% ± 6.0% | 3.12 ± 0.18 |
| Structured | 15 | 252.2 ± 16.3 | 89.8% ± 8.0% | 3.69 ± 0.24 |

## Decode saturation

Runs [GS](#gs), [NS](#ns) use the [sustained-wave method](methodology.md#decode-saturation):
293 prompt tokens per request, an 8,192-token output budget, and one wave per C.

| Weights ID | C | Steady (s) | Avg batch | Steady decode (tok/s) | MTP acceptance (wave) | Speedup vs. C1 | Wave makespan (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `groupwise-int` | 1 | 43.01 | 1.00 | 185.8 | 68.2% | 1.00× | 44.23 |
| `groupwise-int` | 2 | 65.01 | 2.00 | 247.0 | 69.0% | 1.33× | 66.67 |
| `groupwise-int` | 4 | 102.02 | 4.00 | 309.5 | 68.4% | 1.67× | 107.49 |
| `groupwise-int` | 8 | 118.02 | 8.00 | 535.0 | 68.3% | 2.88× | 125.20 |
| `nvfp4` | 1 | 39.01 | 1.00 | 202.4 | 69.3% | 1.00× | 40.46 |
| `nvfp4` | 2 | 39.01 | 2.00 | 399.7 | 71.4% | 1.97× | 41.82 |
| `nvfp4` | 4 | 44.01 | 4.00 | 699.7 | 69.3% | 3.46× | 47.92 |
| `nvfp4` | 8 | 55.01 | 8.00 | 1,146.9 | 68.6% | 5.67× | 58.57 |

All 30 requests reached their output limit without a request, CUDA, or out-of-memory failure,
producing 245,760 completion tokens. At C=8, available device memory after startup was 2.66 GiB
for groupwise-int and 2.18 GiB for NVFP4. These are sustained-generation stress results.

## Comparisons and limitations

The weight profiles use the same fixtures, seeds, sampling, and output budgets, but revisions,
driver API versions, and MTP3 context ceilings differ as recorded above. Quantization can change
sampled continuations. These are historical fixed-workload results, not an isolated quantization
comparison. MTP0 and MTP3 use different workloads; no per-scenario speculative speedup is reported.

## Reproduction and reports

Build [ninfer-serve](../../README.md#quick-start) and run from the repository root with Python 3.11:

```bash
export NINFER_BENCH_PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
```

See the common [run/report conventions](methodology.md#publishing-and-updating-results).

### G0

Historical reports: `profiles/bench/serve_corpus_20260720/`.

The original directory also contains other target/mode runs. This command selects only the MTP0 profile used here.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_27b_mtp0
```

### N0

Historical reports: `profiles/bench/serve_corpus_27b_nvfp4_w8_20260731/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_27b_nvfp4_w8_20260731
```

### G3

Historical reports: `profiles/bench/serve_corpus_27b_mtp3_20260724/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp3 \
  --output profiles/bench/serve_corpus_27b_mtp3_20260724
```

### N3

Historical reports: `profiles/bench/concurrent_corpus_27b_nvfp4_mtp3_20260811/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_27b_nvfp4_mtp3_20260811
```

### GS

Historical reports: `profiles/bench/concurrent_decode_27b_mtp3_20260811/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_mtp3_20260811
```

### NS

Historical reports: `profiles/bench/concurrent_decode_27b_nvfp4_mtp3_20260811/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_nvfp4_mtp3_20260811
```

[Runner usage](../../tools/bench/README.md#serving-corpus-benchmark) describes the report files.
Older corpus runs may retain only point reports and server logs.
