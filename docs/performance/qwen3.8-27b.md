# Qwen3.8-27B serving performance

[Performance index](../performance.md) · [Measurement and publication rules](methodology.md)

On this page: [run records](#scope-and-run-records), [context profile](#no-speculation-context-profile),
[single-request decode](#single-request-speculative-decode), [corpus makespan](#corpus-makespan),
[decode saturation](#decode-saturation),
[comparisons](#comparisons-and-limitations), [reproduction](#reproduction-and-reports).

## Scope and run records

All runs use the [common RTX 5090 serving profile](methodology.md#common-serving-profile).
The tables below specify differences; all runs use stochastic sampling. Dates identify historical
campaign labels. Run IDs are local to this page and link to the reproduction command, artifact
path, and local report directory.

| Run | Campaign date | Weights ID | Measurement | Context ceiling | C | CUDA driver API | KV capacity (tokens) |
|---|---|---|---|---:|---|---|---|
| [G0](#g0) | 2026-08-31 | `groupwise-int` | MTP0 | 262,144 | 1 | 13.3 | 262,144 |
| [N0](#n0) | 2026-08-17 | `nvfp4` | MTP0 | 262,144 | 1 | 13.3 | 262,144 |
| [G3](#g3) | 2026-08-30 | `groupwise-int` | MTP3 corpus | 131,072 | 1, 2, 4, 8 | 13.3 | auto; see makespan |
| [N3](#n3) | 2026-08-17 | `nvfp4` | MTP3 corpus | 131,072 | 1, 2, 4, 8 | 13.3 | auto; see makespan |
| [GD](#gd) | 2026-09-06 | `groupwise-int` | DFlash2 K=7 corpus | 131,072 | 1 | 13.3 | auto → 131,072 |
| [ND](#nd) | 2026-09-06 | `nvfp4` | DFlash2 K=7 corpus | 131,072 | 1 | 13.3 | auto → 131,072 |
| [NS](#ns) | 2026-08-17 | `nvfp4` | MTP3 saturation | 16,384 | 1, 2, 4, 8 | 13.3 | auto → C × 16,384 |

| Run | Tested Git revision |
|---|---|
| [G0](#g0) | `5e4bf313cb2f8b0603e00bf3b42e7ab3ec6d927a` |
| [N0](#n0) | `f08597d6eaafce5b875934aaa85854fcd5426df8` |
| [G3](#g3) | `d9dbe1ce4d1a53deec2349e669a429a000c54d01` |
| [N3](#n3) | `32c9881b6783949df4999422a764b3dcaa111b13` |
| [GD](#gd) | `03177b910e70f783b00c4f980ce0d1896a6b8592` |
| [ND](#nd) | `03177b910e70f783b00c4f980ce0d1896a6b8592` |
| [NS](#ns) | Not recorded in the retained point reports |

Both weight profiles cover MTP0, MTP3 C=1/2/4/8 corpus results, and DFlash2 K=7 C=1 corpus
results. DFlash2 uses artifacts with included companion weights and the terminal-settlement fix.
MTP3 G3/N3 are historical baselines using earlier artifacts. DFlash2 K=15 and concurrent DFlash2
results are not published on this page.

NS fills in the detailed evidence for the previously published README saturation excerpt. Its
retained point reports establish the configuration and values, but do not identify the tested Git
revision; the nearby N3 corpus revision must not be assumed to apply to NS.

## No-speculation context profile

MTP0 uses the [single-request method](methodology.md#single-request-phases), five samples per context.

### groupwise-int

Run [G0](#g0).

| Prompt tokens | Samples | Prefill phase (tok/s) | Server TTFT (ms) | Decode phase (tok/s) |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 3,274.7 ± 11.3 | 2,349.2 ± 8.4 | 79.8 ± 0.4 |
| 64,512 | 5 | 2,696.1 ± 12.7 | 23,952.3 ± 112.4 | 73.6 ± 0.4 |
| 130,048 | 5 | 2,183.5 ± 11.0 | 59,607.5 ± 299.8 | 66.3 ± 0.4 |
| 260,096 | 5 | 1,609.7 ± 5.3 | 161,674.5 ± 533.7 | 56.2 ± 0.6 |

### nvfp4

Run [N0](#n0).

| Prompt tokens | Samples | Prefill phase (tok/s) | Server TTFT (ms) | Decode phase (tok/s) |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 8,340.4 ± 13.0 | 931.6 ± 1.6 | 71.2 ± 0.1 |
| 64,512 | 5 | 5,297.9 ± 259.2 | 12,281.1 ± 561.5 | 65.7 ± 0.8 |
| 130,048 | 5 | 3,544.7 ± 25.3 | 36,853.5 ± 259.4 | 59.6 ± 0.9 |
| 260,096 | 5 | 2,203.1 ± 13.4 | 118,354.8 ± 717.2 | 52.9 ± 2.3 |

## Single-request speculative decode

These are per-request phase statistics from the C=1 points of the [corpus runs](#corpus-makespan),
not separate experiments. Each reasoning fixture has five seeds; each category pools three
fixtures × five seeds. Values are mean ± sample standard deviation of per-request rates and ratios.
The prompt set, seeds, sampling, and output budgets match across weights and speculative modes;
continuations can differ. AIME 15 reaches 65,536 tokens in all G3 samples and averages 65,414.4
in N3. DFlash2 termination and repetition outcomes follow the tables.

### groupwise-int

MTP3: [G3](#g3) at C=1. DFlash2 K=7: [GD](#gd) at C=1.

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 1,537.4 ± 317.8 | 193.4 ± 5.5 | 72.5% ± 3.1% | 3.17 ± 0.09 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 150.1 ± 2.3 | 53.0% ± 1.3% | 2.59 ± 0.04 |
| `long_decode_aime26_30` | 5 | 46,245.4 ± 10,867.7 | 171.1 ± 27.7 | 63.6% ± 15.7% | 2.91 ± 0.47 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode phase (tok/s) | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 200.3 ± 7.8 | 76.3% ± 4.2% | 3.29 ± 0.12 |
| Story | 15 | 130.4 ± 12.0 | 37.9% ± 6.5% | 2.14 ± 0.19 |
| Translation | 15 | 198.1 ± 10.2 | 74.9% ± 5.5% | 3.25 ± 0.17 |
| Structured | 15 | 224.4 ± 13.6 | 89.5% ± 7.4% | 3.68 ± 0.22 |

#### DFlash2 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | DFlash2 acceptance | DFlash2 tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 1,402.0 ± 165.7 | 224.2 ± 10.8 | 64.6% ± 3.1% | 5.52 ± 0.22 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 133.5 ± 4.9 | 34.7% ± 1.6% | 3.43 ± 0.11 |
| `long_decode_aime26_30` | 5 | 49,412.2 ± 13,931.5 | 175.2 ± 71.5 | 49.7% ± 26.4% | 4.48 ± 1.85 |

The AIME 30 mean includes the [repetition-loop sample](#dflash2-completion-outcomes).

#### DFlash2 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode phase (tok/s) | DFlash2 acceptance | DFlash2 tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 191.4 ± 12.9 | 53.5% ± 5.2% | 4.74 ± 0.36 |
| Story | 15 | 84.0 ± 20.9 | 15.5% ± 7.4% | 2.09 ± 0.52 |
| Translation | 15 | 181.9 ± 34.8 | 50.2% ± 12.1% | 4.51 ± 0.85 |
| Structured | 15 | 267.3 ± 30.7 | 80.7% ± 10.9% | 6.65 ± 0.76 |

### nvfp4

MTP3: [N3](#n3) at C=1. DFlash2 K=7: [ND](#nd) at C=1.

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 1,465.4 ± 417.3 | 195.2 ± 4.6 | 76.0% ± 2.4% | 3.28 ± 0.07 |
| `long_decode_aime26_15` | 5 | 65,414.4 ± 271.9 | 151.4 ± 2.0 | 56.2% ± 1.1% | 2.69 ± 0.03 |
| `long_decode_aime26_30` | 5 | 50,023.4 ± 14,839.1 | 167.5 ± 23.7 | 64.6% ± 14.9% | 2.94 ± 0.45 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode phase (tok/s) | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 194.3 ± 6.1 | 76.4% ± 3.9% | 3.29 ± 0.12 |
| Story | 15 | 126.1 ± 10.9 | 37.4% ± 5.8% | 2.12 ± 0.17 |
| Translation | 15 | 192.3 ± 11.9 | 75.0% ± 6.5% | 3.25 ± 0.19 |
| Structured | 15 | 219.8 ± 8.6 | 90.8% ± 5.1% | 3.72 ± 0.15 |

#### DFlash2 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | DFlash2 acceptance | DFlash2 tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 1,290.4 ± 102.7 | 321.1 ± 15.6 | 67.2% ± 2.9% | 5.70 ± 0.20 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 183.4 ± 8.0 | 35.2% ± 2.8% | 3.47 ± 0.19 |
| `long_decode_aime26_30` | 5 | 38,697.2 ± 7,222.9 | 199.6 ± 10.2 | 38.6% ± 1.9% | 3.70 ± 0.14 |

#### DFlash2 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode phase (tok/s) | DFlash2 acceptance | DFlash2 tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 265.5 ± 21.7 | 53.9% ± 5.5% | 4.77 ± 0.39 |
| Story | 15 | 121.3 ± 30.6 | 16.8% ± 7.9% | 2.17 ± 0.55 |
| Translation | 15 | 255.9 ± 50.6 | 51.1% ± 12.2% | 4.58 ± 0.85 |
| Structured | 15 | 356.8 ± 40.8 | 78.1% ± 10.7% | 6.46 ± 0.75 |

### DFlash2 completion outcomes

Runs [ND](#nd) and [GD](#gd).

Entries are stop-token / output-limit counts. A stop token denotes termination, not task
correctness; this campaign does not assign answer-accuracy or prompt-compliance scores.

| Workload | Samples per profile | NVFP4 stop / limit | Groupwise-int stop / limit |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 5 / 0 | 5 / 0 |
| `long_decode_aime26_15` | 5 | 0 / 5 | 0 / 5 |
| `long_decode_aime26_30` | 5 | 5 / 0 | 4 / 1 |
| Code | 15 | 5 / 10 | 1 / 14 |
| Story | 15 | 10 / 5 | 10 / 5 |
| Translation | 15 | 15 / 0 | 15 / 0 |
| Structured | 15 | 0 / 15 | 1 / 14 |
| **Total** | **75** | **40 / 35** | **36 / 39** |

All ten AIME 15 samples exhaust the 65,536-token output budget. Their rates characterize sustained
long decode rather than completed reasoning. Code, story, and structured-output rates likewise
include the output-limit samples counted above.

The groupwise-int AIME 30 sample with seed `9060622443728853932` repeats `-?` 32,046 times in its
reasoning field, has empty final content, and reaches 65,536 completion tokens. Its 96.8% acceptance
and 302.6 tok/s describe a repetition loop. It remains in the five-sample mean of 175.2 ± 71.5 tok/s;
the other four samples stop naturally and average 143.3 ± 6.3 tok/s. This four-sample statistic is
supplementary and does not replace the fixed-corpus result.

## Corpus makespan

The [fixed-corpus method](methodology.md#corpus-makespan) uses 75 requests per point and shuffle
seed `20260811`. Each row is one complete run. Corpus rates use the full makespan; acceptance is
the ratio of summed accepted and drafted tokens. Average batch covers transitions and drain.
C=1 supplies the phase statistics above.

### MTP3 groupwise-int

Run [G3](#g3).

| C | Requests | Computed prefill tokens | Decode tokens | Makespan (s) | Requests/s | Corpus prefill (tok/s) | Corpus decode (tok/s) | Avg batch | MTP acceptance | Speedup vs. C1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 15,460 | 747,295 | 4,622.59 | 0.0162 | 3.3 | 161.7 | 1.00 | 58.5% | 1.00× |
| 2 | 75 | 15,460 | 766,184 | 3,580.22 | 0.0209 | 4.3 | 214.0 | 1.91 | 59.5% | 1.29× |
| 4 | 75 | 15,460 | 739,692 | 2,864.48 | 0.0262 | 5.4 | 258.2 | 3.67 | 58.3% | 1.61× |
| 8 | 75 | 15,460 | 697,193 | 2,211.20 | 0.0339 | 7.0 | 315.3 | 4.76 | 58.9% | 2.09× |

All 300 requests completed without a request, CUDA, or out-of-memory failure. C=8 gives the
shortest complete-corpus makespan. `--kv-capacity auto` resolved to 131,072, 262,144, 341,952, and
313,984 tokens at C=1, 2, 4, and 8. C=8 reached a maximum of four waiting requests while admitting
long contexts into the shared pool; no spill, owner degradation/eviction, or search-exhaustion
event occurred.

### MTP3 nvfp4

Run [N3](#n3).

| C | Requests | Computed prefill tokens | Decode tokens | Makespan (s) | Requests/s | Corpus prefill (tok/s) | Corpus decode (tok/s) | Avg batch | MTP acceptance | Speedup vs. C1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 15,460 | 752,160 | 4,670.27 | 0.0161 | 3.3 | 161.1 | 1.00 | 60.8% | 1.00× |
| 2 | 75 | 15,460 | 739,951 | 2,510.78 | 0.0299 | 6.2 | 294.7 | 1.98 | 59.2% | 1.86× |
| 4 | 75 | 15,460 | 713,384 | 1,647.74 | 0.0455 | 9.4 | 432.9 | 3.29 | 58.0% | 2.83× |
| 8 | 75 | 15,460 | 723,602 | 2,164.90 | 0.0346 | 7.1 | 334.2 | 2.36 | 57.6% | 2.16× |

All 300 requests completed without a request, CUDA, or out-of-memory failure. C=4 gives the
shortest complete-corpus makespan. C=8 is limited by memory pressure, which constrains effective
batching and makes the end-to-end result slower than C=4.

The groupwise-int weight arena is 16.672 GiB versus 19.729 GiB for NVFP4. At C=8, the smaller
resident profile permits 313,984 tokens of Device KV versus 187,712 and raises the full-corpus
average batch from 2.36 to 4.76. The best measured point is therefore C=8 for groupwise-int and C=4
for NVFP4.

N3 automatic KV capacity at C=1, 2, 4, 8 resolved to 131,072, 255,360, 232,768, and 187,712
tokens respectively.

### DFlash2 K=7

Runs [ND](#nd), then [GD](#gd), on 2026-09-06 after the terminal-settlement fix. Each computed
15,460 prefill tokens with average decode batch 1.00 and automatic KV capacity 131,072 tokens.

| Weights ID | Requests | Completion tokens | Decode tokens | Makespan (s) | Requests/s | Corpus decode (tok/s) | DFlash2 acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nvfp4` | 75 | 695,432 | 695,357 | 3,611.52 | 0.0208 | 192.5 | 37.0% |
| `groupwise-int` | 75 | 761,763 | 761,688 | 5,181.45 | 0.0145 | 147.0 | 39.2% |

All 150 requests completed without request errors. The earlier interrupted NVFP4 run is excluded;
these are the two complete post-fix runs. [Completion outcomes](#dflash2-completion-outcomes)
include natural stops, output limits, and the groupwise-int repetition outlier.

## Decode saturation

Run [NS](#ns), NVFP4 MTP3, uses the [sustained-wave method](methodology.md#decode-saturation).
The rendered prompt is 335 tokens per request, compared with 293 for the published Qwen3.6
profiles. Each request has an 8,192-token output budget; each C point is one wave.

| C | Steady (s) | Avg batch | Steady decode (tok/s) | MTP acceptance (wave) | Speedup vs. C1 | Wave makespan (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 55.15 | 1.00 | 143.8 | 48.9% | 1.00× | 56.95 |
| 2 | 60.04 | 2.00 | 267.6 | 48.1% | 1.86× | 61.29 |
| 4 | 68.17 | 4.00 | 461.1 | 45.8% | 3.21× | 72.14 |
| 8 | 81.09 | 8.00 | 766.6 | 46.0% | 5.33× | 87.29 |

All 15 recorded requests reached the output limit, producing 122,880 completion tokens. There
are 264 selected steady intervals. This is sustained decode, without an answer/structure audit.
The retained point reports do not identify the tested Git revision.

## Comparisons and limitations

### DFlash2 phase rates versus historical MTP3

ND versus N3 and GD versus G3 at C=1. These historical percentages use the displayed rounded
means; future comparisons follow the unrounded-report rule in the methodology.

Changes below compare the displayed per-request mean decode rates in the tables above. G3/N3 use
their separately listed revisions and earlier artifacts; prompts, seeds, output budgets, and shared
sampling/context settings match. Stochastic output lengths differ, and these percentages do not
isolate the backend's effect on identical generated tokens.

| Workload | NVFP4 DFlash2 change | Groupwise-int DFlash2 change |
|---|---:|---:|
| `long_decode_aime26_01` | +64.5% | +15.9% |
| `long_decode_aime26_15` | +21.1% | -11.1% |
| `long_decode_aime26_30` | +19.2% | +2.4% |
| Code | +36.6% | -4.4% |
| Story | -3.8% | -35.6% |
| Translation | +33.1% | -8.2% |
| Structured | +62.3% | +19.1% |

The groupwise-int AIME 30 change includes the [repetition outlier](#dflash2-completion-outcomes) and does not establish a
reasoning-performance improvement.

### DFlash2 corpus versus historical MTP3

ND versus N3 and GD versus G3, using complete-corpus metrics at C=1.
Against the historical C=1 MTP3 corpus, NVFP4's full-makespan decode rate rises 19.5% and makespan
falls 22.7%; groupwise-int's rate falls 9.1% and makespan rises 12.1%. These compare the recorded
campaigns, not an isolated backend change: revisions differ, and the stochastic backends produce
different continuations and token totals. No fresh MTP3 baseline was run.

MTP0 and speculative scenarios measure different workloads. No per-scenario baseline/speculative
speedup is reported. Neither the phase-rate nor makespan comparisons establish answer accuracy.

## Reproduction and reports

Build [ninfer-serve](../../README.md#quick-start) and run from the repository root with Python 3.11:

```bash
export NINFER_BENCH_PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
```

See the common [run/report conventions](methodology.md#publishing-and-updating-results).

### G0

Historical reports: `profiles/bench/serve_corpus_qwen3_8_27b_groupwise_mtp0_20260831/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_qwen3_8_27b_groupwise_mtp0_20260831
```

### N0

Historical reports: `profiles/bench/serve_corpus_qwen3_8_27b_nvfp4_mtp0_20260817/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_qwen3_8_27b_nvfp4_mtp0_20260817
```

### G3

Historical reports: `profiles/bench/concurrent_corpus_qwen3_8_27b_groupwise_mtp3_20260830/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_qwen3_8_27b_groupwise_mtp3_20260830
```

### N3

Historical reports: `profiles/bench/concurrent_corpus_qwen3_8_27b_nvfp4_mtp3_20260817/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_qwen3_8_27b_nvfp4_mtp3_20260817
```

### GD

Historical reports: `profiles/bench/dflash2-single-kv-fix-20260906/groupwise-int/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b.ninfer \
  --mode dflash2_7 --sampling stochastic --suite corpus-makespan --concurrency 1 \
  --max-context 131072 --kv-capacity auto --prefill-chunk 1024 --port 18080 \
  --output profiles/bench/dflash2-single-kv-fix-20260906/groupwise-int
```

### ND

Historical reports: `profiles/bench/dflash2-single-kv-fix-20260906/nvfp4/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode dflash2_7 --sampling stochastic --suite corpus-makespan --concurrency 1 \
  --max-context 131072 --kv-capacity auto --prefill-chunk 1024 --port 18080 \
  --output profiles/bench/dflash2-single-kv-fix-20260906/nvfp4
```

### NS

Historical reports: `profiles/bench/concurrent_decode_qwen3_8_27b_nvfp4_mtp3_20260817/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp3 --sampling stochastic --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_qwen3_8_27b_nvfp4_mtp3_20260817
```

[Runner usage](../../tools/bench/README.md#serving-corpus-benchmark) describes the report files.
Older corpus runs may retain only point reports and server logs.
