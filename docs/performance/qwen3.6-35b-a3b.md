# Qwen3.6-35B-A3B serving performance

[Performance index](../performance.md) · [Methodology](methodology.md)

On this page: [configuration](#configuration-and-runs), [context profile](#no-speculation-context-profile),
[single-request decode](#single-request-speculative-decode), [corpus makespan](#corpus-makespan),
[decode saturation](#decode-saturation), [termination](#termination-and-anomalies),
[reproduction](#reproduction-and-reports).

## Configuration and runs

Measured on 2026-09-07 at revision `487f89773f07cb18a2fb841fe0971ec9634d409b`. All 11 measurement
points and 485 formal requests completed without request, CUDA, or out-of-memory failures.

| Setting | Value |
|---|---|
| Model / weights | `qwen3.6-35b-a3b` / `groupwise-int` |
| Artifact | `out/qwen3_6_35b_a3b.ninfer`, including DFlash companion weights |
| GPU | One NVIDIA GeForce RTX 5090, 32 GiB, device 0 |
| Build | Release, `sm_120a` |
| CUDA compile/runtime / driver API | 13.1 / 13.1 / 13.3 |
| NVIDIA driver | 610.74 |
| Route | Loopback OpenAI Chat Completions, `stream=false`, port 18080 |
| KV | INT8 group-64, `--kv-capacity auto` |
| Prefill chunk | 1,024 tokens |
| CUDA Graph / prefix reuse | Enabled / disabled |
| Stochastic sampling | Temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 1.0, frequency penalty 0 |
| Greedy sampling | Exact argmax |
| Speculative proposal head | Optimized (`--lm-head-draft`) |

The build and artifact remain fixed across all points. Each point starts a fresh server; model
loading and startup warmup complete before measurement. Three short startup checks are outside
the formal request count. Run IDs link to their commands and local report directories.

| Run | Workload | Backend / sampling | C | Requests |
|---|---|---|---|---:|
| [P0](#p0) | Long-context phases | None / stochastic | 1 | 20 |
| [M3](#m3) | Complete corpus | MTP K=3 / stochastic | 1, 2, 4, 8 | 300 |
| [DS](#ds) | Complete corpus | DFlash K=7 / stochastic | 1 | 75 |
| [DG](#dg) | Complete corpus | DFlash K=7 / greedy | 1 | 75 |
| [S3](#s3) | Saturated decode | MTP K=3 / stochastic | 1, 2, 4, 8 | 15 |

| Run | C | Context ceiling | Actual KV capacity (tokens) | Device memory available after startup (GiB) |
|---|---|---|---|---|
| P0 | 1 | 262,144 | 262,144 | 7.86 |
| M3 | 1 | 262,144 | 262,144 | 6.39 |
| M3 | 2 | 262,144 | 524,288 | 3.30 |
| M3 | 4 | 262,144 | 734,336 | 1.06 |
| M3 | 8 | 262,144 | 680,128 | 1.38 |
| DS | 1 | 262,144 | 262,144 | 5.88 |
| DG | 1 | 262,144 | 262,144 | 5.83 |
| S3 | 1 | 16,384 | 16,384 | 9.28 |
| S3 | 2 | 16,384 | 32,768 | 9.04 |
| S3 | 4 | 16,384 | 65,536 | 8.54 |
| S3 | 8 | 16,384 | 131,072 | 7.23 |

## No-speculation context profile

Run [P0](#p0) uses four Long NIAH fixtures with five seeds each, thinking disabled, and a
128-token output budget. Each request stops naturally after 17 completion tokens. Inputs are
submitted serially through the corpus runner's C=1 path with shuffle seed `20260811`.
Values are arithmetic mean ± sample standard deviation of per-request phase metrics.

| Prompt tokens | Samples | Prefill phase (tok/s) | Server TTFT (ms) | Decode phase (tok/s) |
|---|---|---|---|---|
| 7,680 | 5 | 17,705.4 ± 234.6 | 437.7 ± 6.0 | 338.3 ± 5.5 |
| 64,512 | 5 | 11,758.0 ± 122.3 | 5,510.0 ± 57.3 | 298.1 ± 2.0 |
| 130,048 | 5 | 8,317.1 ± 109.3 | 15,685.1 ± 208.0 | 260.6 ± 4.4 |
| 260,096 | 5 | 5,247.0 ± 30.1 | 49,657.3 ± 283.0 | 213.0 ± 3.3 |

## Single-request speculative decode

The C=1 points of [M3](#m3), [DS](#ds), and [DG](#dg) each supply 75 requests: three long-reasoning
fixtures and twelve scenario fixtures, with five fixed seeds per fixture. Reasoning enables
thinking and uses a 65,536-token output budget. Code, Story, Translation, and Structured each
contain three fixtures, disable thinking, and use a 4,096-token output budget.

Tables use arithmetic mean ± sample standard deviation of per-request decode-phase rates and
speculative ratios. Category rows pool 15 samples. Completion lengths are actual generated tokens.
These C=1 runs also supply the corpus-makespan results below.

### MTP3 stochastic

Run [M3](#m3).

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | Spec acceptance | Spec tokens/round |
|---|---|---|---|---|---|
| `long_decode_aime26_01` | 5 | 8,407.2 ± 2,764.1 | 750.6 ± 22.4 | 83.5% ± 3.8% | 3.50 ± 0.11 |
| `long_decode_aime26_15` | 5 | 64,860.2 ± 1,511.1 | 636.5 ± 9.2 | 72.0% ± 1.0% | 3.16 ± 0.03 |
| `long_decode_aime26_30` | 5 | 55,354.6 ± 7,132.4 | 683.3 ± 4.0 | 79.2% ± 1.1% | 3.38 ± 0.03 |

#### Cross-scenario decode

| Category | Samples | Decode phase (tok/s) | Spec acceptance | Spec tokens/round |
|---|---|---|---|---|
| Code | 15 | 677.2 ± 25.6 | 70.5% ± 3.6% | 3.12 ± 0.11 |
| Story | 15 | 465.2 ± 36.3 | 37.8% ± 5.5% | 2.14 ± 0.16 |
| Translation | 15 | 659.0 ± 35.1 | 67.7% ± 5.8% | 3.03 ± 0.17 |
| Structured | 15 | 779.6 ± 44.3 | 87.5% ± 7.1% | 3.63 ± 0.21 |

### DFlash K=7 stochastic

Run [DS](#ds).

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | Spec acceptance | Spec tokens/round |
|---|---|---|---|---|---|
| `long_decode_aime26_01` | 5 | 8,536.2 ± 3,506.4 | 866.8 ± 47.4 | 66.4% ± 4.0% | 5.65 ± 0.28 |
| `long_decode_aime26_15` | 5 | 65,325.0 ± 471.8 | 641.6 ± 62.4 | 49.7% ± 6.2% | 4.48 ± 0.43 |
| `long_decode_aime26_30` | 5 | 53,756.4 ± 5,693.8 | 732.6 ± 13.3 | 58.1% ± 1.4% | 5.07 ± 0.10 |

#### Cross-scenario decode

| Category | Samples | Decode phase (tok/s) | Spec acceptance | Spec tokens/round |
|---|---|---|---|---|
| Code | 15 | 620.6 ± 43.3 | 42.6% ± 4.1% | 3.98 ± 0.28 |
| Story | 15 | 291.6 ± 58.8 | 12.2% ± 5.4% | 1.85 ± 0.38 |
| Translation | 15 | 547.5 ± 74.5 | 35.2% ± 6.8% | 3.47 ± 0.48 |
| Structured | 15 | 906.4 ± 127.3 | 69.7% ± 12.5% | 5.88 ± 0.88 |

### DFlash K=7 greedy

Run [DG](#dg).

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | Spec acceptance | Spec tokens/round |
|---|---|---|---|---|---|
| `long_decode_aime26_01` | 5 | 5,292.0 ± 0.0 | 903.8 ± 7.6 | 69.7% ± 0.0% | 5.88 ± 0.00 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 626.0 ± 4.3 | 48.0% ± 0.0% | 4.36 ± 0.00 |
| `long_decode_aime26_30` | 5 | 50,629.0 ± 0.0 | 738.0 ± 3.7 | 58.4% ± 0.0% | 5.09 ± 0.00 |

#### Cross-scenario decode

| Category | Samples | Decode phase (tok/s) | Spec acceptance | Spec tokens/round |
|---|---|---|---|---|
| Code | 15 | 627.8 ± 59.8 | 43.0% ± 5.4% | 4.01 ± 0.38 |
| Story | 15 | 290.3 ± 58.3 | 12.0% ± 5.3% | 1.84 ± 0.37 |
| Translation | 15 | 587.5 ± 97.0 | 38.9% ± 8.9% | 3.72 ± 0.63 |
| Structured | 15 | 936.8 ± 73.1 | 72.5% ± 7.5% | 6.07 ± 0.53 |

## Corpus makespan

Each point runs the same 75-request corpus with shuffle seed `20260811` and a fixed HTTP send
order. Exactly C persistent clients submit their next request after reading their current response.
Makespan spans client release through the last complete response, including prefill, request
admission, workload transitions, and drain. Corpus decode throughput uses that full interval;
average batch covers the whole run. Acceptance is the ratio of summed accepted and drafted tokens.

### MTP3

| C | Requests | Computed prefill tokens | Decode tokens | Makespan (s) | Requests/s | Corpus decode (tok/s) | Avg batch | MTP acceptance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 75 | 14,830 | 831,378 | 1,279.40 | 0.0586 | 649.8 | 1.00 | 72.2% |
| 2 | 75 | 14,830 | 827,334 | 898.13 | 0.0835 | 921.2 | 2.00 | 73.2% |
| 4 | 75 | 14,830 | 807,280 | 737.87 | 0.1016 | 1,094.1 | 3.45 | 71.7% |
| 8 | 75 | 14,830 | 861,416 | 679.61 | 0.1104 | 1,267.5 | 6.48 | 73.3% |

The stop-token / output-limit counts at C=1, 2, 4, 8 are 37/38, 34/41, 34/41, and 32/43.
Stochastic continuations and token totals can vary with concurrency.

### DFlash K=7, C=1

| Profile | Requests | Completion tokens | Decode tokens | Makespan (s) | Corpus decode (tok/s) | Spec acceptance |
|---|---|---|---|---|---|---|
| DFlash K=7 stochastic | 75 | 828,778 | 828,703 | 1,318.83 | 628.4 | 46.6% |
| DFlash K=7 greedy | 75 | 797,210 | 797,135 | 1,277.36 | 624.0 | 46.1% |

Each DFlash run computed 14,830 prefill tokens and had average decode batch 1.00.

## Decode saturation

Run [S3](#s3) uses `long_decode_aime26_15`, a 293-token rendered prompt, thinking enabled, and
an 8,192-token output budget per request. Each C point is one sustained wave. Steady throughput
uses only complete one-second intervals satisfying the [full-batch criteria](methodology.md#decode-saturation).
Prefill, ramp-up, and drain are excluded from steady throughput and included in wave makespan.
Acceptance covers the complete wave.

| C | Requests | Steady (s) | Avg batch | Steady decode (tok/s) | MTP acceptance (wave) | Wave makespan (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 11.01 | 1.00 | 642.5 | 68.6% | 12.70 |
| 2 | 2 | 15.99 | 2.00 | 907.2 | 66.3% | 18.03 |
| 4 | 4 | 25.00 | 4.00 | 1,213.5 | 69.6% | 27.27 |
| 8 | 8 | 45.00 | 8.00 | 1,380.7 | 68.0% | 47.94 |

All 15 requests reach the output limit, producing 122,880 completion tokens.

## Termination and anomalies

Checks cover execution failures, token-accounting consistency, termination reasons, and obvious
repetition. No obvious short-cycle repetition was found in the 225 complete C=1 speculative
responses. C>1 retains request usage, termination reasons, and server logs; response-text screening
covers C=1. This performance measurement does not score answer accuracy or prompt compliance.

The following entries are stop-token / output-limit counts. Each reasoning row has five samples
per profile; each category has fifteen. All output-limit samples remain in the reported statistics.

| Workload | MTP3 stop / limit | DFlash stochastic stop / limit | DFlash greedy stop / limit |
|---|---|---|---|
| `long_decode_aime26_01` | 5 / 0 | 5 / 0 | 5 / 0 |
| `long_decode_aime26_15` | 1 / 4 | 1 / 4 | 0 / 5 |
| `long_decode_aime26_30` | 5 / 0 | 5 / 0 | 5 / 0 |
| Code | 0 / 15 | 0 / 15 | 0 / 15 |
| Story | 10 / 5 | 10 / 5 | 10 / 5 |
| Translation | 15 / 0 | 15 / 0 | 15 / 0 |
| Structured | 1 / 14 | 1 / 14 | 0 / 15 |
| Total | 37 / 38 | 37 / 38 | 35 / 40 |

## Reproduction and reports

Run from the repository root using Python 3.11 and the explicit artifact above.
Campaign reports: `profiles/bench/qwen3_6_35b_a3b_retest_20260907_130244/`.
Select an unused `NINFER_PERF_OUTPUT` directory for a new run.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target ninfer-serve

NINFER_BENCH_PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
NINFER_PERF_OUTPUT=profiles/bench/qwen3_6_35b_a3b_performance
NINFER_PERF_COMMON=(
  --serve build/apps/ninfer-serve
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer
  --kv-capacity auto --prefill-chunk 1024 --device 0 --port 18080
)
```

### P0

Report subdirectory: `mtp0_context/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py \
  "${NINFER_PERF_COMMON[@]}" \
  --mode mtp0 --sampling stochastic --suite corpus-makespan \
  --concurrency 1 \
  --max-context 262144 \
  --output "$NINFER_PERF_OUTPUT/mtp0_context"
```

### M3

Report subdirectory: `mtp3_corpus/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py \
  "${NINFER_PERF_COMMON[@]}" \
  --mode mtp3 --sampling stochastic --suite corpus-makespan \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --max-context 262144 \
  --output "$NINFER_PERF_OUTPUT/mtp3_corpus"
```

### DS

Report subdirectory: `dflash7_stochastic/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py \
  "${NINFER_PERF_COMMON[@]}" \
  --mode dflash7 --sampling stochastic --suite corpus-makespan \
  --concurrency 1 \
  --max-context 262144 \
  --output "$NINFER_PERF_OUTPUT/dflash7_stochastic"
```

### DG

Report subdirectory: `dflash7_greedy/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py \
  "${NINFER_PERF_COMMON[@]}" \
  --mode dflash7 --sampling greedy --suite corpus-makespan \
  --concurrency 1 \
  --max-context 262144 \
  --output "$NINFER_PERF_OUTPUT/dflash7_greedy"
```

### S3

Report subdirectory: `mtp3_saturation/`.

```bash
"$NINFER_BENCH_PYTHON" tools/bench/run_serve_concurrency.py \
  "${NINFER_PERF_COMMON[@]}" \
  --mode mtp3 --sampling stochastic --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --max-context 16384 --decode-tokens 8192 \
  --output "$NINFER_PERF_OUTPUT/mtp3_saturation"
```

Each group contains `summary.json`, `summary.csv`, `summary.md`, `points/`, and `server/`.
C=1 `corpus/` directories retain complete responses and per-fixture phase summaries. The campaign
`manifest.json` records the revision, artifact identity, and commands; `anomalies.json` records
termination and repetition screening. See [runner usage](../../tools/bench/README.md#concurrent-serving-benchmark).
