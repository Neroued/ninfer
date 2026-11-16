# Serving performance

Recorded measurements cover RTX 5090 and single-/dual-DGX Spark deployments through NInfer's public
HTTP serving route. Each model page contains its run conditions, results, completion outcomes,
and reproduction commands.

Read the [measurement and publication rules](performance/methodology.md) for workload definitions,
metric formulas, statistics, comparison requirements, and the standard result-page format.

## Published coverage

Each cell links to its results. C is configured request concurrency; K is the number of draft tokens.

### DGX Spark

| Model / deployment | Context profile | Single-request speculative decode | Corpus makespan | MTP3 decode saturation |
|---|---|---|---|---|
| Qwen3.8-Flash-Next / one Spark | [MTP0/MTP3, 8K–256K](performance/qwen3.8-flash-next.md#context-profile) | [MTP3](performance/qwen3.8-flash-next.md#single-request-speculative-decode) | [C=1, 2, 4, 8](performance/qwen3.8-flash-next.md#corpus-makespan) | [C=1, 2, 4, 8, 16](performance/qwen3.8-flash-next.md#decode-saturation) |
| Qwen3.8-Flash-Next / two Sparks, TP2 | [MTP0/MTP3, 8K–256K](performance/qwen3.8-flash-next.md#context-profile) | [MTP3](performance/qwen3.8-flash-next.md#single-request-speculative-decode) | [C=1, 2, 4, 8](performance/qwen3.8-flash-next.md#corpus-makespan) | [C=1, 2, 4, 8, 16, 32](performance/qwen3.8-flash-next.md#decode-saturation) |

Both deployments use the mixed-precision NVFP4-expert artifact, FP8 group-128 KV, and a 4,096-token
prefill chunk. Dual Spark uses expert TP and Native communication. C=1 corpus runs also supply the
single-request scenario tables.

### RTX 5090

| Model / weights | MTP0 context profile | Single-request speculative decode | Corpus makespan | MTP3 decode saturation |
|---|---|---|---|---|
| Qwen3.6-27B / `groupwise-int` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-27B / `nvfp4` | [8K–256K](performance/qwen3.6-27b.md#no-speculation-context-profile) | [MTP3](performance/qwen3.6-27b.md#single-request-speculative-decode) | Not published | [C=1, 2, 4, 8](performance/qwen3.6-27b.md#decode-saturation) |
| Qwen3.6-35B-A3B / `groupwise-int` | [8K–256K](performance/qwen3.6-35b-a3b.md#no-speculation-context-profile) | [MTP3; DFlash K=7 stochastic/greedy](performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash C=1](performance/qwen3.6-35b-a3b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.6-35b-a3b.md#decode-saturation) |
| Qwen3.8-27B / `groupwise-int` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |
| Qwen3.8-27B / `nvfp4` | [8K–256K](performance/qwen3.8-27b.md#no-speculation-context-profile) | [MTP3; DFlash2 K=7](performance/qwen3.8-27b.md#single-request-speculative-decode) | [MTP3 C=1, 2, 4, 8; DFlash2 C=1](performance/qwen3.8-27b.md#corpus-makespan) | [C=1, 2, 4, 8](performance/qwen3.8-27b.md#decode-saturation) |

Qwen3.8-27B and Qwen3.6-35B-A3B C=1 corpus points also supply their single-request phase tables.
The Qwen3.6-27B NVFP4 MTP3 phase table comes from a corpus C=1 point whose full makespan is
not published here. These Qwen3.8-27B measurements use FP8 E4M3 row-256 KV; the Qwen3.6 measurements
use INT8 group-64 KV. Each model page records its build and run conditions.

GLM has no published performance page here. Product support is listed in the
[README](../README.md#supported-models-and-deployments).

## Reading the results

| Question | Metric to use |
|---|---|
| How fast is prompt processing or an individual decode phase? | Prefill phase, Server TTFT, Decode phase |
| How long does the full fixed request set take? | Corpus makespan, Corpus decode, Requests/s |
| What aggregate decode rate is sustained at a full batch? | Steady decode |
| How much time does a decode execution take at a given batch? | Full-batch step |

These rates use different time boundaries. Current server TTFT follows the production first-token
boundary; older serial reports used a phase sum. Preserve each run's metric definition, as described
in the [methodology](performance/methodology.md#metrics-and-statistics). External streaming TTFT has
its [own benchmark contract](../tools/bench/ttft/README.md). Stochastic runs
can generate different token totals even with the same prompts and seeds. Output-limit and
repetition samples remain labeled in the measured corpus; throughput alone does not establish
successful task completion. Each model page records completion outcomes with its measurements.

## Related references

- [Serving benchmark runners](../tools/bench/README.md#serving-corpus-benchmark): usage and local report files.
- [Engine and Op benchmarks](../bench/README.md): their separate measurement scopes and commands.
- [Capability evaluation](../eval/README.md): evaluation workflow; published scores live in the
  [model cards](../model-cards/), with a [README summary](../README.md#evaluation).
- [Perplexity](perplexity.md): offline causal-scoring measurement and comparison rules.

Model pages are the detailed result authority. README and model-card performance tables are
excerpts linked to those pages; update them together when replacing an applicable measurement.
