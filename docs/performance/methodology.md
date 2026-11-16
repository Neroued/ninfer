# Serving performance methodology

This document defines measurement and publication for the serving campaigns linked from the
[performance index](../performance.md), including RTX 5090 and single-/dual-Spark deployments.
Model pages own measured values and run-specific conditions. Current resource controls are
described in [startup configuration](../startup-configuration.md).

[Performance index](../performance.md) · [Runner usage](../../tools/bench/README.md#serving-corpus-benchmark)

Engine and Op benchmarks have their own contracts in [bench/README.md](../../bench/README.md).
The results here measure the public HTTP serving route.

## Common serving profile

The published runs share the settings below. Each model page records hardware, node count,
parallel strategy, communication backend, toolchain, weights, KV representation, prefill chunk,
context/concurrency settings, and any sampling exceptions.

| Setting | Value |
|---|---|
| Route | Persistent `ninfer-serve`, loopback OpenAI Chat Completions, `stream=false` |
| CUDA Graph | Enabled |
| Prefix reuse | Disabled |
| Stochastic sampling | Temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 1.0, frequency penalty 0 |
| Greedy sampling | Exact argmax; identified explicitly per run |
| Startup and warmup | Outside measured requests/waves |

| Report label | Runner mode | Server options |
|---|---|---|
| MTP0 (no speculation) | `mtp0` | `--spec none` |
| MTP3 | `mtp3` | `--spec mtp --draft-tokens 3 --proposal-head optimized` |
| DFlash, block=8 | `dflash7` | `--spec dflash --draft-tokens 7 --proposal-head optimized` |
| DFlash2, block=8 | `dflash2_7` | `--spec dflash2 --draft-tokens 7 --proposal-head optimized` |

`K` denotes draft tokens; block size is `K+1`. Weight format, backend, draft count, and proposal
head are separate experimental dimensions.

The per-request context limit and physical capacity are recorded separately. Current engines use
per-rank Context byte pools shared by KV and State; `auto` follows each rank's remaining budget.
Older campaigns record dedicated KV capacity in tokens. Preserve the resource meaning of each run.

## Workloads and measurement boundaries

### Single-request phases

The context-length profile uses `long_niah_8k`, `long_niah_64k`, `long_niah_128k`, and
`long_niah_256k`: thinking disabled, 128 output tokens requested, five seeds per fixture.
It measures prefill, server TTFT, and decode for each selected speculative mode. Tables retain
actual prompt lengths; MTP0 and MTP3 results are reported separately.

The speculative corpus uses three reasoning fixtures (`long_decode_aime26_01`,
`long_decode_aime26_15`, `long_decode_aime26_30`) with thinking enabled and a 65,536-token output
budget. Twelve other fixtures cover Code, Story, Translation, and Structured output, with three
fixtures per category, thinking disabled, and a 4,096-token output budget. Each fixture has five
seeds, giving 75 requests per corpus. Actual completion lengths are measurements, not budgets.

Fixture content is in [examples/cli/manifest.json](../../examples/cli/manifest.json); fixture
selection and the five seeds are in [run_serve_corpus.py](../../tools/bench/run_serve_corpus.py).
Preserve their identity when comparing historical runs. Serial runs submit one request at a time
to one persistent server. A corpus-makespan C=1 point can also supply these phase statistics;
model pages identify that shared source.

### Corpus makespan

Each concurrency point starts a fresh server. The runner shuffles the fixed request set once with
seed `20260811` and preserves its ordered HTTP send sequence at every C. Exactly C persistent
workers submit their next request after reading their current response. Makespan starts at worker
release and ends when the last complete HTTP response has been read. It includes prefill, decode,
workload transitions, admission waits, and drain. Decode step latency is grouped by actual batch;
the full-batch column selects steps at the requested C.

Prompts, seeds, and send order are fixed; stochastic continuations and token totals can vary with
weights, backends, and concurrency. Report token totals alongside makespan and rate.

### Decode saturation

At each C, a fresh server releases C requests with distinct fixed seeds and waits for every
response. Model pages specify the fixture, thinking mode, actual prompt length, output budget,
and resource settings. The runner's `jobs.json` retains every seed and request assignment.

Steady throughput includes only complete one-second server intervals with zero computed prefill
tokens, `running=C`, `prefilling=0`, `decode_ready=C`, at least one decode round, and exactly C rows
in every round. Intervals must have no preemptions, Snapshot/Replay restores or replayed tokens,
and no paused or replaying requests at the end. Ramp-up, prefill, recovery, and drain are excluded.
Wave makespan still spans client release through the last complete response. Acceptance is
aggregated over the full wave, not just steady intervals. Each published point is one wave,
not a repeated-sample mean. Full-batch step latency uses all recorded executions at batch C,
independently of the one-second interval selection used for steady throughput.

## Metrics and statistics

| Published metric | Definition |
|---|---|
| Prefill phase (tok/s) | `prompt_tokens / prefill_seconds` for each serial request |
| Server TTFT (ms) | `1000 * request_done.timings_seconds.ttft` |
| Decode phase (tok/s) | `(completion_tokens - 1) / decode_seconds` for each serial request |
| Corpus prefill (tok/s) | Total computed prefill tokens / full corpus makespan |
| Corpus decode (tok/s) | Total decode tokens / full corpus makespan |
| Steady decode (tok/s) | Sum of committed decode tokens / sum of selected interval seconds |
| Full-batch step (ms) | Total Engine decode execution time at actual batch C / step count at C |
| Per-request step (ms) | Mean Engine decode execution time during one serial request |
| Requests/s | Number of requests / full corpus makespan |
| Spec acceptance | Accepted draft tokens / drafted tokens |
| Spec tokens/round | `1 + accepted_tokens / speculative_rounds` |

Server TTFT follows the production first-token boundary and includes service/Engine preparation
and Engine queueing. It is separate from the [Serve TTFT client's](../../tools/bench/ttft/README.md)
network observation, which starts before writing the HTTP request and ends at the first nonempty
output delta. Execution phases can overlap and must not be added to reconstruct either TTFT.
Serial result schema v8 and earlier used a phase sum; those historical values have a different
boundary from the current server TTFT.

Vision timing is the reported GPU stream interval. Serial records preserve its sample count;
`vision_stream_seconds` is `null` when that count is zero, including when GPU timing is disabled.
A positive sample count with a zero duration represents a measured zero. Request Host exposure
includes all six reported phases, including `executor_flush`.
The three decode rates have different time denominators and must not share an unlabeled column.
Committed decode tokens exclude the first completion token produced by prefill.

Single-request tables report arithmetic mean ± sample standard deviation of per-request values.
Category rows pool 3 fixtures × 5 seeds: their deviation includes differences between fixtures.
For greedy repetitions, within-fixture timing variation remains, while deterministic output and
acceptance values may have zero deviation. A single wave or corpus point has no sample deviation.

Acceptance in a phase table is the mean of per-request ratios. Acceptance in a corpus or wave
table is the ratio of summed accepted and drafted tokens. Do not interchange these aggregations.
Calculate new statistics and comparisons from unrounded reports. Display phase rates and their
deviations to one decimal, seconds to two decimals, acceptance to one percentage decimal, and
tokens/round and speedups to two decimals. Keep token totals and sample counts exact. Historical
comparisons derived from rounded published means must identify that limitation.

## Output checks and optional comparisons

Performance runs check execution failures, token accounting, termination reasons, and obvious
repetition. Record the scope and result of those checks once on the model page. Answer accuracy,
prompt compliance, and other content audits are included only when required by the task; their
conclusions apply only to the identified runs.

Retain pathological samples in fixed-corpus statistics and label them next to affected results.
Supplementary statistics excluding samples must identify the exclusions. A natural stop or high
throughput alone does not establish successful task completion.

Comparisons are optional. When included, identify both run IDs, the metric, and differences in
configuration or workload. A throughput speedup is `new / baseline`; a makespan speedup is
`baseline / new`; relative change is `(new / baseline - 1) * 100%`. C=1 must be the matching
suite/profile baseline. Distinguish fixed-request workloads with varying output lengths from
fixed-token measurements. Historical comparisons do not isolate a backend change. MTP0 context
profiles and speculative scenario suites use different workloads and cannot establish per-scenario
baseline/speculative speedups.

## Publishing and updating results

Use one model page ordered by configuration, measured workloads, completion outcomes, and
reproduction. Group or compare deployments, weight formats, and backends within each workload.
Identify each result's configuration, command, and source reports; a page-local run ID can connect
larger matrices. Omit unmeasured sections.

Record the date, model/weights identity, explicit artifact and relevant companion, known tested
revision, hardware/toolchain, workload, sample count, and runtime settings. Link common settings
rather than repeating them; put exceptions and measured per-rank capacities in the run record.
Identify missing historical metadata without inferring values.

Commands run from the repository root with a selected Python 3.11 interpreter. Use a fresh output
directory for reruns; recorded report paths identify the measured evidence. Reproducing a workload
on another build or artifact does not promise identical historical outputs or timings. New raw
reports stay under `.agent-tmp/<campaign>/`; historical report locations remain recorded on their
model pages. Reuse existing JSON/CSV/Markdown summaries to verify values.

Model pages own detailed serving results. README and model-card excerpts link to them and update
with the relevant measurements. Capability scores belong in model-card Evaluation sections, with
workflow guidance in [eval/README.md](../../eval/README.md); runner usage and report-file descriptions
belong in [tools/bench/README.md](../../tools/bench/README.md).

When replacing measurements, update their run records, applicable findings, coverage index, and
excerpts together. Retain older data only for an explicitly requested historical comparison or
audit. Before publishing, check affected links/anchors, values and material caveats, and
`git diff --check`.
