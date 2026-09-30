# KVMem inference regression pipeline

The model runners require the retrieval-window implementation proposed in
[NInfer #345](https://github.com/Neroued/ninfer/pull/345), including its numerical
primitives. The standalone harness unit tests can run without that engine change.

Run from the NInfer checkout on the CUDA 13.1 / sm_120a machine. Python 3.11,
CMake, Ninja, idle test ports (8095 through 8100), and an explicit v3 model are required.
The runner owns only its child process; it fails if another server owns the port.
It does not stop services, retry failed requests, or treat HTTP 200 as successful
generation. An engine error, missing SSE terminator, missing usage, incorrect
recall, or silent retrieval capture fails the run.

```bash
export PYTHON=python3.11
export NINFER_MODEL='/path/to/model.ninfer'
bash tests/e2e/run_kvmem_pipeline.sh smoke
bash tests/e2e/run_kvmem_pipeline.sh regression
bash tests/e2e/run_kvmem_pipeline.sh long
bash tests/e2e/run_kvmem_pipeline.sh concurrency
```

`smoke` builds affected targets, runs numerical/storage/options tests and real
JSON/SSE generation, tool histories with large output budgets, cancellation, and
subsequent inference. `regression` additionally crosses a small KV window twice,
recalls a historical needle, checks that nonzero Q features selected and restored
Host pages, generates at least 512 tokens across page-growth boundaries, and
exercises 2048-token chunks, ordinary decoding, and dense mode.
It also runs a multi-chunk tool-tail replay with `max_tokens=1`: no spare output
budget may hide omitted replay service quanta. The focused `replay-budget` profile
runs that request and verifies subsequent inference; use the default 16K context,
64-page window and a chunk no larger than 2048 for this fixture.
It also constrains Host KV to 32 MiB, requires an over-budget request to fail
with HTTP 400 before execution, and verifies subsequent inference stays healthy.
`long` adds a 96K-device-window / 256K-logical-context test ladder through
250K nominal tokens, twice on the same engine after cache-producing requests.
Actual token counts and inference latency are recorded, not inferred from bytes.

Reports are saved under `out/kvmem-tests/<UTC timestamp>/`: JSON, JUnit XML, engine
logs, GPU memory/utilization samples, hardware, and source metadata. `NINFER_TEST_OUTPUT` and `NINFER_BUILD_DIR`
override output/build directories. A missing GPU is a failure, not a successful
skip. Existing scripts outside this repository are not needed.
`NINFER_TEST_PORT` overrides the first port; each matrix configuration uses the
next port to avoid TIME_WAIT conflicts between successive server processes.

For a focused run against a freshly built binary:

```bash
"$PYTHON" tests/e2e/kvmem_suite.py --model "$NINFER_MODEL" \
  --output /tmp/kvmem-check-unique --profile regression --window 64 --context 16384
```

For two concurrent sparse lanes, run both decode backends:

```bash
port=8095
for spec in none mtp; do
  "$PYTHON" tests/e2e/kvmem_suite.py --model "$NINFER_MODEL" \
    --output "out/kvmem-tests/dual-$spec-$(date -u +%Y%m%dT%H%M%SZ)" \
    --profile concurrency --concurrency 2 --spec "$spec" \
    --window 64 --context 16384 --chunk 1024 --host-mib 2048 --port "$port"
  port=$((port + 1))
done
```

This profile compares greedy output from isolated and simultaneous requests,
requires two real decode lanes in the same GPU batch, and requires both lanes to
restore scored historical pages from Host KV. Repeated pairs exercise lane reuse;
disconnecting one active lane must leave the other generating and permit another
request afterwards. HTTP overlap alone cannot pass this profile. These controlled
output comparisons are regression checks, not broad model quality equivalence.

To serve with this implementation, use `--max-concurrency 2 --kv-capacity auto`
alongside your KVMem options. The window remains **per lane**: auto capacity is
`lanes * min(ceil(context/64), window_pages + ceil(min(chunk,context)/64) + 16)`
Main KV pages, plus the existing per-lane MTP lead when enabled. Model weights are
shared, while recurrent state and retrieval capture storage grow with concurrency.
`--host-kv-mib` is the shared Host budget; requests can queue if their combined
reservations do not fit. Two lanes do not imply twice the tokens per second.

Numerical oracle tests qualify attention mathematics. The model tests establish
the listed inference behaviors, not LongMemEval quality parity or universal
long-context accuracy. A report is only evidence for its recorded configuration.
Recall requires the correct final answer line, not a substring appearing anywhere.
`exact_answer_format` separately records whether the model omitted all extra prose;
format compliance is reported rather than used as a storage-stability gate.

For self-hosted GPU CI, supply an idle Linux x64 worker, an explicit model path
and the Python 3.11 interpreter, and invoke the same local runner shown above.
Serialize jobs using the GPU and preserve reports after both success and failure.
This test tooling does not install or start a remote workflow; configuring remote
execution is separate from a passing local run.

## Diagnostic quality comparisons

`kvmem_quality.py` uses frozen, seeded fixtures for needles at 10/50/90% history
depth, long tool tails, two-hop questions, updated facts, and unanswerable questions.
It grades the entire answer as a single exact JSON object; prose containing the
correct substring and duplicate JSON keys fail. Each report retains complete
requests, raw responses, gold answers, token usage, engine logs, and GPU samples.

```bash
"$PYTHON" tests/e2e/kvmem_quality.py fixtures --output out/quality/fixtures.json
"$PYTHON" tests/e2e/kvmem_quality.py run --fixtures out/quality/fixtures.json \
  --model "$NINFER_MODEL" --window 0 --port 8105 --output out/quality/dense
"$PYTHON" tests/e2e/kvmem_quality.py run --fixtures out/quality/fixtures.json \
  --model "$NINFER_MODEL" --window 64 --port 8106 --output out/quality/sparse
"$PYTHON" tests/e2e/kvmem_quality.py compare \
  out/quality/dense/quality.json out/quality/sparse/quality.json
```

The reference server can run the same fixtures with `--engine kvmem --binary
/path/llama-kvmem-server --model /path/model.gguf`. Sparse reference runs require
explicit `--reference-budget` and `--reference-reserve` token counts. `--window 0`
selects its dense route. NInfer's current rolling window and the reference's separate
selection/generation budgets differ; record them explicitly when interpreting results.
NInfer INT8 and reference q8_0 KV, and their model weight formats, are not identical.

The comparison rejects incomplete runs or different fixture hashes. It reports
paired regressions/improvements and always labels this small synthetic suite
`diagnostic-only`; it cannot certify KVMem quality equivalence. Representative
long-context datasets and matched reference controls are still required.

## LongMemEval-S full-history predictions

`longmemeval-source.json` pins the official cleaned 500-question source by revision,
size and SHA256, plus the official evaluator revision. Download the `url` in that
lock file to a local data path. Preparation refuses different bytes. The conversational
prompt adaptation preserves every dated session and both user/assistant turns,
removes annotation fields, and places the actual question in the final user message.
It does not truncate or select evidence sessions. This differs from the paper's
generation prompt and must be reported as such.

```bash
"$PYTHON" tests/e2e/kvmem_longmemeval.py prepare --data /path/longmemeval_s_cleaned.json \
  --output out/lme/full-fixtures.json
# Optional pipeline diagnostic: add --per-stratum 1 to freeze ten questions,
# selected by ID hash across task type and abstention, without looking at answers.
"$PYTHON" tests/e2e/kvmem_longmemeval.py run --fixtures out/lme/full-fixtures.json \
  --model "$NINFER_MODEL" --context 262144 --window 512 --host-mib 12288 \
  --port 8120 --output out/lme/ninfer-sparse-32k
```

Reference invocation uses the same `--engine kvmem`, `--binary`, `--reference-budget`
and `--reference-reserve` arguments as the diagnostic runner. Run dense and sparse
configurations serially on one GPU. Context overflow, a length-truncated answer, or
a request error stays in the report and fails the run; there is no silent shortening
or automatic retry. `fixtures.json` preserves the complete prompts;
`predictions.json` preserves responses, usage, timing, gold and request hashes.
`hypotheses.jsonl` is compatible with the official evaluator's input format.

Prediction success does not assign correctness: reports remain `scored: false`.
The official evaluator uses task-specific semantic rubrics. Preserve those scores
separately from stricter review of contradictions and unsupported assertions in the
whole answer; do not replace semantic scoring with substring matching. An incomplete
prediction set or a ten-question pilot cannot establish benchmark equivalence.

`kvmem_quality_stats.py` reports conservative paired accuracy-difference intervals
from exact binomial bounds on discordant pairs and a Bonferroni correction.
The question-only interval assumes independent questions. LongMemEval comparisons
require a separate, source-bound cluster map because original/abstention variants
and reused evidence are dependent candidates:

```bash
"$PYTHON" tests/e2e/kvmem_longmemeval.py clusters --data /path/longmemeval_s_cleaned.json \
  --output out/lme/clusters.json
```

This grouping links question-ID stems and shared answer-session IDs transitively,
without looking at model answers. The frozen source has 466 groups: 432 singletons
and 34 pairs. Group membership does not modify inference fixtures. The interval
allows arbitrary dependence within each group and assumes independent groups with
a common distribution within each group-size stratum. It retains the original
question-weighted accuracy: for a size-s group's net correct-answer change D,
E[D] = sum(P(D >= j)) - sum(P(D <= -j)), j=1..s. Exact binomial bounds on those
tail probabilities, with Bonferroni across strata and tails, yield the reported
conservative interval. Unidentified dependence remains a limitation; the grouping
is a safeguard, not proof of independence. Small strata can make the interval wide.
Even zero disagreements has nonzero uncertainty, and 500 questions do not guarantee
enough power to certify a 1pp margin. Statistics do not establish
comparability of model weights, budgets or grading, and never auto-certify parity.

The comparison reports a paired **95%** interval and an interval-only indicator
against a **1 percentage point** margin. That diagnostic threshold is not an
upstream acceptance policy or proof of lossless output. Select representative
workloads before comparing a change, repeat paired A/B runs, and include repeated
A/A controls to characterize run-to-run variability. Inspect task-family
regressions as well as aggregate results. These repeated-run controls must be
planned and analyzed separately; the interval tool does not perform them.
`equivalence_established` stays false until dataset, execution and comparability
requirements have also been qualified. Small pilot intervals are insufficient.

Whole-answer reviews can be attached without altering the raw predictions:

```bash
"$PYTHON" tests/e2e/kvmem_semantic_review.py --predictions out/lme/run/predictions.json \
  --output out/lme/run/reviews.json
# A reviewer now fills boolean rubric/strict labels and rationales, and identifies
# the actual reviewer and rubric. Null/unknown labels stay unresolved.
"$PYTHON" tests/e2e/kvmem_semantic_review.py --predictions out/lme/run/predictions.json \
  --reviews out/lme/run/reviews.json --output out/lme/run/reviewed.json
```

The importer rejects stale hashes, edited answers, missing IDs, nonboolean or
unresolved labels, and missing rationales. Manual pilot review is labeled as such,
not as the official GPT-4o score. Compare complete `reviewed.json` files with
`kvmem_quality.py compare --clusters out/lme/clusters.json`; all failures remain
in the denominator. Missing/stale clustering is rejected for LongMemEval.
Source adjudication retains the original `judge_report_sha256`, `judge_settings`,
literal evidence and any citation-source corrections, and adds identified
`adjudication` metadata describing the actual reviewer and protocol. Comparison
requires the same adjudication protocol on both sides. The imported report binds
the complete review using SHA256 of its canonical `encode()` JSON; raw predictions
and API reports stay unchanged. An unresolved source/gold conflict remains null,
so it cannot silently enter a completed accuracy report.

## Qwen semantic judge

The user's selected provider is the existing Qwen API. Run the judge where its
credential file already lives; the key is read locally and is never written into
reports. Use the same explicit API/model, rubric and settings for every comparison:

```bash
"$PYTHON" tests/e2e/kvmem_judge_evidence.py --data /path/longmemeval_s_cleaned.json \
  --output out/lme/grading-evidence.json
"$PYTHON" tests/e2e/kvmem_qwen_judge.py --predictions out/lme/run/predictions.json \
  --evidence out/lme/grading-evidence.json \
  --api-url "$QWEN_API_URL" --protocol anthropic \
  --api-key-file "$QWEN_KEY_FILE" --model "$QWEN_JUDGE_MODEL" \
  --output out/lme/run/qwen-review
```

Before scoring a benchmark, run the fixed calibration with the same API arguments,
replacing `--predictions ...` with
`--calibration tests/e2e/qwen-judge-calibration.json` and using a fresh output folder.
Its nine predefined base/strict label pairs are never sent to the API. A mismatch
returns nonzero and is retained in `calibration.json`; passing this small check
does not prove general grading accuracy.

This task-specific rubric adaptation returns separate base/strict labels with
reasons. The judge sees the whole answer and dated source evidence, but no engine
identity. Evidence contains all complete sessions identified by the frozen source's
answer-session annotations, including repeated IDs in history order. It is selected
independently of model answers and used only for grading, never inference. The file
hash is bound to both engines' judge settings. These sessions are not exhaustive
history: omission alone does not prove a historical claim false; unresolved material
claims require review against the full source. `question_date` anchors relative
intervals but does not discard provided records, consistent with the full-history
generation task. Record dates and event dates must be distinguished.
Raw requests, responses, timestamps, returned model and hashes are preserved.
The protocol is explicit and bound to resume/comparison settings. Anthropic bases
append `/v1/messages` and use top-level system/thinking fields; OpenAI bases append
`/chat/completions`. Negative verdicts must include literal candidate and comparison
quotes, checked against the actual input. This detects fabricated quotations, but
does not prove the judge's semantic interpretation; retain calibration and review
evidence before using scores for acceptance.
Malformed JSON, duplicate keys, unresolved labels, invalid quotes, truncated responses
and HTTP errors remain failed attempts; later cases still run. `reviews.json` retains
null labels for unresolved cases, and no completed `reviewed.json` is produced until
every case is valid. `--resume` refuses changed predictions/settings/rubric and retains
semantic/schema-invalid responses without calling the API again. Transport failures
can be retried explicitly; do not resample judgments until a desired label appears.
The audited citation-source policy may correct only a wrong source-field name when
the unchanged quote occurs verbatim in exactly one other supplied field. It never
changes quotes or labels. `--revalidate-from /path/old-report` permits that parser-only
revalidation of identical raw requests/responses; it refuses altered inputs/rubrics
and is mutually exclusive with `--resume`. Original reports remain unchanged.
Model aliases do not guarantee immutable server weights: record this limitation
when no dated snapshot is available. These are Qwen scores, not official GPT-4o scores.
Paired semantic comparisons reject different judges/rubrics/settings, incomplete
execution, missing labels and inconsistent totals.

## SWE mini agentic supplement

`kvmem_swe.py` uses the existing NAS EvalScope 1.12.0 / swebench 4.1.0 installation.
Its source lock binds every byte and ordered instance ID of a local JSONL snapshot.
The current NAS mini set contains 50 instances: 25 Django and 25 Sphinx. The frozen
cache has no verified immutable upstream revision; its content hash is the authority.
This is an agentic executable-test supplement, not a replacement for the 500-question
long-memory acceptance set. Fifty questions cannot by themselves establish a 1pp margin.

```bash
"$PYTHON" tests/e2e/kvmem_swe.py --data /path/swe-mini-frozen.jsonl \
  --source-lock /path/swe-mini-source.json --api-url http://dev-host:8127/v1 \
  --output /path/new-run --prepare-only
# Add --pilot-per-repo 1 for a deterministic two-instance preparation.
# To execute, use another fresh output directory and omit --prepare-only.
```

Preparation validates the actual EvalScope local loader, source hashes, IDs, package
versions and Docker access. A manifest records package-source hashes, image identities,
generation configuration and selected instances. Execution uses one request at a time,
greedy/no-thinking generation, no cached answers, no automatic HTTP retries, and at
most 250 tool-loop steps per instance. Model input follows the agentic adapter's issue
description and tool protocol; oracle source text is not supplied. Keep both engines'
complete traces, patches and executable test results. `evalscope_finished` means only
that EvalScope returned: verify all instance outcomes and environment failures before
computing paired resolved rates. Neither preparation nor process completion certifies parity.

The development endpoint must use an independent port outside 8080/8081. For a
WSL server bound to localhost, run a process-scoped bridge on its Windows host:

```powershell
node tests/e2e/kvmem_nas_bridge.cjs WINDOWS_LAN_IP 8127 WSL_SERVER_PORT NAS_IP
```

Only the selected NAS IP can connect; the bridge does not alter firewall rules
or system forwarding, and rejects production ports 8080/8081. Stop it after the
benchmark. NAS-to-Windows-to-WSL `/health` returned 200 with the reference server
on 8128. Use NO_PROXY for the development host when the NAS has an HTTP proxy.
This is a connectivity check, not an agentic SWE result.

For network-dependent SWE tests, `--container-proxy http://NAS_LAN_IP:7897`
passes an existing reachable proxy to agent sandboxes and the separate grading
containers. It records the setting, preserves local HTTP/HTTPS test servers through
NO_PROXY, and scopes the grading SDK override to this process and `swebench/sweb.eval.*`
images. It changes no host proxy, image, test or scoring rule. Proxy URLs with
credentials are rejected because manifests are retained. Validate gold patches
before attributing a network-dependent failure to either model. On the NAS pilot,
Sphinx's gold patch initially failed three PASS_TO_PASS external-link checks;
with the existing NAS proxy, both fixed pilot instances passed their full gold checks.

`kvmem_suite.py --profile replay-cancel --context 196608 --window 64` tests cancellation
after retrieval with a long tool suffix. It requires the next request to finish within
five seconds. It waits until replay actually begins, then disconnects; this check
is also part of `long`. A 135258-token tool-tail replay originally blocked the
next request for 37.27 seconds. Bounded replay steps restored cancellation; retain
the run reports for backend-specific timing and the final binary's qualification.

Each server run writes `host-memory.csv` with process RSS, anonymous/file memory,
swap, and Linux VM headroom every two seconds. To avoid repeating a WSL-wide OOM,
the runner terminates only its owned server if MemAvailable + SwapFree falls below
3 GiB. It preserves `memory-abort.json` and fails the run; this is not a passing
capacity check. Windows system Commit is a separate host limit and still needs
host-side monitoring. Recorded execution overrides are restricted to the CUDA
Graph and allocator diagnostic flags; credentials are never recorded.

## Two-lane window comparison

`kvmem_window_bench.py` compares 36 Ki-token (`576` pages), 50 Ki-token (`800`
pages) and 72 Ki-token (`1152` pages) selection windows, each with two lanes and
262144-token logical contexts. Run the configurations sequentially with the same
binary and model:

```bash
"$PYTHON" tests/e2e/kvmem_window_bench.py --model /path/model.ninfer \
  --window 576 --port 8098 --output out/window-36k --active-file out/window-active.json
"$PYTHON" tests/e2e/kvmem_window_bench.py --model /path/model.ninfer \
  --window 800 --port 8098 --output out/window-50k --active-file out/window-active.json
"$PYTHON" tests/e2e/kvmem_window_bench.py --model /path/model.ninfer \
  --window 1152 --port 8099 --output out/window-72k --active-file out/window-active.json
```

Each run owns its server and requires a fresh output directory. Defaults use INT8
KV, MTP3, a 1024-token prefill chunk and an 18 GiB shared Host KV arena. The GPU
pool additionally reserves per-lane chunk/slack and MTP lead capacity. Cold paired
inputs target 128K and 256K tokens; reports retain actual tokenizer counts, fixture
hashes, two 2048-token output streams, engine timings and verified two-lane decode
rounds. This repeated-filler/counting workload is a capacity and performance
diagnostic, not a representative quality or general generation-speed benchmark.

For an end-to-end throughput control, add `--concurrency 1` in a fresh output
directory. Both clients still submit together; the one-lane server must queue
them and execute two separate sparse retrievals with exclusively one-lane decode.
Compare the complete pair wall time and total output tokens against concurrency
two with identical prompts, window and output length. Per-request active decode
rates exclude scheduler waiting and must not be summed as end-to-end throughput.

### KVMem + vision regression

Use a v3 artifact containing Vision weights. The `vision` pipeline profile runs both
ordinary decoding and MTP with two lanes, including real sparse retrieval/replay:

```bash
NINFER_MODEL=/path/model.ninfer PYTHON=/path/python3.11 \
  bash tests/e2e/run_kvmem_pipeline.sh vision

"$PYTHON" tests/e2e/kvmem_vision_suite.py --model /path/model.ninfer \
  --window 576 --context 262144 --host-mib 18432 --spec mtp \
  --port 8099 --output out/vision-36k-c2
```

The runner checks exact image colors, multi-image order, cached-media text suffixes,
query clipping inside a media item, tool tails longer than the window, cancellation
at an observed media replay boundary, and oversized-group admission rejection.
Two visual prompts generate 1,024 tokens independently and concurrently; the runner
requires actual two-lane decode/retrieval and matching greedy outputs for these fixtures.
A still image through the native video route checks temporal position plumbing only;
it does not qualify motion understanding or all video codecs. `--window 0` provides
a dense control for the small-window fixtures. Reports save raw completed responses
before assertions, server diagnostics, GPU samples, Host memory and JUnit results.
Each run needs a fresh output directory and a free port; use different ports for
successive processes if accepted sockets remain in TIME_WAIT.

Default tests use a 4K window/16K context/2GiB Host budget. The 36K configuration
tests prompts just beyond its window, not two fully populated 256K visual contexts.
Sparse selection can change answers; these controlled cases do not establish general
quality equivalence to dense attention or reference KVMem. On Windows/WSL, monitor
system commit independently; `active.json` in the output parent identifies only the
owned test process for an external memory guard.

All window benchmark runs enable the same diagnostic logging. `NINFER_KVMEM_TRANSFER_TRACE` emits
actual KV placement payload bytes and existing copy submission/wait times, split
by prefill, retrieval, replay and decode. Demoted pages with current Host replicas
need no copy and are counted separately. The timings are CPU elapsed measurements,
not GPU-only PCIe measurements; total placement time includes bookkeeping and
table publication. Logging adds overhead. Windows Commit must be sampled outside
WSL; Linux swap and application KV transfers do not establish Windows pagefile I/O.


### DFlash2 seven-draft + Vision

The artifact must contain matching `text`, `vision`, and `dflash2` components.
An MTP-only artifact is insufficient. Run the dedicated pipeline (also available
as the manual workflow profile `dflash2-vision`):

```bash
NINFER_MODEL=/absolute/qwen3_8_27b_nvfp4.ninfer \
  bash tests/e2e/run_kvmem_pipeline.sh dflash2-vision
```

It runs dense DFlash2, sparse ordinary, and sparse DFlash2 on the same artifact.
The DFlash2 runner requires real telemetry with `draft_window=7`, nonzero drafted
and accepted counts, and zero/partial/full acceptance coverage in sparse mode.
It also covers a 2304-token decode that wraps the 2048-token draft ring, true
batched text/visual requests, long media query replay, cancellation and lane reuse.
Cross-backend output quality still requires inspecting paired responses; passing
these controlled fixtures is not general quality equivalence.

To exercise two 36K windows with a 256K logical limit per lane:

```bash
python3.11 tests/e2e/kvmem_vision_suite.py \
  --model /absolute/qwen3_8_27b_nvfp4.ninfer --spec dflash2 \
  --window 576 --context 262144 --host-mib 18432 \
  --output out/kvmem-dflash2-36k --port 8099
```

This profile actually prefills beyond the 36K window. It does not fill both 256K
contexts. Check host commit and GPU memory externally on WSL; the runner's Linux
headroom guard does not measure Windows commit.
