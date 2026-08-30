# Serve TTFT benchmark

The measurement runner is a black-box client for an already-running `ninfer-serve`. It constructs a
fixed request graph, sends only public HTTP requests, and measures time to first token (TTFT). It
never starts or configures Serve, calls `Engine`, reads request logs, or labels an observed request
as a cache hit, transfer, or eviction. A separate campaign controller owns the repetitive process
lifecycle around that runner.

The standard campaign has one resident `qwen3.8-27b/nvfp4` artifact, FP8 KV, one GPU, and one to
eight active requests. Artifact, KV dtype, speculative backend, and model are not benchmark axes.
The runner discovers the public model ID through `/v1/models`. The campaign controller applies the
fixed Serve profile and gives every measured sample a fresh process.

## Measurement contract

Request JSON and data URLs are prepared before timing. Every request owns a preconnected HTTP/1.1
connection:

```text
t0 = immediately before the first request bytes are written
tb = request body write completed
ta = first protocol metadata emitted after Serve accepted the request
t1 = first non-empty model text, reasoning, or tool-argument delta
TTFT = t1 - t0
```

Connection establishment and JSON serialization are outside TTFT. Body upload, Serve request
preparation, media acquisition/preprocessing, admission, prefill, and the first model output are
inside it. HTTP headers, assistant-role chunks, `response.created`, `message_start`, usage, empty
deltas, and finish metadata are not model output.

A rejected request, timeout, cancellation, protocol failure, or disconnect before `t1` has no
TTFT. A cancellation case is constructed only when the client transport terminates without a
protocol terminal event within five seconds of the cancellation action. Rejection status/code is
reported separately. A clean request records only its first
accepted, first-output, and terminal events; later output deltas are consumed but do not bloat the
artifact.

`constructed=true` means the requested arrival graph and all required external event inequalities
actually occurred. It does not mean the expected cache or scheduling optimization worked. A
constructed case with unexpectedly high TTFT is a valid regression signal. The client deliberately
does not use internal Serve diagnostics to reinterpret it.

## Layers

```text
bench/fixtures/ttft       frozen text, media, hashes, and token facts
tools/streaming_http      byte transport and incremental SSE timestamps
tools/ninfer_serve        public Chat, Responses, and Anthropic adapters
tools/bench/ttft          request graphs, event coordination, and reports
run_serve_ttft_campaign   profiles, fresh Serve lifecycle, artifacts, summary
```

Protocol is selected by the case, not by a CLI matrix: Chat covers ordinary scheduling and media,
Responses covers `previous_response_id` sessions, and Anthropic Messages covers explicit
`cache_control` boundaries.

## Resource formulas

For a request with expanded prompt tokens `p` and output limit `o`, the Main KV reservation used by
the cases is:

```text
g = o - 1
e = 64 * ceil((p + g) / 64)
```

The first generated token is produced at the prompt frontier, so only the remaining `o-1` tokens
grow KV. Fixed pressure shapes are:

| Shape | `p` | `o` | `e` |
|---|---:|---:|---:|
| short | 30 | 32 | 64 |
| long source | 7680 | 16 | 7744 |
| independent long prefill | 7680 | 32 | 7744 |
| interferer | 127 | 256 | 384 |
| decode holder | 127 | 4096 | 4224 |
| unsafe borrower | 30 | 3000 | 3072 |
| exact context | 8129 | 64 | 8192 |
| 256K input | 260096 | 32 | 260160 |

The resource-pressure graph is identical for all six pressure profiles:

```text
A: stored Responses long source completes
B and C: two disposable interferers start together and both reach first output
A2: previous_response_id=A resumes after B and C complete

e(A) + e(B)     = 8128 <= 8192
e(A) + e(B) + e(C) = 8512 > 8192
```

Changing the profile changes the legal placement of A; it does not change the requests.

## Serve profiles

`tools/bench/ttft/profiles.py` is the executable profile catalog used by the campaign controller.
The controller starts a fresh Serve process for every sample. The low-level runner validates only
the profile label and cannot inspect whether a manually managed process really used these
arguments. Serve completes its internal warmup before listening; that operational request has
context-cache participation disabled regardless of profile, so every sample begins with an empty
logical context cache. Before the first process, the controller stages the immutable standard
artifact once in `/dev/shm/ninfer-artifacts/` with a source-identity-qualified name. Every Serve
process then reads that tmpfs path; a later campaign reuses it while the source device, inode, size,
and mtime remain unchanged. There is no disk fallback or user-facing cache option. The standard
artifact and common arguments are:

```bash
SERVE=build/apps/ninfer-serve
NINFER_WEIGHTS=out/qwen3_8_27b_nvfp4.ninfer
COMMON=(--host 127.0.0.1 --port 18080 --kv-dtype fp8 \
  --no-thinking --greedy --log-stats-interval-ms 0)

"$SERVE" "$NINFER_WEIGHTS" "${COMMON[@]}" <profile arguments>
```

All capacities are tokens except values explicitly suffixed `-mib`. `device-state-slots` is extra
checkpoint capacity beyond active lanes.

| Profile label | Profile arguments |
|---|---|
| `text-cold-8k` | `--max-context 8192 --kv-capacity 8192 --max-concurrency 1 --no-prefix-reuse` |
| `text-cold-64k` | `--max-context 65536 --kv-capacity 65536 --max-concurrency 1 --no-prefix-reuse` |
| `text-cold-256k` | `--max-context 262144 --kv-capacity 262144 --max-concurrency 1 --no-prefix-reuse` |
| `cache-hot` | `--max-context 8192 --kv-capacity 8192 --max-concurrency 1 --device-state-slots 2 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 2 --max-shared-prefixes 0 --max-long-anchors-per-continuation 0` |
| `cache-pressure-device` | `--max-context 8192 --kv-capacity 16384 --max-concurrency 2 --device-state-slots 2 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 4 --max-shared-prefixes 0 --max-long-anchors-per-continuation 0` |
| `cache-pressure-state-host` | `--max-context 8192 --kv-capacity 16384 --max-concurrency 2 --device-state-slots 0 --host-state-slots 4 --host-kv-mib 0 --max-private-continuations 4 --max-shared-prefixes 0 --max-long-anchors-per-continuation 0` |
| `cache-pressure-kv-host` | `--max-context 8192 --kv-capacity 8192 --max-concurrency 2 --device-state-slots 2 --host-state-slots 0 --host-kv-mib 8192 --max-private-continuations 4 --max-shared-prefixes 0 --max-long-anchors-per-continuation 0` |
| `cache-swap-64k-host` | `--max-context 65536 --kv-capacity 65536 --max-concurrency 2 --device-state-slots 4 --host-state-slots 0 --host-kv-mib 4608 --max-private-continuations 4 --max-shared-prefixes 0 --max-long-anchors-per-continuation 0` |
| `cache-pressure-both-host` | `--max-context 8192 --kv-capacity 8192 --max-concurrency 2 --device-state-slots 0 --host-state-slots 4 --host-kv-mib 8192 --max-private-continuations 4 --max-shared-prefixes 0 --max-long-anchors-per-continuation 0` |
| `cache-pressure-evict` | `--max-context 8192 --kv-capacity 8192 --max-concurrency 2 --device-state-slots 1 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 4 --max-shared-prefixes 0 --max-long-anchors-per-continuation 0` |
| `cache-pressure-catalog` | `--max-context 8192 --kv-capacity 16384 --max-concurrency 2 --device-state-slots 2 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 2 --max-shared-prefixes 0 --max-long-anchors-per-continuation 0` |
| `cache-off` | `--max-context 8192 --kv-capacity 8192 --max-concurrency 1 --no-prefix-reuse` |
| `shared-prefix` | `--max-context 8192 --kv-capacity 16384 --max-concurrency 2 --device-state-slots 2 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 2 --max-shared-prefixes 1 --max-long-anchors-per-continuation 0` |
| `shared-value` | `--max-context 16384 --kv-capacity 16384 --max-concurrency 1 --device-state-slots 3 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 1 --max-shared-prefixes 1 --max-long-anchors-per-continuation 0` |
| `session-order` | `--max-context 8192 --kv-capacity 16384 --max-concurrency 2 --device-state-slots 8 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 4 --max-shared-prefixes 0 --max-long-anchors-per-continuation 0` |
| `scheduler-overlap` | `--max-context 8192 --kv-capacity 16384 --max-concurrency 2 --prefill-chunk 1024 --no-prefix-reuse` |
| `scheduler-prefill-128` | `--max-context 8192 --kv-capacity 16384 --max-concurrency 2 --prefill-chunk 128 --no-prefix-reuse` |
| `scheduler-prefill-4096` | `--max-context 8192 --kv-capacity 16384 --max-concurrency 2 --prefill-chunk 4096 --no-prefix-reuse` |
| `scheduler-backfill` | `--max-context 7744 --kv-capacity 7808 --max-concurrency 2 --pending-timeout-ms 120000 --no-prefix-reuse` |
| `lane-limit-8` | `--max-context 4224 --kv-capacity 33792 --max-concurrency 8 --max-pending-requests 1 --pending-timeout-ms 120000 --no-prefix-reuse` |
| `pending-timeout` | `--max-context 4224 --kv-capacity 4224 --max-concurrency 1 --max-pending-requests 1 --pending-timeout-ms 100 --no-prefix-reuse` |
| `context-boundary` | `--max-context 8192 --kv-capacity 8192 --max-concurrency 1 --no-prefix-reuse` |
| `vision-cache` | `--max-context 32768 --kv-capacity 32768 --max-concurrency 1 --device-state-slots 2 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 2 --max-shared-prefixes 0 --max-long-anchors-per-continuation 0 --vision --media-cache-mib 512 --media-live-mib 512` |
| `vision-thread-1` | `--max-context 32768 --kv-capacity 32768 --max-concurrency 1 --device-state-slots 2 --host-state-slots 0 --host-kv-mib 0 --max-private-continuations 2 --max-shared-prefixes 0 --max-long-anchors-per-continuation 0 --vision --media-cache-mib 512 --media-live-mib 512 --media-preprocess-threads 1` |
| `vision-concurrent` | `--max-context 32768 --kv-capacity 65536 --max-concurrency 2 --vision --media-cache-mib 512 --media-live-mib 1024 --no-prefix-reuse` |
| `media-cache-tight` | `--max-context 8192 --kv-capacity 8192 --max-concurrency 1 --vision --media-cache-mib 16 --media-live-mib 128 --no-prefix-reuse` |
| `vision-boundary` | `--max-context 65536 --kv-capacity 65536 --max-concurrency 1 --vision --no-prefix-reuse` |
| `mixed-four` | `--max-context 8192 --kv-capacity 32768 --max-concurrency 4 --vision` |

## Audited cases

Baseline and cache cases:

| Case | Profile | Request graph and covered behavior |
|---|---|---|
| `cold-short` | `text-cold-8k` | One 30-token prompt; ordinary short baseline. |
| `cold-long-8k` | `text-cold-8k` | One 7680-token prefill. |
| `cold-long-64k` | `text-cold-64k` | One 64512-token prefill. |
| `cold-long-256k` | `text-cold-256k` | One 260096-token, hardware-resident extreme input. |
| `mixed-four-ordered` | `mixed-four` | Continuation, independent cold long, short, and image requests are submitted one by one after the preceding request is observably accepted by Serve. The two long prompts diverge at the first system-content token. |
| `mixed-four-concurrent` | `mixed-four` | The same four heterogeneous requests are released through one barrier; no frontend or Engine submission order is assumed. |
| `anonymous-hot-continuation` | `cache-hot` | Chat source then exact full-history continuation; private typed rewrite. |
| `session-hot-continuation` | `cache-hot` | Stored Responses source then `previous_response_id` continuation. |
| `session-alternating` | `cache-pressure-device` | `A1, B1, A2, B2` across two stored Responses lineages. |
| `session-alternating-64k-host-swap` | `cache-swap-64k-host` | Two early-divergent 64512-token sessions run as `A1, B1, A2, B2`; one fits Device, the pair requires two Host KV covers for bidirectional rotation. |
| `unmarked-common-prefix-miss` | `cache-hot` | Two standalone user messages share over 4096 tokens but no legal marker. |
| `resume-after-interference-device` | `cache-pressure-device` | Fixed A/B/C/A2 graph with source placement available on Device. |
| `resume-after-interference-state-host` | `cache-pressure-state-host` | Same graph with checkpoint State available only on Host. |
| `resume-after-interference-kv-host` | `cache-pressure-kv-host` | Same graph with KV pressure and Host KV available. |
| `resume-after-interference-both-host` | `cache-pressure-both-host` | Same graph with State and KV Host capacity. |
| `resume-after-interference-evicted` | `cache-pressure-evict` | Source can publish initially, then pressure has no Host fallback. |
| `resume-after-interference-catalog` | `cache-pressure-catalog` | Logical catalog fills while physical capacity remains. |
| `continuation-cache-off` | `cache-off` | Stored Responses control with Engine prefix reuse disabled. |
| `shared-sequential` | `shared-prefix` | Anthropic-marked system prefix followed by a second suffix. |
| `shared-openai-explicit` | `shared-prefix` | OpenAI explicit system breakpoint followed by a different suffix. |
| `shared-openai-implicit` | `shared-value` | OpenAI's default full-prompt candidate, private displacement, then the identical prompt. |
| `shared-observed-promotion` | `shared-value` | Two independent unmarked Anthropic requests establish demand; a filler removes the private endpoint before a third identical request. |
| `shared-fanout` | `shared-prefix` | Seed then two simultaneous branches from a 4149-token, non-page-aligned frontier. |
| `shared-slot-retention` | `shared-value` | Equal-size marked A and B compete for `S=1` while physical capacity can retain A; a filler removes A's private fallback before A returns. |
| `shared-value-replacement` | `shared-value` | A 4149-token system owner and a 5074-token tool candidate compete for `S=1`; a filler removes B's private fallback before B returns. |
| `shared-tools-sequential` | `shared-prefix` | Two requests with the same marked 32-tool prefix. |
| `shared-tools-changed` | `shared-prefix` | The first tool identity changes, invalidating the marked tool prefix. |

Scheduling and boundary cases:

| Case | Profile | Required external relation or boundary |
|---|---|---|
| `short-during-prefill-128` | `scheduler-prefill-128` | `long.accepted < short.sent < long.first`. |
| `short-during-prefill-1024` | `scheduler-overlap` | Same arrival graph with the standard prefill chunk. |
| `short-during-prefill-4096` | `scheduler-prefill-4096` | Same graph with the largest covered chunk. |
| `short-during-decode` | `scheduler-overlap` | `holder.first < short.sent < holder.completed`. |
| `protected-head-backfill` | `scheduler-backfill` | Safe short borrower passes a resource-blocked FIFO head. |
| `protected-head-no-backfill` | `scheduler-backfill` | 3072-token borrower stays behind the head because it would keep blocking it. |
| `active-lanes-full-8` | `lane-limit-8` | Eight holders reach first output before a ninth probe; covers the product lane ceiling. |
| `session-publication-order` | `session-order` | Older child starts first but finishes after newer child; later continuation follows newer. |
| `cancel-before-first` | `scheduler-overlap` | Cancel after acceptance and require transport termination without first output or terminal event, then run a probe. |
| `cancel-after-first` | `scheduler-overlap` | Cancel an active decode after first output, require transport termination without terminal event, then run a probe. |
| `pending-overflow` | `lane-limit-8` | Eight active + one pending; next request must be 429 `server_overloaded`. |
| `pending-timeout` | `pending-timeout` | Blocked non-streaming request must be 503 `request_queue_timeout`. |
| `context-exact` | `context-boundary` | `p + o - 1 = 8192` succeeds. |
| `context-over` | `context-boundary` | 8193-token prompt is 400 `context_length_exceeded`. |

Media cases:

| Case | Profile | Request graph and covered behavior |
|---|---|---|
| `media-cold-image` | `vision-cache` | One image through acquisition, preprocess, Vision, and text prefill. |
| `media-cold-image-video` | `vision-cache` | Interleaved image and video input. |
| `media-prefix-continuation` | `vision-cache` | Same media and full history followed by text. |
| `media-prefix-append` | `vision-cache` | Reused media history with one new suffix image. |
| `media-prefix-changed` | `vision-cache` | A byte-distinct replacement changes an earlier media identity. |
| `media-preprocess-warm` | `media-cache-tight` | Identical standalone image twice with context reuse disabled. |
| `media-cache-thrash` | `media-cache-tight` | `A, B, C, A` against a 16 MiB preprocess cache. |
| `many-image-28` | `vision-cache` | Legal 28-image request with 28752 expanded prompt tokens. |
| `many-image-28-thread-1` | `vision-thread-1` | Same request with one host preprocessing worker. |
| `text-during-media-prepare` | `vision-concurrent` | Media body completes, then short text arrives before media acceptance and must finish first. |
| `media-during-text-decode` | `vision-concurrent` | Heavy media request arrives while a text holder decodes. |
| `two-heavy-media-arrivals` | `vision-concurrent` | Two byte-distinct 28-image requests start through one barrier. |
| `vision-disabled` | `text-cold-8k` | Image request is 400 `vision_disabled`. |
| `vision-envelope-over` | `vision-boundary` | 33 images exceed the 32768 raw-patch envelope and return `media_budget_exceeded`. |

The protected-head profile uses:

```text
e(holder)=4224, e(head)=7744, e(short)=64, e(unsafe)=3072, K=7808
holder+head > K
holder+short <= K and head+short <= K      # safe backfill
holder+unsafe <= K but head+unsafe > K     # unsafe backfill
```

## Run and summarize

The normal entry point is one managed command:

```bash
python3 tools/bench/run_serve_ttft_campaign.py --campaign resource --samples 5
```

`smoke` runs the short baseline, `resource` runs the six Device/Host/eviction/catalog comparisons
plus the 64K bidirectional Host-swap case, and `full` runs all 55 audited cases. `resource` is the
default and `--samples` defaults to one.
Repeat `--case NAME` instead of `--campaign` to run a focused subset through the same managed
lifecycle, for example:

```bash
python3 tools/bench/run_serve_ttft_campaign.py \
  --case session-hot-continuation \
  --case resume-after-interference-device \
  --case shared-sequential \
  --samples 3
```

The controller uses Qwen3.8-27B NVFP4 with FP8 KV, starts and stops a fresh Serve for every sample,
shows the runner's live progress, retains the RAM artifact across campaigns until reboot or source
replacement, and writes beneath:

```text
profiles/bench/ttft/qwen3_8_27b_nvfp4-fp8/<timestamp>-<campaign>/
  manifest.json
  raw/<profile>/<case>/sample-NNN.json
  progress/<profile>/<case>/sample-NNN.log
  serve/<profile>/<case>/sample-NNN.log
  request-log/<profile>/<case>/sample-NNN.jsonl
  summary.md
  summary.json
  summary.csv
  failures.json                         # only when a run fails
```

The raw JSON is the external TTFT measurement authority. Progress and text Serve logs are
operational diagnostics; the structured request log records internal timings for diagnosis and
does not replace the external measurement. Complete, failed, and interrupted
campaigns automatically produce the same summary bundle. `summary.md` is the human report;
`summary.json` is the complete machine-readable aggregate; `summary.csv` contains typed TTFT,
comparison, symmetric-order, rejection, and optional cross-campaign rows. `manifest.json` records
both the source artifact and the actual tmpfs path used by Serve.

The low-level runner remains available when a separately managed Serve process is intentional:

```bash
python3 tools/bench/run_serve_ttft.py \
  --base-url http://127.0.0.1:18080 \
  --case shared-fanout \
  --profile-label shared-prefix \
  --output profiles/bench/ttft/shared-fanout-01.json
```

Exit status is zero only for a constructed case. Raw results preserve external timestamps,
request body bytes, outcome, HTTP status/code, compact event trace, first-output order, completion
order, public model ID, and failed construction conditions.

The runner prints live progress to `stderr` and flushes every line. It reports profile validation,
corpus loading, model discovery, case-graph construction, barriers, and each request's preparation,
start, body upload, acceptance, first output, terminal outcome, waits, cancellation, and errors.
A five-second heartbeat names the last stage and every active request while a long prefill or decode
has no new public event. Stage lines use elapsed monotonic time and include TTFT or upload duration
when that value becomes observable. The final JSON remains the only content written to `stdout`, so
progress can be displayed interactively or redirected independently without changing report input.

To re-aggregate one complete or partial campaign manually:

```bash
python3 tools/bench/summarize_serve_ttft.py \
  profiles/bench/ttft/qwen3_8_27b_nvfp4-fp8/<campaign>
```

Add `--baseline <campaign>` for matching observations against an explicit prior campaign. The
summarizer reads only runs declared by the campaign manifest, reports missing, invalid, and
unconstructed samples, and links each issue back to raw/progress/Serve/request diagnostics.

Successful constructed TTFT is grouped by public model, architecture category, case, profile, and
request role. Each group reports raw samples, min, median, max, median absolute deviation, and
relative span. Declared comparisons use `subject - baseline`, so a positive delta means the subject
is slower; roles inside one case are paired per run, while cross-case comparisons use independent
group medians. Symmetric barrier roles are ranked inside each run without assigning semantic
meaning to arbitrary role names. The report does not publish QPS, throughput, P95, or inferred
internal cache actions.

## Deliberately absent cases

Public HTTP requests cannot precisely force or prove private long anchors, multiple independent
cache markers, an invalid “KV without State” checkpoint, allocator-fragment geometry, one exact
transaction/cancellation instruction boundary, or a particular partial page residency. Those are
Engine tests, not black-box TTFT cases. Large-scale concurrency, QoS, preemption, multi-GPU,
multi-model serving, artifact comparisons, and speculative-backend sweeps are outside the product
workload and this benchmark.
