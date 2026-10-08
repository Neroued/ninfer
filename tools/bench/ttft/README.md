# Serve TTFT and scheduling benchmark

The runner sends public HTTP requests to an already-running `ninfer-serve`. It records the whole
stream: first output, subsequent output events, completion, errors and cancellation. Optional
structured Serve logs explain cache and scheduling actions. A separate campaign controller starts
a fresh server for each sample and records its effective configuration.

## Measurement contract

Request JSON and media data URLs are prepared before timing. Each request owns a preconnected
HTTP/1.1 connection:

```text
t0 = immediately before writing the request
tb = request body write completed
ta = first protocol acceptance metadata
t1 = first nonempty text, reasoning or tool-argument delta
TTFT = t1 - t0
```

TTFT includes body upload, input preparation, queuing, prefill and first output. HTTP headers,
assistant-role chunks, usage and empty deltas do not count as output. A failure before `t1` has no
TTFT; a later failure retains the TTFT already measured.

The trace retains timestamps and sizes for every nonempty output event, without duplicating its
text. It reports output-event gaps, terminal latency, the tail after the last output, and transport
closure. SSE events are not tokens: a speculative commit can emit several tokens at once, and
several frames can arrive in one socket read. Cancellation records the action and actual closure;
an unterminated stream is distinguished from a protocol terminal.

Four results are separate:

- **Construction:** the declared arrival graph and required external ordering occurred.
- **Request outcome:** success, rejection, cancellation or failure, including partial output.
- **Mechanism coverage:** diagnostics actually observed the required preemption, restore or other
  action. Zero is `not_observed`; missing evidence is `unavailable`.
- **Performance:** measured latency, output gaps, completed work and costs. A constructed case with
  high TTFT remains a valid adverse result.

Completed output throughput is successful-request usage divided by the workload makespan,
including time spent on unsuccessful requests. Missing usage is not replaced with byte or event
counts. Histories built from generated answers and pressure-triggered cancellations are marked as
conditional workloads; their observed rates do not establish matched-input throughput.

## Workloads and resources

[Case definitions](cases.py) own request graphs and their construction checks.
[Profiles](profiles.py) own exact server arguments. `--help` on the runner lists available
cases. Chat, Responses and Anthropic cases exercise their respective public protocols; Responses
cases also cover stored conversation history.

Device KV grows with executed work. For an ordinary request with prompt length `p` and output
limit `o`, `64 * ceil((p + o - 1) / 64)` describes its eventual page-rounded Main KV length, not a
reservation taken at admission. Shared content, partial-page COW, speculative peaks and state
slots must also be accounted for. The profiler's observed allocations and transfers establish
which resource actually became scarce.

The resource campaign covers hot continuations, changing private/shared working sets, interference,
two-session 64K swaps and six-session 55K rotations. Cases retain their workload names when an
older cache policy is replaced; the name alone does not prove a particular victim or placement.
`private-state-working-set-shift` retains the protocol's default automatic shared writes.
`private-only-working-set-shift` uses explicit mode with no shared markers to isolate private
continuations; use that control when comparing an older profile that disabled shared storage.

The main scheduling workloads are:

| Family | Workload and evidence |
|---|---|
| `preemption-replay`, `preemption-snapshot` | Same two growing arrivals with Host disabled/enabled; the snapshot graph also has MTP and DFlash2 profiles. |
| `shared-growth-recovery` | Three shared-prefix branches and two initial short arrivals; two further short requests arrive on an actual Replay restore event. Request-level boundaries measure other requests' actual progress during recovery. |
| `host-history-pressure-cancel` | Two retained histories and three growing requests compete for limited Host capacity. A pause event identifies the request to cancel; subsequent probes measure retained history. |
| `vision-growth-replay` | Require the Vision request itself to be preempted and replayed. |
| `resident-{decode,prefill}-first-{roomy,pressure}` | Reverse the Engine ticket groups of long Prefill and two Decode streams, with enough/insufficient KV for their full growth. Measure stream gaps and completion as well as TTFT. |
| `mixed-arrivals-{sparse,steady,burst}` | The same twelve short, medium/long-input and growing-output requests arrive at fixed intervals, then drain naturally. |
| `agent-continuation-{replay,snapshot}` | Build a multi-turn conversation with a late developer message, recover a pressured generation, then retry and continue. |

Fixed-arrival graphs record planned offsets, actual sends and lateness. Long traces establish each
connection at its scheduled arrival, before that request's `t0`, so unused connections do not
expire at the server. Preparation/connection delays appear as send lateness. Submission never
waits for earlier requests to finish. Normal EOS remains enabled: output limits are not promises
of fixed completed work.

`shared-tools-revisit` measures stable and changed tool identities in one graph.
`media-preprocess-revisit` uses A,A,B,C,A with context reuse disabled, isolating cold, warm and
post-eviction media preparation. `decode-with-short-arrival` and `media-during-decode` observe
background output after the foreground completes, then cancel the background and check another
request. Their whole-run throughput is conditional on that cancellation boundary.

The two-cohort 55K case contains the complete ordinary rotation as its first phase. Phase reports
separate that first cohort and its warm requests; whole-server counters still belong to the full
run. Use the standalone ordinary case when an isolated whole-run resource comparison is needed.

## Run a campaign

The default artifact is `out/qwen3_8_27b_nvfp4.ninfer`, with FP8 KV. GPU campaigns run serially.

```bash
python3 tools/bench/run_serve_ttft_campaign.py --campaign preemption --samples 1

python3 tools/bench/run_serve_ttft_campaign.py \
  --case session-hot-continuation \
  --case private-state-working-set-shift \
  --samples 3
```

Campaigns select workloads by purpose:

| Campaign | Selection |
|---|---|
| `full` | Composite performance run, including the new load/recovery cases; excludes contract checks, specialist controls and the duplicated standalone 55K rotation. |
| `resource` | Cache-pressure and Host rotation workloads, including the private-only control. |
| `preemption` | Recovery, limited Host, Vision and agent graphs, including selected speculative backends. |
| `load` | Resident phase/ticket competition and finite mixed arrivals. |
| `contract` | HTTP rejection, context bounds and session publication checks. |
| `long-context` | The 256K cold case. |
| `preprocess` | Default versus single-thread heavy-media preparation. |
| `smoke` | One short cold case. |

The default is `resource` with one sample. Select
`--serve` and `--artifact` explicitly when using another binary or artifact. Fixture token facts
still need to match the selected tokenizer/template.

`--profile-config FILE` replaces the built-in profile catalog. This allows a preserved baseline
binary to use its own CLI without compatibility flags in the candidate:

```json
{
  "common_args": ["--kv-dtype", "fp8", "--no-thinking", "--greedy"],
  "profiles": {
    "cache-hot": ["--max-context", "8192", "--kv-capacity", "8192",
                  "--max-concurrency", "1", "--device-state-slots", "2",
                  "--host-context-mib", "0"]
  }
}
```

The controller applies a common one-second statistics interval, enables request logging, and
records these effective arguments in its manifest. Host capacity is a unified byte budget for KV
and state; when comparing a baseline with separate pools, match its actual aggregate backing.
`--device-state-slots` remains extra state capacity beyond active lanes.

Before loading servers, the controller stages the selected immutable artifact once under
`/dev/shm/ninfer-artifacts/`. It reuses that copy while the source identity is unchanged; insufficient
tmpfs space is an explicit error. Server warmup completes before measurement and does not populate
the logical prefix cache.

Each sample gets a fresh process. Results are written under the selected `--output-dir`, or the
normal `profiles/bench/ttft/` campaign directory:

```text
manifest.json
raw/<profile>/<case>/sample-NNN.json
progress/<profile>/<case>/sample-NNN.log
serve/<profile>/<case>/sample-NNN.log
request-log/<profile>/<case>/sample-NNN.jsonl
summary.md / summary.json / summary.csv
request-analysis.csv                 # per-request first-output attribution
failures.json                         # when a run fails
```

The raw client JSON owns external measurements. Structured logs own internal work and resource
observations. Failed and interrupted campaigns retain their measurements and produce the same
summary bundle.

Raw runs retain the actual request payload and assembled output text for offline prefix analysis.
After measurement, data URLs are deduplicated by SHA-256 under the adjacent
`<sample>-media/` directory; the JSON retains each item's type, encoding and relative path.

For an independently managed server:

```bash
python3 tools/bench/run_serve_ttft.py \
  --base-url http://127.0.0.1:18080 \
  --case shared-fanout \
  --profile-label shared-prefix \
  --request-log-jsonl profiles/bench/server-requests.jsonl \
  --output profiles/bench/shared-fanout.json
```

The runner does not configure that server; its profile label does not verify the actual settings.
It prints progress and five-second heartbeats to stderr. Exit success means the graph was
constructed; mechanism coverage and performance remain separate report dimensions.

## Interpretation and comparisons

```bash
python3 tools/bench/summarize_serve_ttft.py profiles/bench/candidate \
  --baseline profiles/bench/baseline
```

The report reads runs declared by the manifest and exposes missing, invalid and unconstructed
samples. TTFT groups retain raw samples, median, range and dispersion. Changed request graphs
use new names or a new `measurement_contract`; mismatched contracts are excluded from numerical
baseline comparisons. Request classes and workload phases report TTFT, terminal latency and
output-event gaps separately. Fixed traces also report injection span, send lateness, actual
completed output and drain time. Symmetric simultaneous roles
are compared by ordered observations within each run, rather than treating an arbitrary role's
changed admission order as a performance regression.

Client and server observations join through exact wire request/response IDs within a server
instance. Server request IDs then identify Engine work. Process clocks are not subtracted from
each other. Request statistics distinguish original prefill, replay, pauses, snapshot/replay restores
and transfers. Client cancellation can leave public usage unknown even when the Engine records
committed work; those counts do not establish that the client received it.

`request-analysis.csv` uses the server's
[`first_output_timing`](../../../docs/serving.md#structured-request-log) snapshot to partition
streaming TTFT into preparation, initial Engine queue, initial binding, paused time, remaining
resident time and HTTP residual. Each stage retains its duration and percentage of client TTFT;
the residual includes transport and publication outside the Engine boundary.
Admission columns retain the preferred reuse frontier, source-wait milliseconds, revoked checkpoint
count and fallback reason from Engine. Source waiting is already included in queue time. Historical
logs without these observations leave the columns unavailable.
Missing observations remain unavailable, and negative residuals remain visible as inconsistent
measurements. Aggregate responses finish at a different boundary and do not receive this streaming
attribution.

Separate columns report Host exposure, this request's prefill/replay submission, completion wait,
postprocessing and GPU stream intervals, plus completed context-transfer time and bytes. These
overlapping measurements explain work within the wall-time stages; adding them together would
double-count time. The per-case summary complements the full per-request CSV.

For a theoretical reuse bound, compare the actual prepared input with earlier committed histories,
including token, position, media and rewrite semantics. A matching prefix alone does not establish
that its recurrent state was saved. Distinguish the semantic bound from the deepest complete
checkpoint retainable under the case's physical budget, then explain the actual selected frontier.
Replay of the same paused request is additional work, not a cross-request cache hit.

Sparse `request_scheduling` records use Engine timestamps and cumulative global/request work at
pause and recovery boundaries. Replay coexistence subtracts this request's work from the global
increment over its restored-to-replay-complete interval. Missing boundaries remain unavailable;
periodic phase snapshots do not substitute for that evidence. Request identity is available from
HTTP response headers while the stream is still active, allowing pressure cancellation to target
the observed request.

Global interval statistics include background cache demotion, which has no request owner, and the
Engine's mutually exclusive Host work accounting. Device wait and detailed subsets must not be
added again to Host totals. The final shutdown interval completes the counter stream when its
marker is present. Legacy or incomplete logs remain explicitly unconfirmed.

Periodic resource occupancy is a **sampled maximum**. Only the allocator's actual high-water mark
is called a peak. Reserved destinations count against physical capacity while copies are in flight.

A comparison needs matching workload, physical budget, artifact, backend, observation settings and
hardware conditions. Reported eligibility does not itself establish that these conditions match.
Use repeated interleaved baseline/candidate runs for performance conclusions; preserve adverse
samples and failures. State precisely which mechanisms and workloads the measurements cover.

Exact state/KV contents, lease retirement, allocator geometry and cancellation at particular Native
transaction boundaries belong in behavioral and Native tests. HTTP measurements complement those
checks with real scheduling, protocol and performance evidence.
