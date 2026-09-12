# Operational logging

This document is the authority for repository-owned C++ operational logging. It defines what is a
log, which layer owns it, how records are formatted, and which other output contracts must remain
separate. CLI, Serve, and Perplexity construct the shared product logging runtime; their resident
operational producers follow this contract.

## 1. Output classes

NInfer has four distinct output classes. A shared destination such as stderr does not make them the
same contract.

| Class | Examples | Owner and representation |
|---|---|---|
| Operational log | startup phases, readiness, request lifecycle summaries, throughput, recoverable warnings and errors | application-owned spdlog logger |
| Product result | CLI answer/reasoning/token report, help and local command/input diagnostics, perplexity result table | the relevant application renderer |
| Machine measurement | request JSONL, benchmark raw/summary JSON, conversion and evaluation reports | the defining typed schema and writer |
| Emergency diagnostic | CUDA abort/invariant path, cleanup failure, or logging-sink failure after ordinary logging may be unavailable | minimal direct stderr writer at the owning low-level boundary |

NVTX ranges are profiler annotations, not logs. Tests and benchmark executables may write failure or
result reports directly because those streams are their observable product, not a resident process
operational channel.

Operational logging must never replace, wrap, or reinterpret a machine measurement record. One
typed event may have both a concise operational renderer and an independent machine-schema writer,
but neither consumes the other renderer's text.

## 2. Dependency and ownership

NInfer vendors the compiled spdlog library under `third_party/spdlog`. Configuration never fetches
network content or selects a system version. The `ninfer_product_logging` target is the only NInfer
library that owns logger construction policy.

An application entry point creates one `product::LoggingRuntime` with its executable name and owns
it until every component and worker that can log has stopped. Components receive an explicit
`std::shared_ptr<spdlog::logger>`. They do not select sinks, patterns, levels, global registry names,
or shutdown behavior.

No code may call spdlog's global default logger, mutate its global pattern or level, or register an
application logger in the global registry. `ninfer_core`, `ninfer_artifact`, `ninfer_ops`,
`ninfer_engine`, and target packages do not link product logging. These layers communicate an
observable diagnostic through an owning typed event/callback; the product adapter decides whether
to log it.

The default logger is synchronous and uses one progress-aware stderr color sink whose owning mutex
serializes records and transient-line updates. `Auto` color is selected only for a capable terminal;
redirected logs contain no ANSI escapes. Asynchronous queues, file sinks, rotation, and multiple
destinations require a separate product decision because they introduce durability, ordering,
overflow, and lifecycle semantics.

## 3. Record representations

Operational stderr is a human-readable presentation, not a machine schema. Serve records use local
wall time with millisecond precision and a fixed-width uppercase level; the process name is omitted
because the dedicated process stream already supplies that identity:

```text
2026-09-02 23:12:56.607  INFO  throughput | 5.0s | decode 190.6 tok/s (953 tok) | running 1 (decode-ready 1) | batch 1.00 | host 1.5% (73.4 ms)
2026-09-02 23:13:01.458  INFO  req#13 done | openai-chat | stop token | prompt 2,139 | output 54,088 | cache 129 (6.0%, turn closure) | TTFT 323 ms | total 5m 21.0s | prefill 6.49k tok/s | decode 169.0 tok/s
```

CLI and Perplexity use the Tool presentation: informational progress has no timestamp or level
prefix, while diagnostics use short `debug:`, `warning:`, `error:`, or `fatal:` labels. Both
presentations use ` | ` to separate semantic groups, adaptive durations, IEC byte units,
thousands-separated counts, and rounded rates. Optional zero-valued groups are omitted when their
absence is unambiguous. Trusted identifiers are normalized for display and control characters are
never allowed to create another physical record.

Pretty output is intentionally not parsed. Exact base units, full floating-point precision, all
zero-valued gauges, and stable snake-case field names belong to the independent typed JSON/JSONL
measurement writers. A producer passes the same typed value independently to its pretty renderer
and machine writer; neither consumes text produced by the other.

## 4. Levels

| Level | Contract |
|---|---|
| `trace` | exceptionally fine diagnostic events enabled only for a concrete investigation |
| `debug` | all startup transitions, graph/profile inventory, memory ledgers, and resource-decision detail not needed for normal operation |
| `info` | material startup milestones, readiness, normal request lifecycle, fixed-interval throughput, and orderly shutdown |
| `warning` | recoverable overload, timeout, degraded external input, or an operator-relevant condition that does not invalidate the process |
| `error` | failed operation, internal request failure, or loss of an optional output such as request JSONL |
| `critical` | process-level state cannot safely continue |
| `off` | explicit suppression of operational records |

Expected client mistakes are not automatically errors. The protocol adapter maps their operational
severity from the failure semantics. Logging level is not a substitute for the HTTP status or
machine event status.

Warnings and higher levels flush immediately. Normal shutdown flushes all sinks. A sink failure is
reported once through the emergency stderr path; logging must not recursively log its own failure.

## 5. Data policy

Operational records may contain request IDs, protocol/model identities, non-secret configuration,
counts, timings, cache paths, and summarized state transitions. Resident Serve request, response,
and HTTP records must not contain:

- API keys, authorization headers, cookies, signed URLs, or credentials;
- prompt text, generated content, reasoning text, tool arguments/results, or prior conversation;
- raw image, video, audio, tokenizer, tensor, StateImage, or KV payloads;
- request bodies or arbitrary client-controlled headers;
- full data URLs or unredacted query strings.

Filesystem paths are permitted only when they are operator-selected local configuration or output
paths and are necessary to diagnose the operation. A component that cannot prove a resident-service
value safe emits an identity/count/digest or omits it. Serve request/response/HTTP operational
records never include raw `ApiError.message`, arbitrary `exception.what()`, a request body, or an
unclassified request path. Trusted startup/configuration errors and the local one-shot CLI's own
input diagnostics may retain their detailed exception text. The protocol response and request
JSONL may retain an exact error string when their independent contracts require it.

## 6. Producer rules

A producer owns the event semantics and supplies already-validated values; it does not own logger
configuration. Cross-layer state is reported at its existing ownership boundary. In particular,
startup pinning is reported by Program around the Host State/Host KV construction calls, not by the
generic pinned-buffer primitive. Serve owns HTTP and request lifecycle severity. CLI owns its result
streams and must never prefix generated answer or reasoning data with operational-log metadata.

Progress producers emit a typed begin before a potentially blocking operation, bounded progress
updates when available, and one complete or failed terminal event. This guarantee covers normal
return and C++ exception unwinding; a fatal CUDA `abort()` uses the emergency diagnostic path and
does not unwind typed scopes. Rate limiting belongs to the product renderer. Producers do not print
ad hoc dots, percentages, carriage-return lines, or duplicate completion summaries.

When stderr is a terminal, `ninfer_product_logging` may render one transient startup or offline-tool
progress line below persistent records. The progress-aware stderr sink owns the same mutex as normal
records: it erases the transient line before a log record and redraws it afterward. Only a phase
with real byte progress displays a percentage, rate, or ETA. Completion erases the line and emits a
readable terminal summary. Redirected stderr never receives carriage returns, ANSI escapes, or a
progress bar; long operations instead receive rate-limited persistent progress records.

Serve runtime throughput is never transient. While Engine or context-resource activity exists, one
persistent pretty record and, when request logging is enabled, one full-precision JSONL record are
produced from the same interval at each configured deadline. Periodic scheduling uses absolute
steady-clock deadlines so logging work does not accumulate drift. Missed periods are not replayed
in a burst. A partial shutdown interval may be retained by JSONL, but does not create an irregular
pretty throughput record.

Engine startup is an inclusive typed hierarchy. `engine-startup` contains CUDA initialization,
artifact inspection, target planning, weight materialization, target/frontend/Program construction,
and Engine finalization. `weights-staging-pin` is nested in `weights-materialize`; Host State pin,
Host KV pin, and CUDA Graph preparation are nested in `program-initialize`. Nested durations explain
their parent and must not be added to it. Disabled zero-capacity phases are omitted. Byte progress is
reported as submitted work, while only a synchronized terminal event reports completed bytes.
The normal pretty view retains weight materialization, enabled Host pinning, CUDA Graph preparation,
and the merged Engine-ready summary. Internal staging, planning, frontend, Program aggregate, and
finalization phases remain `debug`; elapsed time alone does not make an internal milestone useful to
an operator. Every failed phase remains an error regardless of its ordinary visibility.

Serve owns request failure classification and severity. A prepared generation request receives one
machine terminal: `request_done` immediately after `GenerationService::run()` returns, or
`request_error` if it does not return an outcome. Response rendering, Responses storage, and terminal
transport happen after that transaction; their failures are operational `response` records and do
not create a second request JSONL terminal.

Tool-call parameter normalization remains a successful request outcome. Empty-argument omissions
and schema mismatches are machine-only counters. If a complete tool marker must be returned to text
because its structure or declared identity cannot be represented, Serve emits one warning carrying
only the stable fallback classification; generated markup and arguments remain excluded.

There is no dual spdlog/custom operational path. Product results, machine measurements, and the
explicit emergency cases above remain direct outputs because they are different contracts.

## 7. Verification policy

Logging tests protect NInfer-owned observable semantics, not private object shape. The request-log
test covers the consumed JSONL schema, representative request/throughput pretty records, Serve
failure severity, and exclusion of arbitrary client error text. The pretty-logging test covers the
observable Service and Tool prefixes. The corpus consumer test protects its exact schema-version
agreement with Serve. Startup progress coordination is verified through terminal and redirected
paths when that code changes; registry, mutex, getter, and constructor tests are not retained.
