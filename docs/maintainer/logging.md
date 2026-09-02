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

The default logger is synchronous and uses a multi-thread-safe stderr color sink. `Auto` color is
selected only for a capable terminal; redirected logs contain no ANSI escapes. Asynchronous queues,
file sinks, rotation, and multiple destinations require a separate product decision because they
introduce durability, ordering, overflow, and lifecycle semantics.

## 3. Record format

The fixed operational pattern is:

```text
[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v
```

It produces local wall time with millisecond precision, the lowercase spdlog level, the executable
logger name, and one complete message. Examples:

```text
[2026-09-01 21:14:50.971] [info] [ninfer-serve] startup phase=host-kv-pin status=begin total_bytes=51539607552
[2026-09-01 21:15:22.774] [info] [ninfer-serve] startup phase=host-kv-pin status=complete completed_bytes=51539607552 total_bytes=51539607552 duration_ms=31803.100
[2026-09-01 21:15:23.102] [info] [ninfer-serve] request id=42 status=done prompt_tokens=55055 prefix_cache_hit_tokens=55048 ttft_ms=75.200
```

Messages start with a stable event noun and then stable `key=value` fields. Field names use
lowercase snake case. Values containing whitespace or control characters must be escaped or quoted
by the owning renderer. A renderer must not create a second timestamp, level label, executable
prefix, or trailing newline inside the message.

Exact counts use base units in their field name: `bytes`, `tokens`, `pages`, `slots`, `duration_ms`,
or `elapsed_ns`. Human-readable GiB and rates may supplement an exact value but never replace it.
Floating-point fields state the same semantic boundary on every record and use enough precision for
operator interpretation; full-precision measurement remains in the machine record.

## 4. Levels

| Level | Contract |
|---|---|
| `trace` | exceptionally fine diagnostic events enabled only for a concrete investigation |
| `debug` | startup planning, graph/profile inventory, and resource-decision detail not needed for normal operation |
| `info` | phase transitions, readiness, normal request lifecycle, periodic throughput, and orderly shutdown |
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

When stderr is a terminal, `ninfer_product_logging` may render one transient startup progress line
below the persistent spdlog records. The progress-aware stderr sink owns the same mutex as normal
records: it erases the transient line before a log record and redraws it afterward. Only a phase
with real byte progress displays a percentage, pipeline rate, or ETA. Completion erases the line and
emits the ordinary structured terminal record. Redirected stderr never receives this transient
line, carriage returns, or progress-bar text; it receives rate-limited structured progress records.

Engine startup is an inclusive typed hierarchy. `engine-startup` contains CUDA initialization,
artifact inspection, target planning, weight materialization, target/frontend/Program construction,
and Engine finalization. `weights-staging-pin` is nested in `weights-materialize`; Host State pin,
Host KV pin, and CUDA Graph preparation are nested in `program-initialize`. Nested durations explain
their parent and must not be added to it. Disabled zero-capacity phases are omitted. Byte progress is
reported as submitted work, while only a synchronized terminal event reports completed bytes.

Serve owns request failure classification and severity. A prepared generation request receives one
machine terminal: `request_done` immediately after `GenerationService::run()` returns, or
`request_error` if it does not return an outcome. Response rendering, Responses storage, and terminal
transport happen after that transaction; their failures are operational `response` records and do
not create a second request JSONL terminal.

There is no dual spdlog/custom operational path. Product results, machine measurements, and the
explicit emergency cases above remain direct outputs because they are different contracts.

## 7. Verification policy

Logging tests protect NInfer-owned observable semantics, not spdlog internals or private object
shape. The request-log test covers the consumed JSONL schema plus Serve failure severity and the
rule that arbitrary client error text is absent from operational output. The corpus consumer test
protects its exact schema-version agreement with Serve. Startup ordering and presentation are
verified through real terminal and redirected startup paths when that code changes; synthetic
observer, formatter snapshot, registry, color, mutex, getter, and constructor tests are not
retained.
