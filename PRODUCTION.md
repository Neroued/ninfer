# Production profile — Qwen3.8-27B NVFP4 on <host>

Operating notes for running this build as a dedicated local LLM server.
Everything below was measured on this box (RTX 5090, 32 GiB, CUDA 13.3,
build `feaf4dd`) rather than taken from upstream's published tables.

> Figures here are serving-performance only. No answers were scored and no
> accuracy is claimed.

## Start it

```sh
export NINFER_API_KEY=…          # or rely on LLAMA_API_KEY already in env
./serve-production.sh
```

Defaults to `127.0.0.1:8080`. The script refuses to start if under 30 GiB of
VRAM is free, because one resident model owns the GPU and launching on top of
another engine OOMs about 25 ms in rather than failing cleanly.

Health: `curl -s localhost:8080/health` (unauthenticated).
Model id: `qwen3.8-27b` — requests must send exactly this in `model`.

## The configuration, and why

| Flag | Value | Reason |
|---|---|---|
| `--max-context` | `131072` | NVFP4's validated ceiling. Costs nothing to set high — see below. |
| `--kv-capacity` | `auto` | Resolves to **196,160 tokens** here, a *shared* pool. |
| `--kv-dtype` | `int8` | Group-64. Only `bf16` and `int8` exist; `int8` doubles the pool. |
| `--max-concurrency` | `8` | The engine's hard maximum; the range is `1..8`. |
| `--spec mtp --draft-tokens 3` | | MTP is the only speculative backend this model supports. |
| `--lm-head-draft` | on | Optimized proposal head; used in every fast configuration. |
| `--min-p` | `0.05` | **Must be a server flag** — there is no per-request `min_p` field, and the registered Qwen3.8 thinking default is `0`. This matches the sampler the llama.cpp host runs. Set `NINFER_MIN_P=0` to use NInfer's registered default instead. |
| `--default-max-tokens` | `131072` | Output cap for clients that omit `max_tokens`; upstream's `8192` truncates long answers silently. **Costs concurrency** — see below. Set `NINFER_DEFAULT_MAX_TOKENS` to lower it. |
| prefix reuse | **on** (default) | Deliberately *not* disabled. The benchmark spec passes `--no-prefix-reuse` to force cold prefill; production wants the opposite, and on this host's traffic it is a large TTFT win. |
| CUDA graphs | on (default) | Never pass `--no-cuda-graph`. |

## Response length

Three separate limits decide how long an answer can get, in this order:

1. **What the client asks for.** `max_tokens` (Anthropic and OpenAI),
   `max_completion_tokens` (OpenAI), or `max_output_tokens` (Responses). A
   client-supplied value always wins; the server default is never consulted for
   such a request. Most agent clients set this themselves, so check what your
   caller sends before blaming the server.
2. **The server default**, used only when the client sends no field at all.
   Upstream's default is 8192; this profile sets it to the full 131,072 so an
   unconfigured client is never truncated below the context ceiling.
3. **The engine's hard clamp** — `max_context - prompt_tokens + 1`.

A response cut short by any of the three comes back with
`finish_reason: "length"` (Anthropic: `stop_reason: "max_tokens"`); the wire
format does not distinguish which one fired. With `--request-log-jsonl` enabled
the `request_done` record settles it: `requested_output_tokens` together with
`requested_output_tokens_source` (`"client"` or `"server_default"`). The
`server_start` record logs the profile's own value as `default_output_tokens`.

### The default is a reservation, not just a ceiling

This is the part worth understanding before raising it. At admission a request
is entitled to `prompt_tokens + requested_output_tokens` worth of KV pages, and
those pages are held against the shared 196,160-token pool for the request's
whole life — not allocated lazily as tokens are produced. The requested figure
is the client's `max_tokens` or, absent one, this default.

So the number of lanes that can run at once is set by `prompt + cap`:

| Default traffic sends | Entitlement each | Lanes admitted |
|---|---:|:--|
| own `max_tokens` = 16K, 8K prompt | ~24.5K | 8 — full batching |
| own `max_tokens` = 32K, 8K prompt | ~40K | 4 |
| no `max_tokens` (this profile) | 131,072 | **1** |

With a 131,072 default, the entitlement works out to exactly `--max-context`
whatever the prompt length, because the clamp in (3) is what caps it. Two such
requests need 262,144 tokens against a 196,160-token pool, so the second one
waits rather than running beside the first — the queue holds 16 and gives up
after 30 s (`--max-pending-requests`, `--pending-timeout-ms`).

This only taxes clients that omit `max_tokens`. Traffic that sets its own cap is
entitled to exactly that cap and keeps batching normally, which is why the C=8
figures below still stand for well-behaved callers. If you need 8-wide batching
from default traffic too, the condition is `prompt + cap <= ~24,520` — i.e.
`NINFER_DEFAULT_MAX_TOKENS=16384` for prompts up to ~8K.

## Measured throughput

Qwen3.8-27B NVFP4, MTP3, int8 KV, 128K ceiling, thinking enabled:

| C | Aggregate decode | Per request |
|---:|---:|---:|
| 1 | 191.9 tok/s | 191.9 |
| 4 | 585.1 tok/s | 146.3 |
| 8 | **903.4 tok/s** | 112.9 |

Single-request decode varies strongly with how predictable the text is, because
MTP acceptance does: structured output and code accept ~76–91% and run near
200 tok/s, prose accepts ~37% and runs near 126. Baseline decode without
speculation is about 71 tok/s, so MTP3 is worth roughly 1.8–3× depending on
workload.

Prefill is roughly 8,300 tok/s at 8K of context, 5,300 at 64K, 3,500 at 130K.

## Context vs concurrency — the one thing to understand

`--max-context` is a *per-sequence ceiling*. `--kv-capacity` is a **shared pool**
of 196,160 tokens. Setting a 128K ceiling does **not** reserve 8 × 128K, and it
does not by itself cost throughput.

What matters is the live working set:

| Concurrent working set | KV needed | Fits? |
|---|---:|:--|
| 8 lanes × ~24K | ~5.2 GiB | ✅ full 8-wide batching |
| 1 × 128K + 7 short | ~5.2 GiB | ✅ |
| 2 × 128K | 8.3 GiB | ❌ |
| 8 × 128K | 33 GiB | ❌ — 5× the whole card |

So the server degrades gracefully: when several long contexts are live the pool
throttles admission, average batch falls, and aggregate throughput drops toward
the 300–400 tok/s range. When lanes are short you get the full ~900 tok/s. This
is a property of the *workload*, not of the ceiling — which is why the ceiling
is set high here.

Memory budget at C=8: weights 19.7 GiB, sequence arena 8.9 GiB (KV 6.6 GiB at
**35.1 KiB/token**, plus ~300 MiB per lane of Gated Delta Net recurrent state),
leaving ~1.6 GiB free. KV is this cheap because only 16 of the 64 layers are
full attention; the other 48 are GDN and hold constant-size state regardless of
context length.

## What this build cannot do

- **No DFlash on this model.** `--spec dflash` loads only the 35B-A3B backend.
  Qwen3.8-27B resolves to the `Qwen3_6_27B` target, whose `DFlashConfig`
  declares `supported = false` — it is compiled out, not gated at runtime. MTP
  with `--draft-tokens 1..5` is the only speculative option.
- **No multi-GPU, no CPU offload.** Single GPU only; a second card cannot
  extend the pool.
- **Vision costs context.** `--vision` allocates the vision weights and scratch
  buffers up front, taking the space directly out of the KV pool. Leave it off
  unless you need image or video input, and consider a separate process for it.

## Tuning levers, in order of effect

1. **Working-set discipline** — the dominant factor, per the table above.
2. **`--draft-tokens`** — `3` is upstream's choice, tuned at C=1. Measured
   acceptance falls to ~58% at C=8, where rejected drafts cost more, so `2` may
   win under sustained load. Worth a sweep against your own traffic.
3. **`--max-concurrency`** — lowering it to 4 raises per-request speed (146 vs
   113 tok/s) at the cost of aggregate. Choose by whether latency or total
   throughput matters more to you.
4. **`--min-p`** — affects acceptance, and so speed, not just output quality.

## Monitoring

`--log-stats-interval-ms 5000` prints periodic throughput and scheduler
occupancy. For per-request records, add:

```sh
./serve-production.sh --request-log-jsonl "$HOME/.local/state/ninfer/requests.jsonl"
```

The parent directory must already exist or startup aborts. Each `request_done`
line carries token counts, unrounded phase seconds, and the full speculative
counters including `accepted_per_position` — that array is the depth-headroom
signal telling you whether a different draft window would help.

## Relationship to the benchmark spec

The cross-engine benchmark spec runs the same engine with two deliberate
differences: prefix reuse disabled (cold prefill by design) and fixed-length
generation. Do not copy that spec's flags here — its job is comparability
across engines, not speed.

## Optional: run it under systemd

```ini
# ~/.config/systemd/user/ninfer.service
[Unit]
Description=NInfer — Qwen3.8-27B NVFP4
After=network.target

[Service]
Type=exec
Environment=NINFER_API_KEY=…
ExecStart=%h/src/ninfer/serve-production.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

`systemctl --user daemon-reload && systemctl --user enable --now ninfer`.
Put the key in an `EnvironmentFile=` with mode 600 rather than inline if the
unit file is ever going to be readable by anything else.
