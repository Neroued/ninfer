# Deployment — Qwen3.8-27B NVFP4 as a container

Operating notes for running this build as a dedicated local LLM server, in a container, on
`<host>` (RTX 5090, 32 GiB, driver 595.84). This is a host-specific operating profile
rather than upstream's published methodology; [`performance.md`](performance.md) holds the latter.

> Figures here are serving-performance only. No answers were scored and no accuracy is claimed.

## Start it

```sh
cp .env.example .env && chmod 600 .env   # then fill in NINFER_API_KEY
docker compose up -d
docker compose logs -f ninfer
```

Compose refuses to create the container without `NINFER_API_KEY`, because the endpoint is reachable
from the public internet through Tailscale Funnel and an unauthenticated start would be a live
regression.

A healthy start logs the model load, then the resolved KV capacity, then `warming up...`,
and finally:

```text
listening on http://0.0.0.0:11434 (model id: qwen3.8-27b, auth: bearer)
```

`auth: bearer` is the confirmation that the key was picked up; `auth: disabled` means
`NINFER_API_KEY` never reached the process. Nothing answers on the port until all of that
finishes — the socket is bound before the weights are read, so a TCP check succeeds long before
the server does.

Health: `curl -s localhost:11434/health` (unauthenticated).
Model id: `qwen3.8-27b` — requests must send exactly this in `model`.

## The configuration, and why

Everything is set through the environment; the container's entrypoint renders the flags. Flags
passed after the image name are appended last and win, because the option parser is last-wins —
that is the escape hatch for the values marked fixed.

| Flag | Value | Override | Reason |
|---|---|---|---|
| `--max-context` | `131072` | `NINFER_MAX_CONTEXT` | NVFP4's validated ceiling. Costs nothing to set high — see below. |
| `--kv-capacity` | `auto` | fixed | A *shared* pool, sized from VRAM left after the weights. It resolved to **193,792 tokens** in the container here; the earlier bare-metal profile resolved 196,160. Read the actual figure off the startup line rather than assuming a constant. |
| `--kv-dtype` | `int8` | fixed | Group-64. Only `bf16` and `int8` exist; `int8` doubles the pool. |
| `--max-concurrency` | `8` | `NINFER_MAX_CONCURRENCY` | The engine's hard maximum; the range is `1..8`. |
| `--spec mtp --draft-tokens 3` | | `NINFER_DRAFT_TOKENS` | MTP is the only speculative backend this model supports. |
| `--lm-head-draft` | on | fixed | Optimized proposal head; used in every fast configuration. |
| `--min-p` | `0.05` | `NINFER_MIN_P` | **Must be a server flag** — there is no per-request `min_p` field, and the registered Qwen3.8 thinking default is `0`. Set `NINFER_MIN_P=0` to use NInfer's registered default instead. |
| `--chat-style` | `sharp-v22.1` | `NINFER_CHAT_STYLE` | Cuts completion tokens sharply without changing correctness. Set `default` for the artifact's stock chat template. |
| `--default-max-tokens` | `131072` | `NINFER_DEFAULT_MAX_TOKENS` | Output cap for clients that omit `max_tokens`; upstream's `8192` truncates long answers silently. **Costs concurrency** — see below. |
| `--host` | `0.0.0.0` | `NINFER_HOST` | Must bind all interfaces *inside* the container for the published port to reach it. The security boundary is the publish spec, not this value. |
| `--port` | `11434` | `NINFER_PORT` | What the existing Tailscale funnel already proxies to. |
| `--request-log-jsonl` | `/var/log/ninfer/requests.jsonl` | `NINFER_REQUEST_LOG_JSONL` | Per-request records, on a named volume. |
| prefix reuse | **on** (default) | — | Deliberately *not* disabled. The benchmark spec passes `--no-prefix-reuse` to force cold prefill; production wants the opposite, and on this host's traffic it is a large TTFT win. |
| CUDA graphs | on (default) | — | Never pass `--no-cuda-graph`. |

## Response length

Three separate limits decide how long an answer can get, in this order:

1. **What the client asks for.** `max_tokens` (Anthropic and OpenAI), `max_completion_tokens`
   (OpenAI), or `max_output_tokens` (Responses). A client-supplied value always wins; the server
   default is never consulted for such a request. Most agent clients set this themselves, so check
   what your caller sends before blaming the server.
2. **The server default**, used only when the client sends no field at all. Upstream's default is
   8192; this profile sets it to the full 131,072 so an unconfigured client is never truncated below
   the context ceiling.
3. **The engine's hard clamp** — `max_context - prompt_tokens + 1`.

A response cut short by any of the three comes back with `finish_reason: "length"` (Anthropic:
`stop_reason: "max_tokens"`); the wire format does not distinguish which one fired. The
`request_done` record settles it: `requested_output_tokens` together with
`requested_output_tokens_source` (`"client"` or `"server_default"`). The `server_start` record logs
the profile's own value as `default_output_tokens`.

### The default is a reservation, not just a ceiling

This is the part worth understanding before raising it. At admission a request is entitled to
`prompt_tokens + requested_output_tokens` worth of KV pages, and those pages are held against the
shared ~193.8K-token pool for the request's whole life — not allocated lazily as tokens are
produced. The requested figure is the client's `max_tokens` or, absent one, this default.

So the number of lanes that can run at once is set by `prompt + cap`:

| Default traffic sends | Entitlement each | Lanes admitted |
|---|---:|:--|
| own `max_tokens` = 16K, 6K prompt | ~22.4K | 8 — full batching |
| own `max_tokens` = 32K, 8K prompt | ~40K | 4 |
| no `max_tokens` (this profile) | 131,072 | **1** |

With a 131,072 default, the entitlement works out to exactly `--max-context` whatever the prompt
length, because the clamp in (3) is what caps it. Two such requests need 262,144 tokens against a
~193.8K-token pool, so the second one waits rather than running beside the first — the queue holds
16 and gives up after 30 s (`--max-pending-requests`, `--pending-timeout-ms`).

This only taxes clients that omit `max_tokens`. Traffic that sets its own cap is entitled to exactly
that cap and keeps batching normally, which is why the C=8 figures below still stand for
well-behaved callers. If you need 8-wide batching from default traffic too, the condition is
`prompt + cap <= pool / 8` — about 24,200 tokens at the pool size above, i.e.
`NINFER_DEFAULT_MAX_TOKENS=16384` for prompts up to roughly 7.8K.

## Measured throughput

Qwen3.8-27B NVFP4, MTP3, int8 KV, 128K ceiling, thinking enabled:

| C | Aggregate decode | Per request |
|---:|---:|---:|
| 1 | 191.9 tok/s | 191.9 |
| 4 | 585.1 tok/s | 146.3 |
| 8 | **903.4 tok/s** | 112.9 |

> Provenance: measured on a **bare-metal** build at commit `feaf4dd` against the host CUDA 13.3
> toolkit, RTX 5090 32 GiB, driver 595.84 — not re-measured under the container. They are expected
> to carry over, and that expectation is the reason `CUDA_VERSION` defaults to the same 13.3 minor:
> the device code is identical ahead-of-time `sm_120a` with no JIT, GPU compute has no container
> overhead, and the artifact is read from the same ext4 device through the same `O_DIRECT` path.
>
> Spot-checked in the container at C=1: **170.3 tok/s** over a 4,096-token completion, at **59.6%**
> MTP acceptance. That acceptance is prose-range rather than the 76–91% code range, and 170 tok/s
> sits inside the 126–200 band the next paragraph describes for it, so the sample neither confirms
> nor contradicts the 191.9 headline — it is one point, not a re-measurement.

Single-request decode varies strongly with how predictable the text is, because MTP acceptance does:
structured output and code accept ~76–91% and run near 200 tok/s, prose accepts ~37% and runs near
126. Baseline decode without speculation is about 71 tok/s, so MTP3 is worth roughly 1.8–3×
depending on workload.

Prefill is roughly 8,300 tok/s at 8K of context, 5,300 at 64K, 3,500 at 130K.

## Context vs concurrency — the one thing to understand

`--max-context` is a *per-sequence ceiling*. `--kv-capacity` is a **shared pool** — 193,792 tokens
as resolved here. Setting a 128K ceiling does **not** reserve 8 × 128K, and it does not by itself
cost throughput.

What matters is the live working set:

| Concurrent working set | KV needed | Fits? |
|---|---:|:--|
| 8 lanes × ~24K | ~5.2 GiB | ✅ full 8-wide batching |
| 1 × 128K + 7 short | ~5.2 GiB | ✅ |
| 2 × 128K | 8.3 GiB | ❌ |
| 8 × 128K | 33 GiB | ❌ — 5× the whole card |

So the server degrades gracefully: when several long contexts are live the pool throttles admission,
average batch falls, and aggregate throughput drops toward the 300–400 tok/s range. When lanes are
short you get the full ~900 tok/s. This is a property of the *workload*, not of the ceiling — which
is why the ceiling is set high here.

Memory budget at C=8: weights 19.7 GiB, sequence arena 8.9 GiB (KV 6.6 GiB at **35.1 KiB/token**,
plus ~300 MiB per lane of Gated Delta Net recurrent state), leaving ~1.6 GiB free. KV is this cheap
because only 16 of the 64 layers are full attention; the other 48 are GDN and hold constant-size
state regardless of context length.

## Networking and Tailscale

The container publishes `127.0.0.1:11434` only. Tailscale already fronts that address from the host
network namespace, and **no tailscaled change is needed**:

- `https://<host>.<tailnet>.ts.net` — Funnel, proxying `/` to `http://127.0.0.1:11434`
- `https://<service>.<tailnet>.ts.net` — tailnet-only service (`svc:ollama`), same target

That configuration lives in host `tailscaled`, not in this repository; do not look for it here.

Two consequences worth stating. First, this **removes** LAN exposure: the previous bare-metal
process bound `0.0.0.0:11434` and answered any host on the network, whereas now the only listener on
a routable interface is Tailscale's. Second, Funnel is the **public internet** — `NINFER_API_KEY` is
the only thing between an anonymous caller and the GPU, which is why Compose refuses to start
without it.

## Secrets

The key reaches the process through `NINFER_API_KEY`, which `ninfer-serve` reads itself. It never
appears on the command line, so it is absent from `ps`, from `/proc/<pid>/cmdline`, and from the
JSONL `startup_argv` — `server_start` reports only `api_key_configured: true`.

It **is** readable through `docker inspect` and `docker compose config` by anyone who can reach the
Docker daemon. The improvement is over `ps`-visibility to every local user, not absolute secrecy.
Keep `.env` at mode 600.

## Lifecycle

`restart: unless-stopped` plus an enabled `docker.service` supplies everything the previous systemd
user unit did — start on boot, restart on failure — with no unit file and no environment file.

| Task | Command |
|---|---|
| start | `docker compose up -d` |
| rebuild after a code change | `docker compose up -d --build` |
| stop, staying stopped across reboot | `docker compose stop` |
| stop and remove | `docker compose down` |
| logs | `docker compose logs -f ninfer` |

Points that surprise people:

- **`unhealthy` does not restart anything.** `restart: unless-stopped` reacts to process exit only.
  Health is a signal for `docker ps`, `docker inspect`, and `depends_on`.
- **Graceful stop drains in-flight generations with no engine-side timeout**, bounded only by
  `stop_grace_period: 5m`. A generation still running at that point is killed. Raise the value
  deliberately if that matters more than restart latency.
- **SIGTERM during model load is not graceful.** Signal handlers are installed only after warmup, so
  an interrupted load dies immediately. That is clean, not a hang.
- **A fatal configuration error becomes a restart loop** under `unless-stopped`, with exponential
  backoff. Read `docker compose logs`.

## Monitoring

`docker compose logs -f ninfer` carries the periodic throughput and scheduler-occupancy lines
(console output goes to stderr, which is where `docker logs` reads from).

Per-request records land in the `ninfer-logs` volume:

```sh
docker compose exec ninfer tail -f /var/log/ninfer/requests.jsonl
```

Each `request_done` line carries token counts, unrounded phase seconds, and the full speculative
counters including `accepted_per_position` — that array is the depth-headroom signal telling you
whether a different draft window would help. Container health:

```sh
docker inspect --format '{{.State.Health.Status}}' ninfer
```

## Tuning levers, in order of effect

1. **Working-set discipline** — the dominant factor, per the table above.
2. **`NINFER_DRAFT_TOKENS`** — `3` is upstream's choice, tuned at C=1. Measured acceptance falls to
   ~58% at C=8, where rejected drafts cost more, so `2` may win under sustained load. Worth a sweep
   against your own traffic.
3. **`NINFER_MAX_CONCURRENCY`** — lowering it to 4 raises per-request speed (146 vs 113 tok/s) at
   the cost of aggregate. Choose by whether latency or total throughput matters more to you.
4. **`NINFER_MIN_P`** — affects acceptance, and so speed, not just output quality.

## What this build cannot do

- **No DFlash on this model.** `--spec dflash` loads only the 35B-A3B backend. Qwen3.8-27B resolves
  to the `Qwen3_6_27B` target, whose `DFlashConfig` declares `supported = false` — it is compiled
  out, not gated at runtime. MTP with `--draft-tokens 1..5` is the only speculative option.
- **No multi-GPU, no CPU offload.** Single GPU only; a second card cannot extend the pool.
- **Vision costs context.** `--vision` allocates the vision weights and scratch buffers up front,
  taking the space directly out of the KV pool. Leave it off unless you need image or video input,
  and consider a separate container for it.

## Troubleshooting

**The container is refused at creation with a CUDA version message.** The base images declare
`NVIDIA_REQUIRE_CUDA=cuda>=<version>`, which libnvidia-container checks against the driver. Driver
595.84 satisfies the 13.3 default, so this does not occur here; it would after a driver downgrade,
or if `NINFER_CUDA_VERSION` were raised past what the driver supports. The fix is to lower the pin —
`NINFER_CUDA_VERSION=13.2.1 docker compose build` — not to set `NVIDIA_DISABLE_REQUIRE`, which
would only defer the failure to runtime. Note that `nvidia-smi`'s "CUDA Version" column reports the
newest runtime the driver ships, and is not the value this check compares against.

**Startup aborts on device memory.** The engine reports exact byte counts before copying any
weights. One resident model owns the GPU, so the usual cause is another engine still holding VRAM.
Name it from the host — `nvidia-smi` inside the container cannot see processes in the host PID
namespace:

```sh
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

**Startup aborts opening the artifact.** The artifact is opened `O_DIRECT` with no fallback. It must
live on a filesystem that supports it: a bind mount from ext4 is fine, tmpfs never works, and an
image layer on overlayfs cannot be relied on. Never bake the `.ninfer` file into the image.

**`--request-log-jsonl` aborts at startup.** Its parent directory must already exist. The image
pre-creates `/var/log/ninfer` owned by uid 1000; a host bind mount to a path Docker has to create
would be root-owned and unwritable by the non-root process — `chown 1000:1000` it first.

## Relationship to the benchmark spec

`<host>-benchmarks/specs/qwen3.8-27b-nvfp4-ninfer-mtp3-c8.yaml` runs the same engine with two
deliberate differences: prefix reuse disabled (cold prefill by design) and fixed-length generation.
Do not copy that spec's flags here — its job is comparability across engines, not speed.
