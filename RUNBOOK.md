# GPU runbook: vLLM-dialect thinking off

Schema and request-parse tests on this branch cover the dialect without loading a
model. This runbook is the remaining full-size check: greedy byte-identity across
the three thinking-off spellings, plus the 400 paths that a live HTTP server must
echo.

Do not run this until the coordinator schedules a GPU window. The registered
Qwen3.8-27B NVFP4 artifact is about 20 GiB of weights and cannot fit the 16 GiB
compact allocation cap.

## Constraints

- Do not stop, pause, or restart docker containers.
- Before any process that allocates GPU memory:
  1. `nvidia-smi` must show at least 20 GiB free.
  2. Acquire `C:\Users\igorl\.ninfer-gpu.lock` with `mkdir` (atomic). If it
     exists, wait 60 s and retry for up to 30 minutes, then stop.
  3. Remove the lock directory immediately after, success or failure.
- Rebuild of `ninfer:seedstore` and the `:8018` lane is coordinator-owned.

## Server

Full-size serving flags (no `--preserve-thinking`):

```text
--max-context 131072 --kv-capacity 1048576 --max-concurrency 8 --spec mtp
--draft-tokens 5 --lm-head-draft --kv-dtype int8 --prefill-chunk 2048 --vision
--cors --prefix-cache-mib 4096
```

Public model ID: `qwen3.8-27b`. Base URL in this runbook: `http://127.0.0.1:8018`.

Use temperature 0, seed 0, and a cold prefix (new prompt text, or restart so the
prefix cache does not hide a template mismatch).

## Prompt set

Three one-shot Chat Completions bodies. Only the thinking-off spelling changes.

Shared fields:

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Reply with the single word ping."}],
  "max_completion_tokens": 32,
  "temperature": 0,
  "seed": 0
}
```

| Name | Extra fields |
|---|---|
| `effort_none` | `"reasoning_effort": "none"` |
| `kwargs_off` | `"chat_template_kwargs": {"enable_thinking": false}` |
| `top_off` | `"enable_thinking": false` |

Also send one Responses request with `"input": "Reply with the single word ping."`,
`"max_output_tokens": 32`, `"temperature": 0`, and
`"chat_template_kwargs": {"enable_thinking": false}`.

## Expected generation

For each thinking-off spelling:

- HTTP 200.
- `choices[0].message.content` is byte-identical across `effort_none`,
  `kwargs_off`, and `top_off` on a cold prompt.
- `choices[0].message.reasoning_content` is absent or empty.
- `usage.prompt_tokens` is identical across the three Chat Completions spellings
  (the chat template must have taken the same thinking-off branch).

A live server with thinking left on (omit all three spellings, default) must
differ: `reasoning_content` is non-empty or `prompt_tokens` is larger.

## Expected 400s

`chat_template_kwargs.enable_thinkng: false` (misspelling):

```json
{"error":{"message":"chat_template_kwargs.enable_thinkng is not supported","type":"invalid_request_error","param":"chat_template_kwargs","code":"chat_template_option_not_supported"}}
```

`enable_thinking: true` together with `chat_template_kwargs.enable_thinking: false`:

```json
{"error":{"message":"conflicting enable_thinking values","type":"invalid_request_error","param":"enable_thinking","code":"conflicting_template_option"}}
```

`reasoning_effort: "high"` against the registered template:

HTTP 400 `reasoning_effort_not_supported`. Do not treat this as a dialect
failure; `high` is a parsed protocol value and is not an alias of `xhigh`.

## Commands

```bash
BASE=http://127.0.0.1:8018
MODEL=qwen3.8-27b

curl -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word ping.\"}],
  \"max_completion_tokens\": 32,
  \"temperature\": 0,
  \"seed\": 0,
  \"reasoning_effort\": \"none\"
}"

curl -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word ping.\"}],
  \"max_completion_tokens\": 32,
  \"temperature\": 0,
  \"seed\": 0,
  \"chat_template_kwargs\": {\"enable_thinking\": false}
}"

curl -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word ping.\"}],
  \"max_completion_tokens\": 32,
  \"temperature\": 0,
  \"seed\": 0,
  \"enable_thinking\": false
}"

curl -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word ping.\"}],
  \"chat_template_kwargs\": {\"enable_thinkng\": false}
}"

curl -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word ping.\"}],
  \"enable_thinking\": true,
  \"chat_template_kwargs\": {\"enable_thinking\": false}
}"
```

Compare the three 200 bodies with `jq -S '.choices[0].message'` (or equivalent).
Pass only when content, reasoning_content, and prompt_tokens match.

---

# GPU runbook: native Windows vs WSL2-container tax

Native `ninfer-serve.exe` now compiles on this box (MSVC 19.44.35228 + CUDA 13.3.33,
`sm_120a`, `NINFER_BUILD_MEDIA=OFF`). Do not boot it or acquire the GPU lock until
the coordinator schedules an exclusive window. The Qwen3.8-27B NVFP4 artifact is
about 20 GiB and does not fit the 16 GiB compact cap.

This runbook measures the WSL2 tax. Decode is expected to be similar (GPU-resident,
bandwidth-bound). Prefill, TTFT, weight load, boot wall time, and seed-store
captures all cross the WSL2 boundary on the container arm.

## Arms

| Arm | Runtime | Binary |
|---|---|---|
| A container | `ninfer:seedstore` under WSL2/docker | container `ninfer-serve` from `feat/prefix-seed-store` @ 352a49c3 plus this Windows port |
| B native | `build-win/apps/ninfer-serve.exe` from `task/issue-6-native-windows` | same git tree, MSVC+CUDA 13.3, text-only (`NINFER_BUILD_MEDIA=OFF`) |

Same checkpoint: `neroued/Qwen3.8-27B-nvfp4-NInfer` / public model id `qwen3.8-27b`.
Same serving flags both arms. Native currently **cannot** honor `--vision` until
FFmpeg+libcurl are installed. Until then, drop `--vision` on **both** arms so the
A/B stays matched, or install the media prefix and rebuild native with
`-DNINFER_BUILD_MEDIA=ON` before the window.

Never stop production containers (`sglang-qwen38` on `:8016`, embeddings, whisper).

## Server flags (matched)

No `--preserve-thinking`. JSONL log required. Port 8018 native or container, one
at a time.

```text
--host 127.0.0.1 --port 8018
--max-context 131072 --kv-capacity 1048576 --max-concurrency 8
--spec mtp --draft-tokens 5 --lm-head-draft
--kv-dtype int8 --prefill-chunk 2048 --cors
--prefix-cache-mib 4096
--request-log-jsonl <arm>-ninfer.jsonl
```

Add `--vision` only when both arms actually load Vision.

Native launch (after vcvars64 + CUDA 13.3 on PATH):

```bat
build-win\apps\ninfer-serve.exe <artifact.ninfer> --host 127.0.0.1 --port 8018 ...
```

Record wall time from process start to first `GET /health` 200. That is boot
wall time. Weight-load time is the `load_progress` / startup log span until the
server accepts connections.

## Metrics

Collect on every request from `request_done` JSONL:

- prefill tok/s = `computed_prefill_tokens / timings_seconds.prefill`
- decode tok/s = `(completion_tokens - 1) / timings_seconds.decode`
- TTFT = `timings_seconds.ttft`
- `prefix_reuse_path`, `prefix_cache_hit_tokens`

### Prefill (~6k and ~57k)

Two prompt lengths, serial, c=1, `--greedy` (prefill is not MTP-luck bound):

- ~6k: a real ~6k-token chat from the serving corpus or long-niah fixture.
- ~57k: a long-context body in the same family. Report `prompt_tokens` from
  `request_done` so the two buckets are actual, not nominal.

Three repetitions each length per arm. Report median prefill tok/s.

### Decode c=1, >= 3 boots

Three full process boots per arm, interleaved A/B/A/B/A/B, c=1 n=16,
`examples/cli/messages/scenario_*.json` cycled to 16, MTP-5, **not** greedy.
Arm score = median of 3 boot medians. Single-boot deltas under ~10% are noise.

### TTFT cold vs seeded

`--greedy`, prefix-cache on. Two identical temp=0 seed=0 requests:

1. Cold: `prefix_reuse_path=full_reset`. Record TTFT.
2. Seeded: `prefix_reuse_path=seed_prefix` (or restore_*). Record TTFT.

Pass the seed-store oracle if content is byte-identical and the second path is
not `full_reset`. Compare TTFT native vs container on both the cold and seeded
requests.

### Weight-load and boot wall time

Three boots per arm. Median seconds from process start to `/health` ok, and
median seconds of the weight-load phase from the startup log.

### Soak (after the A/B, not instead of it)

A multi-hour mixed-traffic soak on native, production-shaped chat + tools, before
native earns any default-local role. The container lane has already survived
hundreds of live agentic requests; native must match that bar. Out of scope for
the first exclusive window if time is short — schedule separately, do not skip.

## Configure (native, already proven on this machine)

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=%CUDA_PATH%\bin;%PATH%"
cmake -S . -B build-win -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" -DNINFER_BUILD_MEDIA=OFF -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build build-win -j --target ninfer-serve
```

## GPU lock

Coordinator only. `nvidia-smi` >= 20 GiB free; `mkdir C:\Users\igorl\.ninfer-gpu.lock`
(retry 60 s, up to 30 min); remove the lock directory after, success or failure.

---

# GPU runbook: prefix-cache usage observability

Drive one conversation so all four Engine reuse paths appear, then assert Chat
Completions `usage` matches `--request-log-jsonl` `request_done.result` for the
same request. Schema tests already cover field shape; this is the live match.

Server: production flags including `--prefix-cache-mib 4096` and
`--request-log-jsonl /tmp/ninfer-usage.jsonl`. Model `qwen3.8-27b`.
`--greedy`. `enable_thinking: false`.

Conversation (serial, same process):

1. Unique user prompt A → expect `prefix_reuse_path=full_reset`,
   `prefix_cache_hit_tokens=0`.
2. Repeat prompt A as a new request → expect `seed_prefix` and
   `prefix_cache_hit_tokens` > 0.
3. Prompt A + assistant reply + new user turn B → expect
   `restore_turn_checkpoint` or `append_frontier` (record whichever the log
   prints; both are valid).
4. Append another user turn C on that history → expect `append_frontier` or
   `restore_turn_checkpoint`.

For each request, parse the HTTP `usage` object and the matching
`request_done` JSONL event. Pass only if:

- `usage.prompt_tokens`, `usage.completion_tokens` match `result.prompt_tokens`
  / `result.completion_tokens`
- `usage.prefix_cache_hit_tokens` == `result.prefix_cache_hit_tokens` ==
  `usage.prompt_tokens_details.cached_tokens`
- `usage.prefix_reuse_path` == `result.prefix_reuse_path` (string equality)
- `usage.total_tokens` == prompt + completion (cached_tokens is not an addend)
- `GET /v1/models` `data[0].max_model_len` equals the process `--max-context`

If a listed path does not appear, do not invent a fifth name; record the
observed path from the log and fail only if usage disagrees with that log.
