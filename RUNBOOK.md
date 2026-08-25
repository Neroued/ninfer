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

---

# Runbook: Warmup Fail-Fast, Exception Logging, and Boot Watchdog (Issue #4)

## Overview

This runbook documents the operational verification procedures for:
1. **Crash / Terminate Logging**: Ensuring unhandled exceptions escaping thread or server boundaries print `typeid(error).name()` and `error.what()` directly through the console logger before calling `std::abort()` (preventing uninformative silent aborts under container PID 1).
2. **Warmup Fail-Fast**: Ensuring any exception during startup warmup (e.g., CUDA OOM, corrupted prefix cache, invalid batch allocation) terminates the process immediately with non-zero exit code (1) instead of leaving the process listening in an alive-but-503 zombie state.
3. **Pre-Listen Boot Watchdog**: A detached timer thread armed before warmup that terminates the process via `std::_Exit(1)` if boot does not reach the listening state within a configurable budget (default 120 s via `--boot-watchdog-timeout-s`), protecting against uncooperative GPU/driver wedges.
4. **Warmup Timeout Decoupling**: Ensuring startup warmup uses an explicit 60-second budget rather than the short client-facing `--pending-timeout-ms`.
5. **Auto KV-Capacity Bounding**: Clarifying `--kv-capacity auto` description in `--help` to state `(bounded by max-context * max-concurrency)`.

---

## 1. Automated Unit Tests (CPU Container Lane)

All serve unit tests run and pass inside the build container without requiring GPU hardware:

```bash
docker run --rm -v "P:\NInfer.gemini:/workspace" -w /workspace \
  -e LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:/usr/local/cuda-13.1/targets/x86_64-linux/lib/stubs \
  ninfer:test-build bash -c \
  "cd /workspace/build && ctest --output-on-failure -R 'ninfer_(serve_options|http_error_handler|openai_schema|responses_schema|response_store|anthropic_schema|tool_call_parser|request_log|kv_capacity)_test'"
```

### Verified Test Cases:
- `ninfer_serve_options_test`: Verifies `--help` text contains `(bounded by max-context * max-concurrency)` for `--kv-capacity auto` and `--boot-watchdog-timeout-s` options.
- `ninfer_http_error_handler_test`: Verifies HTTP error JSON mapping.
- `ninfer_kv_capacity_test`: Verifies sequence capacity curve and page allocation bounds.
- `ninfer_openai_schema_test`, `ninfer_responses_schema_test`, `ninfer_response_store_test`, `ninfer_anthropic_schema_test`, `ninfer_tool_call_parser_test`, `ninfer_request_log_test`: 100% passing.

---

## 2. Induced Failure & Error Path Procedures (GPU Maintenance Window)

When executed in a coordinator-scheduled GPU maintenance window under the cross-agent lock protocol (`C:\Users\igorl\.ninfer-gpu.lock`):

### Procedure A: Induce Warmup Failure (OOM / Allocation Fault)
Run `ninfer-serve` with `--prefix-cache-mib` set higher than available GPU VRAM:
```bash
./build/apps/ninfer-serve /path/to/qwen3_8_27b_nvfp4.ninfer --kv-capacity auto --prefix-cache-mib 60000 --port 8018
```
**Expected Observable Reality**:
- `httplib` binds the port and sets up the socket backlog at startup.
- Engine initialization or warmup throws `std::runtime_error("warmup generation failed: ...")` or allocation exception.
- Stderr log output:
  ```
  [YYYY-MM-DD HH:MM:SS.mmm] [error] ninfer-serve: warmup generation failed: ...
  ```
- The process does NOT enter `server.listen()` (the HTTP accept loop) and terminates immediately with exit code 1.
- Socket is closed upon process exit; no zombie 503 HTTP server remains running.

### Procedure B: Verify Clean Warmup & Normal Boot
Run `ninfer-serve` with standard production options:
```bash
./build/apps/ninfer-serve /path/to/qwen3_8_27b_nvfp4.ninfer --kv-capacity auto --max-context 131072 --port 8018
```
**Expected Observable Reality**:
- Console logs:
  ```
  [YYYY-MM-DD HH:MM:SS.mmm] [info] ninfer-serve: loading model...
  [YYYY-MM-DD HH:MM:SS.mmm] [info] ninfer-serve: model loaded in ... s
  [YYYY-MM-DD HH:MM:SS.mmm] [info] ninfer-serve: KV capacity auto resolved=...
  [YYYY-MM-DD HH:MM:SS.mmm] [info] ninfer-serve: warming up...
  [YYYY-MM-DD HH:MM:SS.mmm] [info] ninfer-serve: listening on http://127.0.0.1:8018 (model id: ..., auth: disabled)
  ```
- Boot watchdog is cleanly disarmed upon reaching the listening state.
- `curl http://127.0.0.1:8018/v1/models` returns HTTP 200 OK with model descriptor.

### Procedure C: Verify PID 1 Terminate Logging Handler
In a test container running without a custom init system:
- Trigger an unhandled exception escaping a thread boundary.
**Expected Observable Reality**:
- Stderr log output:
  ```
  [YYYY-MM-DD HH:MM:SS.mmm] [error] ninfer-serve: terminate called after throwing <type_name>: <message>
  ```
- `std::abort()` terminates the process via `SIGABRT` (exit code 134, or container protection fault).

### Procedure D: Verify Boot Watchdog Hang Protection
To test the uncooperative hang fallback, run with a short watchdog timeout:
```bash
./build/apps/ninfer-serve /path/to/qwen3_8_27b_nvfp4.ninfer --boot-watchdog-timeout-s 1 --port 8018
```
**Expected Observable Reality**:
- If model loading or warmup exceeds 1 second:
  ```
  [YYYY-MM-DD HH:MM:SS.mmm] [error] ninfer-serve: boot watchdog timeout (1 s) exceeded before reaching listening state; terminating process
  ```
- The watchdog terminates the process immediately via `std::_Exit(1)`.

---

## Issue #5: Tool-calling robustness and multimodal tool results

Schema and parser test suites (`ninfer_tool_call_parser_test`, `ninfer_openai_schema_test`)
validate the tolerant parser and content parts on CPU without GPU memory. Live-model
validation for multimodal tool results and parallel tool-calling loops is documented below.

### 1. Multimodal tool result (screenshot in tool message)

Start server with `--vision` and `--tolerant-tool-calls`:

```bash
BASE=http://127.0.0.1:8018
MODEL=qwen3.8-27b

# Turn 1 + 2: User requests screenshot, assistant calls tool, tool returns image data part
curl -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [
    {\"role\": \"user\", \"content\": \"Take a screenshot and describe what you see.\"},
    {
      \"role\": \"assistant\",
      \"content\": null,
      \"tool_calls\": [
        {
          \"id\": \"call_screenshot_001\",
          \"type\": \"function\",
          \"function\": {\"name\": \"take_screenshot\", \"arguments\": \"{}\"}
        }
      ]
    },
    {
      \"role\": \"tool\",
      \"tool_call_id\": \"call_screenshot_001\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"Screenshot taken successfully:\"},
        {
          \"type\": \"image_url\",
          \"image_url\": {\"url\": \"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==\"}
        }
      ]
    }
  ],
  \"max_completion_tokens\": 128,
  \"temperature\": 0,
  \"seed\": 0
}"
```

**Expected Observable Outcome**:
- HTTP 200 OK.
- The Engine decodes the base64 PNG in the tool message turn, preprocesses image patches via the Vision pipeline, and generates a description of the image content.
- `finish_reason` is `"stop"`.
- Token usage reports both text tokens and vision patch tokens in `prompt_tokens`.

### 2. Live tolerant tool-call recovery

When `--tolerant-tool-calls` is enabled on the server, the parser recovers complete tool calls even when the model emits duplicate closing tags or trailing suffixes (e.g. duplicate `</function>`, `</function_invocation>`, or trailing explanatory text after a complete function), or when the model omits the outer `</tool_call>` closing tag.

```bash
curl -sS "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [
    {\"role\": \"user\", \"content\": \"Search for weather in Tokyo using the get_weather tool.\"}
  ],
  \"tools\": [
    {
      \"type\": \"function\",
      \"function\": {
        \"name\": \"get_weather\",
        \"description\": \"Get current weather for a city\",
        \"parameters\": {
          \"type\": \"object\",
          \"properties\": {
            \"city\": {\"type\": \"string\"}
          },
          \"required\": [\"city\"]
        }
      }
    }
  ],
  \"tool_choice\": \"auto\",
  \"max_completion_tokens\": 256,
  \"temperature\": 0
}"
```

**Expected Observable Outcome**:
- HTTP 200 OK.
- If the model generation produces a valid `<tool_call>` block with duplicate closing suffixes (e.g. `</function_invocation>`) or an unclosed `</tool_call>`:
  - `choices[0].finish_reason` is `"tool_calls"`.
  - `choices[0].message.tool_calls` contains the parsed function call (`get_weather` with `{"city":"Tokyo"}`).
  - `choices[0].message.content` contains any text prefix before the `<tool_call>` tag (or null/empty if none).
- If the output contains near-miss tag syntax (e.g. `<function name="...">` or `<call>`) or is cut off mid-parameter by token limits:
  - The turn gracefully degrades to a plain text response with `finish_reason` `"stop"` or `"length"`.
  - No internal 500 errors occur, and no phantom tool calls with empty/corrupted arguments are fabricated.

---

# GPU runbook: decode micro-opts A/B (#69 + #67)

This round is implementation and compile only. Do not boot a server or acquire the
GPU lock until the coordinator schedules an exclusive window. The Qwen3.8-27B
NVFP4 artifact is about 20 GiB and does not fit the 16 GiB compact cap.

## Arms

| Arm | Git | Binary |
|---|---|---|
| A baseline | `9dda66511c81e72686ba6b610256625a8af603a7` (`feat/prefix-seed-store` without the thinking dialect) | rebuild `ninfer-serve` from that commit |
| B treatment | `task/issue-1-decode-micro-opts` HEAD | rebuild `ninfer-serve` from this branch |

Do not base either arm on the vLLM-dialect merge. Decode A/B must stay pure.

Rebuild image `ninfer:seedstore` from the arm under test. Coordinator owns the
rebuild and the `:8018` lane. Never stop production containers
(`sglang-qwen38` on `:8016`, embeddings, whisper).

## Server flags (identical both arms)

No `--preserve-thinking`. Model ID `qwen3.8-27b`. JSONL log required.

```text
--host 127.0.0.1 --port 8018
--max-context 131072 --kv-capacity 1048576 --max-concurrency 8
--spec mtp --draft-tokens 5 --lm-head-draft
--kv-dtype int8 --prefill-chunk 2048 --vision --cors
--prefix-cache-mib 4096
--request-log-jsonl /tmp/ninfer-decode-ab.jsonl
```

## A/B protocol (binding)

- Identical config both arms.
- At least **3 full process boots per arm** (6 boots total). Interleave
  A/B/A/B/A/B in one exclusive GPU session so thermal/clock drift is visible.
- Client load: **c=1, n=16**. One in-flight request. Sixteen serial Chat
  Completions per boot.
- Prompts: production-shaped chat from `examples/cli/messages/scenario_*.json`
  (code, story, translation, structured). Cycle the twelve scenario fixtures
  and repeat the first four to make sixteen. `max_completion_tokens=256`.
  Do not send `enable_thinking` / `chat_template_kwargs` (this arm predates
  the dialect).
- Speculative: MTP-5 as in the server flags above. Do **not** pass `--greedy`
  on the A/B throughput boots (MTP acceptance luck is why boot variance is
  large).
- Metric: per-request decode tok/s from the process log line
  `decode=<rate>` which is `(completion_tokens - 1) / timings_seconds.decode`.
  Boot score = median of the 16 request rates. Arm score = median of the 3
  boot scores. Also record all three boot scores so spread is visible.
- **Noise floor:** boot-to-boot decode on this card has been 133.5–155.9 tok/s
  from stochastic MTP acceptance. A **single-boot** delta under about 10% is
  noise. Do not accept or reject on one boot.
- Pass: arm B median is not a regression versus arm A after three boots.
  The upstream claims (+3.7% MoE L2 prefetch, +2.3% node removal) were
  measured on an RTX 5090 at T=1; they are unverified on the PRO 6000.
  Adopt only if the 3-boot median does not regress. Drop the pick if it
  regresses.

### Throughput commands

```bash
BASE=http://127.0.0.1:8018
MODEL=qwen3.8-27b
PROMPTS=(
  examples/cli/messages/scenario_code_cuda.json
  examples/cli/messages/scenario_code_python.json
  examples/cli/messages/scenario_code_typescript.json
  examples/cli/messages/scenario_story_zh_scifi.json
  examples/cli/messages/scenario_story_en_mystery.json
  examples/cli/messages/scenario_story_zh_dialogue.json
  examples/cli/messages/scenario_translation_zh_en.json
  examples/cli/messages/scenario_translation_en_zh.json
  examples/cli/messages/scenario_translation_markdown.json
  examples/cli/messages/scenario_structured_jsonl.json
  examples/cli/messages/scenario_structured_csv.json
  examples/cli/messages/scenario_structured_sql.json
)
# Repeat first four to reach n=16.
for i in $(seq 0 15); do
  msg="${PROMPTS[$((i % 12))]}"
  python3 - "$BASE" "$MODEL" "$msg" <<'PY'
import json, sys, urllib.request
base, model, path = sys.argv[1], sys.argv[2], sys.argv[3]
body = {
  "model": model,
  "messages": json.load(open(path, encoding="utf-8")),
  "max_completion_tokens": 256,
  "stream": False,
}
req = urllib.request.Request(
    base + "/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=600) as r:
    json.load(r)
PY
done
```

Parse `/tmp/ninfer-decode-ab.jsonl` events with `"event":"request_done"`:

```text
decode_tok_s = (result.completion_tokens - 1) / timings_seconds.decode
```

## Greedy bit-identity: cold vs seeded

Run **once per arm**, not as part of the 3-boot throughput. Use `--greedy`
in addition to the flags above (same prefix-cache). Cold-start the process.

Prompt: a two-message chat so the seed frontier is the first user turn.
Send the same body twice.
Pass only if all of:

1. Both HTTP 200.
2. `choices[0].message.content` is byte-identical across the two responses.
3. First `request_done.prefix_reuse_path` is `full_reset`.
4. Second `request_done.prefix_reuse_path` is `seed_prefix` (or a restore_*
   path). `full_reset` on the second request means the seed-store oracle
   failed — fail the pick even if content matches by chance.
5. Repeat the same pair on arm A and arm B. Content must match **across
   arms** as well (prefetch and node folding must not change tokens).

If (2) or (5) fails, drop the pick. If (4) fails, the seed-store contract
regressed and the pick is not adoptable.

### Long greedy generation

A 32-token probe cannot catch the 1-ulp class recorded in `src/ops/kernel/rope.cuh`:
in-kernel rotary drift that "surfaces as a diverged token deep inside long greedy
generations." Add one long greedy request per arm, same cold-start process as
the short probe (or a third request after it).

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Write a detailed technical explanation of speculative decoding with MTP, including a worked numeric example."}],
  "max_completion_tokens": 2048,
  "temperature": 0,
  "seed": 0
}
```

Send it twice (cold, then seeded) on arm A and arm B. Pass only if:

1. All four HTTP 200, `finish_reason` is `stop` or `length`.
2. `choices[0].message.content` is byte-identical across the four responses
   (both arms, cold and seeded).
3. Seeded `prefix_reuse_path` is not `full_reset`.

A mismatch anywhere in the 2048-token body fails the pick even if the 32-token
probe passed.

## GPU lock

Only the coordinator runs this. Before any process that allocates GPU
memory: `nvidia-smi` shows at least 20 GiB free; `mkdir`
`C:\Users\igorl\.ninfer-gpu.lock` (retry 60 s up to 30 min); remove the
lock directory after, success or failure. Never stop or restart the
production docker stack.
