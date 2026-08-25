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
