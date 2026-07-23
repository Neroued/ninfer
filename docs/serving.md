# HTTP serving

`build/apps/ninfer-serve` loads one registered artifact and exposes OpenAI- and
Anthropic-compatible HTTP endpoints over one resident NInfer Engine.

## Start the server

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --model-id qwen3.6-27b \
  --max-context 16384 \
  --mtp-draft-tokens 3 \
  --lm-head-draft
```

For the 35B-A3B artifact, set both the artifact path and public model alias:

```bash
./build/apps/ninfer-serve models/qwen3_6_35b_a3b.ninfer \
  --model-id qwen3.6-35b-a3b \
  --max-context 16384 \
  --mtp-draft-tokens 3 \
  --lm-head-draft
```

The default `--model-id` is `qwen3.6-27b`; it is an HTTP alias and does not select the artifact.

For a permanently text-only server, add `--no-vision`. Vision weights and its maximum workspace are
then never allocated, and media requests and token-count requests fail with HTTP 400
`vision_disabled`. MTP residency is likewise fixed by `--mtp-draft-tokens`: `0` omits MTP weights
and state, while `--lm-head-draft` additionally loads the optimized proposal head. A later request
cannot enable a capability omitted at startup.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | process health |
| `GET /v1/models` | configured OpenAI model alias |
| `GET /v1/models/{id}` | lookup of the configured alias |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

The endpoint supports:

- `system`, `developer`, `user`, `assistant`, and `tool` history;
- string content and ordered text, `image_url`, and `video_url` parts;
- `max_completion_tokens` and the legacy `max_tokens` spelling;
- `temperature`, `top_p`, `top_k`, presence/frequency penalties, and a nonnegative `seed`;
- one stop string or an array of stop strings;
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`;
- function tools, tool choices, assistant tool-call history, and tool-result messages;
- the `enable_thinking` extension.

The request `model` must equal `--model-id`. Reasoning is returned separately as
`reasoning_content`; answer text remains in `content`.

Streaming begins with an assistant-role chunk, sends separate reasoning and content deltas, then a
finish-reason chunk and `[DONE]`. When `stream_options.include_usage` is true, a final empty
`choices` chunk contains completed usage.

### Multimodal request

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

OpenAI image and video sources may be HTTP(S) URLs or base64 data URLs.

## Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

The endpoint supports system text, user/assistant history, text and image blocks, thinking blocks,
tool-use history, tool results, client-defined tools, non-streaming responses, and Anthropic SSE
events. `thinking.type: "disabled"` disables thinking; other supported values enable it.

Anthropic's `model` field is treated as a response label and does not select the loaded artifact.

`POST /v1/messages/count_tokens` uses the artifact's tokenizer, chat template, and media expansion
without running GPU generation:

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```

## Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic
`x-api-key` header. `GET /health` and CORS preflight requests remain unauthenticated.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

`--cors` adds permissive browser CORS headers. It is disabled by default.

## Server options

| Option | Meaning | Default |
|---|---|---:|
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--model-id ID` | public OpenAI model alias | `qwen3.6-27b` |
| `--max-context N` | engine context capacity | `8192` |
| `--prefill-chunk N` | text-prefill chunk | `1024` |
| `--device N` | CUDA device index | `0` |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--request-log-jsonl FILE` | append full-precision server/request records | disabled |
| `--kv-dtype bf16\|int8` | KV-cache storage | `bf16` |
| `--mtp-draft-tokens N` | MTP draft window, `0..5` | `0` |
| `--lm-head-draft` | optimized MTP proposal head | off |
| `--default-max-tokens N` | output limit when omitted by a request | `8192` |
| `--no-vision` | permanently disable Vision and omit its GPU allocations | Vision on |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-prefix-reuse` | disable compatible-prefix caching | prefix reuse on |
| `--no-thinking` | disable thinking by default | thinking on |
| `--cors` | permissive browser CORS headers | off |
| `--greedy` | force exact argmax for all requests | off |

Server sampling defaults are temperature `0.6`, top-p `0.95`, top-k `20`, presence penalty
`1.0`, and frequency penalty `0`. Supported request fields override those defaults unless the
server was started with `--greedy`.

Run `./build/apps/ninfer-serve --help` for the exact option contract.

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive model or MTP blocks may share one campaign
file. The parent directory must already exist. Failure to open the file aborts startup; the log path
is also rejected if it resolves to the model artifact.

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --model-id qwen3.6-27b \
  --request-log-jsonl profiles/bench/run/server.requests.jsonl
```

Every line is one `ninfer_serve_request_log` schema-v1 JSON object. All events carry
`timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within
that server instance.

| Event | Contents |
|---|---|
| `server_start` | target/artifact, resolved Engine and sampler configuration, memory summary, CUDA/GPU environment, and redacted argv |
| `request_start` | protocol, resolved sampler and seed, thinking mode, output budget, stream/message/tool shape |
| `request_done` | finish reason, prompt/completion/cache tokens, unrounded phase seconds, and complete MTP counters |
| `request_error` | the resolved request configuration and generation error message |

`request_done.timings_seconds` contains `prepare`, `vision`, `prefill`, `decode`, and `total` as
full-precision JSON numbers. Its `mtp` object contains `enabled`, `draft_window`, `rounds`,
`drafted_tokens`, `accepted_tokens`, `fallback_steps`, and `accepted_per_position`. Rates and TTFT
are intentionally derived downstream from raw token counts and seconds instead of being stored as
rounded strings.

The JSONL file contains no generated response text and never records an API-key value; `argv`
replaces that value with `<redacted>`. The existing stderr summaries remain available for operators
but are rounded and are not the aggregation source. Structured request events cover successfully
prepared OpenAI/Anthropic generation requests and errors during their generation; schema rejection
and token-count-only calls are not measurement requests and do not receive request IDs.

## Execution behavior

The server accepts concurrent HTTP connections, but model execution is serialized because one
Engine owns one resident sequence. It does not perform continuous batching.

Compatible resident prefixes are reused for both text and multimodal histories unless the server is
started with `--no-prefix-reuse`. A multimodal hit requires matching token types, three-axis MRoPE
positions, encoded-media digest, grid, and consumer spans; changing an earlier image or video
therefore resets the prefix instead of reusing placeholder-token KV. Media wholly inside a matched
prefix skips Vision execution, while new suffix media is encoded normally. The completion log
reports the reused token count as `cache=`.

MTP is an engine option and does not change protocol output shapes, stop behavior, or usage
accounting. If a stop truncates a multi-token MTP round, the Engine commits the exact accepted target
prefix so a following compatible turn can still reuse it. Output-limit and context-capacity finishes
map to `length`/ `max_tokens`; ordinary model or string stops map to `stop`/ `end_turn`.

Function tools are rendered into the model prompt and generated calls are parsed into protocol
responses. NInfer does not execute tools and does not enforce client JSON Schema through constrained
decoding.

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
from accepted output token IDs, including a stop token whose decoded text may be withheld.
