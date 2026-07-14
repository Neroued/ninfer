# NInfer CLI and HTTP Serving

> Status: current implemented CLI and HTTP behavior.
>
> Authority: this document defines the two product applications, their command-line options, the
> OpenAI- and Anthropic-compatible HTTP surfaces, and the mapping from those surfaces to the public
> `ninfer::Engine` API. It does not define the `.ninfer` container, model mathematics, target-private
> execution, or conversion.
>
> Executable `--help` output remains authoritative for exact option spelling.

## 1. Product boundary

NInfer exposes two applications over the same public engine:

- `apps/cli` builds the local one-shot `ninfer` CLI;
- `apps/serve` builds `ninfer-serve`, which keeps one engine resident and exposes HTTP endpoints.

With the default CMake layout, the corresponding binaries are `build/apps/ninfer` and
`build/apps/ninfer-serve`. Both take one `.ninfer` artifact as their first positional argument;
there are no separate model or tokenizer path arguments.

The product-to-engine route is deliberately small:

```text
CLI text/messages or HTTP request
    -> product parsing and media acquisition
    -> owning PromptInput
    -> Engine::prepare(...)
    -> Engine::generate(..., OutputSink, CancellationView)

Anthropic count_tokens request
    -> product parsing and media acquisition
    -> owning PromptInput
    -> Engine::count_tokens(...)
```

The applications own command-line and protocol schemas, model aliases, media source acquisition,
HTTP streaming, usage objects, and protocol error shapes. `Engine` owns artifact dispatch, the
registered checkpoint frontend, prompt rendering and tokenization, media preprocessing, context
capacity, prefix reuse, target execution, MTP, stop handling, reasoning/content decoding, and final
generation accounting.

`prepare` returns an owning `PreparedPrompt` and rejects a prompt that already exceeds the configured
engine context. `generate` consumes that prompt and a `RequestOptions`; it fits the requested output
budget to the remaining target capacity. `count_tokens` follows the registered checkpoint frontend,
including chat-template and media-token expansion, but does not start GPU generation or apply the
generation context limit.

One `Engine` has one resident sequence program. Calls to `generate` are serialized inside the
engine; the current HTTP transport can serve streaming connections, but it does not continuously
batch their model execution.

## 2. Local CLI

Text input:

```bash
./build/apps/ninfer model.ninfer --prompt "你好" --max-new 128
```

Structured chat and multimodal input:

```bash
./build/apps/ninfer model.ninfer \
  --messages messages.json \
  --max-context 8192 \
  --max-new 256
```

Exactly one of `--prompt` and `--messages` is required. A messages file may be either a nonempty
message array or an object containing `messages` and an optional `tools` array. Supported roles are
`system`, `developer`, `user`, `assistant`, and `tool`. Ordered content parts may contain text,
images, and videos; media sources may be local paths, HTTP(S) URLs, or base64 data URIs.

Answer content is streamed to stdout. Reasoning, loading progress, timings, throughput, memory
statistics, and MTP statistics go to stderr. This keeps stdout usable as generated output.

### 2.1 CLI options

| Option | Meaning | Default |
|---|---|---:|
| `--prompt TEXT` | one plain user prompt | mutually exclusive with `--messages` |
| `--messages FILE` | structured messages JSON | mutually exclusive with `--prompt` |
| `--max-context N` | engine context capacity | `2048` |
| `--prefill-chunk N` | text-prefill chunk, a positive multiple of 128 | `1024` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--kv-dtype bf16\|int8` | KV-cache storage | `bf16` |
| `--mtp-draft-tokens N` | MTP draft window, `0..5` | `0` |
| `--lm-head-draft` | use the optimized proposal head; requires MTP | off |
| `--no-cuda-graph` | disable CUDA Graph execution | graphs on |
| `--no-thinking` | disable thinking in prompt rendering | thinking on |
| `--temperature F` | sampling temperature in `[0,2]` | `0.6` |
| `--top-p F` | nucleus threshold in `[0,1]` | `0.95` |
| `--top-k N` | nonnegative top-k value | `20` |
| `--min-p F` | min-p threshold in `[0,1]` | `0` |
| `--presence-penalty F` | presence penalty in `[-2,2]` | `1.0` |
| `--frequency-penalty F` | frequency penalty in `[-2,2]` | `0` |
| `--seed N` | nonnegative 64-bit sampling seed | `0` |
| `--greedy` | replace sampling options with exact argmax | off |
| `--stop-token-id N` | add a stop token; repeatable | none |
| `--stop TEXT` | add a content-channel stop string; repeatable | none |
| `--reasoning-stop TEXT` | add a reasoning-channel stop string; repeatable | none |
| `--raw-output` | return the frontend's raw output stream | off |
| `--print-token-ids` | print generated token ids in diagnostics | off |

Artifact-defined default stop tokens remain active when CLI stop conditions are added. A matched
stop string is withheld from output.

## 3. HTTP server

```bash
./build/apps/ninfer-serve model.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --model-id qwen3.6-27b
```

The server loads and warms one `Engine`, then exposes:

| Method and path | Behavior |
|---|---|
| `GET /health` | process health |
| `GET /v1/models` | configured OpenAI model alias |
| `GET /v1/models/{id}` | lookup of that alias |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |

### 3.1 Server options

| Option | Meaning | Default |
|---|---|---:|
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | require the configured bearer or `x-api-key` value | unset |
| `--model-id ID` | alias reported by the OpenAI model endpoints | `qwen3.6-27b` |
| `--max-context N` | engine context capacity | `8192` |
| `--prefill-chunk N` | text-prefill chunk, a positive multiple of 128 | `1024` |
| `--device N` | CUDA device index | `0` |
| `--max-request-mib N` | maximum HTTP body before JSON parsing | `384` |
| `--kv-dtype bf16\|int8` | KV-cache storage | `bf16` |
| `--mtp-draft-tokens N` | MTP draft window, accepted target range `0..5` | `0` |
| `--lm-head-draft` | use the optimized proposal head; requires MTP | off |
| `--default-max-tokens N` | output limit when a request omits one | `8192` |
| `--no-cuda-graph` | disable CUDA Graph execution | graphs on |
| `--no-thinking` | make thinking disabled by default | thinking on |
| `--cors` | emit permissive browser CORS headers | off |
| `--temperature F` | default temperature in `[0,2]` | `0.6` |
| `--top-p F` | default top-p in `[0,1]` | `0.95` |
| `--top-k N` | default nonnegative top-k | `20` |
| `--presence-penalty F` | default presence penalty in `[-2,2]` | `1.0` |
| `--frequency-penalty F` | default frequency penalty in `[-2,2]` | `0` |
| `--seed N` | default nonnegative 64-bit seed | fresh seed per request when unset |
| `--greedy` | force exact argmax for every request | off |

`--default-max-tokens` is independent of `--max-context`: it is the protocol fallback, while
`Engine::generate` computes the effective output budget from the prepared prompt and actual context
capacity. A request can therefore finish earlier with a capacity stop.

## 4. OpenAI Chat Completions

`POST /v1/chat/completions` requires the request `model` to equal `--model-id`; a different alias
returns `model_not_found`. The endpoint supports:

- `system`, `developer`, `user`, `assistant`, and `tool` history;
- string content and ordered content-part arrays;
- `image_url` and `video_url` parts using HTTP(S) or data URLs;
- `max_completion_tokens` or the legacy `max_tokens` spelling;
- `stop` as one string or an array of strings;
- `temperature`, `top_p`, `top_k`, presence/frequency penalties, and a nonnegative `seed`;
- `n: 1`, text response format, non-streaming responses, and SSE streaming;
- `stream_options.include_usage`;
- function tools, `tool_choice`, assistant `tool_calls`, and tool-result messages;
- the `enable_thinking` extension.

Non-streaming responses always include prompt, completion, and total-token usage. Reasoning is
returned separately as `message.reasoning_content`; answer text remains in `message.content`.

A stream begins with the assistant-role chunk. It then emits separate `reasoning_content` and
`content` deltas, a final finish-reason chunk, and `[DONE]`. When
`stream_options.include_usage` is true, ordinary chunks carry `usage: null` and a final chunk with
an empty `choices` array carries the completed usage counts.

Output-limit and context-capacity finishes map to `finish_reason: "length"`; model/default stop
tokens and request stop strings map to `"stop"`. A parsed function call maps to `"tool_calls"`.

## 5. Anthropic Messages

`POST /v1/messages` requires a nonempty `model` string but treats it as a protocol label: the
server echoes it instead of using it to select the loaded target. The endpoint supports:

- top-level `system` text or text-block arrays;
- `user`, `assistant`, and in-array `system` messages;
- user text and image blocks with base64 or HTTP(S) URL sources;
- assistant `thinking`, `text`, and `tool_use` history;
- `tool_result` blocks containing text or images;
- `thinking.type`, where `disabled` turns thinking off and other values turn it on;
- `max_tokens`, `stop_sequences`, `temperature`, `top_p`, and `top_k`;
- client-defined tools and Anthropic tool choices;
- non-streaming responses and Anthropic SSE events.

System-role messages inside the message array are folded into the leading system text. Generated
reasoning, answer text, and parsed tools become `thinking`, `text`, and `tool_use` content blocks in
that order.

Streaming uses `message_start`, content-block start/delta/stop, `message_delta`, and
`message_stop` events. Input usage is reported in `message_start`; the final `message_delta` reports
output usage. Output-limit and context-capacity finishes map to `stop_reason: "max_tokens"`, ordinary
stops map to `"end_turn"`, and parsed tools map to `"tool_use"`.

`POST /v1/messages/count_tokens` runs the same protocol translation and registered checkpoint
frontend as generation. Its `input_tokens` therefore includes chat-template tokens and expanded
media tokens. It does not execute the model and does not reject a count merely because it is larger
than the configured generation context.

## 6. Sampling, stopping, and MTP

The shared server defaults are temperature `0.6`, top-p `0.95`, top-k `20`, presence penalty `1.0`,
and frequency penalty `0`. Omitted request fields inherit these values. OpenAI may also provide a
seed; Anthropic has no seed or penalty fields, so those retain server defaults. If neither the
request nor server specifies a seed, the server chooses a fresh seed for that request.

Temperature zero selects exact argmax. CLI `--greedy` and server `--greedy` select the same path;
the server form overrides request sampling fields. Min-p is exposed only by the CLI.

MTP is disabled at draft window zero and enabled by `--mtp-draft-tokens 1..5`. It is an engine
execution choice, not a protocol feature: output channels, stops, usage, and finish reasons remain
the same. `--lm-head-draft` changes only the proposal head and requires a nonzero draft window.

The frontend contributes model-default stop tokens. Callers may add stop token ids and channel-aware
stop strings through `RequestOptions`; the HTTP adapters add their `stop` or `stop_sequences` values
as content-channel strings. The engine performs incremental matching across token boundaries and
does not publish the matched string.

## 7. Media ownership

Paths and URLs are product inputs, not engine inputs. Before calling `Engine::prepare` or
`Engine::count_tokens`, the CLI or server resolves every media source into an owning `OwnedMedia`
byte buffer. The resulting `PromptInput` has no dependency on a temporary download, request-body
view, or caller-owned memory.

The registered checkpoint frontend decodes and preprocesses those bytes, places image/video tokens
in message order, and computes the expanded prompt length. Generation performs Vision work during
prompt processing and then continues text decoding from the target's prepared state.

The CLI accepts files, HTTP(S), and data URIs for images and videos. OpenAI accepts HTTP(S) and data
URLs for image/video parts. Anthropic accepts base64 and HTTP(S) image sources. Unsupported content
types are rejected rather than silently removed.

## 8. Function tools

The HTTP server renders client tool definitions into the checkpoint chat template and parses
generated tool-call blocks after generation. It never executes a tool. Clients execute returned
calls and send tool results in a later request.

OpenAI supports function tools and `tool_choice` values `none`, `auto`, `required`, or a named
function. Anthropic supports client tools with an `input_schema` and choices `none`, `auto`, `any`,
or a named tool. Both protocols preserve assistant call history and tool-result history.

Tool generation is best effort, not constrained decoding. Strict JSON Schema enforcement,
deprecated OpenAI `functions`/`function_call`, built-in server tools, and server-side tool execution
are not implemented. While a tool-capable request is streaming, answer-channel output is buffered
until the final parser decides whether it is ordinary text or tool-call syntax; reasoning may still
stream immediately.

## 9. Errors, cancellation, and accounting

Protocol-schema and sampling validation failures are client errors. A prepared prompt beyond the
engine context is returned as `context_length_exceeded` before an HTTP stream starts. An output
request larger than the remaining context is accepted and ends at capacity with the protocol finish
reason described above. Disconnecting an SSE client requests cancellation at the engine's supported
execution boundary.

Usage is based on the prepared prompt's expanded token count and the engine's accepted generated
token ids. It is the same accounting source for non-streaming responses, streaming responses, and
server request logs. Stop tokens can therefore count as generated tokens even when their decoded
text is withheld from the published answer.
