# Runtime and Serving Behavior

> Status: current user-visible generation and protocol behavior.
>
> Executable `--help` output is authoritative for option spelling and defaults. This document
> describes semantics shared by the CLI and HTTP service.

## 1. Frontends

The project exposes two generation frontends over the same tokenizer, processor, Engine, sampler,
and token-stream decoder:

- `ninfer` accepts one text prompt or one structured messages file and streams decoded text to stdout;
- `ninfer-serve` keeps one Engine resident and exposes OpenAI- and Anthropic-compatible HTTP endpoints.

Both use the tokenizer and generation configuration embedded in the q5090 artifact. Neither accepts
a separate runtime tokenizer directory.

## 2. CLI

Text input:

```bash
./build/src/ninfer MODEL.qus --prompt "你好" --max-new 128
```

Multimodal input:

```bash
./build/src/ninfer MODEL.qus --messages messages.json --no-thinking --max-new 256
```

The CLI requires exactly one of `--prompt` or `--messages`. It writes generated text to stdout and
progress/timing information to stderr. Clean output is the default; diagnostic modes can emit raw
template output or token ids.

Relevant execution choices include:

- `--max-context N` and `--prefill-chunk N`;
- `--kv-dtype bf16|int8`;
- `--mtp-draft-tokens 0..5` and `--lm-head-draft`;
- `--no-cuda-graph`;
- `--no-thinking`;
- sampler fields or `--greedy`.

The CLI defaults to a 2,048-token context and 128 generated tokens. Prefill chunk defaults to 1,024
and must be a nonzero multiple of 128.

## 3. Server

```bash
./build/src/ninfer-serve MODEL.qus --host 127.0.0.1 --port 8080
```

The server exposes:

| Method and path | Behavior |
|---|---|
| `GET /health` | process health |
| `GET /v1/models` | configured model listing |
| `GET /v1/models/{id}` | configured model lookup |
| `POST /v1/chat/completions` | OpenAI-style generation |
| `POST /v1/messages` | Anthropic-style generation |
| `POST /v1/messages/count_tokens` | expanded prompt token count |

The server defaults to `127.0.0.1:8080`, model id `qwen3.6-27b`, and an 8,192-token context. If a
request omits its output limit, the default is `min(max_context / 2, 8192)`, floored at one token.
`--default-max-tokens` overrides that derived value.

An explicit or default output limit is an upper bound, not a reservation. After rendering and
tokenizing the prompt, the server reduces the effective output limit to the remaining KV capacity
when necessary. A prompt that itself exceeds `max_context` is rejected; an otherwise valid request
continues with the reduced limit and reports a length stop if it exhausts that capacity
(`finish_reason: "length"` for OpenAI and `stop_reason: "max_tokens"` for Anthropic). The request
start log records both requested and effective limits when this context clamp occurs.

An API key is optional. When configured, requests must authenticate with the supported bearer-key
surface. Permissive browser CORS is opt-in. The request body limit defaults to 384 MiB and is checked
before JSON parsing; `--max-request-mib` can lower it.

One Engine owns one mutable sequence state. Generation runs under the Engine lock, so request
threads and streaming workers do not interleave GPU inference. Memory-heavy media preprocessing is
also admitted one request at a time.

## 4. OpenAI Chat Completions

`POST /v1/chat/completions` supports:

- system, developer, user, assistant, and tool history after translation to the Qwen template;
- string content or ordered content-part arrays;
- `image_url` and `video_url` content parts;
- non-streaming responses and SSE streaming;
- `max_completion_tokens` or `max_tokens`;
- `stop` as a string or string array;
- `temperature`, `top_p`, `top_k`, presence/frequency penalties, and `seed`;
- `stream_options.include_usage`;
- function tools and assistant `tool_calls` history;
- the project extension controlling thinking mode.

OpenAI image/video URLs may be direct URL strings or objects containing `url`. `data:` URLs are
decoded locally. Unsupported content kinds such as input audio are rejected rather than silently
dropped.

Non-streaming thinking output is returned as `message.reasoning_content`, while the answer remains
in `message.content`. Streaming uses separate `reasoning_content` and `content` deltas, followed by a
final finish-reason chunk and `[DONE]`.

## 5. Anthropic Messages

`POST /v1/messages` supports:

- top-level `system` text or text-block arrays;
- user and assistant text content;
- user image blocks with base64 or URL sources;
- assistant `thinking`, `text`, and `tool_use` blocks;
- user `tool_result` blocks, including text and image result content;
- `thinking.type` values such as `disabled`, `enabled`, and `adaptive`;
- `max_tokens`, `stop_sequences`, `temperature`, `top_p`, and `top_k`;
- non-streaming responses and Anthropic-style SSE events.

System-role messages found inside the message array are folded into the leading system text. This
supports clients such as Claude Code that inject system reminders between turns even though the Qwen
chat template only honors a leading system section.

Non-streaming responses contain, in order, an optional thinking block, optional text block, and tool
use blocks. Streaming emits the corresponding content-block start/delta/stop events and finishes
with message-delta/message-stop events. Tool output sets the Anthropic stop reason to `tool_use`.

`POST /v1/messages/count_tokens` runs the same schema translation, chat template, media-placeholder
expansion, and tokenizer path used by generation, then returns the expanded input-token count.

## 6. Thinking and output channels

Thinking mode is enabled by default to match the Qwen3.6 chat template. `--no-thinking` disables it
for the CLI or sets the server default off; a supported request field may override the server
default.

The token stream decoder separates an initial `<think>...</think>` channel from answer text. Stop
strings apply to the answer channel, not hidden reasoning. Incremental decoding holds back enough
bytes to match a stop string spanning token boundaries, and the matched stop text is not emitted.

Default stop-token ids come from the artifact's embedded `generation_config.json`. The CLI may add
explicit stop ids; protocol requests may add decoded stop strings.

## 7. Sampling

The default sampler follows the Qwen thinking recommendation:

| Field | Default |
|---|---:|
| temperature | 0.6 |
| top-p | 0.95 |
| top-k | 20 |
| presence penalty | 1.0 |
| frequency penalty | 0.0 |

The sampler performs penalties, temperature scaling, top-k, top-p, optional internal min-p, and a
counter-based random draw. The random counter is keyed by seed, absolute position, and sampling
purpose so target and draft sites have deterministic independent streams for a fixed request.

The implementation keeps at most 20 candidates. `top_k` values in `1..20` are honored; `top_k <= 0`
or `top_k > 20` selects 20. Top-p is then applied inside that candidate set. The current CLI and HTTP
schemas do not expose min-p, so it remains disabled there.

OpenAI requests may set all documented sampler fields. Anthropic requests may set temperature,
top-p, and top-k; that API has no presence/frequency penalty or seed fields. Omitted fields inherit
the server defaults. If neither request nor server pins a seed, the server generates a fresh one per
request.

## 8. Greedy mode

`temperature <= 0` selects an exact argmax bypass with lowest-index tie breaking. It does not pass
through the truncated sampling distribution.

- CLI `--greedy` forces this path;
- server `--greedy` overrides request sampler fields;
- an OpenAI request with `temperature: 0` selects the same path.

Use greedy mode for deterministic numerical diagnostics. It is not the recommended default for
normal thinking responses because long greedy thinking can enter repetitive loops.

## 9. MTP and sampling correctness

MTP is disabled unless `--mtp-draft-tokens N` is nonzero. The maximum is five. Draft preparation,
target verification, accept/commit, and fallback all use the resolved request sampler.

Greedy verification accepts a proposal prefix while each draft equals the target argmax. Sampled
verification uses proposal/target distributions and rejection correction so speculative execution
preserves the target distribution. The full target `lm_head` is always used for target decisions.

`--lm-head-draft` replaces only proposal-site projection with the embedded shortlisted Q4 head. It
can alter proposals and acceptance rate, but not which distribution the target model emits.

Both one-token decode and complete MTP rounds use CUDA Graphs by default. Sampling configuration is
stored at a stable device address, so per-request values do not invalidate graph capture.

## 10. Multimodal input

Structured messages preserve text/image/video part order. The native processor supports local files,
allowed remote URLs, and data URLs where the protocol schema provides them. Remote access rejects
private-network targets by default and applies connect, total-time, and redirect limits.

Default processor budgets include:

- at most 16 media items;
- at most 256 MiB of fetched media data;
- approximately 1M pixels per image and 4M sampled pixels per video;
- at most 768 sampled video frames and 600 seconds source duration;
- at most 131,072 raw patches, 32,768 merged Vision tokens, and a bounded attention-work estimate;
- at most 32,768 expanded prompt tokens before the Engine's configured context check.

These are preprocessing safety limits, not a promise that every maximum-sized combination fits GPU
memory. The configured Engine context and available device memory remain final constraints.

Vision executes once during prefill. Generated tokens use the saved MRoPE offset and no longer touch
the Vision tower.

## 11. Function tool calling

The server implements best-effort function calling by rendering tools into the Qwen chat template
and parsing generated `<tool_call>` blocks. The client is responsible for executing tools and sending
results in a later request.

Supported OpenAI tool behavior includes:

- `tools` entries with `type: "function"`;
- `tool_choice` absent, `none`, `auto`, `required`, or a named function choice;
- assistant messages containing `tool_calls`;
- tool-result messages with role `tool`;
- non-streaming `message.tool_calls`;
- streaming `delta.tool_calls`.

Supported Anthropic behavior maps function definitions to `tools`, assistant calls to `tool_use`,
and results to `tool_result`.

The following are not supported:

- strict JSON Schema or constrained decoding;
- a guarantee that `required` or named choice produces a valid tool call;
- deprecated OpenAI `functions` / `function_call`;
- custom/built-in tool types, namespaces, or allowed-tool subsets;
- server-side tool execution.

Malformed or incomplete generated tool blocks may remain ordinary text. This is an intentional
best-effort surface, not a strict tool protocol.

## 12. Prefix reuse and request isolation

For compatible text conversations, the service retains a logical token mirror and can reuse an exact
prefix or a saved assistant-content boundary. Thinking-stripped follow-up turns can therefore avoid
recomputing the stable prefix when the rendered tokens still match.

Prefix reuse falls back to full prefill when identity, boundary, context, state, or modality makes
reuse unsafe. Multimodal resident state is not reused by the text-prefix mechanism. Every request
refreshes sampler configuration and token-count penalties before generation.

## 13. Errors and limits

Schema validation failures return protocol-shaped client errors. Generation is rejected when the
expanded prompt plus requested output budget exceeds configured context. Media budget, remote
fetch, decode, and timeout failures are translated to request errors rather than CUDA failures.

Client disconnect during streaming cancels the sink and stops emitting; it is represented as a
normal stopped stream rather than corrupting the resident Engine. The next request still passes
through the normal prefix-safety checks.
