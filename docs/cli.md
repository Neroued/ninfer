# NInfer CLI

`ninfer` runs one request through the public Engine, on one device or a configured execution group.
Build and weight download instructions are in the [README](../README.md). Direct-path examples use
Qwen3.8-27B NVFP4 on RTX 5090. For Spark, choose a [deployment configuration](../examples/configs/)
and pass it to the Spark executable:

```bash
./build/dgx-spark-release/apps/ninfer --config ninfer.local.toml \
  --prompt "Explain speculative decoding." --max-tokens 2048
```

On two Sparks, run this once on the coordinator. The configured workers start automatically.
The same model, resource and deployment settings work with `ninfer-serve`.

## Text input

```bash
./build/rtx5090-release/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 32768 \
  --max-tokens 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --proposal-head optimized
```

Exactly one of `--prompt` and `--messages` is required. Context capacity defaults to `auto`;
`--max-context` limits each request, while KV and State share the resolved byte capacity.

Answer content is streamed to stdout. Tool calls supplied through `--messages` also stream to stdout
as `{"name":"...","arguments":...}` records, with argument bytes appearing as they are generated.
An interrupted record retains its partial JSON and reports the interruption on stderr; it is not
repaired into a complete call. Human-readable startup milestones and runtime errors are
written to stderr without service timestamps. Reasoning and the CLI result report (timings,
throughput, GPU memory, token IDs when requested, and speculative-decoding statistics) also use
stderr as unprefixed product output, so stdout can be redirected independently. On a terminal,
weight materialization is one transient progress line followed by a compact Engine-ready summary.
Redirected stderr contains persistent readable progress for long loads and no carriage returns or
ANSI escapes. `--log-level debug` exposes every startup phase. Option and local prompt/message input
failures remain direct command diagnostics:

```bash
./build/rtx5090-release/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Return one sentence." \
  --max-context 4096 \
  --max-tokens 64 \
  --kv-dtype fp8 \
  > answer.txt 2> run.log
```

`--chat-template FILE` overrides the artifact's built-in template with a local Jinja file.
Changes to the file take effect after restarting NInfer:

```bash
./build/rtx5090-release/apps/ninfer models/qwen3_8_27b.ninfer \
  --chat-template tools/chat_templates/qwen3_8.jinja --prompt "Hello"
```

## Constrained output

`--grammar-file FILE` constrains the answer with a GBNF grammar whose entry rule is `root`:

```bash
printf 'root ::= "yes" | "no"\n' > answer.gbnf
./build/rtx5090-release/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Is 17 prime?" --no-thinking --grammar-file answer.gbnf --max-tokens 64
```

`--choice TEXT` selects a literal candidate; repeat the flag to supply the candidate set.
`--regex PATTERN` constrains the complete answer to a regular expression:

```bash
./build/rtx5090-release/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Classify this review: the service was excellent." --no-thinking \
  --choice positive --choice neutral --choice negative

./build/rtx5090-release/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Return ticket TASK-0042." --no-thinking --regex '(BUG|TASK)-[0-9]{4}'
```

Candidates preserve exact text; `--choice ''` explicitly permits empty content. `--regex ''`
permits only empty content. See the [language implementation](../src/text/grammar.cpp)
for supported regex syntax.

`--json-object` constrains the answer to a JSON object. `--json-schema-file FILE` applies a JSON
Schema. All output constraint options are mutually exclusive:

```bash
printf '%s\n' '{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}' > answer.schema.json
./build/rtx5090-release/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Return the answer to 6 times 7 as JSON." --no-thinking \
  --json-schema-file answer.schema.json --max-tokens 64
```

GBNF supports recursive rules, Unicode character classes and repetition. All constraint modes work with
ordinary decoding, MTP, DFlash and DFlash2. Thinking may precede the constrained answer; an output
limit or cancellation can leave it incomplete. JSON modes can accompany tools supplied in messages:
the answer is either JSON or a tool-call sequence. GBNF, choice and regex require no active tools.
Constraints reject custom stops and `--raw-output`. JSON permits whitespace and emits declared
properties in schema order. Describe the desired content in the prompt; the schema is not added
to it automatically. See the [supported schema subset](../src/text/json_schema.cpp).

## Thinking and sampling

Omitted thinking, effort and history-preservation settings use the selected template's defaults.
The maintained templates and GLM frontend have these behaviors:

| Model/template | Thinking | Reasoning effort |
|---|---|---|
| Qwen3.6 Dense/MoE | On by default; can be disabled | No effort-specific prompt in the maintained template |
| Qwen3.8-27B | On by default; can be disabled | `low`, `medium`, `xhigh` (default) |
| Qwen3.8-Flash-Next | On by default; can be disabled | `low`, `medium`, `xhigh` (default); `minimal` maps to low, `high`/`max` to xhigh |
| GLM-5.3-Flash | Always enabled | `low`, `high`, `max`; default comes from the artifact template |

For Qwen, `--no-thinking` or `--reasoning-effort none` disables thinking. GLM rejects that choice.
`--temperature 0` selects greedy sampling independently of thinking. `--preserve-thinking` keeps
closed-turn reasoning when the template supports it; the GLM frontend maps this to `clear_thinking=false`.
Qwen3.8 maintained templates retain it by default; Qwen3.6 omits it by default.
A custom `--chat-template` can change template defaults.

`--thinking-budget N` caps accepted model-origin thinking tokens. At the boundary, a natural thinking
close, stop, cancellation or total output/context limit takes precedence. Otherwise the Engine
commits the model's closing suffix and continues generation. Qwen uses its early-close guidance
plus `</think>`; GLM uses `</think>` without the Qwen guidance. The budget cannot be paired with an
explicitly disabled CLI thinking mode.

`--max-tokens` counts every committed generated token, including internally inserted control tokens.
When the effective output capacity extends beyond the thinking budget, it must have room for the
complete tokenizer-derived control suffix plus one post-close model token; an undersized request is
rejected rather than truncating the suffix. Normal output sends any inserted guidance through the
reasoning stream on stderr. `--print-token-ids` includes the inserted IDs, while `--raw-output`
preserves the raw control representation.

For example, this allows at most 512 model-origin thinking tokens while retaining enough total
output capacity for the inserted suffix and the answer:

```bash
./build/rtx5090-release/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain speculative decoding, then give a concise conclusion." \
  --max-context 4096 \
  --max-tokens 1024 \
  --thinking-budget 512 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --proposal-head optimized
```

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages`
and an optional `tools` array.

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "../media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Local media paths are relative to the messages file, so fixtures work independently of the
current directory:

```bash
./build/rtx5090-release/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-tokens 128 \
  --kv-dtype fp8 \
  --vision \
  --spec mtp --draft-tokens 3 \
  --proposal-head optimized
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
The model frontend formats these roles while preserving message order. Qwen maintains system/developer
positions; GLM renders developer turns as system turns at the same positions.

Message content may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. Assistant
history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and
`tool_call_id`.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking,
long-decode, and long-context inputs.

## Speculative decoding

Select a backend at startup; its weights must be present in the artifact:

| Backend | Models | Draft tokens |
|---|---|---|
| `mtp` | All supported models with MTP weights | 1..5 |
| `dflash` | Qwen3.6-35B-A3B with DFlash companion weights | 1..15 |
| `dflash2` | Qwen3.8-27B with DFlash2 companion weights | 1..15 |

```bash
./build/rtx5090-release/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Explain speculative decoding." --max-context 16384 --max-tokens 2048 \
  --kv-dtype fp8 --spec dflash --draft-tokens 7 --proposal-head optimized
```

`--proposal-head full` uses the full head; `optimized` selects additional conversion-time weights.
Qwen3.5-family optimized heads use a selected vocabulary. Flash Next and GLM use an additional
full-vocabulary Q4 A16 head, keeping the main head unchanged. The artifact must contain the selected
head. Only one backend is active per Engine; `--spec none` disables speculation.

All supported backends can follow multimodal prefill with `--vision`. They accelerate generated-text
decode; Vision encoding and prefill remain eager. `--no-cuda-graph` selects eager decode.
Draft count is a workload choice: more proposals add compute and help only when enough are accepted.

## Common options

The complete field/flag mapping, defaults and override rules are maintained in
[Startup configuration](startup-configuration.md). Both CLI and serve accept `--config FILE` and
share the engine controls. The root [ninfer.example.toml](../ninfer.example.toml) lists every setting.

CLI defaults are 8192 context tokens, 128 output tokens, seed 0, no prefix reuse, automatic Device
Context capacity and zero Host context capacity. Explicit cache and concurrency settings are retained.
Use `--temperature 0` for greedy sampling. `--seed random` chooses a fresh seed for the request;
`model` clears a sampling or template override and `default` restores an entry-specific default
where supported.

When a sampling flag is omitted, Engine selects the general-task preset for the loaded architecture
and rendered prompt mode. The current official models use:

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.6-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.8-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | thinking | `1.0` | `0.95` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |

Frequency penalty is `0` in these Qwen3.5-family presets. Task-specific profiles such as Qwen's
precise-coding profile use explicit sampling overrides.

Flash Next and GLM read their sampling defaults from the artifact's `generation_config.json`.
Explicit sampling flags override these values; GLM retains its thinking-only prompt mode.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics.

Run `./build/rtx5090-release/apps/ninfer --help` for the exact option contract.

## Context and memory

The logical context limit comes from `--max-context`; the request's prompt and generation share that
limit. Device KV and continuation state share a separate Context byte pool sized by
`--context-capacity SIZE|auto`. Read the startup token-capacity estimate for the selected deployment.
The [startup reference](startup-configuration.md#context-and-memory) explains capacity, cache,
Vision/media budgets and environment controls, including `NINFER_CUDA_SYNC`.

Vision and speculative residency are fixed at startup. Omitted components use no corresponding
weights or execution resources. An HTTP request cannot enable them later. CLI defaults disable
cross-request prefix retention and Host context backing; explicit settings in a shared config are
retained. `--vision-memory-budget` bounds the enabled Vision encoder/output resources.

## CPU expert offload

On RTX 5090, Qwen3.6-35B-A3B and Flash Next can keep Main experts in Host memory, cache selected
experts on the GPU and stream weights for larger token batches. `NINFER_VRAM_BUDGET_MIB` can limit
the Engine's GPU allocation budget:

```bash
NINFER_VRAM_BUDGET_MIB=16384 ./build/rtx5090-release/apps/ninfer out/qwen3_6_35b_a3b.ninfer \
  --prompt "Explain how a compiler works." --max-context 8192 --max-tokens 1024
```

Flash Next's conversion recipe needs approximately 63.3 GiB of Host Main-expert backing on this
path, plus other runtime/system memory. Its NF4 PLE rows are read from the artifact on disk.
On one Spark, experts are GPU resident and PLE remains disk-backed; on two Sparks, the PLE table is
sharded across Device memory. These placements are selected by the model deployment.

The default prefill chunk is listed in [startup configuration](startup-configuration.md#model-and-execution).
Use a positive multiple of 128 to choose it explicitly. The Host context-cache budget is separate
from expert memory. See [expert execution](maintainer/expert-weight-cache.md) for the implementation.
