# NInfer CLI

`build/apps/ninfer` runs one request against one registered `.ninfer` artifact. Build NInfer and
download an artifact using the [project README](../README.md) before following this guide.

## Text input

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 16384 \
  --max-new 256
```

Exactly one of `--prompt` and `--messages` is required.

Answer content is streamed to stdout. Reasoning, model loading, timings, throughput, GPU memory, and
MTP statistics are written to stderr, so stdout can be redirected independently:

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Return one sentence." --max-new 64 \
  > answer.txt 2> run.log
```

Thinking is enabled by default. Add `--no-thinking` for direct-response prompt rendering or
`--greedy` for exact argmax decoding.

## Startup memory profile

GPU residency is frozen when the Engine starts:

- `--mtp-draft-tokens 0` omits MTP weights, MTP KV/state, and the optimized draft head;
- MTP with the full proposal head omits the optimized draft head;
- Vision is disabled by default, omitting its weights and maximum workspace;
- `--vision` loads those allocations and enables image/video input.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: a
text-only Engine rejects media and cannot enable Vision later. The default MTP and Vision settings
produce the smallest resident profile.

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
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Run message files from the repository root when they contain repository-relative media paths:

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`. Message content
may be a string or an ordered array containing:

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

## MTP

MTP is disabled by default. Enable one to five draft positions with `--mtp-draft-tokens`.
`--lm-head-draft` selects the optimized proposal head and requires MTP:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --mtp-draft-tokens 3 \
  --lm-head-draft
```

The published [performance results](performance.md) use a draft window of three and the optimized
proposal head.

## Common options

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | allocated context capacity | `2048` |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `1024` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--kv-dtype bf16\|int8` | KV-cache storage | `bf16` |
| `--mtp-draft-tokens 0..5` | MTP draft window | `0` |
| `--lm-head-draft` | optimized MTP proposal head | off |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-thinking` | disable thinking in prompt rendering | thinking on |
| `--greedy` | exact argmax decoding | off |
| `--temperature F` | sampling temperature | `0.6` |
| `--top-p F` | nucleus threshold | `0.95` |
| `--top-k N` | top-k threshold | `20` |
| `--min-p F` | min-p threshold | `0` |
| `--presence-penalty F` | presence penalty | `1.0` |
| `--frequency-penalty F` | frequency penalty | `0` |
| `--seed N` | sampling seed | `0` |

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics.

Run `./build/apps/ninfer --help` for the exact option contract.

## Context and memory

Both registered models have a native context limit of 262,144 tokens. The practical allocation on
one RTX 5090 depends on the selected artifact, media workload, output budget, and KV-cache type.
Use `--kv-dtype int8` for large context allocations. The prepared prompt must fit
`--max-context`; generation stops at the remaining context capacity when necessary.

All weight, sequence, workspace, and graph allocations are released when the Engine is destroyed.
