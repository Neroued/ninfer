# `ninfer` one-shot CLI (from NInfer)

`build/apps/ninfer` runs one request against one v3 `.ninfer` artifact. Build NInfer and
download an artifact using the [project README](../README.md) before following this guide.

The examples use Qwen3.8-27B NVFP4 with FP8 KV storage.

## Text input

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Exactly one of `--prompt` and `--messages` is required. The CLI normally omits `--kv-capacity`, so
the shared Main Text KV pool follows the example's 32,768-token `--max-context`.

Answer content is streamed to stdout. Human-readable startup milestones and runtime errors are
written to stderr without service timestamps. Reasoning and the CLI result report (timings,
throughput, GPU memory, token IDs when requested, and speculative-decoding statistics) also use
stderr as unprefixed product output, so stdout can be redirected independently. On a terminal,
weight materialization is one transient progress line followed by a compact Engine-ready summary.
Redirected stderr contains persistent readable progress for long loads and no carriage returns or
ANSI escapes. `--log-level debug` exposes every startup phase. Option and local prompt/message input
failures remain direct command diagnostics:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Return one sentence." \
  --max-context 4096 \
  --max-new 64 \
  --kv-dtype fp8 \
  > answer.txt 2> run.log
```

`--chat-template FILE` overrides the artifact's built-in template with a local Jinja file.
Changes to the file take effect after restarting NInfer:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --chat-template tools/chat_templates/qwen3_8.jinja --prompt "Hello"
```

Omitted thinking and effort options use the selected template's defaults. `--no-thinking` or
`--reasoning-effort none` requests disabled thinking; other effort values cannot be combined with
`--no-thinking`. The template interprets the selected effort. `--greedy` selects exact argmax
decoding independently.

`--thinking-budget N` places a positive upper bound on accepted model-origin tokens while the
new-turn Qwen thinking block remains open. If the model has not emitted `</think>` at that exact
boundary, Engine appends [Qwen's canonical early-close guidance](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md)
and `</think>` to the same resident sequence without sampling, publishes the guidance through the
reasoning stream, then resumes ordinary generation from the updated context. A natural thinking
close, stop condition, cancellation, or total output/context limit at the boundary takes priority
and suppresses this insertion. The option cannot be combined with `--no-thinking`, but it can be
combined with `--reasoning-effort`.

`--max-new` counts every committed generated token, including internally inserted control tokens.
When the effective output capacity extends beyond the thinking budget, it must have room for the
complete tokenizer-derived control suffix plus one post-close model token; an undersized request is
rejected rather than truncating the suffix. Normal output sends the inserted guidance to stderr as
reasoning. `--print-token-ids` includes the inserted IDs, while `--raw-output` preserves the raw
control representation.

For example, this allows at most 512 model-origin thinking tokens while retaining enough total
output capacity for the inserted suffix and the answer:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain speculative decoding, then give a concise conclusion." \
  --max-context 4096 \
  --max-new 1024 \
  --thinking-budget 512 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

## Startup memory profile

GPU residency is frozen when the Engine starts:

- no `--spec` omits MTP/DFlash2 weights and state and the optimized proposal head;
- `--spec mtp` and `--spec dflash2` (Qwen3.8-27B) load only the selected speculative backend;
- a speculative backend with the full proposal head omits the optimized proposal head;
- Vision is disabled by default, omitting its weights and Vision-specific unified-workspace extent;
- `--vision` loads the weights, expands the one Program workspace for Vision encode/handoff, and
  enables image/video input.
- the one-request CLI uses root-only context mode, so it does not reserve an extra Device
  checkpoint StateImage or capture a continuation that no later request could consume.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: an Engine
started without Vision rejects media and cannot enable Vision later. DFlash2 and Vision may
be enabled together; these backends apply to generated-text decode after multimodal prefill and does not
accelerate Vision encode. The default speculative and Vision settings produce the smallest resident
profile.

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
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --kv-dtype fp8 \
  --vision \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`.
The selected template formats these roles. The maintained Qwen templates keep system/developer
messages at their input positions.

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

Speculative decoding is disabled by default. Select MTP with one to five draft positions, or the
Qwen3.8-27B DFlash2 backend with one to fifteen; DFlash2 may be combined with `--vision`.
`--lm-head-draft` selects the optimized proposal head and requires a selected backend:

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For Qwen3.8-27B artifacts containing the DFlash2 companion weights, select
`--spec dflash2 --draft-tokens 7`, optionally with `--lm-head-draft` and `--vision`.
DFlash2 accepts every draft count from 1 through 15; seven is the checkpoint recommendation.
Both `groupwise-int` and `nvfp4` artifacts use the same Engine route, including CUDA Graph,
concurrent requests, sampling penalties, and prefix reuse. An artifact without the companion
weights reports a missing DFlash2 component when selected. Vision and MTP follow the same
rule: their weights are required only when that component is enabled at startup.

Only one speculative backend can be enabled per Engine. The published [performance results](performance.md)
use MTP with three draft tokens and DFlash2 with seven draft tokens (block length eight), both with
the optimized proposal head. Seven forms the measured block length eight, while fifteen uses the
maximum supported block length sixteen.

## Common options

The table lists executable defaults. The examples above select FP8 KV and MTP3.

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling | `2048` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `2048` |
| `--vram-headroom-mib N` | VRAM in MiB that `--kv-capacity auto` leaves free after sizing the KV pool; requires `auto` | `1024` |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `1024` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--tp 1\|2` | tensor-parallel width; see [Two GPUs](#two-gpus) | `1` |
| `--devices A,B` | one CUDA device per rank, rank 0 first; required with `--tp 2` | `--device` |
| `--kv-dtype bf16\|int8\|fp8\|nvfp4\|k8v4` | KV-cache storage | `bf16` |
| `--spec mtp\|dflash2\|off` | speculative backend; `off` (or `none`) also drops `--draft-tokens` and `--lm-head-draft` given before it, so `--profile NAME --spec off` runs a profile without speculation | off |
| `--draft-tokens N` | MTP `1..5`; DFlash2 `1..15` | unset |
| `--lm-head-draft` | optimized proposal head | off |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--vision-device N` | CUDA device that holds the Vision tower and encodes; equal to `--device` on one GPU, one of `--devices` at `--tp 2` | `--device` |
| `--max-vision-tokens N` | merged Vision tokens of one image or video item (`64..16384`); larger media are resized | `16384` |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--no-tp-mailbox` | keep the captured `--tp 2` all-reduces on cross-device copies; see [Two GPUs](#two-gpus). Without it the mailbox is probed once at startup and dropped by itself when the probe times out or exceeds 50 ms (`NINFER_TP_MAILBOX_PROBE=off` skips the probe, `=fail` forces the fallback); an exchange that hangs in a decode graph's first launch moves the MTP draft phase, then everything, to the copies (`NINFER_TP_MAILBOX_DRAFT=copies` starts with the draft phase there; `NINFER_TP_MAILBOX_FAULT=draft\|any` simulates the hang); `NINFER_TP_MAILBOX_LEGACY=1` runs the mailbox with its original exchange kernel (slower, same results); with `--spec mtp --lm-head-draft` the optimized proposal head is split by vocabulary across the two GPUs, and `NINFER_TP_DRAFT_HEAD=primary` keeps it whole on the first (slower, same results) | mailbox on without P2P, probed at startup |
| `--embedding-host` | keep the token embedding table in one copy in pinned host memory that every GPU reads in place over PCIe, instead of one copy per GPU; frees its VRAM (1.18 GiB per GPU for Qwen3.8-27B), identical results ([serving guide](serving.md#two-gpus)) | off |
| `--chat-template FILE` | use a local Jinja template | artifact template |
| `--no-thinking` | disable thinking | template default |
| `--thinking-budget N` | positive model-origin thinking-token cap; omitted means unlimited | unset |
| `--reasoning-effort none\|minimal\|low\|medium\|high\|xhigh\|max` | pass an effort value to the selected template | template default |
| `--greedy` | exact argmax decoding | off |
| `--temperature F` | sampling temperature override | prompt-mode default |
| `--top-p F` | nucleus-threshold override | prompt-mode default |
| `--top-k N` | top-k-threshold override (`0..20`; zero selects the top-20 cap) | prompt-mode default |
| `--min-p F` | min-p-threshold override | prompt-mode default |
| `--presence-penalty F` | presence-penalty override | prompt-mode default |
| `--frequency-penalty F` | frequency-penalty override | prompt-mode default (`0`) |
| `--seed N` | sampling seed | `0` |

When a sampling flag is omitted, Engine selects the general-task preset of the rendered prompt
mode; the presets are the same for every model:

| Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---:|---:|---:|---:|---:|
| thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |

Frequency penalty is `0` in both presets. Task-specific profiles such as Qwen's
precise-coding profile use explicit sampling overrides.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics.

Run `./build/apps/ninfer --help` for the exact option contract.

## CUDA synchronization

`NINFER_CUDA_SYNC` selects the CUDA device synchronization schedule at startup for both the CLI
and HTTP server. When unset, it defaults to `spin`, prioritizing low synchronization latency at
the cost of CPU usage while waiting for the GPU. Use `blocking` to let the waiting thread sleep;
the decode performance cost depends on the host. `yield` yields the CPU while waiting, and `auto`
uses CUDA's scheduling heuristic, not an automatic performance benchmark.

```bash
NINFER_CUDA_SYNC=blocking ./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer --prompt "Hello"
```

The Engine-ready log reports the selected mode. Empty or unrecognized values, or failure to apply
the schedule, fail startup. This controls device scheduling (including stream synchronization);
it does not override individual CUDA event creation flags.

## Context and memory

The official artifacts have a native context limit of 262,144 tokens. The practical allocation
on one RTX 5090 depends on the selected artifact, media workload, output budget, and KV-cache type.
The artifact describes its model configuration and weight representations;
`--kv-dtype` independently selects runtime KV storage. The prepared prompt must fit
`--max-context`; generation stops at the remaining context capacity when necessary.
`--kv-capacity N` controls the shared physical Main Text KV pool independently and is rounded up to
the 64-token page size. `--kv-capacity auto` loads the selected weights, measures the remaining GPU
memory, and directly chooses the largest legal page capacity for the complete enabled runtime
layout. This includes the selected speculative backend, fixed sequence state, unified workspace,
and CUDA Graph allowance, while leaving the default 1 GiB automatic headroom
unallocated. It does not probe allocations or resize the pool at request time. The single-request
CLI normally leaves the option omitted so it follows
`--max-context`; the distinction matters primarily to a concurrent Engine or server.

At Engine startup NInfer reserves model weights, persistent sequence state, one phase-reused
Program workspace, and a separate CUDA Graph driver allowance. With Vision enabled, that one
workspace contains a general execution prefix and a fixed item-output handoff region. Vision encode
may reuse the full backing before producing the output; Text/MTP/decode work remains inside the
general prefix while the handoff is live. The capacity is therefore the maximum legal simultaneous
extent, not the sum of Text, Vision scratch, and Vision output allocations. Text prefill uses
`min(--prefill-chunk,--max-context)`; Vision keeps the existing 32,768-token aggregate prompt budget
but plans Device execution for the registered 16,384-token maximum single item, or for
`--max-vision-tokens N`: images and videos larger than `N` tokens are then resized to at most `N`
(one token per 32x32 pixels of an image or of two video frames), a video too long for `N` tokens is
rejected, and the encode workspace shrinks with `N`. Requests perform no
project-owned device allocation or growth. Context-cache capacity controls are intentionally absent
from this one-request interface; the persistent Engine and server routes own cross-request reuse and
optional Host backing.

All weight, sequence, workspace, and graph allocations are released when the Engine is destroyed.

## Two GPUs

`--tp 2 --devices A,B` splits an artifact across two GPUs of the same compute capability, for
models whose weights do not fit one device (Qwen3.8-27B NVFP4 on two 16 GB boards; Qwen3.8-Flash-Next
also runs at `--tp 2`, but it is served with `ninfer-serve --profile flash-next`, see
[HTTP serving](serving.md#capy-flash-serving)):

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer --tp 2 --devices 0,1 \
  --max-context 8192 --prompt "What is 17*23?"
```

Rank 0 runs on `A` and owns scheduling and sampling; `--device`, when given, must equal `A`.
Attention heads, Gated DeltaNet heads, the MLP intermediate width and the output-head vocabulary
are halved per rank, and every layer ends in two cross-device all-reduces. Each rank holds half of
the KV cache and recurrent state, and both reserve the same runtime layout; `--kv-capacity auto`
sizes it from the rank with less free memory. Direct peer access is used when the driver grants it;
otherwise the transfers are staged through host memory, which is slower but equivalent. In CUDA
Graph decode, a single request's all-reduces instead exchange through a small pinned host mailbox,
one kernel per GPU, with identical results, when the GPUs have no peer access; `--no-tp-mailbox`
keeps them on the staged copies. [`tools/tp2/mailbox_probe.cu`](../tools/README.md#standalone-tp2-mailbox-probe)
checks the mailbox on a machine without loading a model.

Tensor parallelism covers ordinary decoding, `--spec mtp` and `--spec dflash2` of the dense
architecture with `bf16`, `int8` or `nvfp4` KV. The `nvfp4` cache (v0.4.2) is about 1.8× denser than
`int8`; measured on two RTX 5070 Ti with the QUASAR artifact it leaves ~2 GiB more per board at a
262,144-token context with perplexity and GSM8K unchanged, and costs +4…8 % per decode round with
MTP3 and +4…17 % with DFlash2 (whose acceptance also drops 10-15 points) from 55k to 210k prompt
tokens. Keep `int8` where it fits; `nvfp4` buys KV capacity (`--kv-capacity` above `--max-context`
for concurrent requests) or room for a larger artifact. `--max-context` cannot exceed the artifact's
position capacity with either cache. The MTP head is split like a Text layer and verification
runs on both ranks; `--draft-tokens` and `--lm-head-draft` work as on one GPU. The DFlash2 drafter
is split like the Text layers since v0.4.0 (each rank runs its half of the drafter's heads and MLP,
with the recurrent state mirrored on both); its candidate selector sits on rank 1 (v0.4.1), or on
rank 0 with `NINFER_TP_SELECTOR=primary` or when Vision is on rank 1. It requires `--lm-head-draft`,
since the full output head is split by vocabulary across the ranks; a drafter with full-attention
layers is not supported.
The `fp8` and `k8v4` KV types are rejected at startup. `ninfer-perplexity` takes the same `--tp 2 --devices A,B` ([Perplexity](perplexity.md#two-gpus)).
The split attention and Gated DeltaNet projections take FP8 or NVFP4 weights and the split MLP
FP8 or NVFP4, so both the official mixed artifact (FP8 attention and GDN, NVFP4 MLP) and an
all-NVFP4 recipe run at `--tp 2`; the MTP head splits only in Q8, as the official recipes store it.

`--vision` works at `--tp 2` with each of these modes. The Vision tower and its encode workspace
live on one GPU, `--vision-device` (default `A`), which must be one of `--devices`: it encodes each
image or video item once and copies the merged embeddings to the other GPU, which holds only the
buffer they land in. `--kv-capacity auto` sizes the KV pool from each GPU's own share, so the
tower's GPU pays for it alone; `--max-vision-tokens` shrinks the encode workspace.

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer --tp 2 --devices 0,1 \
  --vision --vision-device 1 --max-context 8192 \
  --messages examples/cli/messages/image_chart.json
```
