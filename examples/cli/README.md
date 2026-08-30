# Stable CLI examples

This directory contains committed, offline `--messages` inputs for exercising the product CLI from
the repository root. It covers short text, chat history, images, video, mixed multimodal history,
hard thinking problems, long decode, and four long-context lengths. This is an operator-facing
example set, not a second correctness framework.

[`manifest.json`](manifest.json) lists the frozen comparison cases, their intended observations,
recommended runtime budgets, and prepared-prompt token counts. Standalone exploratory prompts may
also live beside them when they deliberately have no frozen output oracle.

## Quick start

Run from the repository root because media paths in the JSON files are repository-relative:

```bash
CLI=./build/apps/ninfer
MODEL=models/qwen3_6_27b.ninfer

$CLI "$MODEL" \
  --messages examples/cli/messages/text_smoke_zh.json \
  --no-thinking --greedy --max-new 8
```

Expected stdout is exactly `42`. `--no-thinking --greedy` is the normal comparison mode for simple
cases. Reasoning, progress, timings, memory, and MTP statistics are written to stderr; answer content
is written to stdout.

## Send the same fixture to Serve

`send_to_serve` submits one CLI messages file to an already-running OpenAI Chat Completions
endpoint. Run it as a repository module so it uses the maintained Serve client. Local CLI image and
video paths are read on the client and encoded as data URIs; the server never receives a local
filesystem path.

```bash
python3 -m examples.cli.send_to_serve \
  examples/cli/messages/image_chart.json \
  --base-url http://127.0.0.1:8080 \
  --model qwen3.6-27b --no-thinking --max-tokens 64
```

Start `ninfer-serve` with `--vision` for media fixtures. Relative media paths use the current
working directory by default; pass `--media-root` when invoking the command elsewhere. `--dry-run`
prints the exact converted request without contacting a server.

## Text and multimodal cases

```bash
$CLI "$MODEL" --messages examples/cli/messages/text_chat_history.json \
  --no-thinking --greedy --max-new 32

$CLI "$MODEL" --messages examples/cli/messages/text_code_review.json \
  --no-thinking --greedy --max-new 256

for CASE in image_chart image_natural video_temporal multi_image_compare \
            mixed_image_video mixed_multiturn; do
  $CLI "$MODEL" --messages "examples/cli/messages/${CASE}.json" \
    --max-context 8192 --no-thinking --greedy --max-new 128 --vision
done
```

The controlled observations are:

| Case | Expected observation |
|---|---|
| `text_chat_history` | `Cedar|2041-09-17|37` |
| `text_code_review` | empty input divides by zero; add an explicit empty-input policy |
| `image_chart` | `NIFER VISION 731`; three red circles; blue square on the left |
| `image_natural` | mailbox `24`; sun on the right |
| `video_temporal` | red circle moves; green `3`; square still visible when `3` appears; ending `9` |
| `multi_image_compare` | two circles, three circles, and a new yellow star |
| `mixed_image_video` | `NIFER-9` |
| `mixed_multiturn` | `24-9` |

To exercise MTP through the same input path:

```bash
$CLI "$MODEL" --messages examples/cli/messages/text_smoke_zh.json \
  --no-thinking --greedy --max-new 8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

## Thinking cases

Do not pass `--no-thinking` for these inputs. Give reasoning enough room to complete and transition
to answer content; the model is free to stop before the requested maximum.

```bash
$CLI "$MODEL" --messages examples/cli/messages/thinking_logic_grid.json \
  --greedy --max-context 16384 --max-new 8192

$CLI "$MODEL" --messages examples/cli/messages/thinking_multimodal_checksum.json \
  --greedy --max-context 8192 --max-new 4096 --vision

$CLI "$MODEL" --messages examples/cli/messages/reasoning_jacobian_counterexample_3d.json \
  --greedy --max-context 32768 --max-new 16384
```

The logic grid has one solution and must end with `CHECK=4606`. The multimodal case reads independent
facts from two images and one video, then must end with `CHECKSUM=2238`. The Jacobian prompt is a
standalone exact-proof stress case without a frozen model-output oracle, so it is intentionally not
part of the comparison manifest.

## Long decode

The three frozen AIME 2026 cases deliberately get generous budgets. They are meant to run until the
model's stop token, not to discover the smallest `max-new` value that happens to fit one output.

```bash
for CASE in 01 15 30; do
  $CLI "$MODEL" --messages "examples/cli/messages/long_decode_aime26_${CASE}.json" \
    --greedy --max-context 262144 --kv-dtype int8 --max-new 65536
done
```

The boxed integer answers for cases 01, 15, and 30 are respectively `277`, `83`, and `393`.

## Long context

These prompt lengths include the chat template with thinking disabled. Each input freezes one long
document with the same retrieval needle placed at 50% depth.

```bash
$CLI "$MODEL" --messages examples/cli/messages/long_niah_8k.json \
  --max-context 262144 --kv-dtype int8 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 128

$CLI "$MODEL" --messages examples/cli/messages/long_niah_64k.json \
  --max-context 262144 --kv-dtype int8 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 128

$CLI "$MODEL" --messages examples/cli/messages/long_niah_128k.json \
  --max-context 262144 --kv-dtype int8 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 128

$CLI "$MODEL" --messages examples/cli/messages/long_niah_256k.json \
  --max-context 262144 --kv-dtype int8 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 128
```

All four must output:

```text
ORCHID=493817; COLOR=COBALT
```

The committed prompt token counts were validated against the Qwen3.6-27B and Qwen3.6-35B-A3B
frontend profiles, which produce identical sequences for these files. Qwen3.8-27B uses the same CLI
surface but carries its own tokenizer and chat-template resources; inspect its prepared token count
when using these fixtures.

## Fixture construction

All PNG and MP4 media are project-authored deterministic scenes. The video is five seconds at 8 FPS
with forty H.264 frames. Runtime tests never depend on a network URL or mutable external content.

The committed files are the canonical inputs; there is no in-tree regeneration script. A deliberate
replacement must update the file, its manifest hash, prompt-token count, and oracle together.
