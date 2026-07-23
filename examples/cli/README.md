# Stable CLI examples

This directory contains committed, offline `--messages` inputs for exercising the product CLI from
the repository root. It covers short text, chat history, images, video, mixed multimodal history,
hard thinking problems, long decode, and four long-context capacities. This is an operator-facing
example set, not a second correctness framework.

[`manifest.json`](manifest.json) lists each case, its intended observation, recommended runtime
budget, and its prepared-prompt token count.

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
  --mtp-draft-tokens 3 --lm-head-draft
```

## Thinking cases

Do not pass `--no-thinking` for these inputs. Give reasoning enough room to complete and transition
to answer content; the model is free to stop before the requested maximum.

```bash
$CLI "$MODEL" --messages examples/cli/messages/thinking_logic_grid.json \
  --greedy --max-context 16384 --max-new 8192

$CLI "$MODEL" --messages examples/cli/messages/thinking_multimodal_checksum.json \
  --greedy --max-context 8192 --max-new 4096 --vision
```

The logic grid has one solution and must end with `CHECK=4606`. The multimodal case reads independent
facts from two images and one video, then must end with `CHECKSUM=2238`.

## Long decode

This case deliberately gets a generous budget. It is meant to run until the model's stop token, not
to discover the smallest `max-new` value that happens to fit one output.

```bash
$CLI "$MODEL" --messages examples/cli/messages/long_decode_design_review.json \
  --greedy --max-context 32768 --max-new 16384
```

The answer must contain all eight requested design sections plus `设计自检`. Its memory table must
separate raw KV payload, 6.25% metadata, and total KV. The structural/factual oracle is intentional:
the generated prose is not required to be byte-identical.

## Long context

These prompt lengths include the chat template with thinking disabled. The inputs freeze meaningful
NInfer documentation and source excerpts, with four unique records placed across the packet.

```bash
$CLI "$MODEL" --messages examples/cli/messages/long_8k.json \
  --max-context 8192 --kv-dtype bf16 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 64

$CLI "$MODEL" --messages examples/cli/messages/long_64k.json \
  --max-context 65536 --kv-dtype int8 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 64

$CLI "$MODEL" --messages examples/cli/messages/long_128k.json \
  --max-context 131072 --kv-dtype int8 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 64

$CLI "$MODEL" --messages examples/cli/messages/long_256k.json \
  --max-context 262144 --kv-dtype int8 --prefill-chunk 1024 \
  --no-thinking --greedy --max-new 64
```

All four must output:

```text
ORCHID=37; COPPER=8142; HARBOR=KESTREL; COLOR=AMBER; SUM=8179
```

The registered 27B and 35B-A3B tokenizer/chat-template resources produce identical prompt token
sequences for every committed JSON file. Both artifacts run these inputs through the same CLI
surface.

## Fixture construction

All PNG and MP4 media are project-authored deterministic scenes. The video is five seconds at 8 FPS
with forty H.264 frames. Runtime tests never depend on a network URL or mutable external content.

The committed generated files are the actual inputs. To intentionally rebuild media and the four
long-context JSON files from the current source tree:

```bash
python3 examples/cli/make_fixtures.py \
  --tokenizer /path/to/Qwen3.6-27B
```

Regeneration updates the frozen source snapshot when selected project files change, so generated
differences should be reviewed like any other fixture change.
