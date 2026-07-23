---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.6
  - multimodal
  - conversational
  - cuda
  - rtx-5090
model-index:
  - name: Qwen3.6-35B-A3B-NInfer
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2025
          type: aime25
        metrics:
          - type: accuracy
            value: 90.0
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2026
          type: aime26
        metrics:
          - type: accuracy
            value: 90.0
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: GPQA-Diamond
          type: gpqa_diamond
        metrics:
          - type: accuracy
            value: 85.35
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
---

# Qwen3.6-35B-A3B for NInfer

This model card is the version-controlled source for
[neroued/Qwen3.6-35B-A3B-NInfer](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer).

The repository contains
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) converted to the native
[NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format. The artifact is intended
only for NInfer; it is not a Transformers checkpoint, Safetensors distribution, or GGUF file.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_6_35b_a3b.ninfer` |
| Size | 22,783,246,080 bytes (21.22 GiB) |
| SHA-256 | `5194407dd6d3092b8c2f81ce41e014b50ca0d6f1ba4e5d8c1492b8652bfa267f` |
| Container version | 1 |
| NInfer model ID | `qwen3.6-35b-a3b` |
| NInfer target key | `qwen3_6_35b_a3b` |

The file contains the registered Text, Vision, MTP, proposal-head, DFlash, tokenizer,
chat-template, generation, and media-processor objects required by NInfer.

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  '5194407dd6d3092b8c2f81ce41e014b50ca0d6f1ba4e5d8c1492b8652bfa267f' \
  'qwen3_6_35b_a3b.ninfer' | sha256sum --check
```

## Requirements

- [NInfer](https://github.com/Neroued/ninfer) built from source;
- 64-bit Linux;
- NVIDIA GeForce RTX 5090 (`sm_120a`);
- CUDA Toolkit 13.1 or newer.

NInfer does not provide an install target or packaged binary. See the
[repository README](https://github.com/Neroued/ninfer#build) for source-build dependencies.

## Download and run

```bash
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ninfer \
  --local-dir models

./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --mtp-draft-tokens 3 \
  --lm-head-draft
```

For images, videos, structured chat history, and HTTP serving, see the
[NInfer documentation](https://github.com/Neroued/ninfer/tree/master/docs).

## Supported use

The artifact supports:

- text generation in thinking and non-thinking modes;
- image, multi-image, video, and mixed multimodal messages;
- MTP speculative decoding with draft windows from one to five;
- BF16 and INT8 group-64 KV cache;
- CUDA Graph decode and compatible-prefix reuse;
- the NInfer CLI;
- OpenAI Chat Completions and Anthropic Messages serving.

## Performance

The following single-GPU serving measurements were collected on an NVIDIA GeForce RTX 5090 with
CUDA 13.1. Requests were submitted serially to a persistent `ninfer-serve` process with CUDA Graph
enabled, a 1,024-token prefill chunk, INT8 group-64 KV cache, and prefix reuse disabled. Each value
is the arithmetic mean ± sample standard deviation over five fixed seeds; warm-up requests are
excluded.

### Long-context baseline (MTP disabled)

| Prompt tokens | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|
| 7,680 | 15,544.3 ± 242.4 | 500.2 ± 7.8 | 271.1 ± 3.6 |
| 64,512 | 10,809.0 ± 95.3 | 6,009.9 ± 52.6 | 242.9 ± 1.3 |
| 130,048 | 7,828.4 ± 34.1 | 16,693.3 ± 71.2 | 219.4 ± 1.6 |
| 260,096 | 5,157.1 ± 52.4 | 50,598.8 ± 519.7 | 188.2 ± 2.1 |

### MTP=3 long-reasoning decode

Thinking was enabled and the output limit was 65,536 tokens.

| AIME 2026 fixture | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Problem 1 | 8,675.4 ± 1,565.6 | 634.3 ± 14.2 | 82.7% ± 2.6% | 3.48 ± 0.08 |
| Problem 15 | 65,536.0 ± 0.0 | 542.8 ± 12.5 | 73.0% ± 2.5% | 3.19 ± 0.07 |
| Problem 30 | 55,171.0 ± 5,407.1 | 572.9 ± 9.1 | 77.7% ± 1.4% | 3.33 ± 0.04 |

### MTP=3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture (15 samples). Thinking was
disabled and the output limit was 4,096 tokens.

| Category | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|
| Code | 576.5 ± 21.7 | 71.0% ± 4.0% | 3.13 ± 0.12 |
| Story | 395.9 ± 30.9 | 37.7% ± 5.8% | 2.13 ± 0.17 |
| Translation | 559.3 ± 28.1 | 66.6% ± 5.1% | 3.00 ± 0.15 |
| Structured output | 661.2 ± 29.5 | 87.2% ± 6.0% | 3.62 ± 0.18 |

See the
[full methodology and results](https://github.com/Neroued/ninfer/blob/master/docs/performance.md),
including metric definitions and the exact reproduction command.

## Evaluation

The artifact was evaluated through NInfer's OpenAI-compatible serving route with thinking enabled,
MTP=3, and a 262,144-token context limit. EvalScope 1.9.0 used 0-shot prompts, rule-based scoring,
and one sample per problem with temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0, and
seed 42. All configured samples completed and were scored.

| Benchmark | Accuracy | Correct / total |
|---|---:|---:|
| AIME 2025 | 90.00% | 27 / 30 |
| AIME 2026 | 90.00% | 27 / 30 |
| GPQA-Diamond | 85.35% | 169 / 198 |

These are single-sample results under the stated NInfer evaluation profile, not pass@k scores.

## Limits

- The artifact is accepted only by the matching NInfer target.
- NInfer currently executes on one RTX 5090, one CUDA device, and one active request per Engine.
- The artifact contains the DFlash companion weights, but the current Engine does not execute
  DFlash proposals and leaves that weight group nonresident.
- It does not provide continuous batching, multi-GPU execution, CPU/GPU offload, or distributed
  serving.
- Context allocation is subject to GPU memory and the selected KV-cache type.
- NInfer does not execute generated tool calls.

## Provenance

| Field | Value |
|---|---|
| Base source repository | `Qwen/Qwen3.6-35B-A3B` |
| Base source revision | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| DFlash source repository | `z-lab/Qwen3.6-35B-A3B-DFlash` |
| DFlash source revision | `f181eece646affea2c38b2765f1aaa01a9734ccd` |
| Conversion recipe | `qwen3_6_35b_a3b-v2` |
| Converter repository | `https://github.com/Neroued/ninfer` |

The complete object inventory and conversion metadata are published in
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/blob/main/artifact-manifest.json).

## License

This NInfer artifact is distributed under the Apache License 2.0. The source
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) repository is also licensed under
Apache-2.0. Users remain responsible for complying with the license and applicable laws.
