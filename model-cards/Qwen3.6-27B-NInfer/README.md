---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: Qwen/Qwen3.6-27B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.6
  - multimodal
  - conversational
  - cuda
  - rtx-5090
# Replace every TBD evaluation value before publishing this card.
model-index:
  - name: Qwen3.6-27B-NInfer
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2025
          type: aime25
        metrics:
          - type: accuracy
            value: 86.67
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
            value: 93.33
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
            value: 86.87
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
---

# Qwen3.6-27B for NInfer

This model card is the version-controlled source for
[neroued/Qwen3.6-27B-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-NInfer).

The repository contains
[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) converted to the native
[NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format. The artifact is intended
only for NInfer; it is not a Transformers checkpoint, Safetensors distribution, or GGUF file.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_6_27b.ninfer` |
| Size | 17,495,365,888 bytes (16.29 GiB) |
| SHA-256 | `74fac75f3a6b7ab7b52e08c36969c7a33a8ba23465910eccd72d195adb497127` |
| Container version | 1 |
| NInfer model ID | `qwen3.6-27b` |
| NInfer target key | `qwen3_6_27b` |

The file contains the registered Text, Vision, MTP, proposal-head, tokenizer, chat-template,
generation, and media-processor objects required by NInfer.

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  '74fac75f3a6b7ab7b52e08c36969c7a33a8ba23465910eccd72d195adb497127' \
  'qwen3_6_27b.ninfer' | sha256sum --check
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
hf download neroued/Qwen3.6-27B-NInfer \
  qwen3_6_27b.ninfer \
  --local-dir models

./build/apps/ninfer models/qwen3_6_27b.ninfer \
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
| 7,680 | 3,218.1 ± 4.3 | 2,392.4 ± 3.0 | 77.6 ± 0.1 |
| 64,512 | 2,655.9 ± 2.9 | 24,335.7 ± 25.2 | 70.7 ± 0.1 |
| 130,048 | 2,185.3 ± 0.3 | 59,590.3 ± 8.9 | 64.5 ± 0.1 |
| 260,096 | 1,614.8 ± 0.6 | 161,221.8 ± 62.5 | 54.8 ± 0.1 |

### MTP=3 long-reasoning decode

Thinking was enabled and the output limit was 65,536 tokens.

| AIME 2026 fixture | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Problem 1 | 11,009.4 ± 419.1 | 174.2 ± 3.3 | 79.9% ± 2.0% | 3.40 ± 0.06 |
| Problem 15 | 62,652.6 ± 3,000.4 | 158.7 ± 5.2 | 73.3% ± 3.4% | 3.20 ± 0.10 |
| Problem 30 | 47,837.8 ± 5,882.7 | 169.0 ± 2.7 | 79.3% ± 2.0% | 3.38 ± 0.06 |

### MTP=3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture (15 samples). Thinking was
disabled and the output limit was 4,096 tokens.

| Category | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|
| Code | 163.9 ± 6.2 | 72.5% ± 3.9% | 3.18 ± 0.12 |
| Story | 110.4 ± 9.2 | 37.9% ± 6.0% | 2.14 ± 0.18 |
| Translation | 153.6 ± 11.7 | 65.7% ± 7.5% | 2.97 ± 0.23 |
| Structured output | 189.1 ± 15.7 | 88.9% ± 10.2% | 3.67 ± 0.31 |

See the
[full methodology and results](https://github.com/Neroued/ninfer/blob/master/docs/performance.md),
including metric definitions and the exact reproduction command.

## Evaluation

The artifact is being evaluated through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and a 262,144-token context limit. EvalScope 1.9.0 uses 0-shot prompts, rule-based
scoring, and one sample per problem with temperature 0.6, top-p 0.95, top-k 20, presence penalty
1.0, and seed 42. Scores will be filled after all configured samples complete and the results have
been validated.

| Benchmark | Accuracy | Correct / total |
|---|---:|---:|
| AIME 2025 | 86.67% | 26 / 30 |
| AIME 2026 | 93.33% | 28 / 30 |
| GPQA-Diamond | 86.87% | 172 / 198 |

These will be single-sample results under the stated NInfer evaluation profile, not pass@k scores.

## Limits

- The artifact is accepted only by the matching NInfer target.
- NInfer currently executes on one RTX 5090, one CUDA device, and one active request per Engine.
- It does not provide continuous batching, multi-GPU execution, CPU/GPU offload, or distributed
  serving.
- Context allocation is subject to GPU memory and the selected KV-cache type.
- NInfer does not execute generated tool calls.

## Provenance

| Field | Value |
|---|---|
| Source repository | `Qwen/Qwen3.6-27B` |
| Source revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| Conversion recipe | `qwen3_6_27b-v2` |
| Converter repository | `https://github.com/Neroued/ninfer` |
| Converter revision | `d6319426e5ef08fa95c36e75cb3ab8b18e5fb957` |

The complete object inventory and conversion metadata are published in
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.6-27B-NInfer/blob/main/artifact-manifest.json).

## License

This NInfer artifact is distributed under the Apache License 2.0. The source
[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) repository is also licensed under
Apache-2.0. Users remain responsible for complying with the license and applicable laws.
