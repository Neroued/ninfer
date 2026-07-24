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
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For text-only DFlash, the measured block-8 configuration uses seven draft tokens:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec dflash --draft-tokens 7 \
  --lm-head-draft
```

`--draft-tokens` accepts `1..5` for MTP and `1..15` for DFlash. The DFlash value `7` is the
current measured recommendation, not a semantic limit; `15` uses the companion's full native
16-position block. MTP and DFlash are mutually exclusive. DFlash is text-only and cannot be
combined with `--vision`; use a separate non-DFlash process for image or video requests.

For images, videos, structured chat history, and HTTP serving, see the
[NInfer documentation](https://github.com/Neroued/ninfer/tree/master/docs).

## Supported use

The artifact supports:

- text generation in thinking and non-thinking modes;
- image, multi-image, video, and mixed multimodal messages;
- MTP speculative decoding with draft windows from one to five;
- text-only DFlash speculative decoding with draft windows from one to fifteen;
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
| Problem 1 | 7,933.0 ± 1,852.3 | 695.1 ± 17.7 | 83.3% ± 2.8% | 3.50 ± 0.08 |
| Problem 15 | 65,536.0 ± 0.0 | 584.0 ± 10.6 | 72.4% ± 1.7% | 3.17 ± 0.05 |
| Problem 30 | 61,743.6 ± 4,489.5 | 629.4 ± 15.7 | 79.6% ± 3.2% | 3.39 ± 0.10 |

### MTP=3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture (15 samples). Thinking was
disabled and the output limit was 4,096 tokens.

| Category | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|
| Code | 635.0 ± 24.2 | 71.8% ± 4.2% | 3.15 ± 0.13 |
| Story | 434.9 ± 34.8 | 38.2% ± 5.9% | 2.15 ± 0.18 |
| Translation | 598.6 ± 26.6 | 66.1% ± 4.5% | 2.98 ± 0.14 |
| Structured output | 714.3 ± 36.2 | 87.7% ± 6.6% | 3.63 ± 0.20 |

### DFlash block=8 (`k=7`) decode

These stochastic-sampling measurements use the same fixtures, sampling parameters, output limits,
and server configuration as MTP3. Different speculative backends consume random values
differently, so this is a fixed-workload comparison rather than a token-identical output
comparison.

| Workload | Decode tok/s | DFlash acceptance | DFlash tokens/round | Change vs. MTP3 |
|---|---:|---:|---:|---:|
| AIME 2026 problem 1 | 764.1 ± 55.6 | 65.2% ± 5.4% | 5.56 ± 0.38 | +9.9% |
| AIME 2026 problem 15 | 584.0 ± 33.3 | 51.1% ± 3.7% | 4.58 ± 0.26 | 0.0% |
| AIME 2026 problem 30 | 638.3 ± 15.8 | 56.4% ± 2.5% | 4.95 ± 0.17 | +1.4% |
| Code | 562.3 ± 36.2 | 43.0% ± 3.7% | 4.01 ± 0.26 | -11.4% |
| Story | 261.7 ± 51.1 | 12.1% ± 5.3% | 1.85 ± 0.37 | -39.8% |
| Translation | 490.8 ± 62.6 | 34.8% ± 6.3% | 3.44 ± 0.44 | -18.0% |
| Structured output | 786.4 ± 124.7 | 66.5% ± 13.5% | 5.66 ± 0.94 | +10.1% |

DFlash throughput is acceptance-sensitive: block=8 leads on the measured long-reasoning and
structured-output cases with sufficient acceptance, while MTP3 remains faster on the measured
code, story, and translation categories.

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
