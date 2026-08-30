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
| SHA-256 | `7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b` |
| Container version | 2 |
| NInfer model ID | `qwen3.6-27b` |
| NInfer weights ID | `groupwise-int` |
| NInfer target key | `qwen3_6_27b` |

The file contains the registered Text, Vision, MTP, proposal-head, tokenizer, chat-template,
generation, and media-processor objects required by NInfer.

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  '7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b' \
  'qwen3_6_27b.ninfer' | sha256sum --check
```

## Requirements

- [NInfer](https://github.com/Neroued/ninfer) revision
  [`bd265a3`](https://github.com/Neroued/ninfer/commit/bd265a36fe990475bae143d2073d6a6cf67d0da3)
  or later, built from source;
- 64-bit Linux;
- NVIDIA GeForce RTX 5090 (`sm_120a`);
- CUDA Toolkit 13.1 or newer.

NInfer does not provide an install target or packaged binary. See the
[repository README](https://github.com/Neroued/ninfer#quick-start) for source-build dependencies.

## Download and run a CLI example

```bash
hf download neroued/Qwen3.6-27B-NInfer \
  qwen3_6_27b.ninfer \
  --local-dir models

./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For images, videos, and structured chat history, see the
[CLI guide](https://github.com/Neroued/ninfer/blob/master/docs/cli.md).

## Start a local server

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. The shared 240,000-token Device KV pool admits
two active requests when their combined completion reservations fit; either request may use the
full pool while running alone. Two extra Device checkpoint slots, eight pinned Host State slots,
and 8 GiB of pinned Host KV retain reusable continuations under resource pressure.

See the [HTTP serving guide](https://github.com/Neroued/ninfer/blob/master/docs/serving.md) for the
API surface and the [resource scheduling reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/resource-scheduling-and-context-cache.md)
for cache and admission semantics.

## Supported use

The artifact supports:

- text generation in thinking and non-thinking modes;
- image, multi-image, video, and mixed multimodal messages;
- MTP speculative decoding with draft windows from one to five;
- row-scaled FP8 E4M3, INT8 group-64, and BF16 KV cache;
- CUDA Graph decode and compatible-prefix reuse;
- startup-bounded small-scale concurrent serving with true batched decode;
- the NInfer CLI;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages serving.

## Performance

The single-request serving measurements below were collected on an NVIDIA GeForce RTX 5090 with
CUDA 13.1. Requests were submitted serially to a persistent `ninfer-serve` process with CUDA Graph
enabled, a 1,024-token prefill chunk, INT8 group-64 KV cache, and prefix reuse disabled. Each
single-request value is the arithmetic mean ± sample standard deviation over five fixed seeds;
warm-up requests are excluded. These results use `weights_id = groupwise-int`.

### Concurrent MTP=3 decode saturation

The concurrent campaign uses CUDA driver API 13.3 and one 293-token prompt followed by an
8,192-token generation per active request. Each concurrency point starts a fresh server with MTP3,
INT8 group-64 KV, CUDA Graphs, a 16,384-token per-request context limit, and prefix reuse disabled.
Aggregate throughput includes only complete one-second intervals whose actual decode batch remains
equal to C. Each row is one sustained wave.

| C | Steady aggregate decode tok/s | Speedup vs. C1 | Wave makespan |
|---:|---:|---:|---:|
| 1 | 185.8 | 1.00× | 44.23 s |
| 2 | 247.0 | 1.33× | 66.67 s |
| 4 | 309.5 | 1.67× | 107.49 s |
| 8 | 535.0 | 2.88× | 125.20 s |

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
| Problem 1 | 10,686.2 ± 553.8 | 175.4 ± 1.0 | 77.9% ± 0.9% | 3.34 ± 0.03 |
| Problem 15 | 61,604.2 ± 5,677.9 | 161.9 ± 2.8 | 73.4% ± 1.7% | 3.20 ± 0.05 |
| Problem 30 | 47,339.8 ± 9,162.2 | 172.2 ± 0.9 | 78.8% ± 0.8% | 3.36 ± 0.02 |

### MTP=3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture (15 samples). Thinking was
disabled and the output limit was 4,096 tokens.

| Category | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|
| Code | 167.0 ± 5.4 | 72.3% ± 3.5% | 3.17 ± 0.11 |
| Story | 112.6 ± 9.4 | 37.8% ± 5.9% | 2.13 ± 0.18 |
| Translation | 161.5 ± 11.3 | 68.3% ± 7.2% | 3.05 ± 0.22 |
| Structured output | 193.0 ± 18.8 | 88.7% ± 11.7% | 3.66 ± 0.35 |

See the
[full methodology and results](https://github.com/Neroued/ninfer/blob/master/docs/performance.md),
including metric definitions and the exact reproduction command.

## Evaluation

The artifact was evaluated through NInfer's OpenAI-compatible serving route with thinking enabled,
MTP=3, and a 262,144-token context limit. EvalScope 1.9.0 used 0-shot prompts, rule-based scoring,
and one sample per problem with temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0, and
seed 42. All 258 configured samples completed and were scored.

| Benchmark | Accuracy | Correct / total |
|---|---:|---:|
| AIME 2025 | 86.67% | 26 / 30 |
| AIME 2026 | 93.33% | 28 / 30 |
| GPQA-Diamond | 86.87% | 172 / 198 |

These are single-sample results under the stated NInfer evaluation profile, not pass@k scores.

## Limits

- The artifact is accepted only by NInfer revision `bd265a3` or later and the matching registered
  target.
- NInfer executes on one RTX 5090 and one CUDA device, with a startup-fixed capacity of 1–8 active
  requests per Engine.
- It does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  multi-GPU execution, CPU/GPU offload, or distributed serving.
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
| Minimum runtime revision | `bd265a36fe990475bae143d2073d6a6cf67d0da3` |

The artifact identity, summarized object inventory, and conversion provenance are published in
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.6-27B-NInfer/blob/main/artifact-manifest.json).
The exact storage contract is maintained in the
[Qwen3.6-27B artifact reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/qwen3.6-27b-artifact.md).

## License

This NInfer artifact is distributed under the Apache License 2.0. The source
[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) repository is also licensed under
Apache-2.0. Users remain responsible for complying with the license and applicable laws.
