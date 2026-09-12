---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.6-35B-A3B
  - z-lab/Qwen3.6-35B-A3B-DFlash
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
Its optional DFlash companion weights come from
[z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash).

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_6_35b_a3b.ninfer` |
| Size | 22,783,246,080 bytes (21.22 GiB) |
| SHA-256 | `1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2` |
| Container version | 2 |
| NInfer model ID | `qwen3.6-35b-a3b` |
| NInfer weights ID | `groupwise-int` |
| NInfer target key | `qwen3_6_35b_a3b` |

The file contains the registered Text, Vision, MTP, proposal-head, DFlash, tokenizer,
chat-template, generation, and media-processor objects required by NInfer.

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  '1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2' \
  'qwen3_6_35b_a3b.ninfer' | sha256sum --check
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
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ninfer \
  --local-dir models

./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For DFlash, the measured block-8 configuration uses seven draft tokens:

```bash
./build/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec dflash --draft-tokens 7 \
  --lm-head-draft
```

`--draft-tokens` accepts `1..5` for MTP and `1..15` for DFlash. The DFlash value `7` is the
measured block-length-eight profile; `15` uses the companion's full native 16-position block. MTP
and DFlash are mutually exclusive backend selections. DFlash may be combined with `--vision` for
image or video prompts; it accelerates generated-text decode, not Vision encode or target prefill.

For images, videos, and structured chat history, see the
[CLI guide](https://github.com/Neroued/ninfer/blob/master/docs/cli.md).

## Start a local server

```bash
./build/apps/ninfer-serve models/qwen3_6_35b_a3b.ninfer \
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
- DFlash speculative decoding for Text and image/video Vision prompts with draft windows from one
  to fifteen;
- BF16, INT8, FP8, NVFP4, and K8V4 KV cache;
- CUDA Graph decode and compatible-prefix reuse;
- startup-bounded small-scale concurrent serving with true batched decode;
- the NInfer CLI;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages serving.

## Performance

Measured on 2026-09-07 at NInfer revision `487f89773f07cb18a2fb841fe0971ec9634d409b` through the public HTTP
serving route on one RTX 5090. The Release build uses CUDA 13.1 compile/runtime and CUDA driver
API 13.3, INT8 group-64 KV, CUDA Graphs, 1,024-token prefill chunks, and disabled prefix reuse.
Each measurement point starts a fresh server and excludes startup warmup.

### Long-context baseline

No speculative backend; context ceiling 262,144 tokens; five fixed seeds per prompt length.
Values are arithmetic mean ± sample standard deviation.

| Prompt tokens | Prefill phase (tok/s) | Server TTFT (ms) | Decode phase (tok/s) |
| --- | --- | --- | --- |
| 7,680 | 17,705.4 ± 234.6 | 437.7 ± 6.0 | 338.3 ± 5.5 |
| 64,512 | 11,758.0 ± 122.3 | 5,510.0 ± 57.3 | 298.1 ± 2.0 |
| 130,048 | 8,317.1 ± 109.3 | 15,685.1 ± 208.0 | 260.6 ± 4.4 |
| 260,096 | 5,247.0 ± 30.1 | 49,657.3 ± 283.0 | 213.0 ± 3.3 |

### MTP3 single-request decode

Three draft tokens, optimized proposal head, stochastic sampling. The C=1 corpus point supplies
five samples per reasoning fixture and fifteen per scenario category. Reasoning enables thinking
with a 65,536-token output budget; other scenarios disable thinking with a 4,096-token budget.

| Fixture | Completion tokens | Decode phase (tok/s) | Spec acceptance | Spec tokens/round |
| --- | --- | --- | --- | --- |
| `long_decode_aime26_01` | 8,407.2 ± 2,764.1 | 750.6 ± 22.4 | 83.5% ± 3.8% | 3.50 ± 0.11 |
| `long_decode_aime26_15` | 64,860.2 ± 1,511.1 | 636.5 ± 9.2 | 72.0% ± 1.0% | 3.16 ± 0.03 |
| `long_decode_aime26_30` | 55,354.6 ± 7,132.4 | 683.3 ± 4.0 | 79.2% ± 1.1% | 3.38 ± 0.03 |

| Category | Decode phase (tok/s) | Spec acceptance | Spec tokens/round |
| --- | --- | --- | --- |
| Code | 677.2 ± 25.6 | 70.5% ± 3.6% | 3.12 ± 0.11 |
| Story | 465.2 ± 36.3 | 37.8% ± 5.5% | 2.14 ± 0.16 |
| Translation | 659.0 ± 35.1 | 67.7% ± 5.8% | 3.03 ± 0.17 |
| Structured | 779.6 ± 44.3 | 87.5% ± 7.1% | 3.63 ± 0.21 |

### MTP3 corpus makespan

Each C runs the same 75-request corpus. Rates use the full makespan, including prefill, workload
transitions, and drain. The per-request context ceiling is 262,144 tokens and KV capacity is automatic.

| C | Decode tokens | Makespan (s) | Requests/s | Corpus decode (tok/s) | Avg batch | MTP acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 831,378 | 1,279.40 | 0.0586 | 649.8 | 1.00 | 72.2% |
| 2 | 827,334 | 898.13 | 0.0835 | 921.2 | 2.00 | 73.2% |
| 4 | 807,280 | 737.87 | 0.1016 | 1,094.1 | 3.45 | 71.7% |
| 8 | 861,416 | 679.61 | 0.1104 | 1,267.5 | 6.48 | 73.3% |

### DFlash K=7 single-request decode

Seven draft tokens, optimized proposal head, and the same corpus and context ceiling. Values below
use stochastic sampling and per-request decode-phase timings.

| Fixture | Completion tokens | Decode phase (tok/s) | Spec acceptance | Spec tokens/round |
| --- | --- | --- | --- | --- |
| `long_decode_aime26_01` | 8,536.2 ± 3,506.4 | 866.8 ± 47.4 | 66.4% ± 4.0% | 5.65 ± 0.28 |
| `long_decode_aime26_15` | 65,325.0 ± 471.8 | 641.6 ± 62.4 | 49.7% ± 6.2% | 4.48 ± 0.43 |
| `long_decode_aime26_30` | 53,756.4 ± 5,693.8 | 732.6 ± 13.3 | 58.1% ± 1.4% | 5.07 ± 0.10 |

| Category | Decode phase (tok/s) | Spec acceptance | Spec tokens/round |
| --- | --- | --- | --- |
| Code | 620.6 ± 43.3 | 42.6% ± 4.1% | 3.98 ± 0.28 |
| Story | 291.6 ± 58.8 | 12.2% ± 5.4% | 1.85 ± 0.38 |
| Translation | 547.5 ± 74.5 | 35.2% ± 6.8% | 3.47 ± 0.48 |
| Structured | 906.4 ± 127.3 | 69.7% ± 12.5% | 5.88 ± 0.88 |

### MTP3 decode saturation

One 8,192-token generation per active request, a 16,384-token context ceiling, and one wave per C.
Steady rates include only complete intervals with a full decode batch; acceptance covers the full wave.

| C | Steady decode (tok/s) | MTP acceptance (wave) | Wave makespan (s) |
| --- | --- | --- | --- |
| 1 | 642.5 | 68.6% | 12.70 |
| 2 | 907.2 | 66.3% | 18.03 |
| 4 | 1,213.5 | 69.6% | 27.27 |
| 8 | 1,380.7 | 68.0% | 47.94 |

All 485 formal requests across the 11 measurement points completed without request, CUDA, or
out-of-memory failures. No obvious short-cycle repetition was found in the 225 C=1 speculative
responses. Output-limit samples remain in the reported results.

See the [complete performance report](https://github.com/Neroued/ninfer/blob/master/docs/performance/qwen3.6-35b-a3b.md)
for DFlash greedy results, corpus token totals, termination counts, resource settings, and commands.

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
| Base source repository | `Qwen/Qwen3.6-35B-A3B` |
| Base source revision | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| DFlash source repository | [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) |
| DFlash source revision | [`f181eece646affea2c38b2765f1aaa01a9734ccd`](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash/tree/f181eece646affea2c38b2765f1aaa01a9734ccd) |
| Conversion recipe | `qwen3_6_35b_a3b-v2` |
| Converter repository | `https://github.com/Neroued/ninfer` |
| Converter revision | `872b9792b4f43244e38faca5cded79136eca5666` |
| Minimum runtime revision | `bd265a36fe990475bae143d2073d6a6cf67d0da3` |

The artifact identity, summarized object inventory, and conversion provenance are published in
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer/blob/main/artifact-manifest.json).
The exact storage contract is maintained in the
[Qwen3.6-35B-A3B artifact reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/qwen3.6-35b-a3b-artifact.md).

## License

This NInfer artifact is distributed under the Apache License 2.0. The source
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) repository is also licensed under
Apache-2.0. Users remain responsible for complying with the license and applicable laws.
