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

For text-only DFlash, the measured block-8 configuration uses seven draft tokens:

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
and DFlash are mutually exclusive. DFlash is text-only and cannot be combined with `--vision`; use
a separate non-DFlash process for image or video requests.

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
- text-only DFlash speculative decoding with draft windows from one to fifteen;
- row-scaled FP8 E4M3, INT8 group-64, and BF16 KV cache;
- CUDA Graph decode and compatible-prefix reuse;
- startup-bounded small-scale concurrent serving with true batched decode;
- the NInfer CLI;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages serving.

## Performance

The single-request serving measurements below were collected on an NVIDIA GeForce RTX 5090 with
CUDA 13.1 compile/runtime. The refreshed MTP3 measurements use CUDA driver API 13.3. Requests were
submitted serially to a persistent `ninfer-serve` process with CUDA Graph enabled, a 1,024-token
prefill chunk, INT8 group-64 KV cache, and prefix reuse disabled. Each single-request value is the
arithmetic mean ± sample standard deviation over five fixed seeds; server warm-up is excluded.

### Concurrent MTP=3 decode saturation

The concurrent campaign uses CUDA driver API 13.3 and one 293-token prompt followed by an
8,192-token generation per active request. Each concurrency point starts a fresh server with MTP3,
INT8 group-64 KV, CUDA Graphs, a 16,384-token per-request context limit, and prefix reuse disabled.
Aggregate throughput includes only complete one-second intervals whose actual decode batch remains
equal to C. Each row is one sustained wave.

| C | Steady aggregate decode tok/s | Speedup vs. C1 | Wave makespan |
|---:|---:|---:|---:|
| 1 | 593.0 | 1.00× | 13.75 s |
| 2 | 877.7 | 1.48× | 18.87 s |
| 4 | 1,166.0 | 1.97× | 28.43 s |
| 8 | 1,313.8 | 2.22× | 50.20 s |

At C=8, the profile sustains **1,313.8 aggregate decode tok/s**.

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
| Problem 1 | 8,223.0 ± 2,224.1 | 726.2 ± 22.9 | 82.8% ± 3.4% | 3.48 ± 0.10 |
| Problem 15 | 65,536.0 ± 0.0 | 620.3 ± 8.1 | 72.7% ± 1.4% | 3.18 ± 0.04 |
| Problem 30 | 52,977.8 ± 11,849.6 | 671.9 ± 8.8 | 80.1% ± 2.7% | 3.40 ± 0.08 |

### MTP=3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture (15 samples). Thinking was
disabled and the output limit was 4,096 tokens.

| Category | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|
| Code | 657.6 ± 34.3 | 70.3% ± 5.5% | 3.11 ± 0.16 |
| Story | 456.2 ± 36.6 | 38.0% ± 6.0% | 2.14 ± 0.18 |
| Translation | 649.7 ± 33.0 | 67.6% ± 5.1% | 3.03 ± 0.15 |
| Structured output | 770.9 ± 29.3 | 89.1% ± 4.9% | 3.67 ± 0.15 |

### DFlash block=8 (`k=7`) decode

These stochastic-sampling measurements use the same fixtures, sampling parameters, and output
limits as MTP3. Different speculative backends consume random values differently, so this is a
fixed-workload comparison rather than a token-identical output comparison.

| Workload | Decode tok/s | DFlash acceptance | DFlash tokens/round | Change vs. MTP3 |
|---|---:|---:|---:|---:|
| AIME 2026 problem 1 | 764.1 ± 55.6 | 65.2% ± 5.4% | 5.56 ± 0.38 | +5.2% |
| AIME 2026 problem 15 | 584.0 ± 33.3 | 51.1% ± 3.7% | 4.58 ± 0.26 | -5.9% |
| AIME 2026 problem 30 | 638.3 ± 15.8 | 56.4% ± 2.5% | 4.95 ± 0.17 | -5.0% |
| Code | 562.3 ± 36.2 | 43.0% ± 3.7% | 4.01 ± 0.26 | -14.5% |
| Story | 261.7 ± 51.1 | 12.1% ± 5.3% | 1.85 ± 0.37 | -42.6% |
| Translation | 490.8 ± 62.6 | 34.8% ± 6.3% | 3.44 ± 0.44 | -24.5% |
| Structured output | 786.4 ± 124.7 | 66.5% ± 13.5% | 5.66 ± 0.94 | +2.0% |

DFlash throughput is acceptance-sensitive: block=8 leads on AIME problem 1 and structured output,
while MTP3 remains faster on the other measured long-reasoning and cross-scenario cases.

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
