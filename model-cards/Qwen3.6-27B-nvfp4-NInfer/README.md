---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.6-27B
  - rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.6
  - nvfp4
  - w4a4
  - blackwell
  - multimodal
  - conversational
  - cuda
  - rtx-5090
model-index:
  - name: Qwen3.6-27B-nvfp4-NInfer
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2025
          type: aime25
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
            value: 84.34
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
---

# Qwen3.6-27B NVFP4 for NInfer

This model card is the version-controlled source for
[neroued/Qwen3.6-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer).

The repository contains the NVFP4 weight profile of
[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), converted from the fixed packed weights in
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm)
to the native [NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format. The artifact is
intended only for NInfer; it is not a Transformers checkpoint, Safetensors distribution, or GGUF
file.

This is a second weight profile for the existing `qwen3_6_27b` target, not a separate model target.
The version-2 artifact identity selects the NVFP4 binder and execution leaves. NInfer uses W4A4
Tensor Core MMA for prefill and A16 NVFP4 kernels for decode while retaining the same Text, Vision,
MTP, prefix-reuse, CLI, and serving paths as the groupwise-int profile.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_6_27b_nvfp4.ninfer` |
| Size | 18,324,064,000 bytes (17.07 GiB) |
| SHA-256 | `bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a` |
| Container version | 2 |
| NInfer model ID | `qwen3.6-27b` |
| NInfer weights ID | `nvfp4` |
| NInfer target key | `qwen3_6_27b` |
| Stored objects | 1,307 (1,301 tensors and 6 resources) |
| NVFP4 tensors | 247 |

The file contains the registered Text, Vision, MTP, proposal-head, tokenizer, chat-template,
generation, and media-processor objects required by NInfer. Text linears use the source repository's
fixed mixed NVFP4/BF16 allocation; Vision, MTP, and frontend resources retain their registered
NInfer formats.

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  'bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a' \
  'qwen3_6_27b_nvfp4.ninfer' | sha256sum --check
```

## Requirements

- [NInfer](https://github.com/Neroued/ninfer) revision
  [`a85109c`](https://github.com/Neroued/ninfer/commit/a85109c4ef16b0b4a218c61e4641cfd0d11320fe)
  or later, built from source;
- 64-bit Linux;
- NVIDIA GeForce RTX 5090 (`sm_120a`);
- CUDA Toolkit 13.1 or newer.

NInfer does not provide an install target or packaged binary. See the
[repository README](https://github.com/Neroued/ninfer#quick-start) for source-build dependencies.

## Download and run a CLI example

```bash
hf download neroued/Qwen3.6-27B-nvfp4-NInfer \
  qwen3_6_27b_nvfp4.ninfer \
  --local-dir models

./build/apps/ninfer models/qwen3_6_27b_nvfp4.ninfer \
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
./build/apps/ninfer-serve models/qwen3_6_27b_nvfp4.ninfer \
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
CUDA 13.1 compile/runtime and CUDA driver API 13.3. Requests were submitted serially to a persistent
`ninfer-serve` process with CUDA Graph enabled, a 1,024-token prefill chunk, INT8 group-64 KV cache,
and prefix reuse disabled. Each single-request value is the arithmetic mean ± sample standard
deviation over five fixed seeds; server warm-up completes before the measured requests.

### Concurrent MTP=3 decode saturation

The concurrent campaign uses one 293-token prompt followed by an 8,192-token generation per active
request. Each concurrency point starts a fresh server with MTP3, INT8 group-64 KV, CUDA Graphs, a
16,384-token per-request context limit, and prefix reuse disabled. Aggregate throughput includes
only complete one-second intervals whose actual decode batch remains equal to C. Each row is one
sustained wave.

| C | Steady aggregate decode tok/s | Speedup vs. C1 | Wave makespan |
|---:|---:|---:|---:|
| 1 | 202.4 | 1.00× | 40.46 s |
| 2 | 399.7 | 1.97× | 41.82 s |
| 4 | 699.7 | 3.46× | 47.92 s |
| 8 | 1,146.9 | 5.67× | 58.57 s |

At C=8, the profile sustains **1,146.9 aggregate decode tok/s**, or **5.67×** its C=1 throughput.

### Long-context baseline (MTP disabled)

| Prompt tokens | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|
| 7,680 | 11,191.5 ± 70.2 | 692.5 ± 4.3 | 86.4 ± 0.5 |
| 64,512 | 6,298.5 ± 97.6 | 10,288.6 ± 159.3 | 78.0 ± 1.2 |
| 130,048 | 4,204.7 ± 14.1 | 31,012.5 ± 104.6 | 71.2 ± 0.2 |
| 260,096 | 2,510.6 ± 16.8 | 103,761.1 ± 698.8 | 59.9 ± 0.3 |

At 7,680 prompt tokens this is 3.48× the prefill throughput of the published groupwise-int profile;
at 260,096 prompt tokens it is 1.55×.

### MTP=3 long-reasoning decode

Thinking was enabled and the output limit was 65,536 tokens.

| AIME 2026 fixture | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Problem 1 | 12,053.4 ± 820.9 | 231.0 ± 3.0 | 80.2% ± 1.2% | 3.41 ± 0.04 |
| Problem 15 | 63,109.0 ± 5,426.9 | 213.1 ± 4.2 | 76.3% ± 2.0% | 3.29 ± 0.06 |
| Problem 30 | 57,166.4 ± 9,204.9 | 223.3 ± 1.8 | 81.1% ± 1.5% | 3.43 ± 0.04 |

### MTP=3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture (15 samples). Thinking was
disabled and the output limit was 4,096 tokens.

| Category | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|
| Code | 220.3 ± 8.2 | 74.2% ± 4.0% | 3.23 ± 0.12 |
| Story | 148.8 ± 11.6 | 39.2% ± 5.7% | 2.18 ± 0.17 |
| Translation | 213.6 ± 12.2 | 70.5% ± 6.0% | 3.12 ± 0.18 |
| Structured output | 252.2 ± 16.3 | 89.8% ± 8.0% | 3.69 ± 0.24 |

See the
[full methodology and results](https://github.com/Neroued/ninfer/blob/master/docs/performance.md),
including metric definitions, comparison data, and the exact reproduction command.

## Evaluation

The artifact was evaluated through NInfer's OpenAI-compatible serving route with thinking enabled,
MTP=3, and a 262,144-token context limit. EvalScope 1.9.0 used 0-shot prompts, rule-based scoring,
and one sample per problem with temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0, and
seed 42. All 258 configured samples completed and were scored.

| Benchmark | Accuracy | Correct / total |
|---|---:|---:|
| AIME 2025 | 93.33% | 28 / 30 |
| AIME 2026 | 93.33% | 28 / 30 |
| GPQA-Diamond | 84.34% | 167 / 198 |

These are single-sample results under the stated NInfer evaluation profile, not pass@k scores.

## Limits

- The artifact is accepted only by NInfer revision `a85109c` or later and the matching registered
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
| Base repository | `Qwen/Qwen3.6-27B` |
| Base revision | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| NVFP4 source repository | `rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` |
| NVFP4 source revision | `9b5389d4a1e207daab2d47732efea57d7e946dcf` |
| Conversion recipe | `qwen3_6_27b_nvfp4-v1` |
| Converter repository | `https://github.com/Neroued/ninfer` |
| Converter revision | `a85109c4ef16b0b4a218c61e4641cfd0d11320fe` |
| Minimum runtime revision | `a85109c4ef16b0b4a218c61e4641cfd0d11320fe` |
| Ranking input SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |

The artifact identity, summarized object inventory, and conversion provenance are published in
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer/blob/main/artifact-manifest.json).
The exact storage contract is maintained in the
[Qwen3.6-27B artifact reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/qwen3.6-27b-artifact.md).

## License

This NInfer artifact is distributed under the Apache License 2.0. The
[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) base repository and the
[NVFP4 source repository](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm)
are also licensed under Apache-2.0. Users remain responsible for complying with the license and
applicable laws.
