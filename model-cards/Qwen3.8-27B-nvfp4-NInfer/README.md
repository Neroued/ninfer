---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - unsloth/Qwen3.8-27B-NVFP4
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - nvfp4
  - fp8
  - w4a4
  - blackwell
  - multimodal
  - conversational
  - cuda
  - rtx-5090
---

# Qwen3.8-27B NVFP4 for NInfer

This model card is the version-controlled source for
[neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer).

The repository contains the registered NVFP4 weight profile of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). It combines the official BF16 checkpoint
with the fixed packed Text weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) in the native
[NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format. The artifact is intended
only for NInfer; it is not a Transformers checkpoint, Safetensors distribution, or GGUF file.

This is a second weight profile for the existing `qwen3_8_27b` target, not a separate model target.
The version-2 artifact identity selects the NVFP4 binder and execution leaves. The `nvfp4` weights
ID names the complete registered profile rather than claiming that every matrix has one format:
Text layers 0–55 use NVFP4 MLP weights, while the token embedding, attention input/output
projections, GDN Q/K/V/Z and output projections, full output head, and Text layers 56–63 MLP weights
use row-scaled FP8. BF16 control weights and the registered MTP and Vision allocations are retained.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_nvfp4.ninfer` |
| Size | 21,492,695,040 bytes (20.02 GiB) |
| SHA-256 | `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32` |
| Container version | 2 |
| NInfer model ID | `qwen3.8-27b` |
| NInfer weights ID | `nvfp4` |
| NInfer target key | `qwen3_8_27b` |
| Stored objects | 1,124 (1,118 tensors and 6 resources) |
| NVFP4 tensors | 112 |
| Row-scaled FP8 tensors | 146 |

The file contains the registered Text, Vision, MTP, optimized proposal-head, tokenizer,
chat-template, generation, and media-processor objects required by NInfer. Source-derived NVFP4 and
FP8 words are preserved without decode and requantization; only the official BF16 token embedding
is encoded locally as row-scaled FP8.

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  'bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32' \
  'qwen3_8_27b_nvfp4.ninfer' | sha256sum --check
```

## Requirements

- [NInfer](https://github.com/Neroued/ninfer) revision
  [`5d2c1f5`](https://github.com/Neroued/ninfer/commit/5d2c1f5590b8f4c3d106a75f65210eb4efb8f4e1)
  or later, built from source;
- 64-bit Linux;
- NVIDIA GeForce RTX 5090 (`sm_120a`);
- CUDA Toolkit 13.1 or newer.

NInfer does not provide an install target or packaged binary. See the
[repository README](https://github.com/Neroued/ninfer#build) for source-build dependencies.

## Download and run

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models

./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
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
- startup-bounded small-scale concurrent serving with true batched decode;
- the NInfer CLI;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages serving.

## Performance

The MTP0 measurements below were collected at NInfer revision
[`f08597d`](https://github.com/Neroued/ninfer/commit/f08597d6eaafce5b875934aaa85854fcd5426df8),
and the MTP3 measurements at revision
[`32c9881`](https://github.com/Neroued/ninfer/commit/32c9881b6783949df4999422a764b3dcaa111b13).
Both campaigns used one NVIDIA GeForce RTX 5090, CUDA 13.1 compile/runtime, CUDA driver API 13.3,
stochastic sampling, INT8 group-64 KV, CUDA Graphs, a 1,024-token prefill chunk, and prefix reuse
disabled. MTP0 used no speculative backend and a 262,144-token context limit; MTP3 used a
131,072-token per-request context limit and three draft tokens.

### Concurrent MTP=3 corpus makespan

The fixed corpus contains three long-reasoning and twelve cross-scenario fixtures with five seeds
each, for 75 requests. Every concurrency point starts a fresh server and uses the same shuffle seed
and ordered HTTP send sequence. C=1 is the serial single-request corpus. Makespan includes prefill,
decode, workload transitions, and final drain.

| C | Requests | Decode tokens | Makespan | Requests/s | Decode tok/s | Avg batch | MTP acceptance | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 752,160 | 4,670.27 s | 0.0161 | 161.1 | 1.00 | 60.8% | 1.00× |
| 2 | 75 | 739,951 | 2,510.78 s | 0.0299 | 294.7 | 1.98 | 59.2% | 1.86× |
| 4 | 75 | 713,384 | 1,647.74 s | 0.0455 | 432.9 | 3.29 | 58.0% | 2.83× |
| 8 | 75 | 723,602 | 2,164.90 s | 0.0346 | 334.2 | 2.36 | 57.6% | 2.16× |

All 300 requests completed without a request, CUDA, or out-of-memory failure. C=4 gives the
shortest complete-corpus makespan. C=8 is limited by memory pressure, which makes its
complete-corpus result slower than C=4. Sampling is stochastic, so the fixed prompts and seeds do
not imply token-identical continuations across concurrency-specific numerical routes; exact
decode-token totals are shown above.

### Long-context baseline (MTP disabled)

Each value is the arithmetic mean ± sample standard deviation over five fixed seeds after server
warm-up.

| Prompt tokens | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|
| 7,680 | 8,340.4 ± 13.0 | 931.6 ± 1.6 | 71.2 ± 0.1 |
| 64,512 | 5,297.9 ± 259.2 | 12,281.1 ± 561.5 | 65.7 ± 0.8 |
| 130,048 | 3,544.7 ± 25.3 | 36,853.5 ± 259.4 | 59.6 ± 0.9 |
| 260,096 | 2,203.1 ± 13.4 | 118,354.8 ± 717.2 | 52.9 ± 2.3 |

### MTP=3 single-request long-reasoning decode

The C=1 point supplies five samples for each fixture. Values are arithmetic mean ± sample standard
deviation from server phase timings and speculative counters.

| AIME 2026 fixture | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Problem 1 | 1,465.4 ± 417.3 | 195.2 ± 4.6 | 76.0% ± 2.4% | 3.28 ± 0.07 |
| Problem 15 | 65,414.4 ± 271.9 | 151.4 ± 2.0 | 56.2% ± 1.1% | 2.69 ± 0.03 |
| Problem 30 | 50,023.4 ± 14,839.1 | 167.5 ± 23.7 | 64.6% ± 14.9% | 2.94 ± 0.45 |

### MTP=3 single-request cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|
| Code | 194.3 ± 6.1 | 76.4% ± 3.9% | 3.29 ± 0.12 |
| Story | 126.1 ± 10.9 | 37.4% ± 5.8% | 2.12 ± 0.17 |
| Translation | 192.3 ± 11.9 | 75.0% ± 6.5% | 3.25 ± 0.19 |
| Structured output | 219.8 ± 8.6 | 90.8% ± 5.1% | 3.72 ± 0.15 |

See the
[full methodology and results](https://github.com/Neroued/ninfer/blob/master/docs/performance.md)
for metric definitions and the exact reproduction command.

## Limits

- The artifact is accepted only by NInfer revision `5d2c1f5` or later and the matching registered
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
| Base repository | `Qwen/Qwen3.8-27B` |
| Base revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Base download source | `modelscope.cn/models/Qwen/Qwen3.8-27B` |
| Quantized source repository | `unsloth/Qwen3.8-27B-NVFP4` |
| Quantized source revision | `60e813d4dbbdc5d64cf3f5a8caf2897bedf03679` |
| Conversion recipe | `qwen3_8_27b_nvfp4-v1` |
| Embedding encoder | `MAXABS_BF16S_RECIP_E4M3FN_RNE_V1` |
| Converter repository | `https://github.com/Neroued/ninfer` |
| Converter revision | `651d779657988dcb943896983d415ff6d38a21e2` |
| Minimum runtime revision | `5d2c1f5590b8f4c3d106a75f65210eb4efb8f4e1` |
| Ranking input SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |

The artifact identity, summarized object inventory, and conversion provenance are published in
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/blob/main/artifact-manifest.json).
The exact storage contract is maintained in the
[Qwen3.8-27B artifact reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/qwen3.8-27b-artifact.md).

## License

This NInfer artifact is distributed under the Apache License 2.0. The
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) base repository and the
[quantized source repository](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) are also licensed
under Apache-2.0. Users remain responsible for complying with the license and applicable laws.
