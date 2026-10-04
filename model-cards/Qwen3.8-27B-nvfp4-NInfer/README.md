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
model-index:
  - name: Qwen3.8-27B-nvfp4-NInfer
    results:
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: IFBench
          type: ifbench
        metrics:
          - type: accuracy
            value: 77.00
            name: Prompt-level strict (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: text-generation
          name: Text Generation
        dataset:
          name: AIME 2025
          type: aime25
        metrics:
          - type: accuracy
            value: 96.67
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
            value: 96.67
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
            value: 90.40
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: image-text-to-text
          name: Image Text to Text
        dataset:
          name: ERQA
          type: erqa
        metrics:
          - type: accuracy
            value: 66.25
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
      - task:
          type: image-text-to-text
          name: Image Text to Text
        dataset:
          name: RealWorldQA
          type: real_world_qa
        metrics:
          - type: accuracy
            value: 83.53
            name: Accuracy (0-shot, rule)
        source:
          url: https://github.com/Neroued/ninfer/tree/master/eval
          name: NInfer EvalScope 1.9.0
---

# Qwen3.8-27B NVFP4 for NInfer

This model card is the version-controlled source for
[neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer).

The repository contains a mixed NVFP4/FP8 representation of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). It combines the official BF16 checkpoint
with the fixed packed Text weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) in the native
[NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format. The artifact is intended
only for NInfer; it is not a Transformers checkpoint, Safetensors distribution, or GGUF file.

The artifact uses the Qwen3.5 Dense architecture. Text layers 0–55 use NVFP4 MLP weights,
while the token embedding, attention input/output
projections, GDN Q/K/V/Z and output projections, full output head, and Text layers 56–63 MLP weights
use row-scaled FP8. Control weights use BF16, with separate MTP, Vision and DFlash2 weights.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_nvfp4.ninfer` |
| Size | 23,719,715,844 bytes (22.09 GiB) |
| SHA-256 | `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82` |
| Container version | 3 |
| Architecture | `Qwen3_5ForCausalLM` |
| Public model name | `qwen3.8-27b` |
| Chat template | [qwen3_8.jinja](https://github.com/Neroued/ninfer/blob/98dada0e03cb073fd07f905400b5904bc6e82759/tools/chat_templates/qwen3_8.jinja); override with `--chat-template FILE` |
| Template defaults | thinking on; effort `xhigh`; closed-turn reasoning retained |
| Stored objects | 1,246 (1,240 tensors and 6 resources) |
| NVFP4 tensors | 112 |
| Row-scaled FP8 tensors | 146 |

The file contains Text, Vision, MTP, DFlash2, the optimized proposal head and frontend resources.
Vision and speculative weights are loaded only when selected at startup. Source-derived NVFP4 and
FP8 words are preserved without decode and requantization; only the official BF16 token embedding
is encoded locally as row-scaled FP8.

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  '74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82' \
  'qwen3_8_27b_nvfp4.ninfer' | sha256sum --check
```

This release includes the complete DFlash2 companion weights from
`z-lab/Qwen3.8-27B-DFlash2` at revision
`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. Select
`--spec dflash2 --draft-tokens 7 --lm-head-draft`; draft counts 1..15 are supported.
DFlash2 requires the runtime revision listed below. Existing performance and evaluation tables
retain their stated MTP configurations and revisions.

## Requirements

- [NInfer](https://github.com/Neroued/ninfer) revision
  [`04350ba9`](https://github.com/Neroued/ninfer/commit/98dada0e03cb073fd07f905400b5904bc6e82759)
  or later, built from source;
- 64-bit Linux;
- NVIDIA GeForce RTX 5090 (`sm_120a`);
- CUDA Toolkit 13.1 or newer.

Already have the official v2 file? [Upgrade it locally](https://github.com/Neroued/ninfer/blob/master/docs/weight-conversion.md#upgrade-an-existing-v2-artifact)
without downloading the weights again.

NInfer does not provide an install target or packaged binary. See the
[repository README](https://github.com/Neroued/ninfer#quick-start) for source-build dependencies.

## Download and run a CLI example

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models

./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
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
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 \
  --port 8080 \
  --max-context 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --host-cache-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. The shared Device KV pool takes the memory the
model and runtime leave free; it admits two active requests when their combined completion
reservations fit, and either request may use the full pool while running alone. Pages no active
request holds are the Device prefix cache, and 8 GiB of pinned Host memory holds cached KV blocks
and state snapshots beyond it.

See the [HTTP serving guide](https://github.com/Neroued/ninfer/blob/master/docs/serving.md) for the
API surface and the [prefix cache reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/hybrid-prefix-cache.md)
for cache and admission semantics.

## Supported use

The artifact supports:

- text generation in thinking and non-thinking modes;
- image, multi-image, video, and mixed multimodal messages;
- MTP speculative decoding with draft windows from one to five;
- DFlash2 with draft windows from one to fifteen using the included
  DFlash2 companion weights (`--spec dflash2 --draft-tokens 7`, optionally `--lm-head-draft`);
- BF16, INT8, FP8, NVFP4, and K8V4 KV cache;
- CUDA Graph decode and compatible-prefix reuse;
- startup-bounded small-scale concurrent serving with true batched decode;
- the NInfer CLI;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages serving.

## Performance

Measured on September 28–29, 2026 with NInfer revision
[`7f6aafed`](https://github.com/Neroued/ninfer/commit/7f6aafedb5f20200def820cfe51ab81c09c20eeb),
one RTX 5090, driver 617.14, and CUDA 13.4 compile/runtime/driver API.
These serving runs use FP8 E4M3 row-256 KV, CUDA Graphs, a 1,024-token prefill chunk, disabled
prefix reuse, and temperature 0.6 / top-p 0.95 / top-k 20 / min-p 0 / presence penalty 1.0 /
frequency penalty 0. MTP0 has a 262,144-token context ceiling; MTP3 uses 131,072 tokens per
request, three draft tokens, the optimized proposal head, and automatic shared KV capacity.

### Concurrent MTP=3 corpus makespan

Each C is one complete 75-request corpus, with three reasoning and twelve cross-scenario fixtures,
five seeds per fixture, and a fixed shuffled send order. Makespan includes prefill, decode,
admission waits, transitions, and drain; actual output lengths vary.

| C | Requests | Computed prefill tokens | Decode tokens | Makespan (s) | Requests/s | Corpus prefill (tok/s) | Corpus decode (tok/s) | Avg batch | MTP acceptance |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 75 | 15,460 | 693,701 | 4,115.22 | 0.0182 | 3.8 | 168.6 | 1.00 | 59.6% |
| 2 | 75 | 15,460 | 709,989 | 2,394.23 | 0.0313 | 6.5 | 296.5 | 1.90 | 58.9% |
| 4 | 75 | 15,460 | 722,202 | 1,640.17 | 0.0457 | 9.4 | 440.3 | 3.13 | 58.5% |
| 8 | 75 | 15,460 | 708,589 | 1,443.37 | 0.0520 | 10.7 | 490.9 | 3.52 | 59.5% |

All 300 requests completed without request, CUDA, or allocation errors. At C=1/2/4/8,
automatic KV capacity is 131,072 / 262,144 / 253,632 / 225,024 tokens. C=8 has average batch 3.52
and up to five waiting requests in the sampled intervals. The resident MTP3 weights occupy
19.729 GiB, and the workspace arena is 243.3 MiB.

### Long-context serving (MTP disabled)

Values are arithmetic mean ± sample standard deviation over five fixed seeds per fixture.

| Prompt tokens | Samples | Prefill phase (tok/s) | Server TTFT (ms) | Decode phase (tok/s) |
|---|---|---|---|---|
| 7,680 | 5 | 12,819.1 ± 16.8 | 602.7 ± 1.1 | 74.1 ± 0.3 |
| 64,512 | 5 | 8,658.2 ± 44.8 | 7,487.4 ± 37.7 | 68.2 ± 0.2 |
| 130,048 | 5 | 6,198.5 ± 19.6 | 21,055.7 ± 67.2 | 62.5 ± 0.3 |
| 260,096 | 5 | 4,016.4 ± 10.4 | 64,910.0 ± 171.8 | 53.4 ± 0.6 |

### MTP=3 single-request long-reasoning decode

The C=1 corpus supplies these phase statistics, with five samples per reasoning fixture.

| Fixture | Samples | Completion tokens | Decode phase (tok/s) | MTP3 acceptance | MTP3 tokens/round |
|---|---|---|---|---|---|
| `long_decode_aime26_01` | 5 | 1,559.4 ± 727.2 | 207.4 ± 3.6 | 77.2% ± 1.8% | 3.32 ± 0.05 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 161.7 ± 3.8 | 57.2% ± 2.1% | 2.72 ± 0.06 |
| `long_decode_aime26_30` | 5 | 37,978.0 ± 7,474.4 | 170.0 ± 2.3 | 59.9% ± 1.5% | 2.80 ± 0.05 |

### MTP=3 single-request cross-scenario decode

Each category pools three fixtures × five seeds. Values are mean ± sample standard deviation.

| Category | Samples | Decode phase (tok/s) | MTP3 acceptance | MTP3 tokens/round |
|---|---|---|---|---|
| Code | 15 | 205.3 ± 9.5 | 76.8% ± 5.1% | 3.30 ± 0.15 |
| Story | 15 | 132.6 ± 13.1 | 37.6% ± 7.1% | 2.13 ± 0.21 |
| Translation | 15 | 202.9 ± 12.3 | 75.2% ± 6.6% | 3.26 ± 0.20 |
| Structured | 15 | 231.7 ± 10.6 | 90.6% ± 5.6% | 3.72 ± 0.17 |

All five AIME 15 samples reach the 65,536-token output budget. The C=1 corpus contains
40 stop-token and 35 output-limit results; all are retained in the statistics. These measurements
do not score answer accuracy or task completion.

The [full results and reproduction commands](https://github.com/Neroued/ninfer/blob/master/docs/performance/qwen3.8-27b.md)
also cover DFlash2 K=7, MTP3 decode saturation, and completion outcomes.

## Evaluation

The artifact was evaluated through NInfer's OpenAI-compatible serving route with thinking enabled,
MTP=3, and INT8 group-64 KV. EvalScope 1.9.0 used 0-shot prompts, rule-based scoring, and one sample
per problem with temperature 1.0, top-p 0.95, top-k 20, presence penalty 0.0, and seed 42. The text
suite ran at a 252,928-token context limit; the multimodal suite ran with `--vision` at a
81,920-token limit.

| Benchmark | NInfer NVFP4 | Correct / total | Official Qwen3.8-27B BF16 |
|---|---:|---:|---:|
| IFBench (prompt-level strict) | 77.00% | 231 / 300 | 79.5 |
| AIME 2025 | 96.67% | 29 / 30 | — |
| AIME 2026 | 96.67% | 29 / 30 | — |
| GPQA-Diamond | 90.40% | 179 / 198 | 89.2 |
| ERQA | 66.25% | 265 / 400 | 65.5 |
| RealWorldQA | 83.53% | 639 / 765 | 85.9 |

All 1,723 configured samples completed and were scored. IFBench additionally reports 80.50%
instruction-level strict, 80.33% prompt-level loose, and 83.50% instruction-level loose. These are
single-sample results, not pass@k.

The official Qwen3.8-27B BF16 figures come from the
[upstream model card](https://huggingface.co/Qwen/Qwen3.8-27B); its sampling settings and IFBench
metric level are not stated there, so the last column is not a same-protocol comparison. The NVFP4
deltas stay within ±2.5 points on the four overlapping benchmarks, and the upstream card reports no
AIME results.

## Limits

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
| Conversion recipe | `qwen3_8_27b_nvfp4` |
| Embedding encoder | `fp8_row_maxabs` |
| Converter repository | `https://github.com/Neroued/ninfer` |
| Minimum runtime revision | `98dada0e03cb073fd07f905400b5904bc6e82759` |
| Ranking input SHA-256 | `c692dc76388132c910547589b4fb4a0503fbd6ad50aaac6a509bbcb192a8afa5` |

The artifact identity, summarized object inventory, and conversion provenance are published in
[`artifact-manifest.json`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/blob/main/artifact-manifest.json).
The exact storage contract is maintained in the
[v3 container reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/artifact-container.md).

## License

This NInfer artifact is distributed under the Apache License 2.0. The
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) base repository and the
[quantized source repository](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) are also licensed
under Apache-2.0. Users remain responsible for complying with the license and applicable laws.
