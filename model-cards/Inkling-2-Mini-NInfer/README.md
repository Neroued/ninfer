---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: thinkingmachines/Inkling-2-Mini
base_model_relation: quantized
tags:
- ninfer
- inkling2
- nvfp4
- mxfp8
- dspark
- blackwell
- rtx-5090
- multimodal
- conversational
- cuda
model-index:
- name: Inkling-2-Mini-NInfer
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: IFBench
      type: ifbench
    metrics:
    - type: prompt_level_strict
      value: 71.33
      name: Prompt-level strict accuracy (0-shot, rule)
    - type: prompt_level_loose
      value: 76.0
      name: Prompt-level loose accuracy (0-shot, rule)
    source:
      url: https://github.com/Neroued/ninfer/blob/master/model-cards/Inkling-2-Mini-NInfer/evaluation.json
      name: NInfer / EvalScope 1.10.0
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: GPQA-Diamond
      type: gpqa_diamond
    metrics:
    - type: accuracy
      value: 87.88
      name: Accuracy (0-shot, rule)
    source:
      url: https://github.com/Neroued/ninfer/blob/master/model-cards/Inkling-2-Mini-NInfer/evaluation.json
      name: NInfer / EvalScope 1.10.0
  - task:
      type: image-text-to-text
      name: Image Text to Text
    dataset:
      name: MMMU-Pro (standard, 10 options)
      type: mmmu_pro
    metrics:
    - type: accuracy
      value: 75.66
      name: Accuracy (0-shot, rule)
    source:
      url: https://github.com/Neroued/ninfer/blob/master/model-cards/Inkling-2-Mini-NInfer/evaluation.json
      name: NInfer / EvalScope 1.10.0
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: OpenAI MRCR (8-needle, 262K-524K)
      type: openai_mrcr
    metrics:
    - type: score
      value: 51.52
      name: Overall MRCR score (8-needle, 262K-524K)
    source:
      url: https://github.com/Neroued/ninfer/blob/master/model-cards/Inkling-2-Mini-NInfer/evaluation.json
      name: NInfer / EvalScope 1.10.0
---

# Inkling-2-Mini for NInfer

A native NInfer v3 artifact of
[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini),
preserving the checkpoint's native NVFP4_MUL and MXFP8 MoE expert encodings with offline W4A4
activation calibration, the 23-layer PE-Lang Vision encoder, and the 3-layer DSpark speculative
draft model (`block_size=8`, recurrent Markov RNN head).

Use it with [NInfer](https://github.com/Neroued/ninfer) for CLI generation or OpenAI-/Anthropic-compatible
HTTP serving on RTX 5090 (`sm_120a`). The native architecture is `Inkling2ForCausalLM`: a looped
24-physical-layer / 48-logical-layer Block-AttnRes MoE backbone (6 prelude, 12 core layers × 3
passes with rank-64 message gates, 6 coda) with hybrid local sliding-window (`W=1024`) and global
relative-bias attention, causal short convolutions, and 128 routed experts (top-8 sigmoid). The
artifact embeds the tokenizer, chat template, generation config, and image processor config.
`.ninfer` files are consumed directly by NInfer.

## Download

```bash
hf download neroued/Inkling-2-Mini-NInfer \
  inkling2_mini.ninfer --local-dir models
```

The artifact is a single **18.89 GiB** (`20,281,047,044` bytes) v3 file:

```bash
sha256sum -c SHA256SUMS
```

File size, SHA-256, component geometry, and object inventory are recorded in
[artifact-manifest.json](artifact-manifest.json) and [SHA256SUMS](SHA256SUMS).

## Representation

| Weights | Stored representation | Execution policy |
|---|---|---|
| Routed MoE experts (layers 0–19, 40 tensors) | Native `nvfp4_mul` (`batched_block_scale_k16_m128x4_v1`) + offline `max` W4A4 activation multipliers (`per_expert_down`) | `AllowA4` (132 uses) |
| Routed MoE experts (layers 20–23, 8 tensors) | Native `mxfp8` (`row_group_scale_v1`) | `AllowA8` (12 uses) |
| Embeddings, attention/short-conv/output projections, loop controllers, output head, 23-layer PE-Lang Vision encoder, 3-layer DSpark + folded Markov RNN head | BF16 (`contiguous_le_v1`, 752 tensors) | `A16Only` |
| Norms, routers/biases, short-conv taps, relative bias tables, AttnRes queries, clip scales, calibration multipliers | FP32 (`contiguous_le_v1`, 174 tensors) | Retained precision |

Native `nvfp4_mul` and `mxfp8` codes and scales are imported without requantization via the
[`inkling2` recipe](https://github.com/Neroued/ninfer/blob/master/tools/convert/official_recipes.py).
Offline `max` activation calibration attaches per-visit/per-expert FP32 multipliers (`88` tensors,
`min_expert_elements=36864`) enabling Blackwell W4A4 routed MoE execution across layers 0–19.
DSpark shares the text embedding, embedding norm, and output head bindings, with its Markov RNN
unembed projection pre-folded offline in FP64.

## Run on RTX 5090

Build NInfer from source using its
[build instructions](https://github.com/Neroued/ninfer#build) with the `rtx5090-release` preset.
From the NInfer repository root:

```bash
cp examples/configs/rtx5090-inkling2-mini.toml ninfer.local.toml
# Set model.path to the downloaded inkling2_mini.ninfer file.
./build/rtx5090-release/apps/ninfer-serve --config ninfer.local.toml
```

[`rtx5090-inkling2-mini.toml`](https://github.com/Neroued/ninfer/blob/master/examples/configs/rtx5090-inkling2-mini.toml)
enables Vision (`1 GiB` window budget), FP8 full-attention KV (`--kv-dtype fp8`), BF16
sliding-window KV (`--swa-kv-dtype bf16`), DSpark speculative decoding (`--spec dspark
--draft-tokens 7 --proposal-head full`), CUDA Graph decode, a 4,096-token prefill chunk,
`max_context = 131072`, `max_concurrency = 4`, and public model alias `inkling2-mini`.
Because this artifact's embedded `metadata.name` is unset, omitting `--model-id` advertises
`Inkling2ForCausalLM`.

For a single CLI request:

```bash
./build/rtx5090-release/apps/ninfer models/inkling2_mini.ninfer \
  --prompt "Explain looped transformer attention briefly." \
  --max-context 16384 --max-tokens 2048 \
  --kv-dtype fp8 --swa-kv-dtype bf16 \
  --spec dspark --draft-tokens 7 --proposal-head full
```

Resident device weights are **17.58 GiB** (`MTP0`, no Vision) and **18.27 GiB** (`DSpark K=7`, no
Vision), leaving ~10.3–10.6 GiB for the automatic shared KV/State Context pool on a 32 GiB RTX
5090. For high-concurrency serving (e.g. `C=16..32`), set `--swa-kv-dtype fp8` to halve the
per-lane sliding-window KV footprint. Inkling2 uses device-resident Context storage (`host_capacity`
defaults to `0 B`).

## Features

- Text and image-only Vision prompts (23-layer PE-Lang ViT, 20×20 patches with 2×2 spatial merge;
  video input is rejected explicitly), streaming, tool calls (`<|content_invoke_tool_json|>`), and
  constrained output (GBNF, regex, choice, JSON object/schema).
- Native DSpark speculative decoding with `1..7` draft tokens (default `7`), adaptive verification
  length, and `--proposal-head full`.
- CUDA Graph decode, exact prefix reuse, and configurable resident-request capacity.
- CLI, OpenAI Chat Completions / Responses, Anthropic Messages, and offline perplexity scoring
  through the same Engine.
- Thinking enabled by default at effort `high` (`0.9`). Supports `none` (`0.0`, disables thinking),
  `minimal` (`0.1`), `low` (`0.2`), `medium` (`0.7`), `high` (`0.9`), `xhigh`/`max` (`0.99`), or
  continuous numeric values in `[0.0, 0.99]`.

The context limit is per request (up to 1,048,576 positions in model config). Actual shared
KV/State capacity is reported at startup and depends on the selected KV dtypes and enabled
components. NInfer returns tool calls to the client; it does not execute them.

## Performance

Measured on November 27, 2026 on one NVIDIA GeForce RTX 5090 (SM 12.0, 32 GiB, CUDA 13.4, NInfer
revision `b2514a7ae`) with FP8 full-attention KV, BF16 SWA KV, DSpark `K=7` (adaptive), CUDA
Graphs, 4,096-token prefill chunk, and Vision/prefix reuse disabled.

Copy-edit decode saturation (`10,910`-token verbatim C++ prompt, `8,192`-token output budget,
thinking disabled):

| Concurrency | C=1 | C=2 | C=4 | C=8 | C=16 |
|---|---:|---:|---:|---:|---:|
| Steady decode (tok/s) | 568.8 | 970.2 | 1,711.6 | 2,802.3 | 4,360.8 |
| Full-batch step (ms) | 13.37 | 15.54 | 17.73 | 21.60 | 27.81 |
| DSpark acceptance | 94.6% | 94.5% | 94.1% | 94.1% | 94.3% |

A supplementary `C=32` copy-edit wave with `--swa-kv-dtype fp8` sustained **6,439.6 tok/s** across
28 full-length active streams (peak 1s interval **6,804.0 tok/s**, full-batch `C=32` step
**34.56 ms**).

Single-request Long NIAH context profile (mean across 5 seeds, `128`-token output budget):

| Prompt tokens | DSpark K=7 prefill (tok/s) | DSpark K=7 decode (tok/s) | MTP0 decode (tok/s) | Speedup vs MTP0 |
|---:|---:|---:|---:|---:|
| 8,142 | 27,526.6 | 388.5 | 180.4 | 2.15× |
| 68,362 | 20,632.7 | 324.3 | 161.8 | 2.00× |
| 137,668 | 16,083.0 | 296.0 | 143.8 | 2.06× |
| 275,395 | 10,945.2 | 286.4 | 121.1 | 2.36× |

On the 75-request fixed serving corpus (`131,072` context limit), DSpark `K=7` reaches **352.0
tok/s at C=1** and **1,224.5 tok/s at C=8** (3.16× makespan speedup). See the
[complete serving report](https://github.com/Neroued/ninfer/blob/master/docs/performance/inkling2-mini.md)
for per-scenario breakdowns, completion outcomes, and reproduction commands.

## Evaluation

Recorded on **2026-11-26–28** through NInfer's public OpenAI Chat Completions service on one
**RTX 5090** with EvalScope 1.10.0 (0-shot, rule-based scoring, evaluator seed 42, one sample per
question), FP8 full-attention KV, BF16 SWA KV, CUDA Graphs, and a 4,096-token prefill chunk.

| Benchmark | Primary metric | Score | Scored / slice | Deployment & effort |
|---|---|---:|---:|---|
| IFBench | Prompt-level strict (loose) | **71.33%** (76.00%) | 300 / 300 | DSpark K=7, `C=8`, `131K` ctx, `xhigh`, no thinking budget |
| GPQA-Diamond | Accuracy | **87.88%** | 174 / 198 | DSpark K=7, `C=8`, `196K` ctx, `high` (0.9), `131K` budget |
| MMMU-Pro (`standard, 10 options`) | Accuracy | **75.66%** | 1,309 / 1,730 | Vision + DSpark K=7, `C=8`, `163K` ctx, `high` (0.9), `98K` budget |
| OpenAI MRCR (`8-needle, 262K–524K`) | Overall MRCR score | **51.52%** | 99 / 99 | `MTP0`, `C=1`, `589K` ctx, `high` (0.9), `49K` budget |

Additional IFBench metrics: instruction-level strict **74.17%**, instruction-level loose **78.67%**.

All four benchmarks completed with zero request failures. Original EvalScope scorer outputs are
retained in the headline totals without dropping truncated or unparsed samples:

- **IFBench**: generation used `reasoning_effort="xhigh"`, `temperature=1.0`, `top_p=0.95`,
  `top_k=20`, `min_p=0.0`, `max_tokens=131072`. Thirteen prompts hit the 131,072-token context
  ceiling during `xhigh` reasoning loops (12 empty content, 1 passed loose) and remain in the
  reported 300-prompt totals; the 287 non-truncated prompts scored 74.22% prompt strict (213/287)
  and 79.09% prompt loose (227/287).
- **GPQA-Diamond & MMMU-Pro**: generation used template default `high` (`0.9`), `temperature=1.0`,
  `top_p=1.0`, `top_k=20`, `min_p=0.0`. Zero samples truncated; one MMMU-Pro response was unparsed
  and remains in the 1,730-question total.
- **OpenAI MRCR**: evaluated all 99 questions in the 8-needle `262,144–524,288` character slice
  with `--spec none --no-prefix-reuse` and `max_tokens=65536`. Twenty-six items achieved an exact
  `1.0` match; five items hit the 589,824-token context ceiling during reasoning (scored `0.0`) and
  remain in the 51.52% mean (94 non-truncated items averaged 54.26%).

[evaluation.json](evaluation.json) preserves the exact run IDs, timestamps, revisions, server
options, generation parameters, token totals, and diagnostics. A reproducible EvalScope config is
maintained in [`eval/configs/inkling2_mini.yaml`](https://github.com/Neroued/ninfer/blob/master/eval/configs/inkling2_mini.yaml).

## Sources and license

- Base checkpoint and frontend resources:
  [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
  (revision `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`).
- Conversion and runtime: [NInfer](https://github.com/Neroued/ninfer).

Both the upstream Inkling-2-Mini weights and NInfer are licensed under the
[Apache License 2.0](LICENSE).
