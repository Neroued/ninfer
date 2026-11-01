---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model:
- Qwen/Qwen3.8-Flash-Next
- nvidia/Qwen3.8-Flash-Next-NVFP4
base_model_relation: quantized
tags:
- ninfer
- qwen4-exp
- nvfp4
- q8
- nf4
- mtp
- blackwell
- dgx-spark
- multimodal
- conversational
- cuda
model-index:
- name: Qwen3.8-Flash-Next-NInfer
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: GPQA-Diamond
      type: gpqa_diamond
    metrics:
    - type: accuracy
      value: 91.79
      name: Mean sample accuracy (0-shot, rule)
    source:
      url: https://github.com/Neroued/ninfer/blob/master/model-cards/Qwen3.8-Flash-Next-NInfer/evaluation.json
      name: NInfer / EvalScope 1.10.0
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: AIME 2026
      type: aime26
    metrics:
    - type: accuracy
      value: 97.78
      name: Mean sample accuracy (0-shot, rule)
    source:
      url: https://github.com/Neroued/ninfer/blob/master/model-cards/Qwen3.8-Flash-Next-NInfer/evaluation.json
      name: NInfer / EvalScope 1.10.0
  - task:
      type: image-text-to-text
      name: Image Text to Text
    dataset:
      name: ERQA
      type: erqa
    metrics:
    - type: accuracy
      value: 68.0
      name: Mean sample accuracy (0-shot, rule)
    source:
      url: https://github.com/Neroued/ninfer/blob/master/model-cards/Qwen3.8-Flash-Next-NInfer/evaluation.json
      name: NInfer / EvalScope 1.10.0
  - task:
      type: image-text-to-text
      name: Image Text to Text
    dataset:
      name: RealWorldQA
      type: real_world_qa
    metrics:
    - type: accuracy
      value: 85.75
      name: Mean sample accuracy (0-shot, rule)
    source:
      url: https://github.com/Neroued/ninfer/blob/master/model-cards/Qwen3.8-Flash-Next-NInfer/evaluation.json
      name: NInfer / EvalScope 1.10.0
  - task:
      type: image-text-to-text
      name: Image Text to Text
    dataset:
      name: MMMU-Pro (standard, 10 options)
      type: mmmu_pro
    metrics:
    - type: accuracy
      value: 80.81
      name: Mean sample accuracy (0-shot, rule)
    source:
      url: https://github.com/Neroued/ninfer/blob/master/model-cards/Qwen3.8-Flash-Next-NInfer/evaluation.json
      name: NInfer / EvalScope 1.10.0
---

# Qwen3.8-Flash-Next for NInfer

A mixed-precision NInfer v3 artifact of
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), combining the BF16 checkpoint
with calibrated experts and MTP weights from
[NVIDIA's NVFP4 checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4).
It contains Text, Vision, native MTP and an optimized full-vocabulary draft head.

Use it with [NInfer](https://github.com/Neroued/ninfer) for CLI generation or OpenAI-/Anthropic-compatible
serving. The native architecture is Qwen4Exp: Gated DeltaNet, sparse attention, gated residual streams,
MoE and PLE embeddings. The artifact carries its tokenizer, processor resources and the maintained
Flash Next chat template. `.ninfer` files are consumed directly by NInfer.

## Download

```bash
hf download neroued/Qwen3.8-Flash-Next-NInfer \
  --include 'qwen3_8_flash_next.ninfer*' \
  --local-dir models/Qwen3.8-Flash-Next-NInfer
```

The complete artifact is **101.00 GiB** across four files:

```text
qwen3_8_flash_next.ninfer
qwen3_8_flash_next.ninfer.part-0001
qwen3_8_flash_next.ninfer.part-0002
qwen3_8_flash_next.ninfer.part-0003
```

Keep all four in one directory and pass only `qwen3_8_flash_next.ninfer` to NInfer. These are storage
parts; loading selects any required rank shards. File sizes and the artifact identity are recorded
in [artifact-manifest.json](artifact-manifest.json).

## Representation

| Weights | Stored representation |
|---|---|
| Main routed experts | Imported NVIDIA NVFP4, with calibrated scales |
| Large GDN projections; Main QSA query/gate and output | Q8 A16 |
| MTP routed experts | Imported block-FP8 |
| PLE table | NF4 |
| Main head, shared experts, HC, Vision and other retained projections | BF16; source FP32 control fields remain FP32 |
| Optimized MTP head | Additional full-vocabulary Q4 A16; Main head stays BF16 |

Imported NVFP4 codes/scales are retained. Additional quantization is performed by the
[`qwen3_8_flash_next` recipe](https://github.com/Neroued/ninfer/blob/master/tools/convert/official_recipes.py).
The optimized head does not crop the vocabulary. Select it with
`--spec mtp --draft-tokens 3 --proposal-head optimized`.

## Run on Spark

Build the current NInfer checkout using its
[build instructions](https://github.com/Neroued/ninfer#build). One Spark uses the
`dgx-spark-release` preset. For two Sparks, build the same revision on each node with
`-DNINFER_ENABLE_DISTRIBUTED=ON` and NCCL development dependencies installed.

From the NInfer repository root:

```bash
cp examples/configs/spark-qwen3_8-flash-next.toml ninfer.local.toml
# Set model.path to the downloaded entry file.
./build/dgx-spark-release/apps/ninfer-serve --config ninfer.local.toml
```

For two Sparks, start from
[`spark2-qwen3_8-flash-next.toml`](https://github.com/Neroued/ninfer/blob/master/examples/configs/spark2-qwen3_8-flash-next.toml),
place all artifact parts on both nodes, and edit worker paths, model paths and network settings.
Run the server once on the coordinator; it launches both workers. TP2 uses expert TP by default;
`ep = 2` selects whole-expert EP. Both examples enable Vision, FP8 KV, MTP3, the optimized head and
CUDA Graph decode, with a 262,144-token logical context limit.

For one CLI request, use `apps/ninfer` with the same config and `--prompt` or `--messages`.
See the [deployment guide](https://github.com/Neroued/ninfer/tree/master/examples/configs) for commands.

RTX 5090 can run this artifact with CPU expert offload. That path needs approximately 63.3 GiB of
Host Main-expert backing plus other runtime/system memory. On one Spark, experts reside on the GPU
and PLE is disk-backed; dual Spark shards PLE across Device memory. File size is not a runtime
memory requirement: KV/State, workspace and enabled components also affect capacity.

## Features

- Text, image/video prompts, streaming, tool calls and constrained output.
- Native MTP with 1..5 drafts, full or optimized head; enable Vision with `--vision`.
- CUDA Graph decode, prefix reuse and up to eight active requests.
- CLI, OpenAI Chat/Responses and Anthropic Messages through the same Engine.
- Thinking enabled by default. The maintained template supports `low`, `medium`, `xhigh` effort;
  `high`/`max` map to `xhigh`, `minimal` to `low`, and `none` disables thinking.

The context limit is per request. Actual shared KV/State capacity is reported at startup and depends
on the deployment and selected resources. DFlash and DFlash2 are not used by this architecture.
NInfer returns tool calls to the client; it does not execute them.

## Evaluation

Recorded on **2026-10-21–22** through NInfer's public OpenAI-compatible service and EvalScope 1.10.0.
The server used **two DGX Sparks, TP2/EP2, BF16 KV, Vision, MTP3, optimized draft head, CUDA Graphs,
prefill chunk 4096, context limit 262,144 and concurrency 8**.

| Benchmark | Accuracy | Correct / scored samples | Repeats per question |
|---|---:|---:|---:|
| GPQA-Diamond | 91.79% | 727 / 792 | 4 |
| AIME 2026 | 97.78% | 88 / 90 | 3 |
| ERQA | 68.00% | 272 / 400 | 1 |
| RealWorldQA | 85.75% | 656 / 765 | 1 |
| MMMU-Pro (standard, 10 options) | 80.81% | 1,398 / 1,730 | 1 |

Scores are mean accuracy across generated samples, not pass@k. GPQA contains 198 questions with
four samples each; AIME26 contains 30 questions with three samples each. The other datasets use one
sample per question. MMMU-Pro uses the standard 10-option format.

All jobs completed with no failed samples. Two GPQA responses reached the output limit and had no
parsed answer; one MMMU-Pro response was unparsed. They remain in the scored totals. Original dataset
labels and scorer outputs are retained. Vision results were collected before the subsequent JPEG
decoder fix.

Generation used thinking with `xhigh` effort, temperature 1.0, top-p 0.95, top-k 20, min-p 0, zero
presence/frequency penalties and a 131,072-token output budget. Evaluation used 0-shot prompts,
rule-based scoring and evaluator seed 42. [evaluation.json](evaluation.json) preserves the run
identity, configuration, counts and diagnostics.

## Sources and license

- BF16 parameters and frontend resources: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
- Calibrated experts and MTP: [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4).
- Conversion and runtime: [NInfer](https://github.com/Neroued/ninfer).

The source model uses the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
The NVIDIA checkpoint lists the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
and the Qwen source terms. Those source weight terms apply to this derivative. NInfer's Apache-2.0
software license is separate from the model licenses.
