---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: mit
base_model:
- zai-org/GLM-5.3-Flash
- nvidia/GLM-5.3-Flash-NVFP4
base_model_relation: quantized
tags:
- ninfer
- glm5next
- nvfp4
- q8
- mtp
- blackwell
- dgx-spark
- multimodal
- conversational
- cuda
---

# GLM-5.3-Flash for NInfer

A mixed-precision NInfer v3 artifact of
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), converted from
[NVIDIA's NVFP4 checkpoint](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4).
It includes Text, Vision, native MTP and an additional full-vocabulary Q4 draft head.

The native NInfer architecture is GLM5Next, combining KDA linear attention, Kpool sparse MLA,
mHC and MoE. The artifact includes the source tokenizer, GLM chat template and processor resources.
Use it directly with [NInfer](https://github.com/Neroued/ninfer) for CLI generation or
OpenAI-/Anthropic-compatible serving on **two DGX Sparks**.

## Download

```bash
hf download neroued/GLM-5.3-Flash-NInfer \
  --include 'glm5_3_flash.ninfer*' \
  --local-dir models/GLM-5.3-Flash-NInfer
```

The complete artifact is **178.33 GiB** across six files:

```text
glm5_3_flash.ninfer
glm5_3_flash.ninfer.part-0001
glm5_3_flash.ninfer.part-0002
glm5_3_flash.ninfer.part-0003
glm5_3_flash.ninfer.part-0004
glm5_3_flash.ninfer.part-0005
```

Keep all parts beside the entry file and place the complete set on each node. Pass only
`glm5_3_flash.ninfer` to NInfer; rank-specific weight slicing happens during loading.
File sizes and the artifact identity are in [artifact-manifest.json](artifact-manifest.json).

## Representation

| Weights | Stored representation |
|---|---|
| Main routed experts and first three dense MLP layers | Imported NVIDIA NVFP4, with calibrated scales |
| KDA Q/K/V/O, MLA output and shared-expert projections | Q8 A16 |
| MTP routed experts | Block-FP8 converted from source BF16 |
| Main head, embedding, mHC, Vision and other retained parameters | BF16; source FP32 KDA/control fields remain FP32 |
| Optimized MTP head | Additional full-vocabulary Q4 A16; Main head stays BF16 |

The [`glm5next` recipe](https://github.com/Neroued/ninfer/blob/master/tools/convert/official_recipes.py)
retains imported Main NVFP4 codes/scales and performs the additional quantization above.
The optimized draft head does not crop the vocabulary.

## Run on two Sparks

Build the current NInfer revision on both nodes with NCCL development dependencies installed:

```bash
cmake --preset dgx-spark-release -DNINFER_ENABLE_DISTRIBUTED=ON
cmake --build build/dgx-spark-release -j
```

Full dependency and checkout instructions are in the [build guide](https://github.com/Neroued/ninfer#build).
From the repository root, prepare the deployment config:

```bash
cp examples/configs/spark2-glm5_3-flash.toml ninfer.local.toml
# Edit coordinator/worker artifact paths, worker executable paths and network addresses.
./build/dgx-spark-release/apps/ninfer-serve --config ninfer.local.toml
```

The [example](https://github.com/Neroued/ninfer/blob/master/examples/configs/spark2-glm5_3-flash.toml)
uses TP2 with attention DP2, FP8 KV, Vision, MTP3, the optimized head, CUDA Graph decode and up to
eight active requests. Its logical context limit is **1,048,576 tokens**. The coordinator launches
both workers, using SSH for the remote node. Clients connect to its single HTTP endpoint.

Experts use TP by default (`ep = 1`); set `ep = 2` to select whole-expert EP while retaining TP2
and attention DP2 elsewhere. Keep all six artifact files on each node. The complete artifact does
not fit on one Spark. Runtime allocations also include workspace and the shared KV/State pool;
read the startup capacity estimate for the selected configuration.

For CLI use, run `apps/ninfer --config ninfer.local.toml --prompt "..." --max-tokens 2048` on the
coordinator. Detailed startup and network instructions are in the
[deployment guide](https://github.com/Neroued/ninfer/tree/master/examples/configs).

## Model options

- Thinking stays enabled. Supported reasoning efforts are `low`, `high` and `max`; the template
  supplies the default. `--no-thinking` and effort `none` are unsupported.
- Native MTP supports 1..5 drafts. `--proposal-head optimized` selects the Q4 draft head;
  the target head stays BF16. Draft count can be adjusted for workload and concurrency.
- Latent KV uses `--kv-dtype bf16` or `fp8`. The Kpool indexer uses FP8 by default;
  `NINFER_KPOOL_INDEXER_BF16=1` selects BF16 indexer storage independently.
- Vision, tool calls, constrained output, streaming and prefix reuse use the same public Engine
  and protocol endpoints as text generation. Enable Vision at startup for image/video input.
- GLM uses its own chat template and tool framing; retain the included GLM frontend resources.

DFlash and DFlash2 are not used by this architecture. NInfer returns generated tool calls to the
client; it does not execute them.

## Sources and license

The primary conversion source is
[nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4), revision
`da920bb0b9f4a06727223a349e55468e38352348`, based on
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).
The model weights use the upstream [MIT license](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE).
[NInfer](https://github.com/Neroued/ninfer) is Apache-2.0-licensed software.
