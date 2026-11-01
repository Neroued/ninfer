# capy-flash documentation

Start with the [project README](../README.md): which model fits your machine, `install.sh`, the launchers and the
benchmarks. capy-flash is a fork of [NInfer](https://github.com/Neroued/ninfer); the guides marked *upstream* are
NInfer's and still apply to the shared engine, with the capy-flash additions linked from them.

## capy-flash guides

| Document | Purpose |
|---|---|
| [capy-flash serving](serving.md#capy-flash-serving) | launchers, the four profiles in full, which `install.sh --model` runs which, RAM per GPU count, chat template, CPU expert threads, client setup (Claude Code, Codex CLI, OpenCode) |
| [Environment variables](capy-flash/diagnostics.md) | production settings, A/B switches, dumps and validation aids of the engine |
| [Release tarball](../tools/release/README-release.md) | contents of the prebuilt release, requirements, launcher commands |
| [Two-GPU performance](performance/two-gpu.md) | Qwen3.8-27B on 2x RTX 5070 Ti against upstream's RTX 5090 runs (v0.2.2) |
| [Development log](capy-flash/development-log.md) | history of the capy-flash work and the `Tnn` tags in the sources (not current usage) |
| [ninfer-tp2 README](https://github.com/ValerioDolci/capy-flash/blob/v0.4.9/README.md) | the historical README of ninfer-tp2 v0.4.x, at its last tag (not current usage) |

## User guides

| Document | Purpose |
|---|---|
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, tool calls and every `ninfer-serve` option |
| [CLI](cli.md) | upstream: the one-request `ninfer` CLI (text, chat history, image/video input, sampling, MTP, two GPUs); Flash-Next is served with `ninfer-serve` |
| [Performance](performance.md) | upstream: RTX 5090 measurement coverage, per-model serving results, methodology, and publication rules; capy-flash's Flash-Next numbers are in the [README](../README.md#benchmarks) |
| [Weight conversion](weight-conversion.md) | upstream: official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [Perplexity](perplexity.md) | upstream: fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Model artifacts

capy-flash artifacts (what `install.sh` and the launchers use):

| Model | Weights | Download | `install.sh --model` |
|---|---|---|---|
| Qwen3.8-Flash-Next | NVIDIA's NVFP4 experts, text and MTP | [Hugging Face](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-NVFP4-capy-flash) | `flash-next` (default) |
| Qwen3.8-Flash-Next with images | the same plus the Q8 vision tower | [Hugging Face](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-NVFP4-Vision-capy-flash) | none: download by hand, serve with `--vision` |
| Qwen3.8-Flash-Next, 64 GB edition | ISTA-DASLab's GSQ-RCO i-quant experts | [Hugging Face](https://huggingface.co/Feyd89/Qwen3.8-Flash-Next-GSQ-RCO-capy-flash) | `flash-next-64gb` |
| Qwen3.8-27B, one GPU | ISTA-DASLab's 3-bit GSQ weights plus the official MTP layer, vision tower | [Hugging Face](https://huggingface.co/Feyd89/Qwen3.8-27B-GSQ-3bit-capy-flash) | `27b-16gb` |
| Qwen3.8-27B QUASAR-QAT, two GPUs | `nvfp4`, vision tower | [Hugging Face](https://huggingface.co/Feyd89/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer) ([model card](../model-cards/Qwen3.8-27B-QUASAR-QAT-nvfp4-NInfer/README.md)) | `27b` |

Upstream NInfer artifacts (documented upstream for one large GPU; the Qwen3.8-27B `nvfp4` one also runs with `serve-tp2.sh` on two 16 GB boards):

| Model | Weights | Download | Versioned model card source |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | [model card](../model-cards/Qwen3.6-27B-NInfer/README.md) |
| Qwen3.6-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) |
| Qwen3.8-27B | `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| Qwen3.8-27B | `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Maintainer references

The active references under [`maintainer/`](maintainer/) record current architecture, model,
artifact, and maintenance contracts. These files are not additional user workflows or installed
API documentation.

[Engine architecture](maintainer/engine-architecture.md) is the single top-level reference. The
other references own narrower contracts:

| Document | Responsibility |
|---|---|
| [Engine architecture](maintainer/engine-architecture.md) | model/config/weight ownership, loading-to-execution flow, requests, scheduling, transactions and graphs |
| [Build system](maintainer/build-system.md) | CMake targets, explicit source ownership, CUDA compilation boundaries, presets and developer configuration |
| [Artifact container](maintainer/artifact-container.md) | v3 directory, objects, logical bindings, Uses, resources and file framing/sharding |
| [Numeric formats](maintainer/tensor-formats.md) | represented values, codes/scales, conversion arithmetic and numerical interpretation |
| [Storage layouts](maintainer/storage-layouts.md) | packing, plane offsets, padding, encoded sizes and view addressing |
| [Qwen3.5 model](maintainer/qwen3_5-model.md) | Dense mathematics, instance config, logical parameters, MTP, Vision and state semantics |
| [Qwen4Exp model](maintainer/qwen4-exp-model.md) | Qwen3.8-Flash-Next mathematics (hyper-connections, QSA, PLE, MTP, the dense FFN), its artifact names and loading residency, and the one Text forward every model runs (Qwen3.8-27B included) |
| [DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | candidate selection, retention, materialization and Device/Host checkpoint policy |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [Active KV in RAM](maintainer/active-kv-in-ram.md) | `--kv-ram-active`: Flash-Next text K/V in pinned host memory, the QSA staging region and block cache, invalidation, cost |
| [Token embedding in host memory](maintainer/embedding-host.md) | `--embedding-host`: the HostMapped placement, its mapped pinned allocation, CUDA Graph rules, cost and reporting |
| [Two-GPU tensor parallelism](maintainer/tensor-parallel.md) | `--tp 2` weight sharding, rank roles, all-reduce transports, two-device CUDA Graphs, mirrored state, tests and limits |
| [Merging upstream](maintainer/upstream-merge.md) | where the two-GPU layer lives, the hooks it keeps in upstream's files, typical conflicts, the mandatory gate, adding a GPU or a model |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Linear benchmark](maintainer/linear-benchmark.md) | pure Linear measurement, metrics and suites |
| [Linear tuning and reports](maintainer/linear-tuning.md) | tuning ranges, priority points, dispatch tradeoffs and final performance report format |

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.
