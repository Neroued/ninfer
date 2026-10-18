# NInfer documentation

Start with the [project README](../README.md) to build NInfer, download a published artifact, and
run the CLI or HTTP server.

## User guides

| Document | Purpose |
|---|---|
| [Startup configuration](startup-configuration.md) | shared CLI/serve TOML schema, defaults, overrides, resource controls and startup display |
| [CLI](cli.md) | text, chat-history, image/video input, output streams, sampling, MTP, and common runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, authentication, and tool calls |
| [Performance](performance.md) | RTX 5090 measurement coverage, per-model serving results, methodology, and publication rules |
| [Weight conversion](weight-conversion.md) | official recipes, custom formats and sources, conversion methods, optional components and artifact output |
| [Perplexity](perplexity.md) | fixed-corpus and custom-text causal perplexity, comparison rules, progress, and reports |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

Published artifact downloads are listed in the [project README](../README.md).
[Versioned model cards](../model-cards/) own each artifact's identity, provenance and evaluation results.

## Repository-local guides

- [Benchmarks](../bench/README.md)
- [Tests](../tests/README.md)
- [Tools](../tools/README.md)
- [Capability evaluation](../eval/README.md)

## Designs

- [Distributed execution](distributed-execution-design.md) — selected architecture for centralized
  scheduling, local/distributed executors, rank ownership, parallel plans and communication;
  local execution and distributed execution foundations are implemented; multi-rank model
  integration is pending.
- [Qwen4Exp parallel strategy](qwen4_exp-parallel-design.md) — selected initial dual-Spark
  TP2/EP2 strategy, state shards, computation/graph composition and implementation dependencies.
- [Qwen4Exp dual-Spark roadmap](qwen4_exp-distributed-roadmap.md) — D2–D6 implementation sequence,
  shared infrastructure, sharded loading, inference integration and stage completion checklist.
- [GLM5Next design and implementation](designs/glm5next/README.md) — selected dual-Spark
  placement, KDA ReplaySSM, Kpool Sparse MLA, mHC and the existing Op checklist; separates
  decisions, implementation status and retirement of designs after code takes ownership.

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
| [Qwen3.5 model](maintainer/qwen3_5-model.md) | Dense/MoE mathematics, instance config, logical parameters, MTP, Vision and state semantics |
| [Qwen4Exp model and weights](maintainer/qwen4_exp-model.md) | Flash Next architecture, quantization recipe, merged projections, expert banks and PLE representation |
| [Qwen4Exp execution and state](maintainer/qwen4_exp-execution.md) | current support, layer flow, PLE pipeline, MTP/Vision, continuation, resource planning and CUDA Graph |
| [GLM5Next model semantics](maintainer/glm5next-model.md) | architecture, complete mathematics, logical weights, Main/MTP/Vision and history dependencies |
| [GLM5Next quantization and numerics](maintainer/glm5next-quantization.md) | selected source/weight/cache representations, exact codecs, activation policy and rounding boundaries; implementation status belongs to the design index |
| [Expert weight cache](maintainer/expert-weight-cache.md) | Host/GPU bank sources, replica cache, mixed CPU/GPU execution, streaming and graph handoffs |
| [DFlash and DFlash2](maintainer/dflash.md) | conditioning, masked draft computation, proposal distributions and backend state |
| [Constrained decoding](maintainer/constrained-decoding.md) | GBNF/JSON/choice/regex, tool policies, masks, speculative transactions and observations |
| [Resource scheduling and context cache](maintainer/resource-scheduling-and-context-cache.md) | core design for continuation, retention, incremental resources, preemption and recovery |
| [Paged KV context store](maintainer/paged-kv-cache.md) | typed pools, pages, replicas, address spaces, reservations and consumer views |
| [ReplaySSM GDN](maintainer/replayssm-gdn.md) | raw transition records and faithful commitment of the verified state prefix |
| [Op development](maintainer/op-development.md) | semantic boundaries, source ownership, numerical qualification and performance evidence |
| [Operational logging](maintainer/logging.md) | log ownership, presentation, severity and data policy |
| [Linear tuning and measurement](maintainer/linear-tuning.md) | logical metrics, timing, tuning ranges, dispatch tradeoffs and performance reports |

Model cards contain official artifact facts and source provenance. The
[conversion guide](weight-conversion.md) is the entry point for making an artifact. Exact config
fields, parameter expansion and native supported domains are maintained by the code linked from
these references.

Qwen3.8-Flash-Next is the Qwen4Exp instance; Qwen3.8-27B uses the Qwen3.5 architecture.
The model references own mathematical and stored-representation facts, and the execution reference
owns integrated support and its limits. Concrete Op contracts live in `include/ninfer/ops/`.
