# Documentation

Active documents define the current NInfer product, maintained contracts, source tree, and exact
checkpoint references that are still expected to guide later target work. Completed plans, retired
implementations, dated investigations, and performance evidence belong under [`archive/`](archive/).

## Start here

| Document | Authority |
|---|---|
| [`../README.md`](../README.md) | current capabilities, build products, and quick-start commands |
| [`ninfer-project-positioning.md`](ninfer-project-positioning.md) | project mission, exact-target policy, workload, performance priorities, and non-goals |
| [`design.md`](design.md) | implemented component ownership, load/request flows, lifetime boundaries, and build direction |
| [`ninfer-engine-architecture.md`](ninfer-engine-architecture.md) | detailed target-package, Engine, Frontend, Program, generated-token transaction, and source-organization decisions |
| [`serving.md`](serving.md) | CLI, server, sampling, multimodal input, streaming, usage, and tool-call behavior |
| [`op-development.md`](op-development.md) | Op definition and contracts, central implementation ownership, numerical validation, benchmarking, and profiling |
| [`ninfer-naming.md`](ninfer-naming.md) | canonical NInfer project name and `.ninfer` artifact extension |

The executable `--help` output is the exact command-option contract. The public C++ API is
defined by `include/ninfer/engine.h` and `include/ninfer/types.h`.

## Native artifact contracts

| Document | Authority |
|---|---|
| [`ninfer-tensor-formats.md`](ninfer-tensor-formats.md) | registered persistent numeric formats and their logical semantics |
| [`ninfer-storage-layouts.md`](ninfer-storage-layouts.md) | registered persistent byte layouts and resource encoding |
| [`ninfer-container-format.md`](ninfer-container-format.md) | `.ninfer` framing, embedded JSON directory, object geometry, and container/model boundary |
| [`qwen3.6-27b-ninfer-artifact.md`](qwen3.6-27b-ninfer-artifact.md) | complete Qwen3.6-27B object inventory, source transforms, resources, views, and binder obligations |

These contracts are implemented by the generic artifact reader/writer/inspector, the registered
27B converter and verifier, the C++ artifact reader/binder/materializer, the compiled target package,
and the Python correctness reference.

## Model computation references

| Document | Authority |
|---|---|
| [`qwen3.6-27b-architecture.md`](qwen3.6-27b-architecture.md) | current target's fixed dimensions, Text/MTP/Vision math, and state semantics |
| [`qwen3.6-35b-a3b-architecture.md`](qwen3.6-35b-a3b-architecture.md) | exact 35B-A3B source-checkpoint Text/MoE/MTP/Vision reference; not runtime-support status |

Model computation documents do not define product support, artifact framing, source ownership, or
serving behavior. Runtime support requires an explicit compiled exact-checkpoint/GPU target.

## Implemented product boundary

The only registered product target is `qwen3_6_27b_rtx5090`. Its artifact and request route are:

```text
qwen3_6_27b_rtx5090.ninfer
  -> generic artifact reader / binder / materializer
  -> closed qwen3_6_27b_rtx5090 target package
  -> immutable LoadedModel + Frontend + one mutable Program
  -> common generated-token controller
  -> public Engine
  -> apps/ninfer, apps/ninfer-serve, and ninfer_bench
```

The target package owns checkpoint/GPU binding, schedules, and state policy; its schedules compose
repository-internal Op contracts whose implementations are centrally owned. Common runtime code
owns target-independent mechanisms and generated-token policy. Serving code owns protocols and
transport. The public API is opaque and contains no CUDA, artifact, tensor, Op, kernel, or
target-private types.

## Component guides

- [`../bench/README.md`](../bench/README.md) — product-route and CUDA operator benchmarks;
- [`../tests/README.md`](../tests/README.md) — retained tests and their observable risks;
- [`../tools/bench/README.md`](../tools/bench/README.md) — benchmark corpus and matrix tooling;
- [`../tools/README.md`](../tools/README.md) — entry point for artifact, conversion, reference,
  parity, benchmark, and serving-smoke tools;
- [`../tools/artifact/`](../tools/artifact/) — generic `.ninfer` read/write/layout/inspection code;
- [`../tools/convert/qwen3_6_27b_rtx5090/`](../tools/convert/qwen3_6_27b_rtx5090/) — registered
  converter, inventory, recipe, draft head, and source verifier;
- [`../tools/reference/qwen3_6_27b_rtx5090/`](../tools/reference/qwen3_6_27b_rtx5090/) — complete
  artifact-native Python Text/Vision/MTP reference;
- [`../tools/parity/qwen3_6_27b_rtx5090/`](../tools/parity/qwen3_6_27b_rtx5090/) — target-specific
  Text activation, source-BF16 Vision, and combined frontend/Vision/MTP comparison tools;
- [`../eval/README.md`](../eval/README.md) — optional local capability-evaluation coordinator.

## Authority and lifecycle

Each active document is authoritative only for the responsibility named above. If implementation
and an active document disagree, reconcile the document instead of preserving two current routes.

Archive documents may contain old status language because they record a particular development
state. Their archive location overrides those labels. Runtime code must not depend on an archived
plan or retired specification for current behavior.

[`archive/README.md`](archive/README.md) explains the archive eras. Archive commands, paths, APIs,
and measurements are historical evidence and are not maintained as product entry points.
