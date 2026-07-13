# Documentation

This directory contains the small set of documents that describe the current qwen3.6-ultraspeed
system. Active documents are maintained against the current source tree. Completed plans, retired
formats, dated design investigations, and benchmark evidence belong under [`archive/`](archive/).

## Start here

| Document | Authority |
|---|---|
| [`../README.md`](../README.md) | project capabilities, build, and quick-start commands |
| [`design.md`](design.md) | system boundaries, component ownership, runtime flows, and supported scope |
| [`qwen3.6-27b-architecture.md`](qwen3.6-27b-architecture.md) | fixed model dimensions, math, Text/MTP/Vision schedules, and state semantics |
| [`qwen3.6-35b-a3b-architecture.md`](qwen3.6-35b-a3b-architecture.md) | exact 35B-A3B source-checkpoint dimensions, hybrid Text/MoE/MTP/Vision math, and state semantics; not runtime-support status |
| [`q5090_packed_file_format_v4.md`](q5090_packed_file_format_v4.md) | normative q5090 v4.2 binary and tensor-assignment contract |
| [`kernel-development.md`](kernel-development.md) | L1 layering, API ownership, correctness, benchmark, and profiler workflow |
| [`serving.md`](serving.md) | CLI/server generation semantics, sampling, multimodal input, and tool calling |

These documents have deliberately separate responsibilities:

- system design does not duplicate model mathematics or binary layout tables;
- the model references do not prescribe repository history or kernel implementation details;
- the q5090 specification is the only normative artifact-format document;
- public C++ headers, not a hand-maintained catalog, enumerate the current operator API;
- executable `--help` output is the option-level CLI contract.

## Accepted decisions pending implementation

| Document | Authority |
|---|---|
| [`ninfer-naming.md`](ninfer-naming.md) | future project identity `NInfer` and future artifact filename extension `.ninfer` only; no current runtime, container, ABI, or migration authority |
| [`ninfer-project-positioning.md`](ninfer-project-positioning.md) | accepted NInfer mission, target-selection policy, workload, performance priorities, product boundary, and non-goals; no implementation or format authority |
| [`ninfer-tensor-formats.md`](ninfer-tensor-formats.md) | accepted closed registry and logical semantics for future NInfer direct and grouped-quantized persistent tensor formats; no container, physical layout, checkpoint assignment, kernel, or runtime-state authority |
| [`ninfer-container-format.md`](ninfer-container-format.md) | accepted future `.ninfer` v1 typed object directory, model/container boundary, canonical file geometry, loading validation, and evolution rules; no model-specific inventory, conversion recipe, layout, implementation, or migration authority |

The current system documentation above remains authoritative until the corresponding migration is
implemented. A pending decision must not be read as an already available command, API, or file
format.

## Component guides

Operational documentation stays next to the code it describes:

- [`../bench/README.md`](../bench/README.md) — real-weight and per-operator benchmarks;
- [`../tests/README.md`](../tests/README.md) — current test organization and intended risk coverage;
- [`../tools/bench/README.md`](../tools/bench/README.md) — benchmark corpus and matrix tooling;
- [`../tools/q5090/README.md`](../tools/q5090/README.md) — Python reference and diagnostics;
- [`../tools/q5090_convert/README.md`](../tools/q5090_convert/README.md) — converter and verifier.
- [`../eval/README.md`](../eval/README.md) — configurable capability evaluation, progress, logs,
  resume, and normalized reports.

## Authority and lifecycle

An active document may call itself current, canonical, or normative only within the responsibility
listed above. If implementation and an active document disagree, treat that as a documentation bug
and reconcile it rather than preserving both descriptions.

Historical documents may contain words such as “current”, “pending”, or “canonical” because they
recorded the state at the time. Their archive location overrides those old status labels. An active
implementation must not depend on an archived plan or retired specification to define required
behavior; stable requirements must be restated in an active document or expressed by the source
contract itself.

New implementation plans may be written while work is active. Once the work is completed or
superseded, move the plan and its dated evidence into the appropriate archive instead of leaving it
as a competing entrypoint.

## Archive

[`archive/README.md`](archive/README.md) explains the archive eras and how historical material is
organized. Archive files are retained for provenance and performance archaeology; their commands,
paths, APIs, and measurements are not maintained for current use.
