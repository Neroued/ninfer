# Documentation

This directory contains the small set of documents that describe the current NInfer project and
implemented system. Active documents are maintained against the current source tree. Completed
plans, retired formats, dated design investigations, and benchmark evidence belong under
[`archive/`](archive/).

## Start here

| Document | Authority |
|---|---|
| [`../README.md`](../README.md) | project capabilities, build, and quick-start commands |
| [`ninfer-naming.md`](ninfer-naming.md) | canonical project name, native `.ninfer` filename extension, and naming-cutover status; no source/API, container, or ABI authority |
| [`ninfer-project-positioning.md`](ninfer-project-positioning.md) | current project mission, exact-target selection policy, workload, performance priorities, product boundary, and non-goals; no implementation or format authority |
| [`design.md`](design.md) | current implemented system boundaries, component ownership, runtime flows, and supported scope |
| [`qwen3.6-27b-architecture.md`](qwen3.6-27b-architecture.md) | fixed model dimensions, math, Text/MTP/Vision schedules, and state semantics |
| [`qwen3.6-35b-a3b-architecture.md`](qwen3.6-35b-a3b-architecture.md) | exact 35B-A3B source-checkpoint dimensions, hybrid Text/MoE/MTP/Vision math, and state semantics; not runtime-support status |
| [`q5090_packed_file_format_v4.md`](q5090_packed_file_format_v4.md) | normative q5090 v4.2 binary and tensor-assignment contract |
| [`kernel-development.md`](kernel-development.md) | L1 layering, API ownership, correctness, benchmark, and profiler workflow |
| [`serving.md`](serving.md) | CLI/server generation semantics, sampling, multimodal input, and tool calling |

These documents have deliberately separate responsibilities:

- system design does not duplicate model mathematics or binary layout tables;
- the model references do not prescribe repository history or kernel implementation details;
- the q5090 specification is authoritative for the current C++ Engine's `.qus` route;
- public C++ headers, not a hand-maintained catalog, enumerate the current operator API;
- executable `--help` output is the option-level CLI contract.

## Implemented native artifact contracts

| Document | Authority |
|---|---|
| [`ninfer-tensor-formats.md`](ninfer-tensor-formats.md) | closed registry and logical semantics for native NInfer direct and grouped-quantized persistent tensor formats; no container, physical layout, checkpoint assignment, kernel implementation/dispatch, or runtime-state-codec authority |
| [`ninfer-storage-layouts.md`](ninfer-storage-layouts.md) | registered `contiguous-le-v1`, `row-split-k128-v1`, and `raw-bytes-v1` persistent-byte contracts; no checkpoint assignment, conversion recipe, or kernel authority |
| [`ninfer-container-format.md`](ninfer-container-format.md) | `.ninfer` v1 16-byte framing, closed embedded-JSON object directory, payload geometry, registries, and model/container boundary; no model-specific inventory, conversion recipe, model execution, or Engine-construction authority |
| [`qwen3.6-27b-ninfer-artifact.md`](qwen3.6-27b-ninfer-artifact.md) | complete `qwen3.6-27b` `.ninfer` object inventory, storage signatures, logical views, frontend resources, source-checkpoint transforms, and binder obligations |

These contracts are implemented by the native `.ninfer` converter, generic Python
reader/inspector, narrow C++ reader, target verifier/binder, and complete Python Text/Vision/MTP
reference. This is a complete artifact and correctness-reference route, not a claim that the current
C++ Engine consumes `.ninfer`.

## Accepted Engine decision under implementation

| Document | Authority |
|---|---|
| [`ninfer-engine-architecture.md`](ninfer-engine-architecture.md) | accepted future NInfer core engine and source-organization boundary: compiled exact-target packages, load-time construction, ownership and memory lifetimes, checkpoint frontends, one-request program state, transactional generated-token rounds, repository/build dependency direction, and common-versus-target-private interfaces; no current implementation, model math, format details, serving protocol, migration plan, or future scheduler authority |

The current C++ implementation documents above remain authoritative until the Engine migration is
complete. The q5090 v4.2 `.qus` route remains the only implemented C++ Engine artifact input. The
implemented native `.ninfer` tooling and Python reference do not make the accepted multi-target
Engine architecture an available C++ runtime command or API.

## Component guides

Operational documentation stays next to the code it describes:

- [`../bench/README.md`](../bench/README.md) — real-weight and per-operator benchmarks;
- [`../tests/README.md`](../tests/README.md) — current test organization and intended risk coverage;
- [`../tools/bench/README.md`](../tools/bench/README.md) — benchmark corpus and matrix tooling;
- [`../tools/q5090/README.md`](../tools/q5090/README.md) — legacy `.qus` Python reader/codec and
  structural diagnostics retained for the current C++ Engine;
- [`../tools/q5090_convert/README.md`](../tools/q5090_convert/README.md) — converter and verifier;
- [`../tools/artifact/`](../tools/artifact/) — native `.ninfer` reader, writer, inspector, numeric
  formats, and storage layouts;
- [`../tools/convert/qwen3_6_27b_rtx5090/`](../tools/convert/qwen3_6_27b_rtx5090/) — registered
  Qwen3.6-27B native converter and source verifier;
- [`../tools/reference/qwen3_6_27b_rtx5090/`](../tools/reference/qwen3_6_27b_rtx5090/) — complete
  native-artifact Text/Vision/MTP reference and CLI;
- [`../tools/parity/qwen3_6_27b_rtx5090/`](../tools/parity/qwen3_6_27b_rtx5090/) — target-specific
  activation, preprocessing, and source-Vision comparison diagnostics;
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
