# AGENTS.md

## Project scope

These rules apply to the whole repository.

NInfer is a from-scratch C++/CUDA inference engine for maximum single-GPU inference performance on
a small set of explicitly registered checkpoint/GPU targets. The current product supports exactly
`qwen3_6_27b_rtx5090`: Qwen3.6-27B on an NVIDIA GeForce RTX 5090, with Text, image/video Vision,
MTP, prefix reuse, CLI, OpenAI/Anthropic serving, and measurement through one `.ninfer` Engine
route. The current workload is one user, one active request, and one GPU. Continuous batching,
additional targets, and additional hardware are future work, not current behavior.

## Core engineering principles

- This is a local, single-owner project. Registered models, generated artifacts, and the local
  workflow are trusted. Do not invent untrusted-artifact, injection, multi-tenant, or adversarial
  requirements.
- Follow the declared product scope. A possible future scenario or general engineering preference
  is not a current requirement. Discuss new cross-cutting constraints before adding them.
- Prioritize functional correctness, inference performance, clear ownership, and low maintenance
  cost. Generality, defensive hardening, formal completeness, and test coverage are not goals by
  themselves.
- Preserve useful provenance about how an artifact or result was produced, but keep descriptive
  provenance separate from validity requirements. Do not require fixed hashes, clean worktrees,
  byte-identical regeneration, or exact probabilistic outputs unless a concrete contract requires
  them.
- Verification must match the semantic contract: exact comparison for exact formats and
  transformations; numerical or behavioral criteria for floating-point and probabilistic work.
  Keep only tests that protect supported functionality or a realistic regression.

## Local environment

| Purpose | Path |
|---|---|
| repository | `/home/neroued/ninfer` |
| Python 3.11 | `/home/neroued/miniconda3/envs/py311/bin/python` |
| BF16 source checkpoint | `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16` |
| product artifact | `out/qwen3_6_27b_rtx5090.ninfer` |
| conversion report | `out/qwen3_6_27b_rtx5090.ninfer.conversion.json` |
| normal build | `build/` |
| profiler output | `profiles/ncu/`, `profiles/nsys/`, `profiles/bench/` |
| hardware/toolchain | RTX 5090, `sm_120a`, CUDA 13.1 |

Use the canonical Python interpreter explicitly. Do not install or upgrade dependencies unless the
task requires it. Never select an artifact by glob, modification time, or an unqualified “latest”
name. Large artifacts, source checkpoints, and profiler outputs are local prerequisites; do not
download or regenerate them unless that work is in scope.

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
MODEL=/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16
NINFER_WEIGHTS=out/qwen3_6_27b_rtx5090.ninfer
```

## Sources of truth

Read the smallest relevant current authority:

- `README.md` and executable `--help`: delivered capabilities and exact commands;
- `docs/README.md`: active-document map and authority boundaries;
- `docs/ninfer-project-positioning.md`: mission, target policy, workload, and non-goals;
- `docs/design.md` and `docs/ninfer-engine-architecture.md`: implemented ownership, runtime, and
  source/build boundaries;
- `docs/ninfer-container-format.md`, `docs/ninfer-storage-layouts.md`, and
  `docs/ninfer-tensor-formats.md`: generic `.ninfer` contracts;
- `docs/qwen3.6-27b-ninfer-artifact.md`: exact target inventory, conversion, and binding;
- `docs/qwen3.6-27b-architecture.md`: Text/Vision/MTP mathematics and state semantics;
- `docs/kernel-development.md`: kernel correctness and performance workflow;
- `docs/serving.md`: CLI, sampling, multimodal, OpenAI, and Anthropic behavior;
- `include/ninfer/engine.h` and `include/ninfer/types.h`: installed C++ product API.

`docs/archive/` is historical evidence, not current authority.

## Product and ownership boundaries

- `.ninfer` is the only C++ product artifact. Do not add `.qus` fallback, extension detection,
  compatibility shims, or a second product lane.
- `include/ninfer/` contains only the opaque Engine API and owning host values. CUDA, tensors,
  artifact details, kernels, targets, and serving types stay under `src/`.
- `src/core` owns device primitives, tensors/views, checked layouts, arenas, and graph RAII.
- `src/artifact` owns generic `.ninfer` framing, descriptors, binding primitives, and
  materialization. It has no checkpoint execution semantics.
- `src/kernels` owns proven shared mathematical operators. A target-only fixed-shape operation
  stays in that target until real reuse proves a common contract.
- `src/targets/<target_key>` owns the exact checkpoint/GPU storage profile, LoadedModel, Frontend,
  Program, KV/recurrent state, Text/Vision/MTP schedules, target graphs, and diagnostics. It is one
  closed package, not a model family or generic graph.
- `src/runtime` owns common contracts, generated-token transaction/publication policy, and the
  public Engine PIMPL. It does not own model mathematics or target state.
- `src/media/decode` consumes already-owned bytes. URL/path/data acquisition belongs to
  `src/product/media_acquire`, CLI, or serving and is not linked into a target.
- `src/product/prompt_input` owns the shared product-side JSON/message-to-owning-input adapter.
- `src/serve` owns protocol translation and transport. CLI, server, and benchmark call only the
  public Engine for inference.
- `tools/convert/<target>`, `tools/reference/<target>`, and `tools/parity/<target>` remain
  target-private conversion, correctness, and diagnostic implementations.

Prefer direct explicit code over framework-like abstraction. Do not add generic model graphs,
family base classes, plugin discovery, string-driven execution, hidden device allocation, runtime
weight repacking, or placeholders for hypothetical models/hardware.

## Compatibility and lifecycle

Project-owned C++ APIs, CLIs, Python tools, fixtures, reports, formats, and active documentation do
not preserve backward compatibility. Replace obsolete behavior directly and delete aliases,
fallbacks, transition branches, and tests that only protect removed behavior. The advertised
OpenAI and Anthropic protocol surfaces are real external contracts and must change together with
their schema tests and serving documentation.

Integrate stable requirements into the existing active authority. Complex active work may use a
dated file under `docs/plans/`; archive it when the work completes or is abandoned. Do not create
parallel `final`, `v2`, or `new-design` authorities. Update active links and `docs/archive/README.md`
when moving completed plans.

## CUDA, numerical, and performance work

Numerical changes must identify the mathematical oracle, input rounding boundary, accumulation
precision, output tolerance, and real model shapes. Final-token plausibility is not operator
verification. Pay particular attention to numeric-format decode, BF16 fusion order, FP32 GDN state,
BF16/INT8 KV, MTP accept/commit state, arena lifetime, and CUDA Graph address stability.

Use NSYS for whole-inference attribution, then NCU for an identified kernel. An isolated
microbenchmark does not prove end-to-end improvement. A performance claim should retain enough
descriptive context to interpret it—hardware/toolchain, artifact identity, command/matrix, and
reports—but no fixed repository or artifact hash is required by default.

## Tests and verification

Add or retain a test only when it protects a supported observable risk: numerical kernel/model
correctness, `.ninfer` framing/binding, external schema/report behavior, a small real integration
route, GPU lifetime, or a reproduced bug. Do not add tests for coverage, private file/class shape,
getters/constructors, deleted compatibility, source-string scans, hypothetical failures, or test
ceremony.

Run the smallest set that proves the change. Typical evidence is:

| Change | Evidence |
|---|---|
| documentation | active-link/stale-reference review and `git diff --check` |
| C++ runtime/API | affected explicit targets and meaningful tests |
| Python tooling | `py_compile` and affected Python tests |
| `.ninfer` reader/converter/binder | affected contract tests and a real artifact when needed |
| CUDA math | independent numerical oracle at relevant shapes |
| memory/lifetime | the affected execution; sanitizer only when it addresses the actual risk |
| performance | relevant microbenchmark plus end-to-end `ninfer_bench`; NSYS/NCU as warranted |
| serving | OpenAI/Anthropic schema tests and observable request/stream behavior |

Do not replace weak verification with low-value tests. State clearly when a relevant check could not
run and why.

## Commits

Use Conventional Commit-style subjects, for example:

```text
feat(engine): cut over the registered target to native artifacts
```

Use concise lowercase types consistent with repository history (`feat`, `fix`, `perf`, `bench`,
`test`, `build`, `refactor`, `docs`, `chore`).
