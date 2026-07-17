# AGENTS.md

These rules apply to the whole repository.

## Governing objective

Complete the user's explicit deliverable within the applicable product contract. Use the smallest
coherent scope that fully satisfies the requested outcome, preserves supported behavior, and makes
the result credible. “Smallest” means no missing necessary work and no unrelated work; it does not
mean the fewest changed lines or a partial solution.

Correctness, performance, tests, profiling, documentation, provenance, cleanup, and tooling are
means to the requested outcome, not independent objectives. Do not let supporting work replace,
delay, or materially enlarge the requested deliverable.

When choosing between possible work, use this order:

1. respect applicable product and external-contract constraints;
2. satisfy the user's explicit deliverable and acceptance criteria;
3. preserve functional and numerical correctness of supported behavior;
4. satisfy properties explicitly required by the task, including performance where applicable;
5. preserve clear ownership and low maintenance cost;
6. gather only the evidence and provenance needed to support the result.

The product and architecture described here are the current contract for ordinary work. A task may
explicitly change that contract; when it does, update the affected implementation, tests, and active
authorities consistently rather than treating the current description as an immutable prohibition.

## Scope control

Before substantial work, determine the requested output, the behavior or decision it must support,
and the conditions under which it is complete. This is an execution discipline, not a requirement
to create a separate planning artifact.

Work is in scope only when it:

- directly contributes to the requested deliverable;
- is necessary to preserve an applicable product, semantic, or external contract;
- resolves uncertainty that could materially change the result; or
- checks a realistic regression introduced by the change.

Do not expand a task into a broader redesign, audit, cleanup, hardening effort, compatibility
project, benchmark campaign, or documentation project without explicit user direction. General
engineering preferences, possible future scenarios, and concerns outside the declared product
model do not create requirements by themselves. Discuss a newly discovered cross-cutting
constraint before adding it.

Handle incidental findings proportionally:

- address them when they block the requested outcome or make it materially incorrect;
- include them when they are inseparable from a coherent implementation;
- otherwise leave them unchanged and mention them only when they are useful to the user.

For analysis, review, or design work, the requested explanation or design artifact is the
deliverable; experiments and code inspection serve only to resolve material questions. For
implementation work, produce the smallest coherent implementation and validate its supported
observable behavior. For diagnosis, establish the cause and supporting evidence without turning
the task into an unrequested fix or redesign.

## Evidence, provenance, and completion

Select evidence from the claim or decision it supports. The availability of a tool, test suite,
artifact, or profiler does not make its use necessary. Prefer representative evidence over
exhaustive evidence, and do not repeat an experiment unless the previous result is invalid or
inconclusive, or the new result could change a live decision.

Verification must match the semantic contract: use exact comparison for exact formats and
transformations, and numerical or behavioral criteria for floating-point and probabilistic work.
Do not substitute final-output plausibility for verification of an operator or state transition.

Record only the provenance needed to interpret a material result. By default, this is the relevant
target, hardware/toolchain, workload or command, and summarized outcome. Fixed hashes, clean
worktrees, full command transcripts, raw profiler inventories, byte-identical regeneration, and
exact probabilistic outputs are not validity requirements unless a concrete contract or the user
requires them.

Stop when:

- the requested deliverable exists;
- applicable contracts are satisfied;
- material claims have sufficient evidence;
- relevant checks pass, or their limitations are stated clearly; and
- no known in-scope issue prevents the result from being used.

Do not continue merely to eliminate all uncertainty, collect more metrics, complete a process loop,
improve descriptive provenance, investigate unrelated observations, or make working notes
exhaustive. The final result should lead with the deliverable, key decisions, relevant verification,
and material limitations. Raw logs, experiment diaries, exhaustive command histories, hashes, and
intermediate artifacts are excluded unless requested or themselves the deliverable.

## Current product contract

NInfer is a from-scratch C++/CUDA inference engine for maximum single-GPU inference performance on
a small set of explicitly registered checkpoint/GPU targets. The current product supports exactly
`qwen3_6_27b_rtx5090`: Qwen3.6-27B on an NVIDIA GeForce RTX 5090, with Text, image/video Vision,
MTP, prefix reuse, CLI, OpenAI/Anthropic serving, and measurement through one `.ninfer` Engine
route.

The current workload is one user, one active request, and one GPU. Continuous batching, additional
targets, and additional hardware are future work, not current behavior. This is a local,
single-owner project. Registered models, generated artifacts, and the local workflow are trusted.
Requirements derived from a different workload, trust model, or deployment model are out of scope
until that product contract is explicitly changed.

The accepted boundary for adding `qwen3_6_35b_a3b_rtx5090` is deliberate family reuse without a
family execution model. The 27B and 35B-A3B exact targets share one identity-free Qwen3.6 frontend,
owning prepared-prompt/output-session types, and common passive Vision definitions. They do not
share Program state, Text/Vision/MTP schedules, Op selection or fusion, workspace policy, or CUDA
Graphs. Both artifacts embed the same six frontend resources. A Qwen3.6 prepared prompt carries no
exact-target tag or cross-target mismatch machinery.

## Engineering priorities

Prioritize functional correctness, requested inference performance, clear ownership, direct code,
and low maintenance cost. Generality, defensive hardening, formal completeness, broad compatibility,
and test coverage are not goals by themselves.

Prefer explicit target-specific implementation over framework-like abstraction. Do not add generic
model graphs, family base classes, plugin discovery, string-driven execution, hidden device
allocation, runtime weight repacking, or placeholders for hypothetical models or hardware unless an
explicitly changed product contract requires them.

## Sources of truth

Read only the smallest relevant current authority needed for the task. The following list is a
routing map, not a mandatory reading list:

- `README.md` and executable `--help`: delivered capabilities and exact commands;
- `docs/README.md`: active-document map and authority boundaries;
- `docs/ninfer-project-positioning.md`: mission, target policy, workload, and non-goals;
- `docs/design.md` and `docs/ninfer-engine-architecture.md`: implemented ownership, runtime, and
  source/build boundaries;
- `docs/ninfer-container-format.md`, `docs/ninfer-storage-layouts.md`, and
  `docs/ninfer-tensor-formats.md`: generic `.ninfer` contracts;
- `docs/qwen3.6-27b-ninfer-artifact.md`: exact target inventory, conversion, and binding;
- `docs/qwen3.6-27b-architecture.md`: Text/Vision/MTP mathematics and state semantics;
- `docs/op-development.md`: Op boundary, contracts, implementation ownership, correctness, and
  performance workflow;
- `docs/serving.md`: CLI, sampling, multimodal, OpenAI, and Anthropic behavior;
- `include/ninfer/engine.h` and `include/ninfer/types.h`: installed C++ product API.

Do not survey unrelated authorities for completeness. Read additional documents only when they
govern a live decision in the current task. `docs/archive/` is historical evidence, not current
authority.

## Product and ownership boundaries

These boundaries govern ordinary implementation work. An explicit architecture task may revise
them, but must update the corresponding active authorities and affected implementation together.

- `.ninfer` is the only C++ product artifact. Do not add `.qus` fallback, extension detection,
  compatibility shims, or a second product lane.
- `include/ninfer/engine.h` and `include/ninfer/types.h` are the installed opaque Engine API and
  owning host values. `include/ninfer/ops/` contains repository-internal semantic Op contracts; it
  is not installed product ABI.
- `src/core` owns device primitives, tensors/views, checked layouts, arenas, graph RAII, physical
  KV-cache containers, and raw transfer mechanisms.
- `src/artifact` owns generic `.ninfer` framing, descriptors, binding primitives, and
  materialization. It has no checkpoint execution semantics.
- `src/ops` owns every semantically closed Op implementation, including fused, fixed-shape, and
  device-specialized paths. Op ownership follows the mathematical or state-transition contract,
  not its first model caller or demonstrated cross-target reuse.
- `src/targets/qwen3_6` owns only the Qwen3.6-family invariants shared by the 27B and 35B-A3B
  targets: tokenizer/template and output semantics, media preprocessing and MRoPE prompt
  construction, the owning prepared-prompt/output-session types, and passive common Vision
  geometry/binding definitions. It has no target identity, registry entry, Program, execution
  schedule, Op call/fusion policy, workspace, CUDA Graph, or mutable model state.
- `src/targets/<target_key>` owns the exact checkpoint/GPU storage profile, LoadedModel, Program,
  recurrent state, target-width bindings, Text/Vision/MTP schedules, target graph/state policy, and
  diagnostics. A Qwen3.6 target composes the family leaves above. Target schedules compose
  repository-internal Ops but do not own mathematical CUDA implementations. Each exact target is
  one closed execution package, not a family model or generic graph.
- `src/runtime` owns common contracts, generated-token transaction/publication policy, and the
  public Engine PIMPL. It does not own model mathematics or target state.
- `src/media/decode` consumes already-owned bytes. URL/path/data acquisition belongs to
  `src/product/media_acquire`, CLI, or serving and is not linked into a target.
- `src/product/prompt_input` owns the shared product-side JSON/message-to-owning-input adapter.
- `src/serve` owns protocol translation and transport. CLI, server, and benchmark call only the
  public Engine for inference.
- `tools/convert/<target>`, `tools/reference/<target>`, and `tools/parity/<target>` remain
  target-private conversion, correctness, and diagnostic implementations.

## Compatibility and document lifecycle

Project-owned C++ APIs, CLIs, Python tools, fixtures, reports, formats, and active documentation do
not preserve backward compatibility. When a task replaces project-owned behavior, remove the
obsolete aliases, fallbacks, transition branches, and tests in the affected contract instead of
maintaining two paths. Do not turn that rule into unrelated repository-wide cleanup.

The advertised OpenAI and Anthropic protocol surfaces are real external contracts. A change to
their behavior must update the affected schema tests and serving documentation together.

Integrate stable requirements into the existing active authority. Use a dated file under
`docs/plans/` only when active work genuinely needs a separate plan; a plan is not a substitute for
the requested deliverable. Archive it when the work completes or is abandoned. Do not create
parallel `final`, `v2`, or `new-design` authorities. Update active links and
`docs/archive/README.md` when moving completed plans.

## Numerical correctness

When a task changes numerical behavior or makes a numerical claim, identify the mathematical
oracle, represented public inputs, explicit semantic cast/quantization/state boundaries, output
criterion, and real model shapes relevant to that claim. If a route's private precision or
reduction profile matters to the evidence, describe it as an implementation profile rather than a
semantic requirement. Apply exact, tolerance-based, or behavioral comparison according to the
actual semantic contract.

Every floating-point Op has one independent naive FP32/FP64 mathematical oracle; exact transforms
and codecs have one independent exact oracle. The oracle evaluates the complete logical formula
from the represented public inputs and, for packed weights, decodes the signed code with the exact
stored scale. It does not copy a production kernel's staging casts, reduction tree, workspace dtype,
or another implementation's output.

The oracle does not prescribe a production arithmetic path. Unless an intermediate value is an
observable Op output, explicit Cast/quantize/dequantize result, registered codec value, or specified
persistent state, kernels may choose the natural intermediate precision, instruction operands,
reduction association, workspace representation, and kernel decomposition for their route. A fused
kernel is neither required to reproduce an unfused BF16 materialization nor forbidden from using a
lower-precision intermediate when that is the natural qualified implementation. Every production
route is checked directly against the same oracle with a criterion appropriate to its output and
implementation profile; pairwise implementation parity is supplementary evidence only.

Where relevant to the changed behavior, account for numeric-format decode, BF16 fusion order, FP32
GDN state, BF16/INT8 KV, MTP accept/commit state, arena lifetime, and CUDA Graph address stability.
This is a risk map, not a checklist for every numerical task.

## Performance work

Define a performance claim at the level where it matters: operator, schedule, request phase, or
end-to-end inference. Measure that level directly when practical. An isolated microbenchmark can
support an operator-level claim but does not establish an end-to-end improvement.

Use whole-inference profiling when end-to-end attribution remains unresolved. Use kernel profiling
only after a relevant kernel has been identified and a kernel-level answer could materially change
the current design or implementation decision. Do not collect additional profiling data once the
relevant alternatives can be distinguished and the requested claim has adequate support.

Retain concise context sufficient to interpret a reported result: relevant hardware/toolchain,
artifact identity at the descriptive level, workload or command, and summarized measurements. Raw
reports and fixed repository or artifact hashes are not required by default.

## Tests and verification

Add or retain a test only when it protects supported observable behavior or a realistic regression:
numerical kernel/model correctness, `.ninfer` framing/binding, external schema/report behavior, a
small real integration route, GPU lifetime, or a reproduced bug. Do not add tests for coverage,
private file/class shape, getters/constructors, deleted compatibility, source-string scans,
hypothetical failures, or test ceremony.

Run the smallest set of checks sufficient to support the changed behavior and its material claims.
The following are typical choices, not a cumulative checklist:

| Change | Relevant evidence |
|---|---|
| documentation | affected active-link/stale-reference review and `git diff --check` |
| C++ runtime/API | affected explicit targets and meaningful tests |
| Python tooling | `py_compile` and affected Python tests |
| `.ninfer` reader/converter/binder | affected contract tests and a real artifact when semantics require it |
| CUDA math | independent numerical oracle at relevant shapes |
| memory/lifetime | the affected execution; sanitizer only for a concrete lifetime risk |
| performance | measurement at the claimed scope; attribution tools only when needed |
| serving | affected OpenAI/Anthropic schema tests and observable request/stream behavior |

Do not replace weak verification with low-value tests. State clearly when a relevant check could not
run and why.

## Local environment

These are available project resources, not a checklist of resources every task must use:

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

## Commits

Create a commit only when the user requests one. Use Conventional Commit-style subjects, for
example:

```text
feat(engine): cut over the registered target to native artifacts
```

Use concise lowercase types consistent with repository history (`feat`, `fix`, `perf`, `bench`,
`test`, `build`, `refactor`, `docs`, `chore`).
