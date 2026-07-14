# AGENTS.md

## Project Scope

These rules apply to the entire repository.

NInfer is a from-scratch C++/CUDA inference engine that pursues maximum single-GPU inference
performance for a small set of explicitly selected exact checkpoints and GPU targets. It is not a
general-purpose runtime, compatibility layer, or model zoo.

## Project Engineering Principles

- NInfer is a local, single-owner inference engine that deeply optimizes explicitly registered
  model checkpoints for selected hardware. It is not a general-purpose, multi-tenant, or
  untrusted-artifact platform. Treat the registered inputs and project-managed workflow as trusted.
- Follow the declared product scope. Do not silently turn speculative concerns, implementation
  preferences, or possible future scenarios into project requirements. New cross-cutting
  constraints must solve a concrete in-scope problem and be discussed before adoption.
- Prioritize functional correctness, inference performance, architectural clarity, and low
  maintenance cost. Generality, defensive hardening, formal completeness, and test coverage are
  not goals by themselves.
- Preserve useful provenance about how an artifact was produced, but keep provenance separate from
  validity and reproducibility requirements. Recorded information is descriptive unless an explicit
  contract says otherwise; do not default to fixed hashes, clean-worktree requirements,
  byte-identical artifacts, or exact probabilistic outputs.
- Verification must match the semantic contract and its real value: use exact comparison for
  exactly defined formats and transformations, and appropriate numerical or behavioral criteria for
  floating-point and probabilistic computation. Tests must protect supported functionality or
  realistic regressions and justify their maintenance cost.

The current implemented target is Qwen3.6-27B on one RTX 5090. The delivered C++ Engine includes
fixed Text, MTP, native Vision, command-line generation, and OpenAI/Anthropic serving over the q5090
v4.2 `.qus` artifact. In parallel, native `.ninfer` conversion, Python and narrow C++ readers,
target binding/verification, and the complete Text/Vision/MTP Python reference are implemented. The
multi-target C++ Engine has not yet replaced the `.qus` route. The current workload is one user, one
active request, and one GPU; limited continuous batching and additional qualified targets are future
work, not current behavior.

## Local Environment

Use these canonical paths on the current development machine:

| Purpose | Path |
|---|---|
| repository | `/home/neroued/ninfer` |
| Python 3.11 | `/home/neroued/miniconda3/envs/py311/bin/python` |
| BF16 source model | `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16` |
| current C++ Engine artifact | `out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus` |
| current C++ Engine manifest | `out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus.manifest.json` |
| native reference artifact | `out/qwen3_6_27b_rtx5090.ninfer` |
| native conversion report | `out/qwen3_6_27b_rtx5090.ninfer.conversion.json` |
| CMake build | `build/` |
| local profiler output | `profiles/ncu/`, `profiles/nsys/`, `profiles/bench/` |
| hardware/toolchain | RTX 5090, `sm_120a`, CUDA 13.1 |

For Python work, invoke the canonical interpreter explicitly. Do not use an arbitrary ambient
`python` or `pip`, and do not install or upgrade dependencies unless the task requires it.

`out/` contains old experimental artifacts. Never choose an artifact by glob, modification time, or
an unqualified “latest” name. Use the v4.2 `.qus` path for current C++ Engine work and the `.ninfer`
path for native converter, reader, verifier, binder, and Python-reference work. Retained
v1/v2/v3/v4/v4.1 artifacts are historical diagnostics, not valid current runtime inputs.

The model, selected artifact, and profiler outputs are large local prerequisites. If one is missing,
report it clearly; do not download a model, regenerate a large artifact, or launch a long profile
unless that work is in scope. Files under `out/` and `profiles/` are normally local and uncommitted.

Useful shell variables:

```bash
PYTHON=/home/neroued/miniconda3/envs/py311/bin/python
MODEL=/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16
WEIGHTS=out/qwen3_6_27b.q5090_w4g64_mixed_v4_2.qus
NINFER_WEIGHTS=out/qwen3_6_27b_rtx5090.ninfer
```

## Sources of Truth

Use the smallest relevant current source:

- `README.md` — capabilities, build, and quick-start commands;
- `docs/README.md` — active documentation map and authority boundaries;
- `docs/ninfer-naming.md` — canonical project name, native `.ninfer` extension, and cutover status;
- `docs/ninfer-project-positioning.md` — project mission, target-selection policy, workload,
  priorities, and non-goals;
- `docs/design.md` — system ownership, runtime flows, and supported scope;
- `docs/qwen3.6-27b-architecture.md` — implemented 27B Text/MTP/Vision mathematics and state
  semantics;
- `docs/qwen3.6-35b-a3b-architecture.md` — exact 35B-A3B source-checkpoint Text/MoE/MTP/Vision
  mathematics and state semantics; runtime support is defined elsewhere;
- `docs/ninfer-tensor-formats.md`, `docs/ninfer-storage-layouts.md`, and
  `docs/ninfer-container-format.md` — native `.ninfer` numeric, layout, and framing contracts;
- `docs/qwen3.6-27b-ninfer-artifact.md` — native 27B object inventory, conversion, and binding
  contract;
- `docs/q5090_packed_file_format_v4.md` — current C++ Engine `.qus` ABI;
- `docs/ninfer-engine-architecture.md` — accepted but not-yet-implemented multi-target Engine;
- `docs/kernel-development.md` — kernel layering and correctness/performance workflow;
- `docs/serving.md` — CLI, sampling, multimodal, OpenAI, and Anthropic behavior;
- public headers under `include/ninfer/` — current C++ API;
- executable `--help` output — exact current command options.

The native tensor-format, storage-layout, container, and 27B artifact contracts are implemented by
the `.ninfer` converter/readers/verifier/binder and complete Python reference. The Engine
architecture remains pending and does not describe currently available C++ Engine APIs or inputs.

For current `.qus` converter/parser/layout work, read the q5090 specification. For `.ninfer` work,
read the tensor-format, storage-layout, container, and corresponding model-artifact specifications.
Read the system design and model reference before model/runtime/MTP/Vision work; the kernel guide
and relevant model section before CUDA work; and the serving guide before schema/sampling/protocol
work. Do not require unrelated documents to be read for every task.

Documents under `docs/archive/` are historical evidence. They may explain why a decision was made,
but they do not define current behavior.

## Discussion and Authorization Gate

Discussion and alignment with the user are a required phase of work, not an optional courtesy.

Before beginning any non-trivial task, establish the intended outcome, scope and non-goals, unresolved
behavior or interface details, relevant risks and trade-offs, acceptance criteria, and the required
verification. If any of these has material ambiguity, pause and discuss it with the user and obtain
explicit approval before implementation. Do not edit files, run state-changing commands, build, run
broad tests, profile, install dependencies, generate artifacts, launch services, or perform external
operations while waiting. Minimal read-only inspection is allowed only when necessary to frame the
discussion; it must not turn into implementation or an open-ended investigation. Silence, general
repository access, or a vague request to “handle” something is not approval.

Direct execution is allowed when the user gives an explicit, sufficiently scoped instruction with no
material unresolved decisions, or when the requested change is clearly small, low-risk, and
unambiguous (for example, a specified typo, wording, or formatting correction). If new ambiguity,
scope expansion, risk, or conflicting evidence appears, stop and return to discussion before
proceeding. Approval for one action does not authorize adjacent changes or expensive operations.

After discussion and before acting, summarize the agreed goal, scope, and verification. Do not infer
permission from a proposed plan or from the user’s approval of a different action.

## Compatibility Policy

Project-owned interfaces do not preserve backward compatibility. This includes C++ APIs, CLIs,
Python tooling, fixtures, reports, q5090 formats, project extensions, and active documentation.

When changing them:

- replace the old behavior directly;
- delete deprecated flags, fields, aliases, shims, fallback branches, and transition code;
- do not keep old and new behavior side by side;
- do not add project-owned deprecation windows;
- do not retain tests whose only purpose is to protect removed behavior.

The currently advertised OpenAI and Anthropic protocol surfaces are real external contracts.
Changes to them must update schema behavior, schema tests, and `docs/serving.md` together. This does
not require preserving deleted project-specific extensions or obsolete aliases.

Archiving an old specification is provenance, not a promise that the runtime can read it.

## Ownership and Implementation Style

Prefer direct, explicit implementation over framework-like abstraction.

- L0 (`core`) owns devices, streams, arenas, tensors, q5090 loading, KV cache, and recurrent state.
- L1 (`kernels`) owns mathematical operator APIs, dispatch, launchers, and CUDA implementations.
- L2 (`model`) owns fixed Text/MTP/Vision schedules, weight binding, and model-private helpers.
- `runtime` owns Engine lifetime, prefix reuse, sampling state, CUDA Graphs, and MTP rounds.
- `text` and `media` own tokenization, templates, media acquisition, and preprocessing.
- `serve` owns protocol schemas, request translation, streaming, and transport behavior.
- `tools` own conversion, reference inference, parity, and diagnostics.
- `bench` owns executable measurement and report behavior.

Use `docs/design.md` when detailed ownership is unclear. Do not duplicate a competing architecture
inside a local implementation note.

Do not add abstractions for hypothetical models, hardware, formats, or future runtime flexibility.
Do not introduce dynamic model graphs, string-driven model structure, hidden device allocation,
runtime weight repacking, or filesystem access on inference hot paths. Public operator APIs describe
mathematical semantics; kernel strategy and hardware policy remain private. Keep model-specific
bookkeeping out of reusable L1 unless it becomes a genuine operator contract.

## CUDA, Numerical, and Performance Rules

Numerical changes must identify the mathematical oracle, input rounding boundary, accumulation
precision, output tolerance, and real model shapes. Do not copy the CUDA algorithm into a test and
call it an independent oracle.

Treat these as high-risk boundaries:

- q5090 and `.ninfer` code/scale decode, object ranges, tensor assignment, and typed binding;
- BF16 rounding and fusion order;
- zero-centered versus plain RMSNorm;
- FP32 GDN gates and recurrent state;
- BF16/INT8 KV representation;
- MTP proposal, target verification, and state-slot commit;
- arena mark/rewind lifetime;
- CUDA Graph address and shape stability.

Final-token plausibility is not a substitute for operator or block numerical verification.

For performance work:

- use NSYS first for full-inference phase attribution, launch gaps, synchronization, and CPU/GPU
  overlap;
- use NCU after a specific hot kernel has been identified;
- do not use an isolated microbenchmark improvement as proof of end-to-end speed;
- protect numerical behavior and the relevant `ninfer_bench` path together;
- record the git commit, dirty state, GPU/CUDA, artifact identity, complete command, test matrix, and
  before/after reports for any performance claim.

Historical profiler reports are not current baselines after code, artifacts, or measurement
contracts change.

## Testing Policy

Tests are not added by default. A test is allowed only when it protects a real, observable project
risk and can fail for a meaningful regression. Tests that only increase coverage, lock private
implementation shape, or preserve compatibility are not allowed.

### Hard Whitelist

1. Numerical correctness for CUDA kernels, operators, or model-block parity.
   - Use a clear mathematical oracle.
   - Use real project shapes where applicable.
   - Include stress or edge cases that can expose the targeted wrong math.
2. Binary and file-format contracts.
   - Cover the applicable q5090 or `.ninfer` framing, object ranges, pack/unpack, shape, dtype,
     layout, and loading boundaries.
   - Reject contract violations that could load the wrong bytes or corrupt runtime state.
3. Real CLI and report/schema contracts.
   - Cover benchmark reports, tokenizer tools, fixtures, and externally consumed JSON/protocol fields.
   - Validate user-visible or downstream-consumed behavior.

### Conditional Whitelist

4. End-to-end observable behavior.
   - Use a small number of canonical fixtures.
   - Validate final output or downstream artifacts, not private call paths.
5. GPU memory and lifetime risks.
   - Exercise the risky behavior directly or pair the check with `compute-sanitizer`.
   - Examples include OOB access, use-after-rewind, state-slot mismatch, and repeated prefill/decode.
6. Reproduced bug regressions.
   - Record the trigger, expected behavior, and why existing checks missed it.
   - Prefer the smallest layer that exposes the bug.
   - Remove the regression later if the underlying risk disappears.

### Forbidden Tests

Do not add tests that:

- scan source files for strings, symbols, call order, or private layout;
- preserve removed flags, fields, aliases, shims, formats, or command surfaces;
- only test getters, setters, constructors, enum spellings, or trivial mappings;
- only check that a file or document exists;
- duplicate existing coverage without increasing risk protection;
- lock implementation details that should remain free to change;
- exist only to satisfy TDD ceremony or a coverage metric.

If an existing bad test protects a real risk, replace it with numerical, parser/schema, behavioral,
integration, or sanitizer-backed coverage before removing it. If it protects no meaningful risk,
remove it without replacement. Preferred implementation shape belongs in review, not source scans.

## Documentation Lifecycle

The stable active documentation set is defined by `docs/README.md`. New stable requirements must be
integrated into an existing authoritative document whenever possible. Do not create parallel
`final`, `v2`, `updated`, or `new-design` documents for the same current subject.

The q5090 specification is authoritative only for the current C++ Engine's `.qus` ABI. The native
numeric-format, storage-layout, container, and model-artifact documents are authoritative for their
respective `.ninfer` layers. Public headers, not a hand-maintained operator catalog, enumerate C++
APIs. Executable `--help`, not a copied flag list, enumerates command options.

Complex active work may temporarily create `docs/plans/YYYY-MM-DD-topic.md`. A plan must be moved
under the appropriate `docs/archive/` era when the work is completed, abandoned, or superseded,
preferably in the finishing change. Small changes should not create a plan file merely to satisfy a
template. Do not recreate `docs/superpowers/`.

Dated investigations, implementation reports, benchmark evidence, and profiler summaries may exist
while work is active. Distill stable conclusions into the authoritative documents, then archive the
dated material. Current behavior must not depend on an archived report or plan.

Archive rules:

- status words inside archived files describe their historical moment;
- normally preserve historical technical content rather than rewriting it as if it were current;
- repair archive indexes, move notes, and links when useful;
- annotate a material historical error instead of silently rewriting the record;
- active documents may cite archived rationale but must state current normative requirements
  themselves;
- update `docs/archive/README.md` when adding a new archive era.

Documentation changes must check active Markdown links, stale references to moved files, conflicting
current/canonical/normative claims, README navigation, and `git diff --check`. Enforce these through
review and lightweight tooling, not source-scanning unit tests.

## Implementation Plans and Review

A formal implementation plan is appropriate when the user requests one or when work is multi-stage
and high-risk, such as CUDA numerical changes, q5090 ABI changes, GPU lifetime changes, or complex
external schema changes. Ordinary localized work does not need a plan document.

A formal plan must include:

- goal and non-goals;
- current facts and a focused reading list;
- scope, ownership, and shared coordination points;
- dependencies and sequence;
- tasks split by independently verifiable delivery boundaries;
- definition of done and exact verification commands for each phase;
- a risk-scaled final review;
- the archive destination for the completed plan and evidence.

Do not split work by individual file or mechanical step when those pieces cannot be verified
independently. Do not merge unrelated risk areas merely to reduce task count. Do not structure plans
around orchestration mechanics; structure them around behavior, interfaces, artifacts, and risks.

High-risk changes require a separate final review pass after implementation. Re-read the final diff
from the numerical, lifetime, ABI, protocol, or performance perspective involved, then run the final
gate after findings are resolved. Small documentation-only or single-surface changes may use a
focused checklist.

## Verification Before Completion

Run the smallest verification set that proves the changed behavior:

| Change | Minimum verification |
|---|---|
| documentation | active-link and stale-reference audit; `git diff --check` |
| C++ API/runtime | affected build targets and affected tests |
| Python tooling | `py_compile` and affected Python tests |
| q5090 converter/parser/layout | format tests, verifier, malformed-input checks, and real artifact when needed |
| `.ninfer` converter/reader/binder | affected format/target tests, verifier, and one real artifact when needed |
| CUDA numerical behavior | oracle tests at real shapes |
| CUDA memory/lifetime | affected execution plus `compute-sanitizer` when available |
| kernel performance | microbenchmark, NCU, and relevant `ninfer_bench` comparison |
| full inference performance | NSYS and before/after `ninfer_bench` |
| CLI/report/schema | real command/output or affected schema tests |
| serving | OpenAI/Anthropic schema tests and observable response behavior |

Do not add low-value tests to compensate for weak verification. State clearly when a relevant check
could not run and why. Never present an archived report as evidence from the current change.

## Commit Messages

Use Conventional Commit-style subjects:

```text
type(scope): concise imperative summary
```

Use lowercase types consistent with project history, such as `docs`, `feat`, `fix`, `perf`, `bench`,
`test`, `build`, `refactor`, or `chore`. Do not use free-form title-case subjects.
