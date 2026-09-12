# AGENTS.md

These rules apply to the whole repository.

## Objective and scope

Complete the user's explicit deliverable within the applicable product and external contracts.
Choose a coherent solution with functional and numerical correctness, clear ownership, strong
architecture, and maximum performance at the requested scope. Do not sacrifice these goals to
reduce the diff or implementation effort. Evaluate complexity, maintenance cost, and verification
risk as engineering tradeoffs, not reasons to retain a known inferior design.

Before substantial work, identify the deliverable and its completion conditions. Work is relevant
when it completes that deliverable, preserves an applicable contract, resolves a material
uncertainty, or checks a realistic regression. A necessary redesign is in scope; unrelated cleanup,
hardening, compatibility, and benchmark campaigns are not. Address incidental findings when they
block the outcome or are inseparable from the selected implementation.

For analysis or design, deliver the explanation or design. For diagnosis, establish the cause and
supporting evidence; implement a fix when requested. For implementation, complete the selected
design across its affected implementations, callers, tests, tools, and active documentation.

The current product and architecture govern ordinary work. An explicit task may change them;
update the affected contracts and implementation together instead of treating the current design
as an immutable prohibition. Skills provide task-specific methods, not additional deliverables or
approval requirements beyond the user's instructions and the actual execution environment.

## Product and architecture

NInfer is a from-scratch C++/CUDA inference engine for maximum single-GPU performance on explicitly
registered artifacts. Current identities are `qwen3.6-27b/groupwise-int`, `qwen3.6-27b/nvfp4`,
`qwen3.8-27b/groupwise-int`, `qwen3.8-27b/nvfp4`, and `qwen3.6-35b-a3b/groupwise-int`.
The implementation targets `sm_120a` and is tuned on NVIDIA GeForce RTX 5090.

Generation uses one GPU, one resident model, startup-fixed concurrency of one to eight requests,
bounded FIFO ingress, no active-request preemption, and one compact decode batch per round.
Generation and offline CausalScoring use the same public `.ninfer` Engine route. Delivered
capabilities and commands are documented in `README.md`, the product guides, and executable
`--help`. Additional models, execution platforms, large-scale/preemptive continuous batching, and
priority/QoS require an explicit product change.

This is a local, single-owner project with trusted registered models, generated artifacts, and
local workflow. Do not derive requirements from a different deployment or trust model.

Keep these ownership boundaries visible when selecting a design:

- `.ninfer` is the only C++ product artifact; CLI, serving, and inference benchmarks use the public
  Engine. NInfer has no Python model-inference route or installed/exported C++ SDK.
- Core owns physical primitives and raw transfers; artifact owns generic framing and
  materialization; Ops own closed mathematical and state-transition implementations.
- The Qwen3.6 family owns shared frontend semantics and compile-time planning/Program algorithms.
  The 27B and 35B-A3B packages are peer Variants owning their identities, bindings, execution leaves,
  and Program instance storage. Programs share no mutable state or device allocation.
- Runtime owns common execution contracts and Engine publication policy; product/serving own input
  acquisition and protocol translation. Target packages do not acquire media or own transport.

Detailed family/package responsibilities and source ownership are defined in
[Engine architecture](docs/maintainer/engine-architecture.md). Read the relevant boundary before
changing it. Prefer explicit implementations for registered targets. Do not introduce generic model
graphs, family base classes, plugin discovery, string-driven execution, hidden device allocation,
runtime weight repacking, or placeholders for hypothetical targets without a product requirement.

## Change consistency

Project-owned APIs, CLIs, Python tools, fixtures, reports, formats, and documentation do not preserve
backward compatibility. When replacing behavior, remove superseded aliases, fallbacks, transition
branches, and their tests within the affected contract. Leave unrelated paths alone.

Advertised OpenAI and Anthropic protocol behavior is an external contract. Changes update the
affected schema tests and serving documentation together.

Keep stable requirements in their existing active reference. Temporary plans are useful only for
active work; remove them when completed or abandoned. Maintain one current authority rather than
parallel `final`, `v2`, or `new-design` documents.

## Verification and completion

Select evidence to support the changed behavior and material claims. Tests should protect supported
observable behavior, mathematical or state semantics, and realistic regressions, including plausible
boundary failures that have not occurred yet. Avoid tests that merely mirror implementation,
freeze private file/class organization, or increase coverage numbers.

For numerical changes, identify represented public inputs, the independent mathematical oracle,
semantic cast/quantization/state boundaries, output criteria, and relevant real model shapes. Each
floating-point Op uses a naive FP32/FP64 oracle; exact transforms/codecs use an exact oracle. Packed
inputs are independently decoded with their stored scales. Qualify production routes directly
against that oracle, not another kernel or plausible model output. Private arithmetic need not
reproduce unfused materializations unless an intermediate is an observable semantic boundary.
[Op development](docs/maintainer/op-development.md) defines the full qualification contract.

Measure performance at the claimed scope. An Op microbenchmark establishes an Op result, not an
end-to-end improvement. Use whole-inference profiling when an in-scope end-to-end attribution is
unresolved; use kernel profiling when an identified kernel question can change the decision. Reuse
applicable evidence and stop collecting once the relevant alternatives can be distinguished.

Choose the affected checks, rather than running this table as a checklist:

| Change | Typical evidence |
|---|---|
| Documentation | affected links/references and `git diff --check` |
| C++ runtime/API | affected build targets and behavioral tests |
| Python tooling | Python 3.11 `py_compile` and affected tests |
| Artifact framing/binding/conversion | affected contract tests; real artifact when semantics require it |
| CUDA mathematics | independent oracle at relevant shapes and route boundaries |
| Memory or lifetime | affected execution; sanitizer for a concrete lifetime question |
| Performance | measurement at the claimed scope; profiling only for unresolved attribution |
| Serving | affected schema tests and observable request/stream behavior |

Record the target, relevant hardware/toolchain, workload or command, and summarized result needed
to interpret a material claim. Hashes, clean worktrees, full command transcripts, raw report
inventories, and exact probabilistic outputs are not default requirements. Use exact comparison for
exact outputs, and appropriate numerical or behavioral criteria otherwise. State checks that could
not run and their implications.

Finish when the deliverable is usable, applicable contracts are satisfied, material claims have
sufficient evidence, relevant checks pass or their limitations are clear, and no known in-scope
issue blocks use. Expand or repeat verification only for new changes, failures, or unresolved risks
that could change the result. Supporting work is not an independent completion objective.

## Reference navigation

Read the authority relevant to the current decision; this is not a mandatory reading list.

| Decision | Entry point |
|---|---|
| Product capabilities and exact commands | `README.md`, executable `--help`; `docs/cli.md`, `docs/serving.md`, `docs/perplexity.md` |
| Execution, family/package ownership, scheduling, transactions, graphs | `docs/maintainer/engine-architecture.md` |
| Context resources, checkpoints, replicas; physical KV | `docs/maintainer/resource-scheduling-and-context-cache.md`; `docs/maintainer/paged-kv-cache.md` |
| Artifact, layout, codec, or exact target mathematics | model/artifact references linked from `docs/README.md` |
| Op contracts, implementation ownership, numerical/performance qualification | `docs/maintainer/op-development.md` |
| Test/benchmark commands and published performance | `tests/README.md`, `bench/README.md`, `docs/performance.md` |
| In-tree C++ interface | `include/ninfer/engine.h`, `include/ninfer/types.h` |

[Documentation map](docs/README.md) routes to narrower authorities when needed.

## Local operations

Use `cmake --build <build-dir> -j` by default. Adjust parallelism when actual resource pressure
causes failures or interferes with the task, and briefly explain why.

Use the selected Python 3.11 interpreter explicitly. On this machine it is
`/home/neroued/miniconda3/envs/py311/bin/python`; the default shell's `python3` may be a different
version. Use `python3` only after selecting the maintainer environment or checking its version.
Normal resources are `build/`, `out/qwen3_6_27b.ninfer`, its `.conversion.json` report, and
`profiles/ncu/`, `profiles/nsys/`, `profiles/bench/`; the local toolchain is CUDA 13.1.
Select model artifacts by explicit path, never glob order, modification time, or unqualified
“latest”. Source checkpoints and large artifacts are prerequisites; download or regenerate them
only when that work is in scope. Install or upgrade dependencies only when the task needs it.

Create commits only when requested. Use Conventional Commit subjects with concise lowercase types
such as `feat`, `fix`, `perf`, `bench`, `test`, `build`, `refactor`, `docs`, or `chore`.
