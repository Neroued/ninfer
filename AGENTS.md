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

For implementation tasks, record the user's requirements, promised work, affected implementations
and routes, and the quality dimensions to be evaluated before judging results. For comparative
claims, also identify the baseline, workloads, metrics, aggregation method, and acceptance criteria.
Keep this accounting current as the task develops. Explain justified changes and preserve the
earlier findings; never narrow the scope or change the criteria after seeing unfavorable results
to make the task appear successful. This accounting may live in the working conversation or
existing task artifacts; it does not require a new permanent planning document.

For analysis or design, deliver the explanation or design. For diagnosis, establish the cause and
supporting evidence; implement a fix when requested. For implementation, complete the selected
design across its affected implementations, callers, tests, tools, and active documentation.

The current product and architecture govern ordinary work. An explicit task may change them;
update the affected contracts and implementation together instead of treating the current design
as an immutable prohibition. Skills provide task-specific methods, not additional deliverables or
approval requirements beyond the user's instructions and the actual execution environment.

## Product and architecture

NInfer is a from-scratch C++/CUDA inference engine for maximum single-GPU performance. It implements
`Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`; official Qwen3.6/3.8 artifacts and user recipes
use the same architecture, binding and execution path. The implementation targets `sm_120a` and
is tuned on NVIDIA GeForce RTX 5090.

Generation uses one GPU, one resident model, startup-fixed concurrency of one to eight requests,
bounded FIFO ingress, no active-request preemption, and one compact decode batch per round.
Generation and offline CausalScoring use the same public `.ninfer` Engine route. Delivered
capabilities and commands are documented in `README.md`, the product guides, and executable
`--help`. New mathematical architectures, execution platforms, large-scale/preemptive continuous
batching, and priority/QoS require an explicit product change. Another training instance or mixture
of existing representations does not require a checkpoint-specific execution registration.

This is a local, single-owner project with trusted local models, generated artifacts, and
local workflow. Do not derive requirements from a different deployment or trust model.

Keep these ownership boundaries visible when selecting a design:

- v3 `.ninfer` is the only C++ product artifact; CLI, serving, and inference benchmarks use the public
  Engine. NInfer has no Python model-inference route or installed/exported C++ SDK.
- Core owns physical primitives and raw transfers; artifact owns generic framing and
  materialization; Ops own closed mathematical and state-transition implementations.
- Models own fixed mathematics, config interpretation, logical parameter binding, frontend
  semantics and finite execution composition. Immutable Model data owns selected weights and
  resources; native Parameters supply the actual operands to planning and Program execution.
  Program owns mutable state, workspace, context stores and CUDA Graphs. Programs share no mutable
  state or device allocation.
- Converter recipes choose sources, formats, packing and per-input activation permissions. The
  loader validates, uploads and binds the stored representation. Native preparation, resource
  queries and execution enforce actual Op support; there is no whole-artifact capability registry.
- Runtime owns common execution contracts and Engine publication policy; product/serving own input
  acquisition and protocol translation. Model code does not acquire media or own transport.

Detailed model/runtime responsibilities and source ownership are defined in
[Engine architecture](docs/maintainer/engine-architecture.md). Read the relevant boundary before
changing it. Prefer explicit implementations for supported architectures. Do not introduce generic model
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
issue blocks use. Completion also requires the complete, non-selective final report specified
below. Before declaring completion, reconcile the report against the user's requirements, the
affected scope, and all collected findings. If any agreed completion condition remains unmet,
state which parts are complete and which remain incomplete; disclosing an unmet condition does
not satisfy it. Expand or repeat verification only for new changes, failures, or unresolved risks
that could change the result. Supporting work is not an independent completion objective.

## Complete reporting and evidence integrity

These requirements are mandatory for every implementation task. Selective reporting and evidence
gaming are prohibited integrity violations, even when every disclosed statement or number is
true. Never use selection, omission, comparison choices, or aggregation to manufacture an
impression of success that the full results do not support.

1. **Account for the entire deliverable.** The final reply must cover every user requirement,
   promised item, affected implementation or route, and evaluated aspect. State whether each is
   complete, partially complete, incomplete, or unverified. Report different types, components,
   and routes separately when their outcomes differ. A successful subset must never stand in for
   the whole task.
2. **Describe all known effects of the chosen design.** Cover the functionality, mathematical
   semantics, interfaces, architecture, performance, resource usage, complexity, and maintenance
   costs affected or evaluated in this task. Explain benefits, disadvantages, capability losses,
   and retained trade-offs. A favorable metric cannot substitute for the other dimensions.
   Distinguish no observed change, no evaluation, and not applicable; never call an unchecked
   aspect unaffected or not applicable.
3. **Put unfavorable findings in the final reply itself.** Disclose all known failures,
   regressions, slowdowns, additional costs, unresolved problems, and verification gaps. Being
   unfavorable, small, below an alert threshold, or not yet explained is never a reason to omit
   a result. Thresholds determine interpretation and action, not whether a finding is disclosed.
4. **Account for unsuccessful and inferior approaches actually evaluated.** Explain their
   observed problems, why they were rejected or retained, and any disadvantages carried into the
   final implementation. Include problems found and subsequently fixed, together with the
   correction and verification outcome. Repeated experiments may be grouped by common cause or
   outcome, but every failure category, affected scope, and distinct exception must remain
   visible; rejected results must not disappear from the account.
5. **Use comparable baselines and label the nature of each gain.** Distinguish adding a missing
   capability, replacing an inefficient fallback, and improving an existing optimized
   implementation. Never relabel the first two as improvements to an existing optimized
   implementation. Whole-system claims require measurement at that scope; an isolated path's
   gain is insufficient. State the comparison conditions and explain differences in workload,
   precision, semantics, hardware, execution mode, or measurement scope that affect interpretation.
   Do not select a weaker baseline merely because it yields a larger gain.
6. **Summarize the full evaluation set.** Identify the evaluated objects, workloads, and metrics,
   then account for improved, unchanged, worse, and inconclusive results. Quantitative summaries
   must state their coverage, distribution, worst changes, and exceptions. Never report only a
   best case, maximum speedup, or favorable average. Aggregation must not hide a regression or
   imply uniform improvement. Grouping is allowed only when the scope and differing outcomes
   remain clear.
7. **Apply the same evidentiary standard to good and bad results.** A small improvement is not
   automatically a proven gain, and a small slowdown is not automatically noise. Mark unproven
   interpretations as uncertain and explain the missing evidence. Changes to the test set,
   baseline, measurement method, aggregation, or acceptance criteria must have a stated
   justification and preserve earlier adverse findings. Never make such changes to obtain a
   more favorable conclusion.
8. **Make the final reply self-contained.** Attachments, logs, raw data, and progress messages
   support the report; they do not replace it. The user must not have to inspect an attachment
   or reconstruct earlier messages to discover an unfavorable result, omitted work, or a
   limitation. Brevity may compress wording and group repeated findings, but must never remove
   a required aspect, adverse result, or qualification.
9. **Keep completion claims within the evidence.** A successful build, passing tests, or a
   completed commit establishes only the corresponding fact. None substitutes for correctness,
   performance, architectural, or other task-specific acceptance. Do not claim that a task or
   optimization is complete while an agreed completion condition remains unmet. Explain the
   actual status of each part and the remaining work.
10. **Check report completeness before sending it.** Reconcile the final reply with the task
    accounting and all available results, including abandoned approaches and unfavorable
    findings. Correct unsupported conclusions and omissions before claiming completion. Do not
    wait for the user to detect missing evidence or challenge a favorable headline.

Every implementation final reply must contain the following information. The presentation may
combine items, but none may be omitted:

- The user's requirements and the completion status of each.
- The actual changes and their affected scope.
- Design choices, unsuccessful approaches, and retained disadvantages or trade-offs.
- Results across all affected or evaluated dimensions, including unchanged outcomes,
  regressions, and additional costs.
- Verification evidence, failures, checks not run, and coverage limits.
- Remaining issues and an evidence-supported completion conclusion.

Completeness concerns this task's agreed scope, actual work, and evaluated approaches. It does not
require unrelated audits, exhaustive hypothetical configuration sweeps, permanent test matrices,
or new reporting tools. Known unfavorable findings must be reported; unknown or unverified
aspects must remain explicitly unknown or unverified.

## Reference navigation

Read the authority relevant to the current decision; this is not a mandatory reading list.

| Decision | Entry point |
|---|---|
| Product capabilities and exact commands | `README.md`, executable `--help`; `docs/cli.md`, `docs/serving.md`, `docs/perplexity.md` |
| Execution, model/runtime ownership, scheduling, transactions, graphs | `docs/maintainer/engine-architecture.md` |
| Context resources, checkpoints, replicas; physical KV | `docs/maintainer/resource-scheduling-and-context-cache.md`; `docs/maintainer/paged-kv-cache.md` |
| Artifact, layout, codec, conversion, or model mathematics | model/artifact references and conversion guide linked from `docs/README.md` |
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
