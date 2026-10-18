# AGENTS.md

## Product and architecture

NInfer is a C++/CUDA inference engine for maximum single-GPU performance. Execution targets are
selected at compile time: `rtx5090` (Linux x86-64, `sm_120a`) and `dgx_spark` (Linux AArch64,
`sm_121a`). Follow [Build system](docs/maintainer/build-system.md) for target composition.
Choose designs for functional and numerical correctness, clear ownership, and the performance goal
within the requested scope.

It implements `Qwen3_5ForCausalLM`, `Qwen3_5MoeForCausalLM`, and Text/Vision/MTP for
`Qwen4ExpForCausalLM`. Checkpoints and recipe combinations
of existing representations use the same architecture, binding and execution path; do not add
checkpoint-specific execution registration. Generation uses one GPU, one resident model, and
one to eight resident execution lanes fixed at startup.

V3 `.ninfer` is the only C++ product artifact. Generation, offline CausalScoring, CLI, serving, and
inference benchmarks use the public Engine. There is no Python model-inference route or
installed/exported C++ SDK.

This is a local, single-owner project with trusted local models, artifacts, and workflows.
Do not derive requirements from another deployment or trust model.

ModelDefinition owns immutable model metadata and Frontend resources. Each rank owns its
ModelShard, Parameters and RankProgram. ModelSession owns logical continuations, checkpoints and
the ContextDirectory; RankProgram owns physical storage, execution frames and CUDA Graphs.
Executor transfers owning work and completion events between them. Single-device execution uses
LocalExecutor on the Engine worker thread. The loader validates, uploads and binds the stored
representation; actual Parameters and Ops determine execution support, without a whole-artifact
capability registry.
Runtime owns execution and publication policy; product/serving own input acquisition and protocol
translation. Model code does not acquire media or own transport.

Keep architecture, binding, and execution explicit. Without a product requirement, do not introduce
generic model graphs, family base classes, plugin discovery, string-driven execution, hidden device
allocation, or runtime weight repacking. New mathematical architectures, execution platforms,
large-scale continuous batching, or priority/QoS require an explicit product change. When a task
changes product or architecture, update affected contracts and implementation together.

## Change policy

Project-owned contracts do not preserve backward compatibility. Replace behavior completely:
remove superseded aliases, fallbacks, transition branches, and their tests within the affected contract.
Advertised OpenAI and Anthropic protocols are external contracts; update affected schema tests
and serving documentation together.
Update stable requirements in their existing authoritative document; maintain one current authority.
Use Conventional Commit subjects with concise lowercase types when a commit is requested.

## Verification and reporting

Before changing Ops, numerical or state semantics, read
[Op development](docs/maintainer/op-development.md). Qualify affected production routes directly
against the contract's independent mathematical oracle or specified exact reference. Kernel parity
and plausible model output do not establish mathematical correctness.

Measure performance at the claimed scope. An Op microbenchmark establishes an Op result,
not an end-to-end improvement. For comparisons, establish the baseline, workloads, metrics,
aggregation, and acceptance criteria before evaluating results.
Distinguish new capability, fallback replacement, and improvement to an optimized implementation.
Report the baseline, hardware/toolchain, workloads or commands, run conditions, metrics, coverage,
result distribution, worst changes, and exceptions. Include small and unexplained regressions;
do not dismiss slowdowns as noise without evidence.

## References

Read [Engine architecture](docs/maintainer/engine-architecture.md) before changing execution or
ownership. [README](README.md) and executable `--help` define capabilities and exact commands.
The [documentation map](docs/README.md) routes to detailed contracts;
[Tests](tests/README.md) and [Benchmarks](bench/README.md) own their commands and execution details.

## Local environment

Use Python 3.11 via `/home/neroued/miniconda3/envs/py311/bin/python` explicitly; use `python3` only
after selecting the maintainer environment or checking its version.
The usual local model is `out/qwen3_8_27b_nvfp4.ninfer`. Select artifacts by explicit path, never glob order,
modification time, or unqualified “latest”. Source checkpoints and large artifacts are prerequisites;
download or regenerate them only when that work is in scope.

Store all agent-generated temporary files under the repository-root `.agent-tmp/<task>/`,
including temporary plans, experimental code and binaries, test data, logs, profiling captures,
performance reports, and intermediate artifacts. This directory is ignored by Git; do not commit
its contents. Remove temporary files that are no longer needed when the task ends.
Reserve `out/` for model weights, their companion files, and explicitly retained important artifacts.
Do not use `out/` for temporary experiments or task output.
