# NInfer Project Positioning

> Status: accepted on 2026-07-13.
>
> Authority: this document defines why NInfer exists, whom it serves, the targets and workloads it
> chooses, the outcomes it optimizes, and the boundaries that preserve that purpose. It does not
> define current implementation status, model mathematics, artifact or container formats,
> repository structure, APIs, migration work, or implementation strategy. Those subjects belong to
> their own documents.
>
> The project name is defined separately by [`ninfer-naming.md`](ninfer-naming.md). This document is
> an accepted direction for the elevated project, not a claim that the current repository has
> already completed that transition.

## 1. Purpose

NInfer is an end-to-end local inference engine for a small set of deliberately selected model
checkpoints and GPU targets. It pursues maximum practical inference performance on those exact
targets instead of broad model coverage, automatic compatibility, or portability across arbitrary
hardware.

NInfer exists for users who want the highest inference performance obtainable from a single local
GPU. It deliberately trades breadth for depth: each chosen checkpoint and hardware combination is
treated as a concrete target to be understood, implemented, measured, and optimized in depth rather
than as one instance of a generic runtime.

The concise project position is:

> Selected checkpoints. Selected GPUs. Maximum single-GPU inference performance.

The corresponding value judgment is:

> NInfer is not measured by how many models it can load, but by how well it runs the targets it
> deliberately chooses.

## 2. Exact optimization targets

The unit of work in NInfer is an exact optimization target:

```text
exact model checkpoint + selected GPU + defined local workload
```

A model family is not an optimization target. Checkpoints from the same family may differ in
dimensions, layer structure, expert topology, modalities, auxiliary predictors, state semantics, or
other facts that materially change inference. Selecting one checkpoint therefore does not imply
that another checkpoint is compatible, even when their architecture labels or configurations look
similar.

Hardware is selected with the same precision. A target is evaluated on a concrete GPU product, not
inferred from a vendor, generation, or instruction-set label. The NVIDIA RTX 5090 is currently the
only hardware target. Other GPUs may be added later, but each addition is a deliberate project
decision and a new optimization effort rather than an automatic expansion of compatibility.

Checkpoint selection is intentional and maintainer-directed. Relevant considerations include:

- the checkpoint's practical value to local users;
- whether it can run well on one selected GPU without relying on offload;
- whether its complete native inference capabilities can be delivered;
- whether specialization offers a meaningful performance opportunity;
- whether its quality, behavior, memory use, and performance can be evaluated credibly;
- whether the resulting target can be maintained without weakening existing targets.

NInfer does not define a generic onboarding contract for arbitrary third-party checkpoints. It does
not measure progress by the number of model families, configuration variants, or files it happens
to recognize.

## 3. User, workload, and deployment boundary

NInfer is designed first for local, single-user inference. Its primary workload is one active
request on one GPU, with low end-to-end latency and maximum single-stream generation performance.

Limited concurrency may be introduced where it improves practical local use. It remains secondary
to the single-request path and must not redefine the project around aggregate multi-user
throughput. Large-scale concurrency, multi-tenant serving, and data-center scheduling are not target
workloads.

The current deployment unit is one host with one selected GPU. Performance-critical model
execution and model state must fit and run on that GPU. CPU assistance may exist around the
inference path, but CPU weight offload, layer-by-layer transfer, and CPU inference fallback are
outside this boundary. A checkpoint that depends on them is not viable for that GPU target.

Multi-GPU and distributed execution are not planned capabilities, and NInfer does not reserve
complexity for them in advance. If real hardware availability and project goals change, multi-GPU
work would require an explicit revision of this positioning rather than being treated as an
obligation hidden in the current project.

## 4. End-to-end product scope

NInfer is a usable local inference product, not only a collection of accelerated kernels and not
only a model-forward library. Its product boundary extends from the checkpoint's native inputs to
generated output.

For a selected checkpoint, that boundary includes the behavior required to use its full set of
native inference capabilities:

- checkpoint-appropriate input interpretation and preparation;
- native input modalities where the checkpoint has them;
- checkpoint-appropriate tokenization and generation semantics;
- the complete inference and generation path;
- final user-visible output;
- practical local invocation and limited local-service access.

Command-line and local service interfaces are ways to expose the product. Compatibility with a
particular hosted-service protocol may be useful, but protocol breadth is not NInfer's identity and
does not turn it into a general serving platform. End-to-end behavior and performance on the
selected local workload take precedence over serving-framework breadth.

The intended boundary for a chosen checkpoint is its full set of native inference capabilities.
Work may be delivered in stages because modalities and model components differ in complexity and
dependencies. Staging is a development fact, not a hierarchy of support labels: documentation
should state what exists, what has been verified, and what limitations remain without introducing
separate partial- and full-support classifications.

## 5. Performance priorities

NInfer optimizes the complete user-visible inference path. Its priorities are ordered as follows.

### 5.1 Decode efficiency

Single-stream decode is the primary performance objective. NInfer seeks low per-token latency and
high useful output-token throughput for one active request, across the context lengths that matter
for the selected checkpoint.

Internal work does not become user-visible throughput merely because it executes quickly. The
project-level result is the rate and latency of actual generated output through the complete
inference path.

### 5.2 Prefill efficiency

Prompt processing must remain efficient across relevant prompt lengths. Prefill throughput and
time to first token are first-class outcomes, not costs that may be ignored while optimizing
steady-state decode.

### 5.3 Context capacity and long-context behavior

GPU memory use, including the state that grows with context, must be optimized for the selected
single-GPU deployment. The goal is to make the verified usable context approach the checkpoint's
native context limit as closely as practical while preserving acceptable quality and useful
performance.

Two context limits must remain distinct:

- the checkpoint's native context limit, established by the checkpoint's own semantics and
  published behavior;
- NInfer's verified context limit for one exact checkpoint, GPU, and current optimized path.

NInfer aims to make the second reach the first, but it reports only lengths that have actually been
validated. An advertised or theoretical limit is not a delivered result. Long-context decode speed,
stability, and memory behavior are part of the evidence rather than footnotes to a capacity claim.

### 5.4 Limited concurrency

Future low-concurrency operation may improve local utilization, but it ranks after the
single-request goals above. It must not silently sacrifice the primary decode path, usable context,
or local latency in order to maximize aggregate request throughput.

## 6. One current optimized route

For each exact checkpoint and hardware target, NInfer maintains one current optimized product
route. It does not offer parallel high-fidelity, balanced, or maximum-speed editions.

The objective is:

```text
maximize practical inference performance
subject to acceptable model quality and correct checkpoint-native behavior
```

This is a constrained performance objective, not a collection of user-selectable quality tiers.
Operational settings that change request shape or capacity do not create separate quality
editions.

Numerically lossy techniques are permitted when their effect remains acceptable for the selected
checkpoint. Bitwise reproduction of a reference floating-point path is not the project goal. The
exact source checkpoint remains the semantic target, while its NInfer execution may be a validated
high-performance approximation.

At the same time, performance must not be obtained by concealing material quality loss, silently
removing checkpoint-native capabilities, or counting incorrect behavior as useful work. Functional
correctness and implemented product behavior remain hard constraints even when numerical
approximation is allowed. The checkpoint's full set of native capabilities remains the product
boundary even while those capabilities are delivered in stages.

Acceptability is determined with evaluations appropriate to the checkpoint and its real uses, not
with one universal numerical threshold imposed on every model. When a better performance choice
within the accepted quality and capability constraints is found, it replaces the previous project
choice instead of accumulating as another user-facing edition.

## 7. End-to-end evidence

Performance claims must name the exact checkpoint, exact GPU, relevant workload, quality
conditions, and verified context. They must come from the real inference path rather than from an
isolated component presented as a proxy for the whole product.

Evidence should distinguish the outcomes that matter to users:

- useful decode throughput and per-token latency;
- end-to-end generation speed and latency;
- prefill throughput and time to first token;
- behavior at short, representative, and long contexts;
- memory use and maximum verified context;
- limited-concurrency behavior when that capability exists;
- the quality and capability conditions under which the result was obtained.

Component benchmarks and profiler results are valuable for identifying and explaining bottlenecks.
They do not by themselves prove a project-level performance improvement. A faster isolated
operation is meaningful only when it preserves the accepted constraints and improves a relevant
end-to-end outcome.

## 8. Specialization and reuse

Specialization is a deliberate means to NInfer's goal, not a temporary deficiency awaiting a
general framework. Target-specific choices are valid when they produce a meaningful performance or
clarity benefit without violating the accepted quality and capability constraints.

Reuse is valuable when two real targets demonstrably share the same semantics and requirements. It
is not a goal that overrides target performance. Adding a checkpoint or GPU must not force existing
optimized paths into a lowest-common-denominator design, and NInfer does not create complexity
solely for hypothetical future targets.

NInfer's selected target set may change or expand over time. This is not a promise that every target
shares one execution strategy, that family similarity implies compatibility, or that every project
subsystem must become model- and hardware-agnostic.

## 9. Evolution and compatibility

NInfer is not a compatibility-first project. Project-owned interfaces, tools, workflows, and other
contracts may be replaced directly when a clearer design or a materially better result requires
it. Preserving an obsolete project-owned path is not an independent goal.

This freedom does not make advertised external behavior disposable. An external protocol or
user-visible contract that NInfer currently offers is real while it is offered and must be changed
deliberately and coherently. The distinction is between maintaining an intentional product contract
and retaining historical internal choices merely because they once existed.

## 10. Non-goals

NInfer is not intended to be:

- a general-purpose model runtime or dynamic model graph;
- an automatic loader for arbitrary checkpoints;
- a model zoo whose success is measured by recognized architectures;
- a compatibility layer that treats family membership or similar configuration as sufficient;
- a portable backend for every GPU, accelerator, or CPU;
- a CPU-offload system for checkpoints that do not fit the selected GPU;
- a tensor-parallel, pipeline-parallel, multi-GPU, or distributed inference system under the current
  project direction;
- a large-scale batching, multi-tenant, or data-center serving framework;
- a protocol-coverage project for reproducing every hosted inference API;
- a product with parallel high-fidelity, balanced, and maximum-speed editions for the same target;
- a benchmark project that substitutes isolated kernel peaks for real end-to-end results;
- a framework whose abstractions or backward compatibility take priority over performance on its
  selected targets.

These boundaries do not claim that the excluded goals are unimportant. They state that pursuing
them would dilute the reason NInfer exists. Changing a foundational boundary requires an explicit
project decision rather than gradual expansion by implication.

## 11. Success criteria

NInfer succeeds when deep knowledge of a selected checkpoint and GPU produces local, single-GPU
inference performance that a general-purpose engine cannot easily match, while retaining acceptable
quality and the checkpoint's native capabilities, and while pushing the verified usable context as
close as practical to the checkpoint's native limit.

Success is not established by:

- merely loading or executing a checkpoint;
- recognizing many model families;
- matching a general runtime without a meaningful target-level advantage;
- publishing an isolated operation with a high peak number;
- claiming a context length that has not been exercised and verified;
- omitting quality loss or missing capabilities from the performance result.

The relevant result is the complete local user experience on an exact target: useful output speed,
latency, prompt handling, context capacity, quality, and native capabilities working together.

## 12. Decision rule

When a project decision trades generality against a measurable benefit on an actual NInfer target,
NInfer favors the actual target. General mechanisms are justified by demonstrated commonality among
real selected targets, not by the possibility that an unspecified checkpoint, device, or workload
may appear later.

Any proposal that changes the checkpoint-by-checkpoint targeting model, selected-hardware policy, single-GPU
deployment boundary, local low-concurrency workload, single optimized route, or end-to-end product
scope must revise this positioning explicitly rather than expanding the project by implication.

## 13. Deliberately separate decisions

This document does not decide:

- artifact or container semantics;
- binary framing, versioning, identity, integrity, or compatibility mechanisms;
- model architecture mathematics or state behavior;
- repository ownership, module layering, public APIs, or source layout;
- numerical formats, packing, memory layouts, kernels, or execution schedules;
- command-line options, protocol schemas, or transport behavior;
- the active checkpoint list or implementation roadmap;
- project-renaming mechanics or migration order.

Those decisions must be made in their own authoritative documents. They must serve this positioning
without turning NInfer into a general runtime or weakening its selected-target performance goal.
