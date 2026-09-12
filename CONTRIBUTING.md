# Contributing to NInfer

NInfer welcomes precise problem reports, reproducible performance evidence, and carefully scoped
code contributions. The most useful first contribution is often an Issue that establishes what is
happening and why it matters. A pull request is an implementation of an agreed change, not the
place to ask the maintainer to discover the problem, choose its semantics, or redesign its core
approach.

## Project scope and decisions

NInfer is an intentionally focused inference engine. Contributions are evaluated against the
supported product described in [`README.md`](README.md), the public documentation map in
[`docs/README.md`](docs/README.md), and the applicable architecture documents. Broader
compatibility or generality is not a benefit by itself when it adds a product contract or
maintenance surface that the project does not intend to own.

The maintainer makes the final decision about product scope, architecture, and long-term
maintenance. Establishing that a bug is real or that an idea is useful does not imply that a
particular implementation will be accepted. The maintainer may preserve the report or core idea
and implement it independently within the current architecture.

## Start with an Issue

Every bug, performance opportunity, feature request, protocol change, and architecture proposal
must begin with an Issue before implementation starts. Use the Issue to establish:

- the concrete problem or use case;
- whether it occurs within the supported product scope;
- the expected behavior or performance claim;
- the affected contract and ownership boundary; and
- whether an external implementation is appropriate.

Wait for the maintainer to confirm the scope and implementation direction before investing in a
pull request. Opening an Issue does not by itself approve a proposed design. A pull request without
a linked, confirmed Issue may be closed without detailed review.

### Bug reports

A bug report must contain enough information for the maintainer to locate and reason about the
failure. Include, as applicable:

- the exact NInfer commit or release and registered artifact identity;
- the GPU, driver, CUDA toolchain, build configuration, and relevant runtime options;
- the complete command, request, or smallest practical reproduction;
- the expected behavior and the observed behavior;
- relevant logs or error output as text;
- the reproduction frequency and any known trigger or boundary condition; and
- checks already performed, including anything that could not be verified.

State clearly when a reproduction is incomplete or when the behavior occurs outside the advertised
product target. An unsupported use case may still motivate a proposal, but it is not a defect in
the supported product contract.

### Performance reports

A performance report must identify the level of the claim: operator, schedule, request phase, or
end-to-end inference. Include the baseline and candidate measurements under comparable conditions,
the exact workload and commands, hardware and toolchain, warmup and repetition method, and a useful
summary of the results.

Account for relevant tradeoffs such as workspace, resident memory, transfer cost, numerical quality,
and effects on other execution paths. End-to-end results can establish an end-to-end observation;
they do not isolate an operator change. A proposed operator optimization should therefore include a
direct operator benchmark and appropriate correctness evidence.

### Feature and architecture proposals

Describe the real supported use case, the limitation of the current behavior, the desired semantics,
and the new contract or maintenance responsibility the project would acquire. Explain why the
existing architecture cannot satisfy the use case. Do not begin a broad implementation until the
maintainer has accepted both the product direction and the proposed ownership model.

## Pull request scope

A pull request is appropriate only after its linked Issue has established the problem and the
maintainer has agreed to its scope and implementation direction. Target pull requests at the
`master` branch. Before implementation, also review the current `dev` branch to avoid duplicating
or conflicting with maintainer work that has not yet reached `master`.

Each pull request must represent one coherent engineering decision:

- address one confirmed Issue with one consistent design;
- include all implementation, tests, tooling, and active documentation required to close that
  decision correctly;
- keep independent bugs, optimizations, refactors, and product changes in separate pull requests;
- exclude unrelated cleanup and speculative compatibility work; and
- avoid changing public semantics or product scope beyond what was agreed in the Issue.

There is no mechanical line-count or file-count limit. A necessary solution may cross several
files, but every changed part must belong to the same contract and be reviewable and verifiable as
one unit. A large change that combines independent decisions, crosses ownership boundaries without
prior agreement, or requires the maintainer to reconstruct its design is not maintainable and will
not be accepted. Use the Issue, not a draft implementation, to resolve architecture and scope.

## Engineering and verification

Read the active documentation governing the code being changed and preserve its ownership and
semantic contracts. In particular:

- validate exact transformations and formats with exact comparison;
- validate floating-point work against an independent mathematical oracle with a justified
  numerical criterion;
- verify the complete affected transition for stateful behavior;
- exercise relevant supported shapes and execution routes for CUDA changes;
- measure performance at the level being claimed; and
- keep externally observable implementation, schema tests, and documentation consistent.

Use the existing project workflows:

- [`tests/README.md`](tests/README.md) for test organization and commands;
- [`bench/README.md`](bench/README.md) for product and operator benchmarks; and
- [`docs/maintainer/op-development.md`](docs/maintainer/op-development.md) for numerical Op
  development and qualification.

Run the focused checks needed to establish the changed behavior and its material claims. Record the
exact commands and summarize the results. If a relevant check could not be run, state that
limitation and its consequence. Do not claim validation that was not performed.

## Contributor responsibility and AI assistance

The contributor is responsible for every submitted change, regardless of which tools were used to
produce it. Tool choice neither qualifies nor disqualifies a contribution.

The contributor must understand the complete diff, explain the design and its boundary conditions,
review generated output, perform the stated verification, and respond to technical review. Code
that the contributor cannot explain, validate, or maintain is not ready for submission. Maintainer
review is not a substitute for contributor debugging or engineering supervision.

## Pull request description

A pull request description must include:

- the linked Issue and the agreed scope;
- the concrete design and why it fits the current architecture;
- the affected behavior, ownership boundary, or public contract;
- the exact verification commands and summarized results;
- the workload, hardware, toolchain, and methodology for any performance claim; and
- every relevant check that was not run and the resulting limitation.

## Review policy

Review is a limited maintainer resource and is not guaranteed. A pull request may be closed without
a line-by-line review when it:

- was submitted before its problem, scope, or implementation direction was confirmed;
- falls outside the supported product or expands an unapproved contract;
- combines independent changes into an unreviewable implementation;
- contains evident correctness, ownership, lifetime, state, or boundary problems;
- lacks evidence appropriate to its correctness, numerical, or performance claims;
- cannot be explained and supported by its contributor; or
- would require the maintainer to redesign, split, debug, or complete its core implementation.

Closing a pull request does not necessarily reject the underlying report or idea. It means that the
submitted implementation does not meet the project's review and maintenance threshold. The
maintainer may address a confirmed problem through a separate implementation and reference the
original report where appropriate.
