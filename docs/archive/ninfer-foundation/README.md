# NInfer Foundation Documentation

This archive era begins with the coordinated project-identity cutover from QUS and
`qwen3.6-ultraspeed` to NInfer, followed by the first native `.ninfer` artifact and Python-reference
route. It preserves the completed implementation plans, operational boundaries, and verification
evidence for that foundation.

Use the active documentation index at [`../../README.md`](../../README.md) for current behavior and
current authority boundaries. The material here is historical and does not make the old source,
commands, paths, schemas, or project identity compatible with NInfer.

## Contents

- [`2026-07-14-ninfer-naming-cutover.md`](2026-07-14-ninfer-naming-cutover.md) — completed naming
  cutover plan and evidence record.
- [`2026-07-14-ninfer-artifact-toolchain.md`](2026-07-14-ninfer-artifact-toolchain.md) — completed
  `.ninfer` converter/reader/verifier/binder and complete Python reference migration plan.
- [`2026-07-14-ninfer-engine-atomic-cutover.md`](2026-07-14-ninfer-engine-atomic-cutover.md) — atomic
  replacement of the `.qus` product route by the registered `.ninfer` C++ Engine target.
- [`2026-07-14-generation-transaction-simplification.md`](2026-07-14-generation-transaction-simplification.md)
  — replacement of the layered round/candidate protocols by direct exact-prefix resolution.
- [`2026-07-15-ninfer-op-architecture-and-migration.md`](2026-07-15-ninfer-op-architecture-and-migration.md)
  — definition of the central Op boundary and atomic migration of mathematical CUDA implementations
  out of model-owned and temporary kernel directories.
- [`2026-07-15-gqa-contract-and-execution-refactor.md`](2026-07-15-gqa-contract-and-execution-refactor.md)
  — completed separation of physical KV storage, Program frontiers, device positions, GQA execution
  envelopes, cached-only A3, and frontier-tier CUDA Graphs.

The preserved `.qus` references describe the retired q5090 v4.2 route. The current C++ Engine and
Python reference both consume the registered `.ninfer` artifact; archived `.qus` material is
provenance only.
