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

The preserved `.qus` references describe the q5090 v4.2 artifact that still feeds the current C++
Engine. The native `.ninfer` toolchain and Python reference are implemented, while C++ Engine
execution over `.ninfer` remains a later migration.
