# Documentation Archive

This subtree preserves completed plans, retired specifications, dated design investigations, and
benchmark/profiler evidence. It exists for provenance and performance archaeology, not as a current
implementation entrypoint.

Use the active documentation index at [`../README.md`](../README.md) for current behavior.

## Completed correctness audits

- [`2026-07-17-op-numerical-correctness-audit.md`](2026-07-17-op-numerical-correctness-audit.md)
  — independent FP32/FP64 Op-oracle coverage, kernel numerical-path freedom, 27B/35B
  active-document corrections, and the docs-only logical contract for future 35B MoE work.

## Eras

- [`pre-optimization/`](pre-optimization/) — L0/L1/L2 foundation, M2 hardening, and the M2.8
  pre-optimization gate.
- [`optimization-era/`](optimization-era/) — subsequent linear, attention, MTP, INT8 KV, q5090,
  serving, and Vision design/implementation work through the v4.2-native runtime, including the
  completed Linear architecture refactor, inventory, Q4 template design, and experiment record.
- [`ninfer-foundation/`](ninfer-foundation/) — the completed project-identity, native `.ninfer`
  toolchain, Python reference, and C++ Engine cutovers, including implementation plans and evidence.

## Archive rules

- Status labels inside archived files describe their original moment in time.
- Archived documents are not maintained when source paths, commands, APIs, or artifact formats
  change.
- Retired q5090 specifications do not imply compatibility. The current C++ Engine and native Python
  reference both consume the registered `.ninfer` route; archived `.qus` material is provenance
  only.
- Historical benchmark numbers are meaningful only with their recorded commit, artifact, command,
  hardware, and profiler context.
- Active documents may link here for rationale or provenance, but not to outsource a normative
  current requirement.
