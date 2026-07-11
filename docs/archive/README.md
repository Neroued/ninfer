# Documentation Archive

This subtree preserves completed plans, retired specifications, dated design investigations, and
benchmark/profiler evidence. It exists for provenance and performance archaeology, not as a current
implementation entrypoint.

Use the active documentation index at [`../README.md`](../README.md) for current behavior.

## Eras

- [`pre-optimization/`](pre-optimization/) — L0/L1/L2 foundation, M2 hardening, and the M2.8
  pre-optimization gate.
- [`optimization-era/`](optimization-era/) — subsequent linear, attention, MTP, INT8 KV, q5090,
  serving, and Vision design/implementation work through the v4.2-native runtime.

## Archive rules

- Status labels inside archived files describe their original moment in time.
- Archived documents are not maintained when source paths, commands, APIs, or artifact formats
  change.
- Retired q5090 specifications do not imply runtime compatibility; the current runtime accepts only
  the format named by the active q5090 specification.
- Historical benchmark numbers are meaningful only with their recorded commit, artifact, command,
  hardware, and profiler context.
- Active documents may link here for rationale or provenance, but not to outsource a normative
  current requirement.
