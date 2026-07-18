# Documentation Archive

This subtree preserves completed plans, retired specifications, dated design investigations, and
benchmark/profiler evidence. It exists for provenance and performance archaeology, not as a current
implementation entrypoint.

Use the active documentation index at [`../README.md`](../README.md) for current behavior.

## Completed correctness audits

- [`2026-07-17-op-numerical-correctness-audit.md`](2026-07-17-op-numerical-correctness-audit.md)
  — independent FP32/FP64 Op-oracle coverage, kernel numerical-path freedom, 27B/35B
  active-document corrections, and the docs-only logical contract for future 35B MoE work.

## Completed operator plans

- [`2026-07-17-sparse-moe-design-log.md`](2026-07-17-sparse-moe-design-log.md)
  — retained closed-Op decisions for the `SparseMoe` leaf used by the registered 35B-A3B Variant.
- [`2026-07-17-sparse-moe-decode-implementation.md`](2026-07-17-sparse-moe-decode-implementation.md)
  — the closed `SparseMoe` decode `T=1` API/workspace, four-stage implementation, independent
  oracle, CUDA Graph test, and RTX 5090 stage qualification.
- [`optimization-era/bench/qwen3.6-35b-sparse-moe-decode-roofline.md`](optimization-era/bench/qwen3.6-35b-sparse-moe-decode-roofline.md)
  — retained candidate, payload-control, full-route, and Nsight Compute evidence for that domain.
- [`optimization-era/bench/qwen3.6-35b-sparse-moe-small-t.md`](optimization-era/bench/qwen3.6-35b-sparse-moe-small-t.md)
  — retained realistic-route timing, candidate rejection, and Nsight Compute evidence for the
  exact-T `T=2..8` CUDA Core/SIMT route.
- [`optimization-era/bench/qwen3.6-35b-w8-input-projection.md`](optimization-era/bench/qwen3.6-35b-w8-input-projection.md)
  — retained complete-Op and Nsight Compute evidence for the 35B W8 Attention and GDN
  multi-output projections.

## Completed cutovers

- [`2026-07-18-qwen3.6-family-runtime-variants.md`](2026-07-18-qwen3.6-family-runtime-variants.md)
  — moved Program/planning/Text/Vision/MTP/state/workspace/graph algorithms into one Qwen3.6
  family runtime, migrated 27B, and registered 35B-A3B as its peer Variant.
- [`2026-07-18-qwen3.6-family-runtime-mechanism-migration.md`](2026-07-18-qwen3.6-family-runtime-mechanism-migration.md)
  — moved hybrid topology, decoder/GDN and round-state layouts/views, MTP alignment, and Vision
  control into the identity-free family leaf while retaining 27B Program/schedule/graph ownership.
- [`2026-07-18-qwen3.6-family-source-composition.md`](2026-07-18-qwen3.6-family-source-composition.md)
  — moved the shared Qwen3.6 frontend, resources, prepared values, and passive Vision bindings into
  family ownership while compiling family and exact-target sources once in `ninfer_engine`.
- [`2026-07-18-qwen3.6-official-checkpoint-cutover.md`](2026-07-18-qwen3.6-official-checkpoint-cutover.md)
  — replaced the 27B Unsloth-derived small files and artifact resources with the pinned official
  Qwen set while preserving every BF16 shard, and added the shared 27B/35B resource preflight.
- [`optimization-era/plans/2026-07-17-qwen3-6-27b-decode-projection-cutover.md`](optimization-era/plans/2026-07-17-qwen3-6-27b-decode-projection-cutover.md)
  — atomic 27B `gdn/value_z` artifact migration, two-parent full-attention input projection, and
  direct GDN final-buffer output without concat D2D copies.

## Superseded plans

- [`2026-07-18-qwen3.6-35b-a3b-runtime-target.md`](2026-07-18-qwen3.6-35b-a3b-runtime-target.md)
  — the unexecuted independent-Program design replaced by the peer-Variant family-runtime plan.

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
