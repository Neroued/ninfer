# q5090 v2 Landing Roadmap

How the decode-optimal v2 weight format replaces v1 end to end. This is the high-level sequence; each
phase is gated by [q5090_v2_verification_contract.md](q5090_v2_verification_contract.md) and specified
by [q5090_packed_file_format_v2.md](q5090_packed_file_format_v2.md) +
[qwen3_6_27b_q5090_v2_tensor_plan.md](qwen3_6_27b_q5090_v2_tensor_plan.md).

## End state (the goal this roadmap commits to)

When the roadmap is complete, **only v2 exists**:

- **v1 code removed** — no v1 layout/format/plan code in the converter, Python ref, or cpp runtime; no
  v1 path, no fallback, no compatibility shim.
- **v1 docs archived** — `q5090_packed_file_format_v1.md` and
  `qwen3_6_27b_q5090_final_quant_format_v1.md` (and any v1-only notes) moved under `docs/archive/`.
- **v1 artifacts deleted** — the `…mixed_v1.qus` weight file is removed once cpp runs on v2.
- The runtime, converter, reference, tests, and manifests reference **v2 only**.

The v1 weight *file* is kept alive **only** during the window where cpp has not yet migrated (Phase 1),
so the unported cpp runtime keeps working; it is deleted at cutover (Phase 3).

## Phases

| phase | scope | produces | gate (verification contract) |
|---|---|---|---|
| **0. Spec** (done) | binary spec, tensor plan, verification contract | the v2 contract | docs internally consistent |
| **1. Converter + Python ref** (no cpp) | rewrite converter to emit v2; rewrite Python ref to consume v2; remove Python-side v1 code | v2 `.qus` file; proven-correct Python reference | converter G-STRUCT+G-VALUE; Python ref G-STRUCT/G-VALUE/G-DUMP/G-SNAPSHOT (HF diagnostic) |
| **2. CPP runtime** | weight store + reference backend on v2 (correctness), then row-split GEMV / prefill GEMM (perf; fused-projection GEMV deferred); remove cpp v1 layout code | v2-native cpp runtime | cpp G-STRUCT + G-DUMP + G-KERNEL + G-SNAPSHOT, then perf rounds |
| **3. Cutover & cleanup** | delete the v1 weight file; archive v1 docs; remove residual v1 references | v2-only repository | end-to-end v2 parity + the end-state checklist above |

## Ordering rationale

- **Converter first**: nothing downstream can be tested without the v2 file. Quantization is unchanged
  (value-preserving), so the converter change is layout-only and easy to verify ("same values, new
  bytes").
- **Python ref second**: it is the cheap, flexible **correctness firewall** — it proves the v2 format
  infers correctly (parity vs HF) before any cpp kernel is written. The Python ref and the converter
  share `format.py`/`layouts.py`/`qtypes.py`, so adapting them updates both.
- **CPP last**: the largest, most error-prone work (kernels) runs against a known-good file and a
  known-good reference, with a converter/ref **dump** to diff the cpp parse against before any numerics.
- **Cutover only after cpp is proven on v2**: the v1 weight file is the safety net for the unported cpp
  runtime during Phase 1–2; deleting it and archiving v1 docs is the last step.

Phase 1 is detailed in
[plans/2026-06-29-q5090-v2-step1-converter-and-pyref.md](plans/2026-06-29-q5090-v2-step1-converter-and-pyref.md).
Phases 2 and 3 get their own plans when Phase 1's gate is green.
