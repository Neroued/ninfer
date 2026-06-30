# q5090 v2 → v3 plane-split layout migration — roadmap

> **Status:** Phase 1 complete; Phases 2 and 3 pending. This is a **roadmap** (sequencing + scope +
> exit criteria), not an execution plan. Each pending phase becomes its own detailed (subagent-driven)
> plan when it starts; this document only fixes *what* each phase delivers and *in what order*.

## Goal

Migrate the on-disk weight format from v2 (Q5/Q6 codes packed as a concatenated LSB-first bitstream) to
**v3 plane-split** (low-4-bit nibble plane + separate high-bit plane), so the decode GEMVs reconstruct
each Q5/Q6 value with register-only shift/mask instead of cross-byte funnel-shift + cross-lane shuffle.
v3 is a **pure relayout**: identical dequantized values, identical `bytes_per_token`. Spec:
[../q5090_packed_file_format_v3.md](../q5090_packed_file_format_v3.md).

## Why (evidence)

The roofline push-down round
([2026-06-30-q5090-v2-decode-gemv-roofline-pushdown.md](2026-06-30-q5090-v2-decode-gemv-roofline-pushdown.md),
ledger `profiles/ncu-roofline-pushdown/tuning_status.md`) showed that in-kernel tuning lifted the **Q4**
GEMVs to the DRAM roofline (`mlp_gate_up` 62.7%→93.7%, `gdn_in_qk` 41.8%→64%) but every **Q5** GEMV
stalled with the bitstream unpack as the documented `ncu` limiter (`mlp_down` +1.5pp to ~58%, `proj`
~46%, `out` ~46%, `attn_in` Q5 ~50%; all flagged `layout-relevant? = yes`). Q4 is already a clean
nibble plane; Q5/Q6 are not. v3 gives Q5/Q6 the same nibble-plane ergonomics. (llama.cpp reaches the
same conclusion: its `q5_0`/`q5_K`/`q6_K` all store the high bits in a separate `qh` plane.)

## Cross-phase invariants

- **No backward compatibility** (AGENTS.md): the converter emits v3, the runtime reads v3 only, the
  `.qus` is regenerated; no dual-format path, no fallback.
- **Value preservation is the migration gate.** v3 dequant must be **bit-identical** to v2 dequant,
  value-for-value (v3 spec invariant 0.1). Every phase boundary is gated on this, not on tok/s.
- **Performance-first.** v3 removes the unpack wall; capturing the win still requires a per-kernel perf
  loop in Phase 3 (same protocol as the push-down round).
- **Scope discipline.** v3 changes *only* the Q5/Q6 code packing. It does **not** change scales
  (per-group fp16), the activation path (float accumulation), fusion/segments/schedule, or CUDA graphs.

## Phases

### Phase 1 — Format specification  ✅ done

- **Deliverable:** the complete, standalone v3 binary spec
  ([../q5090_packed_file_format_v3.md](../q5090_packed_file_format_v3.md)) — buildable converter +
  loader from it alone; §18 records the v2 delta.
- **Exit criteria:** spec is self-contained and byte-precise (plane sizes/offsets, bit positions,
  decode formula, representative-block byte table). ✔

### Phase 2 — Python: converter + reference + new weights  (pending)

- **Deliverable:** the converter emits v3; a regenerated `out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus`;
  reference/verifier inference confirmed numerically normal on v3.
- **Scope (areas):**
  - `tools/q5090_convert/packing.py` — v3 plane-split pack/unpack for Q5/Q6 (numpy reference + torch
    fast path, kept bit-equivalent to each other); Q4/W8 packing unchanged.
  - `tools/q5090_convert/layouts.py` — ROW_SPLIT three-plane assembly (nibble / high / scale,
    256-aligned).
  - `tools/q5090_convert/format.py` — magic/version bump (`Q5090MIXEDV3`, version 3); TensorEntry plane
    fields (`nibble_plane_bytes` / `high_plane_bytes` / `scale_plane_bytes`); manifest `..._v3`.
  - `tools/q5090_convert/{tensor_plan.py,convert.py,quantize.py}` — byte-size tables / writer for the
    new plane offsets.
  - `tools/q5090_convert/tests/test_packing.py` — `pack ∘ unpack == id`, and **v3 dequant == v2 dequant
    value-for-value** (the value-preservation gate).
  - Reference / verifier (`tools/parity/**`, `tools/q5090_convert/verify.py`) — read v3; confirm
    inference parity vs the v2 reference.
- **Exit criteria:** `pytest tools/q5090_convert/tests/` green; converter regenerates the `.qus`;
  v3-vs-v2 dequant parity proven; reference-model inference on v3 matches the v2 reference.
- **Note:** Phase 2 produces the artifact Phase 3 consumes; it does **not** touch C++.

### Phase 3 — C++: parser + kernels + perf loop  (pending)

- **Deliverable:** the runtime parses and runs v3; Q5/Q6 decode GEMVs adapted to the plane layout and
  re-tuned to the DRAM roofline; v2 path removed.
- **Scope (areas):**
  - Parser/store: `include/qus/core/weight_store_parser.h`, `src/core/weight_store.cpp` — three-plane
    offsets + asserts (`nibble/high/scale`), magic/version check; `tests/kernels/q5090_pack.h` +
    `tests/test_q5090_pack_golden.cpp` + parser tests updated to v3.
  - Generic codec: `src/kernels/linear/codec/linear_codec.cuh` (`Q5Codec`, `Q6Codec`) — two-plane
    unpack; serves the generic lowbit GEMV/GEMM (prefill + fallback).
  - Specialized decode GEMV (unpack body swap; see analysis below): `linear_rowsplit_gemv_mlp_down`,
    `linear_rowsplit_gemv_proj_6144`, `linear_rowsplit_gemv_out_6144`,
    `linear_rowsplit_gemv_attn_in_7168` (Q5 kernel), `linear_rowsplit_gemv_lm_head` (Q6).
  - Embedding: `embed_gather` (Q6 row dequant, now two-plane).
  - Bench: `bench/linear_op_bench.cu` payload generator emits the v3 plane layout.
  - Then the **per-kernel perf loop** (same protocol as the push-down round) on the now-unblocked
    Q5/Q6 kernels.
- **Exit criteria:** build + `ctest` green; fp64 oracle green for every Q5/Q6 shape;
  `compute-sanitizer` clean; e2e smoke on the v3 artifact; nsys decode re-profile shows the Q5/Q6
  kernels reaching (or approaching) the DRAM roofline; v2 reader/code deleted.

## Phase 3 kernel-impact analysis (adapt, don't rewrite)

The integrated push-down work is the *kernel structure* (warp-per-row vs block-per-row, split-K +
reduce, grid sizing, occupancy, vectorized scale/`x` reads). v3 touches **none** of that — only the
inner per-thread code-unpack changes, and it gets *shorter*.

| kernel / area | qtype | v3 impact |
|---|---|---|
| `mlp_gate_up_34816`, `gdn_in_qk_4096`, `attn_in_7168` (Q4 kernel) | Q4 | **none** — Q4 nibble packing unchanged; keep integrated kernels verbatim |
| `mlp_down` | Q5 | keep split-K + reduce; rewrite only `accumulate_group` (nibble load + small high load + `(low\|high<<4)` + sign-extend) |
| `proj_6144`, `out_6144` | Q5 | keep split-K + occupancy; rewrite only `accumulate_q5_group` |
| `attn_in_7168` (Q5 kernel) | Q5 | keep hybrid (warp-per-row + cooperative KV tail); rewrite only the Q5 unpack (Q4 kernel in the same file untouched) |
| `lm_head` | Q6 | keep grid; rewrite only `accumulate_group` (2-bit high plane) |
| `embed_gather` | Q6 | adapt the single-row dequant to two planes |
| `linear_codec.cuh` (`Q5Codec`/`Q6Codec`), generic lowbit GEMV/GEMM | Q5/Q6 | two-plane unpack body |

The nibble plane is structurally identical to a Q4 code plane, so the already-tuned Q4 vectorized
loader is reused for the dominant (low-bit) traffic; the high plane is a small extra coalesced load.
Phase 3 is therefore a localized unpack swap on ~5 GEMVs + the codec + `embed_gather`, then a perf-loop
pass — not a re-derivation of the structural optimizations already on `master`.

## Sequencing

```
Phase 1 (spec, done)
        │  value-preservation gate (v3 dequant == v2 dequant)
Phase 2 (python: converter + ref + v3 weights)
        │  artifact handoff: out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus
Phase 3 (cpp: parser + kernels + perf loop)  → v2 removed
```

Phase 3 depends on the Phase 2 artifact (the runtime needs a v3 file to load and bench). Each pending
phase is turned into its own detailed plan when it starts; this roadmap does not execute them.
