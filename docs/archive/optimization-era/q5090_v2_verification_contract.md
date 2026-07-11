# q5090 v2 Verification Contract

What "correct" means for the v2 weight format, and the pass/fail gates shared by the three landing
phases — **converter**, **Python reference**, **cpp runtime**.

Companion docs: byte contract = [q5090_packed_file_format_v2.md](q5090_packed_file_format_v2.md);
assignment = [qwen3_6_27b_q5090_v2_tensor_plan.md](qwen3_6_27b_q5090_v2_tensor_plan.md).

## 0. Principle (read this first)

Quantized inference is **supposed to diverge from the bf16 reference**, and that divergence
**accumulates with depth**. Measured on this model, per-layer hidden-state cosine vs HF bf16 decays
monotonically to ~0.92 by layer 63 — this is normal quantization behavior, **not** a defect, and a model
in this state still produces the correct tokens. Therefore:

- **Hard gates are EXACTNESS against the model's own reference**: the quantizer values, a recorded
  quantized greedy snapshot, and (for cpp) a CUDA-vs-cpp fp64 kernel oracle. These are deterministic —
  they pass, or they expose a real bug.
- **Every comparison against HF bf16 is DIAGNOSTIC** — reported to localize gross errors, **never** a
  pass/fail threshold.

This split is mandatory and non-negotiable: **an HF-closeness check must never gate**, and **an
exactness check must never be loosened** to absorb quantization error. (Phase 1 was blocked precisely
by violating this: an HF per-layer cosine gate at ≥0.999/0.998 failed on normal quant drift, and an HF
per-row 0.98 floor failed on normal Q4 rows at ≈0.977.)

## 1. Oracles and references

| oracle | kind | role |
|---|---|---|
| **Quantizer** (`tools/q5090_convert/quantize.py`) | exact | value-preservation gate |
| **Recorded quantized greedy snapshot** (e.g. `profiles/e2e/m3-output-gate.json`) | exact | authoritative end-to-end gate — the engine must reproduce it |
| **cpp fp64/CPU reference** (`qus_linear_test`) | bf16 tolerance | cpp kernel cross-validation gate (CUDA↔cpp) |
| **HF bf16** (`/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16`) | **diagnostic only** | report-only sanity; **never** a gate |

The snapshot is the known-good quantized greedy output recorded from the prior runtime; since v2 is
value-preserving, reproducing it proves correctness. The **Python reference is a Phase-1 engine**
(validated by value-preservation + the snapshot); it is **not** a per-op oracle for the cpp runtime.

## 2. Hard gates (exactness — pass/fail)

- **G-STRUCT** — ABI valid + plan-conformance: header/offsets in range and ordered, payloads
  256-aligned and non-overlapping, segments partition `[0,N)`, fusion adjacency + `source_kind` rule,
  plane-byte formulas; and the file matches the tensor-plan expected manifest item-by-item
  (block/segment/fusion identities, names, shapes, row ranges). Converter and offline verification
  tools may recompute per-block `crc32`; the cpp runtime loader does not recompute payload CRC during
  normal model load.
- **G-VALUE** — recovered `(scale16, q_i)` from the file are **bit-identical** to the quantizer output.
- **G-DUMP** — the converter dump and each engine dump (Python, cpp) agree **bit-exact** on
  block/segment/fusion metadata and on sampled `(scale16, q_i)`; sampled dequant agrees within a tiny
  tolerance (compare the recovered codes/scales, not dtype-dependent dequant floats).
- **G-KERNEL** (cpp) — every CUDA GEMV (`T=1`) and GEMM (`T>1`) matches a **cpp fp64/CPU reference of
  the same quantized weights** within a **bf16-appropriate tolerance** (`qus_linear_test`). This is the
  CUDA↔cpp cross-validation; **no Python is involved**.
- **G-SNAPSHOT** — the engine reproduces the recorded quantized greedy snapshot **exactly**, for the
  snapshot's **own** token length (do not assume a fixed count).

## 3. Diagnostics (reported — never gate)

- **D-DEQUANT** — per-tensor dequant vs HF bf16: report max-abs / RMS / per-row cosine. A WARN may flag
  gross outliers (a layout scramble shows as broadly low cosine), but quantization-level per-row dips
  (e.g. Q4 rows at cos ≈ 0.977) are **normal and do not fail**.
- **D-LAYER** — per-layer hidden-state cosine vs HF and the first divergent layer. Expected to decay
  with depth (to ~0.92 deep); informational.
- **D-GREEDY-HF** — first token index where the quantized greedy diverges from HF; informational.

## 4. Per-phase gate sets

| phase | hard gates | diagnostics |
|---|---|---|
| **Converter** | G-STRUCT, G-VALUE | — |
| **Python ref** (Phase 1) | G-STRUCT, G-VALUE, G-DUMP, G-SNAPSHOT | D-DEQUANT, D-LAYER, D-GREEDY-HF |
| **cpp runtime** (Phase 2) | G-STRUCT (parse, excluding payload CRC recomputation), G-DUMP, G-KERNEL, G-SNAPSHOT | D-DEQUANT, D-LAYER, D-GREEDY-HF |

No phase gates on any HF comparison. The cpp runtime is validated by the **kernel oracle (CUDA↔cpp)**
plus the **snapshot**, not by any per-op comparison to the Python reference.

## 5. Canonical fixtures

- The recorded quantized greedy snapshot at its recorded length is the e2e authority; the engine must
  reproduce it exactly.
- HF bf16 is loaded only to produce the diagnostics in §3.
- Fixtures: the existing parity/bench prompt id files; the snapshot file used by `greedy_match`.

## 6. Tolerances

- **G-STRUCT / G-VALUE / G-DUMP** (metadata, codes/scales): **exact**.
- **G-KERNEL**: bf16-appropriate — account for bf16 inputs and the kernel's accumulation dtype; reuse
  the calibrated v1 kernel-test tolerance. **Do not tighten to fp32/fp64 equality** (that fails correct
  bf16 kernels).
- **G-SNAPSHOT**: **exact** token match (greedy is deterministic).
- **Diagnostics**: metrics only; **no pass/fail thresholds**.

## 7. Explicitly NOT gates

- **Performance** (DRAM throughput, tok/s) — tracked separately in the perf rounds.
- **Any HF-bf16 closeness** (D-DEQUANT / D-LAYER / D-GREEDY-HF).
- **Byte-identity to v1** — only value-identity is required (G-VALUE).
- **Per-op cpp-vs-Python comparison** — removed. cpp op correctness is **G-KERNEL** (CUDA↔cpp fp64);
  cpp end-to-end correctness is **G-SNAPSHOT**. The Python ref does not gate cpp.

## 8. Failure handling (localize; never loosen a gate or tighten a diagnostic)

- **G-STRUCT** → producer/parser offset/identity bug; fix the writer or parser.
- **G-VALUE / G-DUMP** → the relayout corrupted a value, or the loader mis-addresses a segment; fix
  packing/loader. (A wrong segment row-range or q_proj/qkv split preserves values but mis-assigns them —
  it shows up here or as a broadly-low D-DEQUANT, not as quant noise.)
- **G-KERNEL** → codec addressing or GEMV/GEMM math; bisect by shape/qtype. First confirm the tolerance
  is bf16-appropriate before suspecting the kernel.
- **G-SNAPSHOT** → integration/dispatch/schedule bug; bisect — G-KERNEL isolates the linear path, and the
  non-linear ops are unchanged from the known-good runtime. Use D-LAYER to localize; it never gates.
