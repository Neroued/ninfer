# q5090 v2 Verification Contract

This document defines what "correct" means for the v2 weight format and pins the acceptance gates
shared by the three landing phases — **converter**, **Python reference**, **cpp runtime**. It is the
single source of pass/fail criteria so each phase is gated on evidence, not assertion.

Companion docs: byte contract = [q5090_packed_file_format_v2.md](q5090_packed_file_format_v2.md);
per-tensor assignment = [qwen3_6_27b_q5090_v2_tensor_plan.md](qwen3_6_27b_q5090_v2_tensor_plan.md).

## 0. Principle

v2 is a **value-preserving relayout** of the quantization policy output. Therefore verification splits
into two distinct kinds of comparison, which must never be conflated:

- **Exactness checks** — against the quantizer output and across implementations. v2 must change *bytes*
  only, never *values*; and Python and cpp readers of the same file must agree. These are bit-exact or
  tight-float and are **hard gates**.
- **Quality checks** — against the HF bf16 reference. The error here is the *quantization* error (it
  exists in v1 too); it is bounded, reported, and gated only at the end-to-end level. A quality check
  is **not** allowed to mask a relayout bug, and an exactness check is **not** allowed to be relaxed to
  hide quantization error.

## 1. Oracles and references

| oracle | what it is | used for |
|---|---|---|
| **Quantizer** | `tools/q5090_convert/quantize.py` output: `(scale16, q_i)` per group | exactness of the relayout (L1) |
| **HF bf16** | `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16` via `tools/parity/hf_reference.py` | quality of the quantized model (L2, L3-HF, L4) |
| **Python ref** | `tools/parity/ref_model.py` running on v2 weights | cross-impl oracle for the cpp runtime (L3-impl) |
| **Reference greedy snapshot** | greedy token sequences the established (v1) quantized model produced on the canonical fixtures | end-to-end regression bar (L4) |

`quantize.py` is unchanged by v2; the same `(scale16, q_i)` values feed both v1 and v2 packing. If a v1
artifact or dump is still available during the transition, `dequant(v2_tensor) == dequant(v1_tensor)`
bit-exact is an additional strong check, but it is **not required** (L1 against the quantizer does not
depend on v1 existing).

## 2. Verification levels

### L0 — Structural / ABI integrity (file is a valid v2 file)

Checked by the converter self-check and again by the cpp weight-store parse. All must hold:

- **Plan conformance (the file equals the authoritative tensor plan).** The tensor-plan templates are
  rendered to an expected per-block/segment/fusion manifest (deterministically, for the modules
  present). Every block and segment in the file matches it **item-by-item**: block `name`/`name_hash`,
  `source_kind`, `source_layer`, `qtype`, `layout`, `[N,K]`; each segment's `(source_kind,
  source_layer, row_begin, row_count, name)`; each block's `fusion_group_id`/`fusion_index`; and each
  `FusionGroupRecord`'s `group_id`/`source_layer`/member set/`total_n`. An assignment error (wrong
  segment identity, swapped segment order, mislabeled fusion index) thus fails L0 **structurally**, not
  only as a numerical symptom in L2/L3. The expected manifest is generated from
  `qwen3_6_27b_q5090_v2_tensor_plan.md` (the converter and the cpp reader share the same generator).
- `magic == "Q5090MIXEDV2"`, `version == 2`, `header_size == 4096`, `endian == 0x01020304`.
- `module_index`/`tensor_index`/`segment_index`/`fusion_group_index`/`string_table`/`payload` offsets
  are in order, non-overlapping, within file size; `payload_offset` is 4096-aligned.
- Counts are validated **per module**, not just at the header. Header
  `tensor_count`/`segment_count`/`fusion_group_count` equal the table sizes and the **sum over the
  modules present**. Via each `ModuleRecord` index range, the TEXT_CORE module has exactly **819 blocks
  / 963 segments / 128 fusion groups**; MTP and VISION module counts are validated against their plan
  templates when those modules are present (deferred; tensor plan §5–6). A TEXT-only file therefore has
  header counts 819 / 963 / 128; a multi-module file has larger header totals.
- Each block: `payload_offset` 256-aligned; `payload_bytes == align_up(code_plane_bytes,256) +
  scale_plane_bytes`; `code_plane_bytes` and `scale_plane_bytes` equal the values computed from
  `(N, K_pad, qtype)` (binary spec §9.2); payloads do not overlap.
- Each block: `crc32` matches a recompute over its payload.
- Segments of a block partition `[0, N)` exactly, in increasing `row_begin`, with `sum(row_count) == N`.
- `source_kind` rule: `segment_count == 1` ⇒ block `source_kind` mirrors the segment; `segment_count >
  1` ⇒ block `source_kind == OTHER (0)`.
- Fusion groups: members are consecutive tensor indices and consecutive payloads; every member has
  matching `fusion_group_id`/`source_layer`/`K` and correct `fusion_index`; `total_n == Σ member N`.
- Every quantized block is `ROW_SPLIT`; every control block is `CONTIGUOUS`; no other `layout` value
  appears.

### L1 — Value preservation (relayout changed no value)

For every quantized block, recover `(scale16, q_i)` from the file (parse planes, unpack codes per §9.1)
and compare to the quantizer output for the same logical projection rows:

- recovered `q_i` integers: **bit-identical**.
- recovered `scale16` half-floats: **bit-identical**.

This is the core "v2 == v1 in value" guarantee. Hard gate for the converter and the Python loader.

### L2 — Per-tensor dequant sanity vs HF (gross-corruption catch)

For each weight, `dequant(v2)` vs the HF bf16 weight:

- per-output-row cosine ≥ **0.98** (catches wrong layout / scrambled rows / transpose bugs).
- report max-abs-err, RMS-err, mean per-row cosine per qtype (informational; bounded by quantization).

L2 is a **sanity** gate, not a quality bar; the magnitude of the dequant error is a property of the
quantizer, not of v2.

### L3 — Per-op / per-layer parity

Two comparisons, both on the canonical fixtures' activations:

- **L3-impl (cpp vs Python ref, same weights):** for each op tap (rmsnorm, each GEMV/projection,
  attention, GDN recurrence, MLP), cosine ≥ **0.9999** and max relative error ≤ **1e-2** (float-order
  differences only; the math and weights are identical).
- **L3-HF (quantized stack vs HF):** per-layer hidden-state cosine ≥ **0.999** through the stack
  (quant-bounded; localizes any layer where quantization or layout disproportionately degrades
  signal). Compared per layer to catch drift, not just at the output.

### L4 — End-to-end inference parity

On the canonical prompts:

- **Greedy regression (hard):** greedy next-token sequence matches the reference greedy snapshot for
  the first **K = 128** generated tokens. Because v2 is value-identical to the prior quantized model,
  the v2 Python ref must reproduce the snapshot exactly; the cpp runtime must match the Python ref
  exactly.
- **HF agreement (reported, calibrated):** first greedy divergence index vs HF bf16 must be **≥** the
  v1 baseline's divergence index (v2 must not regress quantized quality); report top-1 agreement rate
  and mean logit cosine over the window.

## 3. Acceptance gates by phase

| phase | required levels | meaning |
|---|---|---|
| **B. Converter** (`tools/q5090_convert`) | L0 + L1 (+ unit tests `test_tensor_plan`, `test_packing`) | the emitted file is a valid v2 file whose values equal the quantizer |
| **C. Python ref** (`tools/parity`) | L0 + L1 (loader) + L2 + L3-HF + L4-greedy + L4-HF | the v2 format **infers correctly**, independent of cpp — the correctness firewall |
| **D. cpp runtime** | L0 (parse) + L3-impl (cpp vs Python) + L4-greedy (cpp vs snapshot) | cpp reproduces the proven-correct reference; then perf gates apply (separate) |

No phase may proceed past its gate on partial evidence. Phase D must not begin tuned-kernel work
(binary spec §11 perf) until D's L3-impl/L4-greedy pass on a correctness backend.

## 4. Canonical fixtures

- **Prompts:** the existing parity/bench fixtures (`bench/fixtures/prompts/*.ids`, e.g. `cn_short`) and
  the long-decode FEM prompt used in the nsys reports. The fixture manifest pins token ids so all
  phases decode the same sequences.
- **HF reference:** `base-hf-bf16` loaded by `tools/parity/hf_reference.py`; per-tensor weights, per-op
  taps, and greedy tokens are dumped once and reused.
- **Reference greedy snapshot:** stored token sequences (first ≥128 tokens per prompt) used as the L4
  regression bar.
- **Tolerances** are calibrated once against the v1 baseline on these fixtures; the values in §2 are
  the contract, tightened (not loosened) if the baseline is comfortably inside them.

## 5. Cross-phase structural parity (dump mode)

The converter MUST provide a deterministic **dump** (JSON or text): per block — name, `source_kind`,
`source_layer`, `qtype`, `layout`, `[N,K]`, `payload_offset`, `code_plane_bytes`, `scale_plane_bytes`,
`crc32`, `fusion_group_id`, `fusion_index`, segment list; per fusion group — members and `total_n`; and
a small sample of dequantized values at fixed `(block,row,col)` probes.

- The Python loader and the cpp weight-store each emit the same dump; `tools/parity/compare_dumps.py`
  diffs them. Identical dumps prove all three agree on offsets, sizes, CRCs, and sampled values before
  any heavy numerical run — the cheapest place to catch a parse/layout mismatch.

## 6. What is explicitly NOT a correctness gate

- **Performance** (DRAM throughput, tok/s, ncu/nsys roofline) is tracked separately in the perf rounds;
  it is never a substitute for, nor gated together with, the correctness levels above.
- **Byte-identity to v1** is not required (v2 reorders bytes); only value-identity (L1) is.
- **Per-tensor dequant error magnitude vs HF** (L2 metrics beyond the cosine floor) is reported, not
  gated — it is quantizer-defined.

## 7. Failure handling

On any gate failure, localize before changing thresholds:

- L0 fail → a structural/offset bug in the converter writer or the reader parser; fix the producer or
  parser, never relax L0.
- L1 fail → the relayout corrupted a value (wrong row order, wrong plane split, wrong unpack); fix
  packing/layout, never relax L1.
- L2 cosine fail with L1 pass → a *logical* layout error (e.g. wrong segment row-range, q_proj
  de-interleave, qkv split) that preserves values but assigns them to the wrong projection; fix the
  assignment in the tensor plan.
- L3/L4 fail with L0–L2 pass → an execution/consumption mismatch (kernel, dispatch, or schedule);
  bisect by op tap (L3) to the first divergent layer.

Thresholds are tightened on calibration, never loosened to pass a failing run.
