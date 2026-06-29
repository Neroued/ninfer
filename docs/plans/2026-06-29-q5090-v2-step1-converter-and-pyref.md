# q5090 v2 — Step 1: Converter + Python Reference (no cpp)

> Phase 1 of [docs/q5090_v2_roadmap.md](../q5090_v2_roadmap.md). Spec:
> [q5090_packed_file_format_v2.md](../q5090_packed_file_format_v2.md);
> assignment: [qwen3_6_27b_q5090_v2_tensor_plan.md](../qwen3_6_27b_q5090_v2_tensor_plan.md);
> gates: [q5090_v2_verification_contract.md](../q5090_v2_verification_contract.md).

## Goal

Produce a correct **v2 `.qus` weight file** from a rewritten converter, and a **Python reference engine**
that loads v2 and infers correctly (parity vs HF), with **all Python-side v1 code removed**. This is the
correctness firewall before any cpp work.

## Non-goals / hard constraints

- **No cpp changes.** No edits under `src/`, `include/`, `tests/` (cpp), or `bench/` (cpp).
- **Keep the v1 weight file for cpp.** `out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus` must remain
  byte-unchanged at the end of this step (the unported cpp runtime still loads it). The converter no
  longer *can* emit v1 (v1 code removed), but the existing v1 artifact is preserved.
- **Generate the v2 weight file** `out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus` (+ `manifest.json` is
  written next to it; do not clobber the v1 manifest — write `manifest.v2.json` or regenerate manifest
  for the v2 file path).
- **No backward compatibility.** No v1 layout/format/plan code path remains in `tools/` Python.
- **No perf work, no MTP/Vision fusion** (MTP/Vision stay standalone per tensor-plan §5–6).
- **No requantization.** `quantize.py` is unchanged; v2 is a value-preserving relayout.

## Execution mode

Subagent-driven, **sequential** (tasks share `format.py`/`layouts.py`/`qtypes.py`, so ordering matters).
One implementer subagent per task; after each code task dispatch a spec-compliance reviewer then a
code-quality reviewer; fix Critical/Important before proceeding. The final numerical/format review runs
after Task 5.

## Scope and ownership

Python only. Files by area:

- **Converter** (`tools/q5090_convert/`): `qtypes.py`, `format.py`, `layouts.py`, `packing.py`,
  `tensor_plan.py`, `convert.py`, `verify.py`, `tests/test_packing.py`, `tests/test_tensor_plan.py`,
  `README.md`. `quantize.py` is **unchanged**.
- **Python ref** (`tools/parity/`): `ref_model.py`, `block_parity.py`, `greedy_match.py`,
  `compare_dumps.py`. `hf_reference.py` is the oracle (unchanged unless tap plumbing needs it).
- **Artifacts** (`out/`): write `…mixed_v2.qus`; never touch `…mixed_v1.qus`.

**Coordination points** (single-writer, edit in task order):
- `format.py`, `layouts.py`, `qtypes.py` are shared by the converter and the ref (`ref_model.py` imports
  them). They are owned by Tasks 1/3; Task 4 only consumes them.
- `tools/parity/` consumes the converter modules; do not edit converter modules from parity tasks.

## Reference: HF model and paths

- HF bf16 source: `/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16`
- v2 output: `out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus`
- Preserved v1 artifact: `out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus` (do not modify/delete)

---

## Task 1 — Converter format & ROW_SPLIT layout core

**Files:** `qtypes.py`, `format.py`, `layouts.py`, `packing.py`, `tests/test_packing.py`.

**Reading list:** binary spec §2–§9; current `qtypes.py`, `format.py`, `layouts.py`, `packing.py`.

**Requirements:**
- `qtypes.py`: layout enum is exactly `{ROW_SPLIT=0, CONTIGUOUS=1}`. Delete `TILE_N64_K64`,
  `TILE_N64_K128`, `ROW_GROUPED_G64`.
- `format.py`: v2 header (magic `Q5090MIXEDV2`, version 2, the new offset fields), `ModuleRecord`,
  `TensorEntry` (block, with `segment_count`/`segment_begin`/`fusion_group_id`/`fusion_index`/
  `code_plane_bytes`/`scale_plane_bytes`), new `SegmentRecord` (32 B) and `FusionGroupRecord` (64 B),
  pack/unpack for all, string table builder (block + segment names).
- `layouts.py`: `encode_tensor`/`decode_tensor` for `ROW_SPLIT` (code plane + scale plane, §9.2) and
  `CONTIGUOUS`; reuse the §9.1 LSB-first bit-packing from `packing.py` unchanged.
- `packing.py`: keep the value-identical Q4/Q5/Q6 packing and W8 int8; expose plane assembly.

**DoD / verification:**
- `pytest tools/q5090_convert/tests/test_packing.py` passes: ROW_SPLIT `encode → decode` round-trips the
  quantized `(codes, scales)` bit-exactly for Q4/Q5/Q6/W8; `code_plane_bytes`/`scale_plane_bytes` equal
  the §9.2 formulas; row code-runs are 16-byte aligned for in-scope shapes.
- No v1 layout enum or tile/row-grouped code remains (`rg -n "TILE_N64|ROW_GROUPED" tools/` is empty).

## Task 2 — Tensor plan & expected-manifest generator

**Files:** `tensor_plan.py`, `tests/test_tensor_plan.py`.

**Reading list:** tensor-plan doc (full); current `tensor_plan.py`; binary spec §4–§6, §10.

**Requirements:**
- Emit **blocks** with **segments** and **fusion groups** per the tensor-plan templates (full / GDN /
  globals): `ATTN_IN` (q4{q,k}, q5{gate,v}), `GDN_IN` (q4{in_q,in_k}, q5{in_v}); `in_z` standalone;
  `MLP_GATEUP` (q4{gate,up}); o_proj/out_proj/down/lm_head/embed standalone; control `CONTIGUOUS`.
- Carry over the value-defining transforms unchanged (q_proj de-interleave, qkv row split, conv1d
  runtime-native).
- Provide a deterministic **expected-manifest generator** (block/segment/fusion records the file must
  match) used by `verify.py` L0 plan-conformance and by the ref/cpp dump comparison.
- Canonical emission order = tensor-plan §4.5 (fusion members consecutive).

**DoD / verification:**
- `pytest tools/q5090_convert/tests/test_tensor_plan.py` passes: TEXT_CORE counts are exactly
  **819 blocks / 963 segments / 128 fusion groups**; per-template block shapes/qtypes/segment
  row-ranges/fusion ids match the doc; fusion members are consecutive with correct `fusion_index`;
  `source_kind == OTHER` for every multi-segment block.

## Task 3 — Converter orchestration, self-verify, dump, v2 file

**Files:** `convert.py`, `verify.py`, `format.py` (manifest), `README.md`.

**Reading list:** binary spec; verification contract L0/L1 and §5 (dump); current `convert.py`,
`verify.py`.

**Requirements:**
- `convert.py`: walk the plan, quantize (unchanged `quantize.py`), encode `ROW_SPLIT` blocks with fused
  members emitted consecutively, write header + 4 tables + string table + CRC; v2 `manifest`.
- `verify.py`: run **L0** (structural + plan-conformance vs Task 2 generator) and **L1**
  (value-preservation: recovered `(scale16, q_i)` bit-identical to the quantizer) on the emitted file;
  emit the **dump** (per-block metadata + sampled dequant probes) per contract §5.
- Output goes to `…mixed_v2.qus`; the v1 artifact is never opened for write.

**DoD / verification:**
```bash
python -m tools.q5090_convert.convert \
  --model /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --out out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus
python -m tools.q5090_convert.verify out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus   # L0 + L1 pass; dump written
test -f out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus                                 # v1 artifact still present
```
Expected: v2 file + manifest produced; L0+L1 pass; v1 file present and byte-unchanged (compare mtime/size
or a pre-recorded sha256).

## Task 4 — Python reference engine on v2

**Files:** `ref_model.py`, `compare_dumps.py` (and parity wiring).

**Reading list:** current `ref_model.py`; tensor-plan doc; verification contract L0/§5.

**Requirements:**
- Parse v2 via the shared `format.py`; build the `LayerWeights` view table from **segments**
  (`row_begin..row_begin+row_count` of each block), not from per-projection tensors.
- Dequant via `layouts.decode_tensor` (ROW_SPLIT); run the existing forward/greedy schedule on v2.
- Remove all Python-side v1 consumption paths.
- Emit a ref **dump** identical in schema to the converter dump.

**DoD / verification:**
```bash
python -m tools.parity.ref_model --weights out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus --dump out/ref_dump.v2.json
python -m tools.parity.compare_dumps out/conv_dump.v2.json out/ref_dump.v2.json   # identical
```
Expected: ref loads v2 and runs a greedy step without error; converter and ref dumps are identical
(offsets, sizes, CRCs, sampled dequant) — the cheap structural cross-check.

## Task 5 — Numerical parity gates (vs HF)

**Files:** `block_parity.py`, `greedy_match.py`, `hf_reference.py` (oracle), fixtures.

**Reading list:** verification contract L2/L3/L4; current `block_parity.py`, `greedy_match.py`,
`hf_reference.py`.

**Requirements:**
- **L2:** per-weight `dequant(v2)` vs HF bf16 — per-output-row cosine ≥ 0.98; report max/RMS/cosine per
  qtype.
- **L3-HF:** per-layer hidden-state cosine ≥ 0.999 through the stack on the canonical fixtures.
- **L4:** greedy next-token sequence matches the reference snapshot for K = 128 tokens; first divergence
  vs HF is ≥ the v1 baseline's divergence index (no quantized-quality regression — expected exact,
  since v2 values equal v1's).

**DoD / verification:**
```bash
python -m tools.parity.block_parity --weights out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
  --hf /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16
python -m tools.parity.greedy_match --weights out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
  --hf /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --fixture bench/fixtures/prompts/cn_short.ids --tokens 128
```
Expected: L2/L3/L4 thresholds met; greedy matches the snapshot.

## Task 6 — Remove residual Python v1 code (audit)

**Files:** audit `tools/q5090_convert/*`, `tools/parity/*`; delete v1-only tests/helpers.

**Requirements:** no v1 layout enums, v1 magic (`Q5090MIXEDV1`), `q5090_…_v1` format id, or v1-only
helpers remain in Python. The v1 *weight file* and v1 *docs* are untouched (docs archived later, Phase
3).

**DoD / verification:**
```bash
rg -n "TILE_N64|ROW_GROUPED|Q5090MIXEDV1|mixed_v1" tools/   # only the preserved out/ path reference, if any, is allowed; no code paths
pytest tools/q5090_convert/tests/
```

## Definition of done (Step 1)

- v2 file emitted; converter `verify.py` L0+L1 green; converter unit tests green.
- Python ref runs on v2; converter/ref dumps identical; L2+L3-HF+L4 green.
- No Python-side v1 code path remains; `out/…mixed_v1.qus` present and byte-unchanged.
- No cpp file modified.

## Review phase

Risk: numerical correctness + binary ABI. After Task 5, dispatch:

1. **Numerical-correctness reviewer** — L1 value-preservation, L2/L3/L4 thresholds, and that the
   q_proj de-interleave / qkv split / conv1d transforms are preserved (a value swap here passes L1 but
   fails L2; confirm L2 catches it).
2. **Format/ABI reviewer** — header/tables, segment partition, fusion adjacency, plan-conformance, and
   the `source_kind` rule for fused blocks.
3. **Scope reviewer** — no cpp touched; v1 weight file preserved byte-for-byte; Python v1 code removed.

All three must pass before Phase 1 is declared done and Phase 2 (cpp) planning begins.
