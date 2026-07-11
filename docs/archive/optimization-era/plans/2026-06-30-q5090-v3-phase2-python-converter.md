# q5090 v3 migration — Phase 2: Python converter + reference + v3 weights (single atomic task)

> **Status:** Phase 2 of the v3 plane-split migration
> ([2026-06-30-q5090-v3-layout-migration-roadmap.md](2026-06-30-q5090-v3-layout-migration-roadmap.md)).
> Phase 1 (the spec, [../q5090_packed_file_format_v3.md](../q5090_packed_file_format_v3.md)) is done.
> Phase 3 (C++ parser + kernels) is **not** in scope here.

This plan defines **task scope** and the **final verification strategy** only. It is intentionally not
a step-by-step subtask breakdown.

## Goal

Move the Python converter/reference stack from v2 (concatenated-bitstream Q5/Q6) to **v3 plane-split**
(low-nibble plane + separate high-bit plane), regenerate the weight artifact as v3, and prove — in the
Python path — that v3 preserves the quantizer policy values from the original source weights and that
the reference model runs normally on the full artifact. v3 byte contract: the Phase-1 spec
(§2/§4/§7/§9/§13/§17).

## Execution mode — single atomic task, NOT subagent-driven

Per AGENTS.md, a non-subagent mode requires a stated reason:

- This is a **format migration/refactor** across tightly-coupled modules: `packing` (bit layout) ↔
  `layouts` (plane assembly) ↔ `format` (byte structs) ↔ `convert` (driver) ↔ `verify` (structural +
  dequant) ↔ `ref_model` (reader). Each consumes the other's byte contract; an intermediate state where
  only some are on v3 **does not import, run, or verify**. There is no meaningful verifiable subtask
  boundary, so the change must not be split.
- Therefore: **one developer executes the entire Python change in one task**, with **no intermediate
  commits, builds, or reviews**; it is verified against the strategy below and then reviewed once in a
  final read-only subagent review stage.
- No git worktree / no parallelism is needed (single sequential task on `master`; the implementer may
  iterate freely in the working tree and squash to one commit when green).

## Non-goals / hard constraints

- **No C++ changes.** `src/**`, `include/**`, the kernels, the C++ parser (`weight_store*`,
  `q5090_pack.h`), and C++ tests are Phase 3. This task touches Python + docs only.
- **No value change.** v3 dequant MUST equal the policy output from the original source weights
  bit-for-bit (the gate). A v2 artifact comparison is not required.
  The quantizer (`quantize.py` max-abs) is **not** modified.
- **No tensor-plan reassignment.** Blocks/segments/fusion groups/qtypes/shapes/source transforms are
  unchanged (v3 inherits the v2 tensor plan). Only the on-disk **code packing** changes.
- **No v2 path kept.** The converter emits v3 only; the v2 emitter/reader is replaced (AGENTS.md: no
  dual format). The old `.qus` is regenerated, not converted in place.
- **No new abstractions** for hypothetical future layouts; edit the existing modules directly.

## Scope & ownership (files this task changes)

| file | change |
| --- | --- |
| `tools/q5090_convert/packing.py` | v3 plane-split pack/unpack: a low-nibble plane (= Q4 nibble pack of `code & 0xF`) + a high-bit plane (Q5: 1 bit/code → 8 B/group; Q6: 2 bits/code → 16 B/group), per spec §9.1. numpy reference **and** torch fast path, kept bit-equivalent (cross-checked). Q4 nibble path and W8 int8 base plane unchanged. `row_split_plane_sizes` → 3 planes (nibble/high/scale) with 256-aligned relative offsets and `payload_bytes = scale_rel + scale_plane_bytes`. `assemble_row_split_payload` / `split_row_split_payload` → 3 planes. |
| `tools/q5090_convert/layouts.py` | `encode_row_split` emits nibble+high+scale and returns `(payload, nibble_bytes, high_bytes, scale_bytes, …)`; `decode_row_split_quantized` reconstructs codes via `sign_extend(low \| high<<4)`; W8 stays single base plane (`high_bytes=0`). `EncodeResult` tuple gains the high-plane field. |
| `tools/q5090_convert/format.py` | `MAGIC`→`Q5090MIXEDV3…`, `VERSION`→3, manifest name; `_ENTRY_STRUCT` gains one `Q` (116→124 B body, still 128-padded); `TensorEntry` + pack/unpack: `code_plane_bytes`/`scale_plane_bytes` → `nibble_plane_bytes`/`high_plane_bytes`/`scale_plane_bytes`; header `flags` writer (TEXT/MTP/VISION present + CALIBRATED per spec §2); `module_version`=3. |
| `tools/q5090_convert/qtypes.py` | `MODULE_POLICY` strings `…_v2` → `…_v3`; (qtype/layout/module/source_kind/fusion enums are ABI-stable — **do not renumber**). Optional: a `high_plane_bytes_per_group(qtype)` helper. |
| `tools/q5090_convert/convert.py` | driver: thread `nibble/high/scale` plane bytes into each `TensorEntry`; build the header with v3 `VERSION`/flags; `_write_manifest` emits the v3 schema (spec §17: `binary_spec`, `tensor_plan`, `file_bytes`, hex `sha256_safetensors_index`, `modules`/`absent_modules`, `qtypes`, `alignment`, `tensor_count`/`segment_count`/`fusion_group_count`); output path → `…_v3.qus`. |
| `tools/q5090_convert/verify.py` | structural validation to the spec §13 (3-plane sizes via `row_split_plane_sizes`; magic/version; ROW_SPLIT vs CONTIGUOUS `padded_shape` split; plane-byte equality; `source_kind` defined-for-module; zero-fill as verifier check); keep the **dequant value-preservation** check (decoded `.qus` == source-quantized weights) on v3. |
| `tools/q5090_convert/tensor_plan.py` | no assignment change; bump any embedded format/version string only. |
| `tools/q5090_convert/tests/test_packing.py` | add v3 round-trip (`pack∘unpack == codes`, Q4/Q5/Q6/W8) **and** the value-preservation test (v3 dequant == the policy output, value-for-value, random tensors per qtype). |
| `tools/q5090_convert/tests/test_tensor_plan.py` | bump magic/version asserts if present; TEXT_CORE plan counts remain 819/963/128, while the full v3 artifact counts are 1167/1311/128. |
| `tools/parity/ref_model.py` | reads v3 **transparently** through the updated `format`/`layouts`/`packing`; change only if it pins magic/version. |
| `docs/qwen3_6_27b_q5090_v2_tensor_plan.md` (optional) | add a one-line note that the assignment applies to v3 unchanged, or rename/tag a v3 copy; no assignment edits. |

**Out of scope (do not touch):** everything under `src/**`, `include/**`, `tests/**` (C++),
`bench/**`; the quantizer math in `quantize.py`.

## The task (implementation outline — guidance, not verifiable subtasks)

Done in one pass, bottom-up so the working tree only becomes runnable at the end:
`packing.py` (plane pack/unpack + sizes) → `format.py` (magic/version + entry struct) →
`layouts.py` (3-plane encode/decode) → `qtypes.py` (policy strings) → `convert.py` (plane bytes +
manifest + header) → `verify.py` (v3 structural + dequant) → tests → regenerate the artifact →
`ref_model` sanity. There are no commits or gates between these — only the final verification runs.

## Artifact output

Regenerate the full all-module `out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus` (+ sidecar manifest) from
the source safetensors, including TEXT_CORE, MTP_DRAFT, and VISION_ENCODER. The v2 `.qus` is
superseded (not deleted by this task; Phase 3 makes the runtime read v3).

## Reading list

- Spec: [../q5090_packed_file_format_v3.md](../q5090_packed_file_format_v3.md) §2 (header/flags), §4
  (TensorEntry plane fields), §7 (enums), §9.1/§9.2 (bit-plane + 3-plane), §13 (validation), §17
  (manifest); §19 (v2→v3 delta).
- Tensor plan companion: [../qwen3_6_27b_q5090_v2_tensor_plan.md](../qwen3_6_27b_q5090_v2_tensor_plan.md).
- Current code: `tools/q5090_convert/{packing,layouts,format,qtypes,convert,verify,quantize}.py`,
  `tools/q5090_convert/tests/test_packing.py`, `tools/parity/ref_model.py` (mmap + `unpack_header` +
  `decode_tensor`).

## Final verification strategy (the gate — all must pass, run once)

Python: `/home/neroued/miniconda3/envs/vllm-bench/bin/python` (or repo env).

1. **Packing round-trip.** `pytest tools/q5090_convert/tests/test_packing.py` — for Q4/Q5/Q6/W8,
   `unpack(pack(codes)) == codes`; numpy reference and torch fast path agree bit-for-bit.
2. **Value-preservation gate (the migration's reason to exist).** For random tensors per qtype **and**
   on real model tensors: **v3 dequant == the `quantize.py` policy output from the source weights,
   value-for-value**. A single mismatch fails the task. (Test in `test_packing.py` + the verifier's
   dequant probe.)
3. **Converter regen + structural verify.** Run the converter to emit `…_v3.qus` + manifest, then
   `python -m tools.q5090_convert.verify out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus` passes **all** spec
   §13 structural checks (magic V3/version 3, index adjacency, 3-plane sizes, ROW_SPLIT/CONTIGUOUS
   `padded_shape`, `payload_bytes`, segment partition, fusion consistency) **and** the dequant probe.
4. **Header/manifest consistency.** `unpack_header` reports magic `Q5090MIXEDV3`, version 3, full
   all-module counts (`tensor_count`/`segment_count`/`fusion_group_count` = 1167/1311/128,
   `module_count` = 3) and TEXT/MTP/VISION flags; the manifest fields equal the header (spec §17).
5. **Reference-model inference is normal.** Run `tools/parity/ref_model.py` on the v3 artifact using
   the local Qwen3.6 tokenizer, the Qwen chat template, and a real short prompt; confirm it prints a
   non-empty prompt token count, generated token ids, and decoded generated text. This is the
   "inference normal" check; comparing against a v2 artifact is not required in this phase.
   Command shape:
   `python tools/parity/ref_model.py --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus --model /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 --prompt "请用三条短句说明，为什么每天适量喝水很重要？每条不超过 18 个字。" --decode 16`.
6. **Suite green.** `pytest tools/q5090_convert/` fully green.

> **Out of Phase-2 verification scope:** the C++ runtime cannot load v3 until Phase 3, so there is **no
> C++ build / e2e / nsys check here**. Phase 2 is proven entirely on the Python path (steps 1–6); the
> C++ e2e gate belongs to Phase 3.

## Review (final read-only subagent stage)

Before the final commit, run two independent subagent reviews:

1. **v3 design-doc fit** — compare the implementation against
   `docs/q5090_packed_file_format_v3.md` §2/§4/§7/§9/§13/§17 and this plan.
2. **Code implementation quality** — inspect the Python converter/verifier/reference implementation
   for malformed partial artifacts, manifest/header consistency, weight-loading safety, and avoidable
   complexity.

The combined review checklist:

- **Format/ABI** — emitted bytes match spec §2/§4/§9 (magic/version, `_ENTRY_STRUCT` size 124→128,
  three 256-aligned planes, `payload_bytes` = relative span); enums not renumbered.
- **Numerical** — the value-preservation gate (step 2) is real (covers all qtypes incl. sign/edge
  codes and the fp16-scale underflow case), not a trivially-passing test.
- **Weight-loading safety** — `verify.py` rejects malformed/short payloads and wrong plane sizes (a
  bad offset must fail loud, not silently mis-read).
- **Scope** — no C++ touched; quantizer unchanged; tensor-plan assignment unchanged; no v2 emitter
  left behind.

On green + review, squash to one commit (e.g. `feat(q5090): convert emits v3 plane-split weights`) and
regenerate-commit the artifact/manifest per repo convention (`bench(q5090): …` for the artifact if it
is tracked).
