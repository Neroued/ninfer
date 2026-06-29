# Python q5090 Tools Restructure Design

Date: 2026-06-29

## Goal

Restructure the Python side of the project around the three project-owned q5090 workflows:

1. Convert original Qwen3.6-27B bf16 safetensors into the q5090 v2 packed weight format.
2. Load q5090 v2 from Python and run a correctness-first reference inference path.
3. Compare q5090 against original bf16/HF sources for diagnostics, including weight metrics,
   HF greedy reports, structural dumps, and activation tensor dumps for debugging.

The cleanup should make responsibilities explicit, remove duplicated correctness-critical code, and
replace confusing command surfaces directly. The project does not preserve backward compatibility, so
old Python module paths and CLIs can be deleted instead of wrapped.

## Non-Goals

- Do not change the q5090 v2 binary ABI or quantization policy.
- Do not change C++ runtime behavior in this restructuring step.
- Do not add support for other models or generic runtimes.
- Do not make HF bf16 comparisons into correctness gates. They remain diagnostics only.
- Do not add low-value tests that only protect source layout or old command compatibility.

## Current Problems

The current Python code is functional but poorly separated:

- `tools/q5090_convert/convert.py` mixes plan adaptation, HF reading, config checks, source
  materialization, file writing, manifest writing, progress reporting, and CLI parsing.
- `tools/q5090_convert/verify.py` imports converter internals such as `ShardReader`,
  `build_conversion_plan`, and `materialize_block`, so verification is not a clean consumer of shared
  library code.
- `tools/q5090_convert/verify.py` and `tools/parity/ref_model.py` each parse the q5090 file format and
  each build structural dumps.
- HF shard reading and source tensor transforms are duplicated between converter and parity code.
- `tools/parity/ref_model.py` contains q5090 parsing, weight views, PyTorch ops, model schedule,
  state management, structural dump output, and CLI behavior in one large file.
- Activation tensor dump support exists only as an internal dictionary path in `RefModel.forward`; it
  is not a stable CLI or file artifact.
- `tools/parity/block_parity.py` still has gate-like threshold behavior for HF/bf16 weight comparison,
  which conflicts with the q5090 v2 verification contract. HF comparisons must report diagnostics,
  not fail correct quantized outputs.
- Command names do not consistently describe behavior. For example, `ref_model --dump` writes a
  structural dump, not activation tensors.

## Chosen Approach

Create one focused Python package under `tools/q5090/` and move the q5090 Python surface into that
package. Delete the old Python command surfaces instead of keeping compatibility wrappers.

Target package shape:

```text
tools/q5090/
  __init__.py

  format/
    __init__.py
    abi.py
    quantize.py
    records.py
    packing.py
    layout.py
    reader.py
    dump.py

  source/
    __init__.py
    hf.py
    tensor_plan.py

  convert/
    __init__.py
    writer.py
    verify.py
    cli.py

  pyref/
    __init__.py
    ops.py
    model.py
    tensor_dump.py
    cli.py

  diagnostics/
    __init__.py
    bf16_weights.py
    hf_greedy.py
    snapshot.py
    compare_dumps.py

  tests/
```

The old `tools/q5090_convert/*.py` and `tools/parity/*.py` Python files are removed or moved into the
new package. The C++ files currently under `tools/parity/` are not part of this Python cleanup and may
remain in place until a C++ parity reorganization is needed.

## Module Responsibilities

### `tools/q5090/format`

This layer owns q5090 v2 ABI mechanics only. It must not depend on HF, tensor-plan generation, or the
Python model reference.

- `abi.py`: qtype, layout, module, load-policy, fusion, source-kind constants and per-qtype numeric
  parameters.
- `quantize.py`: exact per-row/per-group q5090 quantizer and dequant helper. This is the G-VALUE
  oracle used by converter verification and layout tests.
- `records.py`: q5090 v2 header, module, tensor/block, segment, and fusion record pack/unpack, plus
  hashes, CRC, alignment constants, and string table helpers.
- `packing.py`: low-bit code packing/unpacking and ROW_SPLIT plane size/assembly/splitting.
- `layout.py`: ROW_SPLIT and CONTIGUOUS encode/decode. Encoding may call quantization for converter
  use; decoding is used by verifier, pyref, and diagnostics.
- `reader.py`: the single q5090 parser and mmap-backed reader. It exposes modules, blocks, segments,
  fusion records, raw payloads, logical segment views, ROW_SPLIT row slicing, and row-chunk iterators.
- `dump.py`: canonical structural dump generation from a parsed q5090 file. Converter verification and
  pyref must both call this module so dump schemas cannot drift.

### `tools/q5090/source`

This layer owns original HF bf16 source access and Qwen3.6-specific tensor assignment.

- `hf.py`: safetensors index loading, lazy shard reading, config loading and config lock validation,
  source tensor row reads, full tensor materialization, and source transforms.
- `tensor_plan.py`: the authoritative Qwen3.6-27B q5090 block/segment/fusion plan, migrated from the
  current v2 tensor plan implementation.

The value-defining transforms must exist in one place only:

- full-attention `q_proj` query/gate de-interleave;
- GDN `in_proj_qkv` q/k/v row slicing;
- GDN `conv1d` runtime-native transform.

Converter, verifier, and bf16 diagnostics all consume the same source materialization API.

### `tools/q5090/convert`

This layer produces and verifies q5090 files.

- `writer.py`: build conversion plan, materialize source blocks, encode payloads, write tables,
  payloads, header, and `manifest.v2.json`.
- `verify.py`: implement G-STRUCT and G-VALUE. It reads q5090 through `format.reader`, compares plan
  identity against `source.tensor_plan`, and checks recovered `(scale16, q_i)` bit-identically against
  the quantizer output from the same source tensors.
- `cli.py`: command-line conversion entry point.

`verify.py` must not import converter CLI internals. Shared concepts such as plan building and source
materialization belong in `source/`.

### `tools/q5090/pyref`

This layer owns Python q5090 inference.

- `ops.py`: correctness-first PyTorch helper ops such as linear, RMSNorm, RoPE, GQA attention,
  GDN recurrence, GDN gating, causal conv1d, residual add, SiLU/mul, and sigmoid gate/mul.
- `model.py`: `RefModel`, model state, q5090 weight lookup through segment views, prefill/decode
  schedule, and greedy token generation.
- `tensor_dump.py`: activation tensor dump writer.
- `cli.py`: q5090 inference command.

The pyref model consumes q5090 weights only through `format.reader` segment views. It does not parse
binary records itself and does not know about HF source files.

### `tools/q5090/diagnostics`

This layer owns report-only comparisons and exact external gates.

- `bf16_weights.py`: D-DEQUANT report. Compare q5090 dequantized segment views against original bf16
  tensors, reporting max abs, RMS, row-cos mean/min, and warning counts. This command exits nonzero
  only for invalid inputs or runtime errors, not for normal quantization drift.
- `hf_greedy.py`: slow HF full-model greedy helper used for report-only first-divergence diagnostics.
- `snapshot.py`: G-SNAPSHOT exact gate. Run q5090 pyref greedy and compare exactly against the recorded
  quantized snapshot for the snapshot's own length.
- `compare_dumps.py`: G-DUMP exact structural dump comparison.

## Command Surface

The new commands are module entry points. Old Python commands are removed.

Convert:

```bash
python -m tools.q5090.convert.cli \
  --model /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --out out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus
```

Verify q5090 structure and values:

```bash
python -m tools.q5090.convert.verify \
  out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
  --model /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --dump out/conv_dump.v2.json
```

Run q5090 pyref inference and write a structural dump:

```bash
python -m tools.q5090.pyref.cli \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
  --prompt "1" \
  --decode 1 \
  --structural-dump out/ref_dump.v2.json
```

Run q5090 pyref inference and write activation tensors:

```bash
python -m tools.q5090.pyref.cli \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
  --ids bench/fixtures/prompts/cn_short.ids \
  --decode 26 \
  --tensor-dump out/tensor_dumps/q5090_cn_short
```

Run bf16 weight diagnostics:

```bash
python -m tools.q5090.diagnostics.bf16_weights \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
  --hf /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16
```

Compare structural dumps:

```bash
python -m tools.q5090.diagnostics.compare_dumps \
  out/conv_dump.v2.json \
  out/ref_dump.v2.json
```

Run snapshot exactness gate:

```bash
python -m tools.q5090.diagnostics.snapshot \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
  --fixture bench/fixtures/prompts/cn_short.ids \
  --snapshot-report profiles/e2e/m3-output-gate.json \
  --case cn_short
```

## Dump Contracts

### Structural Dump

Structural dumps keep the current top-level shape:

```text
format
file
header
modules
blocks
fusion_groups
```

Every producer calls `tools.q5090.format.dump`. The dump includes block metadata, segment metadata,
fusion metadata, payload offsets/sizes, CRCs, plane sizes, and deterministic sampled quant/dequant
probes.

### Activation Tensor Dump

Activation tensor dumps use a directory format:

```text
out/tensor_dumps/q5090_cn_short/
  manifest.json
  embed.pt
  layer_00.pt
  layer_01.pt
  ...
  layer_63.pt
  final_norm.pt
  logits_last.pt
```

Default dump level is `layer`, which records embeddings, every layer boundary, final norm, and last
logits. A heavier `op` level may be added in the same implementation if needed for immediate debug
work; when enabled, it writes per-layer subdirectories for intermediate tensors such as attention
q/k/v, GDN q/k/v, GDN conv output, MLP gate/up, and MLP output.

`manifest.json` records:

- q5090 file path and SHA256 if available without an expensive rescan;
- prompt ids source and prompt length;
- requested decode count and actual generated token ids;
- dump level;
- tensor names, filenames, shapes, dtypes, phases, and token-position ranges;
- tool command metadata.

Tensor payloads are written as `.pt` files. This preserves dtype and shape without inventing a custom
array schema and allows direct PyTorch inspection.

## Correctness Semantics

Hard gates:

- **G-STRUCT**: q5090 ABI and plan conformance.
- **G-VALUE**: recovered q5090 scales/codes exactly match the quantizer output for the same source.
- **G-DUMP**: structural dumps from converter, pyref, and later C++ readers match exactly.
- **G-SNAPSHOT**: q5090 greedy output exactly matches the recorded quantized snapshot for the
  snapshot's own length.

Diagnostics:

- **D-DEQUANT**: q5090 dequantized weight metrics against HF bf16.
- **D-LAYER**: future activation/layer metrics against HF if implemented.
- **D-GREEDY-HF**: first divergence versus slow HF greedy.

No HF/bf16 comparison is a pass/fail correctness gate. A diagnostic command may print `WARN` for
gross outliers, but normal quantization drift does not produce a failing exit code.

## Data Flow

Conversion:

```text
HF safetensors
  -> source.hf reader
  -> source.tensor_plan block/segment/fusion plan
  -> source materialize logical tensors
  -> format.layout encode_tensor
  -> convert.writer writes records + payloads + manifest
  -> convert.verify checks G-STRUCT/G-VALUE
```

Python q5090 inference:

```text
q5090 file
  -> format.reader mmap parse
  -> segment views
  -> pyref.model resolves weights by canonical segment name
  -> PyTorch schedule
  -> generated token ids
  -> optional structural dump and activation tensor dump
```

BF16 diagnostics:

```text
q5090 segment view + HF source tensor
  -> shared source transforms
  -> compare dequantized q5090 against bf16 source
  -> report metrics only
```

## Testing and Verification

Allowed tests should protect real contracts:

- binary record pack/unpack and reserved-byte sizing;
- low-bit packing/unpacking and ROW_SPLIT plane geometry;
- layout encode/decode round trips;
- tensor-plan block/segment/fusion counts, shapes, row ranges, and fusion adjacency;
- structural dump compare behavior;
- CLI/report JSON schema only where downstream tools consume it.

Do not add tests that preserve old command paths, scan source layout, or assert implementation-only
function names.

Small verification set after restructuring:

```bash
pytest -q tools/q5090/tests
python -m tools.q5090.convert.verify out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus --quick
python -m tools.q5090.pyref.cli \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
  --prompt "1" \
  --decode 1 \
  --structural-dump out/ref_dump.v2.json
python -m tools.q5090.diagnostics.compare_dumps out/conv_dump.v2.json out/ref_dump.v2.json
python -m tools.q5090.diagnostics.snapshot \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
  --fixture bench/fixtures/prompts/cn_short.ids \
  --snapshot-report profiles/e2e/m3-output-gate.json \
  --case cn_short
```

Full verification on the real model:

```bash
python -m tools.q5090.convert.verify \
  out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
  --model /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16
python -m tools.q5090.diagnostics.bf16_weights \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v2.qus \
  --hf /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16
```

## Documentation Updates

Update docs and READMEs that currently mention old Python paths or v1-primary command examples:

- root `README.md`;
- `tools/q5090_convert/README.md`, replaced by a new `tools/q5090/README.md`;
- q5090 v2 roadmap and verification docs if command examples move;
- implementation plans that are still active and mention old command paths.

Archived historical docs do not need to be rewritten unless they are actively referenced as current
instructions.

## Review Requirements

This is a high-risk Python refactor because it touches binary ABI parsing, quantized value recovery,
source tensor transforms, and the correctness firewall before C++ runtime work. The implementation
plan should include independent review of:

- q5090 ABI and reader/dump equivalence;
- source tensor transforms and value-preservation verification;
- pyref schedule behavior and snapshot exactness;
- command/documentation consistency and removal of old Python compatibility paths.

## Definition of Done

- Python q5090 code lives under `tools/q5090/` with the responsibilities described above.
- Old `tools/q5090_convert` and `tools/parity` Python command surfaces are removed or fully migrated.
- Converter emits v2, verifier passes G-STRUCT/G-VALUE, and the manifest remains v2.
- Pyref loads q5090 through the shared reader and runs greedy inference.
- Converter and pyref structural dumps are generated by the same schema code and compare exactly.
- Activation tensor dump has a stable CLI and manifest.
- BF16/HF comparison commands are diagnostic-only and do not gate on quantization drift.
- Documentation points users to the new commands.
