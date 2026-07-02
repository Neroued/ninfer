# q5090_convert

Offline, GPU-accelerated quantizing converter: original **Qwen3.6-27B** bf16 safetensors
-> a single self-describing packed file in the `q5090_w4g64_mixed_v3` format.

- Binary file contract and MTP_DRAFT v3 assignment:
  [`docs/q5090_packed_file_format_v3.md`](../../docs/q5090_packed_file_format_v3.md)
- TEXT_CORE/VISION assignment companion:
  [`docs/qwen3_6_27b_q5090_v2_tensor_plan.md`](../../docs/qwen3_6_27b_q5090_v2_tensor_plan.md)

The output **always contains all three segments** (TEXT_CORE + MTP_DRAFT + VISION_ENCODER) so
it never needs regenerating; the runtime skips the segments it does not need. There are no runtime
transforms: the converter may canonicalize documented TEXT_CORE shapes into q5090 runtime-native form.

## Layout

| file | role |
|---|---|
| `qtypes.py` | qtype/layout/module/source enums + per-qtype constants (on-disk ABI) |
| `format.py` | header / module / tensor / segment / fusion record serialization, FNV-1a-64, crc32 |
| `quantize.py` | GPU per-group symmetric quantization (fp16 scale, signed codes) |
| `packing.py` | LSB-first two's-complement bit packing (Q4/Q5/Q6) + W8 |
| `layouts.py` | ROW_SPLIT and CONTIGUOUS encoders + decoders |
| `tensor_plan.py` | declarative source->qtype/layout/slice tables for all three segments |
| `convert.py` | CLI: stream shards, assemble the packed file + `.manifest.json` |
| `verify.py` | L0 structure/plan checks, L1 bit-exact quantizer checks, and `conv_dump.v3.json` |
| `tests/` | CPU round-trip tests for packing/quant/layouts/container |

GDN `linear_attn.conv1d.weight` is transformed from HF raw `[10240,1,4]` into q5090 canonical
`[10240,4,1]` with runtime-native tap-major payload order. This is an offline converter transform; the
runtime model bind path does not allocate temporary conv1d storage.

## Usage

Run from the repo root with a CUDA-enabled env:

```bash
/home/neroued/miniconda3/envs/py311/bin/python -m tools.q5090_convert.convert \
  --model /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --out out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus

/home/neroued/miniconda3/envs/py311/bin/python -m tools.q5090_convert.verify \
  out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus
```

The converter writes `<weights>.manifest.json` next to the packed file. The verifier writes
`out/conv_dump.v3.json` unless `--dump FILE` is supplied. Useful flags: `--out FILE`, `--device cpu`,
and `--force` (config mismatch -> warn). The verifier defaults to the local HF model path above; pass
`--model DIR` to verify another source tree.

Tests:

```bash
python -m tools.q5090_convert.tests.test_packing   # or: pytest tools/q5090_convert/tests
```
