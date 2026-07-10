# q5090_convert

Offline, GPU-accelerated quantizing converter: original **Qwen3.6-27B** bf16 safetensors
-> a single self-describing artifact in the `q5090_w4g64_mixed_v4_1` format.

- Binary file contract (self-contained v4.1, including tokenizer and optional draft `lm_head`):
  [`docs/q5090_packed_file_format_v4.md`](../../docs/q5090_packed_file_format_v4.md)
- Draft-head decision (Q4 @ N=131072):
  [`docs/2026-07-06-lm-head-draft-q4-decision.md`](../../docs/2026-07-06-lm-head-draft-q4-decision.md)

The output contains TEXT_CORE + MTP_DRAFT + VISION_ENCODER and, unless disabled, the independent
LM_HEAD_DRAFT module. It also embeds `tokenizer.json`, `merges.txt`, and `generation_config.json` as
CPU-only assets. There are no runtime weight transforms: the converter may canonicalize documented
TEXT_CORE shapes into q5090 runtime-native form.

Unless `--no-draft-head` is given, the independent LM_HEAD_DRAFT module contains a shortlisted `lm_head`
(`[131072,5120]` Q4G64) plus a paired `[131072]` int32 id-map. The shortlist is the top-N tokens by
frequency (from `--ranking`, default `tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64`)
unioned with the tokenizer's special ids; its rows are re-quantized from the original bf16
`lm_head.weight`, and the id-map records each draft row's real vocab id. See `draft_head.py`.

## Layout

| file | role |
|---|---|
| `qtypes.py` | qtype/layout/module/source enums + per-qtype constants (on-disk ABI) |
| `format.py` | header / module / tokenizer / tensor / segment / fusion serialization and hashes |
| `quantize.py` | GPU per-group symmetric quantization (fp16 scale, signed codes) |
| `packing.py` | LSB-first two's-complement bit packing (Q4/Q5/Q6) + W8 |
| `layouts.py` | ROW_SPLIT and CONTIGUOUS encoders + decoders |
| `tensor_plan.py` | declarative source->qtype/layout/slice tables for all four modules |
| `draft_head.py` | deterministic frequency-shortlist selection for the optional draft `lm_head` |
| `convert.py` | CLI: stream shards, assemble the packed file + `.manifest.json` |
| `verify.py` | L0 structure/tokenizer/plan checks, L1 bit-exact checks, and `conv_dump.v4_1.json` |
| `tests/` | CPU round-trip tests for packing/quant/layouts/container/shortlist |

GDN `linear_attn.conv1d.weight` is transformed from HF raw `[10240,1,4]` into q5090 canonical
`[10240,4,1]` with runtime-native tap-major payload order. This is an offline converter transform; the
runtime model bind path does not allocate temporary conv1d storage.

## Usage

Run from the repo root with a CUDA-enabled env:

```bash
/home/neroued/miniconda3/envs/py311/bin/python -m tools.q5090_convert.convert \
  --model /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --out out/qwen3_6_27b.q5090_w4g64_mixed_v4_1.qus

/home/neroued/miniconda3/envs/py311/bin/python -m tools.q5090_convert.verify \
  out/qwen3_6_27b.q5090_w4g64_mixed_v4_1.qus
```

The converter writes `<weights>.manifest.json` next to the packed file. The verifier writes
`out/conv_dump.v4_1.json` unless `--dump FILE` is supplied. Useful flags: `--out FILE`, `--device cpu`,
`--force` (config mismatch -> warn), `--no-draft-head`, `--ranking FILE`, `--draft-n N`, and
`--tokenizer DIR`. The converter reads the three embedded runtime assets and `tokenizer_config.json`
from that build-time directory. The verifier re-derives the shortlist for L1 (`--ranking`/`--tokenizer`) and
defaults to the local HF model path above; pass `--model DIR` to verify another source tree.

Tests:

```bash
python -m tools.q5090_convert.tests.test_packing   # or: pytest tools/q5090_convert/tests
```
