# MTP Foundation Part 1 Verification

This report records the local verification artifacts for
`docs/2026-07-02-mtp-foundation-part1-design.md` Part 1.

## q5090 v3 MTP_DRAFT Layout

Generated artifact:

```text
out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus
out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus.manifest.json
out/conv_dump.v3_mtp_w8g32.json
```

These files are under `out/` and are ignored by git.

Generation command:

```bash
/home/neroued/miniconda3/envs/py311/bin/python -m tools.q5090_convert.convert \
  --model /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --out out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus
```

Converter summary:

```text
blocks=1164 segments=1312 fusion_groups=130 modules=3
TEXT_CORE      blocks=819 payload=16378329088
MTP_DRAFT      blocks=12  payload=451267584
VISION_ENCODER blocks=333 payload=293396992
file_bytes=17123284480
```

Verifier command:

```bash
/home/neroued/miniconda3/envs/py311/bin/python -m tools.q5090_convert.verify \
  out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus \
  --model /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16 \
  --dump out/conv_dump.v3_mtp_w8g32.json
```

Verifier result:

```text
L0 structural checks done; problems=0
L1 value checks done; problems=0
OK: L0 + L1 passed in 198s

Q4G64_F16S   blocks=182 scale_bad=0 code_bad=0 control_bad=0
Q5G64_F16S   blocks=295 scale_bad=0 code_bad=0 control_bad=0
Q6G64_F16S   blocks=2   scale_bad=0 code_bad=0 control_bad=0
W8G128_F16S  blocks=2   scale_bad=0 code_bad=0 control_bad=0
BF16_CTRL    blocks=582 scale_bad=0 code_bad=0 control_bad=0
FP32_CTRL    blocks=96  scale_bad=0 code_bad=0 control_bad=0
W8G32_F16S   blocks=5   scale_bad=0 code_bad=0 control_bad=0
```

The new MTP_DRAFT layout contributes 12 blocks, 16 segments, and 2 fusion groups:
`ATTN_IN` and `MLP_GATEUP`. The five MTP dense/fused blocks use `W8G32_F16S`; the seven
MTP control tensors use `BF16_CTRL`. `TEXT_CORE` has no W8 tensors. The only `W8G128_F16S`
tensors are the existing VISION merger FC weights.

## C++ Scope

This W8G32 MTP precision update is verified in the q5090 Python converter/verifier. C++ runtime and
model-card support for MTP remains outside Part 1.
