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

## Python Ref Model Verified MTP

Verification command:

```bash
/home/neroued/miniconda3/envs/py311/bin/python - <<'PY'
from tools.parity.ref_model import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    DEFAULT_STOP_TOKEN_IDS,
    RefModel,
    chat_prompt_ids,
    load_tokenizer,
    parse_stop_token_ids,
)

weights = 'out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus'
decode = 8
model = RefModel(weights, device='cuda', resident='auto')
tokenizer = load_tokenizer(DEFAULT_MODEL)
prompt_ids, rendered = chat_prompt_ids(tokenizer, [{'role': 'user', 'content': DEFAULT_PROMPT}])
stop = parse_stop_token_ids(DEFAULT_STOP_TOKEN_IDS)
print(f'PROMPT_TOKENS {len(prompt_ids)}')
base = model.forward(prompt_ids, decode, stop_token_ids=stop)
print('BASE', base)
for draft_count in range(1, 6):
    tokens, stats = model.forward_mtp_verified(prompt_ids, decode, draft_count=draft_count, stop_token_ids=stop)
    print(
        f'MTP draft_count={draft_count} match={tokens == base} '
        f'draft_tokens={stats.draft_tokens} accepted_tokens={stats.accepted_tokens} '
        f'acceptance_rate={stats.acceptance_rate:.6f} tokens={tokens}'
    )
PY
```

Result:

```text
PROMPT_TOKENS 31
BASE [103900, 122977, 3709, 98629, 102539, 111038, 1710, 198]
```

| draft tokens requested | output matched baseline | verified draft tokens | accepted tokens | acceptance rate |
|---:|---|---:|---:|---:|
| 1 | yes | 7 | 5 | 0.714286 |
| 2 | yes | 7 | 5 | 0.714286 |
| 3 | yes | 7 | 5 | 0.714286 |
| 4 | yes | 7 | 5 | 0.714286 |
| 5 | yes | 7 | 5 | 0.714286 |

The acceptance rate is not abnormally low for this smoke prompt. The verified greedy path produced the
same token sequence as non-MTP greedy for every requested MTP draft-token count 1..5.

## C++ Scope

This W8G32 MTP precision update is verified in the q5090 Python converter/verifier and Python ref
model. C++ runtime/model-card support for MTP remains outside Part 1.
