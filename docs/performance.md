# Performance

This page records the representative NInfer CLI measurements used in the project README. The
current NInfer runs were captured on 2026-07-19 from mainline revision `fdaf991` in
`profiles/bench/qwen36_mtp_cli_repeat5_20260719_current`. The unchanged llama.cpp rows come from the
2026-07-18 run in `profiles/bench/qwen36_mtp_cli_repeat5_20260718`. Raw run output remains local;
the complete aggregated results and test conditions are preserved here.

## Test conditions

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| Driver reported by the run | 591.86 |
| Prompt source | `examples/cli/messages/long_8k.json` |
| NInfer prepared prompt | 7,678 tokens |
| Context capacity | 16,384 tokens |
| Generated tokens | 512 |
| KV cache | BF16 |
| Sampling | greedy |
| NInfer prefill chunk | 1,024 |
| CUDA Graph | enabled |
| MTP draft window | 0 or 3 |
| NInfer MTP proposal head | optimized when MTP=3 |
| Warm-up | one run per configuration, excluded |
| Measurement | five runs per configuration |
| Reported statistic | mean plus sample standard deviation |
| llama.cpp build | `b201-b369ae383` |

The NInfer and llama.cpp runs used the same source prompt and 512-token generation budget, with each
engine applying its own chat path and reporting its own prefill/decode phase rate. Configurations
were alternated between MTP off and on to reduce ordering bias.

## Full results

| Model | Engine | MTP | Prefill mean ± SD | Decode mean ± SD | MTP acceptance | Accepted length |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | llama.cpp | 0 | 2,966.58 ± 4.51 tok/s | 60.60 ± 0.14 tok/s | — | — |
| Qwen3.6-27B | llama.cpp | 3 | 2,646.64 ± 30.47 tok/s | 140.38 ± 0.70 tok/s | not reported | not reported |
| Qwen3.6-27B | NInfer | 0 | 3,278.79 ± 32.63 tok/s | 76.97 ± 0.58 tok/s | — | — |
| Qwen3.6-27B | NInfer | 3 | 3,260.67 ± 30.85 tok/s | 188.30 ± 1.47 tok/s | 87.47% | 3.62 tok/round |
| Qwen3.6-35B-A3B | llama.cpp | 0 | 6,272.86 ± 268.07 tok/s | 197.80 ± 1.13 tok/s | — | — |
| Qwen3.6-35B-A3B | llama.cpp | 3 | 5,476.82 ± 312.99 tok/s | 285.88 ± 5.28 tok/s | not reported | not reported |
| Qwen3.6-35B-A3B | NInfer | 0 | 15,674.64 ± 39.74 tok/s | 275.38 ± 0.20 tok/s | — | — |
| Qwen3.6-35B-A3B | NInfer | 3 | 15,496.77 ± 140.45 tok/s | 593.60 ± 1.52 tok/s | 79.56% | 3.39 tok/round |

## Comparisons

| Model | Comparison | Prefill ratio | Decode ratio |
|---|---|---:|---:|
| Qwen3.6-27B | NInfer MTP=3 / NInfer MTP=0 | 0.99x | 2.45x |
| Qwen3.6-35B-A3B | NInfer MTP=3 / NInfer MTP=0 | 0.99x | 2.16x |
| Qwen3.6-27B | NInfer / llama.cpp at MTP=3 | 1.23x | 1.34x |
| Qwen3.6-35B-A3B | NInfer / llama.cpp at MTP=3 | 2.83x | 2.08x |

Ratios are ratios of the five-run means before display rounding.

## Artifact boundary

This is an end-to-end configuration comparison, not an identical-format kernel comparison.

NInfer used the published native artifacts:

- `qwen3_6_27b.ninfer`, 17,495,365,888 bytes;
- `qwen3_6_35b_a3b.ninfer`, 22,373,184,256 bytes.

llama.cpp used:

- `Qwen3.6-27B-Q4_K_M-mtp.gguf`;
- `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.

The artifacts use different quantization and storage recipes. The comparison therefore describes
the measured complete-engine combinations and must not be interpreted as an isolated runtime or
kernel result.

## NInfer command

The measured MTP configuration for either registered artifact was:

```bash
./build/apps/ninfer "$ARTIFACT" \
  --messages examples/cli/messages/long_8k.json \
  --device 0 \
  --max-context 16384 \
  --prefill-chunk 1024 \
  --kv-dtype bf16 \
  --max-new 512 \
  --greedy \
  --mtp-draft-tokens 3 \
  --lm-head-draft
```

The MTP=0 control replaced the final two flags with `--mtp-draft-tokens 0`.

## llama.cpp command

The corresponding llama.cpp MTP configuration was:

```bash
PROMPT=$(jq -r '.[0].content' examples/cli/messages/long_8k.json)

llama-cli \
  -m "$GGUF" \
  --prompt "$PROMPT" \
  --conversation \
  --single-turn \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --ctx-size 16384 \
  --n-predict 512 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --gpu-layers all \
  --split-mode none \
  --main-gpu 0 \
  --flash-attn on \
  --cache-type-k bf16 \
  --cache-type-v bf16 \
  --temperature 0 \
  --seed 0 \
  --no-display-prompt \
  --color off \
  --perf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0 \
  --spec-draft-type-k bf16 \
  --spec-draft-type-v bf16
```

The MTP=0 control omitted the five `--spec-*` options.
