# Single-GPU serving performance

Tested Git revisions:

- Qwen3.6-35B-A3B MTP3: `b1a220f028aa750f75bceb3522ac00bbaab7e42d`;
- Qwen3.6-35B-A3B MTP0 and all Qwen3.6-27B results:
  `0795169393cab0f2c16246d4bac20dee735dc2a4`.

These measurements characterize the two registered NInfer targets independently on one NVIDIA
GeForce RTX 5090. They cover long-context prefill and baseline decode with MTP disabled, plus
long-reasoning and cross-scenario decode with MTP enabled.

All requests were submitted serially to a persistent `ninfer-serve` process over the loopback
OpenAI-compatible HTTP endpoint. Each reported fixture used five fixed seeds. Values are arithmetic
mean ± sample standard deviation; warm-up requests are excluded.

## Test method

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GiB |
| CUDA compile/runtime/driver API | 13.1 / 13.1 / 13.1 |
| Request mode | One active request, `stream=false` |
| Maximum context | 262,144 tokens |
| Prefill chunk | 1,024 tokens |
| KV cache | INT8 group-64 |
| CUDA Graph | Enabled |
| Prefix reuse | Disabled |
| Sampling | Temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0 |
| MTP0 | no `--spec` |
| MTP3 | `--spec mtp --draft-tokens 3 --lm-head-draft` |

The MTP0 profile uses four Long NIAH prompts with approximately 8K, 64K, 128K, and 256K tokens.
Thinking is disabled and the output budget is 128 tokens. These runs measure prefill throughput,
server-internal time to first token, and baseline decode throughput at each context length. Content
scenarios are not repeated with MTP disabled because they do not change the baseline decode path.

The MTP3 corpus contains three long-reasoning fixtures with thinking enabled and a 65,536-token
output limit, followed by twelve fixtures covering code, story, translation, and structured output.
The cross-scenario fixtures disable thinking and use a 4,096-token output limit. The tables report
actual completion lengths rather than assuming that every request reaches its limit.

Metrics are computed from the server's unrounded phase timings and MTP counters:

```text
prefill_tok_s = prompt_tokens / prefill_seconds
server_ttft_ms = 1000 * (prepare_seconds + vision_seconds + prefill_seconds)
decode_tok_s = (completion_tokens - 1) / decode_seconds
mtp_acceptance = accepted_tokens / drafted_tokens
mtp_tokens_per_round = 1 + accepted_tokens / mtp_rounds
```

## Reproduction

Build `ninfer-serve` and prepare the registered `.ninfer` artifacts. The refreshed 35B-A3B MTP3
tables use:

```bash
python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --mode mtp3 \
  --output profiles/bench/serve_corpus_35b_mtp3_20260724
```

Omit `--mode` and supply both artifacts to run the complete two-target MTP0/MTP3 campaign:

```bash
python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --output profiles/bench/serve_corpus_20260720
```

## `qwen3_6_35b_a3b`

### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 15,544.3 ± 242.4 | 500.2 ± 7.8 | 271.1 ± 3.6 |
| 64,512 | 5 | 10,809.0 ± 95.3 | 6,009.9 ± 52.6 | 242.9 ± 1.3 |
| 130,048 | 5 | 7,828.4 ± 34.1 | 16,693.3 ± 71.2 | 219.4 ± 1.6 |
| 260,096 | 5 | 5,157.1 ± 52.4 | 50,598.8 ± 519.7 | 188.2 ± 2.1 |

### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 7,933.0 ± 1,852.3 | 695.1 ± 17.7 | 83.3% ± 2.8% | 3.50 ± 0.08 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 584.0 ± 10.6 | 72.4% ± 1.7% | 3.17 ± 0.05 |
| `long_decode_aime26_30` | 5 | 61,743.6 ± 4,489.5 | 629.4 ± 15.7 | 79.6% ± 3.2% | 3.39 ± 0.10 |

### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 635.0 ± 24.2 | 71.8% ± 4.2% | 3.15 ± 0.13 |
| Story | 15 | 434.9 ± 34.8 | 38.2% ± 5.9% | 2.15 ± 0.18 |
| Translation | 15 | 598.6 ± 26.6 | 66.1% ± 4.5% | 2.98 ± 0.14 |
| Structured | 15 | 714.3 ± 36.2 | 87.7% ± 6.6% | 3.63 ± 0.20 |

## `qwen3_6_27b`

### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 3,218.1 ± 4.3 | 2,392.4 ± 3.0 | 77.6 ± 0.1 |
| 64,512 | 5 | 2,655.9 ± 2.9 | 24,335.7 ± 25.2 | 70.7 ± 0.1 |
| 130,048 | 5 | 2,185.3 ± 0.3 | 59,590.3 ± 8.9 | 64.5 ± 0.1 |
| 260,096 | 5 | 1,614.8 ± 0.6 | 161,221.8 ± 62.5 | 54.8 ± 0.1 |

### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 11,009.4 ± 419.1 | 174.2 ± 3.3 | 79.9% ± 2.0% | 3.40 ± 0.06 |
| `long_decode_aime26_15` | 5 | 62,652.6 ± 3,000.4 | 158.7 ± 5.2 | 73.3% ± 3.4% | 3.20 ± 0.10 |
| `long_decode_aime26_30` | 5 | 47,837.8 ± 5,882.7 | 169.0 ± 2.7 | 79.3% ± 2.0% | 3.38 ± 0.06 |

### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 163.9 ± 6.2 | 72.5% ± 3.9% | 3.18 ± 0.12 |
| Story | 15 | 110.4 ± 9.2 | 37.9% ± 6.0% | 2.14 ± 0.18 |
| Translation | 15 | 153.6 ± 11.7 | 65.7% ± 7.5% | 2.97 ± 0.23 |
| Structured | 15 | 189.1 ± 15.7 | 88.9% ± 10.2% | 3.67 ± 0.31 |

The MTP0 and MTP3 suites intentionally measure different supported workloads. No per-scenario
MTP0/MTP3 speedup is reported.
