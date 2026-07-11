# MTP M3 Target Verify Cost Checkpoint

Date: 2026-07-03

Command:

```bash
./build/bench/qus_target_verify_bench \
  --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus \
  --warmup 1 --reps 3 --parity
```

Environment:

- GPU: NVIDIA GeForce RTX 5090
- Timing: CUDA event time on the model stream
- State: zero-state target KV/GDN before each timed sample
- Weights: `out/qwen3_6_27b.q5090_w4g64_mixed_v3_mtp_w8g32.qus`

## Verify Cost

Baseline T=1 decode mean: `13.1765 ms`.

`epsilon = verify_ms / decode_t1_ms - 1`, matching the throughput model in
`docs/2026-07-03-mtp-spec-decode-overview.md` section 7.2.

| T | verify ms | ratio vs decode | epsilon |
|---:|---:|---:|---:|
| 2 | 54.8359 | 4.16163 | 3.16163 |
| 3 | 55.3396 | 4.19986 | 3.19986 |
| 4 | 58.9182 | 4.47145 | 3.47145 |
| 5 | 62.4989 | 4.74320 | 3.74320 |
| 6 | 66.4134 | 5.04028 | 4.04028 |

Current M3 generic verify is much slower than the roofline assumption that small-T verify costs
roughly one target decode step. At k=5, `epsilon ~= 4.04`, so M5 should prioritize small-T target
verify kernels before relying on the speedup model.

## Parity Check

The same run compared `target_verify(T=1..6)` against sequential T=1 decode replay with identical
input ids and positions at both empty context (`cache_offset=0`) and after a four-token prefix
(`cache_offset=4`). Decode replay used the existing fused T=1 path; verify used the new generic
small-T path. Argmax matched for every checked column.

| prefix | T | hidden max abs | logits max abs | argmax mismatches |
|---:|---:|---:|---:|---:|
| 0 | 1 | 0.25 | 0.125 | 0 |
| 0 | 2 | 0.25 | 0.125 | 0 |
| 0 | 3 | 0.25 | 0.125 | 0 |
| 0 | 4 | 0.25 | 0.125 | 0 |
| 0 | 5 | 0.25 | 0.125 | 0 |
| 0 | 6 | 0.25 | 0.136719 | 0 |
| 4 | 1 | 0.136719 | 0.101562 | 0 |
| 4 | 2 | 0.3125 | 0.125 | 0 |
| 4 | 3 | 0.3125 | 0.125 | 0 |
| 4 | 4 | 0.3125 | 0.15625 | 0 |
| 4 | 5 | 0.3125 | 0.15625 | 0 |
| 4 | 6 | 0.3125 | 0.15625 | 0 |

## Memory Budget

For `mtp_draft_tokens=5`, `max_ctx=8192`, and the default `prefill_chunk=1024`:

| arena | bytes | MiB |
|---|---:|---:|
| cache | 1,574,773,732 | 1501.82 |
| workspace | 385,875,968 | 368.00 |

The cache budget includes six GDN snapshot slots. The SSM component grows from 144 MiB at k=0 to
864 MiB at k=5, so the M3 snapshot-slot increment is exactly +720 MiB. The total cache delta from
k=0 to k=5 at the same context and prefill chunk is 768.48 MiB after MTP KV, conv slots, StepState
buffers, alignment slack, and the existing margin are included.
