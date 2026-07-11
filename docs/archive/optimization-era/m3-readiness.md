# M3 Readiness

> Status: complete for the regenerated Qwen3.6 chat-template M2.8 baseline.
> Scope: official M3 entry evidence package for qwen3.6-ultraspeed.
> Note: the M2.8 e2e benchmark apparatus (`qus_e2e_bench`, its JSON report schema, and the
> committed baseline summaries) has been retired. Throughput is now measured with the `qus_bench`
> tool (see [`bench/README.md`](../bench/README.md)); the numbers below are the historical M2.8
> evidence, and the report/summary artifacts they reference no longer live in the tree.

The previous pre-chat-template and whitespace-output smoke observations are superseded. The current
baseline was regenerated on real q5090 weights with `.messages.json` prompt fixtures rendered through
the Qwen3.6 chat template, fixed prompt ids, and stop tokens `[248046, 248044]`.

## Baseline Identity

- Commit: `7f61950236135e6a196928673396995c643df52b`
- Worktree state recorded by benchmark: clean
- GPU: NVIDIA GeForce RTX 5090, driver `591.86`
- q5090: `out/qwen3_6_27b.q5090_w4g64_mixed_v1.qus`
- q5090 SHA256: `9c468418a26d2b243e00bb63eecd0e02be0cbfe577e35559202a00a2918cd2db`
- Fixture manifest: `bench/fixtures/prompts/m2.8-v1.manifest.json`
- Max context: `8192`
- Workspace policy: `block_scoped_mixer_mlp_rewind`
- Hidden device allocations: `false`

## Evidence Artifacts

- Output gate raw report (historical, local): `profiles/e2e/m3-output-gate.json`
- Prefill gate raw report (historical, local): `profiles/e2e/m3-prefill-gate.json`
- L1 microbench snapshot: `profiles/microbench/l1-microbench-20260628-235356.log`

These M2.8 raw reports and microbench logs were local under ignored `profiles/`. The committed
baseline summaries that used to live under `docs/bench/baselines/` were removed together with the
e2e apparatus; the tables below preserve the audited numbers from that baseline.

## Output Gate

Command class: `m3_output_gate`

Required cases were run with `max_new_tokens=96`, `warmup_repeats=1`, and `measured_repeats=3`.
Decoded clean output was nonempty for all measured repeats of `cn_short`, `en_short`, `code_short`,
and `math_short`.

| Case | Prompt tokens | Median prefill s | Median prefill tok/s | Median decode s | Median decode tok/s | Deterministic |
|---|---:|---:|---:|---:|---:|---|
| `cn_short` | 31 | 1.02391 | 30.2761 | 6.44325 | 3.88003 | true |
| `en_short` | 26 | 0.898695 | 28.9308 | 24.6520 | 3.85364 | true |
| `code_short` | 54 | 1.66028 | 32.5247 | 24.8044 | 3.82996 | true |
| `math_short` | 46 | 1.42937 | 32.1820 | 24.7461 | 3.83898 | true |

## Prefill Gate

Command class: `m3_prefill_gate`

The required `long_2k` case was run with `max_new_tokens=1`, `warmup_repeats=0`, and
`measured_repeats=1`.

| Case | Prompt tokens | Median prefill s | Median prefill tok/s | Decode tok/s | Deterministic |
|---|---:|---:|---:|---:|---|
| `long_2k` | 7932 | 243.324 | 32.5984 | n/a | true |

`decode_tok_s` is intentionally not valid for this case because `max_new_tokens=1` produces no
decode loop tokens after prefill.

## Memory Summary

Both official gate summaries recorded complete engine-owned arena accounting and no hidden device
allocations.

| Arena | Capacity bytes | Used bytes | Peak used bytes |
|---|---:|---:|---:|
| `weights` | 17372552192 | 16378329088 | 16378329088 |
| `cache` | 758454024 | 691312128 | 691312128 |
| `workspace` | 4294967296 | 0 | 0 |

## M3 Starting Point

The regenerated baseline is valid as the entry point for M3 performance optimization. The headline
single-stream decode rate across the short output-gate cases is currently about `3.83-3.88 tok/s`,
and long prompt prefill is currently about `32.6 prompt tok/s`. Per-kernel optimization should use the
local L1 microbench snapshot as the convenience baseline and collect `ncu` evidence for each targeted
kernel family before making performance claims.
