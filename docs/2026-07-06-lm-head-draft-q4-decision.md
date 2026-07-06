# Decision: Q4 draft `lm_head` at N=131072 for MTP drafting

Status: accepted (2026-07-06). This document records the final, measured decision to add a
Q4-quantized, frequency-shortlisted draft `lm_head` for MTP drafting, and fixes its size
(`N = 131072`), precision (`Q4G64`), and value source (quantized from the original bf16
`lm_head.weight`). The on-disk representation is specified by the v4 packed-format spec
([q5090_packed_file_format_v4.md](q5090_packed_file_format_v4.md)); the runtime/kernel changes are a
direct port from the measurement branch `feat/lm-head-draft-shortlist`.

## Problem

In MTP speculative decoding, each decode round proposes `k` draft tokens and then verifies them with
one full forward. Every draft step runs a vocab projection over the full `lm_head`
(`[248320, 5120]`, `Q6G64`, `4000` bytes/row = the nibble + high + scale planes of §9 in the format
spec). At `T = 1` this projection is pure HBM streaming: reading all `248320` rows dominates the
draft-step cost. With `k = 3` drafts per round, the full head is streamed ~3 extra times per round on
top of the single verify read.

Two independent levers shrink that streamed traffic:

1. **Shortlist (fewer rows).** Most drafted tokens come from a small, high-frequency slice of the
   vocabulary. A draft head that stores only the top-`N` rows streams `N/248320` of the bytes. A draft
   argmax over the shortlist is remapped to a real vocab id through an index -> id map.
2. **Lower precision (fewer bytes/row).** `Q4G64` stores `2720` bytes/row versus `Q6G64`'s `4000`
   (68%), a further ~32% cut to the per-row GEMV.

Correctness is unconditional: **verify always uses the full `Q6` head**, so the draft head only
changes *which tokens are proposed* (a speed/acceptance knob), never which tokens are *emitted*. A
worse draft only lowers acceptance length; it can never produce a wrong token.

## Decision

Ship a draft head with:

- **N = 131072** rows (frequency-ranked shortlist over the measurement corpus, union a
  force-include set of 21 special/control ids).
- **`Q4G64_F16S`**, `ROW_SPLIT`, `[131072, 5120]`.
- **Quantized from the original bf16 `lm_head.weight`** (`quantize_core`, group 64, `qmax=7`,
  `qmin=-8`), not from the dequantized `Q6` values.
- An **`int32[131072]` id map** (shortlist index -> real vocab id) used to remap the draft argmax.

The draft head is optional: when absent, drafting uses the full head unchanged.

## Evidence

All numbers are from a single same-session A/B (50 held-out prompts, `max_new = 128`, `k = 3`, CUDA
graphs on). Sources: [profiles/lm_head_draft/analysis.json](../profiles/lm_head_draft/analysis.json),
[profiles/lm_head_draft/bench_q4_q6_t1.csv](../profiles/lm_head_draft/bench_q4_q6_t1.csv), and
[out/lm_head_draft.q4.131072.manifest.json](../out/lm_head_draft.q4.131072.manifest.json). Canonical
greedy (`k = 0`) runs at 81.6 tok/s; every MTP config below is far above it.

### End-to-end decode (aggregate tok/s, 50 prompts, k=3)

| config | bytes/row | tok/s (agg) | vs baseline | acc_len | acc_rate |
|---|---:|---:|---:|---:|---:|
| baseline (full `Q6` head) | 4000 | 130.05 | - | 2.699 | 0.566 |
| Q6 draft, N=65536 | 4000 | 130.87 | +0.6% | 2.545 | 0.515 |
| Q6 draft, N=98304 | 4000 | 132.16 | +1.6% | 2.615 | 0.538 |
| Q6 draft, N=131072 | 4000 | 132.58 | +1.9% | 2.656 | 0.552 |
| Q4 draft, N=65536 | 2720 | 131.70 | +1.3% | 2.545 | 0.515 |
| Q4 draft, N=98304 | 2720 | 134.04 | +3.1% | 2.619 | 0.540 |
| **Q4 draft, N=131072** | **2720** | **134.40** | **+3.3%** | **2.662** | **0.554** |

By mean-prompt tok/s the winner is 136.78 vs 132.08 baseline = **+3.6%**.

### Q4 vs Q6 at matched N (same session)

| N | Q4 tok/s | Q6 tok/s | Q4 over Q6 | Q4 acc_len | Q6 acc_len | acc_len delta |
|---:|---:|---:|---:|---:|---:|---:|
| 65536 | 131.70 | 130.87 | +0.6% | 2.5454 | 2.5452 | +0.0002 |
| 98304 | 134.04 | 132.16 | +1.4% | 2.6194 | 2.6154 | +0.0041 |
| 131072 | 134.40 | 132.58 | +1.4% | 2.6617 | 2.6564 | +0.0053 |

**Acceptance is identical between Q4 and Q6 at every N** (delta within run-to-run noise, and slightly
positive). Acceptance depends only on whether the draft's argmax equals the target's argmax; the
argmax of a `5120`-dim dot product is robust to Q4's per-element drift (mean relative element error
~0.284, but `mean_abs` only `1.2e-3`). So Q4 pays no acceptance cost for its smaller rows.

### Kernel bandwidth (T=1 GEMV, `bench_q4_q6_t1.csv`)

| N | Q4 warm us | Q4 DRAM% | Q6 warm us | Q6 DRAM% | Q4/Q6 time |
|---:|---:|---:|---:|---:|---:|
| 65536 | 118.1 | 90.1% | 166.9 | 93.4% | 0.71 |
| 98304 | 170.1 | 92.2% | 245.7 | 95.7% | 0.69 |
| 131072 | 222.2 | 94.7% | 327.1 | 97.5% | 0.68 |

The Q4 per-row GEMV is ~30-32% faster, matching the `2720/4000 = 0.68` byte ratio, at 90-95% of peak
DRAM bandwidth. There is no kernel-efficiency loss that would eat the byte savings.

### Coverage, footprint, safety

- Coverage at N=131072 (`manifest.json`): train covered mass `0.9939`; held-out accepted-token mass
  `0.9919`; worst domain (other_chat) `0.9786`.
- Footprint: payload `356,515,840` B (`340 MiB`) + id map `524,288` B (`0.5 MiB`).
- Numerics: `compute-sanitizer memcheck` clean on a Q4 decode; an fp64-oracle GEMV parity test for
  the `lm_head`-family T1 route passes at rel_l2 ~`1.65e-3` for both Q4 and Q6.

## Why Q4 over Q6

Q4 **strictly dominates** the Q6 draft: same acceptance, 68% of the bytes, ~1.4% higher end-to-end
tok/s at N=131072, and identical correctness (verify uses the full head regardless). There is no
regime in the data where the Q6 draft is preferable.

## Why N=131072

Larger N raises coverage and acceptance but costs bytes; the returns flatten past ~128k:

- N=65536 -> N=98304 gains ~+1.8% tok/s and lifts acc_len 2.545 -> 2.619;
- N=98304 -> N=131072 gains only ~+0.3% tok/s and acc_len 2.619 -> 2.662, while worst-domain
  held-out coverage reaches `0.9786` (~0.99 target with margin only at the larger N).

N=131072 is the knee: it meets the coverage target across domains and captures essentially all of the
available acceptance, at a `340 MiB` cost that is acceptable in the 32 GB budget. Smaller N would trade
measurable acceptance for a footprint the budget does not require; larger N buys almost nothing.

## Divergence (not a correctness concern)

Emitted token ids are **not** bit-identical to canonical greedy (`k = 0`), but this predates and is
independent of the draft head: the stock MTP engine's per-round acceptance pattern is not
float-associative (the GDN chunked state update), so a near-tie argmax can flip when acceptance
changes. Baseline `k = 3` already matches `k = 0` on only 24/50 prompts (71% token prefix). The Q4
draft (N=131072) matches `k = 0` on 22/50 (68% prefix) - the same near-tie class, not a new error
source. Every emitted token remains the greedy argmax of the full head at its position.

## Alternatives rejected

- **Q6 draft head** - dominated by Q4 (above).
- **Smaller N (<=98304)** - measurably lower acceptance for a footprint saving the budget does not need.
- **Keeping the `QUSLMHD1` sidecar** - fine for branch measurement, but on main the draft head is a
  first-class part of the packed weights (v4), avoiding a second file and a separate load path.

## Integration (follow-on work)

1. **Format** - [q5090_packed_file_format_v4.md](q5090_packed_file_format_v4.md) adds the draft-head
   weights block (`Q4G64` ROW_SPLIT, `source_kind = LM_HEAD_DRAFT`) and its id-map block
   (`I32_CTRL` CONTIGUOUS, `source_kind = LM_HEAD_DRAFT_IDMAP`) to the `TEXT_CORE` module, plus a
   `DRAFT_HEAD_PRESENT` header flag.
2. **Converter** - select the shortlist, quantize the bf16 rows to Q4, emit the two blocks and the id
   map (the selection logic exists as `tools/lm_head_draft/build_draft_head.py` on the feat branch).
3. **Runtime/kernel** - the runtime-N `Q4` `lm_head`-family GEMV, the `mtp_remap_draft_token` kernel,
   and the draft-site wiring are direct ports from `feat/lm-head-draft-shortlist` (commit `d999455`);
   the loader binds the two v4 blocks instead of parsing a sidecar.
