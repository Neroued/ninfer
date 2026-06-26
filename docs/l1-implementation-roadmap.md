# L1 Implementation Roadmap — qwen3.6-ultraspeed

> Status: roadmap (brainstorm). Date: 2026-06-26.
> Scope: the **sequencing** for implementing the 13 L1 operators from
> [`l1-operator-catalog.md`](l1-operator-catalog.md), using the api→wrapper→launcher→kernel layout
> from [`l1-kernel-layering.md`](l1-kernel-layering.md) and the `Weight` seam from
> [`weight-handle-design.md`](weight-handle-design.md). Correctness-first, profiler-gated
> (`design.md` §11): get a correct end-to-end forward (M2), then drive to roofline (M3).

This document fixes the **rhythm**; each tier gets its own task-by-task plan under `docs/plans/`.

---

## 1. Tiers (by implementation difficulty)

| Tier | Ops | Source of implementation | v1 performance bar |
|---|---|---|---|
| **1 — Simple** | `rmsnorm`, `residual_add`, `sigmoid_gate_mul`, `silu_and_mul` (exemplar, exists), `l2norm`, `gdn_gating`, `rope`, `embed_gather`, `argmax`, `causal_conv1d` | hand-write; short grid-stride / reduction kernels | **bandwidth roofline now** — memory-bound, so a careful first impl can hit ≥85–90% of peak DRAM BW |
| **2 — Port** | `gated_delta_rule_chunked` (prefill), `gated_delta_rule_recurrent` (decode) | **adapt `~/chunked_gdn`** (`chunked/`, `ar/`) into our `Tensor`/state + the L1 layout | chunked already FLA-competitive; AR adequate (GDN recurrence isn't the decode bottleneck) |
| **3 — Complex, staged** | `linear` (W4A16/W5/W6/W8 + dense GEMV/GEMM, bespoke to the q5090 packed layout), `gqa_attention_prefill`/`_decode` (flash-style causal GQA) | hand-write; reference algorithms only (`flashinfer`, `vllm`, `flash-attention`, the q5090 layout spec) | **correctness baseline now; roofline in M3** — each gets its own multi-stage sub-plan with profiling |

Difficulty is not the same as math complexity: GDN's recurrence math is the hardest, but it is **already
solved** in `~/chunked_gdn`; `linear`/`attention` are "standard" but reaching their roofline is the core
project effort, so they are staged.

---

## 2. Rhythm — five phases

1. **Tier-1 simple ops** (`docs/plans/l1-tier1-simple-ops.md`). Batch, subagent-driven, one op per task.
   CPU-ref parity + multi-shape tests + tune each to the **bandwidth roofline** (they are cheap to get
   right and there is no reason to leave throughput on the table).
2. **Tier-2 GDN port** (`docs/plans/l1-tier2-gdn.md`, later). Adapt the chunked + AR kernels; parity vs
   the `~/chunked_gdn` CPU references and FLA; perf parity vs its bench.
3. **Tier-3 correctness baselines** (`docs/plans/l1-tier3-*.md`, later). *Naive* `linear` (dense + each
   qtype) and *naive* `gqa_attention` — just enough to unblock end-to-end M2.
4. **M2 integration.** Wire the L2 card + engine; validate **per-layer parity** (cosine / max-abs vs
   HF/vLLM dumps) and **greedy token-match** vs the reference (`design.md` §12).
5. **M3 optimization ladder.** Drive `linear` (the decode weight-bandwidth roofline — the headline
   metric) and `gqa_attention` (flash) to peak; then fusion + CUDA-graph (`design.md` §11, M4–M5).

Phases 1–3 produce the kernels; phase 4 is the first correct run; phase 5 is the performance campaign.

---

## 3. Performance bar per tier

- **Tier 1 (now):** memory-bound → acceptance = ncu DRAM throughput **≥ 85% of peak** at a representative
  decode (`T=1`) *and* prefill (`T` large) shape, with an absolute effective-BW sanity check against the
  **1.79 TB/s** roofline. `compute-sanitizer` clean.
- **Tier 2 (now):** numerical parity with the chunked_gdn CPU refs; chunked perf within a small margin of
  the existing optimized kernel (it is the same kernel, ported).
- **Tier 3 (M3):** `linear` decode GEMV → the weight-bandwidth roofline (`~14–15 GB ÷ 1.79 TB/s`,
  `design.md` §1); `gqa_attention` → flash-attention-class. Correctness baseline only in phase 3.

---

## 4. Test & performance methodology (all tiers)

- **CPU reference per op** (fp32 host implementation) — the parity oracle. Pattern after
  `~/chunked_gdn/reference/`.
- **Multi-shape / multi-dim sweeps** — every op is tested across decode (`T=1`), short, and long
  (`T` up to a few K) sequences and its 2-D/3-D head/group views; plus edge cases (non-multiple-of-warp
  sizes, single element).
- **`compute-sanitizer`** (memcheck/racecheck) clean is the correctness gate before any tuning.
- **`nsys`/`ncu` via the `profile-cuda` skill** — `nsys` to confirm one clean launch, `ncu --set
  roofline`/memory metrics for the DRAM-throughput acceptance gate. Reports land in `profiles/`.

---

## 5. Execution model

Subagent-driven (`superpowers:subagent-driven-development`): one **fresh subagent per op-task**, then a
two-stage review (spec compliance, then code quality). Work happens on a feature branch, never `master`.
Each tier is a separate plan so it stays reviewable.

**Plans:** `docs/plans/l1-tier1-simple-ops.md` (next) → `…/l1-tier2-gdn.md` → `…/l1-tier3-linear.md` +
`…/l1-tier3-attention.md`.

---

## 6. Sources

- `l1-operator-catalog.md`, `l1-kernel-layering.md`, `weight-handle-design.md`, `l2-model-card-design.md`,
  `design.md` §11/§12 (ladder + validation).
- `~/chunked_gdn` (GDN chunked + AR + CPU refs + bench), `~/flash-linear-attention` (GDN baseline),
  `~/flashinfer` / `~/vllm` / `~/llama.cpp` (attention + algorithm references).
- `~/.cursor/skills/profile-cuda` (nsys/ncu capture recipes).
