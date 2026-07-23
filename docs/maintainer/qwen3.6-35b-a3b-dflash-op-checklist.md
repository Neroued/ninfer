# Qwen3.6-35B-A3B dFlash operator checklist

This is the working inventory for qualifying and optimizing Qwen3.6-35B-A3B inference before
integrating a dFlash proposal path with at most 15 draft tokens. It is derived from the current
family schedules, the target's three execution leaves, and the semantic Op calls under `src/ops`.

The dFlash proposal algorithm is not yet present in this repository. This document therefore does
not invent its proposal-only operators. It does two concrete things:

1. inventories every Op used by the current 35B Text, MTP, and Vision inference paths; and
2. defines the target-verification work that is independent of how dFlash produces its drafts.

When the dFlash proposal contract is implemented, add its proposal-only Ops to Section 6 and keep
the target verification list in Section 4 as the shared performance authority.

## 1. Workload and completion contract

Use the following notation throughout this checklist:

- `K`: number of proposed draft tokens, `1 <= K <= 15`;
- `T_v = K + 1`: target-verification columns, therefore `T_v = 2..16`;
- `T=1`: ordinary decode and fallback;
- all shapes are logical `[rows, columns]` unless stated otherwise.

An Op is complete for the dFlash verification path only after all four boxes in its row are closed:

- **C**: its production route is checked directly against the independent mathematical oracle at
  every exact `T=1..16`, including route boundaries;
- **B**: its benchmark accepts an explicit `T` sweep and has RTX 5090/CUDA 13.1/sm_120a results for
  every exact `T=1..16`;
- **O**: the selected implementation is the measured winner for the relevant shape and state or
  context regime, with no known cliff inside `T=1..16`;
- **E**: its improvement is visible in the containing layer or complete speculative round, not
  only in an isolated microbenchmark.

For context-dependent attention, **C** and **B** cover BF16 and INT8-G64 KV caches, eager and CUDA
Graph execution, and representative live contexts including graph-frontier boundaries. For
stateful GDN Ops, they cover every legal initial snapshot slot, sequential state publication, and
the slots selected by accept lengths `0..K`.

Do not use pairwise kernel parity as the correctness oracle. Follow
[`op-development.md`](op-development.md): exact transforms use an exact oracle; floating-point Ops
use the complete FP32/FP64 formula from represented public inputs and a route-appropriate output
criterion.

## 2. Exact 35B topology relevant to the inventory

| Component | Registered 35B-A3B shape/profile |
|---|---|
| Text backbone | hidden 2048, 40 layers |
| Layer mix | 30 GDN layers and 10 full-attention layers at indices 3, 7, ..., 39 |
| Full attention | 16 query heads, 2 KV heads, head dimension 256, rotary dimension 64 |
| GDN | 16x128 query/key, 32x128 value, 8192 QKV rows, 4096 z rows, convolution width 4 |
| GDN recurrent state | FP32 `[128,128,32]` per GDN layer and snapshot slot |
| Sparse MoE | 256 routed experts, top-8, routed/shared intermediate 512 |
| Text MoE codecs | 37 Q4 gate/up + Q5 down layers; layers 34, 38, and 39 use Q6 down; shared experts are W8 |
| Token embedding | W8 `[248320,2048]` |
| Text output head | Q6 `[248320,2048]` |
| MTP | one full-attention sparse-MoE layer, W8/W8 routed experts, Q4 shortlist draft head `[131072,2048]` |
| Vision | 27 blocks, hidden 1152, intermediate 4304, 16 heads of dimension 72 |

Primary source routes:

- `src/targets/qwen3_6/impl/runtime/text_context_impl.h`
- `src/targets/qwen3_6/impl/runtime/mtp_impl.h`
- `src/targets/qwen3_6/impl/runtime/vision_context_impl.h`
- `src/targets/qwen3_6_35b_a3b/impl/variant.cpp`
- `src/targets/qwen3_6_35b_a3b/impl/config.h`
- [`qwen3.6-35b-a3b-model.md`](qwen3.6-35b-a3b-model.md)
- [`qwen3.6-35b-a3b-artifact.md`](qwen3.6-35b-a3b-artifact.md)

## 3. Target-verification call graph

One `target_verify(T_v)` call executes the following semantic Ops. Multiplicity is per call, not
per generated request.

| Stage | Op sequence | Calls |
|---|---|---:|
| Input | `offset_i32_positions`, `embedding` | 1 each |
| 10 full-attention layers | `rmsnorm` input; `attn_input_proj`; `rmsnorm` Q and K; `rope`; `gqa_attention`; `sigmoid_mul`; W8 `linear_add`; post `rmsnorm`; `sparse_moe` | 10 layers |
| 30 GDN layers | `gdn_norm_gating_proj`; W8 `gdn_input_proj_conv_snapshot`; `gated_delta_rule_snapshot`; `gated_rmsnorm`; W8 `linear_add`; post `rmsnorm`; `sparse_moe` | 30 layers |
| Output | final `rmsnorm`; Q6 `linear`; `argmax` | 1 each |

The resulting per-call counts are:

| Op | Calls | Main logical shape |
|---|---:|---|
| `rmsnorm` | 71 | `[2048,T_v]`, plus Q/K norms `[4096,T_v]`/`[512,T_v]` |
| `sparse_moe` | 40 | hidden `[2048,T_v]`, 256 routed experts, top-8 |
| W8 `linear_add` | 40 | weight `[2048,4096]` |
| `gdn_norm_gating_proj` | 30 | hidden `[2048,T_v]`, 32 value heads |
| W8 `gdn_input_proj_conv_snapshot` | 30 | parent weight `[12288,2048]`, convolution channels 8192 |
| `gated_delta_rule_snapshot` | 30 | Q/K `[128,16,T_v]`, V `[128,32,T_v]` |
| `gated_rmsnorm` | 30 | `[4096,T_v]` |
| `attn_input_proj` | 10 | parent W8 `[9216,2048]` |
| `rope`, `gqa_attention`, `sigmoid_mul` | 10 each | full-attention shapes above |
| input/output Ops | 1 each | positions, embedding, final head and selection |

This is the first performance scope. Proposal generation matters only after this complete target
pass is efficient for the accept-length distribution that dFlash actually produces.

## 4. Ordered dFlash verification checklist

### P0: dominant and route-changing Ops

Work these rows in order unless whole-round profiling demonstrates a different dominant cost.

| ID | Status | Op and exact role | Current `T=1..16` route/risk | Existing test/benchmark entry points | C | B | O | E |
|---|---|---|---|---|---|---|---|---|
| DV-01 | [x] | W8 `gdn_input_proj_conv_snapshot`, 30 calls | `T=1` fused decode; every exact `T=2..16` uses one split-K MMA kernel with fused projection, convolution, SiLU, Q/K/V split, z store, and snapshot publication; `T>=17` retains the composed fallback. | `test_gdn_input_proj_conv_snapshot`; `w8_input_proj_bench --op gdn-snapshot`; `gdn_layer_bench` | [x] | [x] | [x] | [x] |
| DV-02 | [x] | `gdn_norm_gating_proj`, 30 calls | Every exact `T=1..16` uses one cooperative split-32 MMA kernel with fused RMS reduction, normalized BF16 hidden publication, direct normalized A/B projection, and gate/beta transforms; larger T retains the composed route. | `test_gdn_gating_proj`, `test_gdn_gating_proj_plan`; `gdn_gating_proj_bench --35b --norm-control`, `gdn_layer_bench` | [x] | [x] | [x] | [x] |
| DV-03 | [x] | `gated_delta_rule_snapshot`, 30 calls | Every exact `T=1..16` uses one four-warp recurrent kernel with fused Q/K normalization, register-resident running state, vectorized FP32 snapshot stores, no workspace, and no route boundary. Each 35B layer reads one 2 MiB state and publishes another 2 MiB per column. | `test_gated_delta_rule`; `gated_delta_rule_bench --small-t --H_v 32`, `gdn_layer_bench --qk-norm` | [x] | [x] | [x] | [x] |
| DV-04 | [x] | `gqa_attention`, 10 calls | `T<=6` keeps the direct small-T route. For 35B, `T=7..12` uses prompt through 512 visible keys and otherwise 2 small-T chunks; `T=13..16` uses prompt through 1024 and otherwise 3 chunks. Each chunk has at most six sequential columns and reuses one workspace allocation. BF16 and INT8-G64 share the context policy. | `test_gqa_attention`; `gqa_attention_bench --append-small-t`/`--cached-small-t`, `attention_layer_bench` | [x] | [x] | [x] | [x] |
| DV-05 | [ ] | `sparse_moe`, 40 calls | Main Q4+Q5/Q6 profiles stay on Small-T for all `T=2..16` (Small-T through 44). It has no route cliff here but is invoked after every layer and streams routed/shared weights. | `test_sparse_moe`; `sparse_moe_bench` | [ ] | [ ] | [ ] | [ ] |
| DV-06 | [ ] | Q6 `linear` output head `[248320,2048]`, 1 call | `T=1..4` SIMT C4, `T=5..6` SIMT C8, `T>=7` MMA R64C128. The `T=6/7` transition spans the full vocabulary head. | `test_linear`, `test_q6_linear_plan`; `linear_op_bench` | [ ] | [ ] | [ ] | [ ] |
| DV-07 | [ ] | W8 `attn_input_proj` `[9216,2048]`, 10 calls | `T=1` decode; `T=2..12` SIMT R8C4; `T=13..16` MMA R32C128. The dFlash range crosses `T=12/13`. | `test_linear`, `test_input_proj_plan`; `w8_input_proj_bench`, `attention_layer_bench` | [ ] | [ ] | [ ] | [ ] |
| DV-08 | [ ] | W8 `linear_add` `[2048,4096]`, 40 calls | All `T=1..16` use SIMT R8C4. Verify throughput and residual epilogue efficiency across exact partial tiles. | `test_linear`, `test_linear_add_plan`; `linear_op_bench`, layer benches | [ ] | [ ] | [ ] | [ ] |

#### DV-01 qualification record

The first Op was qualified on NVIDIA GeForce RTX 5090, CUDA 13.1, `sm_120a`. Timed samples used a
256 MiB L2 flush, five warmups, 50 measured repetitions, and the exact `T=1..16` sweep.

- **C:** the production Op passed its independent complete FP64 oracle at every exact `T=1..16`.
  The test exact-decodes the W8 weight, evaluates projection, convolution, SiLU, z, and sequential
  snapshots from represented inputs, poisons every output and inactive state slot, checks full
  writes and unchanged slots, and exercises every legal initial slot `0..16` at the maximum extent.
  The existing Q4/Q5 peer route also passed the expanded exact-T regression.
- **B:** the dedicated benchmark reports the selected route and the semantically equivalent
  composed control. Production medians were 23.84 us at `T=1`, 21.70--21.76 us at `T=2..8`,
  23.81 us at `T=9..12`, and 25.76--25.86 us at `T=13..16`.
- **O:** the former composed route took 31.52--33.70 us over `T=2..16`. The selected fused route is
  23%--31% faster over `T=7..16`, removes four launches, and reduces this Op's maximum-`T=16`
  transient workspace from 512 KiB to zero. A measured projection-plus-post-kernel candidate was
  also 13%--20% slower over `T=7..16` and was not retained.
- **E:** in captured 35B GDN-layer replay, production uses six graph nodes at `T=7..16`, versus ten
  for the explicit composed control. Median layer latency improved from 56.32 to 52.48 us at `T=7`
  (6.8%) and from 72.99 to 70.94 us at `T=16` (2.8%).

The current fixed `K<=5` MTP controller cannot issue `target_verify(T=7..16)`, so this record makes
an Op and containing-layer claim, not a complete dFlash-round claim. The public 35B eager Engine
route passed a `K=5`, `tg8` smoke run. A CUDA Graph smoke at the same current `K` exposed a
pre-existing graph workspace-planning shortfall (39,325,696 bytes consumed versus 37,748,736
planned); it was not used as DV-01 evidence.

#### DV-02 qualification record

This Op was qualified on NVIDIA GeForce RTX 5090, CUDA 13.1, `sm_120a`, with a 256 MiB L2 flush,
ten warmups, 200 measured repetitions, and every exact `T=1..16`.

- **C:** the 35B contiguous-parent public Op passed its complete FP64 oracle for normalized hidden,
  A/B projections, gate, and beta at every exact `T=1..16`. All three outputs are poisoned before
  execution and compared in full. Plan tests cover the new `T=16/17` route and workspace boundary;
  the 27B two-weight peer form retains its existing regression.
- **B:** production medians were 5.38--7.42 us at `T=1..12` and 9.25--9.47 us at `T=13..16`.
  The benchmark reports the complete Op route and an explicit composed control.
- **O:** composed took 11.26--11.30 us at `T=7..16`; fused is 16%--34% faster and removes one
  launch, at the cost of increasing `T=16` transient workspace from 64 KiB to 130 KiB. The
  implementation uses compile-time norm capacities 6/8/12/16 so extending the long-acceptance
  range does not regress the original small-T kernel. Focused ncu captures at `T=12/16` found the
  same 128 registers/thread and 33.3% theoretical occupancy; the remaining duration increase
  follows the additional norm work rather than a new occupancy cliff.
- **E:** captured 35B GDN-layer replay uses five graph nodes instead of six. Median layer latency
  improved from 52.48 to 50.43 us at `T=7` (3.9%), from 54.53 to 50.46 us at `T=8` (7.5%), and
  from 72.96 to 69.02 us at `T=16` (5.4%).

As with DV-01, complete `target_verify(T=7..16)` evidence remains blocked by the current fixed
`K<=5` controller; DV-02 closes the Op and containing-layer criteria only.

#### DV-03 qualification record

This Op was qualified on NVIDIA GeForce RTX 5090, CUDA 13.1, `sm_120a`, with a 256 MiB L2 flush,
20 warmups, 500 measured repetitions, 17 snapshot slots, and every exact `T=1..16`.

- **C:** the 35B fused-normalization snapshot route passed the independent complete FP64 recurrence
  oracle at every exact `T=1..16`. The oracle normalizes raw represented BF16 Q/K independently,
  evaluates every sequential state transition, and compares BF16 output plus all FP32 snapshots.
  Output is poisoned before launch; inactive slots retain their seeded sentinel or initial-state
  contents; and the maximum extent is checked from every legal initial slot `0..16`. The 27B peer
  and near-zero Q/K cases retain focused regression coverage.
- **B:** the benchmark now accepts an explicit exact-T subset, uses the production 17-slot state,
  prints the complete route, and counts the unavoidable initial-state read plus every snapshot
  write. Production medians grow smoothly from 5.41 us at `T=1` to 17.66 us at `T=8` and
  29.95 us at `T=16`; the logical useful-byte rate reaches 1.20 TB/s at `T=16`.
- **O:** the retained route keeps the full FP32 state tile in registers, publishes 128-bit
  coalesced snapshot stores, fuses Q/K normalization, and allocates no workspace. A six-warp
  candidate improved isolated medians by 7%--10% at selected T values but had no stable
  containing-layer benefit; streaming snapshot stores also regressed some T values. Neither was
  retained. The explicit two-L2Norm composition is faster in isolation at large T (23.81 versus
  29.95 us at `T=16`) but loses after its extra launches and staging are included in the layer.
- **E:** across six alternating cold-cache process pairs, the production fused-normalization GDN
  layer had a median-of-medians of 50.45 us versus 52.48 us for the explicit composition at `T=8`
  (3.9% faster), and 69.90 versus 72.75 us at `T=16` (3.9% faster). Both routes used the same other
  four layer kernels and the production route remained five CUDA Graph nodes versus seven.

The write-every-column state contract remains the dominant fact: one `T=16` call publishes 32 MiB
of FP32 snapshots per GDN layer. Whether dFlash needs every intermediate state is a round-level
decision in Section 5, not a legal shortcut inside this Op qualification. Complete
`target_verify(T=7..16)` evidence remains blocked by the current fixed `K<=5` controller.

#### DV-04 qualification record

This Op was qualified on NVIDIA GeForce RTX 5090, CUDA 13.1, `sm_120a`, for A1
append-and-attend and A3 cached attention, BF16 and INT8-G64 KV, every exact `T=1..16`, and live
contexts 128, 1024, and 8192. The containing-layer measurements used CUDA Graph replay, a 256 MiB
L2 flush, five warmups, and 40 measured repetitions.

- **C:** every 35B `T=1..16` A1 and A3 route passed the shared independent FP64 attention oracle
  directly in both cache formats. The test also compares all A1/A2 cache code and scale bits,
  exercises the first chunked point after each prompt frontier, and captures T=16 prompt and
  chunked CUDA Graphs for both cache formats. Sequential chunks preserve causality because each
  chunk publishes its cache rows before the following chunk; every output column is still checked
  against the complete logical cache formula.
- **B:** `gqa_attention_bench` now accepts T=1..16 for append and cached modes, reports
  `small_t`, `prompt`, or `chunked_small_t`, counts chunks, models each chunk's cache and scratch
  traffic separately, and allocates through the public workspace query. At context 8192, append
  BF16 hot-cache medians were 73.24 us at T=7 and 108.97 us at T=16; INT8 measured 65.94 and
  109.25 us. With a 256 MiB L2 flush before each sample, the same points were 86.85/129.86 us and
  73.82/125.34 us. Cached timings remained within 2 us of append. The short-context prompt route
  measured 15--22 us across T=7..16.
- **O:** the former prompt route measured 474--476 us for BF16 and 430--436 us for INT8 at context
  8192. Chunking is therefore 77%--85% faster at T=7..16 there. Two chunks become worthwhile
  beyond 512 visible keys; the third becomes worthwhile beyond 1024. The retained policy avoids
  the measured three-chunk loss below that second frontier, leaves the 27B route unchanged, and
  caps transient storage at the reused T=6 allocation (8,486,400 bytes) instead of scaling it
  with T. The 35B MTP graph ranges include both crossover frontiers so a captured graph never
  spans a route change.
- **E:** against the explicit old prompt control, captured 35B attention-layer latency at context
  8192 improved from 525.54 to 120.10 us (77.1%) at T=7 and from 551.33 to 177.31 us (67.8%) at
  T=16 for BF16. INT8 improved from 480.42 to 107.90 us (77.5%) and from 505.09 to 173.31 us
  (65.7%). At context 1024, the gains were 30%/13% for BF16 and 32%/10% for INT8 at T=7/T=16.
  The route transitions themselves do not introduce a latency cliff: layer latency improves by
  2%--5% at visible 512/513 for T=12 and by 9%--10% at visible 1024/1025 for T=16.

As with the preceding rows, the current controller cannot issue a complete T=7..16 target verify,
so DV-04 closes the Op and full-attention-layer criteria rather than claiming a complete dFlash
round speedup.

### P1: frequent elementwise, normalization, and selection Ops

These Ops are individually smaller, but their aggregate launch and memory cost can become material
after the P0 contractions and state transitions are optimized.

| ID | Status | Op | Calls per verify | Qualification focus | Test/benchmark |
|---|---|---|---:|---|---|
| DV-09 | [ ] | `rmsnorm` | 71 | Exact `T=1..16` for hidden, Q, and K row counts; assess safe fusion only at an owning semantic boundary. | `test_rmsnorm`; `rmsnorm_bench` |
| DV-10 | [ ] | `gated_rmsnorm` | 30 | `[4096,T]` with z gate; check launch/memory scaling and downstream arithmetic criterion. | `test_rmsnorm`; `rmsnorm_bench` |
| DV-11 | [ ] | `rope` | 10 | `[4096,T]` Q and `[512,T]` K, rotary 64, absolute positions; current launcher changes small-token geometry after 6. | `test_rope`; `rope_bench` |
| DV-12 | [ ] | `sigmoid_mul` | 10 | Gate `[4096,T]`; exact odd/partial token extents and graph replay. | `test_sigmoid_mul`; `sigmoid_mul_bench` |
| DV-13 | [ ] | `embedding` | 1 | W8 `[248320,2048]`, gathered ids for `T=1..16`; include repeated ids and physical row decode. | `test_embedding`; `embedding_bench` |
| DV-14 | [ ] | `offset_i32_positions` | 1 | Exact position construction for all `T`, including near context limit. | `test_position`; dedicated benchmark missing |
| DV-15 | [ ] | `argmax` | 1 | Q6-head output with `T=1..16`, physical stride, valid vocabulary rows, stable ties. | `test_argmax`; `argmax_bench` |
| DV-16 | [ ] | `sample` | 1 per ordinary/accepted publication path | Sampling semantics and cost on the final committed column; penalties and seeded behavior remain unchanged. | `test_sampling`; `sampling_select_bench` |

### P0 measurement infrastructure

- [x] Extend `gdn_layer_bench --t-sweep` from the former hard limit `1..6` to exact `1..16`,
  with sufficient snapshot slots for each case.
- [x] Extend `attention_layer_bench --t-sweep` from the current hard limit `1..6` to exact
  `1..16`.
- [x] Add an explicit exact `T=1..16` small-state mode to `gated_delta_rule_bench`, with 17
  snapshot slots, complete route labels, and state-publication byte accounting.
- [ ] Make every P0 benchmark print the selected production route so a fast result cannot hide an
  unintended fallback.
- [x] Add one target-layer benchmark for each of the 35B attention and GDN layers at `T=1..16`.
- [ ] Add a complete `target_verify`/speculative-round measurement before accepting any isolated
  Op optimization as an end-to-end win.

## 5. Stateful and round-level constraints outside an individual Op

These are not semantic Ops, but they can dominate or block a 15-draft-token implementation and
must be resolved before the Op campaign can be interpreted.

### GDN snapshot capacity and bandwidth

The current family layout reserves `K+2` GDN snapshot slots. For all 30 GDN layers:

- FP32 SSM state per slot is exactly 60 MiB;
- BF16 convolution state per slot is 1.40625 MiB;
- total state per slot is 61.40625 MiB;
- `K=15` therefore requires 17 slots, about 1.019 GiB of persistent GDN snapshot storage;
- one `T_v=16` verification publishes about 960 MiB of SSM snapshots and 22.5 MiB of convolution
  snapshots, before counting reads or arithmetic.

Checklist:

- [ ] Decide whether every intermediate draft position must retain a full state snapshot for the
  dFlash acceptance contract.
- [ ] If not, redesign snapshot publication/commit ownership before optimizing the current
  write-every-column kernel.
- [ ] Measure snapshot write/read bandwidth separately from recurrent arithmetic.
- [ ] Validate accepted-state selection for every accept length `0..15`, including all-accepted
  and first-token-rejected cases.
- [ ] Verify CUDA Graph address stability and memory capacity with the final slot layout.

### Current fixed-`K` blockers

- [ ] Replace the target `kMaximumMtpDraftTokens = 5` capacity with the selected dFlash capacity
  without introducing a runtime target branch.
- [ ] Update the `mtp_round` contracts that currently admit only `1 <= K <= 5`, or replace those
  MTP-specific round Ops with dFlash-owned semantic Ops if their formulas differ.
- [ ] Resize/redesign `MtpGqaEnvelopes::ar`, per-round token/state arrays, graph frontiers, and
  workspace layouts that are derived from the current maximum.
- [ ] Redesign the fixed nine-counter acceptance-statistics layout; it encodes five accepted-token
  positions and cannot represent `K=15`.
- [ ] Qualify eager and captured graph construction/replay at every `K=1..15`, including route
  changes at `T_v=7` and `T_v=13`.
- [ ] Update CLI/serving ranges and their external schema documentation only when the new behavior
  is implemented end to end.

## 6. Proposal and speculative-round operator inventory

This section records the current MTP path. It is part of 35B inference today, but it is not a
specification for dFlash proposal generation.

### Current MTP model call

| Status | Stage | Active Ops |
|---|---|---|
| [ ] | Input/stem | optional `embedding`; two `rmsnorm`; `mtp_pack_fc_input`; W8 `linear` `[2048,4096]`; input `rmsnorm` |
| [ ] | Full attention | W8 `attn_input_proj`; Q/K `rmsnorm`; `rope`; `gqa_attention`; `sigmoid_mul`; W8 `linear` `[2048,4096]`; `residual_add` |
| [ ] | MoE/output | `rmsnorm`; W8/W8 `sparse_moe`; final `rmsnorm`; Q4 shortlist or Q6 full-head `linear`; `argmax`; optional `mtp_remap_draft_token` |
| [ ] | Prefill split | `gqa_kv_append`, followed by final-column `gqa_attention_cached` |

The MTP W8/W8 sparse-MoE remains on Small-T for `T<=17`; its prefill route starts at `T=18`.
Generic W8 `[2048,4096]` linear stays on its SIMT route through `T=56`.

### Current speculative-round control

| Status | Active Op | Role |
|---|---|---|
| [ ] | `mtp_prepare_verify_inputs` | Assemble shifted target-verification ids |
| [ ] | `mtp_accept_tokens` | Compare target/proposal tokens, sample the rejection position, and publish acceptance counts |
| [ ] | `mtp_prepare_shifted_ids` | Build the next MTP batch ids |
| [ ] | `mtp_gather_hidden_row` | Select the accepted target hidden row |
| [ ] | `mtp_remap_draft_token` | Convert shortlist head id to the global token domain |
| [ ] | `assign_i32_scalar`, scalar set/increment helpers | Select state slot and advance round scalars |

After the dFlash proposal formula and represented inputs are known:

- [ ] identify which current MTP Ops remain semantically identical;
- [ ] add every new proposal-only Op here with its exact shape and call count;
- [ ] give each new floating-point Op an independent complete-formula oracle;
- [ ] benchmark proposal generation and target verification separately, then measure the complete
  round against ordinary decode at the actual acceptance-length distribution.

## 7. Text prefill inventory

Text prefill is not the first dFlash optimization scope, but it is part of complete 35B inference
and supplies the resident state consumed by verification.

| Status | Stage | Active Ops |
|---|---|---|
| [ ] | Input assembly | `fill_i32_positions` or `offset_i32_positions`; `embedding`; optional `scatter` of Vision embeddings |
| [ ] | Full attention | `rmsnorm`; `attn_input_proj`; Q/K `rmsnorm`; `rope`; `gqa_attention`; `sigmoid_mul`; `linear_add`; post `rmsnorm`; `sparse_moe` |
| [ ] | GDN | `gdn_norm_gating_proj`; W8 `gdn_input_proj`; `causal_conv1d_silu`; three `extract_bf16_columns`; `gated_delta_rule`; `gated_rmsnorm`; `linear_add`; post `rmsnorm`; `sparse_moe` |
| [ ] | Output | final `rmsnorm`; last-column Q6 `linear`; `sample` or `argmax` |

`gated_delta_rule` is the prefill state transition and is distinct from
`gated_delta_rule_snapshot`. Its long-prefill implementation uses an internal `l2norm` route when
the complete extent reaches 64 columns; that private composition is not an additional target-level
Op call.

## 8. Vision inventory

Vision runs only for multimodal prompts and is not on the steady-state dFlash verification path.
It remains in the complete target inventory.

| Status | Stage | Active Ops |
|---|---|---|
| [ ] | Patch/input | `cast_fp32_to_bf16`; Q6 `linear` patch projection `[1152,1536]`; `add_bias`; `vision_pos_embed_add` |
| [ ] | 27 attention blocks | two `layer_norm`; Q4 QKV `linear` `[3456,1152]`; `add_bias`; 2-D `rope`; `vision_attention`; Q5 projection `linear` `[1152,1152]`; `add_bias`; `residual_add` |
| [ ] | 27 MLP blocks | Q4 FC1 `linear` `[4304,1152]`; `add_bias`; tanh-approximate `gelu`; Q5 FC2 `linear` `[1152,4304]`; `add_bias`; `residual_add` |
| [ ] | Merger | `layer_norm`; W8 `linear` `[4608,4608]`; `add_bias`; exact `gelu`; W8 `linear` `[2048,4608]`; `add_bias` |
| [ ] | Text insertion | `scatter` merged `[2048,N]` visual embeddings into the text hidden input |

## 9. Unique active Op families

This normalized list is the completeness check for Sections 3, 6, 7, and 8. A family appearing in
several schedules is listed once.

### Text and common

- [ ] `fill_i32_positions`
- [ ] `offset_i32_positions`
- [ ] `embedding`
- [ ] `scatter`
- [ ] `rmsnorm`
- [ ] `attn_input_proj`
- [ ] `rope`
- [ ] `gqa_attention`
- [ ] `sigmoid_mul`
- [ ] `linear_add`
- [ ] `gdn_norm_gating_proj`
- [ ] `gdn_input_proj`
- [ ] `gdn_input_proj_conv_snapshot`
- [ ] `causal_conv1d_silu`
- [ ] `causal_conv1d_silu_snapshot`
- [ ] `extract_bf16_columns`
- [ ] `gated_delta_rule`
- [ ] `gated_delta_rule_snapshot`
- [ ] `gated_rmsnorm`
- [ ] `sparse_moe`
- [ ] `linear`
- [ ] `argmax`
- [ ] `sample`
- [ ] `residual_add`

### Current MTP/round control

- [ ] `mtp_pack_fc_input`
- [ ] `gqa_kv_append`
- [ ] `gqa_attention_cached`
- [ ] `mtp_prepare_verify_inputs`
- [ ] `mtp_accept_tokens`
- [ ] `mtp_prepare_shifted_ids`
- [ ] `mtp_gather_hidden_row`
- [ ] `mtp_remap_draft_token`
- [ ] `set_i32_scalar`
- [ ] `assign_i32_scalar`
- [ ] `increment_i32_scalar`
- [ ] `increment_i64_scalar`

### Vision-only

- [ ] `cast_fp32_to_bf16`
- [ ] `add_bias`
- [ ] `layer_norm`
- [ ] `vision_pos_embed_add`
- [ ] `vision_attention`
- [ ] `gelu`

Raw CUDA copies, cache-container mutation, arenas, graph capture/replay, and GDN state storage are
runtime/core mechanisms, not Ops. They belong in round-level profiling when they have measurable
cost, but should not be assigned a fabricated mathematical Op contract.

Registered families such as `linear_swiglu`, standalone `gdn_gating`, `silu_mul`, and
`mtp_split_attn_in` are not called by the current 35B schedules and are intentionally absent.
`l2norm` appears only as a private composition inside the long-prefill `gated_delta_rule` route.

## 10. Recommended execution order

1. Remove the `K<=5` and benchmark `T<=6` measurement blockers without changing inference
   semantics.
2. Establish exact `T=1..16` correctness and route-labeled baselines for DV-01 through DV-08.
3. Resolve GDN snapshot ownership and bandwidth before tuning a kernel around a state layout that
   dFlash may not need.
4. Optimize the measured P0 bottlenecks and verify each change in a complete GDN layer, attention
   layer, and target verification pass.
5. Re-profile the complete round, then work DV-09 through DV-16 only where aggregate traces show
   material cost.
6. Add and qualify the actual dFlash proposal-only Ops after their semantic contract exists.
7. Publish an end-to-end speedup only from full Engine inference with the registered 35B artifact,
   stated context, output length, proposal length, and observed acceptance-length distribution.
