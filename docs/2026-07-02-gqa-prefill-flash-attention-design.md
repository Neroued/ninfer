# GQA Prefill Flash Attention — methodology / design (goal)

> Status: design goal (not yet implemented). Date: 2026-07-02.
> Scope: the **target design** for a tensor-core flash-attention kernel that replaces the
> correctness-first `gqa_attention_prefill_kernel`. This document fixes the architecture, the
> hardware envelope, the roofline it must reach (explicit target: **≥80% tensor-core efficiency at
> `pp4096`**), and the correctness/bench bar. It is deliberately a
> "goal", not an executable task list: it locks *what* to build and *why the design can reach
> roofline*, and leaves *how* (tile constants, pipeline depth, scheduling, register/smem trade) as a
> preserved, `ncu`-guided tuning space for codex. Decode is out of scope and left unchanged.

## 1. Context and motivation

With the prefill `linear` path now moved onto a tensor-core GEMM (see
[`2026-07-01-prefill-linear-foundation-design.md`](2026-07-01-prefill-linear-foundation-design.md)),
**GQA prefill attention is the remaining long-context wall.** From the length sweep
([`bench/q5090-v3-prefill-length-nsys-kernel-breakdown.md`](bench/q5090-v3-prefill-length-nsys-kernel-breakdown.md)):

| pp | `gqa_attention_prefill` share | avg launch |
|---:|---:|---:|
| 512 | 17.1% | 3.25 ms |
| 1024 | 31.8% | 12.7 ms |
| 2048 | 49.2% | 50.8 ms |
| 4096 | **66.1%** | **187.3 ms** |

The cause is purely kernel quality. The current kernel launches one block per `(q_head, token)`, one
thread per `head_dim` element, and loops every previous key doing a `__syncthreads` block-reduce for
each dot product — `O(T)` barriers × `O(T)` keys, zero tensor cores, no flash/online reuse. This is a
kernel problem, not a scheduling one: the prefill schedule already runs attention over the full
`[D, T]` activation.

The headroom is large. Causal attention FLOPs grow as `T²` while the streamed K/V bytes grow as `T`,
so long context is a **compute-bound tensor-core problem** — exactly where the design below aims.

## 2. Locked constraints

1. **Fixed I/O contract** (unchanged; the public `gqa_attention_prefill(...)` signature and its
   shape/scale validation stay as-is). All BF16, `head_dim` contiguous:
   - `q:[256,24,T]`, `k/v:[256,4,T]`; output `out:[256,24,T]`.
   - KV cache `[256, padded_context, 4]` — each `(kv_head, pos)` is a contiguous 256-vector, i.e. a
     natural `S×D` matrix per head (the TMA/streaming-friendly layout).
   - `head_dim=256`, `q_heads=24`, `kv_heads=4` (**GQA group = 6**), causal, `scale = 1/√256`,
     fp32 softmax.
   - RoPE (partial NeoX, `rotary_dim=64`) and KV-cache fill happen **before** the kernel. The kernel
     consumes already-RoPE'd Q (`qn`) and cached K (RoPE'd) / V (not RoPE'd). **No RoPE inside the
     kernel.** The existing cheap fill kernel is kept; attention reads K/V from the cache.
2. **Hand-rolled PTX only — no CUTLASS.** `mma.sync` for the matmuls and hand-rolled TMA/`mbarrier`
   for data movement, consistent with the rest of the engine.
3. **BF16 activations (A16).** FP8/FP4 attention is a later, accuracy-gated, `sm_120a`-only axis and
   is out of scope here.
4. **Prefill adapts to the existing cache layout; the layout never adapts to prefill.** No second K/V
   copy, no re-layout.

## 3. Hardware target — `sm_120` (RTX 5090) capability ledger

Confirmed against the NVIDIA CUDA C++ Programming Guide (§5.1, Tables 28/29/31) and the CCCL PTX
instruction reference. The important, counter-intuitive point: consumer Blackwell has **no async
matrix engine**, but it **does** have TMA and asynchronous barriers as *baseline* features.

| Primitive | `sm_120`? | Notes |
|---|---|---|
| `mma.sync.m16n8k16.f32.bf16.bf16.f32` | ✅ baseline | the only bf16 tensor-core path |
| `ldmatrix` / `ldmatrix.trans` | ✅ baseline | Q/K frags; the P·V V-transpose |
| `cp.async` (Ampere LDGSTS) | ✅ baseline | fallback load path |
| **TMA `cp.async.bulk.tensor.2d` (tile)** | ✅ **baseline (`SM_90`)** | CCCL: `PTX ISA 80, SM_90`; Prog. Guide Table 29 lists the TMA unit for `12.x`. Only `.multicast::cluster`/`.cta_group` need `a`/`f` targets (not used) |
| **`mbarrier` (`arrive.expect_tx`/`try_wait`)** | ✅ baseline | Prog. Guide Table 29 "async barriers" for `12.x` |
| Warp specialization (producer/consumer) | ✅ software | `setmaxnreg` reg-rebalance wants `sm_120a` (optional) |
| `wgmma` (async warpgroup MMA) | ❌ | Hopper `sm_90a` only |
| `tcgen05` / TMEM (async MMA) | ❌ | datacenter `sm_100a` only |
| block-scaled FP4/FP8 `mma.sync` | ❌ on plain `120` | needs `sm_120a`; irrelevant (we are bf16) |
| Shared memory / block | **99 KB** | Prog. Guide Table 31 (`12.x`) — the binding tiling limit |

Official RTX 5090 ceilings (NVIDIA *RTX Blackwell GPU Architecture* whitepaper) — the exact roofline
denominators used throughout this document:

| Ceiling | Value | Note |
|---|---|---|
| **BF16 tensor-core, FP32 accumulate, dense** | **209.5 TFLOP/s** | the relevant peak: our attention is bf16 with fp32 accumulate. GeForce Blackwell halves FP32-accumulate vs the 419 TFLOP/s FP16-accumulate rate; we do not use 2:4 sparsity (419 TFLOP/s sparse) |
| Memory bandwidth | **1792 GB/s** | 32 GB GDDR7, 512-bit, 28 Gbps |
| L2 cache | **96 MB** | GB202-300 (full GB202 is 128 MB) |
| Shared memory / block | **99 KB** | CC 12.x |
| FP32 non-tensor (reference) | 104.8 TFLOP/s | 21760 CUDA cores @ 2.41 GHz |

**Consequence for the design.** We can build a **TMA-fed, warp-specialized, `mbarrier`-pipelined**
kernel, but because MMA is synchronous `mma.sync`, warp specialization overlaps *loads and softmax*
with MMA across resident warps — it does **not** overlap MMA with MMA (there is no async MMA to
overlap). The ceiling is `mma.sync` throughput with TMA latency and softmax fully hidden.

**Build.** Since the project targets exactly one RTX 5090, compile `sm_120a` (a superset that runs on
the 5090 with zero downside) to unlock `setmaxnreg` and keep FP8/FP4 open. TMA/`mbarrier`/WS
themselves need only `sm_120`. `cuTensorMapEncodeTiled` is a driver-API call (resolve via
`cudaGetDriverEntryPoint`, or link `-lcuda`).

## 4. Roofline model — why this design can reach it

Per attention layer, causal, over `H_q = 24` query heads and `d = 256`:

- **Compute:** `FLOPs ≈ 2·d·H_q·T(T+1) = 12288·T(T+1)` (QKᵀ and P·V each `d·H_q·T(T+1)`). GQA reduces
  K/V *bandwidth*, not compute — all 24 query heads compute full scores.
- **DRAM bytes (L2-perfect):** `≈ T·(2·H_q·d·2 [Q,O] + 2·H_kv·d·2 [K,V]) ≈ 28 KB/token/layer`. The
  96 MB L2 comfortably holds a layer's K/V (16 MB at 4k = 2·4·4096·256·2 B), so the 6× intra-group and
  per-q-block K/V re-reads are absorbed by L2, keeping DRAM traffic near this floor.

Arithmetic intensity is `≈ FLOPs/bytes ∝ T`, so:

| Regime | Bound | Design response |
|---|---|---|
| short `T` (≤~256) | launch/latency + softmax overhead | enough CTAs (heads × q-blocks) to fill SMs; deferred-scale softmax |
| long `T` (≥~1k) | **tensor-core compute** | keep both GEMMs on `mma.sync` at high SOL; hide all K/V load + softmax behind MMA |

**Tensor-core efficiency** here means achieved attention TFLOP/s ÷ the **209.5 TFLOP/s**
bf16/FP32-accumulate peak. "Reach roofline" therefore means, concretely and as the **explicit target
handed to codex**: at `pp4096`, sustain **≥ 80% tensor-core efficiency** (**≈ 168 TFLOP/s** of the
209.5 TFLOP/s peak), with DRAM well under the 1792 GB/s peak; shorter lengths should trend toward the
same efficiency as they cross into the compute-bound regime. The design is roofline-capable because it
(a) removes all non-essential memory traffic (flash/online softmax — S and P never touch HBM; K/V
streamed once from the cache and reused via smem + L2), (b) runs **both** matmuls on tensor cores,
(c) minimizes the
16×-costlier non-matmul FLOPs (`ex2.approx`, deferred normalization, FA2 split-Q with no cross-warp
softmax reduction), (d) overlaps K/V loads with compute via TMA + `mbarrier` + warp specialization,
and (e) exposes every tile/pipeline constant as a free knob so the compute/BW balance can be tuned to
the ceiling at each `T`. What the *architecture* fixes is the absence of asymptotic waste; what codex
tunes is the constant factor to the ceiling.

## 5. Kernel design (the architecture)

FlashAttention-2 structure (split-Q work partitioning, online softmax, causal block-skipping),
adapted to `sm_120`'s synchronous MMA with a TMA + warp-specialized front end (the transferable part
of FlashAttention-3). Reuse the existing `gdn_common.cuh` primitives (`ldmatrix`, `mma`, XOR swizzle,
`exp2_fast`, warp/block reductions) and add a small TMA/`mbarrier` helper layer.

- **Grid & mapping.** One CTA per `(q_head, q_block)`; grid `= 24 × ⌈T/Br⌉`. `kv_head = q_head / 6`.
  Each CTA produces one `Br×256` output tile.
- **Warp roles.** `Br` split across **consumer warps** (each owns 16 Q rows → FA2 split-Q, so each
  warp finishes its rows with **no inter-warp softmax reduction**), plus a **TMA producer** warp that
  streams K/V from the cache.
- **TMA + `mbarrier` pipeline.** K/V blocks are pulled with `cp.async.bulk.tensor.2d` from a per-layer
  cache tensor map (`[d=256, padded_context, kv_head=4]`, head chosen by TMA coordinate, 128 B
  swizzle) into a **multi-stage smem ring**; the producer signals an `arrive.expect_tx` barrier per
  stage and waits on a "buffer-free" barrier before reuse; consumers `try_wait` the stage. This hides
  K/V latency behind compute and offloads address generation + swizzle to hardware (register relief vs
  `cp.async`).
- **GEMM0 `S = QKᵀ`.** `mma.sync.m16n8k16`, contract `d=256` in 16 k-steps. Q and K are both stored
  `[rows, d]` (d contiguous) → native `.row.col` operands, no transpose. `S[Br,Bc]` fp32 in registers.
- **Online softmax (fp32, FA2).** Per owned row: running `m`, `l`; fold the scale into the exponent
  and use `ex2.approx` — `P = exp2((S − m)·(scale·log2e))`; **deferred normalization** (rescale the
  unnormalized `O` by `exp2((m_old − m_new)·…)` only when `m` grows; divide by `l` once in the
  epilogue). Row stats live in the lanes owning the row.
- **GEMM1 `O += P·V`.** Contract over `Bc` (keys), output width `d=256`. V is `[Bc,256]` (d
  contiguous) but must be contracted over `Bc`, so load V fragments with **`ldmatrix.trans`** (the
  standard FA V-transpose; no global transpose).
- **`head_dim=256` register/smem strategy (S1, split-d-out — the recommended default).** The natural
  mapping puts `O = 16×256` fp32 = **128 regs/thread** (spills) and fat double-buffered K/V tiles
  exceed 99 KB. So compute the 256-wide output in **two 128-wide d-chunks**, keeping the softmax
  weights `P` live across both passes (the S-accumulator→P-operand fragment conversion done via
  register shuffles or a small smem scratch) and reusing resident V → **O = 64 regs/thread**.
  Representative, feasibility-checked starting envelope (all tunable, and deliberately near the
  ceiling): `Br=64`, `Bc=32` → Q resident in smem (32 KB) + K/V TMA double-buffered (64 KB) = 96 KB,
  leaving 3 KB for `mbarrier` state and the S→P conversion scratch (99 KB − 96 KB = 3072 B); O
  split-d = 64 regs/thread; ~1–2 CTAs/SM (occupancy is a tuning outcome). Sitting at the 99 KB edge is
  intentional — `Br`/`Bc`/stages are the first knobs §6/§7 move.
- **Causal block-skipping.** Skip K-blocks strictly in the upper triangle; apply the elementwise mask
  only on diagonal blocks; handle `T` not a multiple of `Br`/`Bc` via tail bounds. Roughly halves the
  work and removes off-diagonal masking.
- **Epilogue.** Normalize `O_row /= l_row`, cast to bf16, write `out[256,24,T]` (plain vectorized
  store is fine; TMA store optional).

## 6. Preserved tune space (knobs, not decisions)

The architecture above is fixed; the following are intentionally *open* and expected to move under
`ncu` evidence. None of them changes correctness.

| Knob | Envelope / options | Bound it trades |
|---|---|---|
| `Br`, `Bc`, pipeline stages | fit 99 KB smem, ≤255 regs/thread | occupancy ↔ reuse ↔ latency hiding |
| register strategy | **S1** split-d-out (default) ↔ **S2** O-in-registers/low-occupancy | reg spill ↔ smem ↔ occupancy |
| producer warp count / WS shape | 1 producer ↔ N; or uniform-warp (elect-one TMA, no role split) | scheduling simplicity ↔ overlap |
| GQA K/V reuse | per-`(q_head,q_block)` (L2 reuse) ↔ per-`(kv_head,q_block)` looping the 6 heads on one smem K/V tile | DRAM/smem traffic ↔ parallelism |
| TMA swizzle mode | 32/64/**128 B**, aligned to the `ldmatrix`/MMA read pattern | smem bank conflicts |
| Q load path | TMA (per-launch tensor map) ↔ `cp.async` | simplicity ↔ uniformity |
| `setmaxnreg` | on (`sm_120a`) ↔ off | producer/consumer reg balance |
| output store | plain vectorized ↔ TMA store | epilogue overlap |

Fallbacks are first-class: the **uniform-warp** variant (no producer/consumer split) is the
correctness-safe path if WS proves fragile; **S2** is the path if smem is the binding limiter.

## 7. `ncu`-guided tuning methodology (mandatory)

Tuning is **Nsight Compute-driven**, not guess-driven. The loop: profile the attention kernel at a
representative long length, read the binding limiter from the metrics, change exactly the knob that
addresses it, re-profile. Baselines and per-change deltas are captured under `profiles/`. (See the
repo `profile-cuda` / `ncu-kernel-profile` skills for the mechanics; mirror the `nsys`→`ncu` flow
already used in [`bench/q5090-v3-prefill-length-nsys-kernel-breakdown.md`](bench/q5090-v3-prefill-length-nsys-kernel-breakdown.md).)

Representative capture (single launch, long context):

```bash
ncu --force-overwrite --target-processes all --replay-mode application \
  --set full \
  --kernel-name regex:'gqa_attention_prefill' --launch-count 1 \
  -o profiles/ncu-gqa-prefill/pp4096_gqa_flash \
  ./build/bench/qus_attention_bench -p 4096 ...
# roofline view: add --set roofline for the SM/DRAM roofline chart
```

Limiter → action decision table (the ordering matters — fix the top stall first):

| `ncu` signal | Likely cause | Knob |
|---|---|---|
| tensor-pipe SOL low, `stall_long_scoreboard` high | K/V load-bound | more pipeline stages; larger TMA tiles; GQA K/V reuse (Approach 2) |
| local-memory traffic > 0 / high reg pressure | O accumulator spilling | S1 split-d-out; smaller `Br`; fewer live frags |
| excessive shared wavefronts / bank conflicts | swizzle mismatch | fix TMA swizzle vs `ldmatrix` layout |
| `stall_barrier` / `stall_mio` high | over-synchronized WS | fewer `mbarrier` phases; rebalance producer/consumer; `setmaxnreg` |
| low achieved occupancy *and* low SOL | not enough latency hiding | more stages or more resident CTAs (tile down) |
| high non-matmul issue | softmax overhead | confirm `ex2.approx`, deferred normalize, no cross-warp reduction |

Target exit criterion: at **`pp4096`, ≥ 80% tensor-core efficiency** (achieved TFLOP/s ÷ the
209.5 TFLOP/s bf16/FP32-accumulate peak, i.e. ≈ 168 TFLOP/s), with the tensor/issue pipe the binding
SOL (compute-bound as intended), DRAM well below 1792 GB/s, and `compute-sanitizer` clean.

## 8. Correctness validation (strengthened)

Attention is a numerical CUDA kernel with a clear oracle and real project shapes — an AGENTS.md
hard-whitelist case. The bar is stronger than the current path:

- **Primary oracle.** fp32 reference causal GQA attention on identical `(qn, cached K, V)`, `scale`,
  fp32 softmax. Pass by the normwise `linear_tc`-style criterion from
  [`l1-op-test-standard.md`](l1-op-test-standard.md): **`rel_l2 ≤ 4e-3`** (the bf16-TC precedent),
  and also report max-abs error. bf16 output.
- **Secondary cross-check.** The existing correctness-first kernel (slow but correct) as an
  independent reference on the same inputs.
- **Shape matrix.** `T ∈ {1, 2, 3, 17, 64, 127, 128, 200, 512, 1024, 4096}` plus one near-`max_context`
  length; full `24`/`4` heads; `d=256`. Deliberately includes non-multiples of `Br`/`Bc`, prime-ish
  lengths, and single-block cases to exercise tail/diagonal paths.
- **Edge / stress.** Diagonal-block masking correctness; `T=1`; large-magnitude `Q·K` (softmax
  stability via the `m` subtraction); near-one-hot and near-uniform attention; verification that
  positions in `[T, padded_context)` are never read; determinism (identical output across runs).
- **Sanitizer (mandatory for the async pipeline).** `compute-sanitizer` **`memcheck` + `racecheck` +
  `initcheck`** clean — the `mbarrier`/smem ring and split-d-out reuse are exactly the
  race/lifetime-risk surface AGENTS.md calls out.

## 9. Benchmark (strengthened)

The bench must **correctly reflect long-context attention** on its own, in the terms that matter:
time, achieved bandwidth, and achieved compute. Today's `qus_bench` emits no per-test NVTX (the
`*_nvtx_sum.csv` are empty), so per-length attribution needs separate traces — the strengthening
fixes this and adds a roofline read, mirroring the dual-roofline `bench/linear_op_bench.cu` pattern.

- **Isolate the kernel.** A standalone attention microbench (`bench/gqa_attention_bench` alongside
  `linear_op_bench`) that runs *only* `gqa_attention_prefill` on synthetic cached K/V + Q across a
  length sweep, so `ncu` can target it cleanly and repeatably; **and** per-test **NVTX ranges** added
  to the end-to-end `qus_bench` prefill path for in-situ attribution.
- **Long-context sweep.** `T ∈ {512, 1024, 2048, 4096, 8192, 16384}` (up to `max_context`), causal.
- **Per-`T` report (dual roofline).**
  - latency (ms), per layer and per token;
  - **achieved compute** `TFLOP/s = 2·d·H_q·T(T+1)·(#attn layers) / time`, and **% of the
    209.5 TFLOP/s** bf16/FP32-accumulate peak;
  - **achieved bandwidth**: both the model/DRAM-floor bytes (`≈28 KB/token/layer`) per time **and**
    the `ncu`-measured `dram__bytes` per time, and **% of the 1792 GB/s** peak;
  - the **binding bound** (min of the two) and the resulting **roofline efficiency**;
  - optionally L2 hit-rate / effective K/V reuse to confirm the GQA/L2 assumption.
- **Report schema.** Append-only CSV/JSON columns
  (`T, ms, tflops, tflops_pct, gbps_model, gbps_dram, gbps_pct, bound, ...`) consumable by scripts and
  docs (AGENTS.md whitelists CLI/report schema). Denominators are the official peaks (209.5 TFLOP/s,
  1792 GB/s); a measured stream-copy / `mma.sync` achievable ceiling may be reported as a secondary
  context column.
- **Success signal.** The sweep should show the attention share of prefill collapse from the
  current 49–66% at `pp2048–4096` toward a small fraction, with `pp4096` reaching the **≥80%
  tensor-core-efficiency target** (§4) and shorter lengths trending toward it.

## 10. Integration surface (context, not a task list)

- Kernel: rewrite `src/kernels/kernel/gqa_attention_prefill.cuh`; add a small TMA/`mbarrier` helper
  (e.g. `src/kernels/kernel/tma_common.cuh`); reuse `gdn_common.cuh`.
- Launcher: `src/kernels/launcher/gqa_attention_prefill.cu` (build/pass per-layer cache tensor maps;
  keep the fill kernel).
- Wrapper `src/kernels/wrapper/gqa_attention.cpp` and model `src/model/qwen3_6_27b.cpp`: unchanged —
  the public API and RoPE/fill ordering are stable.
- Build: `CMAKE_CUDA_ARCHITECTURES 120 → 120a`; driver-API access for `cuTensorMapEncodeTiled`.
- Bench/test: `bench/gqa_attention_bench.*`, prefill NVTX, and the parity harness in
  `tests/kernels/`.

## 11. Non-goals

Decode (keep the existing split-KV kernel), non-causal attention, dropout, sliding-window, variable
head dims, and FP8/FP4 attention (a future `sm_120a` accuracy-gated axis). This document does not
prescribe task breakdown, sequencing, or final tile constants — those belong to codex under the
`ncu`-guided loop above.

## 12. Achieved results (2026-07-02)

Implemented. The kernel is `Wc=4`/`Br=64`/`Bc=32`, per-warp-independent with register-resident
online softmax, `head_dim=256` output held in registers, Q cached in registers with the Q staging
buffer aliased to the K/V tiles so two CTAs are resident per SM. Detailed evolution and `ncu`
evidence: [`../profiles/ncu-gqa-prefill/pp4096_perwarp_2cta_delta.md`](../profiles/ncu-gqa-prefill/pp4096_perwarp_2cta_delta.md).

Useful tensor-core efficiency (% of the 209.5 TFLOP/s bf16/FP32-accumulate peak), `qus_gqa_attention_bench`:

| T | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|---:|---:|
| TC % | 20.0 | 39.9 | 55.1 | 70.7 | **80.9** | **84.1** |

Efficiency rises with context to a **~84% asymptote** — the practical fp32-accumulate tensor-pipe
ceiling on sm_120. The `pp4096` target lands at **70.7%** (up from the 36.5% cooperative baseline);
it sits below the asymptote purely from finite-size overhead (per-CTA pipeline ramp + partial waves),
not a structural flaw, and the long contexts where attention actually dominates prefill already clear
80%. The kernel is tensor-pipe bound (ncu: "Tensor is the highest-utilized pipeline", 64.2%),
occupancy-walled at 2 CTAs/SM by the `head_dim=256` O accumulator (128 regs) + cached Q (64 regs).
Closing the residual `pp4096` gap within fp32 accumulate would require costly causal load-balancing;
`fp16`-accumulate PV (~2× tensor throughput) is a separate lower-precision datapath, intentionally
left out of scope. Correctness `rel_l2 ≤ 1.85e-3`; `compute-sanitizer` memcheck + racecheck clean.

End-to-end full-model prefill (`qus_bench`) vs the scalar-attention baseline: `pp4096` **848 → 2730
tok/s (3.2×)**, `pp2048` 1243 → 2676 (2.15×), `pp1024` 1580 → 2568 (1.63×). Attention is no longer
the prefill bottleneck; the low-bit linear GEMM now dominates.
