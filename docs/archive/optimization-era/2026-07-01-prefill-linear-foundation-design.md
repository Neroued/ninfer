# Prefill Linear Foundation — methodology / design

> Status: design (implemented in the same session). Date: 2026-07-01.
> Scope: the methodology for optimizing the Qwen3.6-27B **prefill** path, and the concrete
> foundation phase (a usable multi-step linear kernel + dispatch seam + bench/test 基建) that the
> later prefill kernel work builds on. Decode is out of scope and left byte-for-byte unchanged.

## 1. Context and motivation

Decode reached ~81 tok/s (ahead of llama.cpp q4_k_m ~74 tok/s) and is initially satisfactory. The
focus now moves to prefill, which was catastrophically slow (20-token prefill ~1.46 s vs llama.cpp
`pp512 ≈ 3561 t/s`). The cause was purely kernel quality: the T>1 low-bit `linear` path used a naive
one-thread-per-output GEMM that **re-dequantized each weight row for every token** (O(N·K·T) weight
+ dequant traffic, no reuse, no tensor cores). Prefill orchestration is already "wide" (the schedule
runs every op on the full `[D, T]` activation), so this is a kernel problem, not a scheduling one.

The lever is `linear` (the dominant prefill cost). Element-wise ops are deferred; GQA and GDN are
their own later phases (GDN already has a chunked prefill algorithm; GQA needs a flash prefill).

## 2. Locked constraints

1. **Single weight copy; decode owns the layout.** Model weights exist as one row-split copy in
   VRAM (~13-20 GB of 32 GB). The decode GEMV is already tuned against that layout (coalesced
   128-bit plane loads, cp.async pipeline). We do **not** duplicate weights (a second layout would
   ~double weight VRAM) and do **not** re-lay-out for prefill (a Marlin-style `ldmatrix` shuffle
   helps MMA but pessimizes the memory-bound decode streaming pattern). **Prefill adapts to the
   storage on-chip; storage never adapts to prefill.**
2. Hand-rolled `mma.sync` only — no CUTLASS.
3. W4/W5/W6 weights with bf16 activations (A16) only. FP8/FP4 activations are a later, accuracy-gated
   axis.
4. Foundation before optimization.

## 3. Roofline model — one layout, three consumers

For W·A16, weight bytes are ~invariant to `T` (each weight tile is loaded once and reused across the
`T` GEMM columns), while compute grows as `2·N·K·T`. So arithmetic intensity rises ~linearly with
`T`, giving two crossovers and three on-chip strategies over the **same** DRAM bytes:

| Regime | T | Bound | On-chip strategy |
|---|---|---|---|
| T1 | =1 | DRAM bandwidth | existing per-family / generic row-split GEMV (decode) |
| SmallT | 2‥τ | DRAM bandwidth | multi-step GEMV: dequant a weight tile once, reuse across T columns (CUDA cores) |
| LargeT | >τ | tensor-core compute | dequant→shared→`mma` tiled GEMM (P2, not yet built) |

`τ(format, shape)` is a per-shape crossover to be calibrated from the bench once the LargeT MMA GEMM
exists. On the RTX 5090 the measured ceilings are ~1.5 TB/s (cold stream-copy) and ~220 TFLOP/s
(dense bf16 `mma.sync` probe); for the big MLP matrices the memory floor is ~tens of µs, so the
crossover is on the order of tens of tokens.

## 4. What the foundation phase delivered

- **Multi-step (T>1) row-split GEMV** (`src/kernels/linear/gemv/linear_rowsplit_gemm_multistep.*`):
  warp-per-row; each lane owns a group value pair; `blockIdx.y` selects a tile of `kTt=8` activation
  columns; each K-group's weights are dequantized once (`Codec::load_pair`, math identical to
  `Codec::load_group`) and reused across the column tile; fp32 accumulate; bf16 out. This is the
  universal T>1 low-bit path for **every** shape (registered families and the unregistered prefill
  projection shapes such as gate/up `17408x5120`, `in_q/in_k 2048x5120`, `k/v 1024x5120`). It is
  correctness-first: still CUDA-core and it re-streams weights once per column tile, so it is
  memory-bound at large T — usable for testing/benchmarking, with tensor cores and a bandwidth
  pushdown as follow-ons. The old naive GEMM was removed (superseded).
- **Regime seam** (`linear_plan.{h,cpp}`, `linear.cpp`): `classify_regime` now has a real
  `T1/SmallT/LargeT` split via `regime_threshold(format, shape)` (placeholder τ; SmallT and LargeT
  both currently route to the multi-step kernel, so the split is latent until P2). The dead
  `GenericLowbitGemm` policy was replaced by `RowsplitLowbitGemmMultistep`
  (`plan_id = linear.rowsplit.gemm.multistep.v1`, `uses_tensor_cores=false`). T1 (decode) routing is
  unchanged.
- **T-swept dual-roofline bench** (`bench/linear_op_bench.cu`): `--t-sweep` over
  `{1..2048}`; per `(shape, qtype, T)` reports achieved GB/s vs a measured stream-copy ceiling and
  achieved TFLOP/s vs a measured dense-bf16 `mma.sync` ceiling, cold (operationally realistic —
  every layer has distinct weights) and warm (intra-call L2 reuse). CSV columns are append-only.
  Baseline snapshot: `profiles/prefill-linear-foundation/baseline.csv` (confirms the multi-step
  kernel sits at ~2-6% of the TC ceiling — the headroom P2 will close).
- **Prefill parity** (`tests/kernels/test_linear.cpp`): the frozen fp64-golden `linear_bf16`
  standard extended across the registered + real prefill shapes × codecs and a prefill T matrix
  (2‥2048, column-tile boundaries around `kTt=8`, unaligned N/K, stress inputs). `compute-sanitizer
  memcheck` clean.

## 5. Roadmap (follow-on phases)

- **SmallT roofline pushdown**: `ncu`-driven optimization of the multi-step GEMV toward the
  weight-bandwidth ceiling (vectorized/pipelined weight loads).
- **P2 — LargeT tensor-core GEMM**: dequant→shared→`mma` on the existing row-split layout; flip the
  LargeT policy + `uses_tensor_cores`, calibrate `τ` from the bench.
- **GQA flash prefill**; **GDN chunked tuning**; **element-wise / fusion** — each reusing the T-swept
  bench and the parity harness established here.

## 6. Verification

- Build: `cmake --build build` (core, bench, tests) — clean.
- Parity: `./build/tests/qus_linear_test` → `OK linear correctness`; `compute-sanitizer --tool
  memcheck` → `0 errors`.
- Bench: `./build/bench/qus_linear_op_bench --csv-out profiles/prefill-linear-foundation/baseline.csv`
  → 96 rows (8 targets × 12 T), both ceilings measured.
- Decode unchanged: T1 kernels/launchers untouched; T1 routing identical.

## 7. P2 — LargeT tensor-core GEMM (status)

Measured checkpoint after the foundation: e2e prefill went from ~1.46 s / 20 tok (naive GEMM) to
0.18 s / 26 tok and 3.72 s / 705 tok; an nsys breakdown at 705 tok showed **linear ~97%** of prefill
(Q5 51.7% + Q4 45.0%), attention 2.4%, everything else <1%, with the multi-step kernel at only
~4-6% of the measured ~220 TFLOP/s tensor-core ceiling. So P2 adds a tensor-core GEMM for the LargeT
regime (`linear_rowsplit_gemm_mma.{cuh,cu}`, `RowsplitLowbitGemmMma`, `uses_tensor_cores=true`).

- **Numerics.** bf16 `mma.sync.aligned.m16n8k16` with on-chip low-bit dequant (identical
  `Codec::load_pair` math), fp32 accumulate. Parity is judged by the normwise `linear_tc` criterion
  (see `l1-op-test-standard.md` §1.3): the fp64 golden is unchanged, but the pass metric is
  `rel_l2` (which is identical, ~2e-3, to the fp32 multi-step path) rather than a per-element
  worst-case cap that mis-fires on near-zero cancellation outputs. bf16 (not tf32) is kept because,
  under the correct criterion, it passes and is ~2x faster. compute-sanitizer clean.
- **Perf status (done, ncu-driven).** The correctness-first kernel was rebuilt into a tiled GEMM
  behind a compile-time `GemmCfg<BM,BN,BK,WM,WN,STAGES,MIN_BLOCKS>` (default `64x128x64/32x32`,
  `STAGES=2`). The pushdown, in the order ncu evidence dictated: (1) warp-per-row **coalesced**
  dequant (fixed 87% excessive global sectors); (2) **`ldmatrix`** A/B fragments + an XOR **swizzle**
  of the shared bf16 tiles (fixed 83% excessive shared wavefronts / bank conflicts); (3) occupancy
  via `__launch_bounds__` min-blocks; (4) a **`cp.async` double-buffered** mainloop that prefetches
  the next K-tile's raw quant + x while the current tile dequants + multiplies (this was the big one:
  it removed the dominant global-load `long_scoreboard` stall). Result on the measured ~220 TFLOP/s
  bf16 `mma.sync` ceiling: **T=512 -> Q4 gate/up 68.2%, Q5 mlp_down 65.9%, Q5 out 62.8%, Q6 lm_head
  60.2%; T=2048 -> 65-74% on every dominant shape** (from ~10-14% for the correctness-first kernel).
  The binding limiter is now the tensor/issue pipe (ncu SM SOL ~65%, DRAM ~8%, occupancy ~32% -- the
  pipeline hides latency at low occupancy), i.e. compute-bound as intended.
- **e2e + `tau`.** `tau` recalibrated 64 -> 16 (mma overtakes the multi-step GEMV at T~16). e2e
  prefill on a 7932-token prompt (`long_2k`): **54.6 s -> 13.7 s (~4x), 145 -> 576 prompt tok/s**
  (multi-step vs mma, same prompt; attention/GDN unchanged). `qus_linear_test` (`linear_tc`) green,
  `compute-sanitizer memcheck` + `racecheck` clean. Details: [`plans/2026-07-01-prefill-tc-gemm.md`](plans/2026-07-01-prefill-tc-gemm.md).
