# Prefill Tensor-Core GEMM (P2) — implementation plan

> Status: executed 2026-07-01 (ncu-driven roofline pushdown complete). Design:
> [`../2026-07-01-prefill-linear-foundation-design.md`](../2026-07-01-prefill-linear-foundation-design.md) §7.

## Goal

Add a tensor-core GEMM for the LargeT regime so prefill linear (nsys: ~97% of prefill, ~4-6% of the
~220 TFLOP/s ceiling) can approach the tensor-core roofline. `out[N,T] = W[N,K] . x[K,T]`, W is
Q4/Q5/Q6 row-split, x/out bf16.

## Non-goals

New weight layout / repack / second copy; CUTLASS; FP8/FP4 activations; epilogue fusion; changes to
decode or the SmallT multi-step path.

## Execution mode

Single-agent, direct sequential; strict review for this high-risk CUDA/numerical kernel.

## Status

- **Done (correct + validated).**
  - `src/kernels/linear/gemm/linear_rowsplit_gemm_mma.{cuh,cu}`: generic (Q4/Q5/Q6) bf16
    `mma.sync.aligned.m16n8k16` GEMM with on-chip low-bit dequant (identical `Codec::load_pair`
    math), fp32 accumulate, N/T/K tails.
  - Seam: `RowsplitLowbitGemmMma` policy (`uses_tensor_cores=true`); `resolve_plan` routes low-bit
    LargeT (`T>tau`, `tau=16`) to it; SmallT stays multi-step; decode/dense unchanged.
  - Numerics methodology: `Tolerance::linear_tc` normwise (`rel_l2<=4e-3`) parity preset for the TC
    path, documented in [`../l1-op-test-standard.md`](../l1-op-test-standard.md) §1.3. Full
    `qus_linear_test` prefill matrix green; `compute-sanitizer memcheck` + `racecheck` clean.
- **Perf bar met (ncu-driven pushdown).** The correctness-first kernel was rebuilt into a tiled GEMM
  behind a compile-time `GemmCfg<BM,BN,BK,WM,WN,STAGES,MIN_BLOCKS>` (default `64x128x64/32x32`,
  `STAGES=2`): warp-per-row coalesced dequant into a swizzled shared bf16 tile, `ldmatrix` A/B
  fragments, register-double-buffered mma, and a `cp.async` double-buffered mainloop that prefetches
  the next K-tile's raw quant + x while the current tile computes. On the measured ~220 TFLOP/s bf16
  `mma.sync` ceiling this reaches **68.2% (Q4 gate/up) and 65.9% (Q5 mlp_down) at T=512, and
  65-74% across every dominant shape at T=2048** (from ~10-14% for the correctness-first kernel).
  The binding limiter is now the tensor/issue pipe (ncu SM SOL ~65%, DRAM ~8%, achieved occupancy
  ~32% -- the software pipeline hides latency at low occupancy), i.e. compute-bound as intended.

## Result (roofline pushdown, done)

- Kernel: [`../../src/kernels/linear/gemm/linear_rowsplit_gemm_mma.cuh`](../../src/kernels/linear/gemm/linear_rowsplit_gemm_mma.cuh)
  (cp.async pipeline + `ldmatrix` + XOR swizzle + coalesced dequant), launcher config table +
  `QUS_GEMM_CFG` sweep override.
- `tau` recalibrated 64 -> 16 from the T-swept bench: the mma GEMM overtakes the multi-step GEMV at
  T~16 (T<=8 its BN-wide token tile is mostly empty; from T=32 it is ~3-6x faster). Baseline refreshed
  at `profiles/prefill-linear-foundation/baseline_p2.csv` (`baseline_p2_pre.csv` = before).
- e2e prefill (long_2k, 7932 prompt tokens): **54.6 s -> 13.7 s (~4x), 145 -> 576 prompt tok/s**
  (multi-step vs mma on the same prompt; attention/GDN are unchanged and grow with T at 7932).
- Correctness: `qus_linear_test` (LargeT judged by `linear_tc`) green; `compute-sanitizer memcheck`
  and `racecheck` clean.

## Verification commands

```
cmake --build build -j
./build/tests/qus_linear_test
compute-sanitizer --tool memcheck ./build/tests/qus_linear_test
compute-sanitizer --tool racecheck ./build/bench/qus_linear_op_bench --shape GdnInQK4096x5120 --qtype Q4 --t-sweep 128 --warmup 0 --repeat 1 --copy-repeat 1
./build/bench/qus_linear_op_bench --t-sweep 1,8,64,128,512,2048 --csv-out profiles/prefill-linear-foundation/baseline_p2.csv
# e2e prefill (LargeT): mma vs the pre-P2 multi-step path on the same 7932-tok prompt
./build/bench/qus_e2e_bench --weights out/qwen3_6_27b.q5090_w4g64_mixed_v3.qus --output-json profiles/prefill-linear-foundation/e2e_long_p2_after.json --case long_2k:bench/fixtures/prompts/long_2k.ids:1 --max-ctx 8192 --warmup-repeats 1 --repeats 3
```
