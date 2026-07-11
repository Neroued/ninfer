# Prefill Linear Foundation — implementation plan

> Status: executed 2026-07-01. Design: [`../2026-07-01-prefill-linear-foundation-design.md`](../2026-07-01-prefill-linear-foundation-design.md).

## Goal

Give prefill a usable low-bit `linear` kernel and the framework/bench/test 基建 to optimize it:
a multi-step (T>1) row-split GEMV, a real T-regime dispatch seam, a T-swept dual-roofline bench, and
prefill parity coverage. Behavior-preserving for decode (T=1).

## Non-goals

- LargeT tensor-core MMA GEMM (follow-on P2); LargeT temporarily reuses the multi-step GEMV.
- GQA/GDN kernels or scaffolding; element-wise/fusion work.
- New weight layout / repack / second copy; CUTLASS; FP8/FP4 activations.
- Any change to the decode (T=1) path or to numerical results.

## Execution mode

Single-agent, direct sequential implementation — not subagent-driven (the AGENTS default is
overridden here at the user's explicit request; the scope is bounded and the decode path is
untouched). Sequence: T1 → T2 → {T3, T4} → T5.

## Locked constraints

Single weight copy; decode owns the row-split layout and prefill adapts on-chip (never re-layout or
duplicate). Hand-rolled `mma.sync` only. W4/W5/W6 with bf16 activations. Foundation before
optimization.

## Tasks (all complete)

- **T1 — Multi-step (T>1) row-split GEMV kernels.** `Codec::load_pair` on Q4/Q5/Q6
  (`src/kernels/linear/codec/linear_codec.cuh`) + `linear_rowsplit_gemm_multistep.{cuh,cu}`
  (`src/kernels/linear/gemv/`). Warp-per-row, `kTt=8` column tile, dequant once per K-group and reuse
  across the tile, fp32 accumulate; math identical to `Codec::load_group`. T=1 decode kernels
  untouched.
- **T2 — Regime seam.** `linear_plan.{h,cpp}` + `linear.cpp`: `regime_threshold(format, shape)` and a
  real `T1/SmallT/LargeT` split; `RowsplitLowbitGemmMultistep` policy for all low-bit T>1 shapes; the
  naive `GenericLowbitGemm` removed (superseded). `uses_tensor_cores=false` until P2.
- **T3 — T-swept dual-roofline bench.** `bench/linear_op_bench.cu`: `--t-sweep`, cold+warm,
  measured stream-copy + dense-bf16 `mma.sync` ceilings, append-only CSV columns
  (`T, achieved_tflops, tc_ceiling_tflops, tc_pct, warm_median_us`); baseline at
  `profiles/prefill-linear-foundation/baseline.csv`.
- **T4 — Prefill parity coverage.** `tests/kernels/test_linear.cpp`: registered + real prefill shapes
  × codecs across a prefill T matrix (2‥2048, column-tile boundaries, unaligned N/K, stress), frozen
  `linear_bf16` fp64 golden; `compute-sanitizer memcheck` clean.
- **T5 — Integration.** This plan + the design doc; self-review; full build/test/bench verification.

## Definition of done

- Multi-step GEMV covers every low-bit shape; naive GEMM removed; decode T=1 unchanged.
- Seam routes T1/SmallT/LargeT correctly; `qus_linear_test` green (no regression).
- Bench emits cold+warm dual-roofline CSV (8 targets × 12 T); baseline recorded.
- Prefill parity green; `compute-sanitizer` clean.

## Verification commands

```
cmake --build build -j
./build/tests/qus_linear_test
compute-sanitizer --tool memcheck ./build/tests/qus_linear_test
./build/bench/qus_linear_op_bench --csv-out profiles/prefill-linear-foundation/baseline.csv
```

## Review

Focused self-review (low-risk, behavior-preserving seam + additive bench/test). Independent review
is reserved for the higher-risk P2 tensor-core GEMM plan.

## Follow-on plans

SmallT roofline pushdown; P2 LargeT tensor-core GEMM (calibrate `τ`); GQA flash prefill; GDN chunked
tuning; element-wise/fusion — each reusing the T3 bench and T4 parity harness.
