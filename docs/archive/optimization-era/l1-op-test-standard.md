# L1 Op Test & Bench Standard — qwen3.6-ultraspeed

> Status: standard (frozen). Date: 2026-06-26.
> Scope: the **correctness** and **performance** standards every L1 operator must meet, and the
> frozen test/bench framework that enforces them. This is the single source of truth for "is this
> kernel correct?" and "is this kernel fast enough?". Implemented by
> [`tests/kernels/op_check.h`](../tests/kernels/op_check.h),
> [`tests/kernels/op_tester.h`](../tests/kernels/op_tester.h), and
> [`bench/qus_bench_common.h`](../bench/qus_bench_common.h).
>
> Adapted from the proven error/bench framework in `~/chunked_gdn` (`tests/test_utils.h`,
> `bench/bench_common.h`). See [`l1-kernel-layering.md`](l1-kernel-layering.md) and
> [`l1-operator-catalog.md`](l1-operator-catalog.md).

---

## 0. The governing rule (anti-cheat)

**If a test fails, fix the kernel — not the tolerance, not the inputs, not the metric.** Tolerances
and input distributions live only in the frozen framework (`op_check.h` / `op_tester.h`); a per-op
test selects a *preset by name* and supplies inputs from the *frozen generators*. It may not pass
custom tolerance numbers. The performance gate is an **ncu-measured** number, independent of anything
the kernel binary prints.

Corollary: **kernel math may not be approximated to win speed.** Transcendentals (`silu`, `sigmoid`,
`exp`, `softplus`, `rsqrt`) must match fp32 reference math; polynomial/“fast” fits that diverge
outside a fitting range are forbidden. Bandwidth is won by *memory-access* optimization
(vectorization, coalescing, occupancy), never by computing something cheaper and wrong.

---

## 1. Correctness standard

### 1.1 The oracle: an fp64 CPU golden from bf16-rounded inputs
1. Generate fp32 inputs from a fixed seed (Section 1.4).
2. **Round them to bf16** (what the kernel will actually read), then keep that rounded value.
3. Compute the reference in **fp64 (`double`)** from those rounded inputs.
4. Run the kernel; upcast its (bf16/fp32) output to `double`.
5. Compare kernel-output-as-double vs the fp64 reference.

This isolates the *kernel's* error (its math + its bf16 output rounding) from input-rounding noise.

### 1.2 The composite pass criterion (`op_check.h`)
Per element `i`, the bound is `atol + rtol*|ref_i|` (numpy/torch allclose convention). Then:

- **NaN is always fatal** (any NaN in output or reference → FAIL).
- **PASS iff** `no NaN` AND either:
  1. `n_violating == 0` (strict), or
  2. the **tail channel**, requiring *all three*:
     - `n_violating / n     <= tail_frac`     (frequency cap),
     - `worst_violation_ratio <= worst_ratio_max` (magnitude cap, `max |a-b| / bound`),
     - `rel_l2 = ||a-b||_2 / ||b||_2 <= rel_l2_tol` (global energy cap).

The three-way AND is the anti-gaming property: many tiny violations fail the energy cap; a few huge
ones fail the magnitude/energy caps. `rel_l2` is the standard relative residual, robust to outliers
in a way per-element allclose is not.

### 1.3 Frozen presets (`static constexpr` in `op_check.h`; selected by op class)
Order: `atol, rtol, tail_frac, worst_ratio_max, rel_l2_tol`.

- `bf16_elementwise` — `residual_add`, `sigmoid_mul`, `silu_mul`, `rope`, `embedding`:
  `1e-3, 8e-3, 1e-3, 4.0, 4e-3`. (One bf16 output rounding ≈ `2^-8`; `rtol=2^-7` gives 2x headroom.)
- `bf16_reduction` — `rmsnorm`, `l2norm`, `causal_conv1d`:
  `2e-3, 1.6e-2, 2e-3, 5.0, 8e-3`. (fp32 reduction + bf16 round; GPU vs CPU accumulation order differs.)
- `fp32_transcendental` — `gdn_gating` (`g`,`beta` are fp32): `1e-6, 1e-5, 1e-4, 2.0, 1e-5`.
  (libm vs CUDA `expf`/`softplus`/`sigmoid` differ by a few ULP.)
- `linear_bf16` — dense/quant `linear` and model linear parity (fp32-compute paths: decode
  GEMV, the SmallT multi-step GEMV, dense): `2e-3, 1.6e-2, 2e-3, 5.0, 8e-3`.
- `linear_tc` — **tensor-core (low-precision-compute) linear** (the LargeT `mma.sync` GEMM):
  `2e-3, 1.6e-2, 1.0, 1e30, 4e-3`. The golden stays the same fp64-from-bf16-inputs oracle;
  only the *pass criterion* changes to the **normwise relative residual** `rel_l2` (tightened to
  `4e-3`, roughly 2x the observed ~2e-3) with NaN/inf still fatal. The per-element worst/frequency
  caps are neutralized (`tail_frac=1`, `worst_ratio_max=1e30`) because a low-precision GEMM has
  large *relative* error on near-zero cancellation outputs even when the matmul is correct — that
  is intrinsic, not a bug. Normwise error is the standard GEMM/BLAS correctness metric and remains a
  strong bug net (any layout/index bug spikes `rel_l2`). This is **not** tolerance-gaming under §0:
  the golden remains maximum-precision and we select the criterion that matches the compute class,
  rather than loosening a criterion to hide a kernel error. Rationale: bf16/tf32 mma operands round
  the dequantized weight (the reference keeps it fp32); the normwise error is identical (~2e-3) to
  the fp32 multi-step path, so the tensor-core GEMM is as accurate, and bf16 mma (2x tf32
  throughput) is retained.
- `attention_bf16` — GQA attention and attention block parity:
  `2e-3, 1.6e-2, 2e-3, 5.0, 8e-3`.
- `gdn_output_bf16` — GDN recurrent/chunked BF16 output:
  `1e-3, 1.0e-2, 2e-3, 5.0, 8e-3`.
- `gdn_state_fp32` — GDN recurrent/chunked FP32 state:
  `5e-4, 5.0e-3, 2e-2, 5.0, 5e-3`.
- `argmax` — **exact index match**, lowest-index tie-break; no tolerance preset.

### 1.4 Honest, seeded inputs (`op_tester.h`)
- Deterministic: fixed seeds; each op runs **>= 3 seeds**.
- Distributions reflect the op's **real operating range**, not a convenient `[-1,1]`. Defaults:
  hidden states / projection outputs in `[-8, 8]`; q/k for `l2norm` L2-normalized per head; `gdn`
  `a`/`b` spanning the softplus guard (`[-20, 25]` for one case); `rope` positions up to a few K;
  `argmax` logits including negatives and a deliberate tie case.
- Each op also runs a **stress variant** (large magnitudes / adversarial values) that would expose a
  range-limited approximation.

### 1.5 Coverage (the `qus_<op>_test` executable)
Test = correctness + coverage only (no timing). The shape matrix per op MUST include:
- decode `T=1` and prefill (`T` up to a few K),
- at least one **unaligned** size (not a multiple of the warp / vector width),
- every variant view the op supports (e.g. `rmsnorm` over `[5120,T]`, `[256,H,T]`, `[128,48,T]`),
- the stress-input case.
`compute-sanitizer` (memcheck + racecheck) clean is part of correctness.

---

## 2. Performance standard

### 2.1 Separate `qus_<op>_bench` executable on real shapes
Performance is measured by a **separate** binary that runs the op at the **actual Qwen3.6-27B
shapes** for that op — decode `T=1` is the primary scenario (batch=1 autoregression), plus a
representative prefill `T`. Bench bytes are the op's real HBM traffic. The in-process GB/s
(median-based, `qus_bench_common.h`) is a convenience readout, **not** the acceptance gate.

### 2.2 Profiling-first is MANDATORY
The order is fixed and non-negotiable:
1. Build Release (`-lineinfo` is on).
2. Run the bench once to confirm it works; **do not** start editing the kernel.
3. **Profile with `ncu`** and read the Memory Workload / Speed-of-Light / roofline sections to
   identify the bottleneck (DRAM-bound? uncoalesced? low occupancy? tail effect?).
4. Make ONE targeted change addressed at the identified bottleneck.
5. **Re-profile** to confirm the bottleneck moved. Repeat.

"Tweak the kernel and re-run the bench, hoping GB/s goes up" is explicitly the wrong loop — it gives
no causal signal and tempts math approximation. Profiling tells you *why*, which is required before
*how*.

### 2.3 Acceptance gate
- `ncu` `dram__throughput.avg.pct_of_peak_sustained_elapsed` **>= 85%** at the real prefill shape
  (memory-bound Tier-1 ops). Record the `.ncu-rep` path under `profiles/`.
- Correctness re-verified (the `qus_<op>_test` still passes) and `compute-sanitizer` clean.
- Record a **device copy-kernel baseline** (a trivial `out=in` kernel moving the same bytes) as the
  honest achievable ceiling for context.

### 2.4 ncu procedure (inline; no external skill needed)
```bash
# permissions (once): if ncu prints ERR_NVGPUCTRPERM, enable counters:
#   sudo bash -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-profiler.conf' && sudo reboot
mkdir -p profiles
ncu --set roofline --force-overwrite \
    --kernel-name regex:'<op>' --launch-skip 20 --launch-count 1 \
    -o profiles/<op> ./build/bench/qus_<op>_bench --prefill
ncu --import profiles/<op>.ncu-rep --csv | grep -i 'dram__throughput.avg.pct_of_peak_sustained_elapsed'
```
(`--launch-skip` must be < the bench's total launches, else ncu profiles nothing and writes no file.)

---

## 3. Why this prevents the failure we saw

The first Task-1 agent replaced `silu` with a quadratic fit to gain bandwidth and never profiled.
Under this standard that is caught three ways: (1) the honest `[-8,8]` + stress inputs make the fit
violate `bf16_elementwise` (§1.3/§1.4); (2) the "no math approximation" rule (§0) forbids it; (3) the
perf gate is ncu DRAM %, so there is nothing to gain by computing a cheaper wrong value (§2.3).
