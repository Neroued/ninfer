# L1 Tier-1 Simple Ops — Implementation Plan (Codex-self-contained)

> This plan is self-contained: it does NOT rely on any Cursor/Claude "skill". Everything an
> executor needs (the subagent workflow, the test framework, the ncu procedure) is described inline
> or in linked repo files. Read this whole file, plus the **frozen framework** and **standard**:
> [`docs/l1-op-test-standard.md`](../l1-op-test-standard.md),
> [`tests/kernels/op_check.h`](../../tests/kernels/op_check.h),
> [`tests/kernels/op_tester.h`](../../tests/kernels/op_tester.h),
> [`bench/qus_bench_common.h`](../../bench/qus_bench_common.h),
> [`docs/l1-kernel-layering.md`](../l1-kernel-layering.md),
> [`docs/l1-operator-catalog.md`](../l1-operator-catalog.md).

**Goal:** Implement the 9 remaining Tier-1 (memory-bound) L1 ops to the frozen standard. The
test/bench framework and `silu_and_mul` already exist (commit `ef7a19a`); `silu_and_mul` still needs
its performance task.

**Architecture:** Each op follows the four-layer L1 layout (`l1-kernel-layering.md`): api header +
`wrapper/<op>.cpp` (validate + dispatch) + `launcher/<op>[_<v>].{h,cu}` + `kernel/<op>[_<v>].cuh`.
Copy [`src/kernels/*/silu_and_mul.*`](../../src/kernels/wrapper/silu_and_mul.cpp) as the template.
**Correctness and performance are SEPARATE tasks** for every op.

**Tech Stack:** C++20, CUDA 13.1 (sm_120), CMake ≥ 3.28, gcc 13.3. Build dir `build/`.

---

## How to execute (inline subagent workflow — no external skills)

The controller (you) drives this loop; do not pollute one subagent's context with another's.

1. Pick the next unchecked task. Dispatch **one implementer subagent** with: the full task text, its
   **Reading list**, and the **Subagent instructions** below. Never have the subagent read this whole
   plan — give it exactly its task + reading list.
2. The implementer follows **TDD**: write the failing test first, build, confirm it fails, implement,
   build, confirm it passes, run `compute-sanitizer`, self-review, then commit.
3. Dispatch a **spec-compliance reviewer** subagent: does the code match the task spec (correct CPU
   ref, correct signature, the assigned preset, the required shapes), nothing missing/extra? It MUST
   verify the CPU reference matches the math in `l1-operator-catalog.md` and that the test uses the
   frozen preset unchanged. Fix-and-re-review until clean.
4. Dispatch a **code-quality reviewer** subagent (only after spec is clean). Fix-and-re-review until
   approved.
5. Mark the task done. Repeat.

**Hard rules (anti-cheat, from `docs/l1-op-test-standard.md` §0):**
- Do NOT modify the frozen framework (`op_check.h`, `op_tester.h`, `qus_bench_common.h`) or any
  tolerance preset. If a test fails, fix the kernel.
- Do NOT approximate kernel math to gain speed (no polynomial/"fast" transcendentals that diverge).
- Work on `master`, one commit per task, with the commit message given in the task.

**Build / test commands:**
```bash
cmake -S . -B build && cmake --build build -j --target qus_<op>_test     # correctness
ctest --test-dir build -R qus_<op>_test --output-on-failure
compute-sanitizer ./build/tests/qus_<op>_test                            # must be clean
cmake --build build -j --target qus_<op>_bench                           # performance
```

## ncu procedure (inline — performance tasks MUST profile before optimizing)
```bash
# one-time permission fix if ncu prints ERR_NVGPUCTRPERM:
#   sudo bash -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-profiler.conf' && sudo reboot
mkdir -p profiles
ncu --set roofline --force-overwrite --kernel-name regex:'<op>' \
    --launch-skip 20 --launch-count 1 -o profiles/<op> ./build/bench/qus_<op>_bench --prefill
ncu --import profiles/<op>.ncu-rep --csv | grep -i 'dram__throughput.avg.pct_of_peak_sustained_elapsed'
```
Read the Memory Workload / Speed-of-Light section to find the bottleneck, make ONE targeted change,
re-profile. `--launch-skip` must be < the bench's total launches.

---

## Op layout (every correctness task)
Four files, copying `silu_and_mul.*`:
- `include/qus/kernels/<op>.h` — api decl(s) (catalog signature, `(inputs…, params…, out, stream)`).
- `src/kernels/wrapper/<op>.cpp` — validate (dtype/shape/contiguity/null) + dispatch.
- `src/kernels/launcher/<op>.h` + `<op>[_<v>].cu` — grid/block + launch + `CUDA_CHECK`.
- `src/kernels/kernel/<op>[_<v>].cuh` — grid-stride `__global__`, fp32 math, vectorize bf16x2.
- `tests/kernels/test_<op>.cpp` registered in `tests/CMakeLists.txt`; `bench/<op>_bench.cu` +
  `qus_add_bench(...)` in `bench/CMakeLists.txt`.

Correctness test skeleton (uses the frozen framework; mirrors `tests/kernels/test_silu_and_mul.cpp`):
generate fp32 honest inputs → `round_to_bf16` → compute the **fp64** CPU ref from the rounded inputs →
upload bf16, run the op, download → `verify(tag, got, ref, Tolerance::<preset>())`. Sweep the shapes,
≥3 seeds, plus the stress case. `compute-sanitizer` clean.

---

## Tasks

Order: simplest first. Each op = an **-A (correctness)** task then a **-B (performance)** task.
`silu_and_mul` has only a -B (its correctness is committed). Shapes are `ne`-order (`ne[0]` fastest).

### Task silu_and_mul-B — performance
- **Reading list:** `docs/l1-op-test-standard.md` §2, `bench/silu_and_mul_bench.cu`, `bench/qus_bench_common.h`, this file's ncu procedure.
- **Subagent instructions:** the bench exists (`[17408,1]` decode, `[17408,4096]` prefill). Profile with ncu FIRST, find the bottleneck, optimize `src/kernels/kernel/silu_and_mul.cuh` and/or the launcher grid sizing, re-profile. DoD: ncu sustained DRAM ≥ 85% at prefill; `qus_silu_and_mul_test` still PASS; sanitizer clean. Commit `perf(kernels): silu_and_mul to DRAM roofline`.

### Task residual_add-A — correctness
- **Reading list:** `l1-operator-catalog.md` §3.4, `silu_and_mul.*` (template), `op_tester.h`, `op_check.h`.
- **api:** `void residual_add(const Tensor& y, Tensor& x, cudaStream_t stream);` (in place: `x += y`).
- **CPU ref (fp64):** `x[i] = double(x[i]) + double(y[i])`.
- **preset:** `bf16_elementwise`.
- **kernel:** grid-stride bf16x2; `x2[j] = __hadd2`-equivalent via fp32 add then `__floats2bfloat162_rn`.
- **validation:** `x`/`y` BF16, identical `ne[]`, contiguous; short-circuit `numel()==0`.
- **shapes:** `[5120,1]`, `[5120,4096]`, `[6144,1]`, `[123457]` (unaligned); ≥3 seeds, range `[-8,8]`; stress `[-60,60]`.
- **commit:** `feat(kernels): residual_add (correctness)`.

### Task residual_add-B — performance
- **Reading list:** ncu procedure; `qus_bench_common.h`.
- **bench:** `bench/residual_add_bench.cu`, real shapes `[5120,1]` (decode), `[5120,4096]` (prefill); bytes `3*N*2`.
- **DoD:** profile-first; ncu DRAM ≥ 85% at prefill; test still PASS; sanitizer clean. Commit `perf(kernels): residual_add to DRAM roofline`.

### Task sigmoid_gate_mul-A — correctness
- **Reading list:** as residual_add-A.
- **api:** `void sigmoid_gate_mul(const Tensor& gate, Tensor& x, cudaStream_t stream);` (`x *= σ(gate)`).
- **CPU ref:** `x[i] = double(x[i]) * (1.0 / (1.0 + std::exp(-double(gate[i]))))`.
- **preset:** `bf16_elementwise`. **kernel:** grid-stride bf16x2, fp32 sigmoid (`__expf`).
- **validation:** BF16, equal `ne[]`, contiguous.
- **shapes:** `[6144,1]`, `[6144,4096]`, `[255]` (unaligned); ≥3 seeds `[-8,8]`; stress `[-40,40]`.
- **commit:** `feat(kernels): sigmoid_gate_mul (correctness)`.

### Task sigmoid_gate_mul-B — performance
- **bench:** `[6144,1]`, `[6144,4096]`; bytes `3*N*2`. DoD as above. Commit `perf(kernels): sigmoid_gate_mul to DRAM roofline`.

### Task rmsnorm-A — correctness
- **Reading list:** `l1-operator-catalog.md` §3.3, `l1-op-test-standard.md` §1, `op_tester.h`/`op_check.h`, `silu_and_mul.*`.
- **api:** `void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, const Tensor* z, Tensor& out, cudaStream_t stream);`
- **Normalizes over `ne[0]` (fastest dim).** **CPU ref (fp64), per row of length `d=ne[0]`:**
  `v = mean(x_i^2); inv = 1/sqrt(v+eps);` then per element `w' = weight_i + (unit_offset?1:0);`
  `o = x_i*inv*w';` if `z`: `o *= z_i/(1+exp(-z_i))` (SiLU gate). Reduction accumulated in double.
- **preset:** `bf16_reduction`. **kernel:** one block (or warp) per row; fp32 reduction over `ne[0]`; then scale; vectorize where possible.
- **validation:** `x`/`out` BF16 same `ne[]`; `weight` BF16 1-D with `ne[0]==x.ne[0]`; if `z`, BF16 same `ne[]` as `x`; `eps>0`.
- **shapes (all four uses):** layer `[5120,1]`&`[5120,4096]` (`unit_offset=true`, `z=null`); q-norm `[256,24,7]`, k-norm `[256,4,7]` (`unit_offset=true`); gdn gated `[128,48,7]` (`unit_offset=false`, `z` set); unaligned `[260,3]`; ≥3 seeds `[-8,8]`; stress `[-60,60]`.
- **commit:** `feat(kernels): rmsnorm (unit_offset + gated, correctness)`.

### Task rmsnorm-B — performance
- **bench:** real shapes `[5120,1]` & `[5120,4096]` (layer norm; the dominant call). bytes `~2*N*2`. DoD profile-first, ncu ≥ 85% at `[5120,4096]`. Commit `perf(kernels): rmsnorm to DRAM roofline`.

### Task l2norm-A — correctness
- **Reading list:** `l1-operator-catalog.md` §3.7, framework headers, `silu_and_mul.*`.
- **api:** `void l2norm(const Tensor& x, float eps, Tensor& out, cudaStream_t stream);`
- **CPU ref (per row `d=ne[0]`):** `inv = 1/sqrt(sum(x_i^2)+eps); o_i = x_i*inv`.
- **preset:** `bf16_reduction`. **kernel:** per-row fp32 reduction.
- **validation:** BF16 same `ne[]`; `eps>0`.
- **shapes:** `[128,16,1]`, `[128,16,4096]`, `[127,5]` (unaligned); ≥3 seeds `[-8,8]`; stress includes a near-zero row.
- **commit:** `feat(kernels): l2norm (correctness)`.

### Task l2norm-B — performance
- **bench:** `[128,16,1]`, `[128,16,4096]`; bytes `2*N*2`. DoD as above. Commit `perf(kernels): l2norm to DRAM roofline`.

### Task gdn_gating-A — correctness
- **Reading list:** `l1-operator-catalog.md` §3.7, `qwen3.6-27b-architecture.md` §6.5/§7.3, framework headers.
- **api:** `void gdn_gating(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias, Tensor& g, Tensor& beta, cudaStream_t stream);`
- **shapes:** `a,b = [48,T]` BF16; `A_log,dt_bias = [48]` FP32; `g,beta = [48,T]` **FP32**.
- **CPU ref (fp64), per (h,t):** `sp = a + dt_bias[h]; sp = (sp>20)? sp : log1p(exp(sp)); g = -exp(A_log[h])*sp; beta = 1/(1+exp(-b))`.
- **preset:** `fp32_transcendental` (outputs are fp32 → use `from_device_f32`). **kernel:** elementwise over `[48,T]`, broadcast `A_log[h]`/`dt_bias[h]`, all fp32.
- **validation:** `a`/`b` BF16 `[48,T]`; `A_log`/`dt_bias` FP32 `[48]`; `g`/`beta` FP32 `[48,T]`; contiguous.
- **shapes:** `T ∈ {1,7,4096}`; one stress case with `a` in `[15,25]` (softplus guard).
- **commit:** `feat(kernels): gdn_gating (fp32, correctness)`.

### Task gdn_gating-B — performance
- **bench:** `[48,1]`, `[48,4096]`; bytes read `2*48*T*2` + write `2*48*T*4`. DoD profile-first; ncu ≥ 85% at `[48,4096]` (note: tiny tensor — confirm it is bandwidth- not launch-bound; if launch-bound at these sizes, record that finding). Commit `perf(kernels): gdn_gating`.

### Task rope-A — correctness
- **Reading list:** `l1-operator-catalog.md` §3.5, `qwen3.6-27b-architecture.md` §8, framework headers.
- **api:** `void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k, cudaStream_t stream);` (in place).
- **shapes:** `q=[256,24,T]`, `k=[256,4,T]`, `positions=I32 [T]`; `rotary_dim=64`, `theta=1e7`.
- **CPU ref (per tensor, head, token `t`, pair `i∈[0,rotary_dim/2)`):** `freq = theta^(-2i/rotary_dim); ang = positions[t]*freq; c=cos(ang); s=sin(ang); x1=v[i]; x2=v[i+rotary_dim/2]; v[i]=x1*c - x2*s; v[i+rotary_dim/2]=x2*c + x1*s;` dims `>=rotary_dim` unchanged.
- **preset:** `bf16_elementwise`. **kernel:** thread per (tensor,head,token,pair); read `positions` from device.
- **validation:** `q`/`k` BF16, `ne[0]==256`, equal `ne[2]` (=T) and `positions.ne[0]`; `positions` I32; `0<rotary_dim<=ne[0]`, even.
- **shapes:** `T ∈ {1,7,4096}`; assert pass-through dims `[64,256)` bit-unchanged; assert `positions=0` is identity.
- **commit:** `feat(kernels): rope (partial NeoX, correctness)`.

### Task rope-B — performance
- **bench:** `T=1`, `T=4096`; bytes `2*(numel_q+numel_k)*2`. DoD as above. Commit `perf(kernels): rope to DRAM roofline`.

### Task embed_gather-A — correctness
- **Reading list:** `l1-operator-catalog.md` §3.2, `weight-handle-design.md` §3/§5, `tools/q5090_convert/layouts.py::decode_row_grouped`, `include/qus/core/weight.h`, framework headers.
- **api:** `void embed_gather(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream);`
- **shapes:** `ids=I32 [T]`, `out=BF16 [d, T]`. Two variants by `table.qtype`: `BF16_CTRL` (row copy via `as_dense`) and `Q6G64_F16S` (per-group dequant-gather).
- **CPU ref:** BF16: `out[:,t]=row ids[t]`. Q6: mirror `decode_row_grouped` (group = fp16 scale + packed 6-bit signed codes; `val=code*scale`).
- **preset:** `bf16_elementwise`. **kernel(s):** `embed_gather_dense`, `embed_gather_q6`; wrapper dispatch on `qtype`.
- **test setup:** build a small `[16,128]` table both as BF16 and as a Q6 payload via a ~30-line inline packer mirroring `quantize_core` for one group; gather `ids={0,5,15,0}`; compare both paths to the float source.
- **shapes:** `T ∈ {1,4,64}`.
- **commit:** `feat(kernels): embed_gather (Q6 dequant + dense, correctness)`.

### Task embed_gather-B — performance
- **bench:** decode `T=1` (Q6 — the real case) and `T=64`; bytes Q6 `T*d*0.875 + T*d*2`. DoD profile-first; report sustained DRAM (decode `T=1` is tiny — note if launch-bound). Commit `perf(kernels): embed_gather`.

### Task argmax-A — correctness
- **Reading list:** `l1-operator-catalog.md` §3.8, framework headers.
- **api:** `void argmax(const Tensor& logits, Tensor& out, cudaStream_t stream);`
- **shapes:** `logits=BF16 [vocab,T]`, `out=I32 [T]` (`T=1` in v1).
- **CPU ref:** per column, index of max; **ties → lowest index** (scan ascending, update only on strictly greater). The kernel MUST match (combine partials lowest-index-wins). **No tolerance preset — exact index match.** Use `from_device_i32` and compare ints.
- **validation:** `logits` BF16; `out` I32 `[logits.ne[1]]`; contiguous.
- **shapes:** `vocab ∈ {1,257,248320}`, `T ∈ {1,3}`; include a deliberate tie (two equal maxima) and negative logits.
- **commit:** `feat(kernels): argmax (greedy, lowest-index tie-break, correctness)`.

### Task argmax-B — performance
- **bench:** `[248320,1]` (decode); bytes `vocab*2`. DoD profile-first; ncu sustained DRAM at the large vocab. Commit `perf(kernels): argmax`.

### Task causal_conv1d-A — correctness
- **Reading list:** `l1-operator-catalog.md` §3.7, `qwen3.6-27b-architecture.md` §6.5/§7.3, framework headers, `silu_and_mul.*`.
- **api (phase-split, one header):**
  `void causal_conv1d_prefill(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out, cudaStream_t stream);`
  `void causal_conv1d_decode (const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out, cudaStream_t stream);`
- **shapes:** `x=[C=10240,T]`, `weight=[C,4]`, `out=[C,T]`, `conv_state=[C,3]` (in-out). Applies **SiLU** to the conv output (documented; do not approximate).
- **CPU ref:** depthwise causal k=4 per channel; left-pad from `conv_state` (zeros for fresh). prefill: `acc=Σ_{j=0..3} w[c,j]*X(t-3+j,c); out=silu(acc)`; then `conv_state[c]=last 3 inputs`. decode (`T=1`): `seq=[state0,state1,state2,x]; acc=Σ w[c,j]*seq[j]; out=silu(acc); state=seq[1..3]`.
- **preset:** `bf16_reduction`.
- **validation:** BF16; `weight.ne==[C,4]`; `conv_state.ne==[C,3]`; `out.ne==x.ne`; `x.ne[0]==C`.
- **shapes:** prefill `T ∈ {1,7,64,4096}`; **decode-chain-equivalence**: 5 chained decode steps must equal one prefill of the concatenated inputs (the key state-passing test); unaligned `C=10241`; ≥3 seeds `[-8,8]`.
- **commit:** `feat(kernels): causal_conv1d (depthwise k=4 + SiLU, prefill+decode, correctness)`.

### Task causal_conv1d-B — performance
- **bench:** prefill `[10240,4096]`, decode `[10240,1]`; bytes `~4*C*T*2`. DoD profile-first; ncu ≥ 85% at prefill. Commit `perf(kernels): causal_conv1d to DRAM roofline`.

---

## Done criteria
All 9 ops have a green `qus_<op>_test` (sanitizer clean) and a perf task whose ncu sustained DRAM is
≥ 85% at the prefill shape (or a recorded finding when the real shape is launch-bound, e.g. tiny
decode tensors). `silu_and_mul`-B closes the existing 61%→roofline gap.

## Self-review notes (author)
- Each op has separate correctness (-A) and performance (-B) tasks per the user's split.
- Every task lists a Reading list and uses the FROZEN framework + a named preset (no inline tolerances).
- bench drives the real Qwen3.6-27B per-op shapes; perf acceptance is ncu sustained DRAM %, not the
  in-process GB/s. The ncu procedure is inline (no external skill).
- CPU refs are given as exact formulas for the spec-reviewer to check against the catalog/architecture.
