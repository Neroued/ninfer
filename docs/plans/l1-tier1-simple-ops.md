# L1 Tier-1 Simple Ops — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to
> implement this plan task-by-task (one fresh subagent per op-task, two-stage review). Steps use
> checkbox (`- [ ]`) syntax. Subagents MUST follow superpowers:test-driven-development and use the
> `profile-cuda` skill for the perf step.

**Goal:** Implement the 9 remaining Tier-1 (memory-bound) L1 operators — `residual_add`,
`sigmoid_gate_mul`, `rmsnorm`, `l2norm`, `gdn_gating`, `rope`, `embed_gather`, `argmax`,
`causal_conv1d` — each with a CPU-reference parity test over multiple shapes, the
api→wrapper→launcher→kernel layout, and `ncu`-verified throughput at the **DRAM bandwidth roofline**.
Task 1 first builds the shared test+bench harness and tunes the existing `silu_and_mul` exemplar.

**Architecture:** Each op is one operator under `l1-kernel-layering.md` (api header +
`wrapper/<op>.cpp` + `launcher/<op>[_<variant>].{h,cu}` + `kernel/<op>[_<variant>].cuh`), consumed
later by the L2 card per `l1-operator-catalog.md` §3. Tests are framework-free `int main()`
executables linked against `qus_core` (the kernel glob already compiles new `.cpp`/`.cu`). A shared
header `tests/kernels/op_harness.h` carries the device-buffer / bf16 / CPU-compare / shape-sweep
helpers and a `--bench` timing loop so the same binary is the `ncu`/`nsys` target.

**Tech Stack:** C++20, CUDA 13.1 (sm_120), CMake ≥ 3.28, gcc 13.3. `ncu`/`nsys` via the
`~/.cursor/skills/profile-cuda` skill. Roofline anchor: **1.79 TB/s** peak DRAM BW (`design.md` §1).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `tests/kernels/op_harness.h` | **create** (Task 1) | device buffers, bf16↔f32, random fill, CPU-compare (max-abs/rel), GPU-guard skip, shape sweep, `--bench` timing loop (effective GB/s) |
| `include/qus/kernels/<op>.h` | **create** per op | public api decl(s) |
| `src/kernels/wrapper/<op>.cpp` | **create** per op | validate + dispatch (follows `silu_and_mul.cpp`) |
| `src/kernels/launcher/<op>[_<v>].{h,cu}` | **create** per op | grid/block + launch + `CUDA_CHECK` |
| `src/kernels/kernel/<op>[_<v>].cuh` | **create** per op | the `__global__` compute |
| `tests/kernels/test_<op>.cpp` | **create** per op | CPU-ref parity (default) + `--bench` mode |
| `tests/CMakeLists.txt` | **modify** per op | register `qus_<op>_test` |
| `CMakeLists.txt` (src) | **modify** (Task 1) | add `-lineinfo` to `qus_core` CUDA (ncu source view) |

**Reference exemplar:** `include/qus/kernels/silu_and_mul.h` + `src/kernels/{wrapper,launcher,kernel}/
silu_and_mul.*` are the canonical four-layer shape — copy their structure for every op.

---

## Conventions (every op-task follows these)

### C1. Signatures & layout
Use the catalog signatures verbatim (`l1-operator-catalog.md` §3), `(inputs…, params…, out/in-place…,
cudaStream_t stream)`. Wrapper validates (dtype, `ne[]`, contiguity, non-null, op invariants) and
throws `std::invalid_argument("<op>: <reason>")`; launcher assumes valid; kernel is pure compute.
Numerics: bf16 in/out, **fp32 accumulate** inside.

### C2. Shared test harness — `tests/kernels/op_harness.h` (created in Task 1)

```cpp
#pragma once
#include "qus/core/device.h"
#include "qus/core/tensor.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace qus::test {

inline bool cuda_unavailable() {
    int n = 0; const cudaError_t e = cudaGetDeviceCount(&n);
    return e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver || e != cudaSuccess || n == 0;
}
inline float bf16_to_f32(std::uint16_t h) { std::uint32_t u = std::uint32_t(h) << 16; float f; std::memcpy(&f, &u, 4); return f; }
inline std::uint16_t f32_to_bf16(float f) {                 // round-to-nearest-even
    std::uint32_t u; std::memcpy(&u, &f, 4);
    const std::uint32_t lsb = (u >> 16) & 1u; u += 0x7fffu + lsb; return std::uint16_t(u >> 16);
}
struct DBuf { void* p = nullptr; explicit DBuf(std::size_t b) { cudaMalloc(&p, b); } ~DBuf() { cudaFree(p); }
              DBuf(const DBuf&) = delete; DBuf& operator=(const DBuf&) = delete; };
inline void fill_uniform(std::vector<float>& v, std::uint32_t seed, float lo = -1.f, float hi = 1.f) {
    std::mt19937 g(seed); std::uniform_real_distribution<float> d(lo, hi); for (auto& x : v) x = d(g);
}
inline std::vector<std::uint16_t> to_bf16(const std::vector<float>& v) {
    std::vector<std::uint16_t> o(v.size()); for (std::size_t i = 0; i < v.size(); ++i) o[i] = f32_to_bf16(v[i]); return o;
}
inline std::vector<float> from_bf16(const std::vector<std::uint16_t>& v) {
    std::vector<float> o(v.size()); for (std::size_t i = 0; i < v.size(); ++i) o[i] = bf16_to_f32(v[i]); return o;
}
// Compare kernel output (already bf16->f32) to fp32 CPU ref. Returns failure count; prints worst.
inline int check_close(const std::vector<float>& got, const std::vector<float>& ref,
                       const char* tag, float rtol = 2e-2f, float atol = 3e-3f) {
    float worst = 0.f; std::size_t at = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        const float tol  = atol + rtol * std::fabs(ref[i]);
        if (diff - tol > worst) { worst = diff - tol; at = i; }
    }
    if (worst > 0.f) { std::cerr << tag << " mismatch at " << at << ": got " << got[at]
                                 << " ref " << ref[at] << " (excess " << worst << ")\n"; return 1; }
    return 0;
}
// Time `launch()` over iters with CUDA events; print effective GB/s given bytes moved.
template <class F>
inline void bench(const char* tag, std::size_t bytes_moved, F&& launch, int warmup = 20, int iters = 200) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < warmup; ++i) launch();
    cudaEventRecord(a); for (int i = 0; i < iters; ++i) launch(); cudaEventRecord(b);
    cudaEventSynchronize(b); float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
    const double s = double(ms) / 1e3 / iters; const double gbps = double(bytes_moved) / s / 1e9;
    std::cout << tag << ": " << (s * 1e6) << " us  " << gbps << " GB/s  ("
              << (gbps / 1790.0 * 100.0) << "% of 1.79 TB/s)\n";
    cudaEventDestroy(a); cudaEventDestroy(b);
}

struct Shape { int ne0, ne1, ne2, ne3; };   // ne-order (ne[0] fastest)
} // namespace qus::test
```

Every `test_<op>.cpp` `main()` is:

```cpp
int main(int argc, char** argv) {
    using namespace qus::test;
    if (cuda_unavailable()) { std::cout << "SKIP: no usable CUDA device\n"; return 0; }
    const bool bench_mode = (argc > 1 && std::string(argv[1]) == "--bench");
    return bench_mode ? run_bench() : run_correctness();
}
```

### C3. Multi-shape test matrix (the "多维度序列" requirement)
`run_correctness()` sweeps a representative set, including **decode and prefill sequence lengths** and
the op's natural multi-dim views, plus non-aligned edge cases. Per-op the exact list is given, but all
include at least: `T ∈ {1, 7, 64, 4096}` and one deliberately unaligned size. `--bench` uses the two
roofline-relevant shapes (decode `T=1` and a large prefill `T`).

### C4. Performance gate (via `profile-cuda`)
After correctness passes:
1. `~/.cursor/skills/profile-cuda/scripts/preflight.sh` → must be `[OK]`/`[WARN]`.
2. `compute-sanitizer ./build/tests/qus_<op>_test` → clean.
3. `ncu --set roofline --kernel-name regex:'<op>' --launch-skip 20 --launch-count 1 -o profiles/<op>.ncu-rep ./build/tests/qus_<op>_test --bench` then `ncu --import profiles/<op>.ncu-rep --csv`.
4. **Acceptance:** `dram__throughput.avg.pct_of_peak_sustained_elapsed` **≥ 85%** at the large shape,
   and the harness `--bench` effective-GB/s line ≥ ~85% of 1.79 TB/s. If below, tune
   (vectorize `__nv_bfloat162`/`float4` loads, grid-stride sizing, occupancy) and re-measure. Save the
   final `.ncu-rep` path in the commit message.

> Memory-bound ops should hit this on the first or second iteration; if a kernel is far off, the usual
> cause is unvectorized bf16 access — load/store `__nv_bfloat162` (2-wide) or wider.

---

## Task 1: Shared harness + `-lineinfo` + bring `silu_and_mul` to roofline

**Files:** Create `tests/kernels/op_harness.h`, `tests/kernels/test_silu_and_mul.cpp`; modify
`tests/CMakeLists.txt`, `src/CMakeLists.txt`.

- [ ] **Step 1: Create `tests/kernels/op_harness.h`** with the exact contents from §C2.

- [ ] **Step 2: Add `-lineinfo` to `qus_core`** — in `src/CMakeLists.txt`, after the `qus_core`
  target is defined, append:

```cmake
target_compile_options(qus_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
```

- [ ] **Step 3: Write `tests/kernels/test_silu_and_mul.cpp`** — CPU ref `out=silu(g)*u`, sweep, bench.
  (`tests/` is on the include path via Step 4, so the harness resolves as `kernels/op_harness.h`.)

```cpp
#include "qus/kernels/silu_and_mul.h"
#include "kernels/op_harness.h"
#include <string>
using namespace qus;
using namespace qus::test;

static void cpu_silu_and_mul(const std::vector<float>& g, const std::vector<float>& u, std::vector<float>& o) {
    for (std::size_t i = 0; i < g.size(); ++i) { const float s = g[i] / (1.f + std::exp(-g[i])); o[i] = s * u[i]; }
}
static int one_shape(int n) {
    std::vector<float> g(n), u(n), ref(n);
    fill_uniform(g, 1u); fill_uniform(u, 2u); cpu_silu_and_mul(g, u, ref);
    auto gb = to_bf16(g), ub = to_bf16(u);
    std::vector<std::uint16_t> ob(n);
    DBuf dg(n * 2), du(n * 2), dout(n * 2);
    cudaMemcpy(dg.p, gb.data(), n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(du.p, ub.data(), n * 2, cudaMemcpyHostToDevice);
    Tensor tg(dg.p, DType::BF16, {n}), tu(du.p, DType::BF16, {n}), tout(dout.p, DType::BF16, {n});
    kernels::silu_and_mul(tg, tu, tout, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(ob.data(), dout.p, n * 2, cudaMemcpyDeviceToHost);
    return check_close(from_bf16(ob), ref, ("silu_and_mul n=" + std::to_string(n)).c_str());
}
static int run_correctness() {
    int f = 0;
    for (int n : {1, 7, 64, 17408, 1114112, 123457}) f += one_shape(n);
    std::cout << (f ? "FAIL" : "OK") << " silu_and_mul correctness\n";
    return f;
}
static int run_bench() {
    const int n = 1114112;                              // ~ intermediate*64, large memory-bound shape
    std::vector<float> g(n), u(n);
    fill_uniform(g, 1u); fill_uniform(u, 2u);
    auto gb = to_bf16(g), ub = to_bf16(u);
    DBuf dg(n * 2), du(n * 2), dout(n * 2);
    cudaMemcpy(dg.p, gb.data(), n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(du.p, ub.data(), n * 2, cudaMemcpyHostToDevice);
    Tensor tg(dg.p, DType::BF16, {n}), tu(du.p, DType::BF16, {n}), tout(dout.p, DType::BF16, {n});
    bench("silu_and_mul", std::size_t(3) * n * 2, [&] { kernels::silu_and_mul(tg, tu, tout, nullptr); });
    return 0;
}
int main(int argc, char** argv) {
    if (cuda_unavailable()) { std::cout << "SKIP: no usable CUDA device\n"; return 0; }
    return (argc > 1 && std::string(argv[1]) == "--bench") ? run_bench() : run_correctness();
}
```

- [ ] **Step 4: Register the test + fix include path** — in `tests/CMakeLists.txt` add:

```cmake
qus_add_test(qus_silu_and_mul_test           SOURCES kernels/test_silu_and_mul.cpp)
```

and make `tests/` resolvable for `kernels/op_harness.h` by adding, once, near the top of
`tests/CMakeLists.txt`:

```cmake
include_directories(${CMAKE_CURRENT_SOURCE_DIR})   # so kernels/op_harness.h resolves
```

- [ ] **Step 5: Build + run correctness** — `cmake -S . -B build && cmake --build build -j && ctest --test-dir build -R qus_silu_and_mul_test --output-on-failure`. Expected: PASS (or SKIP without GPU).

- [ ] **Step 6: Perf gate** — follow §C4 on `qus_silu_and_mul_test`. If `silu_and_mul.cuh` (the grid-stride
  scalar exemplar) is < 85% peak, vectorize to `__nv_bfloat162` (process 2 elems/thread) in
  `src/kernels/kernel/silu_and_mul.cuh` and re-measure.

- [ ] **Step 7: Commit** — `git add tests/kernels/op_harness.h tests/kernels/test_silu_and_mul.cpp tests/CMakeLists.txt src/CMakeLists.txt src/kernels/kernel/silu_and_mul.cuh && git commit -m "test(kernels): add op harness + silu_and_mul parity/bench at roofline"`.

---

## Task 2: `residual_add` (`x += y`, in place)

**Files:** `include/qus/kernels/residual_add.h`, `src/kernels/{wrapper,launcher,kernel}/residual_add.*`,
`tests/kernels/test_residual_add.cpp`, register in `tests/CMakeLists.txt`.

- **api:** `void residual_add(const Tensor& y, Tensor& x, cudaStream_t stream);` (bf16, same shape, contiguous).
- **CPU ref:** `x[i] = x[i] + y[i]` (fp32 accumulate, store bf16).
- **kernel compute** (grid-stride, vectorize `__nv_bfloat162`): `x[i] = f32_to_bf16(bf16_to_f32(x[i]) + bf16_to_f32(y[i]))`.
- **wrapper validation:** `y`/`x` BF16, identical `ne[]`, contiguous, non-null; short-circuit `numel()==0`.
- **test shapes:** `{1},{7},{64},{5120,4096}(=prefill hidden),{5120,1}(=decode hidden),{123457}` (unaligned).
- **bench bytes:** `3 * numel * 2` (read x, read y, write x). Shapes: `[5120,1]` and `[5120,4096]`.

- [ ] Step 1: failing test (`tests/kernels/test_residual_add.cpp`) with CPU ref + sweep + `--bench`, registered.
- [ ] Step 2: build → FAIL (`residual_add.h` missing).
- [ ] Step 3: implement the four layers (copy `silu_and_mul` structure; compute above).
- [ ] Step 4: `ctest -R qus_residual_add_test` → PASS.
- [ ] Step 5: perf gate (§C4) → DRAM ≥ 85% peak.
- [ ] Step 6: commit `feat(kernels): residual_add at bandwidth roofline`.

---

## Task 3: `sigmoid_gate_mul` (`x *= σ(gate)`, in place)

**Files:** `…/sigmoid_gate_mul.*` + test.

- **api:** `void sigmoid_gate_mul(const Tensor& gate, Tensor& x, cudaStream_t stream);`
- **CPU ref:** `x[i] = x[i] * (1.f / (1.f + std::exp(-gate[i])))`.
- **kernel:** grid-stride, fp32 sigmoid, `__nv_bfloat162` I/O.
- **validation:** BF16, equal `ne[]`, contiguous.
- **test shapes:** `{1},{7},{6144,1}(decode attn-gate),{6144,4096}(prefill),{255}` (unaligned).
- **bench bytes:** `3 * numel * 2`. Shapes `[6144,1]`, `[6144,4096]`.

- [ ] Steps 1–6 as Task 2 (test→fail→impl→pass→perf→commit). Commit `feat(kernels): sigmoid_gate_mul at roofline`.

---

## Task 4: `rmsnorm` (unit_offset + optional SiLU(z) gate)

**Files:** `…/rmsnorm.*` + test. Single variant (flags only); normalizes over `ne[0]`.

- **api:** `void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, const Tensor* z, Tensor& out, cudaStream_t stream);`
- **CPU ref (per row of length `d = ne[0]`, `rows = numel/d`):**

```cpp
for r: { double v=0; for i in [0,d): v += xf*xf;  v/=d;  inv=1/sqrt(v+eps);
         for i: { float w = weight[i] + (unit_offset?1.f:0.f);
                  float o = xf*inv*w;  if (z) o *= zf/(1+exp(-zf)); /*SiLU*/  out=bf16(o); } }
```

  `weight` length must equal `d`; `z` (if non-null) same shape as `x`.
- **kernel:** one block per row (or warp per row for small `d`); fp32 reduction over `ne[0]` (shuffle/shared); then scale. Decode rows are few/large → block-per-row; many rows → grid over rows.
- **validation:** `x`/`out` BF16 same `ne[]`; `weight` BF16, `ne[0]==x.ne[0]`, 1-D; if `z`, BF16 same `ne[]` as `x`; `eps>0`.
- **test shapes (cover all four card uses):**
  - layer norm: `x=[5120,1]` and `[5120,4096]`, `unit_offset=true`, `z=null`.
  - q-norm: `x=[256,24,7]`, `unit_offset=true`.
  - k-norm: `x=[256,4,7]`, `unit_offset=true`.
  - gdn gated: `x=[128,48,7]`, `unit_offset=false`, `z=[128,48,7]`.
  - unaligned `d`: `x=[260,3]`.
- **bench bytes:** `~2 * numel * 2` (read x + write out; weight negligible) at `[5120,1]` and `[5120,4096]`.
- **tolerance:** reductions amplify bf16 error — use `rtol=3e-2, atol=5e-3`.

- [ ] Step 1: failing test (all variant shapes + `--bench`).
- [ ] Step 2: build → FAIL.
- [ ] Step 3: implement (reduction kernel; handle `unit_offset` + optional `z` via wrapper-passed flags/pointer).
- [ ] Step 4: `ctest -R qus_rmsnorm_test` → PASS.
- [ ] Step 5: perf gate → DRAM ≥ 85% at `[5120,4096]`.
- [ ] Step 6: commit `feat(kernels): rmsnorm (unit_offset + gated) at roofline`.

---

## Task 5: `l2norm` (per-row L2 normalize over `ne[0]`)

**Files:** `…/l2norm.*` + test.

- **api:** `void l2norm(const Tensor& x, float eps, Tensor& out, cudaStream_t stream);`
- **CPU ref (per row `d=ne[0]`):** `inv = 1/sqrt(sum(xf^2)+eps); out=xf*inv`.
- **kernel:** reduction per row (like `rmsnorm` without the weight/gate); fp32.
- **validation:** BF16 same `ne[]`; `eps>0`.
- **test shapes:** q/k GDN heads `[128,16,1]` (decode), `[128,16,4096]` (prefill), `[127,5]` (unaligned).
- **bench bytes:** `2 * numel * 2` at `[128,16,1]` and `[128,16,4096]`.

- [ ] Steps 1–6 as above. Commit `feat(kernels): l2norm at roofline`.

---

## Task 6: `gdn_gating` (`g=-exp(A_log)·softplus(a+dt_bias)`, `beta=σ(b)`)

**Files:** `…/gdn_gating.*` + test. **fp32 outputs.**

- **api:** `void gdn_gating(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias, Tensor& g, Tensor& beta, cudaStream_t stream);`
- **shapes:** `a,b,g,beta = [H=48, T]`; `A_log,dt_bias = [48]` (per-head). `g`/`beta` are **FP32**; `a`/`b`/`A_log`/`dt_bias` per the card are dense (`a`/`b` bf16 from `in_a`/`in_b`; `A_log`/`dt_bias` fp32 control). Validate accordingly.
- **CPU ref (per (h,t)):**

```cpp
float sp = a + dt_bias[h];  sp = (sp > 20.f) ? sp : std::log1p(std::exp(sp));   // softplus, guarded
g[h,t]    = -std::exp(A_log[h]) * sp;          // fp32
beta[h,t] =  1.f / (1.f + std::exp(-b));       // fp32
```

- **kernel:** elementwise over `[48,T]`; read `A_log[h]`/`dt_bias[h]` (broadcast over T); all math fp32.
- **validation:** `a`/`b` BF16 `[48,T]`; `A_log`/`dt_bias` FP32 `[48]`; `g`/`beta` FP32 `[48,T]`; contiguous.
- **test shapes:** `T ∈ {1,7,4096}`; check the softplus guard with large `a` (fill `a` in `[15,25]` for one case).
- **bench bytes:** read `a,b` (2·48·T·2) + write `g,beta` (2·48·T·4) at `T=1` and `T=4096`.
- **tolerance:** fp32 outputs → `rtol=1e-5, atol=1e-6`.

- [ ] Steps 1–6. Commit `feat(kernels): gdn_gating (fp32) at roofline`.

---

## Task 7: `rope` (partial NeoX rotary, in place on q,k)

**Files:** `…/rope.*` + test. Single variant (v1 plain partial RoPE; MRoPE deferred).

- **api:** `void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k, cudaStream_t stream);`
- **shapes:** `q=[head_dim=256, n_q=24, T]`, `k=[256, n_kv=4, T]`, `positions=I32 [T]`. Rotate the first `rotary_dim=64` dims, pass through `[64,256)`.
- **CPU ref (per tensor `t`, head `h`, token `tk`, pair `i ∈ [0, rotary_dim/2)`):**

```cpp
float freq = std::pow(theta, -2.0f * i / rotary_dim);   // theta=1e7, rotary_dim=64
float ang  = positions[tk] * freq;  float c = std::cos(ang), s = std::sin(ang);
float x1 = v[i], x2 = v[i + rotary_dim/2];               // NeoX split within the rotary block
out[i]              = x1*c - x2*s;
out[i+rotary_dim/2] = x2*c + x1*s;                        // dims >= rotary_dim unchanged
```

- **kernel:** one thread per (tensor,head,token,pair); read `positions` from device; in-place write q,k.
- **validation:** `q`/`k` BF16, `ne[0]==256`, equal `ne[2]` (=T) and matching `positions.ne[0]`; `positions` I32; `0 < rotary_dim ≤ ne[0]` and even.
- **test shapes:** `T ∈ {1,7,4096}`; verify pass-through dims `[64,256)` are bit-unchanged; verify `positions=0` is identity.
- **bench bytes:** read+write q,k rotary portions; approximate `2*(numel_q+numel_k)*2`. Shapes `T=1`, `T=4096`.

- [ ] Steps 1–6. Commit `feat(kernels): rope (partial NeoX) at roofline`.

---

## Task 8: `embed_gather` (Q6 dequant-gather / BF16 row-copy; seam `Weight`)

**Files:** `…/embed_gather.*` + test. Wrapper dispatches on `table.qtype`.

- **api:** `void embed_gather(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream);`
- **shapes:** `ids=I32 [T]`, `out=BF16 [d=5120, T]`. `table` is a `Weight`: `Q6G64_F16S`
  (`ROW_GROUPED_G64`) or `BF16_CTRL`.
- **CPU ref:**
  - `BF16_CTRL`: `out[:,t] = as_dense(table) row ids[t]` (plain copy).
  - `Q6G64`: mirror `tools/q5090_convert/layouts.py::decode_row_grouped` for one row: each row = `kg=ceil(k/64)` groups; group = 2-byte fp16 `scale` + packed 6-bit codes; `val = code * scale` (signed code per `qtypes.py`). Gather row `ids[t]`.
- **kernel(s):** `embed_gather_dense` (copy) and `embed_gather_q6` (per-group unpack+dequant of one row per id). One block per id; threads over `d`.
- **validation:** `ids` I32 1-D; `out` BF16 `[table.k, ids.ne[0]]`; `table.qtype ∈ {Q6G64_F16S, BF16_CTRL}`; `ids` values `< table.n` (checked host-side only if cheap; else documented precondition).
- **test setup:** build a tiny table both ways — (a) BF16 `[n=16, k=128]`, (b) a Q6 payload for the same logical values by packing with the same group layout (provide a 30-line host packer in the test that mirrors `quantize_core` for one group: `code=round(val/scale)` clamped to the Q6 range, fp16 scale). Gather `ids={0, 5, 15, 0}` and compare both paths to the float source.
- **test shapes:** `T ∈ {1, 4, 64}` (decode + small prefill).
- **bench bytes:** Q6 path reads `T*k*0.875B` (6.25 bits/wt + scale) + writes `T*k*2`. Shapes `T=1` (decode — the real case), `T=64`.

- [ ] Step 1: failing test (BF16 path + Q6 path with the inline packer).
- [ ] Step 2: build → FAIL.
- [ ] Step 3: implement both variants + wrapper dispatch on `qtype` (dense arm via `as_dense`).
- [ ] Step 4: `ctest -R qus_embed_gather_test` → PASS.
- [ ] Step 5: perf gate (decode `T=1` is tiny; still confirm no egregious inefficiency + sanitizer clean).
- [ ] Step 6: commit `feat(kernels): embed_gather (Q6 dequant + dense)`.

---

## Task 9: `argmax` (greedy next token over vocab)

**Files:** `…/argmax.*` + test.

- **api:** `void argmax(const Tensor& logits, Tensor& out, cudaStream_t stream);`
- **shapes:** `logits=BF16 [vocab=248320, T]`, `out=I32 [T]` (one id per scored column; v1 uses `T=1`).
- **CPU ref (per column):** index of max; **ties → lowest index** (kernel must match: use `>` not `>=` when updating the running max, scanning ascending; combine partials by lowest-index-wins).
- **kernel:** block/grid reduction over `ne[0]`; emit `argmax` index (value+index reduction); deterministic lowest-index tie-break.
- **validation:** `logits` BF16; `out` I32 `[logits.ne[1]]` (or `[1]` when 1-D); contiguous.
- **test shapes:** `vocab ∈ {1, 257, 248320}`, `T ∈ {1, 3}`; include a deliberate tie (two equal maxima) to verify lowest-index wins; include negative logits.
- **bench bytes:** `vocab*T*2` read. Shape `[248320,1]` (the decode case).

- [ ] Steps 1–6. Commit `feat(kernels): argmax (greedy, lowest-index tie-break)`.

---

## Task 10: `causal_conv1d` (depthwise k=4 + SiLU; prefill/decode)

**Files:** `include/qus/kernels/causal_conv1d.h` (declares `_prefill`/`_decode`),
`src/kernels/wrapper/causal_conv1d.cpp`, `src/kernels/launcher/causal_conv1d.h` +
`causal_conv1d_prefill.cu` + `causal_conv1d_decode.cu`,
`src/kernels/kernel/causal_conv1d_prefill.cuh` + `causal_conv1d_decode.cuh`, test.

- **api:**

```cpp
void causal_conv1d_prefill(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out, cudaStream_t stream);
void causal_conv1d_decode (const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out, cudaStream_t stream);
```

- **shapes:** `x=[C=10240, T]`, `weight=[C, K=4]` (depthwise, per-channel, no bias), `out=[C, T]`,
  `conv_state=[C, K-1=3]` (the last 3 inputs per channel; **in-out**). Applies **SiLU** to the conv output (model convention; documented).
- **CPU ref:**
  - **prefill:** treat `prev[c, -3..-1] = conv_state[c]` (zeros for a fresh sequence). For `t∈[0,T)`,
    `acc = Σ_{j=0..3} weight[c,j] * X(t-3+j, c)` where `X` indexes `prev` for negative time, else `x`;
    `out[c,t] = silu(acc)`. After: `conv_state[c] = last 3 columns of x` (i.e. `x[:, T-3..T-1]`, left-padded from prev if `T<3`).
  - **decode (T=1):** `seq = [conv_state[c,0], conv_state[c,1], conv_state[c,2], x[c,0]]`;
    `acc = Σ_{j=0..3} weight[c,j]*seq[j]`; `out[c,0]=silu(acc)`; then roll: `conv_state[c] = seq[1..3]`.
- **kernel:** one thread per channel (decode) / grid over (channel, tile of T) (prefill); fp32 acc; SiLU.
- **validation:** BF16 `x`/`weight`/`out`/`conv_state`; `weight.ne == [C,4]`; `conv_state.ne == [C,3]`; `out.ne == x.ne`; `x.ne[0]==C`.
- **test shapes:** prefill `T ∈ {1,7,64,4096}`; decode `T=1` chained 5 steps and compared against a single prefill of the concatenated inputs (state-passing equivalence — the key correctness property); unaligned `C=10241` once.
- **bench bytes:** `~4 * C*T * 2` (read x + write out + small weight/state). Shapes prefill `[10240,4096]`, decode `[10240,1]`.

- [ ] Step 1: failing test (prefill sweep + decode-chain-equals-prefill equivalence + `--bench`).
- [ ] Step 2: build → FAIL.
- [ ] Step 3: implement both variants + wrapper (phase = entry point) + launcher prototypes.
- [ ] Step 4: `ctest -R qus_causal_conv1d_test` → PASS.
- [ ] Step 5: perf gate → DRAM ≥ 85% at prefill `[10240,4096]`; decode sanitizer-clean.
- [ ] Step 6: commit `feat(kernels): causal_conv1d (depthwise k=4 + SiLU, prefill+decode)`.

---

## Self-review notes (author)

- **Spec coverage:** Tier-1 ops from `l1-implementation-roadmap.md` §1 → Tasks 2–10; `silu_and_mul` +
  harness → Task 1. All carry CPU ref (the parity oracle), multi-shape sweep (incl. decode `T=1` +
  prefill), and the §C4 `ncu` roofline gate per the user's requirements.
- **Type/signature consistency:** every api signature is copied from `l1-operator-catalog.md` §3
(`(inputs…, params…, out, stream)`, seam ops take `const Weight&`). `op_harness.h` helpers are defined
once in §C2 and referenced by every task.
- **Ordering:** Task 1 builds the harness before any op uses it; `embed_gather` (Task 8) is the only
  Tier-1 seam op and depends on `as_dense` (already in `weight.h`).
- **Known judgement calls:** `embed_gather` Q6 path needs an inline test packer (Task 8 Step 1) since no
  Q6 fixture is wired into unit tests yet; `argmax` tie-break fixed to lowest-index to match a reference
  greedy decoder.
