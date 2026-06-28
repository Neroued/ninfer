# L1 Kernel Layering & File Layout — qwen3.6-ultraspeed

> Status: design (approved in brainstorm). Date: 2026-06-26.
> Scope: the **L1 kernel layer** — how every operator is split into files and layers, where
> validation and dispatch live, and how the layers call each other. See [`design.md`](design.md)
> for project goals (§5 names this the "api/wrapper/launcher/kernel split") and
> [`l0-infrastructure-design.md`](l0-infrastructure-design.md) for the `Tensor` / `DeviceContext`
> / `CUDA_CHECK` plumbing L1 builds on.

---

## 1. Purpose & boundary

L1 is the set of hand-written CUDA operators (rmsnorm, w4a16 gemm, gqa attention, gated-deltanet,
causal conv1d, rope, swiglu, argmax, …) that the model card (L2) calls in a fixed schedule. This
document fixes **how those operators are physically organized**, not what each one computes.

**Design principles**
- **Organize by layer, not by operator family.** No `gemm/`, `attention/`, `gdn/` … subfolders.
  Each layer is one flat folder; an operator's files are distinguished by filename. **One documented
  exception:** the M3 `linear` backend is large enough to own a dedicated `src/kernels/linear/`
  subtree (see §3.1); every other operator stays flat.
- **Four layers per operator**, top to bottom: **api → wrapper → launcher → kernel**. The folder
  order reads as the call chain.
- **One responsibility per layer.** Validation and dispatch are host concerns (wrapper); launch
  configuration is the host↔device seam (launcher); the kernel is pure compute.
- **The host/device compiler boundary is explicit.** The wrapper is host C++ (`.cpp`, gcc) and
  never sees device code; the launcher and kernel are `nvcc`-compiled.

---

## 2. The four layers

| Layer | File | Compiler | Responsibility |
|---|---|---|---|
| **api** | `include/qus/kernels/<op>.h` | host-includable | Declare the public, shape-parameterized entry point(s) L2 calls. No logic. |
| **wrapper** | `src/kernels/wrapper/<op>.cpp` | host (gcc) | Implement the api. **Validate all parameters** (dtype, shape, contiguity, null/size). **Dispatch** to the right launch entry by phase + dims. Call the launcher. Never includes the kernel header. |
| **launcher** | `src/kernels/launcher/<op>[_<variant>].cu` | nvcc | Compute grid/block/shared-mem, bind the stream, launch the kernel, `CUDA_CHECK(cudaGetLastError())`. Assumes inputs are already validated. |
| **kernel** | `src/kernels/kernel/<op>[_<variant>].cuh` | nvcc (via launcher) | The `__global__` / `__device__` compute. Pure, branch-lean, no host-side validation. |

**Why validation sits in the wrapper:** failing fast on the host gives clear C++ exceptions
(`std::invalid_argument`) with no GPU round-trip, and keeps device code free of argument-checking
branches on the hot path.

**Why the kernel is a header (`.cuh`), not a compiled `.cu`:** operators here are aggressively
specialized for one frozen model, so kernels are frequently templated (tile sizes, dtypes,
prefill/decode). A header that the launcher `#include`s lets templates "just work" and lets `nvcc`
fully inline the kernel into the launch — no reliance on relocatable-device-code linking across
translation units.

---

## 3. File layout & naming

```
include/qus/kernels/
  <op>.h                         # api: one header per distinct operator
src/kernels/
  wrapper/
    <op>.cpp                     # one wrapper per operator (it is the dispatcher)
  launcher/
    <op>.h                       # private host prototype(s) for this op's launch entries
    <op>[_<variant>].cu          # one launcher per (operator, variant)
  kernel/
    <op>[_<variant>].cuh         # one kernel per (operator, variant)
```

**Naming rule.** The basename is the operator name; multi-variant operators add a `_<variant>`
suffix at the launcher and kernel layers; single-variant operators have no suffix. The layer is
identified by folder + extension, so the same basename appears across layers and is trivial to
navigate.

**Single-variant operator** (e.g. the `silu_and_mul` example in §6):

```
include/qus/kernels/silu_and_mul.h
src/kernels/wrapper/silu_and_mul.cpp
src/kernels/launcher/silu_and_mul.h
src/kernels/launcher/silu_and_mul.cu
src/kernels/kernel/silu_and_mul.cuh
```

**Multi-variant operator** (illustrative — phase-routed attention):

```
include/qus/kernels/gqa_attention.h          # declares ..._prefill / ..._decode
src/kernels/wrapper/gqa_attention.cpp        # validates, routes prefill vs decode
src/kernels/launcher/gqa_attention.h         # declares both launch entries
src/kernels/launcher/gqa_attention_prefill.cu
src/kernels/launcher/gqa_attention_decode.cu
src/kernels/kernel/gqa_attention_prefill.cuh
src/kernels/kernel/gqa_attention_decode.cuh
```

The wrapper stays **one file per operator** even when there are several variants — it is the single
place that decides which variant to launch.

### 3.1 Exception: the `linear` backend subtree

`linear` is the sole operator that does not use the flat `wrapper/ launcher/ kernel/` folders. The M3
q5090 GEMV/GEMM backend (see [`m3-linear-backend-framework.md`](m3-linear-backend-framework.md)) is
large enough — multiple formats, shape families, regimes, codecs, and tuned backends — that a flat
layout would scatter it across the shared folders. Its whole private implementation therefore lives in
a dedicated `src/kernels/linear/` subtree, organized by responsibility rather than by layer:

```
src/kernels/linear/
  linear.cpp                 # wrapper (host/gcc): validate + classify + switch dispatch
  plan/linear_plan.h         # device-free keys, registry, classify_*, constexpr table + switch
  codec/linear_codec.cuh     # qtype/layout decode primitives (group/tile granularity)
  reference/linear_generic_*.{h,cu,cuh}  # Generic bootstrap GEMV/GEMM (codec-driven)
  gemv/  gemm/               # tuned plans, added per performance phase
```

The four-layer **responsibilities** are unchanged — only their physical grouping is. The host/device
compiler boundary still holds by extension (`.cpp` = host wrapper, `.cu` = launcher, `.cuh` = device
kernel header), the public header stays at `include/qus/kernels/linear.h`, and the recursive build
glob (§7) picks the subtree up with no CMake edit. The public `linear` API does not change. This
exception applies to `linear` only; do not create per-operator subfolders for other ops.

---

## 4. The wrapper → launcher → kernel seam

The wrapper is host-compiled and must not parse `__global__` code, so it cannot include the kernel
header. The launcher therefore exposes a plain **host prototype** that the wrapper calls:

- The launch prototype lives in a **private header** `src/kernels/launcher/<op>.h`, in the
  `qus::kernels::detail` namespace. (`detail` marks it internal — not part of the public api.)
- The **wrapper** `#include`s `kernels/launcher/<op>.h` and calls `detail::<op>_launch(...)`.
- The **launcher** `.cu` includes its own private header *and* the kernel `.cuh`, defines
  `detail::<op>_launch(...)`, and does the `<<<...>>>` launch.
- The **kernel** `.cuh` is included by exactly one launcher `.cu` (its own). This keeps each
  non-inline `__global__` in a single translation unit — no ODR violation.

```
        include/qus/kernels/<op>.h        (public api decl)
                  ▲
   wrapper/<op>.cpp  ──includes──►  launcher/<op>.h   (private detail:: prototype)
        │                                  ▲
        └── calls detail::<op>_launch ─────┘
                                           │ defined in
                              launcher/<op>.cu ──includes──► kernel/<op>.cuh  (__global__)
```

**Include paths.** Public headers resolve as `qus/kernels/<op>.h` (from `include/`). Private
headers resolve as `kernels/launcher/<op>.h` because `src/` is on the target's **private** include
path (see §7).

---

## 5. Validation & dispatch (wrapper contract)

The wrapper is the operator's guard and router. A wrapper, in order:

1. **Validates** every input/output: `dtype`, shape (`ne[]`), contiguity (`is_contiguous()`),
   non-null data, and any op-specific invariant. On violation it throws
   `std::invalid_argument` with an `"<op>: <reason>"` message.
2. **Short-circuits** trivial cases (e.g. empty `numel()`).
3. **Dispatches**: chooses the launch entry by **phase** (prefill vs decode, known at the call
   site) and, where applicable, by **dims** (size-specialized impls). For single-variant ops this
   is a direct call.
4. **Launches** by calling `detail::<op>_launch(...)`.

Launchers and kernels assume inputs are valid — they do **not** re-check. There is no separate
"dispatcher" object: dispatch is just the wrapper's routing logic. (A shared
`enum class Phase { Prefill, Decode }` header is deferred until the first multi-variant op needs
it — see §8.)

---

## 6. Worked example: `silu_and_mul`

`out = silu(gate) * up`, elementwise SwiGLU activation. Single variant; `gate`/`up` are this
model's separate `MlpGate`/`MlpUp` projection outputs. The four layers in full:

**`include/qus/kernels/silu_and_mul.h`**
```cpp
#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>  // cudaStream_t

namespace qus::kernels {

// out = silu(gate) * up, elementwise (SwiGLU activation).
// gate / up / out: identical shape, BF16, contiguous.
void silu_and_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

} // namespace qus::kernels
```

**`src/kernels/wrapper/silu_and_mul.cpp`** — validation + dispatch:
```cpp
#include "qus/kernels/silu_and_mul.h"

#include "kernels/launcher/silu_and_mul.h"  // detail::silu_and_mul_launch

#include <stdexcept>

namespace qus::kernels {

void silu_and_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    if (gate.dtype != DType::BF16 || up.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("silu_and_mul: gate/up/out must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (gate.ne[d] != up.ne[d] || gate.ne[d] != out.ne[d]) {
            throw std::invalid_argument("silu_and_mul: gate/up/out shapes must match");
        }
    }
    if (!gate.is_contiguous() || !up.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("silu_and_mul: gate/up/out must be contiguous");
    }
    if (out.numel() == 0) { return; }

    detail::silu_and_mul_launch(gate, up, out, stream);  // single variant -> direct dispatch
}

} // namespace qus::kernels
```

**`src/kernels/launcher/silu_and_mul.h`** — private host prototype:
```cpp
#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

// Host entry; assumes inputs already validated by the wrapper.
void silu_and_mul_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

} // namespace qus::kernels::detail
```

**`src/kernels/launcher/silu_and_mul.cu`** — launch config:
```cpp
#include "kernels/launcher/silu_and_mul.h"

#include "kernels/kernel/silu_and_mul.cuh"
#include "qus/core/device.h"  // CUDA_CHECK

namespace qus::kernels::detail {

void silu_and_mul_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    const std::int64_t n = out.numel();
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((n + kBlock - 1) / kBlock);

    silu_and_mul_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate.data),
        static_cast<const __nv_bfloat16*>(up.data),
        static_cast<__nv_bfloat16*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
```

**`src/kernels/kernel/silu_and_mul.cuh`** — compute:
```cpp
#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

__global__ void silu_and_mul_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                    __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const float g    = __bfloat162float(gate[i]);
        const float u    = __bfloat162float(up[i]);
        const float silu = g / (1.0f + expf(-g));
        out[i]           = __float2bfloat16(silu * u);
    }
}

} // namespace qus::kernels
```

> This is a reference template; the kernel is a correctness-baseline form (grid-stride,
> fp32 math). Per-kernel optimization (`design.md` §11) happens later and does not change the
> file layout.

---

## 7. Build integration

`CMakeLists.txt` compiles L1 via the recursive glob, plus the wrapper `.cpp` pattern, plus the
private include path:

```cmake
file(GLOB_RECURSE QUS_SOURCES CONFIGURE_DEPENDS
  ...
  ${CMAKE_SOURCE_DIR}/src/kernels/*.cpp  ${CMAKE_SOURCE_DIR}/src/kernels/*.cu
  ...)

target_include_directories(qus_core PRIVATE ${CMAKE_SOURCE_DIR}/src)  # private headers
```

- Wrappers (`*.cpp`) and launchers (`*.cu`) are compiled; kernel `*.cuh` headers are pulled in by
  their launcher and are not globbed directly.
- `src/` on the **private** include path lets internal headers resolve as
  `kernels/launcher/<op>.h` without leaking to library consumers.

---

## 8. Adding a new operator (checklist)

1. `include/qus/kernels/<op>.h` — declare the public entry point(s) (`..._prefill`/`..._decode`
   if phase-split).
2. `src/kernels/kernel/<op>[_<variant>].cuh` — write the `__global__`(s).
3. `src/kernels/launcher/<op>.h` — declare `detail::<op>_<variant>_launch(...)`.
4. `src/kernels/launcher/<op>[_<variant>].cu` — configure + launch; `CUDA_CHECK`.
5. `src/kernels/wrapper/<op>.cpp` — validate, dispatch, call the launcher.

No CMake edit is needed (the glob picks up the new `.cpp`/`.cu`).

---

## 9. Open / deferred
- **Shared `Phase` enum** (`include/qus/kernels/phase.h`, `enum class Phase { Prefill, Decode }`)
  and any common dim-selection helpers — add when the first multi-variant operator lands, not
  before (YAGNI).
- **Cross-operator fusion** (`design.md` §11 ladder, e.g. dequant+GEMV, RMSNorm+QKV, SwiGLU,
  gate+residual) introduces fused operators; each fused op is still a normal
  api/wrapper/launcher/kernel set under these conventions (it just spans what used to be several
  ops).
- **Tensor-typed kernel dtype dispatch** (bf16/fp8/fp4 variants) — naturally a `_<variant>` axis
  if/when prefill low-precision kernels arrive.
