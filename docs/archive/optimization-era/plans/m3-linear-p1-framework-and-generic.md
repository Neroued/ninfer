# M3 Linear — Phase 1: Framework + Generic Backend (Codex-self-contained)

> This plan is self-contained: it does **not** rely on any Cursor/Claude "skill". Everything an
> executor needs (the subagent workflow, the contracts, the verification) is described inline or in
> linked repo files. Read this whole file plus the **spec** and **layering** docs:
> [`docs/m3-linear-backend-framework.md`](../m3-linear-backend-framework.md) (the framework spec),
> [`docs/l1-kernel-layering.md`](../l1-kernel-layering.md) §3.1/§4 (the linear subtree exception and
> the wrapper→launcher→kernel seam),
> [`docs/q5090_packed_file_format_v1.md`](../q5090_packed_file_format_v1.md) §6–§7 (decode math),
> and [`AGENTS.md`](../../AGENTS.md) (project rules: no backward compat, testing policy).

**Goal:** Stand up the M3 `linear` backend **framework** (plan key, classification, `constexpr`
registry + `switch` dispatch, and the q5090 `WeightCodec` seam) and a single **Generic** (reference)
backend that is correct for every supported `(format, shape, regime)`. No performance tuning, no
tuned per-shape kernels, no Tensor Core.

**Architecture:** All `linear` private code moves into the dedicated `src/kernels/linear/` subtree
(see layering §3.1). The public API `kernels::linear(x, w, out, stream)` is unchanged. The wrapper
validates, classifies `(LinearFormat, ShapeFamily, LinearRegime)`, resolves a `LinearPlan` from the
registry, and `switch`es on the plan's policy into a Generic launcher. Q4/Q5/Q6 share one
codec-driven low-bit GEMV/GEMM templated over a per-format codec; dense BF16/FP32 keeps its own
relocated kernels. The existing fp64 parity test `tests/kernels/test_linear.cpp` (public API) is the
correctness oracle — this phase is a behavior-preserving refactor.

**Tech Stack:** C++20, CUDA 13.x (sm_120), CMake ≥ 3.28, gcc 13. Build dir `build/`. Branch `master`.

---

## Non-goals (out of scope for Phase 1)

- No performance tuning, vectorized loads, shared-memory staging, or Tensor Core. Generic kernels may
  be scalar/grid-stride, mirroring today's correctness baseline.
- No tuned per-shape/per-format plans. Every `(format, shape, regime)` resolves to a Generic plan.
- No `SmallT` band yet (no `SmallT`-specific plan exists). `classify_regime` emits only `T1`/`LargeT`;
  the `SmallT` enum value stays reserved.
- No benchmark/`plan_id` schema work beyond a coarse policy-level id + a compose helper for logs.
- No change to the public `linear` API, the q5090 ABI, or the dispatch axes decided in the spec.
- No new tests. The existing `qus_linear_test` oracle plus `compute-sanitizer` are the gate
  (`AGENTS.md` testing policy: tests are not added by default; this refactor adds no new observable
  risk that the existing oracle does not already cover).
- No CMake edits (the `src` glob is `GLOB_RECURSE`; the test/bench targets already exist).

---

## Execution mode (subagent-driven) and controller loop

Per `AGENTS.md`, this plan is **subagent-driven**. The controller (you) drives the loop; do not pollute
one subagent's context with another's. Tasks are sequential: **T1 → T2 → T3 → T4** (T2 and T3 have
disjoint file ownership and may run in parallel, but both must land before T4).

1. Pick the next unchecked task. Dispatch **one implementer subagent** with the **Implementer prompt**
   below, filled with the task's full text + its **Reading list** + this plan's "Shared contracts"
   section. Never have a subagent read the whole plan — give it exactly its task, the shared contracts,
   and its reading list.
2. The implementer creates/edits only its owned files, builds, and (for T4) runs the oracle +
   `compute-sanitizer`, self-reviews, then commits with the task's commit message.
3. Dispatch a **spec-compliance reviewer** subagent: does the code match the task spec and the shared
   contracts exactly (types, signatures, decode math, classification tables, validation parity)?
   Fix-and-re-review until clean.
4. Dispatch a **code-quality reviewer** subagent (only after spec is clean): layering, the host/device
   boundary, grid-stride tail coverage, `CUDA_CHECK` after launch, bf16/fp32 handling, no dead code,
   no duplicate device symbols. Fix-and-re-review until approved.
5. Mark the task done. Repeat.

**Hard rules (anti-cheat):**
- Do **not** modify the oracle or the frozen test framework: `tests/kernels/test_linear.cpp`,
  `tests/kernels/op_tester.h`, `tests/kernels/op_check.h`, `tests/kernels/q5090_pack.h`, or any
  tolerance preset. If `qus_linear_test` fails, fix the new code — never the test.
- Do **not** change the public header `include/qus/kernels/linear.h` or the q5090 ABI.
- No backward compatibility: the old flat `linear` files are **deleted** in T4, not kept beside the new
  ones. Do not add toggles or fallbacks to the legacy kernels.
- Work on `master`, one commit per task, with the commit message given in the task.
- Do not edit any `CMakeLists.txt`. If a new file is not picked up, re-run `cmake -S . -B build` to
  reconfigure (`CONFIGURE_DEPENDS` should already catch it).

## Verification commands (used by tasks)

```bash
cmake -S . -B build
cmake --build build -j                                            # whole library (T1–T3 build gate)
ctest --test-dir build -R qus_linear_test --output-on-failure     # oracle (T4)
compute-sanitizer ./build/tests/qus_linear_test                   # must be clean (T4)
cmake --build build -j --target qus_linear_bench                  # bench still builds (T4)
./build/bench/qus_linear_bench --decode                           # bench still runs (T4)
```

## Subagent prompt templates

Fill `<TASK NAME>` and paste the task block + its Reading list + the "Shared contracts" section.

Implementer prompt:
```
You are implementing ONE task for qwen3.6-ultraspeed, the M3 linear framework. Repo
/home/neroued/qwen3.6-ultraspeed, branch master.
TASK:
<paste the full task block from docs/plans/m3-linear-p1-framework-and-generic.md>
SHARED CONTRACTS (authoritative type/signature/decode definitions — implement EXACTLY these):
<paste the "Shared contracts" section>
READING LIST (read these first; not the whole plan):
<paste the task's Reading list>
RULES: do NOT modify tests/kernels/test_linear.cpp or the frozen test framework or tolerances; do NOT
change include/qus/kernels/linear.h or the q5090 ABI; no backward-compat (T4 deletes the old files);
do NOT edit CMakeLists; one commit per task on master with the given message; touch only the files this
task owns.
DO: implement the task's files per the shared contracts; build with `cmake -S . -B build && cmake
--build build -j`. For T4 also run `ctest --test-dir build -R qus_linear_test --output-on-failure` and
`compute-sanitizer ./build/tests/qus_linear_test` (both must pass/clean). Self-review against the task
spec; commit.
REPORT exactly one of: DONE (summary + build/test result) / DONE_WITH_CONCERNS (+concerns) /
NEEDS_CONTEXT (+what is missing) / BLOCKED (+why).
```

Spec-compliance reviewer prompt:
```
Review ONLY whether <TASK NAME> matches its spec + the shared contracts. Repo
/home/neroued/qwen3.6-ultraspeed. Read: the task block + "Shared contracts" in
docs/plans/m3-linear-p1-framework-and-generic.md; the changed files (git diff); for decode/codec tasks
also src/kernels/kernel/linear_q4.cuh|linear_q5.cuh|linear_q6.cuh and
docs/q5090_packed_file_format_v1.md §7; for the wrapper task also the old wrapper
src/kernels/wrapper/linear.cpp and tests/kernels/test_linear.cpp.
Verify: (1) types/enums/signatures match the shared contracts verbatim; (2) classify_* tables match the
8 known shapes and the format mapping in weight.h/the q5090 spec; (3) codec decode math (tile bytes,
fp16 scale, signed LSB-first unpack) matches the existing kernels and the spec; (4) validation in the
new wrapper is identical to the old wrapper (same checks, order, empty-T short-circuit); (5) the public
header and q5090 ABI are unchanged; (6) nothing missing or extra vs the spec.
Output: "PASS", or a numbered list of concrete gaps. Do NOT fix anything.
```

Code-quality reviewer prompt (only after spec PASS):
```
Review code quality of <TASK NAME> (spec already PASSED). Repo /home/neroued/qwen3.6-ultraspeed. Read
the git diff. Check: the linear subtree layout + host/device boundary (docs/l1-kernel-layering.md
§3.1/§4); linear_plan.h is device-free (host-includable); grid-stride covers all rows/elements incl.
the K tail (kk >= k break) and N tail; CUDA_CHECK after every launch; fp32 accumulate + bf16 store;
new __global__ names do NOT collide with the still-present legacy kernels (separable compilation is ON);
no dead code/UB; for T4: no remaining references to deleted files/symbols (grep), and the old flat
linear files are gone.
Output: "APPROVED", or a numbered list of issues with severity. Do NOT fix anything.
```

---

## Shared contracts

These declarations are authoritative. All tasks must use them verbatim (names, namespaces,
signatures). Everything here lives in namespace `qus::kernels::detail`.

### C1. `src/kernels/linear/plan/linear_plan.h` (device-free; host- and nvcc-includable)

```cpp
#pragma once

#include "qus/core/tensor.h"   // QType, QuantLayout, Weight

#include <cstdint>
#include <string>

namespace qus::kernels::detail {

enum class LinearFormat {
    Q4G64_N64K64,
    Q5G64_N64K64,
    Q6G64_N64K64,
    DenseBF16,
    DenseFP32,
    GenericUnsupported,
};

enum class ShapeFamily {
    DenseCtrl48x5120,
    AttnKV1024x5120,
    GdnQK2048x5120,
    Proj6144x5120,
    Out5120x6144,
    MlpGateUp17408x5120,
    MlpDown5120x17408,
    LmHead248320x5120,
    Generic,
};

enum class LinearRegime { T1, SmallT, LargeT };

enum class LinearBackendKind { Gemv, Gemm, Reference };

enum class LinearPolicyId {
    GenericLowbitGemv,
    GenericLowbitGemm,
    GenericDenseGemv,
    GenericDenseGemm,
};

struct LinearPlanKey {
    LinearFormat format;
    ShapeFamily  shape;
    LinearRegime regime;
};

struct LinearPlan {
    LinearBackendKind backend;
    LinearPolicyId    policy;
    const char*       plan_id;           // stable coarse identity (policy granularity in Phase 1)
    bool              uses_tensor_cores;  // derived metadata, reports only
};

// Host classification. classify_format returns GenericUnsupported for any (qtype, layout) the
// wrapper does not accept; the wrapper validation rejects those before dispatch.
LinearFormat classify_format(const Weight& w);
ShapeFamily  classify_shape(std::int32_t n, std::int32_t k);
LinearRegime classify_regime(LinearFormat fmt, ShapeFamily shape, std::int32_t t);

// Phase 1 registry: every key resolves to a Generic (reference) plan.
LinearPlan resolve_plan(LinearPlanKey key);

// Names / identity for logs and (future) benchmarks. No behavioral role.
const char* format_name(LinearFormat f);
const char* shape_name(ShapeFamily s);
const char* regime_name(LinearRegime r);
const char* policy_name(LinearPolicyId p);
std::string plan_id_string(LinearPlanKey key, const LinearPlan& plan);

} // namespace qus::kernels::detail
```

### C2. `src/kernels/linear/plan/linear_plan.cpp` (host) — required behavior

```cpp
#include "kernels/linear/plan/linear_plan.h"

namespace qus::kernels::detail {

LinearFormat classify_format(const Weight& w) {
    using L = QuantLayout;
    switch (w.qtype) {
    case QType::Q4G64_F16S: return w.layout == L::TileN64K64 ? LinearFormat::Q4G64_N64K64
                                                             : LinearFormat::GenericUnsupported;
    case QType::Q5G64_F16S: return w.layout == L::TileN64K64 ? LinearFormat::Q5G64_N64K64
                                                             : LinearFormat::GenericUnsupported;
    case QType::Q6G64_F16S: return w.layout == L::TileN64K64 ? LinearFormat::Q6G64_N64K64
                                                             : LinearFormat::GenericUnsupported;
    case QType::BF16_CTRL:  return w.layout == L::Contiguous ? LinearFormat::DenseBF16
                                                             : LinearFormat::GenericUnsupported;
    case QType::FP32_CTRL:  return w.layout == L::Contiguous ? LinearFormat::DenseFP32
                                                             : LinearFormat::GenericUnsupported;
    default:                return LinearFormat::GenericUnsupported;
    }
}

ShapeFamily classify_shape(std::int32_t n, std::int32_t k) {
    struct Entry { std::int32_t n, k; ShapeFamily fam; };
    static constexpr Entry kTable[] = {
        {    48,  5120, ShapeFamily::DenseCtrl48x5120     },
        {  1024,  5120, ShapeFamily::AttnKV1024x5120      },
        {  2048,  5120, ShapeFamily::GdnQK2048x5120       },
        {  6144,  5120, ShapeFamily::Proj6144x5120        },
        {  5120,  6144, ShapeFamily::Out5120x6144         },
        { 17408,  5120, ShapeFamily::MlpGateUp17408x5120  },
        {  5120, 17408, ShapeFamily::MlpDown5120x17408    },
        {248320,  5120, ShapeFamily::LmHead248320x5120    },
    };
    for (const auto& e : kTable) {
        if (e.n == n && e.k == k) { return e.fam; }
    }
    return ShapeFamily::Generic;
}

LinearRegime classify_regime(LinearFormat /*fmt*/, ShapeFamily /*shape*/, std::int32_t t) {
    // Phase 1: no SmallT band yet. Activated when the first SmallT-specific plan lands; the
    // threshold then becomes a tunable per (format, shape) per the framework spec §8.3.
    return t == 1 ? LinearRegime::T1 : LinearRegime::LargeT;
}

LinearPlan resolve_plan(LinearPlanKey key) {
    const bool dense = (key.format == LinearFormat::DenseBF16 || key.format == LinearFormat::DenseFP32);
    const bool gemv  = (key.regime == LinearRegime::T1);
    const LinearPolicyId policy =
        dense ? (gemv ? LinearPolicyId::GenericDenseGemv  : LinearPolicyId::GenericDenseGemm)
              : (gemv ? LinearPolicyId::GenericLowbitGemv : LinearPolicyId::GenericLowbitGemm);
    return LinearPlan{ LinearBackendKind::Reference, policy, policy_name(policy),
                       /*uses_tensor_cores=*/false };
}

const char* format_name(LinearFormat f) {
    switch (f) {
    case LinearFormat::Q4G64_N64K64:     return "q4_n64k64";
    case LinearFormat::Q5G64_N64K64:     return "q5_n64k64";
    case LinearFormat::Q6G64_N64K64:     return "q6_n64k64";
    case LinearFormat::DenseBF16:        return "dense_bf16";
    case LinearFormat::DenseFP32:        return "dense_fp32";
    case LinearFormat::GenericUnsupported: return "generic_unsupported";
    }
    return "unknown";
}

const char* shape_name(ShapeFamily s) {
    switch (s) {
    case ShapeFamily::DenseCtrl48x5120:    return "dense_ctrl_48x5120";
    case ShapeFamily::AttnKV1024x5120:     return "attn_kv_1024x5120";
    case ShapeFamily::GdnQK2048x5120:      return "gdn_qk_2048x5120";
    case ShapeFamily::Proj6144x5120:       return "proj_6144x5120";
    case ShapeFamily::Out5120x6144:        return "out_5120x6144";
    case ShapeFamily::MlpGateUp17408x5120: return "mlp_gate_up_17408x5120";
    case ShapeFamily::MlpDown5120x17408:   return "mlp_down_5120x17408";
    case ShapeFamily::LmHead248320x5120:   return "lm_head_248320x5120";
    case ShapeFamily::Generic:             return "generic";
    }
    return "unknown";
}

const char* regime_name(LinearRegime r) {
    switch (r) {
    case LinearRegime::T1:     return "t1";
    case LinearRegime::SmallT: return "small_t";
    case LinearRegime::LargeT: return "large_t";
    }
    return "unknown";
}

const char* policy_name(LinearPolicyId p) {
    switch (p) {
    case LinearPolicyId::GenericLowbitGemv: return "linear.ref.lowbit.gemv.generic.v1";
    case LinearPolicyId::GenericLowbitGemm: return "linear.ref.lowbit.gemm.generic.v1";
    case LinearPolicyId::GenericDenseGemv:  return "linear.ref.dense.gemv.generic.v1";
    case LinearPolicyId::GenericDenseGemm:  return "linear.ref.dense.gemm.generic.v1";
    }
    return "linear.ref.unknown";
}

std::string plan_id_string(LinearPlanKey key, const LinearPlan& plan) {
    return std::string("linear.ref.") + format_name(key.format) + "." + shape_name(key.shape) + "." +
           regime_name(key.regime) + "." + policy_name(plan.policy);
}

} // namespace qus::kernels::detail
```

### C3. Codec — `src/kernels/linear/codec/linear_codec.cuh` (device, low-bit only)

Each low-bit codec is a stateless type for the `TILE_N64_K64` layout. A tile is
`(row/64)*kg + group`; within a tile the first `64*2` bytes are fp16 row scales, then 64 rows of
`kBytesPerRowPerGroup` packed code bytes. `load_group` fills `kGroupK` dequantized fp32 weights for
one `(row, group)`. Codes are signed, LSB-first bit-packed across byte boundaries
(`q5090_packed_file_format_v1.md` §7.1); sign-extend from bit `kBits-1`. This math is identical to the
existing per-code helpers in `src/kernels/kernel/linear_q4.cuh|linear_q5.cuh|linear_q6.cuh`.

```cpp
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace qus::kernels::detail {

struct Q4Codec {
    static constexpr int kBits = 4;
    static constexpr int kGroupK = 64;
    static constexpr int kBytesPerRowPerGroup = 32;            // ceil(64*4/8)
    static constexpr int kTileBytes = 64 * 2 + 64 * kBytesPerRowPerGroup;  // 2176
    __device__ static void load_group(const std::uint8_t* payload, std::int32_t row,
                                      std::int32_t group, std::int32_t kg, float out[kGroupK]) {
        const std::int32_t tile = row / 64;
        const std::int32_t rit  = row - tile * 64;
        const std::int64_t off  = (static_cast<std::int64_t>(tile) * kg + group) * kTileBytes;
        const std::uint16_t sb  = static_cast<std::uint16_t>(payload[off + rit * 2]) |
                                  static_cast<std::uint16_t>(
                                      static_cast<std::uint16_t>(payload[off + rit * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t* packed =
            payload + off + 64 * 2 + static_cast<std::int64_t>(rit) * kBytesPerRowPerGroup;
        for (int lane = 0; lane < kGroupK; ++lane) {
            const std::uint8_t byte = packed[lane >> 1];
            const int u = (lane & 1) ? (byte >> 4) : (byte & 0x0f);
            const int s = (u & 0x08) ? (u - 16) : u;          // sign-extend bit 3
            out[lane] = static_cast<float>(s) * scale;
        }
    }
};

struct Q5Codec {
    static constexpr int kBits = 5;
    static constexpr int kGroupK = 64;
    static constexpr int kBytesPerRowPerGroup = 40;           // ceil(64*5/8)
    static constexpr int kTileBytes = 64 * 2 + 64 * kBytesPerRowPerGroup;  // 2688
    __device__ static void load_group(const std::uint8_t* payload, std::int32_t row,
                                      std::int32_t group, std::int32_t kg, float out[kGroupK]) {
        const std::int32_t tile = row / 64;
        const std::int32_t rit  = row - tile * 64;
        const std::int64_t off  = (static_cast<std::int64_t>(tile) * kg + group) * kTileBytes;
        const std::uint16_t sb  = static_cast<std::uint16_t>(payload[off + rit * 2]) |
                                  static_cast<std::uint16_t>(
                                      static_cast<std::uint16_t>(payload[off + rit * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t* packed =
            payload + off + 64 * 2 + static_cast<std::int64_t>(rit) * kBytesPerRowPerGroup;
        for (int lane = 0; lane < kGroupK; ++lane) {
            std::uint32_t u = 0;
            const int bitpos = lane * kBits;
            for (int b = 0; b < kBits; ++b) {
                if ((packed[(bitpos + b) >> 3] & (1u << ((bitpos + b) & 7))) != 0) { u |= 1u << b; }
            }
            const int s = (u & 0x10u) ? (static_cast<int>(u) - 32) : static_cast<int>(u);  // bit 4
            out[lane] = static_cast<float>(s) * scale;
        }
    }
};

struct Q6Codec {
    static constexpr int kBits = 6;
    static constexpr int kGroupK = 64;
    static constexpr int kBytesPerRowPerGroup = 48;           // ceil(64*6/8)
    static constexpr int kTileBytes = 64 * 2 + 64 * kBytesPerRowPerGroup;  // 3200
    __device__ static void load_group(const std::uint8_t* payload, std::int32_t row,
                                      std::int32_t group, std::int32_t kg, float out[kGroupK]) {
        const std::int32_t tile = row / 64;
        const std::int32_t rit  = row - tile * 64;
        const std::int64_t off  = (static_cast<std::int64_t>(tile) * kg + group) * kTileBytes;
        const std::uint16_t sb  = static_cast<std::uint16_t>(payload[off + rit * 2]) |
                                  static_cast<std::uint16_t>(
                                      static_cast<std::uint16_t>(payload[off + rit * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t* packed =
            payload + off + 64 * 2 + static_cast<std::int64_t>(rit) * kBytesPerRowPerGroup;
        for (int lane = 0; lane < kGroupK; ++lane) {
            std::uint32_t u = 0;
            const int bitpos = lane * kBits;
            for (int b = 0; b < kBits; ++b) {
                if ((packed[(bitpos + b) >> 3] & (1u << ((bitpos + b) & 7))) != 0) { u |= 1u << b; }
            }
            const int s = (u & 0x20u) ? (static_cast<int>(u) - 64) : static_cast<int>(u);  // bit 5
            out[lane] = static_cast<float>(s) * scale;
        }
    }
};

} // namespace qus::kernels::detail
```

### C4. Generic launcher prototypes — `src/kernels/linear/reference/linear_generic.h`

```cpp
#pragma once

#include "kernels/linear/plan/linear_plan.h"   // LinearFormat
#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

// Low-bit (Q4/Q5/Q6): launcher selects the codec by fmt. w carries payload/qdata + padded_shape.
void linear_generic_lowbit_gemv_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       LinearFormat fmt, cudaStream_t stream);
void linear_generic_lowbit_gemm_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       LinearFormat fmt, cudaStream_t stream);

// Dense (BF16/FP32): wrapper passes as_dense(w) as the weight Tensor.
void linear_generic_dense_gemv_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                                      cudaStream_t stream);
void linear_generic_dense_gemm_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                                      cudaStream_t stream);

} // namespace qus::kernels::detail
```

### C5. New wrapper `linear()` body — `src/kernels/linear/linear.cpp`

Copy the anonymous-namespace validation helpers **verbatim** from the old wrapper
`src/kernels/wrapper/linear.cpp` (everything from `numel_allow_zero` through `require_dense_alignment`,
i.e. the file's lines 13–187). Then implement `linear()` exactly as below. The validation block is
unchanged from today; only the dispatch tail is new.

```cpp
#include "qus/kernels/linear.h"

#include "kernels/linear/plan/linear_plan.h"
#include "kernels/linear/reference/linear_generic.h"
#include "qus/core/weight.h"   // as_dense

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace qus::kernels {
namespace {
// ... copy validation helpers verbatim from src/kernels/wrapper/linear.cpp lines 13-187 ...
} // namespace

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)numel_allow_zero(x, "x");
    (void)numel_allow_zero(out, "out");

    // --- validation (identical to the legacy wrapper) ---
    switch (w.qtype) {
    case QType::BF16_CTRL:
    case QType::FP32_CTRL:
        require_dense_metadata(w);
        require_matrix_shapes(x, w, out);
        require_tensor_strides(x, out);
        if (w.qdata == nullptr) {
            throw std::invalid_argument("linear: dense weight data must be non-null");
        }
        if (is_empty_T(x, out)) { return; }
        require_tensor_data(x, out);
        require_dense_alignment(x, w, out);
        break;
    case QType::Q4G64_F16S:
        require_tile_lowbit_metadata(w, "Q4G64_F16S", 32u);
        require_matrix_shapes(x, w, out);
        require_tensor_strides(x, out);
        if (w.payload == nullptr && w.qdata == nullptr) {
            throw std::invalid_argument("linear: Q4G64_F16S payload must be non-null");
        }
        if (is_empty_T(x, out)) { return; }
        require_tensor_data(x, out);
        break;
    case QType::Q5G64_F16S:
        require_tile_lowbit_metadata(w, "Q5G64_F16S", 40u);
        require_matrix_shapes(x, w, out);
        require_tensor_strides(x, out);
        if (w.payload == nullptr && w.qdata == nullptr) {
            throw std::invalid_argument("linear: Q5G64_F16S payload must be non-null");
        }
        if (is_empty_T(x, out)) { return; }
        require_tensor_data(x, out);
        break;
    case QType::Q6G64_F16S:
        require_tile_lowbit_metadata(w, "Q6G64_F16S", 48u);
        require_matrix_shapes(x, w, out);
        require_tensor_strides(x, out);
        if (w.payload == nullptr && w.qdata == nullptr) {
            throw std::invalid_argument("linear: Q6G64_F16S payload must be non-null");
        }
        if (is_empty_T(x, out)) { return; }
        require_tensor_data(x, out);
        break;
    default:
        throw std::invalid_argument("linear: unsupported weight qtype");
    }

    // --- classify + dispatch (M3 framework) ---
    const detail::LinearFormat fmt    = detail::classify_format(w);
    const detail::ShapeFamily  shape  = detail::classify_shape(w.n, w.k);
    const detail::LinearRegime regime = detail::classify_regime(fmt, shape, x.ne[1]);
    const detail::LinearPlan   plan   = detail::resolve_plan(detail::LinearPlanKey{fmt, shape, regime});

    switch (plan.policy) {
    case detail::LinearPolicyId::GenericLowbitGemv:
        detail::linear_generic_lowbit_gemv_launch(x, w, out, fmt, stream);
        break;
    case detail::LinearPolicyId::GenericLowbitGemm:
        detail::linear_generic_lowbit_gemm_launch(x, w, out, fmt, stream);
        break;
    case detail::LinearPolicyId::GenericDenseGemv: {
        const Tensor dense = as_dense(w);
        detail::linear_generic_dense_gemv_launch(x, dense, out, stream);
        break;
    }
    case detail::LinearPolicyId::GenericDenseGemm: {
        const Tensor dense = as_dense(w);
        detail::linear_generic_dense_gemm_launch(x, dense, out, stream);
        break;
    }
    }
}

} // namespace qus::kernels
```

---

## Tasks

### Task 1 — plan key, classification, and the generic registry

**Owns / creates:**
- `src/kernels/linear/plan/linear_plan.h`
- `src/kernels/linear/plan/linear_plan.cpp`

**Reading list:** Shared contracts C1+C2; `include/qus/core/tensor.h` (`QType`/`QuantLayout`/`Weight`);
`docs/m3-linear-backend-framework.md` §8 + §15 (shape table); `docs/qwen3.6-27b-architecture.md`
§11 weight table (shape cross-check); `docs/l1-kernel-layering.md` §3.1.

**Spec:** Implement C1 and C2 verbatim. `linear_plan.h` MUST be device-free (no CUDA headers, no
`__device__`), so it is includable from both the host wrapper and `.cu` launchers. Nothing references
these symbols yet — this task only adds the contract + its host implementation.

**DoD / verify:**
```bash
cmake -S . -B build && cmake --build build -j
```
The library builds (the new `linear_plan.cpp` compiles under gcc, proving the header is device-free).
Spec reviewer confirms the `classify_shape` table matches all 8 known shapes and `classify_format`
matches the qtype/layout pairs in `weight.h`/the q5090 spec.

**Commit:** `feat(linear): add M3 plan key, classify, and generic registry`

---

### Task 2 — q5090 codec + generic low-bit GEMV/GEMM

**Owns / creates:**
- `src/kernels/linear/codec/linear_codec.cuh`
- `src/kernels/linear/reference/linear_generic_lowbit.cuh`
- `src/kernels/linear/reference/linear_generic_lowbit_gemv.cu`
- `src/kernels/linear/reference/linear_generic_lowbit_gemm.cu`
- `src/kernels/linear/reference/linear_generic.h` (the **complete** C4 header — all four prototypes.
  The two dense launchers are only *declared* here; Task 3 *defines* them. Declaring them now is safe
  because nothing calls them until Task 4, and it keeps this header owned by exactly one task.)

**Reading list:** Shared contracts C3+C4; `src/kernels/kernel/linear_q4.cuh`, `linear_q5.cuh`,
`linear_q6.cuh` (the exact decode being folded into the codec); `docs/q5090_packed_file_format_v1.md`
§6–§7; `src/kernels/launcher/linear_q4.cu` (grid/block + `payload?:qdata` pattern to mirror);
`docs/l1-kernel-layering.md` §4.

**Spec:**
- Implement the codec C3 verbatim (Q4/Q5/Q6).
- `linear_generic_lowbit.cuh`: two `template <class Codec> __global__` kernels in
  `qus::kernels::detail`, mirroring today's baseline math but calling `Codec::load_group`:

```cpp
#pragma once

#include "kernels/linear/codec/linear_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels::detail {

template <class Codec>
__global__ void linear_generic_lowbit_gemv_kernel(const __nv_bfloat16* x,
                                                  const std::uint8_t* payload, __nv_bfloat16* out,
                                                  std::int32_t n, std::int32_t k,
                                                  std::int32_t padded_k) {
    const std::int32_t kg     = padded_k / Codec::kGroupK;
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t row64 = start; row64 < n; row64 += stride) {
        const std::int32_t row = static_cast<std::int32_t>(row64);
        float acc              = 0.0f;
        float wbuf[Codec::kGroupK];
        for (std::int32_t group = 0; group < kg; ++group) {
            Codec::load_group(payload, row, group, kg, wbuf);
            for (int lane = 0; lane < Codec::kGroupK; ++lane) {
                const std::int32_t kk = group * Codec::kGroupK + lane;
                if (kk >= k) { break; }
                acc = fmaf(wbuf[lane], __bfloat162float(x[kk]), acc);
            }
        }
        out[row] = __float2bfloat16(acc);
    }
}

template <class Codec>
__global__ void linear_generic_lowbit_gemm_kernel(const __nv_bfloat16* x,
                                                  const std::uint8_t* payload, __nv_bfloat16* out,
                                                  std::int32_t n, std::int32_t k, std::int32_t t,
                                                  std::int32_t padded_k) {
    const std::int32_t kg       = padded_k / Codec::kGroupK;
    const std::int64_t elements = static_cast<std::int64_t>(n) * t;
    const std::int64_t start    = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride   = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < elements; i += stride) {
        const std::int32_t col     = static_cast<std::int32_t>(i / n);
        const std::int32_t row     = static_cast<std::int32_t>(i - static_cast<std::int64_t>(col) * n);
        const __nv_bfloat16* x_col = x + static_cast<std::int64_t>(col) * k;
        float acc                  = 0.0f;
        float wbuf[Codec::kGroupK];
        for (std::int32_t group = 0; group < kg; ++group) {
            Codec::load_group(payload, row, group, kg, wbuf);
            for (int lane = 0; lane < Codec::kGroupK; ++lane) {
                const std::int32_t kk = group * Codec::kGroupK + lane;
                if (kk >= k) { break; }
                acc = fmaf(wbuf[lane], __bfloat162float(x_col[kk]), acc);
            }
        }
        out[i] = __float2bfloat16(acc);
    }
}

} // namespace qus::kernels::detail
```

- The two `.cu` launchers define the C4 prototypes; each computes grid/block (mirror
  `launcher/linear_q4.cu`: `kBlock = 128`, grid = ceil(work / kBlock), `payload = w.payload ? w.payload
  : w.qdata`, `padded_k = w.padded_shape[1]`, `n = out.ne[0]`, `k = x.ne[0]`, gemm `t = x.ne[1]`) and
  `switch (fmt)` to launch the kernel templated on `Q4Codec`/`Q5Codec`/`Q6Codec`; `CUDA_CHECK`
  (`#include "qus/core/device.h"`) after launch. `default:` is unreachable (wrapper guarantees a
  low-bit format) — leave it empty.

**DoD / verify:**
```bash
cmake -S . -B build && cmake --build build -j
```
Builds (the new `.cu` files instantiate and compile the codec + templated kernels). Spec reviewer
diffs the codec decode against `linear_q4/q5/q6.cuh` and the spec (tile bytes 2176/2688/3200, fp16
scale, signed LSB-first unpack). Not wired into `linear()` yet.

**Commit:** `feat(linear): add q5090 codec and generic low-bit GEMV/GEMM`

---

### Task 3 — relocate the dense generic kernels into the subtree

**Owns / creates:**
- `src/kernels/linear/reference/linear_generic_dense.cuh`
- `src/kernels/linear/reference/linear_generic_dense.cu`

(The dense launcher prototypes are already declared in `linear_generic.h` by Task 2; this task only
provides their definitions. Do not edit `linear_generic.h`.)

**Reading list:** Shared contracts C4; `src/kernels/kernel/linear_dense.cuh` and
`src/kernels/launcher/linear_dense.cu` (the source being relocated); `include/qus/core/weight.h`
(`as_dense`); `docs/l1-kernel-layering.md` §4.

**Spec:** Move the dense kernels + launcher logic into the new files, preserving behavior exactly
(GEMV block-per-row, the small-GEMV path with `kDenseSmallGemvMaxN`/`kDenseSmallGemvMaxT`, the 2-D
GEMM, the fp32/bf16 `weight_fp32` handling and `bf16x2` vectorization). **Rename** the `__global__`
kernels and the launch functions to the `linear_generic_dense_*` names (e.g.
`linear_generic_dense_gemv_kernel`, `linear_generic_dense_gemm_launch`) so they do not collide with the
still-present legacy `linear_dense_*` symbols (CUDA separable compilation is ON — duplicate device
symbols would fail device link). Put the launch functions in `qus::kernels::detail` matching C4.

**DoD / verify:**
```bash
cmake -S . -B build && cmake --build build -j
```
Builds (legacy dense kernels still present and untouched; the new renamed copies compile alongside).
Spec reviewer confirms the relocation is faithful and only names changed.

**Commit:** `refactor(linear): relocate dense generic kernels into linear subtree`

---

### Task 4 — flip the wrapper to the registry and delete the legacy kernels

**Owns / creates:**
- `src/kernels/linear/linear.cpp` (new wrapper, C5)

**Owns / deletes (no backward compat):**
- `src/kernels/wrapper/linear.cpp`
- `src/kernels/launcher/linear.h`
- `src/kernels/launcher/linear_q4.cu`, `linear_q5.cu`, `linear_q6.cu`, `linear_dense.cu`
- `src/kernels/kernel/linear_q4.cuh`, `linear_q5.cuh`, `linear_q6.cuh`, `linear_dense.cuh`

**Reading list:** Shared contracts C5; the old wrapper `src/kernels/wrapper/linear.cpp` (copy its
validation helpers verbatim); `tests/kernels/test_linear.cpp` (the oracle — READ ONLY, lists the
validation cases that must still pass: unsupported qtype, dense metadata/alignment, lowbit metadata,
empty-T); `include/qus/core/weight.h`; `docs/l1-kernel-layering.md` §3.1/§4.

**Spec:** Create the new wrapper per C5 (validation helpers copied verbatim; new classify + dispatch
tail). The new and old wrappers both define `qus::kernels::linear`, so they cannot coexist — create the
new file and delete all the legacy files listed above **in this one task**, then build. After deletion,
grep to confirm nothing references the removed files/symbols
(`linear_q4_gemv_launch`, `linear_dense_gemm_kernel`, `kernels/launcher/linear.h`, etc.).

**DoD / verify (this is the numerical + integration gate):**
```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build -R qus_linear_test --output-on-failure       # PASS
compute-sanitizer ./build/tests/qus_linear_test                     # clean (no errors)
cmake --build build -j --target qus_linear_bench
./build/bench/qus_linear_bench --decode                             # runs, no crash
git grep -n "linear_q4_gemv_launch\|linear_dense_gemm_kernel\|kernels/launcher/linear.h" -- src ; echo "expect: no matches"
```
`qus_linear_test` PASS proves byte-for-byte-within-tolerance parity with the pre-refactor behavior
across every qtype/shape/T and every validation case. Spec reviewer confirms validation parity and that
the public header is unchanged; code reviewer confirms no legacy files/symbols remain.

**Commit:** `refactor(linear): route linear through M3 plan registry, drop legacy kernels`

---

## Definition of done (phase)

- `src/kernels/linear/` subtree exists with `plan/`, `codec/`, `reference/`, the wrapper, and no other
  per-operator subfolders introduced.
- `linear()` dispatches via `classify_* → resolve_plan → switch(policy)`; every supported
  `(format, shape, regime)` resolves to a Generic plan.
- The q5090 `WeightCodec` seam (Q4/Q5/Q6) exists at group/tile granularity and is the only place the
  low-bit decode lives.
- All legacy flat `linear` files are deleted; the public API and q5090 ABI are unchanged.
- `qus_linear_test` PASS, `compute-sanitizer` clean, `qus_linear_bench` builds and runs.
- No new tests, no CMake edits, no tolerance/oracle changes.

## Review phase (risk-scaled per AGENTS.md)

This touches CUDA kernels and numerical decode, so every task uses the two-reviewer pattern
(spec-compliance, then code-quality). Task 4 additionally requires the oracle PASS + `compute-sanitizer`
clean as the merge gate. No independent numerical re-derivation is needed beyond the existing fp64
oracle, which already stresses Q4/Q5/Q6/dense at real shapes and T values.

## Later phases (roadmap, not executable here)

Subsequent phases replace Generic entries with tuned plans, one `(format, shape, regime)` at a time,
each gated by parity against the Generic result + a benchmark line (framework spec §15 tiers, §18
benchmark contract): Tier A first (`MlpGateUp/MlpDown/LmHead`), then Tier B (`Proj/Out`), then Tier C1
(`AttnKV/GdnQK`, where the spec §11.2 mandates CTA-internal K-split for occupancy). The `SmallT` band
and per-shape `plan_id` strings activate with their first specialized plan.

## Self-review notes (author)

- Goal/non-goals, subagent execution mode, per-task file ownership, reading lists, DoD + verification
  commands, and a risk-scaled review phase are all present (AGENTS.md plan requirements).
- Tasks split by verifiable boundary: T1 contract (builds), T2 codec+low-bit (builds), T3 dense
  relocation (builds), T4 integration (oracle + sanitizer). T2/T3 have disjoint ownership.
- Shared contracts give complete, copy-paste type/codec/wrapper code so subagents need no cross-task
  state. Decode math is taken verbatim from the existing kernels (oracle-checked), not re-derived.
- Testing follows AGENTS.md: reuse the existing fp64 oracle; add no new tests; never touch the frozen
  framework/tolerances.
- Each file is owned by exactly one task (`linear_generic.h` is created complete in T2; T3 only adds
  the dense definitions), so there is no shared-file append coordination. The duplicate-device-symbol
  hazard (separable compilation is ON) is called out in T3/T4 as the reason for the `linear_generic_*`
  renames and the create-new-plus-delete-old atomicity in T4.
