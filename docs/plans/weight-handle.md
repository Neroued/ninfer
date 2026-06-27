# Weight Handle Implementation Plan

> **Current status:** historical plan. The unified `Weight` handle, `as_dense`, `weight_from_dense`,
> L1 `linear`/`embed_gather` signatures, and L2 card bindings now exist. This file is retained as
> the implementation history and contract rationale, not as a statement that downstream layers are
> missing.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Historical goal:** Make `Weight` the one tagged weight handle for all precisions (dense
`BF16_CTRL`/`FP32_CTRL` *and* quantized Q4/Q5/Q6/W8), and add the `as_dense` /
`weight_from_dense` bridges, so `linear` / `embed_gather` take a single `const Weight&` instead of a
`Tensor`-vs-`QuantWeight` union.

**Architecture:** Pure L0 change. (1) Clean-rename `QuantWeight` → `Weight` (no alias, no
backward-compat shim). (2) Add a small header-only `include/qus/core/weight.h` with two inline
projection helpers. The on-disk packer, the file format, the `Tensor` type, and every existing kernel
are untouched. The L1 operator signatures and the L2 card bindings now consume this handle; §"Downstream
adoption" is retained as the historical contract that those layers implemented.

**Tech Stack:** C++20, CUDA 13.1 (sm_120), CMake ≥ 3.28, gcc 13.3. Tests are framework-free
`int main()` executables returning a failure count (see `tests/test_tensor.cpp`), registered via
`qus_add_test` in `tests/CMakeLists.txt`. Build dir is `build/`.

**Spec:** [`../weight-handle-design.md`](../weight-handle-design.md). Supersedes the `LinearW` seam in
[`../l2-model-card-design.md`](../l2-model-card-design.md) §3.2/§5.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `include/qus/core/tensor.h` | modify | rename `struct QuantWeight` → `struct Weight` (definition only) |
| `include/qus/core/weight_store.h` | modify | `qweight(...)` return type + `QuantRecord::weight` become `Weight` |
| `src/core/weight_store.cpp` | modify | `make_quant_descriptor` returns `Weight`; `qweight(...)` impls return `Weight*` |
| `tests/test_tensor.cpp` | modify | the `QuantWeight` field-preservation check → `Weight` |
| `tests/test_weight_store.cpp` | modify | the 5 `qus::QuantWeight*` lookups → `qus::Weight*` |
| `include/qus/core/weight.h` | **create** | header-only `as_dense(const Weight&)` + `weight_from_dense(const Tensor&)` |
| `tests/test_weight.cpp` | **create** | unit tests for the two bridges (host-only, no CUDA) |
| `tests/CMakeLists.txt` | modify | register `qus_weight_test` |

**Naming note (locked):** the method stays `qweight()` (it still returns the quant-layout records);
only its return *type* changes via the rename. The verb the L2 card calls is `linear`; the
precision-specific kernels are its variants (see Downstream adoption). No identifier other than
`QuantWeight` is renamed.

---

## Task 1: Clean-rename `QuantWeight` → `Weight` (L0 type + store + tests)

This is a refactor: behavior is unchanged, so the verification is "everything compiles and the
existing test suite stays green." Do the whole rename in one commit so the tree never has a dangling
`QuantWeight`.

**Files:**
- Modify: `include/qus/core/tensor.h:119`
- Modify: `include/qus/core/weight_store.h:32,35,60`
- Modify: `src/core/weight_store.cpp:164,166,287,305`
- Modify: `tests/test_tensor.cpp:165,175`
- Modify: `tests/test_weight_store.cpp:129,154,158,163,175`

- [ ] **Step 1: Rename the struct definition in `tensor.h`**

Change the declaration (the only `QuantWeight` token in the file) from `struct QuantWeight {` to
`struct Weight {`. Leave every field unchanged:

```cpp
struct Weight {
    const void* payload          = nullptr;
    std::uint64_t payload_bytes  = 0;
    QType qtype                  = QType::Q4G64_F16S;
    ModuleKind module            = ModuleKind::TextCore;
    ScaleDType q5090_scale_dtype = ScaleDType::FP16;
    std::uint32_t group_size     = 0;
    std::uint32_t source_layer   = 0xFFFFFFFFU;
    std::uint32_t source_kind    = 0;
    std::int32_t shape[4]        = {1, 1, 1, 1};
    std::int32_t padded_shape[4] = {1, 1, 1, 1};
    std::uint32_t ndim           = 0;

    const void* qdata        = nullptr;
    const void* scales       = nullptr;
    std::int32_t n           = 0;
    std::int32_t k           = 0;
    std::int32_t group       = 0;
    QuantLayout layout       = QuantLayout::W4A16KernelPackedV1;
    DType scale_dtype        = DType::FP32;
    std::int32_t scale_ne[4] = {1, 1, 1, 1};
    std::int64_t scale_nb[4] = {0, 0, 0, 0};
};
```

- [ ] **Step 2: Update `weight_store.h`** — replace all 3 occurrences of `QuantWeight` with `Weight`:

```cpp
    const Weight* qweight(std::string_view name) const noexcept;                              // was QuantWeight*
    const Weight* qweight(ModuleKind module, std::uint32_t source_kind,
                          std::uint32_t source_layer) const noexcept;                         // was QuantWeight*
```

and inside `struct QuantRecord`:

```cpp
        Weight weight;   // was: QuantWeight weight;
```

- [ ] **Step 3: Update `src/core/weight_store.cpp`** — replace all 4 occurrences of `QuantWeight`
  with `Weight` (the factory return type + local, and both accessor return types):

```cpp
Weight make_quant_descriptor(const ParsedQ5090Tensor& tensor, void* payload) {
    check_int32_shape(tensor);
    Weight weight{};
    // … body unchanged …
}
```

```cpp
const Weight* WeightStore::qweight(std::string_view name) const noexcept { /* unchanged body */ }
const Weight* WeightStore::qweight(ModuleKind module, std::uint32_t source_kind,
                                   std::uint32_t source_layer) const noexcept { /* unchanged body */ }
```

- [ ] **Step 4: Update `tests/test_tensor.cpp`** — the descriptor field-preservation block
  (currently lines ~165–175):

```cpp
    int qblob    = 0;
    float scales = 0.0f;
    qus::Weight qw{};
    qw.qdata  = &qblob;
    qw.scales = &scales;
    qw.n      = 7;
    qw.k      = 11;
    qw.group  = 128;
    qw.layout = qus::QuantLayout::W4A16KernelPackedV1;
    if (qw.qdata != &qblob || qw.scales != &scales || qw.n != 7 || qw.k != 11 || qw.group != 128 ||
        qw.layout != qus::QuantLayout::W4A16KernelPackedV1) {
        ++failures;
        std::cerr << "Weight did not preserve descriptor fields\n";
    }
```

- [ ] **Step 5: Update `tests/test_weight_store.cpp`** — replace every `qus::QuantWeight*` with
  `qus::Weight*` (5 sites: the two `text_weight`/`by_source` locals, `mtp`, `vision`, and the
  `weight` local in `expect_module_payload`). No other change.

- [ ] **Step 6: Configure + build**

Run: `cmake --build build -j`
Expected: links cleanly; **no** `QuantWeight` errors. If the build dir is stale, reconfigure first
with `cmake -S . -B build` then rebuild.

- [ ] **Step 7: Confirm no `QuantWeight` token survives in code**

Run: `rg -n "QuantWeight" include src tests`
Expected: **no matches** (docs may still mention it historically; code must be clean).

- [ ] **Step 8: Run the affected tests**

Run: `ctest --test-dir build -R "qus_tensor_test|qus_weight_store_test|qus_q5090_parser_test" --output-on-failure`
Expected: all PASS (or `qus_weight_store_test` prints `SKIP: no usable CUDA device` and exits 0 on a
GPU-less box — still a pass for `ctest`).

- [ ] **Step 9: Commit**

```bash
git add include/qus/core/tensor.h include/qus/core/weight_store.h src/core/weight_store.cpp \
        tests/test_tensor.cpp tests/test_weight_store.cpp
git commit -m "refactor(core): rename QuantWeight -> Weight (unified weight handle)"
```

---

## Task 2: Add `as_dense()` bridge (dense `Weight` → `Tensor` view)

TDD: the test comes first and fails to compile (no `weight.h`), then the header makes it pass. This
test is **host-only** (no CUDA), so it runs everywhere.

**Files:**
- Create: `include/qus/core/weight.h`
- Create: `tests/test_weight.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the failing test** — create `tests/test_weight.cpp`:

```cpp
#include "qus/core/weight.h"

#include <cstdint>
#include <iostream>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

int check_i64(std::int64_t actual, std::int64_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int test_as_dense_bf16() {
    int failures = 0;
    int blob     = 0;
    qus::Weight w{};
    w.qtype    = qus::QType::BF16_CTRL;
    w.layout   = qus::QuantLayout::Contiguous;
    w.qdata    = &blob;
    w.ndim     = 2;
    w.shape[0] = 48;
    w.shape[1] = 5120;

    const qus::Tensor t = qus::as_dense(w);
    failures += (t.data == &blob) ? 0 : fail("as_dense bf16: data pointer changed");
    failures += (t.dtype == qus::DType::BF16) ? 0 : fail("as_dense bf16: dtype not BF16");
    failures += check_i64(t.ne[0], 48, "as_dense bf16 ne[0]");
    failures += check_i64(t.ne[1], 5120, "as_dense bf16 ne[1]");
    failures += check_i64(t.ne[2], 1, "as_dense bf16 ne[2]");
    failures += check_i64(t.ne[3], 1, "as_dense bf16 ne[3]");
    failures += t.is_contiguous() ? 0 : fail("as_dense bf16: not contiguous");
    return failures;
}

int test_as_dense_fp32() {
    int failures = 0;
    int blob     = 0;
    qus::Weight w{};
    w.qtype    = qus::QType::FP32_CTRL;
    w.layout   = qus::QuantLayout::Contiguous;
    w.qdata    = &blob;
    w.ndim     = 1;
    w.shape[0] = 64;

    const qus::Tensor t = qus::as_dense(w);
    failures += (t.dtype == qus::DType::FP32) ? 0 : fail("as_dense fp32: dtype not FP32");
    failures += check_i64(t.ne[0], 64, "as_dense fp32 ne[0]");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_as_dense_bf16();
    failures += test_as_dense_fp32();
    return failures == 0 ? 0 : fail("weight bridge test failed");
}
```

- [ ] **Step 2: Register the test** — append to `tests/CMakeLists.txt` after the
  `qus_tensor_test` line:

```cmake
qus_add_test(qus_weight_test                 SOURCES test_weight.cpp)
```

- [ ] **Step 3: Verify it fails** — Run: `cmake --build build -j`
Expected: FAIL — `fatal error: qus/core/weight.h: No such file or directory`.

- [ ] **Step 4: Create `include/qus/core/weight.h` with `as_dense`**

```cpp
#pragma once

#include "qus/core/tensor.h"

namespace qus {

// Project a dense Weight (qtype BF16_CTRL / FP32_CTRL, layout Contiguous) to a non-owning Tensor
// view. Mirrors WeightStore's make_tensor_view: ne[i] = shape[i], contiguous strides. This lets the
// dense arm of a precision-polymorphic op reuse the existing bf16/fp32 kernels (which take Tensor).
inline Tensor as_dense(const Weight& w) {
    const DType dt = (w.qtype == QType::FP32_CTRL) ? DType::FP32 : DType::BF16;
    void* p        = const_cast<void*>(w.qdata);
    switch (w.ndim) {
    case 1:
        return Tensor(p, dt, {w.shape[0]});
    case 2:
        return Tensor(p, dt, {w.shape[0], w.shape[1]});
    case 3:
        return Tensor(p, dt, {w.shape[0], w.shape[1], w.shape[2]});
    default:
        return Tensor(p, dt, {w.shape[0], w.shape[1], w.shape[2], w.shape[3]});
    }
}

} // namespace qus
```

- [ ] **Step 5: Verify it passes** — Run: `cmake --build build -j && ctest --test-dir build -R qus_weight_test --output-on-failure`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add include/qus/core/weight.h tests/test_weight.cpp tests/CMakeLists.txt
git commit -m "feat(core): add as_dense() — project a dense Weight to a Tensor view"
```

---

## Task 3: Add `weight_from_dense()` bridge (dense `Tensor` → `Weight{BF16_CTRL}`) + round-trip

This is the inverse used at bind time to present a contiguous dense seam param (`in_a`/`in_b`) as a
`Weight`. Seam linears are 2-D matrices `[n, k]` with the existing `ne[0]=n`, `ne[1]=k` convention.

**Files:**
- Modify: `include/qus/core/weight.h`
- Modify: `tests/test_weight.cpp`

- [ ] **Step 1: Extend `tests/test_weight.cpp`** — add these two test functions before `main()`:

```cpp
int test_weight_from_dense() {
    int failures = 0;
    int blob     = 0;
    const qus::Tensor t(&blob, qus::DType::BF16, {48, 5120});

    const qus::Weight w = qus::weight_from_dense(t);
    failures += (w.qtype == qus::QType::BF16_CTRL) ? 0 : fail("weight_from_dense: qtype not BF16_CTRL");
    failures += (w.layout == qus::QuantLayout::Contiguous) ? 0 : fail("weight_from_dense: layout not Contiguous");
    failures += (w.qdata == &blob) ? 0 : fail("weight_from_dense: qdata mismatch");
    failures += (w.payload == &blob) ? 0 : fail("weight_from_dense: payload mismatch");
    failures += (w.scales == nullptr) ? 0 : fail("weight_from_dense: scales not null");
    failures += check_i64(w.n, 48, "weight_from_dense n");
    failures += check_i64(w.k, 5120, "weight_from_dense k");
    failures += check_i64(w.group, 0, "weight_from_dense group");
    failures += check_i64(w.ndim, 2, "weight_from_dense ndim");
    failures += check_i64(w.shape[0], 48, "weight_from_dense shape[0]");
    failures += check_i64(w.shape[1], 5120, "weight_from_dense shape[1]");
    return failures;
}

int test_round_trip() {
    int failures = 0;
    int blob     = 0;
    const qus::Tensor t(&blob, qus::DType::BF16, {48, 5120});

    const qus::Tensor back = qus::as_dense(qus::weight_from_dense(t));
    failures += (back.data == t.data) ? 0 : fail("round-trip: data changed");
    failures += (back.dtype == t.dtype) ? 0 : fail("round-trip: dtype changed");
    failures += check_i64(back.ne[0], t.ne[0], "round-trip ne[0]");
    failures += check_i64(back.ne[1], t.ne[1], "round-trip ne[1]");
    return failures;
}
```

and call them from `main()`:

```cpp
int main() {
    int failures = 0;
    failures += test_as_dense_bf16();
    failures += test_as_dense_fp32();
    failures += test_weight_from_dense();
    failures += test_round_trip();
    return failures == 0 ? 0 : fail("weight bridge test failed");
}
```

- [ ] **Step 2: Verify it fails** — Run: `cmake --build build -j`
Expected: FAIL — `'weight_from_dense' is not a member of 'qus'`.

- [ ] **Step 3: Add `weight_from_dense` to `include/qus/core/weight.h`** (inside `namespace qus`,
  after `as_dense`):

```cpp
// Inverse of as_dense: wrap a dense contiguous seam param (e.g. GDN in_a/in_b) as a Weight so the
// one generic linear() verb can consume it. Seam linears are 2-D weight matrices [n, k].
inline Weight weight_from_dense(const Tensor& t) {
    Weight w{};
    w.qtype    = (t.dtype == DType::FP32) ? QType::FP32_CTRL : QType::BF16_CTRL;
    w.layout   = QuantLayout::Contiguous;
    w.qdata    = t.data;
    w.payload  = t.data;
    w.scales   = nullptr;
    w.n        = t.ne[0];
    w.k        = t.ne[1];
    w.group    = 0;
    w.ndim     = 2;
    w.shape[0] = t.ne[0];
    w.shape[1] = t.ne[1];
    return w;
}
```

- [ ] **Step 4: Verify it passes** — Run: `cmake --build build -j && ctest --test-dir build -R qus_weight_test --output-on-failure`
Expected: PASS.

- [ ] **Step 5: Full suite sanity check** — Run: `ctest --test-dir build --output-on-failure`
Expected: all PASS (CUDA-dependent tests may `SKIP` on a GPU-less box; that is still a pass).

- [ ] **Step 6: Commit**

```bash
git add include/qus/core/weight.h tests/test_weight.cpp
git commit -m "feat(core): add weight_from_dense() bridge + as_dense round-trip test"
```

---

## Downstream adoption (implemented contract for the consuming layers)

The L1 `linear`/`embed_gather` ops and the L2 card now consume the handle from this plan as follows
(full detail in [`../weight-handle-design.md`](../weight-handle-design.md) §5–§6). This section is
kept as the locked contract that current code is expected to preserve.

- **L1 `linear` / `embed_gather` api** (`include/qus/kernels/linear.h`, `.../embed_gather.h`): the
  weight argument is `const Weight&` (never a union). The **wrapper** validates then dispatches with
  `switch (w.qtype)`; `Q4G64`/`Q5G64`/`Q6G64`/`W8G128` → the matching quant-GEMV launcher variant,
  `BF16_CTRL`/`FP32_CTRL` → the dense launcher fed `as_dense(w)`. `qtype` is the `_<variant>` axis of
  `l1-kernel-layering.md` §3.
- **L2 bindings** (`include/qus/model/model.h`): no `LinearW`. `MlpW` and the per-layer structs hold
  `const Weight*` on the seam (`q/k/v/o_proj`, `gate/up/down`, `in_q…in_z`, `out_proj`) and
  `const Tensor*` for always-dense params (`*_norm`, `conv1d`, `a_log`, `dt_bias`).
- **L2 `bind()`** (`src/model/qwen3_6_27b.cpp`): quant seam params bind directly from
  `store.qweight(...)`; dense seam params (`in_a`/`in_b`, held by the store as `Tensor` records) are
  synthesized with `weight_from_dense(*store.tensor(...))` into a card-owned `std::vector<Weight>`
  that is `reserve(96)`-d before binding (so `emplace_back` cannot invalidate stored pointers), or
  the seam fields are stored `Weight` by value.

## Follow-ups / out of scope

- **Doc sync:** older historical docs may still mention `QuantWeight` when describing the
  pre-unification state. Current code and active contracts use `Weight`.
- **Deferred store unification (post-M2):** collapse `TensorRecord`/`QuantRecord` into one
  `WeightRecord` so `WeightStore` returns `const Weight*` for *every* param and `weight_from_dense`
  disappears (spec §8). Larger L0 + test change; defer until after correctness.

## Self-review notes (author)

- Spec coverage: §3 (rename, no alias) → Task 1; §4 (`as_dense`/`weight_from_dense`) → Tasks 2–3;
  §5–§6 (op/binding contract) → Downstream adoption; §8 deferred cleanup → Follow-ups.
- Type consistency: `Weight`, `as_dense(const Weight&) -> Tensor`, `weight_from_dense(const Tensor&)
  -> Weight`, accessor stays `qweight()`. Convention `ne[0]=n=shape[0]`, `ne[1]=k=shape[1]` matches
  `make_quant_descriptor`/`make_tensor_view`.
