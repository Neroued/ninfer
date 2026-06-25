# L0 tensor + QuantWeight Plan

## Scope

Implement the model-agnostic tensor view and quantized-weight descriptors from
`docs/l0-infrastructure-design.md` section 4. This component owns no memory and performs no CUDA
calls. It depends only on `DType`.

## Types and API

Modify `include/qus/core/tensor.h`:

```cpp
#include "qus/core/dtype.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace qus {

struct Tensor {
  void* data = nullptr;
  DType dtype = DType::BF16;
  std::int32_t ne[4] = {1, 1, 1, 1};
  std::int64_t nb[4] = {0, 0, 0, 0};

  Tensor() noexcept = default;
  Tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape);

  std::int64_t numel() const;
  std::size_t bytes() const;
  bool is_contiguous() const;

  Tensor view(std::initializer_list<std::int32_t> shape) const;
  Tensor reshape(std::initializer_list<std::int32_t> shape) const;
  Tensor slice(int dim, std::int32_t start, std::int32_t len) const;
  Tensor permute(std::initializer_list<int> order) const;
};

enum class QuantLayout : std::uint8_t {
  W4A16KernelPackedV1 = 0,
};

struct QuantWeight {
  const void* qdata = nullptr;
  const void* scales = nullptr;
  std::int32_t n = 0;
  std::int32_t k = 0;
  std::int32_t group = 0;
  QuantLayout layout = QuantLayout::W4A16KernelPackedV1;
};

}  // namespace qus
```

Implement in `src/core/tensor.cpp`:

- The constructor validates 1 to 4 positive dimensions, fills omitted dimensions with `1`, and
  sets contiguous byte strides with checked `std::int64_t` multiplication:
  - `nb[0] = dtype_size(dtype)`
  - `nb[i] = nb[i-1] * ne[i-1]`
- `numel()` multiplies all four `ne` values with checked `std::int64_t` multiplication and throws
  `std::overflow_error` before overflow.
- `bytes()` returns `numel() * dtype_size(dtype)` and throws `std::overflow_error` if the result
  cannot fit in `std::size_t`.
- `is_contiguous()` checks canonical contiguous byte strides for the stored shape and dtype.
- `view(shape)` returns a metadata-only tensor with the same data pointer and dtype, canonical
  contiguous strides, and requires the source tensor to be contiguous and `numel()` to match.
- `reshape(shape)` delegates to `view(shape)`.
- `slice(dim,start,len)` validates bounds, advances `data` by `start * nb[dim]`, changes only
  `ne[dim]`, and preserves all strides. The byte offset multiplication is checked and throws
  `std::overflow_error` before overflow.
- `permute(order)` requires exactly four unique dims in `[0,3]`, reorders `ne` and `nb`, and keeps
  `data` and `dtype`.

## Memory and Ownership

`Tensor` and `QuantWeight` are non-owning descriptors. They never allocate, free, copy payloads, or
call CUDA. All helpers are metadata-only and preserve the original backing allocation ownership.

## Error Handling

Throw `std::invalid_argument` for invalid ranks, non-positive dimensions, invalid slice dims,
out-of-range slices, invalid permutations, non-contiguous reshape/view sources, and reshape/view
element-count mismatches. Throw `std::overflow_error` for byte-size overflow.

## Edge Cases

- Missing dimensions are treated as length 1, matching the fixed `ne[4]` contract.
- Slicing can produce non-contiguous tensors when it leaves gaps in higher dimensions; helpers do
  not compact or copy.
- `view` and `reshape` are conservative: they require canonical contiguous input rather than
  trying to infer more complex strided reshapes.
- `QuantLayout` names q5090 serialized payload layouts: `TileN64K64`, `TileN64K128`,
  `RowGroupedG64`, and `Contiguous`. `W4A16KernelPackedV1` remains a compatibility alias for
  `TileN64K64`.
- `QuantWeight` defaults to a q5090 descriptor with null payload pointers and zero logical
  dimensions.

## Tests

Create `tests/test_tensor.cpp` as a standalone host test executable with explicit nonzero failure
returns:

- Construct BF16 tensor `[2,3,4]`, verify `ne = {2,3,4,1}`, `nb = {2,4,12,48}`,
  `numel() == 24`, `bytes() == 48`, and contiguous true.
- Call `view([4,6])`, verify same pointer, same numel, canonical strides, and contiguous true.
- Verify `view([5,5])` throws `std::invalid_argument` for element-count mismatch.
- Slice dimension 1 at start 1 length 2, verify data pointer advances by one dim-1 stride, shape
  becomes `{2,2,4,1}`, strides are preserved, and `is_contiguous()` is false.
- Permute `{2,1,0,3}`, verify shape and strides reorder without changing the data pointer.
- Reshape contiguous `[2,3,4]` to `[6,4]`, verify same pointer, same numel, canonical strides.
- Verify reshape of a sliced non-contiguous tensor throws `std::invalid_argument`.
- Verify invalid shape, invalid slice, and duplicate permutation each throw `std::invalid_argument`.
- Verify an oversized shape throws `std::overflow_error` instead of overflowing stride/numel/bytes
  math.
- Verify slicing a large but constructible strided tensor throws `std::overflow_error` when the
  byte offset would overflow.
- Verify `QuantWeight` can store `qdata`, `scales`, `n`, `k`, `group`, and
  `QuantLayout::W4A16KernelPackedV1`.

Update `CMakeLists.txt` with a separate `qus_tensor_test` executable and CTest entry.

Verification commands:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```
