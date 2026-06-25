# L0 arena Plan

## Scope

Implement `DeviceArena` and pinned host staging from `docs/l0-infrastructure-design.md` section 3.
This component depends on `DeviceContext`/`CUDA_CHECK` and `Tensor`. It provides the reusable
one-slab bump allocator used to instantiate the three L0 lifetime regions: Weights, Cache, and
Workspace. It does not implement a general allocator, free list, paging, or static-offset overlay.

## Types and API

Modify `include/qus/core/arena.h`:

```cpp
#include "qus/core/dtype.h"
#include "qus/core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace qus {

class DeviceArena {
public:
  explicit DeviceArena(std::size_t capacity_bytes);
  ~DeviceArena();

  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;
  DeviceArena(DeviceArena&& other) noexcept;
  DeviceArena& operator=(DeviceArena&& other) noexcept;

  Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape,
               std::size_t align = 256);
  void reset() noexcept;

  void* base() const noexcept;
  std::size_t used() const noexcept;
  std::size_t capacity() const noexcept;

private:
  void* base_ = nullptr;
  std::size_t cap_ = 0;
  std::size_t off_ = 0;
};

class PinnedHostBuffer {
public:
  explicit PinnedHostBuffer(std::size_t size_bytes);
  ~PinnedHostBuffer();

  PinnedHostBuffer(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
  PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
  PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

  void* data() const noexcept;
  std::size_t size() const noexcept;

private:
  void* data_ = nullptr;
  std::size_t size_ = 0;
};

using WorkspaceArena = DeviceArena;

}  // namespace qus
```

Implement in `src/core/arena.cu`:

- `DeviceArena(capacity_bytes)` rejects zero capacity, calls `cudaMalloc`, and stores the base,
  capacity, and zero offset. It direct-checks the `cudaMalloc` result and throws
  `std::runtime_error`; it does not use `CUDA_CHECK` because constructors must not abort on
  allocation failure.
- Destructor frees non-null base with `cudaFree` and logs cleanup errors without throwing.
- Move construction/assignment transfer ownership. Move assignment handles self-assignment and
  frees any destination-owned slab first.
- `alloc(dtype, shape, align)`:
  - Validates `align` is a nonzero power of two.
  - Builds a temporary `Tensor(nullptr, dtype, shape)` and computes `bytes = tmp.bytes()` to reuse
    shape and byte-size validation.
  - Aligns the actual address `reinterpret_cast<std::uintptr_t>(base_) + off_` upward with
    checked arithmetic, then derives the aligned offset from the aligned address so returned
    pointers satisfy `reinterpret_cast<std::uintptr_t>(tensor.data) % align == 0`.
  - Throws `std::bad_alloc` if the aligned range would exceed capacity.
  - Returns a `Tensor` pointing at `static_cast<std::byte*>(base_) + aligned_offset` with
    canonical strides from the `Tensor` constructor.
  - Advances `off_` to the end of the allocation.
- `reset()` sets `off_ = 0` and does not clear memory. The runtime/model card chooses to call it
  only on the Workspace region; Weights/Cache use the same primitive but do not call reset.
- `PinnedHostBuffer(size_bytes)` rejects zero size, allocates with `cudaMallocHost`, direct-checks
  the result and throws `std::runtime_error` on failure, and frees with `cudaFreeHost`.
- Pinned host buffer move operations transfer ownership and null the source.
- `WorkspaceArena` is an alias for `DeviceArena`; L0 uses the same bump-reset primitive for all
  three regions and the runtime/model card decides only the Workspace instance is reset per step.

## Memory and Ownership

`DeviceArena` owns exactly one device slab. It never frees individual allocations. Returned
`Tensor` objects are non-owning views into the slab. `PinnedHostBuffer` owns exactly one pinned
host allocation for loader staging.

## Error Handling

- Throw `std::invalid_argument` for zero arena capacity, zero staging size, and invalid alignment.
- Throw `std::runtime_error` for CUDA allocation failures.
- Throw `std::bad_alloc` for arena OOM.
- Propagate `Tensor` shape/overflow exceptions from `alloc`.
- Destructors log cleanup errors to `stderr` and continue.

## Edge Cases

- Alignment upward rounding is checked for overflow before use.
- Allocation end offset is checked for overflow before comparing to capacity.
- Moved-from arenas/buffers have null pointers and zero sizes/offsets and can be destroyed safely.
- `reset()` after move or on an empty moved-from arena is harmless.

## Tests

Create `tests/test_arena.cpp` as a standalone CUDA smoke test executable with explicit nonzero
failure returns:

- Call `cudaGetDeviceCount`; skip cleanly if CUDA reports no device or unusable driver, fail on
  other CUDA errors.
- Construct `DeviceContext ctx(0)` so CUDA device state is initialized.
- Construct `DeviceArena arena(1024)`, verify non-null base, `capacity() == 1024`, `used() == 0`.
- Verify `DeviceArena(0)` throws `std::invalid_argument`.
- Allocate `BF16 [3,5]` at default 256-byte alignment; verify returned data equals `base()`,
  tensor bytes are 30, and `used() == 30`.
- Allocate `U8 [17]` with 64-byte alignment; verify returned data equals `base()+64` and
  `reinterpret_cast<std::uintptr_t>(data) % 64 == 0`, and `used() == 81`.
- Verify invalid alignment throws `std::invalid_argument`.
- Verify an allocation that exceeds capacity throws `std::bad_alloc`.
- Verify an oversized-shape allocation propagates `std::overflow_error`.
- Call `reset()`, verify `used() == 0`, and verify the next small allocation reuses `base()`.
- Move-construct and move-assign arenas, verifying source pointers become null and destination
  state remains usable. Verify moved-from `used() == 0` and `capacity() == 0`, and self-move
  assignment preserves the destination slab.
- Construct `PinnedHostBuffer(128)`, verify non-null data and size, write a byte through the
  host pointer, then move-assign over another pinned buffer and verify destination remains usable.
- Verify `PinnedHostBuffer(0)` throws `std::invalid_argument`, moved-from pinned buffers report
  `data() == nullptr` and `size() == 0`, and pinned self-move assignment remains usable.

Update `CMakeLists.txt` with a separate `qus_arena_test` executable and CTest entry.

Verification commands:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```
