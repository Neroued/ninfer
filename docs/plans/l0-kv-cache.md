# L0 kv_cache Plan

## Scope

Implement the contiguous, position-indexed KV cache from `docs/l0-infrastructure-design.md`
section 5.3. This component depends on `DeviceArena` and `Tensor`. It allocates K/V cache storage
inside the caller-provided Cache arena and owns only host-side descriptor vectors. It does not
implement paging, block tables, prefix cache, eviction, copies, attention kernels, or model-specific
hardcoded dimensions.

## Types and API

Modify `include/qus/core/kv_cache.h`:

```cpp
#include "qus/core/arena.h"
#include "qus/core/tensor.h"

#include <cstdint>
#include <vector>

namespace qus {

struct KVSlot {
  Tensor k;
  Tensor v;
};

struct KVCache {
  std::vector<Tensor> k;
  std::vector<Tensor> v;
  std::uint32_t pos = 0;
  std::uint32_t max_context = 0;
  std::int32_t num_kv_heads = 0;
  std::int32_t head_dim = 0;
  DType dtype = DType::BF16;

  KVCache() = default;
  KVCache(DeviceArena& cache_arena, std::uint32_t full_layers, std::uint32_t max_context,
          std::int32_t num_kv_heads, std::int32_t head_dim, DType dtype = DType::BF16);

  std::uint32_t layer_count() const noexcept;
  KVSlot slot(std::uint32_t layer, std::uint32_t position) const;
  KVSlot append_slot(std::uint32_t layer) const;
  void advance();
  void reset() noexcept;
};

}  // namespace qus
```

Implement in `src/core/kv_cache.cpp`:

- Constructor validates `full_layers > 0`, `max_context > 0`, `num_kv_heads > 0`, and
  `head_dim > 0`, and `max_context <= std::numeric_limits<std::int32_t>::max()` before casting
  `max_context` to the `std::int32_t` shape used by `Tensor`.
- Before allocating, constructor preflights the total bytes required for all K/V layer tensors
  against `cache_arena.capacity() - cache_arena.used()` with conservative 256-byte alignment
  overhead per tensor. If the preflight fails, it throws `std::bad_alloc` before consuming the
  monotonic arena.
- Allocate each layer's K and V with `cache_arena.alloc(dtype, {num_kv_heads, head_dim,
  static_cast<std::int32_t>(max_context)})`.
- Store descriptors in `k` and `v` vectors sized to `full_layers`.
- `slot(layer, position)` validates layer bounds and requires `position < pos`, then returns
  metadata-only read views for an already-written position:
  - K slot shape `{num_kv_heads, head_dim, 1}`
  - V slot shape `{num_kv_heads, head_dim, 1}`
  - Data pointers offset by `position * nb[2]` from the layer tensor.
  - Strides preserve the parent layer tensor strides.
- `append_slot(layer)` validates layer bounds, requires `pos < max_context`, and returns
  metadata-only write-target views for the current append position. L1 attention kernels write
  directly into these returned K/V views; L0 does not copy `k_new`/`v_new` tensors.
- `advance()` increments `pos` once per decode/prefill step and throws if already at
  `max_context`.
- `reset()` sets `pos = 0` without clearing memory.

## Memory and Ownership

`KVCache` owns descriptor vectors only. Device payloads are owned by the caller-provided
`DeviceArena`. Slot views are non-owning metadata and do not allocate or copy.

## Error Handling

Throw `std::invalid_argument` for invalid constructor dimensions. Throw `std::out_of_range` for
invalid layer/position and advancing beyond capacity. Propagate arena allocation exceptions.

## Edge Cases

- The implementation is model-agnostic: Qwen's 16 full-attention layers, 4 KV heads, 256 head dim,
  and 128K context are constructor parameters, not constants in L0.
- Position bookkeeping is separate from layer writes: all full-attention layers can request their
  `append_slot(layer)` at the same `pos`; the runtime/model card calls `advance()` once after the
  step.
- `slot(layer, position)` allows inspecting previous cached positions (`position < pos`) without
  changing `pos`. `append_slot(layer)` is the only public API that exposes `position == pos`.

## Tests

Create `tests/test_kv_cache.cpp` as a standalone CUDA smoke test executable with explicit nonzero
failure returns:

- Skip cleanly if CUDA reports no usable device; fail on unexpected CUDA errors.
- Construct `DeviceContext ctx(0)` and `DeviceArena cache_arena(8192)`.
- Construct `KVCache cache(cache_arena, 2, 8, 4, 16, DType::U8)` as a small byte-addressable
  fixture.
- Verify `layer_count() == 2`, `pos == 0`, K/V vector sizes are 2, each layer tensor has shape
  `{4,16,8,1}`, and K/V allocations are distinct contiguous arena views.
- Verify `append_slot(0)` and `append_slot(1)` use current `pos` and expose shape `{4,16,1,1}`;
  use `cudaMemset` to write distinct byte sentinels into K/V slots for both layers, `advance()`,
  then use `cudaMemcpy` from `slot(layer,0)` to verify layer/K/V storage does not alias.
- Write different sentinels at append position 1, advance again, and verify position 0 sentinels
  remain intact while position 1 contains the new values.
- Verify `reset()` sets `pos` back to 0 without changing layer tensor base pointers.
- Verify invalid constructor dims, including `max_context > INT32_MAX`, throw
  `std::invalid_argument`.
- Verify `slot()` before any advance, invalid layer, invalid read position, `append_slot()` at
  full capacity, and `advance()` at full capacity throw `std::out_of_range`.
- Verify constructing with an undersized arena throws `std::bad_alloc` before changing
  `cache_arena.used()`.

Update `CMakeLists.txt` with a separate `qus_kv_cache_test` executable and CTest entry.

Verification commands:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```
