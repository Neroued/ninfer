# L0 state_store Plan

## Scope

Implement the fixed-size GDN state store from `docs/l0-infrastructure-design.md` section 5.4. This
component depends on `DeviceArena` and `Tensor`. It allocates per-GDN-layer conv and SSM state
tensors inside the caller-provided Cache arena and owns only host-side descriptor vectors. It does
not implement GDN kernels, recurrence math, paging, context-dependent state growth, or hardcoded
Qwen dimensions.

## Types and API

Modify `include/qus/core/state_store.h`:

```cpp
#include "qus/core/arena.h"
#include "qus/core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace qus {

struct GdnState {
  std::vector<Tensor> conv;
  std::vector<Tensor> ssm;
  std::int32_t conv_dim = 0;
  std::int32_t conv_width = 0;
  std::int32_t value_heads = 0;
  std::int32_t value_head_dim = 0;
  std::int32_t key_head_dim = 0;
  DType conv_dtype = DType::BF16;

  GdnState() = default;
  GdnState(DeviceArena& cache_arena, std::uint32_t gdn_layers, std::int32_t conv_dim,
           std::int32_t conv_width, std::int32_t value_heads,
           std::int32_t value_head_dim, std::int32_t key_head_dim,
           DType conv_dtype = DType::BF16);

  std::uint32_t layer_count() const noexcept;
  void reset(cudaStream_t stream = nullptr);
};

}  // namespace qus
```

Implement in `src/core/state_store.cpp`:

- Constructor validates all dimensions are positive, `gdn_layers > 0`, and `conv_dtype` is either
  `DType::BF16` or `DType::FP32`. SSM state is always `DType::FP32` per the design.
- Preflight the total bytes required for all conv/SSM layer tensors against
  `cache_arena.capacity() - cache_arena.used()` with conservative 256-byte alignment overhead per
  tensor and checked aggregate add/multiply arithmetic. If preflight fails or aggregate sizing
  overflows, throw before consuming the monotonic arena.
- Allocate, for each layer:
  - `conv[layer] = cache_arena.alloc(conv_dtype, {conv_dim, conv_width})`
  - `ssm[layer] = cache_arena.alloc(DType::FP32, {value_heads, value_head_dim, key_head_dim})`
- Store descriptors in `conv` and `ssm` vectors sized to `gdn_layers`.
- After allocation, call `reset()` to zero all conv and SSM state tensors with `cudaMemsetAsync`
  on the provided stream (default stream if none is passed) followed by stream synchronization.
- `reset(stream)` zeros every conv and SSM tensor in place. This is used at initialization and can
  be called by runtime/session reset logic.
- `layer_count()` returns descriptor vector size.

## Memory and Ownership

`GdnState` owns descriptor vectors only. Device payloads are owned by the caller-provided
`DeviceArena`. The state is fixed-size and context-independent. Zeroing state does not allocate or
free memory.

## Error Handling

Throw `std::invalid_argument` for invalid dimensions or invalid conv dtype. Throw `std::bad_alloc`
before arena consumption when preflight detects insufficient remaining capacity. Throw
`std::overflow_error` before arena consumption when aggregate preflight sizing overflows. Propagate
arena allocation and tensor shape exceptions. `reset()` uses `CUDA_CHECK` because a failed state
zeroing is unrecoverable for a usable runtime state.

## Edge Cases

- Qwen's 48 GDN layers, conv dim 10240, width 3, 48 value heads, 128 value dim, and 128 key dim
  are constructor parameters, not constants in L0.
- `conv_width` represents `kernel_width - 1`; L0 only stores the shape it is given.
- Conv dtype is parameterized but restricted to BF16/FP32 because the architecture doc leaves
  bf16/fp32 open for conv state; SSM is fixed FP32 per the design.

## Tests

Create `tests/test_state_store.cpp` as a standalone CUDA smoke test executable with explicit
nonzero failure returns:

- Skip cleanly if CUDA reports no usable device; fail on unexpected CUDA errors.
- Construct `DeviceContext ctx(0)` and `DeviceArena cache_arena(16384)`.
- Construct `GdnState state(cache_arena, 3, 10, 3, 4, 5, 6, DType::BF16)` as a small
  model-agnostic fixture.
- Verify `layer_count() == 3`, conv/SSM vector sizes are 3, conv tensors have shape `{10,3,1,1}`
  and BF16 dtype, and SSM tensors have shape `{4,5,6,1}` and FP32 dtype.
- Verify layer conv/SSM tensors are distinct non-aliasing arena views.
- Copy initial conv/SSM tensors to host and verify all bytes are zero after construction.
- Use `cudaMemset`/`cudaMemcpy` byte sentinels to verify writing layer 0 conv and layer 1 ssm does
  not affect the other tensor.
- Call `state.reset(ctx.stream)`, synchronize, and verify the previously written sentinels are
  zeroed.
- Verify invalid constructor dims and invalid conv dtype (`DType::U8`) throw
  `std::invalid_argument`.
- Verify constructing with an undersized arena throws `std::bad_alloc` before changing
  `cache_arena.used()`.
- Verify oversized aggregate dimensions throw `std::overflow_error` before changing
  `cache_arena.used()`.

Update `CMakeLists.txt` with a separate `qus_state_store_test` executable and CTest entry.

Verification commands:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```
