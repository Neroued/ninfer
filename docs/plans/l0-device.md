# L0 device Plan

## Scope

Implement `DeviceContext`, CUDA error checking, and a minimal CUDA-event timing hook from
`docs/l0-infrastructure-design.md` section 5.1. This component owns CUDA device selection,
device properties, one compute stream, one load stream, and a `CUDA_CHECK` macro. It does not
allocate model regions, tensors, caches, or weights.

## Types and API

Modify `include/qus/core/device.h`:

```cpp
#include <cuda_runtime.h>

#include <cstddef>

namespace qus {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::qus::cuda_check((expr), #expr, __FILE__, __LINE__)

struct DeviceContext {
  int device = 0;
  cudaStream_t stream = nullptr;
  cudaStream_t load_stream = nullptr;
  cudaDeviceProp props{};

  explicit DeviceContext(int device_id = 0);
  ~DeviceContext();

  DeviceContext(const DeviceContext&) = delete;
  DeviceContext& operator=(const DeviceContext&) = delete;
  DeviceContext(DeviceContext&& other) noexcept;
  DeviceContext& operator=(DeviceContext&& other) noexcept;

  int sm() const noexcept;
  std::size_t total_vram() const noexcept;
  void synchronize() const;
};

class CudaEventTimer {
public:
  explicit CudaEventTimer(const DeviceContext& ctx);
  ~CudaEventTimer();

  CudaEventTimer(const CudaEventTimer&) = delete;
  CudaEventTimer& operator=(const CudaEventTimer&) = delete;
  CudaEventTimer(CudaEventTimer&& other) noexcept;
  CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

  void start();
  float stop_ms();

private:
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

}  // namespace qus
```

Implement in `src/core/device.cu`:

- `cuda_check` prints `file:line`, the failed expression, CUDA error name, and CUDA error string
  to `stderr`, then aborts. It returns normally on `cudaSuccess`.
- `DeviceContext(int)` validates at least one CUDA device exists, validates the requested
  single-device index, calls `cudaSetDevice`, reads `cudaDeviceProp`, and creates both streams
  with `cudaStreamNonBlocking`.
- Constructor failure after one stream has been created destroys any already-created stream
  before throwing so partial construction does not leak CUDA handles.
- Destructor destroys any non-null streams without throwing.
- Move construction transfers stream ownership and nulls the moved-from streams.
- Move assignment handles self-assignment, destroys currently owned destination streams first,
  then transfers `device`, `props`, `stream`, and `load_stream` and nulls the source streams.
- `synchronize()` synchronizes the compute stream through `CUDA_CHECK`.
- `sm()` returns `props.major * 10 + props.minor`; `total_vram()` returns
  `props.totalGlobalMem`.
- `CudaEventTimer` creates two CUDA events, records on the context compute stream in `start()`,
  records/synchronizes the stop event in `stop_ms()`, and returns elapsed milliseconds. It is a
  minimal profiling hook with no NVTX dependency.
- `CudaEventTimer` sets the current CUDA device to the provided context's `device` before
  creating events so event handles match the context stream's device.
- `CudaEventTimer` constructor failure after one event has been created destroys any
  already-created event before throwing.
- `CudaEventTimer` move assignment handles self-assignment, destroys currently owned destination
  events first, then transfers `stream_`, `start_`, and `stop_` and nulls the source handles.

## Memory and Ownership

`DeviceContext` owns CUDA stream handles only. `CudaEventTimer` owns CUDA event handles only.
Neither owns device allocations. Copy is disabled to avoid double-destroy. Move is enabled so
future runtime owners can store/reassign the context and timers safely.

## Error Handling

- Construction throws `std::runtime_error` for no devices, invalid device id, device-count query
  failure, property query failure, or stream creation failure.
- `CUDA_CHECK` aborts for unrecoverable CUDA API failures, as required by the design macro.
- Destructors log stream/event destroy errors to `stderr` and continue because destructors must
  not throw.

## Edge Cases

- No CUDA device or unavailable driver: constructor fails cleanly with `std::runtime_error`.
- Moved-from contexts and timers have null handles and can be destroyed safely.
- Invalid device ids are rejected before `cudaSetDevice`.

## Tests

Create `tests/test_device.cpp` with a standalone GPU smoke test executable:

- Call `cudaGetDeviceCount`; if CUDA reports no device or no usable driver, print `SKIP` and
  return success. Other `cudaGetDeviceCount` failures return test failure before inspecting
  `count`.
- Construct `DeviceContext ctx(0)`.
- Verify `ctx.device == 0`, `ctx.stream != nullptr`, `ctx.load_stream != nullptr`,
  `ctx.sm() > 0`, and `ctx.total_vram() > 0`.
- Call `CUDA_CHECK(cudaSuccess)` and `ctx.synchronize()`.
- Move-construct another context from `ctx` and verify ownership transferred.
- Move-assign over an existing context and verify ownership transferred.
- Verify `DeviceContext(-1)` and `DeviceContext(count)` throw `std::runtime_error` when CUDA is
  usable.
- Create `CudaEventTimer`, call `start()` / `stop_ms()`, and verify elapsed time is non-negative.
- Move-assign over an existing `CudaEventTimer`, call `start()` / `stop_ms()`, and verify it
  remains usable.
- If at least two CUDA devices are available, switch the current device away from context 0 before
  constructing a timer for context 0, then verify the timer still records successfully.

Update `CMakeLists.txt`:

- Add `find_package(CUDAToolkit REQUIRED)`.
- Link `qus_core` publicly with `CUDA::cudart` because `device.h` exposes CUDA Runtime types.
- Keep `qus_l0_tests` for `tests/test_dtype.cpp`.
- Add a separate `qus_device_test` executable for `tests/test_device.cpp` and register it with
  CTest, avoiding multiple `main()` definitions.

Verification commands:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```
