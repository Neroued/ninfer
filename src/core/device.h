#pragma once

#include "core/memory.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer {

// Only Device allocations use this error; pinned Host allocation failures are distinct.
class DeviceAllocationError : public std::runtime_error {
public:
    DeviceAllocationError(const char* operation, std::size_t requested_bytes = 0);

    [[nodiscard]] std::size_t requested_bytes() const noexcept { return requested_bytes_; }

private:
    std::size_t requested_bytes_;
};

// Raw observations. available_bytes() is the platform's allocation-planning estimate.
struct DeviceMemorySnapshot {
    std::size_t cuda_free_bytes = 0, total_bytes = 0;
    std::size_t system_available_bytes = 0, system_total_bytes = 0;
    bool shared_physical_memory = false;

    [[nodiscard]] std::size_t available_bytes() const noexcept {
        if (!shared_physical_memory) return cuda_free_bytes;
        return system_available_bytes < total_bytes ? system_available_bytes : total_bytes;
    }
};

[[nodiscard]] DeviceMemorySnapshot device_memory_snapshot(Device);
[[nodiscard]] std::int64_t device_memory_growth(DeviceMemorySnapshot before,
                                                DeviceMemorySnapshot after);

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

// Borrowed execution context. Its stream and device facts have one owning source.
class CudaExecutionView {
public:
    [[nodiscard]] Device device() const noexcept { return device_; }

    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

    [[nodiscard]] int multiprocessor_count() const noexcept { return multiprocessor_count_; }

private:
    friend struct DeviceContext;

    CudaExecutionView(Device device, cudaStream_t stream, int multiprocessor_count) noexcept
        : device_(device), stream_(stream), multiprocessor_count_(multiprocessor_count) {}

    Device device_;
    cudaStream_t stream_;
    int multiprocessor_count_;
};

struct DeviceContext {
    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    void bind_to_current_thread() const;
    void bind_to_current_thread_noexcept() const noexcept;
    int compute_capability() const noexcept;
    int multiprocessor_count() const noexcept;

    [[nodiscard]] Device device() const noexcept { return device_; }

    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

    [[nodiscard]] cudaStream_t transfer_stream() const noexcept { return transfer_stream_; }

    [[nodiscard]] const cudaDeviceProp& properties() const noexcept { return props_; }

    CudaExecutionView execution_view() const noexcept;
    CudaExecutionView transfer_view() const noexcept;
    // Admit borrowed streams at their owner, before entering the execution chain.
    // nullptr denotes this device's legacy default stream; callers still bind the device.
    CudaExecutionView execution_view(cudaStream_t borrowed_stream) const;
    std::size_t total_vram() const noexcept;
    const char* sync_mode() const;
    // Waits for the owned compute stream, not the transfer or borrowed streams.
    void synchronize() const;

private:
    Device device_;
    cudaStream_t stream_          = nullptr;
    cudaStream_t transfer_stream_ = nullptr;
    cudaDeviceProp props_{};
};

// Allocation and transfer boundaries temporarily select their recorded device. Op launches
// run on a thread already bound by DeviceContext, without a per-launch CUDA device query.
class CudaDeviceScope {
public:
    explicit CudaDeviceScope(Device device);
    ~CudaDeviceScope();
    CudaDeviceScope(const CudaDeviceScope&)            = delete;
    CudaDeviceScope& operator=(const CudaDeviceScope&) = delete;

private:
    int previous_ = -1;
    bool changed_ = false;
};

// An additional nonblocking stream. Its owner orders and completes borrowed buffer uses.
class CudaStream {
public:
    explicit CudaStream(const DeviceContext& device);
    ~CudaStream();
    CudaStream(const CudaStream&)            = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

private:
    Device device_;
    cudaStream_t stream_ = nullptr;
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

// Reusable non-timing event for worker-driven asynchronous control transactions. The owning
// component records it after enqueueing one transfer batch and polls it from later boundaries.
class CudaCompletionEvent {
public:
    explicit CudaCompletionEvent(const DeviceContext& ctx);
    ~CudaCompletionEvent();

    CudaCompletionEvent(const CudaCompletionEvent&)            = delete;
    CudaCompletionEvent& operator=(const CudaCompletionEvent&) = delete;
    CudaCompletionEvent(CudaCompletionEvent&& other) noexcept;
    CudaCompletionEvent& operator=(CudaCompletionEvent&& other) noexcept;

    void record(cudaStream_t stream);
    // Capture a record node observable by the host while later graph nodes are still running.
    // Only call during stream capture; this event must outlive every graph referencing it.
    void record_external(cudaStream_t stream);
    void wait(cudaStream_t stream) const;
    // Capture an external dependency. Record the current producer generation before replay.
    void wait_external(cudaStream_t stream) const;
    [[nodiscard]] bool ready() const;
    void synchronize() const;

private:
    cudaEvent_t event_ = nullptr;
};

} // namespace ninfer
