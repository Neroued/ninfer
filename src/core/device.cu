#include "core/device.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer {
namespace {

struct SyncSchedule {
    std::string_view name;
    unsigned int flags;
};

constexpr SyncSchedule kSyncSchedules[] = {
    {"spin", cudaDeviceScheduleSpin},
    {"blocking", cudaDeviceScheduleBlockingSync},
    {"yield", cudaDeviceScheduleYield},
    {"auto", cudaDeviceScheduleAuto},
};

unsigned int sync_schedule_from_environment() {
    const char* value = std::getenv("NINFER_CUDA_SYNC");
    if (value == nullptr) { return cudaDeviceScheduleSpin; }
    for (const auto& schedule : kSyncSchedules) {
        if (schedule.name == value) { return schedule.flags; }
    }
    throw std::invalid_argument("NINFER_CUDA_SYNC must be spin, blocking, yield, or auto");
}

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

DeviceAllocationError::DeviceAllocationError(const char* operation, std::size_t requested_bytes)
    : std::runtime_error(std::string(operation) + ": insufficient Device memory"),
      requested_bytes_(requested_bytes) {}

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::abort();
}

DeviceContext::DeviceContext(int device_id) : device_(Device::cuda(device_id)) {
    const unsigned int sync_flags = sync_schedule_from_environment();
    int count                     = 0;
    cudaError_t err               = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_id >= count) { throw std::invalid_argument("invalid CUDA device id"); }

    bind_to_current_thread();

    err = cudaSetDeviceFlags(sync_flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDeviceFlags failed", err));
    }

    err = cudaGetDeviceProperties(&props_, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceProperties failed", err));
    }

    cudaStream_t compute = nullptr;
    cudaStream_t load    = nullptr;
    err                  = cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
    }

    err = cudaStreamCreateWithFlags(&load, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        destroy_stream(compute);
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(transfer_stream) failed", err));
    }

    stream_          = compute;
    transfer_stream_ = load;
}

DeviceContext::~DeviceContext() {
    if (stream_ != nullptr || transfer_stream_ != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream_);
    destroy_stream(stream_);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device_(other.device_), stream_(other.stream_), transfer_stream_(other.transfer_stream_),
      props_(other.props_) {
    other.stream_          = nullptr;
    other.transfer_stream_ = nullptr;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    if (stream_ != nullptr || transfer_stream_ != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream_);
    destroy_stream(stream_);

    device_          = other.device_;
    props_           = other.props_;
    stream_          = other.stream_;
    transfer_stream_ = other.transfer_stream_;

    other.stream_          = nullptr;
    other.transfer_stream_ = nullptr;
    return *this;
}

void DeviceContext::bind_to_current_thread() const {
    const cudaError_t err = cudaSetDevice(device_.index());
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept {
    log_cuda_error("cudaSetDevice", cudaSetDevice(device_.index()));
}

int DeviceContext::compute_capability() const noexcept { return props_.major * 10 + props_.minor; }

int DeviceContext::multiprocessor_count() const noexcept { return props_.multiProcessorCount; }

CudaExecutionView DeviceContext::execution_view() const noexcept {
    return {device(), stream_, multiprocessor_count()};
}

CudaExecutionView DeviceContext::transfer_view() const noexcept {
    return {device(), transfer_stream_, multiprocessor_count()};
}

CudaExecutionView DeviceContext::execution_view(cudaStream_t borrowed_stream) const {
    CudaDeviceScope scope(device());
    int ordinal      = -1;
    const auto error = cudaStreamGetDevice(borrowed_stream, &ordinal);
    if (error != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaStreamGetDevice failed", error));
    }
    if (ordinal != device_.index()) {
        throw std::invalid_argument("stream belongs to another device");
    }
    return {device(), borrowed_stream, multiprocessor_count()};
}

CudaDeviceScope::CudaDeviceScope(Device device) {
    if (device.type() != DeviceType::CUDA) {
        throw std::invalid_argument("CUDA operation requires a CUDA device");
    }
    auto error = cudaGetDevice(&previous_);
    if (error == cudaSuccess && previous_ != device.index()) {
        error    = cudaSetDevice(device.index());
        changed_ = error == cudaSuccess;
    }
    if (error != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("CUDA device selection failed", error));
    }
}

CudaDeviceScope::~CudaDeviceScope() {
    if (changed_) { log_cuda_error("restore CUDA device", cudaSetDevice(previous_)); }
}

CudaStream::CudaStream(const DeviceContext& device) : device_(device.device()) {
    CudaDeviceScope scope(device_);
    const auto error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (error != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaStreamCreateWithFlags failed", error));
    }
}

CudaStream::~CudaStream() {
    try {
        CudaDeviceScope scope(device_);
        destroy_stream(stream_);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CUDA stream cleanup failed: %s\n", error.what());
    }
}

std::size_t DeviceContext::total_vram() const noexcept { return props_.totalGlobalMem; }

const char* DeviceContext::sync_mode() const {
    bind_to_current_thread();
    unsigned int flags    = 0;
    const cudaError_t err = cudaGetDeviceFlags(&flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceFlags failed", err));
    }
    for (const auto& schedule : kSyncSchedules) {
        if ((flags & cudaDeviceScheduleMask) == schedule.flags) { return schedule.name.data(); }
    }
    throw std::runtime_error("unknown CUDA synchronization schedule");
}

void DeviceContext::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx) : CudaEventTimer(ctx, ctx.stream()) {}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream) : stream_(stream) {
    if (stream == nullptr) { throw std::invalid_argument("CUDA timer stream is null"); }
    ctx.bind_to_current_thread();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    cudaError_t err   = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

void CudaEventTimer::start_external() {
    CUDA_CHECK(cudaEventRecordWithFlags(start_, stream_, cudaEventRecordExternal));
}

void CudaEventTimer::record_stop_external() {
    CUDA_CHECK(cudaEventRecordWithFlags(stop_, stream_, cudaEventRecordExternal));
}

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

CudaCompletionEvent::CudaCompletionEvent(const DeviceContext& ctx) {
    ctx.bind_to_current_thread();
    const cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreateWithFlags failed", err));
    }
}

CudaCompletionEvent::~CudaCompletionEvent() { destroy_event(event_); }

CudaCompletionEvent::CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
    : event_(std::exchange(other.event_, nullptr)) {}

CudaCompletionEvent& CudaCompletionEvent::operator=(CudaCompletionEvent&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(event_);
    event_ = std::exchange(other.event_, nullptr);
    return *this;
}

void CudaCompletionEvent::record(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    const auto status = cudaEventRecord(event_, stream);
    if (status != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventRecord failed", status));
    }
}

void CudaCompletionEvent::wait(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not waitable");
    }
    const auto status = cudaStreamWaitEvent(stream, event_, 0);
    if (status != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaStreamWaitEvent failed", status));
    }
}

void CudaCompletionEvent::record_external(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    const auto status = cudaEventRecordWithFlags(event_, stream, cudaEventRecordExternal);
    if (status != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventRecordWithFlags failed", status));
    }
}

void CudaCompletionEvent::wait_external(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr)
        throw std::logic_error("CUDA completion event is not waitable");
    const auto status = cudaStreamWaitEvent(stream, event_, cudaEventWaitExternal);
    if (status != cudaSuccess)
        throw std::runtime_error(cuda_error_message("cudaStreamWaitEvent external failed", status));
}

bool CudaCompletionEvent::ready() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) { return true; }
    if (status == cudaErrorNotReady) { return false; }
    throw std::runtime_error(cuda_error_message("cudaEventQuery failed", status));
}

void CudaCompletionEvent::synchronize() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const auto status = cudaEventSynchronize(event_);
    if (status != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventSynchronize failed", status));
    }
}

} // namespace ninfer
