#include "core/buffer.h"
#include "core/device.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace {

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

void release_device(void* data, Device device) noexcept {
    if (data == nullptr) { return; }
    try {
        CudaDeviceScope scope(device);
        check(cudaFree(data), "cudaFree");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CUDA allocation cleanup failed: %s\n", error.what());
    }
}

void release_pinned(void* data) noexcept {
    if (data == nullptr) { return; }
    const auto error = cudaFreeHost(data);
    if (error != cudaSuccess) {
        std::fprintf(stderr, "cudaFreeHost failed: %s\n", cudaGetErrorString(error));
    }
}

CudaHostView map_host(ConstMemorySpan host, Device device) {
    CudaDeviceScope scope(device);
    void* mapped = nullptr;
    if (host.bytes) {
        check(cudaHostGetDevicePointer(&mapped, const_cast<void*>(host.data), 0),
              "cudaHostGetDevicePointer");
    }
    return {{host.data, host.bytes, Device::cpu(), true}, mapped, device};
}

void release_registration(CudaHostView view) noexcept {
    if (!view.host.bytes) { return; }
    try {
        CudaDeviceScope scope(view.cuda_device);
        check(cudaHostUnregister(const_cast<void*>(view.host.data)), "cudaHostUnregister");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "CUDA registration cleanup failed: %s\n", error.what());
    }
}

} // namespace

CudaHostRegistration::CudaHostRegistration(ConstMemorySpan host, Device cuda_device) {
    validate_memory(host);
    if (host.device != Device::cpu() || host.host_pinned) {
        throw std::invalid_argument("CUDA registration requires ordinary Host backing");
    }
    CudaDeviceScope scope(cuda_device);
    if (host.bytes) {
        check(cudaHostRegister(const_cast<void*>(host.data), host.bytes, cudaHostRegisterDefault),
              "cudaHostRegister");
    }
    try {
        view_ = map_host(host, cuda_device);
    } catch (...) {
        release_registration({host, nullptr, cuda_device});
        throw;
    }
}

CudaHostRegistration::~CudaHostRegistration() { release_registration(view_); }

CudaHostRegistration::CudaHostRegistration(CudaHostRegistration&& other) noexcept
    : view_(std::exchange(other.view_, {})) {}

CudaHostRegistration& CudaHostRegistration::operator=(CudaHostRegistration&& other) noexcept {
    if (this != &other) {
        release_registration(view_);
        view_ = std::exchange(other.view_, {});
    }
    return *this;
}

DeviceBuffer::DeviceBuffer(Device device, std::size_t bytes) : bytes_(bytes), device_(device) {
    CudaDeviceScope scope(device);
    if (bytes != 0) {
        const auto error = cudaMalloc(&data_, bytes);
        if (error == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            throw DeviceAllocationError("cudaMalloc", bytes);
        }
        check(error, "cudaMalloc");
    }
}

DeviceBuffer::~DeviceBuffer() { release_device(data_, device_); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), bytes_(std::exchange(other.bytes_, 0)),
      device_(std::exchange(other.device_, {})) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        release_device(data_, device_);
        data_   = std::exchange(other.data_, nullptr);
        bytes_  = std::exchange(other.bytes_, 0);
        device_ = std::exchange(other.device_, {});
    }
    return *this;
}

void DeviceBuffer::fill(int byte_value) {
    if (bytes_ == 0) { return; }
    CudaDeviceScope scope(device_);
    check(cudaMemset(data_, byte_value, bytes_), "cudaMemset");
    check(cudaStreamSynchronize(nullptr), "buffer initialization synchronization");
}

void DeviceBuffer::copy_from_host(const void* source, std::size_t count, std::size_t offset) {
    const auto destination = span().subspan(offset, count);
    if (count == 0) { return; }
    if (source == nullptr) { throw std::invalid_argument("upload source is null"); }
    CudaDeviceScope scope(device_);
    check(cudaMemcpy(destination.data, source, count, cudaMemcpyHostToDevice), "buffer upload");
    // Pageable H2D can return before DMA completes; settle it before a nonblocking-stream consumer.
    check(cudaStreamSynchronize(nullptr), "buffer upload synchronization");
}

void DeviceBuffer::copy_to_host(void* destination, std::size_t count, std::size_t offset) const {
    const auto source = span().subspan(offset, count);
    if (count == 0) { return; }
    if (destination == nullptr) { throw std::invalid_argument("readback destination is null"); }
    CudaDeviceScope scope(device_);
    check(cudaMemcpy(destination, source.data, count, cudaMemcpyDeviceToHost), "buffer readback");
}

PinnedHostBuffer::PinnedHostBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes != 0) { check(cudaMallocHost(&data_, bytes), "cudaMallocHost"); }
}

PinnedHostBuffer::~PinnedHostBuffer() { release_pinned(data_); }

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this != &other) {
        release_pinned(data_);
        data_  = std::exchange(other.data_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
}

CudaHostView PinnedHostBuffer::cuda_view(Device cuda_device) const {
    return map_host(span(), cuda_device);
}

} // namespace ninfer
