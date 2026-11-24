#pragma once

#include "core/memory.h"

#include <cstddef>

namespace ninfer {

class HostBuffer {
public:
    HostBuffer() noexcept = default;
    explicit HostBuffer(std::size_t bytes, std::size_t alignment = 256);
    ~HostBuffer();
    HostBuffer(const HostBuffer&)            = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    HostBuffer(HostBuffer&& other) noexcept;
    HostBuffer& operator=(HostBuffer&& other) noexcept;

    [[nodiscard]] void* data() noexcept { return data_; }

    [[nodiscard]] const void* data() const noexcept { return data_; }

    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

    [[nodiscard]] Device device() const noexcept { return Device::cpu(); }

    [[nodiscard]] MemorySpan span() noexcept { return {data_, bytes_, device()}; }

    [[nodiscard]] ConstMemorySpan span() const noexcept { return {data_, bytes_, device()}; }

private:
    void* data_            = nullptr;
    std::size_t bytes_     = 0;
    std::size_t alignment_ = 256;
};

// Owns CUDA registration, not the Host allocation. The caller finishes all uses
// before unregistering and keeps the original Host backing alive until then.
class CudaHostRegistration {
public:
    CudaHostRegistration() noexcept = default;
    CudaHostRegistration(ConstMemorySpan host, Device cuda_device);
    ~CudaHostRegistration();
    CudaHostRegistration(const CudaHostRegistration&)            = delete;
    CudaHostRegistration& operator=(const CudaHostRegistration&) = delete;
    CudaHostRegistration(CudaHostRegistration&& other) noexcept;
    CudaHostRegistration& operator=(CudaHostRegistration&& other) noexcept;

    [[nodiscard]] CudaHostView view() const noexcept { return view_; }

private:
    CudaHostView view_;
};

class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;
    DeviceBuffer(Device device, std::size_t bytes);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    [[nodiscard]] void* data() noexcept { return data_; }

    [[nodiscard]] const void* data() const noexcept { return data_; }

    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

    [[nodiscard]] Device device() const noexcept { return device_; }

    [[nodiscard]] MemorySpan span() noexcept { return {data_, bytes_, device_}; }

    [[nodiscard]] ConstMemorySpan span() const noexcept { return {data_, bytes_, device_}; }

    // Synchronous initialization/readback. The caller orders prior accesses first.
    void fill(int byte_value = 0);
    void copy_from_host(const void* source, std::size_t count, std::size_t offset = 0);
    void copy_to_host(void* destination, std::size_t count, std::size_t offset = 0) const;

private:
    void* data_        = nullptr;
    std::size_t bytes_ = 0;
    Device device_;
};

class PinnedHostBuffer {
public:
    PinnedHostBuffer() noexcept = default;
    explicit PinnedHostBuffer(std::size_t bytes);
    ~PinnedHostBuffer();
    PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    [[nodiscard]] void* data() noexcept { return data_; }

    [[nodiscard]] const void* data() const noexcept { return data_; }

    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

    [[nodiscard]] Device device() const noexcept { return Device::cpu(); }

    [[nodiscard]] MemorySpan span() noexcept { return {data_, bytes_, device(), true}; }

    [[nodiscard]] ConstMemorySpan span() const noexcept { return {data_, bytes_, device(), true}; }

    // Establish once at the owning resource's initialization boundary.
    [[nodiscard]] CudaHostView cuda_view(Device cuda_device) const;

private:
    void* data_        = nullptr;
    std::size_t bytes_ = 0;
};

} // namespace ninfer
