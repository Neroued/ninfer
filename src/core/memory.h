#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer {

enum class DeviceType : std::uint8_t { None, CPU, CUDA };

class Device {
public:
    constexpr Device() noexcept = default;

    [[nodiscard]] static constexpr Device cpu() noexcept { return {DeviceType::CPU, 0}; }

    [[nodiscard]] static Device cuda(int ordinal);

    [[nodiscard]] constexpr DeviceType type() const noexcept { return type_; }

    [[nodiscard]] constexpr int index() const noexcept { return index_; }

    constexpr bool operator==(const Device&) const noexcept = default;

private:
    constexpr Device(DeviceType type, int index) noexcept : type_(type), index_(index) {}

    DeviceType type_ = DeviceType::None;
    int index_       = -1;
};

struct ConstMemorySpan {
    constexpr ConstMemorySpan() noexcept = default;

    constexpr ConstMemorySpan(const void* data, std::size_t bytes, Device device,
                              bool host_pinned = false) noexcept
        : data(data), bytes(bytes), device(device), host_pinned(host_pinned) {}

    const void* data  = nullptr;
    std::size_t bytes = 0;
    Device device;
    bool host_pinned = false;

    [[nodiscard]] ConstMemorySpan subspan(std::size_t offset, std::size_t count) const;

    [[nodiscard]] ConstMemorySpan subspan(std::size_t offset) const {
        return subspan(offset, offset <= bytes ? bytes - offset : 0);
    }
};

struct MemorySpan {
    constexpr MemorySpan() noexcept = default;

    constexpr MemorySpan(void* data, std::size_t bytes, Device device,
                         bool host_pinned = false) noexcept
        : data(data), bytes(bytes), device(device), host_pinned(host_pinned) {}

    void* data        = nullptr;
    std::size_t bytes = 0;
    Device device;
    bool host_pinned = false;

    [[nodiscard]] MemorySpan subspan(std::size_t offset, std::size_t count) const;

    [[nodiscard]] MemorySpan subspan(std::size_t offset) const {
        return subspan(offset, offset <= bytes ? bytes - offset : 0);
    }

    [[nodiscard]] operator ConstMemorySpan() const noexcept {
        return {data, bytes, device, host_pinned};
    }
};

// A pinned CPU range and its address in one CUDA device's mapping. The backing and
// registration outlive this view and every transfer using it.
struct CudaHostView {
    ConstMemorySpan host;
    const void* device_address = nullptr;
    Device cuda_device;

    [[nodiscard]] CudaHostView subspan(std::size_t offset, std::size_t count) const;
};

// Checks backing and location, including pinned memory's CPU residency.
void validate_memory(ConstMemorySpan memory);

} // namespace ninfer
