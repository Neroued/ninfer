#include "core/memory.h"

#include <stdexcept>

namespace ninfer {

Device Device::cuda(int ordinal) {
    if (ordinal < 0) { throw std::invalid_argument("CUDA device ordinal must be nonnegative"); }
    return {DeviceType::CUDA, ordinal};
}

void validate_memory(ConstMemorySpan memory) {
    if (memory.bytes != 0 && memory.data == nullptr) {
        throw std::invalid_argument("memory span has no backing");
    }
    if (memory.data != nullptr && memory.device.type() == DeviceType::None) {
        throw std::invalid_argument("memory span has no device");
    }
    if (memory.host_pinned && memory.device != Device::cpu()) {
        throw std::invalid_argument("pinned memory must reside on CPU");
    }
}

ConstMemorySpan ConstMemorySpan::subspan(std::size_t offset, std::size_t count) const {
    if (offset > bytes || count > bytes - offset) {
        throw std::out_of_range("memory subspan exceeds backing");
    }
    const auto* start = static_cast<const std::byte*>(data);
    return {start == nullptr ? nullptr : start + offset, count, device, host_pinned};
}

MemorySpan MemorySpan::subspan(std::size_t offset, std::size_t count) const {
    const auto part = static_cast<ConstMemorySpan>(*this).subspan(offset, count);
    return {const_cast<void*>(part.data), part.bytes, part.device, part.host_pinned};
}

CudaHostView CudaHostView::subspan(std::size_t offset, std::size_t count) const {
    const auto part    = host.subspan(offset, count);
    const auto* mapped = static_cast<const std::byte*>(device_address);
    return {part, mapped ? mapped + offset : nullptr, cuda_device};
}

} // namespace ninfer
