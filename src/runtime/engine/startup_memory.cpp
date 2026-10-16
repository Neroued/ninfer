#include "runtime/engine/startup_memory.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::runtime {

StartupMemoryBudget::StartupMemoryBudget(DeviceMemorySnapshot initial, std::size_t limit,
                                         std::size_t headroom, std::size_t host_reservation)
    : initial_(initial), limit_(std::min(initial.free_bytes, limit)), headroom_(headroom),
      host_reservation_(host_reservation) {
    if (limit_ <= headroom_ || host_reservation_ >= limit_ - headroom_)
        throw std::invalid_argument("Device budget is smaller than runtime headroom");
}

std::size_t StartupMemoryBudget::available(DeviceMemorySnapshot current) const {
    const auto consumed = static_cast<std::size_t>(
        std::max<std::int64_t>(0, device_memory_growth(initial_, current)));
    const auto remaining = std::min(current.free_bytes, consumed < limit_ ? limit_ - consumed : 0);
    return remaining > headroom() ? remaining - headroom() : 0;
}

bool StartupMemoryBudget::fits(DeviceMemorySnapshot current) const noexcept {
    const auto consumed = device_memory_growth(initial_, current);
    return consumed <= static_cast<std::int64_t>(limit_ - host_reservation_);
}

std::int64_t StartupMemoryBudget::retained(DeviceMemorySnapshot current,
                                           std::size_t owned_bytes) const {
    return device_memory_growth(initial_, current) - static_cast<std::int64_t>(owned_bytes);
}

} // namespace ninfer::runtime
