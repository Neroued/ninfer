#include "runtime/engine/startup_memory.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <stdexcept>

namespace ninfer::runtime {

std::size_t device_budget(std::size_t available) {
    const char* value = std::getenv("NINFER_VRAM_BUDGET_MIB");
    if (!value) return available;
    const std::string_view text(value);
    std::uint64_t mib = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), mib);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !mib ||
        mib > std::numeric_limits<std::size_t>::max() / (1024ULL * 1024))
        throw std::invalid_argument("NINFER_VRAM_BUDGET_MIB requires a positive MiB integer");
    return std::min(available, static_cast<std::size_t>(mib * 1024 * 1024));
}

StartupMemoryBudget::StartupMemoryBudget(DeviceMemorySnapshot initial, std::size_t limit,
                                         std::size_t headroom, std::size_t host_reservation)
    : initial_(initial), limit_(std::min(initial.available_bytes(), limit)), headroom_(headroom),
      host_reservation_(host_reservation) {
    if (limit_ <= headroom_ ||
        (initial_.shared_physical_memory && host_reservation_ >= limit_ - headroom_))
        throw std::invalid_argument("Device budget is smaller than runtime headroom");
    if (host_reservation_ > initial_.system_available_bytes)
        throw std::invalid_argument("Host reservation exceeds available RAM");
}

std::size_t StartupMemoryBudget::available(DeviceMemorySnapshot current) const {
    const auto consumed = static_cast<std::size_t>(
        std::max<std::int64_t>(0, device_memory_growth(initial_, current)));
    const auto remaining =
        std::min(current.available_bytes(), consumed < limit_ ? limit_ - consumed : 0);
    return remaining > headroom() ? remaining - headroom() : 0;
}

bool StartupMemoryBudget::fits(DeviceMemorySnapshot current) const noexcept {
    const auto consumed       = device_memory_growth(initial_, current);
    const auto device_reserve = initial_.shared_physical_memory ? host_reservation_ : 0;
    return consumed <= static_cast<std::int64_t>(limit_ - device_reserve) &&
           current.system_available_bytes >= host_reservation_;
}

std::size_t StartupMemoryBudget::available_host(DeviceMemorySnapshot current) const noexcept {
    return current.system_available_bytes > host_reservation_
               ? current.system_available_bytes - host_reservation_
               : 0;
}

std::int64_t StartupMemoryBudget::retained(DeviceMemorySnapshot current,
                                           std::size_t owned_bytes) const {
    return device_memory_growth(initial_, current) - static_cast<std::int64_t>(owned_bytes);
}

} // namespace ninfer::runtime
