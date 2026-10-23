#pragma once

#include "core/device.h"

namespace ninfer::runtime {

[[nodiscard]] std::size_t device_budget(std::size_t available);

// Startup has one Device owner. Snapshots are taken with its transfers and work settled.
// Re-read after destroying a candidate: retained driver storage is already charged.
// Headroom cushions planning; the Host reservation must remain after initialization.
class StartupMemoryBudget {
public:
    StartupMemoryBudget(DeviceMemorySnapshot initial, std::size_t limit, std::size_t headroom,
                        std::size_t host_reservation);
    [[nodiscard]] std::size_t available(DeviceMemorySnapshot current) const;
    [[nodiscard]] std::size_t available_host(DeviceMemorySnapshot current) const noexcept;
    [[nodiscard]] bool fits(DeviceMemorySnapshot current) const noexcept;
    [[nodiscard]] std::int64_t retained(DeviceMemorySnapshot current,
                                        std::size_t owned_bytes) const;

    [[nodiscard]] std::size_t limit() const noexcept { return limit_; }

    [[nodiscard]] std::size_t headroom() const noexcept {
        return headroom_ + (initial_.shared_physical_memory ? host_reservation_ : 0);
    }

private:
    DeviceMemorySnapshot initial_;
    std::size_t limit_, headroom_, host_reservation_;
};

} // namespace ninfer::runtime
