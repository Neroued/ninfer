#pragma once

#include "core/device.h"

namespace ninfer::runtime {

[[nodiscard]] std::size_t device_budget(std::size_t available);
[[nodiscard]] std::size_t host_memory_shortfall(DeviceMemorySnapshot, std::size_t reservation);
[[nodiscard]] std::int64_t retained_startup_bytes(DeviceMemorySnapshot before,
                                                  DeviceMemorySnapshot after,
                                                  std::size_t owned_bytes);

// Startup has one Device owner. Snapshots are taken with its transfers and work settled.
// Re-read after destroying a candidate: retained driver storage is already charged.
// Headroom cushions planning; the Host reservation must remain after initialization.
class StartupMemoryBudget {
public:
    StartupMemoryBudget(DeviceMemorySnapshot initial, std::size_t limit, std::size_t headroom,
                        std::size_t host_reservation);
    // Both allowances preserve mandatory Host capacity. Only planning also reserves headroom.
    [[nodiscard]] std::size_t planning_available(DeviceMemorySnapshot current) const noexcept;
    [[nodiscard]] std::size_t allocation_available(DeviceMemorySnapshot current) const noexcept;
    [[nodiscard]] std::size_t available_host(DeviceMemorySnapshot current) const noexcept;
    [[nodiscard]] bool fits(DeviceMemorySnapshot current) const noexcept;
    [[nodiscard]] std::size_t shortfall(DeviceMemorySnapshot current) const noexcept;
    // The caller supplies the full outstanding node reservation when the pool is shared.
    [[nodiscard]] std::size_t
    device_shortfall(DeviceMemorySnapshot current, std::size_t node_reservation,
                     std::size_t additional_device_bytes = 0) const noexcept;
    // Tightens the next explicit runtime offer after a completed over-budget preparation.
    // Separate Host shortages cannot be recovered by reducing Device backing.
    [[nodiscard]] std::size_t corrected_runtime_limit(DeviceMemorySnapshot current,
                                                      std::size_t previous_runtime_bytes) const;
    [[nodiscard]] std::int64_t retained(DeviceMemorySnapshot current,
                                        std::size_t owned_bytes) const;

    [[nodiscard]] std::size_t limit() const noexcept { return limit_; }

    [[nodiscard]] std::size_t headroom() const noexcept { return headroom_; }

    [[nodiscard]] std::size_t host_reservation() const noexcept { return host_reservation_; }

private:
    DeviceMemorySnapshot initial_;
    std::size_t limit_, headroom_, host_reservation_;
};

} // namespace ninfer::runtime
