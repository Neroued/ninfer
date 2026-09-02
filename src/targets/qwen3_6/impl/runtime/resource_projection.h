#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::targets::qwen3_6::detail {

// Concrete allocator quantities remain inside the Program boundary. These values describe one
// complete physical state or transition; ResourceManager never observes or reconstructs them.
struct PhysicalDeviceResources {
    std::uint32_t active_lanes     = 0;
    std::uint32_t state_slots      = 0;
    std::uint32_t main_kv_pages    = 0;
    std::uint32_t backend_kv_pages = 0;

    [[nodiscard]] friend constexpr bool operator==(PhysicalDeviceResources,
                                                   PhysicalDeviceResources) noexcept = default;
};

struct PhysicalHostResources {
    std::uint32_t state_slots = 0;
    std::size_t kv_bytes      = 0;

    [[nodiscard]] friend constexpr bool operator==(PhysicalHostResources,
                                                   PhysicalHostResources) noexcept = default;
};

struct PhysicalResources {
    PhysicalDeviceResources device;
    PhysicalHostResources host;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalResources&,
                                                   const PhysicalResources&) noexcept = default;
};

struct PhysicalDemand {
    PhysicalResources active_entitlement;
    PhysicalResources reservation_added;
    PhysicalResources reservation_credit;
    PhysicalResources physical_peak_additional;
    PhysicalResources final_removed;
    PhysicalResources final_added;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalDemand&,
                                                   const PhysicalDemand&) noexcept = default;
};

struct PhysicalDelta {
    PhysicalResources removed;
    PhysicalResources added;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalDelta&,
                                                   const PhysicalDelta&) noexcept = default;
};

// Exact result of evaluating one complete target against the joint State/KV reference graph.
// It is deliberately not a sum of owner-local deltas: aliases and last-reference releases are
// settled once by unique physical identity before this value is produced.
struct PressureTargetProjection {
    PhysicalDelta unique_object_delta;
    PhysicalDelta ownership_transfer_delta;
    PhysicalDelta active_entitlement_delta;
    PhysicalResources source_optional_resources_added;
    std::optional<bool> source_state_fork_required;
    std::optional<bool> source_text_prefix_fork_required;
    std::optional<bool> source_backend_prefix_fork_required;

    [[nodiscard]] friend constexpr bool
    operator==(const PressureTargetProjection&, const PressureTargetProjection&) noexcept = default;
};

} // namespace ninfer::targets::qwen3_6::detail
