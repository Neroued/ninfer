#pragma once

#include <cstdint>

namespace ninfer::runtime {

// Placement is local to one execution instance. Its physical ranks belong to the model plan.
struct PlacementId {
    std::uint32_t value = 0;

    friend constexpr bool operator==(PlacementId, PlacementId) noexcept = default;
};

enum class PauseKind : std::uint8_t {
    KeepDevice,
    HostSnapshot,
    Relocate,
    ReleaseAndReplay,
};

struct PauseAction {
    PauseKind kind = PauseKind::ReleaseAndReplay;
    PlacementId placement;

    friend constexpr bool operator==(PauseAction, PauseAction) noexcept = default;
};

} // namespace ninfer::runtime
