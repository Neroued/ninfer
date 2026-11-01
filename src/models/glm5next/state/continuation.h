#pragma once
#include "runtime/contract/placement.h"

#include <optional>

namespace ninfer::models::glm5next {
struct ContinuationMetadata {
    std::uint32_t main_frontier = 0;
    runtime::PlacementId placement;
    std::optional<std::uint32_t> mtp_frontier;
};

[[nodiscard]] inline std::uint32_t mtp_history_frontier(std::uint32_t f) { return f ? f - 1 : 0; }
} // namespace ninfer::models::glm5next
