#pragma once

#include "runtime/contract/placement.h"

#include <cstdint>
#include <optional>

namespace ninfer::models::inkling2 {

struct ContinuationMetadata {
    std::uint32_t main_frontier = 0;
    std::optional<std::uint32_t> dspark_frontier;
    runtime::PlacementId placement;
};

} // namespace ninfer::models::inkling2
