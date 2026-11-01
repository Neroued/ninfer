#pragma once

#include "runtime/contract/placement.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::runtime {

struct PlacementLoad {
    PlacementId placement;
    std::uint64_t tokens   = 0;
    std::uint32_t requests = 0;
};

struct PlacementCandidate {
    PlacementId placement;
    std::uint64_t local_prefix_tokens = 0;
    bool feasible                     = false;
};

// Return feasible candidate indexes. Within 1.5 times the minimum resulting peak token load,
// prefer local prefixes, then lighter loads. Remaining candidates follow peak load order.
// Loads contain each placement exactly once; all candidate placements must be present.
// Equal choices rotate through increasing placement IDs starting at first_tie, then wrap.
[[nodiscard]] std::vector<std::size_t>
order_placements(std::span<const PlacementLoad> loads,
                 std::span<const PlacementCandidate> candidates, std::uint64_t request_tokens,
                 PlacementId first_tie = {});

} // namespace ninfer::runtime
