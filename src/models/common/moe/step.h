#pragma once

#include <cstdint>
#include <vector>

namespace ninfer::models::moe {

// Ordered traversals of a shared expert domain. The prefix runs the native proposal
// banks and each remaining traversal visits the target banks once in model order.
struct ExpertStep {
    std::uint64_t id           = 0;
    std::uint32_t draft_passes = 0;
    std::vector<std::uint32_t> columns;
};

struct ExpertServiceWork {};

struct ExpertServiceResult {
    std::uint64_t step;
};

} // namespace ninfer::models::moe
