#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::models::moe {

inline constexpr int expert_exchange_block_columns = 64;

inline int exchange_blocks(int columns, int block_columns = expert_exchange_block_columns) {
    if (columns <= 0 || block_columns <= 0)
        throw std::invalid_argument("expert exchange needs a positive block extent");
    return (columns - 1) / block_columns + 1;
}

// Ordered traversals of a shared expert domain. The prefix runs the native proposal
// banks and each remaining traversal visits the target banks once in model order.
struct ExpertStep {
    std::uint64_t id           = 0;
    std::uint32_t draft_passes = 0;
    std::vector<std::uint32_t> columns;
    bool graph = false;

    [[nodiscard]] bool covers(const ExpertStep& source) const {
        if (graph != source.graph || draft_passes != source.draft_passes ||
            columns.size() != source.columns.size())
            return false;
        return std::equal(source.columns.begin(), source.columns.end(), columns.begin(),
                          [this](auto actual, auto matched) {
                              return actual <= matched && (!graph || exchange_blocks(actual) ==
                                                                         exchange_blocks(matched));
                          });
    }
};

struct ExpertServiceWork {};

struct ExpertServiceResult {
    std::uint64_t step;
};

} // namespace ninfer::models::moe
