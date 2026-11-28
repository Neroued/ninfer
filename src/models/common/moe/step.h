#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::models::moe {

// The prepared wire geometry travels with the rank description. A zero maximum
// denotes a rank without a cross-replica expert service.
struct ExchangeGeometry {
    std::uint32_t max_block_columns                            = 0;
    friend bool operator==(ExchangeGeometry, ExchangeGeometry) = default;

    [[nodiscard]] std::uint32_t blocks(std::uint32_t columns) const {
        if (!columns || !max_block_columns)
            throw std::invalid_argument("expert exchange needs a prepared positive extent");
        return 1 + (columns - 1) / max_block_columns;
    }

    // Equal keys imply identical send and return envelopes for every bank of this pass.
    [[nodiscard]] std::uint32_t capture_bucket(std::uint32_t columns) const {
        return blocks(columns);
    }
};

// Ordered traversals of a shared expert domain. The prefix runs the native proposal
// banks and each remaining traversal visits the target banks once in model order.
struct ExpertStep {
    std::uint64_t id           = 0;
    std::uint32_t draft_passes = 0;
    std::vector<std::uint32_t> columns;
    bool graph = false;

    [[nodiscard]] bool covers(const ExpertStep& source, ExchangeGeometry geometry) const {
        if (graph != source.graph || draft_passes != source.draft_passes ||
            columns.size() != source.columns.size())
            return false;
        return std::equal(source.columns.begin(), source.columns.end(), columns.begin(),
                          [this, geometry](auto actual, auto matched) {
                              return actual <= matched &&
                                     (!graph || geometry.capture_bucket(actual) ==
                                                    geometry.capture_bucket(matched));
                          });
    }
};

struct ExpertServiceWork {};

struct ExpertServiceResult {
    std::uint64_t step;
};

} // namespace ninfer::models::moe
