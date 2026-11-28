#pragma once

#include <algorithm>
#include <bit>
#include <limits>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::models::moe {

// The prepared wire geometry travels with the rank description. A zero maximum
// denotes a rank without a cross-replica expert service.
struct ExchangeGeometry {
    std::uint32_t max_block_columns                            = 0;
    std::uint32_t blocks_per_window                            = 1;
    friend bool operator==(ExchangeGeometry, ExchangeGeometry) = default;

    [[nodiscard]] std::uint32_t window_columns() const {
        const auto n = std::uint64_t(max_block_columns) * blocks_per_window;
        if (!n || n > std::numeric_limits<int>::max())
            throw std::invalid_argument("invalid expert window capacity");
        return static_cast<std::uint32_t>(n);
    }

    [[nodiscard]] std::uint32_t packet_columns(std::uint32_t columns) const {
        (void)window_columns();
        if (!columns || columns > std::numeric_limits<int>::max())
            throw std::invalid_argument("expert exchange needs a positive column extent");
        return columns < max_block_columns ? std::min(max_block_columns, std::bit_ceil(columns))
                                           : max_block_columns;
    }

    // Equal keys have identical packet lengths, packet counts and window boundaries.
    [[nodiscard]] std::uint32_t capture_bucket(std::uint32_t columns) const {
        const auto b = packet_columns(columns);
        return ((columns - 1) / b + 1) * b;
    }

    [[nodiscard]] unsigned variant(std::uint32_t columns) const {
        const auto width = packet_columns(columns);
        return std::bit_width(width - 1) + (columns > max_block_columns && blocks_per_window > 1);
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
