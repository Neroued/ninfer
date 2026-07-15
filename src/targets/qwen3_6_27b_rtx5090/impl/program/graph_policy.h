#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ninfer::targets::qwen3_6_27b_rtx5090::detail {

struct GraphFrontierRange {
    std::uint32_t min = 0;
    std::uint32_t max = 0;
};

inline std::vector<GraphFrontierRange>
graph_ranges_through(std::uint32_t max_frontier, const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphFrontierRange> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

inline std::vector<GraphFrontierRange> ordinary_graph_ranges(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    static const std::vector<std::uint32_t> ends = {
        127, 511, 2047, 4095, 8197, 16389, 32767,
    };
    return graph_ranges_through(capacity - 1, ends);
}

inline std::vector<GraphFrontierRange> mtp_graph_ranges(std::uint32_t capacity,
                                                        std::uint32_t draft_window) {
    if (draft_window == 0 || 2ULL * draft_window > capacity) { return {}; }
    // Bound the final AR window E+2K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    // Target verify and MTP batch both have T=K+1 and W=E+K+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_ranges_through(capacity - 2 * draft_window, ends);
}

} // namespace ninfer::targets::qwen3_6_27b_rtx5090::detail
