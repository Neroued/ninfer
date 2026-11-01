#include "models/qwen3_5/program/planning/graph_profiles.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {
// Resource tiers bound inactive attention work. They are not kernel/topology boundaries.
constexpr std::array<std::uint32_t, 7> kCausalVisibleTiers{128,  512,   2048, 4096,
                                                           8192, 16384, 32768};

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
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

std::vector<GraphExecutionProfile> causal_resource_profiles(std::uint32_t capacity,
                                                            std::uint32_t visible_offset) {
    std::vector<std::uint32_t> ends;
    for (const auto visible : kCausalVisibleTiers)
        if (visible >= visible_offset) ends.push_back(visible - visible_offset);
    return graph_profiles_through(capacity - 1, ends);
}

} // namespace

std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window; all tiers share one topology per exact B.
    return causal_resource_profiles(capacity, 1);
}

std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                      std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // The final AR call can see E+2K. Target verify and MTP attention are update-compatible
    // across all resource tiers, independently of the selected KV representation.
    return causal_resource_profiles(capacity, 2 * draft_window);
}

std::vector<GraphExecutionProfile> dflash_graph_profiles(std::uint32_t capacity,
                                                         std::uint32_t draft_window) {
    if (capacity == 0 || draft_window == 0 || draft_window > 15) {
        throw std::invalid_argument("invalid masked draft graph dimensions");
    }
    auto profiles = graph_profiles_through(capacity - 1, {96, 511, 2047, 8191, 32767});
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        profiles[i].topology_class = static_cast<std::uint32_t>(i);
    }
    return profiles;
}

} // namespace ninfer::models::qwen3_5::detail
