#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>

namespace ninfer::runtime {

// The model supplies priority tiers and tie order. The capacity estimate uses a common
// physical baseline before candidate-specific communication and graph resources exist.
struct PlacementEstimate {
    std::size_t id, tier, preference, blocks;
};

// prepare returns the measured capacity or an empty result at a safe resource failure.
// release retires the entire attempt, including a partially installed placement, before
// another prepare starts. On success the returned placement is still installed.
template <class Prepare, class Release>
std::optional<std::size_t> select_prepared_placement(std::vector<PlacementEstimate> estimates,
                                                     Prepare prepare, Release release) {
    const auto better = [](const PlacementEstimate& a, const PlacementEstimate& b) {
        if (a.tier != b.tier) return a.tier < b.tier;
        if (a.blocks != b.blocks) return a.blocks > b.blocks;
        return a.preference < b.preference;
    };
    std::stable_sort(estimates.begin(), estimates.end(), better);
    std::vector<PlacementEstimate> measured;
    for (std::size_t i = 0; i < estimates.size(); ++i) {
        auto candidate = estimates[i];
        if (!measured.empty() && !better(candidate, measured.front())) break;
        if (auto blocks = prepare(candidate.id)) {
            candidate.blocks = *blocks;
            measured.push_back(candidate);
            std::stable_sort(measured.begin(), measured.end(), better);
            if (measured.front().id == candidate.id &&
                (i + 1 == estimates.size() || !better(estimates[i + 1], candidate)))
                return candidate.id;
        }
        release();
    }
    for (const auto& previous : measured) {
        if (prepare(previous.id)) return previous.id;
        release();
    }
    return {};
}

} // namespace ninfer::runtime
