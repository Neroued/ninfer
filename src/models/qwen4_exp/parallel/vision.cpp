#include "models/qwen4_exp/parallel/vision.h"

#include <algorithm>
#include <numeric>

namespace ninfer::models::qwen4_exp {

std::vector<std::uint32_t> vision_owners(const qwen::VisionExecutionInput& input,
                                         std::uint32_t ranks) {
    if (!ranks) throw std::invalid_argument("Vision placement has no ranks");
    std::vector<std::uint32_t> owners(input.vision_items.size(), UINT32_MAX);
    std::vector<std::uint64_t> load(ranks);
    auto uses       = input.plan.uses;
    const auto cost = [&](const auto& use) {
        const auto& c = input.plan.control->items.at(use.control_index);
        return std::uint64_t(c.patch_count) * (4096 + c.segment_length);
    };
    std::stable_sort(uses.begin(), uses.end(),
                     [&](const auto& a, const auto& b) { return cost(a) > cost(b); });
    for (const auto& use : uses) {
        const auto rank =
            static_cast<std::uint32_t>(std::min_element(load.begin(), load.end()) - load.begin());
        owners.at(use.prepared_item_index) = rank;
        load[rank] += cost(use);
    }
    return owners;
}

} // namespace ninfer::models::qwen4_exp
