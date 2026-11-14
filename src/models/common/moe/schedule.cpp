#include "models/common/moe/schedule.h"
#include "core/platform.h"
#include <climits>
#include <stdexcept>

#if NINFER_EXPERT_OFFLOAD
#    include "models/common/moe/offload/plan.h"
#endif

namespace ninfer::models::moe {

TokenGeometry decode_geometry(const BatchExecutionPlan& plan, std::uint32_t rows,
                              std::uint32_t width) {
    const auto minimum = plan.minimum_rows(rows);
    const auto tokens  = std::uint64_t(rows) * width;
    if (!width || tokens > INT_MAX) throw std::invalid_argument("MoE token geometry overflow");
    return {static_cast<std::int32_t>(tokens), static_cast<std::int32_t>(minimum)};
}

std::vector<TokenGeometry> decode_geometries(const BatchExecutionPlan& plan, std::uint32_t width) {
    std::vector<TokenGeometry> out;
    const auto add = [&](std::uint32_t rows) {
        out.push_back(decode_geometry(plan, rows, 1));
        if (width > 1) out.push_back(decode_geometry(plan, rows, width));
    };
    if (plan.graph_rows().empty()) {
        for (std::uint32_t rows = 1; rows <= plan.resident_capacity(); ++rows) add(rows);
    } else {
        for (const auto rows : plan.graph_rows()) add(rows);
    }
    return out;
}

std::shared_ptr<const Plan> describe_plan(std::span<const ExpertBankCacheSpec> banks,
                                          std::span<const LayerSpec> layers) {
#if NINFER_EXPERT_OFFLOAD
    return offload::describe_schedule(banks, layers);
#else
    return {};
#endif
}

} // namespace ninfer::models::moe
