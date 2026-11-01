#pragma once

#include "models/common/moe/schedule.h"
#include "models/common/moe/offload/execution.h"

#include <array>
#include <span>
#include <vector>

namespace ninfer::models::moe::offload {

struct LayerPlan {
    LayerSpec spec;
    std::int32_t hybrid_max_tokens = 0;
    std::uint32_t fixed_experts = 0, cache_experts = 0;
    std::vector<ExpertPlacement> stream_layout;

    [[nodiscard]] bool requires_full_bank(std::int32_t tokens) const noexcept {
        return tokens > hybrid_max_tokens;
    }
};

// Layers follow the model's MoE invocation order. Banks may include resident components
// outside that traversal. Each scheduled bank occurs once; model layer numbers stay external.
struct Plan {
    std::vector<LayerPlan> layers;
    ExpertCacheRequirements cache;
    ExecutionRequirements bridge;
    CpuTeamPlan team;
    CpuAffinity caller_cpus;
    std::vector<ExpertPlacement> fixed_layout, extra_layout;
    std::array<LayoutRegion, 2> windows;
    std::uint32_t window_count = 0;
    std::size_t device_bytes = 0, target_bytes = 0;

    [[nodiscard]] std::size_t layer_index(ExpertBankId bank) const;
    [[nodiscard]] const LayerPlan& layer(ExpertBankId bank) const;

private:
    friend Plan describe_plan(std::span<const ExpertBankCacheSpec>, std::span<const LayerSpec>);
    std::vector<std::size_t> layer_indices_;
};

[[nodiscard]] Plan describe_plan(std::span<const ExpertBankCacheSpec> banks,
                                 std::span<const LayerSpec> layers);

// Resolve fixed per-layer shares and streaming capacity together, then fill the reusable
// span with small-T experts. False means even one streaming window cannot fit.
[[nodiscard]] bool plan_cache(Plan&);
[[nodiscard]] std::optional<Plan> resolve_plan(Plan, std::span<const ExpertBankCacheSpec> banks,
                                               std::size_t available_bytes, std::int32_t max_tokens,
                                               const CpuTopology&, std::uint32_t workers);

[[nodiscard]] std::shared_ptr<const moe::Plan>
    describe_schedule(std::span<const ExpertBankCacheSpec>, std::span<const LayerSpec>);

} // namespace ninfer::models::moe::offload
