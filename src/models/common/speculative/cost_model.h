#pragma once

#include "models/common/speculative/round_plan.h"
#include "ninfer/ops/context_bucket.h"

#include <array>

namespace ninfer::models::speculative {

namespace detail {
class RouteCost;
}

struct CostModelConfig {
    SpeculativeBackend backend      = SpeculativeBackend::None;
    std::uint32_t max_drafts        = 0;
    std::uint32_t owner_count       = 1;
    std::uint32_t num_experts       = 0;
    std::uint32_t experts_per_token = 0;
    std::uint32_t hybrid_max_tokens = 0;
};

struct CostQuery {
    PhysicalBatchShape shape{1, 1};
    std::uint32_t logical_rows            = 1;
    std::uint32_t max_owner_rows          = 1;
    std::uint32_t width                   = 1;
    std::uint32_t mtp_steps               = 0;
    std::uint32_t head_columns            = 0;
    std::uint32_t active_target_tokens    = 1;
    std::uint32_t max_owner_target_tokens = 1;
    std::uint32_t active_proposal_rows    = 0;
    std::uint32_t max_owner_proposal_rows = 0;
    ops::ContextBucket target_bucket      = ops::ContextBucket::K2;
    ops::ContextBucket proposal_bucket    = ops::ContextBucket::K2;

    [[nodiscard]] static CostQuery from_plan(SpeculativeBackend, const DecodeRoundPlan&,
                                             std::span<const std::uint32_t> row_frontiers,
                                             std::uint32_t max_context);
    [[nodiscard]] static CostQuery from_split_plan(SpeculativeBackend,
                                                   const DecodeRoundPlan& proposal,
                                                   const DecodeRoundPlan& target,
                                                   std::span<const std::uint32_t> row_frontiers,
                                                   std::uint32_t max_context);
};

struct TimedCostAnchor {
    CostQuery query;
    double total_ms = 0.0;
};

struct CostModelSummary {
    bool fitted                   = false;
    std::size_t anchor_count      = 0;
    double calibration_elapsed_ms = 0.0;
    std::size_t online_cell_count = 0;
};

// One coordinator owns the fit and updates it once per retired step.
class SpeculativeCostModel {
public:
    using Features = std::array<double, 16>;

    explicit SpeculativeCostModel(CostModelConfig);

    [[nodiscard]] const CostModelConfig& config() const noexcept { return config_; }

    [[nodiscard]] Features features(const CostQuery&) const;

    [[nodiscard]] const Features& weights() const noexcept { return weights_; }

    void fit(std::span<const TimedCostAnchor>, double elapsed_ms = 0.0);
    [[nodiscard]] double base_predict_ms(const CostQuery&) const;
    [[nodiscard]] double predict_ms(const CostQuery&) const;
    void observe_retired_step(const CostQuery&, double total_ms);
    void reset_online() noexcept;
    [[nodiscard]] CostModelSummary summary() const noexcept;

private:
    friend class detail::RouteCost;

    struct OnlineCellKey {
        std::uint32_t rows, width, mtp_steps, head_columns;
        ops::ContextBucket target_bucket;
        bool full_resident;
        auto operator<=>(const OnlineCellKey&) const = default;
    };

    struct OnlineCell {
        double weighted_sum   = 0.0;
        double weighted_count = 0.0;
    };

    [[nodiscard]] OnlineCellKey cell_key(const CostQuery&) const noexcept;
    [[nodiscard]] Features default_weights() const;
    [[nodiscard]] double expected_experts(std::uint32_t tokens) const;
    [[nodiscard]] double cell_multiplier(const CostQuery&) const;

    CostModelConfig config_;
    std::vector<double> expert_counts_;
    Features weights_;
    CostModelSummary calibration_;
    std::map<OnlineCellKey, OnlineCell> online_;
};

} // namespace ninfer::models::speculative
