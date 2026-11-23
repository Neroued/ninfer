#pragma once

#include "models/common/speculative/cost_model.h"

#include <algorithm>
#include <bit>

namespace ninfer::models::speculative::detail {

inline double log_context(ops::ContextBucket bucket) {
    // The bucket ceilings are powers of two. Use the Op's ceiling table.
    return std::bit_width(ops::context_bucket_tokens(bucket)) - 12;
}

inline CostQuery route_query(const RoundRoute& route, std::uint32_t rows,
                             std::uint32_t max_owner_rows) {
    return {.shape          = route.shape,
            .logical_rows   = rows,
            .max_owner_rows = max_owner_rows,
            .width          = route.width,
            .mtp_steps      = route.mtp_steps,
            .head_columns   = route.head_columns};
}

inline void use_proposal_cost(CostQuery& target, const CostQuery& proposal) {
    target.mtp_steps               = proposal.mtp_steps;
    target.head_columns            = proposal.head_columns;
    target.active_proposal_rows    = proposal.active_proposal_rows;
    target.max_owner_proposal_rows = proposal.max_owner_proposal_rows;
    target.proposal_bucket         = proposal.proposal_bucket;
}

inline ops::ContextBucket visible_bucket(std::uint32_t frontier, std::uint64_t extra,
                                         std::uint32_t max_context) {
    const auto tokens = std::min<std::uint64_t>(
        max_context, std::max<std::uint64_t>(1, std::uint64_t(frontier) + extra));
    return ops::select_context_bucket(static_cast<std::uint32_t>(tokens));
}

inline ops::ContextBucket target_bucket(std::uint32_t frontier, std::uint32_t extent,
                                        std::uint32_t max_context) {
    return visible_bucket(frontier, std::uint64_t(extent) + 1, max_context);
}

// Called for an active proposal row on a validated speculative route.
inline ops::ContextBucket proposal_bucket(SpeculativeBackend backend, const RoundRoute& route,
                                          std::uint32_t frontier, std::uint32_t max_context) {
    const auto extra = backend == SpeculativeBackend::Mtp ? std::uint64_t(route.mtp_steps) - 1
                                                          : std::uint64_t(route.head_columns) + 1;
    return visible_bucket(frontier, extra, max_context);
}

// Hoist invariant terms of the feature dot product out of the row trial loop.
class RouteCost {
public:
    RouteCost(const SpeculativeCostModel& model, const CostQuery& route);

    [[nodiscard]] bool has_concave_load_cost() const {
        return !threshold_ && model_.online_.empty();
    }

    [[nodiscard]] double base_predict_ms(const CostQuery& q) const {
        const auto& w        = model_.weights_;
        const auto target    = q.max_owner_target_tokens;
        const auto proposals = q.max_owner_proposal_rows;
        double cost          = fixed_ms_ + proposals * proposal_ms_;
        if (threshold_) {
            const auto active = q.active_target_tokens;
            if (active > threshold_) {
                cost += w[8] + (active - threshold_) * w[9];
            } else {
                cost += active * w[6];
                if (w[7]) cost += experts(active) * w[7];
            }
        } else {
            cost += target * w[6];
            if (resident_expert_ms_) {
                cost += experts(target) * resident_expert_ms_;
                if (mtp_expert_ms_) cost += experts(proposals) * mtp_expert_ms_;
            }
        }
        if (q.target_bucket != ops::ContextBucket::K2)
            cost += (target * w[11] + target_column_ms_) * log_context(q.target_bucket);
        if (q.proposal_bucket != ops::ContextBucket::K2)
            cost += proposals * proposal_context_ms_ * log_context(q.proposal_bucket);
        return std::max(0.05, cost);
    }

    [[nodiscard]] double predict_ms(const CostQuery& q) const {
        const auto cost = base_predict_ms(q);
        return model_.online_.empty() ? cost : cost * model_.cell_multiplier(q);
    }

private:
    [[nodiscard]] double experts(std::uint32_t tokens) const {
        if (tokens < model_.expert_counts_.size()) return model_.expert_counts_[tokens];
        return model_.expected_experts(tokens);
    }

    const SpeculativeCostModel& model_;
    double fixed_ms_;
    std::uint32_t threshold_;
    double resident_expert_ms_, mtp_expert_ms_, target_column_ms_;
    double proposal_ms_, proposal_context_ms_;
};

} // namespace ninfer::models::speculative::detail
