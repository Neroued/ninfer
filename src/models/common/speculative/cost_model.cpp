#include "models/common/speculative/cost_model.h"

#include "models/common/speculative/cost_query.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ninfer::models::speculative {
namespace {

double unique_experts(std::uint32_t experts, std::uint32_t per_token, std::uint32_t tokens) {
    if (!experts || !per_token || !tokens) return 0.0;
    const auto miss = double(experts - std::min(experts, per_token)) / double(experts);
    return double(experts) * (1.0 - std::pow(miss, double(tokens)));
}

bool usable_time(double total_ms) { return total_ms > 0.0 && std::isfinite(total_ms); }

} // namespace

CostQuery CostQuery::from_plan(SpeculativeBackend backend, const DecodeRoundPlan& plan,
                               std::span<const std::uint32_t> row_frontiers,
                               std::uint32_t max_context) {
    const auto extents = plan.verify_drafts();
    if (row_frontiers.size() != extents.size())
        throw std::invalid_argument("cost query frontiers differ from round membership");
    if (plan.valid_target_tokens() > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("round target tokens exceed the cost query range");

    auto query = detail::route_query(plan.route(), static_cast<std::uint32_t>(extents.size()), 0);
    query.active_target_tokens = static_cast<std::uint32_t>(plan.valid_target_tokens());
    const auto owner_tokens    = plan.owner_target_tokens();
    query.max_owner_target_tokens =
        static_cast<std::uint32_t>(*std::max_element(owner_tokens.begin(), owner_tokens.end()));
    std::vector<std::uint32_t> owner_proposals(owner_tokens.size());
    for (std::size_t i = 0; i < extents.size(); ++i) {
        const auto slot      = plan.slots()[i];
        query.max_owner_rows = std::max(query.max_owner_rows, slot.local + 1);
        query.target_bucket  = std::max(
            query.target_bucket, detail::target_bucket(row_frontiers[i], extents[i], max_context));
        if (!extents[i]) continue;
        ++query.active_proposal_rows;
        query.max_owner_proposal_rows =
            std::max(query.max_owner_proposal_rows, ++owner_proposals[slot.owner]);
        query.proposal_bucket =
            std::max(query.proposal_bucket,
                     detail::proposal_bucket(backend, plan.route(), row_frontiers[i], max_context));
    }
    return query;
}

CostQuery CostQuery::from_split_plan(SpeculativeBackend backend, const DecodeRoundPlan& proposal,
                                     const DecodeRoundPlan& target,
                                     std::span<const std::uint32_t> row_frontiers,
                                     std::uint32_t max_context) {
    if (proposal.route().shape != target.route().shape ||
        !std::ranges::equal(proposal.slots(), target.slots()))
        throw std::invalid_argument("split cost plans differ in shape or membership");
    auto query = from_plan(backend, target, row_frontiers, max_context);
    detail::use_proposal_cost(query, from_plan(backend, proposal, row_frontiers, max_context));
    return query;
}

SpeculativeCostModel::SpeculativeCostModel(CostModelConfig config)
    : config_(config), weights_(default_weights()) {
    if (config_.num_experts && config_.experts_per_token) {
        // Cover common decode loads and keep prediction available for larger batches.
        constexpr std::uint32_t cached_tokens = 2048;
        expert_counts_.reserve(cached_tokens + 1);
        for (std::uint32_t tokens = 0; tokens <= cached_tokens; ++tokens)
            expert_counts_.push_back(
                unique_experts(config_.num_experts, config_.experts_per_token, tokens));
    }
}

double SpeculativeCostModel::expected_experts(std::uint32_t tokens) const {
    if (tokens < expert_counts_.size()) return expert_counts_[tokens];
    return unique_experts(config_.num_experts, config_.experts_per_token, tokens);
}

SpeculativeCostModel::Features SpeculativeCostModel::default_weights() const {
    Features weights{1.0, .025, .008, .015, .15,  .002, .012, .007,
                     .10, .003, .003, .003, .001, .015, .004, .001};
    // The fallback has no measured crossover. Keep it increasing through the switch.
    const auto threshold = config_.hybrid_max_tokens;
    weights[8] +=
        weights[6] * threshold +
        weights[7] * unique_experts(config_.num_experts, config_.experts_per_token, threshold);
    return weights;
}

SpeculativeCostModel::Features SpeculativeCostModel::features(const CostQuery& q) const {
    const double rows           = std::max(1u, q.shape.rows);
    const double extra_columns  = double(q.width) - 1.0;
    const bool proposal         = q.mtp_steps != 0 || q.head_columns != 0;
    const bool mtp              = config_.backend == SpeculativeBackend::Mtp;
    const bool hybrid           = config_.hybrid_max_tokens > 0;
    const bool full             = hybrid && q.active_target_tokens > config_.hybrid_max_tokens;
    const double owner_tokens   = q.max_owner_target_tokens;
    const double target_context = detail::log_context(q.target_bucket);
    const double local_proposal_tokens =
        double(q.max_owner_proposal_rows) * (mtp ? q.mtp_steps : q.head_columns);
    return {1.0,
            std::log2(rows),
            double(q.shape.rows),
            q.max_owner_rows / rows,
            double(proposal),
            q.shape.rows * extra_columns,
            hybrid ? (full ? 0.0 : double(q.active_target_tokens)) : owner_tokens,
            hybrid && !full ? expected_experts(q.active_target_tokens) : 0.0,
            double(full),
            full ? double(q.active_target_tokens - config_.hybrid_max_tokens) : 0.0,
            !hybrid && config_.num_experts
                ? expected_experts(q.max_owner_target_tokens) +
                      (mtp ? q.mtp_steps * expected_experts(q.max_owner_proposal_rows) : 0.0)
                : 0.0,
            owner_tokens * target_context,
            extra_columns * target_context,
            mtp ? double(q.mtp_steps) : double(proposal),
            local_proposal_tokens,
            local_proposal_tokens * detail::log_context(q.proposal_bucket)};
}

detail::RouteCost::RouteCost(const SpeculativeCostModel& model, const CostQuery& route)
    : model_(model), threshold_(model.config_.hybrid_max_tokens) {
    auto fixed                 = route;
    fixed.active_target_tokens = fixed.max_owner_target_tokens = fixed.max_owner_proposal_rows = 0;
    fixed.target_bucket = fixed.proposal_bucket = ops::ContextBucket::K2;
    const auto x                                = model.features(fixed);
    const auto& w                               = model.weights_;
    fixed_ms_            = std::inner_product(w.begin(), w.end(), x.begin(), 0.0);
    const bool mtp       = model.config_.backend == SpeculativeBackend::Mtp;
    const auto columns   = mtp ? route.mtp_steps : route.head_columns;
    resident_expert_ms_  = model.expert_counts_.empty() ? 0.0 : w[10];
    mtp_expert_ms_       = mtp ? resident_expert_ms_ * route.mtp_steps : 0.0;
    target_column_ms_    = w[12] * (double(route.width) - 1.0);
    proposal_ms_         = w[14] * columns;
    proposal_context_ms_ = w[15] * columns;
}

void SpeculativeCostModel::fit(std::span<const TimedCostAnchor> anchors, double elapsed_ms) {
    constexpr auto columns = std::tuple_size_v<Features>;
    using Row              = std::array<long double, columns>;
    std::vector<Row> relative;
    relative.reserve(anchors.size());
    Row norms{};
    for (const auto& anchor : anchors) {
        if (!usable_time(anchor.total_ms)) continue;
        const auto x = features(anchor.query);
        Row& row     = relative.emplace_back();
        for (std::size_t j = 0; j < columns; ++j) {
            row[j] = static_cast<long double>(x[j]) / anchor.total_ms;
            norms[j] += row[j] * row[j];
        }
    }

    auto weights = default_weights();
    if (!relative.empty()) {
        // Scale X/y by each column's RMS norm and penalize the scaled coefficients.
        // With unit RMS, a column's leverage grows with the number of anchors.
        // The Gram matrix keeps each coordinate sweep independent of the anchor count.
        for (auto& norm : norms) norm = std::sqrt(norm / relative.size());
        std::array<Row, columns> gram{};
        Row rhs{};
        for (auto& row : relative) {
            for (std::size_t j = 0; j < columns; ++j) {
                if (norms[j] > 0.0L) row[j] /= norms[j];
                rhs[j] += row[j];
            }
            for (std::size_t j = 0; j < columns; ++j)
                for (std::size_t k = 0; k < columns; ++k) gram[j][k] += row[j] * row[k];
        }
        Row scaled{};
        for (int sweep = 0; sweep < 400; ++sweep) {
            long double max_change = 0.0L;
            for (std::size_t j = 0; j < columns; ++j) {
                if (norms[j] == 0.0L) continue;
                const auto off_diagonal = std::inner_product(
                    gram[j].begin(), gram[j].end(), scaled.begin(), -gram[j][j] * scaled[j]);
                const auto ridge = j == 0 ? 1e-6L : 0.015L;
                const auto next  = std::max(0.0L, (rhs[j] - off_diagonal) / (gram[j][j] + ridge));
                max_change       = std::max(max_change, std::abs(next - scaled[j]));
                scaled[j]        = next;
            }
            if (max_change <= 1e-10L) break;
        }
        weights = {};
        for (std::size_t j = 0; j < columns; ++j) {
            if (norms[j] > 0.0L) {
                weights[j] = static_cast<double>(
                    std::min(scaled[j] / norms[j],
                             static_cast<long double>(std::numeric_limits<double>::max())));
            }
        }
    }
    weights_     = weights;
    calibration_ = {!relative.empty(), relative.size(), elapsed_ms, 0};
    reset_online();
}

double SpeculativeCostModel::base_predict_ms(const CostQuery& q) const {
    return detail::RouteCost(*this, q).base_predict_ms(q);
}

SpeculativeCostModel::OnlineCellKey
SpeculativeCostModel::cell_key(const CostQuery& q) const noexcept {
    return {q.shape.rows,
            q.width,
            q.mtp_steps,
            q.head_columns,
            q.target_bucket,
            config_.hybrid_max_tokens > 0 && q.active_target_tokens > config_.hybrid_max_tokens};
}

double SpeculativeCostModel::predict_ms(const CostQuery& q) const {
    return detail::RouteCost(*this, q).predict_ms(q);
}

double SpeculativeCostModel::cell_multiplier(const CostQuery& q) const {
    const auto found = online_.find(cell_key(q));
    if (found == online_.end()) return 1.0;
    constexpr double prior_weight = 3.0;
    const auto& cell              = found->second;
    return std::clamp((prior_weight + cell.weighted_sum) / (prior_weight + cell.weighted_count),
                      0.5, 2.0);
}

void SpeculativeCostModel::observe_retired_step(const CostQuery& q, double total_ms) {
    if (!usable_time(total_ms)) return;
    constexpr double decay  = 0.85;
    const auto sample_ratio = std::clamp(total_ms / base_predict_ms(q), 0.35, 3.0);
    auto& cell              = online_[cell_key(q)];
    cell.weighted_sum       = decay * cell.weighted_sum + sample_ratio;
    cell.weighted_count     = decay * cell.weighted_count + 1.0;
}

void SpeculativeCostModel::reset_online() noexcept { online_.clear(); }

CostModelSummary SpeculativeCostModel::summary() const noexcept {
    auto result              = calibration_;
    result.online_cell_count = online_.size();
    return result;
}

} // namespace ninfer::models::speculative
