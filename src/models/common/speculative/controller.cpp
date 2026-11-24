#include "models/common/speculative/controller.h"

#include "models/common/speculative/cost_query.h"
#include "models/load_options.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::models::speculative {
namespace {

constexpr float kPriorAlpha      = 2.4f;
constexpr float kPriorBeta       = 1.0f;
constexpr float kPriorMass       = kPriorAlpha + kPriorBeta;
constexpr float kPriorMean       = kPriorAlpha / kPriorMass;
constexpr float kConfidenceScale = std::numeric_limits<std::uint16_t>::max();

float logit(float p) { return std::log(p / (1.0f - p)); }

float logistic(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float clamp_survival(float p) { return std::clamp(p, 0.02f, 0.98f); }

constexpr auto kFreshBonus = [] {
    std::array<double, kMaxSpeculativeDrafts + 1> bonus{};
    double survival = 1.0;
    for (std::size_t j = 1; j < bonus.size(); ++j) {
        survival *= kPriorMean;
        bonus[j] = bonus[j - 1] + survival;
    }
    return bonus;
}();

struct RowYield {
    std::uint32_t limit;
    std::array<double, kMaxSpeculativeDrafts + 1> bonus;
};

float draft_confidence_logit(float confidence) {
    if (!(confidence >= 0.0f && confidence <= 1.0f))
        throw std::invalid_argument("invalid pre-draw draft confidence");
    return logit(std::clamp(confidence, 0.01f, 0.99f));
}

float blend_live_confidence(float history, float confidence) {
    return logistic(0.35f * logit(history) + 0.65f * draft_confidence_logit(confidence));
}

RowYield expected_yield(SpeculativeBackend backend, const AdaptiveRowInput& row,
                        std::uint32_t limit, std::span<const float> live) {
    if (!row.state && live.empty()) return {limit, kFreshBonus};
    RowYield result{};
    result.limit    = limit;
    result.bonus[0] = 0.0;
    if (!limit) return result;

    constexpr SequenceSpeculativeState fresh;
    std::array<float, kMaxSpeculativeDrafts> probabilities;
    const auto& state = row.state ? *row.state : fresh;
    state.predict_conditional_survival(backend, limit, probabilities);
    double survival      = 1.0;
    float horizon_weight = 0.65f;
    for (std::uint32_t j = 0; j < limit; ++j) {
        auto p = probabilities[j];
        if (!live.empty()) {
            if (backend == SpeculativeBackend::DFlash) {
                // Parallel mask predictions are marginals. Later positions provide less
                // evidence about survival conditioned on all earlier drafts being accepted.
                const auto confidence = live[j];
                const auto weight =
                    horizon_weight * (confidence < p ? state.draft_attempt_weight(j) : 1.0f);
                p = logistic((1.0f - weight) * logit(p) +
                             weight * draft_confidence_logit(confidence));
                horizon_weight *= 0.55f;
            } else {
                p = blend_live_confidence(p, live[j]);
            }
        }
        survival *= p;
        result.bonus[j + 1] = result.bonus[j] + survival;
    }
    return result;
}

// Each trial changes one coordinate to a smaller value. The search below a
// unique maximum is bounded by that reduction (at most a draft window).
class Maximum {
public:
    explicit Maximum(std::span<std::uint32_t> counts) : counts_(counts) {}

    void add(std::uint32_t value) {
        ++counts_[value];
        top_ = std::max(top_, value);
    }

    [[nodiscard]] std::uint32_t top() const { return top_; }

    [[nodiscard]] bool can_lower() const { return top_ != 0 && counts_[top_] == 1; }

    [[nodiscard]] std::uint32_t after_reducing(std::uint32_t old_value,
                                               std::uint32_t new_value) const {
        if (old_value != top_ || old_value == new_value || counts_[top_] > 1) return top_;
        auto top = top_;
        while (top > new_value && !counts_[--top]) {}
        return top;
    }

    void reduce(std::uint32_t old_value, std::uint32_t new_value, std::uint32_t new_top) {
        --counts_[old_value];
        ++counts_[new_value];
        top_ = new_top;
    }

private:
    std::span<std::uint32_t> counts_;
    std::uint32_t top_ = 0;
};

struct OwnerLoad {
    std::uint32_t tokens = 0, proposals = 0;
};

// Keep exactly the same coordinates as CostQuery::from_plan as one row changes.
class RouteLoad {
public:
    RouteLoad(SpeculativeBackend backend, const RoundRoute& route,
              std::span<const AdaptiveRowInput> rows, std::span<const std::uint32_t> extents,
              std::uint32_t owner_count, std::uint32_t max_owner_rows, std::uint32_t max_context,
              bool short_context, const CostQuery* executed_proposal)
        : query_(
              detail::route_query(route, static_cast<std::uint32_t>(rows.size()), max_owner_rows)),
          backend_(backend), route_(route), max_context_(max_context),
          short_context_(short_context), fixed_proposal_(executed_proposal != nullptr),
          owners_(owner_count > 1 ? owner_count : 0),
          token_counts_(
              owners_.empty()
                  ? 0
                  : std::size_t(max_owner_rows) * (route.verification_limit(backend) + 1) + 1),
          proposal_counts_(owners_.empty() ? 0 : std::size_t(max_owner_rows) + 1),
          max_tokens_(token_counts_), max_proposals_(proposal_counts_),
          target_buckets_(target_counts_), proposal_buckets_(proposal_bucket_counts_) {
        query_.active_target_tokens = 0;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto extent = extents[i];
            const auto tokens = extent + 1;
            query_.active_target_tokens += tokens;
            const bool proposing = !fixed_proposal_ && extent != 0;
            query_.active_proposal_rows += proposing;
            if (!short_context_) {
                target_buckets_.add(target_index(rows[i], extent));
                if (proposing) proposal_buckets_.add(proposal_index(rows[i]));
            }
            if (!owners_.empty()) {
                auto& owner = owners_[rows[i].member.owner];
                owner.tokens += tokens;
                owner.proposals += proposing;
            }
        }
        for (const auto owner : owners_) {
            max_tokens_.add(owner.tokens);
            max_proposals_.add(owner.proposals);
        }
        query_.max_owner_target_tokens =
            owners_.empty() ? query_.active_target_tokens : max_tokens_.top();
        query_.max_owner_proposal_rows =
            owners_.empty() ? query_.active_proposal_rows : max_proposals_.top();
        query_.target_bucket   = static_cast<ops::ContextBucket>(target_buckets_.top());
        query_.proposal_bucket = static_cast<ops::ContextBucket>(proposal_buckets_.top());
        if (executed_proposal) detail::use_proposal_cost(query_, *executed_proposal);
    }

    [[nodiscard]] const CostQuery& query() const { return query_; }

    [[nodiscard]] CostQuery trial(const AdaptiveRowInput& row, std::uint32_t old_extent,
                                  std::uint32_t new_extent) const {
        auto next              = query_;
        const auto drop        = old_extent - new_extent;
        const bool no_proposal = !fixed_proposal_ && new_extent == 0;
        next.active_target_tokens -= drop;
        next.active_proposal_rows -= no_proposal;
        if (owners_.empty()) {
            next.max_owner_target_tokens = next.active_target_tokens;
            if (!fixed_proposal_) next.max_owner_proposal_rows = next.active_proposal_rows;
        } else {
            const auto owner = owners_[row.member.owner];
            next.max_owner_target_tokens =
                max_tokens_.after_reducing(owner.tokens, owner.tokens - drop);
            if (!fixed_proposal_)
                next.max_owner_proposal_rows =
                    max_proposals_.after_reducing(owner.proposals, owner.proposals - no_proposal);
        }
        if (target_buckets_.can_lower())
            next.target_bucket = static_cast<ops::ContextBucket>(target_buckets_.after_reducing(
                target_index(row, old_extent), target_index(row, new_extent)));
        if (no_proposal && proposal_buckets_.can_lower())
            next.proposal_bucket = static_cast<ops::ContextBucket>(
                proposal_buckets_.after_reducing(proposal_index(row), 0));
        return next;
    }

    void reduce(const AdaptiveRowInput& row, std::uint32_t old_extent, std::uint32_t new_extent,
                const CostQuery& next) {
        if (!owners_.empty()) {
            auto& owner          = owners_[row.member.owner];
            const auto tokens    = owner.tokens - (old_extent - new_extent);
            const auto proposals = owner.proposals - (!fixed_proposal_ && new_extent == 0);
            max_tokens_.reduce(owner.tokens, tokens, next.max_owner_target_tokens);
            if (!fixed_proposal_)
                max_proposals_.reduce(owner.proposals, proposals, next.max_owner_proposal_rows);
            owner = {tokens, proposals};
        }
        if (!short_context_) {
            target_buckets_.reduce(target_index(row, old_extent), target_index(row, new_extent),
                                   static_cast<std::uint32_t>(next.target_bucket));
            if (!fixed_proposal_ && !new_extent)
                proposal_buckets_.reduce(proposal_index(row), 0,
                                         static_cast<std::uint32_t>(next.proposal_bucket));
        }
        query_ = next;
    }

private:
    [[nodiscard]] std::uint32_t target_index(const AdaptiveRowInput& row,
                                             std::uint32_t extent) const {
        return static_cast<std::uint32_t>(
            detail::target_bucket(row.frontier, extent, max_context_));
    }

    [[nodiscard]] std::uint32_t proposal_index(const AdaptiveRowInput& row) const {
        return static_cast<std::uint32_t>(
            detail::proposal_bucket(backend_, route_, row.frontier, max_context_));
    }

    CostQuery query_;
    SpeculativeBackend backend_;
    RoundRoute route_;
    std::uint32_t max_context_;
    bool short_context_;
    bool fixed_proposal_;
    std::vector<OwnerLoad> owners_;
    std::vector<std::uint32_t> token_counts_, proposal_counts_;
    std::array<std::uint32_t, ops::kContextBucketCount> target_counts_{}, proposal_bucket_counts_{};
    Maximum max_tokens_, max_proposals_, target_buckets_, proposal_buckets_;
};

// A concave load cost lies above its chord. The maximum owner load is
// at least the average, so these chords give a linear lower bound in
// global tokens and proposal rows. Test the relaxed throughput maximum
// before sorting or trying extents in a route that cannot win.
bool can_beat(const detail::RouteCost& pricing, const CostQuery& full,
              std::span<const RowYield> yields, std::span<const std::uint32_t> extents,
              std::uint32_t owners, double score) {
    if (!pricing.has_concave_load_cost() || !score) return true;
    auto q          = full;
    q.target_bucket = q.proposal_bucket = ops::ContextBucket::K2;
    q.max_owner_target_tokens           = q.logical_rows / owners;
    q.max_owner_proposal_rows           = 0;
    const auto base                     = pricing.predict_ms(q);
    if (base <= 0.05 || !std::isfinite(base)) return true;
    q.max_owner_target_tokens = full.max_owner_target_tokens;
    const auto target_cost    = pricing.predict_ms(q);
    q.max_owner_proposal_rows = full.max_owner_proposal_rows;
    const auto full_cost      = pricing.predict_ms(q);
    if (!std::isfinite(full_cost)) return true;
    const auto token_slope =
        (target_cost - base) /
        ((full.max_owner_target_tokens - full.logical_rows / owners) * double(owners));
    const auto proposal_slope =
        (full_cost - target_cost) / (full.max_owner_proposal_rows * double(owners));
    const auto token_price    = token_slope * score;
    const auto proposal_price = proposal_slope * score;
    auto surplus = full.logical_rows - score * (base + (full.logical_rows % owners) * token_slope);
    for (std::size_t i = 0; i < extents.size(); ++i) {
        const auto limit  = extents[i];
        const auto& bonus = yields[i].bonus;
        if (!limit || bonus[1] <= token_price) continue;
        auto v = limit;
        while (v > 1 && bonus[v] - bonus[v - 1] < token_price) --v;
        surplus += std::max(0.0, bonus[v] - v * token_price - proposal_price);
    }
    return surplus >= -1e-10 * full.logical_rows;
}

double refine_route(const SpeculativeCostModel& model, SpeculativeBackend backend,
                    const RoundRoute& route, std::span<const AdaptiveRowInput> rows,
                    std::span<const RowYield> yields, std::span<std::uint32_t> extents,
                    std::vector<std::uint32_t>& order, std::uint32_t owner_count,
                    std::uint32_t max_owner_rows, std::uint32_t max_context, bool short_context,
                    double minimum_score, const CostQuery* executed_proposal) {
    const auto cap              = route.verification_limit(backend);
    const auto required_depth   = route.covers(backend, 1) ? 1 : cap;
    auto expected               = double(rows.size());
    std::uint32_t required_rows = 0;
    order.clear();
    for (std::uint32_t i = 0; i < rows.size(); ++i) {
        const auto extent = extents[i] = std::min(cap, yields[i].limit);
        if (!extent) continue;
        order.push_back(i);
        required_rows += extent >= required_depth;
        expected += yields[i].bonus[extent];
    }
    if (cap && !required_rows) return 0.0;

    RouteLoad load(backend, route, rows, extents, owner_count, max_owner_rows, max_context,
                   short_context, executed_proposal);
    const detail::RouteCost pricing(model, load.query());
    if (cap && !executed_proposal &&
        !can_beat(pricing, load.query(), yields, extents, owner_count, minimum_score))
        return 0.0;
    auto score            = expected / pricing.predict_ms(load.query());
    const auto less_yield = [&](auto a, auto b) {
        const auto lhs = yields[a].bonus[extents[a]], rhs = yields[b].bonus[extents[b]];
        return lhs < rhs || (lhs == rhs && a < b);
    };
    if (!std::ranges::is_sorted(order, less_yield)) std::ranges::sort(order, less_yield);

    for (const auto i : order) {
        const auto extent = extents[i];
        const auto floor  = required_rows == 1 && extent >= required_depth ? required_depth : 0u;
        const auto expected_without_row = expected - yields[i].bonus[extent];
        auto best                       = extent;
        CostQuery best_query;
        const std::array trials{0u, 1u, 3u};
        for (const auto trial : trials) {
            if (trial < floor || trial >= extent) continue;
            const auto query = load.trial(rows[i], extent, trial);
            const auto candidate =
                (expected_without_row + yields[i].bonus[trial]) / pricing.predict_ms(query);
            if (candidate <= score) continue;
            score      = candidate;
            best       = trial;
            best_query = query;
        }
        if (best == extent) continue;
        load.reduce(rows[i], extent, best, best_query);
        expected = expected_without_row + yields[i].bonus[best];
        required_rows -= extent >= required_depth && best < required_depth;
        extents[i] = best;
    }
    return score;
}

std::vector<RoundRoute> candidate_routes(SpeculativeBackend backend, std::uint32_t max_drafts,
                                         PhysicalBatchShape shape, ExecutionPath path,
                                         const DecodeGraphRoutes* routes, std::uint32_t max_limit,
                                         const RoundRoute* initial) {
    std::vector<RoundRoute> candidates;
    if (path != ExecutionPath::Eager) {
        if (!routes) throw std::invalid_argument("graph decode requires prepared routes");
        const auto prepared = routes->at(shape);
        candidates.reserve(prepared.size());
        const auto ordinary = routes->select_route(shape, backend, 0);
        candidates.push_back(ordinary);
        if (max_limit)
            for (const auto& route : prepared)
                if (route.width > 1 && (!initial || route.width <= initial->width))
                    candidates.push_back(route);
    } else {
        candidates.reserve(max_limit + 1);
        candidates.push_back(RoundRoute::eager(shape, backend, max_drafts, 0));
        // A split proposal is already available. Eager target execution can
        // use any width within that block, including non-dyadic causal caps.
        const bool every_depth = initial || backend == SpeculativeBackend::Mtp;
        for (std::uint32_t depth = 1; depth <= max_limit;
             depth               = every_depth ? depth + 1 : 2 * depth + 1) {
            auto route = RoundRoute::eager(shape, backend, max_drafts, depth);
            if (initial && route.width > initial->width) break;
            candidates.push_back(route);
        }
    }
    // Ascending width lets the tie rule compare against the true maximum score.
    if (!std::ranges::is_sorted(candidates, {}, &RoundRoute::width))
        std::ranges::stable_sort(candidates, {}, &RoundRoute::width);
    return candidates;
}

DecodeRoundPlan optimize(SpeculativeBackend backend, std::uint32_t max_drafts,
                         std::uint32_t owner_count, PhysicalBatchShape shape, ExecutionPath path,
                         const DecodeGraphRoutes* routes, const SpeculativeCostModel& cost_model,
                         std::span<const AdaptiveRowInput> rows, std::uint32_t max_context,
                         const DecodeRoundPlan* initial = nullptr, std::span<const float> live = {},
                         std::uint32_t stride = 0) {
    if (!owner_count || rows.empty() || rows.size() > shape.rows)
        throw std::invalid_argument("invalid adaptive round membership");

    std::vector<RoundMember> members;
    members.reserve(rows.size());
    std::uint32_t max_limit = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto member = rows[i].member;
        members.push_back(member);
        const auto limit = initial ? initial->verify_drafts()[i] : max_drafts;
        max_limit        = std::max(max_limit, std::min(member.granted_limit, limit));
    }
    std::vector<std::uint32_t> best_extents(rows.size());
    if (backend == SpeculativeBackend::None || !max_limit)
        return DecodeRoundPlan::select(backend, max_drafts, owner_count, shape, path, routes,
                                       members, best_extents);
    if (max_drafts > kMaxSpeculativeDrafts)
        throw std::invalid_argument("adaptive draft window exceeds its supported limit");
    if (rows.size() > std::numeric_limits<std::uint32_t>::max() / (max_drafts + 1))
        throw std::overflow_error("adaptive target tokens exceed the cost query range");

    std::uint32_t max_owner_rows = static_cast<std::uint32_t>(rows.size());
    std::vector<std::uint32_t> owner_rows(owner_count > 1 ? owner_count : 0);
    if (!owner_rows.empty()) max_owner_rows = 0;
    std::vector<RowYield> yields;
    yields.reserve(rows.size());
    max_limit                  = 0;
    std::uint32_t max_frontier = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto member = rows[i].member;
        if (member.owner >= owner_count || member.granted_limit > member.hard_limit)
            throw std::invalid_argument("invalid adaptive row grant or owner");
        if (!owner_rows.empty())
            max_owner_rows = std::max(max_owner_rows, ++owner_rows[member.owner]);
        const auto limit =
            std::min(member.granted_limit, initial ? initial->verify_drafts()[i] : max_drafts);
        const auto confidence =
            initial ? live.subspan(i * stride, limit) : std::span<const float>{};
        yields.push_back(expected_yield(backend, rows[i], limit, confidence));
        max_limit    = std::max(max_limit, yields.back().limit);
        max_frontier = std::max(max_frontier, rows[i].frontier);
    }
    const bool short_context =
        detail::target_bucket(max_frontier, max_drafts, max_context) == ops::ContextBucket::K2;

    std::optional<CostQuery> executed_proposal;
    if (initial) {
        std::vector<std::uint32_t> frontiers;
        frontiers.reserve(rows.size());
        for (const auto& row : rows) frontiers.push_back(row.frontier);
        executed_proposal = CostQuery::from_plan(backend, *initial, frontiers, max_context);
    }

    std::vector<std::uint32_t> extents(rows.size()), order;
    order.reserve(rows.size());
    const auto candidates = candidate_routes(backend, max_drafts, shape, path, routes, max_limit,
                                             initial ? &initial->route() : nullptr);
    auto best_route       = candidates.front();
    double best_score = 0.0, max_score = 0.0;
    for (const auto& route : candidates) {
        const auto minimum_score = std::min(best_score, 0.999 * max_score);
        const auto score =
            refine_route(cost_model, backend, route, rows, yields, extents, order, owner_count,
                         max_owner_rows, max_context, short_context, minimum_score,
                         executed_proposal ? &*executed_proposal : nullptr);
        if (score <= 0.0) continue;
        max_score = std::max(max_score, score);
        if (score > best_score || (route.width > best_route.width && score >= 0.999 * max_score)) {
            best_score   = score;
            best_route   = route;
            best_extents = extents;
        }
    }
    return DecodeRoundPlan::freeze(backend, max_drafts, owner_count, members, best_route,
                                   std::move(best_extents));
}

} // namespace

void SequenceSpeculativeState::decay(float factor) noexcept {
    for (auto& value : attempts_) value *= factor;
    for (auto& value : accepts_) value *= factor;
}

void SequenceSpeculativeState::observe_speculative_round(std::uint32_t attempted_extent,
                                                         std::uint32_t accepted_drafts,
                                                         std::span<const float> draft_confidences) {
    if (attempted_extent > kMaxSpeculativeDrafts || accepted_drafts > attempted_extent ||
        draft_confidences.size() > kMaxSpeculativeDrafts ||
        std::ranges::any_of(draft_confidences, [](float p) { return !std::isfinite(p); }))
        throw std::invalid_argument("invalid speculative acceptance observation");
    decay(0.92f);
    const auto tested = std::min(attempted_extent, accepted_drafts + 1);
    for (std::uint32_t j = 0; j < tested; ++j) {
        attempts_[j] += 1.0f;
        accepts_[j] += j < accepted_drafts;
    }
    if (!draft_confidences.empty()) {
        last_draft_count_ = static_cast<std::uint8_t>(draft_confidences.size());
        for (std::size_t j = 0; j < draft_confidences.size(); ++j)
            last_draft_confidences_[j] = static_cast<std::uint16_t>(
                std::clamp(draft_confidences[j], 0.0f, 1.0f) * kConfidenceScale + 0.5f);
    }
    last_target_top1_prob_.reset();
}

void SequenceSpeculativeState::observe_ordinary_round(std::optional<float> target_top1_prob) {
    decay(0.95f);
    last_target_top1_prob_ = target_top1_prob;
    if (last_target_top1_prob_ && std::isnan(*last_target_top1_prob_))
        last_target_top1_prob_.reset();
    last_draft_count_ = 0;
}

float SequenceSpeculativeState::draft_attempt_weight(std::uint32_t position) const noexcept {
    assert(position < kMaxSpeculativeDrafts);
    return kPriorMass / (kPriorMass + attempts_[position]);
}

void SequenceSpeculativeState::predict_conditional_survival(
    SpeculativeBackend backend, std::uint32_t max_drafts,
    std::span<float> out_probs) const noexcept {
    assert(max_drafts <= kMaxSpeculativeDrafts && max_drafts <= out_probs.size());
    const auto emp0         = (kPriorAlpha + accepts_[0]) / (kPriorMass + attempts_[0]);
    const auto deeper_alpha = emp0 < kPriorMean ? kPriorMass * emp0 : kPriorAlpha;
    const bool masked       = is_masked_draft_backend(backend);
    const auto base_logit   = logit(kPriorMean);
    auto previous           = emp0;
    const auto empirical    = [&](std::size_t j) {
        if (!j) return emp0;
        // An untested tail inherits evidence from the verified prefix, with decay at each
        // position. A posterior at or below the prior stops the upward inheritance.
        if (masked && attempts_[j] == 0.0f && previous > kPriorMean)
            previous =
                logistic(base_logit + 0.92f * (logit(clamp_survival(previous)) - base_logit));
        else
            previous = (deeper_alpha + accepts_[j]) / (kPriorMass + attempts_[j]);
        return previous;
    };
    if (last_target_top1_prob_) {
        // Propagate the first-position log-odds correction with a 0.85 depth decay.
        auto correction = 0.5f * (logit(clamp_survival(*last_target_top1_prob_)) - logit(emp0));
        for (std::uint32_t j = 0; j < max_drafts; ++j) {
            out_probs[j] = clamp_survival(logistic(logit(empirical(j)) + correction));
            correction *= 0.85f;
        }
        return;
    }
    const auto blended_count = std::min<std::uint32_t>(max_drafts, last_draft_count_);
    for (std::uint32_t j = 0; j < blended_count; ++j) {
        const auto confidence = clamp_survival(last_draft_confidences_[j] / kConfidenceScale);
        const auto emp        = empirical(j);
        auto weight           = 0.25f;
        if (confidence > emp)
            weight *= draft_attempt_weight(j);
        else if (backend == SpeculativeBackend::DFlash && confidence < emp)
            weight *= kPriorMass / (kPriorMass + accepts_[j]);
        const auto blended =
            (1.0f - weight) * logit(clamp_survival(emp)) + weight * logit(confidence);
        out_probs[j] = clamp_survival(logistic(blended));
    }
    for (auto j = blended_count; j < max_drafts; ++j) out_probs[j] = clamp_survival(empirical(j));
}

void record_speculative_commit(SpeculativeStats& stats, SequenceSpeculativeState* adaptive_state,
                               const runtime::CommitDecision& decision, bool could_speculate,
                               std::uint32_t proposed, std::uint32_t drafted,
                               std::uint32_t accepted, std::span<const float> draft_confidences,
                               std::optional<float> target_top1_probability) {
    if (adaptive_state && !decision.cancelled && !decision.failed) {
        if (drafted)
            adaptive_state->observe_speculative_round(drafted, accepted,
                                                      draft_confidences.first(drafted));
        else
            adaptive_state->observe_ordinary_round(target_top1_probability);
    }
    stats.committed_draft_tokens += std::min(decision.accepted_tokens, accepted);
    if (drafted)
        ++stats.rounds;
    else if (could_speculate)
        ++stats.bypass_steps;
    else
        ++stats.fallback_steps;
    stats.proposed_tokens += proposed;
    stats.drafted_tokens += drafted;
    stats.accepted_tokens += accepted;
    for (std::uint32_t i = 0; i < drafted; ++i) ++stats.attempted_per_position[i];
    for (std::uint32_t i = 0; i < accepted; ++i) ++stats.accepted_per_position[i];
}

DecodeRoundPlan select_adaptive_round(SpeculativeBackend backend, std::uint32_t max_drafts,
                                      std::uint32_t owner_count, PhysicalBatchShape shape,
                                      ExecutionPath path, const DecodeGraphRoutes* routes,
                                      const SpeculativeCostModel& cost_model,
                                      std::span<const AdaptiveRowInput> rows,
                                      std::uint32_t max_context) {
    return optimize(backend, max_drafts, owner_count, shape, path, routes, cost_model, rows,
                    max_context);
}

bool DecodeRoundRequest::uses_adaptive_length(SpeculativeLengthPolicy policy) const noexcept {
    return !budgets.empty() && policy == SpeculativeLengthPolicy::Adaptive;
}

bool DecodeRoundRequest::valid_membership(std::size_t rows) const noexcept {
    if (budgets.empty()) return extents.size() == rows;
    return budgets.size() == rows && std::ranges::all_of(budgets, [](auto budget) {
               return budget.generated_tokens_remaining != 0;
           });
}

std::uint32_t DecodeRoundRequest::draft_budget(std::size_t row) const noexcept {
    return budgets.empty() ? extents[row] : budgets[row].generated_tokens_remaining - 1;
}

std::optional<DecodeRoundPlan>
DecodeRoundRequest::select(SpeculativeLengthPolicy policy, PhysicalBatchShape shape,
                           ExecutionPath path, const DecodeGraphRoutes* routes,
                           const SpeculativeCostModel& cost, std::span<const RoundMember> limits,
                           std::span<const AdaptiveRowInput> rows,
                           std::uint32_t max_context) const {
    const auto& config = cost.config();
    const auto backend = config.backend;
    const auto k       = config.max_drafts;
    const auto owners  = config.owner_count;
    if (!budgets.empty()) {
        if (backend != SpeculativeBackend::None && uses_adaptive_length(policy))
            return select_adaptive_round(backend, k, owners, shape, path, routes, cost, rows,
                                         max_context);
        return DecodeRoundPlan::select(backend, k, owners, shape, path, routes, limits);
    }
    for (std::size_t i = 0; i < extents.size(); ++i)
        if (extents[i] > limits[i].granted_limit) return {};
    if (!route)
        return DecodeRoundPlan::select(backend, k, owners, shape, path, routes, limits, extents);
    if (route->shape != shape || (path == ExecutionPath::Eager) == route->graph_entry.has_value())
        throw std::invalid_argument("probe route differs from the prepared execution shape");
    if (route->graph_entry) {
        if (!routes) throw std::logic_error("probe has no prepared decode graph catalog");
        const auto prepared = routes->at(shape);
        if (std::ranges::find(prepared, *route) == prepared.end())
            throw std::invalid_argument("probe route is absent from the prepared catalog");
    }
    return DecodeRoundPlan::freeze(backend, k, owners, limits, *route,
                                   {extents.begin(), extents.end()});
}

NarrowedTargetPlan
narrow_after_proposal(SpeculativeBackend backend, std::uint32_t max_drafts,
                      std::uint32_t owner_count, ExecutionPath path,
                      const DecodeGraphRoutes* routes, const SpeculativeCostModel& cost_model,
                      const DecodeRoundPlan& initial_plan, std::span<const AdaptiveRowInput> rows,
                      std::span<const float> live_draft_confidences,
                      std::uint32_t confidence_stride, std::uint32_t max_context) {
    const auto& initial = initial_plan.route();
    initial.validate(backend, max_drafts);
    if ((path == ExecutionPath::Eager) == initial.graph_entry.has_value() ||
        initial_plan.owner_target_tokens().size() != owner_count)
        throw std::invalid_argument("split proposal execution differs from the initial plan");
    if (max_drafts > kMaxSpeculativeDrafts || rows.size() != initial_plan.verify_drafts().size() ||
        (confidence_stride && live_draft_confidences.size() / confidence_stride < rows.size()))
        throw std::invalid_argument("proposal confidences differ from round membership");
    for (std::size_t i = 0; i < rows.size(); ++i)
        if (initial_plan.slots()[i].owner != rows[i].member.owner ||
            initial_plan.verify_drafts()[i] > confidence_stride)
            throw std::invalid_argument("proposal confidence stride or owner differs from plan");

    if (backend == SpeculativeBackend::DFlash2) {
        constexpr SequenceSpeculativeState fresh;
        std::vector<RoundMember> members;
        std::vector<std::uint32_t> extents;
        members.reserve(rows.size());
        extents.reserve(rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto& row = rows[i];
            members.push_back(row.member);
            const auto limit = std::min(row.member.granted_limit, initial_plan.verify_drafts()[i]);
            std::array<float, kMaxSpeculativeDrafts> history;
            (row.state ? *row.state : fresh).predict_conditional_survival(backend, limit, history);
            double survival = 1.0;
            std::uint32_t j = 0;
            for (; j < limit; ++j) {
                survival *= blend_live_confidence(
                    history[j], live_draft_confidences[i * confidence_stride + j]);
                if (survival < 0.12f) break;
            }
            extents.push_back(j);
        }
        // Route selection may pack the finalized extents but cannot change them. Confidence for
        // a later draft can depend on this row's earlier sampled tokens.
        const auto plan = DecodeRoundPlan::select(backend, max_drafts, owner_count, initial.shape,
                                                  path, routes, members, extents);
        if (plan.route().width > initial.width)
            throw std::invalid_argument("narrowed target exceeds the initial route");
        return {plan.route(), std::move(extents)};
    }
    auto plan = optimize(backend, max_drafts, owner_count, initial_plan.route().shape, path, routes,
                         cost_model, rows, max_context, &initial_plan, live_draft_confidences,
                         confidence_stride);
    return {plan.route(), {plan.verify_drafts().begin(), plan.verify_drafts().end()}};
}

} // namespace ninfer::models::speculative
