#pragma once

#include "runtime/engine/context_cost.h"
#include "runtime/engine/context_portfolio_value.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace ninfer::runtime {

struct MaterializationCheckpointPolicy {
    std::uint32_t owner_ordinal = 0;
    CheckpointRef checkpoint;
    RetentionClass retention_class     = RetentionClass::RecentPrivate;
    std::uint64_t selected_hit_count   = 0;
    std::uint64_t last_hit_epoch       = 0;
    std::uint32_t demand_mask          = 0;
    std::uint64_t rebuild_ns           = 0;
    std::uint64_t baseline_recovery_ns = 0;
};

struct MaterializationOwnerPolicy {
    std::uint32_t ordinal                  = 0;
    RetentionClass retention_class         = RetentionClass::RecentPrivate;
    std::uint64_t selected_hit_count       = 0;
    std::uint64_t last_hit_epoch           = 0;
    std::uint32_t private_retention_weight = 0;
    bool explicit_shared_credit            = false;
};

template <class Package>
class MaterializationPlanner {
public:
    using Program              = typename Package::Program;
    using PreparedPrompt       = typename Package::PreparedPrompt;
    using AdmissionCandidate   = typename Package::AdmissionCandidate;
    using ResourcePlan         = typename Package::ResourcePlan;
    using ContinuationHandle   = typename Package::ContinuationHandle;
    using SharedPrefixHandle   = typename Package::SharedPrefixHandle;
    using PressureTargetHandle = typename Package::PressureTargetHandle;
    using Clock                = std::chrono::steady_clock;

    struct CandidateInput {
        AdmissionCandidate* candidate = nullptr;
        std::uint32_t stable_ordinal  = 0;
        bool current_session_binding  = false;
    };

    struct LogicalGoal {
        std::uint32_t publication_slot = std::numeric_limits<std::uint32_t>::max();
    };

    struct PressureInputs {
        std::span<const ContinuationHandle* const> private_owners;
        std::span<const std::uint32_t> private_owner_ordinals;
        std::span<const SharedPrefixHandle* const> shared_owners;
        std::span<const std::uint32_t> shared_owner_ordinals;
        std::span<const MaterializationOwnerPolicy> owner_policy;
        std::span<const MaterializationCheckpointPolicy> checkpoint_policy;
    };

    struct Result {
        std::optional<ResourcePlan> plan;
        std::uint32_t candidate_index       = 0;
        std::uint32_t publication_slot      = std::numeric_limits<std::uint32_t>::max();
        ClaimDisposition source_disposition = ClaimDisposition::ConsumedToActive;
        std::vector<PressureOwnerOutcome> owner_outcomes;
        MaterializationDiagnostics diagnostics;
    };

    MaterializationPlanner() {
        queue_.reserve(kTargetBudget);
        identity_costs_.reserve(16);
        assessed_.reserve(kTargetBudget);
        expanded_.reserve(kTargetBudget);
        impact_scratch_.reserve(32);
        portfolio_owner_scratch_.reserve(32);
        portfolio_checkpoint_scratch_.reserve(64);
    }

    template <class PressureInputsFn, class LogicalGoalFn>
    [[nodiscard]] std::optional<Result>
    plan(Program& program, const PreparedPrompt& prompt,
         const ContextMachineCostModel& machine_cost, std::span<const CandidateInput> candidates,
         std::uint32_t root_candidate_index, PressureInputsFn&& pressure_inputs,
         LogicalGoalFn&& logical_goal, Clock::time_point planning_started) {
        if (candidates.empty() || root_candidate_index >= candidates.size()) {
            throw std::invalid_argument("materialization planning problem has no root candidate");
        }
        queue_.clear();
        identity_costs_.clear();
        assessed_.clear();
        expanded_.clear();

        std::optional<Incumbent> identity_best;
        std::vector<IdentityRoot> roots;
        roots.reserve(candidates.size());
        std::uint64_t projection_work = 0;
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const CandidateInput& input = candidates[index];
            if (input.candidate == nullptr) {
                throw std::invalid_argument("materialization candidate is null");
            }
            const IdentityMaterializationAssessment& identity =
                input.candidate->identity_assessment();
            planning_saturating_add(projection_work, identity.projection_work);
            const FoldedCost cost = fold_identity(input, identity);
            identity_costs_.push_back(cost);
            std::optional<LogicalGoal> goal;
            if (identity.physical_status == MaterializationPhysicalStatus::Feasible) {
                goal = logical_goal(static_cast<std::uint32_t>(index), identity.source_disposition,
                                    std::span<const PressureOwnerOutcome>{});
            }
            if (goal && (!identity_best || cost.less(identity_best->cost))) {
                identity_best = Incumbent{
                    .candidate_index    = static_cast<std::uint32_t>(index),
                    .publication_slot   = goal->publication_slot,
                    .source_disposition = identity.source_disposition,
                    .cost               = cost,
                    .assessment_digest  = identity.assessment_digest,
                };
            }
            const bool needs_pressure =
                !goal.has_value() &&
                (identity.physical_status == MaterializationPhysicalStatus::Feasible ||
                 identity.expandable);
            const bool pressure_can_improve =
                goal.has_value() &&
                identity.physical_status == MaterializationPhysicalStatus::Feasible &&
                cost.lower_bound_ns < cost.total_ns;
            roots.push_back(IdentityRoot{
                .candidate_index = static_cast<std::uint32_t>(index),
                .lower_bound_ns  = cost.lower_bound_ns,
                .expandable      = needs_pressure || pressure_can_improve,
            });
        }

        if (identity_best) {
            const bool dominates = std::all_of(roots.begin(), roots.end(), [&](const auto& root) {
                return !root.expandable || root.lower_bound_ns > identity_best->cost.total_ns;
            });
            if (dominates) {
                std::optional<ResourcePlan> sealed = program.seal_identity(
                    *candidates[identity_best->candidate_index].candidate, prompt);
                if (!sealed) { return std::nullopt; }
                MaterializationDiagnostics diagnostics = complete_diagnostics(
                    identity_best->cost, static_cast<std::uint32_t>(candidates.size()),
                    projection_work, planning_started, MaterializationStopReason::NoPressure,
                    false);
                Result result;
                result.plan               = std::move(*sealed);
                result.candidate_index    = identity_best->candidate_index;
                result.publication_slot   = identity_best->publication_slot;
                result.source_disposition = identity_best->source_disposition;
                result.diagnostics        = diagnostics;
                return result;
            }
        }

        std::vector<const AdmissionCandidate*> candidate_handles;
        candidate_handles.reserve(candidates.size());
        for (const CandidateInput& input : candidates) {
            candidate_handles.push_back(input.candidate);
        }
        const PressureInputs pressure = pressure_inputs();
        if (pressure.private_owners.size() != pressure.private_owner_ordinals.size() ||
            pressure.shared_owners.size() != pressure.shared_owner_ordinals.size()) {
            throw std::logic_error("materialization pressure owner arrays are not aligned");
        }
        auto session = program.begin_pressure_planning(
            machine_cost, candidate_handles, pressure.private_owners,
            pressure.private_owner_ordinals, pressure.shared_owners,
            pressure.shared_owner_ordinals);

        Incumbent incumbent;
        std::uint32_t targets_evaluated = static_cast<std::uint32_t>(candidates.size());
        if (identity_best) {
            incumbent = *identity_best;
            incumbent.target =
                session.identity_target(*candidates[incumbent.candidate_index].candidate);
        } else {
            PressureTargetHandle root_maximal =
                session.root_maximal_target(*candidates[root_candidate_index].candidate);
            PressureTargetAssessment assessment = session.assess(root_maximal);
            ++targets_evaluated;
            planning_saturating_add(projection_work, assessment.projection_work);
            std::optional<LogicalGoal> goal;
            if (assessment.physical_status == MaterializationPhysicalStatus::Feasible) {
                goal = logical_goal(root_candidate_index, assessment.source_disposition,
                                    assessment.owner_outcomes);
            }
            if (!goal) { return std::nullopt; }
            incumbent = make_incumbent(root_maximal, assessment, candidates[root_candidate_index],
                                       pressure.owner_policy, pressure.checkpoint_policy, *goal);
        }

        for (const IdentityRoot& root : roots) {
            if (!root.expandable || root.lower_bound_ns > incumbent.cost.total_ns) { continue; }
            QueueEntry entry;
            entry.target = session.identity_target(*candidates[root.candidate_index].candidate);
            entry.candidate_index   = root.candidate_index;
            entry.lower_bound_ns    = root.lower_bound_ns;
            entry.remaining_prefill = identity_costs_[root.candidate_index].remaining_text_prefill;
            entry.remaining_vision_prefill =
                identity_costs_[root.candidate_index].remaining_vision_prefill;
            entry.reused_prompt_tokens = identity_costs_[root.candidate_index].reused_prompt_tokens;
            entry.current_session_binding =
                identity_costs_[root.candidate_index].current_session_binding;
            entry.candidate_ordinal     = identity_costs_[root.candidate_index].candidate_ordinal;
            entry.stable_target_ordinal = root.candidate_index;
            assessed_.push_back(entry.target);
            queue_push(std::move(entry));
        }

        const Clock::time_point search_started = Clock::now();
        const std::uint64_t search_budget_ns =
            std::min<std::uint64_t>(5'000'000ULL, incumbent.cost.total_ns / 20U);
        std::uint64_t maximum_expansion_ns    = 0;
        std::uint32_t optional_targets        = 0;
        MaterializationStopReason stop_reason = MaterializationStopReason::QueueExhausted;
        bool model_optimal                    = true;
        bool budget_exhausted                 = false;
        std::uint64_t interrupted_lower_bound = std::numeric_limits<std::uint64_t>::max();

        for (;;) {
            if (queue_.empty()) {
                stop_reason   = MaterializationStopReason::QueueExhausted;
                model_optimal = true;
                break;
            }
            const QueueEntry& next = queue_.front();
            if (next.lower_bound_ns > incumbent.cost.total_ns) {
                stop_reason   = MaterializationStopReason::ModelOptimal;
                model_optimal = true;
                break;
            }
            if (optional_targets >= kTargetBudget) {
                stop_reason      = MaterializationStopReason::TargetBudget;
                model_optimal    = false;
                budget_exhausted = true;
                break;
            }
            const std::uint64_t elapsed = elapsed_ns(search_started, Clock::now());
            if (elapsed >= search_budget_ns) {
                stop_reason      = MaterializationStopReason::TimeBudget;
                model_optimal    = false;
                budget_exhausted = true;
                break;
            }
            const std::uint64_t possible_improvement =
                incumbent.cost.total_ns > next.lower_bound_ns
                    ? incumbent.cost.total_ns - next.lower_bound_ns
                    : 0;
            if (maximum_expansion_ns != 0 && maximum_expansion_ns >= possible_improvement) {
                stop_reason   = MaterializationStopReason::ValueOfNextExpansion;
                model_optimal = false;
                break;
            }

            const QueueEntry parent = queue_pop();
            if (contains(expanded_, parent.target)) { continue; }
            const Clock::time_point expansion_started = Clock::now();
            auto prepared                             = session.prepare_expansion(parent.target);
            if (prepared.new_canonical_count() > kTargetBudget - optional_targets) {
                session.discard_expansion(std::move(prepared));
                interrupted_lower_bound = parent.lower_bound_ns;
                stop_reason             = MaterializationStopReason::ExpansionCapacity;
                model_optimal           = false;
                budget_exhausted        = true;
                break;
            }
            const auto children = session.commit_expansion(std::move(prepared));
            optional_targets += children.new_canonical_count;
            expanded_.push_back(parent.target);

            for (const PressureTargetHandle child : children.children) {
                if (contains(assessed_, child)) { continue; }
                PressureTargetAssessment assessment = session.assess(child);
                assessed_.push_back(child);
                ++targets_evaluated;
                planning_saturating_add(projection_work, assessment.projection_work);
                const FoldedCost cost =
                    fold_assessment(candidates[assessment.candidate_ordinal], assessment,
                                    pressure.owner_policy, pressure.checkpoint_policy);
                std::optional<LogicalGoal> goal;
                if (assessment.physical_status == MaterializationPhysicalStatus::Feasible) {
                    goal = logical_goal(assessment.candidate_ordinal, assessment.source_disposition,
                                        assessment.owner_outcomes);
                }
                if (goal && cost.less(incumbent.cost)) {
                    incumbent =
                        make_incumbent(child, assessment, candidates[assessment.candidate_ordinal],
                                       pressure.owner_policy, pressure.checkpoint_policy, *goal);
                }
                if (assessment.expandable && cost.lower_bound_ns <= incumbent.cost.total_ns) {
                    queue_push(QueueEntry{
                        .target                    = child,
                        .candidate_index           = assessment.candidate_ordinal,
                        .lower_bound_ns            = cost.lower_bound_ns,
                        .affected_selected_hits    = cost.affected_selected_hits,
                        .newest_affected_hit_epoch = cost.newest_affected_hit_epoch,
                        .owner_evictions           = cost.owner_evictions,
                        .checkpoint_drops          = cost.checkpoint_drops,
                        .copy_operations           = cost.copy_operations,
                        .transferred_bytes         = cost.transferred_bytes,
                        .remaining_prefill         = cost.remaining_text_prefill,
                        .remaining_vision_prefill  = cost.remaining_vision_prefill,
                        .reused_prompt_tokens      = cost.reused_prompt_tokens,
                        .current_session_binding   = cost.current_session_binding,
                        .candidate_ordinal         = cost.candidate_ordinal,
                        .stable_target_ordinal     = assessment.stable_target_ordinal,
                    });
                }
            }
            maximum_expansion_ns =
                std::max(maximum_expansion_ns, elapsed_ns(expansion_started, Clock::now()));
        }

        const std::uint64_t search_elapsed_ns        = elapsed_ns(search_started, Clock::now());
        PressureTargetAssessment selected_assessment = session.assess(incumbent.target);
        planning_saturating_add(projection_work, selected_assessment.projection_work);
        const FoldedCost selected_cost =
            fold_assessment(candidates[incumbent.candidate_index], selected_assessment,
                            pressure.owner_policy, pressure.checkpoint_policy);
        if (!equivalent(selected_cost, incumbent.cost) ||
            selected_assessment.assessment_digest != incumbent.assessment_digest) {
            throw std::logic_error("selected pressure target changed before seal");
        }
        std::vector<PressureOwnerOutcome> selected_outcomes(
            selected_assessment.owner_outcomes.begin(), selected_assessment.owner_outcomes.end());
        std::optional<ResourcePlan> sealed = session.seal(incumbent.target, prompt);
        if (!sealed) { throw std::logic_error("selected pressure target could not be sealed"); }

        std::uint64_t best_remaining =
            queue_.empty() ? interrupted_lower_bound : queue_.front().lower_bound_ns;
        best_remaining = std::min(best_remaining, interrupted_lower_bound);
        if (best_remaining == std::numeric_limits<std::uint64_t>::max()) {
            best_remaining = selected_cost.total_ns;
        }
        MaterializationDiagnostics diagnostics = make_diagnostics(
            selected_cost, targets_evaluated, projection_work, planning_started, search_elapsed_ns,
            stop_reason, model_optimal, budget_exhausted, best_remaining,
            selected_assessment.degradation_units, selected_assessment.root_maximal);

        Result result;
        result.plan               = std::move(*sealed);
        result.candidate_index    = incumbent.candidate_index;
        result.publication_slot   = incumbent.publication_slot;
        result.source_disposition = selected_assessment.source_disposition;
        result.owner_outcomes     = std::move(selected_outcomes);
        result.diagnostics        = diagnostics;
        return result;
    }

private:
    static constexpr std::uint32_t kTargetBudget = 4096;

    struct FoldedCost {
        std::uint64_t now_ns                    = 0;
        std::uint64_t future_loss_ns            = 0;
        std::uint64_t total_ns                  = 0;
        std::uint64_t lower_bound_ns            = 0;
        std::uint64_t affected_selected_hits    = 0;
        std::uint64_t newest_affected_hit_epoch = 0;
        std::uint32_t owner_evictions           = 0;
        std::uint32_t checkpoint_drops          = 0;
        std::uint32_t copy_operations           = 0;
        std::uint64_t transferred_bytes         = 0;
        std::uint64_t remaining_text_prefill    = 0;
        std::uint64_t remaining_vision_prefill  = 0;
        std::uint32_t reused_prompt_tokens      = 0;
        bool current_session_binding            = false;
        std::uint32_t candidate_ordinal         = 0;
        std::uint32_t target_ordinal            = 0;

        [[nodiscard]] auto key() const noexcept {
            return std::tuple{
                total_ns,
                affected_selected_hits,
                newest_affected_hit_epoch,
                owner_evictions,
                checkpoint_drops,
                copy_operations,
                transferred_bytes,
                remaining_text_prefill,
                remaining_vision_prefill,
                std::numeric_limits<std::uint32_t>::max() - reused_prompt_tokens,
                current_session_binding ? 0U : 1U,
                candidate_ordinal,
                target_ordinal,
            };
        }

        [[nodiscard]] bool less(const FoldedCost& other) const noexcept {
            return key() < other.key();
        }
    };

    struct Incumbent {
        PressureTargetHandle target{};
        std::uint32_t candidate_index       = 0;
        std::uint32_t publication_slot      = std::numeric_limits<std::uint32_t>::max();
        ClaimDisposition source_disposition = ClaimDisposition::ConsumedToActive;
        FoldedCost cost;
        std::uint64_t assessment_digest = 0;
    };

    struct IdentityRoot {
        std::uint32_t candidate_index = 0;
        std::uint64_t lower_bound_ns  = 0;
        bool expandable               = false;
    };

    struct QueueEntry {
        PressureTargetHandle target{};
        std::uint32_t candidate_index           = 0;
        std::uint64_t lower_bound_ns            = 0;
        std::uint64_t affected_selected_hits    = 0;
        std::uint64_t newest_affected_hit_epoch = 0;
        std::uint32_t owner_evictions           = 0;
        std::uint32_t checkpoint_drops          = 0;
        std::uint32_t copy_operations           = 0;
        std::uint64_t transferred_bytes         = 0;
        std::uint64_t remaining_prefill         = 0;
        std::uint64_t remaining_vision_prefill  = 0;
        std::uint32_t reused_prompt_tokens      = 0;
        bool current_session_binding            = false;
        std::uint32_t candidate_ordinal         = 0;
        std::uint32_t stable_target_ordinal     = 0;
    };

    struct CombinedImpact {
        std::uint32_t owner_ordinal = 0;
        CheckpointRef checkpoint;
        std::uint64_t baseline_ns = 0;
        std::uint64_t target_ns   = 0;
    };

    [[nodiscard]] static std::uint64_t elapsed_ns(Clock::time_point begin,
                                                  Clock::time_point end) noexcept {
        const auto count =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        return count > 0 ? static_cast<std::uint64_t>(count) : 0;
    }

    static void planning_saturating_add(std::uint64_t& value, std::uint64_t add) noexcept {
        value = add > std::numeric_limits<std::uint64_t>::max() - value
                    ? std::numeric_limits<std::uint64_t>::max()
                    : value + add;
    }

    [[nodiscard]] static const MaterializationOwnerPolicy*
    owner_policy_for(std::span<const MaterializationOwnerPolicy> policies,
                     std::uint32_t ordinal) noexcept {
        const auto found = std::find_if(policies.begin(), policies.end(), [&](const auto& policy) {
            return policy.ordinal == ordinal;
        });
        return found == policies.end() ? nullptr : &*found;
    }

    [[nodiscard]] static const MaterializationCheckpointPolicy*
    checkpoint_policy_for(std::span<const MaterializationCheckpointPolicy> policies,
                          std::uint32_t owner_ordinal, CheckpointRef checkpoint) noexcept {
        const auto found = std::find_if(policies.begin(), policies.end(), [&](const auto& policy) {
            return policy.owner_ordinal == owner_ordinal && policy.checkpoint == checkpoint;
        });
        return found == policies.end() ? nullptr : &*found;
    }

    [[nodiscard]] FoldedCost
    fold_identity(const CandidateInput& candidate,
                  const IdentityMaterializationAssessment& assessment) const noexcept {
        FoldedCost cost;
        cost.now_ns                   = assessment.machine.immediate_ns;
        cost.total_ns                 = cost.now_ns;
        cost.lower_bound_ns           = assessment.machine.minimum_request_ns;
        cost.copy_operations          = assessment.machine.copy_operations;
        cost.transferred_bytes        = assessment.machine.transferred_bytes;
        cost.remaining_text_prefill   = assessment.machine.remaining_prefill_work.tokens;
        cost.remaining_vision_prefill = assessment.machine.remaining_prefill_work.vision_patches;
        cost.reused_prompt_tokens     = assessment.machine.reused_prompt_tokens;
        cost.current_session_binding  = candidate.current_session_binding;
        cost.candidate_ordinal        = candidate.stable_ordinal;
        cost.target_ordinal           = candidate.stable_ordinal;
        return cost;
    }

    [[nodiscard]] FoldedCost
    fold_assessment(const CandidateInput& candidate, const PressureTargetAssessment& assessment,
                    std::span<const MaterializationOwnerPolicy> owner_policies,
                    std::span<const MaterializationCheckpointPolicy> checkpoint_policies) {
        FoldedCost cost;
        cost.now_ns                   = assessment.machine.immediate_ns;
        cost.copy_operations          = assessment.machine.copy_operations;
        cost.transferred_bytes        = assessment.machine.transferred_bytes;
        cost.remaining_text_prefill   = assessment.machine.remaining_prefill_work.tokens;
        cost.remaining_vision_prefill = assessment.machine.remaining_prefill_work.vision_patches;
        cost.reused_prompt_tokens     = assessment.machine.reused_prompt_tokens;
        cost.current_session_binding  = candidate.current_session_binding;
        cost.candidate_ordinal        = candidate.stable_ordinal;
        cost.target_ordinal           = assessment.stable_target_ordinal;
        cost.checkpoint_drops         = assessment.dropped_checkpoints;

        for (const PressureOwnerOutcome& outcome : assessment.owner_outcomes) {
            const MaterializationOwnerPolicy* policy =
                owner_policy_for(owner_policies, outcome.owner_ordinal);
            if (policy == nullptr) {
                throw std::logic_error("pressure target references an unknown logical owner");
            }
            if (outcome.disposition == ClaimDisposition::Evicted) { ++cost.owner_evictions; }
        }

        impact_scratch_.clear();
        for (const PressureCheckpointRecoveryImpact& impact : assessment.checkpoint_impacts) {
            const auto found = std::find_if(impact_scratch_.begin(), impact_scratch_.end(),
                                            [&](const CombinedImpact& item) {
                                                return item.owner_ordinal == impact.owner_ordinal &&
                                                       item.checkpoint == impact.checkpoint;
                                            });
            if (found == impact_scratch_.end()) {
                impact_scratch_.push_back(CombinedImpact{
                    .owner_ordinal = impact.owner_ordinal,
                    .checkpoint    = impact.checkpoint,
                    .baseline_ns   = impact.baseline_recovery_ns,
                    .target_ns     = impact.target_recovery_ns,
                });
            } else {
                found->baseline_ns = std::min(found->baseline_ns, impact.baseline_recovery_ns);
                found->target_ns   = std::max(found->target_ns, impact.target_recovery_ns);
            }
        }
        portfolio_owner_scratch_.clear();
        for (const MaterializationOwnerPolicy& policy : owner_policies) {
            portfolio_owner_scratch_.push_back(ContextPortfolioOwnerPolicy{
                .ordinal                  = policy.ordinal,
                .private_retention_weight = policy.private_retention_weight,
                .explicit_shared_credit   = policy.explicit_shared_credit,
            });
        }
        portfolio_checkpoint_scratch_.clear();
        bool portfolio_degraded = false;
        for (const MaterializationCheckpointPolicy& policy : checkpoint_policies) {
            const MaterializationOwnerPolicy* owner =
                owner_policy_for(owner_policies, policy.owner_ordinal);
            if (owner == nullptr) {
                throw std::logic_error("checkpoint policy has no portfolio owner");
            }
            const auto impact = std::find_if(
                impact_scratch_.begin(), impact_scratch_.end(), [&](const CombinedImpact& value) {
                    return value.owner_ordinal == policy.owner_ordinal &&
                           value.checkpoint == policy.checkpoint;
                });
            const std::uint64_t target_recovery =
                impact == impact_scratch_.end() ? policy.baseline_recovery_ns : impact->target_ns;
            if (impact != impact_scratch_.end() &&
                impact->baseline_ns != policy.baseline_recovery_ns) {
                throw std::logic_error("checkpoint baseline recovery changed during planning");
            }
            portfolio_checkpoint_scratch_.push_back(ContextPortfolioCheckpointValue{
                .owner_ordinal        = policy.owner_ordinal,
                .demand_mask          = policy.demand_mask,
                .rebuild_ns           = policy.rebuild_ns,
                .baseline_recovery_ns = policy.baseline_recovery_ns,
                .target_recovery_ns   = target_recovery,
            });
            if (target_recovery > policy.baseline_recovery_ns) {
                portfolio_degraded = true;
                planning_saturating_add(cost.affected_selected_hits, policy.selected_hit_count);
                cost.newest_affected_hit_epoch =
                    std::max(cost.newest_affected_hit_epoch, policy.last_hit_epoch);
            }
        }
        const ContextPortfolioValueResult portfolio =
            portfolio_value_.fold(portfolio_owner_scratch_, portfolio_checkpoint_scratch_);
        if (portfolio.saturated && portfolio_degraded) {
            cost.future_loss_ns = std::numeric_limits<std::uint64_t>::max();
        } else {
            cost.future_loss_ns =
                portfolio.baseline_public_value > portfolio.target_public_value
                    ? portfolio.baseline_public_value - portfolio.target_public_value
                    : 0;
            planning_saturating_add(cost.future_loss_ns, portfolio.private_transition_loss);
        }
        cost.total_ns = cost.now_ns;
        planning_saturating_add(cost.total_ns, cost.future_loss_ns);
        cost.lower_bound_ns = assessment.machine.minimum_request_ns;
        planning_saturating_add(cost.lower_bound_ns, cost.future_loss_ns);
        return cost;
    }

    [[nodiscard]] Incumbent make_incumbent(
        PressureTargetHandle target, const PressureTargetAssessment& assessment,
        const CandidateInput& candidate, std::span<const MaterializationOwnerPolicy> owner_policy,
        std::span<const MaterializationCheckpointPolicy> checkpoint_policy, LogicalGoal goal) {
        return Incumbent{
            .target             = target,
            .candidate_index    = assessment.candidate_ordinal,
            .publication_slot   = goal.publication_slot,
            .source_disposition = assessment.source_disposition,
            .cost = fold_assessment(candidate, assessment, owner_policy, checkpoint_policy),
            .assessment_digest = assessment.assessment_digest,
        };
    }

    [[nodiscard]] static bool equivalent(const FoldedCost& left, const FoldedCost& right) noexcept {
        return left.key() == right.key();
    }

    [[nodiscard]] static auto queue_key(const QueueEntry& entry) noexcept {
        return std::tuple{
            entry.lower_bound_ns,
            entry.affected_selected_hits,
            entry.newest_affected_hit_epoch,
            entry.owner_evictions,
            entry.checkpoint_drops,
            entry.copy_operations,
            entry.transferred_bytes,
            entry.remaining_prefill,
            entry.remaining_vision_prefill,
            std::numeric_limits<std::uint32_t>::max() - entry.reused_prompt_tokens,
            entry.current_session_binding ? 0U : 1U,
            entry.candidate_ordinal,
            entry.stable_target_ordinal,
        };
    }

    void queue_push(QueueEntry entry) {
        queue_.push_back(std::move(entry));
        std::push_heap(queue_.begin(), queue_.end(), [](const auto& left, const auto& right) {
            return queue_key(right) < queue_key(left);
        });
    }

    [[nodiscard]] QueueEntry queue_pop() {
        std::pop_heap(queue_.begin(), queue_.end(), [](const auto& left, const auto& right) {
            return queue_key(right) < queue_key(left);
        });
        QueueEntry result = std::move(queue_.back());
        queue_.pop_back();
        return result;
    }

    [[nodiscard]] static bool contains(std::span<const PressureTargetHandle> handles,
                                       PressureTargetHandle target) noexcept {
        return std::find(handles.begin(), handles.end(), target) != handles.end();
    }

    [[nodiscard]] static MaterializationDiagnostics
    complete_diagnostics(const FoldedCost& cost, std::uint32_t targets_evaluated,
                         std::uint64_t projection_work, Clock::time_point planning_started,
                         MaterializationStopReason reason, bool maximal_fallback) noexcept {
        return make_diagnostics(cost, targets_evaluated, projection_work, planning_started, 0,
                                reason, true, false, cost.total_ns, 0, maximal_fallback);
    }

    [[nodiscard]] static MaterializationDiagnostics
    make_diagnostics(const FoldedCost& cost, std::uint32_t targets_evaluated,
                     std::uint64_t projection_work, Clock::time_point planning_started,
                     std::uint64_t search_elapsed_ns, MaterializationStopReason reason,
                     bool model_optimal, bool budget_exhausted,
                     std::uint64_t best_remaining_lower_bound_ns, std::uint32_t degradation_units,
                     bool maximal_fallback) noexcept {
        const std::uint64_t gap = model_optimal || best_remaining_lower_bound_ns >= cost.total_ns
                                      ? 0
                                      : cost.total_ns - best_remaining_lower_bound_ns;
        return MaterializationDiagnostics{
            .predicted_now_ns              = cost.now_ns,
            .predicted_future_loss_ns      = cost.future_loss_ns,
            .predicted_total_ns            = cost.total_ns,
            .targets_evaluated             = targets_evaluated,
            .projection_work               = projection_work,
            .planning_elapsed_ns           = elapsed_ns(planning_started, Clock::now()),
            .search_elapsed_ns             = search_elapsed_ns,
            .stop_reason                   = reason,
            .model_optimal                 = model_optimal,
            .budget_exhausted              = budget_exhausted,
            .best_remaining_lower_bound_ns = best_remaining_lower_bound_ns,
            .absolute_bound_gap_ns         = gap,
            .relative_bound_gap =
                cost.total_ns == 0 ? 0.0
                                   : static_cast<double>(gap) / static_cast<double>(cost.total_ns),
            .selected_degradation_units = degradation_units,
            .selected_maximal_fallback  = maximal_fallback,
        };
    }

    std::vector<QueueEntry> queue_;
    std::vector<FoldedCost> identity_costs_;
    std::vector<PressureTargetHandle> assessed_;
    std::vector<PressureTargetHandle> expanded_;
    std::vector<CombinedImpact> impact_scratch_;
    ContextPortfolioValue portfolio_value_;
    std::vector<ContextPortfolioOwnerPolicy> portfolio_owner_scratch_;
    std::vector<ContextPortfolioCheckpointValue> portfolio_checkpoint_scratch_;
};

} // namespace ninfer::runtime
