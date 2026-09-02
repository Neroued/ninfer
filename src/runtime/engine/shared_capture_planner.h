#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/engine/context_cost.h"
#include "runtime/engine/context_portfolio_value.h"
#include "runtime/engine/resource_search.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// Shared publication is optional. Unlike materialization, its incumbent is always Skip and a
// target is selectable only when the complete post-state has strictly positive net value.
template <class Package>
class SharedCapturePlanner {
public:
    using Program                = typename Package::Program;
    using CaptureAssessment      = typename Package::CaptureAssessment;
    using CapturePressurePlan    = typename Package::CapturePressurePlan;
    using ContinuationHandle     = typename Package::ContinuationHandle;
    using SharedPrefixHandle     = typename Package::SharedPrefixHandle;
    using PressureTargetHandle   = typename Package::PressureTargetHandle;
    using AssessedPressureTarget = typename Package::AssessedPressureTarget;

    struct OwnerPolicy {
        PlanningOwnerId owner;
        std::uint32_t private_retention_weight = 0;
        bool explicit_shared_credit            = false;
    };

    struct CheckpointPolicy {
        PlanningOwnerId owner;
        CheckpointRef checkpoint;
        std::uint32_t demand_mask          = 0;
        std::uint64_t rebuild_ns           = 0;
        std::uint64_t baseline_recovery_ns = 0;
    };

    struct Input {
        const CaptureAssessment* capture = nullptr;
        std::span<const ContinuationHandle* const> private_owners;
        std::span<const PlanningOwnerId> private_owner_ids;
        std::span<const SharedPrefixHandle* const> shared_owners;
        std::span<const PlanningOwnerId> shared_owner_ids;
        std::span<const OwnerPolicy> owner_policies;
        std::span<const CheckpointPolicy> checkpoint_policies;
        std::optional<PlanningOwnerId> direct_shared_victim;
        std::uint32_t candidate_demand_mask         = 0;
        std::uint64_t candidate_rebuild_ns          = 0;
        std::uint64_t private_baseline_immediate_ns = 0;
        std::uint32_t blocked_runnable_requests     = 0;
        std::uint32_t stable_scenario_ordinal       = 0;
        std::uint32_t target_budget                 = kTargetBudget;
    };

    struct Result {
        std::optional<CapturePressurePlan> pressure;
        std::vector<PressureOwnerOutcome> owner_outcomes;
        std::vector<PressureCheckpointOutcome> checkpoint_outcomes;
        std::uint64_t baseline_value          = 0;
        std::uint64_t target_value            = 0;
        std::uint64_t immediate_ns            = 0;
        std::uint64_t net_gain                = 0;
        std::uint32_t stable_scenario_ordinal = 0;
        std::uint32_t stable_target_ordinal   = 0;
        std::uint32_t targets_evaluated       = 0;
    };

    SharedCapturePlanner() : target_ledger_(kTargetBudget + 1U) {
        queue_.reserve(kTargetBudget);
        impact_scratch_.reserve(64);
        owner_scratch_.reserve(32);
        checkpoint_scratch_.reserve(64);
    }

    [[nodiscard]] std::optional<Result>
    plan(Program& program, const ContextMachineCostModel& machine_cost, const Input& input) {
        validate(input);
        queue_.clear();
        target_ledger_.reset(static_cast<std::size_t>(input.target_budget) + 1U);

        auto session = program.begin_capture_pressure_planning(
            *input.capture, input.private_owners, input.private_owner_ids, input.shared_owners,
            input.shared_owner_ids);
        const PlanningCandidateId candidate_id = session.candidate_id();

        const PressureTargetHandle identity         = session.identity_target();
        const PressureTargetGuidance identity_guide = session.guidance(identity);
        if (identity_guide.candidate != candidate_id) {
            throw std::logic_error("shared capture identity guidance changed candidate identity");
        }
        queue_.push_back(QueuedTarget{
            .target  = identity,
            .ordinal = identity_guide.stable_target_ordinal,
        });
        mark_target(identity_guide.stable_target_ordinal, kTargetDiscovered);
        std::optional<Incumbent> incumbent;
        // The identity target is already canonical in the Program session.  The search budget
        // bounds every canonical target admitted to this scenario, including children committed
        // now but still waiting in the breadth-first queue; assessed targets are only telemetry.
        std::uint32_t canonical_targets = 1;
        std::uint32_t targets_evaluated = 0;
        std::size_t cursor              = 0;

        while (cursor < queue_.size() && targets_evaluated < input.target_budget) {
            const QueuedTarget queued = queue_[cursor++];
            if (target_marked(queued.ordinal, kTargetAssessed)) { continue; }
            AssessedPressureTarget assessed            = session.assess(queued.target);
            const PressureTargetAssessment& assessment = assessed.assessment();
            if (assessment.candidate != candidate_id ||
                assessment.stable_target_ordinal != queued.ordinal) {
                throw std::logic_error("shared capture target changed candidate identity");
            }
            mark_target(queued.ordinal, kTargetAssessed);
            ++targets_evaluated;

            if (assessment.physical_status == MaterializationPhysicalStatus::Feasible) {
                const TransitionValue value = fold_target(input, assessment, machine_cost);
                if (value.positive &&
                    (!incumbent || better(value, assessment, input, *incumbent))) {
                    incumbent = Incumbent{
                        .target              = queued.target,
                        .assessed            = std::move(assessed),
                        .value               = value,
                        .stable_target       = assessment.stable_target_ordinal,
                        .degradation_units   = assessment.degradation_units,
                        .dropped_checkpoints = assessment.dropped_checkpoints,
                        .owner_outcomes      = std::vector<PressureOwnerOutcome>(
                            assessment.owner_outcomes.begin(), assessment.owner_outcomes.end()),
                        .checkpoint_outcomes =
                            [&] {
                                std::vector<PressureCheckpointOutcome> outcomes;
                                outcomes.reserve(assessment.checkpoint_impacts.size());
                                for (const PressureCheckpointRecoveryImpact& impact :
                                     assessment.checkpoint_impacts) {
                                    outcomes.push_back(PressureCheckpointOutcome{
                                        .owner      = impact.owner,
                                        .checkpoint = impact.checkpoint,
                                        .survives   = impact.survives,
                                    });
                                }
                                return outcomes;
                            }(),
                    };
                }
            }

            if (!assessment.expandable || target_marked(queued.ordinal, kTargetExpanded)) {
                continue;
            }
            auto prepared                 = session.prepare_expansion(queued.target);
            const std::uint32_t remaining = input.target_budget - canonical_targets;
            if (prepared.new_canonical_count() > remaining) {
                session.discard_expansion(std::move(prepared));
                continue;
            }
            const auto children = session.commit_expansion(std::move(prepared));
            canonical_targets += children.new_canonical_count;
            mark_target(queued.ordinal, kTargetExpanded);
            for (const PressureTargetHandle child : children.children) {
                const PressureTargetGuidance guidance = session.guidance(child);
                if (guidance.candidate != candidate_id) {
                    throw std::logic_error("shared capture child changed candidate identity");
                }
                if (target_marked(guidance.stable_target_ordinal, kTargetDiscovered)) { continue; }
                mark_target(guidance.stable_target_ordinal, kTargetDiscovered);
                queue_.push_back(QueuedTarget{
                    .target  = child,
                    .ordinal = guidance.stable_target_ordinal,
                });
            }
        }

        if (!incumbent) { return std::nullopt; }
        std::optional<CapturePressurePlan> pressure = session.seal(std::move(*incumbent->assessed));
        if (!pressure) {
            throw std::logic_error("selected shared capture target could not be sealed");
        }
        return Result{
            .pressure                = std::move(*pressure),
            .owner_outcomes          = std::move(incumbent->owner_outcomes),
            .checkpoint_outcomes     = std::move(incumbent->checkpoint_outcomes),
            .baseline_value          = incumbent->value.baseline_public,
            .target_value            = incumbent->value.target_public,
            .immediate_ns            = incumbent->value.immediate,
            .net_gain                = incumbent->value.gain,
            .stable_scenario_ordinal = input.stable_scenario_ordinal,
            .stable_target_ordinal   = incumbent->stable_target,
            .targets_evaluated       = targets_evaluated,
        };
    }

    static constexpr std::uint32_t kTargetBudget = 4096;

private:
    struct QueuedTarget {
        PressureTargetHandle target;
        std::uint32_t ordinal = 0;
    };

    struct CombinedImpact {
        PlanningOwnerId owner;
        CheckpointRef checkpoint;
        std::uint64_t target_ns = 0;
    };

    struct TransitionValue {
        std::uint64_t baseline_public = 0;
        std::uint64_t target_public   = 0;
        std::uint64_t private_loss    = 0;
        std::uint64_t immediate       = 0;
        std::uint64_t gain            = 0;
        bool positive                 = false;
        bool saturated                = false;

        [[nodiscard]] friend constexpr bool operator==(TransitionValue,
                                                       TransitionValue) noexcept = default;
    };

    struct Incumbent {
        PressureTargetHandle target;
        std::optional<AssessedPressureTarget> assessed;
        TransitionValue value;
        std::uint32_t stable_target       = 0;
        std::uint32_t degradation_units   = 0;
        std::uint32_t dropped_checkpoints = 0;
        std::vector<PressureOwnerOutcome> owner_outcomes;
        std::vector<PressureCheckpointOutcome> checkpoint_outcomes;
    };

    static void validate(const Input& input) {
        if (input.capture == nullptr || !input.capture->publishes_shared ||
            input.target_budget == 0 || input.target_budget > kTargetBudget ||
            input.private_owners.size() != input.private_owner_ids.size() ||
            input.shared_owners.size() != input.shared_owner_ids.size()) {
            throw std::invalid_argument("shared capture planning input is malformed");
        }
    }

    static constexpr std::uint8_t kTargetDiscovered = BoundedTargetLedger::Discovered;
    static constexpr std::uint8_t kTargetAssessed   = BoundedTargetLedger::Assessed;
    static constexpr std::uint8_t kTargetExpanded   = BoundedTargetLedger::Expanded;

    [[nodiscard]] bool target_marked(std::uint32_t ordinal, std::uint8_t mark) const {
        return target_ledger_.contains(ordinal, mark);
    }

    void mark_target(std::uint32_t ordinal, std::uint8_t mark) {
        target_ledger_.mark(ordinal, mark);
    }

    [[nodiscard]] static const CheckpointPolicy*
    checkpoint_policy_for(std::span<const CheckpointPolicy> policies, PlanningOwnerId owner,
                          CheckpointRef checkpoint) noexcept {
        const auto found = std::find_if(policies.begin(), policies.end(), [&](const auto& policy) {
            return policy.owner == owner && policy.checkpoint == checkpoint;
        });
        return found == policies.end() ? nullptr : &*found;
    }

    [[nodiscard]] static std::uint64_t saturating_multiply(std::uint64_t value,
                                                           std::uint64_t multiplier) noexcept {
        return multiplier != 0 && value > std::numeric_limits<std::uint64_t>::max() / multiplier
                   ? std::numeric_limits<std::uint64_t>::max()
                   : value * multiplier;
    }

    [[nodiscard]] TransitionValue fold_target(const Input& input,
                                              const PressureTargetAssessment& assessment,
                                              const ContextMachineCostModel& machine_cost) {
        impact_scratch_.clear();
        for (const PressureCheckpointRecoveryImpact& impact : assessment.checkpoint_impacts) {
            if (checkpoint_policy_for(input.checkpoint_policies, impact.owner, impact.checkpoint) ==
                nullptr) {
                throw std::logic_error("capture pressure impact has no portfolio checkpoint");
            }
            const auto found = std::find_if(
                impact_scratch_.begin(), impact_scratch_.end(), [&](const CombinedImpact& value) {
                    return value.owner == impact.owner && value.checkpoint == impact.checkpoint;
                });
            if (found == impact_scratch_.end()) {
                if (impact.target_recovery_work.empty()) {
                    throw std::logic_error("capture pressure impact has no supported recipe");
                }
                impact_scratch_.push_back(CombinedImpact{
                    .owner      = impact.owner,
                    .checkpoint = impact.checkpoint,
                    .target_ns =
                        price_checkpoint_recovery_work(machine_cost, impact.target_recovery_work),
                });
            } else {
                throw std::logic_error("capture pressure impact is duplicated");
            }
        }

        owner_scratch_.clear();
        for (const OwnerPolicy& policy : input.owner_policies) {
            owner_scratch_.push_back(ContextPortfolioOwnerPolicy{
                .owner                    = policy.owner,
                .private_retention_weight = policy.private_retention_weight,
                .explicit_shared_credit   = policy.explicit_shared_credit,
            });
        }
        const PlanningOwnerId candidate_owner = next_candidate_owner(input.owner_policies);
        const bool candidate_credit =
            has_shared_candidate_evidence(input.capture->shared_evidence,
                                          SharedCandidateEvidence::ExplicitBoundary) ||
            has_shared_candidate_evidence(input.capture->shared_evidence,
                                          SharedCandidateEvidence::RequestedAutomatic);
        owner_scratch_.push_back(ContextPortfolioOwnerPolicy{
            .owner                    = candidate_owner,
            .private_retention_weight = 0,
            .explicit_shared_credit   = candidate_credit,
        });

        checkpoint_scratch_.clear();
        for (const CheckpointPolicy& policy : input.checkpoint_policies) {
            std::uint64_t target_recovery = policy.baseline_recovery_ns;
            if (input.direct_shared_victim == policy.owner) {
                target_recovery = policy.rebuild_ns;
            } else {
                const auto impact = std::find_if(impact_scratch_.begin(), impact_scratch_.end(),
                                                 [&](const CombinedImpact& value) {
                                                     return value.owner == policy.owner &&
                                                            value.checkpoint == policy.checkpoint;
                                                 });
                if (impact != impact_scratch_.end()) { target_recovery = impact->target_ns; }
            }
            checkpoint_scratch_.push_back(ContextPortfolioCheckpointValue{
                .owner                = policy.owner,
                .demand_mask          = policy.demand_mask,
                .rebuild_ns           = policy.rebuild_ns,
                .baseline_recovery_ns = policy.baseline_recovery_ns,
                .target_recovery_ns   = target_recovery,
            });
        }
        checkpoint_scratch_.push_back(ContextPortfolioCheckpointValue{
            .owner                = candidate_owner,
            .demand_mask          = input.candidate_demand_mask,
            .rebuild_ns           = input.candidate_rebuild_ns,
            .baseline_recovery_ns = input.candidate_rebuild_ns,
            .target_recovery_ns   = price_checkpoint_recovery_work(
                machine_cost, input.capture->projected_recovery_work),
        });

        const ContextPortfolioValueResult portfolio =
            portfolio_value_.fold(owner_scratch_, checkpoint_scratch_);
        const std::uint64_t multiplier =
            static_cast<std::uint64_t>(input.blocked_runnable_requests) + 1U;
        const std::uint64_t target_immediate = saturating_multiply(
            price_materialization_machine_work(machine_cost, assessment.machine_work).immediate_ns,
            multiplier);
        const std::uint64_t baseline_immediate =
            saturating_multiply(input.private_baseline_immediate_ns, multiplier);
        const std::uint64_t immediate =
            target_immediate > baseline_immediate ? target_immediate - baseline_immediate : 0;
        bool saturated = portfolio.saturated ||
                         target_immediate == std::numeric_limits<std::uint64_t>::max() ||
                         baseline_immediate == std::numeric_limits<std::uint64_t>::max();
        std::uint64_t threshold  = portfolio.baseline_public_value;
        const auto add_threshold = [&](std::uint64_t increment) {
            if (increment > std::numeric_limits<std::uint64_t>::max() - threshold) {
                threshold = std::numeric_limits<std::uint64_t>::max();
                saturated = true;
            } else {
                threshold += increment;
            }
        };
        add_threshold(portfolio.private_transition_loss);
        add_threshold(immediate);
        const bool positive = !saturated && portfolio.target_public_value > threshold;
        return TransitionValue{
            .baseline_public = portfolio.baseline_public_value,
            .target_public   = portfolio.target_public_value,
            .private_loss    = portfolio.private_transition_loss,
            .immediate       = immediate,
            .gain            = positive ? portfolio.target_public_value - threshold : 0,
            .positive        = positive,
            .saturated       = saturated,
        };
    }

    [[nodiscard]] static PlanningOwnerId
    next_candidate_owner(std::span<const OwnerPolicy> policies) {
        std::uint32_t value = 0;
        for (const OwnerPolicy& policy : policies) {
            if (policy.owner.value == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("capture portfolio owner ID overflow");
            }
            value = std::max(value, policy.owner.value + 1U);
        }
        return PlanningOwnerId{.value = value};
    }

    [[nodiscard]] static bool better(const TransitionValue& value,
                                     const PressureTargetAssessment& assessment, const Input& input,
                                     const Incumbent& incumbent) noexcept {
        return std::tuple{
                   value.gain,
                   std::numeric_limits<std::uint32_t>::max() - assessment.degradation_units,
                   std::numeric_limits<std::uint32_t>::max() - assessment.dropped_checkpoints,
                   std::numeric_limits<std::uint32_t>::max() - input.stable_scenario_ordinal,
                   std::numeric_limits<std::uint32_t>::max() - assessment.stable_target_ordinal} >
               std::tuple{incumbent.value.gain,
                          std::numeric_limits<std::uint32_t>::max() - incumbent.degradation_units,
                          std::numeric_limits<std::uint32_t>::max() - incumbent.dropped_checkpoints,
                          std::numeric_limits<std::uint32_t>::max() - input.stable_scenario_ordinal,
                          std::numeric_limits<std::uint32_t>::max() - incumbent.stable_target};
    }

    ContextPortfolioValue portfolio_value_;
    std::vector<QueuedTarget> queue_;
    BoundedTargetLedger target_ledger_;
    std::vector<CombinedImpact> impact_scratch_;
    std::vector<ContextPortfolioOwnerPolicy> owner_scratch_;
    std::vector<ContextPortfolioCheckpointValue> checkpoint_scratch_;
};

} // namespace ninfer::runtime
