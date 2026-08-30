#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/engine/context_cost.h"
#include "runtime/engine/context_portfolio_value.h"

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
    using Program              = typename Package::Program;
    using AdmissionCandidate   = typename Package::AdmissionCandidate;
    using CaptureAssessment    = typename Package::CaptureAssessment;
    using CapturePressurePlan  = typename Package::CapturePressurePlan;
    using ContinuationHandle   = typename Package::ContinuationHandle;
    using SharedPrefixHandle   = typename Package::SharedPrefixHandle;
    using PressureTargetHandle = typename Package::PressureTargetHandle;

    struct OwnerPolicy {
        std::uint32_t ordinal                  = 0;
        std::uint32_t private_retention_weight = 0;
        bool explicit_shared_credit            = false;
    };

    struct CheckpointPolicy {
        std::uint32_t owner_ordinal = 0;
        CheckpointRef checkpoint;
        std::uint32_t demand_mask          = 0;
        std::uint64_t rebuild_ns           = 0;
        std::uint64_t baseline_recovery_ns = 0;
    };

    struct Input {
        const CaptureAssessment* capture = nullptr;
        std::span<const ContinuationHandle* const> private_owners;
        std::span<const std::uint32_t> private_owner_ordinals;
        std::span<const SharedPrefixHandle* const> shared_owners;
        std::span<const std::uint32_t> shared_owner_ordinals;
        std::span<const OwnerPolicy> owner_policies;
        std::span<const CheckpointPolicy> checkpoint_policies;
        std::optional<std::uint32_t> direct_shared_victim_ordinal;
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
        std::uint64_t baseline_value          = 0;
        std::uint64_t target_value            = 0;
        std::uint64_t immediate_ns            = 0;
        std::uint64_t net_gain                = 0;
        std::uint32_t stable_scenario_ordinal = 0;
        std::uint32_t stable_target_ordinal   = 0;
        std::uint32_t targets_evaluated       = 0;
    };

    SharedCapturePlanner() {
        queue_.reserve(kTargetBudget);
        assessed_.reserve(kTargetBudget);
        expanded_.reserve(kTargetBudget);
        impact_scratch_.reserve(64);
        owner_scratch_.reserve(32);
        checkpoint_scratch_.reserve(64);
    }

    [[nodiscard]] std::optional<Result>
    plan(Program& program, const ContextMachineCostModel& machine_cost, const Input& input) {
        validate(input);
        queue_.clear();
        assessed_.clear();
        expanded_.clear();

        AdmissionCandidate candidate =
            program.make_capture_pressure_candidate(*input.capture, machine_cost);
        const AdmissionCandidate* candidate_handle = &candidate;
        auto session                               = program.begin_pressure_planning(
            machine_cost, std::span<const AdmissionCandidate* const>(&candidate_handle, 1),
            input.private_owners, input.private_owner_ordinals, input.shared_owners,
            input.shared_owner_ordinals);

        const PressureTargetHandle identity = session.identity_target(candidate);
        queue_.push_back(identity);
        std::optional<Incumbent> incumbent;
        std::uint32_t targets_evaluated = 0;
        std::size_t cursor              = 0;

        while (cursor < queue_.size() && targets_evaluated < input.target_budget) {
            const PressureTargetHandle target = queue_[cursor++];
            if (contains(assessed_, target)) { continue; }
            const PressureTargetAssessment assessment = session.assess(target);
            assessed_.push_back(target);
            ++targets_evaluated;

            if (assessment.physical_status == MaterializationPhysicalStatus::Feasible) {
                const TransitionValue value = fold_target(input, assessment);
                if (value.positive &&
                    (!incumbent || better(value, assessment, input, *incumbent))) {
                    incumbent = Incumbent{
                        .target              = target,
                        .value               = value,
                        .stable_target       = assessment.stable_target_ordinal,
                        .degradation_units   = assessment.degradation_units,
                        .dropped_checkpoints = assessment.dropped_checkpoints,
                        .assessment_digest   = assessment.assessment_digest,
                    };
                }
            }

            if (!assessment.expandable || contains(expanded_, target)) { continue; }
            auto prepared                 = session.prepare_expansion(target);
            const std::uint32_t remaining = input.target_budget - targets_evaluated;
            if (prepared.new_canonical_count() > remaining) {
                session.discard_expansion(std::move(prepared));
                continue;
            }
            const auto children = session.commit_expansion(std::move(prepared));
            expanded_.push_back(target);
            for (const PressureTargetHandle child : children.children) {
                if (!contains(assessed_, child) && !contains(queue_, child)) {
                    queue_.push_back(child);
                }
            }
        }

        if (!incumbent) { return std::nullopt; }
        const PressureTargetAssessment selected = session.assess(incumbent->target);
        const TransitionValue selected_value    = fold_target(input, selected);
        if (selected.assessment_digest != incumbent->assessment_digest ||
            selected.stable_target_ordinal != incumbent->stable_target ||
            selected_value != incumbent->value) {
            throw std::logic_error("shared capture target changed before seal");
        }
        std::vector<PressureOwnerOutcome> outcomes(selected.owner_outcomes.begin(),
                                                   selected.owner_outcomes.end());
        std::optional<CapturePressurePlan> pressure = session.seal_capture(incumbent->target);
        if (!pressure) {
            throw std::logic_error("selected shared capture target could not be sealed");
        }
        return Result{
            .pressure                = std::move(*pressure),
            .owner_outcomes          = std::move(outcomes),
            .baseline_value          = selected_value.baseline_public,
            .target_value            = selected_value.target_public,
            .immediate_ns            = selected_value.immediate,
            .net_gain                = selected_value.gain,
            .stable_scenario_ordinal = input.stable_scenario_ordinal,
            .stable_target_ordinal   = selected.stable_target_ordinal,
            .targets_evaluated       = targets_evaluated,
        };
    }

    static constexpr std::uint32_t kTargetBudget = 4096;

private:
    struct CombinedImpact {
        std::uint32_t owner_ordinal = 0;
        CheckpointRef checkpoint;
        std::uint64_t baseline_ns = 0;
        std::uint64_t target_ns   = 0;
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
        TransitionValue value;
        std::uint32_t stable_target       = 0;
        std::uint32_t degradation_units   = 0;
        std::uint32_t dropped_checkpoints = 0;
        std::uint64_t assessment_digest   = 0;
    };

    static void validate(const Input& input) {
        if (input.capture == nullptr || !input.capture->publishes_shared ||
            input.target_budget == 0 || input.target_budget > kTargetBudget ||
            input.private_owners.size() != input.private_owner_ordinals.size() ||
            input.shared_owners.size() != input.shared_owner_ordinals.size()) {
            throw std::invalid_argument("shared capture planning input is malformed");
        }
    }

    [[nodiscard]] static bool contains(std::span<const PressureTargetHandle> values,
                                       PressureTargetHandle target) noexcept {
        return std::find(values.begin(), values.end(), target) != values.end();
    }

    [[nodiscard]] static const CheckpointPolicy*
    checkpoint_policy_for(std::span<const CheckpointPolicy> policies, std::uint32_t owner_ordinal,
                          CheckpointRef checkpoint) noexcept {
        const auto found = std::find_if(policies.begin(), policies.end(), [&](const auto& policy) {
            return policy.owner_ordinal == owner_ordinal && policy.checkpoint == checkpoint;
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
                                              const PressureTargetAssessment& assessment) {
        impact_scratch_.clear();
        for (const PressureCheckpointRecoveryImpact& impact : assessment.checkpoint_impacts) {
            if (checkpoint_policy_for(input.checkpoint_policies, impact.owner_ordinal,
                                      impact.checkpoint) == nullptr) {
                throw std::logic_error("capture pressure impact has no portfolio checkpoint");
            }
            const auto found = std::find_if(
                impact_scratch_.begin(), impact_scratch_.end(), [&](const CombinedImpact& value) {
                    return value.owner_ordinal == impact.owner_ordinal &&
                           value.checkpoint == impact.checkpoint;
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

        owner_scratch_.clear();
        for (const OwnerPolicy& policy : input.owner_policies) {
            owner_scratch_.push_back(ContextPortfolioOwnerPolicy{
                .ordinal                  = policy.ordinal,
                .private_retention_weight = policy.private_retention_weight,
                .explicit_shared_credit   = policy.explicit_shared_credit,
            });
        }
        const std::uint32_t candidate_ordinal = next_candidate_ordinal(input.owner_policies);
        const bool candidate_credit =
            has_shared_candidate_evidence(input.capture->shared_evidence,
                                          SharedCandidateEvidence::ExplicitBoundary) ||
            has_shared_candidate_evidence(input.capture->shared_evidence,
                                          SharedCandidateEvidence::RequestedAutomatic);
        owner_scratch_.push_back(ContextPortfolioOwnerPolicy{
            .ordinal                  = candidate_ordinal,
            .private_retention_weight = 0,
            .explicit_shared_credit   = candidate_credit,
        });

        checkpoint_scratch_.clear();
        for (const CheckpointPolicy& policy : input.checkpoint_policies) {
            std::uint64_t target_recovery = policy.baseline_recovery_ns;
            if (input.direct_shared_victim_ordinal == policy.owner_ordinal) {
                target_recovery = policy.rebuild_ns;
            } else {
                const auto impact =
                    std::find_if(impact_scratch_.begin(), impact_scratch_.end(),
                                 [&](const CombinedImpact& value) {
                                     return value.owner_ordinal == policy.owner_ordinal &&
                                            value.checkpoint == policy.checkpoint;
                                 });
                if (impact != impact_scratch_.end()) {
                    if (impact->baseline_ns != policy.baseline_recovery_ns) {
                        throw std::logic_error(
                            "capture pressure baseline recovery changed during planning");
                    }
                    target_recovery = impact->target_ns;
                }
            }
            checkpoint_scratch_.push_back(ContextPortfolioCheckpointValue{
                .owner_ordinal        = policy.owner_ordinal,
                .demand_mask          = policy.demand_mask,
                .rebuild_ns           = policy.rebuild_ns,
                .baseline_recovery_ns = policy.baseline_recovery_ns,
                .target_recovery_ns   = target_recovery,
            });
        }
        checkpoint_scratch_.push_back(ContextPortfolioCheckpointValue{
            .owner_ordinal        = candidate_ordinal,
            .demand_mask          = input.candidate_demand_mask,
            .rebuild_ns           = input.candidate_rebuild_ns,
            .baseline_recovery_ns = input.candidate_rebuild_ns,
            .target_recovery_ns   = input.capture->projected_recovery_ns,
        });

        const ContextPortfolioValueResult portfolio =
            portfolio_value_.fold(owner_scratch_, checkpoint_scratch_);
        const std::uint64_t multiplier =
            static_cast<std::uint64_t>(input.blocked_runnable_requests) + 1U;
        const std::uint64_t target_immediate =
            saturating_multiply(assessment.machine.immediate_ns, multiplier);
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

    [[nodiscard]] static std::uint32_t
    next_candidate_ordinal(std::span<const OwnerPolicy> policies) {
        std::uint32_t ordinal = 0;
        for (const OwnerPolicy& policy : policies) {
            if (policy.ordinal == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("capture portfolio owner ordinal overflow");
            }
            ordinal = std::max(ordinal, policy.ordinal + 1U);
        }
        return ordinal;
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
    std::vector<PressureTargetHandle> queue_;
    std::vector<PressureTargetHandle> assessed_;
    std::vector<PressureTargetHandle> expanded_;
    std::vector<CombinedImpact> impact_scratch_;
    std::vector<ContextPortfolioOwnerPolicy> owner_scratch_;
    std::vector<ContextPortfolioCheckpointValue> checkpoint_scratch_;
};

} // namespace ninfer::runtime
