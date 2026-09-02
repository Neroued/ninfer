#pragma once

#include "runtime/engine/context_cost.h"
#include "runtime/engine/context_portfolio_value.h"
#include "runtime/engine/resource_search.h"

#include <algorithm>
#include <array>
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
    PlanningOwnerId owner;
    CheckpointRef checkpoint;
    RetentionClass retention_class     = RetentionClass::RecentPrivate;
    std::uint64_t selected_hit_count   = 0;
    std::uint64_t last_hit_epoch       = 0;
    std::uint32_t demand_mask          = 0;
    std::uint64_t rebuild_ns           = 0;
    std::uint64_t baseline_recovery_ns = 0;
};

struct MaterializationOwnerPolicy {
    PlanningOwnerId owner;
    RetentionClass retention_class         = RetentionClass::RecentPrivate;
    std::uint64_t selected_hit_count       = 0;
    std::uint64_t last_hit_epoch           = 0;
    std::uint32_t private_retention_weight = 0;
    bool explicit_shared_credit            = false;
};

template <class Package>
class MaterializationPlanner {
public:
    using Program                = typename Package::Program;
    using PreparedPrompt         = typename Package::PreparedPrompt;
    using AdmissionCandidate     = typename Package::AdmissionCandidate;
    using ResourcePlan           = typename Package::ResourcePlan;
    using ContinuationHandle     = typename Package::ContinuationHandle;
    using SharedPrefixHandle     = typename Package::SharedPrefixHandle;
    using PressureTargetHandle   = typename Package::PressureTargetHandle;
    using AssessedPressureTarget = typename Package::AssessedPressureTarget;
    using Clock                  = std::chrono::steady_clock;

    struct CandidateInput {
        AdmissionCandidate* candidate = nullptr;
        PlanningCandidateId id;
        std::uint32_t stable_ordinal = 0;
        bool current_session_binding = false;
    };

    struct LogicalGoal {
        std::uint32_t publication_slot = std::numeric_limits<std::uint32_t>::max();
    };

    struct PressureInputs {
        std::span<const ContinuationHandle* const> private_owners;
        std::span<const PlanningOwnerId> private_owner_ids;
        std::span<const SharedPrefixHandle* const> shared_owners;
        std::span<const PlanningOwnerId> shared_owner_ids;
        std::span<const MaterializationOwnerPolicy> owner_policy;
        std::span<const MaterializationCheckpointPolicy> checkpoint_policy;
    };

    struct Result {
        std::optional<ResourcePlan> plan;
        PlanningCandidateId candidate;
        std::uint32_t publication_slot = std::numeric_limits<std::uint32_t>::max();
        PrivateSourceMode source_mode  = PrivateSourceMode::ConsumeToActive;
        std::vector<PressureOwnerOutcome> owner_outcomes;
        std::vector<PressureCheckpointOutcome> checkpoint_outcomes;
        MaterializationDiagnostics diagnostics;
    };

    MaterializationPlanner() : target_ledger_(kTargetBudget + 17U) {
        queue_.reserve(kTargetBudget);
        pending_.reserve(kTargetBudget);
        guided_.reserve(kTargetBudget);
        identity_costs_.reserve(16);
        candidate_guided_steps_.reserve(16);
        candidate_seed_complete_.reserve(16);
        impact_scratch_.reserve(32);
        portfolio_owner_scratch_.reserve(32);
        portfolio_checkpoint_scratch_.reserve(64);
    }

    template <class PressureInputsFn, class LogicalGoalFn, class FinalScheduleFn>
    [[nodiscard]] std::optional<Result>
    plan(Program& program, const PreparedPrompt& prompt,
         const ContextMachineCostModel& machine_cost, std::span<const CandidateInput> candidates,
         std::uint32_t root_candidate_index, PressureInputsFn&& pressure_inputs,
         LogicalGoalFn&& logical_goal, FinalScheduleFn&& final_schedule,
         Clock::time_point planning_started) {
        if (candidates.empty() || root_candidate_index >= candidates.size()) {
            throw std::invalid_argument("materialization planning problem has no root candidate");
        }
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            if (candidates[index].candidate == nullptr ||
                std::find_if(candidates.begin(), candidates.begin() + index,
                             [&](const CandidateInput& prior) {
                                 return prior.id == candidates[index].id;
                             }) != candidates.begin() + index) {
                throw std::invalid_argument("materialization candidate IDs are invalid");
            }
        }
        queue_.clear();
        pending_.clear();
        guided_.clear();
        const std::size_t frontier_capacity = candidates.size() + 1U + kTargetBudget;
        queue_.reserve(frontier_capacity);
        pending_.reserve(frontier_capacity);
        guided_.reserve(frontier_capacity);
        identity_costs_.clear();
        candidate_guided_steps_.assign(candidates.size(), 0);
        candidate_seed_complete_.assign(candidates.size(), false);
        target_ledger_.reset(candidates.size() + 1U + kTargetBudget);

        std::optional<Incumbent> identity_best;
        std::vector<IdentityRoot> roots;
        roots.reserve(candidates.size());
        std::uint64_t projection_work = 0;
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const CandidateInput& input = candidates[index];
            const IdentityMaterializationAssessment& identity =
                input.candidate->identity_assessment();
            planning_saturating_add(projection_work, identity.projection_work);
            const FoldedCost cost = fold_identity(input, identity, machine_cost);
            identity_costs_.push_back(cost);
            std::optional<LogicalGoal> goal;
            if (identity.physical_status == MaterializationPhysicalStatus::Feasible) {
                goal = logical_goal(input.id, identity.source_mode,
                                    std::span<const PressureOwnerOutcome>{});
            }
            if (goal && (!identity_best || cost.less(identity_best->cost))) {
                identity_best = Incumbent{
                    .candidate_index  = static_cast<std::uint32_t>(index),
                    .publication_slot = goal->publication_slot,
                    .source_mode      = identity.source_mode,
                    .cost             = cost,
                };
            }
            if (goal) { candidate_seed_complete_[index] = true; }
            const bool needs_pressure =
                !goal.has_value() &&
                (identity.physical_status == MaterializationPhysicalStatus::Feasible ||
                 identity.expandable);
            const bool pressure_can_improve =
                goal.has_value() &&
                identity.physical_status == MaterializationPhysicalStatus::Feasible &&
                identity.pressure_may_change_machine_work;
            roots.push_back(IdentityRoot{
                .candidate_index = static_cast<std::uint32_t>(index),
                .lower_bound_ns  = cost.lower_bound_ns,
                .expandable      = needs_pressure || pressure_can_improve,
            });
        }

        if (identity_best) {
            const bool needs_optional_search =
                std::any_of(roots.begin(), roots.end(),
                            [](const IdentityRoot& root) { return root.expandable; });
            if (!needs_optional_search) {
                const CandidateInput& selected = candidates[identity_best->candidate_index];
                const auto price_split         = [&](std::span<const std::uint32_t> frontiers) {
                    const std::uint64_t baseline =
                        machine_cost.prefill_ns(selected.candidate->identity_assessment()
                                                            .machine_work.remaining_prefill_work);
                    const std::uint64_t target =
                        machine_cost.prefill_ns(program.shared_capture_split_prefill_work(
                            *selected.candidate, prompt, frontiers));
                    return target > baseline ? target - baseline : 0;
                };
                std::vector<std::uint32_t> shared_frontiers =
                    final_schedule(selected.id, selected.candidate->summary(), price_split);
                std::optional<ResourcePlan> sealed = program.seal_identity(
                    *selected.candidate, prompt,
                    FinalScheduleIntent{.shared_capture_frontiers = shared_frontiers});
                if (!sealed) { return std::nullopt; }
                MaterializationDiagnostics diagnostics = complete_diagnostics(
                    identity_best->cost, static_cast<std::uint32_t>(candidates.size()),
                    projection_work, planning_started, MaterializationStopReason::NoPressure,
                    false);
                Result result;
                result.plan             = std::move(*sealed);
                result.candidate        = candidates[identity_best->candidate_index].id;
                result.publication_slot = identity_best->publication_slot;
                result.source_mode      = identity_best->source_mode;
                result.diagnostics      = diagnostics;
                return result;
            }
        }

        std::vector<const AdmissionCandidate*> candidate_handles;
        std::vector<PlanningCandidateId> candidate_ids;
        candidate_handles.reserve(candidates.size());
        candidate_ids.reserve(candidates.size());
        for (const CandidateInput& input : candidates) {
            candidate_handles.push_back(input.candidate);
            candidate_ids.push_back(input.id);
        }
        const PressureInputs pressure = pressure_inputs();
        if (pressure.private_owners.size() != pressure.private_owner_ids.size() ||
            pressure.shared_owners.size() != pressure.shared_owner_ids.size()) {
            throw std::logic_error("materialization pressure owner arrays are not aligned");
        }
        auto session = program.begin_pressure_planning(
            candidate_handles, candidate_ids, pressure.private_owners, pressure.private_owner_ids,
            pressure.shared_owners, pressure.shared_owner_ids);
        const auto candidate_index_for = [&](PlanningCandidateId id) -> std::uint32_t {
            const auto found =
                std::find_if(candidates.begin(), candidates.end(),
                             [&](const CandidateInput& input) { return input.id == id; });
            if (found == candidates.end()) {
                throw std::logic_error("pressure target references an unknown candidate ID");
            }
            return static_cast<std::uint32_t>(found - candidates.begin());
        };

        Incumbent incumbent;
        std::uint32_t targets_evaluated = static_cast<std::uint32_t>(candidates.size());
        if (identity_best) {
            incumbent        = std::move(*identity_best);
            incumbent.target = session.identity_target(candidates[incumbent.candidate_index].id);
        } else {
            PressureTargetHandle root_maximal =
                session.root_maximal_target(candidates[root_candidate_index].id);
            AssessedPressureTarget assessed            = session.assess(root_maximal);
            const PressureTargetAssessment& assessment = assessed.assessment();
            if (assessment.candidate != candidates[root_candidate_index].id) {
                throw std::logic_error("maximal pressure target changed admission candidate");
            }
            ++targets_evaluated;
            planning_saturating_add(projection_work, assessment.projection_work);
            std::optional<LogicalGoal> goal;
            if (assessment.physical_status == MaterializationPhysicalStatus::Feasible) {
                goal = logical_goal(assessment.candidate, assessment.source_mode,
                                    assessment.owner_outcomes);
            }
            if (!goal) { return std::nullopt; }
            const FoldedCost cost =
                fold_assessment(candidates[root_candidate_index], assessment, pressure.owner_policy,
                                pressure.checkpoint_policy, machine_cost);
            incumbent = make_incumbent(root_maximal, root_candidate_index, assessment,
                                       std::move(assessed), cost, *goal);
            mark_target(assessment.stable_target_ordinal, kTargetDiscovered | kTargetAssessed);
        }

        const Clock::time_point search_started = Clock::now();
        const std::uint64_t search_budget_ns =
            std::min<std::uint64_t>(5'000'000ULL, incumbent.cost.total_ns / 20U);
        const std::uint64_t guided_watchdog_ns = search_budget_ns;
        std::uint64_t maximum_step_ns          = 0;
        std::uint32_t optional_targets         = 0;
        std::uint32_t guided_assessments       = 0;
        MaterializationStopReason stop_reason  = MaterializationStopReason::QueueExhausted;
        bool budget_exhausted                  = false;

        for (const IdentityRoot& root : roots) {
            if (!root.expandable) { continue; }
            QueueEntry entry;
            entry.target            = session.identity_target(candidates[root.candidate_index].id);
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
            mark_target(entry.stable_target_ordinal, kTargetDiscovered | kTargetAssessed);
            queue_push(entry);
            const PressureTargetGuidance guidance = session.guidance(entry.target);
            if (guidance.candidate != candidates[root.candidate_index].id) {
                throw std::logic_error("pressure guidance changed admission candidate");
            }
            guided_insert(GuidedEntry{
                .target          = entry.target,
                .candidate_index = root.candidate_index,
                .lower_bound_ns  = root.lower_bound_ns,
                .guidance        = fold_guidance(candidates[root.candidate_index], guidance,
                                                 pressure.owner_policy, machine_cost),
                .already_assessed_expandable = true,
            });
        }

        const auto make_queue_entry = [](PressureTargetHandle target, std::uint32_t candidate_index,
                                         const PressureTargetAssessment& assessment,
                                         const FoldedCost& cost) {
            return QueueEntry{
                .target                    = target,
                .candidate_index           = candidate_index,
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
            };
        };

        const auto assess_target =
            [&](PressureTargetHandle target, std::uint32_t expected_candidate,
                std::uint32_t expected_ordinal) -> std::optional<QueueEntry> {
            AssessedPressureTarget assessed            = session.assess(target);
            const PressureTargetAssessment& assessment = assessed.assessment();
            if (assessment.candidate != candidates[expected_candidate].id ||
                candidate_index_for(assessment.candidate) != expected_candidate ||
                assessment.stable_target_ordinal != expected_ordinal) {
                throw std::logic_error("pressure target changed admission candidate");
            }
            mark_target(assessment.stable_target_ordinal, kTargetDiscovered | kTargetAssessed);
            ++targets_evaluated;
            planning_saturating_add(projection_work, assessment.projection_work);
            const FoldedCost cost =
                fold_assessment(candidates[expected_candidate], assessment, pressure.owner_policy,
                                pressure.checkpoint_policy, machine_cost);
            std::optional<LogicalGoal> goal;
            if (assessment.physical_status == MaterializationPhysicalStatus::Feasible) {
                goal = logical_goal(assessment.candidate, assessment.source_mode,
                                    assessment.owner_outcomes);
            }
            if (goal && !assessment.root_maximal) {
                candidate_seed_complete_[expected_candidate] = true;
            }
            if (goal && cost.less(incumbent.cost)) {
                incumbent = make_incumbent(target, expected_candidate, assessment,
                                           std::move(assessed), cost, *goal);
            }
            if (!assessment.expandable) { return std::nullopt; }
            QueueEntry entry = make_queue_entry(target, expected_candidate, assessment, cost);
            queue_push(entry);
            return entry;
        };

        const auto expand_target = [&](const QueueEntry& parent) {
            if (target_marked(parent.stable_target_ordinal, kTargetExpanded)) { return true; }
            if (optional_targets >= kTargetBudget) { return false; }
            auto prepared = session.prepare_expansion(parent.target);
            if (prepared.new_canonical_count() > kTargetBudget - optional_targets) {
                session.discard_expansion(std::move(prepared));
                return false;
            }
            const auto children = session.commit_expansion(std::move(prepared));
            optional_targets += children.new_canonical_count;
            mark_target(parent.stable_target_ordinal, kTargetExpanded);
            for (const PressureTargetHandle child : children.children) {
                const PressureTargetGuidance guidance = session.guidance(child);
                if (guidance.candidate != candidates[parent.candidate_index].id) {
                    throw std::logic_error("pressure guidance changed admission candidate");
                }
                const std::uint32_t candidate_index = candidate_index_for(guidance.candidate);
                if (target_marked(guidance.stable_target_ordinal, kTargetDiscovered)) { continue; }
                mark_target(guidance.stable_target_ordinal, kTargetDiscovered);
                const std::uint64_t lower_bound_ns = std::max(
                    identity_costs_[candidate_index].lower_bound_ns, parent.lower_bound_ns);
                const GuidanceCost cost = fold_guidance(candidates[candidate_index], guidance,
                                                        pressure.owner_policy, machine_cost);
                PendingEntry pending{
                    .target          = child,
                    .candidate_index = candidate_index,
                    .lower_bound_ns  = lower_bound_ns,
                    .guidance        = cost,
                };
                pending_push(pending);
                if (!candidate_seed_complete_[candidate_index]) {
                    guided_insert(GuidedEntry{
                        .target          = child,
                        .candidate_index = candidate_index,
                        .lower_bound_ns  = lower_bound_ns,
                        .guidance        = cost,
                    });
                }
            }
            return true;
        };

        const auto candidate_needs_seed = [&](std::uint32_t candidate_index) {
            return candidate_index < roots.size() && roots[candidate_index].expandable &&
                   !candidate_seed_complete_[candidate_index];
        };
        const auto has_open_seed = [&] {
            return std::any_of(roots.begin(), roots.end(), [&](const IdentityRoot& root) {
                return candidate_needs_seed(root.candidate_index);
            });
        };

        std::vector<const MaterializationOwnerPolicy*> preferred_owners;
        preferred_owners.reserve(pressure.owner_policy.size());
        for (const MaterializationOwnerPolicy& policy : pressure.owner_policy) {
            preferred_owners.push_back(&policy);
        }
        std::sort(preferred_owners.begin(), preferred_owners.end(),
                  [](const auto* left, const auto* right) {
                      return std::tuple{
                                 left->selected_hit_count,
                                 left->explicit_shared_credit ? 1U : 0U,
                                 left->private_retention_weight,
                                 left->last_hit_epoch,
                                 left->owner.value,
                             } < std::tuple{
                                     right->selected_hit_count,
                                     right->explicit_shared_credit ? 1U : 0U,
                                     right->private_retention_weight,
                                     right->last_hit_epoch,
                                     right->owner.value,
                                 };
                  });
        std::vector<PlanningOwnerId> preferred_owner_ids;
        preferred_owner_ids.reserve(preferred_owners.size());
        for (const MaterializationOwnerPolicy* policy : preferred_owners) {
            preferred_owner_ids.push_back(policy->owner);
        }

        std::vector<IdentityRoot> closure_order;
        closure_order.reserve(roots.size());
        for (const IdentityRoot& root : roots) {
            if (candidate_needs_seed(root.candidate_index)) { closure_order.push_back(root); }
        }
        std::sort(closure_order.begin(), closure_order.end(),
                  [](const IdentityRoot& left, const IdentityRoot& right) {
                      return std::tuple{left.lower_bound_ns, left.candidate_index} <
                             std::tuple{right.lower_bound_ns, right.candidate_index};
                  });
        for (const IdentityRoot& root : closure_order) {
            if (!candidate_needs_seed(root.candidate_index) ||
                elapsed_ns(search_started, Clock::now()) >= guided_watchdog_ns ||
                optional_targets >= kTargetBudget) {
                continue;
            }
            const Clock::time_point step_started              = Clock::now();
            const std::optional<PressureTargetHandle> closure = session.guided_closure_target(
                candidates[root.candidate_index].id, preferred_owner_ids);
            maximum_step_ns = std::max(maximum_step_ns, elapsed_ns(step_started, Clock::now()));
            if (!closure) { continue; }
            const PressureTargetGuidance closure_guidance = session.guidance(*closure);
            if (closure_guidance.candidate != candidates[root.candidate_index].id) {
                throw std::logic_error("guided closure changed admission candidate");
            }
            if (target_marked(closure_guidance.stable_target_ordinal, kTargetAssessed)) {
                continue;
            }
            if (!target_marked(closure_guidance.stable_target_ordinal, kTargetDiscovered)) {
                mark_target(closure_guidance.stable_target_ordinal, kTargetDiscovered);
                ++optional_targets;
            }
            const Clock::time_point assessment_started = Clock::now();
            (void)assess_target(*closure, root.candidate_index,
                                closure_guidance.stable_target_ordinal);
            ++guided_assessments;
            maximum_step_ns =
                std::max(maximum_step_ns, elapsed_ns(assessment_started, Clock::now()));
        }

        // Build one ordinary feasible seed per expandable candidate. Estimated machine cost orders
        // independent beams but never excludes a candidate or certifies an incumbent.
        while (has_open_seed() && !guided_.empty() &&
               guided_assessments < kGuidedAssessmentBudget) {
            if (elapsed_ns(search_started, Clock::now()) >= guided_watchdog_ns) { break; }
            const GuidedEntry next = guided_pop();
            if (!candidate_needs_seed(next.candidate_index) ||
                target_marked(next.guidance.stable_target_ordinal, kTargetExpanded)) {
                continue;
            }
            std::optional<QueueEntry> exact;
            if (next.already_assessed_expandable) {
                exact = QueueEntry{
                    .target          = next.target,
                    .candidate_index = next.candidate_index,
                    .lower_bound_ns  = next.lower_bound_ns,
                    .remaining_prefill =
                        identity_costs_[next.candidate_index].remaining_text_prefill,
                    .remaining_vision_prefill =
                        identity_costs_[next.candidate_index].remaining_vision_prefill,
                    .reused_prompt_tokens =
                        identity_costs_[next.candidate_index].reused_prompt_tokens,
                    .current_session_binding =
                        identity_costs_[next.candidate_index].current_session_binding,
                    .candidate_ordinal = identity_costs_[next.candidate_index].candidate_ordinal,
                    .stable_target_ordinal = next.guidance.stable_target_ordinal,
                };
            } else if (!target_marked(next.guidance.stable_target_ordinal, kTargetAssessed)) {
                const Clock::time_point step_started = Clock::now();
                exact = assess_target(next.target, next.candidate_index,
                                      next.guidance.stable_target_ordinal);
                ++guided_assessments;
                maximum_step_ns = std::max(maximum_step_ns, elapsed_ns(step_started, Clock::now()));
            }
            if (!exact || !candidate_needs_seed(next.candidate_index)) { continue; }
            if (elapsed_ns(search_started, Clock::now()) >= guided_watchdog_ns) { break; }
            const Clock::time_point step_started = Clock::now();
            if (!expand_target(*exact)) { break; }
            maximum_step_ns = std::max(maximum_step_ns, elapsed_ns(step_started, Clock::now()));
        }

        for (;;) {
            while (!queue_.empty() &&
                   target_marked(queue_.front().stable_target_ordinal, kTargetExpanded)) {
                (void)queue_pop();
            }
            while (
                !pending_.empty() &&
                target_marked(pending_.front().guidance.stable_target_ordinal, kTargetAssessed)) {
                (void)pending_pop();
            }
            if (queue_.empty() && pending_.empty()) {
                stop_reason = MaterializationStopReason::QueueExhausted;
                break;
            }
            const std::uint64_t queue_bound   = queue_.empty()
                                                    ? std::numeric_limits<std::uint64_t>::max()
                                                    : queue_.front().lower_bound_ns;
            const std::uint64_t pending_bound = pending_.empty()
                                                    ? std::numeric_limits<std::uint64_t>::max()
                                                    : pending_.front().lower_bound_ns;
            const std::uint64_t next_bound    = std::min(queue_bound, pending_bound);
            const std::uint64_t elapsed       = elapsed_ns(search_started, Clock::now());
            if (elapsed >= search_budget_ns) {
                stop_reason      = MaterializationStopReason::TimeBudget;
                budget_exhausted = true;
                break;
            }
            const std::uint64_t possible_improvement =
                incumbent.cost.total_ns > next_bound ? incumbent.cost.total_ns - next_bound : 0;
            if (possible_improvement != 0 && maximum_step_ns != 0 &&
                maximum_step_ns >= possible_improvement) {
                stop_reason = MaterializationStopReason::ValueOfNextExpansion;
                break;
            }

            const bool assess_pending =
                !pending_.empty() && (queue_.empty() || pending_bound <= queue_bound);
            const Clock::time_point step_started = Clock::now();
            if (assess_pending) {
                const PendingEntry next = pending_pop();
                if (!target_marked(next.guidance.stable_target_ordinal, kTargetAssessed)) {
                    (void)assess_target(next.target, next.candidate_index,
                                        next.guidance.stable_target_ordinal);
                }
            } else {
                if (optional_targets >= kTargetBudget) {
                    stop_reason      = MaterializationStopReason::TargetBudget;
                    budget_exhausted = true;
                    break;
                }
                const QueueEntry parent = queue_pop();
                if (!expand_target(parent)) {
                    stop_reason      = MaterializationStopReason::ExpansionCapacity;
                    budget_exhausted = true;
                    break;
                }
            }
            maximum_step_ns = std::max(maximum_step_ns, elapsed_ns(step_started, Clock::now()));
        }

        const std::uint64_t search_elapsed_ns = elapsed_ns(search_started, Clock::now());
        if (!incumbent.assessed) {
            AssessedPressureTarget assessed            = session.assess(incumbent.target);
            const PressureTargetAssessment& assessment = assessed.assessment();
            if (assessment.candidate != candidates[incumbent.candidate_index].id ||
                assessment.physical_status != MaterializationPhysicalStatus::Feasible) {
                throw std::logic_error("selected identity target lost exact feasibility");
            }
            incumbent.assessed.emplace(std::move(assessed));
        }
        const CandidateInput& selected = candidates[incumbent.candidate_index];
        const auto price_split         = [&](std::span<const std::uint32_t> frontiers) {
            const std::uint64_t baseline = machine_cost.prefill_ns(
                incumbent.assessed->assessment().machine_work.remaining_prefill_work);
            const std::uint64_t target = machine_cost.prefill_ns(
                session.shared_capture_split_prefill_work(*incumbent.assessed, prompt, frontiers));
            return target > baseline ? target - baseline : 0;
        };
        std::vector<std::uint32_t> shared_frontiers =
            final_schedule(selected.id, selected.candidate->summary(), price_split);
        std::optional<ResourcePlan> sealed =
            session.seal(std::move(*incumbent.assessed), prompt,
                         FinalScheduleIntent{.shared_capture_frontiers = shared_frontiers});
        if (!sealed) { throw std::logic_error("selected pressure target could not be sealed"); }

        MaterializationDiagnostics diagnostics = make_diagnostics(
            incumbent.cost, targets_evaluated, projection_work, planning_started, search_elapsed_ns,
            stop_reason, budget_exhausted, incumbent.degradation_units, incumbent.root_maximal);

        Result result;
        result.plan                = std::move(*sealed);
        result.candidate           = candidates[incumbent.candidate_index].id;
        result.publication_slot    = incumbent.publication_slot;
        result.source_mode         = incumbent.source_mode;
        result.owner_outcomes      = std::move(incumbent.owner_outcomes);
        result.checkpoint_outcomes = std::move(incumbent.checkpoint_outcomes);
        result.diagnostics         = diagnostics;
        return result;
    }

    template <class PressureInputsFn, class LogicalGoalFn>
    [[nodiscard]] std::optional<Result>
    plan(Program& program, const PreparedPrompt& prompt,
         const ContextMachineCostModel& machine_cost, std::span<const CandidateInput> candidates,
         std::uint32_t root_candidate_index, PressureInputsFn&& pressure_inputs,
         LogicalGoalFn&& logical_goal, Clock::time_point planning_started) {
        const auto no_optional_schedule = [](PlanningCandidateId, const RequestPlanSummary&,
                                             const auto&) { return std::vector<std::uint32_t>{}; };
        return plan(program, prompt, machine_cost, candidates, root_candidate_index,
                    std::forward<PressureInputsFn>(pressure_inputs),
                    std::forward<LogicalGoalFn>(logical_goal), no_optional_schedule,
                    planning_started);
    }

private:
    static constexpr std::uint32_t kTargetBudget           = 4096;
    static constexpr std::uint32_t kGuidedBeamWidth        = 16;
    static constexpr std::uint32_t kGuidedAssessmentBudget = 32;

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
        std::optional<AssessedPressureTarget> assessed;
        std::uint32_t candidate_index  = 0;
        std::uint32_t publication_slot = std::numeric_limits<std::uint32_t>::max();
        PrivateSourceMode source_mode  = PrivateSourceMode::ConsumeToActive;
        FoldedCost cost;
        std::vector<PressureOwnerOutcome> owner_outcomes;
        std::vector<PressureCheckpointOutcome> checkpoint_outcomes;
        std::uint32_t degradation_units = 0;
        bool root_maximal               = false;
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

    struct GuidanceCost {
        std::uint32_t estimated_remaining_steps = 0;
        std::uint32_t unsatisfied_constraints   = 0;
        std::uint64_t normalized_residual_q20   = 0;
        std::uint64_t affected_selected_hits    = 0;
        std::uint64_t newest_affected_hit_epoch = 0;
        std::uint64_t retention_weight          = 0;
        std::uint32_t explicit_shared_losses    = 0;
        std::uint32_t owner_evictions           = 0;
        std::uint32_t checkpoint_drops          = 0;
        std::uint32_t degradation_units         = 0;
        std::uint64_t estimated_immediate_ns    = 0;
        std::uint32_t copy_operations           = 0;
        std::uint64_t transferred_bytes         = 0;
        std::uint64_t remaining_prefill         = 0;
        std::uint64_t remaining_vision_prefill  = 0;
        std::uint32_t reused_prompt_tokens      = 0;
        bool current_session_binding            = false;
        std::uint32_t candidate_ordinal         = 0;
        std::uint32_t stable_target_ordinal     = 0;

        [[nodiscard]] auto key() const noexcept {
            return std::tuple{
                affected_selected_hits,
                explicit_shared_losses,
                owner_evictions,
                checkpoint_drops,
                newest_affected_hit_epoch,
                estimated_remaining_steps,
                unsatisfied_constraints,
                normalized_residual_q20,
                retention_weight,
                degradation_units,
                estimated_immediate_ns,
                copy_operations,
                transferred_bytes,
                remaining_prefill,
                remaining_vision_prefill,
                std::numeric_limits<std::uint32_t>::max() - reused_prompt_tokens,
                current_session_binding ? 0U : 1U,
                candidate_ordinal,
                stable_target_ordinal,
            };
        }
    };

    struct PendingEntry {
        PressureTargetHandle target{};
        std::uint32_t candidate_index = 0;
        std::uint64_t lower_bound_ns  = 0;
        GuidanceCost guidance;
    };

    struct GuidedEntry {
        PressureTargetHandle target{};
        std::uint32_t candidate_index = 0;
        std::uint64_t lower_bound_ns  = 0;
        GuidanceCost guidance;
        bool already_assessed_expandable = false;
    };

    struct CombinedImpact {
        PlanningOwnerId owner;
        CheckpointRef checkpoint;
        std::uint64_t target_ns = 0;
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
                     PlanningOwnerId owner) noexcept {
        const auto found = std::find_if(policies.begin(), policies.end(),
                                        [&](const auto& policy) { return policy.owner == owner; });
        return found == policies.end() ? nullptr : &*found;
    }

    [[nodiscard]] static const MaterializationCheckpointPolicy*
    checkpoint_policy_for(std::span<const MaterializationCheckpointPolicy> policies,
                          PlanningOwnerId owner, CheckpointRef checkpoint) noexcept {
        const auto found = std::find_if(policies.begin(), policies.end(), [&](const auto& policy) {
            return policy.owner == owner && policy.checkpoint == checkpoint;
        });
        return found == policies.end() ? nullptr : &*found;
    }

    [[nodiscard]] FoldedCost
    fold_identity(const CandidateInput& candidate,
                  const IdentityMaterializationAssessment& assessment,
                  const ContextMachineCostModel& machine_cost) const noexcept {
        const PricedMaterializationMachineWork priced =
            price_materialization_machine_work(machine_cost, assessment.machine_work);
        FoldedCost cost;
        cost.now_ns                 = priced.immediate_ns;
        cost.total_ns               = cost.now_ns;
        cost.lower_bound_ns         = priced.optimistic_request_ns;
        cost.copy_operations        = priced.copy_operations;
        cost.transferred_bytes      = priced.transferred_bytes;
        cost.remaining_text_prefill = assessment.machine_work.remaining_prefill_work.tokens;
        cost.remaining_vision_prefill =
            assessment.machine_work.remaining_prefill_work.vision_patches;
        cost.reused_prompt_tokens    = assessment.machine_work.reused_prompt_tokens;
        cost.current_session_binding = candidate.current_session_binding;
        cost.candidate_ordinal       = candidate.stable_ordinal;
        cost.target_ordinal          = candidate.stable_ordinal;
        return cost;
    }

    [[nodiscard]] GuidanceCost
    fold_guidance(const CandidateInput& candidate, const PressureTargetGuidance& guidance,
                  std::span<const MaterializationOwnerPolicy> owner_policies,
                  const ContextMachineCostModel& machine_cost) const {
        const PricedMaterializationMachineWork priced =
            price_materialization_machine_work(machine_cost, guidance.estimated_machine_work);
        GuidanceCost cost;
        cost.estimated_remaining_steps = guidance.physical.estimated_remaining_steps;
        cost.unsatisfied_constraints   = guidance.physical.unsatisfied_constraints;
        cost.normalized_residual_q20   = guidance.physical.normalized_residual_q20;
        cost.checkpoint_drops          = guidance.dropped_checkpoints;
        cost.degradation_units         = guidance.degradation_units;
        cost.estimated_immediate_ns    = priced.immediate_ns;
        cost.copy_operations           = priced.copy_operations;
        cost.transferred_bytes         = priced.transferred_bytes;
        cost.remaining_prefill = guidance.estimated_machine_work.remaining_prefill_work.tokens;
        cost.remaining_vision_prefill =
            guidance.estimated_machine_work.remaining_prefill_work.vision_patches;
        cost.reused_prompt_tokens    = guidance.estimated_machine_work.reused_prompt_tokens;
        cost.current_session_binding = candidate.current_session_binding;
        cost.candidate_ordinal       = candidate.stable_ordinal;
        cost.stable_target_ordinal   = guidance.stable_target_ordinal;

        for (const PressureOwnerOutcome& outcome : guidance.owner_outcomes) {
            const MaterializationOwnerPolicy* policy =
                owner_policy_for(owner_policies, outcome.owner);
            if (policy == nullptr) {
                throw std::logic_error("pressure guidance references an unknown logical owner");
            }
            if (outcome.disposition == VictimDisposition::Evicted) { ++cost.owner_evictions; }
            const bool may_reduce_recovery_value =
                outcome.disposition == VictimDisposition::Evicted ||
                outcome.dropped_checkpoints != 0 || outcome.degradation_units != 0;
            if (!may_reduce_recovery_value) { continue; }
            planning_saturating_add(cost.affected_selected_hits, policy->selected_hit_count);
            cost.newest_affected_hit_epoch =
                std::max(cost.newest_affected_hit_epoch, policy->last_hit_epoch);
            planning_saturating_add(cost.retention_weight, policy->private_retention_weight);
            if (policy->explicit_shared_credit) { ++cost.explicit_shared_losses; }
        }
        return cost;
    }

    [[nodiscard]] FoldedCost
    fold_assessment(const CandidateInput& candidate, const PressureTargetAssessment& assessment,
                    std::span<const MaterializationOwnerPolicy> owner_policies,
                    std::span<const MaterializationCheckpointPolicy> checkpoint_policies,
                    const ContextMachineCostModel& machine_cost) {
        const PricedMaterializationMachineWork priced =
            price_materialization_machine_work(machine_cost, assessment.machine_work);
        FoldedCost cost;
        cost.now_ns                 = priced.immediate_ns;
        cost.copy_operations        = priced.copy_operations;
        cost.transferred_bytes      = priced.transferred_bytes;
        cost.remaining_text_prefill = assessment.machine_work.remaining_prefill_work.tokens;
        cost.remaining_vision_prefill =
            assessment.machine_work.remaining_prefill_work.vision_patches;
        cost.reused_prompt_tokens    = assessment.machine_work.reused_prompt_tokens;
        cost.current_session_binding = candidate.current_session_binding;
        cost.candidate_ordinal       = candidate.stable_ordinal;
        cost.target_ordinal          = assessment.stable_target_ordinal;
        cost.checkpoint_drops        = assessment.dropped_checkpoints;

        for (const PressureOwnerOutcome& outcome : assessment.owner_outcomes) {
            const MaterializationOwnerPolicy* policy =
                owner_policy_for(owner_policies, outcome.owner);
            if (policy == nullptr) {
                throw std::logic_error("pressure target references an unknown logical owner");
            }
            if (outcome.disposition == VictimDisposition::Evicted) { ++cost.owner_evictions; }
        }

        impact_scratch_.clear();
        for (const PressureCheckpointRecoveryImpact& impact : assessment.checkpoint_impacts) {
            const auto found = std::find_if(
                impact_scratch_.begin(), impact_scratch_.end(), [&](const CombinedImpact& item) {
                    return item.owner == impact.owner && item.checkpoint == impact.checkpoint;
                });
            if (found == impact_scratch_.end()) {
                if (impact.target_recovery_work.empty()) {
                    throw std::logic_error("pressure recovery impact has no supported recipe");
                }
                impact_scratch_.push_back(CombinedImpact{
                    .owner      = impact.owner,
                    .checkpoint = impact.checkpoint,
                    .target_ns =
                        price_checkpoint_recovery_work(machine_cost, impact.target_recovery_work),
                });
            } else {
                throw std::logic_error("pressure recovery impact is duplicated");
            }
        }
        portfolio_owner_scratch_.clear();
        for (const MaterializationOwnerPolicy& policy : owner_policies) {
            portfolio_owner_scratch_.push_back(ContextPortfolioOwnerPolicy{
                .owner                    = policy.owner,
                .private_retention_weight = policy.private_retention_weight,
                .explicit_shared_credit   = policy.explicit_shared_credit,
            });
        }
        portfolio_checkpoint_scratch_.clear();
        bool portfolio_degraded = false;
        for (const MaterializationCheckpointPolicy& policy : checkpoint_policies) {
            const MaterializationOwnerPolicy* owner =
                owner_policy_for(owner_policies, policy.owner);
            if (owner == nullptr) {
                throw std::logic_error("checkpoint policy has no portfolio owner");
            }
            const auto impact = std::find_if(
                impact_scratch_.begin(), impact_scratch_.end(), [&](const CombinedImpact& value) {
                    return value.owner == policy.owner && value.checkpoint == policy.checkpoint;
                });
            const std::uint64_t target_recovery =
                impact == impact_scratch_.end() ? policy.baseline_recovery_ns : impact->target_ns;
            portfolio_checkpoint_scratch_.push_back(ContextPortfolioCheckpointValue{
                .owner                = policy.owner,
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
        cost.lower_bound_ns = priced.optimistic_request_ns;
        planning_saturating_add(cost.lower_bound_ns, cost.future_loss_ns);
        return cost;
    }

    [[nodiscard]] static Incumbent make_incumbent(PressureTargetHandle target,
                                                  std::uint32_t candidate_index,
                                                  const PressureTargetAssessment& assessment,
                                                  AssessedPressureTarget&& assessed,
                                                  FoldedCost cost, LogicalGoal goal) {
        return Incumbent{
            .target           = target,
            .assessed         = std::move(assessed),
            .candidate_index  = candidate_index,
            .publication_slot = goal.publication_slot,
            .source_mode      = assessment.source_mode,
            .cost             = std::move(cost),
            .owner_outcomes   = std::vector<PressureOwnerOutcome>(assessment.owner_outcomes.begin(),
                                                                  assessment.owner_outcomes.end()),
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
            .degradation_units = assessment.degradation_units,
            .root_maximal      = assessment.root_maximal,
        };
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

    [[nodiscard]] static auto pending_key(const PendingEntry& entry) noexcept {
        return std::tuple{entry.lower_bound_ns, entry.guidance.key()};
    }

    void pending_push(PendingEntry entry) {
        pending_.push_back(std::move(entry));
        std::push_heap(pending_.begin(), pending_.end(), [](const auto& left, const auto& right) {
            return pending_key(right) < pending_key(left);
        });
    }

    [[nodiscard]] PendingEntry pending_pop() {
        std::pop_heap(pending_.begin(), pending_.end(), [](const auto& left, const auto& right) {
            return pending_key(right) < pending_key(left);
        });
        PendingEntry result = std::move(pending_.back());
        pending_.pop_back();
        return result;
    }

    [[nodiscard]] static bool guidance_dominates(const GuidanceCost& left,
                                                 const GuidanceCost& right) noexcept {
        const std::array<std::uint64_t, 13> left_dimensions{
            left.estimated_remaining_steps, left.unsatisfied_constraints,
            left.normalized_residual_q20,   left.affected_selected_hits,
            left.newest_affected_hit_epoch, left.explicit_shared_losses,
            left.retention_weight,          left.owner_evictions,
            left.checkpoint_drops,          left.estimated_immediate_ns,
            left.degradation_units,         left.copy_operations,
            left.transferred_bytes,
        };
        const std::array<std::uint64_t, 13> right_dimensions{
            right.estimated_remaining_steps, right.unsatisfied_constraints,
            right.normalized_residual_q20,   right.affected_selected_hits,
            right.newest_affected_hit_epoch, right.explicit_shared_losses,
            right.retention_weight,          right.owner_evictions,
            right.checkpoint_drops,          right.estimated_immediate_ns,
            right.degradation_units,         right.copy_operations,
            right.transferred_bytes,
        };
        bool strict = false;
        for (std::size_t index = 0; index < left_dimensions.size(); ++index) {
            if (left_dimensions[index] > right_dimensions[index]) { return false; }
            strict = strict || left_dimensions[index] < right_dimensions[index];
        }
        return strict;
    }

    void guided_insert(GuidedEntry entry) {
        if (std::any_of(guided_.begin(), guided_.end(), [&](const GuidedEntry& existing) {
                return existing.candidate_index == entry.candidate_index &&
                       existing.lower_bound_ns <= entry.lower_bound_ns &&
                       guidance_dominates(existing.guidance, entry.guidance);
            })) {
            return;
        }
        std::erase_if(guided_, [&](const GuidedEntry& existing) {
            return existing.candidate_index == entry.candidate_index &&
                   entry.lower_bound_ns <= existing.lower_bound_ns &&
                   guidance_dominates(entry.guidance, existing.guidance);
        });
        guided_.push_back(std::move(entry));
        const std::uint32_t candidate_index = guided_.back().candidate_index;
        const std::size_t candidate_size    = static_cast<std::size_t>(
            std::count_if(guided_.begin(), guided_.end(), [&](const GuidedEntry& item) {
                return item.candidate_index == candidate_index;
            }));
        if (candidate_size <= kGuidedBeamWidth) { return; }
        auto worst = guided_.end();
        for (auto item = guided_.begin(); item != guided_.end(); ++item) {
            if (item->candidate_index != candidate_index) { continue; }
            if (worst == guided_.end() ||
                std::tuple{worst->lower_bound_ns, worst->guidance.key()} <
                    std::tuple{item->lower_bound_ns, item->guidance.key()}) {
                worst = item;
            }
        }
        if (worst == guided_.end()) {
            throw std::logic_error("candidate guided beam accounting is inconsistent");
        }
        guided_.erase(worst);
    }

    [[nodiscard]] GuidedEntry guided_pop() {
        const auto key = [&](const GuidedEntry& entry) {
            if (entry.candidate_index >= candidate_guided_steps_.size()) {
                throw std::logic_error("guided target candidate index is invalid");
            }
            return std::tuple{
                candidate_guided_steps_[entry.candidate_index] == 0 ? 0U : 1U,
                entry.lower_bound_ns,
                entry.guidance.key(),
            };
        };
        const auto best    = std::min_element(guided_.begin(), guided_.end(),
                                              [&](const GuidedEntry& left, const GuidedEntry& right) {
                                               return key(left) < key(right);
                                           });
        GuidedEntry result = std::move(*best);
        guided_.erase(best);
        ++candidate_guided_steps_[result.candidate_index];
        return result;
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

    [[nodiscard]] static MaterializationDiagnostics
    complete_diagnostics(const FoldedCost& cost, std::uint32_t targets_evaluated,
                         std::uint64_t projection_work, Clock::time_point planning_started,
                         MaterializationStopReason reason, bool maximal_fallback) noexcept {
        return make_diagnostics(cost, targets_evaluated, projection_work, planning_started, 0,
                                reason, false, 0, maximal_fallback);
    }

    [[nodiscard]] static MaterializationDiagnostics
    make_diagnostics(const FoldedCost& cost, std::uint32_t targets_evaluated,
                     std::uint64_t projection_work, Clock::time_point planning_started,
                     std::uint64_t search_elapsed_ns, MaterializationStopReason reason,
                     bool budget_exhausted, std::uint32_t degradation_units,
                     bool maximal_fallback) noexcept {
        return MaterializationDiagnostics{
            .predicted_now_ns           = cost.now_ns,
            .predicted_future_loss_ns   = cost.future_loss_ns,
            .predicted_total_ns         = cost.total_ns,
            .targets_evaluated          = targets_evaluated,
            .projection_work            = projection_work,
            .planning_elapsed_ns        = elapsed_ns(planning_started, Clock::now()),
            .search_elapsed_ns          = search_elapsed_ns,
            .stop_reason                = reason,
            .budget_exhausted           = budget_exhausted,
            .selected_degradation_units = degradation_units,
            .selected_maximal_fallback  = maximal_fallback,
        };
    }

    std::vector<QueueEntry> queue_;
    std::vector<PendingEntry> pending_;
    std::vector<GuidedEntry> guided_;
    std::vector<FoldedCost> identity_costs_;
    std::vector<std::uint32_t> candidate_guided_steps_;
    std::vector<std::uint8_t> candidate_seed_complete_;
    BoundedTargetLedger target_ledger_;
    std::vector<CombinedImpact> impact_scratch_;
    ContextPortfolioValue portfolio_value_;
    std::vector<ContextPortfolioOwnerPolicy> portfolio_owner_scratch_;
    std::vector<ContextPortfolioCheckpointValue> portfolio_checkpoint_scratch_;
};

} // namespace ninfer::runtime
