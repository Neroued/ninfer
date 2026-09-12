#pragma once

#include "core/math_util.h"
#include "runtime/engine/context_cost.h"
#include "runtime/engine/context_portfolio_value.h"
#include "runtime/engine/materialization_budget.h"
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

template <class Package, class SearchClock = std::chrono::steady_clock>
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
    using Clock                  = SearchClock;

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
        identity_costs_.reserve(16);
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
         Clock::time_point planning_started, PlanningAllowance allowance = {}) {
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
        const std::size_t frontier_capacity = candidates.size() + 1U + kTargetBudget;
        queue_.reserve(frontier_capacity);
        pending_.reserve(frontier_capacity);
        identity_costs_.clear();
        target_ledger_.reset(candidates.size() + 1U + kTargetBudget);

        // The mandatory root-maximal fallback does not count as an ordinary feasible seed.
        std::vector<bool> candidate_seeded(candidates.size(), false);
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
            candidate_seeded[index] = goal.has_value();
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
            const bool no_allowance =
                allowance.remaining(planning_now_ns<Clock>()) == 0 ||
                identity_best->cost.total_ns / 20U / std::max(1U, allowance.affected_requests) == 0;
            if (!needs_optional_search || no_allowance) {
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
                    projection_work, planning_started,
                    needs_optional_search ? MaterializationStopReason::TimeBudget
                                          : MaterializationStopReason::NoPressure,
                    false);
                diagnostics.budget_exhausted  = needs_optional_search;
                diagnostics.search_stop_phase = needs_optional_search
                                                    ? MaterializationSearchPhase::Setup
                                                    : MaterializationSearchPhase::None;
                diagnostics.search_boundary_limited =
                    needs_optional_search && allowance.remaining(planning_now_ns<Clock>()) == 0;
                Result result;
                result.plan             = std::move(*sealed);
                result.candidate        = candidates[identity_best->candidate_index].id;
                result.publication_slot = identity_best->publication_slot;
                result.source_mode      = identity_best->source_mode;
                result.diagnostics      = diagnostics;
                return result;
            }
        }

        typename Clock::time_point search_started = Clock::now();
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

        if (!identity_best) { search_started = Clock::now(); }
        const auto search_origin_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(search_started.time_since_epoch())
                .count());
        MaterializationSearchBudget search_budget(allowance, search_origin_ns,
                                                  incumbent.cost.total_ns);
        const auto initial_cost_ns = incumbent.cost.total_ns;
        std::optional<std::uint64_t> first_improvement_ns;
        std::uint32_t incumbent_improvements = 0;
        std::uint64_t option_step_ns = 1'000, assessment_step_ns = 20'000,
                      expansion_step_ns = 20'000;
        std::uint64_t search_work       = 0;
        const auto work_limit           = static_cast<std::uint64_t>(kTargetBudget) *
                                (16U + 16ULL * pressure.owner_policy.size());
        std::uint32_t optional_targets        = 0;
        MaterializationStopReason stop_reason = MaterializationStopReason::QueueExhausted;
        bool budget_exhausted                 = false;
        auto search_phase                     = MaterializationSearchPhase::Setup;
        const auto allow_work = [&](std::uint64_t operation, std::uint64_t completion,
                                    std::uint64_t gain, bool complete,
                                    bool discovery_eligible = true) {
            if (search_work >= work_limit) {
                stop_reason      = MaterializationStopReason::WorkBudget;
                budget_exhausted = true;
                return false;
            }
            if (!search_budget.allow(planning_now_ns<Clock>(), operation, completion, gain,
                                     complete, search_work, discovery_eligible)) {
                stop_reason      = search_budget.stop_reason();
                budget_exhausted = stop_reason == MaterializationStopReason::TimeBudget;
                return false;
            }
            stop_reason      = MaterializationStopReason::QueueExhausted;
            budget_exhausted = false;
            return true;
        };
        const auto observe_step = [&](std::uint64_t& estimate, Clock::time_point started) {
            const auto sample = elapsed_ns(started, Clock::now());
            estimate          = std::max<std::uint64_t>(1, estimate / 2 + sample / 2);
        };


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
            if (goal) {
                mark_target(assessment.stable_target_ordinal, kTargetFeasible);
                candidate_seeded[expected_candidate] = true;
            }
            if (goal && cost.less(incumbent.cost)) {
                if (cost.total_ns < initial_cost_ns && !first_improvement_ns) {
                    first_improvement_ns = elapsed_ns(search_started, Clock::now());
                }
                ++incumbent_improvements;
                incumbent = make_incumbent(target, expected_candidate, assessment,
                                           std::move(assessed), cost, *goal);
            }
            if (!assessment.expandable) { return std::nullopt; }
            mark_target(assessment.stable_target_ordinal, kTargetExpandable);
            QueueEntry entry = make_queue_entry(target, expected_candidate, assessment, cost);
            queue_push(entry);
            return entry;
        };

        const auto expand_target = [&](const QueueEntry& parent) {
            if (target_marked(parent.stable_target_ordinal, kTargetExpanded)) { return true; }
            if (optional_targets >= kTargetBudget) { return false; }
            auto prepared = session.prepare_expansion(parent.target, 8);
            if (prepared.new_canonical_count() > kTargetBudget - optional_targets) {
                session.discard_expansion(std::move(prepared));
                return false;
            }
            const auto children = session.commit_expansion(std::move(prepared));
            optional_targets += children.new_canonical_count;
            if (children.complete) {
                mark_target(parent.stable_target_ordinal, kTargetExpanded);
            } else {
                queue_push(parent);
            }
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
                GuidanceCost cost =
                    fold_guidance(candidates[candidate_index], guidance, pressure.owner_policy,
                                  pressure.checkpoint_policy, machine_cost);
                cost.logical_ready = logical_goal(candidates[candidate_index].id,
                                                  guidance.source_mode, guidance.owner_outcomes)
                                         .has_value();
                PendingEntry pending{
                    .target          = child,
                    .candidate_index = candidate_index,
                    .lower_bound_ns  = lower_bound_ns,
                    .guidance        = cost,
                };
                pending_push(pending);
            }
            return true;
        };

        using Cursor = decltype(session.begin_construction(incumbent.target));

        struct ConstructionPath {
            std::optional<Cursor> cursor;
            std::uint32_t candidate_index = 0;
            bool feasibility_first        = false;
            bool repair                   = false;
            bool restore                  = false;
            GuidanceCost parent;
            std::optional<GuidanceCost> best;
            PressureConstructionOptionId best_option;
            std::vector<std::uint32_t> visited;
        };

        std::array<ConstructionPath, 4> paths;
        std::vector<std::uint32_t> order;
        for (const auto& root : roots) {
            if (root.expandable) { order.push_back(root.candidate_index); }
        }
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
            return identity_costs_[a].lower_bound_ns < identity_costs_[b].lower_bound_ns;
        });
        const auto rank_guidance = [&](std::uint32_t index, const PressureTargetGuidance& guide) {
            auto cost = fold_guidance(candidates[index], guide, pressure.owner_policy,
                                      pressure.checkpoint_policy, machine_cost);
            cost.logical_ready =
                logical_goal(candidates[index].id, guide.source_mode, guide.owner_outcomes)
                    .has_value();
            if (!cost.logical_ready) {
                ++cost.unsatisfied_constraints;
                cost.normalized_residual_q20 += 1ULL << 20U;
                cost.estimated_remaining_steps = std::max(1U, cost.estimated_remaining_steps);
            }
            return cost;
        };
        const auto start_path = [&](ConstructionPath& path, std::uint32_t candidate,
                                    PressureTargetHandle target, bool feasibility, bool restore) {
            path.cursor.reset();
            path.candidate_index   = candidate;
            path.feasibility_first = feasibility;
            path.restore           = restore;
            path.best.reset();
            path.visited.clear();
            path.visited.reserve(kTargetBudget);
            path.parent = rank_guidance(candidate, session.guidance(target));
            path.repair = path.parent.requires_exact_feedback;
            path.visited.push_back(path.parent.stable_target_ordinal);
            path.cursor.emplace(session.begin_construction(target, restore));
        };
        std::size_t next_path  = 0;
        bool search_stopped    = false;
        bool rescue_done       = false;
        bool refinement_seeded = false;
        const auto have_paths  = [&] {
            return std::any_of(paths.begin(), paths.end(),
                                [](const auto& path) { return bool(path.cursor); });
        };
        while (!search_stopped &&
               (next_path < 2U * order.size() || have_paths() || !refinement_seeded)) {
            if (next_path == 2U * order.size() && !have_paths()) {
                if (!rescue_done) {
                    rescue_done = true;
                    const auto promising =
                        std::find_if(order.begin(), order.end(), [&](auto candidate) {
                            return !candidate_seeded[candidate] &&
                                   identity_costs_[candidate].lower_bound_ns <
                                       incumbent.cost.total_ns;
                        });
                    if (promising != order.end() && optional_targets < kTargetBudget) {
                        const auto candidate = *promising;
                        const auto gain =
                            incumbent.cost.total_ns - identity_costs_[candidate].lower_bound_ns;
                        search_phase = MaterializationSearchPhase::Assessment;
                        if (allow_work(assessment_step_ns, assessment_step_ns, gain, false)) {
                            const auto rescue = session.maximal_target(candidates[candidate].id);
                            const auto guide  = session.guidance(rescue);
                            if (!target_marked(guide.stable_target_ordinal, kTargetAssessed)) {
                                if (!target_marked(guide.stable_target_ordinal,
                                                   kTargetDiscovered)) {
                                    ++optional_targets;
                                }
                                const auto started = Clock::now();
                                (void)assess_target(rescue, candidate, guide.stable_target_ordinal);
                                observe_step(assessment_step_ns, started);
                                ++search_work;
                            }
                        }
                    }
                }
                refinement_seeded = true;
                if (incumbent.degradation_units != 0) {
                    start_path(paths[0], incumbent.candidate_index, incumbent.target, false, true);
                }
                if (!have_paths()) { break; }
            }
            for (auto& path : paths) {
                if (!path.cursor && next_path < 2U * order.size()) {
                    const auto candidate = order[next_path / 2];
                    start_path(path, candidate, session.identity_target(candidates[candidate].id),
                               (next_path % 2) != 0, false);
                    ++next_path;
                }
                if (!path.cursor) { continue; }
                for (unsigned slice = 0; slice < 8 && path.cursor; ++slice) {
                    const auto& forecast  = path.best ? *path.best : path.parent;
                    const auto optimistic = identity_costs_[path.candidate_index].lower_bound_ns;
                    const bool complete =
                        forecast.unsatisfied_constraints == 0 && forecast.recovery_complete;
                    const auto estimate = complete ? forecast.estimated_total_ns : optimistic;
                    const auto gain =
                        incumbent.cost.total_ns > estimate ? incumbent.cost.total_ns - estimate : 0;
                    const auto steps =
                        std::min<std::uint64_t>(64, 1ULL + forecast.estimated_remaining_steps);
                    const auto completion =
                        assessment_step_ns +
                        option_step_ns * steps *
                            std::max<std::size_t>(1, pressure.owner_policy.size());
                    search_phase = path.restore ? MaterializationSearchPhase::Refinement
                                                : MaterializationSearchPhase::Construction;
                    if (!allow_work(option_step_ns, completion, gain, complete,
                                    !candidate_seeded[path.candidate_index])) {
                        search_stopped = search_work >= work_limit ||
                                         allowance.remaining(planning_now_ns<Clock>()) == 0;
                        path.cursor.reset();
                        break;
                    }
                    const auto step_started = Clock::now();
                    const auto step         = session.next_construction_option(*path.cursor);
                    observe_step(option_step_ns, step_started);
                    ++search_work;
                    if (step.guidance) {
                        auto cost = rank_guidance(path.candidate_index, *step.guidance);
                        if (construction_option_better(path.parent, path.best, cost,
                                                       path.feasibility_first, path.restore)) {
                            path.best        = cost;
                            path.best_option = step.option;
                        }
                    }
                    if (!step.exhausted) { continue; }
                    if (!path.best || optional_targets >= kTargetBudget) {
                        path.cursor.reset();
                        break;
                    }
                    session.choose_construction(*path.cursor, path.best_option);
                    const auto target = session.construction_target(*path.cursor);
                    if (!target) {
                        stop_reason      = MaterializationStopReason::ExpansionCapacity;
                        budget_exhausted = search_stopped = true;
                        break;
                    }
                    auto chosen = rank_guidance(path.candidate_index, session.guidance(*target));
                    if (std::find(path.visited.begin(), path.visited.end(),
                                  chosen.stable_target_ordinal) != path.visited.end()) {
                        path.cursor.reset();
                        break;
                    }
                    path.visited.push_back(chosen.stable_target_ordinal);
                    if (!target_marked(chosen.stable_target_ordinal, kTargetDiscovered)) {
                        mark_target(chosen.stable_target_ordinal, kTargetDiscovered);
                        ++optional_targets;
                        pending_push({.target          = *target,
                                      .candidate_index = path.candidate_index,
                                      .lower_bound_ns  = optimistic,
                                      .guidance        = chosen});
                    }
                    path.parent = chosen;
                    path.best.reset();
                    if (chosen.unsatisfied_constraints != 0 && !path.repair && !path.restore) {
                        continue;
                    }
                    if (!target_marked(chosen.stable_target_ordinal, kTargetAssessed)) {
                        const auto target_gain =
                            incumbent.cost.total_ns > chosen.estimated_total_ns
                                ? incumbent.cost.total_ns - chosen.estimated_total_ns
                                : 0;
                        search_phase = MaterializationSearchPhase::Assessment;
                        if (!allow_work(assessment_step_ns, assessment_step_ns,
                                        chosen.recovery_complete ? target_gain : gain,
                                        chosen.recovery_complete &&
                                            chosen.unsatisfied_constraints == 0,
                                        !candidate_seeded[path.candidate_index])) {
                            search_stopped = search_work >= work_limit ||
                                             allowance.remaining(planning_now_ns<Clock>()) == 0;
                            path.cursor.reset();
                            break;
                        }
                        const auto started = Clock::now();
                        (void)assess_target(*target, path.candidate_index,
                                            chosen.stable_target_ordinal);
                        observe_step(assessment_step_ns, started);
                        ++search_work;
                    }
                    const bool feasible =
                        target_marked(chosen.stable_target_ordinal, kTargetFeasible);
                    path.cursor.reset();
                    if (!feasible && !path.restore &&
                        target_marked(chosen.stable_target_ordinal, kTargetExpandable)) {
                        // The Program resumes from its exact residual/geometry evidence.
                        path.cursor.emplace(session.begin_construction(*target));
                        path.repair = true;
                    }
                }
                if (search_stopped) { break; }
            }
        }
        for (auto& path : paths) { path.cursor.reset(); }

        // Ordinary alternatives remain available: construction heuristics are not pruning proofs.
        for (;;) {
            if (search_stopped) { break; }
            while (!queue_.empty() &&
                   target_marked(queue_.front().stable_target_ordinal, kTargetExpanded)) {
                (void)queue_pop();
            }
            while (
                !pending_.empty() &&
                target_marked(pending_.front().guidance.stable_target_ordinal, kTargetAssessed)) {
                (void)pending_pop();
            }
            if (queue_.empty() && pending_.empty()) { break; }
            const bool assess_pending =
                !pending_.empty() && (queue_.empty() || pending_.front().lower_bound_ns <=
                                                            queue_.front().lower_bound_ns);
            const auto estimate = assess_pending ? pending_.front().guidance.estimated_total_ns
                                                 : queue_.front().lower_bound_ns;
            const auto gain =
                incumbent.cost.total_ns > estimate ? incumbent.cost.total_ns - estimate : 0;
            const auto predicted_step = assess_pending ? assessment_step_ns : expansion_step_ns;
            search_phase              = assess_pending ? MaterializationSearchPhase::Assessment
                                                       : MaterializationSearchPhase::Expansion;
            if (!allow_work(predicted_step,
                            predicted_step + (assess_pending ? 0 : assessment_step_ns), gain,
                            assess_pending && pending_.front().guidance.recovery_complete &&
                                pending_.front().guidance.unsatisfied_constraints == 0 &&
                                pending_.front().guidance.logical_ready,
                            !candidate_seeded[assess_pending ? pending_.front().candidate_index
                                                             : queue_.front().candidate_index])) {
                if (search_work >= work_limit ||
                    allowance.remaining(planning_now_ns<Clock>()) == 0) {
                    break;
                }
                // A forecast that cannot justify another window must not starve a different source.
                if (assess_pending) {
                    (void)pending_pop();
                } else {
                    (void)queue_pop();
                }
                continue;
            }
            const auto started = Clock::now();
            if (assess_pending) {
                const auto next = pending_pop();
                (void)assess_target(next.target, next.candidate_index,
                                    next.guidance.stable_target_ordinal);
                observe_step(assessment_step_ns, started);
            } else {
                if (optional_targets >= kTargetBudget) {
                    stop_reason      = MaterializationStopReason::TargetBudget;
                    budget_exhausted = true;
                    break;
                }
                if (!expand_target(queue_pop())) {
                    stop_reason      = MaterializationStopReason::ExpansionCapacity;
                    budget_exhausted = true;
                    break;
                }
                observe_step(expansion_step_ns, started);
            }
            ++search_work;
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

        diagnostics.initial_predicted_total_ns = initial_cost_ns;
        diagnostics.first_improvement_ns       = first_improvement_ns;
        diagnostics.incumbent_improvements     = incumbent_improvements;
        diagnostics.search_work                = search_work;
        diagnostics.search_granted_ns          = search_budget.granted_ns();
        diagnostics.search_renewals            = search_budget.renewals();
        diagnostics.search_discovery_used      = search_budget.discovery_used();
        diagnostics.search_stop_phase          = search_phase;
        diagnostics.search_boundary_limited    = search_budget.boundary_limited();
        diagnostics.search_overshoot_ns        = search_elapsed_ns > search_budget.granted_ns()
                                                     ? search_elapsed_ns - search_budget.granted_ns()
                                                     : 0;
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
         LogicalGoalFn&& logical_goal, Clock::time_point planning_started,
         PlanningAllowance allowance = {}) {
        const auto no_optional_schedule = [](PlanningCandidateId, const RequestPlanSummary&,
                                             const auto&) { return std::vector<std::uint32_t>{}; };
        return plan(program, prompt, machine_cost, candidates, root_candidate_index,
                    std::forward<PressureInputsFn>(pressure_inputs),
                    std::forward<LogicalGoalFn>(logical_goal), no_optional_schedule,
                    planning_started, allowance);
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
        std::uint64_t estimated_total_ns        = 0;
        bool recovery_complete                  = false;
        bool requires_exact_feedback            = false;
        bool logical_ready                      = false;
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
                estimated_total_ns,
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

    [[nodiscard]] static bool construction_option_better(const GuidanceCost& parent,
                                                         const std::optional<GuidanceCost>& best,
                                                         const GuidanceCost& cost,
                                                         bool feasibility_first,
                                                         bool restore) noexcept {
        if (!best) { return true; }
        const auto& prior         = *best;
        const bool complete       = cost.unsatisfied_constraints == 0;
        const bool prior_complete = prior.unsatisfied_constraints == 0;
        if (restore || (complete && prior_complete)) { return cost.key() < prior.key(); }
        if (feasibility_first) {
            if (complete != prior_complete) { return complete; }
            return std::tuple{cost.estimated_remaining_steps, cost.normalized_residual_q20,
                              cost.key()} < std::tuple{prior.estimated_remaining_steps,
                                                       prior.normalized_residual_q20, prior.key()};
        }
        const auto relief = [&](const GuidanceCost& item) {
            return parent.normalized_residual_q20 > item.normalized_residual_q20
                       ? parent.normalized_residual_q20 - item.normalized_residual_q20
                       : 0;
        };
        const auto a = relief(cost), b = relief(prior);
        if ((a != 0) != (b != 0)) { return a != 0; }
        if (a && b) {
            const auto delta = [&](const GuidanceCost& item) {
                return item.estimated_total_ns > parent.estimated_total_ns
                           ? item.estimated_total_ns - parent.estimated_total_ns
                           : 0;
            };
            std::uint64_t left_high = 0, right_high = 0;
            const auto left_low = core::u128_mul(delta(cost), b, &left_high);
            const auto right_low = core::u128_mul(delta(prior), a, &right_high);
            if (left_high != right_high) { return left_high < right_high; }
            if (left_low != right_low) { return left_low < right_low; }
        }
        return cost.key() < prior.key();
    }

    struct PendingEntry {
        PressureTargetHandle target{};
        std::uint32_t candidate_index = 0;
        std::uint64_t lower_bound_ns  = 0;
        GuidanceCost guidance;
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
                  std::span<const MaterializationCheckpointPolicy> checkpoint_policies,
                  const ContextMachineCostModel& machine_cost) {
        const PricedMaterializationMachineWork priced =
            price_materialization_machine_work(machine_cost, guidance.estimated_machine_work);
        GuidanceCost cost;
        cost.estimated_remaining_steps = guidance.physical.estimated_remaining_steps;
        cost.unsatisfied_constraints   = guidance.physical.unsatisfied_constraints;
        cost.normalized_residual_q20   = guidance.physical.normalized_residual_q20;
        cost.requires_exact_feedback   = guidance.physical.requires_exact_feedback;
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
        portfolio_owner_scratch_.clear();
        for (const auto& policy : owner_policies) {
            portfolio_owner_scratch_.push_back(
                {.owner                    = policy.owner,
                 .private_retention_weight = policy.private_retention_weight,
                 .explicit_shared_credit   = policy.explicit_shared_credit});
        }
        portfolio_checkpoint_scratch_.clear();
        for (const auto& checkpoint : checkpoint_policies) {
            const auto outcome =
                std::find_if(guidance.owner_outcomes.begin(), guidance.owner_outcomes.end(),
                             [&](const auto& item) { return item.owner == checkpoint.owner; });
            const auto change =
                std::find_if(guidance.checkpoint_changes.begin(), guidance.checkpoint_changes.end(),
                             [&](const auto& item) {
                                 return item.owner == checkpoint.owner &&
                                        item.checkpoint == checkpoint.checkpoint;
                             });
            std::uint64_t recovery = checkpoint.baseline_recovery_ns;
            if ((outcome != guidance.owner_outcomes.end() &&
                 outcome->disposition == VictimDisposition::Evicted) ||
                (change != guidance.checkpoint_changes.end() && !change->survives)) {
                recovery = checkpoint.rebuild_ns;
            } else {
                const auto estimate = std::find_if(
                    guidance.recovery_estimates.begin(), guidance.recovery_estimates.end(),
                    [&](const auto& item) { return item.owner == checkpoint.owner; });
                if (estimate != guidance.recovery_estimates.end()) {
                    for (std::size_t direction = 0; direction < 3; ++direction) {
                        planning_saturating_add(
                            recovery, machine_cost.transfer_ns(
                                          static_cast<ContextTransferDirection>(direction),
                                          estimate->additional_restore[direction]));
                    }
                }
            }
            portfolio_checkpoint_scratch_.push_back(
                {.owner                = checkpoint.owner,
                 .demand_mask          = checkpoint.demand_mask,
                 .rebuild_ns           = checkpoint.rebuild_ns,
                 .baseline_recovery_ns = checkpoint.baseline_recovery_ns,
                 .target_recovery_ns   = recovery});
        }
        const auto value =
            portfolio_value_.fold(portfolio_owner_scratch_, portfolio_checkpoint_scratch_);
        cost.estimated_total_ns = priced.immediate_ns;
        planning_saturating_add(cost.estimated_total_ns,
                                value.baseline_public_value > value.target_public_value
                                    ? value.baseline_public_value - value.target_public_value
                                    : 0);
        planning_saturating_add(cost.estimated_total_ns, value.private_transition_loss);
        cost.recovery_complete = guidance.recovery_estimate_complete && !value.saturated;
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

    static constexpr std::uint8_t kTargetFeasible   = 8U;
    static constexpr std::uint8_t kTargetExpandable = 16U;
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
            .initial_predicted_total_ns = cost.total_ns,
        };
    }

    std::vector<QueueEntry> queue_;
    std::vector<PendingEntry> pending_;
    std::vector<FoldedCost> identity_costs_;
    BoundedTargetLedger target_ledger_;
    std::vector<CombinedImpact> impact_scratch_;
    ContextPortfolioValue portfolio_value_;
    std::vector<ContextPortfolioOwnerPolicy> portfolio_owner_scratch_;
    std::vector<ContextPortfolioCheckpointValue> portfolio_checkpoint_scratch_;
};

} // namespace ninfer::runtime
