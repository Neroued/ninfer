#pragma once

#include "targets/qwen3_6/impl/runtime/program.h"

#include <array>
#include <limits>
#include <tuple>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

inline void planning_saturating_add(std::uint64_t& value, std::uint64_t add) noexcept {
    value = add > std::numeric_limits<std::uint64_t>::max() - value
                ? std::numeric_limits<std::uint64_t>::max()
                : value + add;
}

inline std::uint32_t planning_saturating_u32(std::uint64_t value) noexcept {
    return value > std::numeric_limits<std::uint32_t>::max()
               ? std::numeric_limits<std::uint32_t>::max()
               : static_cast<std::uint32_t>(value);
}

inline std::size_t planning_direction_index(runtime::ContextTransferDirection direction) {
    switch (direction) {
    case runtime::ContextTransferDirection::DeviceToHost:
        return 0;
    case runtime::ContextTransferDirection::HostToDevice:
        return 1;
    case runtime::ContextTransferDirection::DeviceToDevice:
        return 2;
    }
    throw std::logic_error("materialization transfer direction is invalid");
}

struct PlanningTransferAccumulator {
    std::array<TransferWork, 3> work{};
    std::uint64_t bytes      = 0;
    std::uint64_t operations = 0;

    void append(std::span<const runtime::ContextTransferRequirement> requirements) noexcept {
        for (const runtime::ContextTransferRequirement& requirement : requirements) {
            const std::size_t index = planning_direction_index(requirement.direction);
            planning_saturating_add(work[index].payload_bytes, requirement.work.payload_bytes);
            work[index].copy_operations =
                planning_saturating_u32(static_cast<std::uint64_t>(work[index].copy_operations) +
                                        requirement.work.copy_operations);
            planning_saturating_add(bytes, requirement.work.payload_bytes);
            planning_saturating_add(operations, requirement.work.copy_operations);
        }
    }
};

inline std::uint64_t price_transfer_accumulator(const runtime::ContextMachineCostModel& model,
                                                const PlanningTransferAccumulator& accumulator,
                                                runtime::MaterializationCopyPhase phase) noexcept {
    std::array<runtime::TransferBatchWork, 3> batches{};
    std::size_t count = 0;
    constexpr std::array directions{
        runtime::ContextTransferDirection::DeviceToHost,
        runtime::ContextTransferDirection::HostToDevice,
        runtime::ContextTransferDirection::DeviceToDevice,
    };
    for (std::size_t index = 0; index < accumulator.work.size(); ++index) {
        const TransferWork work = accumulator.work[index];
        if (work.payload_bytes == 0 && work.copy_operations == 0) { continue; }
        batches[count++] = runtime::TransferBatchWork{
            .phase     = phase,
            .direction = directions[index],
            .work      = work,
        };
    }
    return model.transfer_batches_ns(
        std::span<const runtime::TransferBatchWork>(batches.data(), count));
}

inline runtime::MaterializationMachineSummary materialization_machine_summary(
    const AdmissionCandidateImpl& candidate,
    std::span<const qwen3_6::detail::PressureDecision> private_decisions,
    std::span<const qwen3_6::detail::PressureDecision> shared_decisions,
    const runtime::ContextMachineCostModel& model) noexcept {
    PlanningTransferAccumulator pressure;
    for (const qwen3_6::detail::PressureDecision& decision : private_decisions) {
        pressure.append(decision.transfer_requirements);
    }
    for (const qwen3_6::detail::PressureDecision& decision : shared_decisions) {
        pressure.append(decision.transfer_requirements);
    }

    PlanningTransferAccumulator request;
    request.append(candidate.transfer_requirements);

    PlanningTransferAccumulator minimum_request;
    for (const runtime::ContextTransferRequirement& requirement : candidate.transfer_requirements) {
        const bool pressure_may_eliminate =
            candidate.has_source &&
            candidate.source_disposition == runtime::ClaimDisposition::ConsumedToActive &&
            requirement.direction == runtime::ContextTransferDirection::DeviceToDevice;
        if (!pressure_may_eliminate) {
            minimum_request.append(
                std::span<const runtime::ContextTransferRequirement>(&requirement, 1));
        }
    }

    const std::uint64_t prefill = model.prefill_ns(candidate.remaining_prefill_work);
    std::uint64_t immediate     = prefill;
    planning_saturating_add(
        immediate, price_transfer_accumulator(model, pressure,
                                              runtime::MaterializationCopyPhase::PressureToHost));
    planning_saturating_add(
        immediate,
        price_transfer_accumulator(model, request, runtime::MaterializationCopyPhase::Candidate));

    std::uint64_t minimum_request_ns = prefill;
    planning_saturating_add(minimum_request_ns, price_transfer_accumulator(
                                                    model, minimum_request,
                                                    runtime::MaterializationCopyPhase::Candidate));

    runtime::MaterializationMachineSummary summary{
        .minimum_request_ns     = minimum_request_ns,
        .immediate_ns           = immediate,
        .remaining_prefill_work = candidate.remaining_prefill_work,
        .transferred_bytes      = pressure.bytes,
        .copy_operations        = planning_saturating_u32(pressure.operations),
        .reused_prompt_tokens   = candidate.summary.reusable_prompt_tokens,
    };
    planning_saturating_add(summary.transferred_bytes, request.bytes);
    summary.copy_operations = planning_saturating_u32(
        static_cast<std::uint64_t>(summary.copy_operations) + request.operations);
    return summary;
}

inline std::uint64_t
recovery_cost_ns(std::span<const runtime::ContextTransferRequirement> requirements,
                 runtime::PrefillWork prefill_work,
                 const runtime::ContextMachineCostModel& model) noexcept {
    PlanningTransferAccumulator transfer;
    transfer.append(requirements);
    std::uint64_t cost =
        price_transfer_accumulator(model, transfer, runtime::MaterializationCopyPhase::Candidate);
    planning_saturating_add(cost, model.prefill_ns(prefill_work));
    return cost;
}

inline std::uint32_t degradation_units(const qwen3_6::detail::PressureDecision& decision) noexcept {
    std::uint64_t units = decision.evicts_continuation ? 1U : 0U;
    units += decision.state_changes.size();
    units += decision.main_kv_changes.size();
    units += decision.backend_kv_changes.size();
    units += decision.checkpoint_drops;
    return planning_saturating_u32(units);
}

inline std::uint32_t
dropped_checkpoint_count(const qwen3_6::detail::PressureDecision& decision) noexcept {
    return decision.checkpoint_drops;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

namespace planning_detail {

inline constexpr std::size_t kOptionalTargetCapacity = 4096;

template <class Node>
[[nodiscard]] bool same_target(const Node& left, const Node& right) noexcept {
    return left.candidate_index == right.candidate_index &&
           left.owner_choices == right.owner_choices;
}

inline void hash_mix(std::uint64_t& hash, std::uint64_t value) noexcept {
    hash ^= value;
    hash *= 1099511628211ULL;
}

} // namespace planning_detail

inline PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::PressurePlanningSessionImpl(
    Core& owner, const runtime::ContextMachineCostModel& cost,
    std::span<const AdmissionCandidate* const> admission_candidates,
    std::span<const ContinuationHandle* const> private_owners,
    std::span<const std::uint32_t> private_owner_ordinals,
    std::span<const SharedPrefixHandle* const> shared_owners,
    std::span<const std::uint32_t> shared_owner_ordinals)
    : program(&owner), machine_cost(&cost), resource_revision(owner.resource_revision()) {
    if (admission_candidates.empty() || private_owners.size() != private_owner_ordinals.size() ||
        shared_owners.size() != shared_owner_ordinals.size() || owner.has_context_transaction() ||
        owner.pending_transaction_ || owner.pressure_planning_active_) {
        throw std::logic_error("pressure planning session cannot start in the current state");
    }

    candidates.assign(admission_candidates.begin(), admission_candidates.end());
    owners.reserve(private_owners.size() + shared_owners.size());
    for (std::size_t index = 0; index < private_owners.size(); ++index) {
        const ContinuationHandle* handle = private_owners[index];
        if (handle == nullptr || !owner.valid_continuation(*handle)) {
            throw std::logic_error("pressure planning private owner is stale");
        }
        owners.push_back(Owner{
            .private_handle = handle, .ordinal = private_owner_ordinals[index], .shared = false});
    }
    for (std::size_t index = 0; index < shared_owners.size(); ++index) {
        const SharedPrefixHandle* handle = shared_owners[index];
        if (handle == nullptr || !owner.valid_shared_prefix(*handle)) {
            throw std::logic_error("pressure planning shared owner is stale");
        }
        owners.push_back(Owner{
            .shared_handle = handle, .ordinal = shared_owner_ordinals[index], .shared = true});
    }
    std::sort(owners.begin(), owners.end(), [](const Owner& left, const Owner& right) {
        return std::tuple{left.ordinal, left.shared} < std::tuple{right.ordinal, right.shared};
    });
    for (std::size_t index = 1; index < owners.size(); ++index) {
        if (owners[index - 1].ordinal == owners[index].ordinal) {
            throw std::logic_error("pressure planning owner ordinal is duplicated");
        }
    }

    candidate_options.resize(candidates.size());
    targets.reserve(candidates.size() + 1U + planning_detail::kOptionalTargetCapacity);
    expansion_scratch.reserve(owners.size() * 8U);
    committed_children.reserve(owners.size() * 8U);
    selected_private_owners.reserve(private_owners.size());
    selected_private_decisions.reserve(private_owners.size());
    selected_shared_owners.reserve(shared_owners.size());
    selected_shared_decisions.reserve(shared_owners.size());
    recovery_private_owners.reserve(private_owners.size());
    recovery_private_decisions.reserve(private_owners.size());
    recovery_private_ordinals.reserve(private_owners.size());
    recovery_shared_owners.reserve(shared_owners.size());
    recovery_shared_decisions.reserve(shared_owners.size());
    recovery_shared_ordinals.reserve(shared_owners.size());
    assessment_outcomes.reserve(owners.size());
    assessment_impacts.reserve(owners.size() * 4U);

    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const AdmissionCandidate* candidate = candidates[index];
        if (candidate == nullptr || candidate->impl_ == nullptr ||
            candidate->impl_->planning_revision != resource_revision) {
            throw std::logic_error("pressure planning candidate is stale");
        }
        targets.push_back(TargetNode{
            .candidate_index = static_cast<std::uint32_t>(index),
            .owner_choices   = std::vector<std::uint16_t>(owners.size(), 0),
            .stable_ordinal  = static_cast<std::uint32_t>(index),
        });
    }

    if (++owner.pressure_planning_generation_ == 0) { ++owner.pressure_planning_generation_; }
    generation                      = owner.pressure_planning_generation_;
    owner.pressure_planning_active_ = true;
}

inline PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::~PressurePlanningSessionImpl() noexcept {
    if (program != nullptr) { program->pressure_planning_active_ = false; }
}

inline bool PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::valid(
    qwen3_6::PressureTargetHandle target) const noexcept {
    return target.session_ == this && target.generation_ == generation &&
           target.index_ < targets.size() && program != nullptr &&
           program->resource_revision() == resource_revision;
}

inline std::uint32_t PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::candidate_index(
    const AdmissionCandidate& candidate) const {
    const auto found = std::find(candidates.begin(), candidates.end(), &candidate);
    if (found == candidates.end()) {
        throw std::invalid_argument("pressure target candidate does not belong to this session");
    }
    return static_cast<std::uint32_t>(found - candidates.begin());
}

inline qwen3_6::PressureTargetHandle
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::identity_target(
    const AdmissionCandidate& candidate) const {
    qwen3_6::PressureTargetHandle handle;
    handle.session_    = this;
    handle.generation_ = generation;
    handle.index_      = candidate_index(candidate);
    return handle;
}

inline void PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::populate_options(
    std::uint32_t selected_candidate) {
    if (selected_candidate >= candidate_options.size()) {
        throw std::out_of_range("pressure candidate index is invalid");
    }
    CandidateOptions& options = candidate_options[selected_candidate];
    if (options.populated) { return; }
    options.owners.resize(owners.size());
    options.eviction_choices.resize(owners.size(), 0);
    const AdmissionCandidate& candidate = *candidates[selected_candidate];
    const std::optional<typename Core::MaterializationSourceProtection> protection =
        program->materialization_source_protection(*candidate.impl_);
    if (!protection) {
        throw std::logic_error("pressure planning candidate source protection is stale");
    }
    for (std::size_t index = 0; index < owners.size(); ++index) {
        const Owner& owner                       = owners[index];
        std::vector<PressureDecision>& decisions = options.owners[index];
        using PlanningContractAccess =
            qwen3_6::detail::RuntimeContractAccess<NINFER_QWEN36_VARIANT>;
        const bool selected_private_source =
            !owner.shared && candidate.impl_->has_source &&
            PlanningContractAccess::index(*owner.private_handle) == candidate.impl_->source_index &&
            PlanningContractAccess::epoch(*owner.private_handle) ==
                candidate.impl_->source_generation;
        const bool selected_shared_source = owner.shared && candidate.impl_->has_shared_source &&
                                            PlanningContractAccess::index(*owner.shared_handle) ==
                                                candidate.impl_->shared_source_index &&
                                            PlanningContractAccess::epoch(*owner.shared_handle) ==
                                                candidate.impl_->shared_source_generation;
        if (selected_private_source || selected_shared_source) { continue; }
        if (owner.shared) {
            PressureDecision eviction = program->inspect_shared_eviction_option(
                program->shared_prefix_states[PlanningContractAccess::index(*owner.shared_handle)]);
            if (!eviction.evicts_continuation || !eviction.shared_owner) {
                throw std::logic_error("shared pressure owner has no maximal outcome");
            }
            decisions.push_back(std::move(eviction));
        } else {
            PressureDecision eviction = program->inspect_eviction_option(
                program->continuation_states[PlanningContractAccess::index(*owner.private_handle)]);
            if (!eviction.evicts_continuation || eviction.shared_owner) {
                throw std::logic_error("private pressure owner has no maximal outcome");
            }
            decisions.push_back(std::move(eviction));
        }
        if (decisions.size() > std::numeric_limits<std::uint16_t>::max()) {
            throw std::overflow_error("pressure owner target count is not representable");
        }
        options.eviction_choices[index] = static_cast<std::uint16_t>(decisions.size());
    }
    options.populated = true;
}

inline qwen3_6::PressureTargetHandle
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::root_maximal_target(
    const AdmissionCandidate& root_candidate) {
    if (scratch_live) { throw std::logic_error("pressure expansion scratch is still live"); }
    const std::uint32_t selected_candidate = candidate_index(root_candidate);
    populate_options(selected_candidate);
    TargetNode maximal{
        .candidate_index = selected_candidate,
        .owner_choices   = std::vector<std::uint16_t>(owners.size(), 0),
        .root_maximal    = true,
    };
    for (std::size_t index = 0; index < owners.size(); ++index) {
        maximal.owner_choices[index] =
            candidate_options[selected_candidate].eviction_choices[index];
    }
    const auto existing = std::find_if(targets.begin(), targets.end(), [&](const TargetNode& node) {
        return planning_detail::same_target(node, maximal);
    });
    std::uint32_t target_index = 0;
    if (existing != targets.end()) {
        target_index           = static_cast<std::uint32_t>(existing - targets.begin());
        existing->root_maximal = true;
    } else {
        maximal.stable_ordinal = static_cast<std::uint32_t>(targets.size());
        targets.push_back(std::move(maximal));
        target_index = static_cast<std::uint32_t>(targets.size() - 1U);
    }
    qwen3_6::PressureTargetHandle handle;
    handle.session_    = this;
    handle.generation_ = generation;
    handle.index_      = target_index;
    return handle;
}

inline runtime::PressureTargetAssessment
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::assess(qwen3_6::PressureTargetHandle target) {
    if (!valid(target) || scratch_live) {
        throw std::logic_error("pressure target assessment is stale or conflicts with expansion");
    }
    const TargetNode& node = targets[target.index_];
    populate_options(node.candidate_index);
    const AdmissionCandidate& candidate = *candidates[node.candidate_index];
    const CandidateOptions& options     = candidate_options[node.candidate_index];
    const bool identity_target = std::all_of(node.owner_choices.begin(), node.owner_choices.end(),
                                             [](std::uint16_t choice) { return choice == 0; });

    selected_private_owners.clear();
    selected_private_decisions.clear();
    selected_shared_owners.clear();
    selected_shared_decisions.clear();
    recovery_private_owners.clear();
    recovery_private_decisions.clear();
    recovery_private_ordinals.clear();
    recovery_shared_owners.clear();
    recovery_shared_decisions.clear();
    recovery_shared_ordinals.clear();
    assessment_outcomes.clear();
    assessment_impacts.clear();

    std::uint64_t projection_work   = 1;
    std::uint32_t total_degradation = 0;
    std::uint32_t total_dropped     = 0;
    for (std::size_t index = 0; index < owners.size(); ++index) {
        const std::uint16_t choice = node.owner_choices[index];
        if (choice > options.owners[index].size()) {
            throw std::logic_error("pressure target owner choice is invalid");
        }
        const Owner& owner = owners[index];
        const PressureDecision* decision =
            choice == 0 ? nullptr : &options.owners[index][choice - 1U];
        if (owner.shared) {
            recovery_shared_owners.push_back(owner.shared_handle);
            recovery_shared_decisions.push_back(decision);
            recovery_shared_ordinals.push_back(owner.ordinal);
        } else {
            recovery_private_owners.push_back(owner.private_handle);
            recovery_private_decisions.push_back(decision);
            recovery_private_ordinals.push_back(owner.ordinal);
        }
        if (decision == nullptr) { continue; }
        if (owner.shared) {
            selected_shared_owners.push_back(owner.shared_handle);
            selected_shared_decisions.push_back(*decision);
        } else {
            selected_private_owners.push_back(owner.private_handle);
            selected_private_decisions.push_back(*decision);
        }
        const std::uint32_t units   = NINFER_QWEN36_RUNTIME_NS::degradation_units(*decision);
        const std::uint32_t dropped = NINFER_QWEN36_RUNTIME_NS::dropped_checkpoint_count(*decision);
        total_degradation           = NINFER_QWEN36_RUNTIME_NS::planning_saturating_u32(
            static_cast<std::uint64_t>(total_degradation) + units);
        total_dropped = NINFER_QWEN36_RUNTIME_NS::planning_saturating_u32(
            static_cast<std::uint64_t>(total_dropped) + dropped);
        assessment_outcomes.push_back(runtime::PressureOwnerOutcome{
            .owner_ordinal     = owner.ordinal,
            .disposition       = decision->evicts_continuation ? runtime::ClaimDisposition::Evicted
                                                               : runtime::ClaimDisposition::Retained,
            .degradation_units = units,
            .dropped_checkpoints = dropped,
            .shared              = owner.shared,
        });
        ++projection_work;
    }

    bool recovery_projection_valid = true;
    if (!identity_target) {
        recovery_projection_valid = program->pressure_checkpoint_recovery_impacts(
            *candidate.impl_, recovery_private_owners, recovery_private_decisions,
            recovery_private_ordinals, recovery_shared_owners, recovery_shared_decisions,
            recovery_shared_ordinals, *machine_cost, assessment_impacts, projection_work);
    }

    runtime::MaterializationPhysicalStatus status =
        runtime::MaterializationPhysicalStatus::StructuralInvalid;
    std::optional<AdmissionCandidate> composed;
    const qwen3_6::detail::AdmissionCandidateImpl<NINFER_QWEN36_VARIANT>* projected =
        candidate.impl_.get();
    if (identity_target) {
        status = candidate.impl_->identity_assessment.physical_status;
    } else if (recovery_projection_valid) {
        AdmissionCandidate copy(
            std::make_unique<qwen3_6::detail::AdmissionCandidateImpl<NINFER_QWEN36_VARIANT>>(
                *candidate.impl_));
        composed = program->compose_materialization(
            std::move(copy), selected_private_owners, selected_private_decisions,
            selected_shared_owners, selected_shared_decisions);
        if (composed) {
            projected = composed->impl_.get();
            status    = runtime::MaterializationPhysicalStatus::Infeasible;
            if (projected->blocked_host_allocation_bytes == 0 &&
                program->physical_peak_fits(projected->demand.physical_peak_additional)) {
                status = runtime::MaterializationPhysicalStatus::Feasible;
            }
        }
    }
    const runtime::MaterializationMachineSummary machine =
        identity_target
            ? candidate.impl_->identity_assessment.machine
            : NINFER_QWEN36_RUNTIME_NS::materialization_machine_summary(
                  *projected, selected_private_decisions, selected_shared_decisions, *machine_cost);

    bool expandable = identity_target || (recovery_projection_valid && composed.has_value());
    if (expandable) {
        expandable = false;
        for (std::size_t index = 0; index < owners.size(); ++index) {
            const std::uint16_t choice = node.owner_choices[index];
            if ((choice == 0 && !options.owners[index].empty()) ||
                (choice != 0 && choice <= options.owners[index].size() &&
                 !options.owners[index][choice - 1U].evicts_continuation)) {
                expandable = true;
                break;
            }
        }
    }

    std::uint64_t digest = candidate.impl_->identity_assessment.assessment_digest;
    if (!identity_target) {
        digest = 1469598103934665603ULL;
        planning_detail::hash_mix(digest, node.candidate_index);
        planning_detail::hash_mix(digest, node.stable_ordinal);
        planning_detail::hash_mix(digest, static_cast<std::uint8_t>(status));
        planning_detail::hash_mix(digest, machine.immediate_ns);
        planning_detail::hash_mix(digest, total_degradation);
        for (const std::uint16_t choice : node.owner_choices) {
            planning_detail::hash_mix(digest, choice);
        }
    }

    return runtime::PressureTargetAssessment{
        .physical_status       = status,
        .source_disposition    = projected->source_disposition,
        .machine               = machine,
        .owner_outcomes        = assessment_outcomes,
        .checkpoint_impacts    = assessment_impacts,
        .candidate_ordinal     = node.candidate_index,
        .stable_target_ordinal = node.stable_ordinal,
        .degradation_units     = total_degradation,
        .dropped_checkpoints   = total_dropped,
        .projection_work       = projection_work,
        .assessment_digest     = digest,
        .expandable            = expandable,
        .root_maximal          = node.root_maximal,
    };
}

inline qwen3_6::PreparedPressureExpansion<NINFER_QWEN36_VARIANT>
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::prepare_expansion(
    qwen3_6::PressureTargetHandle parent) {
    if (!valid(parent) || scratch_live) {
        throw std::logic_error("pressure expansion parent is stale or scratch is busy");
    }
    const TargetNode& node = targets[parent.index_];
    populate_options(node.candidate_index);
    CandidateOptions& options = candidate_options[node.candidate_index];
    expansion_scratch.clear();
    prepared_owner_decisions.clear();
    prepared_new_count = 0;

    const AdmissionCandidate& candidate = *candidates[node.candidate_index];
    const std::optional<typename Core::MaterializationSourceProtection> protection =
        program->materialization_source_protection(*candidate.impl_);
    if (!protection) { throw std::logic_error("pressure expansion source protection is stale"); }
    selected_private_owners.clear();
    selected_private_decisions.clear();
    selected_shared_owners.clear();
    selected_shared_decisions.clear();
    bool identity = true;
    for (std::size_t index = 0; index < owners.size(); ++index) {
        const std::uint16_t choice = node.owner_choices[index];
        if (choice == 0) { continue; }
        identity = false;
        if (choice > options.owners[index].size()) {
            throw std::logic_error("pressure expansion owner choice is invalid");
        }
        const PressureDecision& decision = options.owners[index][choice - 1U];
        if (owners[index].shared) {
            selected_shared_owners.push_back(owners[index].shared_handle);
            selected_shared_decisions.push_back(decision);
        } else {
            selected_private_owners.push_back(owners[index].private_handle);
            selected_private_decisions.push_back(decision);
        }
    }
    detail::PhysicalResources residual = candidate.impl_->identity_pressure_deficit;
    if (!identity) {
        AdmissionCandidate copy(
            std::make_unique<qwen3_6::detail::AdmissionCandidateImpl<NINFER_QWEN36_VARIANT>>(
                *candidate.impl_));
        std::optional<AdmissionCandidate> composed = program->compose_materialization(
            std::move(copy), selected_private_owners, selected_private_decisions,
            selected_shared_owners, selected_shared_decisions);
        if (composed) {
            residual = program->materialization_deficit(*composed->impl_);
            residual.host.kv_bytes =
                std::max(residual.host.kv_bytes, composed->impl_->blocked_host_allocation_bytes);
        }
    }

    const auto append = [&](TargetNode child) {
        const bool duplicate_scratch = std::any_of(
            expansion_scratch.begin(), expansion_scratch.end(), [&](const TargetNode& existing) {
                return planning_detail::same_target(existing, child);
            });
        if (duplicate_scratch) { return; }
        const bool existing =
            std::any_of(targets.begin(), targets.end(), [&](const TargetNode& item) {
                return planning_detail::same_target(item, child);
            });
        if (!existing) { ++prepared_new_count; }
        expansion_scratch.push_back(std::move(child));
    };

    const auto intern_prepared_decision = [&](std::size_t owner_index, PressureDecision decision) {
        std::vector<PressureDecision>& decisions = options.owners[owner_index];
        const auto existing = std::find(decisions.begin(), decisions.end(), decision);
        if (existing != decisions.end()) {
            return static_cast<std::uint16_t>(1U + (existing - decisions.begin()));
        }
        const auto prepared =
            std::find_if(prepared_owner_decisions.begin(), prepared_owner_decisions.end(),
                         [&](const PreparedOwnerDecision& item) {
                             return item.candidate_index == node.candidate_index &&
                                    item.owner_index == owner_index && item.decision == decision;
                         });
        if (prepared != prepared_owner_decisions.end()) { return prepared->choice; }
        const std::size_t staged = static_cast<std::size_t>(
            std::count_if(prepared_owner_decisions.begin(), prepared_owner_decisions.end(),
                          [&](const PreparedOwnerDecision& item) {
                              return item.candidate_index == node.candidate_index &&
                                     item.owner_index == owner_index;
                          }));
        const std::size_t value = decisions.size() + staged + 1U;
        if (value > std::numeric_limits<std::uint16_t>::max()) {
            throw std::overflow_error("pressure owner target count is not representable");
        }
        const std::uint16_t choice = static_cast<std::uint16_t>(value);
        prepared_owner_decisions.push_back(PreparedOwnerDecision{
            .candidate_index = node.candidate_index,
            .owner_index     = static_cast<std::uint32_t>(owner_index),
            .choice          = choice,
            .decision        = std::move(decision),
        });
        return choice;
    };

    const auto successors_for = [&](std::size_t owner_index, const PressureDecision* current) {
        std::vector<PressureDecision> successors;
        using PlanningContractAccess =
            qwen3_6::detail::RuntimeContractAccess<NINFER_QWEN36_VARIANT>;
        if (owners[owner_index].shared) {
            successors = program->inspect_shared_pressure_successors(
                program->shared_prefix_states[PlanningContractAccess::index(
                    *owners[owner_index].shared_handle)],
                residual, &*protection, current);
        } else {
            successors = program->inspect_pressure_successors(
                program->continuation_states[PlanningContractAccess::index(
                    *owners[owner_index].private_handle)],
                residual, &*protection, current);
        }
        const std::uint16_t eviction_choice = options.eviction_choices[owner_index];
        if (eviction_choice != 0) {
            successors.push_back(options.owners[owner_index][eviction_choice - 1U]);
        }
        return successors;
    };

    for (std::size_t owner_index = 0; owner_index < owners.size(); ++owner_index) {
        const std::uint16_t eviction_choice = options.eviction_choices[owner_index];
        if (eviction_choice == 0) { continue; }
        const std::uint16_t current_choice       = node.owner_choices[owner_index];
        std::vector<PressureDecision>& decisions = options.owners[owner_index];
        if (current_choice > decisions.size() ||
            (current_choice != 0 && decisions[current_choice - 1U].evicts_continuation)) {
            continue;
        }
        const PressureDecision* current =
            current_choice == 0 ? nullptr : &decisions[current_choice - 1U];
        std::vector<PressureDecision> successors = successors_for(owner_index, current);
        for (PressureDecision& successor : successors) {
            const std::uint16_t choice =
                intern_prepared_decision(owner_index, std::move(successor));
            if (choice == current_choice) { continue; }
            TargetNode child                 = node;
            child.owner_choices[owner_index] = choice;
            child.root_maximal               = false;
            append(std::move(child));
        }
    }

    if (++scratch_generation == 0) { ++scratch_generation; }
    scratch_live = true;
    return qwen3_6::PreparedPressureExpansion<NINFER_QWEN36_VARIANT>(
        this, generation, scratch_generation, parent.index_, prepared_new_count);
}

inline qwen3_6::PressureExpansionView
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::commit_expansion(
    qwen3_6::PreparedPressureExpansion<NINFER_QWEN36_VARIANT>&& prepared) {
    if (!scratch_live || prepared.session_ != this || prepared.session_generation_ != generation ||
        prepared.scratch_generation_ != scratch_generation ||
        prepared.new_canonical_count_ != prepared_new_count ||
        prepared.parent_index_ >= targets.size()) {
        throw std::logic_error("prepared pressure expansion is stale");
    }
    const std::size_t maximum = candidates.size() + 1U + planning_detail::kOptionalTargetCapacity;
    if (prepared_new_count > maximum - std::min(maximum, targets.size())) {
        throw std::length_error("prepared pressure expansion exceeds the target arena");
    }

    committed_children.clear();
    committed_children.reserve(expansion_scratch.size());
    for (PreparedOwnerDecision& prepared_decision : prepared_owner_decisions) {
        if (prepared_decision.candidate_index >= candidate_options.size()) {
            throw std::logic_error("prepared pressure owner candidate is invalid");
        }
        CandidateOptions& options = candidate_options[prepared_decision.candidate_index];
        if (prepared_decision.owner_index >= options.owners.size()) {
            throw std::logic_error("prepared pressure owner index is invalid");
        }
        std::vector<PressureDecision>& decisions = options.owners[prepared_decision.owner_index];
        if (prepared_decision.choice != decisions.size() + 1U) {
            throw std::logic_error("prepared pressure owner choice is not canonical");
        }
        decisions.push_back(std::move(prepared_decision.decision));
    }
    for (TargetNode& child : expansion_scratch) {
        auto existing = std::find_if(targets.begin(), targets.end(), [&](const TargetNode& item) {
            return planning_detail::same_target(item, child);
        });
        std::uint32_t index = 0;
        if (existing == targets.end()) {
            child.stable_ordinal = static_cast<std::uint32_t>(targets.size());
            targets.push_back(std::move(child));
            index = static_cast<std::uint32_t>(targets.size() - 1U);
        } else {
            index = static_cast<std::uint32_t>(existing - targets.begin());
        }
        qwen3_6::PressureTargetHandle handle;
        handle.session_    = this;
        handle.generation_ = generation;
        handle.index_      = index;
        committed_children.push_back(handle);
    }
    const std::uint32_t new_count = prepared_new_count;
    prepared.session_             = nullptr;
    prepared.session_generation_  = 0;
    prepared.scratch_generation_  = 0;
    expansion_scratch.clear();
    prepared_owner_decisions.clear();
    prepared_new_count = 0;
    scratch_live       = false;
    return qwen3_6::PressureExpansionView{
        .children            = committed_children,
        .new_canonical_count = new_count,
    };
}

inline void PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::discard_expansion(
    qwen3_6::PreparedPressureExpansion<NINFER_QWEN36_VARIANT>&& prepared) noexcept {
    if (scratch_live && prepared.session_ == this && prepared.session_generation_ == generation &&
        prepared.scratch_generation_ == scratch_generation) {
        expansion_scratch.clear();
        prepared_owner_decisions.clear();
        prepared_new_count = 0;
        scratch_live       = false;
    }
    prepared.session_            = nullptr;
    prepared.session_generation_ = 0;
    prepared.scratch_generation_ = 0;
}

inline std::optional<
    typename PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::AdmissionCandidate>
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::seal(
    qwen3_6::PressureTargetHandle target,
    const NINFER_QWEN36_RUNTIME_NS::PreparedPromptData& prompt) {
    if (!valid(target) || scratch_live) {
        throw std::logic_error("pressure target seal is stale or conflicts with expansion");
    }
    const TargetNode& node = targets[target.index_];
    populate_options(node.candidate_index);
    const AdmissionCandidate& candidate = *candidates[node.candidate_index];
    const CandidateOptions& options     = candidate_options[node.candidate_index];
    selected_private_owners.clear();
    selected_private_decisions.clear();
    selected_shared_owners.clear();
    selected_shared_decisions.clear();
    for (std::size_t index = 0; index < owners.size(); ++index) {
        const std::uint16_t choice = node.owner_choices[index];
        if (choice == 0) { continue; }
        if (choice > options.owners[index].size()) {
            throw std::logic_error("pressure target owner choice is invalid at seal");
        }
        const PressureDecision& decision = options.owners[index][choice - 1U];
        if (owners[index].shared) {
            selected_shared_owners.push_back(owners[index].shared_handle);
            selected_shared_decisions.push_back(decision);
        } else {
            selected_private_owners.push_back(owners[index].private_handle);
            selected_private_decisions.push_back(decision);
        }
    }
    return program->seal_materialization(candidate, prompt, selected_private_owners,
                                         selected_private_decisions, selected_shared_owners,
                                         selected_shared_decisions);
}

inline std::optional<
    typename PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::AdmissionCandidate>
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::seal_capture(
    qwen3_6::PressureTargetHandle target) {
    if (!valid(target) || scratch_live) {
        throw std::logic_error("capture pressure target seal is stale or conflicts with expansion");
    }
    const TargetNode& node = targets[target.index_];
    populate_options(node.candidate_index);
    const AdmissionCandidate& candidate = *candidates[node.candidate_index];
    if (!candidate.impl_->capture_pressure) {
        throw std::invalid_argument("capture pressure seal received a request candidate");
    }
    const CandidateOptions& options = candidate_options[node.candidate_index];
    selected_private_owners.clear();
    selected_private_decisions.clear();
    selected_shared_owners.clear();
    selected_shared_decisions.clear();
    for (std::size_t index = 0; index < owners.size(); ++index) {
        const std::uint16_t choice = node.owner_choices[index];
        if (choice == 0) { continue; }
        if (choice > options.owners[index].size()) {
            throw std::logic_error("capture pressure owner choice is invalid at seal");
        }
        const PressureDecision& decision = options.owners[index][choice - 1U];
        if (owners[index].shared) {
            selected_shared_owners.push_back(owners[index].shared_handle);
            selected_shared_decisions.push_back(decision);
        } else {
            selected_private_owners.push_back(owners[index].private_handle);
            selected_private_decisions.push_back(decision);
        }
    }
    AdmissionCandidate copy(
        std::make_unique<qwen3_6::detail::AdmissionCandidateImpl<NINFER_QWEN36_VARIANT>>(
            *candidate.impl_));
    std::optional<AdmissionCandidate> composed = program->compose_materialization(
        std::move(copy), selected_private_owners, selected_private_decisions,
        selected_shared_owners, selected_shared_decisions);
    if (!composed || composed->impl_->blocked_host_allocation_bytes != 0 ||
        !program->physical_peak_fits(composed->impl_->demand.physical_peak_additional)) {
        return std::nullopt;
    }
    return composed;
}

} // namespace ninfer::targets::qwen3_6::detail
