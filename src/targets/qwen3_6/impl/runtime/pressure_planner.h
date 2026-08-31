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

inline detail::PhysicalResources planning_resource_sum(detail::PhysicalResources left,
                                                       detail::PhysicalResources right) {
    const auto add_u32 = [](std::uint32_t lhs, std::uint32_t rhs) {
        if (rhs > std::numeric_limits<std::uint32_t>::max() - lhs) {
            throw std::overflow_error("pressure guidance resource sum overflow");
        }
        return static_cast<std::uint32_t>(lhs + rhs);
    };
    if (right.host.kv_bytes > std::numeric_limits<std::size_t>::max() - left.host.kv_bytes) {
        throw std::overflow_error("pressure guidance Host KV sum overflow");
    }
    return detail::PhysicalResources{
        .device =
            {
                .active_lanes  = add_u32(left.device.active_lanes, right.device.active_lanes),
                .state_slots   = add_u32(left.device.state_slots, right.device.state_slots),
                .main_kv_pages = add_u32(left.device.main_kv_pages, right.device.main_kv_pages),
                .backend_kv_pages =
                    add_u32(left.device.backend_kv_pages, right.device.backend_kv_pages),
            },
        .host =
            {
                .state_slots = add_u32(left.host.state_slots, right.host.state_slots),
                .kv_bytes    = left.host.kv_bytes + right.host.kv_bytes,
            },
    };
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

inline runtime::MaterializationMachineSummary
materialization_machine_summary(const AdmissionCandidateImpl& candidate,
                                const PlanningTransferAccumulator& pressure,
                                const runtime::ContextMachineCostModel& model) noexcept {
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

inline runtime::MaterializationMachineSummary materialization_machine_summary(
    const AdmissionCandidateImpl& candidate,
    std::span<const qwen3_6::detail::PressureDecision* const> private_decisions,
    std::span<const qwen3_6::detail::PressureDecision* const> shared_decisions,
    const runtime::ContextMachineCostModel& model) noexcept {
    PlanningTransferAccumulator pressure;
    for (const qwen3_6::detail::PressureDecision* decision : private_decisions) {
        if (decision != nullptr) { pressure.append(decision->transfer_requirements); }
    }
    for (const qwen3_6::detail::PressureDecision* decision : shared_decisions) {
        if (decision != nullptr) { pressure.append(decision->transfer_requirements); }
    }
    return materialization_machine_summary(candidate, pressure, model);
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

template <class Node>
[[nodiscard]] std::uint64_t target_hash(const Node& node) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    hash_mix(hash, node.candidate_index);
    for (const std::uint16_t choice : node.owner_choices) { hash_mix(hash, choice); }
    return hash;
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
    const std::size_t maximum_targets =
        candidates.size() + 1U + planning_detail::kOptionalTargetCapacity;
    targets.reserve(maximum_targets);
    std::size_t hash_capacity = 1;
    while (hash_capacity < 2U * maximum_targets) { hash_capacity <<= 1U; }
    target_hash_table.assign(hash_capacity, std::numeric_limits<std::uint32_t>::max());
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
    guidance_outcomes.reserve(owners.size());
    baseline_recovery.reserve(
        owners.size() * (2U + owner.context_cache.max_long_anchors_per_continuation.value_or(0)));
    using PlanningContractAccess = qwen3_6::detail::RuntimeContractAccess<NINFER_QWEN36_VARIANT>;
    for (const Owner& pressure_owner : owners) {
        if (pressure_owner.shared) {
            const auto summary = owner.shared_prefix_summary(
                owner.shared_prefix_states[PlanningContractAccess::index(
                    *pressure_owner.shared_handle)]);
            baseline_recovery.push_back(PressureBaselineRecovery{
                .owner_ordinal = pressure_owner.ordinal,
                .checkpoint    = summary.checkpoint.ref,
                .recovery_ns   = owner.checkpoint_recovery_ns(*pressure_owner.shared_handle,
                                                              summary.checkpoint.ref, cost),
            });
            continue;
        }
        const auto summary =
            owner.continuation_summary(owner.continuation_states[PlanningContractAccess::index(
                *pressure_owner.private_handle)]);
        const auto append = [&](const qwen3_6::CheckpointSummary& checkpoint) {
            baseline_recovery.push_back(PressureBaselineRecovery{
                .owner_ordinal = pressure_owner.ordinal,
                .checkpoint    = checkpoint.ref,
                .recovery_ns   = owner.checkpoint_recovery_ns(*pressure_owner.private_handle,
                                                              checkpoint.ref, cost),
            });
        };
        if (summary.endpoint) { append(*summary.endpoint); }
        if (summary.rewrite) { append(*summary.rewrite); }
        for (const qwen3_6::CheckpointSummary& checkpoint : summary.long_anchors) {
            append(checkpoint);
        }
    }

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
        index_target(static_cast<std::uint32_t>(index));
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

inline const typename PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::TargetNode*
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::find_target(
    const TargetNode& target) const noexcept {
    if (target_hash_table.empty()) { return nullptr; }
    const std::size_t mask = target_hash_table.size() - 1U;
    std::size_t slot       = static_cast<std::size_t>(planning_detail::target_hash(target)) & mask;
    for (std::size_t probe = 0; probe < target_hash_table.size(); ++probe) {
        const std::uint32_t index = target_hash_table[slot];
        if (index == std::numeric_limits<std::uint32_t>::max()) { return nullptr; }
        if (index < targets.size() && planning_detail::same_target(targets[index], target)) {
            return &targets[index];
        }
        slot = (slot + 1U) & mask;
    }
    return nullptr;
}

inline typename PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::TargetNode*
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::find_target(TargetNode const& target) noexcept {
    return const_cast<TargetNode*>(std::as_const(*this).find_target(target));
}

inline void
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::index_target(std::uint32_t target_index) {
    if (target_index >= targets.size() || target_hash_table.empty()) {
        throw std::logic_error("pressure target hash index is invalid");
    }
    const std::size_t mask = target_hash_table.size() - 1U;
    std::size_t slot =
        static_cast<std::size_t>(planning_detail::target_hash(targets[target_index])) & mask;
    for (std::size_t probe = 0; probe < target_hash_table.size(); ++probe) {
        std::uint32_t& indexed = target_hash_table[slot];
        if (indexed == std::numeric_limits<std::uint32_t>::max()) {
            indexed = target_index;
            return;
        }
        if (indexed < targets.size() &&
            planning_detail::same_target(targets[indexed], targets[target_index])) {
            if (indexed != target_index) {
                throw std::logic_error("pressure target hash index is duplicated");
            }
            return;
        }
        slot = (slot + 1U) & mask;
    }
    throw std::length_error("pressure target hash table is full");
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
    TargetNode* existing       = find_target(maximal);
    std::uint32_t target_index = 0;
    if (existing != nullptr) {
        target_index           = static_cast<std::uint32_t>(existing - targets.data());
        existing->root_maximal = true;
    } else {
        maximal.stable_ordinal = static_cast<std::uint32_t>(targets.size());
        targets.push_back(std::move(maximal));
        target_index = static_cast<std::uint32_t>(targets.size() - 1U);
        index_target(target_index);
    }
    qwen3_6::PressureTargetHandle handle;
    handle.session_    = this;
    handle.generation_ = generation;
    handle.index_      = target_index;
    return handle;
}

inline std::optional<qwen3_6::PressureTargetHandle>
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::guided_closure_target(
    const AdmissionCandidate& admission, std::span<const std::uint32_t> preferred_owner_ordinals) {
    if (scratch_live) {
        throw std::logic_error("guided pressure closure conflicts with expansion scratch");
    }
    const std::uint32_t selected_candidate = candidate_index(admission);
    populate_options(selected_candidate);
    CandidateOptions& options = candidate_options[selected_candidate];
    const std::optional<typename Core::MaterializationSourceProtection> protection =
        program->materialization_source_protection(*admission.impl_);
    if (!protection) { return std::nullopt; }

    std::vector<std::size_t> owner_order;
    owner_order.reserve(owners.size());
    const auto append_owner = [&](std::size_t owner_index) {
        if (std::find(owner_order.begin(), owner_order.end(), owner_index) == owner_order.end()) {
            owner_order.push_back(owner_index);
        }
    };
    for (const std::uint32_t ordinal : preferred_owner_ordinals) {
        const auto found = std::find_if(owners.begin(), owners.end(), [&](const Owner& owner) {
            return owner.ordinal == ordinal;
        });
        if (found != owners.end()) {
            append_owner(static_cast<std::size_t>(found - owners.begin()));
        }
    }
    for (std::size_t index = 0; index < owners.size(); ++index) { append_owner(index); }

    const auto projected_residual = [&](const TargetNode& node,
                                        std::optional<std::size_t> override_owner,
                                        const PressureDecision* override_decision) {
        detail::PhysicalDelta pressure;
        for (std::size_t index = 0; index < owners.size(); ++index) {
            const PressureDecision* decision = nullptr;
            if (override_owner && *override_owner == index) {
                decision = override_decision;
            } else {
                const std::uint16_t choice = node.owner_choices[index];
                if (choice != 0) {
                    if (choice > options.owners[index].size()) {
                        throw std::logic_error("guided pressure choice is invalid");
                    }
                    decision = &options.owners[index][choice - 1U];
                }
            }
            if (decision == nullptr) { continue; }
            pressure.added = NINFER_QWEN36_RUNTIME_NS::planning_resource_sum(
                pressure.added, decision->effect.added);
            pressure.removed = NINFER_QWEN36_RUNTIME_NS::planning_resource_sum(
                pressure.removed, decision->effect.removed);
        }
        detail::PhysicalResources residual =
            program->guided_materialization_deficit(*admission.impl_, pressure);
        residual.host.kv_bytes =
            std::max(residual.host.kv_bytes, admission.impl_->blocked_host_allocation_bytes);
        return residual;
    };
    const detail::PhysicalResources capacity = program->admission_capacity();
    constexpr std::uint64_t kResidualOne     = 1ULL << 20U;
    const auto normalized                    = [](std::uint64_t value, std::uint64_t limit) {
        if (value == 0) { return std::uint64_t{0}; }
        if (limit == 0 || value >= limit) { return kResidualOne; }
        if (value > std::numeric_limits<std::uint64_t>::max() / kResidualOne) {
            return kResidualOne;
        }
        const std::uint64_t scaled = value * kResidualOne;
        return std::max<std::uint64_t>(1, scaled / limit + (scaled % limit != 0 ? 1U : 0U));
    };
    const auto residual_key = [&](const detail::PhysicalResources& residual) {
        std::uint32_t constraints = 0;
        std::uint64_t total       = 0;
        const auto append         = [&](std::uint64_t value, std::uint64_t limit) {
            if (value == 0) { return; }
            ++constraints;
            NINFER_QWEN36_RUNTIME_NS::planning_saturating_add(total, normalized(value, limit));
        };
        append(residual.device.active_lanes, capacity.device.active_lanes);
        append(residual.device.state_slots, capacity.device.state_slots);
        append(residual.device.main_kv_pages, capacity.device.main_kv_pages);
        append(residual.device.backend_kv_pages, capacity.device.backend_kv_pages);
        append(residual.host.state_slots, capacity.host.state_slots);
        append(residual.host.kv_bytes, capacity.host.kv_bytes);
        return std::tuple{constraints, total};
    };
    const auto transfer_bytes = [](const PressureDecision& decision) {
        std::uint64_t bytes = 0;
        for (const runtime::ContextTransferRequirement& requirement :
             decision.transfer_requirements) {
            NINFER_QWEN36_RUNTIME_NS::planning_saturating_add(bytes,
                                                              requirement.work.payload_bytes);
        }
        return bytes;
    };

    TargetNode target{
        .candidate_index = selected_candidate,
        .owner_choices   = std::vector<std::uint16_t>(owners.size(), 0),
    };
    using PlanningContractAccess = qwen3_6::detail::RuntimeContractAccess<NINFER_QWEN36_VARIANT>;
    const auto successors_for    = [&](std::size_t owner_index,
                                    const detail::PhysicalResources& residual,
                                    const PressureDecision* current) {
        std::vector<PressureDecision> successors;
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
            const PressureDecision& eviction = options.owners[owner_index][eviction_choice - 1U];
            if (std::find(successors.begin(), successors.end(), eviction) == successors.end()) {
                successors.push_back(eviction);
            }
        }
        return successors;
    };

    struct Selection {
        std::size_t owner_index = 0;
        PressureDecision decision;
        detail::PhysicalResources residual;
    };

    const std::size_t maximum_steps = 16U * std::max<std::size_t>(1, owners.size()) + 16U;
    for (std::size_t step = 0; step < maximum_steps; ++step) {
        const detail::PhysicalResources residual =
            projected_residual(target, std::nullopt, nullptr);
        if (residual == detail::PhysicalResources{}) {
            TargetNode* existing       = find_target(target);
            std::uint32_t target_index = 0;
            if (existing != nullptr) {
                target_index = static_cast<std::uint32_t>(existing - targets.data());
            } else {
                const std::size_t maximum =
                    candidates.size() + 1U + planning_detail::kOptionalTargetCapacity;
                if (targets.size() >= maximum) { return std::nullopt; }
                target.stable_ordinal = static_cast<std::uint32_t>(targets.size());
                targets.push_back(std::move(target));
                target_index = static_cast<std::uint32_t>(targets.size() - 1U);
                index_target(target_index);
            }
            qwen3_6::PressureTargetHandle handle;
            handle.session_    = this;
            handle.generation_ = generation;
            handle.index_      = target_index;
            return handle;
        }

        std::optional<Selection> selected;
        for (int destructive = 0; destructive < 2 && !selected; ++destructive) {
            for (const std::size_t owner_index : owner_order) {
                // populate_options gives the admission's source owner no options
                // (eviction_choices == 0), and every other owner walk skips it on that
                // basis. Skip it here as well: proposing pressure decisions for the
                // continuation the admission itself reuses is rejected by
                // compose_materialization through the engine-fatal "materialization
                // pressure owner is duplicated" contract check (#135).
                if (options.eviction_choices[owner_index] == 0) { continue; }
                const std::uint16_t current_choice = target.owner_choices[owner_index];
                const PressureDecision* current =
                    current_choice == 0 ? nullptr
                                        : &options.owners[owner_index][current_choice - 1U];
                if (current != nullptr && current->evicts_continuation) { continue; }
                std::vector<PressureDecision> successors =
                    successors_for(owner_index, residual, current);
                std::optional<Selection> owner_best;
                for (PressureDecision& successor : successors) {
                    const std::uint32_t prior_drops =
                        current == nullptr ? 0 : current->checkpoint_drops;
                    const bool adds_destruction =
                        successor.evicts_continuation || successor.checkpoint_drops > prior_drops;
                    if (adds_destruction != (destructive != 0)) { continue; }
                    const detail::PhysicalResources child_residual =
                        projected_residual(target, owner_index, &successor);
                    if (child_residual == residual) { continue; }
                    Selection candidate{
                        .owner_index = owner_index,
                        .decision    = std::move(successor),
                        .residual    = child_residual,
                    };
                    const auto key = [&](const Selection& value) {
                        return std::tuple{
                            residual_key(value.residual),
                            NINFER_QWEN36_RUNTIME_NS::degradation_units(value.decision),
                            transfer_bytes(value.decision),
                            value.decision.id,
                        };
                    };
                    if (!owner_best || key(candidate) < key(*owner_best)) {
                        owner_best = std::move(candidate);
                    }
                }
                if (owner_best) {
                    selected = std::move(owner_best);
                    break;
                }
            }
        }
        if (!selected) { return std::nullopt; }

        std::vector<PressureDecision>& decisions = options.owners[selected->owner_index];
        const auto existing  = std::find(decisions.begin(), decisions.end(), selected->decision);
        std::uint16_t choice = 0;
        if (existing != decisions.end()) {
            choice = static_cast<std::uint16_t>(1U + (existing - decisions.begin()));
        } else {
            if (decisions.size() >= std::numeric_limits<std::uint16_t>::max()) {
                return std::nullopt;
            }
            decisions.push_back(std::move(selected->decision));
            choice = static_cast<std::uint16_t>(decisions.size());
        }
        target.owner_choices[selected->owner_index] = choice;
    }
    return std::nullopt;
}

inline runtime::PressureTargetGuidance
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::guidance(qwen3_6::PressureTargetHandle target) {
    if (!valid(target) || scratch_live) {
        throw std::logic_error("pressure target guidance is stale or conflicts with expansion");
    }
    TargetNode& node = targets[target.index_];
    populate_options(node.candidate_index);
    const AdmissionCandidate& candidate = *candidates[node.candidate_index];
    const CandidateOptions& options     = candidate_options[node.candidate_index];

    guidance_outcomes.clear();
    NINFER_QWEN36_RUNTIME_NS::PlanningTransferAccumulator estimated_pressure;
    detail::PhysicalDelta approximate_pressure;
    std::uint32_t total_degradation = 0;
    std::uint32_t total_dropped     = 0;
    for (std::size_t index = 0; index < owners.size(); ++index) {
        const std::uint16_t choice = node.owner_choices[index];
        if (choice > options.owners[index].size()) {
            throw std::logic_error("pressure target guidance owner choice is invalid");
        }
        if (choice == 0) { continue; }
        const PressureDecision& decision = options.owners[index][choice - 1U];
        approximate_pressure.added       = NINFER_QWEN36_RUNTIME_NS::planning_resource_sum(
            approximate_pressure.added, decision.effect.added);
        approximate_pressure.removed = NINFER_QWEN36_RUNTIME_NS::planning_resource_sum(
            approximate_pressure.removed, decision.effect.removed);
        estimated_pressure.append(decision.transfer_requirements);
        const std::uint32_t units = NINFER_QWEN36_RUNTIME_NS::degradation_units(decision);
        total_degradation         = NINFER_QWEN36_RUNTIME_NS::planning_saturating_u32(
            static_cast<std::uint64_t>(total_degradation) + units);
        total_dropped = NINFER_QWEN36_RUNTIME_NS::planning_saturating_u32(
            static_cast<std::uint64_t>(total_dropped) + decision.checkpoint_drops);
        guidance_outcomes.push_back(runtime::PressureOwnerOutcome{
            .owner_ordinal       = owners[index].ordinal,
            .disposition         = decision.evicts_continuation ? runtime::ClaimDisposition::Evicted
                                                                : runtime::ClaimDisposition::Retained,
            .degradation_units   = units,
            .dropped_checkpoints = decision.checkpoint_drops,
            .shared              = owners[index].shared,
        });
    }
    detail::PhysicalResources residual =
        program->guided_materialization_deficit(*candidate.impl_, approximate_pressure);
    residual.host.kv_bytes =
        std::max(residual.host.kv_bytes, candidate.impl_->blocked_host_allocation_bytes);

    const detail::PhysicalResources capacity = program->admission_capacity();
    constexpr std::uint64_t kResidualOne     = 1ULL << 20U;
    const auto normalized                    = [](std::uint64_t value, std::uint64_t limit) {
        if (value == 0) { return std::uint64_t{0}; }
        if (limit == 0 || value >= limit) { return kResidualOne; }
        if (value > std::numeric_limits<std::uint64_t>::max() / kResidualOne) {
            return kResidualOne;
        }
        const std::uint64_t scaled = value * kResidualOne;
        return std::max<std::uint64_t>(1, scaled / limit + (scaled % limit != 0 ? 1U : 0U));
    };
    std::uint32_t constraints  = 0;
    std::uint64_t residual_q20 = 0;
    const auto append_residual = [&](std::uint64_t value, std::uint64_t limit) {
        if (value == 0) { return; }
        ++constraints;
        NINFER_QWEN36_RUNTIME_NS::planning_saturating_add(residual_q20, normalized(value, limit));
    };
    append_residual(residual.device.active_lanes, capacity.device.active_lanes);
    append_residual(residual.device.state_slots, capacity.device.state_slots);
    append_residual(residual.device.main_kv_pages, capacity.device.main_kv_pages);
    append_residual(residual.device.backend_kv_pages, capacity.device.backend_kv_pages);
    append_residual(residual.host.state_slots, capacity.host.state_slots);
    append_residual(residual.host.kv_bytes, capacity.host.kv_bytes);

    std::array<std::uint64_t, 6> maximum_additional_relief{};
    const auto update_relief = [&](std::size_t dimension, std::uint64_t eviction_removed,
                                   std::uint64_t eviction_added, std::uint64_t current_removed,
                                   std::uint64_t current_added) {
        const auto saturating_sum = [](std::uint64_t left, std::uint64_t right) {
            return right > std::numeric_limits<std::uint64_t>::max() - left
                       ? std::numeric_limits<std::uint64_t>::max()
                       : left + right;
        };
        const std::uint64_t released         = saturating_sum(eviction_removed, current_added);
        const std::uint64_t consumed         = saturating_sum(eviction_added, current_removed);
        maximum_additional_relief[dimension] = std::max(
            maximum_additional_relief[dimension], released > consumed ? released - consumed : 0U);
    };
    for (std::size_t index = 0; index < owners.size(); ++index) {
        const std::uint16_t eviction_choice = options.eviction_choices[index];
        if (eviction_choice == 0 || node.owner_choices[index] == eviction_choice) { continue; }
        const PressureDecision& eviction = options.owners[index][eviction_choice - 1U];
        const PressureDecision* current =
            node.owner_choices[index] == 0 ? nullptr
                                           : &options.owners[index][node.owner_choices[index] - 1U];
        const detail::PhysicalDelta empty{};
        const detail::PhysicalDelta& prior = current == nullptr ? empty : current->effect;
        update_relief(0, eviction.effect.removed.device.active_lanes,
                      eviction.effect.added.device.active_lanes, prior.removed.device.active_lanes,
                      prior.added.device.active_lanes);
        update_relief(1, eviction.effect.removed.device.state_slots,
                      eviction.effect.added.device.state_slots, prior.removed.device.state_slots,
                      prior.added.device.state_slots);
        update_relief(2, eviction.effect.removed.device.main_kv_pages,
                      eviction.effect.added.device.main_kv_pages,
                      prior.removed.device.main_kv_pages, prior.added.device.main_kv_pages);
        update_relief(3, eviction.effect.removed.device.backend_kv_pages,
                      eviction.effect.added.device.backend_kv_pages,
                      prior.removed.device.backend_kv_pages, prior.added.device.backend_kv_pages);
        update_relief(4, eviction.effect.removed.host.state_slots,
                      eviction.effect.added.host.state_slots, prior.removed.host.state_slots,
                      prior.added.host.state_slots);
        update_relief(5, eviction.effect.removed.host.kv_bytes, eviction.effect.added.host.kv_bytes,
                      prior.removed.host.kv_bytes, prior.added.host.kv_bytes);
    }
    const std::array<std::uint64_t, 6> residual_values{
        residual.device.active_lanes,  residual.device.state_slots,
        residual.device.main_kv_pages, residual.device.backend_kv_pages,
        residual.host.state_slots,     residual.host.kv_bytes,
    };
    std::uint32_t remaining_steps = 0;
    for (std::size_t index = 0; index < residual_values.size(); ++index) {
        if (residual_values[index] == 0) { continue; }
        if (maximum_additional_relief[index] == 0) {
            remaining_steps = std::numeric_limits<std::uint32_t>::max();
            break;
        }
        const std::uint64_t steps =
            1U + (residual_values[index] - 1U) / maximum_additional_relief[index];
        remaining_steps =
            std::max(remaining_steps, NINFER_QWEN36_RUNTIME_NS::planning_saturating_u32(steps));
    }
    return runtime::PressureTargetGuidance{
        .physical =
            {
                .unsatisfied_constraints   = constraints,
                .estimated_remaining_steps = remaining_steps,
                .normalized_residual_q20   = residual_q20,
            },
        .estimated_machine = NINFER_QWEN36_RUNTIME_NS::materialization_machine_summary(
            *candidate.impl_, estimated_pressure, *machine_cost),
        .owner_outcomes        = guidance_outcomes,
        .candidate_ordinal     = node.candidate_index,
        .stable_target_ordinal = node.stable_ordinal,
        .degradation_units     = total_degradation,
        .dropped_checkpoints   = total_dropped,
    };
}

inline runtime::PressureTargetAssessment
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::assess(qwen3_6::PressureTargetHandle target) {
    if (!valid(target) || scratch_live) {
        throw std::logic_error("pressure target assessment is stale or conflicts with expansion");
    }
    TargetNode& node = targets[target.index_];
    latest_projection.reset();
    latest_projection_target = std::numeric_limits<std::uint32_t>::max();
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
            selected_shared_decisions.push_back(decision);
        } else {
            selected_private_owners.push_back(owner.private_handle);
            selected_private_decisions.push_back(decision);
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
            recovery_shared_ordinals, baseline_recovery, *machine_cost, assessment_impacts,
            projection_work);
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
    if (identity_target || composed) {
        node.assessed_residual                = program->materialization_deficit(*projected);
        node.assessed_residual->host.kv_bytes = std::max(node.assessed_residual->host.kv_bytes,
                                                         projected->blocked_host_allocation_bytes);
    } else {
        node.assessed_residual.reset();
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

    runtime::PressureTargetAssessment result{
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
    if (composed) {
        latest_projection        = std::move(*composed);
        latest_projection_target = target.index_;
    }
    return result;
}

inline void PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::retain_assessment(
    qwen3_6::PressureTargetHandle target) {
    if (!valid(target) || scratch_live) {
        throw std::logic_error("retained pressure assessment is stale or conflicts with expansion");
    }
    const TargetNode& node = targets[target.index_];
    const bool identity    = std::all_of(node.owner_choices.begin(), node.owner_choices.end(),
                                         [](std::uint16_t choice) { return choice == 0; });
    if (identity) {
        retained_projection.reset();
        retained_projection_target = target.index_;
        return;
    }
    if (retained_projection_target == target.index_ && retained_projection) { return; }
    if (latest_projection_target != target.index_ || !latest_projection) {
        throw std::logic_error("pressure target has no retainable exact projection");
    }
    retained_projection        = std::move(latest_projection);
    retained_projection_target = target.index_;
    latest_projection_target   = std::numeric_limits<std::uint32_t>::max();
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
    const bool identity = std::all_of(node.owner_choices.begin(), node.owner_choices.end(),
                                      [](std::uint16_t choice) { return choice == 0; });
    detail::PhysicalResources residual;
    if (identity) {
        residual = candidate.impl_->identity_pressure_deficit;
        residual.host.kv_bytes =
            std::max(residual.host.kv_bytes, candidate.impl_->blocked_host_allocation_bytes);
    } else {
        if (!node.assessed_residual) {
            throw std::logic_error("pressure target must be assessed before expansion");
        }
        residual = *node.assessed_residual;
    }

    const auto append = [&](TargetNode child) {
        const bool duplicate_scratch = std::any_of(
            expansion_scratch.begin(), expansion_scratch.end(), [&](const TargetNode& existing) {
                return planning_detail::same_target(existing, child);
            });
        if (duplicate_scratch) { return; }
        const bool existing = find_target(child) != nullptr;
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
            TargetNode child{
                .candidate_index = node.candidate_index,
                .owner_choices   = node.owner_choices,
            };
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
        TargetNode* existing = find_target(child);
        std::uint32_t index  = 0;
        if (existing == nullptr) {
            child.stable_ordinal = static_cast<std::uint32_t>(targets.size());
            targets.push_back(std::move(child));
            index = static_cast<std::uint32_t>(targets.size() - 1U);
            index_target(index);
        } else {
            index = static_cast<std::uint32_t>(existing - targets.data());
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
    if (retained_projection_target == target.index_ && retained_projection) {
        std::optional<AdmissionCandidate> sealed = std::move(retained_projection);
        retained_projection_target               = std::numeric_limits<std::uint32_t>::max();
        if (sealed->impl_->blocked_host_allocation_bytes != 0 ||
            program->revalidate_materialization(*sealed, prompt) !=
                runtime::PreflightStatus::Ready) {
            return std::nullopt;
        }
        return sealed;
    }
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
            selected_shared_decisions.push_back(&decision);
        } else {
            selected_private_owners.push_back(owners[index].private_handle);
            selected_private_decisions.push_back(&decision);
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
    if (retained_projection_target == target.index_ && retained_projection) {
        std::optional<AdmissionCandidate> sealed = std::move(retained_projection);
        retained_projection_target               = std::numeric_limits<std::uint32_t>::max();
        if (!sealed->impl_->capture_pressure || sealed->impl_->blocked_host_allocation_bytes != 0 ||
            !program->physical_peak_fits(sealed->impl_->demand.physical_peak_additional)) {
            return std::nullopt;
        }
        return sealed;
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
            selected_shared_decisions.push_back(&decision);
        } else {
            selected_private_owners.push_back(owners[index].private_handle);
            selected_private_decisions.push_back(&decision);
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
