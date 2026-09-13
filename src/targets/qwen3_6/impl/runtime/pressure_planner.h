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
    runtime::CoalescedTransferWork work{};

    void append(std::span<const runtime::ContextTransferRequirement> requirements) noexcept {
        for (const runtime::ContextTransferRequirement& requirement : requirements) {
            const std::size_t index = planning_direction_index(requirement.direction);
            planning_saturating_add(work[index].payload_bytes, requirement.work.payload_bytes);
            work[index].copy_operations =
                planning_saturating_u32(static_cast<std::uint64_t>(work[index].copy_operations) +
                                        requirement.work.copy_operations);
        }
    }
};

inline runtime::MaterializationMachineWork
materialization_machine_work(const ResourceCandidateState& candidate,
                             const PlanningTransferAccumulator& pressure) noexcept {
    PlanningTransferAccumulator request;
    request.append(candidate.transfer_requirements);

    PlanningTransferAccumulator optimistic_request;
    for (const runtime::ContextTransferRequirement& requirement : candidate.transfer_requirements) {
        const bool pressure_may_eliminate =
            candidate.has_source &&
            candidate.source_mode == runtime::PrivateSourceMode::ConsumeToActive &&
            requirement.direction == runtime::ContextTransferDirection::DeviceToDevice;
        if (!pressure_may_eliminate) {
            optimistic_request.append(
                std::span<const runtime::ContextTransferRequirement>(&requirement, 1));
        }
    }

    return runtime::MaterializationMachineWork{
        .pressure_transfers             = pressure.work,
        .candidate_transfers            = request.work,
        .optimistic_candidate_transfers = optimistic_request.work,
        .remaining_prefill_work         = candidate.remaining_prefill_work,
        .reused_prompt_tokens           = candidate.summary.reusable_prompt_tokens,
    };
}

inline runtime::MaterializationMachineWork materialization_machine_work(
    const ResourceCandidateState& candidate,
    std::span<const qwen3_6::detail::PressureDecision* const> private_decisions,
    std::span<const qwen3_6::detail::PressureDecision* const> shared_decisions) noexcept {
    PlanningTransferAccumulator pressure;
    for (const qwen3_6::detail::PressureDecision* decision : private_decisions) {
        if (decision != nullptr) { pressure.append(decision->transfer_requirements); }
    }
    for (const qwen3_6::detail::PressureDecision* decision : shared_decisions) {
        if (decision != nullptr) { pressure.append(decision->transfer_requirements); }
    }
    return materialization_machine_work(candidate, pressure);
}

inline runtime::CheckpointRecoveryAlternativeWork
recovery_alternative_work(std::span<const runtime::ContextTransferRequirement> requirements,
                          runtime::PrefillWork prefill_work = {}) noexcept {
    PlanningTransferAccumulator transfer;
    transfer.append(requirements);
    return runtime::CheckpointRecoveryAlternativeWork{
        .transfers = transfer.work,
        .prefill   = prefill_work,
    };
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

inline void hash_mix(std::uint64_t& hash, std::uint64_t value) noexcept {
    hash ^= value;
    hash *= 1099511628211ULL;
}

[[nodiscard]] inline std::uint64_t target_hash(std::uint32_t candidate_index,
                                               std::span<const std::uint16_t> choices) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    hash_mix(hash, candidate_index);
    for (const std::uint16_t choice : choices) { hash_mix(hash, choice); }
    return hash;
}

[[nodiscard]] inline bool same_choices(std::span<const std::uint16_t> left,
                                       std::span<const std::uint16_t> right) noexcept {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin());
}

} // namespace planning_detail

inline PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::PressurePlanningSessionImpl(
    Core& owner, std::span<const PhysicalCandidateBinding> physical_candidates,
    std::span<const runtime::PlanningCandidateId> admission_candidate_ids,
    std::span<const ContinuationHandle* const> private_owners,
    std::span<const runtime::PlanningOwnerId> private_owner_ids,
    std::span<const SharedPrefixHandle* const> shared_owners,
    std::span<const runtime::PlanningOwnerId> shared_owner_ids)
    : program(&owner), resource_revision(owner.resource_revision()) {
    if (physical_candidates.empty() ||
        physical_candidates.size() != admission_candidate_ids.size() ||
        private_owners.size() != private_owner_ids.size() ||
        shared_owners.size() != shared_owner_ids.size() || owner.has_context_transaction() ||
        owner.pending_transaction_ || owner.pressure_planning_active_) {
        throw std::logic_error("pressure planning session cannot start in the current state");
    }

    candidates.assign(physical_candidates.begin(), physical_candidates.end());
    candidate_ids.assign(admission_candidate_ids.begin(), admission_candidate_ids.end());
    for (std::size_t index = 0; index < candidate_ids.size(); ++index) {
        if (std::find(candidate_ids.begin(), candidate_ids.begin() + index, candidate_ids[index]) !=
            candidate_ids.begin() + index) {
            throw std::logic_error("pressure planning candidate ID is duplicated");
        }
    }
    owners.reserve(private_owners.size() + shared_owners.size());
    for (std::size_t index = 0; index < private_owners.size(); ++index) {
        const ContinuationHandle* handle = private_owners[index];
        if (handle == nullptr || !owner.valid_continuation(*handle)) {
            throw std::logic_error("pressure planning private owner is stale");
        }
        owners.push_back(
            Owner{.private_handle = handle, .id = private_owner_ids[index], .shared = false});
    }
    for (std::size_t index = 0; index < shared_owners.size(); ++index) {
        const SharedPrefixHandle* handle = shared_owners[index];
        if (handle == nullptr || !owner.valid_shared_prefix(*handle)) {
            throw std::logic_error("pressure planning shared owner is stale");
        }
        owners.push_back(
            Owner{.shared_handle = handle, .id = shared_owner_ids[index], .shared = true});
    }
    std::sort(owners.begin(), owners.end(), [](const Owner& left, const Owner& right) {
        return std::tuple{left.id.value, left.shared} < std::tuple{right.id.value, right.shared};
    });
    for (std::size_t index = 1; index < owners.size(); ++index) {
        if (owners[index - 1].id == owners[index].id) {
            throw std::logic_error("pressure planning owner ID is duplicated");
        }
    }

    candidate_options.resize(candidates.size());
    const std::size_t maximum_targets =
        candidates.size() + 1U + planning_detail::kOptionalTargetCapacity;
    const std::size_t maximum_successors_per_owner =
        11U + owner.context_cache.max_long_anchors_per_continuation.value_or(0);
    if (!owners.empty() &&
        maximum_successors_per_owner > std::numeric_limits<std::size_t>::max() / owners.size()) {
        throw std::overflow_error("pressure expansion arena size overflow");
    }
    const std::size_t maximum_scratch_targets = owners.size() * maximum_successors_per_owner;
    targets.reserve(maximum_targets);
    if (maximum_scratch_targets > std::numeric_limits<std::size_t>::max() - maximum_targets ||
        (!owners.empty() && maximum_targets + maximum_scratch_targets >
                                std::numeric_limits<std::size_t>::max() / owners.size())) {
        throw std::overflow_error("pressure target choice arena size overflow");
    }
    target_choice_arena.reserve((maximum_targets + maximum_scratch_targets) * owners.size());
    choice_scratch.reserve(owners.size());
    std::size_t hash_capacity = 1;
    while (hash_capacity < 2U * maximum_targets) { hash_capacity <<= 1U; }
    target_hash_table.assign(hash_capacity, std::numeric_limits<std::uint32_t>::max());
    expansion_scratch.reserve(maximum_scratch_targets);
    committed_children.reserve(maximum_scratch_targets);
    for (auto& cursor : construction_slots) {
        cursor.choices.reserve(owners.size());
        cursor.options.reserve(maximum_scratch_targets + owners.size());
    }
    guidance_recovery.reserve(owners.size());
    selected_private_owners.reserve(private_owners.size());
    selected_private_owner_ids.reserve(private_owners.size());
    selected_private_decisions.reserve(private_owners.size());
    selected_shared_owners.reserve(shared_owners.size());
    selected_shared_owner_ids.reserve(shared_owners.size());
    selected_shared_decisions.reserve(shared_owners.size());
    recovery_private_owners.reserve(private_owners.size());
    recovery_private_decisions.reserve(private_owners.size());
    recovery_private_owner_ids.reserve(private_owners.size());
    recovery_shared_owners.reserve(shared_owners.size());
    recovery_shared_decisions.reserve(shared_owners.size());
    recovery_shared_owner_ids.reserve(shared_owners.size());
    projected_owner_decisions.assign(owners.size(), nullptr);
    assessment_outcomes.reserve(owners.size());
    guidance_outcomes.reserve(owners.size());
    guidance_checkpoint_changes.reserve(
        owners.size() * (2U + owner.context_cache.max_long_anchors_per_continuation.value_or(0)));
    const std::size_t private_checkpoint_capacity =
        2U + owner.context_cache.max_long_anchors_per_continuation.value_or(0);
    if (private_checkpoint_capacity > std::numeric_limits<std::size_t>::max() - 3U ||
        private_checkpoint_capacity != 0 &&
            private_checkpoint_capacity >
                std::numeric_limits<std::size_t>::max() / (private_checkpoint_capacity + 3U)) {
        throw std::overflow_error("pressure recovery alternative capacity overflow");
    }
    const std::size_t alternatives_per_private =
        private_checkpoint_capacity * (private_checkpoint_capacity + 3U) / 2U;
    if ((!private_owners.empty() &&
         alternatives_per_private >
             std::numeric_limits<std::size_t>::max() / private_owners.size()) ||
        shared_owners.size() > (std::numeric_limits<std::size_t>::max() -
                                private_owners.size() * alternatives_per_private) /
                                   2U) {
        throw std::overflow_error("pressure recovery alternative arena overflow");
    }
    const std::size_t recovery_alternative_capacity =
        private_owners.size() * alternatives_per_private + shared_owners.size() * 2U;
    if ((!private_owners.empty() &&
         private_checkpoint_capacity >
             (std::numeric_limits<std::size_t>::max() - shared_owners.size()) /
                 private_owners.size())) {
        throw std::overflow_error("pressure recovery impact arena overflow");
    }
    const std::size_t recovery_impact_capacity =
        private_owners.size() * private_checkpoint_capacity + shared_owners.size();
    if (owners.size() > std::numeric_limits<std::size_t>::max() / 3U) {
        throw std::overflow_error("pressure recovery State scratch capacity overflow");
    }
    assessment_impact_projections.reserve(recovery_impact_capacity);
    assessment_recovery_alternatives.reserve(recovery_alternative_capacity);
    recovery_scratch.state_placements.reserve(3U * owners.size());
    recovery_scratch.owners.reserve(owners.size());
    recovery_scratch.checkpoints.reserve(private_checkpoint_capacity);
    recovery_scratch.direct_work.reserve(private_checkpoint_capacity);
    recovery_scratch.continuation_summary.long_anchors.reserve(
        owner.context_cache.max_long_anchors_per_continuation.value_or(0));
    for (AssessmentSlot& slot : assessment_slots) {
        slot.owner_outcomes.reserve(owners.size());
        slot.checkpoint_impacts.reserve(recovery_impact_capacity);
        slot.recovery_alternatives.reserve(recovery_alternative_capacity);
    }
    using PlanningContractAccess = qwen3_6::detail::RuntimeContractAccess<NINFER_QWEN36_VARIANT>;

    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const PhysicalCandidateBinding& binding = candidates[index];
        const bool admission                    = binding.admission != nullptr;
        const bool capture                      = binding.capture != nullptr;
        if (binding.state == nullptr || admission == capture ||
            (admission && binding.state != static_cast<const CandidateState*>(binding.admission)) ||
            (capture && binding.state != static_cast<const CandidateState*>(binding.capture)) ||
            binding.state->planning_revision != resource_revision) {
            throw std::logic_error("pressure planning candidate is stale");
        }
        CandidateOptions& options = candidate_options[index];
        options.victims.reserve(owners.size());
        for (std::size_t owner_index = 0; owner_index < owners.size(); ++owner_index) {
            const Owner& pressure_owner = owners[owner_index];
            const bool selected_private_source =
                !pressure_owner.shared && binding.state->has_source &&
                PlanningContractAccess::index(*pressure_owner.private_handle) ==
                    binding.state->source_index &&
                PlanningContractAccess::epoch(*pressure_owner.private_handle) ==
                    binding.state->source_generation;
            const bool selected_shared_source =
                pressure_owner.shared && binding.state->has_shared_source &&
                PlanningContractAccess::index(*pressure_owner.shared_handle) ==
                    binding.state->shared_source_index &&
                PlanningContractAccess::epoch(*pressure_owner.shared_handle) ==
                    binding.state->shared_source_generation;
            if (!selected_private_source && !selected_shared_source) {
                options.victims.push_back(CandidateVictimOptions{
                    .owner_index = static_cast<std::uint32_t>(owner_index),
                });
            }
        }
        choice_scratch.assign(options.victims.size(), 0);
        if (intern_target(static_cast<std::uint32_t>(index), choice_scratch) != index) {
            throw std::logic_error("pressure identity target ordinal changed");
        }
    }

    if (++owner.pressure_planning_generation_ == 0) { ++owner.pressure_planning_generation_; }
    generation                      = owner.pressure_planning_generation_;
    owner.pressure_planning_active_ = true;
}

inline PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::~PressurePlanningSessionImpl() noexcept {
    if (std::any_of(assessment_slots.begin(), assessment_slots.end(),
                    [](const AssessmentSlot& slot) { return slot.leased; })) {
        std::terminate();
    }
    if (program != nullptr) { program->pressure_planning_active_ = false; }
}

inline std::uint32_t PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::acquire_assessment_slot() {
    for (std::uint32_t index = 0; index < assessment_slots.size(); ++index) {
        AssessmentSlot& slot = assessment_slots[index];
        if (slot.leased) { continue; }
        slot.leased = true;
        slot.owner_outcomes.clear();
        slot.checkpoint_impacts.clear();
        slot.recovery_alternatives.clear();
        return index;
    }
    throw std::logic_error("pressure runner retained too many simultaneous assessments");
}

inline void PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::release_assessment_slot(
    const void* owner, std::uint32_t slot_index, std::uint32_t slot_generation) noexcept {
    auto* session = const_cast<PressurePlanningSessionImpl*>(
        static_cast<const PressurePlanningSessionImpl*>(owner));
    if (session == nullptr || slot_index >= session->assessment_slots.size()) { std::terminate(); }
    AssessmentSlot& slot = session->assessment_slots[slot_index];
    if (!slot.leased || slot.generation != slot_generation) { std::terminate(); }
    slot.owner_outcomes.clear();
    slot.checkpoint_impacts.clear();
    slot.recovery_alternatives.clear();
    slot.leased = false;
    if (++slot.generation == 0) { ++slot.generation; }
}

inline bool PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::valid(
    qwen3_6::PressureTargetHandle target) const noexcept {
    return target.session_ == this && target.generation_ == generation &&
           target.index_ < targets.size() && program != nullptr &&
           program->resource_revision() == resource_revision;
}

inline std::uint32_t PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::candidate_index(
    runtime::PlanningCandidateId candidate) const {
    const auto found = std::find(candidate_ids.begin(), candidate_ids.end(), candidate);
    if (found == candidate_ids.end()) {
        throw std::invalid_argument("pressure target candidate does not belong to this session");
    }
    return static_cast<std::uint32_t>(found - candidate_ids.begin());
}

inline const typename PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::TargetNode*
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::find_target(
    std::uint32_t selected_candidate, std::span<const std::uint16_t> choices) const noexcept {
    if (target_hash_table.empty()) { return nullptr; }
    const std::size_t mask = target_hash_table.size() - 1U;
    std::size_t slot =
        static_cast<std::size_t>(planning_detail::target_hash(selected_candidate, choices)) & mask;
    for (std::size_t probe = 0; probe < target_hash_table.size(); ++probe) {
        const std::uint32_t index = target_hash_table[slot];
        if (index == std::numeric_limits<std::uint32_t>::max()) { return nullptr; }
        if (index < targets.size()) {
            const TargetNode& target = targets[index];
            if (target.candidate_index == selected_candidate &&
                target.victim_choice_offset <= target_choice_arena.size() &&
                target.victim_choice_count <=
                    target_choice_arena.size() - target.victim_choice_offset &&
                std::equal(choices.begin(), choices.end(),
                           target_choice_arena.begin() + target.victim_choice_offset,
                           target_choice_arena.begin() + target.victim_choice_offset +
                               target.victim_choice_count)) {
                return &target;
            }
        }
        slot = (slot + 1U) & mask;
    }
    return nullptr;
}

inline typename PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::TargetNode*
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::find_target(
    std::uint32_t selected_candidate, std::span<const std::uint16_t> choices) noexcept {
    return const_cast<TargetNode*>(std::as_const(*this).find_target(selected_candidate, choices));
}

inline std::span<const std::uint16_t>
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::victim_choices(const TargetNode& target) const {
    if (target.victim_choice_offset > target_choice_arena.size() ||
        target.victim_choice_count > target_choice_arena.size() - target.victim_choice_offset) {
        throw std::logic_error("pressure target choice span is invalid");
    }
    return std::span<const std::uint16_t>(target_choice_arena)
        .subspan(target.victim_choice_offset, target.victim_choice_count);
}

inline std::uint32_t PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::intern_target(
    std::uint32_t selected_candidate, std::span<const std::uint16_t> choices, bool root_maximal) {
    if (selected_candidate >= candidate_options.size() ||
        choices.size() != candidate_options[selected_candidate].victims.size()) {
        throw std::logic_error("pressure target does not match its candidate victim domain");
    }
    if (TargetNode* existing = find_target(selected_candidate, choices)) {
        existing->root_maximal = existing->root_maximal || root_maximal;
        return static_cast<std::uint32_t>(existing - targets.data());
    }
    const std::size_t maximum = candidates.size() + 1U + planning_detail::kOptionalTargetCapacity;
    if (targets.size() >= maximum || targets.size() == targets.capacity() ||
        choices.size() > target_choice_arena.capacity() - target_choice_arena.size() ||
        target_choice_arena.size() > std::numeric_limits<std::uint32_t>::max() ||
        choices.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("pressure target arena is full");
    }
    const std::uint32_t offset = static_cast<std::uint32_t>(target_choice_arena.size());
    target_choice_arena.insert(target_choice_arena.end(), choices.begin(), choices.end());
    const std::uint32_t index = static_cast<std::uint32_t>(targets.size());
    targets.push_back(TargetNode{
        .candidate_index      = selected_candidate,
        .victim_choice_offset = offset,
        .victim_choice_count  = static_cast<std::uint32_t>(choices.size()),
        .stable_ordinal       = index,
        .root_maximal         = root_maximal,
    });
    index_target(index);
    return index;
}

inline void
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::index_target(std::uint32_t target_index) {
    if (target_index >= targets.size() || target_hash_table.empty()) {
        throw std::logic_error("pressure target hash index is invalid");
    }
    const std::span<const std::uint16_t> choices = victim_choices(targets[target_index]);
    const std::size_t mask                       = target_hash_table.size() - 1U;
    std::size_t slot = static_cast<std::size_t>(planning_detail::target_hash(
                           targets[target_index].candidate_index, choices)) &
                       mask;
    for (std::size_t probe = 0; probe < target_hash_table.size(); ++probe) {
        std::uint32_t& indexed = target_hash_table[slot];
        if (indexed == std::numeric_limits<std::uint32_t>::max()) {
            indexed = target_index;
            return;
        }
        if (indexed < targets.size() &&
            targets[indexed].candidate_index == targets[target_index].candidate_index &&
            planning_detail::same_choices(victim_choices(targets[indexed]), choices)) {
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
    runtime::PlanningCandidateId candidate) const {
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
    const CandidateState& candidate = *candidates[selected_candidate].state;
    const std::optional<typename Core::MaterializationSourceProtection> protection =
        program->materialization_source_protection(candidate);
    if (!protection) {
        throw std::logic_error("pressure planning candidate source protection is stale");
    }
    for (CandidateVictimOptions& victim : options.victims) {
        if (victim.owner_index >= owners.size()) {
            throw std::logic_error("pressure planning victim owner is invalid");
        }
        const Owner& owner                       = owners[victim.owner_index];
        std::vector<PressureDecision>& decisions = victim.decisions;
        using PlanningContractAccess =
            qwen3_6::detail::RuntimeContractAccess<NINFER_QWEN36_VARIANT>;
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
        victim.eviction_choice = static_cast<std::uint16_t>(decisions.size());
    }
    options.populated = true;
}

inline std::vector<PressureDecision>
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::pressure_successors(
    const CandidateVictimOptions& victim_options, const detail::PhysicalResources& residual,
    const typename Core::MaterializationSourceProtection& protection,
    const PressureDecision* current) const {
    if (victim_options.owner_index >= owners.size()) {
        throw std::out_of_range("pressure successor owner index is invalid");
    }

    using PlanningContractAccess = qwen3_6::detail::RuntimeContractAccess<NINFER_QWEN36_VARIANT>;
    std::vector<PressureDecision> successors;
    const Owner& victim_owner = owners[victim_options.owner_index];
    if (victim_owner.shared) {
        successors = program->inspect_shared_pressure_successors(
            program
                ->shared_prefix_states[PlanningContractAccess::index(*victim_owner.shared_handle)],
            residual, &protection, current);
    } else {
        successors = program->inspect_pressure_successors(
            program
                ->continuation_states[PlanningContractAccess::index(*victim_owner.private_handle)],
            residual, &protection, current);
    }
    if (victim_options.eviction_choice == 0 ||
        victim_options.eviction_choice > victim_options.decisions.size()) {
        throw std::logic_error("eligible pressure owner has no maximal outcome");
    }
    const PressureDecision& eviction =
        victim_options.decisions[victim_options.eviction_choice - 1U];
    if (std::find(successors.begin(), successors.end(), eviction) == successors.end()) {
        successors.push_back(eviction);
    }
    return successors;
}

inline qwen3_6::PressureTargetHandle
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::root_maximal_target(
    runtime::PlanningCandidateId root_candidate) {
    if (scratch_live) { throw std::logic_error("pressure expansion scratch is still live"); }
    const std::uint32_t selected_candidate = candidate_index(root_candidate);
    populate_options(selected_candidate);
    choice_scratch.assign(candidate_options[selected_candidate].victims.size(), 0);
    for (std::size_t index = 0; index < candidate_options[selected_candidate].victims.size();
         ++index) {
        choice_scratch[index] =
            candidate_options[selected_candidate].victims[index].eviction_choice;
    }
    const std::uint32_t target_index = intern_target(selected_candidate, choice_scratch, true);
    qwen3_6::PressureTargetHandle handle;
    handle.session_    = this;
    handle.generation_ = generation;
    handle.index_      = target_index;
    return handle;
}

inline qwen3_6::PressureTargetHandle
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::maximal_target(
    runtime::PlanningCandidateId id) {
    if (scratch_live) { throw std::logic_error("pressure expansion scratch is live"); }
    const auto selected = candidate_index(id);
    populate_options(selected);
    choice_scratch.clear();
    for (const auto& victim : candidate_options[selected].victims) {
        choice_scratch.push_back(victim.eviction_choice);
    }
    const auto index = intern_target(selected, choice_scratch);
    qwen3_6::PressureTargetHandle result;
    result.session_    = this;
    result.generation_ = generation;
    result.index_      = index;
    return result;
}

inline auto PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::construction_slot(
    const qwen3_6::PressureConstructionCursor& cursor) -> ConstructionSlot& {
    if (cursor.session_ != this || cursor.slot_ >= construction_slots.size() || scratch_live ||
        resource_revision != program->resource_revision()) {
        throw std::logic_error("pressure construction cursor is stale");
    }
    auto& slot = construction_slots[cursor.slot_];
    if (!slot.leased || slot.generation != cursor.generation_) {
        throw std::logic_error("pressure construction lease is stale");
    }
    return slot;
}

inline void PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::release_construction(
    const void* owner, std::uint32_t index, std::uint32_t lease) noexcept {
    auto& session = *const_cast<PressurePlanningSessionImpl*>(
        static_cast<const PressurePlanningSessionImpl*>(owner));
    if (index < session.construction_slots.size()) {
        auto& slot = session.construction_slots[index];
        if (slot.generation == lease) {
            slot.leased = false;
            slot.options.clear();
        }
    }
}

inline detail::PhysicalResources
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::construction_residual(
    std::uint32_t index, std::span<const std::uint16_t> choices) const {
    detail::PhysicalDelta pressure;
    const auto& options = candidate_options[index];
    for (std::size_t owner = 0; owner < choices.size(); ++owner) {
        if (choices[owner] == 0) { continue; }
        const auto& decision = options.victims[owner].decisions[choices[owner] - 1];
        pressure.added =
            NINFER_QWEN36_RUNTIME_NS::planning_resource_sum(pressure.added, decision.effect.added);
        pressure.removed = NINFER_QWEN36_RUNTIME_NS::planning_resource_sum(pressure.removed,
                                                                           decision.effect.removed);
    }
    auto result = program->guided_materialization_deficit(*candidates[index].state, pressure);
    result.host.kv_bytes =
        std::max(result.host.kv_bytes, candidates[index].state->blocked_host_allocation_bytes);
    return result;
}

inline qwen3_6::PressureConstructionCursor
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::begin_construction(
    qwen3_6::PressureTargetHandle target, bool restore) {
    if (!valid(target) || scratch_live) { throw std::logic_error("invalid construction parent"); }
    const auto& node = targets[target.index_];
    populate_options(node.candidate_index);
    for (std::uint32_t index = 0; index < construction_slots.size(); ++index) {
        auto& slot = construction_slots[index];
        if (slot.leased) { continue; }
        const auto choices = victim_choices(node);
        slot.choices.assign(choices.begin(), choices.end());
        slot.options.clear();
        slot.next_owner = slot.next_option = 0;
        slot.candidate_index               = node.candidate_index;
        slot.residual =
            node.assessed_residual.value_or(construction_residual(node.candidate_index, choices));
        slot.restore = restore;
        slot.leased  = true;
        if (++construction_generation == 0) { ++construction_generation; }
        slot.generation      = construction_generation;
        slot.scan_generation = 1;
        return qwen3_6::PressureConstructionCursor(this, index, slot.generation,
                                                   &release_construction);
    }
    throw std::length_error("all pressure construction cursors are leased");
}

inline runtime::PressureConstructionStep
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::next_construction_option(
    qwen3_6::PressureConstructionCursor& cursor) {
    auto& slot    = construction_slot(cursor);
    auto& options = candidate_options[slot.candidate_index];
    if (slot.next_option < slot.options.size()) {
        const auto index   = static_cast<std::uint32_t>(slot.next_option++);
        const auto& option = slot.options[index];
        return {.guidance = guidance_choices(slot.candidate_index, slot.choices, 0, option.victim,
                                             option.identity ? nullptr : &option.decision),
                .option   = {.cursor_generation = slot.generation,
                             .scan_generation   = slot.scan_generation,
                             .index             = index}};
    }
    if (slot.next_owner == options.victims.size()) { return {.exhausted = true}; }
    const auto owner_index          = slot.next_owner++;
    const auto& victim              = options.victims[owner_index];
    const auto choice               = slot.choices[owner_index];
    const PressureDecision* current = choice == 0 ? nullptr : &victim.decisions[choice - 1];
    const auto protection =
        program->materialization_source_protection(*candidates[slot.candidate_index].state);
    if (!protection) { throw std::logic_error("construction source protection is stale"); }
    const auto append = [&](PressureDecision decision, bool identity = false) {
        if (slot.options.size() == slot.options.capacity()) {
            throw std::length_error("construction option capacity exceeded");
        }
        slot.options.push_back(
            {.victim = owner_index, .decision = std::move(decision), .identity = identity});
    };
    if (slot.restore) {
        if (current != nullptr) {
            append({}, true);
            // Rebuild ordinary alternatives from the immutable candidate requirement, not from a
            // rewritten post-state. Complete assessment validates every less-destructive neighbor.
            for (auto& decision : pressure_successors(
                     victim, candidates[slot.candidate_index].state->identity_pressure_deficit,
                     *protection, nullptr)) {
                if (decision != *current && !decision.evicts_continuation) {
                    append(std::move(decision));
                }
            }
        }
    } else if (current == nullptr || !current->evicts_continuation) {
        for (auto& decision : pressure_successors(victim, slot.residual, *protection, current)) {
            if (current == nullptr || decision != *current) { append(std::move(decision)); }
        }
    }
    return {}; // one owner's successor generation is a separately metered operation
}

inline void PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::choose_construction(
    qwen3_6::PressureConstructionCursor& cursor, runtime::PressureConstructionOptionId id) {
    auto& slot = construction_slot(cursor);
    if (id.cursor_generation != slot.generation || id.scan_generation != slot.scan_generation ||
        id.index >= slot.options.size()) {
        throw std::logic_error("stale construction option");
    }
    auto& option         = slot.options[id.index];
    auto& decisions      = candidate_options[slot.candidate_index].victims[option.victim].decisions;
    std::uint16_t choice = 0;
    if (!option.identity) {
        const auto found = std::find(decisions.begin(), decisions.end(), option.decision);
        if (found != decisions.end()) {
            choice = static_cast<std::uint16_t>(1 + found - decisions.begin());
        } else {
            if (decisions.size() == std::numeric_limits<std::uint16_t>::max()) {
                throw std::length_error("construction owner decision capacity exceeded");
            }
            decisions.push_back(std::move(option.decision));
            choice = static_cast<std::uint16_t>(decisions.size());
        }
    }
    slot.choices[option.victim] = choice;
    slot.residual               = construction_residual(slot.candidate_index, slot.choices);
    slot.next_owner = slot.next_option = 0;
    slot.options.clear();
    if (++slot.scan_generation == 0) { ++slot.scan_generation; }
}

inline std::optional<qwen3_6::PressureTargetHandle>
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::construction_target(
    const qwen3_6::PressureConstructionCursor& cursor) {
    auto& slot = construction_slot(cursor);
    if (!find_target(slot.candidate_index, slot.choices) &&
        targets.size() >= candidates.size() + 1U + planning_detail::kOptionalTargetCapacity) {
        return std::nullopt;
    }
    const auto index = intern_target(slot.candidate_index, slot.choices);
    qwen3_6::PressureTargetHandle result;
    result.session_    = this;
    result.generation_ = generation;
    result.index_      = index;
    return result;
}

inline runtime::PressureTargetGuidance
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::guidance(qwen3_6::PressureTargetHandle target) {
    if (!valid(target) || scratch_live) {
        throw std::logic_error("pressure target guidance is stale or conflicts with expansion");
    }
    TargetNode& node = targets[target.index_];
    auto result = guidance_choices(node.candidate_index, victim_choices(node), node.stable_ordinal);
    if (node.assessed_residual &&
        *node.assessed_residual !=
            construction_residual(node.candidate_index, victim_choices(node))) {
        result.physical.requires_exact_feedback = true;
    }
    return result;
}

inline runtime::PressureTargetGuidance
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::guidance_choices(
    std::uint32_t candidate_index, std::span<const std::uint16_t> choices, std::uint32_t ordinal,
    std::optional<std::size_t> override_owner, const PressureDecision* override_decision) {
    populate_options(candidate_index);
    const CandidateState& candidate = *candidates[candidate_index].state;
    const CandidateOptions& options = candidate_options[candidate_index];
    if (choices.size() != options.victims.size()) {
        throw std::logic_error("pressure target victim domain changed");
    }

    guidance_outcomes.clear();
    guidance_checkpoint_changes.clear();
    guidance_recovery.clear();
    bool recovery_complete = true;
    NINFER_QWEN36_RUNTIME_NS::PlanningTransferAccumulator estimated_pressure;
    detail::PhysicalDelta approximate_pressure;
    std::uint32_t total_degradation = 0;
    std::uint32_t total_dropped     = 0;
    for (std::size_t index = 0; index < options.victims.size(); ++index) {
        const std::uint16_t choice                   = choices[index];
        const CandidateVictimOptions& victim_options = options.victims[index];
        if (victim_options.owner_index >= owners.size() ||
            choice > victim_options.decisions.size()) {
            throw std::logic_error("pressure target guidance owner choice is invalid");
        }
        const PressureDecision* selected =
            override_owner && *override_owner == index
                ? override_decision
                : (choice == 0 ? nullptr : &victim_options.decisions[choice - 1U]);
        if (selected == nullptr) { continue; }
        const Owner& victim_owner        = owners[victim_options.owner_index];
        const PressureDecision& decision = *selected;
        for (const auto checkpoint : decision.dropped_checkpoints) {
            guidance_checkpoint_changes.push_back(
                {.owner = victim_owner.id, .checkpoint = checkpoint, .survives = false});
        }
        NINFER_QWEN36_RUNTIME_NS::PlanningTransferAccumulator recovery;
        for (auto transfer : decision.transfer_requirements) {
            if (transfer.direction == runtime::ContextTransferDirection::DeviceToHost) {
                transfer.direction = runtime::ContextTransferDirection::HostToDevice;
                recovery.append(std::span<const runtime::ContextTransferRequirement>(&transfer, 1));
            }
        }
        guidance_recovery.push_back(
            {.owner = victim_owner.id, .additional_restore = recovery.work});
        if (!decision.evicts_continuation && decision.transfer_requirements.empty() &&
            (!decision.state_changes.empty() || !decision.main_kv_changes.empty() ||
             !decision.backend_kv_changes.empty())) {
            recovery_complete = false;
        }
        approximate_pressure.added = NINFER_QWEN36_RUNTIME_NS::planning_resource_sum(
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
            .owner             = victim_owner.id,
            .disposition       = decision.evicts_continuation ? runtime::VictimDisposition::Evicted
                                                              : runtime::VictimDisposition::Retained,
            .degradation_units = units,
            .dropped_checkpoints = decision.checkpoint_drops,
        });
    }
    detail::PhysicalResources residual =
        program->guided_materialization_deficit(candidate, approximate_pressure);
    residual.host.kv_bytes =
        std::max(residual.host.kv_bytes, candidate.blocked_host_allocation_bytes);

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
    for (std::size_t index = 0; index < options.victims.size(); ++index) {
        const CandidateVictimOptions& victim_options = options.victims[index];
        const std::uint16_t eviction_choice          = victim_options.eviction_choice;
        if (eviction_choice == 0 || choices[index] == eviction_choice) { continue; }
        const PressureDecision& eviction = victim_options.decisions[eviction_choice - 1U];
        const PressureDecision* current =
            choices[index] == 0 ? nullptr : &victim_options.decisions[choices[index] - 1U];
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
                .requires_exact_feedback   = candidate.blocked_host_allocation_bytes != 0,
            },
        .estimated_machine_work =
            NINFER_QWEN36_RUNTIME_NS::materialization_machine_work(candidate, estimated_pressure),
        .owner_outcomes             = guidance_outcomes,
        .candidate                  = candidate_ids[candidate_index],
        .stable_target_ordinal      = ordinal,
        .degradation_units          = total_degradation,
        .dropped_checkpoints        = total_dropped,
        .source_mode                = candidate.source_mode,
        .checkpoint_changes         = guidance_checkpoint_changes,
        .recovery_estimates         = guidance_recovery,
        .recovery_estimate_complete = recovery_complete,
    };
}

inline qwen3_6::AssessedPressureTarget<NINFER_QWEN36_VARIANT>
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::assess(qwen3_6::PressureTargetHandle target) {
    if (!valid(target) || scratch_live) {
        throw std::logic_error("pressure target assessment is stale or conflicts with expansion");
    }
    TargetNode& node = targets[target.index_];
    populate_options(node.candidate_index);
    const CandidateState& candidate              = *candidates[node.candidate_index].state;
    const CandidateOptions& options              = candidate_options[node.candidate_index];
    const std::span<const std::uint16_t> choices = victim_choices(node);
    if (choices.size() != options.victims.size()) {
        throw std::logic_error("pressure target victim domain changed");
    }
    const bool identity_target = std::all_of(choices.begin(), choices.end(),
                                             [](std::uint16_t choice) { return choice == 0; });

    selected_private_owners.clear();
    selected_private_owner_ids.clear();
    selected_private_decisions.clear();
    selected_shared_owners.clear();
    selected_shared_owner_ids.clear();
    selected_shared_decisions.clear();
    recovery_private_owners.clear();
    recovery_private_decisions.clear();
    recovery_private_owner_ids.clear();
    recovery_shared_owners.clear();
    recovery_shared_decisions.clear();
    recovery_shared_owner_ids.clear();
    assessment_outcomes.clear();
    assessment_impact_projections.clear();
    assessment_recovery_alternatives.clear();
    std::fill(projected_owner_decisions.begin(), projected_owner_decisions.end(), nullptr);

    std::uint64_t projection_work   = 1;
    std::uint32_t total_degradation = 0;
    std::uint32_t total_dropped     = 0;
    for (std::size_t index = 0; index < options.victims.size(); ++index) {
        const std::uint16_t choice                   = choices[index];
        const CandidateVictimOptions& victim_options = options.victims[index];
        if (victim_options.owner_index >= owners.size() ||
            choice > victim_options.decisions.size()) {
            throw std::logic_error("pressure target owner choice is invalid");
        }
        const Owner& owner = owners[victim_options.owner_index];
        const PressureDecision* decision =
            choice == 0 ? nullptr : &victim_options.decisions[choice - 1U];
        projected_owner_decisions[victim_options.owner_index] = decision;
        if (decision == nullptr) { continue; }
        if (owner.shared) {
            selected_shared_owners.push_back(owner.shared_handle);
            selected_shared_owner_ids.push_back(owner.id);
            selected_shared_decisions.push_back(decision);
        } else {
            selected_private_owners.push_back(owner.private_handle);
            selected_private_owner_ids.push_back(owner.id);
            selected_private_decisions.push_back(decision);
        }
        const std::uint32_t units   = NINFER_QWEN36_RUNTIME_NS::degradation_units(*decision);
        const std::uint32_t dropped = NINFER_QWEN36_RUNTIME_NS::dropped_checkpoint_count(*decision);
        total_degradation           = NINFER_QWEN36_RUNTIME_NS::planning_saturating_u32(
            static_cast<std::uint64_t>(total_degradation) + units);
        total_dropped = NINFER_QWEN36_RUNTIME_NS::planning_saturating_u32(
            static_cast<std::uint64_t>(total_dropped) + dropped);
        assessment_outcomes.push_back(runtime::PressureOwnerOutcome{
            .owner             = owner.id,
            .disposition       = decision->evicts_continuation ? runtime::VictimDisposition::Evicted
                                                               : runtime::VictimDisposition::Retained,
            .degradation_units = units,
            .dropped_checkpoints = dropped,
        });
        ++projection_work;
    }
    for (std::size_t index = 0; index < owners.size(); ++index) {
        const Owner& owner               = owners[index];
        const PressureDecision* decision = projected_owner_decisions[index];
        if (owner.shared) {
            recovery_shared_owners.push_back(owner.shared_handle);
            recovery_shared_decisions.push_back(decision);
            recovery_shared_owner_ids.push_back(owner.id);
        } else {
            recovery_private_owners.push_back(owner.private_handle);
            recovery_private_decisions.push_back(decision);
            recovery_private_owner_ids.push_back(owner.id);
        }
    }

    bool recovery_projection_valid = true;
    if (!identity_target) {
        recovery_projection_valid = program->pressure_checkpoint_recovery_impacts(
            candidate, recovery_private_owners, recovery_private_decisions,
            recovery_private_owner_ids, recovery_shared_owners, recovery_shared_decisions,
            recovery_shared_owner_ids, assessment_impact_projections,
            assessment_recovery_alternatives, recovery_scratch, projection_work);
    }

    runtime::MaterializationPhysicalStatus status =
        runtime::MaterializationPhysicalStatus::StructuralInvalid;
    std::optional<AdmissionCandidate> composed;
    std::optional<CapturePressureCandidate> composed_capture;
    const CandidateState* projected         = &candidate;
    const PhysicalCandidateBinding& binding = candidates[node.candidate_index];
    if (identity_target) {
        status = candidate.identity_assessment.physical_status;
    } else if (recovery_projection_valid) {
        if (binding.admission != nullptr) {
            AdmissionCandidate copy(std::make_unique<AdmissionCandidateImpl>(*binding.admission));
            if (program->compose_pressure_candidate(
                    *copy.impl_, selected_private_owners, selected_private_owner_ids,
                    selected_private_decisions, selected_shared_owners, selected_shared_owner_ids,
                    selected_shared_decisions)) {
                projected = copy.impl_.get();
                composed.emplace(std::move(copy));
            }
        } else {
            CapturePressureCandidate copy(
                std::make_unique<CapturePressureCandidateImpl>(*binding.capture));
            if (program->compose_pressure_candidate(
                    *copy.impl_, selected_private_owners, selected_private_owner_ids,
                    selected_private_decisions, selected_shared_owners, selected_shared_owner_ids,
                    selected_shared_decisions)) {
                projected = copy.impl_.get();
                composed_capture.emplace(std::move(copy));
            }
        }
        if (composed || composed_capture) {
            status = runtime::MaterializationPhysicalStatus::Infeasible;
            if (projected->blocked_host_allocation_bytes == 0 &&
                program->physical_peak_fits(projected->demand.physical_peak_additional)) {
                status = runtime::MaterializationPhysicalStatus::Feasible;
            }
        }
    }
    const bool composed_valid = composed.has_value() || composed_capture.has_value();
    if (identity_target || composed_valid) {
        node.assessed_residual                = program->materialization_deficit(*projected);
        node.assessed_residual->host.kv_bytes = std::max(node.assessed_residual->host.kv_bytes,
                                                         projected->blocked_host_allocation_bytes);
    } else {
        node.assessed_residual.reset();
    }
    const runtime::MaterializationMachineWork machine_work =
        identity_target ? candidate.identity_assessment.machine_work
                        : NINFER_QWEN36_RUNTIME_NS::materialization_machine_work(
                              *projected, selected_private_decisions, selected_shared_decisions);

    bool expandable = identity_target || (recovery_projection_valid && composed_valid);
    if (expandable) {
        expandable = false;
        for (std::size_t index = 0; index < options.victims.size(); ++index) {
            const CandidateVictimOptions& victim_options = options.victims[index];
            const std::uint16_t choice                   = choices[index];
            if ((choice == 0 && !victim_options.decisions.empty()) ||
                (choice != 0 && choice <= victim_options.decisions.size() &&
                 !victim_options.decisions[choice - 1U].evicts_continuation)) {
                expandable = true;
                break;
            }
        }
    }

    std::uint64_t digest = candidate.identity_assessment.assessment_digest;
    if (!identity_target) {
        digest = 1469598103934665603ULL;
        planning_detail::hash_mix(digest, node.candidate_index);
        planning_detail::hash_mix(digest, node.stable_ordinal);
        planning_detail::hash_mix(digest, static_cast<std::uint8_t>(status));
        planning_detail::hash_mix(digest, machine_work.remaining_prefill_work.tokens);
        for (const TransferWork transfer : machine_work.pressure_transfers) {
            planning_detail::hash_mix(digest, transfer.payload_bytes);
            planning_detail::hash_mix(digest, transfer.copy_operations);
        }
        for (const TransferWork transfer : machine_work.candidate_transfers) {
            planning_detail::hash_mix(digest, transfer.payload_bytes);
            planning_detail::hash_mix(digest, transfer.copy_operations);
        }
        planning_detail::hash_mix(digest, total_degradation);
        for (const std::uint16_t choice : choices) { planning_detail::hash_mix(digest, choice); }
    }

    runtime::PressureTargetAssessment result{
        .physical_status       = status,
        .source_mode           = projected->source_mode,
        .machine_work          = machine_work,
        .owner_outcomes        = assessment_outcomes,
        .checkpoint_impacts    = {},
        .candidate             = candidate_ids[node.candidate_index],
        .stable_target_ordinal = node.stable_ordinal,
        .degradation_units     = total_degradation,
        .dropped_checkpoints   = total_dropped,
        .projection_work       = projection_work,
        .assessment_digest     = digest,
        .expandable            = expandable,
        .root_maximal          = node.root_maximal,
    };
    std::optional<AdmissionCandidate> executable;
    std::optional<CapturePressureCandidate> capture_executable;
    if (status == runtime::MaterializationPhysicalStatus::Feasible) {
        if (identity_target) {
            if (binding.admission != nullptr) {
                executable.emplace(std::make_unique<AdmissionCandidateImpl>(*binding.admission));
            } else {
                capture_executable.emplace(
                    std::make_unique<CapturePressureCandidateImpl>(*binding.capture));
            }
        } else if (composed) {
            executable.emplace(std::move(*composed));
        } else if (composed_capture) {
            capture_executable.emplace(std::move(*composed_capture));
        }
    }
    const std::uint32_t slot_index = acquire_assessment_slot();
    AssessmentSlot& slot           = assessment_slots[slot_index];
    if (slot.owner_outcomes.capacity() < assessment_outcomes.size() ||
        slot.checkpoint_impacts.capacity() < assessment_impact_projections.size() ||
        slot.recovery_alternatives.capacity() < assessment_recovery_alternatives.size()) {
        release_assessment_slot(this, slot_index, slot.generation);
        throw std::logic_error("pressure assessment exceeded preallocated result storage");
    }
    slot.owner_outcomes.insert(slot.owner_outcomes.end(), assessment_outcomes.begin(),
                               assessment_outcomes.end());
    slot.recovery_alternatives.insert(slot.recovery_alternatives.end(),
                                      assessment_recovery_alternatives.begin(),
                                      assessment_recovery_alternatives.end());
    for (const PressureCheckpointRecoveryProjection& impact : assessment_impact_projections) {
        if (impact.alternative_offset > slot.recovery_alternatives.size() ||
            impact.alternative_count >
                slot.recovery_alternatives.size() - impact.alternative_offset) {
            release_assessment_slot(this, slot_index, slot.generation);
            throw std::logic_error("pressure recovery work span is invalid");
        }
        slot.checkpoint_impacts.push_back(runtime::PressureCheckpointRecoveryImpact{
            .owner      = impact.owner,
            .checkpoint = impact.checkpoint,
            .target_recovery_work =
                std::span<const runtime::CheckpointRecoveryAlternativeWork>(
                    slot.recovery_alternatives)
                    .subspan(impact.alternative_offset, impact.alternative_count),
            .survives = impact.survives,
        });
    }
    result.owner_outcomes     = slot.owner_outcomes;
    result.checkpoint_impacts = slot.checkpoint_impacts;
    return qwen3_6::AssessedPressureTarget<NINFER_QWEN36_VARIANT>(
        this, generation, target.index_, result, slot_index, slot.generation,
        &PressurePlanningSessionImpl::release_assessment_slot, std::move(executable),
        std::move(capture_executable));
}

inline qwen3_6::PreparedPressureExpansion<NINFER_QWEN36_VARIANT>
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::prepare_expansion(
    qwen3_6::PressureTargetHandle parent, std::uint32_t maximum_owners) {
    if (!valid(parent) || scratch_live || maximum_owners == 0) {
        throw std::logic_error("pressure expansion parent is stale or scratch is busy");
    }
    const TargetNode& node = targets[parent.index_];
    populate_options(node.candidate_index);
    CandidateOptions& options                           = candidate_options[node.candidate_index];
    const std::span<const std::uint16_t> parent_choices = victim_choices(node);
    if (parent_choices.size() != options.victims.size()) {
        throw std::logic_error("pressure target victim domain changed");
    }
    expansion_scratch.clear();
    prepared_owner_decisions.clear();
    prepared_new_count  = 0;
    scratch_choice_mark = target_choice_arena.size();

    const CandidateState& candidate = *candidates[node.candidate_index].state;
    const std::optional<typename Core::MaterializationSourceProtection> protection =
        program->materialization_source_protection(candidate);
    if (!protection) { throw std::logic_error("pressure expansion source protection is stale"); }
    const bool identity = std::all_of(parent_choices.begin(), parent_choices.end(),
                                      [](std::uint16_t choice) { return choice == 0; });
    detail::PhysicalResources residual;
    if (identity) {
        residual = candidate.identity_pressure_deficit;
        residual.host.kv_bytes =
            std::max(residual.host.kv_bytes, candidate.blocked_host_allocation_bytes);
    } else {
        if (!node.assessed_residual) {
            throw std::logic_error("pressure target must be assessed before expansion");
        }
        residual = *node.assessed_residual;
    }

    const auto append = [&](std::size_t victim_index, std::uint16_t choice) {
        if (parent_choices.size() > target_choice_arena.capacity() - target_choice_arena.size() ||
            target_choice_arena.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("pressure expansion choice arena is full");
        }
        const std::uint32_t offset = static_cast<std::uint32_t>(target_choice_arena.size());
        target_choice_arena.insert(target_choice_arena.end(), parent_choices.begin(),
                                   parent_choices.end());
        target_choice_arena[offset + victim_index] = choice;
        TargetNode child{
            .candidate_index      = node.candidate_index,
            .victim_choice_offset = offset,
            .victim_choice_count  = static_cast<std::uint32_t>(parent_choices.size()),
        };
        const std::span<const std::uint16_t> child_choices = victim_choices(child);
        const bool duplicate_scratch                       = std::any_of(
            expansion_scratch.begin(), expansion_scratch.end(), [&](const TargetNode& existing) {
                return existing.candidate_index == child.candidate_index &&
                       planning_detail::same_choices(victim_choices(existing), child_choices);
            });
        if (duplicate_scratch) {
            target_choice_arena.resize(offset);
            return;
        }
        const bool existing = find_target(child.candidate_index, child_choices) != nullptr;
        if (!existing) { ++prepared_new_count; }
        expansion_scratch.push_back(child);
    };

    const auto intern_prepared_decision = [&](std::size_t victim_index, PressureDecision decision) {
        std::vector<PressureDecision>& decisions = options.victims[victim_index].decisions;
        const auto existing = std::find(decisions.begin(), decisions.end(), decision);
        if (existing != decisions.end()) {
            return static_cast<std::uint16_t>(1U + (existing - decisions.begin()));
        }
        const auto prepared =
            std::find_if(prepared_owner_decisions.begin(), prepared_owner_decisions.end(),
                         [&](const PreparedOwnerDecision& item) {
                             return item.candidate_index == node.candidate_index &&
                                    item.victim_index == victim_index && item.decision == decision;
                         });
        if (prepared != prepared_owner_decisions.end()) { return prepared->choice; }
        const std::size_t staged = static_cast<std::size_t>(
            std::count_if(prepared_owner_decisions.begin(), prepared_owner_decisions.end(),
                          [&](const PreparedOwnerDecision& item) {
                              return item.candidate_index == node.candidate_index &&
                                     item.victim_index == victim_index;
                          }));
        const std::size_t value = decisions.size() + staged + 1U;
        if (value > std::numeric_limits<std::uint16_t>::max()) {
            throw std::overflow_error("pressure owner target count is not representable");
        }
        const std::uint16_t choice = static_cast<std::uint16_t>(value);
        prepared_owner_decisions.push_back(PreparedOwnerDecision{
            .candidate_index = node.candidate_index,
            .victim_index    = static_cast<std::uint32_t>(victim_index),
            .choice          = choice,
            .decision        = std::move(decision),
        });
        return choice;
    };

    prepared_owner_end = static_cast<std::uint32_t>(std::min<std::size_t>(
        options.victims.size(),
        static_cast<std::size_t>(node.next_expansion_owner) + maximum_owners));
    try {
        for (std::size_t victim_index = node.next_expansion_owner;
             victim_index < prepared_owner_end; ++victim_index) {
            CandidateVictimOptions& victim_options   = options.victims[victim_index];
            const std::uint16_t current_choice       = parent_choices[victim_index];
            std::vector<PressureDecision>& decisions = victim_options.decisions;
            if (current_choice > decisions.size() ||
                (current_choice != 0 && decisions[current_choice - 1U].evicts_continuation)) {
                continue;
            }
            const PressureDecision* current =
                current_choice == 0 ? nullptr : &decisions[current_choice - 1U];
            std::vector<PressureDecision> successors =
                pressure_successors(victim_options, residual, *protection, current);
            for (PressureDecision& successor : successors) {
                const std::uint16_t choice =
                    intern_prepared_decision(victim_index, std::move(successor));
                if (choice == current_choice) { continue; }
                append(victim_index, choice);
            }
        }
    } catch (...) {
        target_choice_arena.resize(scratch_choice_mark);
        expansion_scratch.clear();
        prepared_owner_decisions.clear();
        prepared_new_count = 0;
        throw;
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
        if (prepared_decision.victim_index >= options.victims.size()) {
            throw std::logic_error("prepared pressure owner index is invalid");
        }
        CandidateVictimOptions& victim_options   = options.victims[prepared_decision.victim_index];
        std::vector<PressureDecision>& decisions = victim_options.decisions;
        if (prepared_decision.choice != decisions.size() + 1U) {
            throw std::logic_error("prepared pressure owner choice is not canonical");
        }
        decisions.push_back(std::move(prepared_decision.decision));
    }
    std::size_t choice_write          = scratch_choice_mark;
    std::uint32_t committed_new_count = 0;
    for (TargetNode& child : expansion_scratch) {
        const std::span<const std::uint16_t> child_choices = victim_choices(child);
        TargetNode* existing = find_target(child.candidate_index, child_choices);
        std::uint32_t index  = 0;
        if (existing == nullptr) {
            if (child_choices.size() > target_choice_arena.size() - choice_write ||
                targets.size() == targets.capacity()) {
                throw std::length_error("committed pressure target exceeds its arena");
            }
            for (std::size_t choice = 0; choice < child_choices.size(); ++choice) {
                target_choice_arena[choice_write + choice] = child_choices[choice];
            }
            child.victim_choice_offset = static_cast<std::uint32_t>(choice_write);
            choice_write += child_choices.size();
            child.stable_ordinal = static_cast<std::uint32_t>(targets.size());
            targets.push_back(child);
            index = static_cast<std::uint32_t>(targets.size() - 1U);
            index_target(index);
            ++committed_new_count;
        } else {
            index = static_cast<std::uint32_t>(existing - targets.data());
        }
        qwen3_6::PressureTargetHandle handle;
        handle.session_    = this;
        handle.generation_ = generation;
        handle.index_      = index;
        committed_children.push_back(handle);
    }
    if (committed_new_count != prepared_new_count) {
        throw std::logic_error("prepared pressure target count changed before commit");
    }
    target_choice_arena.resize(choice_write);
    const std::uint32_t new_count = prepared_new_count;
    auto& parent                  = targets[prepared.parent_index_];
    parent.next_expansion_owner   = prepared_owner_end;
    const bool complete =
        prepared_owner_end == candidate_options[parent.candidate_index].victims.size();
    prepared.session_            = nullptr;
    prepared.session_generation_ = 0;
    prepared.scratch_generation_ = 0;
    expansion_scratch.clear();
    prepared_owner_decisions.clear();
    prepared_new_count  = 0;
    scratch_choice_mark = target_choice_arena.size();
    scratch_live        = false;
    return qwen3_6::PressureExpansionView{
        .children            = committed_children,
        .new_canonical_count = new_count,
        .complete            = complete,
    };
}

inline void PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::discard_expansion(
    qwen3_6::PreparedPressureExpansion<NINFER_QWEN36_VARIANT>&& prepared) noexcept {
    if (scratch_live && prepared.session_ == this && prepared.session_generation_ == generation &&
        prepared.scratch_generation_ == scratch_generation) {
        expansion_scratch.clear();
        prepared_owner_decisions.clear();
        target_choice_arena.resize(scratch_choice_mark);
        prepared_new_count = 0;
        scratch_live       = false;
    }
    prepared.session_            = nullptr;
    prepared.session_generation_ = 0;
    prepared.scratch_generation_ = 0;
}

inline runtime::PrefillWork
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::shared_capture_split_prefill_work(
    const qwen3_6::AssessedPressureTarget<NINFER_QWEN36_VARIANT>& assessed,
    const NINFER_QWEN36_RUNTIME_NS::PreparedPromptData& prompt,
    std::span<const std::uint32_t> frontiers) const {
    if (assessed.session_ != this || assessed.session_generation_ != generation || scratch_live ||
        assessed.target_index_ >= targets.size() || !assessed.executable_ ||
        assessed.capture_executable_ ||
        assessed.assessment_.physical_status != runtime::MaterializationPhysicalStatus::Feasible) {
        throw std::logic_error("shared capture cost requires a feasible assessed materialization");
    }
    return program->shared_capture_split_prefill_work(*assessed.executable_, prompt, frontiers);
}

inline std::optional<
    typename PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::AdmissionCandidate>
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::seal(
    qwen3_6::AssessedPressureTarget<NINFER_QWEN36_VARIANT>&& assessed,
    const NINFER_QWEN36_RUNTIME_NS::PreparedPromptData& prompt,
    runtime::FinalScheduleIntent intent) {
    if (assessed.session_ != this || assessed.session_generation_ != generation || scratch_live ||
        assessed.target_index_ >= targets.size() || !assessed.executable_ ||
        assessed.capture_executable_ ||
        assessed.assessment_.physical_status != runtime::MaterializationPhysicalStatus::Feasible) {
        throw std::logic_error("pressure assessment is not sealable as materialization");
    }
    std::optional<AdmissionCandidate> sealed = std::move(assessed.executable_);
    assessed.reset();
    program->select_shared_captures(*sealed, prompt, intent.shared_capture_frontiers);
    if (sealed->impl_->blocked_host_allocation_bytes != 0 ||
        program->revalidate_materialization(*sealed, prompt) != runtime::PreflightStatus::Ready) {
        return std::nullopt;
    }
    return sealed;
}

inline std::optional<
    typename PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::CapturePressureCandidate>
PressurePlanningSessionImpl<NINFER_QWEN36_VARIANT>::seal_capture(
    qwen3_6::AssessedPressureTarget<NINFER_QWEN36_VARIANT>&& assessed) {
    if (assessed.session_ != this || assessed.session_generation_ != generation || scratch_live ||
        assessed.target_index_ >= targets.size() || assessed.executable_ ||
        !assessed.capture_executable_ ||
        assessed.assessment_.physical_status != runtime::MaterializationPhysicalStatus::Feasible) {
        throw std::logic_error("pressure assessment is not sealable as capture");
    }
    std::optional<CapturePressureCandidate> sealed = std::move(assessed.capture_executable_);
    assessed.reset();
    if (sealed->impl_->blocked_host_allocation_bytes != 0 ||
        !program->physical_peak_fits(sealed->impl_->demand.physical_peak_additional)) {
        return std::nullopt;
    }
    return sealed;
}

} // namespace ninfer::targets::qwen3_6::detail
