#pragma once

#include "models/qwen3_5/program/program_impl.h"

#include <array>
#include <limits>
#include <tuple>

namespace ninfer::models::qwen3_5::detail {

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
    std::span<const qwen3_5::detail::PressureDecision* const> private_decisions,
    std::span<const qwen3_5::detail::PressureDecision* const> shared_decisions) noexcept {
    PlanningTransferAccumulator pressure;
    for (const qwen3_5::detail::PressureDecision* decision : private_decisions) {
        if (decision != nullptr) { pressure.append(decision->transfer_requirements); }
    }
    for (const qwen3_5::detail::PressureDecision* decision : shared_decisions) {
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

inline std::uint32_t degradation_units(const qwen3_5::detail::PressureDecision& decision) noexcept {
    std::uint64_t units = decision.evicts_continuation ? 1U : 0U;
    units += decision.state_changes.size();
    units += decision.main_kv_changes.size();
    units += decision.backend_kv_changes.size();
    units += decision.checkpoint_drops;
    return planning_saturating_u32(units);
}

inline std::uint32_t
dropped_checkpoint_count(const qwen3_5::detail::PressureDecision& decision) noexcept {
    return decision.checkpoint_drops;
}

} // namespace ninfer::models::qwen3_5::detail
