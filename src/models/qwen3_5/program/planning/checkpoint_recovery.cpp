#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/planning/pressure_planner.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

namespace {

runtime::PrefillWork interval_rebuild_work(std::uint32_t begin_frontier,
                                           runtime::PrefillWork begin_work,
                                           std::uint32_t end_frontier,
                                           runtime::PrefillWork end_work,
                                           std::uint32_t prefill_chunk);

std::optional<qwen3_5::TargetKVRequirement>
retained_requirement_after_drop(const qwen3_5::ContinuationSummary& summary,
                                runtime::CheckpointRef dropped) noexcept;

runtime::PrefillWork interval_rebuild_work(std::uint32_t begin_frontier,
                                           runtime::PrefillWork begin_work,
                                           std::uint32_t end_frontier,
                                           runtime::PrefillWork end_work,
                                           std::uint32_t prefill_chunk) {
    if (end_frontier < begin_frontier || end_work.vision_items < begin_work.vision_items ||
        end_work.vision_patches < begin_work.vision_patches) {
        throw std::logic_error("checkpoint rebuild interval is not monotonic");
    }
    return runtime::make_prefill_work(begin_frontier, end_frontier - begin_frontier,
                                      end_work.vision_items - begin_work.vision_items,
                                      end_work.vision_patches - begin_work.vision_patches,
                                      prefill_chunk);
}

std::optional<qwen3_5::TargetKVRequirement>
retained_requirement_after_drop(const qwen3_5::ContinuationSummary& summary,
                                runtime::CheckpointRef dropped) noexcept {
    return retained_requirement_after_drops(summary,
                                            std::span<const runtime::CheckpointRef>(&dropped, 1));
}

} // namespace

void ProgramImpl::begin_pressure_page_scratch() const noexcept {
    if (++pressure_page_scratch_generation_ == 0) {
        std::fill(pressure_text_page_scratch_.begin(), pressure_text_page_scratch_.end(),
                  PressurePageScratchSlot{});
        std::fill(pressure_backend_page_scratch_.begin(), pressure_backend_page_scratch_.end(),
                  PressurePageScratchSlot{});
        pressure_page_scratch_generation_ = 1;
    }
    pressure_text_selected_pages_.clear();
    pressure_backend_selected_pages_.clear();
}

ProgramImpl::PressurePageScratchSlot&
ProgramImpl::pressure_page_scratch(const LogicalKVPageStore& store,
                                   LogicalKVPageHandle page) const {
    std::vector<PressurePageScratchSlot>* slots = nullptr;
    if (&store == text_kv_pages.get()) {
        slots = &pressure_text_page_scratch_;
    } else if (&store == backend_kv_pages.get()) {
        slots = &pressure_backend_page_scratch_;
    } else {
        throw std::logic_error("pressure page scratch received a foreign logical store");
    }
    const std::uint32_t index = store.descriptor_index(page);
    if (index >= slots->size()) {
        throw std::logic_error("pressure page scratch descriptor is outside startup capacity");
    }
    PressurePageScratchSlot& slot = (*slots)[index];
    if (slot.generation != pressure_page_scratch_generation_) {
        slot = PressurePageScratchSlot{.generation = pressure_page_scratch_generation_};
    }
    return slot;
}

const ProgramImpl::PressurePageScratchSlot*
ProgramImpl::find_pressure_page_scratch(const LogicalKVPageStore& store,
                                        LogicalKVPageHandle page) const {
    const std::vector<PressurePageScratchSlot>* slots = nullptr;
    if (&store == text_kv_pages.get()) {
        slots = &pressure_text_page_scratch_;
    } else if (&store == backend_kv_pages.get()) {
        slots = &pressure_backend_page_scratch_;
    } else {
        throw std::logic_error("pressure page scratch received a foreign logical store");
    }
    const std::uint32_t index = store.descriptor_index(page);
    if (index >= slots->size()) {
        throw std::logic_error("pressure page scratch descriptor is outside startup capacity");
    }
    const PressurePageScratchSlot& slot = (*slots)[index];
    return slot.generation == pressure_page_scratch_generation_ ? &slot : nullptr;
}

std::vector<runtime::ContextTransferRequirement>
ProgramImpl::checkpoint_restore_requirements(const SequenceKVBundle& kv,
                                             const qwen3_5::TargetKVRequirement& requirement,
                                             StateImageHandle state) const {
    if (!state_store->valid(state)) {
        throw std::logic_error("checkpoint restore requirement source is incomplete");
    }
    std::vector<runtime::ContextTransferRequirement> requirements;
    requirements.reserve(3);
    if (state_store->residency(state) == StateReplicaResidency::HostOnly) {
        if (host_state_images == nullptr) {
            throw std::logic_error("Host-only checkpoint has no Host StateImage pool");
        }
        requirements.push_back(state_transfer_requirement(
            host_state_images->layout(), runtime::ContextTransferDirection::HostToDevice));
    }
    const auto append_kv = [&](const KVAddressSpaceStore& addresses,
                               const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                               std::uint32_t required, runtime::ContextResourceClass resource) {
        if (required == 0) { return; }
        if (required > addresses.mapped_pages(address)) {
            throw std::logic_error("checkpoint KV requirement exceeds its address space");
        }
        std::uint32_t missing = 0;
        std::uint32_t runs    = 0;
        std::optional<HostKVPageReplica> previous;
        for (std::uint32_t page = 0; page < required; ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (pages.device_resident(logical)) { continue; }
            if (!pages.host_resident(logical)) {
                throw std::logic_error("checkpoint KV page has no restorable replica");
            }
            const HostKVPageReplica replica = pages.host_replica(logical);
            if (!previous || previous->extent != replica.extent ||
                previous->page_offset + 1U != replica.page_offset) {
                ++runs;
            }
            previous = replica;
            ++missing;
        }
        if (missing == 0) { return; }
        const HostKVPageLayout layout = plan_host_kv_page_layout(pages.physical_pool().geometry());
        requirements.push_back(kv_transfer_requirement(
            resource, runtime::ContextTransferDirection::HostToDevice, layout, missing, runs));
    };
    append_kv(*text_kv_addresses, *text_kv_pages, kv.text, requirement.main_pages,
              runtime::ContextResourceClass::MainKV);
    if (requirement.backend_pages != 0) {
        if (!kv.backend || !backend_kv_addresses || !backend_kv_pages) {
            throw std::logic_error("checkpoint Backend KV requirement has no typed store");
        }
        append_kv(*backend_kv_addresses, *backend_kv_pages, *kv.backend, requirement.backend_pages,
                  runtime::ContextResourceClass::BackendKV);
    }
    return requirements;
}

bool ProgramImpl::pressure_checkpoint_recovery_impacts(
    const ResourceCandidateState& candidate,
    std::span<const ContinuationHandle* const> private_owners,
    std::span<const qwen3_5::detail::PressureDecision* const> private_decisions,
    std::span<const runtime::PlanningOwnerId> private_owner_ids,
    std::span<const SharedPrefixHandle* const> shared_owners,
    std::span<const qwen3_5::detail::PressureDecision* const> shared_decisions,
    std::span<const runtime::PlanningOwnerId> shared_owner_ids,
    std::vector<qwen3_5::detail::PressureCheckpointRecoveryProjection>& output,
    std::vector<runtime::CheckpointRecoveryAlternativeWork>& alternatives,
    PressureRecoveryScratch& scratch, std::uint64_t& projection_work) const {
    if (private_owners.size() != private_decisions.size() ||
        private_owners.size() != private_owner_ids.size() ||
        shared_owners.size() != shared_decisions.size() ||
        shared_owners.size() != shared_owner_ids.size() ||
        candidate.planning_revision != resource_revision_) {
        return false;
    }

    using StatePlacement       = PressureRecoveryScratch::StatePlacement;
    using OwnerProjection      = PressureRecoveryScratch::OwnerProjection;
    using CheckpointProjection = PressureRecoveryScratch::CheckpointProjection;

    struct PagePlacement {
        bool device              = false;
        bool host                = false;
        std::uint64_t host_group = 0;
    };

    std::vector<StatePlacement>& state_placements  = scratch.state_placements;
    std::vector<OwnerProjection>& projected_owners = scratch.owners;
    std::vector<CheckpointProjection>& checkpoints = scratch.checkpoints;
    std::vector<std::optional<runtime::CheckpointRecoveryAlternativeWork>>& target_direct =
        scratch.direct_work;
    state_placements.clear();
    projected_owners.clear();
    checkpoints.clear();
    target_direct.clear();
    if (projected_owners.capacity() < private_owners.size() + shared_owners.size()) {
        throw std::logic_error("pressure recovery owner scratch is undersized");
    }
    begin_pressure_page_scratch();
    std::uint64_t next_host_group = 1;

    const auto set_state_placement = [&](StateImageHandle state, bool device, bool host) -> bool {
        const auto found = std::find_if(state_placements.begin(), state_placements.end(),
                                        [&](const auto& item) { return item.state == state; });
        if (found != state_placements.end()) {
            return found->device == device && found->host == host;
        }
        if (state_placements.size() == state_placements.capacity()) {
            throw std::logic_error("pressure recovery State scratch is undersized");
        }
        state_placements.push_back(StatePlacement{.state = state, .device = device, .host = host});
        return true;
    };
    const auto set_page_placement = [&](const LogicalKVPageStore& store, LogicalKVPageHandle page,
                                        bool device, bool host, std::uint64_t host_group) -> bool {
        PressurePageScratchSlot& slot = pressure_page_scratch(store, page);
        if (slot.projected) { return slot.device == device && slot.host == host; }
        slot.projected  = true;
        slot.device     = device;
        slot.host       = host;
        slot.host_group = host_group;
        return true;
    };
    const auto apply_kv_action = [&](const KVAddressSpaceStore* addresses,
                                     const LogicalKVPageStore* pages,
                                     std::optional<KVAddressSpaceHandle> address,
                                     const qwen3_5::detail::PressureKVDecision& action) -> bool {
        if (action.kind == qwen3_5::detail::PressureKVDecisionKind::None) {
            return action.page_count == 0;
        }
        if (addresses == nullptr || pages == nullptr || !address || !addresses->valid(*address) ||
            action.page_count == 0) {
            return false;
        }
        const std::uint32_t mapped = addresses->mapped_pages(*address);
        if (action.begin_page > mapped || action.page_count > mapped - action.begin_page) {
            return false;
        }
        const bool drops_host =
            action.kind == qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate;
        const std::uint64_t host_group = drops_host ? 0 : next_host_group++;
        for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
            const LogicalKVPageHandle page =
                addresses->logical_page(*address, action.begin_page + offset);
            if (!set_page_placement(*pages, page, drops_host, !drops_host, host_group)) {
                return false;
            }
        }
        return true;
    };
    const auto apply_decision = [&](const SequenceState* sequence, const SharedPrefixState* shared,
                                    const qwen3_5::detail::PressureDecision* decision) -> bool {
        if (decision == nullptr || decision->evicts_continuation) { return true; }
        for (const qwen3_5::detail::PressureStateDecision change : decision->state_changes) {
            std::optional<StateImageHandle> state;
            switch (change) {
            case qwen3_5::detail::PressureStateDecision::None:
                return false;
            case qwen3_5::detail::PressureStateDecision::DropEndpointDeviceDuplicate:
            case qwen3_5::detail::PressureStateDecision::DemoteEndpointToHost:
            case qwen3_5::detail::PressureStateDecision::DropEndpointHostDuplicate:
                if (sequence == nullptr) { return false; }
                state = sequence->state.read;
                break;
            case qwen3_5::detail::PressureStateDecision::DropRewriteDeviceDuplicate:
            case qwen3_5::detail::PressureStateDecision::DemoteRewriteToHost:
            case qwen3_5::detail::PressureStateDecision::DropRewriteHostDuplicate:
                if (sequence == nullptr || !sequence->rewrite_state) { return false; }
                state = *sequence->rewrite_state;
                break;
            case qwen3_5::detail::PressureStateDecision::DropSharedDeviceDuplicate:
            case qwen3_5::detail::PressureStateDecision::DemoteSharedToHost:
            case qwen3_5::detail::PressureStateDecision::DropSharedHostDuplicate:
                if (shared == nullptr) { return false; }
                state = shared->state;
                break;
            }
            const bool drops_host =
                change == qwen3_5::detail::PressureStateDecision::DropEndpointHostDuplicate ||
                change == qwen3_5::detail::PressureStateDecision::DropRewriteHostDuplicate ||
                change == qwen3_5::detail::PressureStateDecision::DropSharedHostDuplicate;
            if (!set_state_placement(*state, drops_host, !drops_host)) { return false; }
        }
        const SequenceKVBundle* kv =
            sequence != nullptr ? (sequence->kv ? &*sequence->kv : nullptr)
                                : (shared != nullptr && shared->kv ? &*shared->kv : nullptr);
        if (kv == nullptr) { return false; }
        for (const qwen3_5::detail::PressureKVDecision& action : decision->main_kv_changes) {
            if (!apply_kv_action(text_kv_addresses.get(), text_kv_pages.get(), kv->text, action)) {
                return false;
            }
        }
        for (const qwen3_5::detail::PressureKVDecision& action : decision->backend_kv_changes) {
            if (!apply_kv_action(backend_kv_addresses.get(), backend_kv_pages.get(), kv->backend,
                                 action)) {
                return false;
            }
        }
        return true;
    };

    for (std::size_t index = 0; index < private_owners.size(); ++index) {
        const ContinuationHandle* handle = private_owners[index];
        if (handle == nullptr || !valid_continuation(*handle)) { return false; }
        const SequenceState& sequence = continuation_states[ContractAccess::index(*handle)];
        projected_owners.push_back(OwnerProjection{
            .sequence = &sequence,
            .decision = private_decisions[index],
            .owner    = private_owner_ids[index],
        });
        if (!apply_decision(&sequence, nullptr, private_decisions[index])) { return false; }
    }
    for (std::size_t index = 0; index < shared_owners.size(); ++index) {
        const SharedPrefixHandle* handle = shared_owners[index];
        if (handle == nullptr || !valid_shared_prefix(*handle)) { return false; }
        const SharedPrefixState& shared = shared_prefix_states[ContractAccess::index(*handle)];
        projected_owners.push_back(OwnerProjection{
            .shared   = &shared,
            .decision = shared_decisions[index],
            .owner    = shared_owner_ids[index],
        });
        if (!apply_decision(nullptr, &shared, shared_decisions[index])) { return false; }
    }

    const auto final_state_placement = [&](StateImageHandle state) -> StatePlacement {
        const auto found = std::find_if(state_placements.begin(), state_placements.end(),
                                        [&](const auto& item) { return item.state == state; });
        if (found != state_placements.end()) { return *found; }
        const StateReplicaResidency residency = state_store->residency(state);
        return StatePlacement{
            .state  = state,
            .device = residency == StateReplicaResidency::DeviceOnly ||
                      residency == StateReplicaResidency::Both,
            .host = residency == StateReplicaResidency::HostOnly ||
                    residency == StateReplicaResidency::Both,
        };
    };
    const auto final_page_placement = [&](const LogicalKVPageStore& store,
                                          LogicalKVPageHandle page) -> PagePlacement {
        const PressurePageScratchSlot* slot = find_pressure_page_scratch(store, page);
        if (slot != nullptr && slot->projected) {
            return PagePlacement{
                .device     = slot->device,
                .host       = slot->host,
                .host_group = slot->host_group,
            };
        }
        return PagePlacement{
            .device = store.device_resident(page),
            .host   = store.host_resident(page),
        };
    };

    const auto target_direct_work = [&](const SequenceKVBundle& kv,
                                        const CheckpointProjection& checkpoint)
        -> std::optional<runtime::CheckpointRecoveryAlternativeWork> {
        if (!checkpoint.survives) { return std::nullopt; }
        std::array<runtime::ContextTransferRequirement, 3> requirements{};
        std::size_t requirement_count = 0;
        const auto append_requirement = [&](runtime::ContextTransferRequirement requirement) {
            if (requirement_count >= requirements.size()) {
                throw std::logic_error("checkpoint recovery requirement capacity exceeded");
            }
            requirements[requirement_count++] = requirement;
        };
        const StatePlacement state = final_state_placement(checkpoint.state);
        if (!state.device) {
            if (!state.host || host_state_images == nullptr) { return std::nullopt; }
            append_requirement(state_transfer_requirement(
                host_state_images->layout(), runtime::ContextTransferDirection::HostToDevice));
        }
        const auto append_kv = [&](const KVAddressSpaceStore& addresses,
                                   const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                   std::uint32_t required,
                                   runtime::ContextResourceClass resource) -> bool {
            if (required == 0) { return true; }
            if (!addresses.valid(address) || required > addresses.mapped_pages(address)) {
                return false;
            }
            std::uint32_t missing = 0;
            std::uint32_t runs    = 0;
            std::optional<PagePlacement> previous;
            std::optional<HostKVPageReplica> previous_replica;
            for (std::uint32_t offset = 0; offset < required; ++offset) {
                const LogicalKVPageHandle page = addresses.logical_page(address, offset);
                const PagePlacement placement  = final_page_placement(pages, page);
                ++projection_work;
                if (placement.device) {
                    previous.reset();
                    previous_replica.reset();
                    continue;
                }
                if (!placement.host) { return false; }
                bool contiguous = false;
                std::optional<HostKVPageReplica> replica;
                if (pages.host_resident(page)) { replica = pages.host_replica(page); }
                if (previous) {
                    if (placement.host_group != 0 && placement.host_group == previous->host_group) {
                        contiguous = true;
                    } else if (replica && previous_replica &&
                               replica->extent == previous_replica->extent &&
                               replica->page_offset == previous_replica->page_offset + 1U) {
                        contiguous = true;
                    }
                }
                if (!contiguous) { ++runs; }
                ++missing;
                previous         = placement;
                previous_replica = replica;
            }
            if (missing != 0) {
                const HostKVPageLayout layout =
                    plan_host_kv_page_layout(pages.physical_pool().geometry());
                append_requirement(kv_transfer_requirement(
                    resource, runtime::ContextTransferDirection::HostToDevice, layout, missing,
                    runs));
            }
            return true;
        };
        if (!append_kv(*text_kv_addresses, *text_kv_pages, kv.text,
                       checkpoint.checkpoint.required_kv.main_pages,
                       runtime::ContextResourceClass::MainKV)) {
            return std::nullopt;
        }
        if (checkpoint.checkpoint.required_kv.backend_pages != 0) {
            if (!kv.backend || !backend_kv_addresses || !backend_kv_pages ||
                !append_kv(*backend_kv_addresses, *backend_kv_pages, *kv.backend,
                           checkpoint.checkpoint.required_kv.backend_pages,
                           runtime::ContextResourceClass::BackendKV)) {
                return std::nullopt;
            }
        }
        return recovery_alternative_work(std::span<const runtime::ContextTransferRequirement>(
            requirements.data(), requirement_count));
    };

    for (const OwnerProjection& owner : projected_owners) {
        const SequenceKVBundle* kv =
            owner.sequence != nullptr
                ? (owner.sequence->kv ? &*owner.sequence->kv : nullptr)
                : (owner.shared != nullptr && owner.shared->kv ? &*owner.shared->kv : nullptr);
        if (kv == nullptr) { return false; }
        checkpoints.clear();
        target_direct.clear();
        if (owner.sequence != nullptr) {
            qwen3_5::ContinuationSummary& summary = scratch.continuation_summary;
            populate_continuation_summary(*owner.sequence, summary);
            const std::size_t checkpoint_count = summary.endpoint.has_value() +
                                                 summary.rewrite.has_value() +
                                                 summary.long_anchors.size();
            if (checkpoint_count > checkpoints.capacity() ||
                checkpoint_count > target_direct.capacity()) {
                throw std::logic_error("pressure recovery checkpoint scratch is undersized");
            }
            if (summary.endpoint) {
                checkpoints.push_back(CheckpointProjection{
                    .checkpoint = *summary.endpoint,
                    .state      = owner.sequence->state.read,
                });
            }
            if (summary.rewrite) {
                if (!owner.sequence->rewrite_state) { return false; }
                checkpoints.push_back(CheckpointProjection{
                    .checkpoint = *summary.rewrite,
                    .state      = *owner.sequence->rewrite_state,
                });
            }
            if (summary.long_anchors.size() != owner.sequence->long_anchors.size()) {
                return false;
            }
            for (std::size_t index = 0; index < summary.long_anchors.size(); ++index) {
                checkpoints.push_back(CheckpointProjection{
                    .checkpoint = summary.long_anchors[index],
                    .state      = owner.sequence->long_anchors[index].state,
                });
            }
        } else {
            if (checkpoints.capacity() == 0 || target_direct.capacity() == 0) {
                throw std::logic_error("pressure recovery shared scratch is undersized");
            }
            const qwen3_5::CheckpointSummary checkpoint =
                shared_prefix_summary(*owner.shared).checkpoint;
            checkpoints.push_back(CheckpointProjection{
                .checkpoint = checkpoint,
                .state      = owner.shared->state,
            });
        }
        const bool evicted = owner.decision != nullptr && owner.decision->evicts_continuation;
        for (CheckpointProjection& checkpoint : checkpoints) {
            checkpoint.survives =
                !evicted &&
                !(owner.decision != nullptr &&
                  std::find(owner.decision->dropped_checkpoints.begin(),
                            owner.decision->dropped_checkpoints.end(), checkpoint.checkpoint.ref) !=
                      owner.decision->dropped_checkpoints.end());
        }
        std::sort(checkpoints.begin(), checkpoints.end(), [](const auto& left, const auto& right) {
            return std::tuple{left.checkpoint.ref.frontier, left.checkpoint.ref.kind,
                              left.checkpoint.ref.ordinal} <
                   std::tuple{right.checkpoint.ref.frontier, right.checkpoint.ref.kind,
                              right.checkpoint.ref.ordinal};
        });

        target_direct.resize(checkpoints.size());
        for (std::size_t index = 0; index < checkpoints.size(); ++index) {
            target_direct[index] = target_direct_work(*kv, checkpoints[index]);
            ++projection_work;
        }
        const auto append_target_recovery_work = [&](std::size_t selected) {
            const std::size_t offset               = alternatives.size();
            const CheckpointProjection& checkpoint = checkpoints[selected];
            alternatives.push_back(
                recovery_alternative_work({}, checkpoint.checkpoint.rebuild_work));
            if (target_direct[selected]) { alternatives.push_back(*target_direct[selected]); }
            for (std::size_t prior = 0; prior < selected; ++prior) {
                if (!target_direct[prior] || checkpoints[prior].checkpoint.ref.frontier >
                                                 checkpoint.checkpoint.ref.frontier) {
                    continue;
                }
                runtime::CheckpointRecoveryAlternativeWork alternative = *target_direct[prior];
                alternative.prefill                                    = interval_rebuild_work(
                    checkpoints[prior].checkpoint.ref.frontier,
                    checkpoints[prior].checkpoint.rebuild_work, checkpoint.checkpoint.ref.frontier,
                    checkpoint.checkpoint.rebuild_work, prefill_chunk);
                alternatives.push_back(alternative);
            }
            const std::size_t count = alternatives.size() - offset;
            if (offset > std::numeric_limits<std::uint32_t>::max() ||
                count > std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("pressure recovery work is not representable");
            }
            output.push_back(qwen3_5::detail::PressureCheckpointRecoveryProjection{
                .owner              = owner.owner,
                .checkpoint         = checkpoint.checkpoint.ref,
                .alternative_offset = static_cast<std::uint32_t>(offset),
                .alternative_count  = static_cast<std::uint32_t>(count),
                .survives           = checkpoint.survives,
            });
        };
        for (std::size_t index = 0; index < checkpoints.size(); ++index) {
            append_target_recovery_work(index);
            ++projection_work;
        }
    }
    return true;
}

std::optional<qwen3_5::detail::PressureDecision> ProgramImpl::inspect_checkpoint_drop_option(
    const SequenceState& sequence, std::span<const runtime::CheckpointRef> checkpoints) const {
    if (!sequence.kv || checkpoints.empty()) { return std::nullopt; }
    const qwen3_5::ContinuationSummary summary = continuation_summary(sequence);
    qwen3_5::detail::PressureDecision option;
    option.dropped_checkpoints.assign(checkpoints.begin(), checkpoints.end());
    std::sort(option.dropped_checkpoints.begin(), option.dropped_checkpoints.end(),
              [](runtime::CheckpointRef left, runtime::CheckpointRef right) {
                  return std::tuple{left.kind, left.frontier, left.ordinal} <
                         std::tuple{right.kind, right.frontier, right.ordinal};
              });
    if (std::adjacent_find(option.dropped_checkpoints.begin(), option.dropped_checkpoints.end()) !=
        option.dropped_checkpoints.end()) {
        return std::nullopt;
    }
    option.checkpoint_drops = planning_saturating_u32(option.dropped_checkpoints.size());

    struct DroppedState {
        runtime::CheckpointRef checkpoint;
        StateImageHandle state;
    };

    std::vector<DroppedState> dropped_states;
    dropped_states.reserve(option.dropped_checkpoints.size());
    const auto append_checkpoint = [&](runtime::CheckpointRef checkpoint) {
        if (summary.endpoint && summary.endpoint->ref == checkpoint) {
            dropped_states.push_back({.checkpoint = checkpoint, .state = sequence.state.read});
            return true;
        }
        if (summary.rewrite && summary.rewrite->ref == checkpoint && sequence.rewrite_state) {
            dropped_states.push_back({.checkpoint = checkpoint, .state = *sequence.rewrite_state});
            return true;
        }
        for (std::size_t index = 0; index < summary.long_anchors.size(); ++index) {
            if (summary.long_anchors[index].ref == checkpoint) {
                dropped_states.push_back(
                    {.checkpoint = checkpoint, .state = sequence.long_anchors[index].state});
                return true;
            }
        }
        return false;
    };
    for (const runtime::CheckpointRef checkpoint : option.dropped_checkpoints) {
        if (!append_checkpoint(checkpoint) || !state_store->valid(dropped_states.back().state)) {
            return std::nullopt;
        }
    }

    const std::optional<qwen3_5::TargetKVRequirement> remaining =
        retained_requirement_after_drops(summary, option.dropped_checkpoints);
    if (!remaining) { return std::nullopt; }

    std::vector<StateImageHandle> unique_states;
    unique_states.reserve(dropped_states.size());
    for (const DroppedState& dropped : dropped_states) {
        if (std::find(unique_states.begin(), unique_states.end(), dropped.state) ==
            unique_states.end()) {
            unique_states.push_back(dropped.state);
        }
    }
    const auto checkpoint_dropped = [&](runtime::CheckpointRef checkpoint) {
        return std::binary_search(option.dropped_checkpoints.begin(),
                                  option.dropped_checkpoints.end(), checkpoint,
                                  [](runtime::CheckpointRef left, runtime::CheckpointRef right) {
                                      return std::tuple{left.kind, left.frontier, left.ordinal} <
                                             std::tuple{right.kind, right.frontier, right.ordinal};
                                  });
    };
    for (const StateImageHandle state : unique_states) {
        bool survives = summary.endpoint && !checkpoint_dropped(summary.endpoint->ref) &&
                        sequence.state.read == state;
        survives = survives || (summary.rewrite && !checkpoint_dropped(summary.rewrite->ref) &&
                                sequence.rewrite_state && *sequence.rewrite_state == state);
        for (std::size_t index = 0; !survives && index < summary.long_anchors.size(); ++index) {
            survives = !checkpoint_dropped(summary.long_anchors[index].ref) &&
                       sequence.long_anchors[index].state == state;
        }
        if (survives || !state_exclusive_to_sequence(sequence, state)) { continue; }
        const StateReplicaResidency residency = state_store->residency(state);
        if (residency == StateReplicaResidency::DeviceOnly ||
            residency == StateReplicaResidency::Both) {
            ++option.effect.removed.device.state_slots;
        }
        if (residency == StateReplicaResidency::HostOnly ||
            residency == StateReplicaResidency::Both) {
            ++option.effect.removed.host.state_slots;
        }
    }

    const auto append_suffix_effect =
        [&](const KVAddressSpaceStore& addresses, const LogicalKVPageStore& pages,
            KVAddressSpaceHandle address, std::uint32_t retained_frontier,
            std::uint32_t& removed_pages) -> bool {
        if (!addresses.can_truncate_inactive_prefix(address, retained_frontier)) { return false; }
        const std::uint32_t retained_pages = kv_pages_for_frontier(retained_frontier);
        const std::uint32_t mapped         = addresses.mapped_pages(address);
        const std::size_t stride =
            plan_host_kv_page_layout(pages.physical_pool().geometry()).page_stride;
        for (std::uint32_t page = retained_pages; page < mapped; ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (pages.address_references(logical) != 1) { continue; }
            if (pages.device_resident(logical)) { ++removed_pages; }
            if (pages.host_resident(logical)) {
                if (stride >
                    std::numeric_limits<std::size_t>::max() - option.effect.removed.host.kv_bytes) {
                    throw std::overflow_error("checkpoint Host KV release size overflow");
                }
                option.effect.removed.host.kv_bytes += stride;
            }
        }
        return true;
    };
    if (!append_suffix_effect(*text_kv_addresses, *text_kv_pages, sequence.kv->text,
                              remaining->main_frontier,
                              option.effect.removed.device.main_kv_pages)) {
        return std::nullopt;
    }
    if (sequence.kv->backend &&
        !append_suffix_effect(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
                              remaining->backend_frontier,
                              option.effect.removed.device.backend_kv_pages)) {
        return std::nullopt;
    }

    std::uint64_t identity = 0x44524f5043484b50ULL;
    for (const runtime::CheckpointRef checkpoint : option.dropped_checkpoints) {
        identity ^= static_cast<std::uint64_t>(checkpoint.kind) << 56U;
        identity ^= static_cast<std::uint64_t>(checkpoint.frontier) << 16U;
        identity ^= checkpoint.ordinal;
        identity *= 1099511628211ULL;
    }
    option.id                     = identity == 0 ? 1 : identity;
    option.checkpoint_drop_effect = option.effect;
    return option;
}

void ProgramImpl::publish_checkpoint_drop(SequenceState& sequence,
                                          runtime::CheckpointRef checkpoint) {
    if (!sequence.kv) { throw std::logic_error("checkpoint drop owner has no KV bundle"); }
    const qwen3_5::ContinuationSummary before = continuation_summary(sequence);
    const std::optional<qwen3_5::TargetKVRequirement> retained =
        retained_requirement_after_drop(before, checkpoint);
    if (!retained ||
        !text_kv_addresses->can_truncate_inactive_prefix(sequence.kv->text,
                                                         retained->main_frontier) ||
        (sequence.kv->backend &&
         (!backend_kv_addresses || !backend_kv_addresses->can_truncate_inactive_prefix(
                                       *sequence.kv->backend, retained->backend_frontier)))) {
        throw std::logic_error("checkpoint drop release dependencies changed");
    }
    StateImageHandle dropped_state;
    if (checkpoint.kind == runtime::CheckpointKind::SessionEndpoint) {
        if (!sequence.endpoint_valid || sequence.execution_frontier != checkpoint.frontier) {
            throw std::logic_error("endpoint checkpoint changed before drop");
        }
        dropped_state              = sequence.state.read;
        sequence.endpoint_valid    = false;
        sequence.state             = {};
        sequence.tail_hidden       = {};
        sequence.tail_hidden_valid = false;
    } else if (checkpoint.kind == runtime::CheckpointKind::TurnClosure ||
               checkpoint.kind == runtime::CheckpointKind::ResponseReplay) {
        if (!sequence.rewrite_state || !sequence.rewrite_checkpoint.valid ||
            checkpoint_kind(sequence.rewrite_checkpoint.kind) != checkpoint.kind ||
            sequence.rewrite_checkpoint.frontier != checkpoint.frontier) {
            throw std::logic_error("rewrite checkpoint changed before drop");
        }
        dropped_state = *sequence.rewrite_state;
        state_store->release_checkpoint_reference(dropped_state);
        sequence.rewrite_state.reset();
        sequence.rewrite_checkpoint        = {};
        sequence.rewrite_checkpoint_hidden = {};
    } else if (checkpoint.kind == runtime::CheckpointKind::LongAnchor) {
        const auto anchor = std::find_if(sequence.long_anchors.begin(), sequence.long_anchors.end(),
                                         [&](const LongAnchorCheckpoint& candidate) {
                                             return candidate.frontier == checkpoint.frontier &&
                                                    candidate.ordinal == checkpoint.ordinal;
                                         });
        if (anchor == sequence.long_anchors.end()) {
            throw std::logic_error("long-anchor checkpoint changed before drop");
        }
        dropped_state = anchor->state;
        state_store->release_checkpoint_reference(dropped_state);
        sequence.long_anchors.erase(anchor);
    } else {
        throw std::logic_error("shared checkpoint cannot be dropped from a private owner");
    }

    bool retained_state = sequence.endpoint_valid && sequence.state.read == dropped_state;
    retained_state      = retained_state ||
                     (sequence.rewrite_state && *sequence.rewrite_state == dropped_state) ||
                     std::any_of(sequence.long_anchors.begin(), sequence.long_anchors.end(),
                                 [&](const LongAnchorCheckpoint& anchor) {
                                     return anchor.state == dropped_state;
                                 });
    if (!retained_state && state_store->checkpoint_references(dropped_state) == 0 &&
        !state_store->release(dropped_state)) {
        throw std::logic_error("dropped checkpoint StateImage remained pinned");
    }

    text_kv_addresses->truncate_inactive_prefix(sequence.kv->text, retained->main_frontier);
    text_kv_addresses->set_checkpoint_requirement(sequence.kv->text, retained->main_frontier);
    sequence.text_kv_valid = retained->main_frontier;
    if (sequence.kv->backend) {
        backend_kv_addresses->truncate_inactive_prefix(*sequence.kv->backend,
                                                       retained->backend_frontier);
        backend_kv_addresses->set_checkpoint_requirement(*sequence.kv->backend,
                                                         retained->backend_frontier);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) {
        sequence.mtp_kv_valid    = retained->backend_frontier;
        sequence.mtp_draft_count = 0;
    } else if (is_masked_draft_backend(speculative_backend)) {
        sequence.dflash_context_frontier = retained->main_frontier;
    }
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
    refresh_state_views(sequence);
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
ProgramImpl::checkpoint_recovery_work(const ContinuationHandle& owner,
                                      runtime::CheckpointRef checkpoint) const {
    if (!valid_continuation(owner)) {
        throw std::logic_error("checkpoint recovery owner is stale");
    }
    const SequenceState& sequence = continuation_states[ContractAccess::index(owner)];
    if (!sequence.kv) { throw std::logic_error("checkpoint recovery owner has no KV bundle"); }
    const qwen3_5::ContinuationSummary summary = continuation_summary(sequence);

    struct RecoverySource {
        const qwen3_5::CheckpointSummary* checkpoint = nullptr;
        StateImageHandle state;
    };

    std::vector<RecoverySource> sources;
    sources.reserve(summary.endpoint.has_value() + summary.rewrite.has_value() +
                    summary.long_anchors.size());
    if (summary.endpoint) {
        sources.push_back(
            RecoverySource{.checkpoint = &*summary.endpoint, .state = sequence.state.read});
    }
    if (summary.rewrite) {
        if (!sequence.rewrite_state) {
            throw std::logic_error("rewrite checkpoint has no StateImage");
        }
        sources.push_back(
            RecoverySource{.checkpoint = &*summary.rewrite, .state = *sequence.rewrite_state});
    }
    if (summary.long_anchors.size() != sequence.long_anchors.size()) {
        throw std::logic_error("long-anchor recovery profile is misaligned");
    }
    for (std::size_t index = 0; index < summary.long_anchors.size(); ++index) {
        sources.push_back(RecoverySource{.checkpoint = &summary.long_anchors[index],
                                         .state      = sequence.long_anchors[index].state});
    }
    const auto selected = std::find_if(sources.begin(), sources.end(), [&](const auto& source) {
        return source.checkpoint->ref == checkpoint;
    });
    if (selected == sources.end() || !state_store->valid(selected->state)) {
        throw std::logic_error("checkpoint recovery target is unavailable");
    }
    std::vector<runtime::CheckpointRecoveryAlternativeWork> alternatives;
    alternatives.reserve(sources.size() + 1U);
    alternatives.push_back(recovery_alternative_work({}, selected->checkpoint->rebuild_work));
    for (const RecoverySource& source : sources) {
        if (source.checkpoint->ref.frontier > selected->checkpoint->ref.frontier ||
            !state_store->valid(source.state)) {
            continue;
        }
        const runtime::PrefillWork interval = interval_rebuild_work(
            source.checkpoint->ref.frontier, source.checkpoint->rebuild_work,
            selected->checkpoint->ref.frontier, selected->checkpoint->rebuild_work, prefill_chunk);
        alternatives.push_back(recovery_alternative_work(
            checkpoint_restore_requirements(*sequence.kv, source.checkpoint->required_kv,
                                            source.state),
            interval));
    }
    return alternatives;
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
ProgramImpl::checkpoint_recovery_work(const SharedPrefixHandle& owner,
                                      runtime::CheckpointRef checkpoint) const {
    if (!valid_shared_prefix(owner)) {
        throw std::logic_error("shared checkpoint recovery owner is stale");
    }
    const SharedPrefixState& shared = shared_prefix_states[ContractAccess::index(owner)];
    if (!shared.kv) { throw std::logic_error("shared checkpoint recovery owner has no KV bundle"); }
    const qwen3_5::CheckpointSummary summary = shared_prefix_summary(shared).checkpoint;
    if (summary.ref != checkpoint || !state_store->valid(shared.state)) {
        throw std::logic_error("shared checkpoint recovery target is unavailable");
    }
    std::vector<runtime::CheckpointRecoveryAlternativeWork> alternatives;
    alternatives.reserve(2);
    alternatives.push_back(recovery_alternative_work({}, summary.rebuild_work));
    alternatives.push_back(recovery_alternative_work(
        checkpoint_restore_requirements(*shared.kv, summary.required_kv, shared.state)));
    return alternatives;
}


} // namespace ninfer::models::qwen3_5::detail
