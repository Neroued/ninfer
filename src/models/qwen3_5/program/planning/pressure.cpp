#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/planning/pressure_planner.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

namespace {

struct KVPressureSelection {
    std::vector<qwen3_5::detail::PressureKVDecision> actions;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::uint32_t removed_device_pages = 0;
    std::size_t removed_host_bytes     = 0;
    std::size_t added_host_bytes       = 0;
    std::size_t host_bytes_remaining   = 0;
};

std::uint32_t physical_kv_runs(const KVAddressSpaceStore& addresses,
                               const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                               std::uint32_t begin, std::uint32_t count);

void append_pressure_transfer(qwen3_5::detail::PressureDecision& option,
                              runtime::ContextTransferRequirement requirement);

qwen3_5::detail::PressureDecision
combine_checkpoint_and_replica_target(qwen3_5::detail::PressureDecision checkpoint,
                                      qwen3_5::detail::PressureDecision replica);

std::uint64_t
explicit_pressure_identity(const qwen3_5::detail::PressureDecision& decision) noexcept;

qwen3_5::detail::PressureDecision
explicit_pressure_target(const qwen3_5::detail::PressureDecision& complete);

bool logical_page_matches_prefix(const KVAddressSpaceStore& addresses,
                                 std::optional<KVAddressSpaceHandle> prefix,
                                 std::uint32_t prefix_pages, std::uint32_t page_offset,
                                 LogicalKVPageHandle page);

template <class ProtectedPage>
KVPressureSelection
select_kv_pressure_actions(const KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                           HostKVExtentStore* host_extents, bool host_allocation_available,
                           KVAddressSpaceHandle address, std::optional<std::uint32_t> mapped_limit,
                           std::uint32_t requested_device_pages, std::size_t requested_host_bytes,
                           runtime::ContextResourceClass resource,
                           std::span<const qwen3_5::detail::PressureKVDecision> existing_actions,
                           ProtectedPage&& protected_page);

detail::PhysicalResources pressure_residual(detail::PhysicalResources deficit,
                                            const detail::PhysicalDelta& applied);

std::uint32_t physical_kv_runs(const KVAddressSpaceStore& addresses,
                               const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                               std::uint32_t begin, std::uint32_t count) {
    if (count == 0) { return 0; }
    if (begin > addresses.mapped_pages(address) ||
        count > addresses.mapped_pages(address) - begin) {
        throw std::logic_error("physical KV run range is outside its address space");
    }
    std::vector<DeviceKVPageHandle> physical;
    physical.reserve(count);
    for (std::uint32_t offset = 0; offset < count; ++offset) {
        physical.push_back(pages.physical(addresses.logical_page(address, begin + offset)));
    }
    return pages.physical_pool().contiguous_run_count(physical);
}

void append_pressure_transfer(qwen3_5::detail::PressureDecision& option,
                              runtime::ContextTransferRequirement requirement) {
    if (requirement.units != 0) { option.transfer_requirements.push_back(std::move(requirement)); }
}

qwen3_5::detail::PressureDecision
combine_checkpoint_and_replica_target(qwen3_5::detail::PressureDecision checkpoint,
                                      qwen3_5::detail::PressureDecision replica) {
    if (checkpoint.dropped_checkpoints.empty() || !replica.dropped_checkpoints.empty() ||
        checkpoint.evicts_continuation || replica.evicts_continuation ||
        checkpoint.shared_owner != replica.shared_owner || !checkpoint.state_changes.empty() ||
        !checkpoint.main_kv_changes.empty() || !checkpoint.backend_kv_changes.empty()) {
        throw std::logic_error("pressure owner target composition is structurally invalid");
    }
    checkpoint.state_changes      = std::move(replica.state_changes);
    checkpoint.main_kv_changes    = std::move(replica.main_kv_changes);
    checkpoint.backend_kv_changes = std::move(replica.backend_kv_changes);
    checkpoint.effect.removed =
        checked_resource_sum(checkpoint.effect.removed, replica.effect.removed);
    checkpoint.effect.added = checked_resource_sum(checkpoint.effect.added, replica.effect.added);
    checkpoint.transfer_requirements.insert(checkpoint.transfer_requirements.end(),
                                            replica.transfer_requirements.begin(),
                                            replica.transfer_requirements.end());
    std::uint64_t identity = checkpoint.id ^ 0x434f4d42494e4544ULL;
    identity ^= replica.id + 0x9e3779b97f4a7c15ULL + (identity << 6U) + (identity >> 2U);
    checkpoint.id = identity == 0 ? 1 : identity;
    return checkpoint;
}

std::uint64_t
explicit_pressure_identity(const qwen3_5::detail::PressureDecision& decision) noexcept {
    std::uint64_t identity = decision.shared_owner ? 0x5348415245445052ULL : 0x5052495641544550ULL;
    const auto mix         = [&](std::uint64_t value) {
        identity ^= value;
        identity *= 1099511628211ULL;
    };
    for (const qwen3_5::detail::PressureStateDecision change : decision.state_changes) {
        mix(static_cast<std::uint8_t>(change));
    }
    const auto mix_kv = [&](std::span<const qwen3_5::detail::PressureKVDecision> changes,
                            std::uint64_t tag) {
        for (const qwen3_5::detail::PressureKVDecision& change : changes) {
            mix(tag);
            mix(change.begin_page);
            mix(change.page_count);
            mix(static_cast<std::uint8_t>(change.kind));
        }
    };
    mix_kv(decision.main_kv_changes, 0x4d41494eULL);
    mix_kv(decision.backend_kv_changes, 0x4241434bULL);
    return identity == 0 ? 1 : identity;
}

qwen3_5::detail::PressureDecision
explicit_pressure_target(const qwen3_5::detail::PressureDecision& complete) {
    qwen3_5::detail::PressureDecision explicit_target;
    explicit_target.state_changes      = complete.state_changes;
    explicit_target.main_kv_changes    = complete.main_kv_changes;
    explicit_target.backend_kv_changes = complete.backend_kv_changes;
    explicit_target.effect.removed     = checked_resource_difference(
        complete.effect.removed, complete.checkpoint_drop_effect.removed);
    explicit_target.effect.added =
        checked_resource_difference(complete.effect.added, complete.checkpoint_drop_effect.added);
    explicit_target.transfer_requirements = complete.transfer_requirements;
    explicit_target.shared_owner          = complete.shared_owner;
    explicit_target.id                    = explicit_pressure_identity(explicit_target);
    return explicit_target;
}

bool logical_page_matches_prefix(const KVAddressSpaceStore& addresses,
                                 std::optional<KVAddressSpaceHandle> prefix,
                                 std::uint32_t prefix_pages, std::uint32_t page_offset,
                                 LogicalKVPageHandle page) {
    if (!prefix || page_offset >= prefix_pages || !addresses.valid(*prefix) ||
        prefix_pages > addresses.mapped_pages(*prefix)) {
        return false;
    }
    return addresses.logical_page(*prefix, page_offset) == page;
}

template <class ProtectedPage>
KVPressureSelection
select_kv_pressure_actions(const KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                           HostKVExtentStore* host_extents, bool host_allocation_available,
                           KVAddressSpaceHandle address, std::optional<std::uint32_t> mapped_limit,
                           std::uint32_t requested_device_pages, std::size_t requested_host_bytes,
                           runtime::ContextResourceClass resource,
                           std::span<const qwen3_5::detail::PressureKVDecision> existing_actions,
                           ProtectedPage&& protected_page) {
    KVPressureSelection selection;
    selection.host_bytes_remaining = requested_host_bytes;
    const std::uint32_t mapped     = std::min(addresses.mapped_pages(address),
                                              mapped_limit.value_or(addresses.mapped_pages(address)));
    std::vector<std::uint8_t> selected(mapped, 0);
    const HostKVPageLayout layout = plan_host_kv_page_layout(pages.physical_pool().geometry());

    for (const qwen3_5::detail::PressureKVDecision& action : existing_actions) {
        if (action.kind == qwen3_5::detail::PressureKVDecisionKind::None ||
            action.page_count == 0 || action.begin_page > mapped ||
            action.page_count > mapped - action.begin_page) {
            throw std::logic_error("existing pressure KV action is outside the retained target");
        }
        for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
            const std::uint32_t page = action.begin_page + offset;
            if (selected[page] != 0) {
                throw std::logic_error("existing pressure KV actions overlap");
            }
            selected[page] = 1;
        }
    }

    const auto mark_action = [&](qwen3_5::detail::PressureKVDecision action) {
        for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
            const std::uint32_t page = action.begin_page + offset;
            if (page >= selected.size() || selected[page] != 0) {
                throw std::logic_error("pressure KV actions overlap");
            }
            selected[page] = 1;
        }
        selection.actions.push_back(action);
    };

    while (selection.host_bytes_remaining != 0 && host_extents != nullptr) {
        const std::size_t requested_pages =
            1U + (selection.host_bytes_remaining - 1U) / layout.page_stride;
        bool found        = false;
        std::uint32_t end = mapped;
        while (end != 0 && !found) {
            const auto eligible = [&](std::uint32_t page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                return selected[page] == 0 && pages.device_resident(logical) &&
                       pages.host_resident(logical) && pages.writer_references(logical) == 0 &&
                       pages.source_pins(logical) == 0 &&
                       !protected_page(page, logical,
                                       qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate);
            };
            while (end != 0 && !eligible(end - 1U)) { --end; }
            if (end == 0) { break; }
            std::uint32_t begin = end - 1U;
            while (begin != 0 && eligible(begin - 1U)) { --begin; }
            const std::uint32_t count =
                static_cast<std::uint32_t>(std::min<std::size_t>(end - begin, requested_pages));
            const std::uint32_t selected_begin = end - count;
            std::vector<LogicalKVPageHandle> releases;
            releases.reserve(count);
            for (std::uint32_t offset = 0; offset < count; ++offset) {
                releases.push_back(addresses.logical_page(address, selected_begin + offset));
            }
            if (!host_extents->can_release_page_replicas(pages, releases)) {
                end = begin;
                continue;
            }
            mark_action({
                .begin_page = selected_begin,
                .page_count = count,
                .kind       = qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate,
            });
            const std::size_t bytes = layout.page_stride * static_cast<std::size_t>(count);
            if (bytes > std::numeric_limits<std::size_t>::max() - selection.removed_host_bytes) {
                throw std::overflow_error("pressure Host KV release size overflow");
            }
            selection.removed_host_bytes += bytes;
            selection.host_bytes_remaining = bytes >= selection.host_bytes_remaining
                                                 ? 0
                                                 : selection.host_bytes_remaining - bytes;
            found                          = true;
        }
        if (!found) { break; }
    }

    std::uint32_t device_remaining = requested_device_pages;
    const auto select_device_runs  = [&](bool require_host) {
        std::uint32_t end = mapped;
        while (device_remaining != 0 && end != 0) {
            const auto eligible = [&](std::uint32_t page) {
                if (selected[page] != 0) { return false; }
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                const bool replica_safe = require_host ? pages.can_drop_device_replica(logical)
                                                        : !pages.host_resident(logical);
                const qwen3_5::detail::PressureKVDecisionKind action =
                    require_host ? qwen3_5::detail::PressureKVDecisionKind::DropDeviceDuplicate
                                  : qwen3_5::detail::PressureKVDecisionKind::DemoteToHost;
                return pages.device_resident(logical) && pages.writer_references(logical) == 0 &&
                       pages.source_pins(logical) == 0 && replica_safe &&
                       !addresses.has_active_reference(logical) &&
                       !protected_page(page, logical, action);
            };
            while (end != 0 && !eligible(end - 1U)) { --end; }
            if (end == 0) { break; }
            std::uint32_t begin = end - 1U;
            while (begin != 0 && eligible(begin - 1U)) { --begin; }
            const std::uint32_t count = std::min(device_remaining, end - begin);
            qwen3_5::detail::PressureKVDecision action{
                 .begin_page = end - count,
                 .page_count = count,
                 .kind = require_host ? qwen3_5::detail::PressureKVDecisionKind::DropDeviceDuplicate
                                      : qwen3_5::detail::PressureKVDecisionKind::DemoteToHost,
            };
            mark_action(action);
            selection.removed_device_pages += count;
            device_remaining -= count;
            end = action.begin_page;

            if (!require_host) {
                if (count != 0 &&
                    layout.page_stride > std::numeric_limits<std::size_t>::max() / count) {
                    throw std::overflow_error("pressure Host KV extent size overflow");
                }
                const std::size_t bytes = layout.page_stride * static_cast<std::size_t>(count);
                if (bytes > std::numeric_limits<std::size_t>::max() - selection.added_host_bytes) {
                    throw std::overflow_error("pressure KV transfer size overflow");
                }
                selection.added_host_bytes += bytes;
                selection.transfer_requirements.push_back(kv_transfer_requirement(
                    resource, runtime::ContextTransferDirection::DeviceToHost, layout, count,
                    physical_kv_runs(addresses, pages, address, action.begin_page,
                                      action.page_count)));
            }
        }
    };
    select_device_runs(true);
    if (device_remaining != 0 && selection.host_bytes_remaining == 0 && host_allocation_available &&
        host_extents != nullptr) {
        select_device_runs(false);
    }
    return selection;
}

detail::PhysicalResources pressure_residual(detail::PhysicalResources deficit,
                                            const detail::PhysicalDelta& applied) {
    return positive_resource_difference(checked_resource_sum(deficit, applied.added),
                                        applied.removed);
}

} // namespace

std::optional<AdmissionCandidate> ProgramImpl::inspect_admission(
    const PreparedPromptData& prompt, const RequestBasePlan& base, runtime::LaneId destination,
    const ContinuationHandle* source, const SharedPrefixHandle* shared_source,
    std::optional<runtime::CheckpointRef> checkpoint, bool must_retain_private_source) {
    const std::uint32_t lane = destination.value;
    if (lane >= max_concurrency) { throw std::out_of_range("admission lane is out of range"); }
    if (requests[lane].lifecycle != Lifecycle::Empty ||
        active_continuations[lane] < continuation_capacity) {
        throw std::logic_error("admission destination is active");
    }
    if ((source != nullptr && shared_source != nullptr) ||
        ((source == nullptr && shared_source == nullptr) != !checkpoint.has_value())) {
        throw std::invalid_argument("admission source and checkpoint must be specified together");
    }
    const SequenceState* source_state = nullptr;
    if (source != nullptr) {
        if (!valid_continuation(*source)) {
            throw std::logic_error("admission source continuation is stale");
        }
        source_state = &continuation_states[ContractAccess::index(*source)];
    }
    const SharedPrefixState* shared_state = nullptr;
    if (shared_source != nullptr) {
        if (!valid_shared_prefix(*shared_source)) {
            throw std::logic_error("admission shared-prefix source is stale");
        }
        shared_state = &shared_prefix_states[ContractAccess::index(*shared_source)];
    }

    std::optional<AdmissionCandidate> plan = inspect_lane(
        lane, prompt, base, source_state, shared_state, checkpoint, must_retain_private_source);
    if (!plan) { return std::nullopt; }
    plan->impl_->destination       = destination;
    plan->impl_->destination_epoch = lane_epochs[lane];
    plan->impl_->has_source        = source != nullptr;
    plan->impl_->has_shared_source = shared_source != nullptr;
    plan->impl_->source_index      = source != nullptr ? ContractAccess::index(*source) : 0;
    plan->impl_->source_generation = source != nullptr ? ContractAccess::epoch(*source) : 0;
    plan->impl_->shared_source_index =
        shared_source != nullptr ? ContractAccess::index(*shared_source) : 0;
    plan->impl_->shared_source_generation =
        shared_source != nullptr ? ContractAccess::epoch(*shared_source) : 0;
    plan->impl_->planning_revision         = resource_revision_;
    plan->impl_->identity_pressure_deficit = materialization_deficit(*plan->impl_);
    plan->impl_->identity_assessment.machine_work =
        materialization_machine_work(*plan->impl_, {}, {});
    const runtime::PreflightStatus identity_status = revalidate_materialization(*plan, prompt);
    if (identity_status == runtime::PreflightStatus::InvariantFailure) {
        throw std::logic_error("identity materialization assessment is internally invalid");
    }
    plan->impl_->identity_assessment.physical_status =
        identity_status == runtime::PreflightStatus::Ready
            ? runtime::MaterializationPhysicalStatus::Feasible
            : runtime::MaterializationPhysicalStatus::Infeasible;
    plan->impl_->identity_assessment.source_mode = plan->impl_->source_mode;
    plan->impl_->identity_assessment.pressure_may_change_machine_work =
        plan->impl_->has_source &&
        plan->impl_->source_mode == runtime::PrivateSourceMode::ConsumeToActive &&
        std::any_of(
            plan->impl_->transfer_requirements.begin(), plan->impl_->transfer_requirements.end(),
            [](const runtime::ContextTransferRequirement& requirement) {
                return requirement.direction == runtime::ContextTransferDirection::DeviceToDevice;
            });
    plan->impl_->identity_assessment.expandable =
        identity_status != runtime::PreflightStatus::Ready;
    plan->impl_->identity_assessment.projection_work =
        1U + plan->impl_->transfer_requirements.size();
    std::uint64_t digest = 1469598103934665603ULL;
    const auto mix       = [&](std::uint64_t value) {
        digest ^= value;
        digest *= 1099511628211ULL;
    };
    mix(resource_revision_.value);
    mix(plan->impl_->summary.reusable_prompt_tokens);
    const runtime::MaterializationMachineWork& identity_work =
        plan->impl_->identity_assessment.machine_work;
    mix(identity_work.remaining_prefill_work.chunks);
    mix(identity_work.remaining_prefill_work.tokens);
    mix(identity_work.remaining_prefill_work.attention_pairs);
    mix(identity_work.remaining_prefill_work.vision_items);
    mix(identity_work.remaining_prefill_work.vision_patches);
    for (const TransferWork transfer : identity_work.candidate_transfers) {
        mix(transfer.payload_bytes);
        mix(transfer.copy_operations);
    }
    mix(static_cast<std::uint8_t>(plan->impl_->identity_assessment.physical_status));
    plan->impl_->identity_assessment.assessment_digest = digest;
    return plan;
}

std::optional<ProgramImpl::MaterializationSourceProtection>
ProgramImpl::materialization_source_protection(const ResourceCandidateState& admission) const {
    if (admission.has_source && admission.has_shared_source) { return std::nullopt; }

    MaterializationSourceProtection protection;
    const SequenceKVBundle* kv = nullptr;
    if (admission.has_source) {
        if (admission.source_index >= continuation_capacity ||
            continuation_slots[admission.source_index].role != ContinuationSlotRole::Catalogued ||
            continuation_slots[admission.source_index].generation != admission.source_generation) {
            return std::nullopt;
        }
        const SequenceState& source = continuation_states[admission.source_index];
        if (!source.kv) { return std::nullopt; }
        kv                              = &*source.kv;
        protection.private_source_index = admission.source_index;
        protection.state = selected_state(source, admission.reuse, admission.selected_checkpoint);
        if (admission.source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
            protection.consumed_private_source   = true;
            protection.consumed_state_references = selected_state_consumed_references(
                source, admission.reuse, admission.rewrite_disposition,
                admission.selected_checkpoint, admission.reuse_base);
            // Protection describes the stable state from which a complete pressure target is
            // projected.  The sealed candidate may already contain the target-derived Move/Fork
            // result, so do not read that derived result back as the pre-pressure fact.
            protection.state_fork_required =
                state_store->checkpoint_references(*protection.state) !=
                protection.consumed_state_references;

            if (is_rewrite_checkpoint_restore(admission.reuse)) {
                const auto append_optional_state = [&](StateImageHandle state) {
                    if (!state_store->valid(state) || state_exclusive_to_sequence(source, state) ||
                        std::any_of(
                            protection.state_ownership_candidates.begin(),
                            protection.state_ownership_candidates.end(),
                            [&](const auto& candidate) { return candidate.state == state; })) {
                        return;
                    }
                    protection.state_ownership_candidates.push_back({
                        .state                        = state,
                        .source_checkpoint_references = owned_checkpoint_references(source, state),
                    });
                };
                if (admission.rewrite_disposition == RewriteCheckpointDisposition::RetainExisting &&
                    source.rewrite_state) {
                    append_optional_state(*source.rewrite_state);
                }
                for (const LongAnchorCheckpoint& anchor : source.long_anchors) {
                    if (anchor.frontier <= admission.reuse_base) {
                        append_optional_state(anchor.state);
                    }
                }
            }
        }
    } else if (admission.has_shared_source) {
        if (admission.shared_source_index >= shared_prefix_capacity ||
            shared_prefix_slots[admission.shared_source_index].role !=
                SharedPrefixSlotRole::Catalogued ||
            shared_prefix_slots[admission.shared_source_index].generation !=
                admission.shared_source_generation) {
            return std::nullopt;
        }
        const SharedPrefixState& source = shared_prefix_states[admission.shared_source_index];
        if (!source.kv) { return std::nullopt; }
        kv               = &*source.kv;
        protection.state = source.state;
    }
    if (kv == nullptr) { return protection; }

    protection.text       = kv->text;
    protection.text_pages = kv_pages_for_frontier(admission.reuse_base);
    if (protection.consumed_private_source) {
        protection.text_transfer_pages =
            admission.reuse_base / static_cast<std::uint32_t>(kPagedKVPageSize);
    }
    if (!text_kv_addresses->valid(kv->text) ||
        protection.text_pages > text_kv_addresses->mapped_pages(kv->text)) {
        return std::nullopt;
    }
    if (protection.consumed_private_source) {
        protection.text_prefix_fork_required =
            partial_tail_cow_required(*text_kv_addresses, kv->text, admission.reuse_base);
    }
    const std::uint32_t backend_frontier =
        backend_frontier_at(speculative_backend, admission.reuse_base);
    protection.backend_pages = kv_pages_for_frontier(backend_frontier);
    if (protection.consumed_private_source) {
        protection.backend_transfer_pages =
            backend_frontier / static_cast<std::uint32_t>(kPagedKVPageSize);
    }
    if (protection.backend_pages != 0) {
        if (!kv->backend || !backend_kv_addresses || !backend_kv_addresses->valid(*kv->backend) ||
            protection.backend_pages > backend_kv_addresses->mapped_pages(*kv->backend)) {
            return std::nullopt;
        }
        protection.backend = *kv->backend;
        if (protection.consumed_private_source) {
            protection.backend_prefix_fork_required =
                partial_tail_cow_required(*backend_kv_addresses, *kv->backend, backend_frontier);
        }
    }
    return protection;
}

bool ProgramImpl::protected_materialization_page(const MaterializationSourceProtection* protection,
                                                 const KVAddressSpaceStore& addresses,
                                                 std::uint32_t page_offset,
                                                 LogicalKVPageHandle page, bool backend) const {
    if (protection == nullptr) { return false; }
    const std::optional<KVAddressSpaceHandle>& source =
        backend ? protection->backend : protection->text;
    const std::uint32_t required = backend ? protection->backend_pages : protection->text_pages;
    return logical_page_matches_prefix(addresses, source, required, page_offset, page);
}

std::optional<qwen3_5::detail::PressureDecision> ProgramImpl::inspect_shared_pressure_option(
    const SharedPrefixState& shared, detail::PhysicalResources deficit,
    const MaterializationSourceProtection* protection,
    const qwen3_5::detail::PressureDecision* current) const {
    if (!shared.kv || shared.active_references != 0 || deficit.device.active_lanes != 0 ||
        (current != nullptr && (current->evicts_continuation || !current->shared_owner))) {
        return std::nullopt;
    }

    qwen3_5::detail::PressureDecision option;
    if (current != nullptr) { option = *current; }
    option.shared_owner               = true;
    const std::size_t initial_actions = option.state_changes.size() +
                                        option.main_kv_changes.size() +
                                        option.backend_kv_changes.size();
    std::uint64_t identity = 1099511628211ULL;
    const auto mix         = [&](std::uint64_t value) {
        identity ^= value;
        identity *= 1469598103934665603ULL;
    };
    for (const qwen3_5::detail::PressureStateDecision change : option.state_changes) {
        mix(static_cast<std::uint8_t>(change));
    }
    const auto mix_existing_kv = [&](std::span<const qwen3_5::detail::PressureKVDecision> actions,
                                     std::uint64_t tag) {
        for (const qwen3_5::detail::PressureKVDecision& action : actions) {
            mix(tag);
            mix(action.begin_page);
            mix(action.page_count);
            mix(static_cast<std::uint8_t>(action.kind));
        }
    };
    mix_existing_kv(option.main_kv_changes, 0x534d41494eULL);
    mix_existing_kv(option.backend_kv_changes, 0x534241434bULL);

    qwen3_5::detail::PressureStateDecision state_change =
        qwen3_5::detail::PressureStateDecision::None;
    if (option.state_changes.empty() &&
        (deficit.device.state_slots != 0 || deficit.host.state_slots != 0) &&
        state_store->valid(shared.state) &&
        state_store->role(shared.state) == StateImageRole::CheckpointImmutable &&
        state_store->source_pins(shared.state) == 0) {
        const StateReplicaResidency residency = state_store->residency(shared.state);
        const bool protected_state =
            protection != nullptr && protection->state && *protection->state == shared.state;
        if (deficit.host.state_slots != 0 && residency == StateReplicaResidency::Both) {
            state_change = qwen3_5::detail::PressureStateDecision::DropSharedHostDuplicate;
            ++option.effect.removed.host.state_slots;
        } else if (!protected_state && state_store->checkpoint_references(shared.state) == 1 &&
                   deficit.device.state_slots != 0 && residency == StateReplicaResidency::Both) {
            state_change = qwen3_5::detail::PressureStateDecision::DropSharedDeviceDuplicate;
        } else if (!protected_state && state_store->checkpoint_references(shared.state) == 1 &&
                   deficit.device.state_slots != 0 &&
                   residency == StateReplicaResidency::DeviceOnly && host_state_images != nullptr) {
            state_change = qwen3_5::detail::PressureStateDecision::DemoteSharedToHost;
            ++option.effect.added.host.state_slots;
            append_pressure_transfer(option, state_transfer_requirement(
                                                 host_state_images->layout(),
                                                 runtime::ContextTransferDirection::DeviceToHost));
        }
        if (state_change != qwen3_5::detail::PressureStateDecision::None) {
            if (state_change != qwen3_5::detail::PressureStateDecision::DropSharedHostDuplicate) {
                option.effect.removed.device.state_slots = 1;
            }
            option.state_changes.push_back(state_change);
            mix(static_cast<std::uint8_t>(state_change));
        }
    }

    std::size_t host_kv_remaining = deficit.host.kv_bytes;
    const auto add_kv = [&](const KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                            KVAddressSpaceHandle address, std::uint32_t requested,
                            std::vector<qwen3_5::detail::PressureKVDecision>& changes,
                            std::uint32_t& removed_dimension,
                            runtime::ContextResourceClass resource, std::uint64_t tag) {
        const bool backend           = resource == runtime::ContextResourceClass::BackendKV;
        KVPressureSelection selected = select_kv_pressure_actions(
            addresses, pages, host_kv_extents.get(),
            host_kv_arena != nullptr && host_kv_extents != nullptr, address, std::nullopt,
            requested, host_kv_remaining, resource, changes,
            [&](std::uint32_t page, LogicalKVPageHandle logical,
                qwen3_5::detail::PressureKVDecisionKind action) {
                return action != qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate &&
                       protected_materialization_page(protection, addresses, page, logical,
                                                      backend);
            });
        host_kv_remaining = selected.host_bytes_remaining;
        option.effect.removed.host.kv_bytes += selected.removed_host_bytes;
        option.effect.added.host.kv_bytes += selected.added_host_bytes;
        removed_dimension += selected.removed_device_pages;
        option.transfer_requirements.insert(option.transfer_requirements.end(),
                                            selected.transfer_requirements.begin(),
                                            selected.transfer_requirements.end());
        for (const qwen3_5::detail::PressureKVDecision& action : selected.actions) {
            mix(tag);
            mix(action.begin_page);
            mix(action.page_count);
            mix(static_cast<std::uint8_t>(action.kind));
            changes.push_back(action);
        }
    };

    add_kv(*text_kv_addresses, *text_kv_pages, shared.kv->text, deficit.device.main_kv_pages,
           option.main_kv_changes, option.effect.removed.device.main_kv_pages,
           runtime::ContextResourceClass::MainKV, 0x534d41494eULL);
    if (shared.kv->backend && backend_kv_addresses && backend_kv_pages) {
        add_kv(*backend_kv_addresses, *backend_kv_pages, *shared.kv->backend,
               deficit.device.backend_kv_pages, option.backend_kv_changes,
               option.effect.removed.device.backend_kv_pages,
               runtime::ContextResourceClass::BackendKV, 0x534241434bULL);
    }
    if (option.state_changes.size() + option.main_kv_changes.size() +
            option.backend_kv_changes.size() ==
        initial_actions) {
        return std::nullopt;
    }
    option.id = identity == 0 ? 1 : identity;
    return option;
}

std::vector<qwen3_5::detail::PressureDecision> ProgramImpl::inspect_shared_pressure_options(
    const SharedPrefixState& shared, detail::PhysicalResources deficit,
    const MaterializationSourceProtection* protection,
    const qwen3_5::detail::PressureDecision* current) const {
    std::vector<detail::PhysicalResources> endpoints;
    endpoints.reserve(8);
    const auto endpoint = [&](detail::PhysicalResources value) {
        if (value != detail::PhysicalResources{} &&
            std::find(endpoints.begin(), endpoints.end(), value) == endpoints.end()) {
            endpoints.push_back(value);
        }
    };
    endpoint(deficit);
    detail::PhysicalResources device_only;
    device_only.device = deficit.device;
    endpoint(device_only);
    if (deficit.device.state_slots != 0) {
        detail::PhysicalResources state;
        state.device.state_slots = 1;
        endpoint(state);
    }
    if (deficit.host.state_slots != 0) {
        detail::PhysicalResources state;
        state.host.state_slots = 1;
        endpoint(state);
    }
    if (deficit.device.main_kv_pages != 0) {
        detail::PhysicalResources exact;
        exact.device.main_kv_pages = deficit.device.main_kv_pages;
        endpoint(exact);
        detail::PhysicalResources full;
        full.device.main_kv_pages = std::numeric_limits<std::uint32_t>::max();
        endpoint(full);
    }
    if (deficit.device.backend_kv_pages != 0) {
        detail::PhysicalResources exact;
        exact.device.backend_kv_pages = deficit.device.backend_kv_pages;
        endpoint(exact);
        detail::PhysicalResources full;
        full.device.backend_kv_pages = std::numeric_limits<std::uint32_t>::max();
        endpoint(full);
    }
    if (deficit.host.kv_bytes != 0) {
        detail::PhysicalResources host;
        host.host.kv_bytes = deficit.host.kv_bytes;
        endpoint(host);
        detail::PhysicalResources full_host;
        full_host.host.kv_bytes = std::numeric_limits<std::size_t>::max();
        endpoint(full_host);
    }

    std::vector<qwen3_5::detail::PressureDecision> options;
    options.reserve(endpoints.size());
    for (const detail::PhysicalResources requested : endpoints) {
        std::optional<qwen3_5::detail::PressureDecision> option =
            inspect_shared_pressure_option(shared, requested, protection, current);
        if (option && std::find(options.begin(), options.end(), *option) == options.end()) {
            options.push_back(std::move(*option));
        }
    }
    return options;
}

std::optional<qwen3_5::detail::PressureDecision>
ProgramImpl::inspect_pressure_option(const SequenceState& sequence,
                                     detail::PhysicalResources deficit,
                                     const MaterializationSourceProtection* protection,
                                     const qwen3_5::TargetKVRequirement* retained_requirement,
                                     std::span<const runtime::CheckpointRef> dropped_checkpoints,
                                     std::span<const StateImageHandle> released_states,
                                     const qwen3_5::detail::PressureDecision* current) const {
    if (!sequence.kv || deficit.device.active_lanes != 0 ||
        (current != nullptr && current->evicts_continuation)) {
        return std::nullopt;
    }

    qwen3_5::detail::PressureDecision option;
    if (current != nullptr) {
        option.state_changes      = current->state_changes;
        option.main_kv_changes    = current->main_kv_changes;
        option.backend_kv_changes = current->backend_kv_changes;
        option.effect.removed     = checked_resource_difference(
            current->effect.removed, current->checkpoint_drop_effect.removed);
        option.effect.added          = checked_resource_difference(current->effect.added,
                                                                   current->checkpoint_drop_effect.added);
        option.transfer_requirements = current->transfer_requirements;
    }
    const std::size_t initial_actions = option.state_changes.size() +
                                        option.main_kv_changes.size() +
                                        option.backend_kv_changes.size();
    const detail::PhysicalDelta initial_effect = option.effect;
    std::uint64_t identity                     = 1469598103934665603ULL;
    const auto mix                             = [&](std::uint64_t value) {
        identity ^= value;
        identity *= 1099511628211ULL;
    };
    for (const qwen3_5::detail::PressureStateDecision change : option.state_changes) {
        mix(static_cast<std::uint8_t>(change));
    }
    const auto mix_kv = [&](std::span<const qwen3_5::detail::PressureKVDecision> actions,
                            std::uint64_t tag) {
        for (const qwen3_5::detail::PressureKVDecision& action : actions) {
            mix(tag);
            mix(action.begin_page);
            mix(action.page_count);
            mix(static_cast<std::uint8_t>(action.kind));
        }
    };
    mix_kv(option.main_kv_changes, 0x4d41494eULL);
    mix_kv(option.backend_kv_changes, 0x4241434bULL);
    const auto add_state = [&](StateImageHandle state, bool rewrite) {
        const bool checkpoint_was_dropped = std::any_of(
            dropped_checkpoints.begin(), dropped_checkpoints.end(),
            [&](runtime::CheckpointRef checkpoint) {
                return rewrite ? (checkpoint.kind == runtime::CheckpointKind::TurnClosure ||
                                  checkpoint.kind == runtime::CheckpointKind::ResponseReplay)
                               : checkpoint.kind == runtime::CheckpointKind::SessionEndpoint;
            });
        const qwen3_5::detail::PressureStateDecision endpoint_drop =
            rewrite ? qwen3_5::detail::PressureStateDecision::DropRewriteDeviceDuplicate
                    : qwen3_5::detail::PressureStateDecision::DropEndpointDeviceDuplicate;
        const qwen3_5::detail::PressureStateDecision endpoint_demote =
            rewrite ? qwen3_5::detail::PressureStateDecision::DemoteRewriteToHost
                    : qwen3_5::detail::PressureStateDecision::DemoteEndpointToHost;
        const qwen3_5::detail::PressureStateDecision endpoint_host_drop =
            rewrite ? qwen3_5::detail::PressureStateDecision::DropRewriteHostDuplicate
                    : qwen3_5::detail::PressureStateDecision::DropEndpointHostDuplicate;
        const bool already_changed =
            std::find(option.state_changes.begin(), option.state_changes.end(), endpoint_drop) !=
                option.state_changes.end() ||
            std::find(option.state_changes.begin(), option.state_changes.end(), endpoint_demote) !=
                option.state_changes.end() ||
            std::find(option.state_changes.begin(), option.state_changes.end(),
                      endpoint_host_drop) != option.state_changes.end();
        const detail::PhysicalDelta extension_effect{
            .removed = checked_resource_difference(option.effect.removed, initial_effect.removed),
            .added   = checked_resource_difference(option.effect.added, initial_effect.added),
        };
        const detail::PhysicalResources residual = pressure_residual(deficit, extension_effect);
        if ((residual.device.state_slots == 0 && residual.host.state_slots == 0) ||
            checkpoint_was_dropped || already_changed || !state_store->valid(state) ||
            state_store->role(state) != StateImageRole::CheckpointImmutable ||
            state_store->source_pins(state) != 0 ||
            std::find(released_states.begin(), released_states.end(), state) !=
                released_states.end()) {
            return false;
        }
        const StateReplicaResidency residency = state_store->residency(state);
        const bool protected_state =
            protection != nullptr && protection->state && *protection->state == state;
        qwen3_5::detail::PressureStateDecision change =
            qwen3_5::detail::PressureStateDecision::None;
        if (residual.host.state_slots != 0 && residency == StateReplicaResidency::Both) {
            change = endpoint_host_drop;
            ++option.effect.removed.host.state_slots;
        } else if (!protected_state && state_exclusive_to_sequence(sequence, state) &&
                   residual.device.state_slots != 0 && residency == StateReplicaResidency::Both) {
            change = endpoint_drop;
        } else if (!protected_state && state_exclusive_to_sequence(sequence, state) &&
                   residual.device.state_slots != 0 &&
                   residency == StateReplicaResidency::DeviceOnly && host_state_images != nullptr) {
            change = endpoint_demote;
            ++option.effect.added.host.state_slots;
            append_pressure_transfer(option, state_transfer_requirement(
                                                 host_state_images->layout(),
                                                 runtime::ContextTransferDirection::DeviceToHost));
        } else {
            return false;
        }
        const bool drops_host =
            change == qwen3_5::detail::PressureStateDecision::DropEndpointHostDuplicate ||
            change == qwen3_5::detail::PressureStateDecision::DropRewriteHostDuplicate;
        if (!drops_host) { ++option.effect.removed.device.state_slots; }
        option.state_changes.push_back(change);
        mix(static_cast<std::uint8_t>(change));
        return true;
    };

    // A typed rewrite is the less destructive state relief while the endpoint remains usable.
    if (sequence.rewrite_state) { (void)add_state(*sequence.rewrite_state, true); }
    (void)add_state(sequence.state.read, false);

    std::size_t host_kv_remaining = deficit.host.kv_bytes;
    const auto add_kv = [&](const KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                            KVAddressSpaceHandle address, std::uint32_t requested,
                            std::optional<std::uint32_t> mapped_limit,
                            std::vector<qwen3_5::detail::PressureKVDecision>& changes,
                            std::uint32_t& removed_dimension,
                            runtime::ContextResourceClass resource, std::uint64_t tag) {
        const bool backend           = resource == runtime::ContextResourceClass::BackendKV;
        KVPressureSelection selected = select_kv_pressure_actions(
            addresses, pages, host_kv_extents.get(),
            host_kv_arena != nullptr && host_kv_extents != nullptr, address, mapped_limit,
            requested, host_kv_remaining, resource, changes,
            [&](std::uint32_t page, LogicalKVPageHandle logical,
                qwen3_5::detail::PressureKVDecisionKind action) {
                return action != qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate &&
                       protected_materialization_page(protection, addresses, page, logical,
                                                      backend);
            });
        host_kv_remaining = selected.host_bytes_remaining;
        option.effect.removed.host.kv_bytes += selected.removed_host_bytes;
        option.effect.added.host.kv_bytes += selected.added_host_bytes;
        removed_dimension += selected.removed_device_pages;
        option.transfer_requirements.insert(option.transfer_requirements.end(),
                                            selected.transfer_requirements.begin(),
                                            selected.transfer_requirements.end());
        for (const qwen3_5::detail::PressureKVDecision& action : selected.actions) {
            mix(tag);
            mix(action.begin_page);
            mix(action.page_count);
            mix(static_cast<std::uint8_t>(action.kind));
            changes.push_back(action);
        }
    };

    add_kv(*text_kv_addresses, *text_kv_pages, sequence.kv->text, deficit.device.main_kv_pages,
           retained_requirement ? std::optional<std::uint32_t>(retained_requirement->main_pages)
                                : std::nullopt,
           option.main_kv_changes, option.effect.removed.device.main_kv_pages,
           runtime::ContextResourceClass::MainKV, 0x4d41494eULL);
    if (sequence.kv->backend && backend_kv_addresses && backend_kv_pages) {
        add_kv(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
               deficit.device.backend_kv_pages,
               retained_requirement
                   ? std::optional<std::uint32_t>(retained_requirement->backend_pages)
                   : std::nullopt,
               option.backend_kv_changes, option.effect.removed.device.backend_kv_pages,
               runtime::ContextResourceClass::BackendKV, 0x4241434bULL);
    }
    if (option.state_changes.size() + option.main_kv_changes.size() +
            option.backend_kv_changes.size() ==
        initial_actions) {
        return std::nullopt;
    }
    option.id = identity == 0 ? 1 : identity;
    return option;
}

std::vector<qwen3_5::detail::PressureDecision>
ProgramImpl::inspect_pressure_successors(const SequenceState& sequence,
                                         detail::PhysicalResources residual,
                                         const MaterializationSourceProtection* protection,
                                         const qwen3_5::detail::PressureDecision* current) const {
    std::vector<qwen3_5::detail::PressureDecision> successors;
    if (!sequence.kv || (current != nullptr && current->evicts_continuation)) { return successors; }
    const auto append_unique = [&](qwen3_5::detail::PressureDecision option) {
        if (current != nullptr && option == *current) { return; }
        if (std::find(successors.begin(), successors.end(), option) == successors.end()) {
            successors.push_back(std::move(option));
        }
    };

    const qwen3_5::ContinuationSummary summary = continuation_summary(sequence);
    std::vector<runtime::CheckpointRef> dropped =
        current != nullptr ? current->dropped_checkpoints : std::vector<runtime::CheckpointRef>{};
    const auto already_dropped = [&](runtime::CheckpointRef checkpoint) {
        return std::find(dropped.begin(), dropped.end(), checkpoint) != dropped.end();
    };
    const auto state_change_conflicts =
        [](runtime::CheckpointRef checkpoint,
           const qwen3_5::detail::PressureDecision& explicit_target) {
            const auto has = [&](qwen3_5::detail::PressureStateDecision change) {
                return std::find(explicit_target.state_changes.begin(),
                                 explicit_target.state_changes.end(),
                                 change) != explicit_target.state_changes.end();
            };
            if (checkpoint.kind == runtime::CheckpointKind::SessionEndpoint) {
                return has(qwen3_5::detail::PressureStateDecision::DropEndpointDeviceDuplicate) ||
                       has(qwen3_5::detail::PressureStateDecision::DemoteEndpointToHost) ||
                       has(qwen3_5::detail::PressureStateDecision::DropEndpointHostDuplicate);
            }
            if (checkpoint.kind == runtime::CheckpointKind::TurnClosure ||
                checkpoint.kind == runtime::CheckpointKind::ResponseReplay) {
                return has(qwen3_5::detail::PressureStateDecision::DropRewriteDeviceDuplicate) ||
                       has(qwen3_5::detail::PressureStateDecision::DemoteRewriteToHost) ||
                       has(qwen3_5::detail::PressureStateDecision::DropRewriteHostDuplicate);
            }
            return false;
        };
    const auto append_checkpoint_successor = [&](runtime::CheckpointRef checkpoint) {
        if (already_dropped(checkpoint)) { return; }
        std::vector<runtime::CheckpointRef> target_drops = dropped;
        target_drops.push_back(checkpoint);
        std::optional<qwen3_5::detail::PressureDecision> drop =
            inspect_checkpoint_drop_option(sequence, target_drops);
        if (!drop) { return; }
        if (current == nullptr ||
            (current->state_changes.empty() && current->main_kv_changes.empty() &&
             current->backend_kv_changes.empty())) {
            append_unique(std::move(*drop));
            return;
        }
        qwen3_5::detail::PressureDecision explicit_target = explicit_pressure_target(*current);
        if (state_change_conflicts(checkpoint, explicit_target)) { return; }
        const std::optional<qwen3_5::TargetKVRequirement> retained =
            retained_requirement_after_drops(summary, drop->dropped_checkpoints);
        if (!retained) { return; }
        const auto within = [](std::span<const qwen3_5::detail::PressureKVDecision> changes,
                               std::uint32_t pages) {
            return std::all_of(changes.begin(), changes.end(), [&](const auto& action) {
                return action.kind != qwen3_5::detail::PressureKVDecisionKind::None &&
                       action.page_count != 0 && action.begin_page <= pages &&
                       action.page_count <= pages - action.begin_page;
            });
        };
        if (!within(explicit_target.main_kv_changes, retained->main_pages) ||
            !within(explicit_target.backend_kv_changes, retained->backend_pages)) {
            return;
        }
        append_unique(
            combine_checkpoint_and_replica_target(std::move(*drop), std::move(explicit_target)));
    };
    if (summary.endpoint) { append_checkpoint_successor(summary.endpoint->ref); }
    if (summary.rewrite) { append_checkpoint_successor(summary.rewrite->ref); }
    for (const qwen3_5::CheckpointSummary& anchor : summary.long_anchors) {
        append_checkpoint_successor(anchor.ref);
    }

    const std::optional<qwen3_5::TargetKVRequirement> retained =
        dropped.empty() ? std::optional<qwen3_5::TargetKVRequirement>()
                        : retained_requirement_after_drops(summary, dropped);
    if (!dropped.empty() && !retained) { return successors; }
    std::vector<detail::PhysicalResources> endpoints;
    endpoints.reserve(8);
    const auto endpoint = [&](detail::PhysicalResources value) {
        if (value != detail::PhysicalResources{} &&
            std::find(endpoints.begin(), endpoints.end(), value) == endpoints.end()) {
            endpoints.push_back(value);
        }
    };
    endpoint(residual);
    if (residual.device.state_slots != 0) {
        detail::PhysicalResources value;
        value.device.state_slots = residual.device.state_slots;
        endpoint(value);
    }
    if (residual.host.state_slots != 0) {
        detail::PhysicalResources value;
        value.host.state_slots = residual.host.state_slots;
        endpoint(value);
    }
    if (residual.device.main_kv_pages != 0) {
        detail::PhysicalResources exact;
        exact.device.main_kv_pages = residual.device.main_kv_pages;
        endpoint(exact);
        detail::PhysicalResources full;
        full.device.main_kv_pages = std::numeric_limits<std::uint32_t>::max();
        endpoint(full);
    }
    if (residual.device.backend_kv_pages != 0) {
        detail::PhysicalResources exact;
        exact.device.backend_kv_pages = residual.device.backend_kv_pages;
        endpoint(exact);
        detail::PhysicalResources full;
        full.device.backend_kv_pages = std::numeric_limits<std::uint32_t>::max();
        endpoint(full);
    }
    if (residual.host.kv_bytes != 0) {
        detail::PhysicalResources host;
        host.host.kv_bytes = residual.host.kv_bytes;
        endpoint(host);
        detail::PhysicalResources full_host;
        full_host.host.kv_bytes = std::numeric_limits<std::size_t>::max();
        endpoint(full_host);
    }

    for (const detail::PhysicalResources requested : endpoints) {
        std::optional<qwen3_5::detail::PressureDecision> replica = inspect_pressure_option(
            sequence, requested, protection, retained ? &*retained : nullptr, dropped, {}, current);
        if (!replica) { continue; }
        if (dropped.empty()) {
            append_unique(std::move(*replica));
        } else {
            std::optional<qwen3_5::detail::PressureDecision> drop =
                inspect_checkpoint_drop_option(sequence, dropped);
            if (drop) {
                append_unique(
                    combine_checkpoint_and_replica_target(std::move(*drop), std::move(*replica)));
            }
        }
    }
    return successors;
}

std::vector<qwen3_5::detail::PressureDecision> ProgramImpl::inspect_shared_pressure_successors(
    const SharedPrefixState& shared, detail::PhysicalResources residual,
    const MaterializationSourceProtection* protection,
    const qwen3_5::detail::PressureDecision* current) const {
    return inspect_shared_pressure_options(shared, residual, protection, current);
}

bool ProgramImpl::pressure_decision_valid(const SequenceState& sequence,
                                          const qwen3_5::detail::PressureDecision& decision,
                                          const MaterializationSourceProtection* protection) const {
    if (!sequence.kv || decision.evicts_continuation || decision.shared_owner || decision.id == 0 ||
        decision.checkpoint_drops != decision.dropped_checkpoints.size()) {
        return false;
    }
    std::optional<qwen3_5::TargetKVRequirement> retained;
    if (!decision.dropped_checkpoints.empty()) {
        const std::optional<qwen3_5::detail::PressureDecision> canonical =
            inspect_checkpoint_drop_option(sequence, decision.dropped_checkpoints);
        if (!canonical || canonical->dropped_checkpoints != decision.dropped_checkpoints ||
            canonical->checkpoint_drop_effect != decision.checkpoint_drop_effect) {
            return false;
        }
        retained = retained_requirement_after_drops(continuation_summary(sequence),
                                                    decision.dropped_checkpoints);
        if (!retained) { return false; }
    } else if (decision.checkpoint_drop_effect != qwen3_5::detail::PhysicalDelta{}) {
        return false;
    }
    const auto dropped_kind = [&](runtime::CheckpointKind kind) {
        return std::any_of(
            decision.dropped_checkpoints.begin(), decision.dropped_checkpoints.end(),
            [&](runtime::CheckpointRef checkpoint) { return checkpoint.kind == kind; });
    };
    std::vector<StateImageHandle> targeted_states;
    for (const qwen3_5::detail::PressureStateDecision action : decision.state_changes) {
        const bool endpoint =
            action == qwen3_5::detail::PressureStateDecision::DropEndpointDeviceDuplicate ||
            action == qwen3_5::detail::PressureStateDecision::DemoteEndpointToHost ||
            action == qwen3_5::detail::PressureStateDecision::DropEndpointHostDuplicate;
        const bool rewrite =
            action == qwen3_5::detail::PressureStateDecision::DropRewriteDeviceDuplicate ||
            action == qwen3_5::detail::PressureStateDecision::DemoteRewriteToHost ||
            action == qwen3_5::detail::PressureStateDecision::DropRewriteHostDuplicate;
        if ((!endpoint && !rewrite) ||
            (endpoint && dropped_kind(runtime::CheckpointKind::SessionEndpoint)) ||
            (rewrite && (dropped_kind(runtime::CheckpointKind::TurnClosure) ||
                         dropped_kind(runtime::CheckpointKind::ResponseReplay)))) {
            return false;
        }
        const std::optional<StateImageHandle> state =
            pressure_state_source(action, &sequence, nullptr);
        const bool drops_host = pressure_state_drops_host(action);
        if (!state || !state_store->valid(*state) ||
            state_store->role(*state) != StateImageRole::CheckpointImmutable ||
            state_store->source_pins(*state) != 0 ||
            (!drops_host && !state_exclusive_to_sequence(sequence, *state)) ||
            (!drops_host && protection != nullptr && protection->state &&
             *protection->state == *state) ||
            std::find(targeted_states.begin(), targeted_states.end(), *state) !=
                targeted_states.end()) {
            return false;
        }
        targeted_states.push_back(*state);
        const StateReplicaResidency residency = state_store->residency(*state);
        if (drops_host) {
            if (residency != StateReplicaResidency::Both) { return false; }
        } else if (pressure_state_demotes(action)) {
            if (residency != StateReplicaResidency::DeviceOnly || host_state_images == nullptr) {
                return false;
            }
        } else if (residency != StateReplicaResidency::Both) {
            return false;
        }
    }
    const auto valid_kv = [&](const KVAddressSpaceStore* addresses, LogicalKVPageStore* pages,
                              std::optional<KVAddressSpaceHandle> address,
                              std::span<const qwen3_5::detail::PressureKVDecision> changes,
                              std::uint32_t retained_pages,
                              runtime::ContextResourceClass resource) {
        if (changes.empty()) { return true; }
        if (addresses == nullptr || pages == nullptr || !address || !addresses->valid(*address)) {
            return false;
        }
        const std::uint32_t mapped = addresses->mapped_pages(*address);
        const std::uint32_t limit  = std::min(mapped, retained_pages);
        std::vector<LogicalKVPageHandle> targeted;
        for (const qwen3_5::detail::PressureKVDecision& action : changes) {
            if (action.kind == qwen3_5::detail::PressureKVDecisionKind::None ||
                action.page_count == 0 || action.begin_page > limit ||
                action.page_count > limit - action.begin_page) {
                return false;
            }
            for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
                const std::uint32_t page_offset = action.begin_page + offset;
                const LogicalKVPageHandle page  = addresses->logical_page(*address, page_offset);
                const bool protected_page       = protected_materialization_page(
                    protection, *addresses, page_offset, page,
                    resource == runtime::ContextResourceClass::BackendKV);
                if (std::find(targeted.begin(), targeted.end(), page) != targeted.end() ||
                    pages->writer_references(page) != 0 || pages->source_pins(page) != 0 ||
                    (protected_page &&
                     action.kind != qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate)) {
                    return false;
                }
                targeted.push_back(page);
                if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate) {
                    if (!pages->device_resident(page) || !pages->host_resident(page) ||
                        host_kv_extents == nullptr ||
                        !host_kv_extents->can_release_page_replica(*pages, page)) {
                        return false;
                    }
                } else if (addresses->has_active_reference(page)) {
                    return false;
                } else if (action.kind ==
                           qwen3_5::detail::PressureKVDecisionKind::DropDeviceDuplicate) {
                    if (!pages->can_drop_device_replica(page)) { return false; }
                } else if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DemoteToHost) {
                    if (!pages->device_resident(page) || pages->host_resident(page) ||
                        host_kv_extents == nullptr || host_kv_arena == nullptr) {
                        return false;
                    }
                }
            }
        }
        return true;
    };
    const std::uint32_t main_limit =
        retained ? retained->main_pages : text_kv_addresses->mapped_pages(sequence.kv->text);
    const std::uint32_t backend_limit =
        retained ? retained->backend_pages
                 : (sequence.kv->backend && backend_kv_addresses
                        ? backend_kv_addresses->mapped_pages(*sequence.kv->backend)
                        : 0U);
    return valid_kv(text_kv_addresses.get(), text_kv_pages.get(), sequence.kv->text,
                    decision.main_kv_changes, main_limit, runtime::ContextResourceClass::MainKV) &&
           valid_kv(backend_kv_addresses.get(), backend_kv_pages.get(), sequence.kv->backend,
                    decision.backend_kv_changes, backend_limit,
                    runtime::ContextResourceClass::BackendKV) &&
           (!decision.state_changes.empty() || !decision.main_kv_changes.empty() ||
            !decision.backend_kv_changes.empty() || !decision.dropped_checkpoints.empty());
}

bool ProgramImpl::shared_pressure_decision_valid(
    const SharedPrefixState& shared, const qwen3_5::detail::PressureDecision& decision,
    const MaterializationSourceProtection* protection) const {
    if (!shared.kv || shared.active_references != 0 || decision.evicts_continuation ||
        !decision.shared_owner || decision.id == 0 || !decision.dropped_checkpoints.empty() ||
        decision.checkpoint_drops != 0 ||
        decision.checkpoint_drop_effect != qwen3_5::detail::PhysicalDelta{} ||
        decision.state_changes.size() > 1) {
        return false;
    }
    if (!decision.state_changes.empty()) {
        const qwen3_5::detail::PressureStateDecision action = decision.state_changes.front();
        const std::optional<StateImageHandle> state =
            pressure_state_source(action, nullptr, &shared);
        const bool drops_host = pressure_state_drops_host(action);
        if (!state || !state_store->valid(*state) ||
            state_store->role(*state) != StateImageRole::CheckpointImmutable ||
            state_store->source_pins(*state) != 0 ||
            (!drops_host && state_store->checkpoint_references(*state) != 1) ||
            (!drops_host && protection != nullptr && protection->state &&
             *protection->state == *state)) {
            return false;
        }
        const StateReplicaResidency residency = state_store->residency(*state);
        if ((drops_host && residency != StateReplicaResidency::Both) ||
            (pressure_state_demotes(action) &&
             (residency != StateReplicaResidency::DeviceOnly || host_state_images == nullptr)) ||
            (!pressure_state_drops_host(action) && !pressure_state_demotes(action) &&
             residency != StateReplicaResidency::Both)) {
            return false;
        }
    }
    const auto valid_kv = [&](const KVAddressSpaceStore* addresses, LogicalKVPageStore* pages,
                              std::optional<KVAddressSpaceHandle> address,
                              std::span<const qwen3_5::detail::PressureKVDecision> changes,
                              runtime::ContextResourceClass resource) {
        if (changes.empty()) { return true; }
        if (addresses == nullptr || pages == nullptr || !address || !addresses->valid(*address)) {
            return false;
        }
        const std::uint32_t mapped = addresses->mapped_pages(*address);
        std::vector<LogicalKVPageHandle> targeted;
        for (const auto& action : changes) {
            if (action.kind == qwen3_5::detail::PressureKVDecisionKind::None ||
                action.page_count == 0 || action.begin_page > mapped ||
                action.page_count > mapped - action.begin_page) {
                return false;
            }
            for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
                const std::uint32_t page_offset = action.begin_page + offset;
                const LogicalKVPageHandle page  = addresses->logical_page(*address, page_offset);
                const bool protected_page       = protected_materialization_page(
                    protection, *addresses, page_offset, page,
                    resource == runtime::ContextResourceClass::BackendKV);
                if (std::find(targeted.begin(), targeted.end(), page) != targeted.end() ||
                    pages->writer_references(page) != 0 || pages->source_pins(page) != 0 ||
                    (protected_page &&
                     action.kind != qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate)) {
                    return false;
                }
                targeted.push_back(page);
                if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate) {
                    if (!pages->device_resident(page) || !pages->host_resident(page) ||
                        host_kv_extents == nullptr ||
                        !host_kv_extents->can_release_page_replica(*pages, page)) {
                        return false;
                    }
                } else if (addresses->has_active_reference(page)) {
                    return false;
                } else if (action.kind ==
                           qwen3_5::detail::PressureKVDecisionKind::DropDeviceDuplicate) {
                    if (!pages->can_drop_device_replica(page)) { return false; }
                } else if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DemoteToHost) {
                    if (!pages->device_resident(page) || pages->host_resident(page) ||
                        host_kv_extents == nullptr || host_kv_arena == nullptr) {
                        return false;
                    }
                }
            }
        }
        return true;
    };
    return valid_kv(text_kv_addresses.get(), text_kv_pages.get(), shared.kv->text,
                    decision.main_kv_changes, runtime::ContextResourceClass::MainKV) &&
           valid_kv(backend_kv_addresses.get(), backend_kv_pages.get(), shared.kv->backend,
                    decision.backend_kv_changes, runtime::ContextResourceClass::BackendKV) &&
           (!decision.state_changes.empty() || !decision.main_kv_changes.empty() ||
            !decision.backend_kv_changes.empty());
}

qwen3_5::detail::PressureDecision
ProgramImpl::inspect_eviction_option(const SequenceState& sequence) const {
    qwen3_5::detail::PressureDecision option;
    option.id                                  = std::numeric_limits<std::uint64_t>::max();
    option.effect.removed                      = owner_exclusive_resources(sequence);
    const qwen3_5::ContinuationSummary summary = continuation_summary(sequence);
    if (!sequence.kv || summary.long_anchors.size() != sequence.long_anchors.size()) {
        throw std::logic_error("eviction owner checkpoint inventory is incomplete");
    }
    option.checkpoint_drops = planning_saturating_u32(
        summary.endpoint.has_value() + summary.rewrite.has_value() + summary.long_anchors.size());
    option.evicts_continuation = true;
    return option;
}

qwen3_5::detail::PressureDecision
ProgramImpl::inspect_shared_eviction_option(const SharedPrefixState& shared) const {
    qwen3_5::detail::PressureDecision option;
    option.id                  = std::numeric_limits<std::uint64_t>::max() - 1U;
    option.effect.removed      = owner_exclusive_resources(shared);
    option.checkpoint_drops    = 1;
    option.evicts_continuation = true;
    option.shared_owner        = true;
    return option;
}

std::optional<detail::PressureTargetProjection> ProgramImpl::evaluate_pressure_target(
    const MaterializationSourceProtection* protection,
    std::span<const ContinuationHandle* const> pressure_owners,
    std::span<const qwen3_5::detail::PressureDecision> pressure_options,
    std::span<const SharedPrefixHandle* const> shared_pressure_owners,
    std::span<const qwen3_5::detail::PressureDecision> shared_pressure_options,
    std::vector<HostKVPageReplicaRelease>* released_host_pages) const {
    if (pressure_owners.size() != pressure_options.size() ||
        shared_pressure_owners.size() != shared_pressure_options.size()) {
        throw std::invalid_argument("combined pressure selection is not row aligned");
    }
    begin_pressure_page_scratch();

    std::vector<std::uint8_t>& private_owner_state = pressure_private_owner_scratch_;
    std::vector<std::uint8_t>& shared_owner_state  = pressure_shared_owner_scratch_;
    constexpr std::uint8_t kOwnerSelected          = 1U;
    constexpr std::uint8_t kOwnerEvicted           = 2U;
    std::fill(private_owner_state.begin(), private_owner_state.end(), 0);
    std::fill(shared_owner_state.begin(), shared_owner_state.end(), 0);
    std::vector<std::vector<runtime::CheckpointRef>>& dropped_private =
        pressure_private_drop_scratch_;
    for (auto& dropped : dropped_private) { dropped.clear(); }
    // Preserving pressure work is published per option. Aliased physical targets would let the
    // first publication invalidate the next while both effects had already been credited.
    std::vector<PressureSelectedState>& pressure_states = pressure_state_scratch_;
    pressure_states.clear();

    const auto append_pressure_targets = [&](const qwen3_5::detail::PressureDecision& option,
                                             const SequenceState* sequence,
                                             const SharedPrefixState* shared) {
        for (const qwen3_5::detail::PressureStateDecision change : option.state_changes) {
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
            const bool protected_state =
                protection != nullptr && protection->state && *protection->state == *state;
            const auto existing = std::find_if(
                pressure_states.begin(), pressure_states.end(),
                [&](const PressureSelectedState& selected) { return selected.state == *state; });
            if (!state_store->valid(*state) ||
                (protected_state && !pressure_state_drops_host(change)) ||
                existing != pressure_states.end()) {
                return false;
            }
            const StateReplicaResidency residency = state_store->residency(*state);
            PressureSelectedState selected{
                .state  = *state,
                .device = residency == StateReplicaResidency::DeviceOnly ||
                          residency == StateReplicaResidency::Both,
                .host = residency == StateReplicaResidency::HostOnly ||
                        residency == StateReplicaResidency::Both,
            };
            if (pressure_state_drops_host(change)) {
                selected.host = false;
            } else {
                selected.device = false;
                selected.host   = true;
            }
            pressure_states.push_back(selected);
        }

        const SequenceKVBundle* kv =
            sequence != nullptr ? (sequence->kv ? &*sequence->kv : nullptr)
                                : (shared != nullptr && shared->kv ? &*shared->kv : nullptr);
        const auto append_pages = [&](const KVAddressSpaceStore* addresses,
                                      const LogicalKVPageStore* pages,
                                      std::optional<KVAddressSpaceHandle> address,
                                      const qwen3_5::detail::PressureKVDecision& action) {
            if (action.kind == qwen3_5::detail::PressureKVDecisionKind::None) {
                return action.page_count == 0;
            }
            if (addresses == nullptr || pages == nullptr || !address ||
                !addresses->valid(*address) || action.page_count == 0) {
                return false;
            }
            const std::uint32_t mapped = addresses->mapped_pages(*address);
            if (action.begin_page > mapped || action.page_count > mapped - action.begin_page) {
                return false;
            }
            for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
                const LogicalKVPageHandle page =
                    addresses->logical_page(*address, action.begin_page + offset);
                const bool backend            = addresses == backend_kv_addresses.get();
                PressurePageScratchSlot& slot = pressure_page_scratch(*pages, page);
                const bool protected_page     = protected_materialization_page(
                    protection, *addresses, action.begin_page + offset, page, backend);
                if ((protected_page &&
                     action.kind != qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate) ||
                    slot.pressure_targeted) {
                    return false;
                }
                slot.pressure_targeted = true;
                slot.projected         = true;
                slot.device            = pages->device_resident(page);
                slot.host              = pages->host_resident(page);
                if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate) {
                    slot.host = false;
                } else {
                    slot.device = false;
                    slot.host   = true;
                }
            }
            return true;
        };
        const std::optional<KVAddressSpaceHandle> text =
            kv != nullptr ? std::optional<KVAddressSpaceHandle>(kv->text) : std::nullopt;
        const std::optional<KVAddressSpaceHandle> backend =
            kv != nullptr ? kv->backend : std::nullopt;
        for (const qwen3_5::detail::PressureKVDecision& action : option.main_kv_changes) {
            if (!append_pages(text_kv_addresses.get(), text_kv_pages.get(), text, action)) {
                return false;
            }
        }
        for (const qwen3_5::detail::PressureKVDecision& action : option.backend_kv_changes) {
            if (!append_pages(backend_kv_addresses.get(), backend_kv_pages.get(), backend,
                              action)) {
                return false;
            }
        }
        return true;
    };
    detail::PressureTargetProjection projection;
    if (protection != nullptr && protection->consumed_private_source) {
        projection.source_text_prefix_fork_required    = protection->text_prefix_fork_required;
        projection.source_backend_prefix_fork_required = protection->backend_prefix_fork_required;
    }
    for (std::size_t position = 0; position < pressure_owners.size(); ++position) {
        const ContinuationHandle* owner                 = pressure_owners[position];
        const qwen3_5::detail::PressureDecision& option = pressure_options[position];
        if (owner == nullptr || ContractAccess::owner(*owner) != this ||
            !valid_continuation(*owner) || option.shared_owner) {
            return std::nullopt;
        }
        const std::uint32_t index = ContractAccess::index(*owner);
        if ((private_owner_state[index] & kOwnerSelected) != 0 ||
            (protection != nullptr && protection->private_source_index == index)) {
            return std::nullopt;
        }
        private_owner_state[index] |= kOwnerSelected;
        if (option.evicts_continuation) {
            if (option.effect.added != detail::PhysicalResources{}) { return std::nullopt; }
            private_owner_state[index] |= kOwnerEvicted;
        } else {
            if (!append_pressure_targets(option, &continuation_states[index], nullptr)) {
                return std::nullopt;
            }
            dropped_private[index] = option.dropped_checkpoints;
            // Checkpoint release is not owner-additive: StateImages and logical KV pages may be
            // shared by several selected owners.  Strip the complete locally estimated drop
            // effect here and settle it once from the joint post-reference state below.
            projection.unique_object_delta.removed = checked_resource_sum(
                projection.unique_object_delta.removed,
                checked_resource_difference(option.effect.removed,
                                            option.checkpoint_drop_effect.removed));
            projection.unique_object_delta.added =
                checked_resource_sum(projection.unique_object_delta.added, option.effect.added);
        }
    }
    for (std::size_t position = 0; position < shared_pressure_owners.size(); ++position) {
        const SharedPrefixHandle* owner                 = shared_pressure_owners[position];
        const qwen3_5::detail::PressureDecision& option = shared_pressure_options[position];
        if (owner == nullptr || ContractAccess::owner(*owner) != this ||
            !valid_shared_prefix(*owner) || !option.shared_owner) {
            return std::nullopt;
        }
        const std::uint32_t index = ContractAccess::index(*owner);
        if ((shared_owner_state[index] & kOwnerSelected) != 0) { return std::nullopt; }
        shared_owner_state[index] |= kOwnerSelected;
        if (option.evicts_continuation) {
            if (option.effect.added != detail::PhysicalResources{}) { return std::nullopt; }
            shared_owner_state[index] |= kOwnerEvicted;
        } else {
            if (!append_pressure_targets(option, nullptr, &shared_prefix_states[index])) {
                return std::nullopt;
            }
            projection.unique_object_delta.removed =
                checked_resource_sum(projection.unique_object_delta.removed, option.effect.removed);
            projection.unique_object_delta.added =
                checked_resource_sum(projection.unique_object_delta.added, option.effect.added);
        }
    }

    const auto final_state_placement = [&](StateImageHandle state) {
        const auto selected =
            std::find_if(pressure_states.begin(), pressure_states.end(),
                         [&](const PressureSelectedState& item) { return item.state == state; });
        if (selected != pressure_states.end()) {
            return std::pair{selected->device, selected->host};
        }
        const StateReplicaResidency residency = state_store->residency(state);
        return std::pair{
            residency == StateReplicaResidency::DeviceOnly ||
                residency == StateReplicaResidency::Both,
            residency == StateReplicaResidency::HostOnly ||
                residency == StateReplicaResidency::Both,
        };
    };

    std::vector<PressureSelectedPage>& main_pages    = pressure_text_selected_pages_;
    std::vector<PressureSelectedPage>& backend_pages = pressure_backend_selected_pages_;
    const auto append_selected_page = [&](LogicalKVPageStore& store, LogicalKVPageHandle page,
                                          std::vector<PressureSelectedPage>& selected) {
        PressurePageScratchSlot& slot = pressure_page_scratch(store, page);
        if (slot.selected_index == std::numeric_limits<std::uint32_t>::max()) {
            if (selected.size() >= std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("pressure selected page count exceeds uint32");
            }
            slot.selected_index = static_cast<std::uint32_t>(selected.size());
            selected.push_back(PressureSelectedPage{.page = page, .references = 1});
        } else {
            if (slot.selected_index >= selected.size()) {
                throw std::logic_error("pressure selected page scratch is inconsistent");
            }
            ++selected[slot.selected_index].references;
        }
    };
    const auto append_address = [&](const KVAddressSpaceStore& addresses, LogicalKVPageStore& store,
                                    KVAddressSpaceHandle address,
                                    std::vector<PressureSelectedPage>& selected) {
        if (!addresses.valid(address) || addresses.active(address)) {
            throw std::logic_error("evicted KV address is not an inactive publication");
        }
        for (std::uint32_t offset = 0; offset < addresses.mapped_pages(address); ++offset) {
            const LogicalKVPageHandle page = addresses.logical_page(address, offset);
            append_selected_page(store, page, selected);
        }
    };
    const auto append_address_suffix = [&](const KVAddressSpaceStore& addresses,
                                           LogicalKVPageStore& store, KVAddressSpaceHandle address,
                                           std::uint32_t retained_pages,
                                           std::vector<PressureSelectedPage>& selected) {
        if (!addresses.valid(address) || addresses.active(address)) {
            throw std::logic_error("dropped checkpoint KV suffix is invalid");
        }
        const std::uint32_t mapped = addresses.mapped_pages(address);
        if (retained_pages > mapped) {
            throw std::logic_error("dropped checkpoint KV suffix exceeds its address space");
        }
        for (std::uint32_t offset = retained_pages; offset < mapped; ++offset) {
            const LogicalKVPageHandle page = addresses.logical_page(address, offset);
            append_selected_page(store, page, selected);
        }
    };
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if ((private_owner_state[index] & kOwnerEvicted) == 0) { continue; }
        const SequenceState& sequence = continuation_states[index];
        if (!sequence.kv) { return std::nullopt; }
        append_address(*text_kv_addresses, *text_kv_pages, sequence.kv->text, main_pages);
        if (sequence.kv->backend) {
            if (!backend_kv_addresses || !backend_kv_pages) { return std::nullopt; }
            append_address(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
                           backend_pages);
        }
    }
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (dropped_private[index].empty() || (private_owner_state[index] & kOwnerEvicted) != 0) {
            continue;
        }
        const SequenceState& sequence = continuation_states[index];
        if (!sequence.kv) { return std::nullopt; }
        const std::optional<qwen3_5::TargetKVRequirement> retained =
            retained_requirement_after_drops(continuation_summary(sequence),
                                             dropped_private[index]);
        if (!retained) { return std::nullopt; }
        append_address_suffix(*text_kv_addresses, *text_kv_pages, sequence.kv->text,
                              retained->main_pages, main_pages);
        if (sequence.kv->backend) {
            if (!backend_kv_addresses || !backend_kv_pages) { return std::nullopt; }
            append_address_suffix(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
                                  retained->backend_pages, backend_pages);
        }
    }
    for (std::uint32_t index = 0; index < shared_prefix_capacity; ++index) {
        if ((shared_owner_state[index] & kOwnerEvicted) == 0) { continue; }
        const SharedPrefixState& shared = shared_prefix_states[index];
        if (!shared.kv) { return std::nullopt; }
        append_address(*text_kv_addresses, *text_kv_pages, shared.kv->text, main_pages);
        if (shared.kv->backend) {
            if (!backend_kv_addresses || !backend_kv_pages) { return std::nullopt; }
            append_address(*backend_kv_addresses, *backend_kv_pages, *shared.kv->backend,
                           backend_pages);
        }
    }

    const auto dropped_checkpoint_state =
        [&](const SequenceState& sequence,
            runtime::CheckpointRef checkpoint) -> std::optional<StateImageHandle> {
        if (checkpoint.kind == runtime::CheckpointKind::SessionEndpoint) {
            return sequence.endpoint_valid ? std::optional<StateImageHandle>(sequence.state.read)
                                           : std::nullopt;
        }
        if (checkpoint.kind == runtime::CheckpointKind::TurnClosure ||
            checkpoint.kind == runtime::CheckpointKind::ResponseReplay) {
            return sequence.rewrite_state;
        }
        if (checkpoint.kind == runtime::CheckpointKind::LongAnchor) {
            const auto anchor =
                std::find_if(sequence.long_anchors.begin(), sequence.long_anchors.end(),
                             [&](const LongAnchorCheckpoint& candidate) {
                                 return candidate.frontier == checkpoint.frontier &&
                                        candidate.ordinal == checkpoint.ordinal;
                             });
            return anchor == sequence.long_anchors.end()
                       ? std::nullopt
                       : std::optional<StateImageHandle>(anchor->state);
        }
        return std::nullopt;
    };
    const auto removed_state_references = [&](StateImageHandle state) {
        std::uint32_t references = 0;
        for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
            const SequenceState& sequence = continuation_states[index];
            if ((private_owner_state[index] & kOwnerEvicted) != 0) {
                if (sequence.rewrite_state && *sequence.rewrite_state == state) { ++references; }
                references += static_cast<std::uint32_t>(std::count_if(
                    sequence.long_anchors.begin(), sequence.long_anchors.end(),
                    [&](const LongAnchorCheckpoint& anchor) { return anchor.state == state; }));
                continue;
            }
            for (const runtime::CheckpointRef checkpoint : dropped_private[index]) {
                const std::optional<StateImageHandle> dropped =
                    dropped_checkpoint_state(sequence, checkpoint);
                if (dropped && *dropped == state &&
                    checkpoint.kind != runtime::CheckpointKind::SessionEndpoint) {
                    ++references;
                }
            }
        }
        for (std::uint32_t index = 0; index < shared_prefix_capacity; ++index) {
            if ((shared_owner_state[index] & kOwnerEvicted) != 0 &&
                shared_prefix_states[index].state == state) {
                ++references;
            }
        }
        return references;
    };
    if (protection != nullptr && protection->consumed_private_source) {
        if (!protection->state || !protection->text) { return std::nullopt; }
        const std::uint32_t selected_state_references =
            state_store->checkpoint_references(*protection->state);
        const std::uint32_t selected_state_removed = removed_state_references(*protection->state);
        if (selected_state_removed > selected_state_references ||
            selected_state_references - selected_state_removed <
                protection->consumed_state_references) {
            return std::nullopt;
        }
        projection.source_state_fork_required =
            selected_state_references - selected_state_removed !=
            protection->consumed_state_references;

        for (const auto& candidate : protection->state_ownership_candidates) {
            const std::uint32_t references = state_store->checkpoint_references(candidate.state);
            const std::uint32_t removed    = removed_state_references(candidate.state);
            if (candidate.source_checkpoint_references == 0 || removed > references ||
                references - removed < candidate.source_checkpoint_references) {
                return std::nullopt;
            }
            if (references - removed == candidate.source_checkpoint_references) {
                detail::PhysicalResources transferred;
                const auto [device_resident, host_resident] =
                    final_state_placement(candidate.state);
                if (device_resident) { transferred.device.state_slots = 1; }
                if (host_resident) { transferred.host.state_slots = 1; }
                if (transferred == detail::PhysicalResources{}) { return std::nullopt; }
                projection.active_entitlement_delta.added =
                    checked_resource_sum(projection.active_entitlement_delta.added, transferred);
                projection.source_optional_resources_added =
                    checked_resource_sum(projection.source_optional_resources_added, transferred);
                // The allocation already exists. Only its accounting ownership moves from shared
                // cache occupancy into the consumed active lineage.
                projection.ownership_transfer_delta.removed =
                    checked_resource_sum(projection.ownership_transfer_delta.removed, transferred);
                projection.ownership_transfer_delta.added =
                    checked_resource_sum(projection.ownership_transfer_delta.added, transferred);
            }
        }

        const auto removed_page_references = [&](LogicalKVPageStore& pages,
                                                 LogicalKVPageHandle page) {
            const PressurePageScratchSlot* slot = find_pressure_page_scratch(pages, page);
            if (slot == nullptr ||
                slot->selected_index == std::numeric_limits<std::uint32_t>::max()) {
                return 0U;
            }
            const std::vector<PressureSelectedPage>& selected =
                &pages == text_kv_pages.get() ? main_pages : backend_pages;
            if (slot->selected_index >= selected.size()) {
                throw std::logic_error("pressure selected page scratch is inconsistent");
            }
            return selected[slot->selected_index].references;
        };
        const auto append_kv_ownership_transfers = [&](const KVAddressSpaceStore& addresses,
                                                       LogicalKVPageStore& pages,
                                                       KVAddressSpaceHandle address,
                                                       std::uint32_t protected_pages,
                                                       std::uint32_t transferable_pages,
                                                       runtime::ContextResourceClass resource) {
            if (!addresses.valid(address) || protected_pages > addresses.mapped_pages(address) ||
                transferable_pages > protected_pages) {
                return false;
            }
            const std::size_t stride =
                plan_host_kv_page_layout(pages.physical_pool().geometry()).page_stride;
            for (std::uint32_t offset = 0; offset < protected_pages; ++offset) {
                const LogicalKVPageHandle page          = addresses.logical_page(address, offset);
                const std::uint32_t references          = pages.address_references(page);
                const std::uint32_t removed             = removed_page_references(pages, page);
                const PressurePageScratchSlot* selected = find_pressure_page_scratch(pages, page);
                const bool device_resident              = selected != nullptr && selected->projected
                                                              ? selected->device
                                                              : pages.device_resident(page);
                const bool host_resident                = selected != nullptr && selected->projected
                                                              ? selected->host
                                                              : pages.host_resident(page);
                if (removed >= references) { return false; }
                if (references <= 1 || references - removed != 1) { continue; }
                if (offset >= transferable_pages) {
                    // The only protected page outside the transferable full-page prefix is a
                    // partial tail. Once the complete victim set leaves it with one address
                    // reference, the consumed source can mutate that page in place and the COW
                    // destination/copy disappear from the direct target transition.
                    if (offset + 1U != protected_pages) { return false; }
                    std::optional<bool>& prefix_fork =
                        resource == runtime::ContextResourceClass::MainKV
                            ? projection.source_text_prefix_fork_required
                            : projection.source_backend_prefix_fork_required;
                    if (!prefix_fork || !*prefix_fork) { return false; }
                    prefix_fork = false;

                    detail::PhysicalResources transferred;
                    if (resource == runtime::ContextResourceClass::MainKV) {
                        if (device_resident) { transferred.device.main_kv_pages = 1; }
                    } else if (device_resident) {
                        transferred.device.backend_kv_pages = 1;
                    }
                    if (host_resident) { transferred.host.kv_bytes = stride; }
                    if (transferred == detail::PhysicalResources{}) { return false; }
                    projection.ownership_transfer_delta.removed = checked_resource_sum(
                        projection.ownership_transfer_delta.removed, transferred);
                    projection.ownership_transfer_delta.added = checked_resource_sum(
                        projection.ownership_transfer_delta.added, transferred);
                    continue;
                }
                detail::PhysicalResources active_added;
                detail::PhysicalResources transferred;
                if (resource == runtime::ContextResourceClass::MainKV) {
                    active_added.device.main_kv_pages = 1;
                    if (device_resident) { transferred.device.main_kv_pages = 1; }
                } else {
                    active_added.device.backend_kv_pages = 1;
                    if (device_resident) { transferred.device.backend_kv_pages = 1; }
                }
                if (host_resident) {
                    active_added.host.kv_bytes = stride;
                    transferred.host.kv_bytes  = stride;
                } else if (!device_resident) {
                    return false;
                }
                projection.active_entitlement_delta.added =
                    checked_resource_sum(projection.active_entitlement_delta.added, active_added);
                projection.ownership_transfer_delta.removed =
                    checked_resource_sum(projection.ownership_transfer_delta.removed, transferred);
                projection.ownership_transfer_delta.added =
                    checked_resource_sum(projection.ownership_transfer_delta.added, transferred);
            }
            return true;
        };
        if (!append_kv_ownership_transfers(*text_kv_addresses, *text_kv_pages, *protection->text,
                                           protection->text_pages, protection->text_transfer_pages,
                                           runtime::ContextResourceClass::MainKV)) {
            return std::nullopt;
        }
        if (protection->backend &&
            (!backend_kv_addresses || !backend_kv_pages ||
             !append_kv_ownership_transfers(*backend_kv_addresses, *backend_kv_pages,
                                            *protection->backend, protection->backend_pages,
                                            protection->backend_transfer_pages,
                                            runtime::ContextResourceClass::BackendKV))) {
            return std::nullopt;
        }
    }

    std::vector<PressureSelectedState>& selected_states = pressure_state_scratch_;
    const auto append_state                             = [&](StateImageHandle state) {
        const auto selected =
            std::find_if(selected_states.begin(), selected_states.end(),
                                                     [&](const PressureSelectedState& item) { return item.state == state; });
        if (state_store->valid(state) && selected == selected_states.end()) {
            const auto [device_resident, host_resident] = final_state_placement(state);
            selected_states.push_back(PressureSelectedState{
                                            .state  = state,
                                            .device = device_resident,
                                            .host   = host_resident,
            });
        }
    };
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        const SequenceState& sequence = continuation_states[index];
        if ((private_owner_state[index] & kOwnerEvicted) != 0) {
            append_state(sequence.state.write);
            if (!sequence.state.borrows_read() || sequence.state.read == sequence.state.write) {
                append_state(sequence.state.read);
            }
            if (sequence.rewrite_state) { append_state(*sequence.rewrite_state); }
            if (sequence.reserved_state) { append_state(*sequence.reserved_state); }
            for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
                append_state(anchor.state);
            }
        } else if (!dropped_private[index].empty()) {
            for (const runtime::CheckpointRef checkpoint : dropped_private[index]) {
                const std::optional<StateImageHandle> dropped =
                    dropped_checkpoint_state(sequence, checkpoint);
                if (!dropped) { return std::nullopt; }
                append_state(*dropped);
            }
        }
    }
    for (std::uint32_t index = 0; index < shared_prefix_capacity; ++index) {
        if ((shared_owner_state[index] & kOwnerEvicted) != 0) {
            append_state(shared_prefix_states[index].state);
        }
    }

    const auto sequence_references_state = [&](std::uint32_t index, const SequenceState& sequence,
                                               StateImageHandle state) {
        const std::vector<runtime::CheckpointRef>& dropped = dropped_private[index];
        const auto dropped_kind                            = [&](runtime::CheckpointKind kind) {
            return std::any_of(
                dropped.begin(), dropped.end(),
                [&](runtime::CheckpointRef checkpoint) { return checkpoint.kind == kind; });
        };
        const bool endpoint_survives =
            sequence.endpoint_valid && !dropped_kind(runtime::CheckpointKind::SessionEndpoint);
        if ((endpoint_survives &&
             (sequence.state.read == state || sequence.state.write == state)) ||
            (sequence.reserved_state && *sequence.reserved_state == state)) {
            return true;
        }
        const bool rewrite_dropped = dropped_kind(runtime::CheckpointKind::TurnClosure) ||
                                     dropped_kind(runtime::CheckpointKind::ResponseReplay);
        if (!rewrite_dropped && sequence.rewrite_state && *sequence.rewrite_state == state) {
            return true;
        }
        return std::any_of(sequence.long_anchors.begin(), sequence.long_anchors.end(),
                           [&](const LongAnchorCheckpoint& anchor) {
                               const bool is_dropped =
                                   std::any_of(dropped.begin(), dropped.end(),
                                               [&](runtime::CheckpointRef checkpoint) {
                                                   return checkpoint.kind ==
                                                              runtime::CheckpointKind::LongAnchor &&
                                                          checkpoint.frontier == anchor.frontier &&
                                                          checkpoint.ordinal == anchor.ordinal;
                                               });
                               return !is_dropped && anchor.state == state;
                           });
    };
    for (const PressureSelectedState& selected_state : selected_states) {
        const StateImageHandle state = selected_state.state;
        if (state_store->source_pins(state) != 0) { continue; }
        bool referenced_by_survivor                  = false;
        std::uint32_t selected_checkpoint_references = 0;
        for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
            if (continuation_slots[index].role == ContinuationSlotRole::Free) { continue; }
            const SequenceState& sequence = continuation_states[index];
            if ((private_owner_state[index] & kOwnerEvicted) == 0 &&
                sequence_references_state(index, sequence, state)) {
                referenced_by_survivor = true;
                break;
            }
            if ((private_owner_state[index] & kOwnerEvicted) != 0) {
                if (sequence.rewrite_state && *sequence.rewrite_state == state) {
                    ++selected_checkpoint_references;
                }
                selected_checkpoint_references += static_cast<std::uint32_t>(std::count_if(
                    sequence.long_anchors.begin(), sequence.long_anchors.end(),
                    [&](const LongAnchorCheckpoint& anchor) { return anchor.state == state; }));
            } else {
                for (const runtime::CheckpointRef checkpoint : dropped_private[index]) {
                    if (checkpoint.kind == runtime::CheckpointKind::SessionEndpoint) { continue; }
                    const std::optional<StateImageHandle> dropped =
                        dropped_checkpoint_state(sequence, checkpoint);
                    if (dropped && *dropped == state) { ++selected_checkpoint_references; }
                }
            }
        }
        if (referenced_by_survivor) { continue; }
        for (std::uint32_t index = 0; index < shared_prefix_capacity; ++index) {
            if (shared_prefix_slots[index].role == SharedPrefixSlotRole::Free) { continue; }
            if (shared_prefix_states[index].state != state) { continue; }
            if ((shared_owner_state[index] & kOwnerEvicted) == 0) {
                referenced_by_survivor = true;
                break;
            }
            ++selected_checkpoint_references;
        }
        if (referenced_by_survivor ||
            selected_checkpoint_references != state_store->checkpoint_references(state)) {
            continue;
        }
        detail::PhysicalResources released;
        if (selected_state.device) { released.device.state_slots = 1; }
        if (selected_state.host) { released.host.state_slots = 1; }
        projection.unique_object_delta.removed =
            checked_resource_sum(projection.unique_object_delta.removed, released);
    }

    const auto append_released_pages = [&](LogicalKVPageStore& pages,
                                           const std::vector<PressureSelectedPage>& selected,
                                           runtime::ContextResourceClass resource) {
        const std::size_t stride =
            plan_host_kv_page_layout(pages.physical_pool().geometry()).page_stride;
        for (const PressureSelectedPage& item : selected) {
            if (pages.address_references(item.page) != item.references ||
                pages.writer_references(item.page) != 0 || pages.source_pins(item.page) != 0) {
                continue;
            }
            const PressurePageScratchSlot* projected = find_pressure_page_scratch(pages, item.page);
            const bool device_resident               = projected != nullptr && projected->projected
                                                           ? projected->device
                                                           : pages.device_resident(item.page);
            const bool host_resident                 = projected != nullptr && projected->projected
                                                           ? projected->host
                                                           : pages.host_resident(item.page);
            detail::PhysicalResources released;
            if (device_resident) {
                if (resource == runtime::ContextResourceClass::MainKV) {
                    released.device.main_kv_pages = 1;
                } else {
                    released.device.backend_kv_pages = 1;
                }
            }
            if (host_resident) {
                released.host.kv_bytes = stride;
                if (released_host_pages != nullptr) {
                    released_host_pages->push_back(
                        HostKVPageReplicaRelease{.pages = &pages, .page = item.page});
                }
            }
            projection.unique_object_delta.removed =
                checked_resource_sum(projection.unique_object_delta.removed, released);
        }
    };
    append_released_pages(*text_kv_pages, main_pages, runtime::ContextResourceClass::MainKV);
    if (backend_kv_pages) {
        append_released_pages(*backend_kv_pages, backend_pages,
                              runtime::ContextResourceClass::BackendKV);
    }
    return projection;
}

std::optional<AdmissionCandidate> ProgramImpl::seal_materialization(
    const AdmissionCandidate& admission, const PreparedPromptData& prompt,
    std::span<const ContinuationHandle* const> pressure_owners,
    std::span<const runtime::PlanningOwnerId> pressure_owner_ids,
    std::span<const qwen3_5::detail::PressureDecision* const> pressure_options,
    std::span<const SharedPrefixHandle* const> shared_pressure_owners,
    std::span<const runtime::PlanningOwnerId> shared_pressure_owner_ids,
    std::span<const qwen3_5::detail::PressureDecision* const> shared_pressure_options) {
    if (admission.impl_ == nullptr || has_context_transaction() || pending_transaction_) {
        return std::nullopt;
    }
    AdmissionCandidate copy(std::make_unique<AdmissionCandidateImpl>(*admission.impl_));
    if (!compose_pressure_candidate(*copy.impl_, pressure_owners, pressure_owner_ids,
                                    pressure_options, shared_pressure_owners,
                                    shared_pressure_owner_ids, shared_pressure_options) ||
        copy.impl_->blocked_host_allocation_bytes != 0 ||
        revalidate_materialization(copy, prompt) != runtime::PreflightStatus::Ready) {
        return std::nullopt;
    }
    return copy;
}

bool ProgramImpl::compose_pressure_candidate(
    ResourceCandidateState& details, std::span<const ContinuationHandle* const> pressure_owners,
    std::span<const runtime::PlanningOwnerId> pressure_owner_ids,
    std::span<const qwen3_5::detail::PressureDecision* const> pressure_options,
    std::span<const SharedPrefixHandle* const> shared_pressure_owners,
    std::span<const runtime::PlanningOwnerId> shared_pressure_owner_ids,
    std::span<const qwen3_5::detail::PressureDecision* const> shared_pressure_options) {
    if (pressure_owners.size() != pressure_owner_ids.size() ||
        pressure_owners.size() != pressure_options.size() ||
        shared_pressure_owners.size() != shared_pressure_owner_ids.size() ||
        shared_pressure_owners.size() != shared_pressure_options.size() ||
        !details.pressure_options.empty() || !details.shared_pressure_options.empty() ||
        details.blocked_host_allocation_bytes != 0) {
        throw std::invalid_argument("materialization pressure composition is invalid");
    }
    const std::optional<MaterializationSourceProtection> protection =
        materialization_source_protection(details);
    if (!protection) { return false; }
    details.pressure_options.reserve(pressure_options.size());
    details.pressure_owner_ids.reserve(pressure_owner_ids.size());
    details.pressure_indices.reserve(pressure_options.size());
    details.pressure_generations.reserve(pressure_options.size());

    bool pressure_needs_transfer = false;
    std::vector<HostKVPageLayout> host_layouts;
    std::vector<HostKVAllocationRequest> private_host_requests;
    std::vector<HostKVAllocationRequest> shared_host_requests;
    std::vector<HostKVPageReplicaRelease> host_releases;
    std::vector<HostKVPageReplicaRelease> host_last_reference_releases;
    const auto demotion_count = [](const qwen3_5::detail::PressureDecision& option) {
        const auto count = [](const auto& changes) {
            return static_cast<std::size_t>(
                std::count_if(changes.begin(), changes.end(), [](const auto& action) {
                    return action.kind == qwen3_5::detail::PressureKVDecisionKind::DemoteToHost;
                }));
        };
        return count(option.main_kv_changes) + count(option.backend_kv_changes);
    };
    std::size_t private_demotion_count = 0;
    for (const qwen3_5::detail::PressureDecision* option : pressure_options) {
        if (option == nullptr) {
            throw std::invalid_argument("materialization pressure option is null");
        }
        private_demotion_count += demotion_count(*option);
    }
    std::size_t shared_demotion_count = 0;
    for (const qwen3_5::detail::PressureDecision* option : shared_pressure_options) {
        if (option == nullptr) {
            throw std::invalid_argument("materialization shared pressure option is null");
        }
        shared_demotion_count += demotion_count(*option);
    }
    host_layouts.reserve(private_demotion_count + shared_demotion_count);
    private_host_requests.reserve(private_demotion_count);
    shared_host_requests.reserve(shared_demotion_count);
    const auto append_host_releases = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                          KVAddressSpaceHandle address,
                                          const qwen3_5::detail::PressureKVDecision& action) {
        if (action.kind != qwen3_5::detail::PressureKVDecisionKind::DropHostDuplicate) { return; }
        const std::uint32_t mapped = addresses.mapped_pages(address);
        if (action.begin_page > mapped || action.page_count > mapped - action.begin_page) {
            throw std::logic_error("materialization Host KV release region is invalid");
        }
        for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
            host_releases.push_back(HostKVPageReplicaRelease{
                .pages = &pages,
                .page  = addresses.logical_page(address, action.begin_page + offset),
            });
        }
    };
    const auto append_kv_actions = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                       KVAddressSpaceHandle address,
                                       std::span<const qwen3_5::detail::PressureKVDecision> changes,
                                       std::vector<HostKVAllocationRequest>& host_requests) {
        for (const qwen3_5::detail::PressureKVDecision& action : changes) {
            append_host_releases(addresses, pages, address, action);
            if (action.kind == qwen3_5::detail::PressureKVDecisionKind::DemoteToHost) {
                host_layouts.push_back(plan_host_kv_page_layout(pages.physical_pool().geometry()));
                host_requests.push_back(
                    {.layout = &host_layouts.back(), .pages = action.page_count});
            }
        }
    };
    for (std::size_t position = 0; position < pressure_options.size(); ++position) {
        const ContinuationHandle* owner                   = pressure_owners[position];
        const runtime::PlanningOwnerId planning_owner     = pressure_owner_ids[position];
        const qwen3_5::detail::PressureDecision& proposed = *pressure_options[position];
        if (owner == nullptr || ContractAccess::owner(*owner) != this ||
            planning_owner.value == std::numeric_limits<std::uint32_t>::max() ||
            std::find(details.pressure_owner_ids.begin(), details.pressure_owner_ids.end(),
                      planning_owner) != details.pressure_owner_ids.end()) {
            throw std::invalid_argument("materialization pressure owner is invalid");
        }
        if (!valid_continuation(*owner)) { return false; }
        const std::uint32_t index      = ContractAccess::index(*owner);
        const std::uint64_t generation = ContractAccess::epoch(*owner);
        if ((details.has_source && index == details.source_index &&
             generation == details.source_generation) ||
            std::find(details.pressure_indices.begin(), details.pressure_indices.end(), index) !=
                details.pressure_indices.end()) {
            throw std::invalid_argument("materialization pressure owner is duplicated");
        }
        qwen3_5::detail::PressureDecision expected;
        if (proposed.evicts_continuation) {
            expected = inspect_eviction_option(continuation_states[index]);
        } else {
            if (!pressure_decision_valid(continuation_states[index], proposed, &*protection)) {
                return false;
            }
            expected = proposed;
        }
        if (expected != proposed || expected.shared_owner) { return false; }
        details.pressure_options.push_back(expected);
        details.pressure_owner_ids.push_back(planning_owner);
        details.pressure_indices.push_back(index);
        details.pressure_generations.push_back(generation);
        pressure_needs_transfer =
            pressure_needs_transfer || !expected.transfer_requirements.empty();
        const SequenceState& pressure_owner = continuation_states[index];
        if (!pressure_owner.kv) { return false; }
        append_kv_actions(*text_kv_addresses, *text_kv_pages, pressure_owner.kv->text,
                          expected.main_kv_changes, private_host_requests);
        if (!expected.backend_kv_changes.empty()) {
            if (!pressure_owner.kv->backend || !backend_kv_addresses || !backend_kv_pages) {
                return false;
            }
            append_kv_actions(*backend_kv_addresses, *backend_kv_pages, *pressure_owner.kv->backend,
                              expected.backend_kv_changes, private_host_requests);
        }
    }

    details.shared_pressure_options.reserve(shared_pressure_options.size());
    details.shared_pressure_owner_ids.reserve(shared_pressure_owner_ids.size());
    details.shared_pressure_indices.reserve(shared_pressure_options.size());
    details.shared_pressure_generations.reserve(shared_pressure_options.size());
    for (std::size_t position = 0; position < shared_pressure_options.size(); ++position) {
        const SharedPrefixHandle* owner                   = shared_pressure_owners[position];
        const runtime::PlanningOwnerId planning_owner     = shared_pressure_owner_ids[position];
        const qwen3_5::detail::PressureDecision& proposed = *shared_pressure_options[position];
        if (owner == nullptr || ContractAccess::owner(*owner) != this ||
            planning_owner.value == std::numeric_limits<std::uint32_t>::max() ||
            std::find(details.pressure_owner_ids.begin(), details.pressure_owner_ids.end(),
                      planning_owner) != details.pressure_owner_ids.end() ||
            std::find(details.shared_pressure_owner_ids.begin(),
                      details.shared_pressure_owner_ids.end(),
                      planning_owner) != details.shared_pressure_owner_ids.end()) {
            throw std::invalid_argument("materialization shared pressure owner is invalid");
        }
        if (!valid_shared_prefix(*owner)) { return false; }
        const std::uint32_t index      = ContractAccess::index(*owner);
        const std::uint64_t generation = ContractAccess::epoch(*owner);
        if ((details.has_shared_source && index == details.shared_source_index &&
             generation == details.shared_source_generation) ||
            std::find(details.shared_pressure_indices.begin(),
                      details.shared_pressure_indices.end(),
                      index) != details.shared_pressure_indices.end()) {
            throw std::invalid_argument("materialization shared pressure owner is duplicated");
        }
        qwen3_5::detail::PressureDecision expected;
        if (proposed.evicts_continuation) {
            expected = inspect_shared_eviction_option(shared_prefix_states[index]);
        } else {
            if (!shared_pressure_decision_valid(shared_prefix_states[index], proposed,
                                                &*protection)) {
                return false;
            }
            expected = proposed;
        }
        if (expected != proposed || !expected.shared_owner) { return false; }
        details.shared_pressure_options.push_back(expected);
        details.shared_pressure_owner_ids.push_back(planning_owner);
        details.shared_pressure_indices.push_back(index);
        details.shared_pressure_generations.push_back(generation);
        pressure_needs_transfer =
            pressure_needs_transfer || !expected.transfer_requirements.empty();
        const SharedPrefixState& pressure_owner = shared_prefix_states[index];
        if (!pressure_owner.kv) { return false; }
        append_kv_actions(*text_kv_addresses, *text_kv_pages, pressure_owner.kv->text,
                          expected.main_kv_changes, shared_host_requests);
        if (!expected.backend_kv_changes.empty()) {
            if (!pressure_owner.kv->backend || !backend_kv_addresses || !backend_kv_pages) {
                return false;
            }
            append_kv_actions(*backend_kv_addresses, *backend_kv_pages, *pressure_owner.kv->backend,
                              expected.backend_kv_changes, shared_host_requests);
        }
    }

    const std::optional<detail::PressureTargetProjection> projection = evaluate_pressure_target(
        &*protection, pressure_owners, details.pressure_options, shared_pressure_owners,
        details.shared_pressure_options, &host_last_reference_releases);
    if (!projection) { return false; }
    const detail::PhysicalResources& removed = projection->unique_object_delta.removed;
    const detail::PhysicalResources& added   = projection->unique_object_delta.added;

    if (projection->source_state_fork_required &&
        details.state_fork_required != *projection->source_state_fork_required) {
        // Reference removal is monotonic, so a complete pressure target may turn Fork into Move
        // but can never turn a valid Move into Fork.  Re-derive every dependent physical fact here
        // before the target is assessed or sealed.
        if (!details.has_source ||
            details.source_mode != runtime::PrivateSourceMode::ConsumeToActive ||
            !details.state_fork_required || *projection->source_state_fork_required) {
            return false;
        }
        const SequenceState& source = continuation_states[details.source_index];
        const StateImageHandle selected =
            selected_state(source, details.reuse, details.selected_checkpoint);
        const StateReplicaResidency residency = state_store->residency(selected);
        details.state_fork_required           = false;
        if (residency == StateReplicaResidency::DeviceOnly ||
            residency == StateReplicaResidency::Both) {
            if (details.demand.reservation_added.device.state_slots == 0 ||
                details.demand.physical_peak_additional.device.state_slots == 0) {
                return false;
            }
            --details.demand.reservation_added.device.state_slots;
            --details.demand.physical_peak_additional.device.state_slots;
            if (details.demand.reservation_credit.device.state_slots ==
                std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("source StateImage Move credit overflow");
            }
            ++details.demand.reservation_credit.device.state_slots;

            if (is_masked_draft_backend(speculative_backend)) {
                const auto copy = std::find_if(
                    details.transfer_requirements.begin(), details.transfer_requirements.end(),
                    [](const runtime::ContextTransferRequirement& requirement) {
                        return requirement.resource == runtime::ContextResourceClass::State &&
                               requirement.direction ==
                                   runtime::ContextTransferDirection::DeviceToDevice;
                    });
                if (copy == details.transfer_requirements.end()) { return false; }
                details.transfer_requirements.erase(copy);
            }
        }
    }

    const auto rederive_prefix_move = [&](std::optional<bool> projected_fork, bool& planned_fork,
                                          const KVAddressSpaceStore& addresses,
                                          const LogicalKVPageStore& pages,
                                          KVAddressSpaceHandle address, std::uint32_t frontier,
                                          runtime::ContextResourceClass resource) {
        if (!projected_fork || planned_fork == *projected_fork) { return true; }
        if (!details.has_source ||
            details.source_mode != runtime::PrivateSourceMode::ConsumeToActive || !planned_fork ||
            *projected_fork || frontier == 0 ||
            frontier % static_cast<std::uint32_t>(kPagedKVPageSize) == 0) {
            return false;
        }
        const std::uint32_t required = kv_pages_for_frontier(frontier);
        if (required == 0 || required > addresses.mapped_pages(address)) { return false; }
        const LogicalKVPageHandle tail = addresses.logical_page(address, required - 1U);
        const bool device_resident     = pages.device_resident(tail);
        if (!device_resident && !pages.host_resident(tail)) { return false; }

        std::uint32_t& added  = resource == runtime::ContextResourceClass::MainKV
                                    ? details.demand.reservation_added.device.main_kv_pages
                                    : details.demand.reservation_added.device.backend_kv_pages;
        std::uint32_t& peak   = resource == runtime::ContextResourceClass::MainKV
                                    ? details.demand.physical_peak_additional.device.main_kv_pages
                                    : details.demand.physical_peak_additional.device.backend_kv_pages;
        std::uint32_t& credit = resource == runtime::ContextResourceClass::MainKV
                                    ? details.demand.reservation_credit.device.main_kv_pages
                                    : details.demand.reservation_credit.device.backend_kv_pages;
        if (added == 0 || peak == 0) { return false; }
        --added;
        --peak;
        if (device_resident) {
            if (credit == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("source KV Move credit overflow");
            }
            ++credit;
        }

        const auto copy = std::find_if(
            details.transfer_requirements.begin(), details.transfer_requirements.end(),
            [&](const runtime::ContextTransferRequirement& requirement) {
                return requirement.resource == resource &&
                       requirement.direction == runtime::ContextTransferDirection::DeviceToDevice &&
                       requirement.page_count == 1;
            });
        if (copy == details.transfer_requirements.end()) { return false; }
        details.transfer_requirements.erase(copy);
        planned_fork = false;
        return true;
    };
    if (details.has_source && details.source_index < continuation_capacity) {
        const SequenceState& source = continuation_states[details.source_index];
        if (!source.kv ||
            !rederive_prefix_move(projection->source_text_prefix_fork_required,
                                  details.text_prefix_fork_required, *text_kv_addresses,
                                  *text_kv_pages, source.kv->text, details.reuse_base,
                                  runtime::ContextResourceClass::MainKV)) {
            return false;
        }
        if (source.kv->backend) {
            if (!backend_kv_addresses || !backend_kv_pages ||
                !rederive_prefix_move(projection->source_backend_prefix_fork_required,
                                      details.backend_prefix_fork_required, *backend_kv_addresses,
                                      *backend_kv_pages, *source.kv->backend,
                                      backend_frontier_at(speculative_backend, details.reuse_base),
                                      runtime::ContextResourceClass::BackendKV)) {
                return false;
            }
        }
    }

    std::vector<HostKVAllocationRequest> host_requests;
    host_requests.reserve(shared_host_requests.size() + private_host_requests.size());
    host_requests.insert(host_requests.end(), shared_host_requests.begin(),
                         shared_host_requests.end());
    host_requests.insert(host_requests.end(), private_host_requests.begin(),
                         private_host_requests.end());
    if (!host_requests.empty()) {
        std::size_t requested_bytes = 0;
        for (const HostKVAllocationRequest& request : host_requests) {
            if (request.layout == nullptr ||
                request.pages >
                    std::numeric_limits<std::size_t>::max() / request.layout->page_stride) {
                throw std::overflow_error("materialization Host KV request size overflow");
            }
            const std::size_t bytes = request.pages * request.layout->page_stride;
            if (bytes > std::numeric_limits<std::size_t>::max() - requested_bytes) {
                throw std::overflow_error("materialization Host KV request total overflow");
            }
            requested_bytes += bytes;
        }
        if (host_kv_extents == nullptr ||
            !host_kv_extents->can_allocate_after_page_releases(
                host_releases, host_last_reference_releases, host_requests)) {
            details.blocked_host_allocation_bytes = std::max<std::size_t>(1, requested_bytes);
        }
    }

    details.demand.reservation_credit =
        checked_resource_sum(details.demand.reservation_credit, removed);
    details.demand.reservation_added =
        checked_resource_sum(details.demand.reservation_added, added);
    details.demand.physical_peak_additional = positive_resource_difference(
        checked_resource_sum(details.demand.physical_peak_additional, added), removed);
    details.demand.final_removed =
        checked_resource_sum(checked_resource_sum(details.demand.final_removed, removed),
                             projection->ownership_transfer_delta.removed);
    details.demand.final_added =
        checked_resource_sum(checked_resource_sum(details.demand.final_added, added),
                             projection->ownership_transfer_delta.added);
    details.demand.active_entitlement = checked_resource_sum(
        checked_resource_difference(details.demand.active_entitlement,
                                    projection->active_entitlement_delta.removed),
        projection->active_entitlement_delta.added);
    details.active_optional_resources = checked_resource_sum(
        details.active_optional_resources, projection->source_optional_resources_added);
    details.needs_transfer = pressure_needs_transfer || !details.transfer_requirements.empty();
    return true;
}

runtime::PreflightStatus
ProgramImpl::revalidate_materialization(const AdmissionCandidate& plan,
                                        const PreparedPromptData& prompt) const {
    if (plan.impl_ == nullptr) { return runtime::PreflightStatus::InvariantFailure; }
    if (has_context_transaction() || pending_transaction_ || has_unsettled_state_fork()) {
        return runtime::PreflightStatus::StalePolicyState;
    }

    const AdmissionCandidateImpl& details = *plan.impl_;
    if (details.blocked_host_allocation_bytes != 0) {
        return runtime::PreflightStatus::StalePolicyState;
    }
    const std::optional<MaterializationSourceProtection> protection =
        materialization_source_protection(details);
    if (!protection) { return runtime::PreflightStatus::StalePolicyState; }
    if (!physical_peak_fits(details.demand.physical_peak_additional)) {
        return runtime::PreflightStatus::StalePolicyState;
    }
    const std::size_t victim_count        = details.pressure_options.size();
    const std::size_t shared_victim_count = details.shared_pressure_options.size();
    if (victim_count > continuation_capacity || shared_victim_count > shared_prefix_capacity ||
        details.pressure_owner_ids.size() != victim_count ||
        details.pressure_indices.size() != victim_count ||
        details.pressure_generations.size() != victim_count ||
        details.shared_pressure_owner_ids.size() != shared_victim_count ||
        details.shared_pressure_indices.size() != shared_victim_count ||
        details.shared_pressure_generations.size() != shared_victim_count) {
        return runtime::PreflightStatus::InvariantFailure;
    }
    const std::uint32_t lane = details.destination.value;
    if (lane >= max_concurrency || (details.has_source && details.has_shared_source)) {
        return runtime::PreflightStatus::InvariantFailure;
    }
    if (details.destination_epoch != lane_epochs[lane] ||
        requests[lane].lifecycle != Lifecycle::Empty ||
        active_continuations[lane] < continuation_capacity) {
        return runtime::PreflightStatus::StalePolicyState;
    }

    const SequenceState* source_state = nullptr;
    if (details.has_source) {
        if (details.source_index >= continuation_capacity ||
            continuation_slots[details.source_index].role != ContinuationSlotRole::Catalogued ||
            continuation_slots[details.source_index].generation != details.source_generation) {
            return runtime::PreflightStatus::StalePolicyState;
        }
        source_state = &continuation_states[details.source_index];
    }
    const SharedPrefixState* shared_state = nullptr;
    if (details.has_shared_source) {
        if (details.shared_source_index >= shared_prefix_capacity ||
            shared_prefix_slots[details.shared_source_index].role !=
                SharedPrefixSlotRole::Catalogued ||
            shared_prefix_slots[details.shared_source_index].generation !=
                details.shared_source_generation) {
            return runtime::PreflightStatus::StalePolicyState;
        }
        shared_state = &shared_prefix_states[details.shared_source_index];
    }
    for (std::size_t victim = 0; victim < victim_count; ++victim) {
        const std::uint32_t index      = details.pressure_indices[victim];
        const std::uint64_t generation = details.pressure_generations[victim];
        if (index >= continuation_capacity ||
            continuation_slots[index].role != ContinuationSlotRole::Catalogued ||
            continuation_slots[index].generation != generation) {
            return runtime::PreflightStatus::StalePolicyState;
        }
        bool matches = false;
        if (details.pressure_options[victim].evicts_continuation) {
            matches = inspect_eviction_option(continuation_states[index]) ==
                      details.pressure_options[victim];
        } else {
            matches = pressure_decision_valid(continuation_states[index],
                                              details.pressure_options[victim], &*protection);
        }
        if (!matches) { return runtime::PreflightStatus::StalePolicyState; }
        if (details.has_source && index == details.source_index &&
            generation == details.source_generation) {
            return runtime::PreflightStatus::InvariantFailure;
        }
        for (std::size_t prior = 0; prior < victim; ++prior) {
            if (details.pressure_indices[prior] == index &&
                details.pressure_generations[prior] == generation) {
                return runtime::PreflightStatus::InvariantFailure;
            }
            if (details.pressure_owner_ids[prior] == details.pressure_owner_ids[victim]) {
                return runtime::PreflightStatus::InvariantFailure;
            }
        }
        if (details.pressure_owner_ids[victim].value == std::numeric_limits<std::uint32_t>::max()) {
            return runtime::PreflightStatus::InvariantFailure;
        }
    }
    for (std::size_t victim = 0; victim < shared_victim_count; ++victim) {
        const std::uint32_t index      = details.shared_pressure_indices[victim];
        const std::uint64_t generation = details.shared_pressure_generations[victim];
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
            shared_prefix_slots[index].generation != generation) {
            return runtime::PreflightStatus::StalePolicyState;
        }
        bool matches = false;
        if (details.shared_pressure_options[victim].evicts_continuation) {
            matches = inspect_shared_eviction_option(shared_prefix_states[index]) ==
                      details.shared_pressure_options[victim];
        } else {
            matches = shared_pressure_decision_valid(
                shared_prefix_states[index], details.shared_pressure_options[victim], &*protection);
        }
        if ((details.has_shared_source && index == details.shared_source_index &&
             generation == details.shared_source_generation) ||
            shared_prefix_states[index].active_references != 0 || !matches) {
            return runtime::PreflightStatus::StalePolicyState;
        }
        for (std::size_t prior = 0; prior < victim; ++prior) {
            if (details.shared_pressure_indices[prior] == index &&
                details.shared_pressure_generations[prior] == generation) {
                return runtime::PreflightStatus::InvariantFailure;
            }
            if (details.shared_pressure_owner_ids[prior] ==
                details.shared_pressure_owner_ids[victim]) {
                return runtime::PreflightStatus::InvariantFailure;
            }
        }
        if (details.shared_pressure_owner_ids[victim].value ==
                std::numeric_limits<std::uint32_t>::max() ||
            std::find(details.pressure_owner_ids.begin(), details.pressure_owner_ids.end(),
                      details.shared_pressure_owner_ids[victim]) !=
                details.pressure_owner_ids.end()) {
            return runtime::PreflightStatus::InvariantFailure;
        }
    }

    std::vector<ContinuationHandle> projected_private_handles;
    std::vector<const ContinuationHandle*> projected_private_owners;
    projected_private_handles.reserve(victim_count);
    projected_private_owners.reserve(victim_count);
    for (std::size_t victim = 0; victim < victim_count; ++victim) {
        projected_private_handles.push_back(ContractAccess::make_continuation(
            this, details.pressure_indices[victim], details.pressure_generations[victim]));
    }
    for (const ContinuationHandle& owner : projected_private_handles) {
        projected_private_owners.push_back(&owner);
    }
    std::vector<SharedPrefixHandle> projected_shared_handles;
    std::vector<const SharedPrefixHandle*> projected_shared_owners;
    projected_shared_handles.reserve(shared_victim_count);
    projected_shared_owners.reserve(shared_victim_count);
    for (std::size_t victim = 0; victim < shared_victim_count; ++victim) {
        projected_shared_handles.push_back(
            ContractAccess::make_shared_prefix(this, details.shared_pressure_indices[victim],
                                               details.shared_pressure_generations[victim]));
    }
    for (const SharedPrefixHandle& owner : projected_shared_handles) {
        projected_shared_owners.push_back(&owner);
    }
    const std::optional<detail::PressureTargetProjection> projected_pressure =
        evaluate_pressure_target(&*protection, projected_private_owners, details.pressure_options,
                                 projected_shared_owners, details.shared_pressure_options, nullptr);
    if (!projected_pressure) { return runtime::PreflightStatus::StalePolicyState; }

    const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (prompt_tokens != details.summary.prompt_tokens ||
        (details.vision.has_value() && !prompt.has_media()) ||
        ((source_state == nullptr && shared_state == nullptr) !=
         (details.reuse == ReusePath::Root))) {
        return runtime::PreflightStatus::InvariantFailure;
    }
    if (source_state != nullptr &&
        !qwen3_5::detail::prefix_matches(prompt, source_state->ledger,
                                         source_state->prefix_identity, details.reuse_base)) {
        return runtime::PreflightStatus::StalePolicyState;
    }
    if (shared_state != nullptr &&
        (!shared_state->identity || shared_state->identity->prefix_identity() == nullptr ||
         !qwen3_5::detail::prefix_matches(prompt, shared_state->identity->ledger(),
                                          *shared_state->identity->prefix_identity(),
                                          details.reuse_base))) {
        return runtime::PreflightStatus::StalePolicyState;
    }
    if (details.reuse == ReusePath::SharedStablePrefix &&
        (!details.selected_checkpoint ||
         details.selected_checkpoint->kind != runtime::CheckpointKind::SharedStablePrefix ||
         details.selected_checkpoint->frontier != shared_state->frontier ||
         details.selected_checkpoint->ordinal != 0)) {
        return runtime::PreflightStatus::StalePolicyState;
    }
    if (is_rewrite_checkpoint_restore(details.reuse) &&
        (!source_state->rewrite_checkpoint.valid ||
         source_state->rewrite_checkpoint.frontier != details.reuse_base ||
         details.reuse != restore_path(source_state->rewrite_checkpoint.kind))) {
        return runtime::PreflightStatus::StalePolicyState;
    }
    if (details.rewrite_disposition == RewriteCheckpointDisposition::RetainExisting &&
        (!prompt.identity.rewrite_checkpoint || source_state == nullptr ||
         !can_retain_rewrite_checkpoint(prompt, *prompt.identity.rewrite_checkpoint, *source_state,
                                        details.reuse, details.reuse_base))) {
        return runtime::PreflightStatus::StalePolicyState;
    }
    if (source_state != nullptr &&
        details.source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
        const bool projected_fork = projected_pressure->source_state_fork_required.value_or(
            protection->state_fork_required);
        const bool projected_text_fork =
            projected_pressure->source_text_prefix_fork_required.value_or(
                protection->text_prefix_fork_required);
        const bool projected_backend_fork =
            projected_pressure->source_backend_prefix_fork_required.value_or(
                protection->backend_prefix_fork_required);
        if (details.state_fork_required != projected_fork ||
            details.text_prefix_fork_required != projected_text_fork ||
            details.backend_prefix_fork_required != projected_backend_fork) {
            return runtime::PreflightStatus::StalePolicyState;
        }
    }
    if (details.reuse == ReusePath::PrivateLongAnchor &&
        (!details.selected_checkpoint ||
         details.selected_checkpoint->kind != runtime::CheckpointKind::LongAnchor ||
         std::none_of(source_state->long_anchors.begin(), source_state->long_anchors.end(),
                      [&](const LongAnchorCheckpoint& anchor) {
                          return anchor.frontier == details.selected_checkpoint->frontier &&
                                 anchor.ordinal == details.selected_checkpoint->ordinal &&
                                 state_store->valid(anchor.state);
                      }))) {
        return runtime::PreflightStatus::StalePolicyState;
    }
    return runtime::PreflightStatus::Ready;
}

detail::PhysicalResources ProgramImpl::admission_capacity() const noexcept {
    const qwen3_5::PagedKVCache* backend = backend_kv_cache();
    return detail::PhysicalResources{
        .device =
            {
                .active_lanes     = max_concurrency,
                .state_slots      = static_cast<std::uint32_t>(state_images->slot_count()),
                .main_kv_pages    = decoder->text_kv.page_pool().capacity_pages(),
                .backend_kv_pages = backend != nullptr ? backend->page_pool().capacity_pages() : 0U,
            },
        .host =
            {
                .state_slots = host_state_images ? host_state_images->capacity() : 0U,
                .kv_bytes    = host_kv_arena ? host_kv_arena->capacity_bytes() : 0U,
            },
    };
}

bool ProgramImpl::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
    if (base.impl_ == nullptr) { return false; }
    const detail::PhysicalResources capacity = admission_capacity();
    const auto fits                          = [](detail::PhysicalResources value,
                         detail::PhysicalResources limit) noexcept {
        return value.device.active_lanes <= limit.device.active_lanes &&
               value.device.state_slots <= limit.device.state_slots &&
               value.device.main_kv_pages <= limit.device.main_kv_pages &&
               value.device.backend_kv_pages <= limit.device.backend_kv_pages &&
               value.host.state_slots <= limit.host.state_slots &&
               value.host.kv_bytes <= limit.host.kv_bytes;
    };
    return fits(base.impl_->root_demand.physical_peak_additional, capacity) &&
           fits(base.impl_->root_demand.final_added, capacity);
}

bool ProgramImpl::persistent_backfill_safe(
    const RequestBasePlan& blocked_head, const AdmissionCandidate& candidate,
    std::span<const SequenceHandle> persistent_borrowers) const {
    if (blocked_head.impl_ == nullptr || candidate.impl_ == nullptr ||
        persistent_borrowers.size() >= max_concurrency) {
        return false;
    }

    detail::PhysicalResources borrowers;
    std::uint32_t observed_lanes = 0;
    for (const SequenceHandle sequence : persistent_borrowers) {
        if (!valid_sequence(sequence)) {
            throw std::logic_error("persistent backfill proof contains a stale sequence");
        }
        const std::uint32_t lane = ContractAccess::lane(sequence).value;
        const std::uint32_t bit  = 1U << lane;
        if ((observed_lanes & bit) != 0) {
            throw std::logic_error("persistent backfill proof contains a duplicate sequence");
        }
        observed_lanes |= bit;
        borrowers = checked_resource_sum(borrowers, requests[lane].active_resources);
    }
    borrowers = checked_resource_sum(borrowers, candidate.impl_->demand.active_entitlement);

    const detail::PhysicalResources capacity = admission_capacity();
    const auto fits                          = [](detail::PhysicalResources value,
                         detail::PhysicalResources limit) noexcept {
        return value.device.active_lanes <= limit.device.active_lanes &&
               value.device.state_slots <= limit.device.state_slots &&
               value.device.main_kv_pages <= limit.device.main_kv_pages &&
               value.device.backend_kv_pages <= limit.device.backend_kv_pages &&
               value.host.state_slots <= limit.host.state_slots &&
               value.host.kv_bytes <= limit.host.kv_bytes;
    };
    const detail::PhysicalDemand& head = blocked_head.impl_->root_demand;
    return fits(checked_resource_sum(borrowers, head.physical_peak_additional), capacity) &&
           fits(checked_resource_sum(borrowers, head.final_added), capacity);
}

qwen3_5::PhysicalUsageSnapshot ProgramImpl::physical_usage() const noexcept {
    const detail::PhysicalResources usage = physical_occupancy();
    return qwen3_5::PhysicalUsageSnapshot{
        .resource_revision       = resource_revision_,
        .device_state_slots      = usage.device.state_slots,
        .host_state_slots        = usage.host.state_slots,
        .device_main_kv_pages    = usage.device.main_kv_pages,
        .device_backend_kv_pages = usage.device.backend_kv_pages,
        .host_kv_bytes           = usage.host.kv_bytes,
    };
}


} // namespace ninfer::models::qwen3_5::detail
