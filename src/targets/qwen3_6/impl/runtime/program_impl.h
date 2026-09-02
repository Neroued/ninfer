#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"
#include "targets/qwen3_6/impl/runtime/rebuild_work.h"

#include "core/nvtx.h"
#include "core/startup.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/target_logprobs.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

std::uint32_t normalized_private_capacity(const ContextCacheOptions& options) {
    if (!options.max_private_continuations || *options.max_private_continuations == 0) {
        throw std::logic_error("Qwen3.6 context cache private capacity is not normalized");
    }
    return *options.max_private_continuations;
}

using Clock = std::chrono::steady_clock;

std::uint64_t elapsed_ns(Clock::time_point started) noexcept {
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count();
    return elapsed > 0 ? static_cast<std::uint64_t>(elapsed) : 0;
}

static_assert(std::is_nothrow_move_assignable_v<SpeculativeStats>);

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t kv_pages_for_frontier(std::uint32_t frontier) noexcept {
    return frontier == 0 ? 0U : 1U + (frontier - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

std::size_t context_resource_index(runtime::ContextResourceClass resource) {
    switch (resource) {
    case runtime::ContextResourceClass::State:
        return 0;
    case runtime::ContextResourceClass::MainKV:
        return 1;
    case runtime::ContextResourceClass::BackendKV:
        return 2;
    }
    throw std::logic_error("unknown context resource class");
}

runtime::PrefillWork validated_rebuild_work(runtime::PrefillWork work, std::uint32_t frontier) {
    if (work.tokens != frontier) {
        throw std::logic_error("checkpoint rebuild work does not match its frontier");
    }
    return work;
}

void validate_long_anchor_ordinals(std::span<const LongAnchorCheckpoint> anchors,
                                   std::size_t capacity) {
    if (anchors.size() > capacity) {
        throw std::logic_error("long-anchor set exceeds configured capacity");
    }
    for (std::size_t index = 0; index < anchors.size(); ++index) {
        const std::uint32_t ordinal = anchors[index].ordinal;
        if (ordinal == 0 || ordinal > capacity) {
            throw std::logic_error("long-anchor ordinal is outside configured slots");
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (anchors[previous].ordinal == ordinal) {
                throw std::logic_error("long-anchor ordinals are not unique");
            }
        }
    }
}

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

void advance_rebuild_work(SequenceState& sequence, std::uint32_t frontier,
                          std::uint32_t prefill_chunk) {
    runtime_support::advance_segmented_rebuild_work(
        sequence.rebuild_work, sequence.rebuild_tail_begin, sequence.execution_frontier, frontier,
        prefill_chunk);
}

std::optional<qwen3_6::TargetKVRequirement>
retained_requirement_after_drops(const qwen3_6::ContinuationSummary& summary,
                                 std::span<const runtime::CheckpointRef> dropped) noexcept {
    if (dropped.empty()) { return std::nullopt; }
    qwen3_6::TargetKVRequirement requirement;
    std::size_t found  = 0;
    bool surviving     = false;
    const auto include = [&](const qwen3_6::CheckpointSummary& checkpoint) {
        const auto match = std::find(dropped.begin(), dropped.end(), checkpoint.ref);
        if (match != dropped.end()) {
            ++found;
            return;
        }
        surviving = true;
        requirement.main_frontier =
            std::max(requirement.main_frontier, checkpoint.required_kv.main_frontier);
        requirement.backend_frontier =
            std::max(requirement.backend_frontier, checkpoint.required_kv.backend_frontier);
        requirement.main_pages =
            std::max(requirement.main_pages, checkpoint.required_kv.main_pages);
        requirement.backend_pages =
            std::max(requirement.backend_pages, checkpoint.required_kv.backend_pages);
    };
    if (summary.endpoint) { include(*summary.endpoint); }
    if (summary.rewrite) { include(*summary.rewrite); }
    for (const qwen3_6::CheckpointSummary& anchor : summary.long_anchors) { include(anchor); }
    if (found != dropped.size() || !surviving || requirement.main_frontier == 0 ||
        requirement.main_pages == 0) {
        return std::nullopt;
    }
    return requirement;
}

std::optional<qwen3_6::TargetKVRequirement>
retained_requirement_after_drop(const qwen3_6::ContinuationSummary& summary,
                                runtime::CheckpointRef dropped) noexcept {
    return retained_requirement_after_drops(summary,
                                            std::span<const runtime::CheckpointRef>(&dropped, 1));
}

runtime::ContextTransferRequirement
state_transfer_requirement(const StateImageHostLayout& layout,
                           runtime::ContextTransferDirection direction,
                           bool dflash_local_only = false) {
    return runtime::ContextTransferRequirement{
        .resource   = runtime::ContextResourceClass::State,
        .direction  = direction,
        .units      = 1,
        .page_count = 0,
        .work       = dflash_local_only ? dflash_local_transfer_work(layout)
                                        : state_image_transfer_work(layout),
    };
}

runtime::ContextTransferRequirement
kv_transfer_requirement(runtime::ContextResourceClass resource,
                        runtime::ContextTransferDirection direction, const HostKVPageLayout& layout,
                        std::uint32_t pages, std::uint32_t contiguous_runs = 1) {
    const TransferWork work = direction == runtime::ContextTransferDirection::DeviceToDevice
                                  ? plan_device_kv_copy_work(layout, pages)
                                  : plan_host_kv_transfer_work(layout, pages, contiguous_runs);
    return runtime::ContextTransferRequirement{
        .resource   = resource,
        .direction  = direction,
        .units      = work.payload_bytes,
        .page_count = pages,
        .work       = work,
    };
}

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

void append_pressure_transfer(qwen3_6::detail::PressureDecision& option,
                              runtime::ContextTransferRequirement requirement) {
    if (requirement.units != 0) { option.transfer_requirements.push_back(std::move(requirement)); }
}

bool pressure_state_drops_host(qwen3_6::detail::PressureStateDecision change) noexcept {
    return change == qwen3_6::detail::PressureStateDecision::DropEndpointHostDuplicate ||
           change == qwen3_6::detail::PressureStateDecision::DropRewriteHostDuplicate ||
           change == qwen3_6::detail::PressureStateDecision::DropSharedHostDuplicate;
}

bool pressure_state_demotes(qwen3_6::detail::PressureStateDecision change) noexcept {
    return change == qwen3_6::detail::PressureStateDecision::DemoteEndpointToHost ||
           change == qwen3_6::detail::PressureStateDecision::DemoteRewriteToHost ||
           change == qwen3_6::detail::PressureStateDecision::DemoteSharedToHost;
}

std::optional<StateImageHandle> pressure_state_source(qwen3_6::detail::PressureStateDecision change,
                                                      const SequenceState* sequence,
                                                      const SharedPrefixState* shared) {
    switch (change) {
    case qwen3_6::detail::PressureStateDecision::None:
        return std::nullopt;
    case qwen3_6::detail::PressureStateDecision::DropEndpointDeviceDuplicate:
    case qwen3_6::detail::PressureStateDecision::DemoteEndpointToHost:
    case qwen3_6::detail::PressureStateDecision::DropEndpointHostDuplicate:
        if (sequence == nullptr) {
            throw std::logic_error("private pressure State action targets a shared owner");
        }
        return sequence->state.read;
    case qwen3_6::detail::PressureStateDecision::DropRewriteDeviceDuplicate:
    case qwen3_6::detail::PressureStateDecision::DemoteRewriteToHost:
    case qwen3_6::detail::PressureStateDecision::DropRewriteHostDuplicate:
        if (sequence == nullptr || !sequence->rewrite_state) {
            throw std::logic_error("pressure rewrite StateImage disappeared");
        }
        return *sequence->rewrite_state;
    case qwen3_6::detail::PressureStateDecision::DropSharedDeviceDuplicate:
    case qwen3_6::detail::PressureStateDecision::DemoteSharedToHost:
    case qwen3_6::detail::PressureStateDecision::DropSharedHostDuplicate:
        if (shared == nullptr) {
            throw std::logic_error("shared pressure State action targets a private owner");
        }
        return shared->state;
    }
    throw std::logic_error("pressure StateImage action is invalid");
}

detail::PhysicalResources checked_resource_sum(detail::PhysicalResources left,
                                               detail::PhysicalResources right);
detail::PhysicalResources checked_resource_difference(detail::PhysicalResources value,
                                                      detail::PhysicalResources removed);

qwen3_6::detail::PressureDecision
combine_checkpoint_and_replica_target(qwen3_6::detail::PressureDecision checkpoint,
                                      qwen3_6::detail::PressureDecision replica) {
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
explicit_pressure_identity(const qwen3_6::detail::PressureDecision& decision) noexcept {
    std::uint64_t identity = decision.shared_owner ? 0x5348415245445052ULL : 0x5052495641544550ULL;
    const auto mix         = [&](std::uint64_t value) {
        identity ^= value;
        identity *= 1099511628211ULL;
    };
    for (const qwen3_6::detail::PressureStateDecision change : decision.state_changes) {
        mix(static_cast<std::uint8_t>(change));
    }
    const auto mix_kv = [&](std::span<const qwen3_6::detail::PressureKVDecision> changes,
                            std::uint64_t tag) {
        for (const qwen3_6::detail::PressureKVDecision& change : changes) {
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

qwen3_6::detail::PressureDecision
explicit_pressure_target(const qwen3_6::detail::PressureDecision& complete) {
    qwen3_6::detail::PressureDecision explicit_target;
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

struct KVPressureSelection {
    std::vector<qwen3_6::detail::PressureKVDecision> actions;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::uint32_t removed_device_pages = 0;
    std::size_t removed_host_bytes     = 0;
    std::size_t added_host_bytes       = 0;
    std::size_t host_bytes_remaining   = 0;
};

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
                           std::span<const qwen3_6::detail::PressureKVDecision> existing_actions,
                           ProtectedPage&& protected_page) {
    KVPressureSelection selection;
    selection.host_bytes_remaining = requested_host_bytes;
    const std::uint32_t mapped     = std::min(addresses.mapped_pages(address),
                                              mapped_limit.value_or(addresses.mapped_pages(address)));
    std::vector<std::uint8_t> selected(mapped, 0);
    const HostKVPageLayout layout = plan_host_kv_page_layout(pages.physical_pool().geometry());

    for (const qwen3_6::detail::PressureKVDecision& action : existing_actions) {
        if (action.kind == qwen3_6::detail::PressureKVDecisionKind::None ||
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

    const auto mark_action = [&](qwen3_6::detail::PressureKVDecision action) {
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
                                       qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate);
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
                .kind       = qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate,
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
                const qwen3_6::detail::PressureKVDecisionKind action =
                    require_host ? qwen3_6::detail::PressureKVDecisionKind::DropDeviceDuplicate
                                  : qwen3_6::detail::PressureKVDecisionKind::DemoteToHost;
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
            qwen3_6::detail::PressureKVDecision action{
                 .begin_page = end - count,
                 .page_count = count,
                 .kind = require_host ? qwen3_6::detail::PressureKVDecisionKind::DropDeviceDuplicate
                                      : qwen3_6::detail::PressureKVDecisionKind::DemoteToHost,
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

detail::PhysicalResources checked_resource_sum(detail::PhysicalResources left,
                                               detail::PhysicalResources right) {
    const auto add_u32 = [](std::uint32_t a, std::uint32_t b, const char* label) {
        if (b > std::numeric_limits<std::uint32_t>::max() - a) { throw std::overflow_error(label); }
        return static_cast<std::uint32_t>(a + b);
    };
    if (right.host.kv_bytes > std::numeric_limits<std::size_t>::max() - left.host.kv_bytes) {
        throw std::overflow_error("Qwen3.6 Host KV resource sum overflow");
    }
    return detail::PhysicalResources{
        .device =
            {
                .active_lanes  = add_u32(left.device.active_lanes, right.device.active_lanes,
                                         "Qwen3.6 active-lane resource sum overflow"),
                .state_slots   = add_u32(left.device.state_slots, right.device.state_slots,
                                         "Qwen3.6 StateImage resource sum overflow"),
                .main_kv_pages = add_u32(left.device.main_kv_pages, right.device.main_kv_pages,
                                         "Qwen3.6 Main KV resource sum overflow"),
                .backend_kv_pages =
                    add_u32(left.device.backend_kv_pages, right.device.backend_kv_pages,
                            "Qwen3.6 Backend KV resource sum overflow"),
            },
        .host =
            {
                .state_slots = add_u32(left.host.state_slots, right.host.state_slots,
                                       "Qwen3.6 Host StateImage resource sum overflow"),
                .kv_bytes    = left.host.kv_bytes + right.host.kv_bytes,
            },
    };
}

detail::PhysicalResources checked_resource_difference(detail::PhysicalResources value,
                                                      detail::PhysicalResources removed) {
    if (removed.device.active_lanes > value.device.active_lanes ||
        removed.device.state_slots > value.device.state_slots ||
        removed.device.main_kv_pages > value.device.main_kv_pages ||
        removed.device.backend_kv_pages > value.device.backend_kv_pages ||
        removed.host.state_slots > value.host.state_slots ||
        removed.host.kv_bytes > value.host.kv_bytes) {
        throw std::logic_error("Qwen3.6 resource subtraction underflow");
    }
    return detail::PhysicalResources{
        .device =
            {
                .active_lanes     = value.device.active_lanes - removed.device.active_lanes,
                .state_slots      = value.device.state_slots - removed.device.state_slots,
                .main_kv_pages    = value.device.main_kv_pages - removed.device.main_kv_pages,
                .backend_kv_pages = value.device.backend_kv_pages - removed.device.backend_kv_pages,
            },
        .host =
            {
                .state_slots = value.host.state_slots - removed.host.state_slots,
                .kv_bytes    = value.host.kv_bytes - removed.host.kv_bytes,
            },
    };
}

detail::PhysicalResources positive_resource_difference(detail::PhysicalResources value,
                                                       detail::PhysicalResources removed) noexcept {
    const auto positive_u32 = [](std::uint32_t left, std::uint32_t right) {
        return left > right ? left - right : 0U;
    };
    return detail::PhysicalResources{
        .device =
            {
                .active_lanes =
                    positive_u32(value.device.active_lanes, removed.device.active_lanes),
                .state_slots = positive_u32(value.device.state_slots, removed.device.state_slots),
                .main_kv_pages =
                    positive_u32(value.device.main_kv_pages, removed.device.main_kv_pages),
                .backend_kv_pages =
                    positive_u32(value.device.backend_kv_pages, removed.device.backend_kv_pages),
            },
        .host =
            {
                .state_slots = positive_u32(value.host.state_slots, removed.host.state_slots),
                .kv_bytes    = value.host.kv_bytes > removed.host.kv_bytes
                                   ? value.host.kv_bytes - removed.host.kv_bytes
                                   : 0U,
            },
    };
}

detail::PhysicalResources pressure_residual(detail::PhysicalResources deficit,
                                            const detail::PhysicalDelta& applied) {
    return positive_resource_difference(checked_resource_sum(deficit, applied.added),
                                        applied.removed);
}

std::array<std::int32_t, 3> prompt_rope_position(const PreparedPromptData& prompt,
                                                 std::uint32_t token) {
    const std::size_t tokens = prompt.token_ids.size();
    if (token >= tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("MTP bridge position is outside prepared prompt metadata");
    }
    return {prompt.positions[token], prompt.positions[tokens + token],
            prompt.positions[2 * tokens + token]};
}

schedule::MtpCausalAttentionEnvelopes mtp_causal_attention_envelopes(std::uint32_t max_frontier,
                                                                     std::uint32_t k,
                                                                     std::uint32_t capacity) {
    const auto visible = [capacity](std::uint64_t value) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, value));
    };
    schedule::MtpCausalAttentionEnvelopes out;
    out.target_verify = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + 1ULL)};
    out.batch         = out.target_verify;
    for (std::uint32_t step = 0; step + 1 < k; ++step) {
        out.ar[step] = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + step + 2ULL)};
    }
    return out;
}

schedule::DFlashEnvelopes dflash_envelopes(std::uint32_t min_frontier, std::uint32_t max_frontier,
                                           std::uint32_t k) {
    (void)min_frontier;
    return schedule::DFlashEnvelopes{
        .local  = {0, max_frontier},
        .full   = {0, max_frontier},
        .append = {0, k + 1},
    };
}

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label) {
    const auto it = std::find_if(
        family.profiles.begin(), family.profiles.end(), [&](const DecodeGraphProfile& profile) {
            return profile.batch_size == batch_size && profile.min_execution_frontier <= frontier &&
                   frontier <= profile.max_execution_frontier;
        });
    if (it == family.profiles.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage is incomplete");
    }
    return *it;
}

void validate_graph_profiles(const std::vector<GraphExecutionProfile>& profiles,
                             std::uint32_t max_frontier, const char* label) {
    if (profiles.empty() || profiles.front().min != 0 || profiles.back().max != max_frontier) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage has invalid endpoints");
    }
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        if (profiles[i].min > profiles[i].max ||
            (i != 0 && profiles[i].min != profiles[i - 1].max + 1)) {
            throw std::logic_error(std::string(label) + " CUDA Graph coverage has a gap");
        }
    }
}

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label) {
    const auto it = std::find_if(family.topologies.begin(), family.topologies.end(),
                                 [topology_class](const DecodeGraphTopology& topology) {
                                     return topology.topology_class == topology_class;
                                 });
    if (it == family.topologies.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph topology is unavailable");
    }
    return *it;
}

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label) {
    DecodeGraphTopology& topology   = select_graph_topology(family, profile.topology_class, label);
    const std::size_t profile_index = static_cast<std::size_t>(&profile - family.profiles.data());
    if (topology.installed_profile != profile_index) {
        topology.executable.update(profile.definition);
        topology.installed_profile = profile_index;
    }
    return topology.executable;
}

template <class Prepare>
void instantiate_graph_family(DecodeGraphFamily& family, const char* label, DeviceContext& device,
                              Prepare&& prepare) {
    if (family.profiles.empty()) {
        throw std::logic_error(std::string(label) + " CUDA Graph family has no profiles");
    }

    for (std::size_t i = 0; i < family.profiles.size(); ++i) {
        DecodeGraphProfile& profile = family.profiles[i];
        if (!profile.definition.ready()) {
            throw std::logic_error(std::string(label) + " CUDA Graph definition is empty");
        }
        const auto existing =
            std::find_if(family.topologies.begin(), family.topologies.end(),
                         [&](const DecodeGraphTopology& topology) {
                             return topology.topology_class == profile.topology_class;
                         });
        if (existing != family.topologies.end()) { continue; }

        family.topologies.emplace_back();
        DecodeGraphTopology& topology = family.topologies.back();
        topology.topology_class       = profile.topology_class;
        topology.executable.instantiate(profile.definition);
        topology.installed_profile = i;
    }

    const auto install_and_upload = [&](DecodeGraphTopology& topology, std::size_t profile_index) {
        DecodeGraphProfile& profile = family.profiles[profile_index];
        if (topology.installed_profile != profile_index) {
            topology.executable.update(profile.definition);
            topology.installed_profile = profile_index;
        }
        topology.executable.upload(device.stream);
        device.synchronize();
    };

    for (DecodeGraphTopology& topology : family.topologies) {
        std::optional<std::size_t> first_profile;
        for (std::size_t i = 0; i < family.profiles.size(); ++i) {
            if (family.profiles[i].topology_class == topology.topology_class) {
                if (!first_profile) {
                    first_profile = i;
                    install_and_upload(topology, i);

                    DecodeGraphProfile& profile = family.profiles[i];
                    prepare(profile.min_execution_frontier, profile.batch_size);
                    device.synchronize();
                    topology.executable.launch(device.stream);
                    device.synchronize();
                    continue;
                }
                install_and_upload(topology, i);
            }
        }
        if (!first_profile) {
            throw std::logic_error(std::string(label) + " CUDA Graph topology has no definitions");
        }
        if (topology.installed_profile != *first_profile) {
            install_and_upload(topology, *first_profile);
        }
    }
}

} // namespace

ProgramImplCore::ProgramImplCore(const LoadedModelData& model_in, const SequencePlanImpl& plan,
                                 DeviceContext& device_in, const StartupObserver& startup_observer)
    : model(model_in), device(device_in), capacity(plan.capacity), kv_capacity(plan.kv_capacity),
      max_concurrency(plan.max_concurrency), context_cache(plan.context_cache),
      continuation_capacity(normalized_private_capacity(plan.context_cache)),
      shared_prefix_capacity(plan.context_cache.max_shared_prefixes.value_or(0)),
      prefill_chunk(plan.prefill_chunk), draft_window(plan.draft_window),
      speculative_backend(plan.speculative_backend), kv_storage(plan.kv_storage),
      proposal_head(plan.proposal_head), vision_enabled(plan.features.vision),
      use_cuda_graph(plan.use_cuda_graph), causal_scoring(plan.causal_scoring),
      kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), plan.workspace.general_capacity}),
      continuation_states(continuation_capacity), continuation_slots(continuation_capacity),
      shared_prefix_states(shared_prefix_capacity), shared_prefix_slots(shared_prefix_capacity),
      round_host(sizeof(TokenId)),
      score_logprobs_host(plan.causal_scoring ? std::make_optional<PinnedHostBuffer>(
                                                    kCausalScoreTile * sizeof(float))
                                              : std::nullopt),
      ordinary_host(
          plan.speculative_backend == SpeculativeBackend::None
              ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_6::OrdinaryDecodeIngress) +
                                                     sizeof(qwen3_6::OrdinaryDecodeEgress))
              : std::nullopt),
      mtp_host(plan.speculative_backend == SpeculativeBackend::Mtp
                   ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_6::MtpDecodeIngress) +
                                                          sizeof(qwen3_6::MtpDecodeEgress))
                   : std::nullopt),
      dflash_host(plan.speculative_backend == SpeculativeBackend::DFlash
                      ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_6::DFlashDecodeIngress) +
                                                             sizeof(qwen3_6::DFlashDecodeEgress))
                      : std::nullopt),
      context_source_ready_(device_in), context_completion_(device_in),
      context_transfer_timers_{CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream),
                               CudaEventTimer(device_in, device_in.transfer_stream)} {
    if (model.weights_arena == nullptr) {
        throw std::invalid_argument("Qwen3.6 model view has no owning weight arena");
    }
    if (model.features != plan.features || model.mtp.has_value() != plan.features.mtp() ||
        model.dflash.has_value() != plan.features.dflash() ||
        model.optimized_proposal.has_value() != plan.features.optimized_proposal() ||
        model.vision.has_value() != plan.features.vision) {
        throw std::invalid_argument(
            "Qwen3.6 loaded weights do not match the frozen startup features");
    }
    if (model.mtp.has_value() && model.dflash.has_value()) {
        throw std::invalid_argument("MTP and DFlash model views are mutually exclusive");
    }
    if (model.dflash.has_value() && model.vision.has_value()) {
        throw std::invalid_argument("DFlash and Vision model views are mutually exclusive");
    }
    if (workspace_plan.general_capacity == 0 ||
        workspace_plan.vision.has_value() != vision_enabled ||
        causal_scoring != plan.persistent.score_hidden.has_value() ||
        causal_scoring != (workspace_plan.causal_score != 0) ||
        (workspace_plan.vision &&
         workspace_plan.vision->general_capacity_bytes != workspace_plan.general_capacity)) {
        throw std::invalid_argument("Qwen3.6 workspace plan does not match startup features");
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    if (!plan.context_cache.max_private_continuations || !plan.context_cache.max_shared_prefixes) {
        throw std::logic_error("Qwen3.6 context cache options are not normalized");
    }
    const std::uint64_t address_capacity64 =
        static_cast<std::uint64_t>(*plan.context_cache.max_private_continuations) +
        *plan.context_cache.max_shared_prefixes;
    if (address_capacity64 == 0 || address_capacity64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.6 KV address-space capacity exceeds uint32");
    }
    // One unpublished descriptor is reserved for the single in-flight active-capture snapshot.
    // Published private/shared address spaces remain bounded by P + S; the transaction slot lets a
    // full shared catalog replace one entry without releasing the old checkpoint before the new
    // snapshot has been prepared.
    if (address_capacity64 == std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.6 KV transaction address capacity exceeds uint32");
    }
    const auto address_capacity      = static_cast<std::uint32_t>(address_capacity64 + 1U);
    const auto logical_page_capacity = [&](const DeviceKVPagePool& pool) {
        const HostKVPageLayout host_layout = plan_host_kv_page_layout(pool.geometry());
        const std::uint64_t host_pages =
            plan.context_cache.host_kv_capacity_bytes / host_layout.page_stride;
        const std::uint64_t total = static_cast<std::uint64_t>(pool.capacity_pages()) + host_pages;
        if (total > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Qwen3.6 logical KV page capacity exceeds uint32");
        }
        return static_cast<std::uint32_t>(total);
    };

    decoder = std::make_unique<qwen3_6::DecoderState>(backing, plan.persistent.decoder);
    text_host_kv_page_stride =
        plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry()).page_stride;
    text_kv_pages = std::make_unique<LogicalKVPageStore>(
        decoder->text_kv.page_pool(), logical_page_capacity(decoder->text_kv.page_pool()));
    text_kv_addresses = std::make_unique<KVAddressSpaceStore>(
        *text_kv_pages, decoder->text_kv.execution_tables(), address_capacity,
        decoder->text_kv.execution_tables().logical_page_capacity());
    state_images =
        std::make_unique<qwen3_6::StateImageDevicePool>(backing, plan.persistent.state_images);
    if (plan.context_cache.host_state_slots != 0) {
        const std::uint64_t host_state_bytes =
            static_cast<std::uint64_t>(state_images->host_layout().image_bytes) *
            plan.context_cache.host_state_slots;
        StartupPhaseScope host_state_phase(startup_observer, StartupPhase::HostStatePin,
                                           StartupProgressUnit::Bytes, host_state_bytes);
        host_state_images = std::make_unique<qwen3_6::HostStatePool>(
            state_images->host_layout(), plan.context_cache.host_state_slots);
        host_state_phase.complete(host_state_bytes, host_state_bytes);
    }
    const std::uint64_t logical_state_capacity =
        static_cast<std::uint64_t>(state_images->slot_count()) +
        plan.context_cache.host_state_slots;
    if (logical_state_capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Qwen3.6 logical StateImage capacity exceeds uint32");
    }
    state_store = std::make_unique<StateImageStore>(
        *state_images, host_state_images.get(), static_cast<std::uint32_t>(logical_state_capacity));
    pressure_private_owner_scratch_.resize(continuation_capacity);
    pressure_shared_owner_scratch_.resize(shared_prefix_capacity);
    pressure_private_drop_scratch_.resize(continuation_capacity);
    const std::size_t pressure_checkpoint_capacity =
        2U + context_cache.max_long_anchors_per_continuation.value_or(0U);
    for (auto& dropped : pressure_private_drop_scratch_) {
        dropped.reserve(pressure_checkpoint_capacity);
    }
    pressure_state_scratch_.reserve(static_cast<std::size_t>(logical_state_capacity));
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
        replay_fold.emplace(*replay_records, state_images->linear().all_layers_view());
    }
    if (replay_records.has_value() != (speculative_backend != SpeculativeBackend::None) ||
        replay_fold.has_value() != replay_records.has_value()) {
        throw std::logic_error("ReplaySSM records do not match the sequence plan");
    }
    if (plan.persistent.dflash) {
        CyclicKVCache* local = state_images->dflash_local();
        if (local == nullptr) {
            throw std::logic_error("DFlash StateImage has no local fixed state");
        }
        dflash.emplace(backing, *plan.persistent.dflash, *local);
    }
    if (dflash.has_value() != plan.features.dflash()) {
        throw std::logic_error("DFlash state does not match the frozen sequence plan");
    }
    if (qwen3_6::PagedKVCache* backend = backend_kv_cache()) {
        backend_host_kv_page_stride =
            plan_host_kv_page_layout(backend->page_pool().geometry()).page_stride;
        backend_kv_pages = std::make_unique<LogicalKVPageStore>(
            backend->page_pool(), logical_page_capacity(backend->page_pool()));
        backend_kv_addresses = std::make_unique<KVAddressSpaceStore>(
            *backend_kv_pages, backend->execution_tables(), address_capacity,
            backend->execution_tables().logical_page_capacity());
    }
    pressure_text_page_scratch_.resize(text_kv_pages->capacity());
    pressure_text_selected_pages_.reserve(text_kv_pages->capacity());
    if (backend_kv_pages) {
        pressure_backend_page_scratch_.resize(backend_kv_pages->capacity());
        pressure_backend_selected_pages_.reserve(backend_kv_pages->capacity());
    }
    if (plan.context_cache.host_kv_capacity_bytes != 0) {
        std::vector<HostKVPageLayout> layouts;
        layouts.push_back(plan_host_kv_page_layout(decoder->text_kv.page_pool().geometry()));
        if (const qwen3_6::PagedKVCache* backend = backend_kv_cache()) {
            HostKVPageLayout backend_layout =
                plan_host_kv_page_layout(backend->page_pool().geometry());
            if (backend_layout != layouts.front()) { layouts.push_back(std::move(backend_layout)); }
        }
        StartupPhaseScope host_kv_phase(
            startup_observer, StartupPhase::HostKvPin, StartupProgressUnit::Bytes,
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes));
        host_kv_arena = std::make_unique<HostKVArena>(
            plan.context_cache.host_kv_capacity_bytes,
            std::span<const HostKVPageLayout>(layouts.data(), layouts.size()));
        host_kv_phase.complete(
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes),
            static_cast<std::uint64_t>(plan.context_cache.host_kv_capacity_bytes));
        std::size_t minimum_stride = layouts.front().page_stride;
        for (const HostKVPageLayout& layout : layouts) {
            minimum_stride = std::min(minimum_stride, layout.page_stride);
        }
        const std::size_t extent_capacity =
            plan.context_cache.host_kv_capacity_bytes / minimum_stride;
        if (extent_capacity > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Qwen3.6 Host KV extent capacity exceeds uint32");
        }
        if (extent_capacity != 0) {
            host_kv_extents = std::make_unique<HostKVExtentStore>(
                *host_kv_arena, static_cast<std::uint32_t>(extent_capacity));
        }
    }

    io = qwen3_6::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    if (io.mtp_decode.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("MTP decode frame does not match the sequence plan");
    }
    if (io.ordinary.has_value() != (speculative_backend == SpeculativeBackend::None)) {
        throw std::logic_error("ordinary decode frame does not match the sequence plan");
    }
    if (io.dflash_prefill.has_value() != (speculative_backend == SpeculativeBackend::DFlash)) {
        throw std::logic_error("DFlash prefill scratch does not match the sequence plan");
    }
    if (io.dflash_decode.has_value() != (speculative_backend == SpeculativeBackend::DFlash)) {
        throw std::logic_error("DFlash decode frame does not match the sequence plan");
    }
    prefill_hidden = plan.persistent.prefill_hidden.bind(backing);
    if (plan.persistent.score_hidden) {
        score_hidden = plan.persistent.score_hidden->bind(backing);
    }
    token_counts    = plan.persistent.token_counts.bind(backing);
    sampling_config = plan.persistent.sampling_config.bind(backing);
    active_continuations.fill(continuation_capacity);
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) { lane_epochs[lane] = 1; }
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        SequenceState& sequence = continuation_states[index];
        sequence.ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_digests.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.long_anchors.reserve(context_cache.max_long_anchors_per_continuation.value_or(0));
        // One retained shared resume source can coexist with every fixed per-request candidate.
        sequence.shared_prefix_references.reserve(8U);
    }
    materialization_ledger_.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    materialization_identity_.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    materialization_prefix_digests_.reserve(static_cast<std::size_t>(capacity) + 1ULL);

    set_device_i32(io.text_kv_table_row, 0);
    set_device_i32(io.backend_kv_table_row, 0);

    host_tokens = static_cast<TokenId*>(round_host.data());
    if (ordinary_host) {
        ordinary_host_ingress = static_cast<qwen3_6::OrdinaryDecodeIngress*>(ordinary_host->data());
        ordinary_host_egress  = reinterpret_cast<qwen3_6::OrdinaryDecodeEgress*>(
            static_cast<unsigned char*>(ordinary_host->data()) +
            sizeof(qwen3_6::OrdinaryDecodeIngress));
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
    }
    if (mtp_host) {
        mtp_host_ingress = static_cast<qwen3_6::MtpDecodeIngress*>(mtp_host->data());
        mtp_host_egress  = reinterpret_cast<qwen3_6::MtpDecodeEgress*>(
            static_cast<unsigned char*>(mtp_host->data()) + sizeof(qwen3_6::MtpDecodeIngress));
        *mtp_host_ingress = {};
        *mtp_host_egress  = {};
    }
    if (dflash_host) {
        dflash_host_ingress = static_cast<qwen3_6::DFlashDecodeIngress*>(dflash_host->data());
        dflash_host_egress  = reinterpret_cast<qwen3_6::DFlashDecodeEgress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_6::DFlashDecodeIngress));
        *dflash_host_ingress = {};
        *dflash_host_egress  = {};
    }
    if (io.dflash_prefill) {
        CUDA_CHECK(cudaMemsetAsync(io.dflash_prefill->produced_count.data, 0,
                                   io.dflash_prefill->produced_count.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream));
    device.synchronize();
    if (use_cuda_graph) {
        StartupPhaseScope graph_phase(startup_observer, StartupPhase::CudaGraphPrepare);
        prepare_graphs();
        graph_phase.complete();
    }
    work.reset();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

ProgramImplCore::~ProgramImplCore() noexcept {
    if (device.transfer_stream != nullptr) { (void)cudaStreamSynchronize(device.transfer_stream); }
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
}

std::vector<float> ProgramImplCore::causal_score(PreparedPromptData&& prompt,
                                                 std::uint32_t first_target) {
    if (!causal_scoring || !score_hidden || !score_logprobs_host ||
        workspace_plan.causal_score == 0) {
        throw std::logic_error("Program was not constructed for causal scoring");
    }
    if (speculative_backend != SpeculativeBackend::None || vision_enabled || use_cuda_graph ||
        context_cache.enabled) {
        throw std::logic_error("causal scoring Program has generation-only startup features");
    }
    const std::size_t token_count_size = prompt.token_ids.size();
    if (token_count_size < 2 || token_count_size > capacity) {
        throw std::invalid_argument("causal score token count must be in [2,capacity]");
    }
    if (first_target == 0 || first_target >= token_count_size) {
        throw std::invalid_argument("causal score first_target is outside the token window");
    }
    if (prompt.has_media()) {
        throw std::invalid_argument("causal scoring accepts text tokens only");
    }

    const auto token_count                     = static_cast<std::uint32_t>(token_count_size);
    const std::uint32_t predictor_count        = token_count - 1U;
    const std::uint32_t scored_predictor_begin = first_target - 1U;
    const std::uint32_t entitlement            = kv_pages_for_frontier(predictor_count);
    if (entitlement == 0) { throw std::logic_error("causal score has no KV entitlement"); }

    std::optional<StateImageHandle> state;
    std::optional<KVAddressSpaceHandle> address;
    const auto cleanup = [&] {
        bool released = true;
        if (address) {
            if (text_kv_addresses->active(*address)) { text_kv_addresses->deactivate(*address); }
            released = text_kv_addresses->release(*address) && released;
            address.reset();
        }
        if (state) {
            released = state_store->release(*state) && released;
            state.reset();
        }
        if (!released) { throw std::logic_error("causal score resources could not be released"); }
    };

    std::vector<float> output;
    output.reserve(token_count_size - first_target);
    std::vector<TokenId> staged_targets;
    staged_targets.reserve(kCausalScoreTile);
    std::uint32_t staged_columns = 0;

    try {
        state = state_store->reserve_reset(device.stream);
        if (!state) { throw std::bad_alloc(); }
        address = text_kv_addresses->create_active(entitlement, 0);
        if (!address) { throw std::bad_alloc(); }
        if (text_kv_addresses->bound_row(*address) != 0) {
            throw std::logic_error("causal score did not bind the unique Main KV row");
        }
        text_kv_addresses->materialize_to_tokens(*address, predictor_count, device.stream);

        const std::int32_t state_slot = state_store->physical_slot(*state);
        const auto flush              = [&] {
            if (staged_columns == 0) { return; }
            if (staged_columns != staged_targets.size() || staged_columns > kCausalScoreTile) {
                throw std::logic_error("causal score staging has an invalid shape");
            }
            work.reset();
            mark_workspace_usage(workspace_plan.causal_score);
            const auto columns = static_cast<std::int32_t>(staged_columns);
            Tensor logits      = work.alloc(DType::BF16, {TextConfig::output_rows, columns});
            Tensor target_ids  = work.alloc(DType::I32, {columns});
            Tensor logprobs    = work.alloc(DType::FP32, {columns});
            Tensor hidden      = score_hidden->slice(1, 0, columns);
            ops::linear(hidden, model.output_head, logits, device.stream);
            CUDA_CHECK(cudaMemcpyAsync(target_ids.data, staged_targets.data(), target_ids.bytes(),
                                                    cudaMemcpyHostToDevice, device.stream));
            ops::target_logprobs(logits, target_ids, TextConfig::token_domain, logprobs,
                                              device.stream);
            CUDA_CHECK(cudaMemcpyAsync(score_logprobs_host->data(), logprobs.data, logprobs.bytes(),
                                                    cudaMemcpyDeviceToHost, device.stream));
            device.synchronize();
            const auto* host = static_cast<const float*>(score_logprobs_host->data());
            output.insert(output.end(), host, host + staged_columns);
            staged_targets.clear();
            staged_columns = 0;
            work.reset();
        };

        std::uint32_t cursor = 0;
        while (cursor < predictor_count) {
            const std::uint32_t nominal = std::min(prefill_chunk, predictor_count - cursor);
            schedule::PrefillContext schedule_state{
                {device, model, work, state_images->linear(), nullptr, io, prefill_hidden,
                 prefill_chunk, proposal_head},
                decoder->text_kv.execution_view(text_kv_addresses->execution_row(*address)),
                {},
                decoder->text_kv,
                nullptr,
                nullptr,
                cursor,
                nullptr,
                nullptr,
                state_slot,
                state_slot,
                0,
                nullptr};
            mark_workspace_usage(workspace_plan.text_prefill);
            const schedule::PrefillChunkResult result = schedule::prefill_text_chunk(
                schedule_state, std::span<const TokenId>(prompt.token_ids), nominal, std::nullopt,
                false);
            if (result.finalized || result.processed_tokens == 0 ||
                result.processed_tokens > nominal) {
                throw std::logic_error("causal score Prefill made invalid progress");
            }
            const std::uint32_t chunk_begin = cursor;
            cursor += result.processed_tokens;
            text_kv_addresses->commit_frontier(*address, cursor);

            std::uint32_t selected = std::max(chunk_begin, scored_predictor_begin);
            while (selected < cursor) {
                const std::uint32_t available = cursor - selected;
                const std::uint32_t room      = kCausalScoreTile - staged_columns;
                const std::uint32_t count     = std::min(available, room);
                Tensor source =
                    prefill_hidden.slice(1, static_cast<std::int32_t>(selected - chunk_begin),
                                         static_cast<std::int32_t>(count));
                Tensor destination = score_hidden->slice(
                    1, static_cast<std::int32_t>(staged_columns), static_cast<std::int32_t>(count));
                CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                                           cudaMemcpyDeviceToDevice, device.stream));
                for (std::uint32_t column = 0; column < count; ++column) {
                    staged_targets.push_back(prompt.token_ids[selected + column + 1U]);
                }
                selected += count;
                staged_columns += count;
                if (staged_columns == kCausalScoreTile) { flush(); }
            }
        }
        flush();
        if (output.size() != token_count_size - first_target) {
            throw std::logic_error("causal score produced the wrong number of logprobs");
        }
        cleanup();
        return output;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        try {
            cleanup();
        } catch (...) {}
        throw;
    }
}

void ProgramImplCore::start_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].start();
}

void ProgramImplCore::stop_context_transfer_timer(runtime::ContextResourceClass resource) {
    context_transfer_timers_[context_resource_index(resource)].record_stop();
}

runtime::ContextTransferObservation ProgramImplCore::context_transfer_observation(
    runtime::ContextResourceClass resource, runtime::ContextTransferDirection direction,
    TransferWork work, std::uint32_t page_count, std::uint64_t state_images) const {
    const double elapsed_ns =
        static_cast<double>(
            context_transfer_timers_[context_resource_index(resource)].elapsed_ms()) *
        1'000'000.0;
    const std::uint64_t measured_ns =
        elapsed_ns >= static_cast<double>(std::numeric_limits<std::uint64_t>::max())
            ? std::numeric_limits<std::uint64_t>::max()
            : std::max<std::uint64_t>(1, static_cast<std::uint64_t>(elapsed_ns + 0.5));
    return runtime::ContextTransferObservation{
        .resource  = resource,
        .direction = direction,
        .units =
            resource == runtime::ContextResourceClass::State ? state_images : work.payload_bytes,
        .page_count = page_count,
        .work       = work,
        .elapsed_ns = measured_ns,
    };
}

std::optional<AdmissionCandidate> ProgramImplCore::inspect_admission(
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

std::optional<ProgramImplCore::MaterializationSourceProtection>
ProgramImplCore::materialization_source_protection(const ResourceCandidateState& admission) const {
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

bool ProgramImplCore::protected_materialization_page(
    const MaterializationSourceProtection* protection, const KVAddressSpaceStore& addresses,
    std::uint32_t page_offset, LogicalKVPageHandle page, bool backend) const {
    if (protection == nullptr) { return false; }
    const std::optional<KVAddressSpaceHandle>& source =
        backend ? protection->backend : protection->text;
    const std::uint32_t required = backend ? protection->backend_pages : protection->text_pages;
    return logical_page_matches_prefix(addresses, source, required, page_offset, page);
}

std::optional<qwen3_6::detail::PressureDecision> ProgramImplCore::inspect_shared_pressure_option(
    const SharedPrefixState& shared, detail::PhysicalResources deficit,
    const MaterializationSourceProtection* protection,
    const qwen3_6::detail::PressureDecision* current) const {
    if (!shared.kv || shared.active_references != 0 || deficit.device.active_lanes != 0 ||
        (current != nullptr && (current->evicts_continuation || !current->shared_owner))) {
        return std::nullopt;
    }

    qwen3_6::detail::PressureDecision option;
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
    for (const qwen3_6::detail::PressureStateDecision change : option.state_changes) {
        mix(static_cast<std::uint8_t>(change));
    }
    const auto mix_existing_kv = [&](std::span<const qwen3_6::detail::PressureKVDecision> actions,
                                     std::uint64_t tag) {
        for (const qwen3_6::detail::PressureKVDecision& action : actions) {
            mix(tag);
            mix(action.begin_page);
            mix(action.page_count);
            mix(static_cast<std::uint8_t>(action.kind));
        }
    };
    mix_existing_kv(option.main_kv_changes, 0x534d41494eULL);
    mix_existing_kv(option.backend_kv_changes, 0x534241434bULL);

    qwen3_6::detail::PressureStateDecision state_change =
        qwen3_6::detail::PressureStateDecision::None;
    if (option.state_changes.empty() &&
        (deficit.device.state_slots != 0 || deficit.host.state_slots != 0) &&
        state_store->valid(shared.state) &&
        state_store->role(shared.state) == StateImageRole::CheckpointImmutable &&
        state_store->source_pins(shared.state) == 0) {
        const StateReplicaResidency residency = state_store->residency(shared.state);
        const bool protected_state =
            protection != nullptr && protection->state && *protection->state == shared.state;
        if (deficit.host.state_slots != 0 && residency == StateReplicaResidency::Both) {
            state_change = qwen3_6::detail::PressureStateDecision::DropSharedHostDuplicate;
            ++option.effect.removed.host.state_slots;
        } else if (!protected_state && state_store->checkpoint_references(shared.state) == 1 &&
                   deficit.device.state_slots != 0 && residency == StateReplicaResidency::Both) {
            state_change = qwen3_6::detail::PressureStateDecision::DropSharedDeviceDuplicate;
        } else if (!protected_state && state_store->checkpoint_references(shared.state) == 1 &&
                   deficit.device.state_slots != 0 &&
                   residency == StateReplicaResidency::DeviceOnly && host_state_images != nullptr) {
            state_change = qwen3_6::detail::PressureStateDecision::DemoteSharedToHost;
            ++option.effect.added.host.state_slots;
            append_pressure_transfer(option, state_transfer_requirement(
                                                 host_state_images->layout(),
                                                 runtime::ContextTransferDirection::DeviceToHost));
        }
        if (state_change != qwen3_6::detail::PressureStateDecision::None) {
            if (state_change != qwen3_6::detail::PressureStateDecision::DropSharedHostDuplicate) {
                option.effect.removed.device.state_slots = 1;
            }
            option.state_changes.push_back(state_change);
            mix(static_cast<std::uint8_t>(state_change));
        }
    }

    std::size_t host_kv_remaining = deficit.host.kv_bytes;
    const auto add_kv = [&](const KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                            KVAddressSpaceHandle address, std::uint32_t requested,
                            std::vector<qwen3_6::detail::PressureKVDecision>& changes,
                            std::uint32_t& removed_dimension,
                            runtime::ContextResourceClass resource, std::uint64_t tag) {
        const bool backend           = resource == runtime::ContextResourceClass::BackendKV;
        KVPressureSelection selected = select_kv_pressure_actions(
            addresses, pages, host_kv_extents.get(),
            host_kv_arena != nullptr && host_kv_extents != nullptr, address, std::nullopt,
            requested, host_kv_remaining, resource, changes,
            [&](std::uint32_t page, LogicalKVPageHandle logical,
                qwen3_6::detail::PressureKVDecisionKind action) {
                return action != qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate &&
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
        for (const qwen3_6::detail::PressureKVDecision& action : selected.actions) {
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

std::vector<qwen3_6::detail::PressureDecision> ProgramImplCore::inspect_shared_pressure_options(
    const SharedPrefixState& shared, detail::PhysicalResources deficit,
    const MaterializationSourceProtection* protection,
    const qwen3_6::detail::PressureDecision* current) const {
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

    std::vector<qwen3_6::detail::PressureDecision> options;
    options.reserve(endpoints.size());
    for (const detail::PhysicalResources requested : endpoints) {
        std::optional<qwen3_6::detail::PressureDecision> option =
            inspect_shared_pressure_option(shared, requested, protection, current);
        if (option && std::find(options.begin(), options.end(), *option) == options.end()) {
            options.push_back(std::move(*option));
        }
    }
    return options;
}

std::optional<qwen3_6::detail::PressureDecision> ProgramImplCore::inspect_pressure_option(
    const SequenceState& sequence, detail::PhysicalResources deficit,
    const MaterializationSourceProtection* protection,
    const qwen3_6::TargetKVRequirement* retained_requirement,
    std::span<const runtime::CheckpointRef> dropped_checkpoints,
    std::span<const StateImageHandle> released_states,
    const qwen3_6::detail::PressureDecision* current) const {
    if (!sequence.kv || deficit.device.active_lanes != 0 ||
        (current != nullptr && current->evicts_continuation)) {
        return std::nullopt;
    }

    qwen3_6::detail::PressureDecision option;
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
    for (const qwen3_6::detail::PressureStateDecision change : option.state_changes) {
        mix(static_cast<std::uint8_t>(change));
    }
    const auto mix_kv = [&](std::span<const qwen3_6::detail::PressureKVDecision> actions,
                            std::uint64_t tag) {
        for (const qwen3_6::detail::PressureKVDecision& action : actions) {
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
        const qwen3_6::detail::PressureStateDecision endpoint_drop =
            rewrite ? qwen3_6::detail::PressureStateDecision::DropRewriteDeviceDuplicate
                    : qwen3_6::detail::PressureStateDecision::DropEndpointDeviceDuplicate;
        const qwen3_6::detail::PressureStateDecision endpoint_demote =
            rewrite ? qwen3_6::detail::PressureStateDecision::DemoteRewriteToHost
                    : qwen3_6::detail::PressureStateDecision::DemoteEndpointToHost;
        const qwen3_6::detail::PressureStateDecision endpoint_host_drop =
            rewrite ? qwen3_6::detail::PressureStateDecision::DropRewriteHostDuplicate
                    : qwen3_6::detail::PressureStateDecision::DropEndpointHostDuplicate;
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
        qwen3_6::detail::PressureStateDecision change =
            qwen3_6::detail::PressureStateDecision::None;
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
            change == qwen3_6::detail::PressureStateDecision::DropEndpointHostDuplicate ||
            change == qwen3_6::detail::PressureStateDecision::DropRewriteHostDuplicate;
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
                            std::vector<qwen3_6::detail::PressureKVDecision>& changes,
                            std::uint32_t& removed_dimension,
                            runtime::ContextResourceClass resource, std::uint64_t tag) {
        const bool backend           = resource == runtime::ContextResourceClass::BackendKV;
        KVPressureSelection selected = select_kv_pressure_actions(
            addresses, pages, host_kv_extents.get(),
            host_kv_arena != nullptr && host_kv_extents != nullptr, address, mapped_limit,
            requested, host_kv_remaining, resource, changes,
            [&](std::uint32_t page, LogicalKVPageHandle logical,
                qwen3_6::detail::PressureKVDecisionKind action) {
                return action != qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate &&
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
        for (const qwen3_6::detail::PressureKVDecision& action : selected.actions) {
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

std::vector<qwen3_6::detail::PressureDecision> ProgramImplCore::inspect_pressure_successors(
    const SequenceState& sequence, detail::PhysicalResources residual,
    const MaterializationSourceProtection* protection,
    const qwen3_6::detail::PressureDecision* current) const {
    std::vector<qwen3_6::detail::PressureDecision> successors;
    if (!sequence.kv || (current != nullptr && current->evicts_continuation)) { return successors; }
    const auto append_unique = [&](qwen3_6::detail::PressureDecision option) {
        if (current != nullptr && option == *current) { return; }
        if (std::find(successors.begin(), successors.end(), option) == successors.end()) {
            successors.push_back(std::move(option));
        }
    };

    const qwen3_6::ContinuationSummary summary = continuation_summary(sequence);
    std::vector<runtime::CheckpointRef> dropped =
        current != nullptr ? current->dropped_checkpoints : std::vector<runtime::CheckpointRef>{};
    const auto already_dropped = [&](runtime::CheckpointRef checkpoint) {
        return std::find(dropped.begin(), dropped.end(), checkpoint) != dropped.end();
    };
    const auto state_change_conflicts =
        [](runtime::CheckpointRef checkpoint,
           const qwen3_6::detail::PressureDecision& explicit_target) {
            const auto has = [&](qwen3_6::detail::PressureStateDecision change) {
                return std::find(explicit_target.state_changes.begin(),
                                 explicit_target.state_changes.end(),
                                 change) != explicit_target.state_changes.end();
            };
            if (checkpoint.kind == runtime::CheckpointKind::SessionEndpoint) {
                return has(qwen3_6::detail::PressureStateDecision::DropEndpointDeviceDuplicate) ||
                       has(qwen3_6::detail::PressureStateDecision::DemoteEndpointToHost) ||
                       has(qwen3_6::detail::PressureStateDecision::DropEndpointHostDuplicate);
            }
            if (checkpoint.kind == runtime::CheckpointKind::TurnClosure ||
                checkpoint.kind == runtime::CheckpointKind::ResponseReplay) {
                return has(qwen3_6::detail::PressureStateDecision::DropRewriteDeviceDuplicate) ||
                       has(qwen3_6::detail::PressureStateDecision::DemoteRewriteToHost) ||
                       has(qwen3_6::detail::PressureStateDecision::DropRewriteHostDuplicate);
            }
            return false;
        };
    const auto append_checkpoint_successor = [&](runtime::CheckpointRef checkpoint) {
        if (already_dropped(checkpoint)) { return; }
        std::vector<runtime::CheckpointRef> target_drops = dropped;
        target_drops.push_back(checkpoint);
        std::optional<qwen3_6::detail::PressureDecision> drop =
            inspect_checkpoint_drop_option(sequence, target_drops);
        if (!drop) { return; }
        if (current == nullptr ||
            (current->state_changes.empty() && current->main_kv_changes.empty() &&
             current->backend_kv_changes.empty())) {
            append_unique(std::move(*drop));
            return;
        }
        qwen3_6::detail::PressureDecision explicit_target = explicit_pressure_target(*current);
        if (state_change_conflicts(checkpoint, explicit_target)) { return; }
        const std::optional<qwen3_6::TargetKVRequirement> retained =
            retained_requirement_after_drops(summary, drop->dropped_checkpoints);
        if (!retained) { return; }
        const auto within = [](std::span<const qwen3_6::detail::PressureKVDecision> changes,
                               std::uint32_t pages) {
            return std::all_of(changes.begin(), changes.end(), [&](const auto& action) {
                return action.kind != qwen3_6::detail::PressureKVDecisionKind::None &&
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
    for (const qwen3_6::CheckpointSummary& anchor : summary.long_anchors) {
        append_checkpoint_successor(anchor.ref);
    }

    const std::optional<qwen3_6::TargetKVRequirement> retained =
        dropped.empty() ? std::optional<qwen3_6::TargetKVRequirement>()
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
        std::optional<qwen3_6::detail::PressureDecision> replica = inspect_pressure_option(
            sequence, requested, protection, retained ? &*retained : nullptr, dropped, {}, current);
        if (!replica) { continue; }
        if (dropped.empty()) {
            append_unique(std::move(*replica));
        } else {
            std::optional<qwen3_6::detail::PressureDecision> drop =
                inspect_checkpoint_drop_option(sequence, dropped);
            if (drop) {
                append_unique(
                    combine_checkpoint_and_replica_target(std::move(*drop), std::move(*replica)));
            }
        }
    }
    return successors;
}

std::vector<qwen3_6::detail::PressureDecision> ProgramImplCore::inspect_shared_pressure_successors(
    const SharedPrefixState& shared, detail::PhysicalResources residual,
    const MaterializationSourceProtection* protection,
    const qwen3_6::detail::PressureDecision* current) const {
    return inspect_shared_pressure_options(shared, residual, protection, current);
}

void ProgramImplCore::begin_pressure_page_scratch() const noexcept {
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

ProgramImplCore::PressurePageScratchSlot&
ProgramImplCore::pressure_page_scratch(const LogicalKVPageStore& store,
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

const ProgramImplCore::PressurePageScratchSlot*
ProgramImplCore::find_pressure_page_scratch(const LogicalKVPageStore& store,
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
ProgramImplCore::checkpoint_restore_requirements(const SequenceKVBundle& kv,
                                                 const qwen3_6::TargetKVRequirement& requirement,
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

bool ProgramImplCore::pressure_checkpoint_recovery_impacts(
    const ResourceCandidateState& candidate,
    std::span<const ContinuationHandle* const> private_owners,
    std::span<const qwen3_6::detail::PressureDecision* const> private_decisions,
    std::span<const runtime::PlanningOwnerId> private_owner_ids,
    std::span<const SharedPrefixHandle* const> shared_owners,
    std::span<const qwen3_6::detail::PressureDecision* const> shared_decisions,
    std::span<const runtime::PlanningOwnerId> shared_owner_ids,
    std::vector<qwen3_6::detail::PressureCheckpointRecoveryProjection>& output,
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
                                     const qwen3_6::detail::PressureKVDecision& action) -> bool {
        if (action.kind == qwen3_6::detail::PressureKVDecisionKind::None) {
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
            action.kind == qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate;
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
                                    const qwen3_6::detail::PressureDecision* decision) -> bool {
        if (decision == nullptr || decision->evicts_continuation) { return true; }
        for (const qwen3_6::detail::PressureStateDecision change : decision->state_changes) {
            std::optional<StateImageHandle> state;
            switch (change) {
            case qwen3_6::detail::PressureStateDecision::None:
                return false;
            case qwen3_6::detail::PressureStateDecision::DropEndpointDeviceDuplicate:
            case qwen3_6::detail::PressureStateDecision::DemoteEndpointToHost:
            case qwen3_6::detail::PressureStateDecision::DropEndpointHostDuplicate:
                if (sequence == nullptr) { return false; }
                state = sequence->state.read;
                break;
            case qwen3_6::detail::PressureStateDecision::DropRewriteDeviceDuplicate:
            case qwen3_6::detail::PressureStateDecision::DemoteRewriteToHost:
            case qwen3_6::detail::PressureStateDecision::DropRewriteHostDuplicate:
                if (sequence == nullptr || !sequence->rewrite_state) { return false; }
                state = *sequence->rewrite_state;
                break;
            case qwen3_6::detail::PressureStateDecision::DropSharedDeviceDuplicate:
            case qwen3_6::detail::PressureStateDecision::DemoteSharedToHost:
            case qwen3_6::detail::PressureStateDecision::DropSharedHostDuplicate:
                if (shared == nullptr) { return false; }
                state = shared->state;
                break;
            }
            const bool drops_host =
                change == qwen3_6::detail::PressureStateDecision::DropEndpointHostDuplicate ||
                change == qwen3_6::detail::PressureStateDecision::DropRewriteHostDuplicate ||
                change == qwen3_6::detail::PressureStateDecision::DropSharedHostDuplicate;
            if (!set_state_placement(*state, drops_host, !drops_host)) { return false; }
        }
        const SequenceKVBundle* kv =
            sequence != nullptr ? (sequence->kv ? &*sequence->kv : nullptr)
                                : (shared != nullptr && shared->kv ? &*shared->kv : nullptr);
        if (kv == nullptr) { return false; }
        for (const qwen3_6::detail::PressureKVDecision& action : decision->main_kv_changes) {
            if (!apply_kv_action(text_kv_addresses.get(), text_kv_pages.get(), kv->text, action)) {
                return false;
            }
        }
        for (const qwen3_6::detail::PressureKVDecision& action : decision->backend_kv_changes) {
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
            qwen3_6::ContinuationSummary& summary = scratch.continuation_summary;
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
            const qwen3_6::CheckpointSummary checkpoint =
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
            output.push_back(qwen3_6::detail::PressureCheckpointRecoveryProjection{
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

std::optional<qwen3_6::detail::PressureDecision> ProgramImplCore::inspect_checkpoint_drop_option(
    const SequenceState& sequence, std::span<const runtime::CheckpointRef> checkpoints) const {
    if (!sequence.kv || checkpoints.empty()) { return std::nullopt; }
    const qwen3_6::ContinuationSummary summary = continuation_summary(sequence);
    qwen3_6::detail::PressureDecision option;
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

    const std::optional<qwen3_6::TargetKVRequirement> remaining =
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

bool ProgramImplCore::pressure_decision_valid(
    const SequenceState& sequence, const qwen3_6::detail::PressureDecision& decision,
    const MaterializationSourceProtection* protection) const {
    if (!sequence.kv || decision.evicts_continuation || decision.shared_owner || decision.id == 0 ||
        decision.checkpoint_drops != decision.dropped_checkpoints.size()) {
        return false;
    }
    std::optional<qwen3_6::TargetKVRequirement> retained;
    if (!decision.dropped_checkpoints.empty()) {
        const std::optional<qwen3_6::detail::PressureDecision> canonical =
            inspect_checkpoint_drop_option(sequence, decision.dropped_checkpoints);
        if (!canonical || canonical->dropped_checkpoints != decision.dropped_checkpoints ||
            canonical->checkpoint_drop_effect != decision.checkpoint_drop_effect) {
            return false;
        }
        retained = retained_requirement_after_drops(continuation_summary(sequence),
                                                    decision.dropped_checkpoints);
        if (!retained) { return false; }
    } else if (decision.checkpoint_drop_effect != qwen3_6::detail::PhysicalDelta{}) {
        return false;
    }
    const auto dropped_kind = [&](runtime::CheckpointKind kind) {
        return std::any_of(
            decision.dropped_checkpoints.begin(), decision.dropped_checkpoints.end(),
            [&](runtime::CheckpointRef checkpoint) { return checkpoint.kind == kind; });
    };
    std::vector<StateImageHandle> targeted_states;
    for (const qwen3_6::detail::PressureStateDecision action : decision.state_changes) {
        const bool endpoint =
            action == qwen3_6::detail::PressureStateDecision::DropEndpointDeviceDuplicate ||
            action == qwen3_6::detail::PressureStateDecision::DemoteEndpointToHost ||
            action == qwen3_6::detail::PressureStateDecision::DropEndpointHostDuplicate;
        const bool rewrite =
            action == qwen3_6::detail::PressureStateDecision::DropRewriteDeviceDuplicate ||
            action == qwen3_6::detail::PressureStateDecision::DemoteRewriteToHost ||
            action == qwen3_6::detail::PressureStateDecision::DropRewriteHostDuplicate;
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
                              std::span<const qwen3_6::detail::PressureKVDecision> changes,
                              std::uint32_t retained_pages,
                              runtime::ContextResourceClass resource) {
        if (changes.empty()) { return true; }
        if (addresses == nullptr || pages == nullptr || !address || !addresses->valid(*address)) {
            return false;
        }
        const std::uint32_t mapped = addresses->mapped_pages(*address);
        const std::uint32_t limit  = std::min(mapped, retained_pages);
        std::vector<LogicalKVPageHandle> targeted;
        for (const qwen3_6::detail::PressureKVDecision& action : changes) {
            if (action.kind == qwen3_6::detail::PressureKVDecisionKind::None ||
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
                     action.kind != qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate)) {
                    return false;
                }
                targeted.push_back(page);
                if (action.kind == qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate) {
                    if (!pages->device_resident(page) || !pages->host_resident(page) ||
                        host_kv_extents == nullptr ||
                        !host_kv_extents->can_release_page_replica(*pages, page)) {
                        return false;
                    }
                } else if (addresses->has_active_reference(page)) {
                    return false;
                } else if (action.kind ==
                           qwen3_6::detail::PressureKVDecisionKind::DropDeviceDuplicate) {
                    if (!pages->can_drop_device_replica(page)) { return false; }
                } else if (action.kind == qwen3_6::detail::PressureKVDecisionKind::DemoteToHost) {
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

bool ProgramImplCore::shared_pressure_decision_valid(
    const SharedPrefixState& shared, const qwen3_6::detail::PressureDecision& decision,
    const MaterializationSourceProtection* protection) const {
    if (!shared.kv || shared.active_references != 0 || decision.evicts_continuation ||
        !decision.shared_owner || decision.id == 0 || !decision.dropped_checkpoints.empty() ||
        decision.checkpoint_drops != 0 ||
        decision.checkpoint_drop_effect != qwen3_6::detail::PhysicalDelta{} ||
        decision.state_changes.size() > 1) {
        return false;
    }
    if (!decision.state_changes.empty()) {
        const qwen3_6::detail::PressureStateDecision action = decision.state_changes.front();
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
                              std::span<const qwen3_6::detail::PressureKVDecision> changes,
                              runtime::ContextResourceClass resource) {
        if (changes.empty()) { return true; }
        if (addresses == nullptr || pages == nullptr || !address || !addresses->valid(*address)) {
            return false;
        }
        const std::uint32_t mapped = addresses->mapped_pages(*address);
        std::vector<LogicalKVPageHandle> targeted;
        for (const auto& action : changes) {
            if (action.kind == qwen3_6::detail::PressureKVDecisionKind::None ||
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
                     action.kind != qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate)) {
                    return false;
                }
                targeted.push_back(page);
                if (action.kind == qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate) {
                    if (!pages->device_resident(page) || !pages->host_resident(page) ||
                        host_kv_extents == nullptr ||
                        !host_kv_extents->can_release_page_replica(*pages, page)) {
                        return false;
                    }
                } else if (addresses->has_active_reference(page)) {
                    return false;
                } else if (action.kind ==
                           qwen3_6::detail::PressureKVDecisionKind::DropDeviceDuplicate) {
                    if (!pages->can_drop_device_replica(page)) { return false; }
                } else if (action.kind == qwen3_6::detail::PressureKVDecisionKind::DemoteToHost) {
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

void ProgramImplCore::publish_checkpoint_drop(SequenceState& sequence,
                                              runtime::CheckpointRef checkpoint) {
    if (!sequence.kv) { throw std::logic_error("checkpoint drop owner has no KV bundle"); }
    const qwen3_6::ContinuationSummary before = continuation_summary(sequence);
    const std::optional<qwen3_6::TargetKVRequirement> retained =
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
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        sequence.dflash_context_frontier = retained->backend_frontier;
    }
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
    refresh_state_views(sequence);
}

qwen3_6::detail::PressureDecision
ProgramImplCore::inspect_eviction_option(const SequenceState& sequence) const {
    qwen3_6::detail::PressureDecision option;
    option.id                                  = std::numeric_limits<std::uint64_t>::max();
    option.effect.removed                      = resident_resources(sequence);
    const qwen3_6::ContinuationSummary summary = continuation_summary(sequence);
    if (!sequence.kv || summary.long_anchors.size() != sequence.long_anchors.size()) {
        throw std::logic_error("eviction owner checkpoint inventory is incomplete");
    }
    option.checkpoint_drops = planning_saturating_u32(
        summary.endpoint.has_value() + summary.rewrite.has_value() + summary.long_anchors.size());
    option.evicts_continuation = true;
    return option;
}

qwen3_6::detail::PressureDecision
ProgramImplCore::inspect_shared_eviction_option(const SharedPrefixState& shared) const {
    qwen3_6::detail::PressureDecision option;
    option.id                  = std::numeric_limits<std::uint64_t>::max() - 1U;
    option.effect.removed      = resident_resources(shared);
    option.checkpoint_drops    = 1;
    option.evicts_continuation = true;
    option.shared_owner        = true;
    return option;
}

std::optional<detail::PressureTargetProjection> ProgramImplCore::evaluate_pressure_target(
    const MaterializationSourceProtection* protection,
    std::span<const ContinuationHandle* const> pressure_owners,
    std::span<const qwen3_6::detail::PressureDecision> pressure_options,
    std::span<const SharedPrefixHandle* const> shared_pressure_owners,
    std::span<const qwen3_6::detail::PressureDecision> shared_pressure_options,
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

    const auto append_pressure_targets = [&](const qwen3_6::detail::PressureDecision& option,
                                             const SequenceState* sequence,
                                             const SharedPrefixState* shared) {
        for (const qwen3_6::detail::PressureStateDecision change : option.state_changes) {
            std::optional<StateImageHandle> state;
            switch (change) {
            case qwen3_6::detail::PressureStateDecision::None:
                return false;
            case qwen3_6::detail::PressureStateDecision::DropEndpointDeviceDuplicate:
            case qwen3_6::detail::PressureStateDecision::DemoteEndpointToHost:
            case qwen3_6::detail::PressureStateDecision::DropEndpointHostDuplicate:
                if (sequence == nullptr) { return false; }
                state = sequence->state.read;
                break;
            case qwen3_6::detail::PressureStateDecision::DropRewriteDeviceDuplicate:
            case qwen3_6::detail::PressureStateDecision::DemoteRewriteToHost:
            case qwen3_6::detail::PressureStateDecision::DropRewriteHostDuplicate:
                if (sequence == nullptr || !sequence->rewrite_state) { return false; }
                state = *sequence->rewrite_state;
                break;
            case qwen3_6::detail::PressureStateDecision::DropSharedDeviceDuplicate:
            case qwen3_6::detail::PressureStateDecision::DemoteSharedToHost:
            case qwen3_6::detail::PressureStateDecision::DropSharedHostDuplicate:
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
                                      const qwen3_6::detail::PressureKVDecision& action) {
            if (action.kind == qwen3_6::detail::PressureKVDecisionKind::None) {
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
                     action.kind != qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate) ||
                    slot.pressure_targeted) {
                    return false;
                }
                slot.pressure_targeted = true;
                slot.projected         = true;
                slot.device            = pages->device_resident(page);
                slot.host              = pages->host_resident(page);
                if (action.kind == qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate) {
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
        for (const qwen3_6::detail::PressureKVDecision& action : option.main_kv_changes) {
            if (!append_pages(text_kv_addresses.get(), text_kv_pages.get(), text, action)) {
                return false;
            }
        }
        for (const qwen3_6::detail::PressureKVDecision& action : option.backend_kv_changes) {
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
        const qwen3_6::detail::PressureDecision& option = pressure_options[position];
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
        const qwen3_6::detail::PressureDecision& option = shared_pressure_options[position];
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
        const std::optional<qwen3_6::TargetKVRequirement> retained =
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
            if (!sequence.state_source_retained || sequence.state.read == sequence.state.write) {
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

std::optional<AdmissionCandidate> ProgramImplCore::seal_materialization(
    const AdmissionCandidate& admission, const PreparedPromptData& prompt,
    std::span<const ContinuationHandle* const> pressure_owners,
    std::span<const runtime::PlanningOwnerId> pressure_owner_ids,
    std::span<const qwen3_6::detail::PressureDecision* const> pressure_options,
    std::span<const SharedPrefixHandle* const> shared_pressure_owners,
    std::span<const runtime::PlanningOwnerId> shared_pressure_owner_ids,
    std::span<const qwen3_6::detail::PressureDecision* const> shared_pressure_options) {
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

bool ProgramImplCore::compose_pressure_candidate(
    ResourceCandidateState& details, std::span<const ContinuationHandle* const> pressure_owners,
    std::span<const runtime::PlanningOwnerId> pressure_owner_ids,
    std::span<const qwen3_6::detail::PressureDecision* const> pressure_options,
    std::span<const SharedPrefixHandle* const> shared_pressure_owners,
    std::span<const runtime::PlanningOwnerId> shared_pressure_owner_ids,
    std::span<const qwen3_6::detail::PressureDecision* const> shared_pressure_options) {
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
    const auto demotion_count = [](const qwen3_6::detail::PressureDecision& option) {
        const auto count = [](const auto& changes) {
            return static_cast<std::size_t>(
                std::count_if(changes.begin(), changes.end(), [](const auto& action) {
                    return action.kind == qwen3_6::detail::PressureKVDecisionKind::DemoteToHost;
                }));
        };
        return count(option.main_kv_changes) + count(option.backend_kv_changes);
    };
    std::size_t private_demotion_count = 0;
    for (const qwen3_6::detail::PressureDecision* option : pressure_options) {
        if (option == nullptr) {
            throw std::invalid_argument("materialization pressure option is null");
        }
        private_demotion_count += demotion_count(*option);
    }
    std::size_t shared_demotion_count = 0;
    for (const qwen3_6::detail::PressureDecision* option : shared_pressure_options) {
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
                                          const qwen3_6::detail::PressureKVDecision& action) {
        if (action.kind != qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate) { return; }
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
                                       std::span<const qwen3_6::detail::PressureKVDecision> changes,
                                       std::vector<HostKVAllocationRequest>& host_requests) {
        for (const qwen3_6::detail::PressureKVDecision& action : changes) {
            append_host_releases(addresses, pages, address, action);
            if (action.kind == qwen3_6::detail::PressureKVDecisionKind::DemoteToHost) {
                host_layouts.push_back(plan_host_kv_page_layout(pages.physical_pool().geometry()));
                host_requests.push_back(
                    {.layout = &host_layouts.back(), .pages = action.page_count});
            }
        }
    };
    for (std::size_t position = 0; position < pressure_options.size(); ++position) {
        const ContinuationHandle* owner                   = pressure_owners[position];
        const runtime::PlanningOwnerId planning_owner     = pressure_owner_ids[position];
        const qwen3_6::detail::PressureDecision& proposed = *pressure_options[position];
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
        qwen3_6::detail::PressureDecision expected;
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
        const qwen3_6::detail::PressureDecision& proposed = *shared_pressure_options[position];
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
        qwen3_6::detail::PressureDecision expected;
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

            if (speculative_backend == SpeculativeBackend::DFlash) {
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
ProgramImplCore::revalidate_materialization(const AdmissionCandidate& plan,
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
        !qwen3_6::detail::prefix_matches(prompt, source_state->ledger,
                                         source_state->prefix_identity, details.reuse_base)) {
        return runtime::PreflightStatus::StalePolicyState;
    }
    if (shared_state != nullptr &&
        (!shared_state->identity || shared_state->identity->prefix_identity() == nullptr ||
         !qwen3_6::detail::prefix_matches(prompt, shared_state->identity->ledger(),
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

runtime::ContextTransactionReserveStatus
ProgramImplCore::reserve_materialization(AdmissionCandidate&& plan, PreparedPromptData&& prompt,
                                         runtime::CancellationFlagView cancellation) {
    if (cancellation.requested()) { return runtime::ContextTransactionReserveStatus::Aborted; }
    const runtime::PreflightStatus preflight = revalidate_materialization(plan, prompt);
    if (preflight != runtime::PreflightStatus::Ready) {
        throw std::logic_error("materialization changed after successful preflight");
    }
    if (has_context_transaction() || pending_transaction_) {
        throw std::logic_error("Program already owns a physical transaction");
    }
    if (plan.impl_ == nullptr) {
        throw std::invalid_argument("materialization reservation is invalid");
    }

    const AdmissionCandidateImpl& details = *plan.impl_;
    const std::uint32_t lane              = details.destination.value;
    if (lane >= max_concurrency || details.destination_epoch != lane_epochs[lane] ||
        requests[lane].lifecycle != Lifecycle::Empty ||
        active_continuations[lane] < continuation_capacity) {
        throw std::logic_error("materialization activation is stale");
    }

    const SequenceState* source_state =
        details.has_source ? &continuation_states[details.source_index] : nullptr;
    const SharedPrefixState* shared_state =
        details.has_shared_source ? &shared_prefix_states[details.shared_source_index] : nullptr;
    MaterializationTransaction transaction;
    transaction.id                  = next_materialization_id_++;
    transaction.destination         = details.destination;
    transaction.has_source          = details.has_source;
    transaction.has_shared_source   = details.has_shared_source;
    transaction.source_mode         = details.source_mode;
    transaction.source_index        = details.has_source ? details.source_index : 0;
    transaction.source_generation   = details.has_source ? details.source_generation : 0;
    transaction.shared_source_index = details.has_shared_source ? details.shared_source_index : 0;
    transaction.shared_source_generation =
        details.has_shared_source ? details.shared_source_generation : 0;
    if (source_state != nullptr) {
        transaction.source_result.emplace();
        transaction.source_result->final_summary.emplace();
        transaction.source_result->final_summary->long_anchors.reserve(
            source_state->long_anchors.size());
    }
    if (shared_state != nullptr) { transaction.shared_source_result.emplace(); }
    const std::size_t victim_count        = details.pressure_options.size();
    const std::size_t shared_victim_count = details.shared_pressure_options.size();
    transaction.victim_count              = victim_count;
    transaction.victim_indices.resize(victim_count);
    transaction.victim_generations.resize(victim_count);
    transaction.victim_released.resize(victim_count, false);
    transaction.pressure.reserve(victim_count);
    transaction.pressure_results.resize(victim_count);
    transaction.shared_victim_count = shared_victim_count;
    transaction.shared_victim_indices.resize(shared_victim_count);
    transaction.shared_victim_generations.resize(shared_victim_count);
    transaction.shared_victim_released.resize(shared_victim_count, false);
    transaction.shared_pressure_results.resize(shared_victim_count);
    transaction.shared_pressure.reserve(shared_victim_count);
    if (victim_count + shared_victim_count > (std::numeric_limits<std::size_t>::max() - 3U) / 3U) {
        throw std::overflow_error("materialization transfer observation capacity overflow");
    }
    transaction.transfer_observations.reserve(3U * (victim_count + shared_victim_count) + 3U);
    const SequenceKVBundle* source_kv =
        source_state != nullptr
            ? (source_state->kv ? &*source_state->kv : nullptr)
            : (shared_state != nullptr && shared_state->kv ? &*shared_state->kv : nullptr);
    if ((source_state != nullptr || shared_state != nullptr) && source_kv == nullptr) {
        throw std::logic_error("materialization source has no KV address space");
    }
    if (source_kv != nullptr) {
        const std::uint32_t text_pages = text_kv_addresses->mapped_pages(source_kv->text);
        transaction.text_restores.reserve(text_pages);
        transaction.text_restore_destinations.reserve(text_pages);
        if (source_kv->backend) {
            const std::uint32_t backend_pages =
                backend_kv_addresses->mapped_pages(*source_kv->backend);
            transaction.backend_restores.reserve(backend_pages);
            transaction.backend_restore_destinations.reserve(backend_pages);
        }
    }
    for (std::size_t victim = 0; victim < victim_count; ++victim) {
        const std::uint32_t index      = details.pressure_indices[victim];
        const std::uint64_t generation = details.pressure_generations[victim];
        if (details.has_source && index == transaction.source_index &&
            generation == transaction.source_generation) {
            throw std::logic_error("materialization source was also selected as a victim");
        }
        for (std::size_t prior = 0; prior < victim; ++prior) {
            if (transaction.victim_indices[prior] == index &&
                transaction.victim_generations[prior] == generation) {
                throw std::logic_error("materialization victim capability is duplicated");
            }
        }
        transaction.victim_indices[victim]         = index;
        transaction.victim_generations[victim]     = generation;
        transaction.pressure_results[victim].owner = details.pressure_owner_ids[victim];
        transaction.pressure_results[victim].final_summary.emplace();
        transaction.pressure_results[victim].final_summary->long_anchors.reserve(
            continuation_states[index].long_anchors.size());
        transaction.pressure.push_back(MaterializationTransaction::PressureWork{
            .option                  = details.pressure_options[victim],
            .continuation_index      = index,
            .continuation_generation = generation,
        });
        prepare_pressure_bookkeeping(transaction.pressure.back());
    }
    for (std::size_t victim = 0; victim < shared_victim_count; ++victim) {
        const std::uint32_t index      = details.shared_pressure_indices[victim];
        const std::uint64_t generation = details.shared_pressure_generations[victim];
        if ((details.has_shared_source && index == transaction.shared_source_index &&
             generation == transaction.shared_source_generation) ||
            shared_prefix_states[index].active_references != 0) {
            throw std::logic_error("materialization shared source was also selected as a victim");
        }
        for (std::size_t prior = 0; prior < victim; ++prior) {
            if (transaction.shared_victim_indices[prior] == index &&
                transaction.shared_victim_generations[prior] == generation) {
                throw std::logic_error("materialization shared victim capability is duplicated");
            }
        }
        transaction.shared_victim_indices[victim]     = index;
        transaction.shared_victim_generations[victim] = generation;
        transaction.shared_pressure_results[victim].owner =
            details.shared_pressure_owner_ids[victim];
        transaction.shared_pressure.push_back(MaterializationTransaction::PressureWork{
            .option                  = details.shared_pressure_options[victim],
            .continuation_index      = index,
            .continuation_generation = generation,
            .shared_owner            = true,
        });
        prepare_pressure_bookkeeping(transaction.shared_pressure.back());
    }
    if (transaction.id == 0) { transaction.id = next_materialization_id_++; }

    if (!details.has_source || details.source_mode == runtime::PrivateSourceMode::Retain) {
        for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
            if (continuation_slots[index].role != ContinuationSlotRole::Free) { continue; }
            transaction.root_continuation_index = index;
            break;
        }
        if (!transaction.root_continuation_index) {
            const auto eviction =
                std::find_if(details.pressure_options.begin(), details.pressure_options.end(),
                             [](const qwen3_6::detail::PressureDecision& option) {
                                 return option.evicts_continuation;
                             });
            if (eviction == details.pressure_options.end()) {
                throw std::logic_error(
                    "preserving materialization has no continuation destination");
            }
            const std::size_t position =
                static_cast<std::size_t>(eviction - details.pressure_options.begin());
            transaction.root_continuation_index = transaction.victim_indices[position];
            transaction.root_waiting_for_victim = true;
        }
    }

    const auto host_started = Clock::now();
    transaction.plan.emplace(std::move(plan));
    AdmissionCandidateImpl& request_plan = *transaction.plan->impl_;
    RequestControl& request              = requests[lane];
    try {
        const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
        if (prompt_tokens != request_plan.summary.prompt_tokens ||
            (request_plan.vision.has_value() && !prompt.has_media())) {
            throw std::invalid_argument("request plan does not describe the prepared prompt");
        }
        if (prompt.identity.rewrite_checkpoint &&
            (prompt.identity.rewrite_checkpoint->frontier == 0 ||
             prompt.identity.rewrite_checkpoint->frontier > prompt_tokens)) {
            throw std::invalid_argument("prepared prompt has an invalid rewrite checkpoint");
        }
        const bool suffix_has_visual = std::any_of(
            prompt.token_types.begin() + static_cast<std::ptrdiff_t>(request_plan.reuse_base),
            prompt.token_types.end(), [](std::uint8_t type) { return type != 0; });
        if (suffix_has_visual != request_plan.vision.has_value()) {
            throw std::invalid_argument(
                "request plan does not describe the prompt suffix modality");
        }
        if (((source_state == nullptr && shared_state == nullptr) !=
             (request_plan.reuse == ReusePath::Root))) {
            throw std::logic_error("materialization source does not match the selected reuse path");
        }
        if (source_state != nullptr &&
            !qwen3_6::detail::prefix_matches(prompt, source_state->ledger,
                                             source_state->prefix_identity,
                                             request_plan.reuse_base)) {
            throw std::logic_error("planned resident prefix is no longer reusable");
        }
        if (shared_state != nullptr &&
            (!shared_state->identity || shared_state->identity->prefix_identity() == nullptr ||
             !qwen3_6::detail::prefix_matches(prompt, shared_state->identity->ledger(),
                                              *shared_state->identity->prefix_identity(),
                                              request_plan.reuse_base))) {
            throw std::logic_error("planned shared prefix is no longer reusable");
        }
        if (request_plan.reuse == ReusePath::SharedStablePrefix &&
            (!request_plan.selected_checkpoint ||
             request_plan.selected_checkpoint->kind !=
                 runtime::CheckpointKind::SharedStablePrefix ||
             request_plan.selected_checkpoint->frontier != shared_state->frontier ||
             request_plan.selected_checkpoint->ordinal != 0)) {
            throw std::logic_error("planned shared-prefix checkpoint is unavailable");
        }
        if (is_rewrite_checkpoint_restore(request_plan.reuse) &&
            (!source_state->rewrite_checkpoint.valid ||
             source_state->rewrite_checkpoint.frontier != request_plan.reuse_base ||
             request_plan.reuse != restore_path(source_state->rewrite_checkpoint.kind))) {
            throw std::logic_error("planned rewrite checkpoint is unavailable");
        }
        if (request_plan.reuse == ReusePath::PrivateLongAnchor &&
            (!request_plan.selected_checkpoint ||
             request_plan.selected_checkpoint->kind != runtime::CheckpointKind::LongAnchor ||
             std::none_of(source_state->long_anchors.begin(), source_state->long_anchors.end(),
                          [&](const LongAnchorCheckpoint& anchor) {
                              return anchor.frontier ==
                                         request_plan.selected_checkpoint->frontier &&
                                     anchor.ordinal == request_plan.selected_checkpoint->ordinal &&
                                     state_store->valid(anchor.state);
                          }))) {
            throw std::logic_error("planned long-anchor checkpoint is unavailable");
        }
        if (request_plan.rewrite_disposition == RewriteCheckpointDisposition::RetainExisting &&
            (!prompt.identity.rewrite_checkpoint || source_state == nullptr ||
             !can_retain_rewrite_checkpoint(prompt, *prompt.identity.rewrite_checkpoint,
                                            *source_state, request_plan.reuse,
                                            request_plan.reuse_base))) {
            throw std::logic_error("planned rewrite checkpoint retention is unavailable");
        }
        if (request_plan.rewrite_disposition ==
                RewriteCheckpointDisposition::ReplaceAtCommittedFrontier &&
            (!prompt.identity.rewrite_checkpoint ||
             std::none_of(request_plan.capture_groups.begin(), request_plan.capture_groups.end(),
                          [&](const CaptureGroup& group) {
                              return group.rewrite &&
                                     *group.rewrite == prompt.identity.rewrite_checkpoint->kind &&
                                     group.frontier == prompt.identity.rewrite_checkpoint->frontier;
                          }))) {
            throw std::logic_error("planned rewrite checkpoint capture is invalid");
        }
        for (const CaptureGroup& group : request_plan.capture_groups) {
            const bool base_shared_promotion = group.frontier == request_plan.reuse_base &&
                                               group.shared && !group.rewrite && !group.long_anchor;
            if (!group.identity ||
                (group.frontier <= request_plan.reuse_base && !base_shared_promotion) ||
                group.frontier > prompt_tokens ||
                group.identity->shortlist_key.frontier != group.frontier ||
                group.identity->prefix_identity() == nullptr ||
                !qwen3_6::detail::prefix_matches(prompt, group.identity->ledger(),
                                                 *group.identity->prefix_identity(),
                                                 group.frontier)) {
                throw std::logic_error("planned capture identity is invalid");
            }
        }

        if (request.prefill) {
            throw std::logic_error("free request lane retained prefill bookkeeping");
        }
        if (request_plan.vision) {
            std::vector<bool> used(prompt.media_payloads.size(), false);
            for (const VisionUseSpan& use : request_plan.vision->uses) {
                if (use.prepared_item_index >= used.size()) {
                    throw std::logic_error("Vision plan references a missing media payload");
                }
                used[use.prepared_item_index] = true;
            }
            for (std::size_t index = 0; index < used.size(); ++index) {
                if (!used[index]) {
                    prompt.media_payloads[index].reset();
                    continue;
                }
            }
            VisionPrefillPlan& vision      = *request_plan.vision;
            const std::uint32_t first_item = vision.uses.front().prepared_item_index;
            if (!vision.control_plan) {
                throw std::logic_error("Vision suffix plan has no prepared metadata");
            }
            auto control = std::make_shared<qwen3_6::VisionControl>(
                qwen3_6::build_vision_control(prompt, *vision.control_plan, first_item));
            for (VisionUseSpan& use : vision.uses) {
                if (use.prepared_item_index < first_item) {
                    throw std::logic_error("Vision suffix item order changed during admission");
                }
                use.control_index = use.prepared_item_index - first_item;
                if (use.control_index >= control->items.size()) {
                    throw std::logic_error("Vision suffix control does not cover a planned item");
                }
            }
            vision.control = std::move(control);
            vision.control_plan.reset();
        }
        if (prompt.has_media() && !request_plan.vision) { prompt.release_all_media_payloads(); }

        materialization_ledger_.assign(prompt.token_ids.begin(), prompt.token_ids.end());
        materialization_identity_.assign(prompt);
        materialization_prefix_digests_.assign(prompt);

        const std::uint32_t initial_mtp_extent =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min({draft_window,
                            request_plan.summary.effective_output_tokens > 1
                                ? request_plan.summary.effective_output_tokens - 2
                                : 0U,
                            capacity - prompt_tokens > 0 ? capacity - prompt_tokens - 1 : 0U})
                : 0U;
        RequestControl::Prefill prefill{
            .prompt             = std::move(prompt),
            .vision_plan        = std::move(request_plan.vision),
            .vision             = nullptr,
            .capture_groups     = std::move(request_plan.capture_groups),
            .base               = request_plan.reuse_base,
            .cursor             = request_plan.reuse_base,
            .prompt_tokens      = prompt_tokens,
            .initial_mtp_extent = initial_mtp_extent,
            .elapsed_seconds    = 0.0,
            .prepare_mtp        = request_plan.prepare_mtp,
            .reuse              = request_plan.reuse,
            .mtp_bridge         = request_plan.mtp_bridge,
        };
        request.prefill.emplace(std::move(prefill));
        if (request.prefill->vision_plan) {
            if (!workspace_plan.vision) {
                throw std::logic_error("Vision prefill has no startup workspace plan");
            }
            request.prefill->vision = std::make_unique<schedule::VisionPrefillSession>(
                device, model, DeviceSpan{workspace_storage.base(), workspace_storage.capacity()},
                *workspace_plan.vision, request.prefill->prompt, *request.prefill->vision_plan,
                vision_handoff_peak_bytes);
        }
        request.prefill->elapsed_seconds =
            std::chrono::duration<double>(Clock::now() - host_started).count();
        static_assert(std::is_nothrow_move_constructible_v<MaterializationTransaction>);
        if (transaction.root_continuation_index && !transaction.root_waiting_for_victim) {
            ContinuationSlot& destination =
                continuation_slots[*transaction.root_continuation_index];
            if (destination.role != ContinuationSlotRole::Free) {
                throw std::logic_error("materialization continuation destination changed");
            }
            destination.role = ContinuationSlotRole::ReservedMaterialization;
        }
        advance_resource_revision();
        context_transaction_.emplace<MaterializationTransaction>(std::move(transaction));
        return runtime::ContextTransactionReserveStatus::Reserved;
    } catch (...) {
        release_materialization_staging(transaction);
        throw;
    }
}

void ProgramImplCore::release_materialization_staging(
    MaterializationTransaction& transaction) noexcept {
    const std::uint32_t lane = transaction.destination.value;
    if (lane < max_concurrency && requests[lane].lifecycle == Lifecycle::Empty) {
        requests[lane].prefill.reset();
    }
    for (std::size_t position = transaction.shared_pressure_cursor;
         position < transaction.shared_pressure.size(); ++position) {
        MaterializationTransaction::PressureWork& work = transaction.shared_pressure[position];
        if (work.submitted) {
            try {
                context_completion_.synchronize();
            } catch (...) { std::terminate(); }
        }
        abort_pressure_work(work);
    }

    for (std::size_t position = transaction.pressure_cursor; position < transaction.pressure.size();
         ++position) {
        MaterializationTransaction::PressureWork& work = transaction.pressure[position];
        if (work.submitted) {
            try {
                context_completion_.synchronize();
            } catch (...) { std::terminate(); }
        }
        abort_pressure_work(work);
    }

    abort_materialization_transfers(transaction);
    transaction.backend_retained_tail_backup.reset();
    transaction.text_retained_tail_backup.reset();
    transaction.backend_retained_tail.reset();
    transaction.text_retained_tail.reset();
    transaction.backend_prefix_fork.reset();
    transaction.text_prefix_fork.reset();
    transaction.backend_source_restore_reservation.reset();
    transaction.text_source_restore_reservation.reset();
    transaction.backend_activation.reset();
    transaction.text_activation.reset();
    if (transaction.root_backend_address && backend_kv_addresses) {
        (void)backend_kv_addresses->release(*transaction.root_backend_address);
        transaction.root_backend_address.reset();
    }
    if (transaction.root_text_address && text_kv_addresses) {
        (void)text_kv_addresses->release(*transaction.root_text_address);
        transaction.root_text_address.reset();
    }
    if (transaction.state_fork_destination) {
        if (state_store) { (void)state_store->release(*transaction.state_fork_destination); }
        transaction.state_fork_destination.reset();
    }
    for (std::size_t index = 0; index < transaction.reserved_state_count; ++index) {
        if (state_store) { (void)state_store->release(transaction.reserved_states[index]); }
        transaction.reserved_states[index] = {};
    }
    transaction.reserved_state_count = 0;

    if (transaction.root_continuation_index) {
        const std::uint32_t index = *transaction.root_continuation_index;
        if (index < continuation_capacity &&
            continuation_slots[index].role == ContinuationSlotRole::ReservedMaterialization) {
            release_continuation_slot_best_effort(index);
        }
        transaction.root_continuation_index.reset();
    }
    transaction.prepared                       = false;
    transaction.prefix_tail_submitted          = false;
    transaction.retained_tail_backup_submitted = false;
    transaction.prefix_forks_ready             = false;
    materialization_ledger_.clear();
    materialization_identity_.clear();
    materialization_prefix_digests_.clear();
}

void ProgramImplCore::prepare_consumed_source(MaterializationTransaction& transaction) {
    if (transaction.source_prepared || !transaction.plan || transaction.plan->impl_ == nullptr) {
        throw std::logic_error("materialization source preparation state is invalid");
    }
    transaction.source_prepared           = true;
    const AdmissionCandidateImpl& details = *transaction.plan->impl_;
    if (!transaction.has_source ||
        details.source_mode != runtime::PrivateSourceMode::ConsumeToActive) {
        return;
    }
    if (transaction.source_index >= continuation_capacity ||
        continuation_slots[transaction.source_index].role != ContinuationSlotRole::Catalogued ||
        continuation_slots[transaction.source_index].generation != transaction.source_generation) {
        throw std::logic_error("materialization source changed before dependency release");
    }
    SequenceState& source = continuation_states[transaction.source_index];
    if (!source.kv || details.reuse == ReusePath::Root ||
        details.reuse == ReusePath::SharedStablePrefix) {
        throw std::logic_error("consumed materialization source is incomplete");
    }

    const detail::PhysicalResources before = resident_resources(source);
    const auto retained_state              = [&](StateImageHandle handle) {
        if (source.endpoint_valid && source.state.read == handle) { return true; }
        if (source.rewrite_state && *source.rewrite_state == handle) { return true; }
        return std::any_of(
            source.long_anchors.begin(), source.long_anchors.end(),
            [&](const LongAnchorCheckpoint& anchor) { return anchor.state == handle; });
    };
    const auto release_if_unreferenced = [&](StateImageHandle handle) {
        if (!state_store->valid(handle) || retained_state(handle) ||
            state_store->checkpoint_references(handle) != 0) {
            return;
        }
        if (!state_store->release(handle)) {
            throw std::logic_error("superseded source StateImage remained pinned");
        }
    };

    if (source.endpoint_valid && source.execution_frontier > details.reuse_base) {
        const StateImageHandle endpoint = source.state.read;
        source.endpoint_valid           = false;
        source.state                    = {};
        source.tail_hidden              = {};
        source.tail_hidden_valid        = false;
        release_if_unreferenced(endpoint);
    }
    for (std::size_t index = source.long_anchors.size(); index != 0; --index) {
        LongAnchorCheckpoint& anchor = source.long_anchors[index - 1U];
        if (anchor.frontier <= details.reuse_base) { continue; }
        const StateImageHandle state = anchor.state;
        state_store->release_checkpoint_reference(state);
        source.long_anchors.erase(source.long_anchors.begin() +
                                  static_cast<std::ptrdiff_t>(index - 1U));
        release_if_unreferenced(state);
    }
    if (details.reuse == ReusePath::PrivateEndpoint &&
        details.rewrite_disposition != RewriteCheckpointDisposition::RetainExisting &&
        source.rewrite_state) {
        const StateImageHandle rewrite = *source.rewrite_state;
        state_store->release_checkpoint_reference(rewrite);
        source.rewrite_state.reset();
        source.rewrite_checkpoint        = {};
        source.rewrite_checkpoint_hidden = {};
        release_if_unreferenced(rewrite);
    }

    struct TruncateTarget {
        KVAddressSpaceStore* addresses = nullptr;
        LogicalKVPageStore* pages      = nullptr;
        KVAddressSpaceHandle address;
        std::uint32_t frontier        = 0;
        bool prefix_fork              = false;
        bool releases_stale_host_tail = false;
    };

    std::array<TruncateTarget, 2> targets{};
    std::size_t target_count = 0;
    targets[target_count++]  = TruncateTarget{
         .addresses   = text_kv_addresses.get(),
         .pages       = text_kv_pages.get(),
         .address     = source.kv->text,
         .frontier    = details.reuse_base,
         .prefix_fork = details.text_prefix_fork_required,
    };
    if (source.kv->backend) {
        targets[target_count++] = TruncateTarget{
            .addresses   = backend_kv_addresses.get(),
            .pages       = backend_kv_pages.get(),
            .address     = *source.kv->backend,
            .frontier    = backend_frontier_at(speculative_backend, details.reuse_base),
            .prefix_fork = details.backend_prefix_fork_required,
        };
    }

    std::array<HostKVPageReplicaRelease, 2> host_tail_releases{};
    std::size_t host_tail_release_count = 0;
    for (TruncateTarget& target : std::span(targets.data(), target_count)) {
        if (target.prefix_fork) {
            if (!target.addresses->can_truncate_inactive_prefix(target.address, target.frontier)) {
                throw std::logic_error("COW source KV suffix is not releasable");
            }
            continue;
        }
        const std::uint32_t target_pages = kv_pages_for_frontier(target.frontier);
        if (target_pages != 0) {
            const LogicalKVPageHandle tail =
                target.addresses->logical_page(target.address, target_pages - 1U);
            const std::uint32_t columns =
                target.frontier -
                (target_pages - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
            target.releases_stale_host_tail = columns != target.pages->committed_columns(tail) &&
                                              target.pages->host_resident(tail);
            if (target.releases_stale_host_tail) {
                if (!host_kv_extents || host_tail_release_count == host_tail_releases.size()) {
                    throw std::logic_error("stale source Host KV tail is not releasable");
                }
                host_tail_releases[host_tail_release_count++] =
                    HostKVPageReplicaRelease{.pages = target.pages, .page = tail};
            }
        }
        if (!target.addresses->can_destructive_truncate_inactive(target.address, target.frontier,
                                                                 target.releases_stale_host_tail)) {
            throw std::logic_error("consumed source KV is not destructively truncatable");
        }
    }
    if (host_tail_release_count != 0) {
        const std::span<const HostKVPageReplicaRelease> releases(host_tail_releases.data(),
                                                                 host_tail_release_count);
        if (!host_kv_extents->release_page_replicas(releases)) {
            throw std::logic_error("stale source Host KV tails cannot be released atomically");
        }
    }
    for (TruncateTarget& target : std::span(targets.data(), target_count)) {
        if (target.prefix_fork) {
            target.addresses->truncate_inactive_prefix(target.address, target.frontier);
        } else {
            target.addresses->destructive_truncate_inactive(target.address, target.frontier);
        }
        target.addresses->set_checkpoint_requirement(target.address, target.frontier);
    }
    source.text_kv_valid = details.reuse_base;
    if (speculative_backend == SpeculativeBackend::Mtp) {
        source.mtp_kv_valid = backend_frontier_at(speculative_backend, details.reuse_base);
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        source.dflash_context_frontier = details.reuse_base;
    }
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
    refresh_state_views(source);

    const detail::PhysicalResources after   = resident_resources(source);
    const detail::PhysicalResources removed = checked_resource_difference(before, after);
    (void)checked_resource_difference(details.demand.final_removed, removed);
}

void ProgramImplCore::prepare_materialization(MaterializationTransaction& transaction) {
    if (transaction.prepared || !transaction.plan ||
        transaction.destination.value >= max_concurrency ||
        !requests[transaction.destination.value].prefill || !transaction.source_prepared) {
        throw std::logic_error("materialization preparation state is invalid");
    }
    for (std::size_t victim = 0; victim < transaction.victim_count; ++victim) {
        if (!transaction.victim_released[victim]) {
            throw std::logic_error("materialization preparation has an unreleased victim");
        }
    }

    const auto prepare_started            = Clock::now();
    const AdmissionCandidateImpl& details = *transaction.plan->impl_;
    const detail::PhysicalDemand& demand  = details.demand;
    const std::uint32_t lane              = transaction.destination.value;
    if (transaction.has_source &&
        (transaction.source_index >= continuation_capacity ||
         continuation_slots[transaction.source_index].role != ContinuationSlotRole::Catalogued ||
         continuation_slots[transaction.source_index].generation !=
             transaction.source_generation)) {
        throw std::logic_error("materialization source changed during capacity preparation");
    }
    if (transaction.has_shared_source &&
        (transaction.shared_source_index >= shared_prefix_capacity ||
         shared_prefix_slots[transaction.shared_source_index].role !=
             SharedPrefixSlotRole::Catalogued ||
         shared_prefix_slots[transaction.shared_source_index].generation !=
             transaction.shared_source_generation)) {
        throw std::logic_error("materialization shared source changed during capacity preparation");
    }
    SequenceState* source_state =
        transaction.has_source ? &continuation_states[transaction.source_index] : nullptr;
    SharedPrefixState* shared_state = transaction.has_shared_source
                                          ? &shared_prefix_states[transaction.shared_source_index]
                                          : nullptr;
    if (source_state != nullptr && resident_resources(*source_state).device.state_slots == 0 &&
        resident_resources(*source_state).host.state_slots == 0) {
        throw std::logic_error("materialization source has no resident state");
    }

    std::uint32_t state_count = demand.reservation_added.device.state_slots;
    std::optional<StateImageHandle> host_state_restore;
    std::optional<StateImageHandle> host_state_fork_destination;
    if (source_state != nullptr || shared_state != nullptr) {
        const StateImageHandle state =
            source_state != nullptr
                ? selected_state(*source_state, details.reuse, details.selected_checkpoint)
                : shared_state->state;
        const bool consuming_fork =
            source_state != nullptr &&
            details.source_mode == runtime::PrivateSourceMode::ConsumeToActive &&
            details.state_fork_required;
        if (state_store->residency(state) == StateReplicaResidency::HostOnly) {
            host_state_restore = state;
            if (state_count == 0) {
                throw std::logic_error("Host StateImage restore has no Device reservation");
            }
            --state_count;
            if (details.source_mode == runtime::PrivateSourceMode::Retain || consuming_fork) {
                std::optional<StateImageHandle> destination =
                    state_store->reserve_logical_destination();
                if (!destination) { throw std::bad_alloc(); }
                if (consuming_fork) {
                    transaction.state_fork_destination = *destination;
                } else {
                    transaction.reserved_states[transaction.reserved_state_count++] = *destination;
                }
                host_state_fork_destination = *destination;
            }
        } else if (consuming_fork) {
            if (state_count == 0) {
                throw std::logic_error("StateImage Fork has no Device reservation");
            }
            --state_count;
            transaction.state_fork_destination = state_store->reserve_destination();
            if (!transaction.state_fork_destination) { throw std::bad_alloc(); }
        } else if (source_state != nullptr &&
                   details.source_mode == runtime::PrivateSourceMode::Retain &&
                   state_store->residency(state) == StateReplicaResidency::Both) {
            if (state_count == 0) {
                throw std::logic_error("Both StateImage split has no active destination");
            }
            --state_count;
            std::optional<StateImageHandle> destination =
                state_store->reserve_logical_destination();
            if (!destination) { throw std::bad_alloc(); }
            transaction.reserved_states[transaction.reserved_state_count++] = *destination;
            transaction.split_state_identity                                = true;
        }
    }
    if (state_count > transaction.reserved_states.size() - transaction.reserved_state_count) {
        throw std::logic_error("materialization state reservation exceeds the active contract");
    }
    for (std::uint32_t index = 0; index < state_count; ++index) {
        std::optional<StateImageHandle> state = state_store->reserve_destination();
        if (!state) { throw std::bad_alloc(); }
        transaction.reserved_states[transaction.reserved_state_count++] = *state;
    }
    if (!transaction.has_source && !transaction.has_shared_source) {
        if (!transaction.root_continuation_index || transaction.root_waiting_for_victim ||
            continuation_slots[*transaction.root_continuation_index].role !=
                ContinuationSlotRole::ReservedMaterialization ||
            transaction.reserved_state_count == 0) {
            throw std::logic_error("root materialization destination is not reserved");
        }
        state_store->activate_reset(transaction.reserved_states[0], device.stream);
    }

    KVAddressSpaceHandle text_address;
    std::optional<KVAddressSpaceHandle> backend_address;
    const bool retained_source = (source_state != nullptr || shared_state != nullptr) &&
                                 details.source_mode == runtime::PrivateSourceMode::Retain;
    if (source_state != nullptr || shared_state != nullptr) {
        const SequenceKVBundle* source_kv = source_state != nullptr
                                                ? (source_state->kv ? &*source_state->kv : nullptr)
                                                : (shared_state->kv ? &*shared_state->kv : nullptr);
        if (source_kv == nullptr) {
            throw std::logic_error("materialization source has no KV address space");
        }
        text_address    = source_kv->text;
        backend_address = source_kv->backend;
        if (retained_source || details.text_prefix_fork_required) {
            transaction.root_text_address = text_kv_addresses->create_inactive();
            if (!transaction.root_text_address) {
                throw std::logic_error("Text KV prefix-fork destination is unavailable");
            }
        }
        if (backend_address && (retained_source || details.backend_prefix_fork_required)) {
            transaction.root_backend_address = backend_kv_addresses->create_inactive();
            if (!transaction.root_backend_address) {
                throw std::logic_error("Backend KV prefix-fork destination is unavailable");
            }
        }
    } else {
        transaction.root_text_address = text_kv_addresses->create_inactive();
        if (!transaction.root_text_address) {
            throw std::logic_error("root Text KV address descriptor is unavailable");
        }
        text_address = *transaction.root_text_address;
        if (details.backend_kv_page_entitlement != 0) {
            if (!backend_kv_addresses) {
                throw std::logic_error("root Backend KV store is unavailable");
            }
            transaction.root_backend_address = backend_kv_addresses->create_inactive();
            if (!transaction.root_backend_address) {
                throw std::logic_error("root Backend KV address descriptor is unavailable");
            }
            backend_address = *transaction.root_backend_address;
        }
    }
    if (details.text_kv_page_entitlement == 0 ||
        backend_address.has_value() != (details.backend_kv_page_entitlement != 0)) {
        throw std::logic_error("materialization KV addresses do not match their entitlements");
    }

    if (source_state != nullptr || shared_state != nullptr) {
        transaction.text_activation_frontier = details.reuse_base;
        if (backend_address) {
            transaction.backend_activation_frontier =
                speculative_backend == SpeculativeBackend::Mtp && details.reuse_base != 0
                    ? details.reuse_base - 1U
                    : details.reuse_base;
        }
    }

    const bool text_prefix_fork =
        (source_state != nullptr || shared_state != nullptr) && details.text_prefix_fork_required;
    const bool backend_prefix_fork = (source_state != nullptr || shared_state != nullptr) &&
                                     details.backend_prefix_fork_required;
    if (text_prefix_fork) {
        transaction.text_source_restore_reservation.emplace(
            text_kv_pages->physical_pool().make_empty_reservation());
    } else {
        const KVAddressSpaceHandle activation_address =
            retained_source ? *transaction.root_text_address : text_address;
        transaction.text_activation.emplace(text_kv_addresses->prepare_activation(
            activation_address, details.text_kv_page_entitlement, static_cast<std::int32_t>(lane),
            transaction.text_activation_frontier));
    }
    if (backend_address && backend_prefix_fork) {
        transaction.backend_source_restore_reservation.emplace(
            backend_kv_pages->physical_pool().make_empty_reservation());
    } else if (backend_address) {
        const KVAddressSpaceHandle activation_address =
            retained_source ? *transaction.root_backend_address : *backend_address;
        transaction.backend_activation.emplace(backend_kv_addresses->prepare_activation(
            activation_address, details.backend_kv_page_entitlement,
            static_cast<std::int32_t>(lane), transaction.backend_activation_frontier));
    }

    const auto prepare_kv_restores =
        [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages, KVAddressSpaceHandle address,
            std::optional<std::uint32_t> activation_frontier, bool source_reservation,
            DeviceKVPageReservation& reservation,
            std::vector<MaterializationTransaction::KVRestorePage>& restores,
            std::vector<DeviceKVPageHandle>& destinations) {
            const std::uint32_t mapped = activation_frontier
                                             ? kv_pages_for_frontier(*activation_frontier)
                                             : addresses.mapped_pages(address);
            if (mapped > addresses.mapped_pages(address)) {
                throw std::logic_error("KV activation frontier exceeds address membership");
            }
            std::uint32_t missing = 0;
            for (std::uint32_t page = 0; page < mapped; ++page) {
                if (!pages.device_resident(addresses.logical_page(address, page))) { ++missing; }
            }
            if (source_reservation) {
                pages.physical_pool().resize_reservation(reservation, missing);
            }
            for (std::uint32_t page = 0; page < mapped; ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                if (pages.device_resident(logical)) { continue; }
                if (!pages.host_resident(logical) || !host_kv_extents) {
                    throw std::logic_error("checkpoint KV page has no restorable replica");
                }
                const HostKVPageReplica replica = pages.host_replica(logical);
                const DeviceKVPageHandle destination =
                    pages.reserve_device_replica(logical, reservation);
                restores.push_back(MaterializationTransaction::KVRestorePage{
                    .logical     = logical,
                    .extent      = replica.extent,
                    .extent_page = replica.page_offset,
                });
                destinations.push_back(destination);
            }
        };
    DeviceKVPageReservation& text_restore_reservation =
        text_prefix_fork ? *transaction.text_source_restore_reservation
                         : text_kv_addresses->page_reservation(*transaction.text_activation);
    prepare_kv_restores(*text_kv_addresses, *text_kv_pages, text_address,
                        transaction.text_activation_frontier, text_prefix_fork,
                        text_restore_reservation, transaction.text_restores,
                        transaction.text_restore_destinations);
    if (backend_address) {
        DeviceKVPageReservation& backend_restore_reservation =
            backend_prefix_fork
                ? *transaction.backend_source_restore_reservation
                : backend_kv_addresses->page_reservation(*transaction.backend_activation);
        prepare_kv_restores(*backend_kv_addresses, *backend_kv_pages, *backend_address,
                            transaction.backend_activation_frontier, backend_prefix_fork,
                            backend_restore_reservation, transaction.backend_restores,
                            transaction.backend_restore_destinations);
    }
    if (host_state_restore) {
        start_context_transfer_timer(runtime::ContextResourceClass::State);
        std::optional<StateImageTransfer> restore =
            host_state_fork_destination
                ? state_store->begin_host_fork(*host_state_restore, *host_state_fork_destination,
                                               device.transfer_stream)
                : state_store->begin_host_to_device(*host_state_restore, device.transfer_stream);
        if (!restore) { throw std::bad_alloc(); }
        transaction.state_restore.emplace(std::move(*restore));
        stop_context_transfer_timer(runtime::ContextResourceClass::State);
        transaction.transfer_timer_mask |=
            1U << context_resource_index(runtime::ContextResourceClass::State);
    }
    transaction.prepared = true;
    requests[lane].prefill->elapsed_seconds +=
        std::chrono::duration<double>(Clock::now() - prepare_started).count();
}

void ProgramImplCore::prepare_prefix_forks(MaterializationTransaction& transaction) {
    if (!transaction.plan || transaction.plan->impl_ == nullptr ||
        (transaction.has_source == transaction.has_shared_source) ||
        (transaction.has_source && transaction.source_index >= continuation_capacity) ||
        (transaction.has_shared_source &&
         transaction.shared_source_index >= shared_prefix_capacity) ||
        transaction.prefix_forks_ready || transaction.prefix_tail_submitted) {
        throw std::logic_error("prefix fork preparation is invalid");
    }
    const AdmissionCandidateImpl& details = *transaction.plan->impl_;
    if ((!details.text_prefix_fork_required && !details.backend_prefix_fork_required) ||
        (details.text_prefix_fork_required &&
         (!transaction.root_text_address || transaction.text_prefix_fork)) ||
        (details.backend_prefix_fork_required &&
         (!transaction.root_backend_address || transaction.backend_prefix_fork))) {
        throw std::logic_error("planned prefix fork destinations are incomplete");
    }
    const SequenceKVBundle* source_kv =
        transaction.has_source ? (continuation_states[transaction.source_index].kv
                                      ? &*continuation_states[transaction.source_index].kv
                                      : nullptr)
                               : (shared_prefix_states[transaction.shared_source_index].kv
                                      ? &*shared_prefix_states[transaction.shared_source_index].kv
                                      : nullptr);
    if (source_kv == nullptr || !transaction.text_activation_frontier) {
        throw std::logic_error("prefix fork source is incomplete");
    }
    if (transaction.text_source_restore_reservation &&
        transaction.text_source_restore_reservation->pages() != 0) {
        throw std::logic_error("retained Text KV restores are incomplete");
    }
    if (transaction.backend_source_restore_reservation &&
        transaction.backend_source_restore_reservation->pages() != 0) {
        throw std::logic_error("retained Backend KV restores are incomplete");
    }
    const auto prepare_retained_tail_backup = [&](KVAddressSpaceStore& addresses,
                                                  LogicalKVPageStore& pages,
                                                  KVPrefixForkReservation& fork, bool staged,
                                                  std::optional<LogicalKVPageHandle>& retained_tail,
                                                  std::optional<HostKVExtentReservation>& backup) {
        if (!staged) { return; }
        const LogicalKVPageHandle tail = addresses.prefix_fork_tail_logical_source(fork);
        if (pages.address_references(tail) != 1 || !pages.device_resident(tail) ||
            pages.writer_references(tail) != 0) {
            throw std::logic_error("retained KV tail changed before staged release");
        }
        retained_tail = tail;
        if (pages.host_resident(tail)) { return; }
        if (host_kv_extents == nullptr) {
            throw std::logic_error("retained KV tail has no Host extent store");
        }
        const std::array membership{tail};
        std::optional<HostKVExtentReservation> reserved =
            host_kv_extents->prepare(pages, membership);
        if (!reserved) { throw std::bad_alloc(); }
        backup.emplace(std::move(*reserved));
    };
    bool copied_tail = false;
    if (details.text_prefix_fork_required) {
        transaction.text_source_restore_reservation.reset();
        transaction.text_prefix_fork.emplace(text_kv_addresses->prepare_prefix_fork(
            source_kv->text, *transaction.root_text_address, *transaction.text_activation_frontier,
            details.text_kv_page_entitlement,
            static_cast<std::int32_t>(transaction.destination.value),
            details.text_retained_tail_release));
        prepare_retained_tail_backup(
            *text_kv_addresses, *text_kv_pages, *transaction.text_prefix_fork,
            details.text_retained_tail_release, transaction.text_retained_tail,
            transaction.text_retained_tail_backup);
        if (*transaction.text_activation_frontier % static_cast<std::uint32_t>(kPagedKVPageSize) !=
            0) {
            start_context_transfer_timer(runtime::ContextResourceClass::MainKV);
            text_kv_pages->physical_pool().copy_page(
                text_kv_addresses->prefix_fork_tail_source(*transaction.text_prefix_fork),
                text_kv_addresses->prefix_fork_tail_destination(*transaction.text_prefix_fork),
                device.transfer_stream);
            stop_context_transfer_timer(runtime::ContextResourceClass::MainKV);
            transaction.transfer_timer_mask |=
                1U << context_resource_index(runtime::ContextResourceClass::MainKV);
            ++transaction.operations.partial_tail_cow_pages;
            copied_tail = true;
        }
    }

    if (details.backend_prefix_fork_required) {
        if (!transaction.root_backend_address || !transaction.backend_activation_frontier) {
            throw std::logic_error("Backend KV prefix-fork destination is incomplete");
        }
        if (!source_kv->backend) {
            throw std::logic_error("Backend KV prefix-fork source is unavailable");
        }
        transaction.backend_source_restore_reservation.reset();
        transaction.backend_prefix_fork.emplace(backend_kv_addresses->prepare_prefix_fork(
            *source_kv->backend, *transaction.root_backend_address,
            *transaction.backend_activation_frontier, details.backend_kv_page_entitlement,
            static_cast<std::int32_t>(transaction.destination.value),
            details.backend_retained_tail_release));
        prepare_retained_tail_backup(
            *backend_kv_addresses, *backend_kv_pages, *transaction.backend_prefix_fork,
            details.backend_retained_tail_release, transaction.backend_retained_tail,
            transaction.backend_retained_tail_backup);
        if (*transaction.backend_activation_frontier %
                static_cast<std::uint32_t>(kPagedKVPageSize) !=
            0) {
            start_context_transfer_timer(runtime::ContextResourceClass::BackendKV);
            backend_kv_pages->physical_pool().copy_page(
                backend_kv_addresses->prefix_fork_tail_source(*transaction.backend_prefix_fork),
                backend_kv_addresses->prefix_fork_tail_destination(
                    *transaction.backend_prefix_fork),
                device.transfer_stream);
            stop_context_transfer_timer(runtime::ContextResourceClass::BackendKV);
            transaction.transfer_timer_mask |=
                1U << context_resource_index(runtime::ContextResourceClass::BackendKV);
            ++transaction.operations.partial_tail_cow_pages;
            copied_tail = true;
        }
    }

    if (copied_tail) {
        context_completion_.record(device.transfer_stream);
        transaction.prefix_tail_submitted = true;
        transaction.transfer_submitted    = true;
    } else {
        transaction.prefix_forks_ready = true;
    }
}

void ProgramImplCore::enqueue_materialization_transfers(MaterializationTransaction& transaction) {
    if (!transaction.prepared || transaction.transfer_submitted) {
        throw std::logic_error("materialization transfer batch is not enqueueable");
    }
    const auto enqueue_kv =
        [&](LogicalKVPageStore& pages,
            const std::vector<MaterializationTransaction::KVRestorePage>& restores,
            const std::vector<DeviceKVPageHandle>& destinations,
            runtime::ContextResourceClass resource) {
            if (restores.size() != destinations.size()) {
                throw std::logic_error("KV restore bookkeeping is not row aligned");
            }
            if (restores.empty()) { return; }
            start_context_transfer_timer(resource);
            std::size_t begin = 0;
            while (begin < restores.size()) {
                std::size_t end = begin + 1;
                while (end < restores.size() && restores[end].extent == restores[begin].extent &&
                       restores[end].extent_page == restores[end - 1].extent_page + 1U) {
                    ++end;
                }
                const HostKVAllocationConstView source =
                    host_kv_extents->view(restores[begin].extent)
                        .subview(restores[begin].extent_page,
                                 static_cast<std::uint32_t>(end - begin));
                pages.physical_pool().copy_from_host(
                    source,
                    std::span<const DeviceKVPageHandle>(destinations.data() + begin, end - begin),
                    device.transfer_stream);
                begin = end;
            }
            stop_context_transfer_timer(resource);
            transaction.transfer_timer_mask |= 1U << context_resource_index(resource);
        };
    enqueue_kv(*text_kv_pages, transaction.text_restores, transaction.text_restore_destinations,
               runtime::ContextResourceClass::MainKV);
    if (!transaction.backend_restores.empty()) {
        enqueue_kv(*backend_kv_pages, transaction.backend_restores,
                   transaction.backend_restore_destinations,
                   runtime::ContextResourceClass::BackendKV);
    }
    const bool any = transaction.state_restore.has_value() || !transaction.text_restores.empty() ||
                     !transaction.backend_restores.empty();
    if (any) {
        context_completion_.record(device.transfer_stream);
        transaction.transfer_submitted = true;
    } else if (transaction.plan && transaction.plan->impl_ &&
               (transaction.plan->impl_->text_prefix_fork_required ||
                transaction.plan->impl_->backend_prefix_fork_required)) {
        prepare_prefix_forks(transaction);
    }
}

void ProgramImplCore::record_materialization_transfer_observations(
    MaterializationTransaction& transaction) {
    if (!transaction.transfer_submitted || !context_completion_.ready()) {
        throw std::logic_error("materialization transfer observation is not complete");
    }
    const auto record = [&](runtime::ContextResourceClass resource,
                            runtime::ContextTransferDirection direction, TransferWork transfer_work,
                            std::uint32_t pages) {
        const std::uint8_t bit = static_cast<std::uint8_t>(1U << context_resource_index(resource));
        if ((transaction.transfer_timer_mask & bit) == 0) { return; }
        transaction.transfer_observations.push_back(
            context_transfer_observation(resource, direction, transfer_work, pages));
        transaction.transfer_timer_mask &= static_cast<std::uint8_t>(~bit);
    };
    const auto host_layout = [](const LogicalKVPageStore& pages) {
        return plan_host_kv_page_layout(pages.physical_pool().geometry());
    };
    const auto restore_copy_runs = [](const auto& restores, const auto& destinations,
                                      const LogicalKVPageStore& pages) {
        if (restores.size() != destinations.size()) {
            throw std::logic_error("KV restore observation is not row aligned");
        }
        std::uint32_t runs = 0;
        std::size_t begin  = 0;
        while (begin < restores.size()) {
            std::size_t end = begin + 1U;
            while (end < restores.size() && restores[end].extent == restores[begin].extent &&
                   restores[end].extent_page == restores[end - 1U].extent_page + 1U) {
                ++end;
            }
            runs += pages.physical_pool().contiguous_run_count(
                std::span<const DeviceKVPageHandle>(destinations.data() + begin, end - begin));
            begin = end;
        }
        return runs;
    };
    if (transaction.prefix_tail_submitted) {
        if (transaction.text_prefix_fork && transaction.text_prefix_fork->needs_tail_copy()) {
            record(runtime::ContextResourceClass::MainKV,
                   runtime::ContextTransferDirection::DeviceToDevice,
                   plan_device_kv_copy_work(host_layout(*text_kv_pages), 1), 1);
        }
        if (backend_kv_pages && transaction.backend_prefix_fork &&
            transaction.backend_prefix_fork->needs_tail_copy()) {
            record(runtime::ContextResourceClass::BackendKV,
                   runtime::ContextTransferDirection::DeviceToDevice,
                   plan_device_kv_copy_work(host_layout(*backend_kv_pages), 1), 1);
        }
        return;
    }
    if (transaction.retained_tail_backup_submitted) {
        if (transaction.text_retained_tail_backup) {
            record(runtime::ContextResourceClass::MainKV,
                   runtime::ContextTransferDirection::DeviceToHost,
                   plan_host_kv_transfer_work(host_layout(*text_kv_pages), 1, 1), 1);
        }
        if (backend_kv_pages && transaction.backend_retained_tail_backup) {
            record(runtime::ContextResourceClass::BackendKV,
                   runtime::ContextTransferDirection::DeviceToHost,
                   plan_host_kv_transfer_work(host_layout(*backend_kv_pages), 1, 1), 1);
        }
        return;
    }
    if (transaction.state_restore) {
        record(runtime::ContextResourceClass::State,
               runtime::ContextTransferDirection::HostToDevice,
               state_image_transfer_work(host_state_images->layout()), 0);
    }
    record(runtime::ContextResourceClass::MainKV, runtime::ContextTransferDirection::HostToDevice,
           plan_host_kv_transfer_work(host_layout(*text_kv_pages),
                                      static_cast<std::uint32_t>(transaction.text_restores.size()),
                                      restore_copy_runs(transaction.text_restores,
                                                        transaction.text_restore_destinations,
                                                        *text_kv_pages)),
           static_cast<std::uint32_t>(transaction.text_restores.size()));
    if (backend_kv_pages) {
        record(runtime::ContextResourceClass::BackendKV,
               runtime::ContextTransferDirection::HostToDevice,
               plan_host_kv_transfer_work(
                   host_layout(*backend_kv_pages),
                   static_cast<std::uint32_t>(transaction.backend_restores.size()),
                   restore_copy_runs(transaction.backend_restores,
                                     transaction.backend_restore_destinations, *backend_kv_pages)),
               static_cast<std::uint32_t>(transaction.backend_restores.size()));
    }
}

void ProgramImplCore::publish_materialization_transfers(MaterializationTransaction& transaction) {
    record_materialization_transfer_observations(transaction);
    const auto enqueue_retained_tail_backups = [&]() {
        bool submitted     = false;
        const auto enqueue = [&](LogicalKVPageStore& pages,
                                 std::optional<HostKVExtentReservation>& backup,
                                 runtime::ContextResourceClass resource) {
            if (!backup) { return; }
            if (host_kv_extents == nullptr || host_kv_extents->page_count(*backup) != 1) {
                throw std::logic_error("retained KV tail Host reservation changed");
            }
            std::array<DeviceKVPageHandle, 1> source{};
            host_kv_extents->device_sources(*backup, source);
            start_context_transfer_timer(resource);
            pages.physical_pool().copy_to_host(source, host_kv_extents->writable_view(*backup),
                                               device.transfer_stream);
            stop_context_transfer_timer(resource);
            transaction.transfer_timer_mask |= 1U << context_resource_index(resource);
            submitted = true;
        };
        enqueue(*text_kv_pages, transaction.text_retained_tail_backup,
                runtime::ContextResourceClass::MainKV);
        if (backend_kv_pages) {
            enqueue(*backend_kv_pages, transaction.backend_retained_tail_backup,
                    runtime::ContextResourceClass::BackendKV);
        }
        if (submitted) {
            context_completion_.record(device.transfer_stream);
            transaction.retained_tail_backup_submitted = true;
            transaction.transfer_submitted             = true;
        }
        return submitted;
    };
    const auto publish_retained_tail_releases = [&]() {
        if (!transaction.plan || transaction.plan->impl_ == nullptr) {
            throw std::logic_error("retained KV tail release lost its admission plan");
        }
        const AdmissionCandidateImpl& details = *transaction.plan->impl_;
        const auto publish = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                 std::optional<KVPrefixForkReservation>& fork, bool staged,
                                 std::optional<LogicalKVPageHandle>& retained_tail,
                                 std::optional<HostKVExtentReservation>& backup) {
            if (!staged) {
                if (retained_tail || backup) {
                    throw std::logic_error("unstaged KV prefix fork owns a retained tail release");
                }
                return;
            }
            if (!fork || !retained_tail ||
                addresses.prefix_fork_tail_logical_source(*fork) != *retained_tail) {
                throw std::logic_error("staged KV prefix-fork tail identity changed");
            }
            if (backup) {
                if (host_kv_extents == nullptr) {
                    throw std::logic_error("retained KV tail Host store disappeared");
                }
                (void)host_kv_extents->publish(std::move(*backup));
                backup.reset();
            }
            addresses.settle_prefix_fork_tail_source(*fork);
            if (!pages.drop_device_replica(*retained_tail)) {
                throw std::logic_error("retained KV tail Device replica is not releasable");
            }
            addresses.complete_prefix_fork_after_tail_release(*fork);
            retained_tail.reset();
        };
        publish(*text_kv_addresses, *text_kv_pages, transaction.text_prefix_fork,
                details.text_retained_tail_release, transaction.text_retained_tail,
                transaction.text_retained_tail_backup);
        if (details.backend_retained_tail_release) {
            if (!backend_kv_addresses || !backend_kv_pages) {
                throw std::logic_error("staged Backend KV tail store is unavailable");
            }
            publish(*backend_kv_addresses, *backend_kv_pages, transaction.backend_prefix_fork, true,
                    transaction.backend_retained_tail, transaction.backend_retained_tail_backup);
        } else if (transaction.backend_retained_tail || transaction.backend_retained_tail_backup) {
            throw std::logic_error("unstaged Backend KV tail release was prepared");
        }
        transaction.prefix_forks_ready = true;
    };
    if (transaction.prefix_tail_submitted) {
        transaction.prefix_tail_submitted = false;
        transaction.transfer_submitted    = false;
        if (enqueue_retained_tail_backups()) { return; }
        publish_retained_tail_releases();
        return;
    }
    if (transaction.retained_tail_backup_submitted) {
        transaction.retained_tail_backup_submitted = false;
        transaction.transfer_submitted             = false;
        publish_retained_tail_releases();
        return;
    }
    if (transaction.state_restore) {
        state_store->publish_transfer(std::move(*transaction.state_restore), true);
        transaction.state_restore.reset();
        transaction.state_restored = true;
    }
    for (const MaterializationTransaction::KVRestorePage& restore : transaction.text_restores) {
        text_kv_pages->publish_device_replica(restore.logical);
    }
    for (const MaterializationTransaction::KVRestorePage& restore : transaction.backend_restores) {
        backend_kv_pages->publish_device_replica(restore.logical);
    }
    transaction.text_restores.clear();
    transaction.text_restore_destinations.clear();
    transaction.backend_restores.clear();
    transaction.backend_restore_destinations.clear();
    transaction.transfer_submitted = false;
    if (transaction.plan && transaction.plan->impl_ &&
        (transaction.plan->impl_->text_prefix_fork_required ||
         transaction.plan->impl_->backend_prefix_fork_required)) {
        prepare_prefix_forks(transaction);
    }
}

void ProgramImplCore::abort_materialization_transfers(
    MaterializationTransaction& transaction) noexcept {
    try {
        if (transaction.transfer_submitted) {
            context_completion_.synchronize();
            record_materialization_transfer_observations(transaction);
        }
        if (transaction.state_restore) {
            state_store->abort_transfer(std::move(*transaction.state_restore));
            transaction.state_restore.reset();
        }
        if (transaction.text_activation || transaction.text_source_restore_reservation) {
            DeviceKVPageReservation& reservation =
                transaction.text_source_restore_reservation
                    ? *transaction.text_source_restore_reservation
                    : text_kv_addresses->page_reservation(*transaction.text_activation);
            for (const MaterializationTransaction::KVRestorePage& restore :
                 transaction.text_restores) {
                text_kv_pages->abort_device_replica(restore.logical, reservation);
            }
        }
        if (transaction.backend_activation || transaction.backend_source_restore_reservation) {
            DeviceKVPageReservation& reservation =
                transaction.backend_source_restore_reservation
                    ? *transaction.backend_source_restore_reservation
                    : backend_kv_addresses->page_reservation(*transaction.backend_activation);
            for (const MaterializationTransaction::KVRestorePage& restore :
                 transaction.backend_restores) {
                backend_kv_pages->abort_device_replica(restore.logical, reservation);
            }
        }
    } catch (...) { std::terminate(); }
    transaction.text_restores.clear();
    transaction.text_restore_destinations.clear();
    transaction.backend_restores.clear();
    transaction.backend_restore_destinations.clear();
    transaction.transfer_timer_mask = 0;
    transaction.transfer_submitted  = false;
}

void ProgramImplCore::prepare_pressure_bookkeeping(MaterializationTransaction::PressureWork& work) {
    work.state_changes.clear();
    work.main_kv_changes.clear();
    work.backend_kv_changes.clear();
    if (work.option.evicts_continuation) { return; }

    work.state_changes.resize(work.option.state_changes.size());

    const SequenceState* sequence =
        work.shared_owner ? nullptr : &continuation_states[work.continuation_index];
    const SharedPrefixState* shared =
        work.shared_owner ? &shared_prefix_states[work.continuation_index] : nullptr;
    const SequenceKVBundle* kv = sequence != nullptr ? (sequence->kv ? &*sequence->kv : nullptr)
                                                     : (shared->kv ? &*shared->kv : nullptr);
    if (kv == nullptr) { throw std::logic_error("pressure owner has no KV address space"); }

    const auto prepare =
        [&](KVAddressSpaceStore* addresses, LogicalKVPageStore* pages,
            std::optional<KVAddressSpaceHandle> address,
            std::span<const qwen3_6::detail::PressureKVDecision> actions,
            std::vector<MaterializationTransaction::PressureWork::KVChangeWork>& changes) {
            changes.reserve(actions.size());
            for (const qwen3_6::detail::PressureKVDecision& action : actions) {
                changes.emplace_back();
                MaterializationTransaction::PressureWork::KVChangeWork& change = changes.back();
                if (action.kind == qwen3_6::detail::PressureKVDecisionKind::None) {
                    throw std::logic_error("pressure KV action has no operation kind");
                }
                if (addresses == nullptr || pages == nullptr || !address) {
                    throw std::logic_error("pressure KV action has no typed address space");
                }
                const std::uint32_t mapped = addresses->mapped_pages(*address);
                if (action.page_count == 0 || action.begin_page > mapped ||
                    action.page_count > mapped - action.begin_page) {
                    throw std::logic_error("pressure KV action range is invalid");
                }
                change.pages.reserve(action.page_count);
                for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
                    change.pages.push_back(
                        addresses->logical_page(*address, action.begin_page + offset));
                }
                if (action.kind == qwen3_6::detail::PressureKVDecisionKind::DemoteToHost) {
                    change.sources.resize(action.page_count);
                }
            }
        };
    prepare(text_kv_addresses.get(), text_kv_pages.get(), kv->text, work.option.main_kv_changes,
            work.main_kv_changes);
    prepare(backend_kv_addresses.get(), backend_kv_pages.get(), kv->backend,
            work.option.backend_kv_changes, work.backend_kv_changes);
}

void ProgramImplCore::publish_pressure_host_releases(
    MaterializationTransaction::PressureWork& work) {
    detail::PhysicalDelta delta;
    if (work.option.evicts_continuation || work.completed || work.submitted) { return; }
    const bool valid_owner =
        work.shared_owner ? (work.continuation_index < shared_prefix_capacity &&
                             shared_prefix_slots[work.continuation_index].role ==
                                 SharedPrefixSlotRole::Catalogued &&
                             shared_prefix_slots[work.continuation_index].generation ==
                                 work.continuation_generation &&
                             shared_prefix_states[work.continuation_index].active_references == 0)
                          : (work.continuation_index < continuation_capacity &&
                             continuation_slots[work.continuation_index].role ==
                                 ContinuationSlotRole::Catalogued &&
                             continuation_slots[work.continuation_index].generation ==
                                 work.continuation_generation);
    if (!valid_owner || work.option.shared_owner != work.shared_owner) {
        throw std::logic_error("pressure Host release owner changed before publication");
    }
    SequenceState* sequence =
        work.shared_owner ? nullptr : &continuation_states[work.continuation_index];
    SharedPrefixState* shared =
        work.shared_owner ? &shared_prefix_states[work.continuation_index] : nullptr;

    if (!work.option.dropped_checkpoints.empty() && !work.checkpoint_drop_published) {
        if (sequence == nullptr) {
            throw std::logic_error("checkpoint drop targets a shared pressure owner");
        }
        for (const runtime::CheckpointRef checkpoint : work.option.dropped_checkpoints) {
            publish_checkpoint_drop(*sequence, checkpoint);
        }
        delta.removed =
            checked_resource_sum(delta.removed, work.option.checkpoint_drop_effect.removed);
        delta.added = checked_resource_sum(delta.added, work.option.checkpoint_drop_effect.added);
        work.checkpoint_drop_published = true;
        work.mutation_published        = true;
        const bool pure_drop           = work.option.state_changes.empty() &&
                               work.option.main_kv_changes.empty() &&
                               work.option.backend_kv_changes.empty();
        if (pure_drop) {
            work.committed_delta = delta;
            work.completed       = true;
            return;
        }
    }

    if (work.state_changes.size() != work.option.state_changes.size()) {
        throw std::logic_error("pressure State bookkeeping is not action aligned");
    }
    for (std::size_t index = 0; index < work.option.state_changes.size(); ++index) {
        const qwen3_6::detail::PressureStateDecision action = work.option.state_changes[index];
        auto& change                                        = work.state_changes[index];
        if (!pressure_state_drops_host(action) || change.host_released) { continue; }
        const std::optional<StateImageHandle> state =
            pressure_state_source(action, sequence, shared);
        if (!state || !state_store->drop_host_replica(*state)) {
            throw std::logic_error("pressure Host State duplicate is no longer releasable");
        }
        change.host_released    = true;
        work.mutation_published = true;
        ++delta.removed.host.state_slots;
    }

    const SequenceKVBundle* kv = sequence != nullptr ? (sequence->kv ? &*sequence->kv : nullptr)
                                                     : (shared->kv ? &*shared->kv : nullptr);
    if (kv == nullptr) { throw std::logic_error("pressure Host release owner has no KV bundle"); }
    const auto release_kv = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                KVAddressSpaceHandle address,
                                const qwen3_6::detail::PressureKVDecision& action,
                                MaterializationTransaction::PressureWork::KVChangeWork& change) {
        if (action.kind != qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate ||
            change.host_released) {
            return;
        }
        const std::uint32_t mapped = addresses.mapped_pages(address);
        if (action.begin_page > mapped || action.page_count > mapped - action.begin_page ||
            change.pages.size() != action.page_count) {
            throw std::logic_error("pressure Host KV release region changed");
        }
        for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
            if (change.pages[offset] !=
                addresses.logical_page(address, action.begin_page + offset)) {
                throw std::logic_error("pressure Host KV release membership changed");
            }
        }
        if (!host_kv_extents || !host_kv_extents->release_page_replicas(pages, change.pages)) {
            throw std::logic_error("pressure Host KV duplicates are no longer releasable");
        }
        const std::size_t page_stride =
            &pages == text_kv_pages.get() ? text_host_kv_page_stride : backend_host_kv_page_stride;
        if (action.page_count != 0 &&
            page_stride > std::numeric_limits<std::size_t>::max() / action.page_count) {
            throw std::overflow_error("pressure Host KV release size overflow");
        }
        const std::size_t bytes = page_stride * static_cast<std::size_t>(action.page_count);
        if (bytes > std::numeric_limits<std::size_t>::max() - delta.removed.host.kv_bytes) {
            throw std::overflow_error("pressure Host KV release sum overflow");
        }
        delta.removed.host.kv_bytes += bytes;
        change.host_released    = true;
        work.mutation_published = true;
    };
    if (work.main_kv_changes.size() != work.option.main_kv_changes.size() ||
        work.backend_kv_changes.size() != work.option.backend_kv_changes.size()) {
        throw std::logic_error("pressure KV bookkeeping is not action aligned");
    }
    for (std::size_t index = 0; index < work.option.main_kv_changes.size(); ++index) {
        release_kv(*text_kv_addresses, *text_kv_pages, kv->text, work.option.main_kv_changes[index],
                   work.main_kv_changes[index]);
    }
    if (!work.option.backend_kv_changes.empty()) {
        if (!kv->backend || !backend_kv_addresses || !backend_kv_pages) {
            throw std::logic_error("pressure Host Backend KV release has no typed store");
        }
        for (std::size_t index = 0; index < work.option.backend_kv_changes.size(); ++index) {
            release_kv(*backend_kv_addresses, *backend_kv_pages, *kv->backend,
                       work.option.backend_kv_changes[index], work.backend_kv_changes[index]);
        }
    }
    work.committed_delta.removed =
        checked_resource_sum(work.committed_delta.removed, delta.removed);
    work.committed_delta.added = checked_resource_sum(work.committed_delta.added, delta.added);
}

void ProgramImplCore::prepare_pressure_work(MaterializationTransaction::PressureWork& work,
                                            runtime::ContextResourceClass resource) {
    const bool valid_owner =
        work.shared_owner ? (work.continuation_index < shared_prefix_capacity &&
                             shared_prefix_slots[work.continuation_index].role ==
                                 SharedPrefixSlotRole::Catalogued &&
                             shared_prefix_slots[work.continuation_index].generation ==
                                 work.continuation_generation &&
                             shared_prefix_states[work.continuation_index].active_references == 0)
                          : (work.continuation_index < continuation_capacity &&
                             continuation_slots[work.continuation_index].role ==
                                 ContinuationSlotRole::Catalogued &&
                             continuation_slots[work.continuation_index].generation ==
                                 work.continuation_generation);
    if (work.completed || !valid_owner || work.option.shared_owner != work.shared_owner) {
        throw std::logic_error("pressure work source changed before transfer");
    }
    if (work.option.evicts_continuation) { return; }
    SequenceState* sequence =
        work.shared_owner ? nullptr : &continuation_states[work.continuation_index];
    SharedPrefixState* shared =
        work.shared_owner ? &shared_prefix_states[work.continuation_index] : nullptr;
    if (work.state_changes.size() != work.option.state_changes.size()) {
        throw std::logic_error("pressure State bookkeeping is not action aligned");
    }
    if (resource == runtime::ContextResourceClass::State) {
        for (std::size_t index = 0; index < work.option.state_changes.size(); ++index) {
            const qwen3_6::detail::PressureStateDecision action = work.option.state_changes[index];
            auto& change                                        = work.state_changes[index];
            if (!pressure_state_demotes(action)) { continue; }
            if (change.transfer) {
                throw std::logic_error("pressure State transfer was prepared more than once");
            }
            const std::optional<StateImageHandle> source =
                pressure_state_source(action, sequence, shared);
            if (!source) { throw std::logic_error("pressure State transfer has no source"); }
            std::optional<StateImageTransfer> transfer =
                state_store->begin_device_to_host(*source, device.transfer_stream);
            if (!transfer) { throw std::bad_alloc(); }
            change.transfer.emplace(std::move(*transfer));
        }
    }

    const auto prepare_kv = [&](KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                KVAddressSpaceHandle address,
                                const qwen3_6::detail::PressureKVDecision& action,
                                MaterializationTransaction::PressureWork::KVChangeWork& change) {
        if (action.page_count == 0) { return; }
        if (action.kind == qwen3_6::detail::PressureKVDecisionKind::None) {
            throw std::logic_error("pressure KV action has no operation kind");
        }
        if (action.kind == qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate &&
            change.host_released) {
            return;
        }
        const std::uint32_t mapped = addresses.mapped_pages(address);
        if (action.begin_page > mapped || action.page_count > mapped - action.begin_page ||
            change.pages.size() != action.page_count) {
            throw std::logic_error("pressure KV region changed before transfer");
        }
        for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
            const LogicalKVPageHandle logical =
                addresses.logical_page(address, action.begin_page + offset);
            const bool host_resident = pages.host_resident(logical);
            const bool valid_residency =
                action.kind == qwen3_6::detail::PressureKVDecisionKind::DemoteToHost
                    ? !host_resident
                    : host_resident;
            const bool removes_device =
                action.kind != qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate;
            if (!pages.device_resident(logical) || pages.writer_references(logical) != 0 ||
                pages.source_pins(logical) != 0 || !valid_residency ||
                (removes_device && addresses.has_active_reference(logical))) {
                throw std::logic_error("pressure KV replica changed before transfer");
            }
            if (change.pages[offset] != logical) {
                throw std::logic_error("pressure KV membership changed before transfer");
            }
        }
        if (action.kind == qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate) {
            if (!host_kv_extents) { throw std::logic_error("Host KV extent store is unavailable"); }
            if (!host_kv_extents->can_release_page_replicas(pages, change.pages)) {
                throw std::logic_error("pressure Host KV replicas are no longer releasable");
            }
            return;
        }
        if (action.kind == qwen3_6::detail::PressureKVDecisionKind::DropDeviceDuplicate) { return; }
        if (!host_kv_extents) { throw std::logic_error("Host KV extent store is unavailable"); }
        std::optional<HostKVExtentReservation> reserved =
            host_kv_extents->prepare(pages, change.pages);
        if (!reserved) { throw std::bad_alloc(); }
        if (change.sources.size() != change.pages.size()) {
            throw std::logic_error("pressure KV source backing was not prepared");
        }
        host_kv_extents->device_sources(*reserved, change.sources);
        pages.physical_pool().copy_to_host(
            change.sources, host_kv_extents->writable_view(*reserved), device.transfer_stream);
        change.backup.emplace(std::move(*reserved));
    };
    const SequenceKVBundle* kv = sequence != nullptr ? (sequence->kv ? &*sequence->kv : nullptr)
                                                     : (shared->kv ? &*shared->kv : nullptr);
    if (kv == nullptr) { throw std::logic_error("pressure owner has no KV address space"); }
    if (resource == runtime::ContextResourceClass::MainKV) {
        if (work.main_kv_changes.size() != work.option.main_kv_changes.size()) {
            throw std::logic_error("pressure Main KV bookkeeping is not action aligned");
        }
        for (std::size_t index = 0; index < work.option.main_kv_changes.size(); ++index) {
            prepare_kv(*text_kv_addresses, *text_kv_pages, kv->text,
                       work.option.main_kv_changes[index], work.main_kv_changes[index]);
        }
    }
    if (resource == runtime::ContextResourceClass::BackendKV &&
        !work.option.backend_kv_changes.empty()) {
        if (!kv->backend || !backend_kv_addresses || !backend_kv_pages) {
            throw std::logic_error("pressure owner has no Backend KV address space");
        }
        if (work.backend_kv_changes.size() != work.option.backend_kv_changes.size()) {
            throw std::logic_error("pressure Backend KV bookkeeping is not action aligned");
        }
        for (std::size_t index = 0; index < work.option.backend_kv_changes.size(); ++index) {
            prepare_kv(*backend_kv_addresses, *backend_kv_pages, *kv->backend,
                       work.option.backend_kv_changes[index], work.backend_kv_changes[index]);
        }
    }
    work.submitted = std::any_of(work.state_changes.begin(), work.state_changes.end(),
                                 [](const auto& change) { return change.transfer.has_value(); }) ||
                     std::any_of(work.main_kv_changes.begin(), work.main_kv_changes.end(),
                                 [](const auto& change) { return change.backup.has_value(); }) ||
                     std::any_of(work.backend_kv_changes.begin(), work.backend_kv_changes.end(),
                                 [](const auto& change) { return change.backup.has_value(); });
}

void ProgramImplCore::publish_pressure_work(
    MaterializationTransaction::PressureWork& work) noexcept {
    try {
        if (work.option.evicts_continuation || work.completed) { std::terminate(); }
        SequenceState* sequence =
            work.shared_owner ? nullptr : &continuation_states[work.continuation_index];
        SharedPrefixState* shared =
            work.shared_owner ? &shared_prefix_states[work.continuation_index] : nullptr;
        if (work.state_changes.size() != work.option.state_changes.size()) { std::terminate(); }
        for (std::size_t index = 0; index < work.option.state_changes.size(); ++index) {
            const qwen3_6::detail::PressureStateDecision action = work.option.state_changes[index];
            auto& change                                        = work.state_changes[index];
            const std::optional<StateImageHandle> source =
                pressure_state_source(action, sequence, shared);
            if (!source) { std::terminate(); }
            if (change.transfer) {
                state_store->publish_transfer(std::move(*change.transfer), false);
                change.transfer.reset();
                work.mutation_published = true;
            } else if (!change.host_released) {
                if (pressure_state_drops_host(action)
                        ? !state_store->drop_host_replica(*source)
                        : !state_store->drop_device_replica(*source)) {
                    std::terminate();
                }
                work.mutation_published = true;
            }
        }

        const auto publish_kv =
            [&](LogicalKVPageStore& pages, const qwen3_6::detail::PressureKVDecision& action,
                MaterializationTransaction::PressureWork::KVChangeWork& change) {
                if (action.kind == qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate) {
                    if (change.host_released) { return; }
                    if (!host_kv_extents || change.backup) { std::terminate(); }
                    if (!host_kv_extents->release_page_replicas(pages, change.pages)) {
                        std::terminate();
                    }
                    work.mutation_published = true;
                    return;
                }
                if (change.backup) {
                    if (!host_kv_extents) { std::terminate(); }
                    (void)host_kv_extents->publish(std::move(*change.backup));
                    change.backup.reset();
                }
                for (const LogicalKVPageHandle page : change.pages) {
                    if (!pages.drop_device_replica(page)) { std::terminate(); }
                }
                work.mutation_published = true;
            };
        if (work.main_kv_changes.size() != work.option.main_kv_changes.size() ||
            work.backend_kv_changes.size() != work.option.backend_kv_changes.size()) {
            std::terminate();
        }
        for (std::size_t index = 0; index < work.option.main_kv_changes.size(); ++index) {
            publish_kv(*text_kv_pages, work.option.main_kv_changes[index],
                       work.main_kv_changes[index]);
            if (work.option.main_kv_changes[index].kind ==
                qwen3_6::detail::PressureKVDecisionKind::DemoteToHost) {
                work.spill_pages += work.main_kv_changes[index].pages.size();
            }
        }
        if (!work.option.backend_kv_changes.empty()) {
            if (!backend_kv_pages) { std::terminate(); }
            for (std::size_t index = 0; index < work.option.backend_kv_changes.size(); ++index) {
                publish_kv(*backend_kv_pages, work.option.backend_kv_changes[index],
                           work.backend_kv_changes[index]);
                if (work.option.backend_kv_changes[index].kind ==
                    qwen3_6::detail::PressureKVDecisionKind::DemoteToHost) {
                    work.spill_pages += work.backend_kv_changes[index].pages.size();
                }
            }
        }
        work.submitted = false;
        work.completed = true;
    } catch (...) { std::terminate(); }
}

void ProgramImplCore::abort_pressure_work(MaterializationTransaction::PressureWork& work) noexcept {
    try {
        if (work.completed) { return; }
        for (auto& change : work.state_changes) {
            if (change.transfer) {
                state_store->abort_transfer(std::move(*change.transfer));
                change.transfer.reset();
            }
        }
        for (auto& change : work.main_kv_changes) { change.backup.reset(); }
        for (auto& change : work.backend_kv_changes) { change.backup.reset(); }
        work.state_changes.clear();
        work.main_kv_changes.clear();
        work.backend_kv_changes.clear();
        work.submitted = false;
    } catch (...) { std::terminate(); }
}

ProgramImplCore::PhysicalReleaseResult
ProgramImplCore::release_materialization_victim(MaterializationTransaction& transaction,
                                                std::size_t position) {
    PhysicalReleaseResult out;
    if (position >= transaction.victim_count || transaction.victim_released[position]) {
        return out;
    }
    const std::uint32_t index      = transaction.victim_indices[position];
    const std::uint64_t generation = transaction.victim_generations[position];
    if (index >= continuation_capacity ||
        continuation_slots[index].role != ContinuationSlotRole::Catalogued ||
        continuation_slots[index].generation != generation) {
        return out;
    }
    if (!can_release_continuation_slot_strict(index)) {
        throw std::logic_error("materialization victim is not strictly releasable");
    }

    out.delta.removed = resident_resources(continuation_states[index]);
    release_continuation_slot_strict(index);
    if (transaction.root_waiting_for_victim && transaction.root_continuation_index == index) {
        continuation_slots[index].role      = ContinuationSlotRole::ReservedMaterialization;
        transaction.root_waiting_for_victim = false;
    }
    transaction.victim_released[position] = true;
    out.status                            = runtime::ConsumeStatus::Consumed;
    return out;
}

MaterializationResult
ProgramImplCore::progress_materialization_transaction(runtime::CancellationFlagView cancellation) {
    MaterializationResult out;
    MaterializationTransaction* transaction_ptr =
        std::get_if<MaterializationTransaction>(&context_transaction_);
    if (transaction_ptr == nullptr || transaction_ptr->terminal) {
        throw std::logic_error("Program has no progressable context transaction");
    }
    MaterializationTransaction& transaction = *transaction_ptr;
    PressureTransition& pressure_transition = transaction.pressure_transition;
    const auto collect_pressure_operations  = [&](MaterializationTransaction::PressureWork& work) {
        if (work.spill_pages > std::numeric_limits<std::uint64_t>::max() -
                                   transaction.operations.pressure_spill_pages) {
            transaction.operations.pressure_spill_pages = std::numeric_limits<std::uint64_t>::max();
        } else {
            transaction.operations.pressure_spill_pages += work.spill_pages;
        }
        work.spill_pages = 0;
    };
    const auto retain_private_result = [&](auto& result, const SequenceState& state) {
        if (!result.final_summary) {
            throw std::logic_error("private acknowledgement backing was not reserved");
        }
        using Result = std::remove_cvref_t<decltype(result)>;
        if constexpr (std::is_same_v<Result, MaterializationSourceResult>) {
            result.mode = runtime::PrivateSourceMode::Retain;
        } else {
            result.disposition = runtime::VictimDisposition::Retained;
        }
        populate_continuation_summary(state, *result.final_summary);
    };
    const auto evict_private_result = [&](MaterializationVictimResult& result) {
        result.disposition        = runtime::VictimDisposition::Evicted;
        result.pressure_committed = true;
        result.final_summary.reset();
    };
    const auto complete_victim_acknowledgement = [&]() {
        for (std::size_t position = 0; position < transaction.victim_count; ++position) {
            if (transaction.victim_released[position]) { continue; }
            const std::uint32_t index      = transaction.victim_indices[position];
            const std::uint64_t generation = transaction.victim_generations[position];
            if (index >= continuation_capacity ||
                continuation_slots[index].role != ContinuationSlotRole::Catalogued ||
                continuation_slots[index].generation != generation) {
                throw std::logic_error("unmodified pressure claim is unavailable");
            }
            retain_private_result(transaction.pressure_results[position],
                                  continuation_states[index]);
            const MaterializationTransaction::PressureWork& work = transaction.pressure[position];
            transaction.pressure_results[position].pressure_committed = work.mutation_published;
        }
        out.victims = std::move(transaction.pressure_results);
    };
    const auto complete_source_acknowledgement = [&](bool published) {
        if (!transaction.has_source) { return; }
        if (published && transaction.source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
            out.source.emplace(MaterializationSourceResult{
                .mode = runtime::PrivateSourceMode::ConsumeToActive,
            });
            return;
        }
        if (transaction.source_index >= continuation_capacity ||
            continuation_slots[transaction.source_index].role != ContinuationSlotRole::Catalogued ||
            continuation_slots[transaction.source_index].generation !=
                transaction.source_generation) {
            throw std::logic_error("retained materialization source is unavailable");
        }
        SequenceState& source = continuation_states[transaction.source_index];
        if (!transaction.source_result) {
            throw std::logic_error("materialization source backing was not reserved");
        }
        retain_private_result(*transaction.source_result, source);
        out.source.emplace(std::move(*transaction.source_result));
    };
    const auto complete_shared_source_acknowledgement = [&](bool published) {
        if (!transaction.has_shared_source) { return; }
        if (transaction.shared_source_index >= shared_prefix_capacity ||
            shared_prefix_slots[transaction.shared_source_index].role !=
                SharedPrefixSlotRole::Catalogued ||
            shared_prefix_slots[transaction.shared_source_index].generation !=
                transaction.shared_source_generation) {
            throw std::logic_error("retained materialization shared source is unavailable");
        }
        const SharedPrefixState& source = shared_prefix_states[transaction.shared_source_index];
        if (!transaction.shared_source_result) {
            throw std::logic_error("materialization shared-source backing was not reserved");
        }
        transaction.shared_source_result->final_summary = shared_prefix_summary(source);
        out.shared_source.emplace(std::move(*transaction.shared_source_result));
        if (published && out.shared_source->final_summary->active_references == 0) {
            throw std::logic_error("published shared source lost its active reference");
        }
    };
    const auto complete_shared_victim_acknowledgement = [&]() {
        for (std::size_t position = 0; position < transaction.shared_victim_count; ++position) {
            if (transaction.shared_victim_released[position]) { continue; }
            const std::uint32_t index      = transaction.shared_victim_indices[position];
            const std::uint64_t generation = transaction.shared_victim_generations[position];
            if (index >= shared_prefix_capacity ||
                shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
                shared_prefix_slots[index].generation != generation) {
                throw std::logic_error("unmodified shared pressure claim is unavailable");
            }
            transaction.shared_pressure_results[position] = MaterializationSharedVictimResult{
                .owner              = transaction.shared_pressure_results[position].owner,
                .disposition        = runtime::VictimDisposition::Retained,
                .pressure_committed = transaction.shared_pressure[position].mutation_published,
                .final_summary      = shared_prefix_summary(shared_prefix_states[index]),
            };
        }
        out.shared_victims = std::move(transaction.shared_pressure_results);
    };
    const auto abort_transaction = [&]() {
        release_materialization_staging(transaction);
        transaction.terminal      = true;
        out.status                = runtime::ContextTransactionStatus::Aborted;
        out.transfer_observations = std::move(transaction.transfer_observations);
        out.operations            = transaction.operations;
        complete_source_acknowledgement(false);
        complete_shared_source_acknowledgement(false);
        complete_victim_acknowledgement();
        complete_shared_victim_acknowledgement();
    };

    if (cancellation.requested()) { transaction.cancel_pending = true; }

    if (pressure_transition.phase == PressureTransitionPhase::HostReleases) {
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }
        for (std::size_t position = 0; position < transaction.shared_victim_count; ++position) {
            MaterializationTransaction::PressureWork& work = transaction.shared_pressure[position];
            if (work.option.evicts_continuation) {
                const std::uint32_t index      = transaction.shared_victim_indices[position];
                const std::uint64_t generation = transaction.shared_victim_generations[position];
                if (index >= shared_prefix_capacity ||
                    shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
                    shared_prefix_slots[index].generation != generation ||
                    shared_prefix_states[index].active_references != 0) {
                    throw std::logic_error("shared pressure victim changed before release");
                }
                const detail::PhysicalResources resident =
                    resident_resources(shared_prefix_states[index]);
                if (work.option.effect.added != detail::PhysicalResources{}) {
                    throw std::logic_error("shared pressure eviction changed after reservation");
                }
                if (!can_release_shared_prefix_state(index, SharedPrefixSlotRole::Catalogued)) {
                    throw std::logic_error("shared pressure victim is not strictly releasable");
                }
                const detail::PhysicalResources released =
                    release_shared_prefix_state_strict(index, SharedPrefixSlotRole::Catalogued);
                if (released != resident) {
                    throw std::logic_error("shared pressure eviction acknowledgement is invalid");
                }
                work.committed_delta    = detail::PhysicalDelta{.removed = released};
                work.completed          = true;
                work.mutation_published = true;
                transaction.shared_pressure_results[position] = MaterializationSharedVictimResult{
                    .owner              = transaction.shared_pressure_results[position].owner,
                    .disposition        = runtime::VictimDisposition::Evicted,
                    .pressure_committed = true,
                };
                transaction.shared_victim_released[position] = true;
            } else {
                publish_pressure_host_releases(work);
            }
        }
        for (std::size_t position = 0; position < transaction.victim_count; ++position) {
            MaterializationTransaction::PressureWork& work = transaction.pressure[position];
            if (work.option.evicts_continuation) {
                const PhysicalReleaseResult released =
                    release_materialization_victim(transaction, position);
                if (released.status != runtime::ConsumeStatus::Consumed ||
                    released.delta.added != detail::PhysicalResources{} ||
                    work.option.effect.added != detail::PhysicalResources{}) {
                    throw std::logic_error("materialization eviction changed after reservation");
                }
                work.committed_delta    = released.delta;
                work.completed          = true;
                work.mutation_published = true;
                evict_private_result(transaction.pressure_results[position]);
            } else {
                publish_pressure_host_releases(work);
                if (work.completed) {
                    SequenceState& victim = continuation_states[work.continuation_index];
                    retain_private_result(transaction.pressure_results[position], victim);
                    transaction.pressure_results[position].pressure_committed = true;
                    transaction.victim_released[position]                     = true;
                }
            }
        }
        pressure_transition.phase = PressureTransitionPhase::CopyPreparation;
        if (cancellation.requested()) { transaction.cancel_pending = true; }
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }
    }

    const auto complete_pressure_delta = [&](MaterializationTransaction::PressureWork& work) {
        (void)checked_resource_difference(work.option.effect.removed, work.committed_delta.removed);
        (void)checked_resource_difference(work.option.effect.added, work.committed_delta.added);
        work.committed_delta = work.option.effect;
    };

    const auto for_each_pending_pressure = [&](auto&& callback) {
        for (MaterializationTransaction::PressureWork& work : transaction.shared_pressure) {
            if (!work.completed) { callback(work); }
        }
        for (MaterializationTransaction::PressureWork& work : transaction.pressure) {
            if (!work.completed) { callback(work); }
        }
    };

    if (pressure_transition.phase == PressureTransitionPhase::CopyPreparation) {
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }

        constexpr std::array pressure_resources{
            runtime::ContextResourceClass::State,
            runtime::ContextResourceClass::MainKV,
            runtime::ContextResourceClass::BackendKV,
        };
        try {
            for (const runtime::ContextResourceClass resource : pressure_resources) {
                bool has_copy = false;
                for_each_pending_pressure(
                    [&](const MaterializationTransaction::PressureWork& work) {
                        has_copy =
                            has_copy ||
                            std::any_of(
                                work.option.transfer_requirements.begin(),
                                work.option.transfer_requirements.end(),
                                [&](const auto& requirement) {
                                    return requirement.resource == resource &&
                                           requirement.direction ==
                                               runtime::ContextTransferDirection::DeviceToHost;
                                });
                    });
                if (has_copy) { start_context_transfer_timer(resource); }
                try {
                    for_each_pending_pressure([&](MaterializationTransaction::PressureWork& work) {
                        prepare_pressure_work(work, resource);
                    });
                } catch (...) {
                    if (has_copy) { stop_context_transfer_timer(resource); }
                    throw;
                }
                if (!has_copy) { continue; }
                stop_context_transfer_timer(resource);
                const std::size_t resource_index = context_resource_index(resource);
                pressure_transition.timer_mask |= static_cast<std::uint8_t>(1U << resource_index);
                for_each_pending_pressure(
                    [&](const MaterializationTransaction::PressureWork& work) {
                        for (const runtime::ContextTransferRequirement& requirement :
                             work.option.transfer_requirements) {
                            if (requirement.resource != resource ||
                                requirement.direction !=
                                    runtime::ContextTransferDirection::DeviceToHost) {
                                continue;
                            }
                            TransferWork& total = pressure_transition.transfer_work[resource_index];
                            total.payload_bytes =
                                requirement.work.payload_bytes >
                                        std::numeric_limits<std::uint64_t>::max() -
                                            total.payload_bytes
                                    ? std::numeric_limits<std::uint64_t>::max()
                                    : total.payload_bytes + requirement.work.payload_bytes;
                            const std::uint64_t operations =
                                static_cast<std::uint64_t>(total.copy_operations) +
                                requirement.work.copy_operations;
                            total.copy_operations =
                                operations > std::numeric_limits<std::uint32_t>::max()
                                    ? std::numeric_limits<std::uint32_t>::max()
                                    : static_cast<std::uint32_t>(operations);
                            const std::uint64_t pages =
                                static_cast<std::uint64_t>(
                                    pressure_transition.transfer_pages[resource_index]) +
                                requirement.page_count;
                            pressure_transition.transfer_pages[resource_index] =
                                pages > std::numeric_limits<std::uint32_t>::max()
                                    ? std::numeric_limits<std::uint32_t>::max()
                                    : static_cast<std::uint32_t>(pages);
                            if (resource == runtime::ContextResourceClass::State) {
                                pressure_transition.state_images =
                                    requirement.units > std::numeric_limits<std::uint64_t>::max() -
                                                            pressure_transition.state_images
                                        ? std::numeric_limits<std::uint64_t>::max()
                                        : pressure_transition.state_images + requirement.units;
                            }
                        }
                    });
            }
        } catch (...) {
            (void)cudaStreamSynchronize(device.transfer_stream);
            for_each_pending_pressure(
                [&](MaterializationTransaction::PressureWork& work) { abort_pressure_work(work); });
            throw;
        }

        bool copies_submitted = false;
        for_each_pending_pressure([&](const MaterializationTransaction::PressureWork& work) {
            copies_submitted = copies_submitted || work.submitted;
        });
        pressure_transition.phase = copies_submitted ? PressureTransitionPhase::CopiesInFlight
                                                     : PressureTransitionPhase::CopyPublication;
        if (copies_submitted) {
            context_completion_.record(device.transfer_stream);
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
    }

    if (pressure_transition.phase == PressureTransitionPhase::CopiesInFlight) {
        if (!context_completion_.ready()) {
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
        pressure_transition.phase = PressureTransitionPhase::CopyPublication;
    }
    if (transaction.cancel_pending) {
        // D2H destinations are still private reservations.  Waiting for the stream and aborting
        // them leaves only the already committed PreRelease changes visible.
        abort_transaction();
        return out;
    }

    if (pressure_transition.phase == PressureTransitionPhase::CopyPublication) {
        for (std::size_t position = 0; position < transaction.shared_pressure.size(); ++position) {
            MaterializationTransaction::PressureWork& work = transaction.shared_pressure[position];
            if (work.completed) { continue; }
            publish_pressure_work(work);
            collect_pressure_operations(work);
            const std::uint32_t index = transaction.shared_victim_indices[position];
            transaction.shared_pressure_results[position] = MaterializationSharedVictimResult{
                .owner              = transaction.shared_pressure_results[position].owner,
                .disposition        = runtime::VictimDisposition::Retained,
                .pressure_committed = true,
                .final_summary      = shared_prefix_summary(shared_prefix_states[index]),
            };
            complete_pressure_delta(work);
            transaction.shared_victim_released[position] = true;
        }
        transaction.shared_pressure_cursor = transaction.shared_pressure.size();

        for (std::size_t position = 0; position < transaction.pressure.size(); ++position) {
            MaterializationTransaction::PressureWork& work = transaction.pressure[position];
            if (work.completed) { continue; }
            publish_pressure_work(work);
            collect_pressure_operations(work);
            retain_private_result(transaction.pressure_results[position],
                                  continuation_states[work.continuation_index]);
            transaction.pressure_results[position].pressure_committed = true;
            complete_pressure_delta(work);
            transaction.victim_released[position] = true;
        }
        transaction.pressure_cursor = transaction.pressure.size();

        constexpr std::array pressure_resources{
            runtime::ContextResourceClass::State,
            runtime::ContextResourceClass::MainKV,
            runtime::ContextResourceClass::BackendKV,
        };
        for (const runtime::ContextResourceClass resource : pressure_resources) {
            const std::size_t index = context_resource_index(resource);
            const std::uint8_t bit  = static_cast<std::uint8_t>(1U << index);
            if ((pressure_transition.timer_mask & bit) == 0) { continue; }
            transaction.transfer_observations.push_back(context_transfer_observation(
                resource, runtime::ContextTransferDirection::DeviceToHost,
                pressure_transition.transfer_work[index], pressure_transition.transfer_pages[index],
                pressure_transition.state_images));
        }
        pressure_transition.timer_mask = 0;
        pressure_transition.phase      = PressureTransitionPhase::Committed;
        if (cancellation.requested()) { transaction.cancel_pending = true; }
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }
    }
    if (pressure_transition.phase != PressureTransitionPhase::Committed) {
        throw std::logic_error("materialization pressure transition did not reach a stable phase");
    }

    if (!transaction.source_prepared) {
        prepare_consumed_source(transaction);
        if (cancellation.requested()) { transaction.cancel_pending = true; }
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }
    }

    if (transaction.transfer_submitted) {
        if (!context_completion_.ready()) {
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
        if (transaction.cancel_pending) {
            abort_transaction();
            return out;
        }
        publish_materialization_transfers(transaction);
        if (transaction.transfer_submitted) {
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
    }

    if (transaction.cancel_pending) {
        abort_transaction();
        return out;
    }

    if (!transaction.prepared) {
        prepare_materialization(transaction);
        enqueue_materialization_transfers(transaction);
        if (transaction.transfer_submitted) {
            out.status = runtime::ContextTransactionStatus::InProgress;
            return out;
        }
    }
    if (cancellation.requested()) {
        abort_transaction();
        return out;
    }

    // This is the unique physical publication point. ResourceManager still owns the logical
    // catalog capabilities and adopts them only after validating this terminal result.
    try {
        out.published.emplace(start_request(transaction));
        materialization_ledger_.clear();
        materialization_identity_.clear();
        materialization_prefix_digests_.clear();
    } catch (...) {
        release_materialization_staging(transaction);
        throw;
    }
    transaction.terminal      = true;
    out.status                = runtime::ContextTransactionStatus::Published;
    out.transfer_observations = std::move(transaction.transfer_observations);
    out.operations            = transaction.operations;
    complete_source_acknowledgement(true);
    complete_shared_source_acknowledgement(true);
    complete_victim_acknowledgement();
    complete_shared_victim_acknowledgement();
    return out;
}

ContextTransactionProgress<Variant>
ProgramImplCore::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    const auto terminal_or_pending =
        []<class Result>(Result&& result) -> ContextTransactionProgress<Variant> {
        if (result.status == runtime::ContextTransactionStatus::InProgress) {
            return runtime::ContextTransactionInProgress{};
        }
        if (result.status != runtime::ContextTransactionStatus::Published &&
            result.status != runtime::ContextTransactionStatus::Aborted) {
            throw std::logic_error("context transaction returned an invalid status");
        }
        return ContextTransactionProgress<Variant>(std::forward<Result>(result));
    };
    return std::visit(
        [&](auto& transaction) -> ContextTransactionProgress<Variant> {
            using Transaction = std::decay_t<decltype(transaction)>;
            if constexpr (std::is_same_v<Transaction, std::monostate>) {
                throw std::logic_error("Program has no progressable context transaction");
            } else if constexpr (std::is_same_v<Transaction, MaterializationTransaction>) {
                return terminal_or_pending(progress_materialization_transaction(cancellation));
            } else {
                return terminal_or_pending(progress_active_capture_transaction(cancellation));
            }
        },
        context_transaction_);
}

void ProgramImplCore::finalize_context_transaction() noexcept {
    const bool terminal = std::visit(
        [](const auto& transaction) {
            using T = std::decay_t<decltype(transaction)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return false;
            } else if constexpr (std::is_same_v<T, ActiveCaptureTransaction>) {
                return transaction.published;
            } else {
                return transaction.terminal;
            }
        },
        context_transaction_);
    if (terminal) { context_transaction_.emplace<std::monostate>(); }
}

bool ProgramImplCore::has_context_transaction() const noexcept {
    return !std::holds_alternative<std::monostate>(context_transaction_);
}

bool ProgramImplCore::valid_sequence(SequenceHandle handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t lane = ContractAccess::lane(handle).value;
    if (lane >= max_concurrency || ContractAccess::epoch(handle) != lane_epochs[lane]) {
        return false;
    }
    if (active_continuations[lane] >= continuation_capacity ||
        continuation_slots[active_continuations[lane]].role != ContinuationSlotRole::Active) {
        return false;
    }
    const Lifecycle lifecycle = requests[lane].lifecycle;
    return lifecycle == Lifecycle::Prefilling || lifecycle == Lifecycle::Active ||
           lifecycle == Lifecycle::Pending || lifecycle == Lifecycle::Finishable;
}

bool ProgramImplCore::valid_continuation(const ContinuationHandle& handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t index = ContractAccess::index(handle);
    return index < continuation_capacity &&
           ContractAccess::epoch(handle) == continuation_slots[index].generation &&
           continuation_slots[index].role == ContinuationSlotRole::Catalogued;
}

bool ProgramImplCore::valid_shared_prefix(const SharedPrefixHandle& handle) const noexcept {
    if (ContractAccess::owner(handle) != this) { return false; }
    const std::uint32_t index = ContractAccess::index(handle);
    return index < shared_prefix_capacity &&
           ContractAccess::epoch(handle) == shared_prefix_slots[index].generation &&
           shared_prefix_slots[index].role == SharedPrefixSlotRole::Catalogued;
}

bool ProgramImplCore::valid_capture_offer(const CaptureOffer& offer) const noexcept {
    if (ContractAccess::owner(offer) != this) { return false; }
    const std::uint32_t lane = ContractAccess::lane(offer).value;
    if (lane >= max_concurrency || ContractAccess::epoch(offer) != lane_epochs[lane] ||
        (requests[lane].lifecycle != Lifecycle::Prefilling &&
         requests[lane].lifecycle != Lifecycle::Active) ||
        !requests[lane].prefill) {
        return false;
    }
    const RequestControl::Prefill& prefill = *requests[lane].prefill;
    return prefill.pending_capture_offer != 0 &&
           prefill.pending_capture_offer == ContractAccess::id(offer) &&
           prefill.next_capture < prefill.capture_groups.size() &&
           prefill.cursor == prefill.capture_groups[prefill.next_capture].frontier;
}

bool ProgramImplCore::materialization_pins(std::uint32_t index,
                                           std::uint64_t generation) const noexcept {
    const MaterializationTransaction* transaction_ptr =
        std::get_if<MaterializationTransaction>(&context_transaction_);
    if (transaction_ptr == nullptr) { return false; }
    const MaterializationTransaction& transaction = *transaction_ptr;
    if (transaction.has_source && transaction.source_index == index &&
        transaction.source_generation == generation) {
        return true;
    }
    for (std::size_t victim = 0; victim < transaction.victim_count; ++victim) {
        if (!transaction.victim_released[victim] && transaction.victim_indices[victim] == index &&
            transaction.victim_generations[victim] == generation) {
            return true;
        }
    }
    return false;
}

bool ProgramImplCore::valid_pending(const PendingBatch& pending) const noexcept {
    if (ContractAccess::owner(pending) != this || !pending_transaction_ ||
        ContractAccess::transaction(pending) != pending_transaction_->id) {
        return false;
    }
    const auto rows = ContractAccess::rows(pending);
    if (rows.size() != pending_transaction_->size) { return false; }
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (!valid_sequence(rows[row]) ||
            ContractAccess::lane(rows[row]).value != pending_transaction_->lanes[row] ||
            ContractAccess::epoch(rows[row]) != pending_transaction_->epochs[row] ||
            requests[pending_transaction_->lanes[row]].lifecycle != Lifecycle::Pending) {
            return false;
        }
    }
    return true;
}

void ProgramImplCore::invalidate_lane(std::uint32_t lane) noexcept {
    if (lane >= max_concurrency) { return; }
    ++lane_epochs[lane];
    if (lane_epochs[lane] == 0) { ++lane_epochs[lane]; }
}

SequenceState& ProgramImplCore::active_sequence(std::uint32_t lane) {
    if (lane >= max_concurrency) { throw std::out_of_range("active lane is out of range"); }
    const std::uint32_t index = active_continuations[lane];
    if (index >= continuation_capacity ||
        continuation_slots[index].role != ContinuationSlotRole::Active) {
        throw std::logic_error("active lane has no continuation binding");
    }
    return continuation_states[index];
}

const SequenceState& ProgramImplCore::active_sequence(std::uint32_t lane) const {
    if (lane >= max_concurrency) { throw std::out_of_range("active lane is out of range"); }
    const std::uint32_t index = active_continuations[lane];
    if (index >= continuation_capacity ||
        continuation_slots[index].role != ContinuationSlotRole::Active) {
        throw std::logic_error("active lane has no continuation binding");
    }
    return continuation_states[index];
}

std::optional<std::uint32_t> ProgramImplCore::allocate_continuation_slot() noexcept {
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role == ContinuationSlotRole::Free) {
            continuation_slots[index].role = ContinuationSlotRole::Active;
            return index;
        }
    }
    return std::nullopt;
}

bool ProgramImplCore::can_release_continuation_slot_strict(std::uint32_t index) const {
    if (index >= continuation_capacity || !state_store || !text_kv_addresses || !text_kv_pages ||
        continuation_slots[index].role != ContinuationSlotRole::Catalogued) {
        return false;
    }
    const SequenceState& sequence = continuation_states[index];
    if (sequence.state.fork_pending || !sequence.shared_prefix_references.empty() || !sequence.kv ||
        !text_kv_addresses->can_release(sequence.kv->text)) {
        return false;
    }
    if (sequence.kv->backend) {
        if (!backend_kv_addresses || !backend_kv_pages ||
            !backend_kv_addresses->can_release(*sequence.kv->backend)) {
            return false;
        }
    }

    const auto validate_state = [&](StateImageHandle handle, bool release_object) {
        if (!state_store->valid(handle)) { return false; }
        const std::uint32_t owned = owned_checkpoint_references(sequence, handle);
        const std::uint32_t total = state_store->checkpoint_references(handle);
        if (owned > total ||
            (owned != 0 && state_store->role(handle) != StateImageRole::CheckpointImmutable)) {
            return false;
        }
        return !release_object || total != owned ||
               state_store->can_release_after_checkpoint_references(handle, owned);
    };
    const auto repeated_before_anchor = [&](std::size_t anchor_index, StateImageHandle handle) {
        if (handle == sequence.state.read || handle == sequence.state.write ||
            (sequence.rewrite_state && handle == *sequence.rewrite_state)) {
            return true;
        }
        for (std::size_t prior = 0; prior < anchor_index; ++prior) {
            if (sequence.long_anchors[prior].state == handle) { return true; }
        }
        return false;
    };

    if (sequence.endpoint_valid) {
        if (!validate_state(sequence.state.read, !sequence.state_source_retained ||
                                                     sequence.state.read == sequence.state.write)) {
            return false;
        }
        if (sequence.state.write != sequence.state.read &&
            !validate_state(sequence.state.write, true)) {
            return false;
        }
    } else if (state_store->valid(sequence.state.read) ||
               state_store->valid(sequence.state.write) || sequence.state_source_retained) {
        return false;
    }
    if (sequence.rewrite_state && *sequence.rewrite_state != sequence.state.read &&
        *sequence.rewrite_state != sequence.state.write &&
        !validate_state(*sequence.rewrite_state, true)) {
        return false;
    }
    for (std::size_t anchor = 0; anchor < sequence.long_anchors.size(); ++anchor) {
        const StateImageHandle handle = sequence.long_anchors[anchor].state;
        if (!repeated_before_anchor(anchor, handle) && !validate_state(handle, true)) {
            return false;
        }
    }
    if (sequence.reserved_state) {
        const StateImageHandle handle = *sequence.reserved_state;
        bool repeated = handle == sequence.state.read || handle == sequence.state.write ||
                        (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            repeated = repeated || anchor.state == handle;
        }
        if (!repeated && !validate_state(handle, true)) { return false; }
    }
    return true;
}

void ProgramImplCore::release_continuation_slot_strict(std::uint32_t index) noexcept {
    try {
        if (!can_release_continuation_slot_strict(index)) { std::terminate(); }
    } catch (...) { std::terminate(); }
    SequenceState& sequence = continuation_states[index];
    release_sequence_kv_strict(sequence);
    release_sequence_state_strict(sequence);
    retire_continuation_slot(index);
}

void ProgramImplCore::release_continuation_slot_best_effort(std::uint32_t index) noexcept {
    if (index >= continuation_capacity ||
        continuation_slots[index].role == ContinuationSlotRole::Free) {
        return;
    }
    SequenceState& sequence = continuation_states[index];
    release_active_shared_references(sequence);
    release_sequence_kv(sequence);
    release_sequence_state(sequence);
    retire_continuation_slot(index);
}

void ProgramImplCore::retire_continuation_slot(std::uint32_t index) noexcept {
    if (index >= continuation_capacity) { std::terminate(); }
    SequenceState& sequence     = continuation_states[index];
    sequence.execution_frontier = 0;
    sequence.ledger_frontier    = 0;
    sequence.ledger.clear();
    sequence.prefix_identity.clear();
    sequence.prefix_digests.clear();
    sequence.rope_delta              = 0;
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
    sequence.mtp_draft_count         = 0;
    sequence.tail_hidden_valid       = false;
    sequence.endpoint_valid          = false;
    sequence.rewrite_checkpoint      = {};
    sequence.rebuild_work            = {};
    sequence.rebuild_tail_begin      = 0;
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (active_continuations[lane] == index) {
            active_continuations[lane] = continuation_capacity;
        }
    }
    ContinuationSlot& slot = continuation_slots[index];
    slot.role              = ContinuationSlotRole::Free;
    if (++slot.generation == 0) { ++slot.generation; }
}

detail::PhysicalResources ProgramImplCore::resident_resources(const SequenceState& sequence) const {
    if (!state_store || !text_kv_addresses || !text_kv_pages) {
        throw std::logic_error("resident sequence resources have no physical stores");
    }
    detail::PhysicalResources out;
    {
        std::array<StateImageHandle, 4> states{};
        std::uint32_t state_count = 0;
        const auto add_state      = [&](StateImageHandle handle) {
            if (!state_store->valid(handle)) {
                throw std::logic_error("resident sequence has a stale StateImage");
            }
            if (!state_exclusive_to_sequence(sequence, handle)) { return; }
            for (std::uint32_t index = 0; index < state_count; ++index) {
                if (states[index] == handle) { return; }
            }
            states[state_count++]                 = handle;
            const StateReplicaResidency residency = state_store->residency(handle);
            if (residency == StateReplicaResidency::DeviceOnly ||
                residency == StateReplicaResidency::Both) {
                ++out.device.state_slots;
            }
            if (residency == StateReplicaResidency::HostOnly ||
                residency == StateReplicaResidency::Both) {
                ++out.host.state_slots;
            }
        };
        const bool has_read_state  = sequence.state.read.valid();
        const bool has_write_state = sequence.state.write.valid();
        if (has_read_state != has_write_state) {
            throw std::logic_error("resident sequence has a partial primary StateImage pair");
        }
        if (has_read_state) {
            if (!sequence.state_source_retained || sequence.state.read == sequence.state.write) {
                add_state(sequence.state.read);
            }
            add_state(sequence.state.write);
        }
        if (sequence.rewrite_state) { add_state(*sequence.rewrite_state); }
        if (sequence.reserved_state) { add_state(*sequence.reserved_state); }
        for (std::size_t anchor_index = 0; anchor_index < sequence.long_anchors.size();
             ++anchor_index) {
            const StateImageHandle handle = sequence.long_anchors[anchor_index].state;
            if (!state_store->valid(handle)) {
                throw std::logic_error("resident sequence has a stale long-anchor StateImage");
            }
            if (!state_exclusive_to_sequence(sequence, handle)) { continue; }
            bool seen = false;
            for (std::uint32_t index = 0;
                 index < std::min<std::uint32_t>(state_count, states.size()); ++index) {
                if (states[index] == handle) { seen = true; }
            }
            for (std::size_t prior = 0; !seen && prior < anchor_index; ++prior) {
                if (sequence.long_anchors[prior].state == handle) { seen = true; }
            }
            if (seen) { continue; }
            const StateReplicaResidency residency = state_store->residency(handle);
            if (residency == StateReplicaResidency::DeviceOnly ||
                residency == StateReplicaResidency::Both) {
                ++out.device.state_slots;
            }
            if (residency == StateReplicaResidency::HostOnly ||
                residency == StateReplicaResidency::Both) {
                ++out.host.state_slots;
            }
        }

        if (!sequence.kv) { throw std::logic_error("resident sequence has no KV address bundle"); }
        const auto add_kv = [&](const KVAddressSpaceStore& addresses,
                                const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                std::uint32_t& device_pages) {
            if (!addresses.valid(address)) { throw std::logic_error("stale KV address space"); }
            for (std::uint32_t page = 0; page < addresses.mapped_pages(address); ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                // A shared logical page contributes to aggregate occupancy once. Releasing this
                // address cannot free either replica while another address still references it,
                // so it is not part of this owner's exact transition effect.
                if (pages.address_references(logical) > 1) { continue; }
                if (pages.device_resident(logical)) { ++device_pages; }
                if (pages.host_resident(logical)) {
                    if (!host_kv_extents) {
                        throw std::logic_error("missing Host KV extent store");
                    }
                    const HostKVPageReplica& replica = pages.host_replica(logical);
                    const std::size_t stride =
                        host_kv_extents->view(replica.extent).layout().page_stride;
                    if (stride > std::numeric_limits<std::size_t>::max() - out.host.kv_bytes) {
                        throw std::overflow_error("resident Host KV byte count overflow");
                    }
                    out.host.kv_bytes += stride;
                }
            }
            if (addresses.active(address)) {
                const std::uint32_t mapped      = addresses.mapped_pages(address);
                const std::uint32_t entitlement = addresses.entitlement(address);
                if (entitlement < mapped ||
                    entitlement - mapped >
                        std::numeric_limits<std::uint32_t>::max() - device_pages) {
                    throw std::logic_error("resident active KV entitlement is inconsistent");
                }
                device_pages += entitlement - mapped;
            }
        };
        add_kv(*text_kv_addresses, *text_kv_pages, sequence.kv->text, out.device.main_kv_pages);
        if (sequence.kv->backend) {
            if (!backend_kv_addresses || !backend_kv_pages) {
                throw std::logic_error("missing Backend KV stores");
            }
            add_kv(*backend_kv_addresses, *backend_kv_pages, *sequence.kv->backend,
                   out.device.backend_kv_pages);
        }
    }
    return out;
}

detail::PhysicalResources
ProgramImplCore::resident_resources(const SharedPrefixState& shared) const {
    if (!state_store || !text_kv_addresses || !text_kv_pages) {
        throw std::logic_error("shared resident resources have no physical stores");
    }
    detail::PhysicalResources out;
    {
        if (!shared.kv || !shared.identity || !state_store->valid(shared.state)) {
            throw std::logic_error("shared prefix has incomplete resident physical state");
        }
        if (state_store->checkpoint_references(shared.state) == 0) {
            throw std::logic_error("shared prefix StateImage has no checkpoint reference");
        }
        const StateReplicaResidency residency = state_store->residency(shared.state);
        if (state_store->checkpoint_references(shared.state) == 1) {
            if (residency == StateReplicaResidency::DeviceOnly ||
                residency == StateReplicaResidency::Both) {
                ++out.device.state_slots;
            }
            if (residency == StateReplicaResidency::HostOnly ||
                residency == StateReplicaResidency::Both) {
                ++out.host.state_slots;
            }
        }
        const auto add_kv = [&](const KVAddressSpaceStore& addresses,
                                const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                                std::uint32_t& device_pages) {
            if (!addresses.valid(address)) { throw std::logic_error("stale shared KV address"); }
            for (std::uint32_t page = 0; page < addresses.mapped_pages(address); ++page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                if (pages.address_references(logical) != 1) { continue; }
                if (pages.device_resident(logical)) { ++device_pages; }
                if (pages.host_resident(logical)) {
                    if (!host_kv_extents) {
                        throw std::logic_error("missing Host KV extent store");
                    }
                    const HostKVPageReplica& replica = pages.host_replica(logical);
                    const std::size_t stride =
                        host_kv_extents->view(replica.extent).layout().page_stride;
                    if (stride > std::numeric_limits<std::size_t>::max() - out.host.kv_bytes) {
                        throw std::overflow_error("shared Host KV byte count overflow");
                    }
                    out.host.kv_bytes += stride;
                }
            }
        };
        add_kv(*text_kv_addresses, *text_kv_pages, shared.kv->text, out.device.main_kv_pages);
        if (shared.kv->backend) {
            if (!backend_kv_addresses || !backend_kv_pages) {
                throw std::logic_error("missing shared Backend KV stores");
            }
            add_kv(*backend_kv_addresses, *backend_kv_pages, *shared.kv->backend,
                   out.device.backend_kv_pages);
        }
    }
    return out;
}

detail::PhysicalResources ProgramImplCore::physical_occupancy() const noexcept {
    detail::PhysicalResources out;
    for (const RequestControl& request : requests) {
        if (request.lifecycle != Lifecycle::Empty) { ++out.device.active_lanes; }
    }
    if (state_store) {
        out.device.state_slots = state_store->device_occupied();
        out.host.state_slots   = state_store->host_occupied();
    }
    if (text_kv_pages) {
        const DeviceKVPagePool& pool = text_kv_pages->physical_pool();
        out.device.main_kv_pages     = pool.allocated_pages() + pool.reserved_pages();
    }
    if (backend_kv_pages) {
        const DeviceKVPagePool& pool = backend_kv_pages->physical_pool();
        out.device.backend_kv_pages  = pool.allocated_pages() + pool.reserved_pages();
    }
    if (host_kv_arena) { out.host.kv_bytes = host_kv_arena->occupied_bytes(); }
    return out;
}

detail::PhysicalResources
ProgramImplCore::materialization_deficit(const ResourceCandidateState& admission) const {
    // Pressure is relative to this candidate's real peak. Treating every dimension as scarce
    // would forbid Device-to-Host demotion even when Host capacity is available.
    const detail::PhysicalResources required =
        checked_resource_sum(physical_occupancy(), admission.demand.physical_peak_additional);
    return positive_resource_difference(required, admission_capacity());
}

detail::PhysicalResources
ProgramImplCore::guided_materialization_deficit(const ResourceCandidateState& admission,
                                                const detail::PhysicalDelta& pressure) const {
    // Pressure acts on the candidate's complete peak, not on its already-clamped deficit.  Applying
    // a demotion directly to a zero Host deficit would otherwise manufacture Host pressure even
    // when the arena has ample slack and steer the heuristic toward unnecessary destruction.
    const detail::PhysicalResources projected_peak = positive_resource_difference(
        checked_resource_sum(admission.demand.physical_peak_additional, pressure.added),
        pressure.removed);
    const detail::PhysicalResources required =
        checked_resource_sum(physical_occupancy(), projected_peak);
    return positive_resource_difference(required, admission_capacity());
}

bool ProgramImplCore::physical_peak_fits(detail::PhysicalResources peak) const noexcept {
    const detail::PhysicalResources occupied = physical_occupancy();
    const detail::PhysicalResources limits   = admission_capacity();
    const auto fits_u32 = [](std::uint32_t used, std::uint32_t added, std::uint32_t capacity) {
        return added <= capacity && used <= capacity - added;
    };
    const auto fits_size = [](std::size_t used, std::size_t added, std::size_t capacity) {
        return added <= capacity && used <= capacity - added;
    };
    return fits_u32(occupied.device.active_lanes, peak.device.active_lanes,
                    limits.device.active_lanes) &&
           fits_u32(occupied.device.state_slots, peak.device.state_slots,
                    limits.device.state_slots) &&
           fits_u32(occupied.device.main_kv_pages, peak.device.main_kv_pages,
                    limits.device.main_kv_pages) &&
           fits_u32(occupied.device.backend_kv_pages, peak.device.backend_kv_pages,
                    limits.device.backend_kv_pages) &&
           fits_u32(occupied.host.state_slots, peak.host.state_slots, limits.host.state_slots) &&
           fits_size(occupied.host.kv_bytes, peak.host.kv_bytes, limits.host.kv_bytes);
}

StateImageHandle
ProgramImplCore::selected_state(const SequenceState& sequence, ReusePath reuse,
                                std::optional<runtime::CheckpointRef> checkpoint) const {
    if (reuse == ReusePath::PrivateEndpoint) {
        if (!sequence.endpoint_valid || !state_store->valid(sequence.state.read)) {
            throw std::logic_error("private endpoint StateImage is stale");
        }
        return sequence.state.read;
    }
    if (is_rewrite_checkpoint_restore(reuse) && sequence.rewrite_state &&
        state_store->valid(*sequence.rewrite_state)) {
        return *sequence.rewrite_state;
    }
    if (reuse == ReusePath::PrivateLongAnchor) {
        if (!checkpoint || checkpoint->kind != runtime::CheckpointKind::LongAnchor) {
            throw std::logic_error("long-anchor materialization has no selected checkpoint");
        }
        const auto anchor = std::find_if(sequence.long_anchors.begin(), sequence.long_anchors.end(),
                                         [&](const LongAnchorCheckpoint& candidate) {
                                             return candidate.frontier == checkpoint->frontier &&
                                                    candidate.ordinal == checkpoint->ordinal;
                                         });
        if (anchor != sequence.long_anchors.end() && state_store->valid(anchor->state)) {
            return anchor->state;
        }
    }
    throw std::logic_error("materialization path has no selected StateImage");
}

std::uint32_t ProgramImplCore::selected_state_consumed_references(
    const SequenceState& sequence, ReusePath reuse,
    RewriteCheckpointDisposition rewrite_disposition,
    std::optional<runtime::CheckpointRef> checkpoint, std::uint32_t reuse_base) const {
    const StateImageHandle selected   = selected_state(sequence, reuse, checkpoint);
    std::uint32_t consumed_references = 0;
    if (is_rewrite_checkpoint_restore(reuse) &&
        rewrite_disposition != RewriteCheckpointDisposition::RetainExisting) {
        if (!sequence.rewrite_state || *sequence.rewrite_state != selected) {
            throw std::logic_error("selected rewrite StateImage is unavailable");
        }
        consumed_references = 1;
    } else if (reuse == ReusePath::PrivateEndpoint &&
               rewrite_disposition != RewriteCheckpointDisposition::RetainExisting &&
               sequence.rewrite_state && *sequence.rewrite_state == selected) {
        consumed_references = 1;
    }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        if (anchor.frontier > reuse_base && anchor.state == selected) {
            if (consumed_references == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("consumed StateImage reference inventory overflow");
            }
            ++consumed_references;
        }
    }
    const std::uint32_t references = state_store->checkpoint_references(selected);
    if (consumed_references > references) {
        throw std::logic_error("selected StateImage reference inventory is inconsistent");
    }
    return consumed_references;
}

bool ProgramImplCore::selected_state_requires_fork(const SequenceState& sequence, ReusePath reuse,
                                                   RewriteCheckpointDisposition rewrite_disposition,
                                                   std::optional<runtime::CheckpointRef> checkpoint,
                                                   std::uint32_t reuse_base) const {
    const StateImageHandle selected = selected_state(sequence, reuse, checkpoint);
    return state_store->checkpoint_references(selected) !=
           selected_state_consumed_references(sequence, reuse, rewrite_disposition, checkpoint,
                                              reuse_base);
}

bool ProgramImplCore::can_retain_rewrite_checkpoint(const PreparedPromptData& prompt,
                                                    const RewriteCheckpointSpec& desired,
                                                    const SequenceState& sequence, ReusePath reuse,
                                                    std::uint32_t reuse_base) const {
    if (!sequence.rewrite_checkpoint.valid || !sequence.rewrite_state ||
        !state_store->valid(*sequence.rewrite_state) ||
        !qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                         sequence.rewrite_checkpoint.frontier)) {
        return false;
    }
    if (sequence.rewrite_checkpoint.frontier == desired.frontier) { return true; }
    return is_rewrite_checkpoint_restore(reuse) &&
           sequence.rewrite_checkpoint.frontier == reuse_base && desired.frontier <= reuse_base;
}

std::uint32_t ProgramImplCore::device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                      KVAddressSpaceHandle address,
                                                      std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t resident = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        if (pages.device_resident(addresses.logical_page(address, page))) { ++resident; }
    }
    return resident;
}

std::uint32_t ProgramImplCore::shared_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                      KVAddressSpaceHandle address,
                                                      std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t shared = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        if (pages.address_references(addresses.logical_page(address, page)) <= 1) { continue; }
        if (page + 1U == required && frontier % static_cast<std::uint32_t>(kPagedKVPageSize) != 0) {
            continue;
        }
        ++shared;
    }
    return shared;
}

std::uint32_t ProgramImplCore::shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                             KVAddressSpaceHandle address,
                                                             std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t resident = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        const LogicalKVPageHandle logical = addresses.logical_page(address, page);
        if (pages.address_references(logical) > 1 && pages.device_resident(logical)) { ++resident; }
    }
    return resident;
}

bool ProgramImplCore::partial_tail_cow_required(const KVAddressSpaceStore& addresses,
                                                KVAddressSpaceHandle address,
                                                std::uint32_t frontier) const {
    if (frontier == 0 || frontier % static_cast<std::uint32_t>(kPagedKVPageSize) == 0) {
        return false;
    }
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    const LogicalKVPageHandle tail = addresses.logical_page(address, required - 1U);
    return pages.address_references(tail) > 1 || !pages.device_resident(tail);
}

std::uint32_t
ProgramImplCore::missing_shared_device_kv_prefix_pages(const KVAddressSpaceStore& addresses,
                                                       KVAddressSpaceHandle address,
                                                       std::uint32_t frontier) const {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required > addresses.mapped_pages(address)) {
        throw std::logic_error("checkpoint KV requirement exceeds address membership");
    }
    const LogicalKVPageStore& pages =
        (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
    std::uint32_t missing = 0;
    for (std::uint32_t page = 0; page < required; ++page) {
        const LogicalKVPageHandle logical = addresses.logical_page(address, page);
        if (pages.address_references(logical) > 1 && !pages.device_resident(logical)) { ++missing; }
    }
    return missing;
}

std::size_t ProgramImplCore::host_kv_prefix_bytes(const KVAddressSpaceStore& addresses,
                                                  KVAddressSpaceHandle address,
                                                  std::uint32_t frontier) const noexcept {
    if (!host_kv_extents) { return 0; }
    try {
        const LogicalKVPageStore& pages =
            (&addresses == text_kv_addresses.get()) ? *text_kv_pages : *backend_kv_pages;
        const std::uint32_t required_pages = kv_pages_for_frontier(frontier);
        if (required_pages > addresses.mapped_pages(address)) { return 0; }
        std::size_t bytes = 0;
        for (std::uint32_t page = 0; page < required_pages; ++page) {
            const LogicalKVPageHandle logical = addresses.logical_page(address, page);
            if (pages.address_references(logical) > 1) { continue; }
            if (!pages.host_resident(logical)) { continue; }
            if (page + 1U == required_pages &&
                frontier % static_cast<std::uint32_t>(kPagedKVPageSize) != 0 &&
                partial_tail_cow_required(addresses, address, frontier)) {
                continue;
            }
            const std::uint32_t begin = page * static_cast<std::uint32_t>(kPagedKVPageSize);
            const std::uint32_t selected_columns =
                std::min(static_cast<std::uint32_t>(kPagedKVPageSize), frontier - begin);
            if (selected_columns != pages.committed_columns(logical)) {
                // A destructive private rewrite changes this tail page's content epoch, so its
                // old Host replica cannot remain part of the active entitlement.
                continue;
            }
            const std::size_t stride =
                host_kv_extents->view(pages.host_replica(logical).extent).layout().page_stride;
            if (stride > std::numeric_limits<std::size_t>::max() - bytes) { return 0; }
            bytes += stride;
        }
        return bytes;
    } catch (...) { return 0; }
}

PendingBatch ProgramImplCore::wrap_pending(std::span<const std::uint32_t> lanes,
                                           const runtime::BatchedGeneratedRound& round) {
    if (pending_transaction_ || lanes.empty() || lanes.size() > max_concurrency) {
        throw std::logic_error("Program already owns a pending transaction");
    }
    PendingTransaction transaction;
    transaction.id   = next_transaction_id_++;
    transaction.size = lanes.size();
    std::array<SequenceHandle, kMaximumConcurrency> handles{};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending) {
            throw std::logic_error("pending transaction membership is invalid");
        }
        transaction.lanes[row]  = lane;
        transaction.epochs[row] = lane_epochs[lane];
        handles[row] =
            ContractAccess::make_sequence(this, runtime::LaneId{lane}, lane_epochs[lane]);
    }
    pending_transaction_ = transaction;
    return ContractAccess::make_pending(
        this, transaction.id, std::span<const SequenceHandle>(handles.data(), lanes.size()),
        round.tokens, round.row_counts, round.row_stride, round.timing);
}

PrefillProgress ProgramImplCore::wrap_prefill(std::uint32_t lane, runtime::PrefillStepResult step) {
    PrefillProgress out;
    out.summary                 = step.summary;
    out.processed_prompt_tokens = step.processed_prompt_tokens;
    out.complete                = step.complete;
    out.timing                  = step.timing;
    if (step.complete) {
        const std::array<std::uint32_t, 1> lanes{lane};
        const runtime::BatchedGeneratedRound round{
            .tokens     = step.round.tokens,
            .row_counts = {},
            .row_stride = 1,
        };
        out.pending.emplace(wrap_pending(lanes, round));
    } else if (requests[lane].prefill && requests[lane].prefill->pending_capture_offer != 0) {
        out.capture.emplace(
            ContractAccess::make_capture_offer(this, runtime::LaneId{lane}, lane_epochs[lane],
                                               requests[lane].prefill->pending_capture_offer));
    }
    return out;
}

StartResult ProgramImplCore::start_request(MaterializationTransaction& transaction) {
    std::optional<std::uint32_t> destination = transaction.destination.value;
    std::optional<std::uint32_t> continuation_index;
    try {
        if (!transaction.prepared || !transaction.plan || !destination ||
            *destination >= max_concurrency) {
            throw std::invalid_argument("materialization transaction is not publishable");
        }
        const std::uint32_t lane              = *destination;
        const AdmissionCandidateImpl& details = *transaction.plan->impl_;
        if (details.destination_epoch != lane_epochs[lane] ||
            details.has_source != transaction.has_source ||
            details.has_shared_source != transaction.has_shared_source) {
            throw std::logic_error("admission plan physical epoch is stale");
        }
        if (requests[lane].lifecycle != Lifecycle::Empty ||
            active_continuations[lane] < continuation_capacity) {
            throw std::logic_error("admission destination is not free");
        }
        if (transaction.has_source &&
            transaction.source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
            if (transaction.source_index >= continuation_capacity ||
                continuation_slots[transaction.source_index].role !=
                    ContinuationSlotRole::Catalogued ||
                continuation_slots[transaction.source_index].generation !=
                    transaction.source_generation ||
                transaction.source_index != details.source_index ||
                transaction.source_generation != details.source_generation) {
                throw std::logic_error("admission source capability is stale");
            }
            continuation_index                           = transaction.source_index;
            continuation_slots[*continuation_index].role = ContinuationSlotRole::Active;
        } else {
            continuation_index = transaction.root_continuation_index;
            if (!continuation_index || transaction.root_waiting_for_victim ||
                continuation_slots[*continuation_index].role !=
                    ContinuationSlotRole::ReservedMaterialization) {
                throw std::logic_error("materialization continuation reservation is unavailable");
            }
            continuation_slots[*continuation_index].role = ContinuationSlotRole::Active;
        }

        const detail::PhysicalResources active = details.demand.active_entitlement;
        active_continuations[lane]             = *continuation_index;
        SequenceState& sequence                = continuation_states[*continuation_index];
        sequence.lane                          = lane;
        transaction.root_continuation_index.reset();
        start_sequence(lane, sequence, transaction);
        detail::PhysicalResources actual         = resident_resources(sequence);
        actual.device.active_lanes               = 1;
        const detail::PhysicalResources expected = active;
        if (actual != expected) {
            throw std::logic_error("materialized sequence does not match its active entitlement");
        }
        if (details.reuse != ReusePath::Root) {
            if (transaction.state_restored) {
                ++transaction.operations.state_restores;
            } else if (details.source_mode == runtime::PrivateSourceMode::Retain ||
                       transaction.has_shared_source || details.state_fork_required) {
                ++transaction.operations.state_forks;
                ++transaction.operations.historical_fork_hits;
            } else {
                ++transaction.operations.state_moves;
            }
        }
        requests[lane].active_resources   = active;
        requests[lane].optional_resources = details.active_optional_resources;
        invalidate_lane(lane);
        const SequenceHandle handle =
            ContractAccess::make_sequence(this, runtime::LaneId{lane}, lane_epochs[lane]);
        return StartResult{.sequence = handle};
    } catch (...) {
        if (destination && *destination < max_concurrency) {
            const std::uint32_t lane = *destination;
            if (active_continuations[lane] < continuation_capacity) {
                clear_lane_best_effort(active_sequence(lane), requests[lane]);
            } else if (continuation_index) {
                release_continuation_slot_best_effort(*continuation_index);
            }
            invalidate_lane(*destination);
        }
        throw;
    }
}

qwen3_6::CheckpointSummary
ProgramImplCore::checkpoint_summary(const SequenceState& sequence,
                                    runtime::CheckpointRef checkpoint, StateImageHandle state,
                                    runtime::PrefillWork rebuild_work) const {
    if (!sequence.kv) { throw std::logic_error("checkpoint summary has no KV address space"); }
    if (checkpoint.frontier == 0) {
        throw std::logic_error("checkpoint summary has an empty frontier");
    }
    if (!state_store->valid(state)) {
        throw std::logic_error("checkpoint summary has a stale StateImage");
    }
    const StateReplicaResidency state_location = state_store->residency(state);
    runtime::ReplicaResidency residency        = runtime::ReplicaResidency::DeviceOnly;
    if (state_location == StateReplicaResidency::HostOnly) {
        residency = runtime::ReplicaResidency::HostOnly;
    } else if (state_location == StateReplicaResidency::Both) {
        residency = runtime::ReplicaResidency::Both;
    } else if (state_location != StateReplicaResidency::DeviceOnly) {
        throw std::logic_error("checkpoint StateImage has no published replica");
    }
    const std::uint32_t backend_frontier =
        speculative_backend == SpeculativeBackend::Mtp      ? checkpoint.frontier - 1U
        : speculative_backend == SpeculativeBackend::DFlash ? checkpoint.frontier
                                                            : 0U;
    const std::uint32_t identity_tag = static_cast<std::uint32_t>(speculative_backend) |
                                       (static_cast<std::uint32_t>(proposal_head) << 8U) |
                                       (static_cast<std::uint32_t>(kv_storage) << 16U);
    return qwen3_6::CheckpointSummary{
        .ref   = checkpoint,
        .scope = runtime::CheckpointScope::Private,
        .shortlist_key =
            {
                .digests      = sequence.prefix_digests.at(checkpoint.frontier),
                .frontier     = checkpoint.frontier,
                .identity_tag = identity_tag,
            },
        .state_residency = residency,
        .required_kv =
            {
                .main_frontier    = checkpoint.frontier,
                .backend_frontier = backend_frontier,
                .main_pages       = kv_pages_for_frontier(checkpoint.frontier),
                .backend_pages    = kv_pages_for_frontier(backend_frontier),
            },
        .rebuild_work = validated_rebuild_work(rebuild_work, checkpoint.frontier),
    };
}

qwen3_6::ContinuationSummary
ProgramImplCore::continuation_summary(const SequenceState& sequence) const {
    qwen3_6::ContinuationSummary summary;
    summary.long_anchors.reserve(sequence.long_anchors.size());
    populate_continuation_summary(sequence, summary);
    return summary;
}

void ProgramImplCore::populate_continuation_summary(const SequenceState& sequence,
                                                    qwen3_6::ContinuationSummary& summary) const {
    validate_long_anchor_ordinals(sequence.long_anchors,
                                  context_cache.max_long_anchors_per_continuation.value_or(0));
    if (summary.long_anchors.capacity() < sequence.long_anchors.size()) {
        throw std::logic_error("continuation summary backing was not reserved");
    }
    summary.endpoint.reset();
    summary.rewrite.reset();
    summary.long_anchors.clear();
    summary.active_references = 0;
    if (sequence.endpoint_valid) {
        const runtime::CheckpointRef endpoint{
            .kind     = runtime::CheckpointKind::SessionEndpoint,
            .frontier = sequence.execution_frontier,
        };
        runtime::PrefillWork endpoint_work = sequence.rebuild_work;
        summary.endpoint =
            checkpoint_summary(sequence, endpoint, sequence.state.read, endpoint_work);
    }
    if (sequence.rewrite_checkpoint.valid) {
        if (!sequence.rewrite_state) {
            throw std::logic_error("rewrite checkpoint has no StateImage");
        }
        const runtime::CheckpointRef rewrite{
            .kind     = checkpoint_kind(sequence.rewrite_checkpoint.kind),
            .frontier = sequence.rewrite_checkpoint.frontier,
        };
        summary.rewrite = checkpoint_summary(sequence, rewrite, *sequence.rewrite_state,
                                             sequence.rewrite_checkpoint.rebuild_work);
    }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        summary.long_anchors.push_back(
            checkpoint_summary(sequence,
                               runtime::CheckpointRef{.kind = runtime::CheckpointKind::LongAnchor,
                                                      .frontier = anchor.frontier,
                                                      .ordinal  = anchor.ordinal},
                               anchor.state, anchor.rebuild_work));
    }
    if (!summary.endpoint && !summary.rewrite && summary.long_anchors.empty()) {
        throw std::logic_error("private continuation has no checkpoint");
    }
    const auto* begin = continuation_states.data();
    const auto* end   = begin + continuation_capacity;
    if (&sequence >= begin && &sequence < end) {
        const std::size_t index = static_cast<std::size_t>(&sequence - begin);
        summary.active_references =
            continuation_slots[index].role == ContinuationSlotRole::Active ? 1U : 0U;
    }
}

qwen3_6::SharedPrefixSummary
ProgramImplCore::shared_prefix_summary(const SharedPrefixState& shared) const {
    if (!shared.kv || !shared.identity || shared.frontier == 0 ||
        !state_store->valid(shared.state)) {
        throw std::logic_error("shared-prefix summary source is incomplete");
    }
    const StateReplicaResidency state_location = state_store->residency(shared.state);
    runtime::ReplicaResidency residency        = runtime::ReplicaResidency::DeviceOnly;
    if (state_location == StateReplicaResidency::HostOnly) {
        residency = runtime::ReplicaResidency::HostOnly;
    } else if (state_location == StateReplicaResidency::Both) {
        residency = runtime::ReplicaResidency::Both;
    } else if (state_location != StateReplicaResidency::DeviceOnly) {
        throw std::logic_error("shared-prefix StateImage has no published replica");
    }
    return qwen3_6::SharedPrefixSummary{
        .checkpoint =
            {
                .ref =
                    {
                        .kind     = runtime::CheckpointKind::SharedStablePrefix,
                        .frontier = shared.frontier,
                    },
                .scope           = runtime::CheckpointScope::Shared,
                .shortlist_key   = shared.identity->shortlist_key,
                .state_residency = residency,
                .required_kv =
                    {
                        .main_frontier    = shared.frontier,
                        .backend_frontier = shared.backend_frontier,
                        .main_pages       = kv_pages_for_frontier(shared.frontier),
                        .backend_pages    = kv_pages_for_frontier(shared.backend_frontier),
                    },
                .rebuild_work = validated_rebuild_work(shared.rebuild_work, shared.frontier),
            },
        .active_references = shared.active_references,
    };
}

PrefillProgress ProgramImplCore::advance_prefill(SequenceHandle sequence,
                                                 runtime::ExecutionTiming* failed_timing) {
    if (pending_transaction_ || !valid_sequence(sequence)) {
        throw std::logic_error("prefill sequence capability is invalid");
    }
    const std::uint32_t lane = ContractAccess::lane(sequence).value;
    if (requests[lane].lifecycle != Lifecycle::Prefilling) {
        throw std::logic_error("prefill advance requires a prefilling sequence");
    }
    try {
        runtime::PrefillStepResult step = advance_prefill_raw(lane, failed_timing);
        if (failed_timing != nullptr) { *failed_timing += step.timing; }
        return wrap_prefill(lane, std::move(step));
    } catch (...) {
        const Clock::time_point cleanup_started = Clock::now();
        clear_execution_failure_lanes(std::span<const std::uint32_t>(&lane, 1));
        if (failed_timing != nullptr) {
            failed_timing->post_host_ns += elapsed_ns(cleanup_started);
        }
        throw;
    }
}

bool ProgramImplCore::shared_capture_matches(const CaptureOffer& offer,
                                             const SharedPrefixHandle& shared) const {
    if (!valid_capture_offer(offer) || !valid_shared_prefix(shared)) { return false; }
    const std::uint32_t lane               = ContractAccess::lane(offer).value;
    const RequestControl::Prefill& prefill = *requests[lane].prefill;
    const CaptureGroup& group              = prefill.capture_groups[prefill.next_capture];
    const SharedPrefixState& candidate     = shared_prefix_states[ContractAccess::index(shared)];
    return group.shared && group.identity && candidate.identity &&
           group.frontier == candidate.frontier &&
           group.identity->shortlist_key == candidate.identity->shortlist_key &&
           group.identity->prefix_equals(*candidate.identity);
}

CaptureAssessment
ProgramImplCore::inspect_capture(const CaptureOffer& offer, const SharedPrefixHandle* exact_shared,
                                 const SharedPrefixHandle* replacement,
                                 std::optional<runtime::CheckpointRef> private_replacement,
                                 bool permit_shared_publication) const {
    if (!valid_capture_offer(offer)) { throw std::logic_error("capture offer is stale"); }
    if (exact_shared != nullptr && replacement != nullptr) {
        throw std::invalid_argument("capture cannot deduplicate and replace simultaneously");
    }
    if (exact_shared != nullptr && !shared_capture_matches(offer, *exact_shared)) {
        throw std::logic_error("capture dedup source is not exact");
    }
    if (replacement != nullptr) {
        if (!valid_shared_prefix(*replacement)) {
            throw std::logic_error("capture replacement capability is stale");
        }
        const SharedPrefixState& victim = shared_prefix_states[ContractAccess::index(*replacement)];
        if (victim.active_references != 0) {
            throw std::logic_error("active-referenced shared prefix is not replaceable");
        }
    }
    const std::uint32_t lane               = ContractAccess::lane(offer).value;
    const RequestControl::Prefill& prefill = *requests[lane].prefill;
    const CaptureGroup& group              = prefill.capture_groups[prefill.next_capture];
    if (!group.identity) { throw std::logic_error("capture identity backing is missing"); }
    const bool publish_private =
        group.rewrite.has_value() ||
        (group.long_anchor && context_cache.max_long_anchors_per_continuation.value_or(0) != 0);
    const bool publish_shared = group.shared && permit_shared_publication &&
                                exact_shared == nullptr && shared_prefix_capacity != 0;

    CaptureAssessment assessment;
    assessment.shortlist_key   = group.identity->shortlist_key;
    assessment.shared_evidence = group.shared_evidence;
    assessment.protected_rebuild_work =
        validated_rebuild_work(group.identity->rebuild_work, group.frontier);
    assessment.frontier          = group.frontier;
    assessment.publishes_private = publish_private;
    assessment.publishes_shared  = publish_shared;
    if (!publish_private && !publish_shared) {
        if (private_replacement) {
            throw std::invalid_argument("empty capture has a private replacement");
        }
        assessment.physically_feasible = true;
        return assessment;
    }

    const SequenceState& sequence = active_sequence(lane);
    if (!sequence.kv) { throw std::logic_error("capture source has no KV bundle"); }
    const std::size_t anchor_limit = context_cache.max_long_anchors_per_continuation.value_or(0);
    const bool anchor_replacement_required =
        group.long_anchor && anchor_limit != 0 && sequence.long_anchors.size() == anchor_limit;
    const LongAnchorCheckpoint* selected_anchor_replacement = nullptr;
    if (anchor_replacement_required) {
        assessment.private_replacement_candidates.reserve(sequence.long_anchors.size());
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            assessment.private_replacement_candidates.push_back(runtime::CheckpointRef{
                .kind     = runtime::CheckpointKind::LongAnchor,
                .frontier = anchor.frontier,
                .ordinal  = anchor.ordinal,
            });
            if (private_replacement &&
                *private_replacement == assessment.private_replacement_candidates.back()) {
                selected_anchor_replacement = &anchor;
            }
        }
        if (private_replacement && selected_anchor_replacement == nullptr) {
            throw std::logic_error("private capture replacement is stale");
        }
        if (!private_replacement) { return assessment; }
    } else if (private_replacement) {
        throw std::invalid_argument("capture has no replaceable private anchor");
    }

    const bool replaces_rewrite =
        group.rewrite && sequence.rewrite_state && sequence.rewrite_checkpoint.valid;
    assessment.recycles_private_state =
        replaces_rewrite && *sequence.rewrite_state != sequence.state.write &&
        state_store->can_recycle_checkpoint_destination(*sequence.rewrite_state);
    detail::PhysicalResources added;
    detail::PhysicalResources active_removed;
    std::optional<KVActiveSnapshotShape> text_snapshot_shape;
    std::optional<KVActiveSnapshotShape> backend_snapshot_shape;
    if (publish_shared) {
        if (!sequence.kv) { throw std::logic_error("capture source has no KV bundle"); }
        text_snapshot_shape =
            text_kv_addresses->active_snapshot_shape(sequence.kv->text, sequence.text_kv_valid);
        active_removed.device.main_kv_pages = text_snapshot_shape->unique_full_pages;
        added.device.main_kv_pages          = text_snapshot_shape->copied_pages();

        if (sequence.kv->backend) {
            const std::uint32_t backend_frontier = backend_kv_valid(sequence);
            backend_snapshot_shape               = backend_kv_addresses->active_snapshot_shape(
                *sequence.kv->backend, backend_frontier);
            active_removed.device.backend_kv_pages = backend_snapshot_shape->unique_full_pages;
            added.device.backend_kv_pages          = backend_snapshot_shape->copied_pages();
        }
    }

    detail::PhysicalResources replaced_private;
    if (publish_private) {
        struct DroppedReference {
            StateImageHandle state;
            std::uint32_t count = 0;
        };

        std::array<DroppedReference, 2> drops{};
        std::size_t drop_count = 0;
        const auto add_drop    = [&](StateImageHandle state) {
            for (std::size_t index = 0; index < drop_count; ++index) {
                if (drops[index].state == state) {
                    ++drops[index].count;
                    return;
                }
            }
            drops[drop_count++] = DroppedReference{.state = state, .count = 1};
        };
        if (group.rewrite && sequence.rewrite_state) { add_drop(*sequence.rewrite_state); }
        if (selected_anchor_replacement != nullptr) {
            add_drop(selected_anchor_replacement->state);
        }
        for (std::size_t index = 0; index < drop_count; ++index) {
            const DroppedReference& drop = drops[index];
            if (!state_store->valid(drop.state) ||
                state_store->checkpoint_references(drop.state) != drop.count) {
                continue;
            }
            const StateReplicaResidency residency = state_store->residency(drop.state);
            if (residency == StateReplicaResidency::DeviceOnly ||
                residency == StateReplicaResidency::Both) {
                ++replaced_private.device.state_slots;
            }
            if (residency == StateReplicaResidency::HostOnly ||
                residency == StateReplicaResidency::Both) {
                ++replaced_private.host.state_slots;
            }
        }
    }
    detail::PhysicalResources replaced_shared;
    if (publish_shared && replacement != nullptr) {
        replaced_shared =
            resident_resources(shared_prefix_states[ContractAccess::index(*replacement)]);
    }
    if (replaced_shared.device.state_slots > state_store->device_occupied()) {
        throw std::logic_error("shared capture replacement exceeds Device State occupancy");
    }
    const std::uint32_t device_state_after_preparation =
        state_store->device_occupied() - replaced_shared.device.state_slots;
    const bool device_destination_available =
        assessment.recycles_private_state ||
        device_state_after_preparation < state_store->device_capacity();
    if (device_destination_available || host_state_images == nullptr) {
        assessment.state_placement = qwen3_6::CaptureStatePlacement::DeviceFork;
        added.device.state_slots   = 1;
    } else {
        // A capture must not require a third Device image when the active image and a retained
        // checkpoint already occupy the C+H pool.  Snapshot the frozen logical checkpoint to
        // Host, then transfer ownership of its unchanged Device replica to the continuing active
        // identity.  This preserves both logical checkpoints without assigning fixed slot roles.
        assessment.state_placement = qwen3_6::CaptureStatePlacement::HostSnapshot;
        added.host.state_slots     = 1;
    }
    const detail::PhysicalResources replaced =
        checked_resource_sum(replaced_private, replaced_shared);
    assessment.implementation->capacity_preparation_removed = replaced_shared;
    assessment.implementation->demand                       = detail::PhysicalDemand{
                              .reservation_added  = added,
                              .reservation_credit = replaced_shared,
                              .final_removed      = replaced,
                              .final_added        = added,
    };
    if (assessment.recycles_private_state) {
        if (assessment.state_placement != qwen3_6::CaptureStatePlacement::DeviceFork) {
            throw std::logic_error("recycled rewrite capture selected Host placement");
        }
        if (replaced_private.device.state_slots == 0) {
            throw std::logic_error("recycled rewrite capture has no Device state replacement");
        }
        if (assessment.implementation->demand.reservation_credit.device.state_slots ==
            std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("capture StateImage reservation credit overflow");
        }
        ++assessment.implementation->demand.reservation_credit.device.state_slots;
    }
    assessment.implementation->demand.physical_peak_additional =
        positive_resource_difference(assessment.implementation->demand.reservation_added,
                                     assessment.implementation->demand.reservation_credit);
    assessment.implementation->active_entitlement_delta.removed =
        checked_resource_sum(active_removed, replaced_private);
    if (publish_private && !publish_shared) {
        assessment.implementation->active_entitlement_delta.added = added;
    }
    assessment.transfer_requirements.reserve(3);
    if (assessment.state_placement == qwen3_6::CaptureStatePlacement::HostSnapshot) {
        assessment.transfer_requirements.push_back(state_transfer_requirement(
            state_images->host_layout(), runtime::ContextTransferDirection::DeviceToHost));
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        assessment.transfer_requirements.push_back(state_transfer_requirement(
            state_images->host_layout(), runtime::ContextTransferDirection::DeviceToDevice, true));
    }
    if (added.device.main_kv_pages != 0) {
        assessment.transfer_requirements.push_back(kv_transfer_requirement(
            runtime::ContextResourceClass::MainKV,
            runtime::ContextTransferDirection::DeviceToDevice,
            plan_host_kv_page_layout(text_kv_pages->physical_pool().geometry()),
            added.device.main_kv_pages));
    }
    if (added.device.backend_kv_pages != 0) {
        assessment.transfer_requirements.push_back(kv_transfer_requirement(
            runtime::ContextResourceClass::BackendKV,
            runtime::ContextTransferDirection::DeviceToDevice,
            plan_host_kv_page_layout(backend_kv_pages->physical_pool().geometry()),
            added.device.backend_kv_pages));
    }
    assessment.needs_transfer = !assessment.transfer_requirements.empty();
    assessment.physically_feasible =
        physical_peak_fits(assessment.implementation->demand.physical_peak_additional);
    if (publish_shared) {
        std::vector<runtime::ContextTransferRequirement> recovery;
        recovery.reserve(3);
        if (assessment.state_placement == qwen3_6::CaptureStatePlacement::HostSnapshot) {
            recovery.push_back(state_transfer_requirement(
                state_images->host_layout(), runtime::ContextTransferDirection::HostToDevice));
        } else if (speculative_backend == SpeculativeBackend::DFlash) {
            recovery.push_back(state_transfer_requirement(
                state_images->host_layout(), runtime::ContextTransferDirection::DeviceToDevice,
                true));
        }
        if (text_snapshot_shape->copied_pages() != 0) {
            recovery.push_back(kv_transfer_requirement(
                runtime::ContextResourceClass::MainKV,
                runtime::ContextTransferDirection::DeviceToDevice,
                plan_host_kv_page_layout(text_kv_pages->physical_pool().geometry()),
                text_snapshot_shape->copied_pages()));
        }
        if (backend_snapshot_shape && backend_snapshot_shape->copied_pages() != 0) {
            recovery.push_back(kv_transfer_requirement(
                runtime::ContextResourceClass::BackendKV,
                runtime::ContextTransferDirection::DeviceToDevice,
                plan_host_kv_page_layout(backend_kv_pages->physical_pool().geometry()),
                backend_snapshot_shape->copied_pages()));
        }
        assessment.projected_recovery_work.reserve(2);
        assessment.projected_recovery_work.push_back(
            NINFER_QWEN36_RUNTIME_NS::recovery_alternative_work({},
                                                                assessment.protected_rebuild_work));
        assessment.projected_recovery_work.push_back(
            NINFER_QWEN36_RUNTIME_NS::recovery_alternative_work(recovery));
    }
    return assessment;
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
ProgramImplCore::checkpoint_recovery_work(const ContinuationHandle& owner,
                                          runtime::CheckpointRef checkpoint) const {
    if (!valid_continuation(owner)) {
        throw std::logic_error("checkpoint recovery owner is stale");
    }
    const SequenceState& sequence = continuation_states[ContractAccess::index(owner)];
    if (!sequence.kv) { throw std::logic_error("checkpoint recovery owner has no KV bundle"); }
    const qwen3_6::ContinuationSummary summary = continuation_summary(sequence);

    struct RecoverySource {
        const qwen3_6::CheckpointSummary* checkpoint = nullptr;
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
    alternatives.push_back(NINFER_QWEN36_RUNTIME_NS::recovery_alternative_work(
        {}, selected->checkpoint->rebuild_work));
    for (const RecoverySource& source : sources) {
        if (source.checkpoint->ref.frontier > selected->checkpoint->ref.frontier ||
            !state_store->valid(source.state)) {
            continue;
        }
        const runtime::PrefillWork interval = interval_rebuild_work(
            source.checkpoint->ref.frontier, source.checkpoint->rebuild_work,
            selected->checkpoint->ref.frontier, selected->checkpoint->rebuild_work, prefill_chunk);
        alternatives.push_back(NINFER_QWEN36_RUNTIME_NS::recovery_alternative_work(
            checkpoint_restore_requirements(*sequence.kv, source.checkpoint->required_kv,
                                            source.state),
            interval));
    }
    return alternatives;
}

std::vector<runtime::CheckpointRecoveryAlternativeWork>
ProgramImplCore::checkpoint_recovery_work(const SharedPrefixHandle& owner,
                                          runtime::CheckpointRef checkpoint) const {
    if (!valid_shared_prefix(owner)) {
        throw std::logic_error("shared checkpoint recovery owner is stale");
    }
    const SharedPrefixState& shared = shared_prefix_states[ContractAccess::index(owner)];
    if (!shared.kv) { throw std::logic_error("shared checkpoint recovery owner has no KV bundle"); }
    const qwen3_6::CheckpointSummary summary = shared_prefix_summary(shared).checkpoint;
    if (summary.ref != checkpoint || !state_store->valid(shared.state)) {
        throw std::logic_error("shared checkpoint recovery target is unavailable");
    }
    std::vector<runtime::CheckpointRecoveryAlternativeWork> alternatives;
    alternatives.reserve(2);
    alternatives.push_back(
        NINFER_QWEN36_RUNTIME_NS::recovery_alternative_work({}, summary.rebuild_work));
    alternatives.push_back(NINFER_QWEN36_RUNTIME_NS::recovery_alternative_work(
        checkpoint_restore_requirements(*shared.kv, summary.required_kv, shared.state)));
    return alternatives;
}

std::unique_ptr<CapturePressureCandidateImpl>
ProgramImplCore::make_capture_physical_candidate(const CaptureAssessment& assessment) const {
    if (assessment.implementation == nullptr || !assessment.publishes_shared ||
        assessment.frontier == 0) {
        throw std::invalid_argument("capture pressure candidate is incomplete");
    }
    auto details                       = std::make_unique<CapturePressureCandidateImpl>();
    details->planning_revision         = resource_revision_;
    details->summary.prompt_tokens     = assessment.frontier;
    details->demand                    = assessment.implementation->demand;
    details->transfer_requirements     = assessment.transfer_requirements;
    details->needs_transfer            = assessment.needs_transfer;
    details->identity_pressure_deficit = materialization_deficit(*details);
    details->identity_assessment.machine_work =
        NINFER_QWEN36_RUNTIME_NS::materialization_machine_work(*details, {}, {});
    details->identity_assessment.physical_status =
        physical_peak_fits(details->demand.physical_peak_additional)
            ? runtime::MaterializationPhysicalStatus::Feasible
            : runtime::MaterializationPhysicalStatus::Infeasible;
    details->identity_assessment.source_mode = runtime::PrivateSourceMode::ConsumeToActive;
    details->identity_assessment.expandable  = details->identity_assessment.physical_status !=
                                              runtime::MaterializationPhysicalStatus::Feasible;
    details->identity_assessment.projection_work = 1U + details->transfer_requirements.size();
    std::uint64_t digest                         = 1469598103934665603ULL;
    const auto mix                               = [&](std::uint64_t value) {
        digest ^= value;
        digest *= 1099511628211ULL;
    };
    mix(resource_revision_.value);
    mix(assessment.frontier);
    mix(static_cast<std::uint8_t>(details->identity_assessment.physical_status));
    details->identity_assessment.assessment_digest = digest;
    return details;
}

void ProgramImplCore::skip_capture(CaptureOffer&& offer) {
    if (!valid_capture_offer(offer)) { throw std::logic_error("capture offer is not skippable"); }
    const std::uint32_t lane = ContractAccess::lane(offer).value;
    ContractAccess::consume(offer);
    RequestControl::Prefill& prefill = *requests[lane].prefill;
    prefill.pending_capture_offer    = 0;
    ++prefill.next_capture;
    if (prefill.cursor == prefill.prompt_tokens &&
        requests[lane].lifecycle != Lifecycle::Prefilling) {
        requests[lane].prefill.reset();
    }
}

runtime::ContextTransactionReserveStatus ProgramImplCore::reserve_active_capture(
    CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
    const SharedPrefixHandle* replacement,
    std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
    runtime::CancellationFlagView cancellation) {
    return reserve_active_capture_impl(std::move(offer), exact_shared, replacement,
                                       private_replacement, permit_shared_publication, std::nullopt,
                                       cancellation);
}

runtime::ContextTransactionReserveStatus ProgramImplCore::reserve_active_capture_with_pressure(
    CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
    const SharedPrefixHandle* replacement,
    std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
    CapturePressureCandidate&& pressure, runtime::CancellationFlagView cancellation) {
    std::optional<CapturePressureCandidate> owned;
    owned.emplace(std::move(pressure));
    return reserve_active_capture_impl(std::move(offer), exact_shared, replacement,
                                       private_replacement, permit_shared_publication,
                                       std::move(owned), cancellation);
}

runtime::ContextTransactionReserveStatus ProgramImplCore::reserve_active_capture_impl(
    CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
    const SharedPrefixHandle* replacement,
    std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
    std::optional<CapturePressureCandidate> pressure, runtime::CancellationFlagView cancellation) {
    if (has_context_transaction() || has_unsettled_state_fork() || !valid_capture_offer(offer)) {
        throw std::logic_error("capture transaction is not reservable");
    }
    if (cancellation.requested()) {
        skip_capture(std::move(offer));
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    const CaptureAssessment assessment = inspect_capture(
        offer, exact_shared, replacement, private_replacement, permit_shared_publication);
    if (!assessment.publishes_private && !assessment.publishes_shared) {
        skip_capture(std::move(offer));
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    const CapturePressureCandidateImpl* pressure_details =
        pressure && pressure->impl_ ? pressure->impl_.get() : nullptr;
    if (pressure &&
        (pressure_details == nullptr || pressure_details->planning_revision != resource_revision_ ||
         pressure_details->summary.prompt_tokens != assessment.frontier ||
         pressure_details->blocked_host_allocation_bytes != 0 ||
         !physical_peak_fits(pressure_details->demand.physical_peak_additional))) {
        skip_capture(std::move(offer));
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    if (!pressure && !assessment.physically_feasible) {
        skip_capture(std::move(offer));
        return runtime::ContextTransactionReserveStatus::Aborted;
    }

    const std::uint32_t lane         = ContractAccess::lane(offer).value;
    RequestControl::Prefill& prefill = *requests[lane].prefill;
    SequenceState& sequence          = active_sequence(lane);
    ActiveCaptureTransaction transaction;
    transaction.id                  = ContractAccess::id(offer);
    transaction.lane                = lane;
    transaction.lane_epoch          = lane_epochs[lane];
    transaction.group               = prefill.capture_groups[prefill.next_capture];
    transaction.publish_private     = assessment.publishes_private;
    transaction.publish_shared      = assessment.publishes_shared;
    transaction.private_replacement = private_replacement;
    transaction.resource_delta      = detail::PhysicalDelta{
             .removed = assessment.implementation->demand.final_removed,
             .added   = assessment.implementation->demand.final_added,
    };
    transaction.active_entitlement_delta = assessment.implementation->active_entitlement_delta;
    transaction.capacity_preparation_removed =
        assessment.implementation->capacity_preparation_removed;
    transaction.recycles_private_state = assessment.recycles_private_state;
    transaction.state_placement        = assessment.state_placement;
    transaction.transfer_requirements  = assessment.transfer_requirements;
    if (pressure_details != nullptr) {
        if (pressure_details->pressure_options.size() !=
                pressure_details->pressure_owner_ids.size() ||
            pressure_details->pressure_options.size() !=
                pressure_details->pressure_indices.size() ||
            pressure_details->pressure_options.size() !=
                pressure_details->pressure_generations.size() ||
            pressure_details->shared_pressure_options.size() !=
                pressure_details->shared_pressure_owner_ids.size() ||
            pressure_details->shared_pressure_options.size() !=
                pressure_details->shared_pressure_indices.size() ||
            pressure_details->shared_pressure_options.size() !=
                pressure_details->shared_pressure_generations.size()) {
            throw std::logic_error("capture pressure plan is not row aligned");
        }
        transaction.victim_indices     = pressure_details->pressure_indices;
        transaction.victim_generations = pressure_details->pressure_generations;
        transaction.pressure_results.resize(pressure_details->pressure_options.size());
        transaction.pressure.reserve(pressure_details->pressure_options.size());
        for (std::size_t index = 0; index < pressure_details->pressure_options.size(); ++index) {
            transaction.pressure_results[index].owner = pressure_details->pressure_owner_ids[index];
            transaction.pressure_results[index].final_summary.emplace();
            transaction.pressure_results[index].final_summary->long_anchors.reserve(
                continuation_states[transaction.victim_indices[index]].long_anchors.size());
            transaction.pressure.push_back(MaterializationTransaction::PressureWork{
                .option                  = pressure_details->pressure_options[index],
                .continuation_index      = transaction.victim_indices[index],
                .continuation_generation = transaction.victim_generations[index],
            });
            prepare_pressure_bookkeeping(transaction.pressure.back());
        }
        transaction.shared_victim_indices     = pressure_details->shared_pressure_indices;
        transaction.shared_victim_generations = pressure_details->shared_pressure_generations;
        transaction.shared_pressure_results.resize(
            pressure_details->shared_pressure_options.size());
        transaction.shared_pressure.reserve(pressure_details->shared_pressure_options.size());
        for (std::size_t index = 0; index < pressure_details->shared_pressure_options.size();
             ++index) {
            transaction.shared_pressure_results[index].owner =
                pressure_details->shared_pressure_owner_ids[index];
            const std::uint32_t victim = transaction.shared_victim_indices[index];
            if (replacement != nullptr && ContractAccess::index(*replacement) == victim) {
                throw std::logic_error("capture logical replacement is duplicated by pressure");
            }
            transaction.shared_pressure.push_back(MaterializationTransaction::PressureWork{
                .option                  = pressure_details->shared_pressure_options[index],
                .continuation_index      = victim,
                .continuation_generation = transaction.shared_victim_generations[index],
                .shared_owner            = true,
            });
            prepare_pressure_bookkeeping(transaction.shared_pressure.back());
        }
    }
    if (transaction.publish_private) {
        transaction.active_summary.long_anchors.reserve(sequence.long_anchors.capacity());
    }
    transaction.transfer_observations.reserve(
        3U * (transaction.pressure.size() + transaction.shared_pressure.size()) + 3U);
    ContractAccess::consume(offer);

    try {
        if (transaction.publish_shared) {
            if (replacement != nullptr) {
                const std::uint32_t index = ContractAccess::index(*replacement);
                if (index >= shared_prefix_capacity ||
                    shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
                    shared_prefix_states[index].active_references != 0) {
                    throw std::logic_error("shared capture replacement changed before reserve");
                }
                transaction.shared_index           = index;
                transaction.replaces_shared        = true;
                transaction.replacement_generation = shared_prefix_slots[index].generation;
                shared_prefix_slots[index].role    = SharedPrefixSlotRole::ReservedReplacement;
            } else {
                for (std::uint32_t index = 0; index < shared_prefix_capacity; ++index) {
                    if (shared_prefix_slots[index].role == SharedPrefixSlotRole::Free) {
                        shared_prefix_slots[index].role = SharedPrefixSlotRole::ReservedCapture;
                        transaction.shared_index        = index;
                        break;
                    }
                }
            }
            if (!transaction.shared_index) {
                throw std::logic_error("shared capture descriptor was not reserved by policy");
            }
        }

        transaction.transfer_enqueue_pending = assessment.needs_transfer;
        advance_resource_revision();
        context_transaction_.emplace<ActiveCaptureTransaction>(std::move(transaction));
        return runtime::ContextTransactionReserveStatus::Reserved;
    } catch (...) {
        abort_active_capture(transaction);
        prefill.pending_capture_offer = 0;
        ++prefill.next_capture;
        throw;
    }
}

detail::PhysicalResources
ProgramImplCore::release_checkpoint_reference(StateImageHandle checkpoint) noexcept {
    detail::PhysicalResources removed;
    if (!state_store->valid(checkpoint)) { return removed; }
    try {
        const StateReplicaResidency residency = state_store->residency(checkpoint);
        const std::uint32_t references        = state_store->checkpoint_references(checkpoint);
        if (references != 0) { state_store->release_checkpoint_reference(checkpoint); }
        if (state_store->checkpoint_references(checkpoint) != 0 ||
            state_store->source_pins(checkpoint) != 0) {
            return removed;
        }
        if (!state_store->release(checkpoint)) { return removed; }
        if (residency == StateReplicaResidency::DeviceOnly ||
            residency == StateReplicaResidency::Both) {
            removed.device.state_slots = 1;
        }
        if (residency == StateReplicaResidency::HostOnly ||
            residency == StateReplicaResidency::Both) {
            removed.host.state_slots = 1;
        }
    } catch (...) {}
    return removed;
}

detail::PhysicalResources
ProgramImplCore::install_private_capture(SequenceState& sequence, const CaptureGroup& group,
                                         StateImageHandle checkpoint,
                                         std::optional<runtime::CheckpointRef> replacement) {
    detail::PhysicalResources removed;
    if (group.rewrite) {
        if (sequence.rewrite_state && *sequence.rewrite_state != checkpoint) {
            removed = checked_resource_sum(removed,
                                           release_checkpoint_reference(*sequence.rewrite_state));
        }
        state_store->retain_checkpoint_reference(checkpoint);
        sequence.rewrite_state      = checkpoint;
        sequence.rewrite_checkpoint = RewriteCheckpoint{
            .valid        = true,
            .kind         = *group.rewrite,
            .frontier     = group.frontier,
            .rebuild_work = validated_rebuild_work(group.identity->rebuild_work, group.frontier),
        };
    }
    if (group.long_anchor && context_cache.max_long_anchors_per_continuation.value_or(0) != 0) {
        const std::size_t capacity_limit = context_cache.max_long_anchors_per_continuation.value();
        validate_long_anchor_ordinals(sequence.long_anchors, capacity_limit);
        std::uint32_t ordinal = 0;
        if (sequence.long_anchors.size() == capacity_limit) {
            if (!replacement || replacement->kind != runtime::CheckpointKind::LongAnchor) {
                throw std::logic_error("full long-anchor set has no selected replacement");
            }
            const auto victim =
                std::find_if(sequence.long_anchors.begin(), sequence.long_anchors.end(),
                             [&](const LongAnchorCheckpoint& anchor) {
                                 return anchor.frontier == replacement->frontier &&
                                        anchor.ordinal == replacement->ordinal;
                             });
            if (victim == sequence.long_anchors.end()) {
                throw std::logic_error("selected long-anchor replacement changed");
            }
            ordinal = victim->ordinal;
            removed = checked_resource_sum(removed, release_checkpoint_reference(victim->state));
            sequence.long_anchors.erase(victim);
        } else {
            if (replacement) {
                throw std::logic_error("non-full long-anchor set has a replacement");
            }
            for (std::size_t candidate = 1; candidate <= capacity_limit; ++candidate) {
                if (std::none_of(sequence.long_anchors.begin(), sequence.long_anchors.end(),
                                 [candidate](const LongAnchorCheckpoint& anchor) {
                                     return anchor.ordinal == candidate;
                                 })) {
                    ordinal = static_cast<std::uint32_t>(candidate);
                    break;
                }
            }
        }
        if (ordinal == 0 || ordinal > capacity_limit) {
            throw std::logic_error("long-anchor capture has no valid ordinal");
        }
        state_store->retain_checkpoint_reference(checkpoint);
        sequence.long_anchors.push_back(LongAnchorCheckpoint{
            .state        = checkpoint,
            .frontier     = group.frontier,
            .ordinal      = ordinal,
            .rebuild_work = validated_rebuild_work(group.identity->rebuild_work, group.frontier),
        });
        validate_long_anchor_ordinals(sequence.long_anchors, capacity_limit);
    }
    return removed;
}

void ProgramImplCore::prepare_active_capture(ActiveCaptureTransaction& transaction) {
    if (transaction.prepared || transaction.lane >= max_concurrency ||
        transaction.lane_epoch != lane_epochs[transaction.lane]) {
        throw std::logic_error("active capture capacity preparation is stale");
    }
    SequenceState& sequence = active_sequence(transaction.lane);
    if (transaction.publish_shared) {
        if (!transaction.shared_index || *transaction.shared_index >= shared_prefix_capacity) {
            throw std::logic_error("shared capture has no reserved descriptor");
        }
        SharedPrefixSlot& slot = shared_prefix_slots[*transaction.shared_index];
        if (transaction.replaces_shared) {
            if (transaction.replacement_removed ||
                slot.role != SharedPrefixSlotRole::ReservedReplacement ||
                slot.generation != transaction.replacement_generation) {
                throw std::logic_error("shared capture replacement changed before preparation");
            }
            if (!can_release_shared_prefix_state(*transaction.shared_index,
                                                 SharedPrefixSlotRole::ReservedReplacement)) {
                throw std::logic_error("shared capture replacement is not strictly releasable");
            }
            const detail::PhysicalResources removed = release_shared_prefix_state_strict(
                *transaction.shared_index, SharedPrefixSlotRole::ReservedReplacement);
            if (removed != transaction.capacity_preparation_removed) {
                throw std::logic_error("shared capture preparation release changed");
            }
            transaction.replacement_removed    = true;
            transaction.replacement_generation = slot.generation;
            slot.role                          = SharedPrefixSlotRole::ReservedCapture;
        } else if (slot.role != SharedPrefixSlotRole::ReservedCapture ||
                   transaction.capacity_preparation_removed != detail::PhysicalResources{}) {
            throw std::logic_error("shared capture vacant descriptor changed before preparation");
        }
    } else if (transaction.capacity_preparation_removed != detail::PhysicalResources{}) {
        throw std::logic_error("private-only capture has shared preparation resources");
    }

    transaction.source_state = sequence.state.write;
    if (transaction.state_placement == qwen3_6::CaptureStatePlacement::HostSnapshot) {
        if (transaction.recycles_private_state || host_state_images == nullptr) {
            throw std::logic_error("Host capture placement has no valid backing");
        }
        std::optional<StateImageHandle> destination = state_store->reserve_logical_destination();
        if (!destination) {
            throw std::logic_error("selected capture has no prepared logical State capacity");
        }
        transaction.destination_state = *destination;
    } else if (transaction.recycles_private_state) {
        if (!sequence.rewrite_state) {
            throw std::logic_error("recycled rewrite destination is unavailable");
        }
        transaction.destination_state = *sequence.rewrite_state;
        transaction.recycled_state_epoch =
            state_store->recycle_checkpoint_destination(transaction.destination_state);
    } else {
        std::optional<StateImageHandle> destination = state_store->reserve_destination();
        if (!destination) {
            throw std::logic_error("selected capture has no prepared Device State capacity");
        }
        transaction.destination_state = *destination;
    }

    if (transaction.publish_shared) {
        if (!sequence.kv || sequence.state.fork_pending ||
            sequence.state.read != sequence.state.write ||
            state_store->role(sequence.state.write) != StateImageRole::ActiveMutable) {
            throw std::logic_error("active capture source is not an in-place writer");
        }
        trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
        transaction.active_text_destination = text_kv_addresses->create_inactive();
        if (!transaction.active_text_destination) {
            throw std::logic_error("selected capture has no Text KV address descriptor");
        }
        transaction.text_snapshot.emplace(text_kv_addresses->prepare_active_snapshot(
            sequence.kv->text, *transaction.active_text_destination, sequence.text_kv_valid));
        if (sequence.kv->backend) {
            transaction.active_backend_destination = backend_kv_addresses->create_inactive();
            if (!transaction.active_backend_destination) {
                throw std::logic_error("selected capture has no Backend KV address descriptor");
            }
            transaction.backend_snapshot.emplace(backend_kv_addresses->prepare_active_snapshot(
                *sequence.kv->backend, *transaction.active_backend_destination,
                backend_kv_valid(sequence)));
        }
    }

    state_store->freeze(transaction.source_state);
    if (transaction.state_placement == qwen3_6::CaptureStatePlacement::DeviceFork) {
        (void)state_store->begin_fork(transaction.source_state, transaction.destination_state);
        sequence.state = ActiveStateBinding{.read         = transaction.source_state,
                                            .write        = transaction.destination_state,
                                            .fork_pending = true};
        refresh_state_views(sequence);
    }
    transaction.prepared = true;
}

void ProgramImplCore::enqueue_active_capture_transfers(ActiveCaptureTransaction& transaction) {
    if (!transaction.prepared || !transaction.transfer_enqueue_pending ||
        transaction.transfer_submitted) {
        throw std::logic_error("active capture transfer batch is not enqueueable");
    }
    context_source_ready_.record(device.stream);
    context_source_ready_.wait(device.transfer_stream);
    if (transaction.state_placement == qwen3_6::CaptureStatePlacement::HostSnapshot) {
        start_context_transfer_timer(runtime::ContextResourceClass::State);
        std::optional<StateImageTransfer> snapshot =
            state_store->begin_device_to_host(transaction.source_state, device.transfer_stream);
        if (!snapshot) {
            throw std::logic_error("selected Host capture has no prepared State target");
        }
        transaction.state_snapshot.emplace(std::move(*snapshot));
        stop_context_transfer_timer(runtime::ContextResourceClass::State);
        transaction.transfer_timer_mask |=
            1U << context_resource_index(runtime::ContextResourceClass::State);
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        const StateImageSelectors state_fork =
            state_store->selectors(transaction.source_state, transaction.destination_state);
        start_context_transfer_timer(runtime::ContextResourceClass::State);
        state_images->copy_dflash_local(state_fork.source, state_fork.destination,
                                        device.transfer_stream);
        stop_context_transfer_timer(runtime::ContextResourceClass::State);
        transaction.transfer_timer_mask |=
            1U << context_resource_index(runtime::ContextResourceClass::State);
    }
    if (transaction.text_snapshot && transaction.text_snapshot->needs_tail_copy()) {
        start_context_transfer_timer(runtime::ContextResourceClass::MainKV);
        decoder->text_kv.page_pool().copy_page(
            text_kv_addresses->active_snapshot_tail_source(*transaction.text_snapshot),
            text_kv_addresses->active_snapshot_tail_destination(*transaction.text_snapshot),
            device.transfer_stream);
        stop_context_transfer_timer(runtime::ContextResourceClass::MainKV);
        transaction.transfer_timer_mask |=
            1U << context_resource_index(runtime::ContextResourceClass::MainKV);
        ++transaction.operations.partial_tail_cow_pages;
    }
    if (transaction.backend_snapshot && transaction.backend_snapshot->needs_tail_copy()) {
        start_context_transfer_timer(runtime::ContextResourceClass::BackendKV);
        backend_kv_cache()->page_pool().copy_page(
            backend_kv_addresses->active_snapshot_tail_source(*transaction.backend_snapshot),
            backend_kv_addresses->active_snapshot_tail_destination(*transaction.backend_snapshot),
            device.transfer_stream);
        stop_context_transfer_timer(runtime::ContextResourceClass::BackendKV);
        transaction.transfer_timer_mask |=
            1U << context_resource_index(runtime::ContextResourceClass::BackendKV);
        ++transaction.operations.partial_tail_cow_pages;
    }
    context_completion_.record(device.transfer_stream);
    transaction.transfer_enqueue_pending = false;
    transaction.transfer_submitted       = true;
}

void ProgramImplCore::abort_active_capture(ActiveCaptureTransaction& transaction) noexcept {
    if (transaction.lane < max_concurrency &&
        active_continuations[transaction.lane] < continuation_capacity) {
        SequenceState& sequence = active_sequence(transaction.lane);
        if (transaction.backend_snapshot) {
            backend_kv_addresses->abort_active_snapshot(*transaction.backend_snapshot);
            transaction.backend_snapshot.reset();
        }
        if (transaction.text_snapshot) {
            text_kv_addresses->abort_active_snapshot(*transaction.text_snapshot);
            transaction.text_snapshot.reset();
        }
        if (transaction.active_backend_destination &&
            backend_kv_addresses->valid(*transaction.active_backend_destination)) {
            (void)backend_kv_addresses->release(*transaction.active_backend_destination);
        }
        if (transaction.active_text_destination &&
            text_kv_addresses->valid(*transaction.active_text_destination)) {
            (void)text_kv_addresses->release(*transaction.active_text_destination);
        }
        if (transaction.state_snapshot) {
            state_store->abort_transfer(std::move(*transaction.state_snapshot));
            transaction.state_snapshot.reset();
        }
        if (state_store->valid(transaction.source_state) &&
            state_store->valid(transaction.destination_state)) {
            try {
                if (transaction.state_placement == qwen3_6::CaptureStatePlacement::HostSnapshot) {
                    (void)state_store->release(transaction.destination_state);
                } else {
                    if (sequence.state.fork_pending &&
                        sequence.state.read == transaction.source_state &&
                        sequence.state.write == transaction.destination_state) {
                        state_store->abort_fork(transaction.source_state,
                                                transaction.destination_state);
                        sequence.state = ActiveStateBinding{.read  = transaction.source_state,
                                                            .write = transaction.source_state};
                    }
                    if (transaction.recycles_private_state) {
                        state_store->restore_recycled_checkpoint(transaction.destination_state,
                                                                 transaction.recycled_state_epoch);
                    } else {
                        (void)state_store->release(transaction.destination_state);
                    }
                }
                state_store->thaw(transaction.source_state);
                refresh_state_views(sequence);
            } catch (...) {}
        }
    }
    if (transaction.shared_index && *transaction.shared_index < shared_prefix_capacity) {
        SharedPrefixSlot& slot = shared_prefix_slots[*transaction.shared_index];
        if (transaction.replaces_shared && transaction.replacement_removed &&
            slot.role == SharedPrefixSlotRole::ReservedCapture &&
            slot.generation == transaction.replacement_generation) {
            slot.role = SharedPrefixSlotRole::Free;
        } else if (transaction.replaces_shared && !transaction.replacement_removed &&
                   slot.role == SharedPrefixSlotRole::ReservedReplacement &&
                   slot.generation == transaction.replacement_generation) {
            slot.role = SharedPrefixSlotRole::Catalogued;
        } else if (!transaction.replaces_shared &&
                   slot.role == SharedPrefixSlotRole::ReservedCapture) {
            slot.role = SharedPrefixSlotRole::Free;
        }
    }
    transaction.prepared = false;
}

ActiveCaptureResult ProgramImplCore::publish_active_capture(ActiveCaptureTransaction& transaction) {
    if (!transaction.prepared || transaction.lane >= max_concurrency ||
        transaction.lane_epoch != lane_epochs[transaction.lane] || transaction.published) {
        throw std::logic_error("active capture transaction is stale");
    }
    SequenceState& sequence          = active_sequence(transaction.lane);
    RequestControl& request          = requests[transaction.lane];
    RequestControl::Prefill& prefill = *request.prefill;
    if (prefill.pending_capture_offer != transaction.id ||
        prefill.next_capture >= prefill.capture_groups.size()) {
        throw std::logic_error("active capture offer ownership changed");
    }

    if (transaction.state_placement == qwen3_6::CaptureStatePlacement::HostSnapshot) {
        if (!transaction.state_snapshot || sequence.state.fork_pending ||
            sequence.state.read != transaction.source_state ||
            sequence.state.write != transaction.source_state) {
            throw std::logic_error("Host capture snapshot is not publishable");
        }
        state_store->publish_transfer(std::move(*transaction.state_snapshot), true);
        transaction.state_snapshot.reset();
        state_store->split_device_replica_identity(transaction.source_state,
                                                   transaction.destination_state);
        sequence.state = ActiveStateBinding{.read  = transaction.destination_state,
                                            .write = transaction.destination_state};
        refresh_state_views(sequence);
    }

    std::optional<SequenceKVBundle> shared_bundle;
    if (transaction.publish_shared) {
        shared_bundle = *sequence.kv;
        text_kv_addresses->commit_active_snapshot(std::move(*transaction.text_snapshot),
                                                  device.stream);
        transaction.text_snapshot.reset();
        SequenceKVBundle active_bundle{.text = *transaction.active_text_destination};
        transaction.active_text_destination.reset();
        if (transaction.backend_snapshot) {
            backend_kv_addresses->commit_active_snapshot(std::move(*transaction.backend_snapshot),
                                                         device.stream);
            transaction.backend_snapshot.reset();
            active_bundle.backend = *transaction.active_backend_destination;
            transaction.active_backend_destination.reset();
        }
        sequence.kv = active_bundle;
        bind_sequence_kv(sequence);
        const std::uint32_t backend_materialized =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min(capacity,
                           prefill.prompt_tokens + (prefill.initial_mtp_extent == 0
                                                        ? 0U
                                                        : prefill.initial_mtp_extent - 1U))
            : speculative_backend == SpeculativeBackend::DFlash ? prefill.prompt_tokens
                                                                : 0U;
        materialize_sequence_kv(sequence, prefill.prompt_tokens, backend_materialized);
    }

    detail::PhysicalResources removed = transaction.capacity_preparation_removed;
    if (transaction.publish_shared) {
        state_store->retain_checkpoint_reference(transaction.source_state);
    }
    if (transaction.publish_private) {
        if (transaction.recycles_private_state) {
            if (!sequence.rewrite_state ||
                *sequence.rewrite_state != transaction.destination_state ||
                !sequence.rewrite_checkpoint.valid) {
                throw std::logic_error("recycled rewrite metadata changed before publication");
            }
            sequence.rewrite_state.reset();
            sequence.rewrite_checkpoint        = {};
            sequence.rewrite_checkpoint_hidden = {};
            removed.device.state_slots         = 1;
        }
        removed = checked_resource_sum(
            removed, install_private_capture(sequence, transaction.group, transaction.source_state,
                                             transaction.private_replacement));
    }
    if (transaction.replaces_shared) {
        if (!transaction.shared_index) {
            throw std::logic_error("shared replacement has no descriptor");
        }
        const std::uint32_t index = *transaction.shared_index;
        if (!transaction.replacement_removed ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::ReservedCapture ||
            shared_prefix_slots[index].generation != transaction.replacement_generation) {
            throw std::logic_error("shared replacement generation changed before publication");
        }
    }
    if (removed != transaction.resource_delta.removed) {
        throw std::logic_error("active capture replacement effect changed after reservation");
    }

    const detail::PhysicalResources private_replacement_removed =
        checked_resource_difference(removed, transaction.capacity_preparation_removed);
    request.optional_resources =
        checked_resource_difference(request.optional_resources, private_replacement_removed);
    if (transaction.publish_private && !transaction.publish_shared) {
        request.optional_resources =
            checked_resource_sum(request.optional_resources, transaction.resource_delta.added);
    }
    request.active_resources = checked_resource_sum(
        checked_resource_difference(request.active_resources,
                                    transaction.active_entitlement_delta.removed),
        transaction.active_entitlement_delta.added);

    if (transaction.state_placement == qwen3_6::CaptureStatePlacement::DeviceFork) {
        ++transaction.operations.state_forks;
    }
    ActiveCaptureResult out;
    out.status                         = runtime::ContextTransactionStatus::Published;
    out.capacity_preparation_committed = transaction.replacement_removed;
    if (transaction.publish_private) {
        populate_continuation_summary(sequence, transaction.active_summary);
        out.active_summary = std::move(transaction.active_summary);
    }
    out.victims               = std::move(transaction.pressure_results);
    out.shared_victims        = std::move(transaction.shared_pressure_results);
    out.transfer_observations = std::move(transaction.transfer_observations);
    out.operations            = transaction.operations;
    if (transaction.publish_shared) {
        if (!transaction.shared_index || !shared_bundle) {
            throw std::logic_error("shared capture publication has no reserved descriptor");
        }
        const std::uint32_t index = *transaction.shared_index;
        SharedPrefixSlot& slot    = shared_prefix_slots[index];
        SharedPrefixState& shared = shared_prefix_states[index];
        if (slot.role != SharedPrefixSlotRole::ReservedCapture || shared.kv || shared.identity) {
            throw std::logic_error("shared capture descriptor changed before publication");
        }
        shared.kv       = *shared_bundle;
        shared.state    = transaction.source_state;
        shared.identity = transaction.group.identity;
        shared.frontier = transaction.group.frontier;
        shared.backend_frontier =
            speculative_backend == SpeculativeBackend::Mtp      ? transaction.group.frontier - 1U
            : speculative_backend == SpeculativeBackend::DFlash ? transaction.group.frontier
                                                                : 0U;
        shared.rope_delta        = sequence.rope_delta;
        shared.tail_hidden_valid = sequence.tail_hidden_valid;
        shared.rebuild_work      = validated_rebuild_work(transaction.group.identity->rebuild_work,
                                                          transaction.group.frontier);
        shared.active_references = 1;
        sequence.shared_prefix_references.push_back(index);
        slot.role = SharedPrefixSlotRole::Catalogued;
        out.shared.emplace(SharedPrefixPublication{
            .handle  = ContractAccess::make_shared_prefix(this, index, slot.generation),
            .summary = shared_prefix_summary(shared),
        });
    }

    prefill.pending_capture_offer = 0;
    const bool post_begin_prompt_frontier_capture =
        prefill.cursor == prefill.prompt_tokens && request.lifecycle != Lifecycle::Prefilling;
    ++prefill.next_capture;
    if (post_begin_prompt_frontier_capture) { request.prefill.reset(); }
    transaction.published = true;
    return out;
}

ActiveCaptureResult
ProgramImplCore::progress_active_capture_transaction(runtime::CancellationFlagView cancellation) {
    ActiveCaptureTransaction* transaction_ptr =
        std::get_if<ActiveCaptureTransaction>(&context_transaction_);
    if (transaction_ptr == nullptr) {
        throw std::logic_error("Program has no active capture transaction");
    }
    ActiveCaptureTransaction& transaction   = *transaction_ptr;
    PressureTransition& pressure_transition = transaction.pressure_transition;
    if (transaction.published) {
        throw std::logic_error("active capture terminal result was already returned");
    }
    const auto abort = [&]() -> ActiveCaptureResult {
        abort_active_capture(transaction);
        if (transaction.lane < max_concurrency && requests[transaction.lane].prefill) {
            RequestControl::Prefill& prefill = *requests[transaction.lane].prefill;
            const bool post_begin_prompt_frontier_capture =
                prefill.cursor == prefill.prompt_tokens &&
                requests[transaction.lane].lifecycle != Lifecycle::Prefilling;
            prefill.pending_capture_offer = 0;
            ++prefill.next_capture;
            if (post_begin_prompt_frontier_capture) { requests[transaction.lane].prefill.reset(); }
        }
        transaction.published = true;
        ActiveCaptureResult out;
        out.status                         = runtime::ContextTransactionStatus::Aborted;
        out.capacity_preparation_committed = transaction.replacement_removed;
        out.victims                        = std::move(transaction.pressure_results);
        out.shared_victims                 = std::move(transaction.shared_pressure_results);
        out.transfer_observations          = std::move(transaction.transfer_observations);
        out.operations                     = transaction.operations;
        return out;
    };
    const auto has_pressure = [&]() {
        return !transaction.pressure.empty() || !transaction.shared_pressure.empty();
    };
    const auto for_each_pending_pressure = [&](auto&& callback) {
        for (MaterializationTransaction::PressureWork& work : transaction.shared_pressure) {
            if (!work.completed) { callback(work); }
        }
        for (MaterializationTransaction::PressureWork& work : transaction.pressure) {
            if (!work.completed) { callback(work); }
        }
    };
    const auto collect_spill = [&](MaterializationTransaction::PressureWork& work) {
        transaction.operations.pressure_spill_pages =
            work.spill_pages > std::numeric_limits<std::uint64_t>::max() -
                                   transaction.operations.pressure_spill_pages
                ? std::numeric_limits<std::uint64_t>::max()
                : transaction.operations.pressure_spill_pages + work.spill_pages;
        work.spill_pages = 0;
    };

    if (has_pressure() && pressure_transition.phase == PressureTransitionPhase::HostReleases) {
        if (cancellation.requested()) { return abort(); }
        for (std::size_t position = 0; position < transaction.shared_pressure.size(); ++position) {
            auto& work                     = transaction.shared_pressure[position];
            const std::uint32_t index      = transaction.shared_victim_indices[position];
            const std::uint64_t generation = transaction.shared_victim_generations[position];
            if (work.option.evicts_continuation) {
                if (index >= shared_prefix_capacity ||
                    shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
                    shared_prefix_slots[index].generation != generation ||
                    shared_prefix_states[index].active_references != 0) {
                    throw std::logic_error("capture shared pressure victim changed before release");
                }
                const detail::PhysicalResources resident =
                    resident_resources(shared_prefix_states[index]);
                if (!can_release_shared_prefix_state(index, SharedPrefixSlotRole::Catalogued)) {
                    throw std::logic_error(
                        "capture shared pressure victim is not strictly releasable");
                }
                const detail::PhysicalResources released =
                    release_shared_prefix_state_strict(index, SharedPrefixSlotRole::Catalogued);
                if (released != resident ||
                    work.option.effect.added != detail::PhysicalResources{}) {
                    throw std::logic_error("capture shared pressure eviction changed");
                }
                work.committed_delta    = detail::PhysicalDelta{.removed = released};
                work.completed          = true;
                work.mutation_published = true;
                transaction.shared_pressure_results[position] = MaterializationSharedVictimResult{
                    .owner              = transaction.shared_pressure_results[position].owner,
                    .disposition        = runtime::VictimDisposition::Evicted,
                    .pressure_committed = true,
                };
            } else {
                publish_pressure_host_releases(work);
                if (work.completed) {
                    transaction.shared_pressure_results[position] =
                        MaterializationSharedVictimResult{
                            .owner       = transaction.shared_pressure_results[position].owner,
                            .disposition = runtime::VictimDisposition::Retained,
                            .pressure_committed = true,
                            .final_summary = shared_prefix_summary(shared_prefix_states[index]),
                        };
                }
            }
        }
        for (std::size_t position = 0; position < transaction.pressure.size(); ++position) {
            auto& work                     = transaction.pressure[position];
            const std::uint32_t index      = transaction.victim_indices[position];
            const std::uint64_t generation = transaction.victim_generations[position];
            if (work.option.evicts_continuation) {
                if (index >= continuation_capacity ||
                    continuation_slots[index].role != ContinuationSlotRole::Catalogued ||
                    continuation_slots[index].generation != generation ||
                    work.option.effect.added != detail::PhysicalResources{}) {
                    throw std::logic_error(
                        "capture private pressure victim changed before release");
                }
                if (!can_release_continuation_slot_strict(index)) {
                    throw std::logic_error("capture private victim is not strictly releasable");
                }
                const detail::PhysicalResources resident =
                    resident_resources(continuation_states[index]);
                release_continuation_slot_strict(index);
                work.committed_delta                   = detail::PhysicalDelta{.removed = resident};
                work.completed                         = true;
                work.mutation_published                = true;
                transaction.pressure_results[position] = MaterializationVictimResult{
                    .owner              = transaction.pressure_results[position].owner,
                    .disposition        = runtime::VictimDisposition::Evicted,
                    .pressure_committed = true,
                };
            } else {
                publish_pressure_host_releases(work);
                if (work.completed) {
                    transaction.pressure_results[position] = MaterializationVictimResult{
                        .owner              = transaction.pressure_results[position].owner,
                        .disposition        = runtime::VictimDisposition::Retained,
                        .pressure_committed = true,
                        .final_summary      = continuation_summary(continuation_states[index]),
                    };
                }
            }
        }
        pressure_transition.phase = PressureTransitionPhase::CopyPreparation;
    }

    if (has_pressure() && pressure_transition.phase == PressureTransitionPhase::CopyPreparation) {
        constexpr std::array resources{
            runtime::ContextResourceClass::State,
            runtime::ContextResourceClass::MainKV,
            runtime::ContextResourceClass::BackendKV,
        };
        context_source_ready_.record(device.stream);
        context_source_ready_.wait(device.transfer_stream);
        try {
            for (const runtime::ContextResourceClass resource : resources) {
                bool has_copy = false;
                for_each_pending_pressure([&](const auto& work) {
                    has_copy =
                        has_copy ||
                        std::any_of(work.option.transfer_requirements.begin(),
                                    work.option.transfer_requirements.end(),
                                    [&](const auto& requirement) {
                                        return requirement.resource == resource &&
                                               requirement.direction ==
                                                   runtime::ContextTransferDirection::DeviceToHost;
                                    });
                });
                if (has_copy) { start_context_transfer_timer(resource); }
                for_each_pending_pressure(
                    [&](auto& work) { prepare_pressure_work(work, resource); });
                if (!has_copy) { continue; }
                stop_context_transfer_timer(resource);
                const std::size_t resource_index = context_resource_index(resource);
                pressure_transition.timer_mask |= static_cast<std::uint8_t>(1U << resource_index);
                for_each_pending_pressure([&](const auto& work) {
                    for (const auto& requirement : work.option.transfer_requirements) {
                        if (requirement.resource != resource ||
                            requirement.direction !=
                                runtime::ContextTransferDirection::DeviceToHost) {
                            continue;
                        }
                        TransferWork& total = pressure_transition.transfer_work[resource_index];
                        total.payload_bytes =
                            requirement.work.payload_bytes >
                                    std::numeric_limits<std::uint64_t>::max() - total.payload_bytes
                                ? std::numeric_limits<std::uint64_t>::max()
                                : total.payload_bytes + requirement.work.payload_bytes;
                        const std::uint64_t operations =
                            static_cast<std::uint64_t>(total.copy_operations) +
                            requirement.work.copy_operations;
                        total.copy_operations =
                            operations > std::numeric_limits<std::uint32_t>::max()
                                ? std::numeric_limits<std::uint32_t>::max()
                                : static_cast<std::uint32_t>(operations);
                        const std::uint64_t pages =
                            static_cast<std::uint64_t>(
                                pressure_transition.transfer_pages[resource_index]) +
                            requirement.page_count;
                        pressure_transition.transfer_pages[resource_index] =
                            pages > std::numeric_limits<std::uint32_t>::max()
                                ? std::numeric_limits<std::uint32_t>::max()
                                : static_cast<std::uint32_t>(pages);
                        if (resource == runtime::ContextResourceClass::State) {
                            pressure_transition.state_images =
                                requirement.units > std::numeric_limits<std::uint64_t>::max() -
                                                        pressure_transition.state_images
                                    ? std::numeric_limits<std::uint64_t>::max()
                                    : pressure_transition.state_images + requirement.units;
                        }
                    }
                });
            }
        } catch (...) {
            (void)cudaStreamSynchronize(device.transfer_stream);
            for_each_pending_pressure([&](auto& work) { abort_pressure_work(work); });
            throw;
        }
        bool copies_submitted = false;
        for_each_pending_pressure(
            [&](const auto& work) { copies_submitted = copies_submitted || work.submitted; });
        pressure_transition.phase = copies_submitted ? PressureTransitionPhase::CopiesInFlight
                                                     : PressureTransitionPhase::CopyPublication;
        if (copies_submitted) {
            context_completion_.record(device.transfer_stream);
            return ActiveCaptureResult{.status = runtime::ContextTransactionStatus::InProgress};
        }
    }

    if (pressure_transition.phase == PressureTransitionPhase::CopiesInFlight) {
        if (!context_completion_.ready()) {
            return ActiveCaptureResult{.status = runtime::ContextTransactionStatus::InProgress};
        }
        pressure_transition.phase = PressureTransitionPhase::CopyPublication;
    }
    if (has_pressure() && pressure_transition.phase == PressureTransitionPhase::CopyPublication) {
        for (std::size_t position = 0; position < transaction.shared_pressure.size(); ++position) {
            auto& work = transaction.shared_pressure[position];
            if (!work.completed) {
                publish_pressure_work(work);
                collect_spill(work);
                transaction.shared_pressure_results[position] = MaterializationSharedVictimResult{
                    .owner              = transaction.shared_pressure_results[position].owner,
                    .disposition        = runtime::VictimDisposition::Retained,
                    .pressure_committed = true,
                    .final_summary      = shared_prefix_summary(
                        shared_prefix_states[transaction.shared_victim_indices[position]]),
                };
            }
        }
        for (std::size_t position = 0; position < transaction.pressure.size(); ++position) {
            auto& work = transaction.pressure[position];
            if (!work.completed) {
                publish_pressure_work(work);
                collect_spill(work);
                transaction.pressure_results[position] = MaterializationVictimResult{
                    .owner              = transaction.pressure_results[position].owner,
                    .disposition        = runtime::VictimDisposition::Retained,
                    .pressure_committed = true,
                    .final_summary      = continuation_summary(
                        continuation_states[transaction.victim_indices[position]]),
                };
            }
        }
        constexpr std::array resources{
            runtime::ContextResourceClass::State,
            runtime::ContextResourceClass::MainKV,
            runtime::ContextResourceClass::BackendKV,
        };
        for (const runtime::ContextResourceClass resource : resources) {
            const std::size_t index = context_resource_index(resource);
            if ((pressure_transition.timer_mask & (1U << index)) == 0) { continue; }
            transaction.transfer_observations.push_back(context_transfer_observation(
                resource, runtime::ContextTransferDirection::DeviceToHost,
                pressure_transition.transfer_work[index], pressure_transition.transfer_pages[index],
                pressure_transition.state_images));
        }
        pressure_transition.timer_mask = 0;
        pressure_transition.phase      = PressureTransitionPhase::Committed;
    }
    if (has_pressure() && pressure_transition.phase != PressureTransitionPhase::Committed) {
        throw std::logic_error("capture pressure transition did not reach a stable phase");
    }
    if (cancellation.requested()) { return abort(); }
    if (!transaction.prepared) {
        if (cancellation.requested()) { return abort(); }
        try {
            prepare_active_capture(transaction);
        } catch (...) {
            abort_active_capture(transaction);
            transaction.published = true;
            throw;
        }
    }
    if (transaction.transfer_enqueue_pending) {
        if (cancellation.requested()) { return abort(); }
        try {
            enqueue_active_capture_transfers(transaction);
        } catch (...) {
            if (device.transfer_stream != nullptr) {
                (void)cudaStreamSynchronize(device.transfer_stream);
            }
            abort_active_capture(transaction);
            transaction.published = true;
            throw;
        }
        return ActiveCaptureResult{.status = runtime::ContextTransactionStatus::InProgress};
    }
    if (transaction.transfer_submitted && !context_completion_.ready()) {
        return ActiveCaptureResult{.status = runtime::ContextTransactionStatus::InProgress};
    }
    if (transaction.transfer_submitted) {
        const auto record = [&](runtime::ContextResourceClass resource,
                                runtime::ContextTransferDirection direction, TransferWork work,
                                std::uint32_t pages) {
            const std::uint8_t bit =
                static_cast<std::uint8_t>(1U << context_resource_index(resource));
            if ((transaction.transfer_timer_mask & bit) == 0) { return; }
            transaction.transfer_observations.push_back(
                context_transfer_observation(resource, direction, work, pages));
            transaction.transfer_timer_mask &= static_cast<std::uint8_t>(~bit);
        };
        const auto planned_work = [&](runtime::ContextResourceClass resource,
                                      runtime::ContextTransferDirection direction) {
            const auto found = std::find_if(
                transaction.transfer_requirements.begin(), transaction.transfer_requirements.end(),
                [&](const auto& requirement) {
                    return requirement.resource == resource && requirement.direction == direction;
                });
            return found == transaction.transfer_requirements.end() ? TransferWork{} : found->work;
        };
        const runtime::ContextTransferDirection state_direction =
            transaction.state_placement == qwen3_6::CaptureStatePlacement::HostSnapshot
                ? runtime::ContextTransferDirection::DeviceToHost
                : runtime::ContextTransferDirection::DeviceToDevice;
        record(runtime::ContextResourceClass::State, state_direction,
               planned_work(runtime::ContextResourceClass::State, state_direction), 0);
        record(runtime::ContextResourceClass::MainKV,
               runtime::ContextTransferDirection::DeviceToDevice,
               planned_work(runtime::ContextResourceClass::MainKV,
                            runtime::ContextTransferDirection::DeviceToDevice),
               1);
        if (backend_kv_pages) {
            record(runtime::ContextResourceClass::BackendKV,
                   runtime::ContextTransferDirection::DeviceToDevice,
                   planned_work(runtime::ContextResourceClass::BackendKV,
                                runtime::ContextTransferDirection::DeviceToDevice),
                   1);
        }
        transaction.transfer_submitted = false;
    }
    if (cancellation.requested()) { return abort(); }
    return publish_active_capture(transaction);
}

PendingBatch ProgramImplCore::decode(std::span<const SequenceHandle> members,
                                     std::span<const runtime::RoundBudget> budgets,
                                     runtime::ExecutionTiming* failed_timing) {
    if (pending_transaction_ || members.empty() || members.size() > max_concurrency ||
        budgets.size() != members.size()) {
        throw std::invalid_argument("decode membership is invalid");
    }
    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    for (std::size_t row = 0; row < members.size(); ++row) {
        if (!valid_sequence(members[row])) {
            throw std::logic_error("decode sequence capability is invalid");
        }
        const std::uint32_t lane = ContractAccess::lane(members[row]).value;
        if (requests[lane].lifecycle != Lifecycle::Active ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("decode membership is duplicate or not active");
        }
        lanes[row] = lane;
    }
    const auto lane_span = std::span<const std::uint32_t>(lanes.data(), members.size());
    try {
        runtime::BatchedGeneratedRound round = decode_raw(lane_span, budgets, failed_timing);
        if (failed_timing != nullptr) { *failed_timing += round.timing; }
        return wrap_pending(lane_span, std::move(round));
    } catch (...) {
        const Clock::time_point cleanup_started = Clock::now();
        clear_execution_failure_lanes(lane_span);
        pending_transaction_.reset();
        if (failed_timing != nullptr) {
            failed_timing->post_host_ns += elapsed_ns(cleanup_started);
        }
        throw;
    }
}

runtime::ExecutionTiming ProgramImplCore::append_forced_tokens(
    std::span<const SequenceHandle> members, std::span<const TokenId> row_major_tokens,
    std::uint32_t row_stride, runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (pending_transaction_ || members.empty() || members.size() > max_concurrency ||
        row_stride == 0 ||
        row_major_tokens.size() != static_cast<std::size_t>(row_stride) * members.size()) {
        throw std::invalid_argument("forced-token membership is invalid");
    }

    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    for (std::size_t row = 0; row < members.size(); ++row) {
        if (!valid_sequence(members[row])) {
            throw std::logic_error("forced-token sequence capability is invalid");
        }
        const std::uint32_t lane = ContractAccess::lane(members[row]).value;
        if (requests[lane].lifecycle != Lifecycle::Active ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("forced-token membership is duplicate or not active");
        }
        const SequenceState& sequence = active_sequence(lane);
        if (sequence.execution_frontier == std::numeric_limits<std::uint32_t>::max() ||
            sequence.ledger_frontier != sequence.execution_frontier + 1U ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            (speculative_backend == SpeculativeBackend::Mtp &&
             sequence.mtp_kv_valid != sequence.execution_frontier) ||
            (speculative_backend == SpeculativeBackend::DFlash &&
             sequence.dflash_context_frontier > sequence.execution_frontier) ||
            static_cast<std::uint64_t>(sequence.execution_frontier) + row_stride > capacity) {
            throw std::logic_error("forced-token sequence frontier is invalid");
        }
        validate_licensed_tokens(row_major_tokens.subspan(row * row_stride, row_stride));
        lanes[row] = lane;
    }

    const bool count_forced_tokens = std::any_of(
        lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(members.size()),
        [&](std::uint32_t lane) { return requests[lane].sampling_host.token_counts != nullptr; });
    if (count_forced_tokens) {
        work.reset();
        Tensor forced_ids =
            work.alloc(DType::I32, {checked_i32(static_cast<std::uint32_t>(row_major_tokens.size()),
                                                "forced-token batch exceeds int32")});
        CUDA_CHECK(cudaMemcpyAsync(forced_ids.data, row_major_tokens.data(), forced_ids.bytes(),
                                   cudaMemcpyHostToDevice, device.stream));
        for (std::size_t row = 0; row < members.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            if (requests[lane].sampling_host.token_counts == nullptr) { continue; }
            Tensor ids    = forced_ids.slice(0, static_cast<std::int32_t>(row * row_stride),
                                             static_cast<std::int32_t>(row_stride));
            Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(lane), 1)
                                .view({TextConfig::token_domain});
            ops::increment_token_counts(ids, counts, device.stream);
        }
        work.reset();
    }

    try {
        for (std::size_t row = 0; row < members.size(); ++row) {
            timing.resume_submit();
            const std::uint32_t lane = lanes[row];
            SequenceState& sequence  = active_sequence(lane);
            RequestControl& request  = requests[lane];
            const std::span<const TokenId> forced =
                row_major_tokens.subspan(row * row_stride, row_stride);
            const std::uint32_t base = sequence.execution_frontier;
            const std::uint32_t end  = base + row_stride;
            const auto started       = Clock::now();

            if (speculative_backend == SpeculativeBackend::DFlash &&
                sequence.dflash_context_frontier < base) {
                const std::array<std::uint32_t, 1> append_lanes{lane};
                const std::array<std::uint32_t, 1> append_starts{sequence.dflash_context_frontier};
                const std::array<std::uint32_t, 1> append_counts{base -
                                                                 sequence.dflash_context_frontier};
                enqueue_dflash_context_append(append_lanes, append_starts, append_counts);
                timing.begin_wait();
                device.synchronize();
                timing.end_wait();
                sequence.dflash_context_frontier = base;
                commit_sequence_kv(sequence, sequence.text_kv_valid,
                                   sequence.dflash_context_frontier);
                work.reset();
                timing.resume_submit();
            }

            materialize_sequence_kv(sequence, end,
                                    speculative_backend == SpeculativeBackend::None ? 0U : end);

            sequence.ledger.insert(sequence.ledger.end(), forced.begin(), forced.end());
            if (sequence.ledger.size() != static_cast<std::size_t>(end) + 1U) {
                throw std::logic_error("forced-token continuation ledger has an invalid shape");
            }

            if (speculative_backend == SpeculativeBackend::DFlash) {
                if (!dflash || !io.dflash_decode || !sequence.kv || !sequence.kv->backend) {
                    throw std::logic_error("DFlash forced continuation state is incomplete");
                }
                *dflash_host_ingress                            = {};
                dflash_host_ingress->active_lanes[0]            = static_cast<std::int32_t>(lane);
                const StateImageSelectors selectors             = state_selectors(sequence);
                dflash_host_ingress->state_source_slots[0]      = selectors.source;
                dflash_host_ingress->state_destination_slots[0] = selectors.destination;
                dflash_host_ingress->dflash_kv_table_rows[0] =
                    backend_kv_addresses->bound_row(*sequence.kv->backend);
                CUDA_CHECK(cudaMemcpyAsync(io.dflash_decode->ingress.data, dflash_host_ingress,
                                           sizeof(qwen3_6::DFlashDecodeIngress),
                                           cudaMemcpyHostToDevice, device.stream));
            }

            std::uint32_t cursor = base;
            while (cursor < end) {
                const std::uint32_t count           = std::min(prefill_chunk, end - cursor);
                const StateImageSelectors selectors = state_selectors(sequence);
                schedule::PrefillContext schedule_state{
                    {device, model, work, state_images->linear(),
                     replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
                     proposal_head},
                    text_kv_view(sequence),
                    mtp_kv_view(sequence),
                    decoder->text_kv,
                    decoder->mtp_cache(),
                    dflash ? &*dflash : nullptr,
                    cursor,
                    nullptr,
                    nullptr,
                    selectors.source,
                    selectors.destination,
                    0,
                    dflash_host_ingress};
                mark_workspace_usage(speculative_backend == SpeculativeBackend::Mtp
                                         ? workspace_plan.mtp_prefill
                                         : workspace_plan.text_prefill);
                if (speculative_backend == SpeculativeBackend::DFlash) {
                    mark_workspace_usage(workspace_plan.dflash_context);
                }
                const schedule::PrefillChunkResult result = schedule::prefill_text_chunk(
                    schedule_state, sequence.ledger, count, std::nullopt, false);
                if (result.finalized || result.processed_tokens == 0 ||
                    result.processed_tokens > count) {
                    throw std::logic_error("forced-token prefill made invalid progress");
                }
                cursor += result.processed_tokens;
                sequence.text_kv_valid = cursor;
                if (speculative_backend == SpeculativeBackend::Mtp) {
                    sequence.mtp_kv_valid = cursor;
                } else if (speculative_backend == SpeculativeBackend::DFlash) {
                    sequence.dflash_context_frontier = cursor;
                }
                commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
                settle_state_fork(sequence);
                copy_tail(sequence,
                          prefill_hidden.slice(
                              1, static_cast<std::int32_t>(result.processed_tokens) - 1, 1));
            }
            timing.begin_wait();
            device.synchronize();
            timing.end_wait();
            work.reset();

            sequence.prefix_identity.append_generated(row_stride, sequence.rope_delta);
            sequence.prefix_digests.append_generated(forced, sequence.rope_delta);
            advance_rebuild_work(sequence, end, prefill_chunk);
            sequence.execution_frontier = end;
            sequence.ledger_frontier    = end + 1U;
            sequence.mtp_draft_count    = 0;
            sequence.tail_hidden_valid  = true;
            if (sequence.ledger.size() != sequence.ledger_frontier ||
                sequence.prefix_identity.size() != sequence.ledger_frontier ||
                sequence.prefix_digests.size() != sequence.ledger_frontier ||
                sequence.ledger.back() != forced.back()) {
                throw std::logic_error("forced-token commit did not establish a valid frontier");
            }
            trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            request.timings.decode_seconds +=
                std::chrono::duration<double>(Clock::now() - started).count();
        }
        return timing.finish();
    } catch (...) {
        timing.begin_wait();
        try {
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        work.reset();
        clear_execution_failure_lanes(std::span<const std::uint32_t>(lanes.data(), members.size()));
        throw;
    }
}

CommitResult ProgramImplCore::commit(PendingBatch&& pending,
                                     std::span<const runtime::CommitDecision> decisions,
                                     runtime::CommitObservation observation,
                                     runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    std::array<SequenceHandle, kMaximumConcurrency> members{};
    const auto input_rows       = ContractAccess::rows(pending);
    const std::size_t row_count = input_rows.size();
    for (std::size_t row = 0; row < row_count; ++row) { members[row] = input_rows[row]; }
    const bool valid = valid_pending(pending);
    ContractAccess::consume(pending);

    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    std::array<GenerationTimings, kMaximumConcurrency> timings{};
    std::array<SpeculativeStats, kMaximumConcurrency> speculative{};
    std::array<PendingKind, kMaximumConcurrency> pending_kinds{};
    const auto release_members = [&]() noexcept {
        std::array<std::uint32_t, kMaximumConcurrency> failed_lanes{};
        std::size_t failed_count = 0;
        for (std::size_t row = 0; row < row_count; ++row) {
            if (ContractAccess::owner(members[row]) != this) { continue; }
            const std::uint32_t lane = ContractAccess::lane(members[row]).value;
            if (lane >= max_concurrency) { continue; }
            failed_lanes[failed_count++] = lane;
        }
        clear_execution_failure_lanes(
            std::span<const std::uint32_t>(failed_lanes.data(), failed_count));
        pending_transaction_.reset();
    };

    try {
        if (!valid || row_count == 0 || row_count > max_concurrency ||
            decisions.size() != row_count) {
            throw std::logic_error("pending transaction capability or decision shape is invalid");
        }
        std::array<std::uint32_t, kMaximumConcurrency> accepted{};
        std::array<std::uint8_t, kMaximumConcurrency> terminal{};
        std::array<std::uint8_t, kMaximumConcurrency> cancelled{};
        for (std::size_t row = 0; row < row_count; ++row) {
            const std::uint32_t lane                = ContractAccess::lane(members[row]).value;
            lanes[row]                              = lane;
            const PendingCandidate& candidate       = requests[lane].pending;
            pending_kinds[row]                      = candidate.kind;
            const runtime::CommitDecision& decision = decisions[row];
            if (decision.cancelled && has_context_transaction()) {
                throw std::logic_error(
                    "active cancellation overlaps the global context transaction");
            }
            if ((decision.cancelled && (decision.accepted_tokens != 0 || !decision.terminal)) ||
                (!decision.cancelled &&
                 (decision.accepted_tokens == 0 || decision.accepted_tokens > candidate.produced ||
                  (!decision.terminal && decision.accepted_tokens != candidate.produced)))) {
                throw std::logic_error("pending transaction decision is invalid");
            }
            accepted[row]  = decision.accepted_tokens;
            terminal[row]  = decision.terminal ? 1U : 0U;
            cancelled[row] = decision.cancelled ? 1U : 0U;
            if (decision.cancelled) {
                timings[row]     = requests[lane].timings;
                speculative[row] = std::move(requests[lane].speculative_stats);
            }
        }

        timing.pause();
        timing.include(resolve_pending_raw(
            std::span<const std::uint32_t>(lanes.data(), row_count),
            std::span<const std::uint32_t>(accepted.data(), row_count),
            std::span<const std::uint8_t>(terminal.data(), row_count),
            std::span<const std::uint8_t>(cancelled.data(), row_count), failed_timing));
        timing.resume_post();
        pending_transaction_.reset();

        CommitResult out;
        out.row_count          = row_count;
        bool released_resource = false;
        for (std::size_t row = 0; row < row_count; ++row) {
            if (decisions[row].cancelled) {
                invalidate_lane(lanes[row]);
                released_resource = true;
                out.rows[row]     = CommitRowResult{
                        .disposition = runtime::CommitDisposition::CancelledReleased,
                        .timings     = timings[row],
                        .speculative = std::move(speculative[row]),
                };
            } else if (decisions[row].terminal) {
                out.rows[row].disposition = runtime::CommitDisposition::Finishable;
                if (observation == runtime::CommitObservation::AllRows) {
                    out.rows[row].timings     = requests[lanes[row]].timings;
                    out.rows[row].speculative = requests[lanes[row]].speculative_stats;
                }
            } else {
                out.rows[row].disposition = runtime::CommitDisposition::Active;
                if (observation == runtime::CommitObservation::AllRows) {
                    out.rows[row].timings     = requests[lanes[row]].timings;
                    out.rows[row].speculative = requests[lanes[row]].speculative_stats;
                }
            }

            if (pending_kinds[row] != PendingKind::Begin || decisions[row].cancelled) { continue; }
            RequestControl& request = requests[lanes[row]];
            if (decisions[row].terminal) {
                request.prefill.reset();
                continue;
            }
            if (!request.prefill) { continue; }
            RequestControl::Prefill& prefill = *request.prefill;
            if (prefill.cursor != prefill.prompt_tokens ||
                prefill.next_capture >= prefill.capture_groups.size() ||
                prefill.capture_groups[prefill.next_capture].frontier != prefill.prompt_tokens ||
                prefill.pending_capture_offer != 0) {
                throw std::logic_error("prompt-frontier capture carrier is inconsistent");
            }
            if (++next_capture_offer_id_ == 0) { ++next_capture_offer_id_; }
            prefill.pending_capture_offer = next_capture_offer_id_;
            out.captures[row].emplace(ContractAccess::make_capture_offer(
                this, runtime::LaneId{lanes[row]}, lane_epochs[lanes[row]],
                prefill.pending_capture_offer));
        }
        if (released_resource) { advance_resource_revision(); }
        out.timing = timing.finish();
        return out;
    } catch (...) {
        timing.resume_post();
        release_members();
        throw;
    }
}

DiscardResult ProgramImplCore::abort_pending(PendingBatch&& pending) noexcept {
    DiscardResult out;
    const auto rows  = ContractAccess::rows(pending);
    const bool valid = valid_pending(pending);
    out.row_count    = std::min<std::size_t>(rows.size(), kMaximumConcurrency);
    std::array<SequenceHandle, kMaximumConcurrency> members{};
    for (std::size_t row = 0; row < out.row_count; ++row) { members[row] = rows[row]; }
    ContractAccess::consume(pending);
    if (!valid) { return out; }
    std::array<std::uint32_t, kMaximumConcurrency> failed_lanes{};
    for (std::size_t row = 0; row < out.row_count; ++row) {
        failed_lanes[row] = ContractAccess::lane(members[row]).value;
    }
    const bool deferred_to_fail_all = has_context_transaction();
    clear_execution_failure_lanes(
        std::span<const std::uint32_t>(failed_lanes.data(), out.row_count));
    pending_transaction_.reset();
    if (deferred_to_fail_all) { return out; }
    if (out.row_count != 0) { advance_resource_revision(); }
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

FinishResult ProgramImplCore::finish(SequenceHandle sequence) noexcept {
    FinishResult out;
    if (has_context_transaction() || pending_transaction_ || !valid_sequence(sequence)) {
        return out;
    }
    const std::uint32_t lane               = ContractAccess::lane(sequence).value;
    RequestControl& request                = requests[lane];
    SequenceState& state                   = active_sequence(lane);
    const std::uint32_t continuation_index = active_continuations[lane];
    if (request.lifecycle != Lifecycle::Finishable) { return out; }
    if (!request.publish_continuation) {
        if (!clear_lane_strict(state, request)) { return out; }
        out.disposition = runtime::FinishDisposition::Released;
        out.timings     = request.timings;
        out.speculative = std::move(request.speculative_stats);
        invalidate_lane(lane);
        advance_resource_revision();
        out.status = runtime::ConsumeStatus::Consumed;
        return out;
    }
    try {
        out.summary.long_anchors.reserve(state.long_anchors.size());
    } catch (...) { return out; }
    try {
        if (state.state.fork_pending) {
            const StateImageHandle source      = state.state.read;
            const StateImageHandle destination = state.state.write;
            state_store->abort_fork(source, destination);
            if (!state_store->release(destination)) { return out; }
            state.state = ActiveStateBinding{.read = source, .write = source};
        }
        if (state.reserved_state) {
            if (!state_store->release(*state.reserved_state)) { return out; }
            state.reserved_state.reset();
        }
        if (state.rewrite_state && *state.rewrite_state == state.state.read) {
            if (state_store->checkpoint_references(*state.rewrite_state) == 0) { return out; }
            state_store->release_checkpoint_reference(*state.rewrite_state);
            state.rewrite_state.reset();
            state.rewrite_checkpoint = {};
        }
        if (state_store->role(state.state.read) == StateImageRole::ActiveMutable) {
            state_store->freeze(state.state.read);
        } else if (state_store->role(state.state.read) != StateImageRole::CheckpointImmutable) {
            return out;
        }
        state.endpoint_valid = true;
        refresh_state_views(state);
        text_kv_addresses->set_checkpoint_requirement(state.kv->text, state.execution_frontier);
        if (state.kv->backend) {
            backend_kv_addresses->set_checkpoint_requirement(*state.kv->backend,
                                                             backend_kv_valid(state));
        }
        populate_continuation_summary(state, out.summary);
        out.summary.active_references = 0;
    } catch (...) { return out; }
    release_active_shared_references(state);
    release_sequence_growth_entitlement(state);
    unbind_sequence_kv(state);
    request.active_resources                    = {};
    request.optional_resources                  = {};
    request.lifecycle                           = Lifecycle::Empty;
    request.pending                             = {};
    continuation_slots[continuation_index].role = ContinuationSlotRole::Catalogued;
    active_continuations[lane]                  = continuation_capacity;
    invalidate_lane(lane);
    out.continuation.emplace(ContractAccess::make_continuation(
        this, continuation_index, continuation_slots[continuation_index].generation));
    out.timings     = request.timings;
    out.speculative = std::move(request.speculative_stats);
    out.disposition = runtime::FinishDisposition::Catalogued;
    advance_resource_revision();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

AbortResult ProgramImplCore::abort(SequenceHandle sequence) noexcept {
    AbortResult out;
    if (has_context_transaction() || pending_transaction_ || !valid_sequence(sequence)) {
        return out;
    }
    const std::uint32_t lane = ContractAccess::lane(sequence).value;
    RequestControl& request  = requests[lane];
    if (request.lifecycle == Lifecycle::Pending || request.lifecycle == Lifecycle::Empty) {
        return out;
    }
    SequenceState& state = active_sequence(lane);
    if (!clear_lane_strict(state, request)) { return out; }
    out.timings     = request.timings;
    out.speculative = std::move(request.speculative_stats);
    invalidate_lane(lane);
    advance_resource_revision();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

ReleaseResult ProgramImplCore::release_continuation(ContinuationHandle&& continuation) noexcept {
    ReleaseResult out;
    const std::uint32_t index      = ContractAccess::index(continuation);
    const std::uint64_t generation = ContractAccess::epoch(continuation);
    const bool valid               = !has_context_transaction() && !pending_transaction_ &&
                       valid_continuation(continuation) && !materialization_pins(index, generation);
    if (!valid) { return out; }
    try {
        if (!can_release_continuation_slot_strict(index)) { return out; }
    } catch (...) { return out; }
    release_continuation_slot_strict(index);
    ContractAccess::consume(continuation);
    advance_resource_revision();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

bool ProgramImplCore::can_release_shared_prefix_state(std::uint32_t index,
                                                      SharedPrefixSlotRole expected_role) const {
    if (index >= shared_prefix_capacity || !state_store || !text_kv_addresses ||
        shared_prefix_slots[index].role != expected_role) {
        return false;
    }
    const SharedPrefixState& shared = shared_prefix_states[index];
    if (shared.active_references != 0 || !shared.kv || !shared.identity ||
        !state_store->valid(shared.state) || !text_kv_addresses->can_release(shared.kv->text) ||
        (shared.kv->backend &&
         (!backend_kv_addresses || !backend_kv_addresses->can_release(*shared.kv->backend)))) {
        return false;
    }
    const std::uint32_t state_references = state_store->checkpoint_references(shared.state);
    return state_references != 0 &&
           (state_references != 1 ||
            state_store->can_release_after_checkpoint_references(shared.state, 1));
}

detail::PhysicalResources
ProgramImplCore::release_shared_prefix_state_strict(std::uint32_t index,
                                                    SharedPrefixSlotRole expected_role) noexcept {
    try {
        if (!can_release_shared_prefix_state(index, expected_role)) { std::terminate(); }
        SharedPrefixState& shared               = shared_prefix_states[index];
        SharedPrefixSlot& slot                  = shared_prefix_slots[index];
        const detail::PhysicalResources removed = resident_resources(shared);
        const bool last_state_reference = state_store->checkpoint_references(shared.state) == 1;
        if (shared.kv->backend && !backend_kv_addresses->release(*shared.kv->backend)) {
            std::terminate();
        }
        if (!text_kv_addresses->release(shared.kv->text)) { std::terminate(); }
        state_store->release_checkpoint_reference(shared.state);
        if (last_state_reference && !state_store->release(shared.state)) { std::terminate(); }

        shared    = SharedPrefixState{};
        slot.role = SharedPrefixSlotRole::Free;
        if (++slot.generation == 0) { ++slot.generation; }
        if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
        return removed;
    } catch (...) { std::terminate(); }
}

ReleaseResult ProgramImplCore::release_shared_prefix(SharedPrefixHandle&& handle) noexcept {
    ReleaseResult out;
    const std::uint32_t index      = ContractAccess::index(handle);
    const std::uint64_t generation = ContractAccess::epoch(handle);
    const bool valid =
        !has_context_transaction() && !pending_transaction_ && valid_shared_prefix(handle);
    if (!valid || index >= shared_prefix_capacity ||
        shared_prefix_slots[index].generation != generation) {
        return out;
    }
    try {
        if (!can_release_shared_prefix_state(index, SharedPrefixSlotRole::Catalogued)) {
            return out;
        }
    } catch (...) { return out; }
    (void)release_shared_prefix_state_strict(index, SharedPrefixSlotRole::Catalogued);
    ContractAccess::consume(handle);
    advance_resource_revision();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

void ProgramImplCore::fail_all_cleanup() noexcept {
    pending_transaction_.reset();
    if (auto* transaction = std::get_if<ActiveCaptureTransaction>(&context_transaction_)) {
        if (transaction->transfer_submitted && device.transfer_stream != nullptr) {
            (void)cudaStreamSynchronize(device.transfer_stream);
        }
        abort_active_capture(*transaction);
    }
    if (auto* transaction = std::get_if<MaterializationTransaction>(&context_transaction_)) {
        if (transaction->transfer_submitted && device.transfer_stream != nullptr) {
            (void)cudaStreamSynchronize(device.transfer_stream);
        }
        release_materialization_staging(*transaction);
    }
    context_transaction_.emplace<std::monostate>();
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (active_continuations[lane] < continuation_capacity) {
            clear_lane_best_effort(active_sequence(lane), requests[lane]);
        }
        invalidate_lane(lane);
    }
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role != ContinuationSlotRole::Free) {
            release_continuation_slot_best_effort(index);
        }
    }
    for (std::uint32_t index = 0; index < shared_prefix_capacity; ++index) {
        if (shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued) { continue; }
        shared_prefix_states[index].active_references = 0;
        auto handle =
            ContractAccess::make_shared_prefix(this, index, shared_prefix_slots[index].generation);
        (void)release_shared_prefix(std::move(handle));
    }
}

detail::PhysicalResources ProgramImplCore::admission_capacity() const noexcept {
    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
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

bool ProgramImplCore::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
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

bool ProgramImplCore::persistent_backfill_safe(
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

qwen3_6::PhysicalUsageSnapshot ProgramImplCore::physical_usage() const noexcept {
    const detail::PhysicalResources usage = physical_occupancy();
    return qwen3_6::PhysicalUsageSnapshot{
        .resource_revision       = resource_revision_,
        .device_state_slots      = usage.device.state_slots,
        .host_state_slots        = usage.host.state_slots,
        .device_main_kv_pages    = usage.device.main_kv_pages,
        .device_backend_kv_pages = usage.device.backend_kv_pages,
        .host_kv_bytes           = usage.host.kv_bytes,
    };
}

void ProgramImplCore::start_sequence(std::uint32_t lane, SequenceState& sequence,
                                     MaterializationTransaction& transaction) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    RequestControl& request = requests[lane];
    if (!transaction.plan || transaction.plan->impl_ == nullptr || !transaction.prepared ||
        !request.prefill) {
        throw std::invalid_argument("materialization staging is incomplete");
    }
    AdmissionCandidateImpl& request_plan = *transaction.plan->impl_;
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("staged prefill requires a free request lane");
    }
    auto& staged                           = *request.prefill;
    const auto started                     = Clock::now();
    const std::uint32_t prompt_tokens      = staged.prompt_tokens;
    const std::uint32_t base               = staged.base;
    const std::uint32_t initial_mtp_extent = staged.initial_mtp_extent;
    request.lifecycle                      = Lifecycle::Empty;
    try {
        const std::uint32_t state_slots = request_plan.demand.active_entitlement.device.state_slots;
        const bool preserving_source =
            (transaction.has_source || transaction.has_shared_source) &&
            transaction.source_mode == runtime::PrivateSourceMode::Retain;
        const bool text_prefix_fork    = request_plan.text_prefix_fork_required;
        const bool backend_prefix_fork = request_plan.backend_prefix_fork_required;
        if (request_plan.reuse == ReusePath::Root) {
            if (transaction.reserved_state_count != state_slots || state_slots == 0 ||
                !transaction.root_text_address || !transaction.text_activation ||
                transaction.root_backend_address.has_value() !=
                    (request_plan.backend_kv_page_entitlement != 0) ||
                transaction.backend_activation.has_value() !=
                    (request_plan.backend_kv_page_entitlement != 0)) {
                throw std::logic_error("root materialization reservations are incomplete");
            }
            release_sequence_kv(sequence);
            release_sequence_state(sequence);
            sequence.state = ActiveStateBinding{.read  = transaction.reserved_states[0],
                                                .write = transaction.reserved_states[0]};
            transaction.reserved_states[0] = {};
            if (state_slots == 2) {
                sequence.reserved_state        = transaction.reserved_states[1];
                transaction.reserved_states[1] = {};
            }
            transaction.reserved_state_count = 0;

            SequenceKVBundle bundle{.text = *transaction.root_text_address};
            transaction.root_text_address.reset();
            if (transaction.root_backend_address) {
                bundle.backend = *transaction.root_backend_address;
                transaction.root_backend_address.reset();
            }
            sequence.kv.emplace(bundle);
        } else if (preserving_source) {
            const bool private_source_ready = transaction.has_source &&
                                              transaction.source_index < continuation_capacity &&
                                              continuation_slots[transaction.source_index].role ==
                                                  ContinuationSlotRole::Catalogued;
            const bool shared_source_ready =
                transaction.has_shared_source &&
                transaction.shared_source_index < shared_prefix_capacity &&
                shared_prefix_slots[transaction.shared_source_index].role ==
                    SharedPrefixSlotRole::Catalogued;
            if (private_source_ready == shared_source_ready ||
                transaction.reserved_state_count != state_slots || state_slots == 0 ||
                !transaction.root_text_address || !transaction.text_prefix_fork ||
                !transaction.prefix_forks_ready ||
                transaction.root_backend_address.has_value() !=
                    (request_plan.backend_kv_page_entitlement != 0)) {
                throw std::logic_error("retained materialization is incomplete");
            }
            const StateImageHandle selected =
                private_source_ready
                    ? selected_state(continuation_states[transaction.source_index],
                                     request_plan.reuse, request_plan.selected_checkpoint)
                    : shared_prefix_states[transaction.shared_source_index].state;
            const StateImageHandle current = transaction.reserved_states[0];
            if (state_store->residency(selected) == StateReplicaResidency::HostOnly) {
                if (state_store->role(current) != StateImageRole::ActiveMutable) {
                    throw std::logic_error("Host retained Fork destination was not published");
                }
                sequence.state = ActiveStateBinding{.read = current, .write = current};
            } else if (transaction.split_state_identity) {
                if (!private_source_ready ||
                    state_store->residency(selected) != StateReplicaResidency::Both) {
                    throw std::logic_error("StateImage identity split source changed");
                }
                state_store->split_device_replica_identity(selected, current);
                sequence.state = ActiveStateBinding{.read = current, .write = current};
            } else {
                const StateImageSelectors selectors = state_store->begin_fork(selected, current);
                if (speculative_backend == SpeculativeBackend::DFlash) {
                    state_images->copy_dflash_local(selectors.source, selectors.destination,
                                                    device.stream);
                }
                sequence.state =
                    ActiveStateBinding{.read = selected, .write = current, .fork_pending = true};
                sequence.state_source_retained = true;
            }
            transaction.reserved_states[0]   = {};
            transaction.split_state_identity = false;
            if (state_slots == 2) {
                sequence.reserved_state        = transaction.reserved_states[1];
                transaction.reserved_states[1] = {};
            }
            transaction.reserved_state_count = 0;
            sequence.rewrite_state.reset();
            sequence.rewrite_checkpoint = {};

            SequenceKVBundle bundle{.text = *transaction.root_text_address};
            transaction.root_text_address.reset();
            if (transaction.root_backend_address) {
                bundle.backend = *transaction.root_backend_address;
                transaction.root_backend_address.reset();
            }
            sequence.kv.emplace(bundle);
        } else {
            if (request_plan.state_fork_required !=
                transaction.state_fork_destination.has_value()) {
                throw std::logic_error("private materialization StateImage Fork is incomplete");
            }
            if (transaction.reserved_state_count > 1 ||
                (transaction.reserved_state_count != 0 && sequence.reserved_state)) {
                throw std::logic_error("private materialization StateImage reservation is invalid");
            }
            if (transaction.reserved_state_count == 1) {
                sequence.reserved_state          = transaction.reserved_states[0];
                transaction.reserved_states[0]   = {};
                transaction.reserved_state_count = 0;
            }
        }

        if (!preserving_source) {
            std::array<HostKVPageReplicaRelease, 2> stale_tail_replicas{};
            std::size_t stale_tail_count           = 0;
            const auto preflight_inactive_truncate = [&](KVAddressSpaceStore& addresses,
                                                         LogicalKVPageStore& pages,
                                                         KVAddressSpaceHandle address,
                                                         std::optional<std::uint32_t> frontier) {
                if (!frontier ||
                    (addresses.committed_frontier(address) == *frontier &&
                     addresses.mapped_pages(address) == kv_pages_for_frontier(*frontier))) {
                    return;
                }
                bool releases_tail               = false;
                const std::uint32_t target_pages = kv_pages_for_frontier(*frontier);
                if (target_pages != 0) {
                    const LogicalKVPageHandle tail =
                        addresses.logical_page(address, target_pages - 1U);
                    const std::uint32_t columns =
                        *frontier -
                        (target_pages - 1U) * static_cast<std::uint32_t>(kPagedKVPageSize);
                    if (columns != pages.committed_columns(tail) && pages.host_resident(tail)) {
                        if (host_kv_extents == nullptr ||
                            stale_tail_count == stale_tail_replicas.size()) {
                            throw std::logic_error("stale Host KV tail replica is not releasable");
                        }
                        stale_tail_replicas[stale_tail_count++] =
                            HostKVPageReplicaRelease{.pages = &pages, .page = tail};
                        releases_tail = true;
                    }
                }
                if (!addresses.can_destructive_truncate_inactive(address, *frontier,
                                                                 releases_tail)) {
                    throw std::logic_error(
                        "selected private KV frontier is not destructively materializable");
                }
            };
            if (!sequence.kv) {
                throw std::logic_error("materialization destination has no KV address space");
            }
            if (!text_prefix_fork) {
                preflight_inactive_truncate(*text_kv_addresses, *text_kv_pages, sequence.kv->text,
                                            transaction.text_activation_frontier);
            }
            if (sequence.kv->backend && !backend_prefix_fork) {
                preflight_inactive_truncate(*backend_kv_addresses, *backend_kv_pages,
                                            *sequence.kv->backend,
                                            transaction.backend_activation_frontier);
            }
            if (stale_tail_count != 0) {
                const std::span<const HostKVPageReplicaRelease> releases(stale_tail_replicas.data(),
                                                                         stale_tail_count);
                if (!host_kv_extents->release_page_replicas(releases)) {
                    throw std::logic_error(
                        "stale Host KV tail replicas cannot be released atomically");
                }
            }
            if (!text_prefix_fork && transaction.text_activation_frontier &&
                (text_kv_addresses->committed_frontier(sequence.kv->text) !=
                     *transaction.text_activation_frontier ||
                 text_kv_addresses->mapped_pages(sequence.kv->text) !=
                     kv_pages_for_frontier(*transaction.text_activation_frontier))) {
                text_kv_addresses->destructive_truncate_inactive(
                    sequence.kv->text, *transaction.text_activation_frontier);
            }
            if (!backend_prefix_fork && transaction.backend_activation_frontier &&
                sequence.kv->backend &&
                (backend_kv_addresses->committed_frontier(*sequence.kv->backend) !=
                     *transaction.backend_activation_frontier ||
                 backend_kv_addresses->mapped_pages(*sequence.kv->backend) !=
                     kv_pages_for_frontier(*transaction.backend_activation_frontier))) {
                backend_kv_addresses->destructive_truncate_inactive(
                    *sequence.kv->backend, *transaction.backend_activation_frontier);
            }
            if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
        }
        if ((text_prefix_fork || backend_prefix_fork) && !transaction.prefix_forks_ready) {
            throw std::logic_error("materialization prefix forks are incomplete");
        }
        if (text_prefix_fork) {
            text_kv_addresses->commit_prefix_fork(std::move(*transaction.text_prefix_fork),
                                                  device.stream);
            transaction.text_prefix_fork.reset();
            if (!preserving_source) {
                const KVAddressSpaceHandle source_address = sequence.kv->text;
                sequence.kv->text                         = *transaction.root_text_address;
                transaction.root_text_address.reset();
                if (!text_kv_addresses->release(source_address)) {
                    throw std::logic_error("consumed Text KV source remained pinned after COW");
                }
            }
        } else {
            text_kv_addresses->commit_activation(std::move(*transaction.text_activation),
                                                 device.stream);
            transaction.text_activation.reset();
        }
        if (backend_prefix_fork) {
            backend_kv_addresses->commit_prefix_fork(std::move(*transaction.backend_prefix_fork),
                                                     device.stream);
            transaction.backend_prefix_fork.reset();
            if (!preserving_source) {
                const KVAddressSpaceHandle source_address = *sequence.kv->backend;
                sequence.kv->backend                      = *transaction.root_backend_address;
                transaction.root_backend_address.reset();
                if (!backend_kv_addresses->release(source_address)) {
                    throw std::logic_error("consumed Backend KV source remained pinned after COW");
                }
            }
        } else if (transaction.backend_activation) {
            backend_kv_addresses->commit_activation(std::move(*transaction.backend_activation),
                                                    device.stream);
            transaction.backend_activation.reset();
        }
        transaction.prefix_forks_ready = false;
        transaction.text_activation_frontier.reset();
        transaction.backend_activation_frontier.reset();
        transaction.prepared = false;

        const bool preserve_rewrite =
            request_plan.rewrite_disposition == RewriteCheckpointDisposition::RetainExisting;
        const auto activate_consumed_state = [&](StateImageHandle selected) {
            if (!request_plan.state_fork_required) {
                if (transaction.state_fork_destination ||
                    state_store->checkpoint_references(selected) != 0) {
                    throw std::logic_error("planned StateImage Move is no longer valid");
                }
                state_store->move_checkpoint_to_active(selected);
                sequence.state = ActiveStateBinding{.read = selected, .write = selected};
                return;
            }
            if (!transaction.state_fork_destination ||
                state_store->checkpoint_references(selected) == 0) {
                throw std::logic_error("planned StateImage Fork is no longer valid");
            }
            const StateImageHandle destination = *transaction.state_fork_destination;
            if (transaction.state_restored) {
                if (state_store->role(destination) != StateImageRole::ActiveMutable) {
                    throw std::logic_error("restored StateImage Fork destination is unavailable");
                }
                sequence.state = ActiveStateBinding{.read = destination, .write = destination};
            } else {
                const StateImageSelectors selectors =
                    state_store->begin_fork(selected, destination);
                if (speculative_backend == SpeculativeBackend::DFlash) {
                    state_images->copy_dflash_local(selectors.source, selectors.destination,
                                                    device.stream);
                }
                sequence.state = ActiveStateBinding{
                    .read = selected, .write = destination, .fork_pending = true};
                sequence.state_source_retained = true;
            }
            transaction.state_fork_destination.reset();
        };
        if (request_plan.reuse == ReusePath::Root) {
            sequence.rewrite_checkpoint = {};
            ordered_reset(sequence);
            sequence.ledger.clear();
            sequence.prefix_digests.clear();
            sequence.text_kv_valid = 0;
            sequence.mtp_kv_valid  = 0;
        } else if (preserving_source) {
            const SequenceState* private_source =
                transaction.has_source ? &continuation_states[transaction.source_index] : nullptr;
            SharedPrefixState* shared_source =
                transaction.has_shared_source
                    ? &shared_prefix_states[transaction.shared_source_index]
                    : nullptr;
            const std::uint32_t source_text_frontier =
                private_source != nullptr ? private_source->text_kv_valid : shared_source->frontier;
            if (!sequence.kv || source_text_frontier < base) {
                throw std::logic_error("retained prefix has incomplete Text KV");
            }
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base       = base == 0 ? 0 : base - 1U;
                const std::uint32_t source_backend = private_source != nullptr
                                                         ? private_source->mtp_kv_valid
                                                         : shared_source->backend_frontier;
                if (!request_plan.prepare_mtp || source_backend < mtp_base) {
                    throw std::logic_error("retained prefix has incomplete MTP KV");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash) {
                const std::uint32_t source_backend = private_source != nullptr
                                                         ? private_source->dflash_context_frontier
                                                         : shared_source->backend_frontier;
                if (source_backend < base) {
                    throw std::logic_error("retained prefix has incomplete DFlash KV");
                }
                sequence.dflash_context_frontier = base;
            }
            sequence.tail_hidden_valid =
                base == prompt_tokens &&
                (private_source != nullptr ? private_source->tail_hidden_valid
                                           : shared_source->tail_hidden_valid);
            if (shared_source != nullptr) {
                if (shared_source->active_references == std::numeric_limits<std::uint32_t>::max()) {
                    throw std::overflow_error("shared-prefix active reference overflow");
                }
                ++shared_source->active_references;
                sequence.shared_prefix_references.push_back(transaction.shared_source_index);
            }
            refresh_state_views(sequence);
            bind_sequence_kv(sequence);
        } else if (request_plan.reuse == ReusePath::PrivateEndpoint) {
            if (!state_store->valid(sequence.state.read) ||
                sequence.state.read != sequence.state.write || sequence.state.fork_pending ||
                state_store->role(sequence.state.read) != StateImageRole::CheckpointImmutable) {
                throw std::logic_error("resident endpoint StateImage is not movable");
            }
            if (!preserve_rewrite && sequence.rewrite_state) {
                const StateImageHandle dropped = *sequence.rewrite_state;
                state_store->release_checkpoint_reference(dropped);
                sequence.rewrite_state.reset();
                sequence.rewrite_checkpoint = {};
                if (dropped != sequence.state.read &&
                    state_store->checkpoint_references(dropped) == 0 &&
                    !state_store->release(dropped)) {
                    throw std::logic_error("dropped rewrite StateImage could not be released");
                }
            }
            activate_consumed_state(sequence.state.read);
            if (!sequence.kv) {
                throw std::logic_error("resident prefix has no KV allocation bundle");
            }
            if (sequence.text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than the append frontier");
            }
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error("resident MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash &&
                       sequence.dflash_context_frontier != base) {
                throw std::logic_error("resident DFlash context is not at the append frontier");
            }
            bind_sequence_kv(sequence);
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            sequence.text_kv_valid = base;
            sequence.ledger.resize(base);
            sequence.prefix_digests.truncate(base);
            reserve_state_entitlement(sequence, state_slots);
            refresh_state_views(sequence);
        } else if (is_rewrite_checkpoint_restore(request_plan.reuse)) {
            if (!sequence.kv || sequence.text_kv_valid < base) {
                throw std::logic_error("resident rewrite checkpoint has no complete KV allocation");
            }
            if (!sequence.rewrite_state || !state_store->valid(*sequence.rewrite_state) ||
                state_store->role(*sequence.rewrite_state) != StateImageRole::CheckpointImmutable ||
                (sequence.endpoint_valid &&
                 (!state_store->valid(sequence.state.read) ||
                  sequence.state.read != sequence.state.write || sequence.state.fork_pending ||
                  state_store->role(sequence.state.read) != StateImageRole::CheckpointImmutable))) {
                throw std::logic_error("resident rewrite StateImage is not movable");
            }
            const StateImageHandle checkpoint = *sequence.rewrite_state;
            if (sequence.endpoint_valid && sequence.state.read == checkpoint) {
                throw std::logic_error("resident endpoint aliases its rewrite StateImage");
            }
            if (sequence.endpoint_valid && !state_store->release(sequence.state.read)) {
                throw std::logic_error("superseded endpoint StateImage could not be released");
            }
            if (!preserve_rewrite) {
                state_store->release_checkpoint_reference(checkpoint);
                sequence.rewrite_state.reset();
                sequence.rewrite_checkpoint = {};
            }
            activate_consumed_state(checkpoint);
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error(
                        "rewrite-checkpoint MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash) {
                if (!dflash || !sequence.kv->backend || sequence.dflash_context_frontier < base) {
                    throw std::logic_error("planned DFlash rewrite checkpoint is unavailable");
                }
                sequence.dflash_context_frontier = base;
            }
            bind_sequence_kv(sequence);
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            sequence.tail_hidden_valid = base == prompt_tokens;
            sequence.ledger.resize(base);
            sequence.prefix_digests.truncate(base);
            reserve_state_entitlement(sequence, state_slots);
            refresh_state_views(sequence);
        } else {
            throw std::logic_error("request plan has an invalid prefix reuse path");
        }

        sequence.endpoint_valid = false;
        if (!preserving_source) { trim_sequence_kv(sequence, base, backend_kv_valid(sequence)); }
        bind_sequence_kv(sequence);
        const std::uint32_t backend_materialized =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min(capacity,
                           prompt_tokens + (initial_mtp_extent == 0 ? 0U : initial_mtp_extent - 1U))
            : speculative_backend == SpeculativeBackend::DFlash ? prompt_tokens
                                                                : 0U;
        materialize_sequence_kv(sequence, prompt_tokens, backend_materialized);
        install_sampling(sequence, request, request_plan.sampling);
        sequence.rope_delta = staged.prompt.rope_delta;
        set_device_i32(io.rope_delta, sequence.rope_delta);

        request.timings              = {};
        request.pending              = {};
        request.publish_continuation = request_plan.summary.publish_continuation;
        sequence.mtp_draft_count     = 0;
        sequence.tail_hidden_valid   = base == prompt_tokens && sequence.tail_hidden_valid;
        sequence.ledger.swap(materialization_ledger_);
        sequence.prefix_identity.swap(materialization_identity_);
        sequence.prefix_digests.swap(materialization_prefix_digests_);
        sequence.rebuild_work       = request_plan.root_rebuild_work;
        sequence.rebuild_tail_begin = request_plan.root_rebuild_tail_begin;

        if (speculative_backend == SpeculativeBackend::DFlash) {
            if (!dflash || !io.dflash_decode || !sequence.kv->backend) {
                throw std::logic_error("DFlash prefill state is incomplete");
            }
            *dflash_host_ingress                       = {};
            dflash_host_ingress->active_lanes[0]       = static_cast<std::int32_t>(sequence.lane);
            const StateImageSelectors selectors        = state_selectors(sequence);
            dflash_host_ingress->state_source_slots[0] = selectors.source;
            dflash_host_ingress->state_destination_slots[0] = selectors.destination;
            dflash_host_ingress->dflash_kv_table_rows[0] =
                backend_kv_addresses->bound_row(*sequence.kv->backend);
            CUDA_CHECK(cudaMemcpyAsync(io.dflash_decode->ingress.data, dflash_host_ingress,
                                       sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                       device.stream));
        }

        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        request.lifecycle = Lifecycle::Prefilling;
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        clear_lane_best_effort(sequence, request);
        throw;
    }
}

runtime::PrefillStepResult
ProgramImplCore::advance_prefill_raw(std::uint32_t lane, runtime::ExecutionTiming* failed_timing) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    return advance_prefill(active_sequence(lane), requests[lane], failed_timing);
}

runtime::ExecutionTiming
ProgramImplCore::resolve_prefill_raw(std::uint32_t lane, bool terminal,
                                     runtime::ExecutionTiming* failed_timing) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (requests[lane].pending.kind != PendingKind::Begin) {
        throw std::logic_error("prefill resolution requires a pending prefill token");
    }
    return resolve_non_speculative_pending(active_sequence(lane), requests[lane], 1, terminal,
                                           failed_timing);
}

runtime::ExecutionTiming ProgramImplCore::resolve_pending_raw(
    std::span<const std::uint32_t> lanes, std::span<const std::uint32_t> accepted_tokens,
    std::span<const std::uint8_t> terminal, std::span<const std::uint8_t> cancelled,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    if (lanes.empty() || lanes.size() > max_concurrency || accepted_tokens.size() != lanes.size() ||
        terminal.size() != lanes.size() || cancelled.size() != lanes.size()) {
        throw std::invalid_argument("pending batch resolution has inconsistent membership");
    }

    if (lanes.size() == 1 && lanes.front() < max_concurrency &&
        requests[lanes.front()].pending.kind == PendingKind::Begin) {
        const std::uint32_t lane = lanes.front();
        if (requests[lane].lifecycle != Lifecycle::Pending) {
            throw std::logic_error("prefill pending token no longer matches Program state");
        }
        if (cancelled.front()) {
            if (accepted_tokens.front() != 0 || !terminal.front()) {
                throw std::logic_error("cancelled prefill pending decision is invalid");
            }
            if (!clear_lane_strict(active_sequence(lane), requests[lane])) {
                throw std::logic_error("cancelled prefill lane is not strictly releasable");
            }
        } else {
            timing.pause();
            timing.include(resolve_non_speculative_pending(active_sequence(lane), requests[lane],
                                                           accepted_tokens.front(),
                                                           terminal.front() != 0, failed_timing));
            timing.resume_post();
        }
        return timing.finish();
    }

    if (speculative_backend == SpeculativeBackend::None) {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
                requests[lane].pending.kind != PendingKind::Ordinary) {
                throw std::logic_error("ordinary pending batch no longer matches Program state");
            }
            if (cancelled[row]) {
                if (!clear_lane_strict(active_sequence(lane), requests[lane])) {
                    throw std::logic_error("cancelled decode lane is not strictly releasable");
                }
            } else {
                timing.pause();
                timing.include(resolve_non_speculative_pending(active_sequence(lane),
                                                               requests[lane], accepted_tokens[row],
                                                               terminal[row] != 0, failed_timing));
                timing.resume_post();
            }
        }
        return timing.finish();
    }

    if (!replay_fold) {
        throw std::logic_error("speculative pending batch has no ReplaySSM records");
    }

    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold_rows{};
    std::array<std::int32_t, kMaximumConcurrency> hidden_selectors{};
    bool needs_hidden_correction = false;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
            requests[lane].pending.kind != PendingKind::Speculative) {
            throw std::logic_error("speculative pending batch no longer matches Program state");
        }
        const PendingCandidate& pending = requests[lane].pending;
        const SequenceState& sequence   = active_sequence(lane);
        if (sequence.execution_frontier != pending.base_E ||
            sequence.ledger_frontier != pending.base_S ||
            sequence.ledger.size() != pending.base_S ||
            sequence.prefix_identity.size() != pending.base_S ||
            sequence.prefix_digests.size() != pending.base_S ||
            sequence.text_kv_valid != pending.base_E ||
            (speculative_backend == SpeculativeBackend::Mtp &&
             sequence.mtp_kv_valid != pending.base_E) ||
            (speculative_backend == SpeculativeBackend::DFlash &&
             sequence.dflash_context_frontier != pending.base_E)) {
            throw std::logic_error("speculative pending row is not at its recorded base");
        }
        const std::uint32_t committed = cancelled[row] ? 0U : accepted_tokens[row];
        if ((cancelled[row] && accepted_tokens[row] != 0) ||
            (!cancelled[row] && (committed == 0 || committed > pending.produced ||
                                 (!terminal[row] && committed != pending.produced)))) {
            throw std::logic_error("speculative pending row has an invalid committed prefix");
        }
        const StateImageSelectors selectors = state_selectors(sequence);
        fold_rows[row] =
            ops::GdnReplayFoldRow{.source_state_slot      = selectors.source,
                                  .destination_state_slot = selectors.destination,
                                  .commit_columns         = static_cast<std::int32_t>(committed)};
        const bool partial_terminal =
            !cancelled[row] && terminal[row] && committed < pending.produced;
        hidden_selectors[row] =
            static_cast<std::int32_t>(partial_terminal ? committed - 1U : pending.produced - 1U);
        needs_hidden_correction = needs_hidden_correction || partial_terminal;
    }

    const auto tail_started = Clock::now();
    try {
        timing.resume_submit();
        replay_fold->execute(std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), lanes.size()),
                             device.stream);

        if (needs_hidden_correction) {
            const auto batch = static_cast<std::int32_t>(lanes.size());
            Tensor selector_tensor;
            Tensor hidden;
            Tensor selected;
            Tensor destinations;
            if (speculative_backend == SpeculativeBackend::Mtp && io.mtp_decode) {
                qwen3_6::MtpDecodeState& frame = *io.mtp_decode;
                selector_tensor                = frame.current_extents.slice(0, 0, batch);
                hidden                         = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.state_destination_slots.slice(0, 0, batch);
            } else if (speculative_backend == SpeculativeBackend::DFlash && io.dflash_decode) {
                qwen3_6::DFlashDecodeState& frame = *io.dflash_decode;
                selector_tensor                   = frame.proposal_extents.slice(0, 0, batch);
                hidden                            = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.state_destination_slots.slice(0, 0, batch);
            } else {
                throw std::logic_error("partial speculative commit has no target frame");
            }
            CUDA_CHECK(cudaMemcpyAsync(selector_tensor.data, hidden_selectors.data(),
                                       lanes.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       device.stream));
            ops::speculative_select_accepted_hidden(hidden, selector_tensor, selected,
                                                    device.stream);
            ops::scatter(selected, destinations, state_images->continuation_hidden_store(),
                         device.stream);
        }

        if (speculative_backend == SpeculativeBackend::DFlash) {
            std::array<std::uint32_t, kMaximumConcurrency> append_lanes{};
            std::array<std::uint32_t, kMaximumConcurrency> append_starts{};
            std::array<std::uint32_t, kMaximumConcurrency> append_counts{};
            std::size_t append_size = 0;
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                if (!cancelled[row] && terminal[row]) {
                    append_lanes[append_size]  = lanes[row];
                    append_starts[append_size] = requests[lanes[row]].pending.base_E;
                    append_counts[append_size] = accepted_tokens[row];
                    ++append_size;
                }
            }
            if (append_size != 0) {
                enqueue_dflash_context_append(
                    std::span<const std::uint32_t>(append_lanes.data(), append_size),
                    std::span<const std::uint32_t>(append_starts.data(), append_size),
                    std::span<const std::uint32_t>(append_counts.data(), append_size));
            }
        }

        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        work.reset();
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        work.reset();
        clear_execution_failure_lanes(lanes);
        throw;
    }

    const double tail_seconds = std::chrono::duration<double>(Clock::now() - tail_started).count();
    const std::uint32_t width = draft_window + 1U;
    try {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = active_sequence(lanes[row]);
            RequestControl& request = requests[lanes[row]];
            if (cancelled[row]) {
                if (!clear_lane_strict(sequence, request)) {
                    throw std::logic_error("cancelled speculative lane is not strictly releasable");
                }
                continue;
            }

            const PendingCandidate pending = request.pending;
            const std::uint32_t committed  = accepted_tokens[row];
            settle_state_fork(sequence);
            const TokenId* token_base =
                speculative_backend == SpeculativeBackend::Mtp
                    ? mtp_host_egress->licensed_tokens.data() + row * width
                    : dflash_host_egress->licensed_tokens.data() + row * width;
            sequence.ledger.insert(sequence.ledger.end(), token_base, token_base + committed);
            sequence.prefix_identity.append_generated(committed, sequence.rope_delta);
            sequence.prefix_digests.append_generated(
                std::span<const TokenId>(token_base, committed), sequence.rope_delta);
            advance_rebuild_work(sequence, pending.base_E + committed, prefill_chunk);
            sequence.execution_frontier = pending.base_E + committed;
            sequence.ledger_frontier    = pending.base_S + committed;
            sequence.text_kv_valid      = sequence.execution_frontier;
            sequence.tail_hidden_valid  = true;

            if (speculative_backend == SpeculativeBackend::Mtp) {
                sequence.mtp_kv_valid = sequence.execution_frontier;
                if (terminal[row]) {
                    sequence.mtp_draft_count = 0;
                } else {
                    const std::int32_t next  = mtp_host_egress->next_extents[row];
                    sequence.mtp_draft_count = static_cast<std::uint32_t>(next);
                    for (std::uint32_t step = 0; step < sequence.mtp_draft_count; ++step) {
                        sequence.mtp_drafts[step] =
                            mtp_host_egress->next_drafts[step * max_concurrency + row];
                    }
                }
            } else {
                sequence.dflash_context_frontier =
                    terminal[row] ? sequence.execution_frontier : pending.base_E;
            }

            commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            if (terminal[row]) {
                request.lifecycle = Lifecycle::Finishable;
            } else {
                request.lifecycle = Lifecycle::Active;
            }
            request.pending = {};
            request.timings.decode_seconds += tail_seconds;
        }
    } catch (...) {
        clear_execution_failure_lanes(lanes);
        throw;
    }
    return timing.finish();
}

bool ProgramImplCore::can_clear_lane_strict(const SequenceState& sequence) const {
    const auto* begin = continuation_states.data();
    const auto* end   = begin + continuation_capacity;
    if (&sequence < begin || &sequence >= end || !state_store || !text_kv_addresses ||
        !text_kv_pages || !sequence.kv) {
        return false;
    }
    const std::uint32_t continuation = static_cast<std::uint32_t>(&sequence - begin);
    if (continuation_slots[continuation].role != ContinuationSlotRole::Active ||
        !text_kv_addresses->can_release_after_deactivate(sequence.kv->text) ||
        (sequence.kv->backend &&
         (!backend_kv_addresses || !backend_kv_pages ||
          !backend_kv_addresses->can_release_after_deactivate(*sequence.kv->backend)))) {
        return false;
    }

    for (std::size_t position = 0; position < sequence.shared_prefix_references.size();
         ++position) {
        const std::uint32_t index = sequence.shared_prefix_references[position];
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued) {
            return false;
        }
        const std::uint32_t required = static_cast<std::uint32_t>(std::count(
            sequence.shared_prefix_references.begin(),
            sequence.shared_prefix_references.begin() + static_cast<std::ptrdiff_t>(position + 1U),
            index));
        if (shared_prefix_states[index].active_references < required) { return false; }
    }

    if (!state_store->valid(sequence.state.read) || !state_store->valid(sequence.state.write) ||
        (sequence.state.fork_pending &&
         (!sequence.state_source_retained ||
          !state_store->can_abort_fork(sequence.state.read, sequence.state.write)))) {
        return false;
    }
    const auto validate_state = [&](StateImageHandle handle, bool release_object,
                                    bool fork_destination = false) {
        if (!state_store->valid(handle)) { return false; }
        const std::uint32_t owned = owned_checkpoint_references(sequence, handle);
        const std::uint32_t total = state_store->checkpoint_references(handle);
        if (owned > total ||
            (owned != 0 && state_store->role(handle) != StateImageRole::CheckpointImmutable &&
             !fork_destination)) {
            return false;
        }
        if (!release_object || total != owned) { return true; }
        if (fork_destination) {
            return state_store->can_release_after_fork_abort(sequence.state.read,
                                                             sequence.state.write, owned);
        }
        return state_store->can_release_after_checkpoint_references(handle, owned);
    };
    const auto duplicates_binding = [&](StateImageHandle handle) {
        return handle == sequence.state.read || handle == sequence.state.write;
    };

    if (!validate_state(sequence.state.read, !sequence.state_source_retained ||
                                                 sequence.state.read == sequence.state.write)) {
        return false;
    }
    if (sequence.state.write != sequence.state.read &&
        !validate_state(sequence.state.write, true, sequence.state.fork_pending)) {
        return false;
    }
    if (sequence.rewrite_state && !duplicates_binding(*sequence.rewrite_state) &&
        !validate_state(*sequence.rewrite_state, true)) {
        return false;
    }
    for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
        const StateImageHandle handle = sequence.long_anchors[index].state;
        bool repeated                 = duplicates_binding(handle) ||
                        (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (std::size_t prior = 0; !repeated && prior < index; ++prior) {
            repeated = sequence.long_anchors[prior].state == handle;
        }
        if (!repeated && !validate_state(handle, true)) { return false; }
    }
    if (sequence.reserved_state) {
        const StateImageHandle handle = *sequence.reserved_state;
        bool repeated                 = duplicates_binding(handle) ||
                        (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            repeated = repeated || anchor.state == handle;
        }
        if (!repeated && !validate_state(handle, true)) { return false; }
    }
    return true;
}

void ProgramImplCore::release_active_shared_references_strict(SequenceState& sequence) noexcept {
    for (const std::uint32_t index : sequence.shared_prefix_references) {
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
            shared_prefix_states[index].active_references == 0) {
            std::terminate();
        }
        --shared_prefix_states[index].active_references;
    }
    sequence.shared_prefix_references.clear();
}

bool ProgramImplCore::clear_lane_strict(SequenceState& sequence, RequestControl& request) noexcept {
    try {
        if (!can_clear_lane_strict(sequence)) { return false; }
    } catch (...) { return false; }
    const auto* begin                = continuation_states.data();
    const std::uint32_t continuation = static_cast<std::uint32_t>(&sequence - begin);
    release_active_shared_references_strict(sequence);
    release_active_sequence_kv_strict(sequence);
    release_active_sequence_state_strict(sequence);
    retire_continuation_slot(continuation);
    request.prefill.reset();
    request.lifecycle            = Lifecycle::Empty;
    request.pending              = {};
    request.active_resources     = {};
    request.optional_resources   = {};
    request.publish_continuation = true;
    return true;
}

void ProgramImplCore::clear_execution_failure_lanes(std::span<const std::uint32_t> lanes) noexcept {
    // A concurrent resource transaction may pin or inspect these active owners. Engine-wide
    // cleanup aborts that transaction before releasing lanes, preserving the only safe order.
    if (has_context_transaction()) { return; }
    for (const std::uint32_t lane : lanes) {
        if (lane >= max_concurrency || active_continuations[lane] >= continuation_capacity) {
            continue;
        }
        clear_lane_best_effort(active_sequence(lane), requests[lane]);
        invalidate_lane(lane);
    }
}

void ProgramImplCore::clear_lane_best_effort(SequenceState& sequence,
                                             RequestControl& request) noexcept {
    request.prefill.reset();
    request.lifecycle            = Lifecycle::Empty;
    request.pending              = {};
    request.active_resources     = {};
    request.optional_resources   = {};
    request.publish_continuation = true;
    const auto* begin            = continuation_states.data();
    const auto* end              = begin + continuation_capacity;
    if (&sequence >= begin && &sequence < end) {
        release_continuation_slot_best_effort(static_cast<std::uint32_t>(&sequence - begin));
    }
}

StateImageSelectors ProgramImplCore::state_selectors(const SequenceState& sequence) const {
    if (!state_store || !state_store->valid(sequence.state.read) ||
        !state_store->valid(sequence.state.write)) {
        throw std::logic_error("sequence has no active StateImage binding");
    }
    return state_store->selectors(sequence.state.read, sequence.state.write);
}

std::uint32_t ProgramImplCore::state_footprint(const SequenceState& sequence) const noexcept {
    if (!state_store) { return 0; }
    std::array<StateImageHandle, 4> unique{};
    std::uint32_t count = 0;
    const auto add      = [&](StateImageHandle handle) {
        if (!state_store->valid(handle)) { return; }
        const StateReplicaResidency residency = state_store->residency(handle);
        if (residency != StateReplicaResidency::DeviceOnly &&
            residency != StateReplicaResidency::Both) {
            return;
        }
        for (std::uint32_t index = 0; index < count; ++index) {
            if (unique[index] == handle) { return; }
        }
        unique[count++] = handle;
    };
    add(sequence.state.read);
    add(sequence.state.write);
    if (sequence.rewrite_state) { add(*sequence.rewrite_state); }
    if (sequence.reserved_state) { add(*sequence.reserved_state); }
    for (std::size_t anchor_index = 0; anchor_index < sequence.long_anchors.size();
         ++anchor_index) {
        const StateImageHandle handle = sequence.long_anchors[anchor_index].state;
        if (!state_store->valid(handle)) { continue; }
        const StateReplicaResidency residency = state_store->residency(handle);
        if (residency != StateReplicaResidency::DeviceOnly &&
            residency != StateReplicaResidency::Both) {
            continue;
        }
        bool seen = false;
        for (std::uint32_t index = 0; index < std::min<std::uint32_t>(count, unique.size());
             ++index) {
            if (unique[index] == handle) { seen = true; }
        }
        for (std::size_t prior = 0; !seen && prior < anchor_index; ++prior) {
            if (sequence.long_anchors[prior].state == handle) { seen = true; }
        }
        if (!seen) { ++count; }
    }
    return count;
}

std::uint32_t ProgramImplCore::owned_checkpoint_references(const SequenceState& sequence,
                                                           StateImageHandle state) const noexcept {
    std::uint32_t references = 0;
    if (sequence.rewrite_state && *sequence.rewrite_state == state) { ++references; }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        if (anchor.state == state) { ++references; }
    }
    return references;
}

bool ProgramImplCore::state_exclusive_to_sequence(const SequenceState& sequence,
                                                  StateImageHandle state) const noexcept {
    if (!state_store || !state_store->valid(state)) { return false; }
    return state_store->checkpoint_references(state) ==
           owned_checkpoint_references(sequence, state);
}

void ProgramImplCore::refresh_state_views(SequenceState& sequence) {
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
    if (state_store->valid(sequence.state.read) && state_store->valid(sequence.state.write) &&
        state_store->residency(sequence.state.read) != StateReplicaResidency::HostOnly &&
        state_store->residency(sequence.state.write) != StateReplicaResidency::HostOnly) {
        const StateImageHandle committed =
            sequence.state.fork_pending ? sequence.state.read : sequence.state.write;
        sequence.tail_hidden =
            state_images->continuation_hidden_slot(state_store->physical_slot(committed));
    }
    if (sequence.rewrite_state && state_store->valid(*sequence.rewrite_state) &&
        state_store->residency(*sequence.rewrite_state) != StateReplicaResidency::HostOnly) {
        sequence.rewrite_checkpoint_hidden = state_images->continuation_hidden_slot(
            state_store->physical_slot(*sequence.rewrite_state));
    }
}

void ProgramImplCore::reserve_state_entitlement(SequenceState& sequence, std::uint32_t slots) {
    const std::uint32_t footprint = state_footprint(sequence);
    if (slots == 0 || footprint > slots) {
        throw std::logic_error("sequence StateImage entitlement is inconsistent");
    }
    if (footprint == slots) { return; }
    if (slots - footprint != 1 || sequence.reserved_state) {
        throw std::logic_error("sequence StateImage reservation is not a single destination");
    }
    std::optional<StateImageHandle> reserved = state_store->reserve_destination();
    if (!reserved) { throw std::bad_alloc(); }
    sequence.reserved_state = *reserved;
    if (state_footprint(sequence) != slots) {
        throw std::logic_error("sequence StateImage entitlement did not materialize exactly");
    }
}

void ProgramImplCore::settle_state_fork(SequenceState& sequence) {
    if (!sequence.state.fork_pending) { return; }
    if (has_context_transaction()) {
        throw std::logic_error("StateImage Fork settlement overlaps a resource transaction");
    }
    const StateImageHandle source      = sequence.state.read;
    const StateImageHandle destination = sequence.state.write;
    state_store->commit_fork(source, destination);
    sequence.state.read         = destination;
    sequence.state.write        = destination;
    sequence.state.fork_pending = false;
    if (!sequence.state_source_retained && state_store->checkpoint_references(source) == 0 &&
        !state_store->release(source)) {
        throw std::logic_error("unreferenced StateImage fork source could not be released");
    }
    sequence.state_source_retained = false;
    refresh_state_views(sequence);
}

bool ProgramImplCore::has_unsettled_state_fork() const noexcept {
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        const std::uint32_t continuation = active_continuations[lane];
        if (continuation < continuation_capacity &&
            continuation_states[continuation].state.fork_pending) {
            return true;
        }
    }
    return false;
}

void ProgramImplCore::release_active_sequence_state_strict(SequenceState& sequence) noexcept {
    const auto fail = []() noexcept { std::terminate(); };
    if (!state_store) { fail(); }
    try {
        if (sequence.state.fork_pending) {
            state_store->abort_fork(sequence.state.read, sequence.state.write);
            sequence.state.fork_pending = false;
        }
        if (sequence.rewrite_state) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            state_store->release_checkpoint_reference(anchor.state);
        }

        const auto release_if_unreferenced = [&](StateImageHandle handle, bool lifetime_owned) {
            if (!lifetime_owned || !state_store->valid(handle) ||
                state_store->checkpoint_references(handle) != 0) {
                return;
            }
            if (!state_store->release(handle)) { fail(); }
        };
        const auto duplicates_binding = [&](StateImageHandle handle) {
            return handle == sequence.state.read || handle == sequence.state.write;
        };

        release_if_unreferenced(sequence.state.write, true);
        if (sequence.state.read != sequence.state.write) {
            release_if_unreferenced(sequence.state.read, !sequence.state_source_retained);
        }
        if (sequence.rewrite_state) {
            release_if_unreferenced(*sequence.rewrite_state,
                                    !duplicates_binding(*sequence.rewrite_state));
        }
        for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
            const StateImageHandle handle = sequence.long_anchors[index].state;
            bool repeated                 = duplicates_binding(handle) ||
                            (sequence.rewrite_state && handle == *sequence.rewrite_state);
            for (std::size_t prior = 0; !repeated && prior < index; ++prior) {
                repeated = sequence.long_anchors[prior].state == handle;
            }
            release_if_unreferenced(handle, !repeated);
        }
        if (sequence.reserved_state) {
            const StateImageHandle handle = *sequence.reserved_state;
            bool repeated                 = duplicates_binding(handle) ||
                            (sequence.rewrite_state && handle == *sequence.rewrite_state);
            for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
                repeated = repeated || anchor.state == handle;
            }
            release_if_unreferenced(handle, !repeated);
        }
    } catch (...) { fail(); }

    sequence.state          = {};
    sequence.rewrite_state  = std::nullopt;
    sequence.reserved_state = std::nullopt;
    sequence.endpoint_valid = false;
    sequence.long_anchors.clear();
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
    sequence.state_source_retained     = false;
}

void ProgramImplCore::release_sequence_state_strict(SequenceState& sequence) noexcept {
    const auto fail = []() noexcept { std::terminate(); };
    if (!state_store || sequence.state.fork_pending) { fail(); }

    try {
        if (sequence.rewrite_state) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            state_store->release_checkpoint_reference(anchor.state);
        }

        const auto release_if_unreferenced = [&](StateImageHandle handle, bool lifetime_owned) {
            if (!lifetime_owned || !state_store->valid(handle) ||
                state_store->checkpoint_references(handle) != 0) {
                return;
            }
            if (!state_store->release(handle)) { fail(); }
        };
        const auto repeated_before_anchor = [&](std::size_t anchor_index, StateImageHandle handle) {
            if ((sequence.endpoint_valid &&
                 (handle == sequence.state.read || handle == sequence.state.write)) ||
                (sequence.rewrite_state && handle == *sequence.rewrite_state)) {
                return true;
            }
            for (std::size_t prior = 0; prior < anchor_index; ++prior) {
                if (sequence.long_anchors[prior].state == handle) { return true; }
            }
            return false;
        };

        if (sequence.endpoint_valid) {
            release_if_unreferenced(sequence.state.write, true);
            if (sequence.state.read != sequence.state.write) {
                release_if_unreferenced(sequence.state.read, !sequence.state_source_retained);
            }
        }
        if (sequence.rewrite_state) {
            const StateImageHandle handle = *sequence.rewrite_state;
            const bool duplicates_endpoint =
                sequence.endpoint_valid &&
                (handle == sequence.state.read || handle == sequence.state.write);
            release_if_unreferenced(handle, !duplicates_endpoint);
        }
        for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
            const StateImageHandle handle = sequence.long_anchors[index].state;
            release_if_unreferenced(handle, !repeated_before_anchor(index, handle));
        }
        if (sequence.reserved_state) {
            const StateImageHandle handle = *sequence.reserved_state;
            bool repeated                 = sequence.endpoint_valid &&
                            (handle == sequence.state.read || handle == sequence.state.write);
            repeated = repeated || (sequence.rewrite_state && handle == *sequence.rewrite_state);
            for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
                repeated = repeated || anchor.state == handle;
            }
            release_if_unreferenced(handle, !repeated);
        }
    } catch (...) { fail(); }

    sequence.state          = {};
    sequence.rewrite_state  = std::nullopt;
    sequence.reserved_state = std::nullopt;
    sequence.endpoint_valid = false;
    sequence.long_anchors.clear();
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
    sequence.state_source_retained     = false;
}

void ProgramImplCore::release_sequence_state(SequenceState& sequence) noexcept {
    if (!state_store) { return; }
    if (sequence.state.fork_pending && state_store->valid(sequence.state.read) &&
        state_store->valid(sequence.state.write)) {
        try {
            state_store->abort_fork(sequence.state.read, sequence.state.write);
        } catch (...) {}
    }

    try {
        if (sequence.rewrite_state && state_store->valid(*sequence.rewrite_state) &&
            state_store->checkpoint_references(*sequence.rewrite_state) != 0) {
            state_store->release_checkpoint_reference(*sequence.rewrite_state);
        }
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            if (state_store->valid(anchor.state) &&
                state_store->checkpoint_references(anchor.state) != 0) {
                state_store->release_checkpoint_reference(anchor.state);
            }
        }
    } catch (...) {}

    const auto releasable = [&](StateImageHandle handle) { return state_store->valid(handle); };
    if (releasable(sequence.state.write)) { (void)state_store->release(sequence.state.write); }
    if (!sequence.state_source_retained && sequence.state.read != sequence.state.write &&
        releasable(sequence.state.read)) {
        (void)state_store->release(sequence.state.read);
    }
    if (sequence.rewrite_state) {
        const StateImageHandle handle = *sequence.rewrite_state;
        const bool duplicates_binding =
            handle == sequence.state.write ||
            (!sequence.state_source_retained && handle == sequence.state.read);
        if (!duplicates_binding && releasable(handle)) { (void)state_store->release(handle); }
    }
    for (std::size_t index = 0; index < sequence.long_anchors.size(); ++index) {
        const StateImageHandle handle = sequence.long_anchors[index].state;
        bool duplicate                = handle == sequence.state.write ||
                         (!sequence.state_source_retained && handle == sequence.state.read) ||
                         (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (std::size_t previous = 0; !duplicate && previous < index; ++previous) {
            duplicate = sequence.long_anchors[previous].state == handle;
        }
        if (!duplicate && releasable(handle)) { (void)state_store->release(handle); }
    }
    if (sequence.reserved_state) {
        const StateImageHandle handle = *sequence.reserved_state;
        bool duplicate                = handle == sequence.state.write ||
                         (!sequence.state_source_retained && handle == sequence.state.read) ||
                         (sequence.rewrite_state && handle == *sequence.rewrite_state);
        for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
            duplicate = duplicate || anchor.state == handle;
        }
        if (!duplicate && releasable(handle)) { (void)state_store->release(handle); }
    }
    sequence.state          = {};
    sequence.rewrite_state  = std::nullopt;
    sequence.reserved_state = std::nullopt;
    sequence.endpoint_valid = false;
    sequence.long_anchors.clear();
    sequence.tail_hidden               = {};
    sequence.rewrite_checkpoint_hidden = {};
    sequence.state_source_retained     = false;
}

void ProgramImplCore::release_active_shared_references(SequenceState& sequence) noexcept {
    for (const std::uint32_t index : sequence.shared_prefix_references) {
        if (index >= shared_prefix_capacity ||
            shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued ||
            shared_prefix_states[index].active_references == 0) {
            continue;
        }
        --shared_prefix_states[index].active_references;
    }
    sequence.shared_prefix_references.clear();
}

qwen3_6::PagedKVCache* ProgramImplCore::backend_kv_cache() noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash) { return &dflash->full; }
    return nullptr;
}

const qwen3_6::PagedKVCache* ProgramImplCore::backend_kv_cache() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash) { return &dflash->full; }
    return nullptr;
}

std::uint32_t ProgramImplCore::backend_kv_valid(const SequenceState& sequence) const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return sequence.mtp_kv_valid; }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        return sequence.dflash_context_frontier;
    }
    return 0;
}

void ProgramImplCore::resize_sequence_kv_entitlement(SequenceState& sequence,
                                                     std::uint32_t text_pages,
                                                     std::uint32_t backend_pages) {
    if (!sequence.kv || text_pages == 0 ||
        (sequence.kv->backend.has_value() != (backend_pages != 0))) {
        throw std::invalid_argument("KV resize entitlement does not match the sequence bundle");
    }
    text_kv_addresses->resize_entitlement(sequence.kv->text, text_pages);
    if (sequence.kv->backend) {
        backend_kv_addresses->resize_entitlement(*sequence.kv->backend, backend_pages);
    }
}

void ProgramImplCore::bind_sequence_kv(SequenceState& sequence) {
    if (!sequence.kv) { throw std::logic_error("KV allocation bundle is unavailable"); }
    const std::int32_t row = static_cast<std::int32_t>(sequence.lane);
    const bool text_active = text_kv_addresses->active(sequence.kv->text);
    const bool backend_active =
        sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend);
    if (sequence.kv->backend && text_active != backend_active) {
        throw std::logic_error("KV address-space activation is not bundle-atomic");
    }
    try {
        if (!text_active) {
            text_kv_addresses->activate(sequence.kv->text,
                                        text_kv_addresses->mapped_pages(sequence.kv->text), row);
            if (sequence.kv->backend) {
                backend_kv_addresses->activate(
                    *sequence.kv->backend,
                    backend_kv_addresses->mapped_pages(*sequence.kv->backend), row);
            }
        }
        set_device_i32(io.text_kv_table_row, text_kv_addresses->bound_row(sequence.kv->text));
        set_device_i32(io.backend_kv_table_row,
                       sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend)
                                            : 0);
    } catch (...) {
        if (!text_active) {
            if (sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend)) {
                backend_kv_addresses->deactivate(*sequence.kv->backend);
            }
            if (text_kv_addresses->active(sequence.kv->text)) {
                text_kv_addresses->deactivate(sequence.kv->text);
            }
        }
        throw;
    }
}

void ProgramImplCore::unbind_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    try {
        if (sequence.kv->backend && backend_kv_addresses->active(*sequence.kv->backend)) {
            backend_kv_addresses->deactivate(*sequence.kv->backend);
        }
    } catch (...) {}
    try {
        if (text_kv_addresses->active(sequence.kv->text)) {
            text_kv_addresses->deactivate(sequence.kv->text);
        }
    } catch (...) {}
}

void ProgramImplCore::materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                              std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity) {
        throw std::logic_error("KV materialization request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV materialization requested without an allocation");
    }
    text_kv_addresses->materialize_to_tokens(sequence.kv->text, main_tokens, device.stream);
    if (backend_tokens != 0) {
        backend_kv_addresses->materialize_to_tokens(*sequence.kv->backend, backend_tokens,
                                                    device.stream);
    }
}

void ProgramImplCore::commit_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                         std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity ||
        (backend_tokens != 0 && !sequence.kv->backend)) {
        throw std::logic_error("KV commit request is outside the sequence bundle");
    }
    text_kv_addresses->commit_frontier(sequence.kv->text, main_tokens);
    if (sequence.kv->backend) {
        backend_kv_addresses->commit_frontier(*sequence.kv->backend, backend_tokens);
    }
}

void ProgramImplCore::trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                       std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > main_tokens) {
        throw std::logic_error("KV trim request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV trim requested without an allocation");
    }
    text_kv_addresses->destructive_truncate(sequence.kv->text, main_tokens);
    if (sequence.kv->backend) {
        backend_kv_addresses->destructive_truncate(*sequence.kv->backend, backend_tokens);
    }
}

void ProgramImplCore::release_sequence_growth_entitlement(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    try {
        text_kv_addresses->release_growth_entitlement(sequence.kv->text);
        if (sequence.kv->backend) {
            backend_kv_addresses->release_growth_entitlement(*sequence.kv->backend);
        }
    } catch (...) {}
}

void ProgramImplCore::release_active_sequence_kv_strict(SequenceState& sequence) noexcept {
    if (!sequence.kv || !text_kv_addresses ||
        !text_kv_addresses->can_release_after_deactivate(sequence.kv->text) ||
        (sequence.kv->backend &&
         (!backend_kv_addresses ||
          !backend_kv_addresses->can_release_after_deactivate(*sequence.kv->backend)))) {
        std::terminate();
    }
    if (sequence.kv->backend &&
        !backend_kv_addresses->release_after_deactivate(*sequence.kv->backend)) {
        std::terminate();
    }
    if (!text_kv_addresses->release_after_deactivate(sequence.kv->text)) { std::terminate(); }
    sequence.kv.reset();
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

void ProgramImplCore::release_sequence_kv_strict(SequenceState& sequence) noexcept {
    if (!sequence.kv || !text_kv_addresses || !text_kv_addresses->can_release(sequence.kv->text)) {
        std::terminate();
    }
    if (sequence.kv->backend &&
        (!backend_kv_addresses || !backend_kv_addresses->can_release(*sequence.kv->backend))) {
        std::terminate();
    }
    if (sequence.kv->backend && !backend_kv_addresses->release(*sequence.kv->backend)) {
        std::terminate();
    }
    if (!text_kv_addresses->release(sequence.kv->text)) { std::terminate(); }
    sequence.kv.reset();
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

void ProgramImplCore::release_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    unbind_sequence_kv(sequence);
    if (sequence.kv->backend && backend_kv_addresses) {
        (void)backend_kv_addresses->release(*sequence.kv->backend);
    }
    if (text_kv_addresses) { (void)text_kv_addresses->release(sequence.kv->text); }
    sequence.kv.reset();
    if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
}

qwen3_6::PagedKVCacheView ProgramImplCore::text_kv_view(const SequenceState& sequence) const {
    if (!sequence.kv || !text_kv_addresses->active(sequence.kv->text)) {
        throw std::logic_error("sequence has no active KV execution mapping");
    }
    return decoder->text_kv.execution_view(text_kv_addresses->execution_row(sequence.kv->text));
}

qwen3_6::PagedKVCacheView ProgramImplCore::mtp_kv_view(const SequenceState& sequence) const {
    if (speculative_backend != SpeculativeBackend::Mtp) { return {}; }
    if (decoder->mtp_cache() == nullptr || !sequence.kv || !sequence.kv->backend ||
        !backend_kv_addresses->active(*sequence.kv->backend)) {
        throw std::logic_error("sequence has no active MTP KV execution mapping");
    }
    return decoder->mtp_cache()->execution_view(
        backend_kv_addresses->execution_row(*sequence.kv->backend));
}

void ProgramImplCore::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

void ProgramImplCore::ordered_reset(SequenceState& sequence) {
    if (!state_store->valid(sequence.state.write)) {
        throw std::logic_error("pre-reset StateImage reservation is missing");
    } else {
        if (sequence.state.fork_pending || sequence.state.read != sequence.state.write ||
            state_store->role(sequence.state.write) != StateImageRole::ActiveMutable) {
            throw std::logic_error("StateImage reset requires a private mutable destination");
        }
    }
    refresh_state_views(sequence);
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    if (io.mtp) { set_device_i32(io.mtp->position, 0); }
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
}

void ProgramImplCore::prepare_graphs() {
    if (!use_cuda_graph) { return; }
    nvtx::ScopedRange prepare_range(nvtx::Name::CudaGraphPrepare, nvtx::Category::Graph);

    std::array<StateImageHandle, kMaximumConcurrency> capture_states{};
    for (std::uint32_t row = 0; row < max_concurrency; ++row) {
        std::optional<StateImageHandle> state = state_store->reserve_reset(device.stream);
        if (!state) { throw std::bad_alloc(); }
        capture_states[row] = *state;
    }
    const auto capture_state_slot = [&](std::uint32_t row) {
        return state_store->physical_slot(capture_states.at(row));
    };

    std::vector<KVAddressSpaceHandle> text_capture_allocations;
    std::vector<KVAddressSpaceHandle> mtp_capture_allocations;
    std::vector<KVAddressSpaceHandle> dflash_capture_allocations;
    const auto reserve_capture_rows = [&](qwen3_6::PagedKVCache& cache,
                                          KVAddressSpaceStore& addresses,
                                          std::vector<KVAddressSpaceHandle>& allocations,
                                          const char* label) {
        DeviceKVPagePool& pool       = cache.page_pool();
        KVExecutionTablePool& tables = cache.execution_tables();
        if (pool.capacity_pages() < max_concurrency) {
            throw std::invalid_argument(std::string(label) +
                                        " cannot provide one Paged KV page per concurrent request");
        }
        allocations.reserve(max_concurrency);
        for (std::uint32_t row = 0; row < max_concurrency; ++row) {
            std::optional<KVAddressSpaceHandle> allocation =
                addresses.create_active(1, static_cast<std::int32_t>(row));
            if (!allocation) { throw std::bad_alloc(); }
            allocations.push_back(*allocation);
            addresses.materialize_to_tokens(*allocation, 1, device.stream);

            // Capture profiles exercise arbitrary context envelopes. Repeating each row's private
            // page across its temporary table keeps every dummy read/write address valid without
            // reserving C full contexts solely for graph construction.
            tables.publish_repeated(addresses.execution_row(*allocation).handle(),
                                    addresses.physical_page(*allocation, 0),
                                    tables.logical_page_capacity(), device.stream);
        }
    };
    reserve_capture_rows(decoder->text_kv, *text_kv_addresses, text_capture_allocations,
                         "target KV cache");
    if (speculative_backend == SpeculativeBackend::Mtp) {
        reserve_capture_rows(*decoder->mtp_cache(), *backend_kv_addresses, mtp_capture_allocations,
                             "MTP KV cache");
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        reserve_capture_rows(dflash->full, *backend_kv_addresses, dflash_capture_allocations,
                             "DFlash Full KV cache");
    }
    device.synchronize();

    const auto clear_stable_controls = [&] {
        std::vector<Tensor> controls{
            io.token,
            io.pos,
            io.rope_pos,
            io.rope_delta,
        };
        if (io.mtp) {
            controls.push_back(io.mtp->position);
            controls.push_back(io.mtp->draft_tokens);
            controls.push_back(io.mtp->target_input_ids);
            controls.push_back(io.mtp->target_positions);
        }
        if (io.dflash_prefill) { controls.push_back(io.dflash_prefill->produced_count); }
        for (const Tensor& tensor : controls) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
    };
    const auto zero_capture_pages =
        [&](qwen3_6::PagedKVCache& cache, const KVAddressSpaceStore& addresses,
            const std::vector<KVAddressSpaceHandle>& allocations, std::uint32_t batch_size) {
            std::vector<DeviceKVPageHandle> pages;
            pages.reserve(batch_size);
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                pages.push_back(addresses.physical_page(allocations[row], 0));
            }
            cache.page_pool().zero_pages(pages, device.stream);
        };
    const auto prepare_representative = [&](std::uint32_t frontier, std::uint32_t batch_size) {
        if (batch_size == 0 || batch_size > max_concurrency) {
            throw std::logic_error("CUDA Graph representative batch is invalid");
        }
        work.reset();
        clear_stable_controls();
        zero_capture_pages(decoder->text_kv, *text_kv_addresses, text_capture_allocations,
                           batch_size);
        if (decoder->mtp_cache() != nullptr) {
            zero_capture_pages(*decoder->mtp_cache(), *backend_kv_addresses,
                               mtp_capture_allocations, batch_size);
        }
        if (dflash) {
            zero_capture_pages(dflash->full, *backend_kv_addresses, dflash_capture_allocations,
                               batch_size);
        }
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            state_images->zero_slot(capture_state_slot(row), device.stream);
            if (dflash) {
                const Tensor pending =
                    dflash->pending_features.slice(2, static_cast<std::int32_t>(row), 1);
                CUDA_CHECK(cudaMemsetAsync(pending.data, 0, pending.bytes(), device.stream));
            }
        }
        set_device_i32(io.pos, checked_i32(frontier, "graph representative position"));
        set_device_i32(io.rope_pos, checked_i32(frontier, "graph representative rope position"));
        if (io.mtp) {
            set_device_i32(io.mtp->position,
                           checked_i32(frontier, "graph representative MTP position"));
        }
        if (io.dflash_decode) {
            *dflash_host_ingress       = {};
            *dflash_host_egress        = {};
            const std::uint32_t extent = std::min(draft_window, capacity - frontier - 1U);
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                dflash_host_ingress->anchors[row] = 0;
                dflash_host_ingress->execution_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash frontier");
                dflash_host_ingress->context_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash context frontier");
                dflash_host_ingress->proposal_extents[row] = static_cast<std::int32_t>(extent);
                dflash_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                dflash_host_ingress->text_kv_table_rows[row]      = static_cast<std::int32_t>(row);
                dflash_host_ingress->dflash_kv_table_rows[row]    = static_cast<std::int32_t>(row);
                dflash_host_ingress->active_lanes[row]            = static_cast<std::int32_t>(row);
                dflash_host_ingress->state_source_slots[row]      = capture_state_slot(row);
                dflash_host_ingress->state_destination_slots[row] = capture_state_slot(row);
                dflash_host_ingress->sampling[row]                = {};
            }
        }
        if (io.mtp_decode) {
            *mtp_host_ingress          = {};
            *mtp_host_egress           = {};
            const std::uint32_t extent = std::min(draft_window, capacity - frontier - 1U);
            const std::uint32_t width  = draft_window + 1U;
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                mtp_host_ingress->anchors[row] = 0;
                mtp_host_ingress->base_frontiers[row] =
                    checked_i32(frontier, "graph representative MTP frontier");
                mtp_host_ingress->remaining_budgets[row] =
                    checked_i32(capacity, "graph representative MTP budget");
                mtp_host_ingress->current_extents[row] = static_cast<std::int32_t>(extent);
                mtp_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                for (std::uint32_t step = 0; step < draft_window; ++step) {
                    mtp_host_ingress->current_drafts[row * draft_window + step] = 0;
                }
                for (std::uint32_t column = 0; column < width; ++column) {
                    mtp_host_ingress->target_rope_positions[row * width + column] =
                        checked_i32(frontier + std::min(column, extent),
                                    "graph representative MTP RoPE position");
                }
                mtp_host_ingress->text_kv_table_rows[row]      = static_cast<std::int32_t>(row);
                mtp_host_ingress->mtp_kv_table_rows[row]       = static_cast<std::int32_t>(row);
                mtp_host_ingress->state_source_slots[row]      = capture_state_slot(row);
                mtp_host_ingress->state_destination_slots[row] = capture_state_slot(row);
                mtp_host_ingress->rope_deltas[row]             = 0;
                mtp_host_ingress->sampling[row]                = {};
            }
        }
        if (io.ordinary) {
            *ordinary_host_ingress = {};
            *ordinary_host_egress  = {};
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                ordinary_host_ingress->tokens[row] = 0;
                ordinary_host_ingress->cache_positions[row] =
                    checked_i32(frontier, "graph representative ordinary position");
                ordinary_host_ingress->rope_positions[row] =
                    checked_i32(frontier, "graph representative ordinary RoPE position");
                ordinary_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
                ordinary_host_ingress->state_source_slots[row] = capture_state_slot(row);
                ordinary_host_ingress->state_destination_slots[row] = capture_state_slot(row);
                ordinary_host_ingress->sampling[row]                = {};
            }
        }
    };
    const auto execution_core = [&] {
        return schedule::ExecutionCore{device,
                                       model,
                                       work,
                                       state_images->linear(),
                                       replay_records ? &*replay_records : nullptr,
                                       io,
                                       prefill_hidden,
                                       prefill_chunk,
                                       proposal_head};
    };

    if (speculative_backend == SpeculativeBackend::None) {
        const auto ordinary_profiles = ordinary_graph_profiles(capacity);
        validate_graph_profiles(ordinary_profiles, capacity - 1, "ordinary");
        const std::uint32_t ordinary_batch_limit = max_concurrency;
        schedule::OrdinaryBatchContext ordinary_state{
            execution_core(),      decoder->text_kv,
            *io.ordinary,          *ordinary_host_ingress,
            *ordinary_host_egress, state_images->continuation_hidden_store()};
        const GraphExecutionProfile code_warm = ordinary_profiles.front();
        prepare_representative(code_warm.min, 1);
        device.synchronize();
        schedule::ordinary_decode_batch(ordinary_state, 1, {code_warm.min + 1, code_warm.max + 1},
                                        nullptr);
        device.synchronize();

        ordinary_graphs.profiles.reserve(ordinary_profiles.size() * ordinary_batch_limit);
        for (std::uint32_t batch_size = 1; batch_size <= ordinary_batch_limit; ++batch_size) {
            for (const GraphExecutionProfile planned : ordinary_profiles) {
                ordinary_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = ordinary_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class =
                    planned.topology_class * ordinary_batch_limit + (batch_size - 1U);
                const ops::CausalAttentionExecutionEnvelope envelope{planned.min + 1,
                                                                     planned.max + 1};
                schedule::capture_ordinary_decode_batch(ordinary_state,
                                                        static_cast<std::int32_t>(batch_size),
                                                        envelope, profile.definition);
            }
        }
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        const auto planned_profiles = mtp_graph_profiles(capacity, draft_window);
        validate_graph_profiles(planned_profiles, capacity - 1, "MTP");
        schedule::MtpBatchContext mtp_state{execution_core(),
                                            decoder->text_kv,
                                            *decoder->mtp_cache(),
                                            *io.mtp_decode,
                                            *mtp_host_ingress,
                                            *mtp_host_egress,
                                            state_images->continuation_hidden_store()};
        const GraphExecutionProfile code_warm = planned_profiles.front();
        prepare_representative(code_warm.min, 1);
        device.synchronize();
        schedule::mtp_decode_batch(
            mtp_state, 1, draft_window,
            mtp_causal_attention_envelopes(code_warm.max, draft_window, capacity), nullptr);
        device.synchronize();

        mtp_graphs.profiles.reserve(planned_profiles.size() * max_concurrency);
        for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
            for (const GraphExecutionProfile planned : planned_profiles) {
                mtp_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = mtp_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class =
                    planned.topology_class * max_concurrency + (batch_size - 1U);
                schedule::capture_mtp_decode_batch(
                    mtp_state, static_cast<std::int32_t>(batch_size), draft_window,
                    mtp_causal_attention_envelopes(planned.max, draft_window, capacity),
                    profile.definition);
            }
        }
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        const auto batch_one_profiles = dflash_graph_profiles(capacity, draft_window, 1);
        validate_graph_profiles(batch_one_profiles, capacity - 1, "DFlash");
        schedule::DFlashBatchContext dflash_state{execution_core(),
                                                  decoder->text_kv,
                                                  *dflash,
                                                  *io.dflash_decode,
                                                  *dflash_host_ingress,
                                                  *dflash_host_egress,
                                                  state_images->continuation_hidden_store()};
        const GraphExecutionProfile code_warm = batch_one_profiles.front();
        const ops::CausalAttentionExecutionEnvelope code_warm_target{
            1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                   capacity, static_cast<std::uint64_t>(code_warm.max) + draft_window + 1ULL))};
        prepare_representative(code_warm.min, 1);
        device.synchronize();
        schedule::dflash_decode_batch(dflash_state, 1, draft_window,
                                      dflash_envelopes(code_warm.min, code_warm.max, draft_window),
                                      code_warm_target, nullptr);
        device.synchronize();

        dflash_graphs.profiles.reserve(batch_one_profiles.size() * max_concurrency);
        for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
            const auto planned_profiles =
                batch_size == 1 ? batch_one_profiles
                                : dflash_graph_profiles(capacity, draft_window, batch_size);
            validate_graph_profiles(planned_profiles, capacity - 1, "DFlash");
            for (const GraphExecutionProfile planned : planned_profiles) {
                dflash_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = dflash_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class =
                    planned.topology_class * max_concurrency + (batch_size - 1U);
                const ops::CausalAttentionExecutionEnvelope target_envelope{
                    1,
                    static_cast<std::uint32_t>(std::min<std::uint64_t>(
                        capacity, static_cast<std::uint64_t>(planned.max) + draft_window + 1ULL))};

                schedule::capture_dflash_decode_batch(
                    dflash_state, static_cast<std::int32_t>(batch_size), draft_window,
                    dflash_envelopes(planned.min, planned.max, draft_window), target_envelope,
                    profile.definition);
            }
        }
    }

    if (!ordinary_graphs.profiles.empty()) {
        instantiate_graph_family(ordinary_graphs, "ordinary", device, prepare_representative);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) {
        instantiate_graph_family(mtp_graphs, "MTP", device, prepare_representative);
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        instantiate_graph_family(dflash_graphs, "DFlash", device, prepare_representative);
    }

    clear_stable_controls();
    state_images->zero_all(device.stream);
    if (dflash) {
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_features.data, 0,
                                   dflash->prefill_features.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_positions.data, 0,
                                   dflash->prefill_positions.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->pending_features.data, 0,
                                   dflash->pending_features.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    device.synchronize();
    for (std::uint32_t row = 0; row < max_concurrency; ++row) {
        if (!state_store->release(capture_states[row])) {
            throw std::logic_error("CUDA Graph capture StateImage could not be released");
        }
    }

    const auto release_capture_rows = [](KVAddressSpaceStore& addresses,
                                         std::vector<KVAddressSpaceHandle>& allocations) {
        for (const KVAddressSpaceHandle allocation : allocations) {
            addresses.deactivate(allocation);
            if (!addresses.release(allocation)) {
                throw std::logic_error("CUDA Graph capture KV address space could not be released");
            }
        }
        allocations.clear();
    };
    if (!dflash_capture_allocations.empty()) {
        release_capture_rows(*backend_kv_addresses, dflash_capture_allocations);
    }
    if (!mtp_capture_allocations.empty()) {
        release_capture_rows(*backend_kv_addresses, mtp_capture_allocations);
    }
    release_capture_rows(*text_kv_addresses, text_capture_allocations);
}

void ProgramImplCore::install_sampling(SequenceState& sequence, RequestControl& request,
                                       const ops::SamplingConfig& config) {
    Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(sequence.lane), 1)
                        .view({TextConfig::token_domain});
    request.sampling_host     = config;
    request.speculative_stats = SpeculativeStats{
        .backend               = speculative_backend,
        .enabled               = speculative_backend != SpeculativeBackend::None,
        .draft_window          = draft_window,
        .accepted_per_position = std::vector<std::uint64_t>(draft_window, 0),
    };
    const bool penalties = request.sampling_host.presence_penalty != 0.0F ||
                           request.sampling_host.frequency_penalty != 0.0F;
    if (penalties) { CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream)); }
    request.sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(counts.data) : nullptr;
    Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
    CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &request.sampling_host,
                               sizeof(request.sampling_host), cudaMemcpyHostToDevice,
                               device.stream));
}

void ProgramImplCore::copy_tail(SequenceState& sequence, const Tensor& source) {
    if (source.dtype != DType::BF16 || source.ne[0] != TextConfig::hidden || source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(sequence.tail_hidden.data, source.data, sequence.tail_hidden.bytes(),
                               cudaMemcpyDeviceToDevice, device.stream));
    sequence.tail_hidden_valid = true;
}

void ProgramImplCore::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

void ProgramImplCore::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

void ProgramImplCore::enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                                    std::span<const std::uint32_t> starts,
                                                    std::span<const std::uint32_t> counts) {
    if (speculative_backend != SpeculativeBackend::DFlash || !dflash || !io.dflash_decode ||
        lanes.empty() || lanes.size() > max_concurrency || starts.size() != lanes.size() ||
        counts.size() != lanes.size()) {
        throw std::logic_error("DFlash context append has invalid membership");
    }

    std::uint32_t minimum_count = draft_window + 1U;
    std::uint32_t maximum_count = 0;
    *dflash_host_ingress        = {};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || counts[row] == 0 || counts[row] > draft_window + 1U ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("DFlash context append contains an invalid row");
        }
        SequenceState& sequence   = active_sequence(lane);
        const std::uint32_t start = starts[row];
        const std::uint64_t end64 = static_cast<std::uint64_t>(start) + counts[row];
        const std::uint32_t end   = static_cast<std::uint32_t>(end64);
        if (!sequence.kv || !sequence.kv->backend ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            backend_kv_addresses->bound_row(*sequence.kv->backend) < 0 || end64 > capacity) {
            throw std::logic_error("DFlash context append is outside retained target storage");
        }
        dflash_host_ingress->context_frontiers[row] =
            checked_i32(start, "DFlash append context frontier");
        dflash_host_ingress->execution_frontiers[row] =
            checked_i32(end, "DFlash append target frontier");
        dflash_host_ingress->dflash_kv_table_rows[row] =
            backend_kv_addresses->bound_row(*sequence.kv->backend);
        dflash_host_ingress->active_lanes[row]            = static_cast<std::int32_t>(lane);
        const StateImageSelectors selectors               = state_selectors(sequence);
        dflash_host_ingress->state_source_slots[row]      = selectors.source;
        dflash_host_ingress->state_destination_slots[row] = selectors.destination;
        materialize_sequence_kv(sequence, std::max(sequence.text_kv_valid, end), end);
        minimum_count = std::min(minimum_count, counts[row]);
        maximum_count = std::max(maximum_count, counts[row]);
    }

    qwen3_6::DFlashDecodeState& frame = *io.dflash_decode;
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, dflash_host_ingress,
                               sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                               device.stream));
    const auto batch                = static_cast<std::int32_t>(lanes.size());
    Tensor active_lane_tensor       = frame.active_lanes.slice(0, 0, batch);
    Tensor state_destination_tensor = frame.state_destination_slots.slice(0, 0, batch);
    Tensor device_starts            = frame.context_frontiers.slice(0, 0, batch);
    Tensor device_ends              = frame.execution_frontiers.slice(0, 0, batch);
    Tensor table_rows               = frame.dflash_kv_table_rows.slice(0, 0, batch);
    Tensor positions                = frame.append_positions.slice(1, 0, batch);
    Tensor device_counts            = frame.append_counts.slice(0, 0, batch);

    work.reset();
    Tensor features =
        work.alloc(DType::BF16, {DFlashConfig::feature_rows,
                                 static_cast<std::int32_t>(draft_window + 1U), batch});
    ops::prepare_ragged_prefix(dflash->pending_features, active_lane_tensor, device_starts,
                               device_ends, features, positions, device_counts, device.stream);

    schedule::DFlashAppendContext state{{device, model, work, state_images->linear(),
                                         replay_records ? &*replay_records : nullptr, io,
                                         prefill_hidden, prefill_chunk, proposal_head},
                                        *dflash};
    mark_workspace_usage(workspace_plan.dflash_context);
    schedule::dflash_append_context(state, features, positions, device_counts,
                                    state_destination_tensor, table_rows,
                                    {minimum_count, maximum_count});
}

void ProgramImplCore::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= TextConfig::token_domain) {
            throw std::runtime_error("target returned a token outside the 248077-token domain");
        }
    }
}

runtime::PrefillStepResult
ProgramImplCore::advance_prefill(SequenceState& sequence, RequestControl& request,
                                 runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) {
        throw std::logic_error("staged prefill step requires an active concurrent request");
    }

    RequestControl::Prefill& staged = *request.prefill;
    if (staged.pending_capture_offer != 0) {
        throw std::logic_error("prefill cannot advance while a capture offer is pending");
    }
    const runtime::BeginSummary summary{.prompt_tokens        = staged.prompt_tokens,
                                        .reused_prompt_tokens = staged.base,
                                        .prefix_reuse_path    = staged.reuse};
    std::uint32_t processed_prompt_tokens = 0;
    const auto started                    = Clock::now();
    try {
        if (staged.next_capture < staged.capture_groups.size() &&
            staged.capture_groups[staged.next_capture].frontier == staged.cursor) {
            if (staged.cursor != staged.base ||
                !staged.capture_groups[staged.next_capture].shared ||
                staged.capture_groups[staged.next_capture].rewrite ||
                staged.capture_groups[staged.next_capture].long_anchor) {
                throw std::logic_error("zero-prefill capture is not a shared base promotion");
            }
            if (++next_capture_offer_id_ == 0) { ++next_capture_offer_id_; }
            staged.pending_capture_offer = next_capture_offer_id_;
            return runtime::PrefillStepResult{
                .summary = summary,
                .timing  = timing.finish(),
            };
        }
        StateImageSelectors selectors = state_selectors(sequence);
        Tensor rewrite_capture_hidden;
        Tensor* rewrite_capture_hidden_ptr = nullptr;
        if (staged.next_capture < staged.capture_groups.size()) {
            rewrite_capture_hidden = state_images->continuation_hidden_slot(selectors.destination);
            rewrite_capture_hidden_ptr = &rewrite_capture_hidden;
        }
        schedule::PrefillContext schedule_state{
            {device, model, work, state_images->linear(),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head},
            text_kv_view(sequence),
            mtp_kv_view(sequence),
            decoder->text_kv,
            decoder->mtp_cache(),
            dflash ? &*dflash : nullptr,
            staged.cursor,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1).data),
            rewrite_capture_hidden_ptr,
            selectors.source,
            selectors.destination,
            staged.initial_mtp_extent,
            dflash_host_ingress};

        if (staged.mtp_bridge == MtpBridgeMode::BeforeSuffix) {
            if (staged.cursor != staged.base || staged.base == 0 ||
                staged.cursor >= staged.prompt_tokens) {
                throw std::logic_error("staged MTP bridge is outside the reusable suffix");
            }
            mark_workspace_usage(workspace_plan.mtp_prefill);
            const Tensor& previous_hidden = sequence.tail_hidden;
            const schedule::MtpBridgeInput bridge{
                .previous_hidden = &previous_hidden,
                .position        = checked_i32(staged.base - 1, "MTP bridge position"),
                .rope_position   = prompt_rope_position(staged.prompt, staged.base - 1),
            };
            if (staged.vision) {
                schedule::mtp_bridge_multimodal(schedule_state, staged.prompt, *staged.vision,
                                                bridge);
            } else {
                Tensor bridge_token = io.mtp->target_input_ids.slice(0, 0, 1);
                const TokenId token = staged.prompt.token_ids[staged.base];
                CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                           cudaMemcpyHostToDevice, device.stream));
                schedule::mtp_bridge_and_propose(schedule_state, bridge_token, previous_hidden,
                                                 bridge.position, bridge.rope_position, false);
            }
            sequence.mtp_kv_valid = staged.base;
            commit_sequence_kv(sequence, sequence.text_kv_valid, sequence.mtp_kv_valid);
            staged.mtp_bridge = MtpBridgeMode::None;
        }

        if (staged.cursor < staged.prompt_tokens) {
            const std::uint32_t nominal =
                std::min(prefill_chunk, staged.prompt_tokens - staged.cursor);
            mark_workspace_usage(staged.prepare_mtp ? workspace_plan.mtp_prefill
                                                    : workspace_plan.text_prefill);
            if (speculative_backend == SpeculativeBackend::DFlash) {
                mark_workspace_usage(workspace_plan.dflash_context);
            }
            std::uint32_t remaining          = nominal;
            std::uint32_t final_chunk_tokens = 0;
            bool finalized                   = false;
            while (remaining != 0) {
                schedule_state.text_kv_base           = staged.cursor;
                selectors                             = state_selectors(sequence);
                schedule_state.state_source_slot      = selectors.source;
                schedule_state.state_destination_slot = selectors.destination;
                if (staged.next_capture < staged.capture_groups.size()) {
                    rewrite_capture_hidden =
                        state_images->continuation_hidden_slot(selectors.destination);
                    schedule_state.rewrite_checkpoint_hidden = &rewrite_capture_hidden;
                } else {
                    schedule_state.rewrite_checkpoint_hidden = nullptr;
                }

                const bool final_candidate = staged.cursor + remaining == staged.prompt_tokens;
                const std::optional<std::uint32_t> capture_frontier =
                    staged.next_capture < staged.capture_groups.size()
                        ? std::optional<std::uint32_t>(
                              staged.capture_groups[staged.next_capture].frontier)
                        : std::nullopt;
                std::optional<std::uint32_t> split_frontier = capture_frontier;
                const auto rewrite_split                    = std::upper_bound(
                    staged.prompt.identity.rewrite_execution_frontiers.begin(),
                    staged.prompt.identity.rewrite_execution_frontiers.end(), staged.cursor);
                if (rewrite_split != staged.prompt.identity.rewrite_execution_frontiers.end() &&
                    (!split_frontier || *rewrite_split < *split_frontier)) {
                    split_frontier = *rewrite_split;
                }
                schedule::PrefillChunkResult result;
                timing.pause();
                if (staged.vision) {
                    if (!workspace_plan.vision) {
                        throw std::logic_error("active Vision prefill lost its workspace plan");
                    }
                    mark_workspace_usage(workspace_plan.vision->capacity_bytes);
                    result = schedule::prefill_multimodal_chunk(schedule_state, staged.prompt,
                                                                *staged.vision, remaining,
                                                                split_frontier, final_candidate);
                } else {
                    result = schedule::prefill_text_chunk(
                        schedule_state, std::span<const TokenId>(staged.prompt.token_ids),
                        remaining, split_frontier, final_candidate);
                }
                timing.include(result.timing);
                timing.resume_post();
                if (result.processed_tokens == 0 || result.processed_tokens > remaining) {
                    throw std::logic_error("ordinary prefill chunk made invalid progress");
                }
                if (staged.vision) { staged.vision->release_encoded_media_payloads(); }
                staged.cursor += result.processed_tokens;
                processed_prompt_tokens += result.processed_tokens;
                remaining -= result.processed_tokens;
                final_chunk_tokens     = result.processed_tokens;
                sequence.text_kv_valid = staged.cursor;
                if (staged.prepare_mtp) { sequence.mtp_kv_valid = staged.cursor; }
                if (speculative_backend == SpeculativeBackend::DFlash) {
                    sequence.dflash_context_frontier = staged.cursor;
                }
                commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));

                // Prompt transitions are canonical immediately. If this was the first write after
                // an immutable source, close the Fork before potentially freezing a new rewrite.
                settle_state_fork(sequence);
                const bool reached_capture = capture_frontier && staged.cursor == *capture_frontier;
                if (reached_capture) {
                    if (result.finalized) {
                        // The prompt-frontier state becomes publishable only after the generated
                        // Begin token is committed. commit() emits the offer for this group.
                    } else {
                        staged.elapsed_seconds +=
                            std::chrono::duration<double>(Clock::now() - started).count();
                        if (++next_capture_offer_id_ == 0) { ++next_capture_offer_id_; }
                        staged.pending_capture_offer = next_capture_offer_id_;
                        return runtime::PrefillStepResult{
                            .summary                 = summary,
                            .processed_prompt_tokens = processed_prompt_tokens,
                            .timing                  = timing.finish(),
                        };
                    }
                }

                finalized = result.finalized;
                if (finalized || remaining == 0) { break; }
            }

            if (!finalized) {
                if (staged.cursor == staged.prompt_tokens) {
                    throw std::logic_error("staged prefill reached the prompt without sampling");
                }
                staged.elapsed_seconds +=
                    std::chrono::duration<double>(Clock::now() - started).count();
                return runtime::PrefillStepResult{
                    .summary                 = summary,
                    .processed_prompt_tokens = processed_prompt_tokens,
                    .timing                  = timing.finish(),
                };
            }
            if (staged.cursor != staged.prompt_tokens) {
                throw std::logic_error("staged prefill sampled before the prompt frontier");
            }
            timing.resume_submit();
            copy_tail(sequence, prefill_hidden.slice(
                                    1, static_cast<std::int32_t>(final_chunk_tokens) - 1, 1));
        } else {
            mark_workspace_usage(workspace_plan.ordinary_round);
            if (!sequence.tail_hidden_valid) {
                throw std::logic_error("zero-suffix reuse has no target tail hidden");
            }
            schedule::sample_from_hidden(schedule_state, sequence.tail_hidden,
                                         checked_i32(staged.prompt_tokens, "sample position"),
                                         ops::kSamplePurposePrefill);
            set_device_i32(io.rope_pos, checked_i32(staged.prompt_tokens, "rope position") +
                                            sequence.rope_delta);
            if (staged.prepare_mtp) {
                if (staged.mtp_bridge != MtpBridgeMode::AfterExactHit) {
                    throw std::logic_error("zero-suffix MTP reuse has no exact-hit bridge");
                }
                mark_workspace_usage(workspace_plan.mtp_prefill);
                const auto bridge_rope =
                    prompt_rope_position(staged.prompt, staged.prompt_tokens - 1);
                schedule::mtp_bridge_and_propose(
                    schedule_state, io.token, sequence.tail_hidden,
                    checked_i32(staged.prompt_tokens - 1, "MTP full-prefix bridge position"),
                    bridge_rope, staged.initial_mtp_extent != 0);
                sequence.mtp_kv_valid = staged.prompt_tokens;
                commit_sequence_kv(sequence, sequence.text_kv_valid, sequence.mtp_kv_valid);
                staged.mtp_bridge = MtpBridgeMode::None;
            }
        }

        copy_round_token();
        std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> initial_drafts{};
        if (staged.prepare_mtp && staged.initial_mtp_extent != 0) {
            CUDA_CHECK(cudaMemcpyAsync(initial_drafts.data(), io.mtp->draft_tokens.data,
                                       staged.initial_mtp_extent * sizeof(TokenId),
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        const double vision_seconds       = staged.vision ? staged.vision->elapsed_seconds() : 0.0;
        const std::uint32_t prompt_tokens = staged.prompt_tokens;

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (sequence.ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        sequence.ledger.push_back(host_tokens[0]);
        sequence.prefix_identity.append_generated(1, sequence.rope_delta);
        sequence.prefix_digests.append_generated(std::span<const TokenId>(host_tokens, 1),
                                                 sequence.rope_delta);
        sequence.text_kv_valid = prompt_tokens;
        if (staged.prepare_mtp) {
            if (sequence.mtp_kv_valid != prompt_tokens) {
                throw std::logic_error("staged MTP prefill did not reach the prompt frontier");
            }
            sequence.mtp_draft_count = staged.initial_mtp_extent;
            std::copy_n(initial_drafts.begin(), staged.initial_mtp_extent,
                        sequence.mtp_drafts.begin());
        } else if (speculative_backend == SpeculativeBackend::DFlash &&
                   sequence.dflash_context_frontier != prompt_tokens) {
            throw std::logic_error("staged DFlash prefill did not reach the prompt frontier");
        }
        sequence.tail_hidden_valid      = true;
        request.timings.vision_seconds  = vision_seconds;
        request.timings.prefill_seconds = std::max(0.0, staged.elapsed_seconds - vision_seconds);
        staged.prompt.release_all_media_payloads();
        if (staged.vision) { staged.vision->retire_handoff(); }

        const bool prompt_frontier_capture =
            staged.next_capture < staged.capture_groups.size() &&
            staged.capture_groups[staged.next_capture].frontier == prompt_tokens;
        if (!prompt_frontier_capture) { request.prefill.reset(); }
        request.pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                             .base_E        = 0,
                                             .base_S        = 0,
                                             .prompt_tokens = prompt_tokens,
                                             .produced      = 1};
        request.lifecycle = Lifecycle::Pending;
        return runtime::PrefillStepResult{
            .summary = summary,
            .round   = runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, 1)},
            .processed_prompt_tokens = processed_prompt_tokens,
            .complete                = true,
            .timing                  = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        const std::uint32_t lane = sequence.lane;
        clear_execution_failure_lanes(std::span<const std::uint32_t>(&lane, 1));
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                                       std::span<const runtime::RoundBudget> budgets,
                                       runtime::ExecutionTiming* failed_timing) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeOrdinaryRound, nvtx::Category::Decode,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (speculative_backend != SpeculativeBackend::None) {
        throw std::logic_error("ordinary batch execution requires the ordinary backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("ordinary batch membership is invalid");
    }

    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("ordinary batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier) {
            throw std::logic_error("ordinary batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto start = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeOrdinarySubmit, nvtx::Category::Decode,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable = nullptr;
        ops::CausalAttentionExecutionEnvelope envelope{maximum_frontier + 1, maximum_frontier + 1};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(ordinary_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "ordinary batch");
            executable = &install_graph_profile(ordinary_graphs, profile, "ordinary batch");
            envelope   = {profile.min_execution_frontier + 1, profile.max_execution_frontier + 1};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence            = active_sequence(lanes[row]);
            const RequestControl& request      = requests[lanes[row]];
            const std::uint32_t frontier       = sequence.execution_frontier;
            ordinary_host_ingress->tokens[row] = sequence.ledger.back();
            ordinary_host_ingress->cache_positions[row] =
                checked_i32(frontier, "ordinary batch position");
            ordinary_host_ingress->rope_positions[row] =
                checked_i32(frontier, "ordinary batch RoPE position") + sequence.rope_delta;
            ordinary_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            const StateImageSelectors selectors                 = state_selectors(sequence);
            ordinary_host_ingress->state_source_slots[row]      = selectors.source;
            ordinary_host_ingress->state_destination_slots[row] = selectors.destination;
            ordinary_host_ingress->sampling[row]                = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + 1, 0);
        }

        schedule::OrdinaryBatchContext schedule_state{{device, model, work, state_images->linear(),
                                                       replay_records ? &*replay_records : nullptr,
                                                       io, prefill_hidden, prefill_chunk,
                                                       proposal_head},
                                                      decoder->text_kv,
                                                      *io.ordinary,
                                                      *ordinary_host_ingress,
                                                      *ordinary_host_egress,
                                                      state_images->continuation_hidden_store()};

        mark_workspace_usage(workspace_plan.ordinary_round);
        schedule::ordinary_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                        envelope, executable);
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        }
        timing.end_wait();

        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence    = active_sequence(lanes[row]);
            RequestControl& request    = requests[lanes[row]];
            const std::uint32_t base_E = sequence.execution_frontier;
            const std::uint32_t base_S = sequence.ledger_frontier;
            const TokenId token        = ordinary_host_egress->sampled_tokens[row];
            validate_licensed_tokens(std::span<const TokenId>(&token, 1));
            sequence.text_kv_valid = base_E + 1;
            commit_sequence_kv(sequence, sequence.text_kv_valid, 0);
            sequence.tail_hidden_valid = true;
            sequence.ledger.push_back(token);
            sequence.prefix_identity.append_generated(1, sequence.rope_delta);
            sequence.prefix_digests.append_generated(std::span<const TokenId>(&token, 1),
                                                     sequence.rope_delta);
            request.pending   = PendingCandidate{.kind          = PendingKind::Ordinary,
                                                 .base_E        = base_E,
                                                 .base_S        = base_S,
                                                 .prompt_tokens = 0,
                                                 .produced      = 1};
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens =
                std::span<const TokenId>(ordinary_host_egress->sampled_tokens.data(), lanes.size()),
            .timing = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_mtp_batch(std::span<const std::uint32_t> lanes,
                                  std::span<const runtime::RoundBudget> budgets,
                                  runtime::ExecutionTiming* failed_timing) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeMtpRound, nvtx::Category::Mtp,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (speculative_backend != SpeculativeBackend::Mtp || !io.mtp_decode ||
        decoder->mtp_cache() == nullptr) {
        throw std::logic_error("MTP batch execution requires the MTP backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("MTP batch membership is invalid");
    }

    const std::uint32_t width      = draft_window + 1;
    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("MTP batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv || !sequence.kv->backend ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            backend_kv_addresses->bound_row(*sequence.kv->backend) < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.mtp_kv_valid != sequence.execution_frontier ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier ||
            sequence.mtp_draft_count > draft_window) {
            throw std::logic_error("MTP batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto started = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeMtpSubmit, nvtx::Category::Mtp,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable = nullptr;
        schedule::MtpCausalAttentionEnvelopes envelopes =
            mtp_causal_attention_envelopes(maximum_frontier, draft_window, capacity);
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(mtp_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "MTP batch");
            executable = &install_graph_profile(mtp_graphs, profile, "MTP batch");
            envelopes = mtp_causal_attention_envelopes(profile.max_execution_frontier, draft_window,
                                                       capacity);
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = active_sequence(lanes[row]);
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1
                                                    : 0;
            const std::uint32_t extent =
                std::min({sequence.mtp_draft_count, draft_window, max_by_budget,
                          capacity - sequence.execution_frontier - 1});
            mtp_host_ingress->anchors[row]        = sequence.ledger.back();
            mtp_host_ingress->base_frontiers[row] = checked_i32(frontier, "MTP batch frontier");
            mtp_host_ingress->remaining_budgets[row] =
                checked_i32(budgets[row].generated_tokens_remaining, "MTP batch remaining budget");
            mtp_host_ingress->current_extents[row]      = static_cast<std::int32_t>(extent);
            mtp_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1);
            for (std::uint32_t j = 0; j < draft_window; ++j) {
                mtp_host_ingress->current_drafts[row * draft_window + j] =
                    j < extent ? sequence.mtp_drafts[j] : sequence.ledger.back();
            }
            for (std::uint32_t j = 0; j < width; ++j) {
                const std::uint32_t position = frontier + std::min(j, extent);
                mtp_host_ingress->target_rope_positions[row * width + j] =
                    checked_i32(position, "MTP batch RoPE position") + sequence.rope_delta;
            }
            mtp_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            mtp_host_ingress->mtp_kv_table_rows[row] =
                backend_kv_addresses->bound_row(*sequence.kv->backend);
            const StateImageSelectors selectors            = state_selectors(sequence);
            mtp_host_ingress->state_source_slots[row]      = selectors.source;
            mtp_host_ingress->state_destination_slots[row] = selectors.destination;
            mtp_host_ingress->rope_deltas[row]             = sequence.rope_delta;
            mtp_host_ingress->sampling[row]                = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + extent + 1,
                                    std::min(capacity, frontier + extent + draft_window));
        }

        schedule::MtpBatchContext schedule_state{{device, model, work, state_images->linear(),
                                                  replay_records ? &*replay_records : nullptr, io,
                                                  prefill_hidden, prefill_chunk, proposal_head},
                                                 decoder->text_kv,
                                                 *decoder->mtp_cache(),
                                                 *io.mtp_decode,
                                                 *mtp_host_ingress,
                                                 *mtp_host_egress,
                                                 state_images->continuation_hidden_store()};

        mark_workspace_usage(workspace_plan.mtp_round);
        schedule::mtp_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                   draft_window, envelopes, executable);
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeMtpWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        }
        timing.end_wait();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = active_sequence(lanes[row]);
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = mtp_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = mtp_host_egress->accepted_drafts[row];
            const std::int32_t next_i     = mtp_host_egress->next_extents[row];
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || next_i < 0 ||
                next_i > static_cast<std::int32_t>(draft_window) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("MTP batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(mtp_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            const std::uint32_t pcur =
                static_cast<std::uint32_t>(mtp_host_ingress->current_extents[row]);
            if (pcur == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += pcur;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            request.pending = PendingCandidate{
                .kind          = PendingKind::Speculative,
                .base_E        = base_E,
                .base_S        = base_S,
                .prompt_tokens = 0,
                .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(mtp_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(mtp_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width,
            .timing     = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeMtpWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_dflash_batch(std::span<const std::uint32_t> lanes,
                                     std::span<const runtime::RoundBudget> budgets,
                                     runtime::ExecutionTiming* failed_timing) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeDFlashRound, nvtx::Category::DFlash,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (speculative_backend != SpeculativeBackend::DFlash || !io.dflash_decode || !dflash) {
        throw std::logic_error("DFlash batch execution requires the DFlash backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("DFlash batch membership is invalid");
    }

    const std::uint32_t width           = draft_window + 1U;
    std::uint32_t maximum_frontier      = 0;
    std::uint32_t maximum_target_tokens = 1;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("DFlash batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv || !sequence.kv->backend ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            backend_kv_addresses->bound_row(*sequence.kv->backend) < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            sequence.dflash_context_frontier > sequence.execution_frontier ||
            sequence.execution_frontier - sequence.dflash_context_frontier > width ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier) {
            throw std::logic_error("DFlash batch row is not decode-ready");
        }
        const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                ? budgets[row].generated_tokens_remaining - 1U
                                                : 0U;
        const std::uint32_t extent =
            std::min({draft_window, max_by_budget, capacity - sequence.execution_frontier - 1U});
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
        maximum_target_tokens =
            std::max(maximum_target_tokens, sequence.execution_frontier + extent + 1U);
    }

    const auto started = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeDFlashSubmit, nvtx::Category::DFlash,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable   = nullptr;
        schedule::DFlashEnvelopes envelopes = dflash_envelopes(0, maximum_frontier, draft_window);
        ops::CausalAttentionExecutionEnvelope target_envelope{1, maximum_target_tokens};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(dflash_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "DFlash batch");
            executable      = &install_graph_profile(dflash_graphs, profile, "DFlash batch");
            envelopes       = dflash_envelopes(profile.min_execution_frontier,
                                               profile.max_execution_frontier, draft_window);
            target_envelope = {
                1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                       capacity, static_cast<std::uint64_t>(profile.max_execution_frontier) +
                                     draft_window + 1ULL))};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = active_sequence(lanes[row]);
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1U
                                                    : 0U;
            const std::uint32_t extent =
                std::min({draft_window, max_by_budget, capacity - frontier - 1U});
            dflash_host_ingress->anchors[row] = sequence.ledger.back();
            dflash_host_ingress->execution_frontiers[row] =
                checked_i32(frontier, "DFlash batch frontier");
            dflash_host_ingress->context_frontiers[row] =
                checked_i32(sequence.dflash_context_frontier, "DFlash context frontier");
            dflash_host_ingress->proposal_extents[row]     = static_cast<std::int32_t>(extent);
            dflash_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1U);
            dflash_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            dflash_host_ingress->dflash_kv_table_rows[row] =
                backend_kv_addresses->bound_row(*sequence.kv->backend);
            dflash_host_ingress->active_lanes[row]       = static_cast<std::int32_t>(sequence.lane);
            const StateImageSelectors selectors          = state_selectors(sequence);
            dflash_host_ingress->state_source_slots[row] = selectors.source;
            dflash_host_ingress->state_destination_slots[row] = selectors.destination;
            dflash_host_ingress->sampling[row]                = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + extent + 1U, frontier);
        }

        schedule::DFlashBatchContext schedule_state{{device, model, work, state_images->linear(),
                                                     replay_records ? &*replay_records : nullptr,
                                                     io, prefill_hidden, prefill_chunk,
                                                     proposal_head},
                                                    decoder->text_kv,
                                                    *dflash,
                                                    *io.dflash_decode,
                                                    *dflash_host_ingress,
                                                    *dflash_host_egress,
                                                    state_images->continuation_hidden_store()};

        mark_workspace_usage(workspace_plan.dflash_round);
        schedule::dflash_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                      draft_window, envelopes, target_envelope, executable);
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeDFlashWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        }
        timing.end_wait();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = active_sequence(lanes[row]);
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = dflash_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = dflash_host_egress->accepted_drafts[row];
            const std::uint32_t extent =
                static_cast<std::uint32_t>(dflash_host_ingress->proposal_extents[row]);
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || accepted_i > static_cast<std::int32_t>(extent) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("DFlash batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(dflash_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            if (extent == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += extent;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            sequence.dflash_context_frontier = base_E;
            request.pending                  = PendingCandidate{
                                 .kind          = PendingKind::Speculative,
                                 .base_E        = base_E,
                                 .base_S        = base_S,
                                 .prompt_tokens = 0,
                                 .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(dflash_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(dflash_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width,
            .timing     = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeDFlashWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_raw(std::span<const std::uint32_t> lanes,
                            std::span<const runtime::RoundBudget> budgets,
                            runtime::ExecutionTiming* failed_timing) {
    if (speculative_backend == SpeculativeBackend::None) {
        return decode_ordinary_batch(lanes, budgets, failed_timing);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) {
        return decode_mtp_batch(lanes, budgets, failed_timing);
    }
    return decode_dflash_batch(lanes, budgets, failed_timing);
}

runtime::ExecutionTiming
ProgramImplCore::resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                                 std::uint32_t accepted_tokens, bool terminal,
                                                 runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    if (request.lifecycle != Lifecycle::Pending) {
        throw std::logic_error("pending resolution requires a pending generated round");
    }
    if ((request.pending.kind != PendingKind::Begin &&
         request.pending.kind != PendingKind::Ordinary) ||
        request.pending.produced != 1 || accepted_tokens != 1) {
        throw std::logic_error("non-speculative pending round must commit its single token");
    }

    switch (request.pending.kind) {
    case PendingKind::Begin:
        sequence.execution_frontier = request.pending.prompt_tokens;
        sequence.ledger_frontier    = request.pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
        advance_rebuild_work(sequence, request.pending.base_E + request.pending.produced,
                             prefill_chunk);
        sequence.execution_frontier = request.pending.base_E + request.pending.produced;
        sequence.ledger_frontier    = request.pending.base_S + request.pending.produced;
        break;
    case PendingKind::Speculative:
    case PendingKind::None:
        throw std::logic_error("non-speculative pending round has an invalid kind");
    }
    if (sequence.ledger_frontier != sequence.execution_frontier + 1 ||
        sequence.ledger.size() != sequence.ledger_frontier ||
        sequence.prefix_identity.size() != sequence.ledger_frontier ||
        sequence.prefix_digests.size() != sequence.ledger_frontier) {
        throw std::logic_error("resolved round did not establish a valid frontier");
    }
    // Begin publishes a sampled token but does not execute it through the target. An exact-hit
    // Fork therefore still names an immutable read source and an unwritten destination here; the
    // first state-mutating decode commit closes it. A suffix prefill already closed its Fork at
    // the committed prefill frontier.
    if (request.pending.kind == PendingKind::Begin && terminal && sequence.state.fork_pending) {
        const StateImageSelectors selectors = state_selectors(sequence);
        timing.resume_submit();
        state_images->copy_slot(selectors.source, selectors.destination, device.stream);
        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        settle_state_fork(sequence);
    } else if (request.pending.kind == PendingKind::Ordinary) {
        settle_state_fork(sequence);
    }
    trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
    if (terminal) { sequence.mtp_draft_count = 0; }
    request.lifecycle = terminal ? Lifecycle::Finishable : Lifecycle::Active;
    request.pending   = {};
    return timing.finish();
}

MemorySummary ProgramImplCore::memory_summary() const noexcept {
    MemorySummary out;
    out.device           = device.device;
    out.max_context      = capacity;
    out.kv_capacity      = kv_capacity;
    out.kv_cache         = kv_storage;
    DeviceArena& weights = *model.weights_arena;
    out.weights = ArenaMemorySummary{weights.capacity(), weights.used(), weights.peak_used()};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
    }
    std::size_t active_workspace_bytes = work.used();
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        active_workspace_bytes =
            std::max(active_workspace_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), active_workspace_bytes,
                                       std::max(work.peak_used(), workspace_logical_peak_bytes)};
    if (workspace_plan.vision) {
        out.vision_workspace = VisionWorkspaceMemorySummary{
            .aggregate_prompt_tokens = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(capacity, kMaximumPromptVisionTokens)),
            .max_item_tokens        = workspace_plan.vision->max_merged_tokens,
            .general_capacity_bytes = workspace_plan.vision->general_capacity_bytes,
            .encode_peak_bytes      = workspace_plan.vision->encode_peak_bytes,
            .handoff_offset_bytes   = workspace_plan.vision->handoff_offset_bytes,
            .handoff_capacity_bytes = workspace_plan.vision->handoff_capacity_bytes,
            .handoff_active_bytes   = active_handoff_bytes,
            .handoff_peak_bytes     = vision_handoff_peak_bytes,
        };
    }
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes   = graph_allowance_bytes;
    out.kv_payload_bytes             = kv_payload_bytes;
    if (host_state_images) {
        out.host_state_capacity_slots = host_state_images->capacity();
        out.host_state_occupied_slots = host_state_images->occupied();
    }
    if (host_kv_arena) {
        out.host_kv_capacity_bytes = host_kv_arena->capacity_bytes();
        out.host_kv_occupied_bytes = host_kv_arena->occupied_bytes();
    }
    return out;
}

void ProgramImplCore::reset_memory_peaks() noexcept {
    model.weights_arena->reset_peak();
    persistent.reset_peak();
    work.reset_peak();
    std::size_t active_handoff_bytes = 0;
    for (const RequestControl& request : requests) {
        if (request.prefill && request.prefill->vision) {
            active_handoff_bytes =
                std::max(active_handoff_bytes, request.prefill->vision->active_handoff_bytes());
        }
    }
    vision_handoff_peak_bytes    = active_handoff_bytes;
    workspace_logical_peak_bytes = work.used();
    if (workspace_plan.vision && active_handoff_bytes != 0) {
        workspace_logical_peak_bytes =
            std::max(workspace_logical_peak_bytes,
                     workspace_plan.vision->handoff_offset_bytes + active_handoff_bytes);
    }
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
