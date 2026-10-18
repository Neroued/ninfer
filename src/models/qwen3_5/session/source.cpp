#include "models/qwen3_5/session/internal.h"
#include "models/common/context/work.h"

#include <algorithm>

namespace ninfer::models::qwen3_5::detail {

bool SessionImpl::checkpoint_matches(CheckpointHandle handle, const RequestBasePlan& base) const {
    if (!valid_checkpoint(handle) || !base.impl_ || !base.impl_->allow_prefix_reuse) {
        return false;
    }
    const auto& record = checkpoint(handle);
    if (record.frontier() > base.summary().prompt_tokens ||
        record.key.identity_tag != base.impl_->prefix_identity_tag ||
        base.impl_->prefix_digests.at(record.frontier()) != record.key.digests) {
        return false;
    }
    return prefix_matches(*base.impl_->prompt, record.snapshot.identity().ledger,
                          record.snapshot.identity().prefix_identity, record.frontier());
}

std::optional<SourceCandidate>
SessionImpl::inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> handle,
                            bool consume_source, std::span<const CheckpointHandle> private_points,
                            std::span<const CheckpointHandle> retired_points) const {
    if (!base.impl_ || (handle && !checkpoint_matches(*handle, base))) { return std::nullopt; }
    SourceCandidate candidate{.checkpoint    = handle,
                              .reused_tokens = handle ? checkpoint(*handle).frontier() : 0};
    const auto& prompt       = *base.impl_->prompt;
    candidate.remaining_work = runtime::make_prefill_work(
        candidate.reused_tokens, base.summary().prompt_tokens - candidate.reused_tokens, 0, 0,
        prefill_chunk);
    for (const auto& item : prompt.vision_items) {
        if (std::any_of(item.token_spans.begin(), item.token_spans.end(), [&](const auto& span) {
                return span.begin + span.count > candidate.reused_tokens;
            })) {
            ++candidate.remaining_work.vision_items;
            candidate.remaining_work.vision_patches += item.patch_count;
        }
    }
    if (!handle) { return candidate; }
    const auto& record       = checkpoint(*handle);
    candidate.consume_source = consume_source;
    candidate.take_private   = consume_source;
    for (const auto point : retired_points) {
        if (point == handle || !can_release_checkpoint(point)) { return std::nullopt; }
        if (std::find(candidate.retired_points.begin(), candidate.retired_points.end(), point) ==
            candidate.retired_points.end()) {
            candidate.retired_points.push_back(point);
        }
    }
    const auto retires = [&](CheckpointHandle point) {
        return std::find(candidate.retired_points.begin(), candidate.retired_points.end(), point) !=
               candidate.retired_points.end();
    };
    auto state_references = state_store->checkpoint_references(record.continuation().image);
    for (const auto point : candidate.retired_points) {
        if (checkpoint(point).continuation().image == record.continuation().image) {
            --state_references;
        }
    }
    auto occupied_states = state_store->device_occupied();
    for (std::size_t i = 0; i < candidate.retired_points.size(); ++i) {
        const auto state = checkpoint(candidate.retired_points[i]).continuation().image;
        if (std::any_of(
                candidate.retired_points.begin(), candidate.retired_points.begin() + i,
                [&](auto point) { return checkpoint(point).continuation().image == state; })) {
            continue;
        }
        const auto references = static_cast<std::uint32_t>(std::count_if(
            candidate.retired_points.begin(), candidate.retired_points.end(),
            [&](auto point) { return checkpoint(point).continuation().image == state; }));
        if (references == state_store->checkpoint_references(state) &&
            state_store->device_resident(state) &&
            state_store->can_release_after_checkpoint_references(state, references)) {
            --occupied_states;
        }
    }
    const auto& rewrite = prompt.identity.rewrite_checkpoint;
    const auto desired  = rewrite ? rewrite->recovery_frontier : base.summary().prompt_tokens;
    for (const auto point : private_points) {
        if (!valid_checkpoint(point) || retires(point) || !checkpoint_matches(point, base)) {
            continue;
        }
        const auto& kept = checkpoint(point);
        if (kept.frontier() > record.frontier()) { continue; }
        const bool input =
            kept.role == runtime::CheckpointRole::InputReplay &&
            (kept.frontier() == desired || (point == *handle && kept.frontier() >= desired));
        if (input || kept.role == runtime::CheckpointRole::LongAnchor) {
            candidate.private_points.push_back(point);
        }
    }
    const bool keep_source =
        std::find(candidate.private_points.begin(), candidate.private_points.end(), *handle) !=
        candidate.private_points.end();
    const auto image       = record.continuation().image;
    candidate.state_action = state_store->device_resident(image) ? StateBindingAction::Fork
                                                                 : StateBindingAction::Restore;
    if (consume_source && !keep_source && checkpoints[handle->index].pins == 0 &&
        state_references == 1 && state_store->source_pins(image) == 0) {
        candidate.state_action = StateBindingAction::Move;
    } else if (occupied_states == state_store->device_capacity() &&
               state_store->device_resident(image) && state_store->source_pins(image) == 0) {
        if (state_store->host_resident(image)) {
            candidate.state_action = StateBindingAction::Split;
        } else if (checkpoints[handle->index].pins == 0) {
            if (host_context_arena &&
                host_context_arena->can_allocate(state_store->image_bytes())) {
                candidate.state_action = StateBindingAction::BackupSplit;
                candidate.transfers.push_back(context::state_transfer_requirement(
                    plan.state_transfer, runtime::ContextTransferDirection::DeviceToHost));
            } else if (consume_source && state_references == 1) {
                candidate.state_action = StateBindingAction::Move;
            }
        }
    }
    const auto backend_frontier = record.backend_frontier();
    const auto first            = prefill_unit(base.summary().prompt_tokens, record.frontier());
    std::vector<const context::HistoryPrefix*> retired_main, retired_backend;
    const auto retire = [&](CheckpointHandle point) {
        retired_main.push_back(&checkpoint(point).text_prefix());
        if (const auto* prefix = checkpoint(point).backend_prefix())
            retired_backend.push_back(prefix);
    };
    for (auto point : candidate.retired_points) retire(point);
    if (consume_source && (!keep_source || (candidate.state_action == StateBindingAction::Move)))
        retire(*handle);
    const auto main_binding = text_kv_histories->binding_demand(
        &record.text_prefix(), record.frontier(), std::max(record.frontier(), first.main_frontier),
        retired_main);
    const auto backend_binding =
        backend_kv_histories
            ? backend_kv_histories->binding_demand(
                  record.backend_prefix(), backend_frontier,
                  std::max(backend_frontier, first.backend_frontier), retired_backend)
            : context::HistoryBindingDemand{};
    candidate.move_history =
        main_binding.reuse_history && (!backend_kv_histories || backend_binding.reuse_history);
    if (!state_store->device_resident(record.continuation().image)) {
        candidate.transfers.push_back(context::state_transfer_requirement(
            plan.state_transfer, runtime::ContextTransferDirection::HostToDevice));
    } else if (candidate.state_action != StateBindingAction::Move &&
               candidate.state_action != StateBindingAction::Split &&
               candidate.state_action != StateBindingAction::BackupSplit &&
               is_masked_draft_backend(speculative_backend)) {
        candidate.transfers.push_back(context::state_transfer_requirement(
            plan.dflash_local_transfer.value(), runtime::ContextTransferDirection::DeviceToDevice));
    }
    const auto scan = [&](const context::HistoryDirectory& histories,
                          const context::PageDirectory& pages, context::HistoryId address,
                          std::uint32_t frontier, runtime::ContextTransferResource resource,
                          std::uint32_t tail_copies) {
        const auto count      = context::kv_pages_for_frontier(frontier);
        const auto& transfer  = pages.transfer_profile();
        std::uint32_t missing = 0, run_pages = 0;
        TransferWork restore_work;
        const auto finish_run = [&] {
            const auto work = transfer.host_transfer_run_work(run_pages);
            restore_work.payload_bytes += work.payload_bytes;
            restore_work.copy_operations += work.copy_operations;
            run_pages = 0;
        };
        std::optional<context::HostPageReplica> previous;
        for (std::uint32_t index = 0; index < count; ++index) {
            const auto page = histories.logical_page(address, index);
            if (pages.device_resident(page)) { continue; }
            const auto replica = pages.host_replica(page);
            if (previous && (previous->extent != replica.extent ||
                             previous->page_offset + 1 != replica.page_offset)) {
                finish_run();
            }
            previous = replica;
            ++run_pages;
            ++missing;
        }
        if (missing) {
            finish_run();
            candidate.transfers.push_back(context::kv_transfer_requirement(
                resource, runtime::ContextTransferDirection::HostToDevice, missing, restore_work));
        }
        if (tail_copies) {
            candidate.transfers.push_back(context::kv_transfer_requirement(
                resource, runtime::ContextTransferDirection::DeviceToDevice, 1,
                transfer.device_copy_work(1)));
        }
    };
    scan(*text_kv_histories, *text_kv_pages, record.history().text, record.frontier(),
         runtime::ContextTransferResource{kTextKvPool}, main_binding.tail_copies);
    if (record.history().backend) {
        scan(*backend_kv_histories, *backend_kv_pages, *record.history().backend, backend_frontier,
             runtime::ContextTransferResource{kBackendKvPool}, backend_binding.tail_copies);
    }
    return candidate;
}
} // namespace ninfer::models::qwen3_5::detail
