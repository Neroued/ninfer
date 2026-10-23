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
    ContextDemand demand{.new_states = static_cast<std::uint32_t>(
                             candidate.state_action != StateBindingAction::Move ||
                             !state_store->device_resident(image)),
                         .logical_states = static_cast<std::uint32_t>(candidate.state_action !=
                                                                      StateBindingAction::Move)};
    demand.add_history(kTextKvPool.index, main_binding);
    if (backend_kv_histories) demand.add_history(kBackendKvPool.index, backend_binding);
    const auto releases = checkpoint_release_set(candidate.retired_points);
    if (!releases) return std::nullopt;
    if (!directory.assess(demand, *releases).ready() &&
        candidate.state_action == StateBindingAction::Fork &&
        state_store->source_pins(image) == 0) {
        if (state_store->host_resident(image)) {
            candidate.state_action = StateBindingAction::Split;
        } else if (checkpoints[handle->index].pins == 0) {
            if (host_context_enabled &&
                directory.plan_host(context::HostWriteRequirements{.state_images = 1}, *releases)
                    .ready()) {
                candidate.state_action = StateBindingAction::BackupSplit;
                candidate.transfers.push_back(context::state_transfer_requirement(
                    plan.state_transfer, runtime::ContextTransferDirection::DeviceToHost));
            } else if (consume_source && state_references == 1) {
                candidate.state_action = StateBindingAction::Move;
            }
        }
    }
    candidate.move_history =
        main_binding.reuse_history && (!backend_kv_histories || backend_binding.reuse_history);
    const auto restores = record.snapshot.restore_requirements(directory);
    candidate.transfers.insert(candidate.transfers.end(), restores.begin(), restores.end());
    if (state_store->device_resident(record.continuation().image) &&
        candidate.state_action == StateBindingAction::Fork &&
        is_masked_draft_backend(speculative_backend))
        candidate.transfers.push_back(context::state_transfer_requirement(
            plan.dflash_local_transfer.value(), runtime::ContextTransferDirection::DeviceToDevice));
    const auto tail = [&](const context::PageDirectory& pages, runtime::KvPoolId pool,
                          std::uint32_t copies) {
        if (copies)
            candidate.transfers.push_back(context::kv_transfer_requirement(
                {pool}, runtime::ContextTransferDirection::DeviceToDevice, copies,
                TransferWork{std::uint64_t(pages.layout().parts.front().record.bytes) * copies,
                             copies}));
    };
    tail(*text_kv_pages, kTextKvPool, main_binding.tail_copies);
    if (backend_kv_pages) tail(*backend_kv_pages, kBackendKvPool, backend_binding.tail_copies);
    return candidate;
}
} // namespace ninfer::models::qwen3_5::detail
