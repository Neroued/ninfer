#include "models/qwen3_5/session/internal.h"

#include <stdexcept>
#include <type_traits>

namespace ninfer::models::qwen3_5::detail {

bool SessionImpl::has_context_operation() const noexcept {
    return context_operation_ != nullptr || normalizing_pause_.has_value() || !releases.empty() ||
           release_ticket.has_value();
}

bool SessionImpl::context_blocks(SequenceHandle handle) const noexcept {
    if (normalizing_pause_ && *normalizing_pause_ == handle) return true;
    if (!context_operation_ || !valid_sequence(handle)) return false;
    const auto index = lane(handle);
    if (context_operation_->lane() == index) return true;
    if (!std::holds_alternative<BindOperation>(context_operation_->action)) return false;
    const auto& state = sequences[index];
    if (!state.kv) return false;
    const auto& transaction = context_operation_->transaction;
    const auto* text        = transaction.binding(kTextKvPool.index);
    const auto* backend     = transaction.binding(kBackendKvPool.index);
    return (text && text->blocks_writer(state.kv->text.handle())) ||
           (backend && state.kv->backend && backend->blocks_writer(state.kv->backend->handle()));
}

bool SessionImpl::submit_context(PendingContext& operation, ContextWork&& input) {
    if (!operation.reservation || operation.ticket)
        throw std::logic_error("context work requires its reserved execution frame");
    operation.transaction.prepare_actions();
    Work work{.context = operation.transaction.actions(), .operation = std::move(input)};
    operation.ticket = executor().submit(std::move(*operation.reservation), std::move(work));
    operation.reservation.reset();
    operation.transaction.mark_submitted();
    return true;
}

void SessionImpl::receive_context_result(runtime::ExecutionTicket ticket, ContextResult&& result) {
    if (!context_operation_ || context_operation_->ticket != ticket || context_operation_->result)
        throw std::logic_error("context result does not match its outstanding transaction");
    context_operation_->result.emplace(std::move(result));
}

void SessionImpl::cancel_context() noexcept {
    if (context_operation_) context_operation_->transaction.cancel();
    if (normalizing_pause_) pause_cancelled_ = true;
}

void SessionImpl::retire_context(runtime::DeviceRetired retired) {
    if (!context_operation_ || context_operation_->ticket != retired.ticket ||
        !context_operation_->result)
        throw std::logic_error("context completion has no matching result");
    auto& operation   = *context_operation_;
    auto& transaction = operation.transaction;
    transaction.mark_completed();
    ContextProgress out{.kind = operation.kind()};
    out.transfers                         = std::move(operation.result->transfers);
    out.operations.pressure_spill_pages   = transaction.spill_page_groups();
    out.operations.partial_tail_cow_pages = transaction.tail_copy_page_groups();
    for (const auto& transfer : out.transfers)
        if (transfer.resource.is_state() &&
            transfer.direction == runtime::ContextTransferDirection::HostToDevice)
            out.operations.state_restores += transfer.image_count;
    if (auto* pause = std::get_if<PauseOperation>(&operation.action)) {
        out.request_speculative = pause->paused.impl_->control.speculative_stats;
    }
    if (transaction.cancelled()) {
        abort_context();
        out.complete = true;
    } else {
        transaction.publish();
        std::visit(
            [&](auto& action) {
                using Action = std::decay_t<decltype(action)>;
                if constexpr (std::is_same_v<Action, BindOperation>) {
                    complete_binding(action, transaction, out);
                } else if constexpr (std::is_same_v<Action, CaptureOperation>) {
                    out.captured_checkpoints.reserve(action.points.size());
                    publish_capture(action, transaction, out.operations);
                    for (const auto& [handle, role] : action.points)
                        out.captured_checkpoints.push_back(handle);
                } else if constexpr (std::is_same_v<Action, PauseOperation>) {
                    out.paused.emplace(complete_pause(action));
                }
            },
            operation.action);
        out.complete = out.published = true;
        context_operation_.reset();
    }
    events.push_back({.ticket = retired.ticket, .result = std::move(out)});
}

void SessionImpl::abort_context() noexcept {
    normalizing_pause_.reset();
    pause_reservation_.reset();
    pause_cancelled_ = false;
    if (!context_operation_) return;
    auto& operation = *context_operation_;
    try {
        if (operation.transaction.in_flight()) {
            executor().drain();
            operation.transaction.mark_completed();
        }
        operation.transaction.discard();
        std::visit(
            [&](auto& action) {
                using Action = std::decay_t<decltype(action)>;
                if constexpr (std::is_same_v<Action, BindOperation>) {
                    auto& state = sequences[action.lane];
                    if (!closing && operation.ticket && state.residency.generation)
                        release_residency(state.residency);
                    release_sequence_state(state);
                    release_sequence_kv(state);
                    requests[action.lane] = {};
                    invalidate_lane(action.lane);
                    action.source_lease.reset();
                    action.carried_pins.clear();
                    for (const auto& [from, to] : action.carried_clones) {
                        if (valid_checkpoint(to))
                            (void)release_checkpoint(to);
                        else
                            checkpoints.cancel(to);
                    }
                } else if constexpr (std::is_same_v<Action, CaptureOperation>) {
                    auto& state = sequences[action.lane];
                    if (state_store->role(state.continuation.image) ==
                            context::StateRole::CheckpointImmutable &&
                        !state_store->checkpoint_references(state.continuation.image))
                        state_store->make_mutable(state.continuation.image);
                    for (const auto& [handle, role] : action.points)
                        if (!valid_checkpoint(handle)) checkpoints.cancel(handle);
                } else if constexpr (std::is_same_v<Action, PauseOperation>) {
                    requests[action.lane] = {};
                    invalidate_lane(action.lane);
                }
            },
            operation.action);
        context_operation_.reset();
    } catch (...) { std::terminate(); }
}

} // namespace ninfer::models::qwen3_5::detail
