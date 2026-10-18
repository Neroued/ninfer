#include "models/qwen3_5/session/internal.h"
#include "models/common/context/work.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {
bool has_role(const CaptureGroup& group, runtime::CheckpointRole role) {
    return std::find(group.roles.begin(), group.roles.end(), role) != group.roles.end();
}
} // namespace

CaptureReservation::~CaptureReservation() {
    if (!owner) { return; }
    if (owner->state_store->valid(state) && !owner->state_store->release(state)) {
        std::terminate();
    }
    for (const auto& [handle, role] : points) { owner->checkpoints.cancel(handle); }
}

bool SessionImpl::reclaim_capture_reservation(ContextResources shortage) {
    for (auto& request : requests) {
        const auto& ticket = request.capture_reservation;
        if (!ticket) { continue; }
        const bool relevant =
            (shortage.state_slots && state_store->device_resident(ticket->state)) ||
            (shortage.host_bytes && ticket->host);
        if (relevant) {
            request.capture_reservation.reset();
            return true;
        }
    }
    return false;
}

bool SessionImpl::reserve_capture_destination(std::uint32_t lane, std::uint32_t frontier) {
    auto& request = requests[lane];
    if (request.capture_reservation) { return request.capture_reservation->frontier == frontier; }
    if (request.next_capture >= request.capture_groups.size()) { return false; }
    const auto& group = request.capture_groups[request.next_capture];
    if (group.frontier != frontier) { return false; }
    auto ticket      = std::make_unique<CaptureReservation>();
    ticket->owner    = this;
    ticket->frontier = frontier;
    ticket->points.reserve(group.roles.size());
    for (const auto role : group.roles) {
        const auto point = reserve_checkpoint();
        if (!point) {
            if (role == runtime::CheckpointRole::SharedPrefix && !ticket->points.empty()) {
                continue;
            }
            return false;
        }
        ticket->points.emplace_back(*point, role);
    }
    auto destination = state_store->reserve_destination();
    if (!destination && host_context_arena) {
        ticket->host = host_context_arena->allocate(state_store->image_bytes());
        if (ticket->host) { destination = state_store->reserve_logical_destination(); }
    }
    if (!destination) { return false; }
    ticket->state               = *destination;
    request.capture_reservation = std::move(ticket);
    return true;
}

std::optional<CapturePreparation> SessionImpl::prepare_capture(SequenceHandle handle) {
    if (!valid_sequence(handle)) { throw std::logic_error("capture has a stale lane"); }
    const auto lane = this->lane(handle);
    auto& request   = requests[lane];
    if (!request.base->capture_backing || !request.permit) { return std::nullopt; }
    if (request.recovery) {
        // Recovery owns enough resources to regain old progress and execute new work.
        // Optional saves cannot add a State/COW requirement to that finite guarantee.
        request.capture_reservation.reset();
        while (request.next_capture < request.capture_groups.size() &&
               request.capture_groups[request.next_capture].frontier <=
                   request.permit->main_frontier) {
            ++request.next_capture;
        }
        return std::nullopt;
    }
    if (request.capture_reservation) {
        return CapturePreparation{.frontier = request.capture_reservation->frontier,
                                  .reserved = true};
    }
    const auto cursor =
        request.lifecycle == Lifecycle::Replaying ? request.replay_cursor : request.prefill->cursor;
    while (request.next_capture < request.capture_groups.size()) {
        const auto frontier = request.capture_groups[request.next_capture].frontier;
        if (frontier <= cursor) {
            ++request.next_capture;
            continue;
        }
        if (frontier > request.permit->main_frontier) { return std::nullopt; }
        if (reserve_capture_destination(lane, frontier)) {
            return CapturePreparation{.frontier = frontier, .reserved = true};
        }
        CapturePreparation result{.frontier = frontier};
        if (state_store->device_occupied() == state_store->device_capacity()) {
            result.shortage.state_slots = 1;
            result.host_bytes           = state_store->image_bytes();
        }
        return result;
    }
    return std::nullopt;
}

void SessionImpl::prepare_capture_boundary(std::uint32_t lane) {
    while (const auto prepared = prepare_capture(sequence_handle(lane))) {
        if (prepared->reserved) { return; }
        // Optional snapshots do not move required input identity boundaries.
        skip_capture(sequence_handle(lane));
    }
}

void SessionImpl::skip_capture(SequenceHandle handle) {
    if (!valid_sequence(handle)) { throw std::logic_error("capture has a stale lane"); }
    auto& request = requests[this->lane(handle)];
    request.capture_reservation.reset();
    if (!request.capture_pending) {
        if (request.next_capture < request.capture_groups.size()) { ++request.next_capture; }
        return;
    }
    request.capture_pending = false;
    ++request.next_capture;
    if (request.lifecycle != Lifecycle::Replaying && request.lifecycle != Lifecycle::Prefilling &&
        request.prefill && request.prefill->cursor == request.prefill->prompt_tokens) {
        request.prefill.reset();
    }
}

bool SessionImpl::capture_is_input(SequenceHandle handle) const {
    if (!valid_sequence(handle)) { throw std::logic_error("capture has a stale lane"); }
    const auto& request = requests[this->lane(handle)];
    if (request.next_capture >= request.capture_groups.size()) { return false; }
    const auto& group = request.capture_groups[request.next_capture];
    return has_role(group, runtime::CheckpointRole::InputReplay) &&
           (request.capture_pending ||
            (request.permit && group.frontier <= request.permit->main_frontier));
}

bool SessionImpl::start_capture(SequenceHandle handle) {
    if (has_context_operation() || execution) { return false; }
    if (!valid_sequence(handle)) { throw std::logic_error("capture has a stale lane"); }
    const auto lane = this->lane(handle);
    auto& request   = requests[lane];
    auto& state     = sequences[lane];
    if (!request.capture_pending || request.next_capture >= request.capture_groups.size()) {
        return false;
    }
    if (state.pending_write.valid() || state.continuation.image != state.write_state()) {
        return false;
    }
    const auto frontier = request.capture_groups[request.next_capture].frontier;
    if (frontier != state.continuation.metadata.text_kv_valid ||
        !state.continuation.metadata.tail_hidden_valid) {
        return false;
    }
    if (!reserve_capture_destination(lane, frontier)) { return false; }
    auto execution_reservation = executor().try_reserve(runtime::FrameKind::Context);
    if (!execution_reservation) return false;
    auto ticket = std::move(request.capture_reservation);
    CaptureOperation tx{.lane             = lane,
                        .frontier         = frontier,
                        .backend_frontier = speculative_backend == SpeculativeBackend::Mtp
                                                ? frontier - 1U
                                                : backend_kv_valid(state),
                        .points           = std::move(ticket->points)};
    context_operation_ = std::make_unique<PendingContext>(directory, std::move(tx));
    context_operation_->reservation.emplace(std::move(*execution_reservation));
    auto& migration = context_operation_->transaction;
    try {
        migration.own_state(ticket->state);
        ticket->owner = nullptr;
        state_store->freeze(state.continuation.image);
        if (ticket->host) {
            auto transfer = state_store->reserve_device_to_host(state.continuation.image,
                                                                std::move(*ticket->host));
            ticket->host.reset();
            if (!transfer) throw std::logic_error("capture lost its reserved Host state space");
            migration.add_state_copy(std::move(*transfer), true);
        } else if (is_masked_draft_backend(speculative_backend)) {
            migration.add_state_group(state.continuation.image, plan.dflash_local_group->offset,
                                      plan.dflash_local_group->bytes);
        }
        (void)submit_context(*context_operation_, ContextWork{});
        return true;
    } catch (...) {
        abort_context();
        throw;
    }
}

void SessionImpl::publish_capture(CaptureOperation& tx, context::ContextTransaction& migration,
                                  runtime::ContextOperationCounts& operations) {
    auto& state                          = sequences[tx.lane];
    auto& request                        = requests[tx.lane];
    const auto& group                    = request.capture_groups[request.next_capture];
    const auto source_state              = state.continuation.image;
    Continuation captured                = state.continuation;
    captured.metadata.execution_frontier = tx.frontier;
    captured.metadata.backend_kv_valid   = tx.backend_frontier;
    std::vector<context::HistoryPrefix> prefixes;
    prefixes.push_back(text_kv_histories->retain_prefix(state.kv->text.handle(), tx.frontier));
    if (state.kv->backend)
        prefixes.push_back(
            backend_kv_histories->retain_prefix(state.kv->backend->handle(), tx.backend_frontier));
    const Snapshot snapshot(*state_store, captured, std::move(prefixes), group.identity);
    for (const auto& [handle, role] : tx.points)
        checkpoints.publish(handle,
                            CheckpointState{.snapshot = snapshot, .role = role, .key = group.key});
    if (state_store->residency(migration.state_destination().image()) ==
        context::StateResidency::None) {
        state_store->split_device_replica_identity(source_state,
                                                   migration.state_destination().image());
        state_store->complete_initialization(migration.state_destination().image());
        state.continuation.image = migration.state_destination().image();
        state.pending_write      = {};
    } else {
        ++operations.state_forks;
        (void)migration.state_destination().begin_write(source_state);
        state.continuation.image = source_state;
        state.pending_write      = migration.state_destination().image();
    }
    (void)migration.state_destination().release();
    skip_capture(sequence_handle(tx.lane));
}

} // namespace ninfer::models::qwen3_5::detail
