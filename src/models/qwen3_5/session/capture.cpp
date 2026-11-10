#include "models/qwen3_5/session/internal.h"
#include "models/common/context/work.h"
#include "runtime/engine/context_cache/capture_policy.h"

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

bool SessionImpl::reclaim_capture_reservation(const ContextAssessment& assessment) {
    for (auto& request : requests) {
        const auto& ticket = request.capture_reservation;
        if (!ticket) continue;
        const bool relevant =
            std::any_of(assessment.placement.domains.begin(), assessment.placement.domains.end(),
                        [&](const auto& item) {
                            return item.domain.memory == context::ContextMemory::Device
                                       ? state_store->device_resident(ticket->state)
                                       : ticket->host.has_value();
                        });
        if (relevant) {
            request.capture_reservation.reset();
            return true;
        }
    }
    return false;
}

SessionImpl::CaptureReservationStatus
SessionImpl::reserve_capture_destination(std::uint32_t lane, std::uint32_t frontier,
                                         std::span<const CheckpointHandle> victims,
                                         runtime::CaptureStorage storage) {
    using Status  = CaptureReservationStatus;
    auto& request = requests[lane];
    if (request.capture_reservation)
        return request.capture_reservation->frontier == frontier ? Status::Ready
                                                                 : Status::Unavailable;
    if (request.next_capture >= request.capture_groups.size()) return Status::Unavailable;
    const auto& group = request.capture_groups[request.next_capture];
    if (group.frontier != frontier) return Status::Unavailable;
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
            return Status::Unavailable;
        }
        ticket->points.emplace_back(*point, role);
    }
    if (victims.empty() && storage == runtime::CaptureStorage::Automatic) {
        const auto destination = context::reserve_capture_destination(directory, ticket->host);
        if (!destination) return Status::NeedsContext;
        ticket->state = *destination;
    } else {
        const auto released = checkpoint_release_set(victims);
        if (!released) return Status::NeedsContext;
        ContextDemand demand;
        const bool host   = storage == runtime::CaptureStorage::Host;
        const auto target = demand.add_state(
            {.count         = 1,
             .logical_count = 1,
             .memory = host ? context::ContextMemory::Host : context::ContextMemory::Device});
        if (!directory.assess(demand, *released).ready()) return Status::NeedsContext;
        for (const auto victim : victims)
            if (!release_checkpoint(victim))
                throw std::logic_error("capture victim changed after its complete assessment");
        auto backing = directory.reserve(demand);
        if (!backing || !backing.reservation->ready())
            throw std::logic_error("capture placement changed after its complete assessment");
        auto records           = backing.reservation->take(target);
        const auto destination = host ? state_store->reserve_logical_destination(records)
                                      : state_store->reserve_destination(std::move(records));
        if (!destination) throw std::logic_error("capture lost its reserved State descriptor");
        ticket->state = *destination;
        if (host) ticket->host.emplace(std::move(records));
    }
    request.capture_reservation = std::move(ticket);
    return Status::Ready;
}

std::optional<CapturePreparation>
SessionImpl::prepare_capture(SequenceHandle handle, std::span<const CheckpointHandle> victims,
                             runtime::CaptureStorage storage) {
    if (!valid_sequence(handle)) { throw std::logic_error("capture has a stale lane"); }
    const auto lane = this->lane(handle);
    auto& request   = requests[lane];
    if (!request.base->capture_backing || (!request.permit && !request.capture_pending)) return {};
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
                                  .reserved = true,
                                  .origin   = request.capture_groups[request.next_capture].origin,
                                  .evidence =
                                      request.capture_groups[request.next_capture].evidence};
    }
    const auto cursor = sequences[lane].continuation.metadata.execution_frontier;
    const auto end    = request.capture_pending ? cursor : request.permit->main_frontier;
    while (!request.capture_pending && request.next_capture < request.capture_groups.size() &&
           request.capture_groups[request.next_capture].frontier <= cursor)
        ++request.next_capture;
    const auto& advice = request.base->context_cache;
    if (!request.capture_pending && request.lifecycle == Lifecycle::Prefilling &&
        advice.automatic_checkpoints &&
        request.permit->main_frontier > request.last_intermediate_attempt) {
        const auto next  = request.next_capture < request.capture_groups.size()
                               ? std::optional{request.capture_groups[request.next_capture].frontier}
                               : std::nullopt;
        const auto point = runtime::intermediate_capture(
            cursor, request.permit->main_frontier, request.last_saved, next,
            advice.semantic_boundaries, context_cache.checkpoint_interval);
        if (point && request.base->accepts_capture(*point)) {
            auto group   = request.base->capture_group(*point);
            group.roles  = {runtime::CheckpointRole::SharedPrefix};
            group.origin = runtime::CheckpointOrigin::Intermediate;
            request.capture_groups.insert(request.capture_groups.begin() + request.next_capture,
                                          std::move(group));
            request.last_intermediate_attempt = request.permit->main_frontier;
        }
    }
    while (request.next_capture < request.capture_groups.size()) {
        const auto frontier = request.capture_groups[request.next_capture].frontier;
        if (frontier < cursor || (frontier == cursor && !request.capture_pending)) {
            ++request.next_capture;
            continue;
        }
        if (frontier > end) { return std::nullopt; }
        const auto status = reserve_capture_destination(lane, frontier, victims, storage);
        if (status == CaptureReservationStatus::Unavailable) {
            // Catalog exhaustion ends this optional publication; it is not a State placement quote.
            skip_capture(handle);
            continue;
        }
        if (status == CaptureReservationStatus::Ready) {
            return CapturePreparation{.frontier       = frontier,
                                      .reserved       = true,
                                      .retired_points = {victims.begin(), victims.end()},
                                      .origin = request.capture_groups[request.next_capture].origin,
                                      .evidence =
                                          request.capture_groups[request.next_capture].evidence};
        }
        CapturePreparation result{.frontier = frontier,
                                  .origin   = request.capture_groups[request.next_capture].origin,
                                  .evidence =
                                      request.capture_groups[request.next_capture].evidence};
        ContextDemand demand;
        demand.add_state({.count = 1});
        result.assessment        = directory.assess(demand);
        result.host.state_images = 1;
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

void SessionImpl::skip_capture(SequenceHandle handle, runtime::CaptureSkip reason) {
    if (!valid_sequence(handle)) { throw std::logic_error("capture has a stale lane"); }
    auto& request = requests[this->lane(handle)];
    if (!request.capture_pending && reason == runtime::CaptureSkip::Unavailable &&
        request.next_capture < request.capture_groups.size())
        request.retry_frontier = request.capture_groups[request.next_capture].frontier;
    request.capture_reservation.reset();
    if (reason == runtime::CaptureSkip::Retained &&
        request.next_capture < request.capture_groups.size())
        request.last_saved = request.capture_groups[request.next_capture].frontier;
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

bool SessionImpl::retry_capture(SequenceHandle handle) {
    if (!valid_sequence(handle)) throw std::logic_error("capture has a stale lane");
    const auto lane   = this->lane(handle);
    auto& request     = requests[lane];
    const auto& state = sequences[lane];
    const auto retry  = std::exchange(request.retry_frontier, {});
    if (!retry || request.capture_reservation || request.recovery || !request.next_capture ||
        *retry != state.continuation.metadata.execution_frontier ||
        *retry != state.continuation.metadata.text_kv_valid ||
        !state.continuation.metadata.tail_hidden_valid ||
        request.capture_groups[request.next_capture - 1].frontier != *retry)
        return false;
    --request.next_capture;
    request.capture_pending = true;
    return true;
}

std::vector<CheckpointHandle> SessionImpl::share_capture(SequenceHandle handle,
                                                         CheckpointHandle source) {
    if (!valid_sequence(handle)) throw std::logic_error("capture has a stale lane");
    auto& request = requests[this->lane(handle)];
    if (request.next_capture == request.capture_groups.size() || !valid_checkpoint(source))
        return {};
    const auto& group = request.capture_groups[request.next_capture];
    const auto& prior = checkpoint(source);
    if (prior.frontier() != group.frontier || prior.key != group.key ||
        !prefix_matches(*request.base->prompt, prior.snapshot.identity().ledger,
                        prior.snapshot.identity().prefix_identity, group.frontier))
        return {};
    if (request.capture_reservation &&
        state_store->device_resident(request.capture_reservation->state) &&
        !prior.snapshot.restore_requirements(directory).empty())
        return {};
    std::unique_ptr<CaptureReservation> prepared;
    auto* ticket = request.capture_reservation.get();
    if (!ticket) {
        prepared           = std::make_unique<CaptureReservation>();
        prepared->owner    = this;
        prepared->frontier = group.frontier;
        ticket             = prepared.get();
        ticket->points.reserve(group.roles.size());
        for (const auto role : group.roles) {
            const auto point = reserve_checkpoint();
            if (!point) return {};
            ticket->points.emplace_back(*point, role);
        }
    }
    std::vector<CheckpointState> copies;
    std::vector<CheckpointHandle> result;
    copies.reserve(ticket->points.size());
    result.reserve(ticket->points.size());
    for (const auto& [point, role] : ticket->points) {
        copies.push_back({.snapshot = prior.snapshot,
                          .role     = role,
                          .key      = group.key,
                          .origin   = group.origin,
                          .evidence = group.evidence});
        result.push_back(point);
    }
    for (std::size_t i = 0; i < result.size(); ++i)
        checkpoints.publish(result[i], std::move(copies[i]));
    ticket->points.clear();
    skip_capture(handle, runtime::CaptureSkip::Retained);
    return result;
}

runtime::ContextStart SessionImpl::start_capture(SequenceHandle handle) {
    if (has_context_operation() || execution) return runtime::ContextStart::Deferred;
    if (!valid_sequence(handle)) { throw std::logic_error("capture has a stale lane"); }
    const auto lane = this->lane(handle);
    auto& request   = requests[lane];
    auto& state     = sequences[lane];
    if (!request.capture_pending || request.next_capture >= request.capture_groups.size()) {
        return runtime::ContextStart::Stale;
    }
    if (state.pending_write.valid() || state.continuation.image != state.write_state()) {
        throw std::logic_error("capture boundary has an unsettled state writer");
    }
    const auto frontier = request.capture_groups[request.next_capture].frontier;
    if (frontier != state.continuation.metadata.text_kv_valid ||
        !state.continuation.metadata.tail_hidden_valid) {
        throw std::logic_error("capture boundary does not contain a complete model state");
    }
    if (reserve_capture_destination(lane, frontier) != CaptureReservationStatus::Ready) {
        skip_capture(handle);
        return runtime::ContextStart::Stale;
    }
    auto execution_reservation = executor().try_reserve(runtime::FrameKind::Context);
    if (!execution_reservation) return runtime::ContextStart::Deferred;
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
        ticket->owner   = nullptr;
        const bool host = ticket->host.has_value();
        context::prepare_capture_state(migration, *state_store, state.continuation.image,
                                       ticket->host);
        if (!host && is_masked_draft_backend(speculative_backend))
            migration.add_state_group(state.continuation.image, plan.dflash_local_group->offset,
                                      plan.dflash_local_group->bytes);
        (void)submit_context(*context_operation_, ContextWork{});
        return runtime::ContextStart::Started;
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
        checkpoints.publish(handle, CheckpointState{.snapshot = snapshot,
                                                    .role     = role,
                                                    .key      = group.key,
                                                    .origin   = group.origin,
                                                    .evidence = group.evidence});
    const auto writer =
        context::adopt_capture_writer(*state_store, source_state, migration.state_destination());
    state.continuation.image = writer.image;
    state.pending_write      = writer.write_destination;
    operations.state_forks += writer.write_destination.valid();
    request.last_saved = tx.frontier;
    skip_capture(sequence_handle(tx.lane));
}

} // namespace ninfer::models::qwen3_5::detail
