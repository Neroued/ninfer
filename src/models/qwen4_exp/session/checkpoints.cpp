#include "models/qwen4_exp/session/internal.h"

#include "runtime/engine/context_cache/capture_policy.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

std::optional<CheckpointHandle> SessionImpl::detach(Sequence& s, runtime::CheckpointRole role) {
    const auto f = s.continuation.metadata.main_frontier;
    if (!s.request->can_checkpoint(f)) return {};
    auto identity = capture_identity(s);
    std::vector<context::HistoryPrefix> prefixes;
    for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool)
        prefixes.push_back(directory.pool(pool).histories().retain_prefix(
            s.histories[pool].handle(),
            pool == 0 ? f : s.continuation.metadata.mtp_frontier.value()));
    const auto record = checkpoints.reserve();
    if (!record) return {};
    auto& states     = directory.states();
    const auto image = s.continuation.image;
    if (states.role(image) == context::StateRole::ActiveMutable) states.freeze(image);
    checkpoints.publish(
        *record,
        {Snapshot(states, s.continuation, std::move(prefixes), std::move(identity)), role});
    if (s.write_destination.valid()) {
        states.abort_write(image, s.write_destination);
        if (!states.release(s.write_destination))
            throw std::logic_error("unused state writer remains pinned");
        s.write_destination = {};
    }
    s.continuation.image = {};
    for (auto& history : s.histories) history.reset();
    return record;
}

} // namespace ninfer::models::qwen4_exp::detail

namespace ninfer::models::qwen4_exp {

std::optional<CapturePreparation> Session::prepare_capture(SequenceHandle h) {
    auto& p = *impl_;
    auto& s = p.sequence(h);
    if (!s.permit || !s.request->summary.publish_continuation) return {};
    if (s.recovery_peak) {
        s.capture.reset();
        while (s.next_capture < s.captures.size() &&
               s.captures[s.next_capture].frontier <= s.permitted_frontier)
            ++s.next_capture;
        return {};
    }
    if (s.capture)
        return CapturePreparation{.frontier = s.captures[s.next_capture].frontier,
                                  .reserved = true,
                                  .origin   = s.captures[s.next_capture].origin};
    while (s.next_capture < s.captures.size() &&
           s.captures[s.next_capture].frontier <= s.continuation.metadata.main_frontier)
        ++s.next_capture;
    const auto begin   = s.continuation.metadata.main_frontier;
    const auto& advice = s.request->prompt.context_cache;
    if (s.phase == detail::Phase::Prefill && advice.automatic_checkpoints &&
        s.permitted_frontier > s.last_intermediate_attempt) {
        const auto next = s.next_capture < s.captures.size()
                              ? std::optional{s.captures[s.next_capture].frontier}
                              : std::nullopt;
        const auto point =
            runtime::intermediate_capture(begin, s.permitted_frontier, s.last_saved, next,
                                          advice.semantic_boundaries, p.plan.checkpoint_interval);
        if (point && s.request->can_checkpoint(*point)) {
            s.captures.insert(s.captures.begin() + s.next_capture,
                              {*point,
                               {runtime::CheckpointRole::SharedPrefix},
                               runtime::CheckpointOrigin::Intermediate});
            s.last_intermediate_attempt = s.permitted_frontier;
        }
    }
    if (s.next_capture == s.captures.size()) return {};
    const auto& point = s.captures[s.next_capture];
    if (point.frontier > s.permitted_frontier) return {};
    auto ticket         = std::make_unique<detail::CaptureReservation>();
    ticket->states      = &p.directory.states();
    ticket->checkpoints = &p.checkpoints;
    ticket->records.reserve(point.roles.size());
    for (const auto role : point.roles) {
        const auto record = p.checkpoints.reserve();
        if (!record) return CapturePreparation{.frontier = point.frontier, .origin = point.origin};
        ticket->records.emplace_back(*record, role);
    }
    auto destination = context::reserve_capture_destination(
        *ticket->states, p.directory.host_allocation(), ticket->host);
    if (!destination)
        return CapturePreparation{point.frontier,
                                  false,
                                  {.state_slots = 1},
                                  p.directory.states().image_bytes(),
                                  point.origin};
    ticket->destination = *destination;
    s.capture           = std::move(ticket);
    return CapturePreparation{.frontier = point.frontier, .reserved = true, .origin = point.origin};
}

std::vector<std::uint32_t> Session::capture_frontiers(SequenceHandle h) const {
    std::vector<std::uint32_t> result;
    for (const auto& point : impl_->sequence(h).captures) result.push_back(point.frontier);
    return result;
}

bool Session::capture_is_shared(SequenceHandle h) const {
    const auto& s = impl_->sequence(h);
    if (s.next_capture >= s.captures.size()) return false;
    const auto& roles = s.captures[s.next_capture].roles;
    return roles.size() == 1 && roles.front() == runtime::CheckpointRole::SharedPrefix;
}

bool Session::capture_is_input(SequenceHandle h) const {
    const auto& s = impl_->sequence(h);
    if (s.next_capture >= s.captures.size()) return false;
    const auto& point = s.captures[s.next_capture];
    return (s.capture_pending || (s.permit && point.frontier <= s.permitted_frontier)) &&
           std::find(point.roles.begin(), point.roles.end(),
                     runtime::CheckpointRole::InputReplay) != point.roles.end();
}

void Session::skip_capture(SequenceHandle h, runtime::CaptureSkip reason) {
    auto& s = impl_->sequence(h);
    s.capture.reset();
    s.capture_pending = false;
    if (reason == runtime::CaptureSkip::Retained && s.next_capture < s.captures.size())
        s.last_saved = s.captures[s.next_capture].frontier;
    if (s.next_capture < s.captures.size()) ++s.next_capture;
}

bool Session::reclaim_capture_reservation(ContextResources shortage) {
    for (auto& s : impl_->sequences) {
        if (!s || !s->capture || s->capture_pending) continue;
        if ((shortage.state_slots &&
             impl_->directory.states().device_resident(s->capture->destination)) ||
            (shortage.host_bytes && s->capture->host)) {
            s->capture.reset();
            return true;
        }
    }
    return false;
}

bool Session::start_capture(SequenceHandle h) {
    auto& p = *impl_;
    auto& s = p.sequence(h);
    if (p.operation || execution_pending(h) || !s.capture_pending || !s.capture ||
        s.write_destination.valid())
        return false;
    auto execution = p.executor->try_reserve(runtime::FrameKind::Context);
    if (!execution) return false;
    p.operation = std::make_unique<detail::ContextOperation>(
        p.directory, detail::CaptureOperation{s.lane, std::move(s.capture)}, std::move(*execution));
    auto& migration   = p.operation->migration;
    auto& tx          = std::get<detail::CaptureOperation>(p.operation->action);
    auto& reservation = *tx.reservation;
    try {
        migration.own_state(reservation.destination);
        reservation.destination = {};
        std::vector<context::HistoryPrefix> prefixes;
        for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool)
            prefixes.push_back(p.directory.pool(pool).histories().retain_prefix(
                s.histories[pool].handle(), pool == 0
                                                ? s.continuation.metadata.main_frontier
                                                : s.continuation.metadata.mtp_frontier.value()));
        context::prepare_capture_state(migration, p.directory.states(), s.continuation.image,
                                       reservation.host);
        tx.snapshot.emplace(p.directory.states(), s.continuation, std::move(prefixes),
                            s.request->identity);
        p.submit_context();
    } catch (...) {
        p.cancel_operation();
        throw;
    }
    return true;
}

std::optional<std::size_t> Session::pause_host_bytes(SequenceHandle h) const {
    const auto& p = *impl_;
    const auto& s = p.sequence(h);
    if (!s.request->can_checkpoint(s.continuation.metadata.main_frontier) ||
        !p.directory.host_allocation().capacity_bytes())
        return {};
    std::size_t bytes = p.directory.states().host_resident(s.continuation.image)
                            ? 0
                            : p.directory.states().image_bytes();
    for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool) {
        const auto& storage = p.directory.pool(pool);
        const auto frontier = pool == 0 ? s.continuation.metadata.main_frontier
                                        : s.continuation.metadata.mtp_frontier.value();
        for (std::uint32_t i = 0; i < (frontier + 63) / 64; ++i) {
            const auto page = storage.histories().logical_page(s.histories[pool].handle(), i);
            if (storage.pages().active_history_references(page) <= 1 &&
                !storage.pages().host_replica_current(page))
                bytes += p.pools[pool].host_page_stride;
        }
    }
    return bytes;
}

bool Session::start_pause(SequenceHandle h, bool save) {
    auto& p = *impl_;
    if (p.operation || execution_pending(h)) return false;
    auto& s = p.sequence(h);
    if (s.phase != detail::Phase::Prefill && s.phase != detail::Phase::Replay &&
        s.phase != detail::Phase::Active)
        throw std::logic_error("pause requires a committed boundary");
    if (save) {
        const auto bytes = pause_host_bytes(h);
        if (!bytes || *bytes > p.directory.host_allocation().free_bytes()) return false;
    }
    auto execution = p.executor->try_reserve(runtime::FrameKind::Context);
    if (!execution) return false;
    auto paused   = std::make_unique<detail::Suspended>();
    paused->owner = &p;
    paused->frontier =
        s.phase == detail::Phase::Replay ? s.replay_target : s.continuation.metadata.main_frontier;
    p.operation = std::make_unique<detail::ContextOperation>(
        p.directory, detail::PauseOperation{s.lane, std::move(paused)}, std::move(*execution));
    auto& tx        = std::get<detail::PauseOperation>(p.operation->action);
    auto& migration = p.operation->migration;
    try {
        s.capture.reset();
        s.capture_pending = false;
        s.recovery_peak   = 0;
        p.settle(s);
        if (s.phase == detail::Phase::Replay) s.phase = s.resume_phase;
        if (save) tx.paused->snapshot = p.detach(s, runtime::CheckpointRole::Continuation);
        if (!tx.paused->snapshot) p.clear(s);
        tx.paused->sequence = std::move(p.sequences[s.lane]);
        if (tx.paused->snapshot) {
            const auto demotion =
                p.demotion({&*tx.paused->snapshot, 1},
                           {.state_slots = 1, .kv_page_groups = {UINT32_MAX, UINT32_MAX}});
            if (demotion && !p.prepare_demotion(migration, *demotion)) {
                migration.reset_plan();
                if (!p.checkpoints.erase(*tx.paused->snapshot))
                    throw std::logic_error("failed pause backup still has readers");
                tx.paused->snapshot.reset();
            }
        }
        p.submit_context();
        return true;
    } catch (...) {
        p.cancel_operation();
        throw;
    }
}

bool Session::revoke_snapshot(ResumeState& state) noexcept {
    if (!state.state_ || state.state_->owner != impl_.get() || !state.state_->snapshot)
        return false;
    if (!impl_->checkpoints.erase(*state.state_->snapshot)) return false;
    state.state_->snapshot.reset();
    return true;
}

ContextResources Session::snapshot_resources(const ResumeState& s) const {
    return s.has_snapshot() ? impl_->footprint({&*s.state_->snapshot, 1}, true)
                            : ContextResources{};
}

} // namespace ninfer::models::qwen4_exp
