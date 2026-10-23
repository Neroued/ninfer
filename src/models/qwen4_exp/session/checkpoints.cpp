#include "models/qwen4_exp/session/internal.h"

#include "runtime/engine/context_cache/capture_policy.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

PreparedCheckpoint SessionImpl::prepare_checkpoint(Sequence& s) {
    const auto frontier = s.continuation.metadata.main_frontier;
    if (!s.request->can_checkpoint(frontier)) return {};
    auto record = checkpoints.reserve_owned();
    if (!record.valid()) return {};
    ContextDemand demand;
    for (std::uint32_t pool = 0; pool < pool_count(); ++pool)
        demand.add_growth(directory.pool(pool).histories(), s.histories[pool].handle(), 0);
    if (!directory.reserve(demand)) return {};
    return {std::move(record), capture_identity(s)};
}

std::optional<CheckpointHandle> SessionImpl::detach(Sequence& s, runtime::CheckpointRole role,
                                                    PreparedCheckpoint prepared) {
    const auto f = s.continuation.metadata.main_frontier;
    if (!prepared.record.valid()) prepared = prepare_checkpoint(s);
    if (!prepared.record.valid()) return {};
    std::vector<context::HistoryPrefix> prefixes;
    for (std::uint16_t pool = 0; pool < directory.pool_count(); ++pool)
        prefixes.push_back(directory.pool(pool).histories().retain_prefix(
            s.histories[pool].handle(),
            pool == 0 ? f : s.continuation.metadata.mtp_frontier.value()));
    auto& states     = directory.states();
    const auto image = s.continuation.image;
    if (states.role(image) == context::StateRole::ActiveMutable) states.freeze(image);
    const auto record = prepared.record.publish(
        {Snapshot(states, s.continuation, std::move(prefixes), std::move(prepared.identity)),
         role});
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

std::optional<CapturePreparation>
Session::prepare_capture(SequenceHandle h, std::span<const CheckpointHandle> host_victims) {
    auto& p = *impl_;
    auto& s = p.sequence(h);
    if (p.directory.maintenance_pending()) return {};
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
        if (!record) return {};
        ticket->records.emplace_back(*record, role);
    }
    std::vector<CheckpointHandle> retired;
    std::optional<context::StateId> destination;
    if (host_victims.empty()) {
        destination = context::reserve_capture_destination(p.directory, ticket->host);
    } else {
        for (const auto victim : host_victims)
            if (!p.checkpoints.erasable(victim)) return {};
        const auto released   = p.footprint(host_victims, true);
        const auto assessment = p.directory.plan_host({.state_images = 1}, released);
        if (!assessment.ready())
            return CapturePreparation{.frontier   = point.frontier,
                                      .assessment = assessment,
                                      .host       = {.state_images = 1},
                                      .origin     = point.origin};
        destination = p.directory.states().reserve_logical_destination();
        if (!destination) return {};
        ticket->destination = *destination;
        for (const auto victim : host_victims) {
            if (!p.checkpoints.erase(victim))
                throw std::logic_error("capture Host release changed after its assessment");
            retired.push_back(victim);
        }
        auto reserved = p.directory.reserve(assessment.demand);
        if (!reserved || !reserved.reservation->ready())
            throw std::logic_error("capture Host placement changed after its assessment");
        ticket->host.emplace(reserved.reservation->take_state());
    }
    if (!destination)
        return CapturePreparation{point.frontier,
                                  false,
                                  p.directory.assess({.new_states = 1}),
                                  {.state_images = 1},
                                  point.origin};
    ticket->destination = *destination;
    s.capture           = std::move(ticket);
    return CapturePreparation{.frontier       = point.frontier,
                              .reserved       = true,
                              .origin         = point.origin,
                              .retired_points = std::move(retired)};
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

bool Session::reclaim_capture_reservation(const ContextAssessment& assessment) {
    if (assessment.ready()) return false;
    for (auto& sequence : impl_->sequences) {
        if (!sequence || !sequence->capture || sequence->capture_pending) continue;
        sequence->capture.reset();
        return true;
    }
    return false;
}

bool Session::start_capture(SequenceHandle h) {
    auto& p = *impl_;
    auto& s = p.sequence(h);
    if (p.directory.maintenance_pending() || p.operation || execution_pending(h) ||
        !s.capture_pending || !s.capture || s.write_destination.valid())
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

std::optional<Session::HostRequirement> Session::pause_host_requirements(SequenceHandle h) const {
    const auto& p = *impl_;
    const auto& s = p.sequence(h);
    if (!s.request->can_checkpoint(s.continuation.metadata.main_frontier) ||
        std::none_of(p.plan.context.ranks.begin(), p.plan.context.ranks.end(),
                     [](const auto& rank) { return rank.host_bytes != 0; }))
        return {};
    HostRequirement write;
    write.state_images = p.directory.states().host_resident(s.continuation.image) ? 0 : 1;
    write.kv_pages.resize(p.directory.pool_count());
    for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool) {
        const auto& storage = p.directory.pool(pool);
        const auto frontier = pool == 0 ? s.continuation.metadata.main_frontier
                                        : s.continuation.metadata.mtp_frontier.value();
        for (std::uint32_t i = 0; i < (frontier + 63) / 64; ++i) {
            const auto page = storage.histories().logical_page(s.histories[pool].handle(), i);
            if (storage.pages().active_history_references(page) <= 1 &&
                !storage.pages().host_replica_current(page))
                ++write.kv_pages[pool];
        }
    }
    return write;
}

bool Session::prepare_pause(SequenceHandle h, bool save,
                            std::span<const CheckpointHandle> host_victims) {
    auto& p = *impl_;
    if (p.operation || execution_pending(h)) return false;
    if (p.pause_reservation) throw std::logic_error("pause preparation is already owned");
    const auto& s = p.sequence(h);
    if (s.phase != detail::Phase::Prefill && s.phase != detail::Phase::Replay &&
        s.phase != detail::Phase::Active)
        throw std::logic_error("pause requires a committed boundary");
    const auto write = save ? pause_host_requirements(h) : std::optional(HostRequirement{});
    if (!write) return false;
    ContextReleaseSet released;
    if (!host_victims.empty()) {
        for (const auto victim : host_victims)
            if (!p.checkpoints.erasable(victim)) return false;
        released = p.footprint(host_victims, true);
    }
    const auto assessment = p.directory.plan_host(*write, released);
    if (!assessment.feasible()) return false;
    if (assessment.requires_maintenance() && (p.compute || p.executor->has_pending_work()))
        return false;
    auto execution = p.executor->try_reserve(runtime::FrameKind::Context);
    if (!execution) return false;
    detail::PreparedCheckpoint checkpoint;
    auto& sequence = p.sequence(h);
    if (save && sequence.request->can_checkpoint(sequence.continuation.metadata.main_frontier)) {
        checkpoint = p.prepare_checkpoint(sequence);
        if (!checkpoint.record.valid()) return false;
    }
    p.pause_reservation.emplace(detail::SessionImpl::PauseReservation{
        h, save, p.directory.host_demand(*write), std::move(*execution), std::move(checkpoint)});
    return true;
}

bool Session::start_pause(SequenceHandle h, bool save) {
    auto& p = *impl_;
    if (p.operation || execution_pending(h)) return false;
    if (!p.pause_reservation && !prepare_pause(h, save)) return false;
    if (p.pause_reservation->sequence != h || p.pause_reservation->save_snapshot != save)
        throw std::logic_error("pause does not match its preparation");
    auto targets = p.directory.reserve(p.pause_reservation->demand);
    if (!targets) throw std::logic_error("prepared pause placement changed before reservation");
    auto execution  = std::move(p.pause_reservation->execution);
    auto checkpoint = std::move(p.pause_reservation->checkpoint);
    p.pause_reservation.reset();
    auto& s       = p.sequence(h);
    auto paused   = std::make_unique<detail::Suspended>();
    paused->owner = &p;
    paused->frontier =
        s.phase == detail::Phase::Replay ? s.replay_target : s.continuation.metadata.main_frontier;
    p.operation = std::make_unique<detail::ContextOperation>(
        p.directory, detail::PauseOperation{s.lane, std::move(paused), save, std::move(checkpoint)},
        std::move(execution));
    try {
        if (!targets.reservation->ready()) {
            p.operation->migration.discard();
            auto maintenance = p.directory.prepare_maintenance(std::move(*targets.reservation),
                                                               targets.assessment);
            auto previous    = std::move(p.operation);
            p.operation      = std::make_unique<detail::ContextOperation>(
                std::move(maintenance), std::move(previous->action),
                std::move(previous->reservation));
            p.operation->preparing_resources = true;
        } else {
            p.prepare_pause_resources(*targets.reservation);
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


} // namespace ninfer::models::qwen4_exp
