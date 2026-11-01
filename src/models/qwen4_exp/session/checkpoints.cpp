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
        const auto released = p.footprint(host_victims, true);
        ContextDemand demand;
        const auto targets    = demand.add_host({.state_images = 1});
        const auto assessment = p.directory.assess(demand, released);
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
        ticket->host.emplace(reserved.reservation->take(targets.state));
    }
    if (!destination) {
        ContextDemand demand;
        demand.add_state({.count = 1});
        return CapturePreparation{
            point.frontier, false, p.directory.assess(demand), {.state_images = 1}, point.origin};
    }
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

std::vector<PauseOption> Session::pause_options(SequenceHandle h) const {
    const auto& p = *impl_;
    const auto& s = p.sequence(h);
    if (p.operation || execution_pending(h) ||
        (s.phase != detail::Phase::Prefill && s.phase != detail::Phase::Replay &&
         s.phase != detail::Phase::Active))
        return {};

    const auto& states = p.directory.states();
    const auto image   = s.continuation.image;
    const bool abort_writer =
        s.write_destination.valid() && states.can_abort_write(image, s.write_destination);
    ContextReleaseSet preparation_release;
    if (abort_writer) preparation_release.merge(p.directory.state_release(s.write_destination));
    if (s.capture && states.can_release(s.capture->destination))
        preparation_release.merge(p.directory.state_release(s.capture->destination));

    std::vector<PauseOption> result;
    if (s.request->can_checkpoint(s.continuation.metadata.main_frontier)) {
        ContextDemand checkpoint_demand;
        for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool)
            checkpoint_demand.add_growth(p.directory.pool(pool).histories(),
                                         s.histories[pool].handle(), 0);
        result.push_back({.action   = {runtime::PauseKind::KeepDevice, {}},
                          .demand   = checkpoint_demand,
                          .released = preparation_release});

        const bool has_host = std::any_of(p.plan.context.ranks.begin(), p.plan.context.ranks.end(),
                                          [](const auto& rank) { return rank.host_bytes != 0; });
        if (has_host) {
            HostRequirement write;
            write.state_images  = states.host_resident(image) ? 0 : 1;
            write.state_variant = states.variant(image);
            write.kv_pages.resize(p.pool_count());
            ContextReleaseSet released = preparation_release;
            if (states.source_pins(image) == static_cast<std::uint32_t>(abort_writer))
                released.merge(
                    p.directory.state_release(image).replicas(context::ContextMemory::Device));
            for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool) {
                const auto& storage = p.directory.pool(pool);
                const auto frontier = pool == 0 ? s.continuation.metadata.main_frontier
                                                : s.continuation.metadata.mtp_frontier.value();
                for (std::uint32_t i = 0; i < context::kv_pages_for_frontier(frontier); ++i) {
                    const auto page =
                        storage.histories().logical_page(s.histories[pool].handle(), i);
                    if (storage.pages().active_history_references(page) > 1) continue;
                    if (!storage.pages().host_replica_current(page)) ++write.kv_pages[pool];
                    if (!storage.pages().source_pins(page) && storage.pages().can_pin_source(page))
                        released.merge(p.directory.page_release(page).replicas(
                            context::ContextMemory::Device));
                }
            }
            checkpoint_demand.add_host(write);
            result.push_back({.action     = {runtime::PauseKind::HostSnapshot, {}},
                              .demand     = std::move(checkpoint_demand),
                              .released   = std::move(released),
                              .host_write = std::move(write)});
        }
    }

    ContextReleaseSet released = std::move(preparation_release);
    if (states.can_release(image) ||
        (abort_writer && !states.checkpoint_references(image) && states.source_pins(image) == 1))
        released.merge(p.directory.state_release(image));
    for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool) {
        const auto& histories = p.directory.pool(pool).histories();
        const auto history    = s.histories[pool].handle();
        for (const auto page : histories.released_pages(history))
            released.merge(p.directory.page_release(page));
        const auto retired = histories.released_histories(history);
        released.histories.insert(released.histories.end(), retired.begin(), retired.end());
    }
    released.normalize();
    result.push_back(
        {.action = {runtime::PauseKind::ReleaseAndReplay, {}}, .released = std::move(released)});
    return result;
}

bool Session::pause_contributes(SequenceHandle h, const PauseOption& option,
                                const ContextAssessment& assessment) const {
    const auto& p = *impl_;
    const auto& s = p.sequence(h);
    if (assessment.metadata_contribution(option.released) ||
        p.directory.useful_release_bytes(option.released, assessment))
        return true;
    if (s.capture && s.capture->host &&
        s.capture->host->contributes_to(assessment.placement, assessment.metadata_shortage))
        return true;
    for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool)
        if (p.directory.pool(pool).histories().growth_contributes(
                s.histories[pool].handle(), assessment.placement, assessment.metadata_shortage))
            return true;
    return false;
}

bool Session::prepare_pause(SequenceHandle h, runtime::PauseAction action,
                            std::span<const CheckpointHandle> host_victims) {
    auto& p = *impl_;
    if (p.operation || execution_pending(h)) return false;
    if (p.pause_reservation) throw std::logic_error("pause preparation is already owned");
    const auto options  = pause_options(h);
    const auto selected = std::find_if(options.begin(), options.end(),
                                       [&](const auto& option) { return option.action == action; });
    if (selected == options.end()) return false;
    if (!selected->host_write && !host_victims.empty())
        throw std::invalid_argument("pause action does not require Host victims");
    ContextReleaseSet released;
    if (!host_victims.empty()) {
        for (const auto victim : host_victims)
            if (!p.checkpoints.erasable(victim)) return false;
        released = p.footprint(host_victims, true);
    }
    auto& sequence = p.sequence(h);
    sequence.capture.reset();
    sequence.capture_pending = false;
    ContextDemand demand;
    if (action.kind != runtime::PauseKind::ReleaseAndReplay) {
        for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool)
            demand.add_growth(p.directory.pool(pool).histories(), sequence.histories[pool].handle(),
                              0);
    }
    context::HostDemandIds targets;
    if (selected->host_write) targets = demand.add_host(*selected->host_write);
    const auto assessment = p.directory.assess(demand, released);
    if (!assessment.feasible()) return false;
    if (assessment.requires_maintenance() && (p.compute || p.executor->has_pending_work()))
        return false;
    auto execution = p.executor->try_reserve(runtime::FrameKind::Context);
    if (!execution) return false;
    detail::PreparedCheckpoint checkpoint;
    if (action.kind != runtime::PauseKind::ReleaseAndReplay) {
        checkpoint = p.prepare_checkpoint(p.sequence(h));
        if (!checkpoint.record.valid()) return false;
    }
    p.pause_reservation.emplace(
        detail::SessionImpl::PauseReservation{.sequence   = h,
                                              .action     = action,
                                              .demand     = std::move(demand),
                                              .targets    = std::move(targets),
                                              .execution  = std::move(*execution),
                                              .checkpoint = std::move(checkpoint)});
    return true;
}

bool Session::start_pause(SequenceHandle h, runtime::PauseAction action) {
    auto& p = *impl_;
    if (p.operation || execution_pending(h)) return false;
    if (!p.pause_reservation && !prepare_pause(h, action)) return false;
    if (p.pause_reservation->sequence != h || p.pause_reservation->action != action)
        throw std::logic_error("pause does not match its preparation");
    auto targets = p.directory.reserve(p.pause_reservation->demand);
    if (!targets) throw std::logic_error("prepared pause placement changed before reservation");
    auto execution  = std::move(p.pause_reservation->execution);
    auto checkpoint = std::move(p.pause_reservation->checkpoint);
    auto target_ids = std::move(p.pause_reservation->targets);
    p.pause_reservation.reset();
    auto& s       = p.sequence(h);
    auto paused   = std::make_unique<detail::Suspended>();
    paused->owner = &p;
    paused->frontier =
        s.phase == detail::Phase::Replay ? s.replay_target : s.continuation.metadata.main_frontier;
    p.operation = std::make_unique<detail::ContextOperation>(
        p.directory,
        detail::PauseOperation{s.lane, std::move(paused), action, std::move(checkpoint),
                               std::move(target_ids)},
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
