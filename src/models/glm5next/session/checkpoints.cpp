#include "models/glm5next/session/internal.h"

#include "runtime/engine/context_cache/capture_policy.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {

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

} // namespace ninfer::models::glm5next::detail

namespace ninfer::models::glm5next {

std::optional<CapturePreparation>
Session::prepare_capture(SequenceHandle h, std::span<const CheckpointHandle> victims,
                         runtime::CaptureStorage storage) {
    if (storage == runtime::CaptureStorage::Host)
        throw std::invalid_argument("GLM5Next capture has no Host destination");
    auto& p = *impl_;
    auto& s = p.sequence(h);
    if (p.directory.maintenance_pending()) return {};
    if ((!s.permit && !s.capture_pending) || !s.request->summary.publish_continuation) return {};
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
                                  .origin   = s.captures[s.next_capture].origin,
                                  .evidence = s.captures[s.next_capture].evidence};
    while (!s.capture_pending && s.next_capture < s.captures.size() &&
           s.captures[s.next_capture].frontier <= s.continuation.metadata.main_frontier)
        ++s.next_capture;
    const auto begin   = s.continuation.metadata.main_frontier;
    const auto& advice = s.request->prompt.context_cache;
    if (!s.capture_pending && s.phase == detail::Phase::Prefill && advice.automatic_checkpoints &&
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
    if (point.frontier > (s.capture_pending ? begin : s.permitted_frontier)) return {};
    auto ticket         = std::make_unique<detail::CaptureReservation>();
    ticket->states      = &p.directory.states();
    ticket->checkpoints = &p.checkpoints;
    ticket->records.reserve(point.roles.size());
    for (const auto role : point.roles) {
        const auto record = p.checkpoints.reserve();
        if (!record) return {};
        ticket->records.emplace_back(*record, role);
    }
    ContextDemand demand;
    const auto variant = s.continuation.metadata.placement.value;
    const auto target  = demand.add_state({.count = 1, .variant = variant});
    if (!victims.empty()) {
        for (const auto victim : victims)
            if (!p.checkpoints.erasable(victim)) return {};
        const auto assessment = p.directory.assess(demand, p.footprint(victims, true));
        if (!assessment.ready())
            return CapturePreparation{.frontier   = point.frontier,
                                      .assessment = p.directory.assess(demand),
                                      .origin     = point.origin,
                                      .evidence   = point.evidence};
        for (const auto victim : victims)
            if (!p.checkpoints.erase(victim))
                throw std::logic_error("capture victim changed after its complete assessment");
    }
    auto reserved = p.directory.reserve(demand);
    if (!victims.empty() && (!reserved || !reserved.reservation->ready()))
        throw std::logic_error("capture placement changed after its complete assessment");
    if (!reserved || !reserved.reservation->ready()) {
        // The unused reservation may own an unsubmitted maintenance plan. Return
        // its original assessment after releasing that plan, not its temporary wait.
        reserved.reservation.reset();
        return CapturePreparation{.frontier   = point.frontier,
                                  .assessment = std::move(reserved.assessment),
                                  .origin     = point.origin,
                                  .evidence   = point.evidence};
    }
    ticket->destination = p.directory.states()
                              .reserve_destination(reserved.reservation->take(target), variant)
                              .value();
    s.capture = std::move(ticket);
    return CapturePreparation{.frontier       = point.frontier,
                              .reserved       = true,
                              .origin         = point.origin,
                              .evidence       = point.evidence,
                              .retired_points = {victims.begin(), victims.end()}};
}

std::vector<std::uint32_t> Session::capture_frontiers(SequenceHandle h, std::uint32_t begin,
                                                      std::uint32_t end) const {
    std::vector<std::uint32_t> result;
    const auto& points = impl_->sequence(h).captures;
    for (auto it = std::upper_bound(points.begin(), points.end(), begin,
                                    [](auto f, const auto& point) { return f < point.frontier; });
         it != points.end() && it->frontier <= end; ++it)
        result.push_back(it->frontier);
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

bool Session::retry_capture(SequenceHandle h) {
    auto& s          = impl_->sequence(h);
    const auto retry = std::exchange(s.retry_frontier, {});
    if (!retry || s.capture || s.recovery_peak || !s.next_capture ||
        *retry != s.continuation.metadata.main_frontier ||
        s.captures[s.next_capture - 1].frontier != *retry)
        return false;
    --s.next_capture;
    s.capture_pending = true;
    return true;
}

std::vector<CheckpointHandle> Session::share_capture(SequenceHandle h, CheckpointHandle source) {
    auto& p = *impl_;
    auto& s = p.sequence(h);
    if (s.next_capture == s.captures.size() || !p.checkpoints.valid(source)) return {};
    const auto& point = s.captures[s.next_capture];
    const auto& prior = p.checkpoints.at(source);
    if (prior.frontier() != point.frontier ||
        prior.snapshot.continuation().metadata.placement != s.continuation.metadata.placement ||
        prior.snapshot.identity().tag != s.request->identity->tag ||
        !input_prefix_equal(prior.snapshot.identity().view(), s.request->identity->view(),
                            point.frontier))
        return {};
    std::unique_ptr<detail::CaptureReservation> prepared;
    auto* ticket = s.capture.get();
    if (!ticket) {
        prepared              = std::make_unique<detail::CaptureReservation>();
        prepared->states      = &p.directory.states();
        prepared->checkpoints = &p.checkpoints;
        ticket                = prepared.get();
        ticket->records.reserve(point.roles.size());
        for (const auto role : point.roles) {
            const auto record = p.checkpoints.reserve();
            if (!record) return {};
            ticket->records.emplace_back(*record, role);
        }
    }
    std::vector<detail::Checkpoint> copies;
    std::vector<CheckpointHandle> result;
    copies.reserve(ticket->records.size());
    result.reserve(ticket->records.size());
    for (const auto& [record, role] : ticket->records) {
        copies.push_back({prior.snapshot, role, point.origin, point.evidence});
        result.push_back(record);
    }
    for (std::size_t i = 0; i < result.size(); ++i)
        p.checkpoints.publish(result[i], std::move(copies[i]));
    ticket->records.clear();
    skip_capture(h, runtime::CaptureSkip::Retained);
    return result;
}

void Session::skip_capture(SequenceHandle h, runtime::CaptureSkip reason) {
    auto& s = impl_->sequence(h);
    if (!s.capture_pending && reason == runtime::CaptureSkip::Unavailable &&
        s.next_capture < s.captures.size())
        s.retry_frontier = s.captures[s.next_capture].frontier;
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

runtime::ContextStart Session::start_capture(SequenceHandle h) {
    auto& p = *impl_;
    auto& s = p.sequence(h);
    if (!s.capture_pending) return runtime::ContextStart::Stale;
    if (!s.capture) {
        skip_capture(h);
        return runtime::ContextStart::Stale;
    }
    if (p.directory.maintenance_pending() || p.operation || execution_pending(h))
        return runtime::ContextStart::Deferred;
    if (s.write_destination.valid())
        throw std::logic_error("capture boundary has an unsettled state writer");
    auto execution = p.executor->try_reserve(runtime::FrameKind::Context);
    if (!execution) return runtime::ContextStart::Deferred;
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
        p.directory.states().freeze(s.continuation.image);
        tx.snapshot.emplace(p.directory.states(), s.continuation, std::move(prefixes),
                            s.request->identity);
        for (std::uint32_t rank = 0; rank < p.plan.states.size(); ++rank)
            if (const auto& field = p.plan.states[rank].draft_context)
                migration.add_state_group(s.continuation.image, field->offset, field->bytes, rank);
        p.submit_context();
    } catch (...) {
        p.cancel_operation();
        throw;
    }
    return runtime::ContextStart::Started;
}

} // namespace ninfer::models::glm5next

namespace ninfer::models::glm5next {

std::size_t Session::checkpoint_extent() const noexcept { return impl_->checkpoints.size(); }

std::vector<runtime::ContextTransferRequirement>
Session::checkpoint_restore_requirements(CheckpointHandle h) const {
    return impl_->checkpoints.at(h).snapshot.restore_requirements(impl_->directory);
}

bool Session::valid_checkpoint(CheckpointHandle h) const noexcept {
    return impl_->checkpoints.valid(h);
}

bool Session::release_checkpoint(CheckpointHandle h) noexcept {
    return impl_->checkpoints.erase(h);
}

CheckpointMetadata Session::checkpoint_metadata(CheckpointHandle h) const {
    const auto& p = *impl_;
    const auto& c = p.checkpoints.at(h);
    return {.frontier  = c.frontier(),
            .role      = c.role,
            .leased    = p.checkpoints[h.index].pins != 0,
            .origin    = c.origin,
            .evidence  = c.evidence,
            .placement = c.snapshot.continuation().metadata.placement};
}

PrefixShortlistKey Session::checkpoint_key(CheckpointHandle h, std::uint32_t f) const {
    const auto& c = impl_->checkpoints.at(h);
    if (f > c.frontier()) throw std::out_of_range("checkpoint key exceeds its frontier");
    return detail::key(c.snapshot.identity(), f);
}

bool Session::checkpoint_matches(CheckpointHandle h, const RequestBasePlan& base) const {
    if (!valid_checkpoint(h) || !base.plan_->reuse) return false;
    const auto& c        = impl_->checkpoints.at(h);
    const auto& identity = c.snapshot.identity();
    return identity.tag == base.plan_->identity->tag &&
           input_prefix_equal(base.plan_->prompt.prefix_view(), identity.view(), c.frontier());
}

std::uint32_t Session::checkpoint_recovery_frontier(CheckpointHandle h, const RequestBasePlan& base,
                                                    std::uint32_t target) const {
    return checkpoint_matches(h, base) && impl_->checkpoints.at(h).frontier() <= target
               ? impl_->checkpoints.at(h).frontier()
               : 0;
}

} // namespace ninfer::models::glm5next
