#include "models/glm5next/session/internal.h"

#include <algorithm>

namespace ninfer::models::glm5next {

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
    if (s.request->can_checkpoint(s.continuation.metadata.main_frontier)) {
        for (const auto target : p.placements) {
            if (target == s.continuation.metadata.placement) continue;
            ContextDemand demand;
            demand.add_state({.count = 1, .variant = target.value});
            const std::array placement{target.value};
            for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool) {
                const auto f = pool ? s.continuation.metadata.mtp_frontier.value()
                                    : s.continuation.metadata.main_frontier;
                demand.add_history(pool, p.directory.pool(pool).histories().binding_demand(
                                             nullptr, 0, f, {}, placement));
            }
            result.push_back({.action   = {runtime::PauseKind::Relocate, target},
                              .demand   = std::move(demand),
                              .released = released});
        }
    }
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
    if (!host_victims.empty())
        throw std::invalid_argument("GLM5Next pause has no Host destination");
    ContextReleaseSet released;
    auto& sequence = p.sequence(h);
    sequence.capture.reset();
    sequence.capture_pending = false;
    ContextDemand demand;
    if (action.kind != runtime::PauseKind::ReleaseAndReplay) {
        for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool)
            demand.add_growth(p.directory.pool(pool).histories(), sequence.histories[pool].handle(),
                              0);
    }
    detail::RelocationTargets relocation;
    if (action.kind == runtime::PauseKind::Relocate) {
        relocation.state = demand.add_state({.count = 1, .variant = action.placement.value});
        const std::array placement{action.placement.value};
        for (std::uint32_t pool = 0; pool < p.pool_count(); ++pool) {
            const auto f               = pool ? sequence.continuation.metadata.mtp_frontier.value()
                                              : sequence.continuation.metadata.main_frontier;
            relocation.histories[pool] = demand.add_history(
                pool,
                p.directory.pool(pool).histories().binding_demand(nullptr, 0, f, {}, placement));
        }
    }
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
                                              .relocation = relocation,
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
    auto execution        = std::move(p.pause_reservation->execution);
    auto checkpoint       = std::move(p.pause_reservation->checkpoint);
    const auto relocation = p.pause_reservation->relocation;
    p.pause_reservation.reset();
    auto& s       = p.sequence(h);
    auto paused   = std::make_unique<detail::Suspended>();
    paused->owner = &p;
    paused->frontier =
        s.phase == detail::Phase::Replay ? s.replay_target : s.continuation.metadata.main_frontier;
    p.operation = std::make_unique<detail::ContextOperation>(
        p.directory,
        detail::PauseOperation{s.lane, std::move(paused), action, std::move(checkpoint),
                               relocation},
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

} // namespace ninfer::models::glm5next

namespace ninfer::models::glm5next::detail {

void SessionImpl::prepare_pause_resources(context::ContextReservation& targets) {
    auto& pause    = std::get<PauseOperation>(operation->action);
    auto& sequence = *sequences[pause.lane];
    vision_window.revoke(member(sequence));
    sequence.capture.reset();
    sequence.capture_pending = false;
    sequence.recovery_peak   = 0;
    settle(sequence);
    if (sequence.phase == Phase::Replay) sequence.phase = sequence.resume_phase;
    if (pause.action.kind == runtime::PauseKind::ReleaseAndReplay) {
        clear(sequence);
    } else {
        pause.paused->snapshot =
            detach(sequence, runtime::CheckpointRole::Continuation, std::move(pause.checkpoint));
        if (!pause.paused->snapshot)
            throw std::logic_error("prepared pause lost its checkpoint reservation");
        checkpoints.at(*pause.paused->snapshot).snapshot.set_use(context::ContextUse::Request);
    }
    pause.paused->sequence = std::move(sequences[pause.lane]);
    if (pause.action.kind == runtime::PauseKind::Relocate) {
        const auto& snapshot = checkpoints.at(*pause.paused->snapshot).snapshot;
        const auto from      = snapshot.continuation().metadata.placement.value;
        const auto to        = pause.action.placement.value;
        auto& migration      = operation->migration;
        migration.convert_state(snapshot.continuation().image, to,
                                targets.take(pause.relocation.state),
                                state_conversion(plan.states, from, to));
        const std::array placement{to};
        for (std::uint32_t pool = 0; pool < pool_count(); ++pool) {
            const auto& prefix = snapshot.prefix(pool);
            migration.add_binding(pool,
                                  directory.pool(pool).histories().prepare_conversion(
                                      prefix, prefix.frontier(), prefix.frontier(), pause.lane,
                                      targets.take(pause.relocation.histories[pool]), placement,
                                      history_conversion(plan.caches[pool], from, to,
                                                         prefix.frontier() % kPagedKVPageSize)));
        }
    }
}

} // namespace ninfer::models::glm5next::detail
