#include "models/qwen3_5/session/internal.h"
#include "models/common/context/work.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

PreparedCheckpoint SessionImpl::prepare_checkpoint(SequenceState& state) {
    const auto frontier = state.continuation.metadata.text_kv_valid;
    if (!frontier || !state.kv) return {};
    auto record = checkpoints.reserve_owned();
    if (!record.valid()) return {};
    ContextDemand demand;
    demand.add_growth(*text_kv_histories, state.kv->text.handle(), 0);
    if (state.kv->backend) demand.add_growth(*backend_kv_histories, state.kv->backend->handle(), 0);
    if (!directory.reserve(demand)) return {};
    auto backing     = std::make_shared<PreparedCaptureBacking>();
    backing->digests = state.prefix_digests;
    backing->digests.truncate(frontier);
    backing->ledger.assign(state.ledger.begin(), state.ledger.begin() + frontier);
    backing->prefix_identity = state.prefix_identity;
    backing->prefix_identity.truncate(frontier);
    return {std::move(record), std::move(backing)};
}

std::optional<CheckpointHandle> SessionImpl::detach_checkpoint(SequenceState& state,
                                                               PreparedCheckpoint prepared) {
    if (!prepared.record.valid()) prepared = prepare_checkpoint(state);
    if (!prepared.record.valid()) return {};
    const auto frontier = state.continuation.metadata.text_kv_valid;
    if (state_store->role(state.continuation.image) == context::StateRole::ActiveMutable)
        state_store->freeze(state.continuation.image);
    auto captured                        = state.continuation;
    captured.metadata.execution_frontier = frontier;
    captured.metadata.backend_kv_valid   = backend_kv_valid(state);
    std::vector<context::HistoryPrefix> prefixes;
    prefixes.push_back(text_kv_histories->retain_prefix(state.kv->text.handle(), frontier));
    if (state.kv->backend)
        prefixes.push_back(backend_kv_histories->retain_prefix(state.kv->backend->handle(),
                                                               backend_kv_valid(state)));
    CheckpointState snapshot{.snapshot = Snapshot(*state_store, captured, std::move(prefixes),
                                                  std::move(prepared.identity)),
                             .key      = {state.prefix_digests.at(frontier), frontier,
                                          requests[state.lane].base->prefix_identity_tag}};
    // The complete snapshot now owns the base before an outstanding writer is discarded.
    if (state.pending_write.valid()) {
        state_store->abort_write(state.continuation.image, state.write_state());
        if (!state_store->release(state.write_state()))
            throw std::logic_error("unused writer could not be released");
    }
    const auto handle = prepared.record.publish(std::move(snapshot));
    state.kv.reset();
    state.continuation.image = {};
    state.pending_write      = {};
    return handle;
}

bool SessionImpl::prepare_backup(context::ContextTransaction& migration, CheckpointState& record,
                                 bool shared_device, bool& preserve_state_device,
                                 context::ContextReservation& targets,
                                 const context::HostDemandIds& target_ids) {
    if (!host_context_enabled) return false;
    preserve_state_device =
        shared_device && state_store->source_pins(record.continuation().image) != 0;
    if (!state_store->host_resident(record.continuation().image)) {
        auto transfer = state_store->reserve_device_to_host(record.continuation().image,
                                                            targets.take(target_ids.state));
        if (!transfer) return false;
        migration.add_state_copy(std::move(*transfer), preserve_state_device);
    }
    const auto collect = [&](context::HistoryDirectory& histories, context::PageDirectory& pages,
                             context::HistoryId address, runtime::KvPoolId pool) {
        std::vector<context::PageId> logical;
        for (std::uint32_t i = 0; i < histories.mapped_pages(address); ++i) {
            const auto page = histories.logical_page(address, i);
            if (!pages.device_resident(page) ||
                (shared_device && pages.active_history_references(page)))
                continue;
            if (pages.host_replica_current(page)) {
                if (!pages.drop_device_replica(page)) return false;
            } else {
                logical.push_back(page);
            }
        }
        if (!logical.empty()) {
            migration.add_backups(pool.index, logical,
                                  targets.take(target_ids.histories.at(pool.index)), true);
        }
        return true;
    };
    if (!collect(*text_kv_histories, *text_kv_pages, record.history().text, kTextKvPool))
        return false;
    return !record.history().backend || collect(*backend_kv_histories, *backend_kv_pages,
                                                *record.history().backend, kBackendKvPool);
}

std::vector<PauseOption> SessionImpl::pause_options(SequenceHandle handle) const {
    if (has_context_operation() || execution || !valid_sequence(handle)) return {};
    const auto index    = lane(handle);
    const auto& state   = sequences[index];
    const auto& control = requests[index];
    if (control.lifecycle != Lifecycle::Active && control.lifecycle != Lifecycle::Prefilling &&
        control.lifecycle != Lifecycle::Replaying)
        return {};

    const auto frontier = control.lifecycle == Lifecycle::Prefilling
                              ? control.prefill->cursor
                              : state.continuation.metadata.execution_frontier;
    bool can_snapshot   = control.lifecycle != Lifecycle::Replaying && frontier && state.kv &&
                        state_store->valid(state.continuation.image);
    if (can_snapshot && control.base->vision)
        can_snapshot = control.base->vision->can_checkpoint(frontier);

    ContextReleaseSet optional_storage;
    if (state.pending_write.valid() &&
        state_store->can_abort_write(state.continuation.image, state.write_state()))
        optional_storage = directory.state_release(state.write_state());

    if (control.capture_reservation) {
        const auto destination = control.capture_reservation->state;
        if (state_store->can_release(destination))
            optional_storage.merge(directory.state_release(destination));
    }

    std::vector<PauseOption> result;
    if (can_snapshot) {
        result.push_back(
            {.action = {runtime::PauseKind::KeepDevice, {0}}, .released = optional_storage});
        if (const auto write = pause_host_requirements(handle)) {
            PauseOption host{.action     = {runtime::PauseKind::HostSnapshot, {0}},
                             .released   = optional_storage,
                             .host_write = *write};
            host.demand.add_host(*write);
            const auto image      = state.continuation.image;
            const auto own_reader = state.pending_write.valid() ? 1U : 0U;
            if (state_store->source_pins(image) == own_reader)
                host.released.merge(
                    directory.state_release(image).replicas(context::ContextMemory::Device));
            const auto collect = [&](const context::HistoryDirectory& histories,
                                     const context::PageDirectory& pages,
                                     context::HistoryId history) {
                for (std::uint32_t i = 0; i < histories.mapped_pages(history); ++i) {
                    const auto page = histories.logical_page(history, i);
                    if (pages.active_history_references(page) == 1 && !pages.source_pins(page))
                        host.released.merge(
                            directory.page_release(page).replicas(context::ContextMemory::Device));
                }
            };
            collect(*text_kv_histories, *text_kv_pages, state.kv->text.handle());
            if (state.kv->backend)
                collect(*backend_kv_histories, *backend_kv_pages, state.kv->backend->handle());
            result.push_back(std::move(host));
        }
    }

    PauseOption replay{.action   = {runtime::PauseKind::ReleaseAndReplay, {0}},
                       .released = std::move(optional_storage)};
    const auto image = state.continuation.image;
    if (state_store->valid(image) &&
        (state_store->can_release(image) ||
         (state.pending_write.valid() && !state_store->checkpoint_references(image) &&
          state_store->source_pins(image) == 1)))
        replay.released.merge(directory.state_release(image));
    const auto collect = [&](const context::HistoryDirectory& histories,
                             context::HistoryId history) {
        for (const auto page : histories.released_pages(history))
            replay.released.merge(directory.page_release(page));
        const auto released = histories.released_histories(history);
        replay.released.histories.insert(replay.released.histories.end(), released.begin(),
                                         released.end());
    };
    if (state.kv) {
        collect(*text_kv_histories, state.kv->text.handle());
        if (state.kv->backend) collect(*backend_kv_histories, state.kv->backend->handle());
    }
    replay.released.normalize();
    result.push_back(std::move(replay));
    return result;
}

bool SessionImpl::pause_contributes(SequenceHandle handle, const PauseOption& option,
                                    const ContextAssessment& assessment) const {
    if (!valid_sequence(handle)) return false;
    if (assessment.metadata_contribution(option.released) ||
        directory.useful_release_bytes(option.released, assessment))
        return true;
    const auto index    = lane(handle);
    const auto& capture = requests[index].capture_reservation;
    if (capture && capture->host &&
        capture->host->contributes_to(assessment.placement, assessment.metadata_shortage))
        return true;
    const auto& state = sequences[index];
    if (!state.kv) return false;
    return text_kv_histories->growth_contributes(state.kv->text.handle(), assessment.placement,
                                                 assessment.metadata_shortage) ||
           (state.kv->backend &&
            backend_kv_histories->growth_contributes(
                state.kv->backend->handle(), assessment.placement, assessment.metadata_shortage));
}

bool SessionImpl::prepare_pause(SequenceHandle handle, runtime::PauseAction action,
                                std::span<const CheckpointHandle> host_victims) {
    if (has_context_operation() || execution || !valid_sequence(handle)) return false;
    if (pause_reservation_) throw std::logic_error("pause preparation is already owned");
    const auto options  = pause_options(handle);
    const auto selected = std::find_if(options.begin(), options.end(),
                                       [&](const auto& option) { return option.action == action; });
    if (selected == options.end()) return false;
    const auto index = lane(handle);
    auto& state      = sequences[index];
    auto& control    = requests[index];
    control.capture_reservation.reset();
    const auto released = checkpoint_release_set(host_victims);
    if (!released) return false;
    ContextDemand demand;
    auto target_ids =
        selected->host_write ? demand.add_host(*selected->host_write) : context::HostDemandIds{};
    auto placement = directory.assess(demand, *released);
    if (!placement.feasible()) return false;
    auto execution = executor().try_reserve(runtime::FrameKind::Context);
    if (!execution) return false;
    PreparedCheckpoint checkpoint;
    std::optional<runtime::ExecutionReservation> normalization;
    if (action.kind != runtime::PauseKind::ReleaseAndReplay) {
        checkpoint = prepare_checkpoint(state);
        if (!checkpoint.record.valid()) return false;
        const bool normalize = is_masked_draft_backend(speculative_backend) &&
                               state.continuation.metadata.dflash_context_frontier <
                                   state.continuation.metadata.execution_frontier;
        if (normalize) {
            normalization = executor().try_reserve(runtime::FrameKind::Compute);
            if (!normalization) return false;
            if (control.permit) release_units(std::span(&handle, 1));
            const std::array<ExecutionUnit, 1> units{{handle, ExecutionUnitKind::Normalize, 0}};
            if (!reserve_units(units)) return false;
            placement = directory.assess(demand, *released);
            if (!placement.feasible()) {
                release_units(std::span(&handle, 1));
                return false;
            }
        }
    }
    pause_reservation_.emplace(PauseReservation{.sequence      = handle,
                                                .action        = action,
                                                .placement     = std::move(placement),
                                                .target_ids    = std::move(target_ids),
                                                .execution     = std::move(*execution),
                                                .normalization = std::move(normalization),
                                                .checkpoint    = std::move(checkpoint)});
    return true;
}

bool SessionImpl::start_pause(SequenceHandle handle, runtime::PauseAction action) {
    if (has_context_operation() || execution) return false;
    if (!valid_sequence(handle)) throw std::logic_error("pause has a stale sequence");
    const auto index = lane(handle);
    auto& state      = sequences[index];
    auto& control    = requests[index];
    if (control.lifecycle != Lifecycle::Active && control.lifecycle != Lifecycle::Prefilling &&
        control.lifecycle != Lifecycle::Replaying)
        throw std::logic_error("pause requires a retired committed boundary");
    if (!pause_reservation_ && !prepare_pause(handle, action, {})) return false;
    if (pause_reservation_->sequence != handle || pause_reservation_->action != action)
        throw std::logic_error("pause does not match its preparation");
    auto targets = directory.reserve(pause_reservation_->placement.demand);
    if (!targets) throw std::logic_error("prepared pause placement changed before reservation");
    if (!targets.reservation->ready()) {
        auto maintenance =
            directory.prepare_maintenance(std::move(*targets.reservation), targets.assessment);
        context_operation_ = std::make_unique<PendingContext>(
            std::move(maintenance),
            MaintenanceOperation{.pause_sequence = handle, .pause_action = action});
        context_operation_->reservation.emplace(std::move(pause_reservation_->execution));
        (void)submit_context(*context_operation_, ContextWork{});
        return true;
    }
    pause_reservation_->targets = std::move(*targets.reservation);
    return continue_pause_ready(handle, action);
}

bool SessionImpl::continue_pause_ready(SequenceHandle handle, runtime::PauseAction action) {
    auto& control = requests[lane(handle)];
    control.capture_reservation.reset();
    if (control.capture_pending) skip_capture(handle);
    control.recovery.reset();
    if (pause_reservation_->normalization) {
        auto execution = std::move(*pause_reservation_->normalization);
        pause_reservation_->normalization.reset();
        if (!start_normalize(handle, std::move(execution)))
            throw std::logic_error("prepared pause normalization lost its execution permit");
        normalizing_pause_ = handle;
        return true;
    }
    if (control.permit) release_units(std::span(&handle, 1));
    return begin_pause(handle, action);
}

void SessionImpl::continue_pause_after_normalize(SequenceHandle handle) {
    if (!normalizing_pause_ || *normalizing_pause_ != handle || execution)
        throw std::logic_error("pause normalization has not retired");
    normalizing_pause_.reset();
    const auto action = pause_cancelled_
                            ? runtime::PauseAction{runtime::PauseKind::ReleaseAndReplay, {0}}
                            : pause_reservation_->action;
    if (!begin_pause(handle, action))
        throw std::logic_error("pause lost its execution reservation");
}

bool SessionImpl::begin_pause(SequenceHandle handle, runtime::PauseAction action) {
    const bool save_snapshot = action.kind != runtime::PauseKind::ReleaseAndReplay;
    const auto lane          = this->lane(handle);
    auto& state              = sequences[lane];
    auto& control            = requests[lane];
    if (!pause_reservation_) throw std::logic_error("pause has no context execution reservation");
    const auto residency = state.residency;
    control.capture_reservation.reset();
    auto saved      = std::make_unique<ResumeStateImpl>();
    saved->owner    = this;
    saved->frontier = control.lifecycle == Lifecycle::Prefilling ? control.prefill->cursor
                      : control.lifecycle == Lifecycle::Replaying
                          ? control.replay_target
                          : state.continuation.metadata.execution_frontier;
    if (control.lifecycle == Lifecycle::Replaying) { control.lifecycle = control.resume_lifecycle; }
    if (save_snapshot) {
        saved->snapshot = detach_checkpoint(state, std::move(pause_reservation_->checkpoint));
        if (!saved->snapshot) throw std::logic_error("prepared pause lost its complete checkpoint");
    }
    if (!saved->snapshot) {
        release_sequence_kv(state);
        release_sequence_state(state);
    }
    saved->sequence    = std::move(state);
    saved->control     = std::move(control);
    state              = {};
    state.lane         = lane;
    control            = {};
    control.lifecycle  = Lifecycle::Pausing;
    context_operation_ = std::make_unique<PendingContext>(
        directory, PauseOperation{lane, ResumeState(std::move(saved))});
    auto targets                            = std::move(pause_reservation_->targets);
    auto target_ids                         = std::move(pause_reservation_->target_ids);
    context_operation_->preceding_transfers = std::move(pause_reservation_->preceding_transfers);
    context_operation_->reservation.emplace(std::move(pause_reservation_->execution));
    pause_reservation_.reset();
    auto& tx        = std::get<PauseOperation>(context_operation_->action);
    auto& migration = context_operation_->transaction;
    try {
        if (tx.paused.has_snapshot() && action.kind == runtime::PauseKind::HostSnapshot) {
            auto& snapshot = checkpoint(*tx.paused.impl_->snapshot);
            if (!prepare_backup(migration, snapshot, true, tx.preserve_state_device, targets,
                                target_ids))
                throw std::logic_error("prepared pause backup lost its destination");
        }
        if (pause_cancelled_) migration.cancel();
        pause_cancelled_ = false;
        ContextWork work;
        work.releases.push_back(residency);
        (void)submit_context(*context_operation_, std::move(work));
        return true;
    } catch (...) {
        abort_context();
        throw;
    }
}

ResumeState SessionImpl::complete_pause(PauseOperation& tx) {
    if (tx.paused.has_snapshot()) {
        const auto& record = checkpoint(*tx.paused.impl_->snapshot);
        if (!tx.preserve_state_device &&
            state_store->device_resident(record.continuation().image) &&
            !state_store->drop_device_replica(record.continuation().image)) {
            throw std::logic_error("paused state is still Device-pinned");
        }
    }
    requests[tx.lane]       = {};
    sequences[tx.lane]      = {};
    sequences[tx.lane].lane = tx.lane;
    invalidate_lane(tx.lane);
    return std::move(tx.paused);
}
} // namespace ninfer::models::qwen3_5::detail
