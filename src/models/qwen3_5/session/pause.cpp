#include "models/qwen3_5/session/internal.h"
#include "models/common/context/work.h"

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
                                 context::ContextReservation& targets) {
    if (!host_context_enabled) return false;
    preserve_state_device =
        shared_device && state_store->source_pins(record.continuation().image) != 0;
    if (!state_store->host_resident(record.continuation().image)) {
        auto transfer =
            state_store->reserve_device_to_host(record.continuation().image, targets.take_state());
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
            migration.add_backups(pool.index, logical, targets.take_history(pool.index), true);
        }
        return true;
    };
    if (!collect(*text_kv_histories, *text_kv_pages, record.history().text, kTextKvPool))
        return false;
    return !record.history().backend || collect(*backend_kv_histories, *backend_kv_pages,
                                                *record.history().backend, kBackendKvPool);
}

bool SessionImpl::prepare_pause(SequenceHandle handle, bool save_snapshot,
                                std::span<const CheckpointHandle> host_victims) {
    if (has_context_operation() || execution || !valid_sequence(handle)) return false;
    if (pause_reservation_) throw std::logic_error("pause preparation is already owned");
    const auto write = save_snapshot ? pause_host_requirements(handle)
                                     : std::optional(context::HostWriteRequirements{});
    if (!write) return false;
    const auto released = checkpoint_release_set(host_victims);
    if (!released) return false;
    auto placement = directory.plan_host(*write, *released);
    if (!placement.feasible()) return false;
    auto execution = executor().try_reserve(runtime::FrameKind::Context);
    if (!execution) return false;
    PreparedCheckpoint checkpoint;
    auto& state = sequences[lane(handle)];
    if (save_snapshot && state.continuation.metadata.text_kv_valid) {
        checkpoint = prepare_checkpoint(state);
        if (!checkpoint.record.valid()) return false;
    }
    pause_reservation_.emplace(PauseReservation{handle,
                                                save_snapshot,
                                                std::move(placement),
                                                {},
                                                std::move(*execution),
                                                {},
                                                std::move(checkpoint)});
    return true;
}

bool SessionImpl::start_pause(SequenceHandle handle, bool save_snapshot) {
    if (has_context_operation() || execution) return false;
    if (!valid_sequence(handle)) throw std::logic_error("pause has a stale sequence");
    const auto index = lane(handle);
    auto& state      = sequences[index];
    auto& control    = requests[index];
    if (control.lifecycle != Lifecycle::Active && control.lifecycle != Lifecycle::Prefilling &&
        control.lifecycle != Lifecycle::Replaying)
        throw std::logic_error("pause requires a retired committed boundary");
    if (!pause_reservation_ && !prepare_pause(handle, save_snapshot, {})) return false;
    if (pause_reservation_->sequence != handle ||
        pause_reservation_->save_snapshot != save_snapshot)
        throw std::logic_error("pause does not match its preparation");
    auto targets = directory.reserve(pause_reservation_->placement.demand);
    if (!targets) throw std::logic_error("prepared pause placement changed before reservation");
    if (!targets.reservation->ready()) {
        auto maintenance =
            directory.prepare_maintenance(std::move(*targets.reservation), targets.assessment);
        context_operation_ = std::make_unique<PendingContext>(
            std::move(maintenance),
            MaintenanceOperation{.pause_sequence = handle, .save_snapshot = save_snapshot});
        context_operation_->reservation.emplace(std::move(pause_reservation_->execution));
        (void)submit_context(*context_operation_, ContextWork{});
        return true;
    }
    pause_reservation_->targets = std::move(*targets.reservation);
    return continue_pause_ready(handle, save_snapshot);
}

bool SessionImpl::continue_pause_ready(SequenceHandle handle, bool save_snapshot) {
    const auto index = lane(handle);
    auto& state      = sequences[index];
    auto& control    = requests[index];
    control.capture_reservation.reset();
    if (control.capture_pending) skip_capture(handle);
    control.recovery.reset();
    if (control.permit) release_units(std::span(&handle, 1));
    if (control.lifecycle == Lifecycle::Replaying) save_snapshot = false;
    if (save_snapshot && is_masked_draft_backend(speculative_backend) &&
        state.continuation.metadata.dflash_context_frontier <
            state.continuation.metadata.execution_frontier) {
        const std::array<ExecutionUnit, 1> units{{handle, ExecutionUnitKind::Normalize, 0}};
        if (reserve_units(units)) {
            if (start_normalize(handle)) {
                normalizing_pause_ = handle;
                return true;
            }
            release_units(std::span(&handle, 1));
        }
        save_snapshot               = false;
        pause_reservation_->targets = {};
    }
    return begin_pause(handle, save_snapshot);
}

void SessionImpl::continue_pause_after_normalize(SequenceHandle handle) {
    if (!normalizing_pause_ || *normalizing_pause_ != handle || execution)
        throw std::logic_error("pause normalization has not retired");
    normalizing_pause_.reset();
    if (!begin_pause(handle, !pause_cancelled_))
        throw std::logic_error("pause lost its execution reservation");
}

bool SessionImpl::begin_pause(SequenceHandle handle, bool save_snapshot) {
    const auto lane = this->lane(handle);
    auto& state     = sequences[lane];
    auto& control   = requests[lane];
    if (!pause_reservation_) throw std::logic_error("pause has no context execution reservation");
    const auto residency = state.residency;
    control.capture_reservation.reset();
    auto saved      = std::make_unique<ResumeStateImpl>();
    saved->owner    = this;
    saved->frontier = control.lifecycle == Lifecycle::Prefilling ? control.prefill->cursor
                      : control.lifecycle == Lifecycle::Replaying
                          ? control.replay_target
                          : state.continuation.metadata.execution_frontier;
    if (control.lifecycle == Lifecycle::Replaying) {
        control.lifecycle = control.resume_lifecycle;
        control.replay.reset();
    }
    if (save_snapshot && control.base->vision_control_plan) {
        for (const auto& item : control.base->vision_control_plan->items) {
            if (item.token_begin < saved->frontier && saved->frontier < item.token_end) {
                save_snapshot = false;
                break;
            }
        }
    }
    if (save_snapshot && saved->frontier)
        saved->snapshot = detach_checkpoint(state, std::move(pause_reservation_->checkpoint));
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
    context_operation_->preceding_transfers = std::move(pause_reservation_->preceding_transfers);
    context_operation_->reservation.emplace(std::move(pause_reservation_->execution));
    pause_reservation_.reset();
    auto& tx        = std::get<PauseOperation>(context_operation_->action);
    auto& migration = context_operation_->transaction;
    try {
        if (tx.paused.has_snapshot()) {
            auto& snapshot = checkpoint(*tx.paused.impl_->snapshot);
            if (!prepare_backup(migration, snapshot, true, tx.preserve_state_device, targets))
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
