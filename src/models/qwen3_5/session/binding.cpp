#include "models/common/qwen/vision/control.h"
#include "models/qwen3_5/session/internal.h"
#include "models/common/context/work.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

void SessionImpl::initialize_captures(RequestControl& request, std::uint32_t from,
                                      std::uint32_t through) {
    request.capture_groups.clear();
    request.last_saved                = from;
    request.last_intermediate_attempt = from;
    request.next_capture              = 0;
    request.capture_pending           = false;
    for (const auto& group : request.base->capture_groups) {
        if (group.frontier > from && group.frontier <= through) {
            request.capture_groups.push_back(group);
        }
    }
}

void SessionImpl::initialize_prefill(RequestControl& request, std::uint32_t base) {
    const auto& plan = *request.base;
    auto& staged     = request.prefill.emplace();
    staged.prompt    = *plan.prompt;
    staged.base = staged.cursor = base;
    staged.prompt_tokens        = plan.summary.prompt_tokens;
    staged.reuse                = base ? PrefixReusePath::Checkpoint : PrefixReusePath::Root;
    initialize_captures(request, base, plan.summary.prompt_tokens);
}

BindingPlan SessionImpl::plan_binding(const RequestBasePlan& base, runtime::LaneId lane_id,
                                      const SourceCandidate& candidate, const ResumeState* resume,
                                      ExecutionUnitKind resume_kind,
                                      std::uint32_t resume_tokens) const {
    const auto lane = lane_id.value;
    if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Empty || !base.impl_)
        throw std::logic_error("binding requires a free lane and request plan");
    BindingPlan result;
    const auto actual = inspect_source(base, candidate.checkpoint, candidate.consume_source,
                                       candidate.private_points, candidate.retired_points,
                                       candidate.placement, resume);
    if (!actual) {
        result.evaluation.source_valid = false;
        return result;
    }
    const auto source            = actual->checkpoint;
    auto& transaction            = result.operation;
    transaction.lane             = lane;
    transaction.base             = base.impl_;
    transaction.source           = source;
    transaction.reuse_frontier   = actual->reused_tokens;
    transaction.backend_frontier = source ? checkpoint(*source).backend_frontier() : 0;
    transaction.source_tail_hidden =
        source && checkpoint(*source).continuation().metadata.tail_hidden_valid;
    transaction.resume_snapshot = resume && source && actual->reused_tokens == resume->frontier();
    transaction.take_private    = candidate.take_private;
    transaction.carried_points  = actual->private_points;
    transaction.state_action    = actual->state_action;
    if (source) {
        if (actual->state_action == StateBindingAction::Move)
            std::erase(transaction.carried_points, *source);
        transaction.consume_source =
            actual->consume_source &&
            std::find(transaction.carried_points.begin(), transaction.carried_points.end(),
                      *source) == transaction.carried_points.end();
    }
    UnitDemand first;
    if (!resume)
        first = prefill_unit(base.summary().prompt_tokens, transaction.reuse_frontier);
    else if (!transaction.resume_snapshot) {
        first                  = {.kind = ExecutionUnitKind::Replay,
                                  .main_frontier =
                                      std::min(transaction.reuse_frontier + prefill_chunk, resume->frontier())};
        first.backend_frontier = speculative_backend == SpeculativeBackend::Mtp
                                     ? (first.main_frontier ? first.main_frontier - 1U : 0)
                                     : (has_backend_pool() ? first.main_frontier : 0);
    } else
        first =
            next_unit(resume->impl_->sequence, resume->impl_->control, resume_kind, resume_tokens);
    if (base.impl_->vision &&
        (first.kind == ExecutionUnitKind::Prefill || first.kind == ExecutionUnitKind::Replay)) {
        first.main_frontier =
            vision::Window{}.limit({}, base.impl_->vision->units, transaction.reuse_frontier,
                                   first.main_frontier, plan.vision_max_features);
        first.backend_frontier = speculative_backend == SpeculativeBackend::Mtp
                                     ? (first.main_frontier ? first.main_frontier - 1 : 0)
                                 : has_backend_pool() ? first.main_frontier
                                                      : 0;
    }
    transaction.first_unit = transaction.reservation_demand = first;
    if (resume) {
        auto coverage =
            next_unit(resume->impl_->sequence, resume->impl_->control, resume_kind, resume_tokens);
        coverage.main_frontier = std::max(coverage.main_frontier, resume->frontier());
        if (has_backend_pool())
            coverage.backend_frontier =
                std::max(coverage.backend_frontier,
                         speculative_backend == SpeculativeBackend::Mtp && resume->frontier()
                             ? resume->frontier() - 1U
                             : resume->frontier());
        transaction.recovery =
            RecoveryPermit{.coverage      = coverage,
                           .frontier      = resume->frontier(),
                           .ledger_tokens = resume->impl_->sequence.ledger.size()};
        transaction.reservation_demand = coverage;
    }
    ContextReleaseSet releases;
    if (!candidate.retired_points.empty()) {
        const auto quoted = checkpoint_release_set(candidate.retired_points);
        if (!quoted) {
            result.evaluation.source_valid = false;
            return result;
        }
        releases = *quoted;
    }
    const auto& coverage       = transaction.reservation_demand;
    const auto* main_source    = source ? &checkpoint(*source).text_prefix() : nullptr;
    const auto* backend_source = source ? checkpoint(*source).backend_prefix() : nullptr;
    std::vector<const context::HistoryPrefix*> retired_main, retired_backend;
    const auto retire = [&](CheckpointHandle point) {
        retired_main.push_back(&checkpoint(point).text_prefix());
        if (const auto* prefix = checkpoint(point).backend_prefix())
            retired_backend.push_back(prefix);
    };
    for (auto point : candidate.retired_points) retire(point);
    if (transaction.consume_source) retire(*source);
    const auto main = text_kv_histories->binding_demand(main_source, transaction.reuse_frontier,
                                                        coverage.main_frontier, retired_main);
    const auto backend =
        backend_kv_histories
            ? backend_kv_histories->binding_demand(backend_source, transaction.backend_frontier,
                                                   coverage.backend_frontier, retired_backend)
            : context::HistoryBindingDemand{};
    const auto action = transaction.state_action;
    const bool needs_state =
        action == StateBindingAction::Root || action == StateBindingAction::Fork ||
        action == StateBindingAction::Restore ||
        (action == StateBindingAction::Move &&
         !state_store->device_resident(checkpoint(*source).continuation().image));
    result.state = result.demand.add_state(
        {.count         = static_cast<std::uint32_t>(needs_state),
         .logical_count = static_cast<std::uint32_t>(action != StateBindingAction::Move)});
    result.main = result.demand.add_history(kTextKvPool.index, main);
    if (backend_kv_histories)
        result.backend = result.demand.add_history(kBackendKvPool.index, backend);
    if (action == StateBindingAction::BackupSplit)
        result.backup = result.demand.add_state(
            {.count = 1, .logical_count = 0, .memory = context::ContextMemory::Host});
    result.evaluation.assessment        = directory.assess(result.demand, releases);
    result.evaluation.capacity_possible = !result.evaluation.assessment.impossible();
    return result;
}

BindingReservation SessionImpl::start_binding(const RequestBasePlan& base, runtime::LaneId lane_id,
                                              const SourceCandidate& candidate, ResumeState* resume,
                                              ExecutionUnitKind resume_kind,
                                              std::uint32_t resume_tokens) {
    if (has_context_operation())
        throw std::logic_error("binding requires a free context transaction slot");
    auto planned = plan_binding(base, lane_id, candidate, resume, resume_kind, resume_tokens);
    if (!planned.evaluation.source_valid) return {.source_valid = false};
    if (!planned.evaluation.assessment.ready())
        return {.capacity_possible = planned.evaluation.capacity_possible,
                .assessment        = std::move(planned.evaluation.assessment)};
    auto execution_reservation = executor().try_reserve(runtime::FrameKind::Context);
    if (!execution_reservation)
        return {.assessment = {.status = context::ContextAvailability::Waiting}};
    const auto lane            = lane_id.value;
    auto& request              = requests[lane];
    auto transaction           = std::move(planned.operation);
    transaction.resume         = resume;
    const auto source          = transaction.source;
    const auto* main_source    = source ? &checkpoint(*source).text_prefix() : nullptr;
    const auto* backend_source = source ? checkpoint(*source).backend_prefix() : nullptr;
    BindingReservation result{.reserved       = true,
                              .retired_points = candidate.retired_points,
                              .consumed_source =
                                  transaction.consume_source ? source : std::nullopt};
    RequestControl prepared;
    prepared.base = base.impl_;
    SequenceState prepared_state;
    if (!resume) {
        initialize_prefill(prepared, transaction.reuse_frontier);
        prepared_state.ledger = base.impl_->prompt->token_ids;
        prepared_state.prefix_identity.assign(*base.impl_->prompt);
        prepared_state.prefix_digests = base.impl_->prefix_digests;
        const auto tokens =
            std::size_t(base.summary().prompt_tokens) + base.summary().effective_output_tokens;
        prepared_state.ledger.reserve(tokens);
        prepared_state.prefix_identity.reserve(tokens);
        prepared_state.prefix_digests.reserve(tokens);
        prepared.publish_continuation = base.summary().publish_continuation;
        prepared.sampling_host        = base.impl_->sampling;
        prepared.speculative_stats    = {
               .backend               = speculative_backend,
               .enabled               = speculative_backend != SpeculativeBackend::None,
               .draft_window          = draft_window,
               .accepted_per_position = std::vector<std::uint64_t>(draft_window, 0),
        };
    }
    prepared.lifecycle = Lifecycle::Binding;
    for (auto point : candidate.retired_points) {
        if (!release_checkpoint(point))
            throw std::logic_error("binding retirement changed after its quote");
        transaction.retired_points.push_back(point);
    }
    auto reserved = directory.reserve(planned.demand);
    if (!reserved)
        throw std::logic_error("binding placement changed after its accepted assessment");
    auto& destinations = *reserved.reservation;
    auto state_records = destinations.take(planned.state);
    if (source) transaction.source_lease = checkpoints.lease(*source);
    context_operation_   = std::make_unique<PendingContext>(directory, std::move(transaction));
    request              = std::move(prepared);
    sequences[lane]      = std::move(prepared_state);
    sequences[lane].lane = lane;
    sequences[lane].id   = resume ? resume->impl_->sequence.id
                                  : SequenceHandle{next_sequence++, executor().epoch().value};
    if (++lane_epochs[lane] == 0) ++lane_epochs[lane];
    sequences[lane].residency = {{lane}, lane_epochs[lane]};
    context_operation_->reservation.emplace(std::move(*execution_reservation));
    auto& tx        = std::get<BindOperation>(context_operation_->action);
    auto& migration = context_operation_->transaction;
    try {
        for (auto point : tx.carried_points) {
            if (point != source) tx.carried_pins.push_back(checkpoints.lease(point));
            if (!tx.take_private)
                if (const auto clone = reserve_checkpoint())
                    tx.carried_clones.emplace_back(point, *clone);
        }
        migration.add_binding(
            kTextKvPool.index,
            text_kv_histories->prepare_binding(main_source, tx.reuse_frontier,
                                               tx.reservation_demand.main_frontier, lane,
                                               destinations.take(planned.main), tx.consume_source));
        if (backend_kv_histories)
            migration.add_binding(kBackendKvPool.index,
                                  backend_kv_histories->prepare_binding(
                                      backend_source, tx.backend_frontier,
                                      tx.reservation_demand.backend_frontier, lane,
                                      destinations.take(planned.backend), tx.consume_source));
        const auto source_image =
            source ? checkpoint(*source).continuation().image : context::StateId{};
        if (tx.state_action != StateBindingAction::Move) {
            const bool device = tx.state_action == StateBindingAction::Root ||
                                tx.state_action == StateBindingAction::Fork;
            const auto destination =
                device ? state_store->reserve_destination(std::move(state_records))
                       : state_store->reserve_logical_destination(state_records);
            if (!destination) throw std::logic_error("binding lost its reserved State record");
            migration.own_state(*destination);
        }
        switch (tx.state_action) {
        case StateBindingAction::Root:
            migration.clear_state();
            break;
        case StateBindingAction::Fork:
            if (is_masked_draft_backend(speculative_backend))
                migration.add_state_group(source_image, plan.dflash_local_group->offset,
                                          plan.dflash_local_group->bytes);
            break;
        case StateBindingAction::Move:
            if (state_store->device_resident(source_image)) break;
            [[fallthrough]];
        case StateBindingAction::Restore: {
            const auto destination = tx.state_action == StateBindingAction::Move
                                         ? source_image
                                         : migration.state_destination().image();
            auto transfer          = state_store->reserve_host_to_device(source_image, destination,
                                                                         std::move(state_records));
            if (!transfer) throw std::logic_error("Host State restore reservation failed");
            migration.add_state_copy(std::move(*transfer), true);
            break;
        }
        case StateBindingAction::BackupSplit: {
            auto backup = state_store->reserve_device_to_host(source_image,
                                                              destinations.take(planned.backup));
            if (!backup) throw std::logic_error("source backup lost its Host reservation");
            migration.add_state_copy(std::move(*backup), true);
            break;
        }
        case StateBindingAction::Split:
            break;
        }
        ContextWork work;
        ResidencyBinding binding{{sequences[lane].id, sequences[lane].residency},
                                 base.impl_->sampling};
        if (resume) {
            const auto& ledger = resume->impl_->sequence.ledger;
            if (ledger.size() < base.summary().prompt_tokens)
                throw std::logic_error("sampling restore has no complete generated ledger");
            binding.committed_generated_tokens.assign(ledger.begin() + base.summary().prompt_tokens,
                                                      ledger.end());
        }
        work.bindings.push_back(std::move(binding));
        (void)submit_context(*context_operation_, std::move(work));
        return result;
    } catch (...) {
        abort_context();
        throw;
    }
}

void SessionImpl::install_binding(BindOperation& tx) {
    auto& state   = sequences[tx.lane];
    auto& request = requests[tx.lane];
    if (tx.resume) {
        auto& saved           = *tx.resume->impl_;
        request               = std::move(saved.control);
        const auto residency  = state.residency;
        state                 = std::move(saved.sequence);
        state.lane            = tx.lane;
        state.residency       = residency;
        const auto rope_delta = state.continuation.metadata.rope_delta;
        state.continuation.metadata =
            tx.source ? checkpoint(*tx.source).continuation().metadata : ContinuationMetadata{};
        state.continuation.metadata.rope_delta = rope_delta;
        if (!tx.resume_snapshot) {
            request.resume_lifecycle                            = request.lifecycle;
            request.lifecycle                                   = Lifecycle::Replaying;
            request.replay_target                               = saved.frontier;
            request.replay_cursor                               = tx.reuse_frontier;
            state.continuation.metadata.execution_frontier      = tx.reuse_frontier;
            state.continuation.metadata.text_kv_valid           = tx.reuse_frontier;
            state.continuation.metadata.backend_kv_valid        = tx.backend_frontier;
            state.continuation.metadata.dflash_context_frontier = tx.reuse_frontier;
            initialize_captures(request, tx.reuse_frontier, request.replay_target);
        }
        request.sampling_host = request.base->sampling;
    } else {
        state.continuation.metadata =
            tx.source ? checkpoint(*tx.source).continuation().metadata : ContinuationMetadata{};
        state.continuation.metadata.rope_delta              = tx.base->prompt->rope_delta;
        state.continuation.metadata.text_kv_valid           = tx.reuse_frontier;
        state.continuation.metadata.backend_kv_valid        = tx.backend_frontier;
        state.continuation.metadata.dflash_context_frontier = tx.reuse_frontier;
        state.continuation.metadata.tail_hidden_valid       = tx.source_tail_hidden;
        request.lifecycle                                   = Lifecycle::Prefilling;
    }
    request.permit   = tx.first_unit;
    request.recovery = tx.recovery;
}

void SessionImpl::complete_binding(BindOperation& tx, context::ContextTransaction& migration,
                                   ContextProgress& out) {
    install_binding(tx);
    auto& state = sequences[tx.lane];
    const auto source_image =
        tx.source ? checkpoint(*tx.source).continuation().image : context::StateId{};
    if (tx.state_action == StateBindingAction::Move) {
        if (!tx.consume_source) throw std::logic_error("State move requires a consumed source");
        state.continuation.image = checkpoint(*tx.source).snapshot.take_mutable_image();
        ++out.operations.state_moves;
    } else {
        auto& destination = migration.state_destination();
        if (tx.state_action == StateBindingAction::Fork) {
            (void)destination.begin_write(source_image);
            state.continuation.image = source_image;
            state.pending_write      = destination.release();
            ++out.operations.state_forks;
        } else {
            if (tx.state_action == StateBindingAction::Split ||
                tx.state_action == StateBindingAction::BackupSplit) {
                state_store->split_device_replica_identity(source_image, destination.image());
                ++out.operations.state_moves;
            }
            state_store->complete_initialization(destination.image());
            state.continuation.image = destination.release();
        }
    }
    if (tx.consume_source) {
        const auto old = *tx.source;
        tx.source_lease.reset();
        if (!checkpoints.erase(old))
            throw std::logic_error("consumed checkpoint still has readers");
        tx.retired_points.push_back(old);
        tx.source.reset();
        if (tx.resume && tx.resume->impl_->snapshot == old) tx.resume->impl_->snapshot.reset();
    }
    SequenceKVBundle kv{.text = migration.binding(kTextKvPool.index)->commit()};
    if (auto* backend = migration.binding(kBackendKvPool.index))
        kv.backend.emplace(backend->commit());
    state.kv.emplace(std::move(kv));
    for (const auto& [from, to] : tx.carried_clones) checkpoints.publish(to, checkpoint(from));
    if (!tx.take_private) {
        tx.carried_points.clear();
        for (const auto& [from, to] : tx.carried_clones) tx.carried_points.push_back(to);
    }
    out.private_points      = std::move(tx.carried_points);
    out.retired_checkpoints = std::move(tx.retired_points);
    out.sequence            = sequence_handle(tx.lane);
    out.residency           = state.residency;
    out.replaying           = requests[tx.lane].lifecycle == Lifecycle::Replaying;
}

} // namespace ninfer::models::qwen3_5::detail
