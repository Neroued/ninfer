#include "models/qwen4_exp/session/internal.h"

namespace ninfer::models::qwen4_exp::detail {

ResidencyBinding SessionImpl::context_binding(BindOperation& operation) {
    auto binding   = std::make_unique<Sequence>();
    auto& sequence = *binding;
    if (operation.resume) {
        const auto& previous               = *operation.resume->state_->sequence;
        sequence.id                        = previous.id;
        sequence.epoch                     = previous.epoch;
        sequence.request                   = previous.request;
        sequence.identity                  = previous.identity;
        sequence.reused                    = previous.reused;
        sequence.resume_phase              = previous.phase;
        sequence.next_capture              = previous.next_capture;
        sequence.captures                  = previous.captures;
        sequence.last_saved                = previous.last_saved;
        sequence.last_intermediate_attempt = previous.last_intermediate_attempt;
        sequence.speculative               = previous.speculative;
    } else {
        sequence.id           = next_sequence++;
        sequence.epoch        = identity_generation;
        sequence.request      = operation.request;
        sequence.identity     = *operation.request->identity;
        sequence.reused       = operation.frontier;
        sequence.captures     = operation.request->captures;
        sequence.last_saved   = operation.frontier;
        sequence.resume_phase = Phase::Prefill;
        if (definition->options().mtp()) {
            sequence.speculative.backend      = SpeculativeBackend::Mtp;
            sequence.speculative.enabled      = true;
            sequence.speculative.draft_window = shape.draft_tokens;
            sequence.speculative.attempted_per_position.resize(shape.draft_tokens);
            sequence.speculative.accepted_per_position.resize(shape.draft_tokens);
        }
    }
    sequence.lane                = operation.lane;
    const auto identity_capacity = std::size_t(sequence.request->summary.prompt_tokens) +
                                   sequence.request->summary.effective_output_tokens;
    sequence.identity.tokens.reserve(identity_capacity);
    sequence.identity.positions.reserve(identity_capacity);
    sequence.identity.digests.reserve(identity_capacity);
    sequence.binding_generation = ++epochs[operation.lane];
    sequence.phase = operation.target > operation.frontier ? Phase::Replay : sequence.resume_phase;
    sequence.replay_target      = operation.target;
    sequence.recovery_peak      = operation.resume ? operation.peak : 0;
    sequence.permitted_frontier = operation.peak;
    if (operation.source) {
        sequence.continuation.metadata =
            checkpoints.at(*operation.source).snapshot.continuation().metadata;
        sequence.continuation.metadata.rope_delta = sequence.request->prompt.rope_delta;
    } else {
        const auto eos                 = definition->config().text.ple.eos;
        sequence.continuation.metadata = {
            0,
            sequence.request->prompt.rope_delta,
            {eos, eos},
            definition->options().mtp() ? std::optional<std::uint32_t>{0} : std::nullopt};
    }
    while (sequence.next_capture < sequence.captures.size() &&
           sequence.captures[sequence.next_capture].frontier <= operation.frontier)
        ++sequence.next_capture;

    const auto generated = std::span<const TokenId>(sequence.identity.tokens)
                               .subspan(sequence.request->summary.prompt_tokens);
    ResidencyBinding result{member(sequence), sequence.request->sampling, {}};
    const auto& sampling = sequence.request->sampling;
    if (sampling.presence_penalty || sampling.frequency_penalty)
        result.counts =
            count_builder.build(generated, definition->resources().text.public_token_count);
    operation.binding = std::move(binding);
    return result;
}

void SessionImpl::prepare_binding_resources(context::ContextReservation& reserved) {
    auto& bind             = std::get<BindOperation>(operation->action);
    auto& migration        = operation->migration;
    auto& states           = directory.states();
    const auto* checkpoint = bind.source ? &checkpoints.at(*bind.source) : nullptr;
    for (std::uint32_t pool = 0; pool < pool_count(); ++pool) {
        const auto* prefix = checkpoint ? &checkpoint->snapshot.prefix(pool) : nullptr;
        migration.add_binding(pool,
                              directory.pool(pool).histories().prepare_binding(
                                  prefix, prefix ? prefix->frontier() : 0,
                                  pool ? mtp_history_frontier(bind.peak) : bind.peak, bind.lane,
                                  reserved.take(bind.history_targets[pool]), bind.consume_source));
    }
    if (bind.state_action == StateBindingAction::Move) return;
    const bool materialize = bind.state_action == StateBindingAction::Root ||
                             bind.state_action == StateBindingAction::Fork;
    auto state_records     = reserved.take(bind.state_target);
    const auto destination = materialize ? states.reserve_destination(std::move(state_records))
                                         : states.reserve_logical_destination(state_records);
    if (!destination) throw std::logic_error("binding lost its State metadata reservation");
    migration.own_state(*destination);
    if (bind.state_action == StateBindingAction::Root) {
        migration.clear_state();
    } else if (bind.state_action == StateBindingAction::Restore) {
        auto transfer = states.reserve_host_to_device(bind.source_image, *destination,
                                                      std::move(state_records));
        if (!transfer) throw std::logic_error("binding lost its Host State source");
        migration.add_state_copy(std::move(*transfer), true);
    }
}

void SessionImpl::prepare_pause_resources(context::ContextReservation& targets) {
    auto& pause    = std::get<PauseOperation>(operation->action);
    auto& sequence = *sequences[pause.lane];
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
        checkpoints.at(*pause.paused->snapshot)
            .snapshot.set_use(context::ContextUse::Request,
                              pause.action.kind == runtime::PauseKind::HostSnapshot
                                  ? context::ContextMemory::Host
                                  : context::ContextMemory::Device);
    }
    pause.paused->sequence = std::move(sequences[pause.lane]);
    if (pause.paused->snapshot && pause.action.kind == runtime::PauseKind::HostSnapshot) {
        const auto plan = demotion({&*pause.paused->snapshot, 1});
        if (plan && !prepare_demotion(operation->migration, *plan, targets, pause.targets))
            throw std::logic_error("prepared pause lost its reserved Host targets");
    }
}

void SessionImpl::resume_context(context::ContextReservation targets) {
    if (!operation || !operation->preparing_resources || !targets.ready())
        throw std::logic_error("Context maintenance did not prepare its continuation");
    auto execution = executor->try_reserve(runtime::FrameKind::Context);
    if (!execution) throw std::logic_error("retired Context frame cannot submit its continuation");
    auto previous = std::move(operation);
    operation     = std::make_unique<ContextOperation>(directory, std::move(previous->action),
                                                       std::move(*execution));
    operation->previous_transfers = std::move(previous->previous_transfers);
    if (std::holds_alternative<BindOperation>(operation->action)) {
        prepare_binding_resources(targets);
    } else if (std::holds_alternative<PauseOperation>(operation->action)) {
        prepare_pause_resources(targets);
    } else if (const auto* demote = std::get_if<DemoteOperation>(&operation->action)) {
        if (!prepare_demotion(operation->migration, *demote->plan, targets, demote->targets))
            throw std::logic_error("reserved demotion changed during Context maintenance");
    } else {
        throw std::logic_error("Context maintenance has no continuation");
    }
    submit_context();
}

void SessionImpl::submit_context() {
    try {
        auto& transaction      = *operation;
        const bool first_batch = !transaction.actions_prepared;
        if (first_batch) {
            auto& residency = transaction.residency;
            if (!transaction.preparing_resources) residency.releases = std::move(releases);
            if (auto* binding = std::get_if<BindOperation>(&transaction.action);
                binding && !transaction.preparing_resources)
                residency.bindings.push_back(context_binding(*binding));
            else if (const auto* pause = std::get_if<PauseOperation>(&transaction.action);
                     pause && !transaction.preparing_resources) {
                const auto id = member(*pause->paused->sequence).residency;
                if (std::find(residency.releases.begin(), residency.releases.end(), id) ==
                    residency.releases.end())
                    residency.releases.push_back(id);
            }
            transaction.migration.prepare_actions();
            transaction.actions_prepared = true;
        }
        Work work;
        transaction.final_batch = true;
        std::vector<context::ContextActionBatchSource> sources;
        sources.reserve(plan.context.ranks.size());
        for (const auto& rank : plan.context.ranks) {
            const auto& actions = transaction.migration.actions(rank.rank);
            auto& cursor        = transaction.cursors[rank.rank];
            sources.push_back({rank.rank, &actions, &cursor});
        }
        auto batches = context::take_action_batches(sources);
        for (std::size_t i = 0; i < sources.size(); ++i) {
            if (!batches[i].empty()) work.actions.emplace(sources[i].rank, std::move(batches[i]));
            transaction.final_batch &= sources[i].cursor->complete(*sources[i].actions);
        }
        work.operation = transaction.final_batch ? std::move(transaction.residency) : ContextWork{};
        transaction.ticket = executor->submit(std::move(transaction.reservation), std::move(work));
        if (first_batch) transaction.migration.mark_submitted();
    } catch (const std::exception& error) {
        executor->fail(error.what());
        throw;
    }
}

} // namespace ninfer::models::qwen4_exp::detail
