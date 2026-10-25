#include "models/glm5next/session/internal.h"

#include <algorithm>

namespace ninfer::models::glm5next {

ContextProgress detail::SessionImpl::complete_context() {
    auto& p = *this;
    if (!p.operation || !p.operation->result)
        throw std::logic_error("context operation has no completed result");
    auto& operation = *p.operation;
    auto& migration = operation.migration;
    ContextProgress result;
    result.kind      = operation.kind();
    result.transfers = std::move(operation.previous_transfers);
    result.transfers.insert(result.transfers.end(), operation.result->transfers.begin(),
                            operation.result->transfers.end());
    result.complete = true;
    if (!operation.final_batch) {
        operation.previous_transfers = std::move(result.transfers);
        operation.result.reset();
        auto execution = p.executor->try_reserve(runtime::FrameKind::Context);
        if (!execution)
            throw std::logic_error("retired Context batch cannot continue its transaction");
        operation.reservation = std::move(*execution);
        p.submit_context();
        result.complete = false;
        return result;
    }
    migration.mark_completed();
    if (auto* pause = std::get_if<detail::PauseOperation>(&operation.action);
        pause && pause->paused->sequence)
        pause->paused->sequence->binding_generation = 0;
    if (migration.cancelled()) {
        p.cancel_operation();
        return result;
    }
    const auto tail_copies = migration.tail_copy_page_groups();
    try {
        migration.publish();
        if (operation.preparing_resources) {
            auto targets                 = migration.take_reservation();
            operation.previous_transfers = std::move(result.transfers);
            p.resume_context(std::move(targets));
            result.complete = false;
            return result;
        }
        auto& states = p.directory.states();
        if (auto* tx = std::get_if<detail::BindOperation>(&operation.action)) {
            if (!tx->binding) throw std::logic_error("binding operation has no prepared sequence");
            auto& s = *tx->binding;
            for (std::uint16_t pool = 0; pool < p.directory.pool_count(); ++pool)
                s.histories[pool] = migration.binding(pool)->commit();
            using StateAction = detail::StateBindingAction;
            if (tx->state_action == StateAction::Move) {
                const auto source    = *tx->source;
                s.continuation.image = p.checkpoints.at(source).snapshot.take_mutable_image();
                tx->source_lease.reset();
                if (!p.checkpoints.erase(source))
                    throw std::logic_error("consumed checkpoint still has readers");
                tx->retired.push_back(source);
                tx->source.reset();
                ++result.operations.state_moves;
            } else {
                auto& destination = migration.state_destination();
                if (tx->state_action == StateAction::Fork) {
                    (void)destination.begin_write(tx->source_image);
                    s.continuation.image = tx->source_image;
                    s.write_destination  = destination.release();
                    ++result.operations.state_forks;
                } else {
                    if (tx->state_action != StateAction::Convert)
                        states.complete_initialization(destination.image());
                    s.continuation.image = destination.release();
                }
            }
            for (const auto& [from, to] : tx->clones) {
                p.checkpoints.publish(to, p.checkpoints.at(from));
                tx->carried.push_back(to);
            }
            tx->source_lease.reset();
            if (tx->consume_source && tx->source) {
                if (!p.checkpoints.erase(*tx->source))
                    throw std::logic_error("consumed checkpoint still has readers");
                tx->retired.push_back(*tx->source);
            }
            if (tx->resume) tx->resume->state_.reset();
            result.operations.partial_tail_cow_pages = tail_copies;
            result.sequence                          = p.handle(s);
            result.residency                         = p.member(s).residency;
            result.replaying                         = s.phase == detail::Phase::Replay;
            result.private_points                    = std::move(tx->carried);
            result.retired_checkpoints               = std::move(tx->retired);
            p.sequences[tx->lane]                    = std::move(tx->binding);
            tx->clones.clear();
        } else if (auto* tx = std::get_if<detail::CaptureOperation>(&operation.action)) {
            auto& s = *p.sequences[tx->lane];
            result.captured_checkpoints.reserve(tx->reservation->records.size());
            for (const auto& [record, role] : tx->reservation->records) {
                p.checkpoints.publish(record,
                                      {*tx->snapshot, role, s.captures[s.next_capture].origin});
                result.captured_checkpoints.push_back(record);
            }
            const auto writer    = context::adopt_capture_writer(states, s.continuation.image,
                                                                 migration.state_destination());
            s.continuation.image = writer.image;
            s.write_destination  = writer.write_destination;
            result.operations.state_forks += writer.write_destination.valid();
            tx->reservation->records.clear();
            s.capture_pending = false;
            s.last_saved      = s.captures[s.next_capture].frontier;
            ++s.next_capture;
        } else if (auto* tx = std::get_if<detail::PauseOperation>(&operation.action)) {
            if (tx->action.kind == runtime::PauseKind::Relocate) {
                auto& checkpoint                = p.checkpoints.at(*tx->paused->snapshot);
                auto continuation               = checkpoint.snapshot.continuation();
                continuation.metadata.placement = tx->action.placement;
                auto& destination               = migration.state_destination();
                continuation.image              = destination.image();
                states.freeze(continuation.image);
                std::vector<context::HistoryPrefix> prefixes;
                for (std::uint16_t pool = 0; pool < p.pool_count(); ++pool) {
                    auto history = migration.binding(pool)->commit();
                    prefixes.push_back(p.directory.pool(pool).histories().retain_prefix(
                        history.handle(), checkpoint.snapshot.prefix(pool).frontier()));
                }
                checkpoint.snapshot = detail::Snapshot(states, continuation, std::move(prefixes),
                                                       checkpoint.snapshot.identity_owner());
                (void)destination.release();
                tx->paused->sequence->continuation.metadata.placement = tx->action.placement;
            }
            result.request_speculative = tx->paused->sequence->speculative;
            result.paused.emplace(ResumeState(std::move(tx->paused)));
        }
        result.published = true;
        p.operation.reset();
        return result;
    } catch (...) {
        p.cancel_operation();
        throw;
    }
}

} // namespace ninfer::models::glm5next

namespace ninfer::models::glm5next::detail {

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

void SessionImpl::cancel_operation() noexcept {
    if (!operation) return;
    try {
        auto& tx = *operation;
        tx.migration.discard();
        if (auto* bind = std::get_if<BindOperation>(&tx.action)) {
            if (bind->binding) clear(*bind->binding);
            bind->source_lease.reset();
            bind->carried_leases.clear();
            for (const auto& [from, to] : bind->clones) {
                if (checkpoints.valid(to)) {
                    if (!checkpoints.erase(to)) std::terminate();
                } else
                    checkpoints.cancel(to);
            }
        } else if (auto* capture = std::get_if<CaptureOperation>(&tx.action)) {
            capture->reservation.reset();
            if (auto& sequence = sequences[capture->lane]) {
                auto& states     = directory.states();
                const auto image = sequence->continuation.image;
                if (capture->snapshot && capture->snapshot->continuation().image == image) {
                    if (sequence->write_destination.valid()) {
                        states.abort_write(image, sequence->write_destination);
                        if (!states.release(sequence->write_destination)) std::terminate();
                        sequence->write_destination = {};
                    }
                    sequence->continuation.image = capture->snapshot->take_mutable_image();
                } else if (!capture->snapshot &&
                           states.role(image) == context::StateRole::CheckpointImmutable &&
                           !states.checkpoint_references(image)) {
                    states.make_mutable(image);
                }
                sequence->capture_pending = false;
            }
        } else if (auto* pause = std::get_if<PauseOperation>(&tx.action)) {
            if (pause->paused && pause->paused->sequence) clear(*pause->paused->sequence);
            if (sequences[pause->lane]) {
                clear(*sequences[pause->lane]);
                sequences[pause->lane].reset();
            }
        }
        operation.reset();
    } catch (...) { std::terminate(); }
}

} // namespace ninfer::models::glm5next::detail
