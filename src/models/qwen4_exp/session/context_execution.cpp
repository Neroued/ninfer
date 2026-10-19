#include "models/qwen4_exp/session/internal.h"

namespace ninfer::models::qwen4_exp::detail {

ResidencyBinding SessionImpl::context_binding(BindOperation& operation) {
    auto binding   = std::make_unique<Sequence>();
    auto& sequence = *binding;
    if (operation.resume) {
        const auto& previous  = *operation.resume->state_->sequence;
        sequence.id           = previous.id;
        sequence.epoch        = previous.epoch;
        sequence.request      = previous.request;
        sequence.identity     = previous.identity;
        sequence.reused       = previous.reused;
        sequence.resume_phase = previous.phase;
        sequence.next_capture = previous.next_capture;
        sequence.timings      = previous.timings;
        sequence.speculative  = previous.speculative;
        sequence.started      = previous.started;
        sequence.first_output = previous.first_output;
    } else {
        sequence.id                      = next_sequence++;
        sequence.epoch                   = identity_generation;
        sequence.request                 = operation.request;
        sequence.identity                = *operation.request->identity;
        sequence.reused                  = operation.frontier;
        sequence.resume_phase            = Phase::Prefill;
        sequence.timings.prepare_seconds = operation.request->prompt.prepare.seconds;
        if (definition->options().mtp()) {
            sequence.speculative.backend      = SpeculativeBackend::Mtp;
            sequence.speculative.enabled      = true;
            sequence.speculative.draft_window = shape.draft_tokens;
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
    while (sequence.next_capture < sequence.request->captures.size() &&
           sequence.request->captures[sequence.next_capture].frontier <= operation.frontier)
        ++sequence.next_capture;

    const auto generated = std::span<const TokenId>(sequence.identity.tokens)
                               .subspan(sequence.request->summary.prompt_tokens);
    ResidencyBinding result{member(sequence),
                            sequence.request->sampling,
                            {generated.begin(), generated.end()},
                            sequence.request->vision};
    operation.binding = std::move(binding);
    return result;
}

void SessionImpl::submit_context() {
    try {
        auto& transaction = *operation;
        ContextWork context;
        context.releases = std::move(releases);
        if (auto* binding = std::get_if<BindOperation>(&transaction.action))
            context.bindings.push_back(context_binding(*binding));
        else if (const auto* pause = std::get_if<PauseOperation>(&transaction.action)) {
            const auto residency = member(*pause->paused->sequence).residency;
            if (std::find(context.releases.begin(), context.releases.end(), residency) ==
                context.releases.end())
                context.releases.push_back(residency);
        }
        transaction.migration.prepare_actions();
        Work work{transaction.migration.actions(), std::move(context)};
        transaction.ticket = executor->submit(std::move(transaction.reservation), std::move(work));
        transaction.migration.mark_submitted();
    } catch (const std::exception& error) {
        executor->fail(error.what());
        throw;
    }
}

} // namespace ninfer::models::qwen4_exp::detail
