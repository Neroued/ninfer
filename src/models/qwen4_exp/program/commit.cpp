#include "models/qwen4_exp/program/internal.h"
#include "models/qwen4_exp/execution/mtp.h"
#include "core/transfer.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

void ProgramImpl::commit_verify(std::span<const SequenceHandle> members,
                                std::span<const runtime::CommitDecision> decisions,
                                runtime::ExecutionTimingRecorder& timing) {
    const auto& round   = pending_verify.value();
    const auto& buffers = resources.plan.persistent.buffers;
    const auto& frame   = *buffers.mtp;
    const auto batch = static_cast<int>(members.size()), width = static_cast<int>(round.width);
    const auto channels = dimension(parameters.model.config().text.residual_width());
    const auto hidden   = dimension(parameters.model.config().text.hidden);
    std::array<Sequence*, kMaximumConcurrency> rows;
    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold;
    auto* accepted = static_cast<std::int32_t*>(host_tensor(buffers.batch.accepted_columns).data);
    bool any       = false;
    for (int b = 0; b < batch; ++b) {
        rows[b]          = &sequence(members[b]);
        const auto slots = selectors(*rows[b]);
        accepted[b]      = decisions[b].accepted_tokens;
        any |= accepted[b] != 0;
        fold[b] = {slots.source, slots.destination, accepted[b]};
    }
    if (!any) return;
    auto accept = device_tensor(buffers.batch.accepted_columns).slice(0, 0, batch);
    copy_bytes_async({accept.data, accept.bytes(), cuda.device()},
                     {accepted, accept.bytes(), Device::cpu(), true}, cuda);
    resources.gdn_fold->execute({fold.data(), members.size()}, cuda);
    const auto& metadata = round.batch;
    for (std::uint32_t layer = 0; layer < parameters.model.config().text.qsa_layers; ++layer) {
        const auto records =
            device_tensor(buffers.main_qsa.keys).slice(3, layer, 1).slice(2, 0, batch);
        const auto coordinates =
            device_tensor(buffers.main_qsa.coordinates).slice(3, layer, 1).slice(2, 0, batch);
        ops::qsa_tail_extract(
            resources.states->main_tail(layer, metadata.source_slots),
            {records, coordinates, metadata.base_positions, metadata.valid_columns}, accept,
            resources.states->main_tail(layer, metadata.destination_slots), cuda);
    }
    ops::ple_state_extract(
        resources.states->ple(metadata.source_slots),
        {device_tensor(buffers.ple_records).slice(2, 0, batch), metadata.valid_columns}, accept,
        resources.states->ple(metadata.destination_slots), cuda);

    const auto k = width - 1;
    auto aligned = packed_tensor(frame.alignment_conditioning, {channels, k, batch});
    CUDA_CHECK(cudaMemsetAsync(aligned.data, 0, aligned.bytes(), cuda.stream()));
    auto conditioning = packed_tensor(*buffers.pending_conditioning, {channels, width, batch});
    auto sample       = packed_tensor(buffers.pending_sample, {hidden, width, batch});
    std::array<MtpInputRow, kMaximumConcurrency> inputs;
    for (int b = 0; b < batch; ++b) {
        const auto a     = decisions[b].accepted_tokens;
        const auto count = a ? a - 1 : 0;
        inputs[b]        = {round.bases[b],
                            std::span<const TokenId>(round.inputs).subspan(b * width + 1, count)};
        if (count) {
            auto source      = conditioning.slice(2, b, 1).slice(1, 0, count);
            auto destination = aligned.slice(2, b, 1).slice(1, 0, count);
            copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                             {source.data, source.bytes(), cuda.device()}, cuda);
        }
        if (!a) continue;
        const auto slot = selectors(*rows[b]).destination;
        const auto save = [&](const Tensor& values, Tensor store) {
            const auto source      = values.slice(2, b, 1).slice(1, a - 1, 1);
            const auto destination = store.slice(1, slot, 1);
            copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                             {source.data, source.bytes(), cuda.device()}, cuda);
        };
        save(sample, resources.states->sample_hidden());
        save(conditioning, resources.states->conditioning());
        auto ids =
            packed_tensor(*buffers.sampled_ids, {width, batch}).slice(1, b, 1).slice(0, 0, a);
        auto counts = device_tensor(*buffers.token_counts).slice(1, rows[b]->lane, 1);
        ops::increment_token_counts(ids.view({static_cast<int>(a)}), counts, cuda);
    }
    const auto call = mtp_inputs(frame.history, {rows.data(), members.size()}, k,
                                 {inputs.data(), members.size()});
    auto tail       = mtp_working_tail(1, batch);
    execution::mtp_history(parameters.model.config().text, *parameters.mtp, aligned, call,
                           mtp_working_tail(0, batch), tail, resources.mtp->cache.layer(0), {},
                           resources.workspace, cuda);
    const auto destination = resources.states->mtp_tail(call.destination_slots);
    const auto copy        = [&](const Tensor& source, const Tensor& output) {
        copy_bytes_async({output.data, output.bytes(), cuda.device()},
                                {source.data, source.bytes(), cuda.device()}, cuda);
    };
    for (int b = 0; b < batch; ++b) {
        if (!decisions[b].accepted_tokens) continue;
        const auto slot = selectors(*rows[b]).destination;
        copy(tail.raw_keys.slice(2, b, 1), destination.raw_keys.slice(2, slot, 1));
        copy(tail.first_rope.slice(1, b, 1), destination.first_rope.slice(1, slot, 1));
    }
    timing.begin_wait();
    ple.wait_compute();
    timing.end_wait();
    for (int b = 0; b < batch; ++b) {
        const auto a = decisions[b].accepted_tokens;
        if (!a) continue;
        auto& s     = *rows[b];
        auto& state = s.continuation.metadata;
        for (std::uint32_t j = 0; j < a; ++j)
            state.ple_raw_tail = {state.ple_raw_tail[1], round.inputs[b * width + j]};
        complete_write(s);
        state.main_frontier = round.bases[b] + a;
        state.mtp_frontier  = mtp_history_frontier(state.main_frontier);
        resources.main->histories.commit_frontier(s.histories[0].handle(), state.main_frontier);
        resources.mtp->histories.commit_frontier(s.histories[1].handle(), *state.mtp_frontier);
    }
}

} // namespace ninfer::models::qwen4_exp::detail

namespace ninfer::models::qwen4_exp {
using detail::seconds;

CommitResult Program::commit(PendingBatch&& batch,
                             std::span<const runtime::CommitDecision> decisions,
                             runtime::CommitObservation, runtime::ExecutionTiming* failed) {
    auto& p = *impl_;
    if (batch.owner_ != &p || batch.transaction_ != p.pending_transaction ||
        batch.members_ != p.pending_members || decisions.size() != batch.row_count())
        throw std::logic_error("invalid pending transaction");
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed);
    CommitResult out;
    out.row_count = decisions.size();
    for (std::size_t row = 0; row < decisions.size(); ++row) {
        const auto produced =
            batch.counts_.empty() ? 1U : static_cast<std::uint32_t>(batch.counts_[row]);
        const auto& d = decisions[row];
        if ((d.cancelled && d.failed) || (d.failed && !p.pending_failed[row]) ||
            ((d.cancelled || d.failed)
                 ? (!d.terminal || d.accepted_tokens != 0)
                 : (d.accepted_tokens == 0 || d.accepted_tokens > produced ||
                    (!d.terminal && d.accepted_tokens != produced) || p.pending_failed[row])) ||
            (d.prefix_execution_split_after &&
             (!*d.prefix_execution_split_after ||
              *d.prefix_execution_split_after > d.accepted_tokens)))
            throw std::logic_error("invalid generated-prefix commit decision");
    }
    const auto started = detail::Clock::now();
    if (p.pending_verify) p.commit_verify(batch.members_, decisions, timing);
    const auto post_seconds = detail::seconds(started);
    for (std::size_t row = 0; row < decisions.size(); ++row) {
        const auto h  = batch.members_[row];
        auto& s       = p.sequence(h);
        const auto& d = decisions[row];
        if (batch.decode_ && p.parameters.mtp) {
            auto& stats        = s.speculative;
            const auto drafted = p.pending_verify ? p.pending_verify->extents[row] : 0;
            const auto licensed =
                batch.counts_.empty() ? 1U : static_cast<std::uint32_t>(batch.counts_[row]);
            const auto accepted = std::min(d.accepted_tokens, licensed - 1);
            ++stats.rounds;
            stats.drafted_tokens += drafted;
            stats.accepted_tokens += accepted;
            stats.fallback_steps += drafted == 0;
            for (std::uint32_t i = 0; i < accepted; ++i) ++stats.accepted_per_position[i];
            s.timings.decode_seconds += post_seconds;
        }
        const auto& stats                  = s.speculative;
        out.rows[row].speculative_counters = {stats.rounds, stats.drafted_tokens,
                                              stats.accepted_tokens, stats.fallback_steps};
        out.rows[row].speculative          = s.speculative;
        if (d.cancelled || d.failed) {
            out.rows[row].timings     = s.timings;
            out.rows[row].disposition = d.cancelled ? runtime::CommitDisposition::CancelledReleased
                                                    : runtime::CommitDisposition::FailedReleased;
            p.clear(s);
            p.sequences[h.lane].reset();
            continue;
        }
        const auto tokens =
            std::span<const TokenId>(batch.tokens_).subspan(row * batch.stride_, d.accepted_tokens);
        s.identity.tokens.insert(s.identity.tokens.end(), tokens.begin(), tokens.end());
        s.identity.positions.append_generated(tokens.size(), s.continuation.metadata.rope_delta,
                                              d.prefix_execution_split_after);
        s.identity.digests.append_generated(tokens, s.continuation.metadata.rope_delta,
                                            d.prefix_execution_split_after);
        s.phase         = d.terminal ? detail::Phase::Finished : detail::Phase::Active;
        s.recovery_peak = 0;
        p.settle(s);
        if (!s.first_output) {
            s.first_output                = detail::Clock::now();
            s.timings.first_token_seconds = seconds(s.started);
            s.timings.prompt_wall_seconds = s.timings.first_token_seconds;
        }
        s.timings.total_seconds           = seconds(s.started);
        s.timings.generation_wall_seconds = seconds(*s.first_output);
        out.rows[row].timings             = s.timings;
        out.rows[row].disposition         = d.terminal ? runtime::CommitDisposition::Finishable
                                                       : runtime::CommitDisposition::Active;
        out.capture_ready[row]            = s.capture_pending;
    }
    batch.owner_          = nullptr;
    p.pending_transaction = 0;
    p.pending_members.clear();
    p.pending_verify.reset();
    out.timing = timing.finish();
    return out;
}

} // namespace ninfer::models::qwen4_exp
