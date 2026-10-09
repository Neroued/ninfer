#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/execution_context.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>

namespace ninfer::models::qwen3_5::detail {

PendingBatch ProgramImpl::wrap_pending(std::span<const std::uint32_t> lanes,
                                       const runtime::BatchedGeneratedRound& round) {
    if (pending_transaction_ || lanes.empty() || lanes.size() > max_concurrency) {
        throw std::logic_error("Program already owns a pending transaction");
    }
    PendingTransaction transaction;
    transaction.id   = next_transaction_id_++;
    transaction.size = lanes.size();
    std::array<SequenceHandle, kMaximumConcurrency> handles{};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending) {
            throw std::logic_error("pending transaction membership is invalid");
        }
        transaction.lanes[row]  = lane;
        transaction.epochs[row] = lane_epochs[lane];
        handles[row] =
            ContractAccess::make_sequence(this, runtime::LaneId{lane}, lane_epochs[lane]);
    }
    pending_transaction_ = transaction;
    auto pending         = ContractAccess::make_pending(
        this, transaction.id, std::span<const SequenceHandle>(handles.data(), lanes.size()),
        round.tokens, round.row_counts, round.row_stride, round.timing);
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const auto count = round.row_counts.empty() ? 1 : round.row_counts[row];
        ContractAccess::constraint_failed(pending, row,
                                          (grammar_dead_positions[row] & ((1u << count) - 1)) != 0);
    }
    return pending;
}

PrefillProgress ProgramImpl::wrap_prefill(std::uint32_t lane, runtime::PrefillStepResult step) {
    PrefillProgress out;
    out.summary                 = step.summary;
    out.processed_prompt_tokens = step.processed_prompt_tokens;
    out.complete                = step.complete;
    out.timing                  = step.timing;
    if (step.complete) {
        const std::array<std::uint32_t, 1> lanes{lane};
        const runtime::BatchedGeneratedRound round{
            .tokens     = step.round.tokens,
            .row_counts = {},
            .row_stride = 1,
        };
        out.pending.emplace(wrap_pending(lanes, round));
    } else if (requests[lane].prefill && requests[lane].capture_pending) {
        out.capture_ready = true;
    }
    return out;
}

PendingBatch ProgramImpl::decode(std::span<const SequenceHandle> members,
                                 std::span<const runtime::RoundBudget> budgets,
                                 runtime::ExecutionTiming* failed_timing,
                                 runtime::TokenMaskProvider* masks) {
    if (pending_transaction_ || members.empty() || members.size() > max_concurrency ||
        budgets.size() != members.size()) {
        throw std::invalid_argument("decode membership is invalid");
    }
    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    for (std::size_t row = 0; row < members.size(); ++row) {
        if (!valid_sequence(members[row])) {
            throw std::logic_error("decode sequence capability is invalid");
        }
        const std::uint32_t lane = ContractAccess::lane(members[row]).value;
        if (requests[lane].lifecycle != Lifecycle::Active ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("decode membership is duplicate or not active");
        }
        require_unit(lane, ExecutionUnitKind::Decode, budgets[row].generated_tokens_remaining);
        lanes[row] = lane;
    }
    const auto lane_span = std::span<const std::uint32_t>(lanes.data(), members.size());
    try {
        runtime::BatchedGeneratedRound round = decode_raw(lane_span, budgets, failed_timing, masks);
        if (failed_timing != nullptr) { *failed_timing += round.timing; }
        return wrap_pending(lane_span, std::move(round));
    } catch (...) {
        const Clock::time_point cleanup_started = Clock::now();
        clear_execution_failure_lanes(lane_span);
        pending_transaction_.reset();
        if (failed_timing != nullptr) {
            failed_timing->post_host_ns += elapsed_ns(cleanup_started);
        }
        throw;
    }
}

// Begin and ordinary rounds may already have provisional identity through the accepted extent;
// speculative and forced spans arrive with identity at their base. Both are Program-owned pending
// states, and this is their single accepted-prefix identity commit.
void ProgramImpl::commit_generated_prefix_identity(
    SequenceState& sequence, std::uint32_t base_ledger_frontier,
    std::span<const TokenId> accepted_tokens,
    std::optional<std::uint32_t> prefix_execution_split_after) {
    if (base_ledger_frontier > sequence.ledger.size() ||
        accepted_tokens.size() > sequence.ledger.size() - base_ledger_frontier ||
        sequence.ledger.size() != base_ledger_frontier + accepted_tokens.size() ||
        !std::equal(accepted_tokens.begin(), accepted_tokens.end(),
                    sequence.ledger.begin() + static_cast<std::ptrdiff_t>(base_ledger_frontier)) ||
        (prefix_execution_split_after &&
         (*prefix_execution_split_after == 0 ||
          *prefix_execution_split_after > accepted_tokens.size()))) {
        throw std::logic_error("committed generated-prefix identity has an invalid span");
    }
    const bool already_appended = sequence.prefix_identity.size() == sequence.ledger.size() &&
                                  sequence.prefix_digests.size() == sequence.ledger.size();
    const bool awaits_append = sequence.prefix_identity.size() == base_ledger_frontier &&
                               sequence.prefix_digests.size() == base_ledger_frontier;
    if (!already_appended && !awaits_append) {
        throw std::logic_error("generated-prefix identity is not at its base or committed extent");
    }
    if (already_appended && !prefix_execution_split_after) { return; }
    sequence.prefix_identity.truncate(base_ledger_frontier);
    sequence.prefix_digests.truncate(base_ledger_frontier);
    sequence.prefix_identity.append_generated(accepted_tokens.size(), sequence.rope_delta,
                                              prefix_execution_split_after);
    sequence.prefix_digests.append_generated(accepted_tokens, sequence.rope_delta,
                                             prefix_execution_split_after);
    if (sequence.prefix_identity.size() != sequence.ledger.size() ||
        sequence.prefix_digests.size() != sequence.ledger.size()) {
        throw std::logic_error("committed generated-prefix identity changed the ledger shape");
    }
}

runtime::ExecutionTiming ProgramImpl::append_forced_tokens(
    std::span<const SequenceHandle> members, std::span<const TokenId> row_major_tokens,
    std::uint32_t row_stride, std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (pending_transaction_ || members.empty() || members.size() > max_concurrency ||
        row_stride == 0 || prefix_execution_splits.size() != members.size() ||
        row_major_tokens.size() != static_cast<std::size_t>(row_stride) * members.size()) {
        throw std::invalid_argument("forced-token membership is invalid");
    }

    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    for (std::size_t row = 0; row < members.size(); ++row) {
        if (!valid_sequence(members[row])) {
            throw std::logic_error("forced-token sequence capability is invalid");
        }
        const std::uint32_t lane = ContractAccess::lane(members[row]).value;
        if (requests[lane].lifecycle != Lifecycle::Active ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("forced-token membership is duplicate or not active");
        }
        const SequenceState& sequence = active_sequence(lane);
        if (sequence.execution_frontier == std::numeric_limits<std::uint32_t>::max() ||
            sequence.ledger_frontier != sequence.execution_frontier + 1U ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            (speculative_backend == SpeculativeBackend::Mtp &&
             sequence.mtp_kv_valid != sequence.execution_frontier) ||
            (is_masked_draft_backend(speculative_backend) &&
             sequence.dflash_context_frontier > sequence.execution_frontier) ||
            static_cast<std::uint64_t>(sequence.execution_frontier) + row_stride > capacity) {
            throw std::logic_error("forced-token sequence frontier is invalid");
        }
        validate_licensed_tokens(row_major_tokens.subspan(row * row_stride, row_stride));
        if (prefix_execution_splits[row] &&
            (*prefix_execution_splits[row] == 0 || *prefix_execution_splits[row] > row_stride)) {
            throw std::logic_error("forced-token execution split is outside its row");
        }
        require_unit(lane, ExecutionUnitKind::Control, row_stride);
        lanes[row] = lane;
    }

    const bool count_forced_tokens = std::any_of(
        lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(members.size()),
        [&](std::uint32_t lane) { return requests[lane].sampling_host.token_counts != nullptr; });
    if (count_forced_tokens) {
        work.reset();
        Tensor forced_ids =
            work.alloc(DType::I32, {checked_i32(static_cast<std::uint32_t>(row_major_tokens.size()),
                                                "forced-token batch exceeds int32")});
        CUDA_CHECK(cudaMemcpyAsync(forced_ids.data, row_major_tokens.data(), forced_ids.bytes(),
                                   cudaMemcpyHostToDevice, device.stream));
        for (std::size_t row = 0; row < members.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            if (requests[lane].sampling_host.token_counts == nullptr) { continue; }
            Tensor ids    = forced_ids.slice(0, static_cast<std::int32_t>(row * row_stride),
                                             static_cast<std::int32_t>(row_stride));
            Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(lane), 1)
                                .view({dimension(parameters.model.resources().public_token_count)});
            ops::increment_token_counts(ids, counts, device.stream);
        }
        work.reset();
    }

    try {
        for (std::size_t row = 0; row < members.size(); ++row) {
            timing.resume_submit();
            const std::uint32_t lane = lanes[row];
            SequenceState& sequence  = active_sequence(lane);
            RequestControl& request  = requests[lane];
            const std::span<const TokenId> forced =
                row_major_tokens.subspan(row * row_stride, row_stride);
            const std::uint32_t base_ledger_frontier = sequence.ledger_frontier;
            const std::uint32_t base                 = sequence.execution_frontier;
            const std::uint32_t end                  = base + row_stride;
            const auto started                       = Clock::now();

            ensure_sequence_kv_mapped(sequence, end, backend_kv_cache() ? end : 0U);
            if (is_masked_draft_backend(speculative_backend) &&
                sequence.dflash_context_frontier < base) {
                const std::array<std::uint32_t, 1> append_lanes{lane};
                const std::array<std::uint32_t, 1> append_starts{sequence.dflash_context_frontier};
                const std::array<std::uint32_t, 1> append_counts{base -
                                                                 sequence.dflash_context_frontier};
                enqueue_dflash_context_append(append_lanes, append_starts, append_counts);
                timing.begin_wait();
                device.synchronize();
                timing.end_wait();
                sequence.dflash_context_frontier = base;
                commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
                work.reset();
                timing.resume_submit();
            }

            ensure_sequence_kv_mapped(sequence, end, backend_kv_cache() ? end : 0U);

            sequence.ledger.insert(sequence.ledger.end(), forced.begin(), forced.end());
            if (sequence.ledger.size() != static_cast<std::size_t>(end) + 1U) {
                throw std::logic_error("forced-token continuation ledger has an invalid shape");
            }

            if (is_masked_draft_backend(speculative_backend)) {
                if (!dflash || !io.dflash_prefill || !dflash_prefill_host_ingress || !sequence.kv ||
                    (backend_kv_cache() && !sequence.kv->backend)) {
                    throw std::logic_error("DFlash forced continuation state is incomplete");
                }
            }

            set_device_i32(io.text_kv_table_row, text_kv_addresses->bound_row(sequence.kv->text));
            if (sequence.kv->backend) {
                set_device_i32(io.backend_kv_table_row,
                               backend_kv_addresses->bound_row(*sequence.kv->backend));
            }
            std::uint32_t cursor = base;
            while (cursor < end) {
                const std::uint32_t count           = std::min(prefill_chunk, end - cursor);
                const StateImageSelectors selectors = state_selectors(sequence);
                execution::PrefillContext schedule_state{
                    {device, parameters, work, state_images->linear(),
                     replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
                     proposal_head},
                    text_kv_view(sequence),
                    mtp_kv_view(sequence),
                    decoder->text_kv,
                    decoder->mtp_cache(),
                    dflash ? &*dflash : nullptr,
                    cursor,
                    nullptr,
                    nullptr,
                    selectors.source,
                    selectors.destination,
                    0,
                    sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend)
                                         : 0,
                    dflash_prefill_host_ingress,
                    sequence.rope_delta};
                schedule_state.prefill_gpu_timer = &prefill_gpu_timer_;
                mark_workspace_usage(speculative_backend == SpeculativeBackend::Mtp
                                         ? workspace_plan.mtp_prefill
                                         : workspace_plan.text_prefill);
                if (is_masked_draft_backend(speculative_backend)) {
                    mark_workspace_usage(workspace_plan.dflash_context);
                }
                timing.pause();
                const execution::PrefillChunkResult result = execution::prefill_text_chunk(
                    schedule_state, sequence.ledger, count, std::nullopt, false);
                timing.include(result.timing);
                timing.resume_submit();
                if (result.finalized || result.processed_tokens == 0 ||
                    result.processed_tokens > count) {
                    throw std::logic_error("forced-token prefill made invalid progress");
                }
                cursor += result.processed_tokens;
                sequence.text_kv_valid = cursor;
                if (speculative_backend == SpeculativeBackend::Mtp) {
                    sequence.mtp_kv_valid = cursor;
                } else if (is_masked_draft_backend(speculative_backend)) {
                    sequence.dflash_context_frontier = cursor;
                }
                commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
                settle_state_fork(sequence);
                copy_tail(sequence,
                          prefill_hidden.slice(
                              1, static_cast<std::int32_t>(result.processed_tokens) - 1, 1));
            }
            timing.begin_wait();
            device.synchronize();
            timing.end_wait();
            work.reset();

            commit_generated_prefix_identity(sequence, base_ledger_frontier, forced,
                                             prefix_execution_splits[row]);
            sequence.execution_frontier = end;
            sequence.ledger_frontier    = end + 1U;
            sequence.mtp_draft_count    = 0;
            sequence.tail_hidden_valid  = true;
            if (sequence.ledger.size() != sequence.ledger_frontier ||
                sequence.prefix_identity.size() != sequence.ledger_frontier ||
                sequence.prefix_digests.size() != sequence.ledger_frontier ||
                sequence.ledger.back() != forced.back()) {
                throw std::logic_error("forced-token commit did not establish a valid frontier");
            }
            trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            settle_unit(lane);
            request.timings.decode_seconds +=
                std::chrono::duration<double>(Clock::now() - started).count();
        }
        return timing.finish();
    } catch (...) {
        timing.begin_wait();
        try {
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        work.reset();
        clear_execution_failure_lanes(std::span<const std::uint32_t>(lanes.data(), members.size()));
        throw;
    }
}

CommitResult ProgramImpl::commit(PendingBatch&& pending,
                                 std::span<const runtime::CommitDecision> decisions,
                                 runtime::CommitObservation observation,
                                 runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    std::array<SequenceHandle, kMaximumConcurrency> members{};
    const auto input_rows       = ContractAccess::rows(pending);
    const std::size_t row_count = input_rows.size();
    for (std::size_t row = 0; row < row_count; ++row) { members[row] = input_rows[row]; }
    std::array<bool, kMaximumConcurrency> constraint_failed{};
    for (std::size_t row = 0; row < row_count; ++row) {
        constraint_failed[row] = pending.constraint_failed(row);
    }
    const bool valid = valid_pending(pending);
    ContractAccess::consume(pending);

    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    std::array<GenerationTimings, kMaximumConcurrency> timings{};
    std::array<SpeculativeStats, kMaximumConcurrency> speculative{};
    std::array<PendingKind, kMaximumConcurrency> pending_kinds{};
    const auto release_members = [&]() noexcept {
        std::array<std::uint32_t, kMaximumConcurrency> failed_lanes{};
        std::size_t failed_count = 0;
        for (std::size_t row = 0; row < row_count; ++row) {
            if (ContractAccess::owner(members[row]) != this) { continue; }
            const std::uint32_t lane = ContractAccess::lane(members[row]).value;
            if (lane >= max_concurrency) { continue; }
            failed_lanes[failed_count++] = lane;
        }
        clear_execution_failure_lanes(
            std::span<const std::uint32_t>(failed_lanes.data(), failed_count));
        pending_transaction_.reset();
    };

    try {
        if (!valid || row_count == 0 || row_count > max_concurrency ||
            decisions.size() != row_count) {
            throw std::logic_error("pending transaction capability or decision shape is invalid");
        }
        std::array<std::uint32_t, kMaximumConcurrency> accepted{};
        std::array<std::uint8_t, kMaximumConcurrency> terminal{};
        std::array<std::uint8_t, kMaximumConcurrency> discarded{};
        std::array<std::optional<std::uint32_t>, kMaximumConcurrency> prefix_execution_splits{};
        for (std::size_t row = 0; row < row_count; ++row) {
            const std::uint32_t lane                = ContractAccess::lane(members[row]).value;
            lanes[row]                              = lane;
            const PendingCandidate& candidate       = requests[lane].pending;
            pending_kinds[row]                      = candidate.kind;
            const runtime::CommitDecision& decision = decisions[row];
            if ((decision.cancelled && decision.failed) ||
                (!decision.cancelled && decision.failed != constraint_failed[row]) ||
                ((decision.cancelled || decision.failed) &&
                 (decision.accepted_tokens != 0 || !decision.terminal)) ||
                (!(decision.cancelled || decision.failed) &&
                 (decision.accepted_tokens == 0 || decision.accepted_tokens > candidate.produced ||
                  (!decision.terminal && decision.accepted_tokens != candidate.produced))) ||
                (decision.prefix_execution_split_after &&
                 ((decision.cancelled || decision.failed) ||
                  *decision.prefix_execution_split_after == 0 ||
                  *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                throw std::logic_error("pending transaction decision is invalid");
            }
            accepted[row]                = decision.accepted_tokens;
            terminal[row]                = decision.terminal ? 1U : 0U;
            discarded[row]               = (decision.cancelled || decision.failed) ? 1U : 0U;
            prefix_execution_splits[row] = decision.prefix_execution_split_after;
            if (discarded[row]) {
                timings[row]     = requests[lane].timings;
                speculative[row] = std::move(requests[lane].speculative_stats);
            }
        }

        timing.pause();
        timing.include(
            resolve_pending_raw(std::span<const std::uint32_t>(lanes.data(), row_count),
                                std::span<const std::uint32_t>(accepted.data(), row_count),
                                std::span<const std::uint8_t>(terminal.data(), row_count),
                                std::span<const std::uint8_t>(discarded.data(), row_count),
                                std::span<const std::optional<std::uint32_t>>(
                                    prefix_execution_splits.data(), row_count),
                                failed_timing));
        timing.resume_post();
        pending_transaction_.reset();

        for (std::size_t row = 0; row < row_count; ++row) {
            if (!discarded[row]) { settle_unit(lanes[row]); }
        }
        CommitResult out;
        out.row_count = row_count;
        for (std::size_t row = 0; row < row_count; ++row) {
            if (discarded[row]) {
                invalidate_lane(lanes[row]);
                out.rows[row] = CommitRowResult{
                    .disposition = decisions[row].failed
                                       ? runtime::CommitDisposition::FailedReleased
                                       : runtime::CommitDisposition::CancelledReleased,
                    .timings     = timings[row],
                    .speculative = std::move(speculative[row]),
                };
            } else if (decisions[row].terminal) {
                out.rows[row].disposition = runtime::CommitDisposition::Finishable;
                if (observation == runtime::CommitObservation::AllRows) {
                    out.rows[row].timings     = requests[lanes[row]].timings;
                    out.rows[row].speculative = requests[lanes[row]].speculative_stats;
                }
            } else {
                out.rows[row].disposition = runtime::CommitDisposition::Active;
                if (observation == runtime::CommitObservation::AllRows) {
                    out.rows[row].timings     = requests[lanes[row]].timings;
                    out.rows[row].speculative = requests[lanes[row]].speculative_stats;
                }
            }

            const auto& stats =
                discarded[row] ? out.rows[row].speculative : requests[lanes[row]].speculative_stats;
            out.rows[row].speculative_counters = {stats.rounds, stats.drafted_tokens,
                                                  stats.accepted_tokens, stats.fallback_steps};
            if (pending_kinds[row] != PendingKind::Begin || discarded[row]) { continue; }
            RequestControl& request = requests[lanes[row]];
            if (decisions[row].terminal) {
                request.prefill.reset();
                continue;
            }
            if (!request.prefill) { continue; }
            RequestControl::Prefill& prefill = *request.prefill;
            if (prefill.cursor != prefill.prompt_tokens ||
                request.next_capture >= request.capture_groups.size() ||
                request.capture_groups[request.next_capture].frontier != prefill.prompt_tokens ||
                request.capture_pending) {
                throw std::logic_error("prompt-frontier capture carrier is inconsistent");
            }

            request.capture_pending = true;
            out.capture_ready[row]  = true;
        }
        out.timing = timing.finish();
        return out;
    } catch (...) {
        timing.resume_post();
        release_members();
        throw;
    }
}

DiscardResult ProgramImpl::abort_pending(PendingBatch&& pending) noexcept {
    DiscardResult out;
    const auto rows  = ContractAccess::rows(pending);
    const bool valid = valid_pending(pending);
    out.row_count    = std::min<std::size_t>(rows.size(), kMaximumConcurrency);
    std::array<SequenceHandle, kMaximumConcurrency> members{};
    for (std::size_t row = 0; row < out.row_count; ++row) { members[row] = rows[row]; }
    ContractAccess::consume(pending);
    if (!valid) { return out; }
    std::array<std::uint32_t, kMaximumConcurrency> failed_lanes{};
    for (std::size_t row = 0; row < out.row_count; ++row) {
        failed_lanes[row] = ContractAccess::lane(members[row]).value;
    }
    clear_execution_failure_lanes(
        std::span<const std::uint32_t>(failed_lanes.data(), out.row_count));
    pending_transaction_.reset();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

FinishResult ProgramImpl::finish(SequenceHandle handle) noexcept {
    FinishResult out;
    if (pending_transaction_ || !valid_sequence(handle)) { return out; }
    const auto lane = ContractAccess::lane(handle).value;
    auto& request   = requests[lane];
    if (request.lifecycle != Lifecycle::Finishable) { return out; }
    out.timings     = request.timings;
    out.speculative = std::move(request.speculative_stats);
    try {
        device.synchronize();
        if (request.publish_continuation) { out.checkpoint = detach_checkpoint(sequences[lane]); }
        clear_lane(sequences[lane], request);
        out.status = runtime::ConsumeStatus::Consumed;
    } catch (...) { clear_lane(sequences[lane], request); }
    return out;
}

AbortResult ProgramImpl::abort(SequenceHandle handle) noexcept {
    AbortResult out;
    if (pending_transaction_ || !valid_sequence(handle)) { return out; }
    const auto lane = ContractAccess::lane(handle).value;
    out.timings     = requests[lane].timings;
    out.speculative = std::move(requests[lane].speculative_stats);
    try {
        device.synchronize();
    } catch (...) { return out; }
    if (context_transaction_ && context_transaction_->kind != ContextOperationKind::Demote &&
        context_transaction_->lane == lane) {
        abort_context();
    }
    clear_lane(sequences[lane], requests[lane]);
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

void ProgramImpl::fail_all_cleanup() noexcept {
    try {
        device.synchronize();
    } catch (...) {}
    try {
        release_branch_cache();
    } catch (...) {}
    abort_context();
    pending_transaction_.reset();
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (requests[lane].lifecycle != Lifecycle::Empty) {
            clear_lane(sequences[lane], requests[lane]);
        }
    }
    // Published checkpoints belong to Runtime or ResumeState. After readers settle, their
    // owners release them; clearing this directory here would invalidate those live owners.
}
} // namespace ninfer::models::qwen3_5::detail
