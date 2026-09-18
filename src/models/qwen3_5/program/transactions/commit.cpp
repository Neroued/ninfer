#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
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
    return ContractAccess::make_pending(
        this, transaction.id, std::span<const SequenceHandle>(handles.data(), lanes.size()),
        round.tokens, round.row_counts, round.row_stride, round.timing);
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
    } else if (requests[lane].prefill && requests[lane].prefill->pending_capture_offer != 0) {
        out.capture.emplace(
            ContractAccess::make_capture_offer(this, runtime::LaneId{lane}, lane_epochs[lane],
                                               requests[lane].prefill->pending_capture_offer));
    }
    return out;
}

StartResult ProgramImpl::start_request(MaterializationTransaction& transaction) {
    std::optional<std::uint32_t> destination = transaction.destination.value;
    std::optional<std::uint32_t> continuation_index;
    try {
        if (!transaction.prepared || !transaction.plan || !destination ||
            *destination >= max_concurrency) {
            throw std::invalid_argument("materialization transaction is not publishable");
        }
        const std::uint32_t lane              = *destination;
        const AdmissionCandidateImpl& details = *transaction.plan->impl_;
        if (details.destination_epoch != lane_epochs[lane] ||
            details.has_source != transaction.has_source ||
            details.has_shared_source != transaction.has_shared_source) {
            throw std::logic_error("admission plan physical epoch is stale");
        }
        if (requests[lane].lifecycle != Lifecycle::Empty ||
            active_continuations[lane] < continuation_capacity) {
            throw std::logic_error("admission destination is not free");
        }
        if (transaction.has_source &&
            transaction.source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
            if (transaction.source_index >= continuation_capacity ||
                continuation_slots[transaction.source_index].role !=
                    ContinuationSlotRole::Catalogued ||
                continuation_slots[transaction.source_index].generation !=
                    transaction.source_generation ||
                transaction.source_index != details.source_index ||
                transaction.source_generation != details.source_generation) {
                throw std::logic_error("admission source capability is stale");
            }
            continuation_index                           = transaction.source_index;
            continuation_slots[*continuation_index].role = ContinuationSlotRole::Active;
        } else {
            continuation_index = transaction.root_continuation_index;
            if (!continuation_index || transaction.root_waiting_for_victim ||
                continuation_slots[*continuation_index].role !=
                    ContinuationSlotRole::ReservedMaterialization) {
                throw std::logic_error("materialization continuation reservation is unavailable");
            }
            continuation_slots[*continuation_index].role = ContinuationSlotRole::Active;
        }

        const detail::PhysicalResources active = details.demand.active_entitlement;
        active_continuations[lane]             = *continuation_index;
        SequenceState& sequence                = continuation_states[*continuation_index];
        sequence.lane                          = lane;
        transaction.root_continuation_index.reset();
        start_sequence(lane, sequence, transaction);
        detail::PhysicalResources actual         = owner_exclusive_resources(sequence);
        actual.device.active_lanes               = 1;
        const detail::PhysicalResources expected = active;
        if (actual != expected) {
            throw std::logic_error("materialized sequence does not match its active entitlement");
        }
        if (details.reuse != ReusePath::Root) {
            if (transaction.state_restored) {
                ++transaction.operations.state_restores;
            } else if (details.source_mode == runtime::PrivateSourceMode::Retain ||
                       transaction.has_shared_source || details.state_fork_required) {
                ++transaction.operations.state_forks;
                ++transaction.operations.historical_fork_hits;
            } else {
                ++transaction.operations.state_moves;
            }
        }
        requests[lane].active_resources   = active;
        requests[lane].optional_resources = details.active_optional_resources;
        invalidate_lane(lane);
        const SequenceHandle handle =
            ContractAccess::make_sequence(this, runtime::LaneId{lane}, lane_epochs[lane]);
        return StartResult{.sequence = handle};
    } catch (...) {
        if (destination && *destination < max_concurrency) {
            const std::uint32_t lane = *destination;
            if (active_continuations[lane] < continuation_capacity) {
                clear_lane_best_effort(active_sequence(lane), requests[lane]);
            } else if (continuation_index) {
                release_continuation_slot_best_effort(*continuation_index);
            }
            invalidate_lane(*destination);
        }
        throw;
    }
}

PendingBatch ProgramImpl::decode(std::span<const SequenceHandle> members,
                                 std::span<const runtime::RoundBudget> budgets,
                                 runtime::ExecutionTiming* failed_timing) {
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
        lanes[row] = lane;
    }
    const auto lane_span = std::span<const std::uint32_t>(lanes.data(), members.size());
    try {
        runtime::BatchedGeneratedRound round = decode_raw(lane_span, budgets, failed_timing);
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
                if (!dflash || !io.dflash_decode || !sequence.kv ||
                    (backend_kv_cache() && !sequence.kv->backend)) {
                    throw std::logic_error("DFlash forced continuation state is incomplete");
                }
                *dflash_host_ingress                            = {};
                dflash_host_ingress->active_lanes[0]            = static_cast<std::int32_t>(lane);
                const StateImageSelectors selectors             = state_selectors(sequence);
                dflash_host_ingress->state_source_slots[0]      = selectors.source;
                dflash_host_ingress->state_destination_slots[0] = selectors.destination;
                dflash_host_ingress->dflash_kv_table_rows[0] =
                    sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend)
                                         : 0;
                CUDA_CHECK(cudaMemcpyAsync(io.dflash_decode->ingress.data, dflash_host_ingress,
                                           sizeof(qwen3_5::DFlashDecodeIngress),
                                           cudaMemcpyHostToDevice, device.stream));
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
                    dflash_host_ingress};
                mark_workspace_usage(speculative_backend == SpeculativeBackend::Mtp
                                         ? workspace_plan.mtp_prefill
                                         : workspace_plan.text_prefill);
                if (is_masked_draft_backend(speculative_backend)) {
                    mark_workspace_usage(workspace_plan.dflash_context);
                }
                const execution::PrefillChunkResult result = execution::prefill_text_chunk(
                    schedule_state, sequence.ledger, count, std::nullopt, false);
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
            advance_rebuild_work(sequence, end, prefill_chunk);
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
        std::array<std::uint8_t, kMaximumConcurrency> cancelled{};
        std::array<std::optional<std::uint32_t>, kMaximumConcurrency> prefix_execution_splits{};
        for (std::size_t row = 0; row < row_count; ++row) {
            const std::uint32_t lane                = ContractAccess::lane(members[row]).value;
            lanes[row]                              = lane;
            const PendingCandidate& candidate       = requests[lane].pending;
            pending_kinds[row]                      = candidate.kind;
            const runtime::CommitDecision& decision = decisions[row];
            if (decision.cancelled && has_context_transaction()) {
                throw std::logic_error(
                    "active cancellation overlaps the global context transaction");
            }
            if ((decision.cancelled && (decision.accepted_tokens != 0 || !decision.terminal)) ||
                (!decision.cancelled &&
                 (decision.accepted_tokens == 0 || decision.accepted_tokens > candidate.produced ||
                  (!decision.terminal && decision.accepted_tokens != candidate.produced))) ||
                (decision.prefix_execution_split_after &&
                 (decision.cancelled || *decision.prefix_execution_split_after == 0 ||
                  *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                throw std::logic_error("pending transaction decision is invalid");
            }
            accepted[row]                = decision.accepted_tokens;
            terminal[row]                = decision.terminal ? 1U : 0U;
            cancelled[row]               = decision.cancelled ? 1U : 0U;
            prefix_execution_splits[row] = decision.prefix_execution_split_after;
            if (decision.cancelled) {
                timings[row]     = requests[lane].timings;
                speculative[row] = std::move(requests[lane].speculative_stats);
            }
        }

        timing.pause();
        timing.include(
            resolve_pending_raw(std::span<const std::uint32_t>(lanes.data(), row_count),
                                std::span<const std::uint32_t>(accepted.data(), row_count),
                                std::span<const std::uint8_t>(terminal.data(), row_count),
                                std::span<const std::uint8_t>(cancelled.data(), row_count),
                                std::span<const std::optional<std::uint32_t>>(
                                    prefix_execution_splits.data(), row_count),
                                failed_timing));
        timing.resume_post();
        pending_transaction_.reset();

        CommitResult out;
        out.row_count          = row_count;
        bool released_resource = false;
        for (std::size_t row = 0; row < row_count; ++row) {
            if (decisions[row].cancelled) {
                invalidate_lane(lanes[row]);
                released_resource = true;
                out.rows[row]     = CommitRowResult{
                        .disposition = runtime::CommitDisposition::CancelledReleased,
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

            if (pending_kinds[row] != PendingKind::Begin || decisions[row].cancelled) { continue; }
            RequestControl& request = requests[lanes[row]];
            if (decisions[row].terminal) {
                request.prefill.reset();
                continue;
            }
            if (!request.prefill) { continue; }
            RequestControl::Prefill& prefill = *request.prefill;
            if (prefill.cursor != prefill.prompt_tokens ||
                prefill.next_capture >= prefill.capture_groups.size() ||
                prefill.capture_groups[prefill.next_capture].frontier != prefill.prompt_tokens ||
                prefill.pending_capture_offer != 0) {
                throw std::logic_error("prompt-frontier capture carrier is inconsistent");
            }
            if (++next_capture_offer_id_ == 0) { ++next_capture_offer_id_; }
            prefill.pending_capture_offer = next_capture_offer_id_;
            out.captures[row].emplace(ContractAccess::make_capture_offer(
                this, runtime::LaneId{lanes[row]}, lane_epochs[lanes[row]],
                prefill.pending_capture_offer));
        }
        if (released_resource) { advance_resource_revision(); }
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
    const bool deferred_to_fail_all = has_context_transaction();
    clear_execution_failure_lanes(
        std::span<const std::uint32_t>(failed_lanes.data(), out.row_count));
    pending_transaction_.reset();
    if (deferred_to_fail_all) { return out; }
    if (out.row_count != 0) { advance_resource_revision(); }
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

FinishResult ProgramImpl::finish(SequenceHandle sequence) noexcept {
    FinishResult out;
    if (has_context_transaction() || pending_transaction_ || !valid_sequence(sequence)) {
        return out;
    }
    const std::uint32_t lane               = ContractAccess::lane(sequence).value;
    RequestControl& request                = requests[lane];
    SequenceState& state                   = active_sequence(lane);
    const std::uint32_t continuation_index = active_continuations[lane];
    if (request.lifecycle != Lifecycle::Finishable) { return out; }
    if (!request.publish_continuation) {
        if (!clear_lane_strict(state, request)) { return out; }
        out.disposition = runtime::FinishDisposition::Released;
        out.timings     = request.timings;
        out.speculative = std::move(request.speculative_stats);
        invalidate_lane(lane);
        advance_resource_revision();
        out.status = runtime::ConsumeStatus::Consumed;
        return out;
    }
    // Every valid terminal execution path settles a borrowed materialization Fork first. A
    // borrowed source cannot become this continuation's direct endpoint; fall back to terminal
    // discard if that publication invariant was not established.
    if (state.state.fork_pending && state.state.borrows_read()) { return out; }
    try {
        out.summary.long_anchors.reserve(state.long_anchors.size());
    } catch (...) { return out; }
    try {
        if (state.state.fork_pending) {
            const StateImageHandle source      = state.state.read;
            const StateImageHandle destination = state.state.write;
            state_store->abort_fork(source, destination);
            if (!state_store->release(destination)) { return out; }
            // An active-capture source is still this sequence's primary lifetime. Publishing it
            // as the endpoint retains that direct ownership; surviving checkpoint references
            // still prevent exclusive attribution and release.
            state.state = ActiveStateBinding{.read = source, .write = source};
        }
        if (state.reserved_state) {
            if (!state_store->release(*state.reserved_state)) { return out; }
            state.reserved_state.reset();
        }
        if (state.rewrite_state && *state.rewrite_state == state.state.read) {
            if (state_store->checkpoint_references(*state.rewrite_state) == 0) { return out; }
            state_store->release_checkpoint_reference(*state.rewrite_state);
            state.rewrite_state.reset();
            state.rewrite_checkpoint = {};
        }
        if (state_store->role(state.state.read) == StateImageRole::ActiveMutable) {
            state_store->freeze(state.state.read);
        } else if (state_store->role(state.state.read) != StateImageRole::CheckpointImmutable) {
            return out;
        }
        state.endpoint_valid = true;
        refresh_state_views(state);
        text_kv_addresses->set_checkpoint_requirement(state.kv->text, state.execution_frontier);
        if (state.kv->backend) {
            backend_kv_addresses->set_checkpoint_requirement(*state.kv->backend,
                                                             backend_kv_valid(state));
        }
        populate_continuation_summary(state, out.summary);
        out.summary.active_references = 0;
    } catch (...) { return out; }
    release_active_shared_references(state);
    release_sequence_growth_entitlement(state);
    unbind_sequence_kv(state);
    request.active_resources                    = {};
    request.optional_resources                  = {};
    request.lifecycle                           = Lifecycle::Empty;
    request.pending                             = {};
    continuation_slots[continuation_index].role = ContinuationSlotRole::Catalogued;
    active_continuations[lane]                  = continuation_capacity;
    invalidate_lane(lane);
    out.continuation.emplace(ContractAccess::make_continuation(
        this, continuation_index, continuation_slots[continuation_index].generation));
    out.timings     = request.timings;
    out.speculative = std::move(request.speculative_stats);
    out.disposition = runtime::FinishDisposition::Catalogued;
    advance_resource_revision();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

AbortResult ProgramImpl::abort(SequenceHandle sequence) noexcept {
    AbortResult out;
    if (has_context_transaction() || pending_transaction_ || !valid_sequence(sequence)) {
        return out;
    }
    const std::uint32_t lane = ContractAccess::lane(sequence).value;
    RequestControl& request  = requests[lane];
    if (request.lifecycle == Lifecycle::Pending || request.lifecycle == Lifecycle::Empty) {
        return out;
    }
    SequenceState& state = active_sequence(lane);
    if (!clear_lane_strict(state, request)) { return out; }
    out.timings     = request.timings;
    out.speculative = std::move(request.speculative_stats);
    invalidate_lane(lane);
    advance_resource_revision();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

ReleaseResult ProgramImpl::release_continuation(ContinuationHandle&& continuation) noexcept {
    ReleaseResult out;
    const std::uint32_t index      = ContractAccess::index(continuation);
    const std::uint64_t generation = ContractAccess::epoch(continuation);
    const bool valid               = !has_context_transaction() && !pending_transaction_ &&
                       valid_continuation(continuation) && !materialization_pins(index, generation);
    if (!valid) { return out; }
    try {
        if (!can_release_continuation_slot_strict(index)) { return out; }
    } catch (...) { return out; }
    release_continuation_slot_strict(index);
    ContractAccess::consume(continuation);
    advance_resource_revision();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

bool ProgramImpl::can_release_shared_prefix_state(std::uint32_t index,
                                                  SharedPrefixSlotRole expected_role) const {
    if (index >= shared_prefix_capacity || !state_store || !text_kv_addresses ||
        shared_prefix_slots[index].role != expected_role) {
        return false;
    }
    const SharedPrefixState& shared = shared_prefix_states[index];
    if (shared.active_references != 0 || !shared.kv || !shared.identity ||
        !state_store->valid(shared.state) || !text_kv_addresses->can_release(shared.kv->text) ||
        (shared.kv->backend &&
         (!backend_kv_addresses || !backend_kv_addresses->can_release(*shared.kv->backend)))) {
        return false;
    }
    const std::uint32_t state_references = state_store->checkpoint_references(shared.state);
    return state_references != 0 &&
           (state_references != 1 ||
            state_store->can_release_after_checkpoint_references(shared.state, 1));
}

detail::PhysicalResources
ProgramImpl::release_shared_prefix_state_strict(std::uint32_t index,
                                                SharedPrefixSlotRole expected_role) noexcept {
    try {
        if (!can_release_shared_prefix_state(index, expected_role)) { std::terminate(); }
        SharedPrefixState& shared               = shared_prefix_states[index];
        SharedPrefixSlot& slot                  = shared_prefix_slots[index];
        const detail::PhysicalResources removed = owner_exclusive_resources(shared);
        const bool last_state_reference = state_store->checkpoint_references(shared.state) == 1;
        if (shared.kv->backend && !backend_kv_addresses->release(*shared.kv->backend)) {
            std::terminate();
        }
        if (!text_kv_addresses->release(shared.kv->text)) { std::terminate(); }
        state_store->release_checkpoint_reference(shared.state);
        if (last_state_reference && !state_store->release(shared.state)) { std::terminate(); }

        shared    = SharedPrefixState{};
        slot.role = SharedPrefixSlotRole::Free;
        if (++slot.generation == 0) { ++slot.generation; }
        if (host_kv_extents) { (void)host_kv_extents->release_unreferenced(); }
        return removed;
    } catch (...) { std::terminate(); }
}

ReleaseResult ProgramImpl::release_shared_prefix(SharedPrefixHandle&& handle) noexcept {
    ReleaseResult out;
    const std::uint32_t index      = ContractAccess::index(handle);
    const std::uint64_t generation = ContractAccess::epoch(handle);
    const bool valid =
        !has_context_transaction() && !pending_transaction_ && valid_shared_prefix(handle);
    if (!valid || index >= shared_prefix_capacity ||
        shared_prefix_slots[index].generation != generation) {
        return out;
    }
    try {
        if (!can_release_shared_prefix_state(index, SharedPrefixSlotRole::Catalogued)) {
            return out;
        }
    } catch (...) { return out; }
    (void)release_shared_prefix_state_strict(index, SharedPrefixSlotRole::Catalogued);
    ContractAccess::consume(handle);
    advance_resource_revision();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

void ProgramImpl::fail_all_cleanup() noexcept {
    pending_transaction_.reset();
    if (auto* transaction = std::get_if<ActiveCaptureTransaction>(&context_transaction_)) {
        if (transaction->transfer_submitted && device.transfer_stream != nullptr) {
            (void)cudaStreamSynchronize(device.transfer_stream);
        }
        abort_active_capture(*transaction);
    }
    if (auto* transaction = std::get_if<MaterializationTransaction>(&context_transaction_)) {
        if (transaction->transfer_submitted && device.transfer_stream != nullptr) {
            (void)cudaStreamSynchronize(device.transfer_stream);
        }
        release_materialization_staging(*transaction);
    }
    context_transaction_.emplace<std::monostate>();
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (active_continuations[lane] < continuation_capacity) {
            clear_lane_best_effort(active_sequence(lane), requests[lane]);
        }
        invalidate_lane(lane);
    }
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role != ContinuationSlotRole::Free) {
            release_continuation_slot_best_effort(index);
        }
    }
    for (std::uint32_t index = 0; index < shared_prefix_capacity; ++index) {
        if (shared_prefix_slots[index].role != SharedPrefixSlotRole::Catalogued) { continue; }
        shared_prefix_states[index].active_references = 0;
        auto handle =
            ContractAccess::make_shared_prefix(this, index, shared_prefix_slots[index].generation);
        (void)release_shared_prefix(std::move(handle));
    }
}


} // namespace ninfer::models::qwen3_5::detail
