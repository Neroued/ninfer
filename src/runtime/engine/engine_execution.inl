#pragma once

namespace ninfer::runtime {

template <class Instance>
void EngineCore<Instance>::commit_pending(PendingBatch&& pending, ComputeFlight flight) {
    if (retiring_) throw std::logic_error("another model commit is still retiring");
    EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
    const auto lane_indices     = std::span<const std::uint32_t>(flight.lanes);
    const bool decode_round     = flight.kind == UnitKind::Decode;
    const std::size_t row_count = flight.requests.size();
    for (const auto& request : flight.requests)
        request->pending_execution = pending.execution_sample();
    if (row_count == 0 || row_count != pending.row_count() || pending.row_stride() == 0 ||
        (!pending.row_counts().empty() && pending.row_counts().size() != row_count) ||
        pending.tokens().size() < static_cast<std::size_t>(pending.row_stride()) * row_count) {
        (void)instance_.session->discard(std::move(pending));
        throw std::logic_error("pending batch returned an invalid ragged layout");
    }

    std::array<CommitDecision, kMaximumConcurrency> decisions{};
    std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
    std::array<ContinuationAction, kMaximumConcurrency> continuations{};
    std::array<std::size_t, kMaximumConcurrency> generated_sizes{};
    std::array<bool, kMaximumConcurrency> cancelled{};
    std::array<bool, kMaximumConcurrency> failed{};
    bool generated_staged         = false;
    const auto rollback_generated = [&]() {
        for (std::size_t row = 0; row < row_count; ++row) {
            const auto& request = flight.requests[row];
            if (request != nullptr) { request->output.discard_preview(); }
            if (generated_staged && request != nullptr &&
                request->generated.size() >= generated_sizes[row]) {
                request->generated.resize(generated_sizes[row]);
            }
        }
        generated_staged = false;
    };
    try {
        for (std::size_t row = 0; row < row_count; ++row) {
            const std::uint32_t lane = lane_indices[row];
            const auto& request      = flight.requests[row];
            if (request == nullptr || !request->sequence || !request->lane ||
                request->lane->value != lane || !request->budget ||
                request->sequence != flight.sequences[row] ||
                request->residency != flight.residencies[row]) {
                throw std::logic_error("pending row has no active Engine request");
            }
            if (decode_round) { ++request->host_timing.decode_rounds; }
            cancelled[row] = flight.cancelled[row];
            failed[row]    = !cancelled[row] && pending.constraint_failed(row);
            const std::int32_t raw_count =
                pending.row_counts().empty() ? 1 : pending.row_counts()[row];
            if (raw_count <= 0 || raw_count > static_cast<std::int32_t>(pending.row_stride())) {
                throw std::logic_error("pending row has an invalid licensed extent");
            }
            const std::uint32_t count = static_cast<std::uint32_t>(raw_count);
            const auto row_tokens     = pending.tokens().subspan(row * pending.row_stride(),
                                                                 static_cast<std::size_t>(count));
            generated_sizes[row]      = request->generated.size();
            if (failed[row]) {
                decisions[row] = CommitDecision{.terminal = true, .failed = true};
                continue;
            }
            if (cancelled[row]) {
                (void)request->output.preview_terminal(FinishReason::Cancelled);
                decisions[row] = CommitDecision{
                    .accepted_tokens = 0,
                    .terminal        = true,
                    .cancelled       = true,
                };
                finish_reasons[row] = FinishReason::Cancelled;
                continue;
            }
            const OutputDecision decision = request->output.preview_model(
                row_tokens, request->budget->remaining(), request->budget->limit_reason());
            if (decision.accepted_tokens == 0 || decision.accepted_tokens > count ||
                (!decision.finished() && decision.accepted_tokens != count) ||
                (decision.finished() && decision.continuation != ContinuationAction::Decode) ||
                (decision.prefix_execution_split_after &&
                 (*decision.prefix_execution_split_after == 0 ||
                  *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                throw std::logic_error("output policy returned an invalid licensed prefix");
            }
            decisions[row] = CommitDecision{
                .accepted_tokens              = decision.accepted_tokens,
                .terminal                     = decision.finished(),
                .cancelled                    = false,
                .prefix_execution_split_after = decision.prefix_execution_split_after,
            };
            finish_reasons[row] = decision.finish_reason;
            continuations[row]  = decision.continuation;
        }
        generated_staged = true;
        for (std::size_t row = 0; row < row_count; ++row) {
            const std::uint32_t accepted = decisions[row].accepted_tokens;
            if (accepted == 0) { continue; }
            const auto& request = flight.requests[row];
            if (request->generated.size() > request->generated.capacity() ||
                accepted > request->generated.capacity() - request->generated.size()) {
                throw std::logic_error("admission did not reserve generated-token capacity");
            }
            const auto first =
                pending.tokens().begin() + static_cast<std::ptrdiff_t>(row * pending.row_stride());
            request->generated.insert(request->generated.end(), first,
                                      first + static_cast<std::ptrdiff_t>(accepted));
        }
    } catch (...) {
        const std::exception_ptr error = std::current_exception();
        rollback_generated();
        const auto discarded = instance_.session->discard(std::move(pending));
        if (discarded.status != ConsumeStatus::Consumed) {
            throw std::logic_error("failed pending batch could not be discarded");
        }
        std::rethrow_exception(error);
    }

    std::optional<typename ModelContract::CommitResult> committed_storage;
    try {
        phase.pause_range();
        ExecutionCallScope program_call(*this);
        auto committed = instance_.session->commit(
            std::move(pending), std::span<const CommitDecision>(decisions.data(), row_count));
        program_call.finish();
        committed_storage.emplace(std::move(committed));
        phase.resume_range();
    } catch (...) {
        rollback_generated();
        throw;
    }
    generated_staged = false;
    auto& committed  = *committed_storage;
    if (committed.row_count != row_count) {
        throw std::logic_error("Runtime commit result is not row aligned");
    }
    for (std::size_t row = 0; row < row_count; ++row) {
        const CommitDisposition expected = failed[row]      ? CommitDisposition::FailedReleased
                                           : cancelled[row] ? CommitDisposition::CancelledReleased
                                           : decisions[row].terminal ? CommitDisposition::Finishable
                                                                     : CommitDisposition::Active;
        if (committed.rows[row].disposition != expected) {
            throw std::logic_error("Runtime commit row disposition is invalid");
        }
        if (committed.capture_ready[row] &&
            (decode_round || expected != CommitDisposition::Active)) {
            throw std::logic_error("Runtime exposed a capture outside a committed Begin row");
        }
    }
    const bool terminal_in_batch =
        std::any_of(decisions.begin(), decisions.begin() + static_cast<std::ptrdiff_t>(row_count),
                    [](const CommitDecision& decision) { return decision.terminal; });

    for (std::size_t row = 0; row < row_count; ++row) {
        const auto& request  = flight.requests[row];
        auto& observed       = request->speculative_stats;
        const auto& counters = committed.rows[row].speculative;
        cumulative_stats_.speculative_rounds += counters.rounds - observed.rounds;
        cumulative_stats_.speculative_draft_tokens +=
            counters.drafted_tokens - observed.drafted_tokens;
        cumulative_stats_.speculative_accepted_tokens +=
            counters.accepted_tokens - observed.accepted_tokens;
        cumulative_stats_.speculative_fallback_steps +=
            counters.fallback_steps - observed.fallback_steps;
        observed.rounds                 = counters.rounds;
        observed.drafted_tokens         = counters.drafted_tokens;
        observed.accepted_tokens        = counters.accepted_tokens;
        observed.fallback_steps         = counters.fallback_steps;
        observed.committed_draft_tokens = counters.committed_draft_tokens;
        if (cancelled[row] || failed[row]) {
            request->speculative_stats = std::move(committed.rows[row].speculative);
        }
    }

    if (decode_round) {
        for (std::size_t row = 0; row < row_count; ++row) {
            if (!cancelled[row]) {
                cumulative_stats_.committed_decode_tokens += decisions[row].accepted_tokens;
                flight.requests[row]->committed_decode_tokens += decisions[row].accepted_tokens;
            }
        }
    }

    for (std::size_t row = 0; row < row_count; ++row) {
        const std::uint32_t lane     = lane_indices[row];
        const auto& request          = flight.requests[row];
        const std::uint32_t accepted = decisions[row].accepted_tokens;
        if (failed[row]) {
            request->terminal_error     = std::make_exception_ptr(RequestError(
                RequestErrorKind::ConstraintDeadEnd, "grammar has no valid next token"));
            request->consumed_by_commit = true;
            request->model_state        = EngineRequestState::ModelFinished;
            continue;
        }
        if (!cancelled[row]) { request->budget->commit(accepted); }
        request->consumed_by_commit = cancelled[row];
        update_recovery(request, cancelled[row]);
        auto published = request->output.commit_preview();
        auto timing    = record_committed_output(request, accepted);
        append_output(request, std::move(published), std::move(timing), &phase);
        if (request->output.output_error()) {
            if (committed.capture_ready[row]) {
                instance_.session->skip_capture(*request->sequence);
                committed.capture_ready[row] = false;
            }
            if (fail_output(request, lane)) continue;
        }
        if (decisions[row].terminal) {
            request->model_state     = EngineRequestState::ModelFinished;
            request->terminal_reason = finish_reasons[row];
        } else if (committed.capture_ready[row]) {
            if (!request->is_prefilling()) {
                throw std::logic_error("prompt-frontier capture lost its prefill owner");
            }
            const EngineRequestState post_capture_state =
                continuations[row] == ContinuationAction::ApplyTargetControl
                    ? EngineRequestState::ControlReady
                    : EngineRequestState::DecodeReady;
            if (terminal_in_batch) {
                instance_.session->skip_capture(*request->sequence);
                request->model_state = post_capture_state;
            } else {
                reserve_active_capture(request, post_capture_state);
            }
            committed.capture_ready[row] = false;
            if (!request->capture_pending) {
                request->model_state = continuations[row] == ContinuationAction::ApplyTargetControl
                                           ? EngineRequestState::ControlReady
                                           : EngineRequestState::DecodeReady;
            }
        } else {
            request->model_state = continuations[row] == ContinuationAction::ApplyTargetControl
                                       ? EngineRequestState::ControlReady
                                       : EngineRequestState::DecodeReady;
        }
    }
    phase.finish();
    retiring_.emplace(std::move(flight));
}

template <class Instance>
void EngineCore<Instance>::resolve_prefill_progress(
    const std::shared_ptr<Request>& request, typename ModelContract::PrefillProgress&& progress,
    ComputeFlight flight) {
    EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
    ++request->host_timing.prefill_units;
    cumulative_stats_.computed_prefill_tokens += progress.processed_prompt_tokens;
    if (!request->admitted_begin) {
        throw std::logic_error("prefill progress has no committed admission summary");
    }
    const BeginSummary& begin = *request->admitted_begin;
    if (begin.reused_prompt_tokens > begin.prompt_tokens) {
        throw std::logic_error("admitted prefix exceeds its prompt");
    }
    const std::uint32_t suffix_tokens = begin.prompt_tokens - begin.reused_prompt_tokens;
    if (request->computed_prompt_tokens > suffix_tokens ||
        progress.processed_prompt_tokens > suffix_tokens - request->computed_prompt_tokens) {
        throw std::logic_error("prefill unit exceeded the admitted prompt suffix");
    }
    const auto computed_begin = begin.reused_prompt_tokens + request->computed_prompt_tokens;
    request->computed_prompt_tokens += progress.processed_prompt_tokens;
    if (!request->cancelled.load(std::memory_order_acquire)) {
        resources_.observe_committed(*request->base_plan, request->id, computed_begin,
                                     computed_begin + progress.processed_prompt_tokens,
                                     instance_.session->capture_frontiers(*request->sequence));
    }
    if (progress.complete && request->computed_prompt_tokens != suffix_tokens) {
        throw std::logic_error("completed prefill did not reach the admitted prompt frontier");
    }
    if (progress.processed_prompt_tokens != 0) { publish_prompt_progress(request); }
    if (progress.capture_ready) {
        if (progress.complete || progress.pending) {
            throw std::logic_error("prefill capture offer overlaps prompt completion");
        }
        reserve_active_capture(request, EngineRequestState::Prefill);
        return;
    }
    if (!progress.complete) { return; }
    if (!request->lane || !progress.pending) {
        throw std::logic_error("completed prefill has no lane or pending token");
    }
    if (!request->admitted_begin || progress.summary != *request->admitted_begin) {
        throw std::logic_error("runtime Begin summary differs from committed admission");
    }
    request->begin = progress.summary;
    phase.finish();
    commit_pending(std::move(*progress.pending), std::move(flight));
    progress.pending.reset();
}

template <class Instance>
[[nodiscard]] typename EngineCore<Instance>::ComputeFlight
EngineCore<Instance>::freeze_members(std::span<const std::uint32_t> lanes, UnitKind kind) const {
    ComputeFlight flight;
    flight.kind = kind;
    flight.requests.reserve(lanes.size());
    flight.sequences.reserve(lanes.size());
    flight.residencies.reserve(lanes.size());
    flight.lanes.assign(lanes.begin(), lanes.end());
    flight.cancelled.reserve(lanes.size());
    for (const auto lane : lanes) {
        const auto& request = slots_.at(lane);
        if (!request || !request->sequence || !request->residency || !request->lane ||
            request->lane->value != lane) {
            throw std::logic_error("execution membership has no resident request");
        }
        flight.requests.push_back(request);
        flight.sequences.push_back(*request->sequence);
        flight.residencies.push_back(*request->residency);
        flight.cancelled.push_back(request->cancelled.load(std::memory_order_acquire));
    }
    return flight;
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::start_prefill(std::uint32_t lane) {
    const std::array<std::uint32_t, 1> lanes{lane};
    const auto& request = slots_[lane];
    auto flight =
        freeze_members(lanes, request->is_replaying() ? UnitKind::Replay : UnitKind::Prefill);
    ExecutionCallScope call(*this);
    const auto ticket = request->is_replaying()
                            ? instance_.session->start_replay(flight.sequences.front())
                            : instance_.session->start_prefill(flight.sequences.front(),
                                                               request->output.constrained());
    call.finish();
    if (!ticket) return false;
    flight.ticket = *ticket;
    compute_.emplace(std::move(flight));
    reserved_units_[lane].reset();
    scheduler_.prefill_executed(lane, max_concurrency_);
    return true;
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::start_decode(const RoundMembership& membership) {
    auto flight = freeze_members(membership.lane_span(), UnitKind::Decode);
    std::array<bool, kMaximumConcurrency> constrained{};
    for (std::size_t row = 0; row < flight.requests.size(); ++row) {
        constrained[row] = flight.requests[row]->output.constrained();
    }
    ExecutionCallScope call(*this);
    const auto ticket = instance_.session->start_decode(
        flight.sequences, membership.budget_span(), {constrained.data(), flight.requests.size()});
    call.finish();
    if (!ticket) return false;
    flight.ticket = *ticket;
    for (const auto lane : flight.lanes) reserved_units_[lane].reset();
    compute_.emplace(std::move(flight));
    return true;
}

template <class Instance>
void EngineCore<Instance>::discard_control_preview(const ComputeFlight& flight) noexcept {
    for (std::size_t row = 0; row < flight.requests.size(); ++row) {
        const auto& request = flight.requests[row];
        request->output.discard_preview();
        if (row < flight.generated_sizes.size()) {
            request->generated.resize(flight.generated_sizes[row]);
        }
    }
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::start_control(const ControlMembership& membership) {
    if (membership.empty() || membership.row_stride == 0 ||
        membership.tokens.size() != membership.size * membership.row_stride) {
        throw std::logic_error("thinking control membership is invalid");
    }
    auto flight           = freeze_members(membership.lane_span(), UnitKind::Control);
    flight.control_tokens = membership.tokens;
    flight.control_stride = membership.row_stride;
    flight.generated_sizes.reserve(flight.requests.size());
    for (const auto& request : flight.requests) {
        flight.generated_sizes.push_back(request->generated.size());
    }
    std::array<std::optional<std::uint32_t>, kMaximumConcurrency> splits{};
    try {
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        for (std::size_t row = 0; row < flight.requests.size(); ++row) {
            const auto& request = flight.requests[row];
            const auto tokens   = std::span<const TokenId>(flight.control_tokens)
                                    .subspan(row * flight.control_stride, flight.control_stride);
            const auto decision =
                request->output.preview_control(tokens, request->budget->remaining());
            if (decision.accepted_tokens != flight.control_stride || decision.finished() ||
                decision.continuation != ContinuationAction::Decode ||
                (decision.prefix_execution_split_after &&
                 (*decision.prefix_execution_split_after == 0 ||
                  *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                throw std::logic_error("target control preview returned an invalid decision");
            }
            splits[row] = decision.prefix_execution_split_after;
            if (tokens.size() > request->generated.capacity() - request->generated.size()) {
                throw std::logic_error("admission did not reserve control token capacity");
            }
            request->generated.insert(request->generated.end(), tokens.begin(), tokens.end());
        }
        phase.finish();
        ExecutionCallScope call(*this);
        const auto ticket = instance_.session->start_control(
            flight.sequences, flight.control_tokens, flight.control_stride,
            {splits.data(), flight.requests.size()});
        call.finish();
        if (!ticket) {
            discard_control_preview(flight);
            return false;
        }
        flight.ticket = *ticket;
    } catch (...) {
        discard_control_preview(flight);
        throw;
    }
    for (const auto lane : flight.lanes) {
        reserved_units_[lane].reset();
        control_executed_[lane] = true;
    }
    compute_.emplace(std::move(flight));
    return true;
}

template <class Instance>
void EngineCore<Instance>::resolve_control(const ComputeFlight& flight) {
    EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
    cumulative_stats_.committed_decode_tokens += flight.requests.size() * flight.control_stride;
    for (std::size_t row = 0; row < flight.requests.size(); ++row) {
        const auto& request = flight.requests[row];
        ++request->host_timing.control_units;
        request->committed_decode_tokens += flight.control_stride;
        request->budget->commit(flight.control_stride);
        auto timing = record_committed_output(request, flight.control_stride);
        append_output(request, request->output.commit_preview(), std::move(timing), &phase);
        if (fail_output(request, flight.lanes[row])) continue;
        update_recovery(request);
        request->model_state = EngineRequestState::DecodeReady;
    }
}

template <class Instance>
void EngineCore<Instance>::supply_decision(ExecutionTicket ticket,
                                           const DecisionRequest& decision) {
    if (!compute_ || compute_->ticket != ticket || decision.vocabulary == 0) {
        throw std::logic_error("decision input does not belong to an active execution");
    }
    DecisionInput input;
    input.stage = decision.stage;
    input.rows.reserve(decision.rows.size());
    std::array<bool, kMaximumConcurrency> seen{};
    const std::size_t words_per_position = (std::size_t(decision.vocabulary) + 31) / 32;
    for (const auto& row : decision.rows) {
        if (row.member >= compute_->requests.size() || seen[row.member]) {
            throw std::logic_error("decision request has invalid compact membership");
        }
        seen[row.member]    = true;
        const auto& request = compute_->requests[row.member];
        if (!request->output.constrained()) {
            throw std::logic_error("decision requested a mask for an unconstrained row");
        }
        auto& result     = input.rows.emplace_back();
        result.member    = row.member;
        result.positions = static_cast<std::uint32_t>(row.drafts.size() + 1);
        result.words.resize(words_per_position * result.positions);
        result.dead_positions = request->output.grammar_masks(row.drafts, result.words);
    }
    ExecutionCallScope call(*this);
    instance_.session->supply(ticket, std::move(input));
    call.finish();
}

template <class Instance>
void EngineCore<Instance>::record_decision_upload(const DecisionConsumed& consumed) {
    if (!compute_ || compute_->ticket != consumed.ticket) {
        throw std::logic_error("decision upload does not belong to an active execution");
    }
    for (const auto& row : consumed.rows) {
        if (row.member >= compute_->requests.size()) {
            throw std::logic_error("decision upload has invalid compact membership");
        }
        compute_->requests[row.member]->output.constraint_uploaded(row.bytes);
    }
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::poll_execution(HostPhaseMeasurement& boundary) {
    const bool context_was_pending = instance_.session->has_context_operation();
    const auto owner               = materializing_ ? materializing_->request : context_owner_;
    if (owner && owner->cancelled.load(std::memory_order_acquire) &&
        instance_.session->has_context_operation()) {
        instance_.session->cancel_context();
    }
    bool progressed = false;
    for (;;) {
        ExecutionCallScope call(*this, CoordinatorCall::Poll);
        auto event = instance_.session->poll();
        call.finish();
        if (!event) break;
        progressed = true;
        if (auto* decision = std::get_if<DecisionRequest>(&event->result)) {
            supply_decision(event->ticket, *decision);
            continue;
        }
        if (const auto* consumed = std::get_if<DecisionConsumed>(&event->result)) {
            record_decision_upload(*consumed);
            continue;
        }
        if (auto* context = std::get_if<typename ModelContract::ContextProgress>(&event->result)) {
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            const auto work_class   = current_host_work_class_;
            const auto decode_lanes = current_decode_lanes_;
            const auto decode_count = current_decode_lane_count_;
            set_host_work_class(HostWorkClass::Control);
            boundary = begin_host_phase();
            resolve_context_progress(std::move(*context), boundary);
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            set_host_work_class(work_class, {decode_lanes.data(), decode_count});
            boundary = begin_host_phase();
            continue;
        }
        if (const auto* complete = std::get_if<ExecutionComplete>(&event->result)) {
            const bool retiring = retiring_ && retiring_->ticket == event->ticket;
            auto& flight        = retiring ? retiring_ : compute_;
            if (!flight || flight->ticket != event->ticket || flight->sequences != event->members ||
                complete->row_count != flight->requests.size())
                throw std::logic_error("execution completion changed its frozen membership");
            const auto kind = static_cast<std::size_t>(complete->sample.work.kind);
            for (const auto& request : flight->requests) {
                request->executions[kind].observe(complete->sample.latency);
                if (complete->sample.work.kind == ExecutionKind::Decode)
                    request->decode_by_batch[flight->requests.size() - 1].observe(
                        complete->sample.latency);
                accumulate_gpu(request->max_rank_gpu, complete->sample.max_rank_gpu);
                request->pending_execution.reset();
            }
            if (retiring) retiring_.reset();
            continue;
        }
        if (!compute_ || compute_->ticket != event->ticket ||
            compute_->sequences != event->members) {
            throw std::logic_error("execution result changed its frozen membership");
        }
        auto flight = std::move(*compute_);
        compute_.reset();
        set_host_work_class(flight.kind == UnitKind::Decode    ? HostWorkClass::Decode
                            : flight.kind == UnitKind::Control ? HostWorkClass::Control
                                                               : HostWorkClass::Prefill,
                            flight.kind == UnitKind::Decode
                                ? std::span<const std::uint32_t>(flight.lanes)
                                : std::span<const std::uint32_t>{});
        if (auto* pending = std::get_if<PendingBatch>(&event->result)) {
            commit_pending(std::move(*pending), std::move(flight));
        } else if (auto* prefill =
                       std::get_if<typename ModelContract::PrefillProgress>(&event->result)) {
            const auto request = flight.requests.front();
            resolve_prefill_progress(request, std::move(*prefill), std::move(flight));
            update_recovery(request);
        } else if (auto* replay =
                       std::get_if<typename ModelContract::ReplayProgress>(&event->result)) {
            const auto& request = flight.requests.front();
            request->replayed_tokens += replay->processed_tokens;
            cumulative_stats_.replayed_tokens += replay->processed_tokens;
            if (replay->capture_ready) reserve_active_capture(request, EngineRequestState::Replay);
            if (replay->complete) {
                ++cumulative_stats_.replay_restores;
                ++request->replay_restores;
                request->model_state = request->resume_phase;
                observe_scheduling(request, GenerationSchedulingTransition::ReplayComplete);
            }
            update_recovery(request);
        } else if (std::holds_alternative<typename ModelContract::ControlComplete>(event->result)) {
            resolve_control(flight);
        } else {
            throw std::logic_error("generation Session emitted a scoring result");
        }
    }
    if (context_was_pending && !instance_.session->has_context_operation()) {
        scheduler_.capacity_released();
        request_admission_check();
        progressed = true;
    }
    return progressed;
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::execution_outstanding() const noexcept {
    return compute_ || retiring_ || instance_.session->has_context_operation() ||
           instance_.executor->has_pending_work();
}

template <class Instance>
bool EngineCore<Instance>::compute_pending() const noexcept {
    if (compute_ || retiring_) return true;
    return std::any_of(slots_.begin(), slots_.end(), [&](const auto& request) {
        return request && request->sequence &&
               instance_.session->execution_pending(*request->sequence);
    });
}

template <class Instance>
void EngineCore<Instance>::fail_all_locked(std::exception_ptr error) noexcept {
    std::deque<std::shared_ptr<Request>> pending;
    {
        std::lock_guard lock(queue_mutex_);
        failed_ = true;
        pending.swap(pending_);
    }
    scheduler_ = Scheduling{};
    const std::shared_ptr<Request> materializing_request =
        materializing_ ? materializing_->request : nullptr;
    // Native retires all transfer readers before any external checkpoint owner is destroyed.
    instance_.session->fail_all_cleanup();
    if (compute_ && compute_->kind == UnitKind::Control) discard_control_preview(*compute_);
    compute_.reset();
    retiring_.reset();
    cycle_step_ = CycleStep::Boundary;
    materializing_.reset();
    admission_decision_.reset();
    for (auto& decision : capture_decisions_) { decision.reset(); }
    context_owner_.reset();
    for (auto& request : paused_) {
        request->suspended.reset();
        complete_error(request, error);
    }
    paused_.clear();
    resources_.release_all(*instance_.session);
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        if (slots_[lane] != nullptr) {
            complete_error(slots_[lane], error);
            slots_[lane].reset();
        }
    }
    if (materializing_request != nullptr) { complete_error(materializing_request, error); }
    for (const auto& request : pending) { complete_error(request, error); }
    publish_runtime_stats();
}

} // namespace ninfer::runtime
