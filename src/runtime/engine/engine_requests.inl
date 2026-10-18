#pragma once

namespace ninfer::runtime {

template <class Instance>
GenerationResult EngineCore<Instance>::wait_for_request(std::shared_ptr<Request> request,
                                                        OutputSink* sink,
                                                        const CancellationView& cancellation) {
    struct ConsumerGuard {
        EngineCore* owner;
        std::shared_ptr<Request> request;

        ~ConsumerGuard() { owner->release_consumer(request); }
    } guard{this, request};

    std::exception_ptr caller_error;
    bool scheduling_failed   = false;
    const auto fail_consumer = [&] {
        if (caller_error == nullptr) { caller_error = std::current_exception(); }
        request->cancelled.store(true, std::memory_order_release);
        request_admission_check();
        queue_cv_.notify_one();
    };
    std::optional<GenerationStart> start;
    std::optional<PromptProgress> progress;
    std::vector<typename Request::StreamEvent> events;
    for (;;) {
        start.reset();
        progress.reset();
        events.clear();
        bool done = false;
        {
            std::unique_lock lock(request->mutex);
            request->cv.wait_for(lock, std::chrono::milliseconds(10), [&] {
                return request->response_done || request->stream_start.has_value() ||
                       request->stream_progress.has_value() || !request->events.empty();
            });
            start = std::move(request->stream_start);
            request->stream_start.reset();
            progress = std::move(request->stream_progress);
            request->stream_progress.reset();
            events.swap(request->events);
            done = request->response_done;
        }

        if (caller_error == nullptr && sink != nullptr) {
            try {
                if (start) { sink->start(std::move(*start)); }
                if (progress) { sink->progress(std::move(*progress)); }
            } catch (...) { fail_consumer(); }
        }
        for (auto& event : events) {
            if (const auto* scheduling =
                    std::get_if<std::unique_ptr<GenerationSchedulingObservation>>(&event)) {
                if (scheduling_failed) { continue; }
                try {
                    request->observation.scheduling(**scheduling);
                } catch (...) {
                    scheduling_failed = true;
                    fail_consumer();
                }
            } else if (const auto* first_token =
                           std::get_if<GenerationFirstTokenObservation>(&event)) {
                if (caller_error == nullptr) {
                    try {
                        request->observation.first_token(*first_token);
                    } catch (...) { fail_consumer(); }
                }
            } else if (caller_error == nullptr && sink != nullptr) {
                try {
                    if (auto* timing = std::get_if<GenerationTimingObservation>(&event)) {
                        sink->timing(std::move(*timing));
                    } else {
                        sink->publish(std::move(std::get<OutputEvent>(event)));
                    }
                } catch (...) { fail_consumer(); }
            }
        }

        if (caller_error == nullptr) {
            try {
                if (cancellation.requested()) {
                    request->cancelled.store(true, std::memory_order_release);
                    request_admission_check();
                    queue_cv_.notify_one();
                }
            } catch (...) { fail_consumer(); }
        }
        if (!done) { continue; }

        if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
        std::lock_guard lock(request->mutex);
        if (request->error != nullptr) { std::rethrow_exception(request->error); }
        return std::move(request->result);
    }
}

template <class Instance>
[[nodiscard]] std::optional<GenerationTimingObservation>
EngineCore<Instance>::record_committed_output(const std::shared_ptr<Request>& request,
                                              std::uint32_t accepted_tokens) {
    if (accepted_tokens == 0) { return std::nullopt; }
    cumulative_stats_.generated_tokens += accepted_tokens;
    const bool observe_wall =
        request->observation.phase_timings || request->observation.live_timings;
    const bool need_now         = !request->first_token || observe_wall;
    const Clock::time_point now = need_now ? Clock::now() : Clock::time_point{};
    if (!request->first_token) {
        request->first_token = now;
        if (request->observation.first_token) {
            GenerationFirstTokenObservation observation{
                .prepare_seconds = request->prepare_seconds,
                .elapsed_since_submit_seconds =
                    std::chrono::duration<double>(now - request->submitted).count(),
                .queue_wait_seconds =
                    request->admitted_at
                        ? std::chrono::duration<double>(*request->admitted_at - request->submitted)
                              .count()
                        : 0.0,
            };
            {
                std::lock_guard lock(request->mutex);
                request->events.emplace_back(observation);
            }
            request->cv.notify_one();
        }
    }
    if (!observe_wall) { return std::nullopt; }
    if (!request->admitted_at || !request->first_token) {
        throw std::logic_error("committed output has no observed admission boundary");
    }
    request->last_token = now;
    if (!request->observation.live_timings) { return std::nullopt; }
    if (request->generated.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("generated token count exceeds observation domain");
    }
    return GenerationTimingObservation{
        .generated_tokens      = static_cast<std::uint32_t>(request->generated.size()),
        .prompt_elapsed_ns     = elapsed_ns(*request->admitted_at, *request->first_token),
        .generation_elapsed_ns = elapsed_ns(*request->first_token, *request->last_token),
    };
}

template <class Instance>
void EngineCore<Instance>::append_output(const std::shared_ptr<Request>& request,
                                         PublishedOutput output,
                                         std::optional<GenerationTimingObservation> timing,
                                         EnginePhaseScope* phase) {
    if (output.empty() && !timing) { return; }
    if (!request->first_output_timing &&
        std::any_of(output.begin(), output.end(), [](const OutputEvent& event) {
            if (const auto* text = std::get_if<OutputDelta>(&event)) return !text->text.empty();
            if (const auto* arguments = std::get_if<ToolArgumentsDelta>(&event))
                return !arguments->text.empty();
            return false;
        })) {
        if (phase) { phase->checkpoint(); }
        record_first_output(request);
    }
    const bool streaming = request->consumer_mode == OutputConsumerMode::Streaming;
    {
        std::lock_guard lock(request->mutex);
        if (streaming && timing) { request->events.emplace_back(std::move(*timing)); }
        for (OutputEvent& event : output) {
            if (const auto* delta = std::get_if<OutputDelta>(&event)) {
                auto& full = delta->channel == OutputChannel::Reasoning ? request->reasoning
                                                                        : request->content;
                full += delta->text;
            }
            if (streaming) { request->events.emplace_back(std::move(event)); }
        }
    }
    if (streaming) { request->cv.notify_one(); }
}

template <class Instance>
bool EngineCore<Instance>::fail_output(const std::shared_ptr<Request>& request,
                                       std::uint32_t lane) {
    const auto error = request->output.output_error();
    if (!error) return false;
    if (slots_[lane] != request) {
        throw std::logic_error("invalid tool output lost its request binding");
    }
    request->terminal_error = error->exception();
    request->model_state    = EngineRequestState::ModelFinished;
    return true;
}

template <class Instance>
void EngineCore<Instance>::publish_prompt_progress(const std::shared_ptr<Request>& request) {
    if (!request->observation.prompt_progress) { return; }
    if (!request->admitted_begin || !request->admitted_at) {
        throw std::logic_error("prompt progress has no admitted request boundary");
    }
    const BeginSummary& begin = *request->admitted_begin;
    if (begin.reused_prompt_tokens > begin.prompt_tokens ||
        request->computed_prompt_tokens > begin.prompt_tokens - begin.reused_prompt_tokens) {
        throw std::logic_error("prompt progress exceeds the admitted prompt frontier");
    }
    const PromptProgress progress{
        .total_prompt_tokens     = begin.prompt_tokens,
        .reused_prompt_tokens    = begin.reused_prompt_tokens,
        .processed_prompt_tokens = begin.reused_prompt_tokens + request->computed_prompt_tokens,
        .elapsed_ns              = elapsed_ns(*request->admitted_at, Clock::now()),
    };
    {
        std::lock_guard lock(request->mutex);
        if (request->response_done) { return; }
        request->stream_progress = progress;
    }
    request->cv.notify_one();
}

template <class Instance>
void EngineCore<Instance>::publish_generation_start(const std::shared_ptr<Request>& request,
                                                    BeginSummary begin) {
    if (request->admitted_begin) {
        throw std::logic_error("request admission published generation start twice");
    }
    request->admitted_begin = begin;
    if (request->observation.phase_timings || request->observation.live_timings ||
        request->observation.prompt_progress) {
        if (!request->admitted_at) {
            throw std::logic_error("generation start has no binding timestamp");
        }
    }
    if (request->consumer_mode != OutputConsumerMode::Streaming) { return; }
    {
        std::lock_guard lock(request->mutex);
        if (request->stream_start || request->response_done) {
            throw std::logic_error("streaming request has an invalid generation-start state");
        }
        request->stream_start = GenerationStart{
            .prompt               = request->prompt_summary,
            .reused_prompt_tokens = begin.reused_prompt_tokens,
        };
    }
    request->cv.notify_one();
}

template <class Instance>
void EngineCore<Instance>::release_reserved_capacity() noexcept {
    std::lock_guard lock(queue_mutex_);
    if (outstanding_ != 0) { --outstanding_; }
}

template <class Instance>
void EngineCore<Instance>::release_consumer(const std::shared_ptr<Request>& request) noexcept {
    bool release = false;
    {
        std::lock_guard lock(request->mutex);
        request->consumer_released = true;
        if (request->response_done && !request->capacity_released) {
            request->capacity_released = true;
            release                    = true;
        }
    }
    if (release) { release_reserved_capacity(); }
}

template <class Instance>
void EngineCore<Instance>::abandon_request(std::shared_ptr<Request> request) noexcept {
    request->cancelled.store(true, std::memory_order_release);
    request_admission_check();
    queue_cv_.notify_one();
    release_consumer(request);
}

template <class Instance>
bool EngineCore<Instance>::mark_completed(const std::shared_ptr<Request>& request) noexcept {
    bool release = false;
    {
        std::lock_guard lock(request->mutex);
        if (request->consumer_released && !request->capacity_released) {
            request->capacity_released = true;
            release                    = true;
        }
    }
    return release;
}

template <class Instance>
void EngineCore<Instance>::release_planning_state(
    const std::shared_ptr<Request>& request) noexcept {
    if (admission_decision_ && admission_decision_->request == request) {
        admission_decision_.reset();
    }
    if (!request->admitted_at) {
        request->admission.revoked_checkpoints = resources_.source_revocations(request->id);
        finish_source_wait(request, Clock::now());
    }
    resources_.release_source(*instance_.session, request->id);
    if (request->continuation_owner) {
        resources_.abandon(*instance_.session, request->continuation_owner);
        request->continuation_owner = 0;
    }
    request->suspended.reset();
    request->base_plan.reset();
}

template <class Instance>
void EngineCore<Instance>::complete_error(const std::shared_ptr<Request>& request,
                                          std::exception_ptr error) {
    if (request->preemption_count) {
        observe_scheduling(request, GenerationSchedulingTransition::Terminal);
    }
    release_planning_state(request);
    request->prompt      = {};
    request->model_state = EngineRequestState::ModelFinished;
    request->sequence.reset();
    request->lane.reset();
    request->residency.reset();
    request->budget.reset();
    request->terminal_reason.reset();
    {
        std::lock_guard lock(request->mutex);
        if (request->response_done) { return; }
        request->error         = std::move(error);
        request->response_done = true;
    }
    if (mark_completed(request)) { release_reserved_capacity(); }
    request->cv.notify_one();
}

template <class Instance>
void EngineCore<Instance>::complete_success(const std::shared_ptr<Request>& request,
                                            FinishReason reason) {
    HostPhaseMeasurement completion = begin_host_phase();
    if (request->preemption_count) {
        observe_scheduling(request, GenerationSchedulingTransition::Terminal);
    }
    double prompt_wall_seconds     = 0.0;
    double generation_wall_seconds = 0.0;
    if (request->observation.phase_timings && request->first_token) {
        if (!request->admitted_at || !request->last_token) {
            throw std::logic_error("observed request completed without stable timing bounds");
        }
        prompt_wall_seconds =
            std::chrono::duration<double>(*request->first_token - *request->admitted_at).count();
        generation_wall_seconds =
            std::chrono::duration<double>(*request->last_token - *request->first_token).count();
    }
    release_planning_state(request);
    request->prompt      = {};
    request->model_state = EngineRequestState::ModelFinished;
    if (!request->queue_wait_recorded) {
        request->host_timing.queue_wait_ns = elapsed_ns(request->submitted, Clock::now());
        request->queue_wait_recorded       = true;
    }
    GenerationResult result;
    result.engine_request_id            = request->id;
    result.admission                    = request->admission;
    result.computed_prefill_tokens      = request->computed_prompt_tokens;
    result.scheduling.preemptions       = request->preemption_count;
    result.scheduling.replay_restores   = request->replay_restores;
    result.scheduling.snapshot_restores = request->snapshot_restores;
    result.scheduling.replayed_tokens   = request->replayed_tokens;
    if (request->paused_at) {
        request->paused_ns += elapsed_ns(*request->paused_at, Clock::now());
        request->paused_at.reset();
    }
    result.scheduling.paused_ns            = request->paused_ns;
    result.scheduling.device_to_host_bytes = request->device_to_host_bytes;
    result.scheduling.host_to_device_bytes = request->host_to_device_bytes;
    result.prompt                          = request->prompt_summary;
    result.generated_token_ids             = std::move(request->generated);
    result.content                         = std::move(request->content);
    result.reasoning                       = std::move(request->reasoning);
    result.constraint                      = request->output.constraint_observation();
    result.tool_calls                      = request->output.take_tool_calls();
    result.tool_call_parse                 = request->output.tool_call_parse_diagnostics();
    result.reasoning_tokens                = request->output.reasoning_tokens();
    result.finish_reason                   = reason;
    result.matched_stop_string             = request->output.matched_stop_string();
    if (request->begin) {
        result.reused_prompt_tokens = request->begin->reused_prompt_tokens;
        result.prefix_reuse_path    = request->begin->prefix_reuse_path;
    }
    result.timings                 = request->generation_timings;
    result.timings.prepare_seconds = request->prepare_seconds;
    result.speculative             = std::move(request->speculative_stats);
    result.thinking                = request->output.thinking_stats();
    if (request->first_token) {
        result.timings.first_token_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(*request->first_token - request->submitted).count();
    }
    result.timings.prompt_wall_seconds     = prompt_wall_seconds;
    result.timings.generation_wall_seconds = generation_wall_seconds;
    result.timings.total_seconds =
        request->prepare_seconds +
        std::chrono::duration<double>(Clock::now() - request->submitted).count();
    request->sequence.reset();
    request->lane.reset();
    request->residency.reset();
    request->budget.reset();
    request->terminal_reason.reset();
    finish_engine_phase(completion, EngineHostPhase::CommitOutput);
    result.engine_timing       = request->host_timing.public_snapshot();
    result.first_output_timing = request->first_output_timing;
    {
        std::lock_guard lock(request->mutex);
        if (request->response_done) { return; }
        request->result        = std::move(result);
        request->response_done = true;
    }
    if (mark_completed(request)) { release_reserved_capacity(); }
    request->cv.notify_one();
}

template <class Instance>
void EngineCore<Instance>::complete_cancelled(const std::shared_ptr<Request>& request) {
    (void)request->output.preview_terminal(FinishReason::Cancelled);
    append_output(request, request->output.commit_preview());
    complete_success(request, FinishReason::Cancelled);
}

template <class Instance>
void EngineCore<Instance>::complete_detached_cancelled(const std::shared_ptr<Request>& request) {
    try {
        complete_cancelled(request);
    } catch (...) {
        const std::exception_ptr error = std::current_exception();
        complete_error(request, error);
        throw;
    }
}

template <class Instance>
void EngineCore<Instance>::remove_completed_slot(std::uint32_t lane) {
    capture_decisions_[lane].reset();
    reserved_units_[lane].reset();
    slots_[lane].reset();
    request_admission_check();
    scheduler_.capacity_released();
}

template <class Instance>
[[nodiscard]] std::array<bool, kMaximumConcurrency>
EngineCore<Instance>::snapshot_cancellations() const noexcept {
    std::array<bool, kMaximumConcurrency> cancelled{};
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        if (slots_[lane] != nullptr) {
            cancelled[lane] = slots_[lane]->cancelled.load(std::memory_order_acquire);
        }
    }
    return cancelled;
}

template <class Instance>
bool EngineCore<Instance>::settle_terminal_requests(HostPhaseMeasurement& boundary) {
    if (instance_.session->has_context_operation()) { return false; }

    bool changed = false;
    for (;;) {
        std::optional<std::uint32_t> selected;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || (!request->terminal_reason && !request->terminal_error) ||
                (request->sequence && instance_.session->execution_pending(*request->sequence))) {
                continue;
            }
            if (!selected || request->publication_order < slots_[*selected]->publication_order) {
                selected = lane;
            }
        }
        if (!selected) { break; }

        const std::uint32_t lane = *selected;
        const auto request       = slots_[lane];
        if (!request->is_model_finished() || request->capture_pending || !request->sequence ||
            !request->lane || request->lane->value != lane) {
            throw std::logic_error("terminal-pending request has invalid ownership");
        }
        if (request->consumed_by_commit) {
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            if (request->terminal_error) {
                complete_error(request, std::exchange(request->terminal_error, {}));
            } else {
                complete_success(request, *request->terminal_reason);
            }
            remove_completed_slot(lane);
            boundary = begin_host_phase();
            changed  = true;
            continue;
        }
        if (request->terminal_error) {
            auto aborted = instance_.session->abort(*request->sequence);
            if (aborted.status != ConsumeStatus::Consumed) {
                throw std::logic_error("invalid output sequence could not be aborted");
            }
            request->generation_timings = aborted.timings;
            request->speculative_stats  = std::move(aborted.speculative);
            const auto error            = std::exchange(request->terminal_error, {});
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            complete_error(request, error);
            remove_completed_slot(lane);
            boundary = begin_host_phase();
            changed  = true;
            continue;
        }
        const FinishReason reason = *request->terminal_reason;
        auto finished             = instance_.session->finish(*request->sequence);
        if (finished.status != ConsumeStatus::Consumed) {
            throw std::logic_error("terminal native sequence could not finish");
        }
        if (finished.checkpoint) {
            resources_.publish(*instance_.session, request->continuation_owner,
                               *finished.checkpoint);
        }
        resources_.finish(*instance_.session, request->continuation_owner, *request->base_plan,
                          request->publication_order);
        request->continuation_owner = 0;
        request->generation_timings = finished.timings;
        request->speculative_stats  = std::move(finished.speculative);
        request->terminal_reason.reset();

        finish_engine_phase(boundary, EngineHostPhase::Boundary);
        complete_success(request, reason);
        remove_completed_slot(lane);
        boundary = begin_host_phase();
        changed  = true;
    }
    if (changed) { publish_runtime_stats(); }
    return changed;
}

template <class Instance>
void EngineCore<Instance>::cancel_active_requests(
    const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary,
    HostPhaseMeasurement& boundary) {
    if (instance_.session->has_context_operation()) { return; }
    bool changed = false;
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        const auto& request = slots_[lane];
        if (request == nullptr || !cancelled_at_boundary[lane]) { continue; }
        if (request->capture_pending ||
            (request->sequence && instance_.session->execution_pending(*request->sequence))) {
            continue;
        }
        if (!request->sequence || !request->lane || request->lane->value != lane) {
            throw std::logic_error("active cancellation has no sequence binding");
        }
        (void)request->output.preview_terminal(FinishReason::Cancelled);
        auto aborted = instance_.session->abort(*request->sequence);
        if (aborted.status != ConsumeStatus::Consumed) {
            throw std::logic_error("cancelled sequence could not be aborted");
        }
        request->generation_timings = aborted.timings;
        request->speculative_stats  = std::move(aborted.speculative);
        finish_engine_phase(boundary, EngineHostPhase::Boundary);
        append_output(request, request->output.commit_preview());
        complete_success(request, FinishReason::Cancelled);
        remove_completed_slot(lane);
        boundary = begin_host_phase();
        changed  = true;
    }
    if (changed) { publish_runtime_stats(); }
}

template <class Instance>
[[nodiscard]] bool EngineCore<Instance>::expire_pending_requests() {
    std::vector<std::shared_ptr<Request>> cancelled;
    std::vector<std::shared_ptr<Request>> expired;
    bool have_pending = false;
    {
        std::lock_guard lock(queue_mutex_);
        const auto now    = Clock::now();
        pending_deadline_ = Clock::time_point::max();
        for (auto it = pending_.begin(); it != pending_.end();) {
            if ((*it)->cancelled.load(std::memory_order_acquire)) {
                cancelled.push_back(*it);
                it = pending_.erase(it);
            } else if (now >= (*it)->deadline) {
                expired.push_back(*it);
                it = pending_.erase(it);
            } else {
                pending_deadline_ = std::min(pending_deadline_, (*it)->deadline);
                ++it;
            }
        }
        have_pending = !pending_.empty();
    }
    try {
        for (const auto& request : cancelled) { complete_detached_cancelled(request); }
        for (const auto& request : expired) {
            complete_error(request, std::make_exception_ptr(RequestError(
                                        RequestErrorKind::QueueTimeout,
                                        "inference request expired while waiting for admission")));
        }
    } catch (...) {
        const std::exception_ptr error = std::current_exception();
        for (const auto& request : cancelled) { complete_error(request, error); }
        for (const auto& request : expired) { complete_error(request, error); }
        throw;
    }
    if (!cancelled.empty() || !expired.empty()) {
        request_admission_check();
        publish_runtime_stats();
    }
    return have_pending;
}

} // namespace ninfer::runtime
