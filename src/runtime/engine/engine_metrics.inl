#pragma once

namespace ninfer::runtime {

template <class Instance>
nvtx::Name EngineCore<Instance>::phase_range_name(EngineHostPhase phase) noexcept {
    switch (phase) {
    case EngineHostPhase::Boundary:
        return nvtx::Name::EngineBoundary;
    case EngineHostPhase::CommitOutput:
        return nvtx::Name::EngineCommitOutput;
    case EngineHostPhase::Maintenance:
        return nvtx::Name::EngineMaintenance;
    }
    return nvtx::Name::EngineBoundary;
}

template <class Instance>
std::uint64_t EngineCore<Instance>::elapsed_ns(Clock::time_point started,
                                               Clock::time_point finished) noexcept {
    const auto count =
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count();
    return count > 0 ? static_cast<std::uint64_t>(count) : 0;
}

template <class Instance>
typename EngineCore<Instance>::ExposureSnapshot EngineCore<Instance>::active_exposure_set() const {
    if (completion_exposure_) return completion_exposure_->exposed;
    if (active_exposure_) return active_exposure_;
    auto result        = std::make_shared<ActiveExposureSet>();
    result->work_class = current_host_work_class_;
    result->entries.reserve(max_concurrency_);
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        const auto& request = slots_[lane];
        if (!request) continue;
        result->entries.push_back({request, lane,
                                   current_host_work_class_ == HostWorkClass::Decode &&
                                       current_decode_requests_[lane] == request->id});
    }
    active_exposure_ = std::move(result);
    return active_exposure_;
}

template <class Instance>
typename EngineCore<Instance>::HostPhaseMeasurement EngineCore<Instance>::begin_host_phase() const {
    return HostPhaseMeasurement{
        .started          = Clock::now(),
        .accounted_before = worker_accounted_elapsed_ns_,
        .exposed          = active_exposure_set(),
    };
}

template <class Instance>
HostExposureClock EngineCore<Instance>::exposure_clock() const noexcept {
    const auto& t = cumulative_stats_.host_work;
    return {{t.engine_boundary_ns, t.engine_commit_output_ns, t.engine_maintenance_ns},
            {t.session_call_ns, t.executor_poll_ns, t.executor_flush_ns, t.scheduler_yield_ns}};
}

template <class Instance>
RequestHostTiming EngineCore<Instance>::request_host_timing(const Request& request) const noexcept {
    auto timing = request.host_timing;
    if (completion_exposure_ && request.completion_exposure_row) {
        const auto& row = completion_exposure_->exposed->entries[*request.completion_exposure_row];
        timing.expose_since(completion_exposure_->first, exposure_clock(), row.decode);
    }
    return timing;
}

template <class Instance>
void EngineCore<Instance>::finish_completion_exposure() noexcept {
    if (!completion_exposure_) return;
    const auto last = exposure_clock();
    for (const auto& row : completion_exposure_->exposed->entries) {
        const auto* completed = row.request->completion_index
                                    ? &pending_completions_[*row.request->completion_index]
                                    : nullptr;
        row.request->host_timing.expose_since(
            completion_exposure_->first,
            completed && completed->ready ? completed->exposure_end : last, row.decode);
        row.request->completion_exposure_row.reset();
    }
    completion_exposure_.reset();
    for (auto& completed : pending_completions_)
        if (completed.ready && !completed.error)
            completed.result.stats.engine_timing = completed.request->host_timing.public_snapshot();
}

template <class Instance>
void EngineCore<Instance>::set_resident(std::uint32_t lane,
                                        std::shared_ptr<Request> request) noexcept {
    slots_[lane] = std::move(request);
    active_exposure_.reset();
}

template <class Instance>
void EngineCore<Instance>::set_host_work_class(HostWorkClass work_class,
                                               std::span<const std::uint32_t> decode_lanes) {
    if (work_class != HostWorkClass::Decode && !decode_lanes.empty())
        throw std::logic_error("non-decode host work has decode members");
    bool unchanged = current_host_work_class_ == work_class &&
                     std::equal(current_decode_lanes_.begin(), current_decode_lanes_.end(),
                                decode_lanes.begin(), decode_lanes.end());
    for (auto lane : decode_lanes) {
        if (lane >= slots_.size() || !slots_[lane])
            throw std::logic_error("decode host work has no resident member");
        unchanged &= current_decode_requests_[lane] == slots_[lane]->id;
    }
    if (unchanged) return;
    for (auto lane : current_decode_lanes_) current_decode_requests_[lane] = 0;
    current_decode_lanes_.assign(decode_lanes.begin(), decode_lanes.end());
    for (auto lane : current_decode_lanes_) current_decode_requests_[lane] = slots_[lane]->id;
    current_host_work_class_ = work_class;
    active_exposure_.reset();
}

template <class Instance>
void EngineCore<Instance>::restore_host_work_class(const ActiveExposureSet& previous) noexcept {
    for (auto lane : current_decode_lanes_) current_decode_requests_[lane] = 0;
    current_decode_lanes_.clear();
    for (const auto& entry : previous.entries) {
        if (!entry.decode) continue;
        current_decode_lanes_.push_back(entry.lane);
        current_decode_requests_[entry.lane] = entry.request->id;
    }
    current_host_work_class_ = previous.work_class;
    active_exposure_.reset();
}

template <class Instance>
void EngineCore<Instance>::add_class_host_time(HostWorkClass work_class, std::uint64_t host_ns,
                                               std::uint64_t scheduler_yield_ns) noexcept {
    RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
    switch (work_class) {
    case HostWorkClass::Decode:
        stats.decode_host_ns += host_ns;
        stats.decode_scheduler_yield_ns += scheduler_yield_ns;
        break;
    case HostWorkClass::Prefill:
        stats.prefill_host_ns += host_ns;
        stats.prefill_scheduler_yield_ns += scheduler_yield_ns;
        break;
    case HostWorkClass::Control:
        stats.control_host_ns += host_ns;
        stats.control_scheduler_yield_ns += scheduler_yield_ns;
        break;
    }
}

template <class Instance>
void EngineCore<Instance>::expose_engine_phase(const ActiveExposureSet& exposed,
                                               EngineHostPhase phase,
                                               std::uint64_t elapsed) noexcept {
    if (completion_exposure_ && completion_exposure_->exposed.get() == &exposed) return;
    for (const auto& exposure : exposed.entries)
        exposure.request->host_timing.expose_engine(phase, elapsed, exposure.decode);
}

template <class Instance>
void EngineCore<Instance>::finish_engine_phase(const HostPhaseMeasurement& measurement,
                                               EngineHostPhase phase) noexcept {
    const std::uint64_t wall    = elapsed_ns(measurement.started, Clock::now());
    const std::uint64_t nested  = worker_accounted_elapsed_ns_ - measurement.accounted_before;
    const std::uint64_t own     = wall > nested ? wall - nested : 0;
    RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
    switch (phase) {
    case EngineHostPhase::Boundary:
        stats.engine_boundary_ns += own;
        break;
    case EngineHostPhase::CommitOutput:
        stats.engine_commit_output_ns += own;
        break;
    case EngineHostPhase::Maintenance:
        stats.engine_maintenance_ns += own;
        break;
    }
    add_class_host_time(measurement.exposed->work_class, own, 0);
    expose_engine_phase(*measurement.exposed, phase, own);
    worker_accounted_elapsed_ns_ += own;
}

template <class Instance>
void EngineCore<Instance>::record_coordinator_call(CoordinatorCall call, std::uint64_t elapsed,
                                                   const ActiveExposureSet& exposed) noexcept {
    auto& stats = cumulative_stats_.host_work;
    switch (call) {
    case CoordinatorCall::Session:
        stats.session_call_ns += elapsed;
        break;
    case CoordinatorCall::Poll:
        stats.executor_poll_ns += elapsed;
        break;
    case CoordinatorCall::Flush:
        stats.executor_flush_ns += elapsed;
        break;
    case CoordinatorCall::Yield:
        stats.scheduler_yield_ns += elapsed;
        break;
    }
    const bool yielding = call == CoordinatorCall::Yield;
    add_class_host_time(exposed.work_class, yielding ? 0 : elapsed, yielding ? elapsed : 0);
    if (!completion_exposure_ || completion_exposure_->exposed.get() != &exposed)
        for (const auto& exposure : exposed.entries)
            exposure.request->host_timing.expose_call(call, elapsed, exposure.decode);
    worker_accounted_elapsed_ns_ += elapsed;
}

template <class Instance>
void EngineCore<Instance>::finish_execution_call(const HostPhaseMeasurement& measurement,
                                                 CoordinatorCall call) noexcept {
    const auto wall   = elapsed_ns(measurement.started, Clock::now());
    const auto nested = worker_accounted_elapsed_ns_ - measurement.accounted_before;
    record_coordinator_call(call, wall > nested ? wall - nested : 0, *measurement.exposed);
}

template <class Instance>
void EngineCore<Instance>::record_detail(std::uint64_t RuntimeHostWorkStats::*elapsed_member,
                                         std::uint64_t RuntimeHostWorkStats::*invocation_member,
                                         Clock::time_point started) noexcept {
    RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
    stats.*elapsed_member += elapsed_ns(started, Clock::now());
    ++(stats.*invocation_member);
}

template <class Instance>
RequestSnapshot EngineCore<Instance>::observe_request(const Request& request) const {
    RequestSnapshot out;
    out.id = request.id;
    switch (request.model_state) {
    case EngineRequestState::Waiting:
        out.phase = RequestPhase::Waiting;
        break;
    case EngineRequestState::Materializing:
        out.phase = RequestPhase::Materializing;
        break;
    case EngineRequestState::Prefill:
        out.phase = RequestPhase::Prefill;
        break;
    case EngineRequestState::DecodeReady:
        out.phase = RequestPhase::Decode;
        break;
    case EngineRequestState::ControlReady:
        out.phase = RequestPhase::Control;
        break;
    case EngineRequestState::Replay:
        out.phase = RequestPhase::Replay;
        break;
    case EngineRequestState::Pausing:
        out.phase = RequestPhase::Pausing;
        break;
    case EngineRequestState::Paused:
        out.phase = RequestPhase::Paused;
        break;
    case EngineRequestState::ModelFinished:
        out.phase = RequestPhase::Finishing;
        break;
    }
    if (request.terminal_reason || request.terminal_error) out.phase = RequestPhase::Finishing;
    if (request.placement) out.placement = request.placement->value;
    out.submitted_ns = elapsed_ns({}, request.submitted);
    if (request.admitted_at) out.admitted_ns = elapsed_ns({}, *request.admitted_at);
    if (request.first_token) out.first_token_ns = elapsed_ns({}, *request.first_token);
    out.prompt_tokens = request.prompt_summary.prompt_tokens;
    if (request.admitted_begin)
        out.reused_prompt_tokens = request.admitted_begin->reused_prompt_tokens;
    out.processed_prompt_tokens = out.reused_prompt_tokens + request.computed_prompt_tokens;
    out.generated_tokens        = static_cast<std::uint32_t>(request.generated.size());
    out.injected_control_tokens = request.injected_control_tokens;
    out.logical_context_tokens  = out.processed_prompt_tokens + out.generated_tokens;
    if (request.budget) out.output_token_limit = out.generated_tokens + request.budget->remaining();
    out.capture_pending  = request.capture_pending;
    out.resident         = request.lane && slots_[request.lane->value].get() == &request;
    out.preemptions      = request.preemption_count;
    out.replayed_tokens  = request.replayed_tokens;
    out.recovery         = request.recovery;
    out.recovery_pending = request.recovery_pending;
    if (request.last_progress) out.last_progress_ns = elapsed_ns({}, *request.last_progress);
    out.execution_in_flight = request.execution_in_flight;
    return out;
}

template <class Instance>
void EngineCore<Instance>::record_completion(const std::shared_ptr<Request>& request,
                                             FinishReason reason, bool failed) {
    if (request->completion_observed) return;
    request->completion_observed = true;
    auto& counters               = cumulative_stats_.requests;
    if (failed)
        ++counters.failed;
    else if (reason == FinishReason::Cancelled)
        ++counters.cancelled;
    else
        ++counters.completed;
}

template <class Instance>
void EngineCore<Instance>::publish_runtime_stats() {
    const auto published_revision    = schedule_revision_.load(std::memory_order_acquire);
    HostPhaseMeasurement measurement = begin_host_phase();
    std::optional<nvtx::ScopedRange> phase_range;
    phase_range.emplace(nvtx::Name::EngineMaintenance, nvtx::Category::Runtime);
    const Clock::time_point detail_started = Clock::now();
    std::optional<nvtx::ScopedRange> detail_range;
    detail_range.emplace(nvtx::Name::StatsPublication, nvtx::Category::Control);
    auto& snapshot             = stats_snapshot_;
    snapshot                   = cumulative_stats_;
    const auto& executions     = instance_.executor->execution_statistics();
    snapshot.decode_rounds     = executions[static_cast<std::size_t>(ExecutionKind::Decode)].units;
    snapshot.decode_row_rounds = executions[static_cast<std::size_t>(ExecutionKind::Decode)].rows;
    const auto physical        = instance_.session->physical_usage();
    snapshot.context_pools     = physical.pools;
    snapshot.context_resources = physical.resources;
    snapshot.prefix_cache      = resources_.statistics();
    snapshot.context_metadata  = physical.metadata;
    snapshot.state_objects     = physical.state_objects;
    snapshot.history_objects   = physical.history_objects;
    for (auto& pool : snapshot.kv_pools) {
        pool.occupied_page_groups = physical.kv_page_groups.at(pool.id);
    }

    request_observations_.clear();
    execution_observations_.clear();
    instance_.executor->observe_in_flight(execution_observations_);
    EngineStatus status;
    {
        std::lock_guard lock(queue_mutex_);
        snapshot.waiting_requests   = static_cast<std::uint32_t>(pending_.size());
        snapshot.requests.submitted = submitted_requests_;
        status                      = stopping_ ? EngineStatus::Stopping
                                      : failed_ ? EngineStatus::Failed
                                                : EngineStatus::Ready;
        for (const auto& request : pending_)
            request_observations_.push_back(observe_request(*request));
    }
    snapshot.prefilling_requests    = 0;
    snapshot.paused_requests        = static_cast<std::uint32_t>(paused_.size());
    snapshot.replaying_requests     = 0;
    snapshot.materializing_requests = materializing_.has_value() ? 1U : 0U;
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        if (slots_[lane] == nullptr) { continue; }
        request_observations_.push_back(observe_request(*slots_[lane]));
        ++snapshot.running_requests;
        if (slots_[lane]->is_prefilling()) { ++snapshot.prefilling_requests; }
        if (slots_[lane]->is_replaying()) { ++snapshot.replaying_requests; }
        if (slots_[lane]->is_decode_ready()) { ++snapshot.decode_ready_requests; }
        if (slots_[lane]->capture_pending) { ++snapshot.capture_pending_requests; }
        if (slots_[lane]->terminal_reason) { ++snapshot.terminal_pending_requests; }
    }
    for (const auto& request : paused_) request_observations_.push_back(observe_request(*request));
    if (materializing_) request_observations_.push_back(observe_request(*materializing_->request));
    std::sort(request_observations_.begin(), request_observations_.end(),
              [](const auto& a, const auto& b) { return a.id < b.id; });
    request_observations_.erase(
        std::unique(request_observations_.begin(), request_observations_.end(),
                    [](const auto& a, const auto& b) { return a.id == b.id; }),
        request_observations_.end());
    for (auto& placement : snapshot.placements) {
        const auto id = placement.id;
        placement     = {.id = id};
    }
    for (const auto& request : request_observations_) {
        if (!request.placement) continue;
        auto found = std::find_if(snapshot.placements.begin(), snapshot.placements.end(),
                                  [&](const auto& item) { return item.id == *request.placement; });
        if (found == snapshot.placements.end()) continue;
        found->resident += request.resident;
        found->prefilling += request.phase == RequestPhase::Prefill;
        found->decoding += request.phase == RequestPhase::Decode;
        found->replaying += request.phase == RequestPhase::Replay;
        found->paused += request.phase == RequestPhase::Paused;
        found->context_tokens += request.logical_context_tokens;
    }
    detail_range.reset();
    record_detail(&RuntimeHostWorkStats::stats_publication_ns,
                  &RuntimeHostWorkStats::stats_publication_invocations, detail_started);
    phase_range.reset();
    finish_engine_phase(measurement, EngineHostPhase::Maintenance);
    snapshot.host_work = cumulative_stats_.host_work;
    snapshot.host_work.prefill_units =
        executions[static_cast<std::size_t>(ExecutionKind::Prefill)].units;
    snapshot.host_work.control_units =
        executions[static_cast<std::size_t>(ExecutionKind::Control)].units;
    std::lock_guard lock(stats_mutex_);
    published_snapshot_.stamp   = {published_snapshot_.stamp.sequence + 1, observation_time_ns()};
    published_snapshot_.runtime = snapshot;
    published_snapshot_.executions = instance_.executor->execution_statistics();
    const auto batches = instance_.executor->decode_statistics().first(max_concurrency_);
    published_snapshot_.decode_by_batch.assign(batches.begin(), batches.end());
    published_snapshot_.in_flight      = execution_observations_;
    published_snapshot_.status         = status;
    published_snapshot_.progress_owner = scheduler_.progress_owner();
    published_requests_                = request_observations_;
    published_revision_                = published_revision;
    const auto ranks                   = instance_.executor->observations();
    published_snapshot_.ranks.assign(ranks.begin(), ranks.end());
}

template <class Instance>
void EngineCore<Instance>::record_context_work(
    const typename ModelContract::ContextProgress& progress,
    const std::shared_ptr<Request>& owner) {
    const auto& op = progress.operations;
    cumulative_stats_.state_moves += op.state_moves;
    cumulative_stats_.state_forks += op.state_forks;
    if (progress.kind == decltype(progress.kind)::Bind) {
        cumulative_stats_.materialization_state_forks += op.state_forks;
    }
    cumulative_stats_.state_restores += op.state_restores;
    cumulative_stats_.pressure_spill_pages += op.pressure_spill_pages;
    cumulative_stats_.partial_tail_cow_pages += op.partial_tail_cow_pages;
    for (const auto& transfer : progress.transfers) {
        const auto direction = static_cast<std::size_t>(transfer.direction);
        auto& series =
            transfer.resource.is_state()
                ? cumulative_stats_.context_transfers.state
                : cumulative_stats_.context_transfers.kv.at(transfer.resource.pool->index);
        accumulate_context_transfer(series.at(direction), transfer);
        if (owner) {
            if (!owner->first_output_timing) {
                auto& observed =
                    owner->context_transfers.at(transfer.resource.index()).at(direction);
                accumulate_context_transfer(observed, transfer);
            }
            if (transfer.direction == ContextTransferDirection::DeviceToHost) {
                owner->device_to_host_bytes += transfer.work.payload_bytes;
            }
            if (transfer.direction == ContextTransferDirection::HostToDevice) {
                owner->host_to_device_bytes += transfer.work.payload_bytes;
            }
        }
    }
    std::uint64_t longest_rank_ns = 0;
    for (std::size_t first = 0; first < progress.transfers.size(); ++first) {
        const auto rank = progress.transfers[first].rank;
        if (std::any_of(progress.transfers.begin(), progress.transfers.begin() + first,
                        [=](const auto& transfer) { return transfer.rank == rank; }))
            continue;
        std::uint64_t rank_ns = 0;
        for (const auto& transfer : progress.transfers) {
            if (transfer.rank != rank) continue;
            rank_ns = transfer.elapsed_ns > UINT64_MAX - rank_ns ? UINT64_MAX
                                                                 : rank_ns + transfer.elapsed_ns;
        }
        longest_rank_ns = std::max(longest_rank_ns, rank_ns);
    }
    if (owner && materializing_ && materializing_->resumed &&
        progress.kind == decltype(progress.kind)::Bind)
        scheduler_.recovery_work(owner->id, longest_rank_ns);

    cumulative_stats_.actual_context_transfer_seconds += longest_rank_ns * 1.0e-9;
}

template <class Instance>
void EngineCore<Instance>::record_first_output(const std::shared_ptr<Request>& request) {
    const auto now                   = Clock::now();
    auto& snapshot                   = request->first_output_timing.emplace();
    snapshot.elapsed_seconds         = elapsed_ns(request->submitted, now) * 1.0e-9;
    snapshot.initial_binding_seconds = request->initial_binding_ns * 1.0e-9;
    snapshot.engine                  = request_host_timing(*request).public_snapshot();
    snapshot.prefill = request->executions[static_cast<std::size_t>(ExecutionKind::Prefill)];
    snapshot.replay  = request->executions[static_cast<std::size_t>(ExecutionKind::Replay)];
    snapshot.pending_execution = request->pending_execution;
    snapshot.scheduling        = {
               .preemptions       = request->preemption_count,
               .snapshot_restores = request->snapshot_restores,
               .direct_restores   = request->direct_restores,
               .replay_restores   = request->replay_restores,
               .replayed_tokens   = request->replayed_tokens,
               .paused_ns =
            request->paused_ns + (request->paused_at ? elapsed_ns(*request->paused_at, now) : 0),
               .device_to_host_bytes = request->device_to_host_bytes,
               .host_to_device_bytes = request->host_to_device_bytes,
    };
    snapshot.computed_prefill_tokens = request->computed_prompt_tokens;
    snapshot.context_transfers.state = request->context_transfers[0];
    snapshot.context_transfers.kv.assign(request->context_transfers.begin() + 1,
                                         request->context_transfers.begin() + 1 +
                                             instance_.session->kv_pools().size());
}

} // namespace ninfer::runtime
