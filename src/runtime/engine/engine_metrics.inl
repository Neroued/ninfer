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
typename EngineCore<Instance>::ActiveExposureSet EngineCore<Instance>::active_exposure_set() const {
    ActiveExposureSet result;
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        if (slots_[lane] == nullptr) { continue; }
        result.entries[result.size++] = ActiveExposure{.request = slots_[lane], .lane = lane};
    }
    return result;
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
void EngineCore<Instance>::set_host_work_class(
    HostWorkClass work_class, std::span<const std::uint32_t> decode_lanes) noexcept {
    current_host_work_class_   = work_class;
    current_decode_lane_count_ = decode_lanes.size();
    for (std::size_t i = 0; i < decode_lanes.size(); ++i) {
        current_decode_lanes_[i] = decode_lanes[i];
    }
}

template <class Instance>
bool EngineCore<Instance>::current_decode_contains(std::uint32_t lane) const noexcept {
    return std::find(current_decode_lanes_.begin(),
                     current_decode_lanes_.begin() +
                         static_cast<std::ptrdiff_t>(current_decode_lane_count_),
                     lane) !=
           current_decode_lanes_.begin() + static_cast<std::ptrdiff_t>(current_decode_lane_count_);
}

template <class Instance>
void EngineCore<Instance>::add_class_host_time(std::uint64_t host_ns,
                                               std::uint64_t scheduler_yield_ns) noexcept {
    RuntimeHostWorkStats& stats = cumulative_stats_.host_work;
    switch (current_host_work_class_) {
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
    for (std::size_t i = 0; i < exposed.size; ++i) {
        const ActiveExposure& exposure = exposed.entries[i];
        RequestHostTiming& timing      = exposure.request->host_timing;
        timing.expose_engine(phase, elapsed,
                             current_host_work_class_ == HostWorkClass::Decode &&
                                 current_decode_contains(exposure.lane));
    }
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
    add_class_host_time(own, 0);
    expose_engine_phase(measurement.exposed, phase, own);
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
    add_class_host_time(yielding ? 0 : elapsed, yielding ? elapsed : 0);
    for (std::size_t i = 0; i < exposed.size; ++i) {
        const auto& exposure = exposed.entries[i];
        exposure.request->host_timing.expose_call(call, elapsed,
                                                  current_host_work_class_ ==
                                                          HostWorkClass::Decode &&
                                                      current_decode_contains(exposure.lane));
    }
    worker_accounted_elapsed_ns_ += elapsed;
}

template <class Instance>
void EngineCore<Instance>::finish_execution_call(const HostPhaseMeasurement& measurement,
                                                 CoordinatorCall call) noexcept {
    const auto wall   = elapsed_ns(measurement.started, Clock::now());
    const auto nested = worker_accounted_elapsed_ns_ - measurement.accounted_before;
    record_coordinator_call(call, wall > nested ? wall - nested : 0, measurement.exposed);
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
void EngineCore<Instance>::publish_runtime_stats() {
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
    snapshot.device_state_occupied_slots = physical.occupied.state_slots;
    for (auto& pool : snapshot.kv_pools) {
        pool.occupied_page_groups = physical.occupied.kv_page_groups[pool.id];
        pool.capacity_page_groups = physical.capacity.kv_page_groups[pool.id];
    }

    {
        std::lock_guard lock(queue_mutex_);
        snapshot.waiting_requests = static_cast<std::uint32_t>(pending_.size());
    }
    snapshot.prefilling_requests              = 0;
    snapshot.paused_requests                  = static_cast<std::uint32_t>(paused_.size());
    snapshot.replaying_requests               = 0;
    snapshot.host_context_occupied_bytes      = physical.occupied.host_bytes;
    snapshot.host_state_occupied_slots        = physical.host_state_slots;
    snapshot.host_kv_occupied_bytes           = physical.host_kv_bytes;
    snapshot.host_context_reserved_bytes      = physical.host_reserved_bytes;
    snapshot.host_context_peak_occupied_bytes = physical.host_peak_occupied_bytes;
    snapshot.materializing_requests           = materializing_.has_value() ? 1U : 0U;
    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
        if (slots_[lane] == nullptr) { continue; }
        ++snapshot.running_requests;
        if (slots_[lane]->is_prefilling()) { ++snapshot.prefilling_requests; }
        if (slots_[lane]->is_replaying()) { ++snapshot.replaying_requests; }
        if (slots_[lane]->is_decode_ready()) { ++snapshot.decode_ready_requests; }
        if (slots_[lane]->capture_pending) { ++snapshot.capture_pending_requests; }
        if (slots_[lane]->terminal_reason) { ++snapshot.terminal_pending_requests; }
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
    const auto ranks               = instance_.executor->observations();
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
        auto& total           = series.at(direction);
        const auto accumulate = [&](GenerationTransferTiming& timing) {
            timing.bytes += transfer.work.payload_bytes;
            timing.seconds += transfer.elapsed_ns * 1.0e-9;
            timing.images += transfer.image_count;
            timing.page_groups += transfer.page_groups;
            timing.copy_operations += transfer.work.copy_operations;
        };
        accumulate(total);
        cumulative_stats_.actual_context_transfer_seconds += transfer.elapsed_ns * 1.0e-9;
        if (owner) {
            if (!owner->first_output_timing) {
                auto& observed =
                    owner->context_transfers.at(transfer.resource.index()).at(direction);
                accumulate(observed);
            }
            if (transfer.direction == ContextTransferDirection::DeviceToHost) {
                owner->device_to_host_bytes += transfer.work.payload_bytes;
            }
            if (transfer.direction == ContextTransferDirection::HostToDevice) {
                owner->host_to_device_bytes += transfer.work.payload_bytes;
            }
        }
    }
}

template <class Instance>
void EngineCore<Instance>::record_first_output(const std::shared_ptr<Request>& request) {
    const auto now                   = Clock::now();
    auto& snapshot                   = request->first_output_timing.emplace();
    snapshot.elapsed_seconds         = elapsed_ns(request->submitted, now) * 1.0e-9;
    snapshot.initial_binding_seconds = request->initial_binding_ns * 1.0e-9;
    snapshot.engine                  = request->host_timing.public_snapshot();
    snapshot.prefill = request->executions[static_cast<std::size_t>(ExecutionKind::Prefill)];
    snapshot.replay  = request->executions[static_cast<std::size_t>(ExecutionKind::Replay)];
    snapshot.pending_execution = request->pending_execution;
    snapshot.scheduling        = {
               .preemptions       = request->preemption_count,
               .snapshot_restores = request->snapshot_restores,
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
