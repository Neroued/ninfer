#include "runtime/observability/execution.h"
#include "runtime/contract/resources.h"

namespace ninfer::runtime {

void accumulate_context_transfer(GenerationTransferTiming& timing,
                                 const ContextTransferObservation& transfer) noexcept {
    timing.seconds += transfer.elapsed_ns * 1.0e-9;
    if (transfer.peer && transfer.rank != transfer.peer->source_rank) return;
    timing.bytes += transfer.work.payload_bytes;
    timing.images += transfer.image_count;
    timing.page_groups += transfer.page_groups;
    timing.copy_operations += transfer.work.copy_operations;
}

void merge_host_observations(RankHostStatistics& into, const RankHostStatistics& from) noexcept {
    for (auto field : {&RankHostStatistics::submit_ns, &RankHostStatistics::progress_ns,
                       &RankHostStatistics::supply_ns, &RankHostStatistics::commit_ns,
                       &RankHostStatistics::discard_ns})
        if (into.*field < from.*field) into.*field = from.*field;
}

void merge_gpu_intervals(GpuExecutionObservation& into,
                         const GpuExecutionObservation& from) noexcept {
    for (auto field : kGpuIntervalFields)
        if (from.*field && (!(into.*field) || *(from.*field) > *(into.*field)))
            into.*field = from.*field;
}

void accumulate_gpu(GpuIntervalCounters& totals, const GpuExecutionObservation& value) noexcept {
    for (std::size_t i = 0; i < kGpuIntervalCount; ++i)
        if (value.*kGpuIntervalFields[i]) totals[i].observe(*(value.*kGpuIntervalFields[i]));
}

void ExecutionClock::start(ExecutionWork work, std::uint64_t now) noexcept {
    work_    = work;
    started_ = now;
    result_.reset();
    resolved_.reset();
}

void ExecutionClock::result(std::uint64_t now) noexcept { result_ = now; }

void ExecutionClock::resolve(std::uint64_t now) noexcept { resolved_ = now; }

ExecutionSample ExecutionClock::finish(std::uint64_t now) const noexcept {
    ExecutionLatency latency;
    if (result_) latency.result_ns = *result_ - started_;
    if (resolved_) {
        if (result_ && *resolved_ >= *result_) latency.resolution_ns = *resolved_ - *result_;
        latency.retirement_ns = now - *resolved_;
    }
    latency.total_ns = now - started_;
    return {work_, latency};
}

void accumulate(ExecutionStatistics& statistics, const ExecutionSample& sample) noexcept {
    statistics[static_cast<std::size_t>(sample.work.kind)].observe(sample);
}

} // namespace ninfer::runtime
