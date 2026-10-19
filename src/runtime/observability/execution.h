#pragma once

#include "ninfer/observability/execution.h"
#include "runtime/observability/snapshot.h"

#include <utility>

namespace ninfer::runtime {

enum class CoordinatorCall : std::uint8_t { Session, Poll, Flush, Yield };

// Coordinator-local timestamps. The existing execution protocol defines these boundaries;
// observing them neither advances a frame nor changes its publication policy.
class ExecutionClock {
public:
    void start(ExecutionWork work, std::uint64_t now) noexcept;
    void result(std::uint64_t now) noexcept;
    void resolve(std::uint64_t now) noexcept;
    [[nodiscard]] ExecutionSample finish(std::uint64_t now) const noexcept;
private:
    ExecutionWork work_;
    std::uint64_t started_ = 0;
    std::optional<std::uint64_t> result_, resolved_;
};

void accumulate(ExecutionStatistics&, const ExecutionSample&) noexcept;
void merge_gpu_intervals(GpuExecutionObservation&, const GpuExecutionObservation&) noexcept;
void accumulate_gpu(std::array<DurationCounter, 4>&, const GpuExecutionObservation&) noexcept;
void merge_host_observations(RankHostStatistics&, const RankHostStatistics&) noexcept;

template <class Call>
decltype(auto) observe_host_call(std::uint64_t& counter, Call&& call) {
    struct Charge {
        std::uint64_t& total;
        std::uint64_t start = observation_time_ns();

        ~Charge() { total += observation_time_ns() - start; }
    } charge{counter};

    return std::forward<Call>(call)();
}

} // namespace ninfer::runtime
