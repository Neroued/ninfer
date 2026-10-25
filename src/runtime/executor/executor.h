#pragma once

#include "runtime/executor/execution_pool.h"
#include "runtime/executor/measurement.h"
#include "runtime/observability/execution.h"

namespace ninfer::runtime {

template <class Types>
class Executor {
public:
    virtual ~Executor()                                                              = default;
    virtual void bind_to_current_thread()                                            = 0;
    [[nodiscard]] virtual SessionEpoch epoch() const noexcept                        = 0;
    [[nodiscard]] virtual ExecutionCapacity capacity() const noexcept                = 0;
    [[nodiscard]] virtual std::optional<ExecutionReservation> try_reserve(FrameKind) = 0;
    [[nodiscard]] virtual ExecutionTicket submit(ExecutionReservation, typename Types::Work) = 0;
    virtual void supply(ExecutionTicket, DecisionInput)                                      = 0;
    [[nodiscard]] virtual CommitReceipt commit(ExecutionTicket, typename Types::Commit)      = 0;
    virtual void discard(ExecutionTicket)                                                    = 0;
    [[nodiscard]] virtual std::optional<ExecutionEvent<Types>> poll()                        = 0;
    virtual void flush()                                                                     = 0;
    // Closes submission and completes all physical readers before control owners are released.
    virtual void drain()                                         = 0;
    virtual void fail(std::string message)                       = 0;
    [[nodiscard]] virtual bool failed() const noexcept           = 0;
    [[nodiscard]] virtual bool has_pending_work() const noexcept = 0;
    virtual void measure(MeasurementCommand command)             = 0;

    [[nodiscard]] std::span<const RankSnapshot> observations() const noexcept {
        return observations_;
    }

    [[nodiscard]] const ExecutionStatistics& execution_statistics() const noexcept {
        return execution_statistics_;
    }
protected:
    void observe_retirement(std::uint32_t rank, const RankExecutionObservation& value,
                            bool compute) {
        auto& snapshot = observations_.at(rank);
        merge_host_observations(snapshot.host, value.host);
        snapshot.memory.workspace.required_peak_bytes =
            std::max(snapshot.memory.workspace.required_peak_bytes, value.workspace_peak_bytes);
        snapshot.memory.workspace.measurement_peak_bytes = std::max(
            snapshot.memory.workspace.measurement_peak_bytes, value.measurement_peak_bytes);
        if (snapshot.memory.experts)
            snapshot.memory.experts->uploaded_bytes =
                std::max(snapshot.memory.experts->uploaded_bytes, value.expert_upload_bytes);
        if (snapshot.memory.ple)
            snapshot.memory.ple->uploaded_bytes =
                std::max(snapshot.memory.ple->uploaded_bytes, value.ple_upload_bytes);
        accumulate_gpu(snapshot.gpu_intervals, value.gpu);
        snapshot.communication_eager = value.communication_eager;
        snapshot.communication_graph = value.communication_graph;
        if (compute) ++snapshot.compute_paths[static_cast<std::size_t>(value.path)];
    }

    std::vector<RankSnapshot> observations_;
    ExecutionStatistics execution_statistics_{};
};

} // namespace ninfer::runtime
