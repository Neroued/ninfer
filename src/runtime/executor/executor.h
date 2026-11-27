#pragma once

#include "runtime/executor/execution_pool.h"
#include "runtime/executor/measurement.h"
#include "runtime/observability/execution.h"
#include "ninfer/types.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <functional>
#include <stdexcept>
#include <thread>

namespace ninfer::runtime {

template <class Types>
class Executor {
public:
    explicit Executor(std::uint32_t max_decode_rows) : decode_statistics_(max_decode_rows) {
        if (!max_decode_rows)
            throw std::invalid_argument(
                "executor decode statistics require a positive row capacity");
    }

    virtual ~Executor()                                                              = default;
    virtual void bind_to_current_thread()                                            = 0;
    [[nodiscard]] virtual SessionEpoch epoch() const noexcept                        = 0;
    [[nodiscard]] virtual ExecutionCapacity capacity() const noexcept                = 0;
    [[nodiscard]] virtual std::optional<ExecutionReservation> try_reserve(FrameKind) = 0;
    [[nodiscard]] virtual ExecutionTicket submit(ExecutionReservation, typename Types::Work) = 0;

    // A true result returns ownership of a compute admission before any worker could see it.
    // The caller may roll back its prepared resources; there will be no device retirement.
    virtual bool try_withdraw(ExecutionTicket) { return false; }

    virtual void set_peer_wakeup(std::function<void()>) {}

    // Generation owners enter this boundary before releasing request resources on stop.
    virtual void finish_published_work() {}

    virtual void supply(ExecutionTicket, DecisionInput)                                 = 0;
    [[nodiscard]] virtual CommitReceipt commit(ExecutionTicket, typename Types::Commit) = 0;
    virtual void discard(ExecutionTicket)                                               = 0;
    [[nodiscard]] virtual std::optional<ExecutionEvent<Types>> poll()                   = 0;
    virtual void flush()                                                                = 0;

    virtual void wait_for_activity(std::chrono::milliseconds) const { std::this_thread::yield(); }

    // Closes submission and completes all physical readers before control owners are released.
    virtual void drain()                                                  = 0;
    virtual void fail(std::string message)                                = 0;
    [[nodiscard]] virtual bool failed() const noexcept                    = 0;
    [[nodiscard]] virtual bool has_pending_work() const noexcept          = 0;
    virtual void measure(MeasurementCommand command)                      = 0;
    virtual void observe_in_flight(std::vector<ExecutionProgress>&) const = 0;

    void fail_from_exception(std::exception_ptr error) noexcept {
        try {
            std::string reason = "executor failed";
            try {
                if (error) std::rethrow_exception(error);
            } catch (const std::exception& e) { reason = e.what(); } catch (...) {
            }
            fail(std::move(reason));
        } catch (...) {}
    }

    // Each owner calls this after its local calibration sources have retired. Peers may
    // still need service before startup observations can be reset.
    virtual void finish_startup(bool calibrated) {
        if (!calibrated) return;
        measure({MeasurementAction::ResetStartup});
        execution_statistics_ = {};
        std::ranges::fill(decode_statistics_, ExecutionCounters{});
        for (auto& rank : observations_) {
            rank.host                = {};
            rank.gpu_intervals       = {};
            rank.attention           = {};
            rank.communication_eager = {};
            rank.communication_graph = {};
            rank.compute_paths       = {};
        }
    }

    [[nodiscard]] virtual std::span<const RankSnapshot> observations() const noexcept {
        return observations_;
    }

    [[nodiscard]] virtual std::span<const RankDescriptor> rank_descriptions() const noexcept {
        return rank_descriptions_;
    }

    [[nodiscard]] virtual const ExecutionStatistics& execution_statistics() const noexcept {
        return execution_statistics_;
    }

    [[nodiscard]] virtual std::span<const ExecutionCounters> decode_statistics() const noexcept {
        return decode_statistics_;
    }
protected:
    void observe_execution(const ExecutionSample& sample) {
        accumulate(execution_statistics_, sample);
        if (sample.work.kind == ExecutionKind::Decode)
            decode_statistics_.at(sample.work.rows - 1).observe(sample);
    }

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
        if (value.attention)
            snapshot.attention.at(static_cast<std::size_t>(value.attention->kind))
                .add(value.attention->work);
        snapshot.communication_eager = value.communication_eager;
        snapshot.communication_graph = value.communication_graph;
        if (compute) ++snapshot.compute_paths[static_cast<std::size_t>(value.path)];
    }

    std::vector<RankSnapshot> observations_;
    std::vector<RankDescriptor> rank_descriptions_;
    ExecutionStatistics execution_statistics_{};
    std::vector<ExecutionCounters> decode_statistics_;
};

} // namespace ninfer::runtime
