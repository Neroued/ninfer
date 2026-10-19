#pragma once

#include "core/device.h"
#include "ninfer/observability/execution.h"

namespace ninfer::runtime {

// Optional stream intervals read only after the Program's existing completion event is ready.
// They include stream idle gaps and overlap Host work; they are not kernel-active time.
class ExecutionGpuTimer {
public:
    ExecutionGpuTimer(const DeviceContext& device, bool enabled) {
        if (enabled) {
            result_.emplace(device);
            retirement_.emplace(device);
        }
    }

    void begin() {
        retirement_started_ = retirement_recorded_ = false;
        if (result_) result_->start();
    }

    void record_result() {
        if (result_) result_->record_stop();
    }

    void observe_result(GpuExecutionObservation& observed) const {
        if (result_) observed.result_stream_ns = nanoseconds(*result_);
    }

    void resolve() {
        if (retirement_ && !retirement_started_) {
            retirement_->start();
            retirement_started_ = true;
        }
    }

    void record_retirement() {
        if (retirement_started_) {
            retirement_->record_stop();
            retirement_recorded_ = true;
        }
    }

    void observe_retirement(GpuExecutionObservation& observed) const {
        if (retirement_recorded_) observed.commit_stream_ns = nanoseconds(*retirement_);
    }

private:
    static std::uint64_t nanoseconds(const CudaEventTimer& timer) {
        return static_cast<std::uint64_t>(double(timer.elapsed_ms()) * 1.0e6 + 0.5);
    }

    std::optional<CudaEventTimer> result_, retirement_;
    bool retirement_started_ = false, retirement_recorded_ = false;
};

} // namespace ninfer::runtime
