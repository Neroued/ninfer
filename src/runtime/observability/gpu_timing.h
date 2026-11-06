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
            draft_.emplace(device);
            target_.emplace(device);
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

    void begin_draft(bool capture = false) {
        if (draft_) capture ? draft_->start_external() : draft_->start();
    }

    void end_draft(bool capture = false) {
        if (draft_) capture ? draft_->record_stop_external() : draft_->record_stop();
    }

    void begin_target(bool capture = false) {
        if (target_) capture ? target_->start_external() : target_->start();
    }

    void end_target(bool capture = false) {
        if (target_) capture ? target_->record_stop_external() : target_->record_stop();
    }

    // Call only for a decode round whose target has completed. Graph replay records the same
    // events as eager execution; capture-time Host flags cannot describe which events ran.
    void observe_decode(GpuExecutionObservation& observed, bool drafted) const {
        if (target_) observed.target_stream_ns = nanoseconds(*target_);
        if (draft_ && drafted) observed.draft_stream_ns = nanoseconds(*draft_);
    }

private:
    static std::uint64_t nanoseconds(const CudaEventTimer& timer) {
        return static_cast<std::uint64_t>(double(timer.elapsed_ms()) * 1.0e6 + 0.5);
    }

    std::optional<CudaEventTimer> result_, retirement_;
    std::optional<CudaEventTimer> draft_, target_;
    bool retirement_started_ = false, retirement_recorded_ = false;
};

} // namespace ninfer::runtime
