#include "runtime/executor/measurement.h"

#include "core/device.h"

#include <cuda_profiler_api.h>

#include <stdexcept>

namespace ninfer::runtime {

RankMeasurement::~RankMeasurement() { cancel(); }

void RankMeasurement::validate(MeasurementCommand command) const {
    if (command.profile && command.action != MeasurementAction::Begin)
        throw std::invalid_argument("profiling can only be enabled when beginning a measurement");
    switch (command.action) {
    case MeasurementAction::Reset:
    case MeasurementAction::Begin:
        if (active_) throw std::logic_error("a measurement window is already active");
        break;
    case MeasurementAction::End:
        if (!active_) throw std::logic_error("no measurement window is active");
        break;
    case MeasurementAction::Cancel:
        break;
    default:
        throw std::invalid_argument("invalid measurement action");
    }
}

void RankMeasurement::begin(bool profile) {
    active_    = true;
    profiling_ = profile;
    if (profiling_) CUDA_CHECK(cudaProfilerStart());
}

void RankMeasurement::end() {
    if (profiling_) CUDA_CHECK(cudaProfilerStop());
    profiling_ = false;
    active_    = false;
}

void RankMeasurement::cancel() noexcept {
    if (profiling_) (void)cudaProfilerStop();
    profiling_ = false;
    active_    = false;
}

} // namespace ninfer::runtime
