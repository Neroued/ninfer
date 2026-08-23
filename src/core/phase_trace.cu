#include "core/phase_trace.h"

#include "core/device.h"

namespace ninfer {

PhaseTrace& PhaseTrace::instance() noexcept {
    static PhaseTrace trace;
    return trace;
}

PhaseTrace::~PhaseTrace() {
    for (cudaEvent_t event : events_) {
        if (event != nullptr) { cudaEventDestroy(event); }
    }
}

void PhaseTrace::enable(cudaStream_t stream, std::size_t capacity) {
    disable();
    stream_ = stream;
    // Allocated once up front: creating events inside a timed round would itself be timed.
    events_.reserve(capacity);
    while (events_.size() < capacity) {
        cudaEvent_t event = nullptr;
        CUDA_CHECK(cudaEventCreate(&event));
        events_.push_back(event);
    }
    labels_.clear();
    used_        = 0;
    saw_capture_ = false;
    enabled_     = true;
}

void PhaseTrace::disable() noexcept {
    enabled_ = false;
    labels_.clear();
    used_ = 0;
}

bool PhaseTrace::capturing() const noexcept {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream_, &status) != cudaSuccess) { return true; }
    return status != cudaStreamCaptureStatusNone;
}

void PhaseTrace::begin() noexcept {
    if (capturing()) {
        saw_capture_ = true;
        return;
    }
    labels_.clear();
    used_ = 0;
    if (events_.empty()) { return; }
    if (cudaEventRecord(events_[0], stream_) != cudaSuccess) { return; }
    used_ = 1;
}

void PhaseTrace::mark(const char* label) noexcept {
    if (capturing()) {
        saw_capture_ = true;
        return;
    }
    // Silently drop marks past capacity rather than reallocate mid-round.
    if (used_ == 0 || used_ >= events_.size()) { return; }
    if (cudaEventRecord(events_[used_], stream_) != cudaSuccess) { return; }
    labels_.emplace_back(label);
    ++used_;
}

std::vector<PhaseTrace::Interval> PhaseTrace::collect() {
    std::vector<Interval> intervals;
    if (used_ < 2) { return intervals; }
    CUDA_CHECK(cudaEventSynchronize(events_[used_ - 1]));
    intervals.reserve(labels_.size());
    for (std::size_t index = 0; index + 1 < used_; ++index) {
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, events_[index], events_[index + 1]));
        intervals.push_back(Interval{labels_[index], milliseconds});
    }
    return intervals;
}

} // namespace ninfer
