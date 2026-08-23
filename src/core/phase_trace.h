#pragma once

// Diagnostic CUDA-event phase timing for the decode schedules.
//
// The round benchmarks time one whole `decode_batch` plus its resolve, which says how long a
// speculative round takes but not which side of it owns the time -- target verification, the
// per-draft proposal evaluations, or acceptance and commit. This records an event at each phase
// boundary so the round can be attributed without a profiler, which matters on hardware where
// nsys drops launches (see the V100 performance summary).
//
// Disabled by default, and when disabled every call site is one relaxed bool load and a
// not-taken branch, on the order of ten per round. It is a process-wide singleton rather than a
// parameter threaded through `Program::decode_batch` because it is a benchmark-only diagnostic
// and does not belong in the public scheduling API.
//
// Not graph-safe by design. A captured graph replays without re-recording host-visible events, so
// `mark()` silently does nothing while the stream is capturing and the schedule must be run with
// CUDA graphs disabled for the trace to see anything. `enabled_during_capture()` reports whether
// that happened, so a caller can fail loudly rather than print an empty table.

#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

namespace ninfer {

class PhaseTrace {
public:
    struct Interval {
        std::string label;
        float milliseconds = 0.0F;
    };

    static PhaseTrace& instance() noexcept;

    // `capacity` is the maximum number of marks in one round, including the opening one.
    void enable(cudaStream_t stream, std::size_t capacity);
    void disable() noexcept;
    [[nodiscard]] bool enabled() const noexcept { return enabled_; }
    [[nodiscard]] bool saw_capture() const noexcept { return saw_capture_; }

    // Opens a round: records the first event and clears the previous round's labels.
    void begin() noexcept;
    // Closes the interval that just ended and names it.
    void mark(const char* label) noexcept;
    // Synchronises on the last event and returns one entry per closed interval.
    [[nodiscard]] std::vector<Interval> collect();

    PhaseTrace(const PhaseTrace&)            = delete;
    PhaseTrace& operator=(const PhaseTrace&) = delete;

private:
    PhaseTrace() = default;
    ~PhaseTrace();

    [[nodiscard]] bool capturing() const noexcept;

    bool enabled_        = false;
    bool saw_capture_    = false;
    cudaStream_t stream_ = nullptr;
    std::vector<cudaEvent_t> events_;
    std::vector<std::string> labels_;
    std::size_t used_ = 0;
};

// Call-site helpers. Written so the disabled path is a bool test and nothing else.
inline void phase_trace_begin() noexcept {
    PhaseTrace& trace = PhaseTrace::instance();
    if (trace.enabled()) { trace.begin(); }
}

inline void phase_trace_mark(const char* label) noexcept {
    PhaseTrace& trace = PhaseTrace::instance();
    if (trace.enabled()) { trace.mark(label); }
}

} // namespace ninfer
