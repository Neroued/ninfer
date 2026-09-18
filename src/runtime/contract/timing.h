#pragma once

#include "core/nvtx.h"
#include <chrono>
#include <cstdint>
#include <optional>

namespace ninfer::runtime {

// One Program execution may alternate between Host submission, a blocking Device completion
// wait, and Host post-processing. The three monotonic components are returned to Engine as part of
// the execution capability; serve never infers them from total request time.
struct ExecutionTiming {
    std::uint64_t submit_host_ns = 0;
    std::uint64_t device_wait_ns = 0;
    std::uint64_t post_host_ns   = 0;

    ExecutionTiming& operator+=(ExecutionTiming other) noexcept {
        submit_host_ns += other.submit_host_ns;
        device_wait_ns += other.device_wait_ns;
        post_host_ns += other.post_host_ns;
        return *this;
    }

    [[nodiscard]] std::uint64_t host_ns() const noexcept { return submit_host_ns + post_host_ns; }

    [[nodiscard]] std::uint64_t elapsed_ns() const noexcept { return host_ns() + device_wait_ns; }
};

enum class ExecutionTimingPhase : std::uint8_t {
    Submit,
    Wait,
    Post,
    Paused,
};

// Fixed-cost coarse recorder used only at Program phase boundaries, never in token/page/layer
// loops. It starts in Submit, permits explicit Submit -> Wait -> Post transitions, and can resume
// Submit for a later segment in the same execution unit.
class ExecutionTimingRecorder {
public:
    using Clock = std::chrono::steady_clock;

    explicit ExecutionTimingRecorder(
        ExecutionTimingPhase initial_phase = ExecutionTimingPhase::Submit,
        ExecutionTiming* abandoned_timing  = nullptr) noexcept
        : started_(Clock::now()), phase_(initial_phase), abandoned_timing_(abandoned_timing) {
        open_range();
    }

    ~ExecutionTimingRecorder() noexcept {
        if (finished_) { return; }
        const ExecutionTiming timing = finish();
        if (abandoned_timing_ != nullptr) { *abandoned_timing_ += timing; }
    }

    ExecutionTimingRecorder(const ExecutionTimingRecorder&)            = delete;
    ExecutionTimingRecorder& operator=(const ExecutionTimingRecorder&) = delete;

    void begin_wait() noexcept { transition(ExecutionTimingPhase::Wait); }

    void end_wait() noexcept { transition(ExecutionTimingPhase::Post); }

    void resume_submit() noexcept { transition(ExecutionTimingPhase::Submit); }

    void resume_post() noexcept { transition(ExecutionTimingPhase::Post); }

    void pause() noexcept { transition(ExecutionTimingPhase::Paused); }

    void include(ExecutionTiming timing) noexcept { timing_ += timing; }

    [[nodiscard]] ExecutionTiming finish() noexcept {
        if (finished_) { return timing_; }
        accumulate(Clock::now());
        range_.reset();
        phase_    = ExecutionTimingPhase::Paused;
        finished_ = true;
        return timing_;
    }

private:
    [[nodiscard]] static nvtx::Name range_name(ExecutionTimingPhase phase) noexcept {
        switch (phase) {
        case ExecutionTimingPhase::Submit:
            return nvtx::Name::ProgramSubmit;
        case ExecutionTimingPhase::Wait:
            return nvtx::Name::DeviceWait;
        case ExecutionTimingPhase::Post:
            return nvtx::Name::ProgramPost;
        case ExecutionTimingPhase::Paused:
            break;
        }
        return nvtx::Name::ProgramSubmit;
    }

    void open_range() noexcept {
        if (phase_ != ExecutionTimingPhase::Paused) {
            range_.emplace(range_name(phase_), nvtx::Category::Runtime);
        }
    }

    void transition(ExecutionTimingPhase next) noexcept {
        if (finished_ || next == phase_) { return; }
        const Clock::time_point now = Clock::now();
        accumulate(now);
        range_.reset();
        phase_   = next;
        started_ = Clock::now();
        open_range();
    }

    void accumulate(Clock::time_point now) noexcept {
        const auto count =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - started_).count();
        const std::uint64_t elapsed = count > 0 ? static_cast<std::uint64_t>(count) : 0;
        switch (phase_) {
        case ExecutionTimingPhase::Submit:
            timing_.submit_host_ns += elapsed;
            break;
        case ExecutionTimingPhase::Wait:
            timing_.device_wait_ns += elapsed;
            break;
        case ExecutionTimingPhase::Post:
            timing_.post_host_ns += elapsed;
            break;
        case ExecutionTimingPhase::Paused:
            break;
        }
    }

    Clock::time_point started_;
    ExecutionTimingPhase phase_ = ExecutionTimingPhase::Submit;
    ExecutionTiming timing_;
    std::optional<nvtx::ScopedRange> range_;
    ExecutionTiming* abandoned_timing_ = nullptr;
    bool finished_                     = false;
};

} // namespace ninfer::runtime
