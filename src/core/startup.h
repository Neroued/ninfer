#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <cstdint>

namespace ninfer {

inline void publish_startup_event(const StartupObserver& observer,
                                  const StartupEvent& event) noexcept {
    if (!observer.callback) { return; }
    try {
        observer.callback(event);
    } catch (...) {}
}

class StartupPhaseScope {
public:
    using Clock = std::chrono::steady_clock;

    StartupPhaseScope(const StartupObserver& observer, StartupPhase phase,
                      StartupProgressUnit unit = StartupProgressUnit::None,
                      std::uint64_t total      = 0) noexcept
        : observer_(&observer), phase_(phase), unit_(unit), total_(total), started_(Clock::now()) {
        publish(StartupStatus::Begin, 0, total_);
    }

    ~StartupPhaseScope() noexcept {
        if (!terminal_) { publish(StartupStatus::Failed, current_, total_); }
    }

    StartupPhaseScope(const StartupPhaseScope&)            = delete;
    StartupPhaseScope& operator=(const StartupPhaseScope&) = delete;
    StartupPhaseScope(StartupPhaseScope&&)                 = delete;
    StartupPhaseScope& operator=(StartupPhaseScope&&)      = delete;

    void progress(std::uint64_t current, std::uint64_t total) noexcept {
        current_ = current;
        total_   = total;
        publish(StartupStatus::Progress, current_, total_);
    }

    void complete(std::uint64_t current = 0, std::uint64_t total = 0) noexcept {
        if (unit_ != StartupProgressUnit::None) {
            if (total != 0) { total_ = total; }
            current_ = current;
        }
        terminal_ = true;
        publish(StartupStatus::Complete, current_, total_);
    }

private:
    void publish(StartupStatus status, std::uint64_t current, std::uint64_t total) const noexcept {
        const std::uint64_t elapsed_ns =
            status == StartupStatus::Begin || status == StartupStatus::Progress
                ? 0
                : static_cast<std::uint64_t>(
                      std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started_)
                          .count());
        publish_startup_event(*observer_, StartupEvent{
                                              .phase         = phase_,
                                              .status        = status,
                                              .progress_unit = unit_,
                                              .current       = current,
                                              .total         = total,
                                              .elapsed_ns    = elapsed_ns,
                                          });
    }

    const StartupObserver* observer_ = nullptr;
    StartupPhase phase_              = StartupPhase::EngineStartup;
    StartupProgressUnit unit_        = StartupProgressUnit::None;
    std::uint64_t current_           = 0;
    std::uint64_t total_             = 0;
    Clock::time_point started_;
    bool terminal_ = false;
};

} // namespace ninfer
