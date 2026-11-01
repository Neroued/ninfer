#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <atomic>
#include <cstdint>

namespace ninfer {

inline constexpr std::size_t kStartupPhaseCount =
    static_cast<std::size_t>(StartupPhase::EngineFinalize) + 1;

// Orders updates for one source and phase, including coalesced or repeated snapshots.
inline bool startup_event_follows(const StartupEvent& event,
                                  const StartupEvent& previous) noexcept {
    if (event.attempt != previous.attempt) return event.attempt > previous.attempt;
    if (event.invocation != previous.invocation) return event.invocation > previous.invocation;
    if (previous.status >= StartupStatus::Complete || event.status < previous.status) return false;
    return event.status != previous.status || event.elapsed_ns > previous.elapsed_ns ||
           event.current > previous.current || event.total != previous.total;
}

inline const char* startup_phase_name(StartupPhase phase) noexcept {
    switch (phase) {
    case StartupPhase::EngineStartup:
        return "engine startup";
    case StartupPhase::CudaInitialize:
        return "CUDA initialization";
    case StartupPhase::ArtifactInspect:
        return "artifact inspection";
    case StartupPhase::TargetPlan:
        return "target planning";
    case StartupPhase::WeightsMaterialize:
        return "weight loading";
    case StartupPhase::WeightsStagingPin:
        return "staging pinning";
    case StartupPhase::TargetFinalize:
        return "target finalization";
    case StartupPhase::FrontendInitialize:
        return "frontend initialization";
    case StartupPhase::ProgramInitialize:
        return "runtime preparation";
    case StartupPhase::HostContextPin:
        return "Host context pinning";
    case StartupPhase::CudaGraphPrepare:
        return "CUDA Graph preparation";
    case StartupPhase::CudaGraphCalibration:
        return "CUDA Graph calibration";
    case StartupPhase::CommunicationInitialize:
        return "communication initialization";
    case StartupPhase::CommunicationWarmup:
        return "communication warmup";
    case StartupPhase::RuntimeMemoryCheck:
        return "runtime memory check";
    case StartupPhase::RuntimeRelease:
        return "runtime release";
    case StartupPhase::EngineFinalize:
        return "engine finalization";
    }
    return "unknown startup phase";
}

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
        static std::atomic<std::uint64_t> next{0};
        invocation_ = next.fetch_add(1, std::memory_order_relaxed) + 1;
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

    void fail() noexcept {
        terminal_ = true;
        publish(StartupStatus::Failed, current_, total_);
    }

private:
    void publish(StartupStatus status, std::uint64_t current, std::uint64_t total) const noexcept {
        const std::uint64_t elapsed_ns =
            status == StartupStatus::Begin
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
                                              .invocation    = invocation_,
                                          });
    }

    const StartupObserver* observer_ = nullptr;
    StartupPhase phase_              = StartupPhase::EngineStartup;
    StartupProgressUnit unit_        = StartupProgressUnit::None;
    std::uint64_t current_           = 0;
    std::uint64_t total_             = 0;
    Clock::time_point started_;
    bool terminal_            = false;
    std::uint64_t invocation_ = 0;
};

} // namespace ninfer
