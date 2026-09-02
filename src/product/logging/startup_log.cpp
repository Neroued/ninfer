#include "product/logging/startup_log.h"

#include "product/logging/logging.h"

#include <spdlog/logger.h>

#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace ninfer::product {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kStartupPhaseCount =
    static_cast<std::size_t>(StartupPhase::EngineFinalize) + 1;
constexpr auto kInteractiveRefresh = std::chrono::milliseconds(200);
constexpr auto kPersistentRefresh  = std::chrono::seconds(10);

const char* phase_name(StartupPhase phase) {
    switch (phase) {
    case StartupPhase::EngineStartup:
        return "engine-startup";
    case StartupPhase::CudaInitialize:
        return "cuda-initialize";
    case StartupPhase::ArtifactInspect:
        return "artifact-inspect";
    case StartupPhase::TargetPlan:
        return "target-plan";
    case StartupPhase::WeightsMaterialize:
        return "weights-materialize";
    case StartupPhase::WeightsStagingPin:
        return "weights-staging-pin";
    case StartupPhase::TargetFinalize:
        return "target-finalize";
    case StartupPhase::FrontendInitialize:
        return "frontend-initialize";
    case StartupPhase::ProgramInitialize:
        return "program-initialize";
    case StartupPhase::HostStatePin:
        return "host-state-pin";
    case StartupPhase::HostKvPin:
        return "host-kv-pin";
    case StartupPhase::CudaGraphPrepare:
        return "cuda-graph-prepare";
    case StartupPhase::EngineFinalize:
        return "engine-finalize";
    }
    return "unknown";
}

std::size_t terminal_columns() noexcept {
    winsize size{};
    if (::ioctl(STDERR_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col != 0) { return size.ws_col; }
    return 120;
}

std::string human_bytes(double bytes) {
    constexpr std::array<const char*, 7> units = {
        "B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB",
    };
    std::size_t unit = 0;
    while (bytes >= 1024.0 && unit + 1 < units.size()) {
        bytes /= 1024.0;
        ++unit;
    }
    std::ostringstream out;
    if (unit == 0) {
        out << std::fixed << std::setprecision(0) << bytes << ' ' << units[unit];
    } else {
        out << std::fixed << std::setprecision(2) << bytes << ' ' << units[unit];
    }
    return out.str();
}

std::string progress_bar(double ratio, std::size_t width) {
    const std::size_t completed = static_cast<std::size_t>(ratio * static_cast<double>(width));
    std::string bar(width, '.');
    for (std::size_t index = 0; index < std::min(completed, width); ++index) { bar[index] = '='; }
    if (completed < width) { bar[completed] = '>'; }
    return bar;
}

struct PhaseProgress {
    Clock::time_point last_persistent;
    Clock::time_point last_interactive;
    Clock::time_point sample_time;
    std::uint64_t sample_bytes       = 0;
    double smoothed_bytes_per_second = 0.0;
};

void update_rate(PhaseProgress& state, std::uint64_t current, Clock::time_point now) {
    const double seconds = std::chrono::duration<double>(now - state.sample_time).count();
    if (current >= state.sample_bytes && seconds > 0.0) {
        const double instantaneous = static_cast<double>(current - state.sample_bytes) / seconds;
        if (instantaneous > 0.0) {
            constexpr double alpha = 0.25;
            state.smoothed_bytes_per_second =
                state.smoothed_bytes_per_second > 0.0
                    ? alpha * instantaneous + (1.0 - alpha) * state.smoothed_bytes_per_second
                    : instantaneous;
        }
    }
    state.sample_time  = now;
    state.sample_bytes = current;
}

std::string progress_percentage(double ratio) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << std::setw(5) << 100.0 * ratio << '%';
    return out.str();
}

std::string progress_line_candidate(const char* phase, const StartupEvent& event,
                                    const PhaseProgress& state, double ratio, std::size_t bar_width,
                                    bool include_bytes, bool include_rate, bool include_eta) {
    std::ostringstream out;
    out << "  -> " << phase << ' ';
    if (bar_width != 0) { out << '[' << progress_bar(ratio, bar_width) << "] "; }
    out << progress_percentage(ratio);
    if (include_bytes) {
        out << "  submitted " << human_bytes(static_cast<double>(event.current)) << '/'
            << human_bytes(static_cast<double>(event.total));
    }
    if (include_rate && state.smoothed_bytes_per_second > 0.0) {
        out << "  pipe " << human_bytes(state.smoothed_bytes_per_second) << "/s";
        if (include_eta && event.current < event.total) {
            const double eta =
                static_cast<double>(event.total - event.current) / state.smoothed_bytes_per_second;
            out << "  eta " << std::fixed << std::setprecision(1) << eta << 's';
        }
    }
    return out.str();
}

std::string interactive_progress_line(const char* phase, const StartupEvent& event,
                                      const PhaseProgress& state) {
    const double ratio =
        event.total == 0
            ? 0.0
            : std::clamp(static_cast<double>(event.current) / static_cast<double>(event.total), 0.0,
                         1.0);
    const std::size_t columns   = terminal_columns();
    const std::array candidates = {
        progress_line_candidate(phase, event, state, ratio, 20, true, true, true),
        progress_line_candidate(phase, event, state, ratio, 16, true, true, false),
        progress_line_candidate(phase, event, state, ratio, 12, true, false, false),
        progress_line_candidate(phase, event, state, ratio, 0, true, false, false),
    };
    for (const std::string& candidate : candidates) {
        if (candidate.size() < columns) { return candidate; }
    }
    std::string compact = ' ' + progress_percentage(ratio) + ' ' + phase;
    if (columns > 1 && compact.size() >= columns) { compact.resize(columns - 1); }
    return compact;
}

} // namespace

struct StartupLogRenderer::Impl {
    Impl(std::shared_ptr<spdlog::logger> logger_in, std::shared_ptr<TerminalProgress> progress_in)
        : logger(std::move(logger_in)), progress(std::move(progress_in)) {}

    ~Impl() { progress->clear(); }

    void render(const StartupEvent& event) {
        const char* phase       = phase_name(event.phase);
        const std::size_t index = static_cast<std::size_t>(event.phase);
        switch (event.status) {
        case StartupStatus::Begin: {
            const Clock::time_point now = Clock::now();
            phases[index]               = PhaseProgress{
                              .last_persistent  = now,
                              .last_interactive = now - kInteractiveRefresh,
                              .sample_time      = now,
            };
            if (event.progress_unit == StartupProgressUnit::Bytes && event.total != 0) {
                logger->info("startup phase={} status=begin total_bytes={}", phase, event.total);
            } else {
                logger->info("startup phase={} status=begin", phase);
            }
            return;
        }
        case StartupStatus::Progress: {
            const Clock::time_point now = Clock::now();
            PhaseProgress& state        = phases[index];
            update_rate(state, event.current, now);
            if (progress->enabled()) {
                if (now - state.last_interactive < kInteractiveRefresh) { return; }
                state.last_interactive = now;
                if (event.progress_unit == StartupProgressUnit::Bytes && event.total != 0) {
                    progress->update(interactive_progress_line(phase, event, state));
                }
                return;
            }
            if (now - state.last_persistent < kPersistentRefresh) { return; }
            state.last_persistent = now;
            if (event.progress_unit == StartupProgressUnit::Bytes) {
                logger->info("startup phase={} status=progress submitted_bytes={} total_bytes={}",
                             phase, event.current, event.total);
            }
            return;
        }
        case StartupStatus::Complete:
            progress->clear();
            if (event.progress_unit == StartupProgressUnit::Bytes) {
                logger->info("startup phase={} status=complete completed_bytes={} total_bytes={} "
                             "duration_ms={:.3f}",
                             phase, event.current, event.total,
                             static_cast<double>(event.elapsed_ns) / 1.0e6);
            } else {
                logger->info("startup phase={} status=complete duration_ms={:.3f}", phase,
                             static_cast<double>(event.elapsed_ns) / 1.0e6);
            }
            return;
        case StartupStatus::Failed:
            progress->clear();
            if (event.progress_unit == StartupProgressUnit::Bytes) {
                logger->error("startup phase={} status=failed submitted_bytes={} total_bytes={} "
                              "duration_ms={:.3f}",
                              phase, event.current, event.total,
                              static_cast<double>(event.elapsed_ns) / 1.0e6);
            } else {
                logger->error("startup phase={} status=failed duration_ms={:.3f}", phase,
                              static_cast<double>(event.elapsed_ns) / 1.0e6);
            }
            return;
        }
    }

    std::shared_ptr<spdlog::logger> logger;
    std::shared_ptr<TerminalProgress> progress;
    std::array<PhaseProgress, kStartupPhaseCount> phases;
};

StartupLogRenderer::StartupLogRenderer(LoggingRuntime& logging)
    : impl_(std::make_shared<Impl>(logging.logger(), logging.terminal_progress())) {}

StartupLogRenderer::~StartupLogRenderer() = default;

StartupObserver StartupLogRenderer::observer() {
    const std::shared_ptr<Impl> state = impl_;
    return StartupObserver{
        .callback = [state](const StartupEvent& event) { state->render(event); }};
}

void StartupLogRenderer::engine_ready(const LoadSummary& load) {
    impl_->progress->clear();
    impl_->logger->info(
        "engine status=ready target={} model_id={} weights_id={} target_load_ms={:.3f} "
        "materialization_pipeline_ms={:.3f} artifact_bytes_read={} host_to_device_bytes={} "
        "peak_staging_bytes={} tensors={} resources={}",
        quote_log_value(load.target), quote_log_value(load.model_id),
        quote_log_value(load.weights_id), load.load_seconds * 1000.0, load.upload_seconds * 1000.0,
        load.artifact_bytes_read, load.host_to_device_bytes, load.peak_staging_bytes,
        load.tensor_count, load.resource_count);
}

} // namespace ninfer::product
