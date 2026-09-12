#include "product/logging/startup_log.h"

#include "product/logging/logging.h"
#include "product/logging/pretty_format.h"

#include <spdlog/logger.h>

#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
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

enum class PhaseVisibility : std::uint8_t {
    Info,
    Debug,
};

struct PhasePresentation {
    const char* active;
    const char* complete;
    PhaseVisibility visibility;
    bool potentially_long;
    bool show_rate;
};

PhasePresentation phase_presentation(StartupPhase phase) noexcept {
    switch (phase) {
    case StartupPhase::EngineStartup:
        return {"starting engine", "engine startup", PhaseVisibility::Info, true, false};
    case StartupPhase::CudaInitialize:
        return {"initializing CUDA", "CUDA initialized", PhaseVisibility::Debug, false, false};
    case StartupPhase::ArtifactInspect:
        return {"inspecting artifact", "artifact inspected", PhaseVisibility::Debug, false, false};
    case StartupPhase::TargetPlan:
        return {"planning runtime", "runtime planned", PhaseVisibility::Debug, false, false};
    case StartupPhase::WeightsMaterialize:
        return {"loading weights", "weights ready", PhaseVisibility::Info, true, true};
    case StartupPhase::WeightsStagingPin:
        return {"pinning staging buffers", "staging buffers pinned", PhaseVisibility::Debug, false,
                false};
    case StartupPhase::TargetFinalize:
        return {"finalizing target", "target finalized", PhaseVisibility::Debug, false, false};
    case StartupPhase::FrontendInitialize:
        return {"initializing frontend", "frontend ready", PhaseVisibility::Debug, false, false};
    case StartupPhase::ProgramInitialize:
        return {"initializing runtime", "runtime initialized", PhaseVisibility::Debug, false,
                false};
    case StartupPhase::HostStatePin:
        return {"pinning host state", "host state pinned", PhaseVisibility::Info, true, false};
    case StartupPhase::HostKvPin:
        return {"pinning host KV", "host KV pinned", PhaseVisibility::Info, true, false};
    case StartupPhase::CudaGraphPrepare:
        return {"preparing CUDA graphs", "CUDA graphs ready", PhaseVisibility::Info, false, false};
    case StartupPhase::EngineFinalize:
        return {"finalizing engine", "engine finalized", PhaseVisibility::Debug, false, false};
    }
    return {"starting unknown phase", "unknown phase complete", PhaseVisibility::Debug, false,
            false};
}

std::size_t terminal_columns() noexcept {
    winsize size{};
    if (::ioctl(STDERR_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col != 0) { return size.ws_col; }
    return 120;
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

std::string begin_line(const PhasePresentation& phase, const StartupEvent& event) {
    std::string line = phase.active;
    if (event.progress_unit == StartupProgressUnit::Bytes && event.total != 0) {
        line += " | " + format_pretty_bytes(event.total);
    }
    return line;
}

std::string progress_line_candidate(const PhasePresentation& phase, const StartupEvent& event,
                                    const PhaseProgress& state, double ratio, std::size_t bar_width,
                                    bool include_bytes, bool include_rate, bool include_eta) {
    std::ostringstream out;
    out << "  " << phase.active << ' ';
    if (bar_width != 0) { out << '[' << progress_bar(ratio, bar_width) << "] "; }
    out << format_pretty_percent(ratio);
    if (include_bytes) {
        out << " | " << format_pretty_bytes(event.current) << '/'
            << format_pretty_bytes(event.total);
    }
    if (include_rate && state.smoothed_bytes_per_second > 0.0) {
        out << " | "
            << format_pretty_bytes(static_cast<std::uint64_t>(state.smoothed_bytes_per_second))
            << "/s";
        if (include_eta && event.current < event.total) {
            const double eta =
                static_cast<double>(event.total - event.current) / state.smoothed_bytes_per_second;
            out << " | ETA " << format_pretty_duration(eta);
        }
    }
    return out.str();
}

std::string interactive_progress_line(const PhasePresentation& phase, const StartupEvent& event,
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
    std::string compact = "  " + std::string(phase.active) + " " + format_pretty_percent(ratio);
    if (columns > 1 && compact.size() >= columns) { compact.resize(columns - 1); }
    return compact;
}

std::string complete_line(const PhasePresentation& phase, const StartupEvent& event) {
    std::ostringstream out;
    out << phase.complete;
    if (event.progress_unit == StartupProgressUnit::Bytes) {
        const std::uint64_t bytes = event.current != 0 ? event.current : event.total;
        out << " | " << format_pretty_bytes(bytes);
    }
    const double seconds = static_cast<double>(event.elapsed_ns) * 1.0e-9;
    out << " | " << format_pretty_duration(seconds);
    if (phase.show_rate && seconds > 0.0 && event.current != 0) {
        out << " | "
            << format_pretty_bytes(
                   static_cast<std::uint64_t>(static_cast<double>(event.current) / seconds))
            << "/s";
    }
    return out.str();
}

} // namespace

struct StartupLogRenderer::Impl {
    Impl(std::shared_ptr<spdlog::logger> logger_in, std::shared_ptr<TerminalProgress> progress_in)
        : logger(std::move(logger_in)), progress(std::move(progress_in)) {}

    ~Impl() { progress->clear(); }

    void render(const StartupEvent& event) {
        const PhasePresentation presentation = phase_presentation(event.phase);
        const std::size_t index              = static_cast<std::size_t>(event.phase);
        switch (event.status) {
        case StartupStatus::Begin: {
            const Clock::time_point now = Clock::now();
            phases[index]               = PhaseProgress{
                              .last_persistent  = now,
                              .last_interactive = now - kInteractiveRefresh,
                              .sample_time      = now,
            };
            if (event.phase == StartupPhase::EngineStartup) {
                logger->info("starting engine");
            } else if (presentation.visibility == PhaseVisibility::Debug) {
                logger->debug("{}", begin_line(presentation, event));
            } else if (progress->enabled()) {
                progress->update("  " + begin_line(presentation, event));
            } else if (presentation.potentially_long) {
                logger->info("{}", begin_line(presentation, event));
            } else {
                logger->debug("{}", begin_line(presentation, event));
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
                    progress->update(interactive_progress_line(presentation, event, state));
                }
                return;
            }
            if (now - state.last_persistent < kPersistentRefresh) { return; }
            state.last_persistent = now;
            if (event.progress_unit == StartupProgressUnit::Bytes && event.total != 0) {
                logger->info("{}", progress_line_candidate(presentation, event, state,
                                                           static_cast<double>(event.current) /
                                                               static_cast<double>(event.total),
                                                           0, true, true, true));
            }
            return;
        }
        case StartupStatus::Complete: {
            progress->clear();
            if (event.phase == StartupPhase::EngineStartup) {
                engine_elapsed_ns = event.elapsed_ns;
                return;
            }
            const std::string line = complete_line(presentation, event);
            if (presentation.visibility == PhaseVisibility::Info) {
                logger->info("{}", line);
            } else {
                logger->debug("{}", line);
            }
            return;
        }
        case StartupStatus::Failed:
            progress->clear();
            if (failure_logged) { return; }
            failure_logged = true;
            logger->error("startup failed | {} | {}", presentation.active,
                          format_pretty_duration(static_cast<double>(event.elapsed_ns) * 1.0e-9));
            return;
        }
    }

    std::shared_ptr<spdlog::logger> logger;
    std::shared_ptr<TerminalProgress> progress;
    std::array<PhaseProgress, kStartupPhaseCount> phases;
    std::uint64_t engine_elapsed_ns = 0;
    bool failure_logged             = false;
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
    const double total_seconds = impl_->engine_elapsed_ns != 0
                                     ? static_cast<double>(impl_->engine_elapsed_ns) * 1.0e-9
                                     : load.load_seconds;
    impl_->logger->info("engine ready | {}/{} | total {} | weights {}",
                        format_pretty_text(load.model_id), format_pretty_text(load.weights_id),
                        format_pretty_duration(total_seconds),
                        format_pretty_bytes(load.host_to_device_bytes));
    impl_->logger->debug(
        "load detail | target {} | artifact read {} | H2D {} | staging peak {} | tensors {} | "
        "resources {}",
        format_pretty_text(load.target), format_pretty_bytes(load.artifact_bytes_read),
        format_pretty_bytes(load.host_to_device_bytes),
        format_pretty_bytes(load.peak_staging_bytes), format_pretty_count(load.tensor_count),
        format_pretty_count(load.resource_count));
}

} // namespace ninfer::product
