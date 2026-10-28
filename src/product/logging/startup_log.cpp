#include "product/logging/startup_log.h"

#include "core/startup.h"
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
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::product {
namespace {

using Clock = std::chrono::steady_clock;

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
        return {"initializing runtime", "runtime initialized", PhaseVisibility::Info, true, false};
    case StartupPhase::HostContextPin:
        return {"pinning host context", "host context pinned", PhaseVisibility::Info, true, false};
    case StartupPhase::CudaGraphPrepare:
        return {"preparing CUDA graphs", "CUDA graphs ready", PhaseVisibility::Info, false, false};
    case StartupPhase::CudaGraphCalibration:
        return {"calibrating CUDA graph memory", "CUDA graph memory calibrated",
                PhaseVisibility::Info, false, false};
    case StartupPhase::CommunicationInitialize:
        return {"connecting communication", "communication connected", PhaseVisibility::Info, false,
                false};
    case StartupPhase::CommunicationWarmup:
        return {"warming communication", "communication ready", PhaseVisibility::Info, true, false};
    case StartupPhase::RuntimeMemoryCheck:
        return {"checking runtime memory", "runtime memory checked", PhaseVisibility::Debug, false,
                false};
    case StartupPhase::RuntimeRelease:
        return {"releasing runtime", "runtime released", PhaseVisibility::Debug, false, false};
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
    std::optional<StartupEvent> event;
    std::uint64_t sample_elapsed_ns  = 0;
    std::uint64_t sample_bytes       = 0;
    double smoothed_bytes_per_second = 0.0;
};

void update_rate(PhaseProgress& state, std::uint64_t current, std::uint64_t elapsed_ns) {
    const double seconds = elapsed_ns > state.sample_elapsed_ns
                               ? static_cast<double>(elapsed_ns - state.sample_elapsed_ns) * 1e-9
                               : 0.0;
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
    state.sample_elapsed_ns = elapsed_ns;
    state.sample_bytes      = current;
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
                                      const PhaseProgress& state, std::size_t columns) {
    const double ratio =
        event.total == 0
            ? 0.0
            : std::clamp(static_cast<double>(event.current) / static_cast<double>(event.total), 0.0,
                         1.0);
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

    struct SourceProgress {
        std::array<PhaseProgress, kStartupPhaseCount> phases;
        std::uint32_t attempt = 0;
        bool failure_logged   = false;
        std::string prefix;
    };

    void refresh(Clock::time_point now, bool force) {
        if (!progress->enabled() || (!force && now - last_interactive < kInteractiveRefresh))
            return;
        last_interactive = now;
        std::vector<std::pair<const SourceProgress*, const PhaseProgress*>> active;
        for (const auto& [key, source] : sources) {
            const PhaseProgress* latest = nullptr;
            for (const auto& phase : source.phases)
                if (phase.event && phase.event->status <= StartupStatus::Progress &&
                    (!latest || phase.event->invocation > latest->event->invocation))
                    latest = &phase;
            if (latest) active.emplace_back(&source, latest);
        }
        if (active.size() > 1)
            std::erase_if(active, [](const auto& item) {
                const auto& event = *item.second->event;
                return !event.origin && event.phase == StartupPhase::EngineStartup;
            });
        if (active.empty()) {
            progress->clear();
            return;
        }
        const auto columns = std::max<std::size_t>(1, terminal_columns() / active.size());
        std::string line;
        for (const auto& [source, phase] : active) {
            const auto& event       = *phase->event;
            const auto presentation = phase_presentation(event.phase);
            const auto available =
                columns > source->prefix.size() + 3 ? columns - source->prefix.size() - 3 : 1;
            std::string part =
                event.progress_unit == StartupProgressUnit::Bytes && event.total
                    ? interactive_progress_line(presentation, event, *phase, available)
                    : begin_line(presentation, event);
            if (part.size() > available) part.resize(available);
            if (!line.empty()) line += " | ";
            line += source->prefix + part;
        }
        progress->update(line);
    }

    void render(const StartupEvent& event) {
        std::lock_guard lock(mutex);
        const auto key = event.origin ? std::pair<std::int64_t, std::int64_t>{event.origin->rank,
                                                                              event.origin->node}
                                      : std::pair<std::int64_t, std::int64_t>{-1, -1};
        auto& source   = sources[key];
        if (event.attempt < source.attempt) return;
        if (event.attempt > source.attempt) source = {};
        source.attempt = event.attempt;
        source.prefix  = event.origin ? "[rank " + std::to_string(event.origin->rank) + "/node " +
                                           std::to_string(event.origin->node) + "] "
                                      : "";
        if (event.attempt) source.prefix += "[attempt " + std::to_string(event.attempt + 1) + "] ";
        auto& state = source.phases.at(static_cast<std::size_t>(event.phase));
        if (state.event && !startup_event_follows(event, *state.event)) return;
        const auto now = Clock::now();
        if (!state.event || state.event->invocation != event.invocation) {
            state                 = {};
            state.last_persistent = now;
        }
        state.event             = event;
        const auto presentation = phase_presentation(event.phase);
        switch (event.status) {
        case StartupStatus::Begin:
            if (event.phase == StartupPhase::EngineStartup) {
                logger->info("{}starting engine", source.prefix);
            } else if (presentation.visibility == PhaseVisibility::Info &&
                       (event.origin || presentation.potentially_long) && !progress->enabled()) {
                logger->info("{}{}", source.prefix, begin_line(presentation, event));
            } else {
                logger->debug("{}{}", source.prefix, begin_line(presentation, event));
            }
            break;
        case StartupStatus::Progress:
            update_rate(state, event.current, event.elapsed_ns);
            if (!progress->enabled() && now - state.last_persistent >= kPersistentRefresh &&
                event.progress_unit == StartupProgressUnit::Bytes && event.total) {
                state.last_persistent = now;
                logger->info(
                    "{}{}", source.prefix,
                    progress_line_candidate(presentation, event, state,
                                            static_cast<double>(event.current) / event.total, 0,
                                            true, true, true));
            }
            break;
        case StartupStatus::Complete:
            if (event.phase == StartupPhase::EngineStartup) {
                if (!event.origin) engine_elapsed_ns = event.elapsed_ns;
            } else if (presentation.visibility == PhaseVisibility::Info) {
                logger->info("{}{}", source.prefix, complete_line(presentation, event));
            } else {
                logger->debug("{}{}", source.prefix, complete_line(presentation, event));
            }
            break;
        case StartupStatus::Failed:
            if (!source.failure_logged) {
                source.failure_logged = true;
                logger->error("{}{} | {} | {}", source.prefix,
                              event.origin ? "startup phase failed" : "startup failed",
                              presentation.active,
                              format_pretty_duration(static_cast<double>(event.elapsed_ns) * 1e-9));
            }
            break;
        }
        refresh(now, event.status != StartupStatus::Progress);
    }

    std::shared_ptr<spdlog::logger> logger;
    std::shared_ptr<TerminalProgress> progress;
    std::mutex mutex;
    std::map<std::pair<std::int64_t, std::int64_t>, SourceProgress> sources;
    Clock::time_point last_interactive{};
    std::uint64_t engine_elapsed_ns = 0;
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
    impl_->logger->info("engine ready | {} | total {} | weights {} | CUDA sync {}",
                        format_pretty_text(load.model_name), format_pretty_duration(total_seconds),
                        format_pretty_bytes(load.host_to_device_bytes), load.cuda_sync_mode);
    impl_->logger->debug(
        "load detail | architecture {} | artifact read {} | H2D {} | staging peak {} | device "
        "objects {} | host objects {}",
        format_pretty_text(load.architecture), format_pretty_bytes(load.artifact_bytes_read),
        format_pretty_bytes(load.host_to_device_bytes),
        format_pretty_bytes(load.max_rank_staging_bytes),
        format_pretty_count(load.device_object_count), format_pretty_count(load.host_object_count));
    impl_->logger->debug("context cost | transfer {} | prefill {} | signature {}",
                         context_cost_preset_source_name(load.context_cost.transfer_source),
                         context_cost_preset_source_name(load.context_cost.prefill_source),
                         load.prefill_signature);
}

} // namespace ninfer::product
