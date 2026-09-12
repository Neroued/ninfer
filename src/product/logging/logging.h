#pragma once

#include <memory>
#include <string>
#include <string_view>

namespace spdlog {
class logger;
}

namespace ninfer::product {

enum class LogLevel {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Critical,
    Off,
};

enum class LogColorMode {
    Auto,
    Always,
    Never,
};

enum class LogPresentation {
    Service,
    Tool,
};

struct LoggingOptions {
    std::string logger_name;
    LogLevel level               = LogLevel::Info;
    LogColorMode color           = LogColorMode::Auto;
    LogPresentation presentation = LogPresentation::Service;
};

[[nodiscard]] LogLevel parse_log_level(std::string_view value);

// One transient terminal line coordinated with the operational stderr sink. It is enabled only
// when stderr is a terminal; redirected output remains persistent spdlog records only.
class TerminalProgress {
public:
    ~TerminalProgress();

    TerminalProgress(const TerminalProgress&)            = delete;
    TerminalProgress& operator=(const TerminalProgress&) = delete;
    TerminalProgress(TerminalProgress&&)                 = delete;
    TerminalProgress& operator=(TerminalProgress&&)      = delete;

    [[nodiscard]] bool enabled() const noexcept;
    void update(std::string line);
    void clear() noexcept;

private:
    friend class LoggingRuntime;
    struct Impl;
    explicit TerminalProgress(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

// Application-owned operational logger lifetime. Construction does not mutate spdlog's global
// default logger or registry; producers receive and retain the returned explicit shared handle.
class LoggingRuntime {
public:
    explicit LoggingRuntime(LoggingOptions options);
    ~LoggingRuntime();

    LoggingRuntime(const LoggingRuntime&)            = delete;
    LoggingRuntime& operator=(const LoggingRuntime&) = delete;
    LoggingRuntime(LoggingRuntime&&)                 = delete;
    LoggingRuntime& operator=(LoggingRuntime&&)      = delete;

    [[nodiscard]] std::shared_ptr<spdlog::logger> logger() const noexcept;
    [[nodiscard]] std::shared_ptr<TerminalProgress> terminal_progress() const noexcept;
    void flush() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::product
