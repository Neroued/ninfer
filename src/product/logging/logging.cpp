#include "product/logging/logging.h"

#include <spdlog/formatter.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iterator>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::product {
namespace {

spdlog::level::level_enum to_spdlog_level(LogLevel level) {
    switch (level) {
    case LogLevel::Trace:
        return spdlog::level::trace;
    case LogLevel::Debug:
        return spdlog::level::debug;
    case LogLevel::Info:
        return spdlog::level::info;
    case LogLevel::Warning:
        return spdlog::level::warn;
    case LogLevel::Error:
        return spdlog::level::err;
    case LogLevel::Critical:
        return spdlog::level::critical;
    case LogLevel::Off:
        return spdlog::level::off;
    }
    throw std::invalid_argument("LoggingOptions level is invalid");
}

std::string_view service_level_name(spdlog::level::level_enum level) noexcept {
    switch (level) {
    case spdlog::level::trace:
        return "TRACE";
    case spdlog::level::debug:
        return "DEBUG";
    case spdlog::level::info:
        return "INFO ";
    case spdlog::level::warn:
        return "WARN ";
    case spdlog::level::err:
        return "ERROR";
    case spdlog::level::critical:
        return "FATAL";
    case spdlog::level::off:
        return "OFF  ";
    case spdlog::level::n_levels:
        break;
    }
    return "UNKWN";
}

std::string_view tool_level_name(spdlog::level::level_enum level) noexcept {
    switch (level) {
    case spdlog::level::trace:
        return "trace: ";
    case spdlog::level::debug:
        return "debug: ";
    case spdlog::level::info:
        return {};
    case spdlog::level::warn:
        return "warning: ";
    case spdlog::level::err:
        return "error: ";
    case spdlog::level::critical:
        return "fatal: ";
    case spdlog::level::off:
    case spdlog::level::n_levels:
        break;
    }
    return {};
}

class PrettyLogFormatter final : public spdlog::formatter {
public:
    explicit PrettyLogFormatter(LogPresentation presentation) : presentation_(presentation) {}

    void format(const spdlog::details::log_msg& message,
                spdlog::memory_buf_t& destination) override {
        if (presentation_ == LogPresentation::Service) {
            const auto since_epoch = message.time.time_since_epoch();
            const auto whole_seconds =
                std::chrono::duration_cast<std::chrono::seconds>(since_epoch);
            const auto milliseconds =
                std::chrono::duration_cast<std::chrono::milliseconds>(since_epoch - whole_seconds)
                    .count();
            const std::time_t wall_seconds = std::chrono::system_clock::to_time_t(
                std::chrono::system_clock::time_point(whole_seconds));
            std::tm local{};
            localtime_r(&wall_seconds, &local);
            fmt::format_to(std::back_inserter(destination),
                           "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}  ", local.tm_year + 1900,
                           local.tm_mon + 1, local.tm_mday, local.tm_hour, local.tm_min,
                           local.tm_sec, milliseconds);
            message.color_range_start    = destination.size();
            const std::string_view level = service_level_name(message.level);
            destination.append(level.data(), level.data() + level.size());
            message.color_range_end = destination.size();
            destination.push_back(' ');
        } else {
            const std::string_view level = tool_level_name(message.level);
            message.color_range_start    = destination.size();
            destination.append(level.data(), level.data() + level.size());
            message.color_range_end = destination.size();
        }
        destination.append(message.payload.data(), message.payload.data() + message.payload.size());
        destination.push_back('\n');
    }

    [[nodiscard]] std::unique_ptr<spdlog::formatter> clone() const override {
        return std::make_unique<PrettyLogFormatter>(presentation_);
    }

private:
    LogPresentation presentation_;
};

spdlog::color_mode to_spdlog_color_mode(LogColorMode mode) {
    switch (mode) {
    case LogColorMode::Auto:
        return spdlog::color_mode::automatic;
    case LogColorMode::Always:
        return spdlog::color_mode::always;
    case LogColorMode::Never:
        return spdlog::color_mode::never;
    }
    throw std::invalid_argument("LoggingOptions color mode is invalid");
}

void report_logging_error(const std::string& message) noexcept {
    static std::atomic_flag reported = ATOMIC_FLAG_INIT;
    if (reported.test_and_set(std::memory_order_relaxed)) { return; }
    std::fprintf(stderr, "ninfer logging failure: %s\n", message.c_str());
    std::fflush(stderr);
}

class ProgressAwareStderrSink final : public spdlog::sinks::sink {
public:
    explicit ProgressAwareStderrSink(spdlog::color_mode color)
        : sink_(color), interactive_(::isatty(STDERR_FILENO) == 1) {}

    ~ProgressAwareStderrSink() override { clear(); }

    void log(const spdlog::details::log_msg& message) override {
        std::lock_guard lock(mutex_);
        erase_unlocked();
        try {
            sink_.log(message);
        } catch (...) {
            draw_unlocked();
            throw;
        }
        draw_unlocked();
    }

    void flush() override {
        std::lock_guard lock(mutex_);
        sink_.flush();
        if (interactive_) { std::fflush(stderr); }
    }

    void set_pattern(const std::string& pattern) override {
        std::lock_guard lock(mutex_);
        sink_.set_pattern(pattern);
    }

    void set_formatter(std::unique_ptr<spdlog::formatter> formatter) override {
        std::lock_guard lock(mutex_);
        sink_.set_formatter(std::move(formatter));
    }

    [[nodiscard]] bool interactive() const noexcept { return interactive_; }

    void update(std::string line) {
        if (!interactive_) { return; }
        std::lock_guard lock(mutex_);
        erase_unlocked();
        active_line_ = std::move(line);
        draw_unlocked();
    }

    void clear() noexcept {
        if (!interactive_) { return; }
        try {
            std::lock_guard lock(mutex_);
            erase_unlocked();
            active_line_.clear();
            std::fflush(stderr);
        } catch (...) {}
    }

private:
    void erase_unlocked() {
        if (!interactive_ || rendered_width_ == 0) { return; }
        std::fputc('\r', stderr);
        for (std::size_t index = 0; index < rendered_width_; ++index) { std::fputc(' ', stderr); }
        std::fputc('\r', stderr);
        rendered_width_ = 0;
    }

    void draw_unlocked() {
        if (!interactive_ || active_line_.empty()) { return; }
        std::fwrite(active_line_.data(), 1, active_line_.size(), stderr);
        std::fflush(stderr);
        rendered_width_ = active_line_.size();
    }

    std::mutex mutex_;
    spdlog::sinks::stderr_color_sink_st sink_;
    bool interactive_ = false;
    std::string active_line_;
    std::size_t rendered_width_ = 0;
};

} // namespace

LogLevel parse_log_level(std::string_view value) {
    if (value == "trace") { return LogLevel::Trace; }
    if (value == "debug") { return LogLevel::Debug; }
    if (value == "info") { return LogLevel::Info; }
    if (value == "warning" || value == "warn") { return LogLevel::Warning; }
    if (value == "error") { return LogLevel::Error; }
    if (value == "critical" || value == "fatal") { return LogLevel::Critical; }
    if (value == "off") { return LogLevel::Off; }
    throw std::invalid_argument("invalid log level: " + std::string(value));
}

struct TerminalProgress::Impl {
    std::shared_ptr<ProgressAwareStderrSink> sink;
    bool info_enabled = true;
};

TerminalProgress::TerminalProgress(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

TerminalProgress::~TerminalProgress() { clear(); }

bool TerminalProgress::enabled() const noexcept {
    return impl_->info_enabled && impl_->sink->interactive();
}

void TerminalProgress::update(std::string line) {
    if (impl_->info_enabled) { impl_->sink->update(std::move(line)); }
}

void TerminalProgress::clear() noexcept { impl_->sink->clear(); }

struct LoggingRuntime::Impl {
    explicit Impl(LoggingOptions options) {
        const spdlog::level::level_enum level = to_spdlog_level(options.level);
        auto sink = std::make_shared<ProgressAwareStderrSink>(to_spdlog_color_mode(options.color));
        progress  = std::shared_ptr<TerminalProgress>(new TerminalProgress(
            std::make_unique<TerminalProgress::Impl>(sink, level <= spdlog::level::info)));
        logger    = std::make_shared<spdlog::logger>(std::move(options.logger_name), sink);
        logger->set_formatter(std::make_unique<PrettyLogFormatter>(options.presentation));
        logger->set_level(level);
        logger->flush_on(spdlog::level::warn);
        logger->set_error_handler([sink](const std::string& message) noexcept {
            sink->clear();
            report_logging_error(message);
        });
    }

    std::shared_ptr<spdlog::logger> logger;
    std::shared_ptr<TerminalProgress> progress;
};

LoggingRuntime::LoggingRuntime(LoggingOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

LoggingRuntime::~LoggingRuntime() { flush(); }

std::shared_ptr<spdlog::logger> LoggingRuntime::logger() const noexcept { return impl_->logger; }

std::shared_ptr<TerminalProgress> LoggingRuntime::terminal_progress() const noexcept {
    return impl_->progress;
}

void LoggingRuntime::flush() noexcept {
    try {
        impl_->logger->flush();
    } catch (const std::exception& exception) {
        impl_->progress->clear();
        report_logging_error(exception.what());
    } catch (...) {
        impl_->progress->clear();
        report_logging_error("unknown flush error");
    }
}

} // namespace ninfer::product
