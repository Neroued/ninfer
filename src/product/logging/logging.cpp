#include "product/logging/logging.h"

#include <spdlog/logger.h>
#include <spdlog/sinks/sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <unistd.h>

#include <atomic>
#include <array>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::product {
namespace {

constexpr const char* kOperationalPattern = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v";

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

std::string quote_log_value(std::string_view value) {
    constexpr std::array<char, 16> hex = {'0', '1', '2', '3', '4', '5', '6', '7',
                                          '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (ch < 0x20 || ch == 0x7f) {
                out += "\\x";
                out.push_back(hex[ch >> 4]);
                out.push_back(hex[ch & 0x0f]);
            } else {
                out.push_back(static_cast<char>(ch));
            }
            break;
        }
    }
    out.push_back('"');
    return out;
}

struct TerminalProgress::Impl {
    std::shared_ptr<ProgressAwareStderrSink> sink;
};

TerminalProgress::TerminalProgress(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

TerminalProgress::~TerminalProgress() { clear(); }

bool TerminalProgress::enabled() const noexcept { return impl_->sink->interactive(); }

void TerminalProgress::update(std::string line) { impl_->sink->update(std::move(line)); }

void TerminalProgress::clear() noexcept { impl_->sink->clear(); }

struct LoggingRuntime::Impl {
    explicit Impl(LoggingOptions options) {
        auto sink = std::make_shared<ProgressAwareStderrSink>(to_spdlog_color_mode(options.color));
        progress  = std::shared_ptr<TerminalProgress>(
            new TerminalProgress(std::make_unique<TerminalProgress::Impl>(sink)));
        logger = std::make_shared<spdlog::logger>(std::move(options.logger_name), sink);
        logger->set_pattern(kOperationalPattern);
        logger->set_level(to_spdlog_level(options.level));
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
