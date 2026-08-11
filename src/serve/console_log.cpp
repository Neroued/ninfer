#include "serve/console_log.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace ninfer::serve {
namespace {

const char* level_name(ConsoleLogLevel level) noexcept {
    switch (level) {
    case ConsoleLogLevel::Info:
        return "info";
    case ConsoleLogLevel::Warning:
        return "warning";
    case ConsoleLogLevel::Error:
        return "error";
    }
    return "info";
}

std::mutex& console_mutex() {
    static std::mutex mutex;
    return mutex;
}

} // namespace

std::string format_console_log_prefix(std::chrono::system_clock::time_point timestamp,
                                      ConsoleLogLevel level) {
    const auto since_epoch   = timestamp.time_since_epoch();
    const auto whole_seconds = std::chrono::duration_cast<std::chrono::seconds>(since_epoch);
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(since_epoch - whole_seconds).count();
    const std::time_t wall_seconds =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::time_point(whole_seconds));
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &wall_seconds);
#else
    localtime_r(&wall_seconds, &local);
#endif

    std::ostringstream out;
    out << '[' << std::put_time(&local, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0')
        << std::setw(3) << milliseconds << "] [" << level_name(level) << "] ninfer-serve: ";
    return out.str();
}

std::string current_console_log_prefix(ConsoleLogLevel level) {
    return format_console_log_prefix(std::chrono::system_clock::now(), level);
}

void write_console_log(ConsoleLogLevel level, std::string_view message) {
    std::lock_guard lock(console_mutex());
    std::cerr << current_console_log_prefix(level) << message << '\n';
}

} // namespace ninfer::serve
