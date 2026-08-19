#pragma once

// Minimal stderr logger for host-KV cache events.
//
// The engine and program layers sit below serve, so they cannot use the
// serve-layer console_log facility (dependency direction: serve -> engine).
// The prefix mimics the console_log format for grep-ability.

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string_view>

namespace ninfer {

namespace host_kv_log_detail {

inline std::mutex& mutex() {
    static std::mutex mutex;
    return mutex;
}

inline std::string prefix() {
    const auto now    = std::chrono::system_clock::now();
    const auto epoch  = now.time_since_epoch();
    const auto whole  = std::chrono::duration_cast<std::chrono::seconds>(epoch);
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(epoch - whole).count();
    const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_r(&seconds, &local);
    std::ostringstream out;
    out << '[' << std::put_time(&local, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0')
        << std::setw(3) << millis << "] [info] ninfer-host-kv: ";
    return out.str();
}

} // namespace host_kv_log_detail

inline void host_kv_log(std::string_view message) {
    std::lock_guard<std::mutex> lock(host_kv_log_detail::mutex());
    std::cerr << host_kv_log_detail::prefix() << message << '\n';
}

} // namespace ninfer
