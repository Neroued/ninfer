#pragma once

#include "config.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

namespace ninfer::supervisor {

enum class EngineState : std::uint8_t {
    Stopped,
    Starting,
    Running,
    Stopping,
    BackingOff,
    Halted,
};

struct EngineStatus {
    EngineState state          = EngineState::Stopped;
    std::uint64_t pid          = 0;
    std::int64_t started_unix_ms = 0;
    int restart_count          = 0;
    int last_exit_code         = 0;
    bool crash_loop_halted     = false;
    std::string last_event;
    std::string health; // ok / unhealthy / unreachable
    int health_fails           = 0;
};

class EngineChild {
public:
    explicit EngineChild(SupervisorConfig cfg);
    ~EngineChild();

    EngineChild(const EngineChild&)            = delete;
    EngineChild& operator=(const EngineChild&) = delete;

    void start();
    void stop();
    void restart();
    void request_quit();
    void observe_health(int http_status);

    [[nodiscard]] EngineStatus status() const;
    [[nodiscard]] std::string log_tail(std::size_t max_bytes) const;
    [[nodiscard]] const SupervisorConfig& config() const noexcept { return cfg_; }

    void run_loop();

private:
    void spawn();
    void capture_wait();
    void append_log(const char* data, std::size_t n);
    void rotate_logs_if_needed();

    SupervisorConfig cfg_;
    RestartGate gate_;
    mutable std::mutex mu_;
    EngineStatus st_;
    std::atomic<bool> stop_child_{false};
    std::atomic<bool> quit_{false};
    std::atomic<bool> auto_restart_{true};
    void* process_handle_ = nullptr; // HANDLE
    void* job_handle_     = nullptr;
    std::string log_path_;
};

} // namespace ninfer::supervisor
