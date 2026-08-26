#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>

namespace ninfer::supervisor {

inline bool is_loopback_host(std::string_view host) {
    return host == "127.0.0.1" || host == "::1" || host == "localhost" || host == "localhost.";
}

inline bool is_loopback_peer(std::string_view addr) {
    if (addr.empty()) { return false; }
    if (is_loopback_host(addr)) { return true; }
    // httplib may report IPv4-mapped IPv6.
    return addr == "::ffff:127.0.0.1";
}

struct RestartPolicy {
    int initial_backoff_s     = 1;
    int max_backoff_s         = 60;
    int crash_loop_max        = 5;
    int crash_loop_window_s   = 60;
    int health_fail_threshold = 3;
};

class RestartGate {
public:
    explicit RestartGate(RestartPolicy policy = {}) : policy_(policy), backoff_s_(policy.initial_backoff_s) {}

    // Record an engine exit. Returns false if auto-restart is halted (crash loop).
    bool note_exit(std::chrono::steady_clock::time_point now) {
        if (halted_) { return false; }
        const auto window = std::chrono::seconds(policy_.crash_loop_window_s);
        while (!exits_.empty() && now - exits_.front() > window) { exits_.pop_front(); }
        exits_.push_back(now);
        if (static_cast<int>(exits_.size()) >= policy_.crash_loop_max) {
            halted_ = true;
            return false;
        }
        return true;
    }

    [[nodiscard]] int backoff_seconds() const noexcept { return backoff_s_; }

    void advance_backoff() {
        if (backoff_s_ < policy_.max_backoff_s) {
            const int next = backoff_s_ * 2;
            backoff_s_     = next > policy_.max_backoff_s ? policy_.max_backoff_s : next;
        }
    }

    void note_healthy() {
        backoff_s_ = policy_.initial_backoff_s;
        health_fails_ = 0;
    }

    bool note_health_fail() {
        ++health_fails_;
        return health_fails_ >= policy_.health_fail_threshold;
    }

    void clear_health_fails() { health_fails_ = 0; }

    void reset_halt() {
        halted_ = false;
        exits_.clear();
        backoff_s_    = policy_.initial_backoff_s;
        health_fails_ = 0;
    }

    [[nodiscard]] bool halted() const noexcept { return halted_; }
    [[nodiscard]] int recent_exits() const noexcept { return static_cast<int>(exits_.size()); }
    [[nodiscard]] int health_fails() const noexcept { return health_fails_; }
    [[nodiscard]] const RestartPolicy& policy() const noexcept { return policy_; }

private:
    RestartPolicy policy_;
    std::deque<std::chrono::steady_clock::time_point> exits_;
    int backoff_s_    = 1;
    int health_fails_ = 0;
    bool halted_      = false;
};

} // namespace ninfer::supervisor
