#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <vector>

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

inline constexpr std::string_view kSupervisorControlHeader      = "X-NInfer-Supervisor";
inline constexpr std::string_view kSupervisorControlHeaderValue = "1";

inline std::string_view trim_sv(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' ||
                          s.front() == '\n')) {
        s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' ||
                          s.back() == '\n')) {
        s.remove_suffix(1);
    }
    return s;
}

// Split Host into name and optional port. IPv6 literals must be bracketed when a
// port is present (`[::1]:8099`).
inline bool split_host_header(std::string_view host, std::string& name, int& port, bool& has_port) {
    host     = trim_sv(host);
    name.clear();
    port     = 0;
    has_port = false;
    if (host.empty()) { return false; }
    if (host.front() == '[') {
        const auto rb = host.find(']');
        if (rb == std::string_view::npos) { return false; }
        name = std::string(host.substr(0, rb + 1));
        if (rb + 1 == host.size()) { return true; }
        if (host[rb + 1] != ':') { return false; }
        const auto p = host.substr(rb + 2);
        if (p.empty()) { return false; }
        int value = 0;
        for (char c : p) {
            if (c < '0' || c > '9') { return false; }
            value = value * 10 + (c - '0');
            if (value > 65535) { return false; }
        }
        port     = value;
        has_port = true;
        return true;
    }
    const auto colon = host.rfind(':');
    if (colon != std::string_view::npos && host.find(':') == colon) {
        name = std::string(host.substr(0, colon));
        const auto p = host.substr(colon + 1);
        if (p.empty() || name.empty()) { return false; }
        int value = 0;
        for (char c : p) {
            if (c < '0' || c > '9') { return false; }
            value = value * 10 + (c - '0');
            if (value > 65535) { return false; }
        }
        port     = value;
        has_port = true;
        return true;
    }
    name = std::string(host);
    return !name.empty();
}

inline bool is_loopback_host_name(std::string_view name) {
    return is_loopback_host(name) || name == "[::1]";
}

// DNS-rebinding defense: only the listen port's loopback names, plus the
// configured bind host when --bind-any names a specific interface. Binding
// 0.0.0.0 does not open the Host allowlist.
inline bool host_header_allowed(std::string_view host_header, int listen_port,
                                std::string_view bind_host, bool bind_any) {
    std::string name;
    int port         = 0;
    bool has_port    = false;
    if (!split_host_header(host_header, name, port, has_port)) { return false; }
    if (has_port && port != listen_port) { return false; }
    if (is_loopback_host_name(name)) { return true; }
    if (!bind_any) { return false; }
    if (bind_host.empty() || bind_host == "0.0.0.0" || bind_host == "::" || bind_host == "[::]") {
        return false;
    }
    return name == bind_host;
}

inline bool supervisor_control_header_ok(std::string_view value) {
    return trim_sv(value) == kSupervisorControlHeaderValue;
}

struct NvidiaSmiMemory {
    bool ok                 = false;
    int index               = -1;
    std::uint64_t used_mib  = 0;
    std::uint64_t total_mib = 0;
    std::string error;
};

// Parses `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits`.
// Values are mebibytes. Picks the row whose index equals `device`.
inline NvidiaSmiMemory parse_nvidia_smi_memory_csv(std::string_view csv, int device) {
    NvidiaSmiMemory out;
    std::string_view rest = csv;
    bool saw_row          = false;
    while (!rest.empty()) {
        auto nl    = rest.find_first_of("\n\r");
        auto line  = trim_sv(nl == std::string_view::npos ? rest : rest.substr(0, nl));
        rest       = nl == std::string_view::npos ? std::string_view{}
                                                  : rest.substr(nl + 1);
        if (line.empty()) { continue; }
        saw_row = true;
        const auto c1 = line.find(',');
        if (c1 == std::string_view::npos) { continue; }
        const auto c2 = line.find(',', c1 + 1);
        if (c2 == std::string_view::npos) { continue; }
        const auto idx_s  = trim_sv(line.substr(0, c1));
        const auto used_s = trim_sv(line.substr(c1 + 1, c2 - c1 - 1));
        const auto tot_s  = trim_sv(line.substr(c2 + 1));
        int idx           = 0;
        std::uint64_t used = 0;
        std::uint64_t tot  = 0;
        try {
            idx  = std::stoi(std::string(idx_s));
            used = std::stoull(std::string(used_s));
            tot  = std::stoull(std::string(tot_s));
        } catch (...) { continue; }
        if (idx != device) { continue; }
        out.ok       = true;
        out.index    = idx;
        out.used_mib = used;
        out.total_mib = tot;
        return out;
    }
    out.error = saw_row ? "nvidia-smi csv has no row for the configured device"
                        : "nvidia-smi csv is empty";
    return out;
}

inline std::uint64_t mib_to_bytes(std::uint64_t mib) { return mib * 1024ull * 1024ull; }

// Pre-filter that agrees with the engine JSONL schema: the field is "event",
// not "type". A substring on the value without the key name is how a panel
// can look populated while every record is then discarded.
inline bool jsonl_event_is(std::string_view line, std::string_view event) {
    const std::string compact = std::string("\"event\":\"") + std::string(event) + "\"";
    const std::string spaced  = std::string("\"event\": \"") + std::string(event) + "\"";
    return line.find(compact) != std::string_view::npos ||
           line.find(spaced) != std::string_view::npos;
}

struct VramSample {
    std::int64_t t_ms                = 0;
    std::uint64_t budget_bytes       = 0;
    std::uint64_t nvidia_used_bytes  = 0;
};

struct VramSeriesEvent {
    std::int64_t t_ms = 0;
    std::string kind;
    std::string label;
};

// Ring of raw 10 Hz samples. No averaging. Oldest is dropped on overflow.
struct VramSeriesRing {
    explicit VramSeriesRing(std::size_t cap = 6000) : cap_(cap), buf_(cap) {}

    void push(VramSample s) {
        if (cap_ == 0) { return; }
        buf_[head_] = s;
        head_       = (head_ + 1) % cap_;
        if (size_ < cap_) { ++size_; }
    }

    void push_event(VramSeriesEvent e, std::size_t event_cap = 128) {
        events_.push_back(std::move(e));
        while (events_.size() > event_cap) { events_.pop_front(); }
    }

    [[nodiscard]] std::vector<VramSample> samples() const {
        std::vector<VramSample> out;
        out.reserve(size_);
        const std::size_t start = size_ < cap_ ? 0 : head_;
        for (std::size_t i = 0; i < size_; ++i) { out.push_back(buf_[(start + i) % cap_]); }
        return out;
    }

    [[nodiscard]] std::vector<VramSeriesEvent> events() const {
        return {events_.begin(), events_.end()};
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    std::size_t cap_  = 0;
    std::size_t head_ = 0;
    std::size_t size_ = 0;
    std::vector<VramSample> buf_;
    std::deque<VramSeriesEvent> events_;
};

inline std::string extract_kv_capacity_line(std::string_view log) {
    const auto key = std::string_view("KV capacity ");
    const auto pos = log.rfind(key);
    if (pos == std::string_view::npos) { return {}; }
    auto start = log.find_last_of("\n", pos);
    start      = start == std::string_view::npos ? 0 : start + 1;
    auto end   = log.find('\n', pos);
    auto line  = log.substr(start, end == std::string_view::npos ? log.size() - start : end - start);
    if (!line.empty() && line.back() == '\r') { line.remove_suffix(1); }
    return std::string(trim_sv(line));
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
