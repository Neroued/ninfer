#include "logic.hpp"
#include "config.hpp"

#include <iostream>
#include <string>

namespace {

int fail(const std::string& m) {
    std::cerr << "FAIL: " << m << '\n';
    return 1;
}
int check(bool c, const std::string& m) { return c ? 0 : fail(m); }

int test_loopback() {
    using namespace ninfer::supervisor;
    int f = 0;
    f += check(is_loopback_host("127.0.0.1") && is_loopback_host("localhost") &&
                   is_loopback_host("::1"),
               "loopback hosts");
    f += check(!is_loopback_host("0.0.0.0") && !is_loopback_host("192.168.1.2"),
               "non-loopback hosts");
    f += check(is_loopback_peer("127.0.0.1") && is_loopback_peer("::ffff:127.0.0.1"),
               "loopback peers");
    f += check(!is_loopback_peer("10.0.0.8") && !is_loopback_peer(""), "off-box peers");
    return f;
}

int test_crash_loop() {
    using clock = std::chrono::steady_clock;
    ninfer::supervisor::RestartPolicy p;
    p.crash_loop_max      = 3;
    p.crash_loop_window_s = 60;
    ninfer::supervisor::RestartGate g(p);
    const auto t0 = clock::now();
    int f         = 0;
    f += check(g.note_exit(t0) && g.note_exit(t0 + std::chrono::seconds(1)), "first exits allowed");
    f += check(!g.note_exit(t0 + std::chrono::seconds(2)) && g.halted(),
               "third exit in window must halt");
    g.reset_halt();
    f += check(!g.halted() && g.note_exit(t0 + std::chrono::seconds(120)),
               "reset allows restart");
    return f;
}

int test_backoff() {
    ninfer::supervisor::RestartGate g;
    int f = 0;
    f += check(g.backoff_seconds() == 1, "initial backoff 1s");
    g.advance_backoff();
    f += check(g.backoff_seconds() == 2, "backoff 2s");
    g.advance_backoff();
    g.advance_backoff();
    g.advance_backoff();
    g.advance_backoff();
    g.advance_backoff();
    f += check(g.backoff_seconds() == 60, "backoff caps at 60s");
    g.note_healthy();
    f += check(g.backoff_seconds() == 1, "healthy resets backoff");
    return f;
}

int test_config_bind() {
    int f = 0;
    const char* ok =
        R"({"engine":{"executable":"C:/ninfer-serve.exe"},"supervisor":{"host":"127.0.0.1"}})";
    try {
        const auto c = ninfer::supervisor::load_config_json(ok);
        f += check(c.host == "127.0.0.1" && !c.bind_any, "loopback config");
    } catch (...) { f += fail("loopback config threw"); }
    bool rejected = false;
    try {
        (void)ninfer::supervisor::load_config_json(
            R"({"engine":{"executable":"x"},"supervisor":{"host":"0.0.0.0"}})");
    } catch (const std::invalid_argument&) { rejected = true; }
    f += check(rejected, "0.0.0.0 without bind_any must be rejected");
    bool any_ok = false;
    try {
        const auto c = ninfer::supervisor::load_config_json(
            R"({"engine":{"executable":"x"},"supervisor":{"host":"0.0.0.0","bind_any":true}})");
        any_ok = c.bind_any;
    } catch (...) {}
    f += check(any_ok, "bind_any allows 0.0.0.0");
    return f;
}

int test_health_threshold() {
    ninfer::supervisor::RestartPolicy p;
    p.health_fail_threshold = 3;
    ninfer::supervisor::RestartGate g(p);
    int f = 0;
    f += check(!g.note_health_fail() && !g.note_health_fail(), "below threshold");
    f += check(g.note_health_fail(), "threshold trips restart");
    g.note_healthy();
    f += check(!g.note_health_fail(), "healthy clears fail count");
    return f;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_loopback();
    failures += test_crash_loop();
    failures += test_backoff();
    failures += test_config_bind();
    failures += test_health_threshold();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
