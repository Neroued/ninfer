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

int test_host_header() {
    using namespace ninfer::supervisor;
    int f = 0;
    f += check(host_header_allowed("127.0.0.1:8099", 8099, "127.0.0.1", false),
               "loopback ipv4 host");
    f += check(host_header_allowed("localhost:8099", 8099, "127.0.0.1", false),
               "localhost host");
    f += check(host_header_allowed("[::1]:8099", 8099, "127.0.0.1", false), "ipv6 loopback host");
    f += check(host_header_allowed("127.0.0.1", 8099, "127.0.0.1", false),
               "loopback host without port");
    f += check(!host_header_allowed("attacker.example", 8099, "127.0.0.1", false),
               "rebinding host rejected");
    f += check(!host_header_allowed("attacker.example:8099", 8099, "127.0.0.1", false),
               "rebinding host:port rejected");
    f += check(!host_header_allowed("127.0.0.1:8080", 8099, "127.0.0.1", false),
               "wrong port rejected");
    f += check(!host_header_allowed("", 8099, "127.0.0.1", false), "empty host rejected");
    f += check(!host_header_allowed("192.168.1.5:8099", 8099, "0.0.0.0", true),
               "bind-any 0.0.0.0 does not open Host allowlist");
    f += check(host_header_allowed("192.168.1.5:8099", 8099, "192.168.1.5", true),
               "bind-any named host is allowed");
    f += check(!host_header_allowed("192.168.1.5:8099", 8099, "192.168.1.5", false),
               "named host without bind_any rejected");
    // Suffix/prefix traps. These pass a naive substring or starts_with check and
    // are the classic way a rebinding defense gets reintroduced as a bug: an
    // attacker controls the whole label, so "localhost.evil.com" is evil.com.
    f += check(!host_header_allowed("localhost.evil.com:8099", 8099, "127.0.0.1", false),
               "localhost-prefixed attacker domain rejected");
    f += check(!host_header_allowed("127.0.0.1.evil.com:8099", 8099, "127.0.0.1", false),
               "ip-prefixed attacker domain rejected");
    f += check(!host_header_allowed("evil-localhost:8099", 8099, "127.0.0.1", false),
               "localhost-suffixed attacker domain rejected");
    f += check(!host_header_allowed("192.168.1.5.evil.com:8099", 8099, "192.168.1.5", true),
               "bind-any named host is matched exactly, not as a prefix");
    f += check(supervisor_control_header_ok("1") && !supervisor_control_header_ok("") &&
                   !supervisor_control_header_ok("true"),
               "control header is exactly 1");
    return f;
}

int test_nvidia_csv() {
    using namespace ninfer::supervisor;
    int f     = 0;
    const auto a = parse_nvidia_smi_memory_csv("0, 24576, 32607\n1, 10, 20\n", 0);
    f += check(a.ok && a.used_mib == 24576 && a.total_mib == 32607, "device 0 csv");
    const auto b = parse_nvidia_smi_memory_csv("0, 1, 2\n1, 99, 100\n", 1);
    f += check(b.ok && b.used_mib == 99 && b.total_mib == 100, "device 1 csv");
    const auto c = parse_nvidia_smi_memory_csv("0, 1, 2\n", 3);
    f += check(!c.ok && !c.error.empty(), "missing device");
    const auto d = parse_nvidia_smi_memory_csv("", 0);
    f += check(!d.ok, "empty csv");
    f += check(mib_to_bytes(1) == 1048576, "mib_to_bytes");
    return f;
}

int test_kv_line() {
    using namespace ninfer::supervisor;
    int f = 0;
    const char* log =
        "[info] ninfer-serve: model loaded in 1.2 s\n"
        "[info] ninfer-serve: KV capacity auto resolved=8192 tokens pages=1/2 "
        "runtime=1 prefix-cache=2 free-after-weights=3 free-after-startup=4 "
        "headroom=5 slack=6 graphs=7/8\n"
        "later line\n";
    const auto line = extract_kv_capacity_line(log);
    f += check(line.find("KV capacity auto resolved=8192") != std::string::npos,
               "extracts last KV capacity line");
    f += check(extract_kv_capacity_line("no capacity here").empty(), "missing line");
    return f;
}

int test_monitor_only_config() {
    int f = 0;
    try {
        const auto c = ninfer::supervisor::load_config_json(
            R"({"engine":{"unmanaged":true,"engine_port":8010},"supervisor":{"host":"127.0.0.1"}})");
        f += check(!ninfer::supervisor::manages_engine_process(c) && c.engine.unmanaged,
                   "unmanaged does not require executable");
    } catch (...) { f += fail("unmanaged config threw"); }
    try {
        const auto c = ninfer::supervisor::load_config_json(
            R"({"engine":{"engine_port":8010},"supervisor":{"host":"127.0.0.1"}})", true);
        f += check(c.monitor_only && !ninfer::supervisor::manages_engine_process(c),
                   "CLI monitor_only does not require executable");
    } catch (...) { f += fail("monitor_only cli config threw"); }
    bool rejected = false;
    try {
        (void)ninfer::supervisor::load_config_json(
            R"({"engine":{},"supervisor":{"host":"127.0.0.1"}})");
    } catch (const std::invalid_argument&) { rejected = true; }
    f += check(rejected, "managed config still requires executable");
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
    failures += test_host_header();
    failures += test_nvidia_csv();
    failures += test_kv_line();
    failures += test_monitor_only_config();
    failures += test_health_threshold();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
