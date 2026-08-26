#include "logic.hpp"
#include "config.hpp"
#include "insights.hpp"

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

int test_insights_honesty() {
    using namespace ninfer::supervisor;
    int f = 0;
    const auto missing = insights_from_request_log_path("");
    f += check(missing.at("source").at("request_log") == "unconfigured", "unconfigured source");
    f += check(missing.at("insights").size() >= 1 &&
                   missing.at("insights").at(0).at("availability") == "unavailable",
               "missing log is unavailable, not a zero");
    f += check(missing.at("insights").at(0).at("statement").get<std::string>().find(
                   "no request_done records") != std::string::npos,
               "unavailable statement");

    const auto typed = analyze_request_log_jsonl(
        R"({"type":"request_done","timestamp_unix_ms":1})"
        "\n",
        "mem");
    f += check(typed.at("insights").at(0).at("availability") == "unavailable",
               "type-key records do not count as request_done");

    const char* jsonl =
        R"({"event":"request_start","server_instance_id":"a","timestamp_unix_ms":1000,"request":{"request_id":1,"enable_thinking":true,"tool_count":0,"requested_output_tokens":8}})"
        "\n"
        R"({"event":"request_done","server_instance_id":"a","timestamp_unix_ms":1600,"request":{"request_id":1,"enable_thinking":true,"tool_count":0,"requested_output_tokens":8},"result":{"finish_reason":"output_limit","completion_tokens":8},"timings_seconds":{"prepare":0.01,"prefill":0.02,"decode":0.02,"ttft":0.03,"total":0.1}})"
        "\n"
        R"({"event":"request_start","server_instance_id":"b","timestamp_unix_ms":2000,"request":{"request_id":1,"enable_thinking":false,"tool_count":0,"requested_output_tokens":16}})"
        "\n"
        R"({"event":"throughput","server_instance_id":"a","timestamp_unix_ms":1601,"scheduler":{"waiting":2,"running":1,"prefilling":0}})"
        "\n";
    const auto r = analyze_request_log_jsonl(jsonl, "mem");
    bool saw_sat = false, saw_limit = false, saw_content = false, saw_ttft = false;
    for (const auto& it : r.at("insights")) {
        const auto id = it.at("id").get<std::string>();
        if (id.find("latency.") == 0 && id.find("ttft") == std::string::npos) {
            saw_sat = true;
            f += check(it.at("availability") == "available" && it.at("confidence") == "measured",
                       "saturation is measured");
            f += check(it.at("measured_over").at("requests") == 1, "measured_over.requests is 1");
            f += check(it.at("evidence").at("queued") == 1, "0.5s queue wait classifies queued");
        }
        if (id.find("ttft") != std::string::npos) {
            saw_ttft = true;
            f += check(it.at("availability") == "available", "ttft split is measured");
            f += check(it.at("evidence").contains("mean_prepare_s") &&
                           it.at("evidence").contains("mean_prefill_s"),
                       "ttft evidence has the split");
        }
        if (id == "client.output_limit_while_thinking") {
            saw_limit = true;
            f += check(it.at("evidence").at("output_limit_thinking") == 1, "output_limit counted");
            f += check(it.at("evidence").at("sample_request_ids").at(0) == 1, "request_id evidence");
        }
        if (id == "client.content_fields") {
            saw_content = true;
            f += check(it.at("availability") == "unavailable",
                       "content fields unavailable, not fabricated");
            f += check(it.at("measured_over").at("requests") == 1,
                       "unavailable measured_over is examined count, not 0");
            f += check(!it.contains("recommendation"), "empty recommendation is omitted");
        }
    }
    f += check(saw_sat && saw_limit && saw_content && saw_ttft, "required insight ids present");
    return f;
}

int test_admin_vram_markers() {
    using namespace ninfer::supervisor;
    AdminVramCursor c;
    std::string kind;
    int f = 0;
    f += check(!c.observe("", "", kind), "first poll is baseline, does not fire");
    f += check(c.observe("release", "seed store released", kind) && kind == "vram_release",
               "empty -> release fires vram_release");
    f += check(c.observe("reclaim", "seed store reclaimed", kind) && kind == "vram_reclaim",
               "release -> reclaim fires vram_reclaim");
    f += check(!c.observe("reclaim", "seed store reclaimed", kind), "unchanged does not fire");
    return f;
}

int test_insights_pinned_tier() {
    using namespace ninfer::supervisor;
    nlohmann::json report = {{"insights", nlohmann::json::array()}};
    nlohmann::json admin  = {
        {"last_transition", ""},
        {"last_reason", ""},
        {"tiers",
         nlohmann::json::array(
             {{{"name", "seed"},
               {"min_bytes", 4294967296ull},
               {"max_bytes", 4294967296ull},
               {"reclaimable_bytes", 0},
               {"released", false}}})}};
    append_admin_vram_insights(report, admin, "");
    int f     = 0;
    bool saw  = false;
    for (const auto& it : report.at("insights")) {
        if (it.at("id") == "vram.tier_pinned_unreleasable") {
            saw = true;
            f += check(it.at("severity") == "warning", "pinned tier is warning");
            f += check(it.at("availability") == "available", "config trap is measured");
        }
    }
    f += check(saw, "pinned min==max insight present");

    nlohmann::json elastic_report = {{"insights", nlohmann::json::array()}};
    nlohmann::json elastic        = {
        {"last_transition", ""},
        {"last_reason", ""},
        {"tiers",
         nlohmann::json::array(
             {{{"name", "seed"},
               {"min_bytes", 0},
               {"max_bytes", 4294967296ull},
               {"reclaimable_bytes", 4294967296ull},
               {"released", false}}})}};
    append_admin_vram_insights(elastic_report, elastic, "");
    bool elastic_fired = false;
    for (const auto& it : elastic_report.at("insights")) {
        if (it.at("id") == "vram.tier_pinned_unreleasable") { elastic_fired = true; }
    }
    f += check(!elastic_fired, "elastic min!=max does not fire pinned insight");
    return f;
}

int test_insights_prefix() {
    using namespace ninfer::supervisor;
    const char* jsonl =
        R"({"event":"request_start","server_instance_id":"a","timestamp_unix_ms":1,"request":{"request_id":1,"message_count":1}})"
        "\n"
        R"({"event":"request_done","server_instance_id":"a","timestamp_unix_ms":2,"request":{"request_id":1,"message_count":1},"result":{"prefix_reuse_path":"full_reset","prompt_tokens":100,"prefix_cache_hit_tokens":0},"timings_seconds":{"total":0.05,"prepare":0.01,"prefill":0.02,"decode":0.02,"ttft":0.03,"vision":0}})"
        "\n"
        R"({"event":"request_start","server_instance_id":"a","timestamp_unix_ms":3,"request":{"request_id":2,"message_count":5}})"
        "\n"
        R"({"event":"request_done","server_instance_id":"a","timestamp_unix_ms":4,"request":{"request_id":2,"message_count":5},"result":{"prefix_reuse_path":"full_reset","prompt_tokens":7749,"prefix_cache_hit_tokens":0},"timings_seconds":{"total":0.05,"prepare":0.01,"prefill":0.02,"decode":0.02,"ttft":0.03,"vision":0}})"
        "\n";
    const auto r = analyze_request_log_jsonl(jsonl, "mem");
    int f = 0;
    bool mix = false, miss = false;
    for (const auto& it : r.at("insights")) {
        const auto id = it.at("id").get<std::string>();
        if (id == "prefix.reuse_mix") {
            mix = true;
            f += check(it.at("evidence").at("full_reset_single_turn") == 1, "single-turn reset");
            f += check(it.at("evidence").at("full_reset_multi_turn") == 1, "multi-turn reset");
            f += check(it.at("availability") == "available", "mix is measured");
        }
        if (id == "prefix.multiturn_full_reset") {
            miss = true;
            f += check(it.at("severity") == "warning", "multi-turn reset is a warning");
            f += check(it.at("evidence").at("samples").at(0).at("prompt_tokens") == 7749,
                       "live-shaped sample");
            f += check(it.at("measured_over").at("requests") == 2, "examined both dones");
        }
    }
    f += check(mix && miss, "prefix insights present");
    return f;
}

int test_jsonl_event_key() {
    using namespace ninfer::supervisor;
    int f = 0;
    f += check(jsonl_event_is(R"({"event":"request_done","result":{}})", "request_done"),
               "compact event key");
    f += check(jsonl_event_is(R"({"event": "request_done"})", "request_done"), "spaced event key");
    f += check(!jsonl_event_is(R"({"type":"request_done"})", "request_done"),
               "type key is not the event key");
    f += check(!jsonl_event_is(R"({"event":"server_start"})", "request_done"), "other event");
    return f;
}

int test_series_persist() {
    using namespace ninfer::supervisor;
    int f = 0;
    VramSample s{};
    f += check(parse_series_sample_line(R"({"t_ms":10,"budget_bytes":20,"nvidia_used_bytes":30})", s) &&
                   s.t_ms == 10 && s.budget_bytes == 20 && s.nvidia_used_bytes == 30,
               "parse sample");
    VramSeriesEvent e{};
    f += check(parse_series_event_line(R"({"t_ms":11,"kind":"vram_release","label":"seed store released"})", e) &&
                   e.kind == "vram_release",
               "parse event");
    VramSeriesRing r(8);
    r.load_jsonl("{\"t_ms\":1,\"budget_bytes\":2,\"nvidia_used_bytes\":3}\n"
                 "{\"t_ms\":4,\"kind\":\"engine_up\",\"label\":\"health 200\"}\n");
    f += check(r.size() == 1 && r.events().size() == 1 && r.events()[0].kind == "engine_up",
               "load_jsonl restores samples and events");
    return f;
}

int test_series_ring() {
    ninfer::supervisor::VramSeriesRing r(3);
    int f = 0;
    r.push({1, 10, 4});
    r.push({2, 20, 5});
    r.push({3, 30, 6});
    r.push({4, 40, 7});
    const auto s = r.samples();
    f += check(s.size() == 3 && s[0].t_ms == 2 && s[2].t_ms == 4 && s[2].budget_bytes == 40,
               "ring drops oldest, keeps raw values");
    r.push_event({4, "admin_vram", "release"}, 2);
    r.push_event({5, "engine_down", "health 0"}, 2);
    r.push_event({6, "engine_up", "health 200"}, 2);
    const auto e = r.events();
    f += check(e.size() == 2 && e[0].kind == "engine_down" && e[1].kind == "engine_up",
               "event ring cap");
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
    failures += test_insights_honesty();
    failures += test_insights_prefix();
    failures += test_admin_vram_markers();
    failures += test_insights_pinned_tier();
    failures += test_jsonl_event_key();
    failures += test_series_persist();
    failures += test_series_ring();
    failures += test_health_threshold();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
