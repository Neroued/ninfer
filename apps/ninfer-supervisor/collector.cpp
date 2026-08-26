#include "collector.hpp"
#include "insights.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#define CPPHTTPLIB_NO_EXCEPTIONS
#include <httplib.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

namespace ninfer::supervisor {
namespace {

std::string load_key(const std::string& path) {
    try {
        return read_api_key(path);
    } catch (...) { return {}; }
}

httplib::Client engine_client(const EngineSpec& spec) {
    httplib::Client cli(spec.engine_host, spec.engine_port);
    cli.set_connection_timeout(1, 0);
    cli.set_read_timeout(2, 0);
    const std::string key = load_key(spec.api_key_file);
    if (!key.empty()) { cli.set_bearer_token_auth(key); }
    return cli;
}

} // namespace

void Collector::poll_health(Collected& out) {
    auto cli = engine_client(spec_);
    if (auto res = cli.Get("/health")) {
        out.health_status = res->status;
        out.health_body   = res->body;
    } else {
        out.health_status = 0;
        out.health_body   = "unreachable";
    }
}

void Collector::poll_admin(Collected& out) {
    auto cli = engine_client(spec_);
    if (auto res = cli.Get("/admin/vram")) {
        if (res->status == 200) {
            try {
                out.admin_vram = nlohmann::json::parse(res->body);
            } catch (...) {
                out.admin_vram_note = "admin/vram returned unreadable JSON";
            }
        } else if (res->status == 401 || res->status == 403) {
            out.admin_vram_note = "admin VRAM unavailable (enable --admin-vram and --api-key)";
        } else if (res->status == 404) {
            out.admin_vram_note = "admin VRAM not registered on this engine";
        } else {
            out.admin_vram_note = "admin/vram HTTP " + std::to_string(res->status);
        }
    } else {
        out.admin_vram_note = "engine unreachable for admin/vram";
    }
}

void Collector::poll_nvidia_smi(Collected& out) {
    FILE* pipe = _popen(
        "nvidia-smi --query-gpu=index,memory.used,memory.total "
        "--format=csv,noheader,nounits",
        "rt");
    if (pipe == nullptr) {
        out.nvidia.error = "nvidia-smi not found";
        return;
    }
    std::string csv;
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe) != nullptr) { csv += buf; }
    const int rc = _pclose(pipe);
    if (rc != 0 && csv.empty()) {
        out.nvidia.error = "nvidia-smi exited " + std::to_string(rc);
        return;
    }
    out.nvidia = parse_nvidia_smi_memory_csv(csv, spec_.device);
}

void Collector::poll_request_log(Collected& out) {
    if (spec_.request_log.empty()) {
        out.requests.log_error = "request log path not configured";
        return;
    }
    std::ifstream in(spec_.request_log);
    if (!in) {
        out.requests.log_error = "request log not present";
        return;
    }
    out.requests.log_available = true;
    std::vector<std::string> lines;
    std::string last_start;
    std::string line;
    while (std::getline(in, line)) {
        if (jsonl_event_is(line, "request_done")) { lines.push_back(line); }
        if (jsonl_event_is(line, "server_start")) { last_start = std::move(line); }
    }
    if (!last_start.empty()) {
        try {
            const auto j = nlohmann::json::parse(last_start);
            const auto& eng = j.at("engine");
            const auto& mem = j.at("memory");
            auto gib = [](const nlohmann::json& obj, const char* key) {
                const auto n = obj.value(key, std::uint64_t{0});
                return std::to_string(n / 1048576) + " MiB";
            };
            out.engine_capacity_line =
                std::string("KV capacity ") + eng.value("kv_capacity_mode", std::string("?")) +
                " resolved=" + std::to_string(eng.value("kv_capacity", 0)) +
                " tokens pages=" + std::to_string(eng.value("kv_capacity_page_groups", 0)) + "/" +
                std::to_string(eng.value("kv_capacity_max_page_groups", 0)) +
                " runtime=" + gib(mem, "runtime_reservation_bytes") +
                " prefix-cache=" + gib(mem, "prefix_cache_bytes") +
                " free-after-weights=" + gib(mem, "available_after_weights_bytes") +
                " free-after-startup=" + gib(mem, "available_after_startup_bytes") +
                " headroom=" + gib(mem, "kv_capacity_headroom_bytes") +
                " slack=" + gib(mem, "planned_slack_bytes") +
                " graphs=" + gib(mem, "cuda_graph_observed_bytes") + "/" +
                gib(mem, "cuda_graph_allowance_bytes") + " (from request-log server_start)";
        } catch (...) {}
    }
    const std::size_t start = lines.size() > 32 ? lines.size() - 32 : 0;
    double ttft_sum = 0;
    double decode_sum = 0;
    int n_ttft = 0;
    int n_dec  = 0;
    for (std::size_t i = start; i < lines.size(); ++i) {
        try {
            const auto j = nlohmann::json::parse(lines[i]);
            // The engine writes {"event":"request_done"}, not "type". Reading the wrong
            // key made every record fall through and the panel read a permanent 0.
            if (j.value("event", "") != "request_done") { continue; }
            ++out.requests.done;
            if (j.contains("speculative") && j.at("speculative").is_object()) {
                const auto& sp = j.at("speculative");
                out.requests.mtp_backend      = sp.value("backend", out.requests.mtp_backend);
                out.requests.mtp_draft_window = sp.value("draft_window", out.requests.mtp_draft_window);
                const auto drafted  = sp.value("drafted_tokens", 0);
                const auto accepted = sp.value("accepted_tokens", 0);
                out.requests.mtp_drafted += drafted;
                out.requests.mtp_accepted += accepted;
                out.requests.mtp_fallback_steps += sp.value("fallback_steps", 0);
                out.requests.mtp_rounds += sp.value("rounds", 0);
                if (drafted > 0) {
                    out.requests.mtp_last_accept_rate =
                        static_cast<double>(accepted) / static_cast<double>(drafted);
                }
                if (sp.contains("accepted_per_position") && sp.at("accepted_per_position").is_array()) {
                    const auto& pos = sp.at("accepted_per_position");
                    if (out.requests.mtp_accepted_per_position.size() < pos.size()) {
                        out.requests.mtp_accepted_per_position.resize(pos.size(), 0);
                    }
                    for (std::size_t p = 0; p < pos.size(); ++p) {
                        out.requests.mtp_accepted_per_position[p] += pos.at(p).get<std::uint64_t>();
                    }
                }
            }
            if (j.contains("timings_seconds") && j.at("timings_seconds").contains("ttft")) {
                ttft_sum += j.at("timings_seconds").at("ttft").get<double>() * 1000.0;
                ++n_ttft;
            }
            const auto& result = j.at("result");
            const double dec_s =
                j.contains("timings_seconds") ? j.at("timings_seconds").value("decode", 0.0) : 0.0;
            const int gen = result.value("completion_tokens", 0);
            if (dec_s > 0.0 && gen > 1) {
                decode_sum += static_cast<double>(gen - 1) / dec_s;
                ++n_dec;
            }
            const std::string reuse = result.value("prefix_reuse_path", "");
            out.requests.last_reuse = reuse;
            if (reuse == "full_reset") {
                ++out.requests.reuse_full_reset;
            } else if (reuse.find("append") != std::string::npos) {
                ++out.requests.reuse_append;
            } else if (reuse.find("seed") != std::string::npos ||
                       reuse.find("restore") != std::string::npos) {
                ++out.requests.reuse_seed;
            } else if (!reuse.empty()) {
                ++out.requests.reuse_other;
            }
        } catch (...) {}
    }
    if (n_ttft != 0) { out.requests.ttft_ms_mean = ttft_sum / n_ttft; }
    if (n_dec != 0) { out.requests.decode_tok_s_mean = decode_sum / n_dec; }
}

std::int64_t Collector::now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

void Collector::load_persisted_series() {
    if (logs_dir_.empty()) { return; }
    std::filesystem::create_directories(logs_dir_);
    series_path_ = (std::filesystem::path(logs_dir_) / "series.jsonl").string();
    std::ifstream in(series_path_, std::ios::binary);
    if (in) {
        in.seekg(0, std::ios::end);
        const auto sz = static_cast<std::int64_t>(in.tellg());
        const std::int64_t keep = 2 * 1024 * 1024;
        if (sz > keep) { in.seekg(sz - keep, std::ios::beg); }
        else {
            in.seekg(0, std::ios::beg);
        }
        std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto nl = body.find('\n');
        if (sz > keep && nl != std::string::npos) { body.erase(0, nl + 1); }
        series_.load_jsonl(body);
    }
    series_file_.open(series_path_, std::ios::app);
}

void Collector::persist_sample(const VramSample& s) {
    if (!series_file_.is_open()) { return; }
    series_file_ << format_series_sample_line(s) << '\n';
    series_file_.flush();
}

void Collector::persist_event(const VramSeriesEvent& e) {
    if (!series_file_.is_open()) { return; }
    series_file_ << format_series_event_line(e) << '\n';
    series_file_.flush();
}

void Collector::start_series() {
    bool expected = false;
    if (!series_run_.compare_exchange_strong(expected, true)) { return; }
    load_persisted_series();
    series_thread_  = std::thread([this] { series_loop(); });
    observe_thread_ = std::thread([this] { observe_loop(); });
}

void Collector::stop_series() {
    series_run_ = false;
    if (series_thread_.joinable()) { series_thread_.join(); }
    if (observe_thread_.joinable()) { observe_thread_.join(); }
}

void Collector::observe_loop() {
    // Health and /admin/vram are HTTP. They do not belong on the 10 Hz DXGI
    // loop. They also cannot live only inside snapshot() — that is demand-driven
    // by /api/state, so a dashboard that is closed records nothing. 1 Hz is the
    // heartbeat: slow enough not to compete with the engine, fast enough that a
    // 5 s reclaim is visible even with nobody watching.
    while (series_run_.load()) {
        const auto t0 = std::chrono::steady_clock::now();
        try {
            Collected tmp;
            poll_health(tmp);
            poll_admin(tmp);
            record_transitions(tmp);
            std::lock_guard lock(mu_);
            detector_last_ran_ms_ = now_ms();
        } catch (...) {
            std::lock_guard lock(mu_);
            detector_last_ran_ms_ = now_ms();
        }
        const auto elapsed = std::chrono::steady_clock::now() - t0;
        const auto period  = std::chrono::milliseconds(1000);
        if (elapsed < period) { std::this_thread::sleep_for(period - elapsed); }
    }
}

void Collector::series_loop() {
    // DXGI is an in-process API call, cheap enough to sample at the full rate --
    // and the budget oscillation IS the finding, so it must not be decimated.
    // nvidia-smi is a PROCESS SPAWN measured at ~51 ms on this box; polling it
    // every tick cost ~10 spawns/s and ~48% of one core, continuously. That does
    // not just waste CPU, it perturbs the machine this series exists to observe --
    // the game-test workload it is meant to measure would be competing with it.
    // Device totals move slowly, so sample them at 1 Hz and carry the last
    // reading forward into the fast series.
    constexpr int kNvidiaEvery = 10;
    int nvidia_tick            = 0;
    NvidiaSmiMemory nvidia_last;
    while (series_run_.load()) {
        const auto t0 = std::chrono::steady_clock::now();
        try {
            VramSample sample;
            sample.t_ms = now_ms();
            DxgiSnapshot dxgi = query_dxgi_local(spec_.device);
            if (nvidia_tick == 0 && dxgi.ok) {
                Collected nv;
                poll_nvidia_smi(nv);
                nvidia_last = nv.nvidia;
            }
            nvidia_tick = (nvidia_tick + 1) % kNvidiaEvery;
            sample.budget_bytes      = dxgi.budget_bytes;
            sample.nvidia_used_bytes = mib_to_bytes(nvidia_last.used_mib);
            {
                std::lock_guard lock(mu_);
                last_dxgi_   = dxgi;
                last_nvidia_ = nvidia_last;
                series_.push(sample);
                persist_sample(sample);
            }
        } catch (...) {}
        const auto elapsed = std::chrono::steady_clock::now() - t0;
        const auto period  = std::chrono::milliseconds(100);
        if (elapsed < period) { std::this_thread::sleep_for(period - elapsed); }
    }
}

void Collector::record_transitions(const Collected& snap) {
    const auto t = now_ms();
    std::lock_guard lock(mu_);
    if (last_health_status_ != -1 && last_health_status_ != snap.health_status) {
        if (snap.health_status == 200) {
            const VramSeriesEvent ev{t, "engine_up", "health 200"};
            series_.push_event(ev);
            persist_event(ev);
        } else if (last_health_status_ == 200) {
            const VramSeriesEvent ev{t, "engine_down",
                                     "health " + std::to_string(snap.health_status)};
            series_.push_event(ev);
            persist_event(ev);
        }
    }
    last_health_status_ = snap.health_status;
    last_health_body_   = snap.health_body;
    last_admin_vram_    = snap.admin_vram;
    last_admin_note_    = snap.admin_vram_note;
    if (snap.admin_vram.is_object()) {
        const std::string trans  = snap.admin_vram.value("last_transition", "");
        const std::string reason = snap.admin_vram.value("last_reason", "");
        std::string kind;
        if (admin_cursor_.observe(trans, reason, kind)) {
            std::string released;
            if (snap.admin_vram.contains("tiers") && snap.admin_vram.at("tiers").is_array()) {
                for (const auto& tier : snap.admin_vram.at("tiers")) {
                    if (tier.value("released", false)) {
                        if (!released.empty()) { released += ","; }
                        released += tier.value("name", "?");
                    }
                }
            }
            std::string label = std::string(trans);
            if (!reason.empty()) {
                if (!label.empty()) { label += " "; }
                label += reason;
            }
            if (!released.empty()) { label += " released=" + released; }
            const VramSeriesEvent ev{t, kind, label};
            series_.push_event(ev);
            persist_event(ev);
            if (kind == "vram_release") { last_release_ms_ = t; }
            if (kind == "vram_reclaim") { last_release_ms_ = 0; }
        }
        bool any_released = false;
        if (snap.admin_vram.contains("tiers") && snap.admin_vram.at("tiers").is_array()) {
            for (const auto& tier : snap.admin_vram.at("tiers")) {
                if (tier.value("released", false)) { any_released = true; }
            }
        }
        if (!any_released && trans == "reclaim") { last_release_ms_ = 0; }
    }
}

void Collector::note_engine_state(const std::string& state, const std::string& last_event) {
    std::lock_guard lock(mu_);
    if (!last_engine_state_.empty() && state != last_engine_state_) {
        const auto t = now_ms();
        if (state == "Running" || state == "Starting") {
            const VramSeriesEvent ev{t, "engine_start", last_event.empty() ? state : last_event};
            series_.push_event(ev);
            persist_event(ev);
        } else if (state == "Stopped" || state == "Stopping" || state == "Halted") {
            const VramSeriesEvent ev{t, "engine_stop", last_event.empty() ? state : last_event};
            series_.push_event(ev);
            persist_event(ev);
        }
    }
    last_engine_state_ = state;
}

nlohmann::json Collector::series_json() {
    std::lock_guard lock(mu_);
    const auto samples = series_.samples();
    nlohmann::json t_ms            = nlohmann::json::array();
    nlohmann::json budget          = nlohmann::json::array();
    nlohmann::json nvidia_used     = nlohmann::json::array();
    for (const auto& s : samples) {
        t_ms.push_back(s.t_ms);
        budget.push_back(s.budget_bytes);
        nvidia_used.push_back(s.nvidia_used_bytes);
    }
    nlohmann::json events = nlohmann::json::array();
    for (const auto& e : series_.events()) {
        events.push_back({{"t_ms", e.t_ms}, {"kind", e.kind}, {"label", e.label}});
    }
    return {{"hz", 10},
            {"raw", true},
            {"t_ms", std::move(t_ms)},
            {"budget_bytes", std::move(budget)},
            {"nvidia_used_bytes", std::move(nvidia_used)},
            {"events", std::move(events)},
            {"detector_last_ran_ms", detector_last_ran_ms_}};
}

nlohmann::json Collector::vram_control_json() {
    std::lock_guard lock(mu_);
    nlohmann::json tiers = nlohmann::json::array();
    bool any_released    = false;
    if (last_admin_vram_.is_object() && last_admin_vram_.contains("tiers") &&
        last_admin_vram_.at("tiers").is_array()) {
        for (const auto& tier : last_admin_vram_.at("tiers")) {
            const bool released = tier.value("released", false);
            if (released) { any_released = true; }
            tiers.push_back({{"name", tier.value("name", "")},
                             {"released", released},
                             {"held_bytes", tier.value("held_bytes", 0)},
                             {"min_bytes", tier.value("min_bytes", 0)},
                             {"max_bytes", tier.value("max_bytes", 0)},
                             {"reclaimable_bytes", tier.value("reclaimable_bytes", 0)}});
        }
    }
    const auto now = now_ms();
    nlohmann::json out = {
        {"last_transition", admin_cursor_.last_transition},
        {"last_reason", admin_cursor_.last_reason},
        {"any_released", any_released},
        {"since_release_s",
         (any_released && last_release_ms_ > 0)
             ? nlohmann::json((now - last_release_ms_) / 1000)
             : nlohmann::json(nullptr)},
        {"tiers", std::move(tiers)},
        {"note", last_admin_note_},
        {"detector_last_ran_ms", detector_last_ran_ms_},
        {"detector_age_s",
         detector_last_ran_ms_ > 0 ? nlohmann::json((now - detector_last_ran_ms_) / 1000)
                                   : nlohmann::json(nullptr)},
    };
    return out;
}

nlohmann::json Collector::insights_report() {
    auto report = insights_from_request_log_path(spec_.request_log);
    nlohmann::json admin;
    std::string note;
    {
        std::lock_guard lock(mu_);
        admin = last_admin_vram_;
        note  = last_admin_note_;
    }
    append_admin_vram_insights(report, admin, note);
    return report;
}

Collected Collector::snapshot() {
    Collected out;
    poll_request_log(out);
    if (series_run_.load()) {
        std::lock_guard lock(mu_);
        out.health_status    = last_health_status_ < 0 ? 0 : last_health_status_;
        out.health_body      = last_health_body_;
        out.admin_vram       = last_admin_vram_;
        out.admin_vram_note  = last_admin_note_;
        out.dxgi             = last_dxgi_;
        out.nvidia           = last_nvidia_;
        return out;
    }
    poll_health(out);
    poll_admin(out);
    out.dxgi = query_dxgi_local(spec_.device);
    poll_nvidia_smi(out);
    record_transitions(out);
    return out;
}

} // namespace ninfer::supervisor
