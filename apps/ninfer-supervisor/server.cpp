#include "server.hpp"
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
#include <iostream>
#include <thread>

namespace ninfer::supervisor {
namespace {

const char* state_name(EngineState s) {
    switch (s) {
    case EngineState::Stopped: return "Stopped";
    case EngineState::Starting: return "Starting";
    case EngineState::Running: return "Running";
    case EngineState::Stopping: return "Stopping";
    case EngineState::BackingOff: return "BackingOff";
    case EngineState::Halted: return "Halted";
    }
    return "?";
}

std::int64_t now_unix_s() {
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

} // namespace

DashboardServer::DashboardServer(SupervisorConfig cfg, EngineChild& child, Collector& collector)
    : cfg_(std::move(cfg)), child_(child), collector_(collector) {}

DashboardServer::~DashboardServer() { stop(); }

bool DashboardServer::control_allowed(const std::string& remote) const {
    return is_loopback_peer(remote);
}

nlohmann::json DashboardServer::state_json() {
    const EngineStatus st = child_.status();
    Collected snap        = collector_.snapshot();
    nlohmann::json engine = {
        {"state", state_name(st.state)},
        {"pid", st.pid},
        {"restart_count", st.restart_count},
        {"last_exit_code", st.last_exit_code},
        {"last_event", st.last_event},
        {"crash_loop_halted", st.crash_loop_halted},
        {"uptime_s", st.started_unix_ms == 0
                         ? 0
                         : now_unix_s() - st.started_unix_ms / 1000},
    };
    nlohmann::json dxgi = {
        {"ok", snap.dxgi.ok},
        {"error", snap.dxgi.error},
        {"adapter_name", snap.dxgi.adapter_name},
        {"budget_bytes", snap.dxgi.budget_bytes},
        {"supervisor_usage_bytes", snap.dxgi.current_usage_bytes},
        {"supervisor_usage_note", "DXGI CurrentUsage of the supervisor process, not the engine"},
    };
    nlohmann::json nvidia = {{"ok", snap.nvidia.ok},
                             {"error", snap.nvidia.error},
                             {"index", snap.nvidia.index},
                             {"used_bytes", mib_to_bytes(snap.nvidia.used_mib)},
                             {"total_bytes", mib_to_bytes(snap.nvidia.total_mib)}};
    nlohmann::json req  = {{"done", snap.requests.done},
                           {"ttft_ms_mean", snap.requests.ttft_ms_mean},
                           {"decode_tok_s_mean", snap.requests.decode_tok_s_mean},
                           {"reuse_full_reset", snap.requests.reuse_full_reset},
                           {"reuse_append", snap.requests.reuse_append},
                           {"reuse_seed", snap.requests.reuse_seed},
                           {"last_reuse", snap.requests.last_reuse},
                           {"log_available", snap.requests.log_available},
                           {"log_error", snap.requests.log_error},
                           {"mtp_backend", snap.requests.mtp_backend},
                           {"mtp_draft_window", snap.requests.mtp_draft_window},
                           {"mtp_drafted", snap.requests.mtp_drafted},
                           {"mtp_accepted", snap.requests.mtp_accepted},
                           {"mtp_fallback_steps", snap.requests.mtp_fallback_steps},
                           {"mtp_rounds", snap.requests.mtp_rounds},
                           {"mtp_accepted_per_position", snap.requests.mtp_accepted_per_position},
                           {"mtp_last_accept_rate", snap.requests.mtp_last_accept_rate}};
    nlohmann::json health = {{"status", snap.health_status}, {"body", snap.health_body}};
    child_.observe_health(snap.health_status);
    collector_.note_engine_state(state_name(st.state), st.last_event);
    const std::string log = child_.log_tail(16 * 1024);
    std::string cap       = extract_kv_capacity_line(log);
    if (cap.empty()) { cap = snap.engine_capacity_line; }
    nlohmann::json insights = collector_.insights_report();
    return {{"monitor_only", !manages_engine_process(cfg_)},
            {"engine", std::move(engine)},
            {"dxgi", std::move(dxgi)},
            {"nvidia_smi", std::move(nvidia)},
            {"engine_capacity_line", cap},
            {"admin_vram", snap.admin_vram},
            {"admin_vram_note", snap.admin_vram_note},
            {"vram_control", collector_.vram_control_json()},
            {"requests", std::move(req)},
            {"insights", insights},
            {"series", collector_.series_json()},
            {"health", std::move(health)},
            {"log_tail", log}};
}

void DashboardServer::stop() {
    stop_ = true;
    if (server_ != nullptr) { static_cast<httplib::Server*>(server_)->stop(); }
}

void DashboardServer::run() {
    httplib::Server svr;
    server_ = &svr;
    // No Access-Control-Allow-Origin. The custom mutating header is a CSRF
    // brake only because cross-origin preflight then fails closed.
    svr.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        if (!host_header_allowed(req.get_header_value("Host"), cfg_.port, cfg_.host, cfg_.bind_any)) {
            res.status = 403;
            res.set_content(nlohmann::json{{"error", "host not allowed"}}.dump(), "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(std::string(kDashboardHtml), "text/html; charset=utf-8");
    });
    svr.Get("/api/state", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(state_json().dump(), "application/json");
    });
    svr.Get("/api/insights", [this](const httplib::Request&, httplib::Response& res) {
        (void)collector_.snapshot();
        res.set_content(collector_.insights_report().dump(), "application/json");
    });
    svr.Get("/api/events", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider("text/event-stream", [this](std::size_t, httplib::DataSink& sink) {
            if (stop_.load()) {
                sink.done();
                return false;
            }
            const std::string payload = "data: " + state_json().dump() + "\n\n";
            sink.write(payload.data(), payload.size());
            for (int i = 0; i < 10 && !stop_.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            return !stop_.load();
        });
    });
    auto control = [this](const httplib::Request& req, httplib::Response& res, auto fn) {
        if (!control_allowed(req.remote_addr)) {
            res.status = 403;
            res.set_content(nlohmann::json{{"error", "control is loopback-only"}}.dump(),
                            "application/json");
            return;
        }
        if (!supervisor_control_header_ok(req.get_header_value(std::string(kSupervisorControlHeader)))) {
            res.status = 403;
            res.set_content(nlohmann::json{{"error", "missing X-NInfer-Supervisor header"}}.dump(),
                            "application/json");
            return;
        }
        if (!manages_engine_process(cfg_)) {
            res.status = 409;
            res.set_content(nlohmann::json{{"error", "engine is unmanaged"}}.dump(),
                            "application/json");
            return;
        }
        fn();
        res.set_content(state_json().dump(), "application/json");
    };
    svr.Post("/api/start", [this, control](const httplib::Request& req, httplib::Response& res) {
        control(req, res, [this] { child_.start(); });
    });
    svr.Post("/api/stop", [this, control](const httplib::Request& req, httplib::Response& res) {
        control(req, res, [this] { child_.stop(); });
    });
    svr.Post("/api/restart", [this, control](const httplib::Request& req, httplib::Response& res) {
        control(req, res, [this] { child_.restart(); });
    });

    const std::string host = cfg_.bind_any ? "0.0.0.0" : cfg_.host;
    if (cfg_.bind_any || !is_loopback_host(host)) {
        std::cerr << "WARNING: ninfer-supervisor is binding " << host
                  << " — control endpoints start/stop the engine. Loopback is the default.\n";
    }
    svr.listen(host, cfg_.port);
    server_ = nullptr;
}

} // namespace ninfer::supervisor
