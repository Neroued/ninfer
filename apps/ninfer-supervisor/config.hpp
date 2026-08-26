#pragma once

#include "logic.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::supervisor {

struct EngineSpec {
    std::string executable;
    std::vector<std::string> args;
    std::string workdir;
    std::string api_key_file;
    std::string engine_host = "127.0.0.1";
    int engine_port         = 8010;
    std::string request_log;
    int device      = 0;
    bool unmanaged  = false; // observe an engine this process did not spawn
};

struct SupervisorConfig {
    EngineSpec engine;
    std::string host = "127.0.0.1";
    int port         = 8099;
    bool bind_any    = false;
    bool monitor_only = false; // never spawn/stop/restart; HTTP observe only
    std::string logs_dir;
    bool run_at_login = false;
    RestartPolicy restart;
};

inline bool manages_engine_process(const SupervisorConfig& cfg) noexcept {
    return !cfg.monitor_only && !cfg.engine.unmanaged;
}

inline std::string read_file_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

inline std::string read_api_key(const std::string& path) {
    if (path.empty()) { return {}; }
    std::string raw = read_file_text(path);
    while (!raw.empty() && (raw.back() == '\n' || raw.back() == '\r' || raw.back() == ' ' ||
                            raw.back() == '\t')) {
        raw.pop_back();
    }
    return raw;
}

inline SupervisorConfig load_config_json(const std::string& json_text,
                                         bool monitor_only_cli = false) {
    const auto body = nlohmann::json::parse(json_text);
    SupervisorConfig cfg;
    if (body.contains("engine") && body.at("engine").is_object()) {
        const auto& e = body.at("engine");
        cfg.engine.executable   = e.value("executable", "");
        cfg.engine.workdir      = e.value("workdir", "");
        cfg.engine.api_key_file = e.value("api_key_file", "");
        cfg.engine.engine_host  = e.value("engine_host", "127.0.0.1");
        cfg.engine.engine_port  = e.value("engine_port", 8010);
        cfg.engine.request_log  = e.value("request_log", "");
        cfg.engine.device       = e.value("device", 0);
        cfg.engine.unmanaged    = e.value("unmanaged", false);
        if (e.contains("args") && e.at("args").is_array()) {
            for (const auto& a : e.at("args")) {
                if (a.is_string()) { cfg.engine.args.push_back(a.get<std::string>()); }
            }
        }
    }
    if (body.contains("supervisor") && body.at("supervisor").is_object()) {
        const auto& s = body.at("supervisor");
        cfg.host         = s.value("host", "127.0.0.1");
        cfg.port         = s.value("port", 8099);
        cfg.bind_any      = s.value("bind_any", false);
        cfg.monitor_only  = s.value("monitor_only", false) || monitor_only_cli;
        cfg.logs_dir      = s.value("logs_dir", "");
        cfg.run_at_login = s.value("run_at_login", false);
        if (s.contains("restart") && s.at("restart").is_object()) {
            const auto& r = s.at("restart");
            cfg.restart.max_backoff_s         = r.value("max_backoff_s", 60);
            cfg.restart.crash_loop_window_s   = r.value("crash_loop_window_s", 60);
            cfg.restart.crash_loop_max        = r.value("crash_loop_max", 5);
            cfg.restart.health_fail_threshold = r.value("health_fail_threshold", 3);
        }
    }
    if (monitor_only_cli) { cfg.monitor_only = true; }
    if (manages_engine_process(cfg) && cfg.engine.executable.empty()) {
        throw std::invalid_argument(
            "engine.executable is required unless monitor_only or engine.unmanaged");
    }
    if (!cfg.bind_any && !is_loopback_host(cfg.host)) {
        throw std::invalid_argument(
            "supervisor host must be loopback unless bind_any is true");
    }
    return cfg;
}

} // namespace ninfer::supervisor
