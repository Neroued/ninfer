#include "collector.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#define CPPHTTPLIB_NO_EXCEPTIONS
#include <httplib.h>

#include <cstdio>
#include <fstream>
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
        if (line.find("\"request_done\"") != std::string::npos) { lines.push_back(line); }
        if (line.find("\"server_start\"") != std::string::npos) { last_start = std::move(line); }
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

Collected Collector::snapshot() {
    Collected out;
    out.dxgi = query_dxgi_local(spec_.device);
    poll_nvidia_smi(out);
    poll_health(out);
    poll_admin(out);
    poll_request_log(out);
    return out;
}

} // namespace ninfer::supervisor
