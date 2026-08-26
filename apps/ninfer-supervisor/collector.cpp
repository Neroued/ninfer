#include "collector.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#define CPPHTTPLIB_NO_EXCEPTIONS
#include <httplib.h>

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
    std::string line;
    while (std::getline(in, line)) {
        if (line.find("\"request_done\"") != std::string::npos) { lines.push_back(std::move(line)); }
    }
    const std::size_t start = lines.size() > 32 ? lines.size() - 32 : 0;
    double ttft_sum = 0;
    double decode_sum = 0;
    int n_ttft = 0;
    int n_dec  = 0;
    for (std::size_t i = start; i < lines.size(); ++i) {
        try {
            const auto j = nlohmann::json::parse(lines[i]);
            if (j.value("type", "") != "request_done") { continue; }
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
    poll_health(out);
    poll_admin(out);
    poll_request_log(out);
    return out;
}

} // namespace ninfer::supervisor
