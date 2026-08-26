#pragma once

#include "config.hpp"
#include "dxgi_query.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <mutex>
#include <string>

namespace ninfer::supervisor {

struct RequestMix {
    std::uint64_t done                 = 0;
    double ttft_ms_mean                = 0;
    double decode_tok_s_mean           = 0;
    std::uint64_t reuse_full_reset     = 0;
    std::uint64_t reuse_append         = 0;
    std::uint64_t reuse_seed           = 0;
    std::uint64_t reuse_other          = 0;
    std::string last_reuse;
    bool log_available                 = false;
    std::string log_error;
};

struct Collected {
    DxgiSnapshot dxgi;
    NvidiaSmiMemory nvidia;
    nlohmann::json admin_vram = nullptr;
    std::string admin_vram_note;
    RequestMix requests;
    std::string health_body;
    std::string engine_capacity_line;
    int health_status = 0;
};

class Collector {
public:
    explicit Collector(EngineSpec spec) : spec_(std::move(spec)) {}
    Collected snapshot();

private:
    void poll_health(Collected& out);
    void poll_admin(Collected& out);
    void poll_nvidia_smi(Collected& out);
    void poll_request_log(Collected& out);

    EngineSpec spec_;
    std::mutex mu_;
};

} // namespace ninfer::supervisor
