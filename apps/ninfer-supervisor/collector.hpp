#pragma once

#include "config.hpp"
#include "dxgi_query.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
    std::string mtp_backend;
    int mtp_draft_window               = 0;
    std::uint64_t mtp_drafted          = 0;
    std::uint64_t mtp_accepted         = 0;
    std::uint64_t mtp_fallback_steps   = 0;
    std::uint64_t mtp_rounds           = 0;
    std::vector<std::uint64_t> mtp_accepted_per_position;
    double mtp_last_accept_rate        = 0;
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
    explicit Collector(EngineSpec spec) : spec_(std::move(spec)), series_(6000) {}
    ~Collector() { stop_series(); }

    Collector(const Collector&)            = delete;
    Collector& operator=(const Collector&) = delete;

    void start_series();
    void stop_series();
    Collected snapshot();
    nlohmann::json series_json();
    void note_engine_state(const std::string& state, const std::string& last_event);

private:
    void poll_health(Collected& out);
    void poll_admin(Collected& out);
    void poll_nvidia_smi(Collected& out);
    void poll_request_log(Collected& out);
    void series_loop();
    void record_transitions(const Collected& snap);
    static std::int64_t now_ms();

    EngineSpec spec_;
    std::mutex mu_;
    VramSeriesRing series_;
    std::atomic<bool> series_run_{false};
    std::thread series_thread_;
    int last_health_status_          = -1;
    std::string last_admin_transition_;
    std::string last_admin_reason_;
    std::string last_admin_released_;
    std::string last_engine_state_;
    DxgiSnapshot last_dxgi_;
    NvidiaSmiMemory last_nvidia_;
};

} // namespace ninfer::supervisor
