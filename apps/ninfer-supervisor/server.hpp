#pragma once

#include "collector.hpp"
#include "dashboard.hpp"
#include "engine_child.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <thread>

namespace ninfer::supervisor {

class DashboardServer {
public:
    DashboardServer(SupervisorConfig cfg, EngineChild& child, Collector& collector);
    ~DashboardServer();

    void run();
    void stop();

private:
    nlohmann::json state_json();
    bool control_allowed(const std::string& remote) const;

    SupervisorConfig cfg_;
    EngineChild& child_;
    Collector& collector_;
    std::atomic<bool> stop_{false};
    void* server_ = nullptr; // httplib::Server*
};

} // namespace ninfer::supervisor
