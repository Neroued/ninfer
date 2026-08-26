#pragma once

#include "engine_child.hpp"

#include <string>

namespace ninfer::supervisor {

class TrayIcon {
public:
    TrayIcon(EngineChild& child, std::string dashboard_url);
    ~TrayIcon();
    void run();
    void request_quit();
    void open_dashboard() const;
    EngineChild& child();

private:
    EngineChild& child_;
    std::string dashboard_url_;
    void* hwnd_ = nullptr;
};

} // namespace ninfer::supervisor
