#pragma once

#include "engine_child.hpp"

#include <cstdint>
#include <string>

namespace ninfer::supervisor {

// What the tray icon says at a glance. In monitor-only mode this still reflects
// the OBSERVED health of the engine rather than a neutral "not my process":
// EngineChild keeps st_.health current for unmanaged engines too, so the colour
// is measured, not invented. The tooltip carries the managed/observing
// distinction instead, because that belongs in words rather than in a hue.
enum class TrayStatus : std::uint8_t {
    Idle,     // grey  — nothing running, or nothing known yet
    Working,  // green — engine running and answering /health
    Pending,  // amber — starting, stopping, backing off, or unreachable
    Failed,   // red   — halted, crash-looped, or reporting unhealthy
};

class TrayIcon {
public:
    TrayIcon(EngineChild& child, std::string dashboard_url, bool manages_engine);
    ~TrayIcon();
    void run();
    void request_quit();
    void open_dashboard() const;
    EngineChild& child();

    // Repaints the tray icon when the status changes. Called on a timer; cheap
    // because it reads EngineChild's in-memory status and never polls the
    // engine or shells out to nvidia-smi.
    void refresh_icon();

private:
    TrayStatus current_status();
    std::wstring tooltip(TrayStatus status) const;

    EngineChild& child_;
    std::string dashboard_url_;
    bool manages_engine_ = true;
    void* hwnd_          = nullptr;
    void* hicon_         = nullptr;
    int last_status_     = -1;
};

} // namespace ninfer::supervisor
