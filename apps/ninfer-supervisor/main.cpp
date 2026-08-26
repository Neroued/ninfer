#include "collector.hpp"
#include "config.hpp"
#include "engine_child.hpp"
#include "logic.hpp"
#include "server.hpp"
#include "tray.hpp"

#include <windows.h>

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

void install_run_at_login(const std::string& command) {
    HKEY key = nullptr;
    if (RegCreateKeyExW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\CurrentVersion\\Run", 0,
                        nullptr, 0, KEY_SET_VALUE, nullptr, &key, nullptr) != ERROR_SUCCESS) {
        throw std::runtime_error("cannot open Run key");
    }
    std::wstring w(command.begin(), command.end());
    const LONG st =
        RegSetValueExW(key, L"NInferSupervisor", 0, REG_SZ,
                       reinterpret_cast<const BYTE*>(w.c_str()),
                       static_cast<DWORD>((w.size() + 1) * sizeof(wchar_t)));
    RegCloseKey(key);
    if (st != ERROR_SUCCESS) { throw std::runtime_error("cannot write Run key"); }
}

void uninstall_run_at_login() {
    HKEY key = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\CurrentVersion\\Run", 0,
                      KEY_SET_VALUE, &key) != ERROR_SUCCESS) {
        return;
    }
    RegDeleteValueW(key, L"NInferSupervisor");
    RegCloseKey(key);
}

void usage() {
    std::cout
        << "usage: ninfer-supervisor --config FILE [--host 127.0.0.1] [--port 8099] [--bind-any]\n"
           "                         [--monitor-only] [--install-login] [--uninstall-login]\n"
           "  Dashboard binds loopback by default. --bind-any is required for 0.0.0.0 and prints\n"
           "  a warning. Control POST /api/start|stop|restart is loopback-peer only, requires\n"
           "  header X-NInfer-Supervisor: 1, and returns 409 in --monitor-only / unmanaged mode.\n"
           "  Host is allowlisted on every route. Do not send CORS headers.\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string config_path;
        std::string host_override;
        int port_override = -1;
        bool bind_any     = false;
        bool monitor_only = false;
        bool install      = false;
        bool uninstall    = false;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto need           = [&](const char* name) -> const char* {
                if (i + 1 >= argc) { throw std::invalid_argument(std::string("missing ") + name); }
                return argv[++i];
            };
            if (a == "--help" || a == "-h") {
                usage();
                return 0;
            } else if (a == "--config") {
                config_path = need("--config");
            } else if (a == "--host") {
                host_override = need("--host");
            } else if (a == "--port") {
                port_override = std::stoi(need("--port"));
            } else if (a == "--bind-any") {
                bind_any = true;
            } else if (a == "--monitor-only") {
                monitor_only = true;
            } else if (a == "--install-login") {
                install = true;
            } else if (a == "--uninstall-login") {
                uninstall = true;
            } else {
                throw std::invalid_argument("unknown argument: " + a);
            }
        }
        if (uninstall) {
            uninstall_run_at_login();
            std::cout << "removed HKCU Run\\NInferSupervisor\n";
            return 0;
        }
        if (config_path.empty()) {
            usage();
            return 2;
        }
        auto cfg = ninfer::supervisor::load_config_json(
            ninfer::supervisor::read_file_text(config_path), monitor_only);
        if (!host_override.empty()) { cfg.host = host_override; }
        if (port_override > 0) { cfg.port = port_override; }
        if (bind_any) { cfg.bind_any = true; }
        if (monitor_only) { cfg.monitor_only = true; }
        if (cfg.bind_any) {
            std::cerr << "WARNING: binding beyond loopback; engine start/stop is exposed on "
                      << (cfg.host.empty() ? "0.0.0.0" : cfg.host) << ":" << cfg.port << "\n";
        } else if (!ninfer::supervisor::is_loopback_host(cfg.host)) {
            throw std::invalid_argument("--host must be loopback without --bind-any");
        }
        if (install) {
            char module[MAX_PATH]{};
            GetModuleFileNameA(nullptr, module, MAX_PATH);
            const std::string cmd = std::string("\"") + module + "\" --config \"" + config_path + "\"";
            install_run_at_login(cmd);
            std::cout << "installed HKCU Run\\NInferSupervisor\n";
        }

        ninfer::supervisor::EngineChild child(cfg);
        ninfer::supervisor::Collector collector(cfg.engine, cfg.logs_dir);
        collector.start_series();
        ninfer::supervisor::DashboardServer server(cfg, child, collector);
        std::thread engine_thread([&] { child.run_loop(); });
        std::thread http_thread([&] { server.run(); });
        const std::string url =
            "http://" + (cfg.bind_any ? std::string("127.0.0.1") : cfg.host) + ":" +
            std::to_string(cfg.port) + "/";
        std::cout << "ninfer-supervisor dashboard " << url << "\n";
        ninfer::supervisor::TrayIcon tray(child, url, ninfer::supervisor::manages_engine_process(cfg));
        tray.run();
        server.stop();
        collector.stop_series();
        child.request_quit();
        child.stop();
        if (http_thread.joinable()) { http_thread.join(); }
        if (engine_thread.joinable()) { engine_thread.join(); }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ninfer-supervisor: " << ex.what() << "\n";
        return 1;
    }
}
