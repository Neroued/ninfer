#include "engine_child.hpp"

#include <windows.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace ninfer::supervisor {
namespace {

std::int64_t unix_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

std::wstring utf8_to_wide(const std::string& s) {
    if (s.empty()) { return {}; }
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring out(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, out.data(), n);
    if (!out.empty() && out.back() == L'\0') { out.pop_back(); }
    return out;
}

std::wstring quote_arg(const std::string& a) {
    std::wstring w = utf8_to_wide(a);
    if (w.find_first_of(L" \t\"") == std::wstring::npos) { return w; }
    std::wstring q = L"\"";
    for (wchar_t c : w) {
        if (c == L'"') { q += L"\\\""; } else { q += c; }
    }
    q += L'"';
    return q;
}

void close_handle(void*& h) {
    if (h != nullptr) {
        CloseHandle(static_cast<HANDLE>(h));
        h = nullptr;
    }
}

} // namespace

EngineChild::EngineChild(SupervisorConfig cfg) : cfg_(std::move(cfg)), gate_(cfg_.restart) {
    if (cfg_.logs_dir.empty()) { cfg_.logs_dir = "ninfer-supervisor-logs"; }
    std::filesystem::create_directories(cfg_.logs_dir);
    log_path_ = (std::filesystem::path(cfg_.logs_dir) / "engine.log").string();
    if (!manages_engine_process(cfg_)) {
        auto_restart_  = false;
        st_.last_event = "monitor-only: not managing engine process";
    }
}

EngineChild::~EngineChild() {
    quit_ = true;
    stop();
    close_handle(job_handle_);
}

void EngineChild::request_quit() { quit_ = true; }

EngineStatus EngineChild::status() const {
    std::lock_guard lock(mu_);
    EngineStatus s     = st_;
    s.crash_loop_halted = gate_.halted();
    s.health_fails      = gate_.health_fails();
    return s;
}

std::string EngineChild::log_tail(std::size_t max_bytes) const {
    std::ifstream in(log_path_, std::ios::binary);
    if (!in) { return {}; }
    in.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(in.tellg());
    const std::size_t off = size > max_bytes ? size - max_bytes : 0;
    in.seekg(static_cast<std::streamoff>(off));
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

void EngineChild::append_log(const char* data, std::size_t n) {
    rotate_logs_if_needed();
    std::ofstream out(log_path_, std::ios::binary | std::ios::app);
    if (out) { out.write(data, static_cast<std::streamsize>(n)); }
}

void EngineChild::rotate_logs_if_needed() {
    std::error_code ec;
    const auto sz = std::filesystem::file_size(log_path_, ec);
    if (ec || sz < (8ULL << 20)) { return; }
    const auto rotated = log_path_ + ".1";
    std::filesystem::remove(rotated, ec);
    std::filesystem::rename(log_path_, rotated, ec);
}

void EngineChild::start() {
    if (!manages_engine_process(cfg_)) { return; }
    auto_restart_ = true;
    gate_.reset_halt();
    std::lock_guard lock(mu_);
    if (st_.state == EngineState::Running || st_.state == EngineState::Starting) { return; }
    st_.last_event = "start requested";
}

void EngineChild::stop() {
    if (!manages_engine_process(cfg_)) { return; }
    auto_restart_ = false;
    stop_child_   = true;
    HANDLE proc   = nullptr;
    {
        std::lock_guard lock(mu_);
        proc           = static_cast<HANDLE>(process_handle_);
        st_.state      = EngineState::Stopping;
        st_.last_event = "stop requested";
    }
    if (proc != nullptr) { TerminateProcess(proc, 1); }
}

void EngineChild::observe_health(int http_status) {
    if (!manages_engine_process(cfg_)) {
        std::lock_guard lock(mu_);
        if (http_status == 200) {
            st_.health     = "ok";
            st_.state      = EngineState::Running;
            st_.last_event = "unmanaged engine reachable";
        } else if (http_status == 503) {
            st_.health     = "unhealthy";
            st_.state      = EngineState::Running;
            st_.last_event = "unmanaged engine unhealthy";
        } else {
            st_.health     = "unreachable";
            st_.state      = EngineState::Stopped;
            st_.last_event = "unmanaged engine unreachable";
        }
        return;
    }
    bool restart_now = false;
    {
        std::lock_guard lock(mu_);
        if (http_status == 200) {
            gate_.note_healthy();
            st_.health = "ok";
            return;
        }
        if (http_status == 503) {
            st_.health = "unhealthy";
            if (gate_.note_health_fail() && auto_restart_.load()) {
                st_.last_event = "health restart threshold";
                restart_now    = true;
            }
        } else {
            st_.health = "unreachable";
        }
    }
    if (restart_now) { restart(); }
}

void EngineChild::restart() {
    if (!manages_engine_process(cfg_)) { return; }
    auto_restart_ = true;
    gate_.reset_halt();
    stop_child_ = true;
    HANDLE proc = nullptr;
    {
        std::lock_guard lock(mu_);
        proc           = static_cast<HANDLE>(process_handle_);
        st_.last_event = "restart requested";
    }
    if (proc != nullptr) { TerminateProcess(proc, 1); }
}

void EngineChild::spawn() {
    stop_child_ = false;
    std::wstring cmd = quote_arg(cfg_.engine.executable);
    for (const auto& a : cfg_.engine.args) {
        cmd += L' ';
        cmd += quote_arg(a);
    }
    std::vector<wchar_t> cmd_buf(cmd.begin(), cmd.end());
    cmd_buf.push_back(L'\0');

    SECURITY_ATTRIBUTES sa{};
    sa.nLength        = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE out_r = nullptr;
    HANDLE out_w = nullptr;
    if (!CreatePipe(&out_r, &out_w, &sa, 0)) {
        throw std::runtime_error("CreatePipe failed");
    }
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si{};
    si.cb          = sizeof(si);
    si.dwFlags     = STARTF_USESTDHANDLES;
    si.hStdOutput  = out_w;
    si.hStdError   = out_w;
    si.hStdInput   = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION pi{};
    const std::wstring cwd = utf8_to_wide(cfg_.engine.workdir);
    const wchar_t* cwd_ptr = cwd.empty() ? nullptr : cwd.c_str();
    if (!CreateProcessW(nullptr, cmd_buf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                        nullptr, cwd_ptr, &si, &pi)) {
        CloseHandle(out_r);
        CloseHandle(out_w);
        throw std::runtime_error("CreateProcessW failed");
    }
    CloseHandle(out_w);
    CloseHandle(pi.hThread);

    if (job_handle_ == nullptr) {
        HANDLE job = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION lim{};
        lim.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &lim, sizeof(lim));
        job_handle_ = job;
    }
    AssignProcessToJobObject(static_cast<HANDLE>(job_handle_), pi.hProcess);

    {
        std::lock_guard lock(mu_);
        process_handle_      = pi.hProcess;
        st_.pid              = static_cast<std::uint64_t>(pi.dwProcessId);
        st_.state            = EngineState::Running;
        st_.started_unix_ms  = unix_ms();
        ++st_.restart_count;
        st_.last_event = "engine started";
    }

    std::thread reader([this, out_r] {
        char buf[4096];
        DWORD n = 0;
        while (ReadFile(out_r, buf, sizeof(buf), &n, nullptr) && n > 0) { append_log(buf, n); }
        CloseHandle(out_r);
    });
    reader.detach();
}

void EngineChild::capture_wait() {
    HANDLE proc = nullptr;
    {
        std::lock_guard lock(mu_);
        proc = static_cast<HANDLE>(process_handle_);
    }
    if (proc == nullptr) { return; }
    WaitForSingleObject(proc, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(proc, &code);
    {
        std::lock_guard lock(mu_);
        st_.last_exit_code = static_cast<int>(code);
        st_.pid            = 0;
        process_handle_    = nullptr;
        st_.state          = EngineState::Stopped;
        st_.last_event     = "engine exited";
    }
    CloseHandle(proc);
}

void EngineChild::run_loop() {
    while (!quit_.load()) {
        const bool running = [&] {
            std::lock_guard lock(mu_);
            return process_handle_ != nullptr;
        }();
        if (running) {
            capture_wait();
            const bool intentional = stop_child_.exchange(false);
            if (quit_.load() || !auto_restart_.load() || intentional) { continue; }
            bool allow = false;
            {
                std::lock_guard lock(mu_);
                allow = gate_.note_exit(std::chrono::steady_clock::now());
            }
            if (!allow) {
                std::lock_guard lock(mu_);
                st_.state      = EngineState::Halted;
                st_.last_event = "crash-loop breaker: too many exits";
                auto_restart_  = false;
                continue;
            }
            int wait_s = 0;
            {
                std::lock_guard lock(mu_);
                gate_.advance_backoff();
                wait_s = gate_.backoff_seconds();
            }
            {
                std::lock_guard lock(mu_);
                st_.state      = EngineState::BackingOff;
                st_.last_event = "backing off";
            }
            for (int i = 0; i < wait_s * 10 && !quit_.load() && auto_restart_.load(); ++i) {
                Sleep(100);
            }
            continue;
        }
        if (manages_engine_process(cfg_) && auto_restart_.load() && !gate_.halted() &&
            !quit_.load()) {
            try {
                {
                    std::lock_guard lock(mu_);
                    st_.state = EngineState::Starting;
                }
                spawn();
            } catch (const std::exception& ex) {
                std::lock_guard lock(mu_);
                st_.state      = EngineState::Stopped;
                st_.last_event = std::string("spawn failed: ") + ex.what();
                auto_restart_  = false;
            }
            continue;
        }
        Sleep(200);
    }
    stop();
}

} // namespace ninfer::supervisor
