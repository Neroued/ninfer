#include "tray.hpp"

#include <windows.h>
#include <shellapi.h>

namespace ninfer::supervisor {
namespace {

constexpr UINT kTrayMsg   = WM_APP + 1;
constexpr UINT kIdOpen    = 1;
constexpr UINT kIdStart   = 2;
constexpr UINT kIdStop    = 3;
constexpr UINT kIdRestart = 4;
constexpr UINT kIdQuit    = 5;
constexpr wchar_t kClass[] = L"NInferSupervisorTray";

struct TrayWnd {
    TrayIcon* self = nullptr;
};

LRESULT CALLBACK tray_wnd(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    TrayIcon* self = nullptr;
    if (msg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTW*>(lparam);
        self     = static_cast<TrayIcon*>(cs->lpCreateParams);
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
    } else {
        self = reinterpret_cast<TrayIcon*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    }
    if (self == nullptr) { return DefWindowProcW(hwnd, msg, wparam, lparam); }
    if (msg == kTrayMsg && (LOWORD(lparam) == WM_RBUTTONUP || LOWORD(lparam) == WM_LBUTTONUP)) {
        POINT pt{};
        GetCursorPos(&pt);
        HMENU menu = CreatePopupMenu();
        AppendMenuW(menu, MF_STRING, kIdOpen, L"Open dashboard");
        AppendMenuW(menu, MF_STRING, kIdStart, L"Start engine");
        AppendMenuW(menu, MF_STRING, kIdStop, L"Stop engine");
        AppendMenuW(menu, MF_STRING, kIdRestart, L"Restart engine");
        AppendMenuW(menu, MF_SEPARATOR, 0, nullptr);
        AppendMenuW(menu, MF_STRING, kIdQuit, L"Quit supervisor");
        SetForegroundWindow(hwnd);
        const int cmd =
            TrackPopupMenu(menu, TPM_RETURNCMD | TPM_NONOTIFY, pt.x, pt.y, 0, hwnd, nullptr);
        DestroyMenu(menu);
        if (cmd == kIdOpen) { self->open_dashboard(); }
        if (cmd == kIdStart) { self->child().start(); }
        if (cmd == kIdStop) { self->child().stop(); }
        if (cmd == kIdRestart) { self->child().restart(); }
        if (cmd == kIdQuit) { PostQuitMessage(0); }
        return 0;
    }
    if (msg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}

} // namespace

TrayIcon::TrayIcon(EngineChild& child, std::string dashboard_url)
    : child_(child), dashboard_url_(std::move(dashboard_url)) {}

TrayIcon::~TrayIcon() {
    if (hwnd_ != nullptr) { DestroyWindow(static_cast<HWND>(hwnd_)); }
}

EngineChild& TrayIcon::child() { return child_; }

void TrayIcon::open_dashboard() const {
    ShellExecuteA(nullptr, "open", dashboard_url_.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

void TrayIcon::request_quit() {
    if (hwnd_ != nullptr) { PostMessageW(static_cast<HWND>(hwnd_), WM_CLOSE, 0, 0); }
}

void TrayIcon::run() {
    WNDCLASSEXW wc{};
    wc.cbSize        = sizeof(wc);
    wc.lpfnWndProc   = tray_wnd;
    wc.hInstance     = GetModuleHandleW(nullptr);
    wc.lpszClassName = kClass;
    RegisterClassExW(&wc);
    HWND hwnd = CreateWindowExW(0, kClass, L"NInfer supervisor", 0, 0, 0, 0, 0, HWND_MESSAGE,
                                nullptr, wc.hInstance, this);
    hwnd_     = hwnd;
    NOTIFYICONDATAW nid{};
    nid.cbSize           = sizeof(nid);
    nid.hWnd             = hwnd;
    nid.uID              = 1;
    nid.uFlags           = NIF_MESSAGE | NIF_TIP | NIF_ICON;
    nid.uCallbackMessage = kTrayMsg;
    nid.hIcon            = LoadIconW(nullptr, MAKEINTRESOURCEW(32512));
    lstrcpyW(nid.szTip, L"NInfer supervisor");
    Shell_NotifyIconW(NIM_ADD, &nid);
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    Shell_NotifyIconW(NIM_DELETE, &nid);
}

} // namespace ninfer::supervisor
