#include "tray.hpp"

#include <windows.h>
#include <shellapi.h>

#include <string>

namespace ninfer::supervisor {
namespace {

constexpr UINT kTrayMsg    = WM_APP + 1;
constexpr UINT kIdOpen     = 1;
constexpr UINT kIdStart    = 2;
constexpr UINT kIdStop     = 3;
constexpr UINT kIdRestart  = 4;
constexpr UINT kIdQuit     = 5;
constexpr UINT_PTR kTimer  = 1;
constexpr UINT kTrayUid    = 1;
constexpr wchar_t kClass[] = L"NInferSupervisorTray";

COLORREF status_fill(TrayStatus status) {
    switch (status) {
    case TrayStatus::Working: return RGB(0x2F, 0x9E, 0x54);
    case TrayStatus::Pending: return RGB(0xD1, 0x8B, 0x12);
    case TrayStatus::Failed: return RGB(0xC5, 0x3B, 0x33);
    case TrayStatus::Idle: break;
    }
    return RGB(0x6B, 0x72, 0x80);
}

const wchar_t* status_word(TrayStatus status) {
    switch (status) {
    case TrayStatus::Working: return L"running";
    case TrayStatus::Pending: return L"pending";
    case TrayStatus::Failed: return L"failed";
    case TrayStatus::Idle: break;
    }
    return L"idle";
}

int small_icon_size() {
    const int size = GetSystemMetrics(SM_CXSMICON);
    return size > 0 ? size : 16;
}

// Drawn rather than shipped as an .ico resource: no build-system change, no
// binary asset in the tree, correct at whatever SM_CXSMICON the display scaling
// reports, and the fill colour can carry status.
//
// The colour bitmap is an explicit 24bpp DIB section, NOT CreateCompatibleBitmap.
// A screen-compatible DDB is 32bpp on any modern display, and CreateIconIndirect
// then reads its alpha channel as per-pixel alpha. GDI never writes alpha, so
// every pixel would come out fully transparent and the icon would vanish. At
// 24bpp there is no alpha channel to misread and the 1bpp mask decides shape.
HICON make_tray_icon(TrayStatus status, int size) {
    HDC screen = GetDC(nullptr);
    if (screen == nullptr) { return nullptr; }

    BITMAPINFO bi{};
    bi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    bi.bmiHeader.biWidth       = size;
    bi.bmiHeader.biHeight      = -size; // top-down
    bi.bmiHeader.biPlanes      = 1;
    bi.bmiHeader.biBitCount    = 24;
    bi.bmiHeader.biCompression = BI_RGB;

    void* bits        = nullptr;
    HBITMAP color_bmp = CreateDIBSection(screen, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    HBITMAP mask_bmp  = CreateBitmap(size, size, 1, 1, nullptr);
    HDC color_dc      = CreateCompatibleDC(screen);
    HDC mask_dc       = CreateCompatibleDC(screen);
    if (color_bmp == nullptr || mask_bmp == nullptr || color_dc == nullptr || mask_dc == nullptr) {
        if (color_dc != nullptr) { DeleteDC(color_dc); }
        if (mask_dc != nullptr) { DeleteDC(mask_dc); }
        if (color_bmp != nullptr) { DeleteObject(color_bmp); }
        if (mask_bmp != nullptr) { DeleteObject(mask_bmp); }
        ReleaseDC(nullptr, screen);
        return nullptr;
    }
    auto* old_color = static_cast<HBITMAP>(SelectObject(color_dc, color_bmp));
    auto* old_mask  = static_cast<HBITMAP>(SelectObject(mask_dc, mask_bmp));

    // Full-bleed fill; the rounded corners are cut by the mask, so the glyph
    // antialiases against the fill rather than fringing against a background.
    RECT rc{0, 0, size, size};
    HBRUSH fill = CreateSolidBrush(status_fill(status));
    FillRect(color_dc, &rc, fill);
    DeleteObject(fill);

    LOGFONTW lf{};
    lf.lfHeight  = -(size * 3 / 4);
    lf.lfWeight  = FW_BOLD;
    lf.lfQuality = ANTIALIASED_QUALITY;
    lf.lfCharSet = DEFAULT_CHARSET;
    lstrcpyW(lf.lfFaceName, L"Segoe UI");
    HFONT font = CreateFontIndirectW(&lf);
    if (font != nullptr) {
        auto* old_font = static_cast<HFONT>(SelectObject(color_dc, font));
        SetBkMode(color_dc, TRANSPARENT);
        SetTextColor(color_dc, RGB(0xFF, 0xFF, 0xFF));
        DrawTextW(color_dc, L"N", 1, &rc, DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOCLIP);
        SelectObject(color_dc, old_font);
        DeleteObject(font);
    }

    // 1bpp mask: white (1) transparent, black (0) opaque.
    PatBlt(mask_dc, 0, 0, size, size, WHITENESS);
    HPEN pen         = CreatePen(PS_SOLID, 1, RGB(0, 0, 0));
    auto* old_brush  = static_cast<HBRUSH>(SelectObject(mask_dc, GetStockObject(BLACK_BRUSH)));
    auto* old_pen    = static_cast<HPEN>(SelectObject(mask_dc, pen));
    const int radius = size / 3 < 2 ? 2 : size / 3;
    RoundRect(mask_dc, 0, 0, size, size, radius, radius);
    SelectObject(mask_dc, old_brush);
    SelectObject(mask_dc, old_pen);
    DeleteObject(pen);

    SelectObject(color_dc, old_color);
    SelectObject(mask_dc, old_mask);

    ICONINFO info{};
    info.fIcon    = TRUE;
    info.hbmMask  = mask_bmp;
    info.hbmColor = color_bmp;
    HICON icon    = CreateIconIndirect(&info);

    DeleteObject(color_bmp);
    DeleteObject(mask_bmp);
    DeleteDC(color_dc);
    DeleteDC(mask_dc);
    ReleaseDC(nullptr, screen);
    return icon;
}

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
    if (msg == WM_TIMER && wparam == kTimer) {
        self->refresh_icon();
        return 0;
    }
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

TrayIcon::TrayIcon(EngineChild& child, std::string dashboard_url, bool manages_engine)
    : child_(child), dashboard_url_(std::move(dashboard_url)), manages_engine_(manages_engine) {}

TrayIcon::~TrayIcon() {
    if (hwnd_ != nullptr) { DestroyWindow(static_cast<HWND>(hwnd_)); }
    if (hicon_ != nullptr) { DestroyIcon(static_cast<HICON>(hicon_)); }
}

EngineChild& TrayIcon::child() { return child_; }

void TrayIcon::open_dashboard() const {
    ShellExecuteA(nullptr, "open", dashboard_url_.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

void TrayIcon::request_quit() {
    if (hwnd_ != nullptr) { PostMessageW(static_cast<HWND>(hwnd_), WM_CLOSE, 0, 0); }
}

TrayStatus TrayIcon::current_status() {
    const EngineStatus status = child_.status();
    if (!manages_engine_) {
        // Unmanaged: no process of ours has a state, so observed health is the
        // only thing actually measured. EngineChild keeps it current for
        // unmanaged engines through observe_health().
        if (status.health == "ok") { return TrayStatus::Working; }
        if (status.health == "unhealthy") { return TrayStatus::Failed; }
        if (status.health == "unreachable") { return TrayStatus::Pending; }
        return TrayStatus::Idle;
    }
    if (status.crash_loop_halted) { return TrayStatus::Failed; }
    switch (status.state) {
    case EngineState::Running:
        return status.health == "unhealthy" ? TrayStatus::Failed : TrayStatus::Working;
    case EngineState::Starting:
    case EngineState::Stopping:
    case EngineState::BackingOff: return TrayStatus::Pending;
    case EngineState::Halted: return TrayStatus::Failed;
    case EngineState::Stopped: break;
    }
    return TrayStatus::Idle;
}

std::wstring TrayIcon::tooltip(TrayStatus status) const {
    std::wstring tip = L"NInfer supervisor - ";
    tip += status_word(status);
    if (!manages_engine_) { tip += L" (monitor-only)"; }
    return tip;
}

void TrayIcon::refresh_icon() {
    if (hwnd_ == nullptr) { return; }
    const TrayStatus status = current_status();
    if (static_cast<int>(status) == last_status_) { return; }
    last_status_ = static_cast<int>(status);

    HICON icon = make_tray_icon(status, small_icon_size());
    if (icon == nullptr) { return; }

    const std::wstring tip = tooltip(status);
    NOTIFYICONDATAW nid{};
    nid.cbSize = sizeof(nid);
    nid.hWnd   = static_cast<HWND>(hwnd_);
    nid.uID    = kTrayUid;
    nid.uFlags = NIF_ICON | NIF_TIP;
    nid.hIcon  = icon;
    lstrcpynW(nid.szTip, tip.c_str(), ARRAYSIZE(nid.szTip));
    Shell_NotifyIconW(NIM_MODIFY, &nid);

    if (hicon_ != nullptr) { DestroyIcon(static_cast<HICON>(hicon_)); }
    hicon_ = icon;
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

    const TrayStatus status = current_status();
    last_status_            = static_cast<int>(status);
    HICON icon              = make_tray_icon(status, small_icon_size());
    hicon_                  = icon;
    const std::wstring tip  = tooltip(status);

    NOTIFYICONDATAW nid{};
    nid.cbSize           = sizeof(nid);
    nid.hWnd             = hwnd;
    nid.uID              = kTrayUid;
    nid.uFlags           = NIF_MESSAGE | NIF_TIP | NIF_ICON;
    nid.uCallbackMessage = kTrayMsg;
    nid.hIcon            = icon != nullptr ? icon : LoadIconW(nullptr, MAKEINTRESOURCEW(32512));
    lstrcpynW(nid.szTip, tip.c_str(), ARRAYSIZE(nid.szTip));
    Shell_NotifyIconW(NIM_ADD, &nid);
    SetTimer(hwnd, kTimer, 1000, nullptr);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    KillTimer(hwnd, kTimer);
    Shell_NotifyIconW(NIM_DELETE, &nid);
}

} // namespace ninfer::supervisor
