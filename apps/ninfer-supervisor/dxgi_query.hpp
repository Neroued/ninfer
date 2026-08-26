#pragma once

// Cached DXGI QueryVideoMemoryInfo. CUDA is used at most once to bind an
// adapter LUID; the 10 Hz path is DXGI-only. Every HRESULT is checked. Failure
// degrades to ok=false rather than throwing or aborting. The factory is
// recreated only after a failed query so a TDR/device-removed does not take
// the process down.

#include <dxgi1_4.h>
#include <windows.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

namespace ninfer::supervisor {

struct DxgiSnapshot {
    std::uint64_t budget_bytes                    = 0;
    std::uint64_t current_usage_bytes             = 0;
    std::uint64_t available_for_reservation_bytes = 0;
    std::uint64_t current_reservation_bytes       = 0;
    std::string adapter_name;
    bool ok = false;
    std::string error;
};

class DxgiBudgetSource {
public:
    DxgiBudgetSource() = default;
    ~DxgiBudgetSource() { reset(); }

    DxgiBudgetSource(const DxgiBudgetSource&)            = delete;
    DxgiBudgetSource& operator=(const DxgiBudgetSource&) = delete;

    DxgiSnapshot query(int device_index) {
        DxgiSnapshot out;
        try {
            std::lock_guard lock(mu_);
            if (!query_locked(device_index, out)) {
                reset_locked();
                query_locked(device_index, out);
            }
        } catch (...) {
            out = DxgiSnapshot{};
            out.error = "DXGI query threw";
            try {
                std::lock_guard lock(mu_);
                reset_locked();
            } catch (...) {}
        }
        return out;
    }

    void reset() {
        std::lock_guard lock(mu_);
        reset_locked();
    }

private:
    static constexpr UINT kNvidia = 0x10DE;

    void reset_locked() {
        if (adapter3_ != nullptr) {
            adapter3_->Release();
            adapter3_ = nullptr;
        }
        if (factory_ != nullptr) {
            factory_->Release();
            factory_ = nullptr;
        }
        bound_device_ = -1;
        adapter_name_.clear();
    }

    bool bind_locked(int device_index, DxgiSnapshot& out) {
        IDXGIFactory1* factory = nullptr;
        HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory));
        if (FAILED(hr) || factory == nullptr) {
            out.error = "CreateDXGIFactory1 failed hr=" + std::to_string(static_cast<unsigned>(hr));
            return false;
        }

        // Do not call CUDA here. cudaGetDeviceProperties during a TDR can abort
        // the process (0xC0000409), which is the failure mode we exist to record.
        UINT index = 0;
        int gpu_seen = 0;
        IDXGIAdapter3* chosen = nullptr;
        std::string chosen_name;
        for (;;) {
            IDXGIAdapter1* a1 = nullptr;
            hr                = factory->EnumAdapters1(index, &a1);
            ++index;
            if (hr == DXGI_ERROR_NOT_FOUND) { break; }
            if (FAILED(hr) || a1 == nullptr) { break; }
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(a1->GetDesc1(&desc))) {
                a1->Release();
                continue;
            }
            if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
                a1->Release();
                continue;
            }
            if (desc.VendorId != kNvidia || gpu_seen++ != device_index) {
                a1->Release();
                continue;
            }
            IDXGIAdapter3* a3 = nullptr;
            hr = a1->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&a3));
            a1->Release();
            if (FAILED(hr) || a3 == nullptr) { continue; }
            char name[128]{};
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name,
                                static_cast<int>(sizeof(name)), nullptr, nullptr);
            chosen      = a3;
            chosen_name = name;
            break;
        }
        if (chosen == nullptr) {
            factory->Release();
            out.error = "no DXGI adapter matched";
            return false;
        }
        factory_      = factory;
        adapter3_     = chosen;
        adapter_name_ = chosen_name;
        bound_device_ = device_index;
        return true;
    }

    bool query_locked(int device_index, DxgiSnapshot& out) {
        if (factory_ == nullptr || adapter3_ == nullptr || bound_device_ != device_index) {
            if (!bind_locked(device_index, out)) { return false; }
        }
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        const HRESULT hr =
            adapter3_->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info);
        if (FAILED(hr)) {
            out.error = "QueryVideoMemoryInfo hr=" + std::to_string(static_cast<unsigned>(hr));
            return false;
        }
        out.budget_bytes                    = info.Budget;
        out.current_usage_bytes             = info.CurrentUsage;
        out.available_for_reservation_bytes = info.AvailableForReservation;
        out.current_reservation_bytes       = info.CurrentReservation;
        out.adapter_name                    = adapter_name_;
        out.ok                              = true;
        out.error.clear();
        return true;
    }

    std::mutex mu_;
    IDXGIFactory1* factory_  = nullptr;
    IDXGIAdapter3* adapter3_ = nullptr;
    int bound_device_        = -1;
    std::string adapter_name_;
};

inline DxgiSnapshot query_dxgi_local(int cuda_device) {
    static DxgiBudgetSource source;
    return source.query(cuda_device);
}

} // namespace ninfer::supervisor
