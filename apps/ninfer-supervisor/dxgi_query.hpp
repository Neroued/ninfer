#pragma once

// Lifted from tools/vram-probes/common.hpp: DXGI adapter match by CUDA LUID
// and QueryVideoMemoryInfo. No D3D device is created.

#include <cuda_runtime.h>
#include <dxgi1_4.h>
#include <windows.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

namespace ninfer::supervisor {

struct DxgiSnapshot {
    std::uint64_t budget_bytes                   = 0;
    std::uint64_t current_usage_bytes            = 0;
    std::uint64_t available_for_reservation_bytes = 0;
    std::uint64_t current_reservation_bytes      = 0;
    std::string adapter_name;
    bool ok = false;
    std::string error;
};

inline DxgiSnapshot query_dxgi_local(int cuda_device) {
    DxgiSnapshot out;
    cudaDeviceProp prop{};
    const cudaError_t cu = cudaGetDeviceProperties(&prop, cuda_device);
    if (cu != cudaSuccess) {
        out.error = std::string("cudaGetDeviceProperties: ") + cudaGetErrorString(cu);
        return out;
    }

    IDXGIFactory1* factory = nullptr;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory));
    if (FAILED(hr) || factory == nullptr) {
        out.error = "CreateDXGIFactory1 failed";
        return out;
    }

    UINT index = 0;
    for (;;) {
        IDXGIAdapter1* adapter1 = nullptr;
        hr                      = factory->EnumAdapters1(index, &adapter1);
        if (hr == DXGI_ERROR_NOT_FOUND) { break; }
        if (FAILED(hr) || adapter1 == nullptr) { break; }
        ++index;
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter1->GetDesc1(&desc))) {
            adapter1->Release();
            continue;
        }
        if (std::memcmp(&desc.AdapterLuid, prop.luid, sizeof(LUID)) != 0) {
            adapter1->Release();
            continue;
        }
        IDXGIAdapter3* adapter3 = nullptr;
        hr = adapter1->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter3));
        if (SUCCEEDED(hr) && adapter3 != nullptr) {
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                                                         &info))) {
                out.budget_bytes                    = info.Budget;
                out.current_usage_bytes             = info.CurrentUsage;
                out.available_for_reservation_bytes = info.AvailableForReservation;
                out.current_reservation_bytes       = info.CurrentReservation;
                char name[128]{};
                WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name,
                                    static_cast<int>(sizeof(name)), nullptr, nullptr);
                out.adapter_name = name;
                out.ok           = true;
            }
            adapter3->Release();
        }
        adapter1->Release();
        break;
    }
    factory->Release();
    if (!out.ok && out.error.empty()) { out.error = "no DXGI adapter LUID matched the CUDA device"; }
    return out;
}

} // namespace ninfer::supervisor
