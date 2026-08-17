#pragma once

#include "core/device.h"

namespace ninfer::ops::detail {

// Multiprocessor count of the current device, cached for the process. The
// engine owns one GPU and one resident model, so one cached query is the
// correct scope; launchers consult it instead of assuming a fixed GPU.
inline int device_sm_count() {
    static const int cached = [] {
        int device          = 0;
        cuda_check(cudaGetDevice(&device), "cudaGetDevice", __FILE__, __LINE__);
        int multiprocessors = 0;
        cuda_check(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device),
                   "cudaDeviceGetAttribute", __FILE__, __LINE__);
        return multiprocessors;
    }();
    return cached;
}

} // namespace ninfer::ops::detail
