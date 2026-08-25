#include "ops/common/device_info.h"

#include <cstdio>
#include <cuda_runtime.h>

namespace ninfer::ops {
namespace {

constexpr int kReferenceSmCount = 170; // RTX 5090

int query_sm_count() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return kReferenceSmCount; }
    int count = 0;
    if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        return kReferenceSmCount;
    }
    if (count <= 0) { return kReferenceSmCount; }
    std::fprintf(stderr, "ninfer: persistent grids sized for %d SMs\n", count);
    return count;
}

} // namespace

int device_sm_count() {
    static const int count = query_sm_count();
    return count;
}

} // namespace ninfer::ops
