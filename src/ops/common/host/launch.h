#pragma once

#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::common {

// AlwaysDynamic is for kernels that declare extern shared storage at every size.
// Their static shared storage also counts against the default limit, so explicitly
// admit the requested dynamic capacity even when it alone fits within 48 KiB.
template <int Bytes, auto Kernel, bool AlwaysDynamic = false>
int prepare_shared() {
    static_assert(Bytes >= 0 && Bytes <= 99 * 1024);
    if constexpr (Bytes > 48 * 1024 || (AlwaysDynamic && Bytes > 0)) {
        static const cudaError_t status =
            cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes);
        CUDA_CHECK(status);
        return Bytes;
    } else {
        return AlwaysDynamic ? Bytes : 0;
    }
}

// Slice only when the selected kernel would exceed grid.y; this is not model chunking.
inline constexpr std::int32_t kCudaGridYLimit = 65'535;

template <class Launch>
void for_each_token_slice(std::int32_t tokens, std::int32_t columns_per_block, Launch&& launch) {
    if (tokens <= 0 || columns_per_block <= 0) {
        throw std::invalid_argument("token slicing requires positive dimensions");
    }
    const std::int64_t capacity = static_cast<std::int64_t>(columns_per_block) * kCudaGridYLimit;
    std::int32_t offset         = 0;
    while (offset < tokens) {
        const std::int32_t count = static_cast<std::int32_t>(
            std::min<std::int64_t>(capacity, static_cast<std::int64_t>(tokens) - offset));
        launch(offset, count);
        offset += count;
    }
}

} // namespace ninfer::ops::common
