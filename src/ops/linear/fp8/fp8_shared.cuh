#pragma once
#include "core/device.h"

namespace ninfer::ops::detail {
template <int Bytes>
__device__ __forceinline__ unsigned char* fp8_shared_storage() {
    static_assert(Bytes > 0 && Bytes <= 99 * 1024);
    if constexpr (Bytes > 48 * 1024) {
        extern __shared__ __align__(16) unsigned char shared_dynamic[];
        return shared_dynamic;
    } else {
        __shared__ __align__(16) unsigned char shared_static[Bytes];
        return shared_static;
    }
}

template <int Bytes, auto Kernel, bool Dynamic = false>
int fp8_prepare_shared() {
    static_assert(Bytes <= 99 * 1024);
    if constexpr (Bytes > 48 * 1024) {
        static const cudaError_t status =
            cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes);
        CUDA_CHECK(status);
        return Bytes;
    } else {
        return Dynamic ? Bytes : 0;
    }
}

template <class Schedule, class Epilogue>
inline constexpr int fp8_mma_shared_bytes = [] {
    constexpr int extra = [] {
        if constexpr (requires { Epilogue::template kSharedBytes<Schedule>; })
            return Epilogue::template kSharedBytes<Schedule>;
        else
            return 0;
    }();
    return Schedule::kSharedBytes > extra ? Schedule::kSharedBytes : extra;
}();
} // namespace ninfer::ops::detail
