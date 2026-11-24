#pragma once

#include "ninfer/kv_cache_format.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/quant/fp8.cuh"

namespace ninfer::ops::detail::kv {

inline constexpr int kFp8CacheGroup =
    kv_cache_format_info(KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0).group_size;

// One warp holds a complete vector, with feature lane+32*j in values[j].
// The caller owns its basis transform and provides separate contiguous planes.
template <bool DirectQuantization = false>
__device__ __forceinline__ void store_fp8_d256(const float (&values)[8], std::uint8_t* codes,
                                               std::uint8_t* scales, int lane) {
    constexpr int Group = kFp8CacheGroup;
#pragma unroll
    for (int group = 0; group < 256 / Group; ++group) {
        float maximum = 0;
#pragma unroll
        for (int j = 0; j < Group / 32; ++j)
            maximum = fmaxf(maximum, fabsf(values[group * (Group / 32) + j]));
        // Nonnegative finite FP32 values have the same unsigned-bit ordering. Native
        // warp reduction keeps two G128 maxima from doubling the shuffle sequence.
        const auto absmax =
            __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(maximum)));
        const auto quant = common::fp8_ue8m0_scale(absmax);
#pragma unroll
        for (int j = 0; j < Group / 32; ++j) {
            const float value = values[group * (Group / 32) + j];
            if constexpr (DirectQuantization)
                // UE8M0 always produces a positive inverse, including all-zero groups.
                codes[group * Group + lane + 32 * j] =
                    __nv_cvt_float_to_fp8(value * quant.inverse_scale, __NV_SATFINITE, __NV_E4M3);
            else
                codes[group * Group + lane + 32 * j] =
                    common::fp8_quant_code(value, quant.inverse_scale);
        }
        if (lane == 0) scales[group] = quant.scale;
    }
}

} // namespace ninfer::ops::detail::kv
