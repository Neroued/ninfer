#pragma once

#include "ops/common/numeric/math.cuh"
#include "ops/common/quant/nvfp4.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4 {

struct ActivationQuantizer {
    float multiplier;
    float inverse;

    __device__ explicit ActivationQuantizer(float scale)
        : multiplier(scale),
          inverse(scale >= 0x1p-100F && scale <= 0x1p100F ? common::rcp_approx_ftz(scale) : 0.F) {}

    __device__ __forceinline__ std::uint8_t scale(float maximum) const {
        return inverse ? __nv_cvt_float_to_fp8(maximum * inverse * (1.F / 6.F), __NV_SATFINITE,
                                               __NV_E4M3)
                       : common::nvfp4_mul_activation_scale(maximum, multiplier);
    }

    __device__ __forceinline__ float factor(std::uint8_t block_scale) const {
        return block_scale ? inverse * common::rcp_approx_ftz(common::decode_e4m3(block_scale)) : 0;
    }

    __device__ __forceinline__ unsigned pair(float2 values, std::uint8_t block_scale) const {
        if (!inverse)
            return common::quantize_nvfp4_mul_activation_pair(values, multiplier, block_scale);
        const auto scale_inverse = factor(block_scale);
        return __nv_cvt_float2_to_fp4x2(
            make_float2(values.x * scale_inverse, values.y * scale_inverse), __NV_E2M1,
            cudaRoundNearest);
    }

    __device__ __forceinline__ common::Nvfp4QuantizedK16 block(const __nv_bfloat16* source) const {
        if (!inverse) return common::quantize_nvfp4_mul_k16(source, multiplier);
        const uint4 lo = common::load_vec<uint4>(source), hi = common::load_vec<uint4>(source + 8);
        const unsigned packed[8]{lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        float2 values[8];
        float maximum = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            values[i] = common::bf16x2_bits_to_float2(packed[i]);
            maximum   = fmaxf(maximum, fmaxf(fabsf(values[i].x), fabsf(values[i].y)));
        }
        common::Nvfp4QuantizedK16 result{};
        result.scale              = scale(maximum);
        const float scale_inverse = factor(result.scale);
#pragma unroll
        for (auto& value : values) {
            value.x *= scale_inverse;
            value.y *= scale_inverse;
        }
        common::pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
        return result;
    }
};

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4
