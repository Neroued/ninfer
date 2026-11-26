#pragma once

#include "ninfer/ops/moe.h"
#include "ops/common/numeric/math.cuh"

#include <cfloat>

namespace ninfer::ops::detail::moe_cuda::block_scaled {

// The caller handles the normal FP32 product. Widen only when combining finite scales in
// FP32 overflows, so a zero accumulator stays zero and small accumulators can still rescale.
__device__ __forceinline__ float scale_accumulator_wide(float value, float first, float second,
                                                        float coefficient = 1.F) {
    return static_cast<float>(static_cast<double>(value) * first * second * coefficient);
}

struct ClampedActivation {
    float limit;

    __host__ constexpr ClampedActivation(float value) : limit(value) {}

    __device__ __forceinline__ float operator()(float gate, float up) const {
        return common::silu(fminf(gate, limit)) * fmaxf(-limit, fminf(up, limit));
    }
};

struct SoftcappedActivation {
    SwiGLUSoftcap caps;
    float gate_inverse, up_inverse;

    __host__ explicit SoftcappedActivation(SwiGLUSoftcap value)
        : caps(value), gate_inverse(inverse(value.gate_cap)), up_inverse(inverse(value.up_cap)) {}

    __device__ __forceinline__ float operator()(float gate, float up) const {
        const float z       = common::exp2_approx(-fabsf(gate) * common::kLog2E);
        const float sigmoid = (gate >= 0 ? 1.F : z) * common::rcp_approx_ftz(1.F + z);
        return (cap(gate, caps.gate_cap, gate_inverse) * sigmoid) *
               cap(up, caps.up_cap, up_inverse);
    }

private:
    static float inverse(float cap) {
        const float value = 1.F / cap;
        return value >= FLT_MIN && value <= FLT_MAX ? value : 0.F;
    }

    __device__ __forceinline__ static float cap(float raw, float cap, float inverse) {
        const float argument = inverse ? raw * inverse : __fdiv_rn(raw, cap);
        float bounded;
        asm("tanh.approx.f32 %0, %1;" : "=f"(bounded) : "f"(argument));
        return fabsf(argument) < 0x1p-12F ? raw : cap * bounded;
    }
};

} // namespace ninfer::ops::detail::moe_cuda::block_scaled
