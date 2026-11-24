#pragma once

#include "ops/common/numeric/math.cuh"
#include "ops/common/cuda/reduce.cuh"
#include <math_constants.h>
#include <limits>

namespace ninfer::ops::detail {

inline constexpr float kAttentionNegativeInfinity = -std::numeric_limits<float>::infinity();

// Keep the maximum in raw QK units in the mainloop, allowing scale to fold into
// the exponent's FMA. Exported partial maxima are always in log2 units.
struct OnlineSoftmaxRow {
    float maximum;
    float sum;
    float scaled_maximum;

    __device__ OnlineSoftmaxRow()
        : maximum(kAttentionNegativeInfinity), sum(0.0f), scaled_maximum(0.0f) {}

    template <bool MayBeEmpty = true>
    __device__ __forceinline__ float update(float tile_maximum, float scale_log2) {
        const float previous = maximum;
        maximum              = fmaxf(maximum, tile_maximum);
        scaled_maximum       = maximum * scale_log2;
        if constexpr (MayBeEmpty)
            if (maximum == kAttentionNegativeInfinity) scaled_maximum = 0.0f;
        const float alpha = common::exp2_approx(__fmaf_rn(previous, scale_log2, -scaled_maximum));
        return alpha;
    }

    __device__ __forceinline__ void accumulate(float alpha, float tile_sum) {
        sum = __fmaf_rn(sum, alpha, tile_sum);
    }

    __device__ __forceinline__ float probability(float score, float scale_log2) const {
        return common::exp2_approx(__fmaf_rn(score, scale_log2, -scaled_maximum));
    }

    __device__ __forceinline__ void finish() { sum = common::warp_sum<4>(sum, 0xffffffffu); }
};

__device__ __forceinline__ float softmax_log2_state_weight(float maximum, float sum,
                                                           float combined) {
    return sum > 0.0f ? common::exp2_approx(maximum - combined) : 0.0f;
}

// Grouped scores already include attention_scale; tiled scores fold it into exp2.
__device__ __forceinline__ float causal_exp_difference(float score, float maximum, float scale) {
    return common::exp2_approx((score - maximum) * scale);
}

__device__ __forceinline__ float causal_exp_scaled(float score, float scaled_maximum, float scale) {
    return common::exp2_approx(__fmaf_rn(score, scale, -scaled_maximum));
}

} // namespace ninfer::ops::detail
