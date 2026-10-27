#pragma once

#include "ops/gated_residual/mix/bf16/operands.h"
#include "ops/common/numeric/math.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail::hc::mix::bf16 {
template <Kind Form>
struct DownOutput {
    __nv_bfloat16* activation;
    float* injection;

    // Public FP32 injection uses expf sigmoid; BF16 mixing uses the tanh approximation.
    __device__ __forceinline__ void store(int row, int token, float dot) const {
        if constexpr (Form == Kind::Block) {
            if (row >= kRank) {
                injection[std::int64_t(token) * kStreams + row - kRank] =
                    2.F * common::sigmoid(dot * 0.25F);
                return;
            }
        }
        const float a = dot * 0.25F;
        activation[std::int64_t(token) * kRank + row] =
            __float2bfloat16_rn(a * common::sigmoid_approx(a));
    }
};

// MMA lanes vary eight rows and four token pairs. Separating token parity and grouping
// eight rows avoids the eight-way conflicts of a row-major contribution tile.
template <int Rows, int Tokens>
__device__ __forceinline__ int contribution_index(int row, int token) {
    static_assert(Rows % 8 == 0 && Tokens % 2 == 0);
    return ((row / 8) * (Tokens / 2) + token / 2) * 8 + row % 8 + (token & 1) * (Rows * Tokens / 2);
}

} // namespace ninfer::ops::detail::hc::mix::bf16
