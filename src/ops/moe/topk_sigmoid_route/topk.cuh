#pragma once
#include "ops/moe/kernels/cuda/routing/topk.cuh"

namespace ninfer::ops::detail::moe_cuda::routing {
template <int Warps>
__global__ __launch_bounds__(Warps * 32) void sigmoid_correction_top8(
    const float* logits, const float* bias, int* ids, float* coefficients, int tokens,
    float scaling_factor, float normalization_epsilon) {
    const int token = blockIdx.x * Warps + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (token >= tokens) return;
    const float* scores = logits + std::int64_t(token) * 288;
    float value[9];
#pragma unroll
    for (int j = 0; j < 9; ++j)
        value[j] = common::sigmoid(scores[lane + 32 * j]) + bias[lane + 32 * j];
    int chosen[8];
    float numerator[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        RankedExpert best{-INFINITY, INT_MAX};
#pragma unroll
        for (int j = 0; j < 9; ++j) {
            RankedExpert v{value[j], lane + 32 * j};
            if (better(v, best)) best = v;
        }
#pragma unroll
        for (int offset = 16; offset; offset /= 2) {
            RankedExpert other{__shfl_down_sync(0xffffffff, best.score, offset),
                               __shfl_down_sync(0xffffffff, best.id, offset)};
            if (lane + offset < 32 && better(other, best)) best = other;
        }
        const int selected = __shfl_sync(0xffffffff, best.id, 0);
        chosen[k]          = selected;
        numerator[k]       = common::sigmoid(scores[selected]);
#pragma unroll
        for (int j = 0; j < 9; ++j)
            if (lane + 32 * j == selected) value[j] = -INFINITY;
    }
    float den = normalization_epsilon;
#pragma unroll
    for (int k = 0; k < 8; ++k) den += numerator[k];
    if (lane < 8) {
        ids[std::int64_t(token) * 8 + lane]          = chosen[lane];
        coefficients[std::int64_t(token) * 8 + lane] = scaling_factor * numerator[lane] / den;
    }
}
} // namespace ninfer::ops::detail::moe_cuda::routing
