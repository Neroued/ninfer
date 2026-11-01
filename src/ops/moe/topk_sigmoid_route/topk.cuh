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
    float value[9], probability[9];
#pragma unroll
    for (int j = 0; j < 9; ++j) {
        probability[j] = common::sigmoid(scores[lane + 32 * j]);
        value[j]       = probability[j] + bias[lane + 32 * j];
    }
    int chosen      = 0;
    float numerator = 0, den = normalization_epsilon;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        RankedExpert best{-INFINITY, INT_MAX};
        float best_probability = 0;
#pragma unroll
        for (int j = 0; j < 9; ++j) {
            RankedExpert v{value[j], lane + 32 * j};
            if (better(v, best)) {
                best             = v;
                best_probability = probability[j];
            }
        }
        const auto score   = common::ordered_score_bits(best.score);
        const auto maximum = __reduce_max_sync(0xffffffff, score);
        const int selected = __reduce_min_sync(0xffffffff, score == maximum ? best.id : INT_MAX);
        const float selected_probability = __shfl_sync(0xffffffff, best_probability, selected & 31);
        den += selected_probability;
        if (lane == k) {
            chosen    = selected;
            numerator = selected_probability;
        }
#pragma unroll
        for (int j = 0; j < 9; ++j)
            if (lane + 32 * j == selected) value[j] = -INFINITY;
    }
    if (lane < 8) {
        ids[std::int64_t(token) * 8 + lane]          = chosen;
        coefficients[std::int64_t(token) * 8 + lane] = scaling_factor * numerator / den;
    }
}
} // namespace ninfer::ops::detail::moe_cuda::routing
