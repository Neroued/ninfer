#pragma once
#include "ops/mhc/kernel/residual.cuh"
#include "ops/common/cuda/reduce.cuh"

namespace ninfer::ops::detail::mhc {
template <int Threads, int Values>
__global__ void post_mean_norm_vector_kernel(Operands o) {
    constexpr int Groups = kHidden / (Threads * Values);
    const auto m         = std::int64_t(blockIdx.x);
    const int tid        = threadIdx.x;
    __shared__ float coeff[20], sums[Threads / 32], inverse;
    if (tid < 4) coeff[tid] = o.post[m * 4 + tid];
    if (tid < 16) coeff[4 + tid] = o.mix[m * 16 + tid];
    __syncthreads();
    float values[Groups][Values], square = 0;
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
        common::Bf16Vector<Values> residual[4];
        mixed_vectors(o, m, (group * Threads + tid) * Values, coeff, coeff + 4, residual);
#pragma unroll
        for (int v = 0; v < Values; ++v) {
            float mean = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) mean += __bfloat162float(residual[j].word[v]);
            values[group][v] = mean * 0.25F;
            square           = fmaf(values[group][v], values[group][v], square);
        }
    }
    square = common::block_reduce_sum<Threads>(square, sums);
    if (tid == 0) inverse = rsqrtf(square / kHidden + o.parameters.branch_epsilon);
    __syncthreads();
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
        const int d = (group * Threads + tid) * Values;
        const auto gamma =
            common::Bf16Vector<Values>::load(static_cast<const __nv_bfloat16*>(o.gamma) + d);
        common::Bf16Vector<Values> out;
#pragma unroll
        for (int v = 0; v < Values; ++v)
            out.word[v] =
                __float2bfloat16_rn(__bfloat162float(gamma.word[v]) * values[group][v] * inverse);
        out.store(static_cast<__nv_bfloat16*>(o.output) + m * kHidden + d);
    }
}

} // namespace ninfer::ops::detail::mhc
