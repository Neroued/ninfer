#pragma once

#include "ops/normalization/rmsnorm/fused_launch.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail::fused_rmsnorm {

enum class Form { Paired, Residual };

template <Form F>
struct Row {
    const __nv_bfloat16 *input, *branch, *gain;
    __nv_bfloat16 *out, *residual;
    bool enabled;

    __device__ explicit Row(Arguments a, std::int64_t index) {
        if constexpr (F == Form::Paired) {
            const bool embedding = (index & 1) == 0;
            const auto column    = index / 2;
            input                = (embedding ? a.first : a.second) + column * a.dimension;
            gain                 = embedding ? a.gain_first : a.gain_second;
            enabled              = !embedding || a.enabled[column] != 0;
        } else {
            input    = a.first + index * a.dimension;
            branch   = a.second + index * a.dimension;
            residual = a.residual + index * a.dimension;
            gain     = a.gain_first;
            enabled  = true;
        }
        out = a.out + index * a.dimension;
    }
};

template <Form F, int D, int Threads, int Vector>
__global__ __launch_bounds__(Threads) void vector_kernel(Arguments a) {
    static_assert(D % (Threads * Vector) == 0);
    constexpr int Groups     = D / (Threads * Vector);
    const std::int64_t count = std::int64_t(a.columns) * (F == Form::Paired ? 2 : 1);
    __shared__ float sums[Threads / 32];
    for (std::int64_t index = blockIdx.x; index < count; index += gridDim.x) {
        const Row<F> row(a, index);
        if (!row.enabled) {
            common::Bf16Vector<Vector> zero{};
#pragma unroll
            for (int group = 0; group < Groups; ++group)
                zero.store(row.out + (group * Threads + threadIdx.x) * Vector);
            continue;
        }
        common::Bf16Vector<Vector> values[Groups], gains[Groups];
        float square = 0;
#pragma unroll
        for (int group = 0; group < Groups; ++group) {
            const int d   = (group * Threads + threadIdx.x) * Vector;
            values[group] = common::Bf16Vector<Vector>::load(row.input + d);
            gains[group]  = common::Bf16Vector<Vector>::load(row.gain + d);
            if constexpr (F == Form::Residual) {
                const auto branch = common::Bf16Vector<Vector>::load(row.branch + d);
#pragma unroll
                for (int v = 0; v < Vector; ++v)
                    values[group].word[v] = __float2bfloat16_rn(
                        __bfloat162float(values[group].word[v]) + __bfloat162float(branch.word[v]));
                values[group].store(row.residual + d);
            }
#pragma unroll
            for (int v = 0; v < Vector; ++v) {
                const float value = __bfloat162float(values[group].word[v]);
                square            = fmaf(value, value, square);
            }
        }
        square = common::warp_reduce_sum(square);
        if constexpr (Threads > 32) {
            if (threadIdx.x % 32 == 0) sums[threadIdx.x / 32] = square;
            __syncthreads();
            square = threadIdx.x % 32 < Threads / 32 ? sums[threadIdx.x % 32] : 0.F;
            square = common::warp_reduce_sum<Threads / 32>(square);
        }
        const float inv = rsqrtf(__shfl_sync(common::kFullWarpMask, square, 0) / D + a.epsilon);
#pragma unroll
        for (int group = 0; group < Groups; ++group) {
            common::Bf16Vector<Vector> output;
#pragma unroll
            for (int v = 0; v < Vector; ++v)
                output.word[v] = __float2bfloat16_rn(__bfloat162float(values[group].word[v]) *
                                                     __bfloat162float(gains[group].word[v]) * inv);
            output.store(row.out + (group * Threads + threadIdx.x) * Vector);
        }
        if (index + gridDim.x < count) __syncthreads();
    }
}

// Natural BF16 alignment and arbitrary positive D, including odd row strides.
template <Form F>
__global__ void generic_kernel(Arguments a) {
    const std::int64_t count = std::int64_t(a.columns) * (F == Form::Paired ? 2 : 1);
    __shared__ float sums[8], inverse;
    for (std::int64_t index = blockIdx.x; index < count; index += gridDim.x) {
        const Row<F> row(a, index);
        if (!row.enabled) {
            for (std::int64_t d = threadIdx.x; d < a.dimension; d += 256)
                row.out[d] = __float2bfloat16_rn(0.F);
            continue;
        }
        float square = 0;
        for (std::int64_t d = threadIdx.x; d < a.dimension; d += 256) {
            auto value = row.input[d];
            if constexpr (F == Form::Residual) {
                value =
                    __float2bfloat16_rn(__bfloat162float(value) + __bfloat162float(row.branch[d]));
                row.residual[d] = value;
            }
            const float x = __bfloat162float(value);
            square        = fmaf(x, x, square);
        }
        square = common::block_reduce_sum<256>(square, sums);
        if (threadIdx.x == 0) inverse = rsqrtf(square / a.dimension + a.epsilon);
        __syncthreads();
        for (std::int64_t d = threadIdx.x; d < a.dimension; d += 256) {
            const float x = __bfloat162float(F == Form::Residual ? row.residual[d] : row.input[d]);
            row.out[d]    = __float2bfloat16_rn(x * __bfloat162float(row.gain[d]) * inverse);
        }
        __syncthreads();
    }
}
} // namespace ninfer::ops::detail::fused_rmsnorm
