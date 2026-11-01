#pragma once
#include "ops/mhc/implementation.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/math.cuh"
#include <cuda_bf16.h>

namespace ninfer::ops::detail::mhc {
// Each warp reduces a disjoint subset of the K partitions, preserving coalesced reads.
// The first warp receives all 24 logits and the full residual square sum.
template <int Threads, int StaticParts = 0>
__device__ __forceinline__ float reduce_projection(const float* partial, int parts,
                                                   std::int64_t m) {
    constexpr int Warps = Threads / 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    __shared__ float sums[Warps][32];
    float value = 0;
    if constexpr (StaticParts > 0) {
        // Issue the independent partial reads together for a fixed partition count.
        constexpr int Count = (StaticParts + Warps - 1) / Warps;
        float chunks[Count] = {};
        if (lane <= 24) {
#pragma unroll
            for (int i = 0; i < Count; ++i) {
                const int p = warp + i * Warps;
                if (p < StaticParts) chunks[i] = partial[(m * StaticParts + p) * 32 + lane];
            }
        }
#pragma unroll
        for (int i = 0; i < Count; ++i) value += chunks[i];
    } else if (lane <= 24) {
        for (int p = warp; p < parts; p += Warps) value += partial[(m * parts + p) * 32 + lane];
    }
    sums[warp][lane] = value;
    __syncthreads();
    value = 0;
    if (warp == 0) {
#pragma unroll
        for (int w = 0; w < Warps; ++w) value += sums[w][lane];
    }
    return value;
}

// A warp holds [pre:4 | post:4 | mix:16 | square sum] after the projection reduction.
// The returned lane value is a mix logit, consumed by sinkhorn after gates are published.
__device__ __forceinline__ float activate_coefficients(Operands o, float value, std::int64_t m,
                                                       float* pre, bool publish_post = true) {
    const int lane = threadIdx.x & 31;
    const float r  = rsqrtf(__shfl_sync(0xffffffffu, value, 24) / float(kWidth) +
                            o.parameters.coefficient_epsilon);
    float logit    = 0;
    if (lane < 24) {
        const int group = lane < 4 ? 0 : lane < 8 ? 1 : 2;
        logit = fmaf(__bfloat162float(static_cast<const __nv_bfloat16*>(o.scale)[group]), value * r,
                     __bfloat162float(static_cast<const __nv_bfloat16*>(o.base)[lane]));
    }
    if (lane < 4)
        pre[lane] =
            common::rcp_approx_ftz(1.F + common::exp_approx_ftz(-logit)) + o.parameters.pre_epsilon;
    if (publish_post && lane >= 4 && lane < 8)
        o.next_post[m * 4 + lane - 4] =
            o.parameters.post_multiplier *
            common::rcp_approx_ftz(1.F + common::exp_approx_ftz(-logit));
    return __shfl_sync(0xffffffffu, logit, (lane + 8) & 31);
}

template <int FixedIterations = 0>
__device__ __forceinline__ void sinkhorn(Operands o, float logit, std::int64_t m) {
    const int lane = threadIdx.x & 31;
    float values[4];
#pragma unroll
    for (int row = 0; row < 4; ++row) {
        const float z = __shfl_sync(0xffffffffu, logit, row * 4 + (lane & 3));
        values[row]   = common::exp_approx_ftz(z - common::warp_max<4>(z));
    }
#pragma unroll
    for (int row = 0; row < 4; ++row)
        values[row] = values[row] * common::rcp_approx_ftz(common::warp_sum<4>(values[row])) +
                      o.parameters.sinkhorn_epsilon;
    // Four rows per lane expose independent row reductions; column sums stay in registers.
    const auto normalize_columns = [&] {
        const float sum     = (values[0] + values[1]) + (values[2] + values[3]);
        const float inverse = common::rcp_approx_ftz(sum + o.parameters.sinkhorn_epsilon);
#pragma unroll
        for (int row = 0; row < 4; ++row) values[row] *= inverse;
    };
    normalize_columns();
    const auto round = [&] {
        float sums[4];
#pragma unroll
        for (int row = 0; row < 4; ++row) sums[row] = common::warp_sum<4>(values[row]);
#pragma unroll
        for (int row = 0; row < 4; ++row)
            values[row] *= common::rcp_approx_ftz(sums[row] + o.parameters.sinkhorn_epsilon);
        normalize_columns();
    };
    if constexpr (FixedIterations > 0) {
#pragma unroll
        for (int iteration = 1; iteration < FixedIterations; ++iteration) round();
    } else {
        for (int iteration = 1; iteration < o.parameters.sinkhorn_iterations; ++iteration) round();
    }
    if (lane < 16) o.next_mix[m * 16 + lane] = values[lane / 4];
}
} // namespace ninfer::ops::detail::mhc
