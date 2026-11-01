#pragma once

#include "ops/gated_residual/mix/operands.h"
#include "ops/common/numeric/math.cuh"
#include "ops/linear/kernels/bf16/bf16_a16_gemv.cuh"

namespace ninfer::ops::detail::hc::mix {

// Eight lanes cover one native 320-element row without a padded tensor-core tile.
// Each group finishes all four streams locally, including the final mean.
__global__ __launch_bounds__(128) void up_gemv_mix_kernel(Operands o,
                                                          const __nv_bfloat16* activation) {
    constexpr int Group = 8, Features = 16;
    const int d    = int(blockIdx.x) * Features + int(threadIdx.x) / Group;
    const int lane = threadIdx.x % Group;
    __shared__ __align__(16) __nv_bfloat16 u[kRank];
    if (threadIdx.x < kRank / 8)
        common::store_vec(u + threadIdx.x * 8,
                          common::load_vec<uint4>(activation + threadIdx.x * 8));
    __syncthreads();
    float sum = 0;
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        float accum[4]{};
#pragma unroll
        for (int k = lane * 8; k < kRank; k += Group * 8) {
            const auto w = load_bf16_weight_pack<Bf16WeightCache::Default, 8>(
                static_cast<const __nv_bfloat16*>(o.up) + (s * kWidth + d) * kRank + k);
            accumulate_bf16_packs(w, load_bf16_pack<8>(u + k), accum);
        }
        const float dot = common::warp_reduce_sum<Group>(accum[0] + accum[1] + accum[2] + accum[3]);
        if (lane == 0) {
            const float z =
                __bfloat162float(static_cast<const __nv_bfloat16*>(o.normalized)[s * kWidth + d]);
            sum += common::sigmoid_approx(dot) * z;
        }
    }
    if (lane == 0) static_cast<__nv_bfloat16*>(o.output)[d] = __float2bfloat16_rn(sum * 0.25F);
}

} // namespace ninfer::ops::detail::hc::mix
