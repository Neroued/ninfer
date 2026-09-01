#include "ops/linear_add/bf16/bf16_linear_add_plan.h"

#include "core/device.h"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

inline constexpr int kRows       = 5120;
inline constexpr int kColumns    = 6144;
inline constexpr int kTokenTile  = 4;
inline constexpr int kThreads    = 256;

__launch_bounds__(kThreads, 2) __global__ void bf16_linear_add_volta_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ residual, int tokens) {
    __shared__ float warp_sums[kThreads / kWarpSize];

    const int row         = static_cast<int>(blockIdx.x);
    const int token_begin = static_cast<int>(blockIdx.y) * kTokenTile;
    const int tid         = static_cast<int>(threadIdx.x);
    const int active      = min(kTokenTile, tokens - token_begin);

    float accumulators[kTokenTile] = {};
    for (int column = tid; column < kColumns; column += kThreads) {
        const float w = __bfloat162float(
            weight[static_cast<std::int64_t>(row) * kColumns + column]);
#pragma unroll
        for (int local = 0; local < kTokenTile; ++local) {
            if (local < active) {
                const int token = token_begin + local;
                const float a = __bfloat162float(
                    x[static_cast<std::int64_t>(token) * kColumns + column]);
                accumulators[local] = fmaf(a, w, accumulators[local]);
            }
        }
    }

#pragma unroll
    for (int local = 0; local < kTokenTile; ++local) {
        if (local < active) {
            const float sum = block_reduce_sum<kThreads>(accumulators[local], warp_sums);
            if (tid == 0) {
                const std::int64_t offset =
                    static_cast<std::int64_t>(token_begin + local) * kRows + row;
                residual[offset] = __float2bfloat16_rn(
                    sum + __bfloat162float(residual[offset]));
            }
        }
    }
}

} // namespace

void bf16_linear_add_volta_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  cudaStream_t stream) {
    const dim3 grid(kRows, static_cast<unsigned>((x.ne[1] + kTokenTile - 1) / kTokenTile), 1u);
    bf16_linear_add_volta_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<__nv_bfloat16*>(residual.data), x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
