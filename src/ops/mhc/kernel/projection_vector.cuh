#pragma once

#include "ops/mhc/kernel/coefficients.cuh"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail::mhc {
__device__ __forceinline__ uint4 projection_vector_load(const __nv_bfloat16* source) {
    uint4 value;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(source));
    return value;
}

// Single-column projection reads contiguous segments of native F rows. Each
// row-zero CTA also owns its residual square sum and any exact expansion stores.
template <Form F, int Parts, int Threads>
__global__ __launch_bounds__(Threads) void projection_vector_kernel(Operands o, float* partial) {
    constexpr int Chunk = kWidth / Parts, Warps = Threads / 32;
    static_assert(F == Form::Pre || F == Form::ExpandPre);
    static_assert(Chunk == Threads * 8 && kWidth % Parts == 0);
    cudaTriggerProgrammaticLaunchCompletion();
    const int part = blockIdx.x, row = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32, k = part * Chunk + tid * 8;
    const auto fw =
        projection_vector_load(static_cast<const __nv_bfloat16*>(o.projection) + row * kWidth + k);
    const auto xw = projection_vector_load(static_cast<const __nv_bfloat16*>(o.residual) +
                                           (F == Form::Pre ? k : k % kHidden));
    if constexpr (F == Form::ExpandPre) {
        if (row == 0) {
            auto* target = static_cast<__nv_bfloat16*>(o.updated) + k;
            asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(target), "r"(xw.x),
                         "r"(xw.y), "r"(xw.z), "r"(xw.w)
                         : "memory");
        }
    }
    const unsigned f[4] = {fw.x, fw.y, fw.z, fw.w}, x[4] = {xw.x, xw.y, xw.z, xw.w};
    float sum = 0, square = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const auto weight = common::bf16x2_bits_to_float2(f[i]);
        const auto value  = common::bf16x2_bits_to_float2(x[i]);
        sum               = fmaf(weight.x, value.x, sum);
        sum               = fmaf(weight.y, value.y, sum);
        if (row == 0) {
            square = fmaf(value.x, value.x, square);
            square = fmaf(value.y, value.y, square);
        }
    }
    __shared__ float sums[Warps], squares[Warps];
    sum = common::warp_reduce_sum(sum);
    if (lane == 0) sums[warp] = sum;
    if (row == 0) {
        square = common::warp_reduce_sum(square);
        if (lane == 0) squares[warp] = square;
    }
    __syncthreads();
    if (warp == 0) {
        sum = common::warp_reduce_sum<Warps>(lane < Warps ? sums[lane] : 0.F);
        if (lane == 0) partial[part * 32 + row] = sum;
        if (row == 0) {
            square = common::warp_reduce_sum<Warps>(lane < Warps ? squares[lane] : 0.F);
            if (lane == 0) partial[part * 32 + 24] = square;
        }
    }
}
} // namespace ninfer::ops::detail::mhc
