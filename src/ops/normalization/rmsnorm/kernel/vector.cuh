#pragma once
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"

namespace ninfer::ops::detail::rmsnorm {
// Fixed-width plain RMSNorm retains input and gain in packed registers across reduction.
template <int D, int Threads>
__global__ __launch_bounds__(Threads) void rmsnorm_vector_kernel(const __nv_bfloat16* x,
                                                                 const __nv_bfloat16* w,
                                                                 __nv_bfloat16* y, float eps) {
    constexpr int Groups = D / (Threads * 8);
    static_assert(Groups > 0 && D % (Threads * 8) == 0);
    auto row = std::int64_t(blockIdx.x);
    common::Bf16Vector<8> values[Groups], weights[Groups];
    float a = 0, b = 0;
#pragma unroll
    for (int g = 0; g < Groups; ++g) {
        int d      = (g * Threads + threadIdx.x) * 8;
        values[g]  = common::Bf16Vector<8>::load(x + row * D + d);
        weights[g] = common::Bf16Vector<8>::load(w + d);
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            float u = __bfloat162float(values[g].word[j]),
                  v = __bfloat162float(values[g].word[j + 1]);
            a       = fmaf(u, u, a);
            b       = fmaf(v, v, b);
        }
    }
    __shared__ float sums[Threads / 32];
    float sum = common::block_reduce_sum<Threads>(a + b, sums);
    if (threadIdx.x == 0) sums[0] = rsqrtf(sum / float(D) + eps);
    __syncthreads();
    float inv = sums[0];
#pragma unroll
    for (int g = 0; g < Groups; ++g) {
        common::Bf16Vector<8> out;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            out.word[j] = __float2bfloat16_rn(__bfloat162float(values[g].word[j]) * inv *
                                              __bfloat162float(weights[g].word[j]));
        out.store(y + row * D + (g * Threads + threadIdx.x) * 8);
    }
}
} // namespace ninfer::ops::detail::rmsnorm
