#pragma once

#include "ops/rope/kernels/axial64.h"
#include "ops/common/cuda/reduce.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {

// Each half warp owns a complete D64 head. A CTA shares the two axes' phases
// across its heads; the optional norm retains the unrounded values in registers.
template <bool Normalize, int HeadsPerBlock>
__global__ __launch_bounds__(16 * HeadsPerBlock) void axial_rope64_kernel(AxialRope64Arguments a) {
    constexpr int Threads = 16 * HeadsPerBlock;
    static_assert(Threads >= 32 && Threads <= 1024);
    const int groups          = (a.q_heads + a.k_heads + HeadsPerBlock - 1) / HeadsPerBlock;
    const std::int64_t blocks = std::int64_t(a.tokens) * groups;
    __shared__ float sine[32], cosine[32];
    for (std::int64_t block = blockIdx.x; block < blocks; block += gridDim.x) {
        const auto token     = block / groups;
        const int head       = int(block % groups) * HeadsPerBlock + threadIdx.x / 16;
        const int lane       = threadIdx.x % 16;
        const bool query     = head < a.q_heads;
        __nv_bfloat162* data = nullptr;
        float2 first{}, second{};
        if (head < a.q_heads + a.k_heads) {
            data = reinterpret_cast<__nv_bfloat162*>(query ? a.q + token * a.q_stride + head * 64
                                                           : a.k + token * a.k_stride +
                                                                 (head - a.q_heads) * 64);
            if constexpr (Normalize) {
                first  = __bfloat1622float2(data[lane]);
                second = __bfloat1622float2(data[lane + 16]);
            }
        }
        if (threadIdx.x < 32) {
            const int pair        = threadIdx.x;
            const float frequency = exp2f(a.frequency_exponent * (pair % 16));
            const float angle =
                float(a.positions[token + std::int64_t(pair / 16) * a.tokens]) * frequency;
            sincosf(angle, sine + pair, cosine + pair);
        }
        if (head < a.q_heads + a.k_heads) {
            if constexpr (Normalize) {
                float square = first.x * first.x + first.y * first.y + second.x * second.x +
                               second.y * second.y;
                square = common::warp_sum<16>(square, 0xffffu << (threadIdx.x % 32 / 16 * 16));
                const float inverse = rsqrtf(square * (1.F / 64.F) + a.epsilon);
                const auto* weight  = query ? a.q_weight : a.k_weight;
                first.x *= inverse * __bfloat162float(weight[lane * 2]);
                first.y *= inverse * __bfloat162float(weight[lane * 2 + 1]);
                second.x *= inverse * __bfloat162float(weight[lane * 2 + 32]);
                second.y *= inverse * __bfloat162float(weight[lane * 2 + 33]);
            }
        }
        // Phase evaluation overlaps the other warps' input loads and normalization.
        __syncthreads();
        if (head < a.q_heads + a.k_heads) {
            if constexpr (!Normalize) {
                first  = __bfloat1622float2(data[lane]);
                second = __bfloat1622float2(data[lane + 16]);
            }
            const float c0 = cosine[lane * 2], c1 = cosine[lane * 2 + 1];
            const float s0 = sine[lane * 2], s1 = sine[lane * 2 + 1];
            data[lane] =
                __floats2bfloat162_rn(first.x * c0 - second.x * s0, first.y * c1 - second.y * s1);
            data[lane + 16] =
                __floats2bfloat162_rn(second.x * c0 + first.x * s0, second.y * c1 + first.y * s1);
        }
        if (block + gridDim.x < blocks) __syncthreads();
    }
}

template <bool Normalize, int HeadsPerBlock>
void launch_axial_rope64(AxialRope64Arguments a, cudaStream_t stream) {
    const int groups    = (a.q_heads + a.k_heads + HeadsPerBlock - 1) / HeadsPerBlock;
    const auto count    = std::int64_t(a.tokens) * groups;
    const unsigned grid = count < 2147483647 ? unsigned(count) : 2147483647;
    axial_rope64_kernel<Normalize, HeadsPerBlock><<<grid, 16 * HeadsPerBlock, 0, stream>>>(a);
}
} // namespace ninfer::ops::detail
