#pragma once

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace qus::kernels {

inline constexpr int kVisionAttentionHeadDim = 72;
inline constexpr int kVisionAttentionHeads   = 16;

__device__ __forceinline__ const __nv_bfloat16*
vision_attention_ptr(const __nv_bfloat16* data, std::int64_t stride_d, std::int64_t stride_h,
                     std::int64_t stride_t, int d, int h, int t) {
    return data + static_cast<std::int64_t>(d) * stride_d +
           static_cast<std::int64_t>(h) * stride_h + static_cast<std::int64_t>(t) * stride_t;
}

template <int Block>
__global__ void vision_attention_reference_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const std::int32_t* cu_seqlens, std::int32_t segments, __nv_bfloat16* out, std::int32_t patches,
    std::int64_t q_stride_d, std::int64_t q_stride_h, std::int64_t q_stride_t,
    std::int64_t k_stride_d, std::int64_t k_stride_h, std::int64_t k_stride_t,
    std::int64_t v_stride_d, std::int64_t v_stride_h, std::int64_t v_stride_t) {
    const int query = static_cast<int>(blockIdx.x);
    const int head  = static_cast<int>(blockIdx.y);
    if (query >= patches || head >= kVisionAttentionHeads) { return; }

    int lo = 0;
    int hi = segments;
    while (lo + 1 < hi) {
        const int mid = (lo + hi) / 2;
        if (cu_seqlens[mid] <= query) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const int begin = cu_seqlens[lo];
    const int end   = cu_seqlens[lo + 1];
    if (begin < 0 || begin > query || end <= query || end > patches) { return; }

    __shared__ float reduction[Block];
    __shared__ float alpha;
    __shared__ float beta;
    __shared__ float normalizer;
    float accumulator     = 0.0f;
    float running_max     = -INFINITY;
    float running_sum     = 0.0f;
    constexpr float scale = 0.11785113019775792073f; // 1/sqrt(72)

    for (int key = begin; key < end; ++key) {
        float product = 0.0f;
        for (int d = static_cast<int>(threadIdx.x); d < kVisionAttentionHeadDim; d += Block) {
            const float qv = __bfloat162float(
                *vision_attention_ptr(q, q_stride_d, q_stride_h, q_stride_t, d, head, query));
            const float kv = __bfloat162float(
                *vision_attention_ptr(k, k_stride_d, k_stride_h, k_stride_t, d, head, key));
            product += qv * kv;
        }
        reduction[threadIdx.x] = product;
        __syncthreads();
        for (int offset = Block / 2; offset > 0; offset >>= 1) {
            if (static_cast<int>(threadIdx.x) < offset) {
                reduction[threadIdx.x] += reduction[threadIdx.x + offset];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score    = reduction[0] * scale;
            const float next_max = fmaxf(running_max, score);
            alpha                = isinf(running_max) ? 0.0f : expf(running_max - next_max);
            beta                 = expf(score - next_max);
            running_sum          = running_sum * alpha + beta;
            running_max          = next_max;
            normalizer           = running_sum;
        }
        __syncthreads();
        if (threadIdx.x < kVisionAttentionHeadDim) {
            const float value = __bfloat162float(*vision_attention_ptr(
                v, v_stride_d, v_stride_h, v_stride_t, static_cast<int>(threadIdx.x), head, key));
            accumulator       = accumulator * alpha + beta * value;
        }
        __syncthreads();
    }

    if (threadIdx.x < kVisionAttentionHeadDim) {
        const std::int64_t index =
            (static_cast<std::int64_t>(query) * kVisionAttentionHeads + head) *
                kVisionAttentionHeadDim +
            static_cast<int>(threadIdx.x);
        out[index] = __float2bfloat16_rn(accumulator / normalizer);
    }
}

} // namespace qus::kernels
