#pragma once

// qus::kernels - gqa_attention decode kernels. The op appends current k/v to
// KVCache at device scalar pos, then computes single-query GQA attention over
// cached positions [0..pos] with fp32 stable softmax.

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

inline constexpr int kGqaHeadDim   = 256;
inline constexpr int kGqaQHeads    = 24;
inline constexpr int kGqaKVHeads   = 4;
inline constexpr int kGqaGroupSize = 6;

__device__ __forceinline__ std::int64_t gqa_cache_index(int kv_head, int d, int position) {
    return (static_cast<std::int64_t>(position) * kGqaHeadDim + d) * kGqaKVHeads + kv_head;
}

__device__ __forceinline__ std::int64_t gqa_q_index(int q_head, int d) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaHeadDim) * q_head;
}

__device__ __forceinline__ float gqa_warp_sum(float value) {
    constexpr unsigned mask = 0xffffffffu;
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);
    return value;
}

__device__ __forceinline__ float gqa_block_sum_256(float value, float* scratch) {
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    value = gqa_warp_sum(value);
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();

    value = (tid < 8) ? scratch[lane] : 0.0f;
    if (warp == 0) { value = gqa_warp_sum(value); }
    if (tid == 0) { scratch[0] = value; }
    __syncthreads();
    value = scratch[0];
    __syncthreads();
    return value;
}

__global__ void gqa_attention_decode_append_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                                   const std::int32_t* pos, __nv_bfloat16* cache_k,
                                                   __nv_bfloat16* cache_v,
                                                   std::int32_t max_context) {
    const std::int32_t p = pos[0];
    if (p < 0 || p >= max_context) { return; }

    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int n = kGqaHeadDim * kGqaKVHeads;
    if (idx >= n) { return; }

    const int kv_head            = idx / kGqaHeadDim;
    const int d                  = idx - kv_head * kGqaHeadDim;
    const std::int64_t cache_off = gqa_cache_index(kv_head, d, p);
    cache_k[cache_off]           = k[idx];
    cache_v[cache_off]           = v[idx];
}

__launch_bounds__(256) __global__
    void gqa_attention_decode_kernel(const __nv_bfloat16* q, const __nv_bfloat16* cache_k,
                                     const __nv_bfloat16* cache_v, const std::int32_t* pos,
                                     float scale, __nv_bfloat16* out, std::int32_t max_context) {
    const int q_head     = blockIdx.x;
    const int d          = threadIdx.x;
    const std::int32_t p = pos[0];
    if (q_head >= kGqaQHeads || d >= kGqaHeadDim || p < 0 || p >= max_context) { return; }

    __shared__ float scratch[kGqaHeadDim];
    __shared__ float max_score_shared;
    __shared__ float denom_shared;

    const int kv_head = q_head / kGqaGroupSize;
    const float q_d   = __bfloat162float(q[gqa_q_index(q_head, d)]);

    float max_score = -3.4028234663852886e38f;
    for (std::int32_t j = 0; j <= p; ++j) {
        const float k_d      = __bfloat162float(cache_k[gqa_cache_index(kv_head, d, j)]);
        const float dot_part = q_d * k_d;
        const float score    = gqa_block_sum_256(dot_part, scratch) * scale;
        if (d == 0 && score > max_score) { max_score = score; }
        __syncthreads();
    }
    if (d == 0) { max_score_shared = max_score; }
    __syncthreads();

    float denom = 0.0f;
    for (std::int32_t j = 0; j <= p; ++j) {
        const float k_d      = __bfloat162float(cache_k[gqa_cache_index(kv_head, d, j)]);
        const float dot_part = q_d * k_d;
        const float score    = gqa_block_sum_256(dot_part, scratch) * scale;
        if (d == 0) { denom += expf(score - max_score_shared); }
        __syncthreads();
    }
    if (d == 0) { denom_shared = denom; }
    __syncthreads();

    float acc = 0.0f;
    for (std::int32_t j = 0; j <= p; ++j) {
        const float k_d      = __bfloat162float(cache_k[gqa_cache_index(kv_head, d, j)]);
        const float dot_part = q_d * k_d;
        const float score    = gqa_block_sum_256(dot_part, scratch) * scale;
        const float prob     = expf(score - max_score_shared) / denom_shared;
        const float v_d      = __bfloat162float(cache_v[gqa_cache_index(kv_head, d, j)]);
        acc += prob * v_d;
        __syncthreads();
    }

    out[gqa_q_index(q_head, d)] = __float2bfloat16(acc);
}

} // namespace qus::kernels
