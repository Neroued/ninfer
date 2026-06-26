#pragma once

// qus::kernels - gqa_attention prefill kernels. The op writes the full prompt
// k/v into KVCache positions [0..T-1], then computes causal GQA attention for
// every prompt token with fp32 online softmax and fp32 AV accumulation.

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

inline constexpr int kGqaPrefillHeadDim   = 256;
inline constexpr int kGqaPrefillQHeads    = 24;
inline constexpr int kGqaPrefillKVHeads   = 4;
inline constexpr int kGqaPrefillGroupSize = 6;

__device__ __forceinline__ std::int64_t gqa_prefill_cache_index(int kv_head, int d, int position) {
    return (static_cast<std::int64_t>(position) * kGqaPrefillHeadDim + d) * kGqaPrefillKVHeads +
           kv_head;
}

__device__ __forceinline__ std::int64_t gqa_prefill_q_index(int q_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaPrefillHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(kGqaPrefillQHeads) * token);
}

__device__ __forceinline__ std::int64_t gqa_prefill_kv_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaPrefillHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(kGqaPrefillKVHeads) * token);
}

__device__ __forceinline__ float gqa_prefill_warp_sum(float value) {
    constexpr unsigned mask = 0xffffffffu;
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);
    return value;
}

__device__ __forceinline__ float gqa_prefill_block_sum_256(float value, float* scratch) {
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    value = gqa_prefill_warp_sum(value);
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();

    value = (tid < 8) ? scratch[lane] : 0.0f;
    if (warp == 0) { value = gqa_prefill_warp_sum(value); }
    if (tid == 0) { scratch[0] = value; }
    __syncthreads();
    value = scratch[0];
    __syncthreads();
    return value;
}

__global__ void gqa_attention_prefill_fill_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                                  __nv_bfloat16* cache_k, __nv_bfloat16* cache_v,
                                                  std::int32_t tokens) {
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t n =
        static_cast<std::int64_t>(tokens) * kGqaPrefillKVHeads * kGqaPrefillHeadDim;
    if (idx >= n) { return; }

    const int d       = static_cast<int>(idx % kGqaPrefillHeadDim);
    const int tmp     = static_cast<int>(idx / kGqaPrefillHeadDim);
    const int kv_head = tmp % kGqaPrefillKVHeads;
    const int token   = tmp / kGqaPrefillKVHeads;

    const std::int64_t cache_off = gqa_prefill_cache_index(kv_head, d, token);
    cache_k[cache_off]           = k[idx];
    cache_v[cache_off]           = v[idx];
}

// PERF: Replace this correctness-first online-softmax kernel with a flash
// prefill implementation in the Tier-3 performance milestone.
__launch_bounds__(256) __global__
    void gqa_attention_prefill_kernel(const __nv_bfloat16* q, const __nv_bfloat16* cache_k,
                                      const __nv_bfloat16* cache_v, float scale, __nv_bfloat16* out,
                                      std::int32_t tokens) {
    const int block  = static_cast<int>(blockIdx.x);
    const int q_head = block % kGqaPrefillQHeads;
    const int token  = block / kGqaPrefillQHeads;
    const int d      = threadIdx.x;
    if (q_head >= kGqaPrefillQHeads || token >= tokens || d >= kGqaPrefillHeadDim) { return; }

    __shared__ float scratch[kGqaPrefillHeadDim];

    const int kv_head = q_head / kGqaPrefillGroupSize;
    const float q_d   = __bfloat162float(q[gqa_prefill_q_index(q_head, d, token)]);

    float max_score = -3.4028234663852886e38f;
    float denom     = 0.0f;
    float acc       = 0.0f;
    for (std::int32_t j = 0; j <= token; ++j) {
        const float k_d      = __bfloat162float(cache_k[gqa_prefill_cache_index(kv_head, d, j)]);
        const float dot_part = q_d * k_d;
        const float score    = gqa_prefill_block_sum_256(dot_part, scratch) * scale;
        const float next_max = fmaxf(max_score, score);
        const float old_w    = expf(max_score - next_max);
        const float new_w    = expf(score - next_max);
        const float v_d      = __bfloat162float(cache_v[gqa_prefill_cache_index(kv_head, d, j)]);
        acc                  = acc * old_w + new_w * v_d;
        denom                = denom * old_w + new_w;
        max_score            = next_max;
    }

    out[gqa_prefill_q_index(q_head, d, token)] = __float2bfloat16(acc / denom);
}

} // namespace qus::kernels
