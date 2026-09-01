#pragma once

// Correctness-first sm_70 causal prompt attention. One CTA owns one
// (query-token, query-head) row, cooperates across D=256 for QK, and keeps one
// output dimension per thread through the online-softmax key walk. This consumes
// the current paged BF16/INT8 cache contracts directly; the Ampere+ tiled kernels
// remain the production path on their native architectures.

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace ninfer::ops {

inline constexpr int kCausalPromptVoltaThreads = kCausalPromptHeadDim;

template <typename Geometry, typename Metadata, bool Int8>
__launch_bounds__(kCausalPromptVoltaThreads, 1) __global__ void
causal_attention_prompt_volta_kernel(
    const __nv_bfloat16* __restrict__ q, const void* __restrict__ cache_k_raw,
    const void* __restrict__ cache_v_raw, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale,
    __nv_bfloat16* __restrict__ out, std::int32_t width) {
    constexpr int D = kCausalPromptHeadDim;
    static_assert(D == kCausalPromptVoltaThreads);

    __shared__ float q_rotated[D];
    __shared__ float warp_sums[kCausalPromptVoltaThreads / kWarpSize];
    __shared__ float score_s;

    const int token  = static_cast<int>(blockIdx.x);
    const int q_head = static_cast<int>(blockIdx.y);
    const int d      = static_cast<int>(threadIdx.x);
    if (token >= width || q_head >= Geometry::QHeads) { return; }

    const int valid_tokens = metadata.valid_tokens(width);
    if (token >= valid_tokens) {
        out[causal_prompt_q_index<Geometry>(q_head, d, token)] = __float2bfloat16(0.0f);
        return;
    }

    if constexpr (Int8) {
        if (d < kWarpSize) {
            float values[8];
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                const int q_d = d + kWarpSize * item;
                values[item] =
                    __bfloat162float(q[causal_prompt_q_index<Geometry>(q_head, q_d, token)]);
            }
            normalized_hadamard_d256_inplace(values, d);
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                q_rotated[d + kWarpSize * item] = values[item];
            }
        }
        __syncthreads();
    }

    const int kv_head               = q_head / Geometry::GroupSize;
    const int last_key              = positions[token];
    const std::int32_t* block_table = metadata.block_table();
    const float q_value = Int8
                              ? q_rotated[d]
                              : __bfloat162float(
                                    q[causal_prompt_q_index<Geometry>(q_head, d, token)]);

    const auto* cache_k_bf16 = static_cast<const __nv_bfloat16*>(cache_k_raw);
    const auto* cache_v_bf16 = static_cast<const __nv_bfloat16*>(cache_v_raw);
    const auto* cache_k_i8    = static_cast<const std::int8_t*>(cache_k_raw);
    const auto* cache_v_i8    = static_cast<const std::int8_t*>(cache_v_raw);

    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    float acc     = 0.0f;
    for (int key = 0; key <= last_key; ++key) {
        const int physical_page = paged_kv_physical_page(block_table, key);
        const int page_offset   = key & kPagedKVPageMask;
        const std::int64_t cache_index = Int8
            ? kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d, page_offset)
            : paged_kv_element_offset<D, Geometry::KVHeads>(block_table, kv_head, key, d);
        float key_value;
        float value;
        if constexpr (Int8) {
            const int group = d / kKVCacheInt8Group;
            const std::int64_t scale_index = kv_cache_int8_quant_scale_index<Geometry>(
                physical_page, kv_head, group, page_offset);
            key_value = static_cast<float>(cache_k_i8[cache_index]) *
                        __half2float(cache_k_scale[scale_index]);
            value = static_cast<float>(cache_v_i8[cache_index]) *
                    __half2float(cache_v_scale[scale_index]);
        } else {
            key_value = __bfloat162float(cache_k_bf16[cache_index]);
            value     = __bfloat162float(cache_v_bf16[cache_index]);
        }

        const float dot = block_reduce_sum<kCausalPromptVoltaThreads>(q_value * key_value,
                                                                      warp_sums);
        if (d == 0) { score_s = dot * scale; }
        __syncthreads();

        const float score   = score_s;
        const float next_max = fmaxf(row_max, score);
        const float old_scale =
            row_max == -CUDART_INF_F ? 0.0f : exp2_approx((row_max - next_max) * 1.4426950408889634f);
        const float probability = exp2_approx((score - next_max) * 1.4426950408889634f);
        acc     = acc * old_scale + probability * value;
        row_sum = row_sum * old_scale + probability;
        row_max = next_max;
        __syncthreads();
    }

    const float result = row_sum > 0.0f ? acc / row_sum : 0.0f;
    out[causal_prompt_q_index<Geometry>(q_head, d, token)] = __float2bfloat16_rn(result);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(kCausalPromptVoltaThreads, 1) __global__ void
causal_attention_prompt_fp8_volta_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale,
    __nv_bfloat16* __restrict__ out, std::int32_t width) {
    constexpr int D = kCausalPromptHeadDim;

    __shared__ float q_rotated[D];
    __shared__ float warp_sums[kCausalPromptVoltaThreads / kWarpSize];
    __shared__ float score_s;

    const int token  = static_cast<int>(blockIdx.x);
    const int q_head = static_cast<int>(blockIdx.y);
    const int d      = static_cast<int>(threadIdx.x);
    if (token >= width || q_head >= Geometry::QHeads) { return; }

    const int valid_tokens = metadata.valid_tokens(width);
    if (token >= valid_tokens) {
        out[causal_prompt_q_index<Geometry>(q_head, d, token)] = __float2bfloat16(0.0f);
        return;
    }

    if (d < kWarpSize) {
        float values[8];
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const int q_d = d + kWarpSize * item;
            values[item] =
                __bfloat162float(q[causal_prompt_q_index<Geometry>(q_head, q_d, token)]);
        }
        normalized_hadamard_d256_inplace(values, d);
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            q_rotated[d + kWarpSize * item] = values[item];
        }
    }
    __syncthreads();

    const int kv_head               = q_head / Geometry::GroupSize;
    const int last_key              = positions[token];
    const std::int32_t* block_table = metadata.block_table();
    const float q_value             = q_rotated[d];
    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    float acc     = 0.0f;
    for (int key = 0; key <= last_key; ++key) {
        const int physical_page = paged_kv_physical_page(block_table, key);
        const int page_offset   = key & kPagedKVPageMask;
        const std::int64_t code_index =
            kv_cache_fp8_code_index<Geometry>(physical_page, kv_head, d, page_offset);
        const std::int64_t scale_index =
            kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head, page_offset);
        __nv_fp8x2_e4m3 k_code;
        __nv_fp8x2_e4m3 v_code;
        k_code.__x = static_cast<std::uint16_t>(cache_k[code_index]);
        v_code.__x = static_cast<std::uint16_t>(cache_v[code_index]);
        const float key_value = static_cast<float2>(k_code).x *
                                __half2float(cache_k_scale[scale_index]);
        const float value = static_cast<float2>(v_code).x *
                            __half2float(cache_v_scale[scale_index]);

        const float dot = block_reduce_sum<kCausalPromptVoltaThreads>(q_value * key_value,
                                                                      warp_sums);
        if (d == 0) { score_s = dot * scale; }
        __syncthreads();
        const float score       = score_s;
        const float next_max    = fmaxf(row_max, score);
        const float old_scale   = row_max == -CUDART_INF_F
                                      ? 0.0f
                                      : exp2_approx((row_max - next_max) * 1.4426950408889634f);
        const float probability = exp2_approx((score - next_max) * 1.4426950408889634f);
        acc     = acc * old_scale + probability * value;
        row_sum = row_sum * old_scale + probability;
        row_max = next_max;
        __syncthreads();
    }

    out[causal_prompt_q_index<Geometry>(q_head, d, token)] =
        __float2bfloat16_rn(row_sum > 0.0f ? acc / row_sum : 0.0f);
}

} // namespace ninfer::ops
