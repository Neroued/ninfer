#pragma once

// qus::kernels - signed int8, per-token group-wise KV cache append.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace qus::kernels {

inline constexpr int kGqaKvQuantHeadDim   = 256;
inline constexpr int kGqaKvQuantKVHeads   = 4;
inline constexpr int kGqaKvQuantGroup     = 64;
inline constexpr int kGqaKvQuantGroups    = kGqaKvQuantHeadDim / kGqaKvQuantGroup;
inline constexpr int kGqaKvQuantBlockSize = kGqaKvQuantGroup;

__device__ __forceinline__ std::int64_t gqa_kv_quant_code_index(int kv_head, int d, int position,
                                                                int padded_context) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaKvQuantHeadDim) *
                                              (static_cast<std::int64_t>(position) +
                                               static_cast<std::int64_t>(padded_context) * kv_head);
}

__device__ __forceinline__ std::int64_t gqa_kv_quant_scale_index(int kv_head, int group,
                                                                 int position, int padded_context) {
    return static_cast<std::int64_t>(group) +
           static_cast<std::int64_t>(kGqaKvQuantGroups) *
               (static_cast<std::int64_t>(position) +
                static_cast<std::int64_t>(padded_context) * kv_head);
}

__device__ __forceinline__ std::int64_t gqa_kv_quant_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaKvQuantHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(kGqaKvQuantKVHeads) * token);
}

__device__ __forceinline__ std::int8_t gqa_kv_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

__device__ __forceinline__ unsigned gqa_kv_quant_pack_bf16(float lo, float hi) {
    unsigned out;
    const unsigned lo_bits = __float_as_uint(lo);
    const unsigned hi_bits = __float_as_uint(hi);
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(out) : "r"(hi_bits), "r"(lo_bits));
    return out;
}

__device__ __forceinline__ int4 gqa_kv_dequant_i8x8(const std::int8_t* __restrict__ cache,
                                                    const __half* __restrict__ scale, int kv_head,
                                                    int d, int position, int padded_context) {
    const int group            = d / kGqaKvQuantGroup;
    const std::int64_t scale_i = gqa_kv_quant_scale_index(kv_head, group, position, padded_context);
    const float s              = __half2float(scale[scale_i]);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d0 = d + 2 * i;
        const int d1 = d0 + 1;
        const float x0 =
            static_cast<float>(
                cache[gqa_kv_quant_code_index(kv_head, d0, position, padded_context)]) *
            s;
        const float x1 =
            static_cast<float>(
                cache[gqa_kv_quant_code_index(kv_head, d1, position, padded_context)]) *
            s;
        packed[i] = gqa_kv_quant_pack_bf16(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

static __global__ void gqa_attention_kv_quantize_append_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, std::int8_t* __restrict__ cache_k,
    std::int8_t* __restrict__ cache_v, __half* __restrict__ scale_k, __half* __restrict__ scale_v,
    std::int32_t tokens, std::int32_t padded_context, std::int32_t max_context,
    std::int32_t positions_are_base) {
    __shared__ float k_abs[kGqaKvQuantGroup];
    __shared__ float v_abs[kGqaKvQuantGroup];
    __shared__ float k_inv_scale;
    __shared__ float v_inv_scale;

    const int group   = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int token   = static_cast<int>(blockIdx.z);
    const int lane    = static_cast<int>(threadIdx.x);

    if (group >= kGqaKvQuantGroups || kv_head >= kGqaKvQuantKVHeads || token >= tokens ||
        lane >= kGqaKvQuantGroup) {
        return;
    }

    const int position = positions_are_base ? positions[0] + token : positions[token];
    if (position < 0 || position >= max_context) { return; }

    const int d                = group * kGqaKvQuantGroup + lane;
    const std::int64_t src_off = gqa_kv_quant_src_index(kv_head, d, token);
    const float k_value        = __bfloat162float(k[src_off]);
    const float v_value        = __bfloat162float(v[src_off]);
    k_abs[lane]                = fabsf(k_value);
    v_abs[lane]                = fabsf(v_value);
    __syncthreads();

    for (int stride = kGqaKvQuantGroup / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            k_abs[lane] = fmaxf(k_abs[lane], k_abs[lane + stride]);
            v_abs[lane] = fmaxf(v_abs[lane], v_abs[lane + stride]);
        }
        __syncthreads();
    }

    if (lane == 0) {
        const std::int64_t scale_off =
            gqa_kv_quant_scale_index(kv_head, group, position, padded_context);
        const __half k_scale  = __float2half_rn(k_abs[0] > 0.0f ? k_abs[0] / 127.0f : 0.0f);
        const __half v_scale  = __float2half_rn(v_abs[0] > 0.0f ? v_abs[0] / 127.0f : 0.0f);
        scale_k[scale_off]    = k_scale;
        scale_v[scale_off]    = v_scale;
        const float k_scale_f = __half2float(k_scale);
        const float v_scale_f = __half2float(v_scale);
        k_inv_scale           = k_scale_f > 0.0f ? 1.0f / k_scale_f : 0.0f;
        v_inv_scale           = v_scale_f > 0.0f ? 1.0f / v_scale_f : 0.0f;
    }
    __syncthreads();

    const std::int64_t cache_off = gqa_kv_quant_code_index(kv_head, d, position, padded_context);
    cache_k[cache_off]           = gqa_kv_quant_code(k_value, k_inv_scale);
    cache_v[cache_off]           = gqa_kv_quant_code(v_value, v_inv_scale);
}

} // namespace qus::kernels
