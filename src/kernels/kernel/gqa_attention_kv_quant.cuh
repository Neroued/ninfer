#pragma once

// qus::kernels - signed int8, per-token group-wise KV cache codec (shared device
// helpers). Quantization (append) and dequantization (stage) are FUSED into the
// GQA attention kernels themselves (decode partial kernel, prefill fill/attention);
// this header only provides the index math, the vectorized dequant, and the scalar
// quantize helper they share. There is deliberately no standalone quant/dequant
// kernel: that would defeat the halved-bandwidth goal.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace qus::kernels {

inline constexpr int kGqaKvQuantHeadDim = 256;
inline constexpr int kGqaKvQuantKVHeads = 4;
inline constexpr int kGqaKvQuantGroup   = 64;
inline constexpr int kGqaKvQuantGroups  = kGqaKvQuantHeadDim / kGqaKvQuantGroup;

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

// Quantize one bf16 value with a precomputed 1/scale (scale is the FP16-rounded
// per-group absmax/127). Round-to-nearest-even + symmetric clamp to keep codes
// bit-identical to the CPU oracle and to bf16 parity.
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

// Dequantize 8 consecutive int8 codes (dims [d, d+8), aligned to a multiple of 8
// so they lie inside one 64-group) into 8 bf16 packed as an int4. The 8 codes are
// read with ONE 64-bit (int2) coalesced load instead of 8 scalar byte loads, so
// the global traffic is exactly half of the bf16 path and stays fully coalesced.
__device__ __forceinline__ int4 gqa_kv_dequant_i8x8(const std::int8_t* __restrict__ cache,
                                                    const __half* __restrict__ scale, int kv_head,
                                                    int d, int position, int padded_context) {
    const int group            = d >> 6; // d / 64
    const std::int64_t scale_i = gqa_kv_quant_scale_index(kv_head, group, position, padded_context);
    const float s              = __half2float(scale[scale_i]);
    const std::int64_t code_off = gqa_kv_quant_code_index(kv_head, d, position, padded_context);
    const int2 raw              = *reinterpret_cast<const int2*>(&cache[code_off]);
    const std::int8_t* c        = reinterpret_cast<const std::int8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = gqa_kv_quant_pack_bf16(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

} // namespace qus::kernels
