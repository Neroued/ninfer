#pragma once

#include "ninfer/types.h"
#include "ops/common/quant/fp8.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"
#include "ops/common/quant/nvfp4.cuh"
#include "ops/softmax_attention/kernels/causal/tile_io.cuh"
#include "ops/softmax_attention/qsa/sparse_gqa/arguments.h"

namespace ninfer::ops::detail::qsa {

template <KvCacheStorage Storage>
inline constexpr bool kSparseQuantized = Storage != KvCacheStorage::BFloat16;

template <KvCacheStorage Storage>
inline constexpr bool kSparseRotatedValue =
    Storage == KvCacheStorage::Nvfp4Group16 || Storage == KvCacheStorage::Fp8KeyNvfp4Value;

template <KvCacheStorage Storage, bool Key>
struct SparseGqaCodec {
    static constexpr auto profile   = kv_cache_profile_info(Storage);
    static constexpr auto format    = Key ? profile.key : profile.value;
    static constexpr auto info      = kv_cache_format_info(format);
    static constexpr int code_bytes = 256 * info.value_bits / 8;
    static constexpr int scale_bytes =
        info.group_size ? 256 / info.group_size * info.scale_bytes() : 0;
    // FP8 exponents stay in registers; only codes occupy the in-place payload.
    static constexpr int packed_row = format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0
                                          ? code_bytes
                                          : (code_bytes + scale_bytes + 15) / 16 * 16;
};

template <KvCacheStorage Storage, bool Key>
__device__ __forceinline__ float sparse_gqa_decode(const std::uint8_t* data,
                                                   const std::uint8_t* scales, int d) {
    constexpr auto format = SparseGqaCodec<Storage, Key>::format;
    if constexpr (format == KvCacheVectorFormat::BFloat16 ||
                  format == KvCacheVectorFormat::Float16) {
        const auto word = reinterpret_cast<const std::uint16_t*>(data)[d];
        if constexpr (Key)
            return __bfloat162float(__ushort_as_bfloat16(word));
        else
            return __half2float(__ushort_as_half(word));
    } else if constexpr (format == KvCacheVectorFormat::Int8Group64F16) {
        return float(reinterpret_cast<const std::int8_t*>(data)[d]) *
               __half2float(reinterpret_cast<const __half*>(scales)[d / 64]);
    } else if constexpr (format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0) {
        __nv_fp8_e4m3 code;
        code.__x = data[d];
        return float(code) * common::ue8m0_to_float(scales[d / 128]);
    } else {
        static_assert(format == KvCacheVectorFormat::Nvfp4Group16Ue4m3);
        __nv_fp4x2_e2m1 code;
        code.__x           = data[d / 2];
        const __half2 pair = static_cast<__half2>(code);
        __nv_fp8_e4m3 scale;
        scale.__x = scales[d / 16];
        return __half2float(d % 2 ? __high2half(pair) : __low2half(pair)) * float(scale);
    }
}

template <KvCacheStorage Storage, bool Key, bool Vector>
__device__ __forceinline__ int4 sparse_gqa_expand(const std::uint8_t* data,
                                                  const std::uint8_t* scales, int d) {
    constexpr auto format = SparseGqaCodec<Storage, Key>::format;
    if constexpr (format == KvCacheVectorFormat::BFloat16 ||
                  format == KvCacheVectorFormat::Float16) {
        if constexpr (Vector)
            return common::load_vec<int4>(data + 2 * d);
        else {
            int4 result;
            auto* words = reinterpret_cast<std::uint16_t*>(&result);
#pragma unroll
            for (int i = 0; i < 8; ++i)
                words[i] = reinterpret_cast<const std::uint16_t*>(data)[d + i];
            return result;
        }
    } else if constexpr (format == KvCacheVectorFormat::Nvfp4Group16Ue4m3 && Vector) {
        return common::nvfp4_dequant_f16x8(data + d / 2, scales[d / 16]);
    } else if constexpr (format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0 && Vector) {
        return common::fp8_dequant_f16x8(data + d, scales[d / 128]);
    } else if constexpr (format == KvCacheVectorFormat::Int8Group64F16 && Vector) {
        const auto packed = common::load_vec<int2>(data + d);
        const auto scale  = reinterpret_cast<const __half*>(scales)[d / 64];
        unsigned out[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            __half2 value;
            // Offset-binary bytes become exact halves 1024 + (code + 128). Subtracting
            // 1152 recovers signed INT8 without scalar integer-to-float conversions.
            const auto word     = reinterpret_cast<const unsigned*>(&packed)[i / 2] ^ 0x80808080u;
            const unsigned bits = __byte_perm(word, 0x64646464u, i % 2 ? 0x4342 : 0x4140);
            const unsigned bias = 0x64806480u;
            const auto codes    = __hsub2(*reinterpret_cast<const __half2*>(&bits),
                                          *reinterpret_cast<const __half2*>(&bias));
            value               = __hmul2(codes, __halves2half2(scale, scale));
            out[i] = *reinterpret_cast<const unsigned*>(&value);
        }
        return make_int4(out[0], out[1], out[2], out[3]);
    } else {
        unsigned out[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float x = sparse_gqa_decode<Storage, Key>(data, scales, d + 2 * i);
            const float y = sparse_gqa_decode<Storage, Key>(data, scales, d + 2 * i + 1);
            out[i]        = common::pack_f16x2(x, y);
        }
        return make_int4(out[0], out[1], out[2], out[3]);
    }
}

template <KvCacheStorage Storage, bool Key>
__device__ __forceinline__ int4 sparse_gqa_expand_prepared(const std::uint8_t* data,
                                                           const std::uint8_t* scales, int d,
                                                           unsigned fp8_scale) {
    if constexpr (SparseGqaCodec<Storage, Key>::format ==
                  KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0) {
        return common::fp8_dequant_f16x8_scaled(
            data + d, __ushort_as_half(static_cast<unsigned short>(fp8_scale)));
    } else
        return sparse_gqa_expand<Storage, Key, true>(data, scales, d);
}

// Q is transformed only in registers and the existing shared prologue space. No A8/A4 operand
// or additional global Q tensor is formed. Row order matches the MMA's padded 16-head tiles.
template <KvCacheStorage Storage, int Warps, int Queries, int KvHeads>
__device__ void sparse_gqa_prepare_query(const SparseGqaKernelArguments<KvHeads>& a,
                                         std::int64_t first, std::int64_t last, int head,
                                         std::uint16_t* target) {
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    if constexpr (kSparseQuantized<Storage>) {
        for (int row = warp; row < Queries * 16; row += Warps) {
            const int h  = row % 16;
            const auto q = first + row / 16;
            float values[8]{};
            if (q < last && h < 12) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    values[i] = __bfloat162float(
                        a.query[(q * a.query_heads + head * 12 + h) * 256 + lane + 32 * i]);
                kv::normalized_hadamard_d256_inplace(values, lane);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
                target[row * 256 + common::swizzle_xor_64(row, lane + 32 * i)] =
                    __half_as_ushort(__float2half_rn(values[i]));
        }
    } else {
        for (int i = tid; i < Queries * 16 * 32; i += Warps * 32) {
            const int row = i / 32, d = i % 32 * 8, h = row % 16;
            auto* dst = target + row * 256 + common::swizzle_xor_64(row, d);
            if (first + row / 16 < last && h < 12)
                common::cp_async<16>(
                    dst, a.query + ((first + row / 16) * a.query_heads + head * 12 + h) * 256 + d);
            else
                common::store_vec(dst, make_int4(0, 0, 0, 0));
        }
        common::cp_commit();
        common::cp_wait<0>();
    }
    __syncthreads();
}

template <KvCacheStorage Storage>
__device__ __forceinline__ void sparse_gqa_qk(float (&score)[4], const unsigned (&q)[4],
                                              const unsigned (&k)[2]) {
    if constexpr (kSparseQuantized<Storage>)
        common::mma_f16(score[0], score[1], score[2], score[3], q[0], q[1], q[2], q[3], k[0], k[1]);
    else
        common::mma_bf16(score[0], score[1], score[2], score[3], q[0], q[1], q[2], q[3], k[0],
                         k[1]);
}

} // namespace ninfer::ops::detail::qsa
