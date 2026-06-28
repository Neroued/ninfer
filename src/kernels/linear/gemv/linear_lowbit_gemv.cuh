#pragma once

#include "kernels/linear/codec/linear_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels::detail {

template <class Codec>
struct LowbitGemvGroup {
    __device__ static float accumulate(const std::uint8_t* payload, std::int32_t row,
                                       std::int32_t group, std::int32_t group_cnt,
                                       const __nv_bfloat16* x_group, std::int32_t limit,
                                       float acc) {
        float wbuf[Codec::kGroupK];
        Codec::load_group(payload, row, group, group_cnt, wbuf);
#pragma unroll
        for (int offset = 0; offset < Codec::kGroupK; ++offset) {
            if (offset < limit) {
                acc = fmaf(wbuf[offset], __bfloat162float(x_group[offset]), acc);
            }
        }
        return acc;
    }
};

template <>
struct LowbitGemvGroup<Q4Codec> {
    __device__ static float accumulate(const std::uint8_t* payload, std::int32_t row,
                                       std::int32_t group, std::int32_t group_cnt,
                                       const __nv_bfloat16* x_group, std::int32_t limit,
                                       float acc) {
        const std::int32_t tile = row / 64;
        const std::int32_t rit  = row - tile * 64;
        const std::int64_t off =
            (static_cast<std::int64_t>(tile) * group_cnt + group) * Q4Codec::kTileBytes;
        const std::uint16_t sb =
            static_cast<std::uint16_t>(payload[off + rit * 2]) |
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(payload[off + rit * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t* packed =
            payload + off + 64 * 2 + static_cast<std::int64_t>(rit) * Q4Codec::kBytesPerRowPerGroup;
        const auto* words = reinterpret_cast<const std::uint32_t*>(packed);
        std::uint32_t w[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { w[j] = words[j]; }
#pragma unroll
        for (int offset = 0; offset < Q4Codec::kGroupK; offset += 2) {
            if (offset + 1 < limit) {
                const std::uint32_t bits = w[offset >> 3] >> ((offset & 7) * Q4Codec::kBits);
                const int s0             = static_cast<int>((bits & 0x0fu) << 28) >> 28;
                const int s1             = static_cast<int>(((bits >> 4) & 0x0fu) << 28) >> 28;
                const float2 xv =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x_group + offset));
                acc = fmaf(static_cast<float>(s0) * scale, xv.x, acc);
                acc = fmaf(static_cast<float>(s1) * scale, xv.y, acc);
            } else if (offset < limit) {
                const std::uint32_t bits =
                    (w[offset >> 3] >> ((offset & 7) * Q4Codec::kBits)) & 0x0fu;
                const int s = static_cast<int>(bits << 28) >> 28;
                acc = fmaf(static_cast<float>(s) * scale, __bfloat162float(x_group[offset]), acc);
            }
        }
        return acc;
    }
};

template <>
struct LowbitGemvGroup<Q5Codec> {
    __device__ static float accumulate(const std::uint8_t* payload, std::int32_t row,
                                       std::int32_t group, std::int32_t group_cnt,
                                       const __nv_bfloat16* x_group, std::int32_t limit,
                                       float acc) {
        const std::int32_t tile = row / 64;
        const std::int32_t rit  = row - tile * 64;
        const std::int64_t off =
            (static_cast<std::int64_t>(tile) * group_cnt + group) * Q5Codec::kTileBytes;
        const std::uint16_t sb =
            static_cast<std::uint16_t>(payload[off + rit * 2]) |
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(payload[off + rit * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t* packed =
            payload + off + 64 * 2 + static_cast<std::int64_t>(rit) * Q5Codec::kBytesPerRowPerGroup;
        const auto* word_pairs = reinterpret_cast<const std::uint64_t*>(packed);
        std::uint32_t w[11];
#pragma unroll
        for (int j = 0; j < 5; ++j) {
            const std::uint64_t pair = word_pairs[j];
            w[j * 2]                 = static_cast<std::uint32_t>(pair);
            w[j * 2 + 1]             = static_cast<std::uint32_t>(pair >> 32);
        }
        w[10] = 0u;
#pragma unroll
        for (int offset = 0; offset < Q5Codec::kGroupK; offset += 2) {
            if (offset + 1 < limit) {
                const int bitpos          = offset * Q5Codec::kBits;
                const int wi              = bitpos >> 5;
                const int sh              = bitpos & 31;
                const int bitpos1         = bitpos + Q5Codec::kBits;
                const int wi1             = bitpos1 >> 5;
                const int sh1             = bitpos1 & 31;
                const std::uint32_t bits0 = __funnelshift_r(w[wi], w[wi + 1], sh);
                const std::uint32_t bits1 = __funnelshift_r(w[wi1], w[wi1 + 1], sh1);
                const int s0              = static_cast<int>(bits0 << 27) >> 27;
                const int s1              = static_cast<int>(bits1 << 27) >> 27;
                const float2 xv =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x_group + offset));
                acc = fmaf(static_cast<float>(s0) * scale, xv.x, acc);
                acc = fmaf(static_cast<float>(s1) * scale, xv.y, acc);
            } else if (offset < limit) {
                const int bitpos         = offset * Q5Codec::kBits;
                const int wi             = bitpos >> 5;
                const int sh             = bitpos & 31;
                const std::uint32_t bits = __funnelshift_r(w[wi], w[wi + 1], sh);
                const int s              = static_cast<int>(bits << 27) >> 27;
                acc = fmaf(static_cast<float>(s) * scale, __bfloat162float(x_group[offset]), acc);
            }
        }
        return acc;
    }
};

template <>
struct LowbitGemvGroup<Q6Codec> {
    __device__ static float accumulate(const std::uint8_t* payload, std::int32_t row,
                                       std::int32_t group, std::int32_t group_cnt,
                                       const __nv_bfloat16* x_group, std::int32_t limit,
                                       float acc) {
        const std::int32_t tile = row / 64;
        const std::int32_t rit  = row - tile * 64;
        const std::int64_t off =
            (static_cast<std::int64_t>(tile) * group_cnt + group) * Q6Codec::kTileBytes;
        const std::uint16_t sb =
            static_cast<std::uint16_t>(payload[off + rit * 2]) |
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(payload[off + rit * 2 + 1]) << 8);
        const float scale = __half2float(__ushort_as_half(sb));
        const std::uint8_t* packed =
            payload + off + 64 * 2 + static_cast<std::int64_t>(rit) * Q6Codec::kBytesPerRowPerGroup;
        const auto* words = reinterpret_cast<const std::uint32_t*>(packed);
        std::uint32_t w[13];
#pragma unroll
        for (int j = 0; j < 12; ++j) { w[j] = words[j]; }
        w[12] = 0u;
#pragma unroll
        for (int offset = 0; offset < Q6Codec::kGroupK; offset += 2) {
            if (offset + 1 < limit) {
                const int bitpos          = offset * Q6Codec::kBits;
                const int wi              = bitpos >> 5;
                const int sh              = bitpos & 31;
                const int bitpos1         = bitpos + Q6Codec::kBits;
                const int wi1             = bitpos1 >> 5;
                const int sh1             = bitpos1 & 31;
                const std::uint32_t bits0 = __funnelshift_r(w[wi], w[wi + 1], sh);
                const std::uint32_t bits1 = __funnelshift_r(w[wi1], w[wi1 + 1], sh1);
                const int s0              = static_cast<int>(bits0 << 26) >> 26;
                const int s1              = static_cast<int>(bits1 << 26) >> 26;
                const float2 xv =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x_group + offset));
                acc = fmaf(static_cast<float>(s0) * scale, xv.x, acc);
                acc = fmaf(static_cast<float>(s1) * scale, xv.y, acc);
            } else if (offset < limit) {
                const int bitpos         = offset * Q6Codec::kBits;
                const int wi             = bitpos >> 5;
                const int sh             = bitpos & 31;
                const std::uint32_t bits = __funnelshift_r(w[wi], w[wi + 1], sh);
                const int s              = static_cast<int>(bits << 26) >> 26;
                acc = fmaf(static_cast<float>(s) * scale, __bfloat162float(x_group[offset]), acc);
            }
        }
        return acc;
    }
};

template <class Codec>
__global__ void linear_tuned_lowbit_gemv_kernel(const __nv_bfloat16* x, const std::uint8_t* payload,
                                                __nv_bfloat16* out, std::int32_t n, std::int32_t k,
                                                std::int32_t padded_k) {
    constexpr int kWarpSize = 32;
    static_assert(Codec::kGroupK == 64);

    const int lane            = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block   = threadIdx.x / kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;
    const std::int32_t row =
        static_cast<std::int32_t>(blockIdx.x * warps_per_block + warp_in_block);
    const std::int32_t group_cnt = padded_k / Codec::kGroupK;

    if (row >= n) { return; }

    float acc = 0.0f;

    for (std::int32_t group = lane; group < group_cnt; group += kWarpSize) {
        const std::int32_t base_k = group * Codec::kGroupK;
        if (base_k >= k) { continue; }

        const std::int32_t remaining = k - base_k;
        const std::int32_t limit     = remaining < Codec::kGroupK ? remaining : Codec::kGroupK;
        acc = LowbitGemvGroup<Codec>::accumulate(payload, row, group, group_cnt, x + base_k, limit,
                                                 acc);
    }

#pragma unroll
    for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, delta);
    }

    if (lane == 0) { out[row] = __float2bfloat16(acc); }
}

__device__ __forceinline__ std::uint32_t q4_load_scale_bits(const std::uint8_t* payload,
                                                            std::int64_t off, std::int32_t rit,
                                                            int lane) {
    std::uint32_t scale_bits = 0u;
    if (lane == 0) {
        scale_bits = static_cast<std::uint16_t>(payload[off + rit * 2]) |
                     static_cast<std::uint32_t>(
                         static_cast<std::uint16_t>(payload[off + rit * 2 + 1]) << 8);
    }
    return __shfl_sync(0xffffffffu, scale_bits, 0);
}

__device__ __forceinline__ float q4_accumulate_lane_pair(
    const std::uint8_t* payload, const __nv_bfloat16* x, std::int32_t tile, std::int32_t rit,
    std::int32_t group, std::int32_t group_cnt, std::int32_t base_k, std::int32_t limit, int lane,
    float acc) {
    const int offset = lane * 2;

    const std::int64_t off =
        (static_cast<std::int64_t>(tile) * group_cnt + group) * Q4Codec::kTileBytes;
    const std::uint32_t scale_bits = q4_load_scale_bits(payload, off, rit, lane);
    const float scale = __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits)));

    const std::uint8_t* packed =
        payload + off + 64 * 2 + static_cast<std::int64_t>(rit) * Q4Codec::kBytesPerRowPerGroup;
    const std::uint32_t bits = packed[lane];
    const int s0             = static_cast<int>((bits & 0x0fu) << 28) >> 28;
    const int s1             = static_cast<int>(((bits >> 4) & 0x0fu) << 28) >> 28;

    if (offset + 1 < limit) {
        const float2 xv =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base_k + offset));
        acc = fmaf(static_cast<float>(s0) * scale, xv.x, acc);
        acc = fmaf(static_cast<float>(s1) * scale, xv.y, acc);
    } else if (offset < limit) {
        acc = fmaf(static_cast<float>(s0) * scale, __bfloat162float(x[base_k + offset]), acc);
    }
    return acc;
}

template <>
__global__ void linear_tuned_lowbit_gemv_kernel<Q4Codec>(const __nv_bfloat16* x,
                                                         const std::uint8_t* payload,
                                                         __nv_bfloat16* out, std::int32_t n,
                                                         std::int32_t k, std::int32_t padded_k) {
    constexpr int kWarpSize = 32;
    static_assert(Q4Codec::kGroupK == 64);

    const int lane            = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block   = threadIdx.x / kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;
    const std::int32_t row =
        static_cast<std::int32_t>(blockIdx.x * warps_per_block + warp_in_block);
    const std::int32_t group_cnt = padded_k / Q4Codec::kGroupK;

    if (row >= n) { return; }

    const std::int32_t tile = row / 64;
    const std::int32_t rit  = row - tile * 64;
    float acc               = 0.0f;

    for (std::int32_t group = 0; group < group_cnt; ++group) {
        const std::int32_t base_k = group * Q4Codec::kGroupK;
        if (base_k >= k) { continue; }

        const std::int32_t remaining = k - base_k;
        const std::int32_t limit =
            remaining < Q4Codec::kGroupK ? remaining : Q4Codec::kGroupK;
        acc = q4_accumulate_lane_pair(payload, x, tile, rit, group, group_cnt, base_k, limit, lane,
                                      acc);
    }

#pragma unroll
    for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, delta);
    }

    if (lane == 0) { out[row] = __float2bfloat16(acc); }
}

__device__ __forceinline__ std::uint32_t q5_load_lane_pair_bits(const std::uint32_t* words,
                                                                int offset) {
    const int bitpos = offset * Q5Codec::kBits;
    const int wi     = bitpos >> 5;
    const int sh     = bitpos & 31;
    const std::uint32_t hi = wi < 9 ? words[wi + 1] : 0u;
    return __funnelshift_r(words[wi], hi, sh);
}

__device__ __forceinline__ std::uint32_t q5_load_scale_bits(const std::uint8_t* payload,
                                                            std::int64_t off, std::int32_t rit,
                                                            int lane) {
    std::uint32_t scale_bits = 0u;
    if (lane == 0) {
        scale_bits = static_cast<std::uint16_t>(payload[off + rit * 2]) |
                     static_cast<std::uint32_t>(
                         static_cast<std::uint16_t>(payload[off + rit * 2 + 1]) << 8);
    }
    return __shfl_sync(0xffffffffu, scale_bits, 0);
}

__device__ __forceinline__ float q5_accumulate_lane_pair_tail(
    const std::uint8_t* payload, const __nv_bfloat16* x, std::int32_t tile, std::int32_t rit,
    std::int32_t group, std::int32_t group_cnt, std::int32_t base_k, std::int32_t limit, int lane,
    float acc) {
    const int offset = lane * 2;

    const std::int64_t off =
        (static_cast<std::int64_t>(tile) * group_cnt + group) * Q5Codec::kTileBytes;
    const std::uint32_t scale_bits = q5_load_scale_bits(payload, off, rit, lane);
    const float scale = __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits)));

    const std::uint8_t* packed =
        payload + off + 64 * 2 + static_cast<std::int64_t>(rit) * Q5Codec::kBytesPerRowPerGroup;
    const auto* words = reinterpret_cast<const std::uint32_t*>(packed);

    const std::uint32_t pair_bits = q5_load_lane_pair_bits(words, offset);
    const int s0                  = static_cast<int>(pair_bits << 27) >> 27;
    const int s1                  = static_cast<int>((pair_bits >> Q5Codec::kBits) << 27) >> 27;

    if (offset + 1 < limit) {
        const float2 xv =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base_k + offset));
        acc = fmaf(static_cast<float>(s0) * scale, xv.x, acc);
        acc = fmaf(static_cast<float>(s1) * scale, xv.y, acc);
    } else if (offset < limit) {
        acc = fmaf(static_cast<float>(s0) * scale, __bfloat162float(x[base_k + offset]), acc);
    }
    return acc;
}

template <>
__global__ void linear_tuned_lowbit_gemv_kernel<Q5Codec>(const __nv_bfloat16* x,
                                                         const std::uint8_t* payload,
                                                         __nv_bfloat16* out, std::int32_t n,
                                                         std::int32_t k, std::int32_t padded_k) {
    constexpr int kWarpSize = 32;
    static_assert(Q5Codec::kGroupK == 64);

    const int lane            = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block   = threadIdx.x / kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;
    const std::int32_t row =
        static_cast<std::int32_t>(blockIdx.x * warps_per_block + warp_in_block);
    const std::int32_t group_cnt = padded_k / Q5Codec::kGroupK;

    if (row >= n) { return; }

    const std::int32_t tile = row / 64;
    const std::int32_t rit  = row - tile * 64;
    const int offset        = lane * 2;
    float acc               = 0.0f;

    std::int32_t group = 0;
    const std::int32_t full_group_cnt = k / Q5Codec::kGroupK;
    for (; group + 3 < full_group_cnt; group += 4) {
        const std::int32_t base_k0 = group * Q5Codec::kGroupK;
        const std::int32_t base_k1 = base_k0 + Q5Codec::kGroupK;
        const std::int32_t base_k2 = base_k1 + Q5Codec::kGroupK;
        const std::int32_t base_k3 = base_k2 + Q5Codec::kGroupK;
        const std::int64_t off0 =
            (static_cast<std::int64_t>(tile) * group_cnt + group) * Q5Codec::kTileBytes;
        const std::int64_t off1 = off0 + Q5Codec::kTileBytes;
        const std::int64_t off2 = off1 + Q5Codec::kTileBytes;
        const std::int64_t off3 = off2 + Q5Codec::kTileBytes;

        const std::uint32_t scale_bits0 = q5_load_scale_bits(payload, off0, rit, lane);
        const std::uint32_t scale_bits1 = q5_load_scale_bits(payload, off1, rit, lane);
        const std::uint32_t scale_bits2 = q5_load_scale_bits(payload, off2, rit, lane);
        const std::uint32_t scale_bits3 = q5_load_scale_bits(payload, off3, rit, lane);

        const std::uint8_t* packed0 = payload + off0 + 64 * 2 +
                                      static_cast<std::int64_t>(rit) *
                                          Q5Codec::kBytesPerRowPerGroup;
        const std::uint8_t* packed1 = payload + off1 + 64 * 2 +
                                      static_cast<std::int64_t>(rit) *
                                          Q5Codec::kBytesPerRowPerGroup;
        const std::uint8_t* packed2 = payload + off2 + 64 * 2 +
                                      static_cast<std::int64_t>(rit) *
                                          Q5Codec::kBytesPerRowPerGroup;
        const std::uint8_t* packed3 = payload + off3 + 64 * 2 +
                                      static_cast<std::int64_t>(rit) *
                                          Q5Codec::kBytesPerRowPerGroup;
        const auto* words0 = reinterpret_cast<const std::uint32_t*>(packed0);
        const auto* words1 = reinterpret_cast<const std::uint32_t*>(packed1);
        const auto* words2 = reinterpret_cast<const std::uint32_t*>(packed2);
        const auto* words3 = reinterpret_cast<const std::uint32_t*>(packed3);

        const std::uint32_t pair_bits0 = q5_load_lane_pair_bits(words0, offset);
        const std::uint32_t pair_bits1 = q5_load_lane_pair_bits(words1, offset);
        const std::uint32_t pair_bits2 = q5_load_lane_pair_bits(words2, offset);
        const std::uint32_t pair_bits3 = q5_load_lane_pair_bits(words3, offset);
        const float2 xv0 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base_k0 + offset));
        const float2 xv1 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base_k1 + offset));
        const float2 xv2 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base_k2 + offset));
        const float2 xv3 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base_k3 + offset));

        const int s00 = static_cast<int>(pair_bits0 << 27) >> 27;
        const int s01 = static_cast<int>((pair_bits0 >> Q5Codec::kBits) << 27) >> 27;
        const int s10 = static_cast<int>(pair_bits1 << 27) >> 27;
        const int s11 = static_cast<int>((pair_bits1 >> Q5Codec::kBits) << 27) >> 27;
        const int s20 = static_cast<int>(pair_bits2 << 27) >> 27;
        const int s21 = static_cast<int>((pair_bits2 >> Q5Codec::kBits) << 27) >> 27;
        const int s30 = static_cast<int>(pair_bits3 << 27) >> 27;
        const int s31 = static_cast<int>((pair_bits3 >> Q5Codec::kBits) << 27) >> 27;
        const float scale0 =
            __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits0)));
        const float scale1 =
            __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits1)));
        const float scale2 =
            __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits2)));
        const float scale3 =
            __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits3)));

        acc = fmaf(static_cast<float>(s00) * scale0, xv0.x, acc);
        acc = fmaf(static_cast<float>(s01) * scale0, xv0.y, acc);
        acc = fmaf(static_cast<float>(s10) * scale1, xv1.x, acc);
        acc = fmaf(static_cast<float>(s11) * scale1, xv1.y, acc);
        acc = fmaf(static_cast<float>(s20) * scale2, xv2.x, acc);
        acc = fmaf(static_cast<float>(s21) * scale2, xv2.y, acc);
        acc = fmaf(static_cast<float>(s30) * scale3, xv3.x, acc);
        acc = fmaf(static_cast<float>(s31) * scale3, xv3.y, acc);
    }

    for (; group < group_cnt; ++group) {
        const std::int32_t base_k = group * Q5Codec::kGroupK;
        if (base_k >= k) { continue; }

        const std::int32_t remaining = k - base_k;
        const std::int32_t limit =
            remaining < Q5Codec::kGroupK ? remaining : Q5Codec::kGroupK;
        acc = q5_accumulate_lane_pair_tail(payload, x, tile, rit, group, group_cnt, base_k, limit,
                                           lane, acc);
    }

#pragma unroll
    for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, delta);
    }

    if (lane == 0) { out[row] = __float2bfloat16(acc); }
}

} // namespace qus::kernels::detail
