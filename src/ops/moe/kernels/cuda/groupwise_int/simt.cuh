#pragma once

#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/quant/integer.cuh"

namespace ninfer::ops::detail::moe_cuda::groupwise_int {
using common::Q5SimtDecodeAtom;
using common::Q6SimtDecodeAtom;
using common::Q8ScalarDecodeAtom;
using common::load_vec;
using common::warp_reduce_sum;
using common::bf16x2_bits_to_float2;

struct Q4Codec {
    static constexpr int kGroupK             = 64;
    static constexpr bool kGateUpPackedWord8 = true;
};

struct Q5Codec : Q5SimtDecodeAtom {
    static constexpr int kGroupK       = 64;
    static constexpr bool kPackedWord8 = true;
};

struct Q6Codec : Q6SimtDecodeAtom {
    static constexpr int kGroupK       = 64;
    static constexpr bool kPackedWord8 = true;
};

struct Q8Codec : Q8ScalarDecodeAtom {
    static constexpr bool kGateUpSingleValuePerLane = true;
    static constexpr bool kGateUpPackedWord8        = false;
    static constexpr bool kPackedWord8              = false;
};

template <class Codec, int K>
__device__ __forceinline__ void dot_two_rows(const std::uint8_t* codes, const std::uint8_t* high,
                                             const std::uint8_t* scales, int row0, int row1,
                                             const __nv_bfloat16* x, int k_begin, int k_end,
                                             float& result0, float& result1) {
    constexpr int kGroups = K / Codec::kGroupK;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc0            = 0.0f;
    float acc1            = 0.0f;
    const int first_group = k_begin / Codec::kGroupK;
    const int last_group  = k_end / Codec::kGroupK;
    if constexpr (Codec::kGateUpPackedWord8) {
        // Four adjacent Q4 groups form one 128-byte warp transaction. Each lane owns eight
        // consecutive K values, so one mantissa decode feeds eight FP32 FMAs instead of issuing
        // four scalar code-pair/decode iterations.
        const int lane_group    = lane >> 3;
        const int lane_in_group = lane & 7;
        for (int group_base = first_group; group_base < last_group; group_base += 4) {
            const int group           = group_base + lane_group;
            const std::int64_t index0 = static_cast<std::int64_t>(row0) * kGroups + group;
            const std::int64_t index1 = static_cast<std::int64_t>(row1) * kGroups + group;
            const std::uint32_t packed0 =
                *reinterpret_cast<const std::uint32_t*>(codes + index0 * 32 + lane_in_group * 4);
            const std::uint32_t packed1 =
                *reinterpret_cast<const std::uint32_t*>(codes + index1 * 32 + lane_in_group * 4);
            const auto scale0 = *reinterpret_cast<const std::uint16_t*>(scales + index0 * 2);
            const auto scale1 = *reinterpret_cast<const std::uint16_t*>(scales + index1 * 2);
            float weights0[8];
            float weights1[8];
            common::Q4SimtDecodeAtom::decode_eight(packed0, scale0, weights0);
            common::Q4SimtDecodeAtom::decode_eight(packed1, scale1, weights1);
            const uint4 input     = load_vec<uint4>(x + group * Codec::kGroupK + lane_in_group * 8);
            const float2 x0       = bf16x2_bits_to_float2(input.x);
            const float2 x1       = bf16x2_bits_to_float2(input.y);
            const float2 x2       = bf16x2_bits_to_float2(input.z);
            const float2 x3       = bf16x2_bits_to_float2(input.w);
            const float values[8] = {x0.x, x0.y, x1.x, x1.y, x2.x, x2.y, x3.x, x3.y};
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                acc0 = fmaf(weights0[item], values[item], acc0);
                acc1 = fmaf(weights1[item], values[item], acc1);
            }
        }
    } else if constexpr (Codec::kGateUpSingleValuePerLane) {
        for (int group = first_group; group < last_group; ++group) {
            const std::int64_t index0 = static_cast<std::int64_t>(row0) * kGroups + group;
            const std::int64_t index1 = static_cast<std::int64_t>(row1) * kGroups + group;
            const float w0            = Codec::load_one(codes, scales, index0, lane);
            const float w1            = Codec::load_one(codes, scales, index1, lane);
            const float xv            = __bfloat162float(x[group * Codec::kGroupK + lane]);
            acc0                      = fmaf(w0, xv, acc0);
            acc1                      = fmaf(w1, xv, acc1);
        }
    } else if (lane < Codec::kGroupK / 2) {
        for (int group = first_group; group < last_group; ++group) {
            float w00, w01, w10, w11;
            Codec::load_pair(codes, high, scales, static_cast<std::int64_t>(row0) * kGroups + group,
                             lane, w00, w01);
            Codec::load_pair(codes, high, scales, static_cast<std::int64_t>(row1) * kGroups + group,
                             lane, w10, w11);
            const int k     = group * Codec::kGroupK + lane * 2;
            const float2 xv = __bfloat1622float2(load_vec<__nv_bfloat162>(x + k));
            acc0            = fmaf(w00, xv.x, acc0);
            acc0            = fmaf(w01, xv.y, acc0);
            acc1            = fmaf(w10, xv.x, acc1);
            acc1            = fmaf(w11, xv.y, acc1);
        }
    }
    result0 = warp_reduce_sum(acc0);
    result1 = warp_reduce_sum(acc1);
}

template <class Codec, int Rows, int K>
__device__ __forceinline__ void dot_fp32_rows(const std::uint8_t* codes, const std::uint8_t* high,
                                              const std::uint8_t* scales, int row_base,
                                              const float* x, int first_group, int last_group,
                                              float (&result)[Rows]) {
    constexpr int kGroups = K / Codec::kGroupK;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc[Rows];
#pragma unroll
    for (int row = 0; row < Rows; ++row) { acc[row] = 0.0f; }
    if constexpr (Codec::kPackedWord8) {
        // Q5/Q6 use the same eight-value lane ownership as gate/up. The high plane and FP16 scale
        // are decoded exactly from their registered row-split codec before FP32 accumulation.
        const int lane_group    = lane >> 3;
        const int lane_in_group = lane & 7;
        for (int group_base = first_group; group_base < last_group; group_base += 4) {
            const int group = group_base + lane_group;
            const float4 x0 = load_vec<float4>(x + group * Codec::kGroupK + lane_in_group * 8);
            const float4 x1 = load_vec<float4>(x + group * Codec::kGroupK + lane_in_group * 8 + 4);
            const float values[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                const std::int64_t group_index =
                    static_cast<std::int64_t>(row_base + row) * kGroups + group;
                float weights[8];
                Codec::load_eight(codes, high, scales, group_index, lane_in_group, weights);
#pragma unroll
                for (int item = 0; item < 8; ++item) {
                    acc[row] = fmaf(weights[item], values[item], acc[row]);
                }
            }
        }
    } else if (lane < Codec::kGroupK / 2) {
        for (int group = first_group; group < last_group; ++group) {
            const int k     = group * Codec::kGroupK + lane * 2;
            const float2 xv = load_vec<float2>(x + k);
#pragma unroll
            for (int row = 0; row < Rows; ++row) {
                float w0, w1;
                Codec::load_pair(codes, high, scales,
                                 static_cast<std::int64_t>(row_base + row) * kGroups + group, lane,
                                 w0, w1);
                acc[row] = fmaf(w0, xv.x, acc[row]);
                acc[row] = fmaf(w1, xv.y, acc[row]);
            }
        }
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row) { result[row] = warp_reduce_sum(acc[row]); }
}

} // namespace ninfer::ops::detail::moe_cuda::groupwise_int
