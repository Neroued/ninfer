#include "ops/sparse_moe/decode/sparse_moe_decode.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"
#include "ops/sparse_moe/sparse_moe_route.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden       = 2048;
constexpr int kExperts      = 256;
constexpr int kRouterRows   = kExperts + 1;
constexpr int kTopK         = 8;
constexpr int kIntermediate = 512;

using RankedValue = SparseMoeRankedValue;

__device__ __forceinline__ float dot_bf16_eight(const __nv_bfloat16* a, const __nv_bfloat16* b) {
    const uint4 av  = load_vec<uint4>(a);
    const uint4 bv  = load_vec<uint4>(b);
    const float2 a0 = bf16x2_bits_to_float2(av.x);
    const float2 a1 = bf16x2_bits_to_float2(av.y);
    const float2 a2 = bf16x2_bits_to_float2(av.z);
    const float2 a3 = bf16x2_bits_to_float2(av.w);
    const float2 b0 = bf16x2_bits_to_float2(bv.x);
    const float2 b1 = bf16x2_bits_to_float2(bv.y);
    const float2 b2 = bf16x2_bits_to_float2(bv.z);
    const float2 b3 = bf16x2_bits_to_float2(bv.w);
    float sum       = 0.0f;
    sum             = fmaf(a0.x, b0.x, sum);
    sum             = fmaf(a0.y, b0.y, sum);
    sum             = fmaf(a1.x, b1.x, sum);
    sum             = fmaf(a1.y, b1.y, sum);
    sum             = fmaf(a2.x, b2.x, sum);
    sum             = fmaf(a2.y, b2.y, sum);
    sum             = fmaf(a3.x, b3.x, sum);
    sum             = fmaf(a3.y, b3.y, sum);
    return sum;
}

template <int Warps>
__device__ __forceinline__ float router_row_dot(const __nv_bfloat16* x, const __nv_bfloat16* row) {
    static_assert(Warps == 4 || Warps == 8);
    constexpr int kSlice = kHidden / Warps;
    constexpr int kVecs  = kSlice / (32 * 8);
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    float sum            = 0.0f;
#pragma unroll
    for (int vector = 0; vector < kVecs; ++vector) {
        const int k = warp * kSlice + vector * 32 * 8 + lane * 8;
        sum += dot_bf16_eight(row + k, x + k);
    }
    return warp_reduce_sum(sum);
}

template <int Warps>
__global__ void sparse_moe_d1_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ router,
                                     float* __restrict__ scores) {
    __shared__ float partial[Warps];
    const int row   = static_cast<int>(blockIdx.x);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const float dot = router_row_dot<Warps>(x, router + static_cast<std::int64_t>(row) * kHidden);
    if (lane == 0) { partial[warp] = dot; }
    __syncthreads();
    if (warp == 0) {
        float value = lane < Warps ? partial[lane] : 0.0f;
        value       = warp_reduce_sum<Warps>(value);
        if (lane == 0) { scores[row] = value; }
    }
}

__global__ void sparse_moe_d2_warp_kernel(const float* __restrict__ scores, int* __restrict__ ids,
                                          float* __restrict__ alpha,
                                          float* __restrict__ shared_scale) {
    __shared__ float selected_logits[kTopK];
    sparse_moe_select_top8_warp(scores, ids, alpha, shared_scale, selected_logits);
}

__global__ void sparse_moe_d2_serial_control_kernel(const float* __restrict__ scores,
                                                    int* __restrict__ ids,
                                                    float* __restrict__ alpha,
                                                    float* __restrict__ shared_scale) {
    if (threadIdx.x != 0) { return; }
    float selected[kTopK];
    int selected_ids[kTopK];
#pragma unroll
    for (int i = 0; i < kTopK; ++i) {
        selected[i]     = -CUDART_INF_F;
        selected_ids[i] = 0x7fffffff;
    }
    for (int id = 0; id < kExperts; ++id) {
        const RankedValue value{scores[id], id, 0};
        const RankedValue tail{selected[kTopK - 1], selected_ids[kTopK - 1], 0};
        if (!sparse_moe_ranked_better(value, tail)) { continue; }
        int position = kTopK - 1;
        while (position > 0 &&
               sparse_moe_ranked_better(
                   value, RankedValue{selected[position - 1], selected_ids[position - 1], 0})) {
            selected[position]     = selected[position - 1];
            selected_ids[position] = selected_ids[position - 1];
            --position;
        }
        selected[position]     = value.value;
        selected_ids[position] = value.id;
    }
    float denominator = 0.0f;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        ids[rank]   = selected_ids[rank];
        alpha[rank] = expf(selected[rank] - selected[0]);
        denominator += alpha[rank];
    }
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) { alpha[rank] /= denominator; }
    *shared_scale = sigmoid(scores[kExperts]);
}

struct Q4Codec {
    static constexpr int kGroupK         = 64;
    static constexpr bool kD3PackedWord8 = true;
};

struct Q5Codec {
    static constexpr int kGroupK       = 64;
    static constexpr bool kPackedWord8 = true;

    __device__ static __forceinline__ void
    load_eight(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
               std::int64_t group_index, int lane_in_group, float (&weights)[8]) {
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            codes + group_index * Q5RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4);
        const std::uint8_t high_bits =
            high[group_index * Q5RowSplitStorage::kHighBytesPerGroup + lane_in_group];
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scales + group_index * Q5RowSplitStorage::kScaleBytesPerGroup);
        Q5SimtDecodeAtom::decode_eight(packed, high_bits, scale_bits, weights);
    }
};

struct Q6Codec {
    static constexpr int kGroupK       = 64;
    static constexpr bool kPackedWord8 = true;

    __device__ static __forceinline__ void
    load_eight(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
               std::int64_t group_index, int lane_in_group, float (&weights)[8]) {
        const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
            codes + group_index * Q6RowSplitStorage::kCodeBytesPerGroup + lane_in_group * 4);
        const std::uint16_t high_bits = *reinterpret_cast<const std::uint16_t*>(
            high + group_index * Q6RowSplitStorage::kHighBytesPerGroup + lane_in_group * 2);
        const auto scale_bits = *reinterpret_cast<const std::uint16_t*>(
            scales + group_index * Q6RowSplitStorage::kScaleBytesPerGroup);
        Q6SimtDecodeAtom::decode_eight(packed, high_bits, scale_bits, weights);
    }
};

struct W8Codec {
    static constexpr int kGroupK                = 32;
    static constexpr bool kD3SingleValuePerLane = true;
    static constexpr bool kD3PackedWord8        = false;
    static constexpr bool kPackedWord8          = false;

    __device__ static __forceinline__ float load_one(const std::uint8_t* codes,
                                                     const std::uint8_t* scales,
                                                     std::int64_t group_index, int lane) {
        const float scale = __half2float(
            __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scales + group_index * 2)));
        return static_cast<float>(static_cast<std::int8_t>(codes[group_index * kGroupK + lane])) *
               scale;
    }

    __device__ static __forceinline__ void
    load_pair(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
              std::int64_t group_index, int lane, float& w0, float& w1) {
        W8ScalarDecodeAtom::load_pair(codes, high, scales, group_index, lane, w0, w1);
    }
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
    if constexpr (Codec::kD3PackedWord8) {
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
            Q4SimtDecodeAtom::decode_eight(packed0, scale0, weights0);
            Q4SimtDecodeAtom::decode_eight(packed1, scale1, weights1);
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
    } else if constexpr (Codec::kD3SingleValuePerLane) {
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

template <class RoutedCodec>
__global__ void sparse_moe_d3_nine_warp_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, float* __restrict__ act) {
    __shared__ __align__(16) __nv_bfloat16 x_shared[kHidden];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid < 256) { store_vec(x_shared + tid * 8, load_vec<uint4>(x + tid * 8)); }
    __syncthreads();

    const int j = static_cast<int>(blockIdx.x);
    float gate  = 0.0f;
    float up    = 0.0f;
    if (warp < kTopK) {
        const int expert   = ids[warp];
        const int row_base = expert * 1024;
        dot_two_rows<RoutedCodec, kHidden>(routed_codes, routed_high, routed_scales, row_base + j,
                                           row_base + kIntermediate + j, x_shared, 0, kHidden, gate,
                                           up);
    } else {
        dot_two_rows<W8Codec, kHidden>(shared_codes, nullptr, shared_scales, j, kIntermediate + j,
                                       x_shared, 0, kHidden, gate, up);
    }
    if (lane == 0) { act[static_cast<std::int64_t>(warp) * kIntermediate + j] = silu(gate) * up; }
}

template <class RoutedCodec, int PathsPerBlock>
__global__ void sparse_moe_d3_path_tiled_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ token_ids,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, float* __restrict__ token_activations) {
    // Three path CTAs per token/output row expose enough blocks for the 170-SM target and keep the
    // heavier shared W8 path from holding eight completed routed warps resident.
    static_assert(PathsPerBlock > 0 && (kTopK + 1) % PathsPerBlock == 0);
    constexpr int kPathBlocks = (kTopK + 1) / PathsPerBlock;
    __shared__ __align__(16) __nv_bfloat16 x_shared[kHidden];
    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int token      = static_cast<int>(blockIdx.y) / kPathBlocks;
    const int path_block = static_cast<int>(blockIdx.y) - token * kPathBlocks;
    const int path       = path_block * PathsPerBlock + warp;
    const auto* input    = x + static_cast<std::int64_t>(token) * kHidden;
    for (int vector = tid; vector < kHidden / 8; vector += static_cast<int>(blockDim.x)) {
        store_vec(x_shared + vector * 8, load_vec<uint4>(input + vector * 8));
    }
    __syncthreads();

    const int j = static_cast<int>(blockIdx.x);
    float gate  = 0.0f;
    float up    = 0.0f;
    if (path < kTopK) {
        const int expert   = token_ids[token * kTopK + path];
        const int row_base = expert * (2 * kIntermediate);
        dot_two_rows<RoutedCodec, kHidden>(routed_codes, routed_high, routed_scales, row_base + j,
                                           row_base + kIntermediate + j, x_shared, 0, kHidden, gate,
                                           up);
    } else {
        dot_two_rows<W8Codec, kHidden>(shared_codes, nullptr, shared_scales, j, kIntermediate + j,
                                       x_shared, 0, kHidden, gate, up);
    }
    if (lane == 0) {
        token_activations[(static_cast<std::int64_t>(token) * (kTopK + 1) + path) * kIntermediate +
                          j] = silu(gate) * up;
    }
}

template <class RoutedCodec>
__global__ void sparse_moe_d3_balanced_kernel(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, float* __restrict__ act) {
    __shared__ __align__(16) __nv_bfloat16 x_shared[kHidden];
    __shared__ float shared_gate_partial[kTopK];
    __shared__ float shared_up_partial[kTopK];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    store_vec(x_shared + tid * 8, load_vec<uint4>(x + tid * 8));
    __syncthreads();

    const int j        = static_cast<int>(blockIdx.x);
    const int expert   = ids[warp];
    const int row_base = expert * 1024;
    float gate         = 0.0f;
    float up           = 0.0f;
    dot_two_rows<RoutedCodec, kHidden>(routed_codes, routed_high, routed_scales, row_base + j,
                                       row_base + kIntermediate + j, x_shared, 0, kHidden, gate,
                                       up);
    if (lane == 0) { act[static_cast<std::int64_t>(warp) * kIntermediate + j] = silu(gate) * up; }

    float shared_gate = 0.0f;
    float shared_up   = 0.0f;
    dot_two_rows<W8Codec, kHidden>(shared_codes, nullptr, shared_scales, j, kIntermediate + j,
                                   x_shared, warp * 256, (warp + 1) * 256, shared_gate, shared_up);
    if (lane == 0) {
        shared_gate_partial[warp] = shared_gate;
        shared_up_partial[warp]   = shared_up;
    }
    __syncthreads();
    if (warp == 0) {
        float gate_sum = lane < kTopK ? shared_gate_partial[lane] : 0.0f;
        float up_sum   = lane < kTopK ? shared_up_partial[lane] : 0.0f;
        gate_sum       = warp_reduce_sum<kTopK>(gate_sum);
        up_sum         = warp_reduce_sum<kTopK>(up_sum);
        if (lane == 0) {
            act[static_cast<std::int64_t>(kTopK) * kIntermediate + j] = silu(gate_sum) * up_sum;
        }
    }
}

template <class Codec, int Rows>
__device__ __forceinline__ void dot_fp32_rows(const std::uint8_t* codes, const std::uint8_t* high,
                                              const std::uint8_t* scales, int row_base,
                                              const float* x, int first_group, int last_group,
                                              float (&result)[Rows]) {
    constexpr int kGroups = kIntermediate / Codec::kGroupK;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    float acc[Rows];
#pragma unroll
    for (int row = 0; row < Rows; ++row) { acc[row] = 0.0f; }
    if constexpr (Codec::kPackedWord8) {
        // Q5/Q6 use the same eight-value lane ownership as D3. The high plane and FP16 scale are
        // decoded exactly from their registered row-split codec before FP32 accumulation.
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

template <class RoutedCodec, int Rows, SparseMoeEpilogue Epilogue>
__global__ void sparse_moe_d4_nine_warp_kernel(
    const int* __restrict__ ids, const float* __restrict__ alpha,
    const float* __restrict__ shared_scale, const float* __restrict__ act,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, __nv_bfloat16* __restrict__ destination) {
    __shared__ float paths[kTopK + 1][Rows];
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int row_base = static_cast<int>(blockIdx.x) * Rows;
    if (warp < kTopK) {
        const int expert = ids[warp];
        float dot[Rows];
        dot_fp32_rows<RoutedCodec, Rows>(routed_codes, routed_high, routed_scales,
                                         expert * kHidden + row_base,
                                         act + static_cast<std::int64_t>(warp) * kIntermediate, 0,
                                         kIntermediate / RoutedCodec::kGroupK, dot);
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < Rows; ++row) { paths[warp][row] = alpha[warp] * dot[row]; }
        }
    } else {
        float dot[Rows];
        dot_fp32_rows<W8Codec, Rows>(shared_codes, nullptr, shared_scales, row_base,
                                     act + static_cast<std::int64_t>(kTopK) * kIntermediate, 0,
                                     kIntermediate / W8Codec::kGroupK, dot);
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < Rows; ++row) { paths[kTopK][row] = *shared_scale * dot[row]; }
        }
    }
    __syncthreads();
    if (warp == 0 && lane < Rows) {
        static_assert(Epilogue == SparseMoeEpilogue::AddResidual);
        float value = __bfloat162float(destination[row_base + lane]);
#pragma unroll
        for (int path = 0; path < kTopK + 1; ++path) { value += paths[path][lane]; }
        destination[row_base + lane] = __float2bfloat16_rn(value);
    }
}

template <class RoutedCodec>
__global__ void sparse_moe_d4_token_kernel(
    const int* __restrict__ token_ids, const float* __restrict__ token_alpha,
    const float* __restrict__ shared_scale, const float* __restrict__ token_activations,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, __nv_bfloat16* __restrict__ destination) {
    // Token is a grid dimension rather than an in-CTA serial loop. This removes the small-T wave
    // tail while retaining the deterministic rank-order FP32 epilogue below.
    __shared__ float paths[kTopK + 1];
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row   = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const float* act =
        token_activations + static_cast<std::int64_t>(token) * (kTopK + 1) * kIntermediate;
    if (warp < kTopK) {
        const int expert = token_ids[token * kTopK + warp];
        float dot[1];
        dot_fp32_rows<RoutedCodec, 1>(routed_codes, routed_high, routed_scales,
                                      expert * kHidden + row,
                                      act + static_cast<std::int64_t>(warp) * kIntermediate, 0,
                                      kIntermediate / RoutedCodec::kGroupK, dot);
        if (lane == 0) { paths[warp] = token_alpha[token * kTopK + warp] * dot[0]; }
    } else {
        float dot[1];
        dot_fp32_rows<W8Codec, 1>(shared_codes, nullptr, shared_scales, row,
                                  act + static_cast<std::int64_t>(kTopK) * kIntermediate, 0,
                                  kIntermediate / W8Codec::kGroupK, dot);
        if (lane == 0) { paths[kTopK] = shared_scale[token] * dot[0]; }
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        __nv_bfloat16* output = destination + static_cast<std::int64_t>(token) * kHidden + row;
        float value           = __bfloat162float(*output);
#pragma unroll
        for (int path = 0; path < kTopK + 1; ++path) { value += paths[path]; }
        *output = __float2bfloat16_rn(value);
    }
}

template <class RoutedCodec, SparseMoeEpilogue Epilogue>
__global__ void sparse_moe_d4_balanced_r4_kernel(
    const int* __restrict__ ids, const float* __restrict__ alpha,
    const float* __restrict__ shared_scale, const float* __restrict__ act,
    const std::uint8_t* __restrict__ routed_codes, const std::uint8_t* __restrict__ routed_high,
    const std::uint8_t* __restrict__ routed_scales, const std::uint8_t* __restrict__ shared_codes,
    const std::uint8_t* __restrict__ shared_scales, __nv_bfloat16* __restrict__ destination) {
    __shared__ float routed_partial[kTopK][4];
    __shared__ float shared_partial[kTopK][4];
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int row_base = static_cast<int>(blockIdx.x) * 4;

    const int expert = ids[warp];
    float routed[4];
    dot_fp32_rows<RoutedCodec, 4>(routed_codes, routed_high, routed_scales,
                                  expert * kHidden + row_base,
                                  act + static_cast<std::int64_t>(warp) * kIntermediate, 0,
                                  kIntermediate / RoutedCodec::kGroupK, routed);

    float shared[4];
    constexpr int kSharedGroupsPerWarp = 64 / W8Codec::kGroupK;
    dot_fp32_rows<W8Codec, 4>(shared_codes, nullptr, shared_scales, row_base,
                              act + static_cast<std::int64_t>(kTopK) * kIntermediate,
                              warp * kSharedGroupsPerWarp, (warp + 1) * kSharedGroupsPerWarp,
                              shared);
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            routed_partial[warp][row] = routed[row];
            shared_partial[warp][row] = shared[row];
        }
    }
    __syncthreads();

    if (warp == 0 && lane < 4) {
        static_assert(Epilogue == SparseMoeEpilogue::AddResidual);
        float value = __bfloat162float(destination[row_base + lane]);
#pragma unroll
        for (int route = 0; route < kTopK; ++route) {
            value += alpha[route] * routed_partial[route][lane];
            value += *shared_scale * shared_partial[route][lane];
        }
        destination[row_base + lane] = __float2bfloat16_rn(value);
    }
}

template <class Codec>
void launch_d3_codec(const Tensor& x, const SparseMoeWeights& weights,
                     const SparseMoeDecodeWorkspace& workspace, SparseMoeD3Schedule schedule,
                     cudaStream_t stream) {
    const auto* input         = static_cast<const __nv_bfloat16*>(x.data);
    const auto* ids           = static_cast<const int*>(workspace.ids.data);
    auto* act                 = static_cast<float*>(workspace.scratch.data);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata);
    const auto* routed_high   = static_cast<const std::uint8_t*>(weights.routed_gate_up.qhigh);
    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    switch (schedule) {
    case SparseMoeD3Schedule::NineWarp:
        sparse_moe_d3_nine_warp_kernel<Codec><<<kIntermediate, 9 * 32, 0, stream>>>(
            input, ids, routed_codes, routed_high, routed_scales, shared_codes, shared_scales, act);
        CUDA_CHECK(cudaGetLastError());
        return;
    case SparseMoeD3Schedule::BalancedEightWarp:
        sparse_moe_d3_balanced_kernel<Codec><<<kIntermediate, 8 * 32, 0, stream>>>(
            input, ids, routed_codes, routed_high, routed_scales, shared_codes, shared_scales, act);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    throw std::logic_error("sparse_moe: unknown D3 schedule");
}

template <class Codec>
void launch_d4_codec(const SparseMoeWeights& weights, Tensor& destination,
                     const SparseMoeDecodeWorkspace& workspace, SparseMoeD4Schedule schedule,
                     cudaStream_t stream) {
    const auto* ids           = static_cast<const int*>(workspace.ids.data);
    const auto* alpha         = static_cast<const float*>(workspace.alpha.data);
    const auto* shared_scale  = static_cast<const float*>(workspace.shared_scale.data);
    const auto* act           = static_cast<const float*>(workspace.scratch.data);
    const auto* routed_codes  = static_cast<const std::uint8_t*>(weights.routed_down.qdata);
    const auto* routed_high   = static_cast<const std::uint8_t*>(weights.routed_down.qhigh);
    const auto* routed_scales = static_cast<const std::uint8_t*>(weights.routed_down.scales);
    const auto* shared_codes  = static_cast<const std::uint8_t*>(weights.shared_down.qdata);
    const auto* shared_scales = static_cast<const std::uint8_t*>(weights.shared_down.scales);
    auto* output              = static_cast<__nv_bfloat16*>(destination.data);
    switch (schedule) {
    case SparseMoeD4Schedule::NineWarpRows1:
        sparse_moe_d4_nine_warp_kernel<Codec, 1, SparseMoeEpilogue::AddResidual>
            <<<kHidden, 9 * 32, 0, stream>>>(ids, alpha, shared_scale, act, routed_codes,
                                             routed_high, routed_scales, shared_codes,
                                             shared_scales, output);
        CUDA_CHECK(cudaGetLastError());
        return;
    case SparseMoeD4Schedule::BalancedEightWarpRows4:
        sparse_moe_d4_balanced_r4_kernel<Codec, SparseMoeEpilogue::AddResidual>
            <<<kHidden / 4, 8 * 32, 0, stream>>>(ids, alpha, shared_scale, act, routed_codes,
                                                 routed_high, routed_scales, shared_codes,
                                                 shared_scales, output);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    throw std::logic_error("sparse_moe: unknown D4 schedule");
}

template <class Codec>
void launch_d3_small_t_codec(const Tensor& x, const SparseMoeWeights& weights, const int* token_ids,
                             float* token_activations, std::int32_t tokens, cudaStream_t stream) {
    constexpr int kPathsPerBlock = 3;
    constexpr int kPathBlocks    = (kTopK + 1) / kPathsPerBlock;
    sparse_moe_d3_path_tiled_kernel<Codec, kPathsPerBlock>
        <<<dim3(kIntermediate, tokens * kPathBlocks), kPathsPerBlock * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), token_ids,
            static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata),
            static_cast<const std::uint8_t*>(weights.routed_gate_up.qhigh),
            static_cast<const std::uint8_t*>(weights.routed_gate_up.scales),
            static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata),
            static_cast<const std::uint8_t*>(weights.shared_gate_up.scales), token_activations);
    CUDA_CHECK(cudaGetLastError());
}

template <class Codec>
void launch_d4_small_t_codec(const SparseMoeWeights& weights, Tensor& destination,
                             const int* token_ids, const float* token_alpha,
                             const float* shared_scale, const float* token_activations,
                             std::int32_t tokens, cudaStream_t stream) {
    sparse_moe_d4_token_kernel<Codec><<<dim3(kHidden, tokens), 9 * 32, 0, stream>>>(
        token_ids, token_alpha, shared_scale, token_activations,
        static_cast<const std::uint8_t*>(weights.routed_down.qdata),
        static_cast<const std::uint8_t*>(weights.routed_down.qhigh),
        static_cast<const std::uint8_t*>(weights.routed_down.scales),
        static_cast<const std::uint8_t*>(weights.shared_down.qdata),
        static_cast<const std::uint8_t*>(weights.shared_down.scales),
        static_cast<__nv_bfloat16*>(destination.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void sparse_moe_decode_launch_d1(const Tensor& x, const Weight& router_shared_gate,
                                 const SparseMoeDecodeWorkspace& workspace,
                                 SparseMoeD1Schedule schedule, cudaStream_t stream) {
    const auto* input  = static_cast<const __nv_bfloat16*>(x.data);
    const auto* router = static_cast<const __nv_bfloat16*>(router_shared_gate.qdata);
    auto* scores       = static_cast<float*>(workspace.scratch.data);
    switch (schedule) {
    case SparseMoeD1Schedule::RowCta4:
        sparse_moe_d1_kernel<4><<<kRouterRows, 4 * 32, 0, stream>>>(input, router, scores);
        CUDA_CHECK(cudaGetLastError());
        return;
    case SparseMoeD1Schedule::RowCta8:
        sparse_moe_d1_kernel<8><<<kRouterRows, 8 * 32, 0, stream>>>(input, router, scores);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    throw std::logic_error("sparse_moe: unknown D1 schedule");
}

void sparse_moe_decode_launch_d2(const SparseMoeDecodeWorkspace& workspace,
                                 SparseMoeD2Schedule schedule, cudaStream_t stream) {
    const auto* scores = static_cast<const float*>(workspace.scratch.data);
    auto* ids          = static_cast<int*>(workspace.ids.data);
    auto* alpha        = static_cast<float*>(workspace.alpha.data);
    auto* shared_scale = static_cast<float*>(workspace.shared_scale.data);
    switch (schedule) {
    case SparseMoeD2Schedule::SerialControl:
        sparse_moe_d2_serial_control_kernel<<<1, 256, 0, stream>>>(scores, ids, alpha,
                                                                   shared_scale);
        CUDA_CHECK(cudaGetLastError());
        return;
    case SparseMoeD2Schedule::WarpRegister:
        sparse_moe_d2_warp_kernel<<<1, 32, 0, stream>>>(scores, ids, alpha, shared_scale);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    throw std::logic_error("sparse_moe: unknown D2 schedule");
}

void sparse_moe_decode_launch_d3(const Tensor& x, const SparseMoeWeights& weights,
                                 const SparseMoeDecodeWorkspace& workspace,
                                 SparseMoeD3Schedule schedule, cudaStream_t stream) {
    switch (weights.routed_gate_up.qtype) {
    case QType::Q4G64_F16S:
        launch_d3_codec<Q4Codec>(x, weights, workspace, schedule, stream);
        return;
    case QType::W8G32_F16S:
        launch_d3_codec<W8Codec>(x, weights, workspace, schedule, stream);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported D3 codec");
    }
}

void sparse_moe_decode_launch_d4(const SparseMoeWeights& weights, Tensor& destination,
                                 const SparseMoeDecodeWorkspace& workspace,
                                 SparseMoeD4Schedule schedule, cudaStream_t stream) {
    switch (weights.routed_down.qtype) {
    case QType::Q5G64_F16S:
        launch_d4_codec<Q5Codec>(weights, destination, workspace, schedule, stream);
        return;
    case QType::Q6G64_F16S:
        launch_d4_codec<Q6Codec>(weights, destination, workspace, schedule, stream);
        return;
    case QType::W8G32_F16S:
        launch_d4_codec<W8Codec>(weights, destination, workspace, schedule, stream);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported D4 codec");
    }
}

void sparse_moe_decode_launch_d3_small_t(const Tensor& x, const SparseMoeWeights& weights,
                                         const int* token_ids, float* token_activations,
                                         std::int32_t tokens, cudaStream_t stream) {
    switch (weights.routed_gate_up.qtype) {
    case QType::Q4G64_F16S:
        launch_d3_small_t_codec<Q4Codec>(x, weights, token_ids, token_activations, tokens, stream);
        return;
    case QType::W8G32_F16S:
        launch_d3_small_t_codec<W8Codec>(x, weights, token_ids, token_activations, tokens, stream);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported small-T D3 codec");
    }
}

void sparse_moe_decode_launch_d4_small_t(const SparseMoeWeights& weights, Tensor& destination,
                                         const int* token_ids, const float* token_alpha,
                                         const float* shared_scale, const float* token_activations,
                                         std::int32_t tokens, cudaStream_t stream) {
    switch (weights.routed_down.qtype) {
    case QType::Q5G64_F16S:
        launch_d4_small_t_codec<Q5Codec>(weights, destination, token_ids, token_alpha, shared_scale,
                                         token_activations, tokens, stream);
        return;
    case QType::Q6G64_F16S:
        launch_d4_small_t_codec<Q6Codec>(weights, destination, token_ids, token_alpha, shared_scale,
                                         token_activations, tokens, stream);
        return;
    case QType::W8G32_F16S:
        launch_d4_small_t_codec<W8Codec>(weights, destination, token_ids, token_alpha, shared_scale,
                                         token_activations, tokens, stream);
        return;
    default:
        throw std::invalid_argument("sparse_moe: unsupported small-T D4 codec");
    }
}

void sparse_moe_decode_launch(const Tensor& x, const SparseMoeWeights& weights, Tensor& destination,
                              const SparseMoeDecodeWorkspace& workspace,
                              const SparseMoeDecodePlan& plan, cudaStream_t stream) {
    sparse_moe_decode_launch_d1(x, weights.router_shared_gate, workspace, plan.d1, stream);
    sparse_moe_decode_launch_d2(workspace, plan.d2, stream);
    sparse_moe_decode_launch_d3(x, weights, workspace, plan.d3, stream);
    sparse_moe_decode_launch_d4(weights, destination, workspace, plan.d4, stream);
}

} // namespace ninfer::ops::detail
