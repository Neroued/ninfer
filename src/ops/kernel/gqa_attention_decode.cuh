#pragma once

// ninfer::ops - split-KV GQA small-T attention shared scaffolding. The bf16 and
// int8 partial kernels live in gqa_attention_decode_bf16.cuh and
// gqa_attention_decode_i8.cuh respectively; they are fully separate kernels (no
// shared body) so each KV format can be optimized independently. This header owns
// only what both share: layout constants, device helpers, and the split reducer.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/gqa_attention_geometry.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaHeadDim = 256;

struct GqaAppendInput {
    static constexpr bool writes_cache = true;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
};

struct GqaCachedInput {
    static constexpr bool writes_cache = false;
};

__device__ __forceinline__ std::int64_t gqa_cache_index(int kv_head, int d, int position,
                                                        int padded_context) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaHeadDim) *
                                              (static_cast<std::int64_t>(position) +
                                               static_cast<std::int64_t>(padded_context) * kv_head);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_q_index(int q_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaHeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(Geometry::QHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_new_index(int kv_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_partial_acc_index(int q_head, int d, int token,
                                                              int split, int tokens) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(Geometry::QHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split));
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_partial_stat_index(int q_head, int token, int split,
                                                               int tokens) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(Geometry::QHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split);
}

template <typename Geometry>
__device__ __forceinline__ bool gqa_valid_q_head(int kv_head, int q_head) {
    return kv_head >= 0 && kv_head < Geometry::KVHeads && q_head >= kv_head * Geometry::GroupSize &&
           q_head < (kv_head + 1) * Geometry::GroupSize && q_head < Geometry::QHeads;
}

template <typename Geometry>
__device__ __forceinline__ int gqa_small_t_default_splits(int window) {
    int target_keys_per_split = 480 / Geometry::DecodeSplitScale;
    if (window <= 4096) {
        target_keys_per_split = 64 / Geometry::DecodeSplitScale;
    } else if (window <= 8198) {
        target_keys_per_split = 128 / Geometry::DecodeSplitScale;
    } else if (window <= 16390) {
        target_keys_per_split = 256 / Geometry::DecodeSplitScale;
    }
    constexpr int kMinSplits = 4 * Geometry::DecodeSplitScale;
    int splits               = div_up(window, target_keys_per_split);
    splits                   = splits > kMinSplits ? splits : kMinSplits;
    return splits < Geometry::DecodeSplits ? splits : Geometry::DecodeSplits;
}

template <typename Geometry, bool Int8>
__device__ __forceinline__ int gqa_small_t_active_splits(int window, int launch_capacity,
                                                         int tokens) {
    if (window <= 0) { return launch_capacity; }
    int splits = 0;
    if constexpr (Int8) {
        if (tokens == 5 && window > 128 && window <= 512) {
            splits = div_up(window, 32 / Geometry::DecodeSplitScale);
        } else if (tokens == 6 && window > 128 && window <= 160) {
            splits = div_up(window, 24 / Geometry::DecodeSplitScale);
        } else if (tokens == 6 && window > 5000 && window <= 8198) {
            splits             = div_up(window, 192 / Geometry::DecodeSplitScale);
            constexpr int kMin = 4 * Geometry::DecodeSplitScale;
            constexpr int kMax = 42 * Geometry::DecodeSplitScale;
            splits             = splits > kMin ? splits : kMin;
            splits             = splits < kMax ? splits : kMax;
        } else {
            splits = gqa_small_t_default_splits<Geometry>(window);
        }
    } else {
        splits = gqa_small_t_default_splits<Geometry>(window);
    }
    return splits < launch_capacity ? splits : launch_capacity;
}

__device__ __forceinline__ int gqa_small_t_tc_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ int gqa_small_t_tc_swz32(int row, int col) {
    return (((col >> 3) ^ (row & 3)) << 3) | (col & 7);
}

// Signed int8 QK MMA, k=32 contraction. A = 16x32 s8 (4 regs/thread, 4 s8 each),
// B = 8x32 s8 col-major (2 regs/thread), D = 16x8 s32 (4 regs/thread). The A/B
// register byte layout is identical to the m16n8k16 bf16 fragments loaded by
// ldmatrix_x4/x2 over a d-contiguous int8 tile reinterpreted as
// b16 (two packed int8 per 16-bit lane), so the same ldmatrix helpers and XOR
// swizzle feed this MMA. The s32 accumulator layout matches the bf16 f32
// accumulator (c0/c1 -> row groupID, c2/c3 -> row groupID+8), so score
// consumption is unchanged; only per-64-group scale rescale differs.
template <typename Geometry>
__device__ __forceinline__ void gqa_small_t_tc_row_to_qt(int row, int tokens, int kv_head,
                                                         int& q_head, int& token) {
    token             = row / Geometry::GroupSize;
    const int local_q = row - token * Geometry::GroupSize;
    q_head            = kv_head * Geometry::GroupSize + local_q;
}

template <typename Geometry, int DChunk, bool Int8>
__launch_bounds__(256) __global__
    void gqa_attention_small_t_reduce_output_kernel(const __nv_bfloat16* partial_acc,
                                                    const float* partial_m, const float* partial_l,
                                                    const std::int32_t* positions,
                                                    std::int32_t tokens, std::int32_t split_count,
                                                    __nv_bfloat16* out) {
    static_assert(DChunk > 0 && DChunk <= kGqaHeadDim);

    const int q_head  = static_cast<int>(blockIdx.x);
    const int d_start = static_cast<int>(blockIdx.y) * DChunk;
    const int token   = static_cast<int>(blockIdx.z);
    const int tid     = threadIdx.x;
    if (q_head >= Geometry::QHeads || token >= tokens) { return; }
    const int last_pos = positions[tokens - 1];
    const int window   = last_pos + 1;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, Int8>(window, split_count, tokens);

    __shared__ float reduce[256];

    float local_m = -CUDART_INF_F;
    for (int split = tid; split < active_split_count; split += blockDim.x) {
        local_m = fmaxf(local_m,
                        partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)]);
    }
    reduce[tid] = local_m;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]); }
        __syncthreads();
    }
    const float head_m = reduce[0];
    __syncthreads();

    if (head_m == -CUDART_INF_F) {
        const int d = d_start + tid;
        if (tid < DChunk && d < kGqaHeadDim) {
            out[gqa_q_index<Geometry>(q_head, d, token)] = __float2bfloat16(0.0f);
        }
        return;
    }

    float local_l = 0.0f;
    for (int split = tid; split < active_split_count; split += blockDim.x) {
        const float tile_l =
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)];
        if (tile_l > 0.0f) {
            local_l +=
                tile_l *
                expf(partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] -
                     head_m);
        }
    }
    reduce[tid] = local_l;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] += reduce[tid + stride]; }
        __syncthreads();
    }
    const float head_l = reduce[0];

    const int d = d_start + tid;
    if (tid >= DChunk || d >= kGqaHeadDim) { return; }

    float numerator = 0.0f;
    if (head_l > 0.0f) {
        for (int split = 0; split < active_split_count; ++split) {
            const float tile_l =
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)];
            if (tile_l <= 0.0f) { continue; }
            const float weight = expf(
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] - head_m);
            numerator +=
                __bfloat162float(
                    partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)]) *
                weight;
        }
    }
    const float value                            = (head_l > 0.0f) ? numerator / head_l : 0.0f;
    out[gqa_q_index<Geometry>(q_head, d, token)] = __float2bfloat16(value);
}

} // namespace ninfer::ops
