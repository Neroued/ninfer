#pragma once

// ninfer::kernels - split-KV GQA small-T attention shared scaffolding. The bf16 and
// int8 partial kernels live in gqa_attention_decode_bf16.cuh and
// gqa_attention_decode_i8.cuh respectively; they are fully separate kernels (no
// shared body) so each KV format can be optimized independently. This header owns
// only what both share: layout constants, device helpers, and the split reducer.

#include "kernels/common/math.cuh"
#include "kernels/common/mma.cuh"
#include "kernels/common/warp.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::kernels {

inline constexpr int kGqaHeadDim   = 256;
inline constexpr int kGqaQHeads    = 24;
inline constexpr int kGqaKVHeads   = 4;
inline constexpr int kGqaGroupSize = 6;

__device__ __forceinline__ std::int64_t gqa_cache_index(int kv_head, int d, int position,
                                                        int padded_context) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaHeadDim) *
                                              (static_cast<std::int64_t>(position) +
                                               static_cast<std::int64_t>(padded_context) * kv_head);
}

__device__ __forceinline__ std::int64_t gqa_q_index(int q_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaHeadDim) *
               (static_cast<std::int64_t>(q_head) + static_cast<std::int64_t>(kGqaQHeads) * token);
}

__device__ __forceinline__ std::int64_t gqa_kv_new_index(int kv_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaHeadDim) *
                                              (static_cast<std::int64_t>(kv_head) +
                                               static_cast<std::int64_t>(kGqaKVHeads) * token);
}

__device__ __forceinline__ std::int64_t gqa_partial_acc_index(int q_head, int d, int token,
                                                              int split, int tokens) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(kGqaQHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split));
}

__device__ __forceinline__ std::int64_t gqa_partial_stat_index(int q_head, int token, int split,
                                                               int tokens) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(kGqaQHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split);
}

__device__ __forceinline__ bool gqa_valid_q_head(int kv_head, int q_head) {
    return kv_head >= 0 && kv_head < kGqaKVHeads && q_head >= kv_head * kGqaGroupSize &&
           q_head < (kv_head + 1) * kGqaGroupSize && q_head < kGqaQHeads;
}

__device__ __forceinline__ int gqa_small_t_active_splits(int window, int max_splits, int tokens) {
    if (window <= 0) { return max_splits; }
    // These int8 launch ranges deliberately use one 32-key tile per split.
    // BF16 passes its ordinary split count here, so returning max_splits leaves
    // the BF16 schedule unchanged while keeping the shared reducer in sync.
    if ((tokens == 5 && window > 128 && window <= 512) ||
        (tokens == 6 && window > 128 && window <= 160)) {
        return max_splits;
    }
    int target_keys_per_split = 480;
    if (window <= 4096) {
        target_keys_per_split = 64;
    } else if (window <= 8198) {
        target_keys_per_split = 128;
    } else if (window <= 16390) {
        target_keys_per_split = 256;
    }
    constexpr int kMinSplits = 4;
    int splits               = div_up(window, target_keys_per_split);
    splits                   = max(kMinSplits, splits);
    return min(max_splits, splits);
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
__device__ __forceinline__ void gqa_small_t_tc_row_to_qt(int row, int tokens, int kv_head,
                                                         int& q_head, int& token) {
    token             = row / kGqaGroupSize;
    const int local_q = row - token * kGqaGroupSize;
    q_head            = kv_head * kGqaGroupSize + local_q;
}

template <int DChunk>
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
    if (q_head >= kGqaQHeads || token >= tokens) { return; }
    const int last_pos           = positions[tokens - 1];
    const int window             = last_pos + 1;
    const int active_split_count = gqa_small_t_active_splits(window, split_count, tokens);

    __shared__ float reduce[256];

    float local_m = -CUDART_INF_F;
    for (int split = tid; split < active_split_count; split += blockDim.x) {
        local_m = fmaxf(local_m, partial_m[gqa_partial_stat_index(q_head, token, split, tokens)]);
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
            out[gqa_q_index(q_head, d, token)] = __float2bfloat16(0.0f);
        }
        return;
    }

    float local_l = 0.0f;
    for (int split = tid; split < active_split_count; split += blockDim.x) {
        const float tile_l = partial_l[gqa_partial_stat_index(q_head, token, split, tokens)];
        if (tile_l > 0.0f) {
            local_l +=
                tile_l *
                expf(partial_m[gqa_partial_stat_index(q_head, token, split, tokens)] - head_m);
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
            const float tile_l = partial_l[gqa_partial_stat_index(q_head, token, split, tokens)];
            if (tile_l <= 0.0f) { continue; }
            const float weight =
                expf(partial_m[gqa_partial_stat_index(q_head, token, split, tokens)] - head_m);
            numerator += __bfloat162float(
                             partial_acc[gqa_partial_acc_index(q_head, d, token, split, tokens)]) *
                         weight;
        }
    }
    const float value                  = (head_l > 0.0f) ? numerator / head_l : 0.0f;
    out[gqa_q_index(q_head, d, token)] = __float2bfloat16(value);
}

} // namespace ninfer::kernels
