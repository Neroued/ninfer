#pragma once

// ninfer::ops - split-KV causal small-T attention shared scaffolding. The BF16 and
// int8 partial kernels live in causal_attention_small_t_bf16.cuh and
// causal_attention_small_t_i8.cuh respectively; they are fully separate kernels (no
// shared body) so each KV format can be optimized independently. This header owns
// only what both share: layout constants, device helpers, and the split reducer.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/dense/causal_cache/geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCausalHeadDim = 256;

struct CausalAppendInput {
    static constexpr bool writes_cache = true;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
};

struct CausalCachedInput {
    static constexpr bool writes_cache = false;
};

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_cache_index(int physical_page, int kv_head, int d,
                                                           int page_offset) {
    return paged_kv_element_offset<kCausalHeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                      page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_q_index(int q_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kCausalHeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(Geometry::QHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_int8_new_index(int kv_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kCausalHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_partial_acc_index(int q_head, int d, int token,
                                                                 int split, int tokens) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kCausalHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(Geometry::QHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split));
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t causal_partial_stat_index(int q_head, int token, int split,
                                                                  int tokens) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(Geometry::QHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split);
}

template <typename Geometry>
__device__ __forceinline__ bool causal_valid_q_head(int kv_head, int q_head) {
    return kv_head >= 0 && kv_head < Geometry::KVHeads && q_head >= kv_head * Geometry::GroupSize &&
           q_head < (kv_head + 1) * Geometry::GroupSize && q_head < Geometry::QHeads;
}

template <typename Geometry>
__device__ __forceinline__ int causal_small_t_default_splits(int window) {
    int target_keys_per_split = 480 / Geometry::SmallTSplitScale;
    if (window <= 4096) {
        target_keys_per_split = 64 / Geometry::SmallTSplitScale;
    } else if (window <= 8198) {
        target_keys_per_split = 128 / Geometry::SmallTSplitScale;
    } else if (window <= 16390) {
        target_keys_per_split = 256 / Geometry::SmallTSplitScale;
    }
    constexpr int kMinSplits = 4 * Geometry::SmallTSplitScale;
    int splits               = div_up(window, target_keys_per_split);
    splits                   = splits > kMinSplits ? splits : kMinSplits;
    return splits < Geometry::SmallTMaximumSplits ? splits : Geometry::SmallTMaximumSplits;
}

template <typename Geometry, bool Int8>
__device__ __forceinline__ int causal_small_t_active_splits(int window, int launch_capacity,
                                                            int tokens) {
    if (window <= 0) { return launch_capacity; }
    int splits = 0;
    if constexpr (Int8) {
        if (tokens == 5 && window > 128 && window <= 512) {
            splits = div_up(window, 32 / Geometry::SmallTSplitScale);
        } else if (tokens == 6 && window > 128 && window <= 160) {
            constexpr int kKeysPerSplit = Geometry::SmallTSplitScale == 2 ? 17 : 24;
            splits                      = div_up(window, kKeysPerSplit);
        } else if (tokens == 6 && window > 5000 && window <= 8198) {
            splits             = div_up(window, 192 / Geometry::SmallTSplitScale);
            constexpr int kMin = 4 * Geometry::SmallTSplitScale;
            constexpr int kMax = 42 * Geometry::SmallTSplitScale;
            splits             = splits > kMin ? splits : kMin;
            splits             = splits < kMax ? splits : kMax;
        } else {
            splits = causal_small_t_default_splits<Geometry>(window);
        }
    } else {
        splits = causal_small_t_default_splits<Geometry>(window);
    }
    return splits < launch_capacity ? splits : launch_capacity;
}

template <typename Geometry>
__device__ __forceinline__ int
causal_small_t_quantized_active_splits(int window, int launch_capacity, int tokens) {
    int splits = causal_small_t_default_splits<Geometry>(window);
    if constexpr (Geometry::SmallTSplitScale == 1) {
        if (tokens == 1 && window > 8198) { splits = Geometry::SmallTMaximumSplits; }
    }
    return splits < launch_capacity ? splits : launch_capacity;
}

__device__ __forceinline__ int causal_small_t_tc_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <typename Byte>
__device__ __forceinline__ void causal_small_t_store_byte_swizzled(Byte* tile, int row, int d,
                                                                   int d_b16_stride, Byte code) {
    const int col_b16 = d >> 1;
    const int byte    = d & 1;
    const int off     = (row * d_b16_stride + causal_small_t_tc_swz(row, col_b16)) * 2 + byte;
    tile[off]         = code;
}

__device__ __forceinline__ int causal_small_t_tc_swz32(int row, int col) {
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
__device__ __forceinline__ void causal_small_t_tc_row_to_qt(int row, int tokens, int kv_head,
                                                            int& q_head, int& token) {
    token             = row / Geometry::GroupSize;
    const int local_q = row - token * Geometry::GroupSize;
    q_head            = kv_head * Geometry::GroupSize + local_q;
}

__device__ __forceinline__ float causal_partial_acc_value(float value) { return value; }

__device__ __forceinline__ float causal_partial_acc_value(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename PartialAcc, typename Geometry, int DChunk, bool Int8, bool MultiBatch,
          bool Masked, bool Offset>
__launch_bounds__(256) __global__ void causal_attention_small_t_reduce_output_kernel(
    const PartialAcc* partial_acc, const float* partial_m, const float* partial_l,
    const std::int32_t* positions, const std::int32_t* valid_columns, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t batch_size,
    std::int32_t split_count, __nv_bfloat16* out) {
    static_assert(DChunk > 0 && DChunk <= kCausalHeadDim);

    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = threadIdx.x;
    if (q_head >= Geometry::QHeads || token >= tokens) { return; }
    if constexpr (MultiBatch) {
        if (batch >= batch_size) { return; }
    }

    if constexpr (Offset) { positions += column_begin; }
    if constexpr (MultiBatch) { positions += batch * full_width; }
    const int last_pos = positions[tokens - 1];
    int output_column  = token;
    if constexpr (Offset) { output_column += column_begin; }
    if constexpr (MultiBatch) { output_column += batch * full_width; }

    if constexpr (MultiBatch) {
        const std::int64_t partial_acc_row = static_cast<std::int64_t>(batch) * kCausalHeadDim *
                                             Geometry::QHeads * tokens * split_count;
        const std::int64_t partial_stat_row =
            static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_acc += partial_acc_row;
        partial_m += partial_stat_row;
        partial_l += partial_stat_row;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        causal_small_t_active_splits<Geometry, Int8>(window, split_count, tokens);

    __shared__ float reduce[256];

    float local_m = -CUDART_INF_F;
    for (int split = tid; split < active_split_count; split += blockDim.x) {
        local_m = fmaxf(
            local_m, partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)]);
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
        if (tid < DChunk && d < kCausalHeadDim) {
            out[causal_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(0.0f);
        }
        return;
    }

    float local_l = 0.0f;
    for (int split = tid; split < active_split_count; split += blockDim.x) {
        const float tile_l =
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)];
        if (tile_l > 0.0f) {
            local_l +=
                tile_l *
                expf(partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] -
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
    if (tid >= DChunk || d >= kCausalHeadDim) { return; }

    float numerator = 0.0f;
    if (head_l > 0.0f) {
        for (int split = 0; split < active_split_count; ++split) {
            const float tile_l =
                partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)];
            if (tile_l <= 0.0f) { continue; }
            const float weight =
                expf(partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, tokens)] -
                     head_m);
            numerator += causal_partial_acc_value(partial_acc[causal_partial_acc_index<Geometry>(
                             q_head, d, token, split, tokens)]) *
                         weight;
        }
    }
    bool valid = true;
    if constexpr (Masked) {
        int absolute_column = token;
        if constexpr (Offset) { absolute_column += column_begin; }
        valid = absolute_column < valid_columns[batch];
    }
    const float value = (valid && head_l > 0.0f) ? numerator / head_l : 0.0f;
    out[causal_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(value);
}

} // namespace ninfer::ops
