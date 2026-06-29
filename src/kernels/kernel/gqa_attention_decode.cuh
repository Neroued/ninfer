#pragma once

// qus::kernels - split-KV GQA decode attention. The partial kernel processes
// one KV head, one query-head subgroup, and one token tile. A fused reducer
// combines tile-local online-softmax partials into the final BF16 output.

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace qus::kernels {

inline constexpr int kGqaHeadDim   = 256;
inline constexpr int kGqaQHeads    = 24;
inline constexpr int kGqaKVHeads   = 4;
inline constexpr int kGqaGroupSize = 6;

__device__ __forceinline__ std::int64_t gqa_cache_index(int kv_head, int d, int position,
                                                        int padded_context) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaHeadDim) *
               (static_cast<std::int64_t>(position) +
                static_cast<std::int64_t>(padded_context) * kv_head);
}

__device__ __forceinline__ std::int64_t gqa_q_index(int q_head, int d) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaHeadDim) * q_head;
}

__device__ __forceinline__ std::int64_t gqa_kv_new_index(int kv_head, int d) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaHeadDim) * kv_head;
}

__device__ __forceinline__ std::int64_t gqa_partial_acc_index(int q_head, int d, int tile) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(kGqaQHeads) * tile);
}

__device__ __forceinline__ std::int64_t gqa_partial_stat_index(int q_head, int tile) {
    return static_cast<std::int64_t>(q_head) + static_cast<std::int64_t>(kGqaQHeads) * tile;
}

__device__ __forceinline__ float gqa_warp_sum(float value) {
    constexpr unsigned mask = 0xffffffffu;
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);
    return value;
}

__device__ __forceinline__ float gqa_warp_sum_broadcast(float value) {
    constexpr unsigned mask = 0xffffffffu;
    const float sum         = gqa_warp_sum(value);
    return __shfl_sync(mask, sum, 0);
}

template <int QHeadsPerCta>
__device__ __forceinline__ void gqa_block_sum_256(float (&values)[QHeadsPerCta],
                                                  float (&reduced)[QHeadsPerCta],
                                                  float* scratch) {
    static_assert(QHeadsPerCta > 0 && QHeadsPerCta <= kGqaGroupSize);
    constexpr int kWarps = kGqaHeadDim / 32;
    const int tid        = threadIdx.x;
    const int lane       = tid & 31;
    const int warp       = tid >> 5;

#pragma unroll
    for (int q = 0; q < QHeadsPerCta; ++q) {
        const float sum = gqa_warp_sum(values[q]);
        if (lane == 0) { scratch[q * kWarps + warp] = sum; }
    }
    __syncthreads();

#pragma unroll
    for (int q = 0; q < QHeadsPerCta; ++q) {
        float sum = (tid < kWarps) ? scratch[q * kWarps + lane] : 0.0f;
        if (warp == 0) { sum = gqa_warp_sum(sum); }
        if (lane == 0 && warp == 0) { scratch[q * kWarps] = sum; }
    }
    __syncthreads();

#pragma unroll
    for (int q = 0; q < QHeadsPerCta; ++q) { reduced[q] = scratch[q * kWarps]; }
    __syncthreads();
}

template <int QHeadsPerCta>
__device__ __forceinline__ int gqa_q_subgroups() {
    return (kGqaGroupSize + QHeadsPerCta - 1) / QHeadsPerCta;
}

template <int QHeadsPerCta>
__device__ __forceinline__ int gqa_global_q_head(int kv_head, int q_subgroup, int local_q) {
    return kv_head * kGqaGroupSize + q_subgroup * QHeadsPerCta + local_q;
}

__device__ __forceinline__ bool gqa_valid_q_head(int kv_head, int q_head) {
    return kv_head >= 0 && kv_head < kGqaKVHeads && q_head >= kv_head * kGqaGroupSize &&
           q_head < (kv_head + 1) * kGqaGroupSize && q_head < kGqaQHeads;
}

template <int QHeadsPerCta>
__device__ __forceinline__ void gqa_write_neutral_partial(__nv_bfloat16* partial_acc,
                                                          float* partial_m, float* partial_l,
                                                          int kv_head, int q_subgroup, int tile) {
    const int d = threadIdx.x;
#pragma unroll
    for (int local_q = 0; local_q < QHeadsPerCta; ++local_q) {
        const int q_head = gqa_global_q_head<QHeadsPerCta>(kv_head, q_subgroup, local_q);
        if (!gqa_valid_q_head(kv_head, q_head)) { continue; }
        if (d == 0) {
            partial_m[gqa_partial_stat_index(q_head, tile)] = -CUDART_INF_F;
            partial_l[gqa_partial_stat_index(q_head, tile)] = 0.0f;
        }
        partial_acc[gqa_partial_acc_index(q_head, d, tile)] = __float2bfloat16(0.0f);
    }
}

template <int QHeadsPerCta>
__device__ __forceinline__ void gqa_write_neutral_partial_warp(__nv_bfloat16* partial_acc,
                                                               float* partial_m,
                                                               float* partial_l, int kv_head,
                                                               int q_subgroup, int tile) {
    constexpr int kDimsPerLane = kGqaHeadDim / 32;
    static_assert(kDimsPerLane == 8);

    const int local_q = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    if (local_q >= QHeadsPerCta) { return; }

    const int q_head = gqa_global_q_head<QHeadsPerCta>(kv_head, q_subgroup, local_q);
    if (!gqa_valid_q_head(kv_head, q_head)) { return; }

    if (lane == 0) {
        partial_m[gqa_partial_stat_index(q_head, tile)] = -CUDART_INF_F;
        partial_l[gqa_partial_stat_index(q_head, tile)] = 0.0f;
    }

#pragma unroll
    for (int k = 0; k < kDimsPerLane; ++k) {
        const int d = lane + 32 * k;
        partial_acc[gqa_partial_acc_index(q_head, d, tile)] = __float2bfloat16(0.0f);
    }
}

template <int TileN, int QHeadsPerCta, bool WarpPerQueryHead = false>
__launch_bounds__(256) __global__ void gqa_attention_decode_partial_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
    const std::int32_t* pos, __nv_bfloat16* cache_k, __nv_bfloat16* cache_v,
    std::int32_t padded_context, std::int32_t max_context, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    static_assert(TileN == 32 || TileN == 64 || TileN == 128);
    static_assert(QHeadsPerCta == 6 || QHeadsPerCta == 3 || QHeadsPerCta == 2);

    constexpr int q_subgroups = (kGqaGroupSize + QHeadsPerCta - 1) / QHeadsPerCta;
    const int kv_head         = static_cast<int>(blockIdx.x) / q_subgroups;
    const int q_subgroup      = static_cast<int>(blockIdx.x) - kv_head * q_subgroups;
    const int tile            = static_cast<int>(blockIdx.y);
    const int tile_start      = tile * TileN;
    const std::int32_t p      = pos[0];

    if (kv_head < 0 || kv_head >= kGqaKVHeads || q_subgroup < 0 || q_subgroup >= q_subgroups) {
        return;
    }
    if (p < 0 || p >= max_context || tile_start > p) {
        if constexpr (WarpPerQueryHead) {
            gqa_write_neutral_partial_warp<QHeadsPerCta>(partial_acc, partial_m, partial_l,
                                                         kv_head, q_subgroup, tile);
        } else {
            gqa_write_neutral_partial<QHeadsPerCta>(partial_acc, partial_m, partial_l, kv_head,
                                                    q_subgroup, tile);
        }
        return;
    }

    if constexpr (WarpPerQueryHead) {
        constexpr int kDimsPerLane = kGqaHeadDim / 32;
        static_assert(kDimsPerLane == 8);

        const int local_q = threadIdx.x >> 5;
        const int lane    = threadIdx.x & 31;
        if (local_q >= QHeadsPerCta) { return; }

        const int q_head = gqa_global_q_head<QHeadsPerCta>(kv_head, q_subgroup, local_q);
        if (!gqa_valid_q_head(kv_head, q_head)) { return; }

        const bool tile_contains_p = p >= tile_start && p < tile_start + TileN;
        if (q_subgroup == 0 && local_q == 0 && tile_contains_p) {
#pragma unroll
            for (int k = 0; k < kDimsPerLane; ++k) {
                const int d                   = lane + 32 * k;
                const std::int64_t new_off    = gqa_kv_new_index(kv_head, d);
                const std::int64_t cache_off  = gqa_cache_index(kv_head, d, p, padded_context);
                cache_k[cache_off]            = k_new[new_off];
                cache_v[cache_off]            = v_new[new_off];
            }
        }

        float q_d[kDimsPerLane];
        float acc[kDimsPerLane];
#pragma unroll
        for (int k = 0; k < kDimsPerLane; ++k) {
            const int d = lane + 32 * k;
            q_d[k]      = __bfloat162float(q[gqa_q_index(q_head, d)]);
            acc[k]      = 0.0f;
        }

        float m = -CUDART_INF_F;
        float l = 0.0f;

        for (int token = tile_start; token < tile_start + TileN; ++token) {
            if (token > p) { break; }
            if (token >= max_context) { break; }

            const bool current_token = token == p;
            float dot                = 0.0f;
#pragma unroll
            for (int k = 0; k < kDimsPerLane; ++k) {
                const int d                = lane + 32 * k;
                const std::int64_t new_off = gqa_kv_new_index(kv_head, d);
                const float k_d = current_token
                                      ? __bfloat162float(k_new[new_off])
                                      : __bfloat162float(cache_k[gqa_cache_index(
                                            kv_head, d, token, padded_context)]);
                dot += q_d[k] * k_d;
            }

            const float score  = gqa_warp_sum_broadcast(dot) * scale;
            const float next_m = fmaxf(m, score);
            const float old_w  = expf(m - next_m);
            const float new_w  = expf(score - next_m);

#pragma unroll
            for (int k = 0; k < kDimsPerLane; ++k) {
                const int d                = lane + 32 * k;
                const std::int64_t new_off = gqa_kv_new_index(kv_head, d);
                const float v_d = current_token
                                      ? __bfloat162float(v_new[new_off])
                                      : __bfloat162float(cache_v[gqa_cache_index(
                                            kv_head, d, token, padded_context)]);
                acc[k] = acc[k] * old_w + new_w * v_d;
            }
            l = l * old_w + new_w;
            m = next_m;
        }

        if (lane == 0) {
            partial_m[gqa_partial_stat_index(q_head, tile)] = m;
            partial_l[gqa_partial_stat_index(q_head, tile)] = l;
        }
#pragma unroll
        for (int k = 0; k < kDimsPerLane; ++k) {
            const int d = lane + 32 * k;
            partial_acc[gqa_partial_acc_index(q_head, d, tile)] = __float2bfloat16(acc[k]);
        }
    } else {
        const int d               = threadIdx.x;
        const bool tile_contains_p = p >= tile_start && p < tile_start + TileN;
        const std::int64_t new_off = gqa_kv_new_index(kv_head, d);
        if (q_subgroup == 0 && tile_contains_p) {
            const std::int64_t cache_off = gqa_cache_index(kv_head, d, p, padded_context);
            cache_k[cache_off]           = k_new[new_off];
            cache_v[cache_off]           = v_new[new_off];
        }

        float q_d[QHeadsPerCta];
        float acc[QHeadsPerCta];
        float m[QHeadsPerCta];
        float l[QHeadsPerCta];
        bool valid_q[QHeadsPerCta];

#pragma unroll
        for (int local_q = 0; local_q < QHeadsPerCta; ++local_q) {
            const int q_head = gqa_global_q_head<QHeadsPerCta>(kv_head, q_subgroup, local_q);
            valid_q[local_q] = gqa_valid_q_head(kv_head, q_head);
            q_d[local_q] =
                valid_q[local_q] ? __bfloat162float(q[gqa_q_index(q_head, d)]) : 0.0f;
            acc[local_q] = 0.0f;
            m[local_q]   = -CUDART_INF_F;
            l[local_q]   = 0.0f;
        }

        __shared__ float reduce_scratch[QHeadsPerCta * 8];

        for (int token = tile_start; token < tile_start + TileN; ++token) {
            if (token > p) { break; }
            if (token >= max_context) { break; }

            const bool current_token = token == p;
            const std::int64_t kv_off =
                current_token ? new_off : gqa_cache_index(kv_head, d, token, padded_context);
            const float k_d = __bfloat162float(current_token ? k_new[new_off] : cache_k[kv_off]);

            float dot[QHeadsPerCta];
#pragma unroll
            for (int local_q = 0; local_q < QHeadsPerCta; ++local_q) {
                dot[local_q] = valid_q[local_q] ? q_d[local_q] * k_d : 0.0f;
            }

            float dot_sum[QHeadsPerCta];
            gqa_block_sum_256<QHeadsPerCta>(dot, dot_sum, reduce_scratch);

            const float v_d = __bfloat162float(current_token ? v_new[new_off] : cache_v[kv_off]);
#pragma unroll
            for (int local_q = 0; local_q < QHeadsPerCta; ++local_q) {
                if (!valid_q[local_q]) { continue; }
                const float score  = dot_sum[local_q] * scale;
                const float next_m = fmaxf(m[local_q], score);
                const float old_w  = expf(m[local_q] - next_m);
                const float new_w  = expf(score - next_m);
                acc[local_q]       = acc[local_q] * old_w + new_w * v_d;
                l[local_q]         = l[local_q] * old_w + new_w;
                m[local_q]         = next_m;
            }
        }

#pragma unroll
        for (int local_q = 0; local_q < QHeadsPerCta; ++local_q) {
            const int q_head = gqa_global_q_head<QHeadsPerCta>(kv_head, q_subgroup, local_q);
            if (!valid_q[local_q]) { continue; }
            if (d == 0) {
                partial_m[gqa_partial_stat_index(q_head, tile)] = m[local_q];
                partial_l[gqa_partial_stat_index(q_head, tile)] = l[local_q];
            }
            partial_acc[gqa_partial_acc_index(q_head, d, tile)] = __float2bfloat16(acc[local_q]);
        }
    }
}

template <int DChunk>
__launch_bounds__(256) __global__ void gqa_attention_decode_reduce_output_kernel(
    const __nv_bfloat16* partial_acc, const float* partial_m, const float* partial_l,
    std::int32_t tile_count, __nv_bfloat16* out) {
    static_assert(DChunk > 0 && DChunk <= kGqaHeadDim);

    const int q_head  = static_cast<int>(blockIdx.x);
    const int d_start = static_cast<int>(blockIdx.y) * DChunk;
    const int tid     = threadIdx.x;
    if (q_head >= kGqaQHeads) { return; }

    __shared__ float reduce[256];

    float local_m = -CUDART_INF_F;
    for (int tile = tid; tile < tile_count; tile += blockDim.x) {
        local_m = fmaxf(local_m, partial_m[gqa_partial_stat_index(q_head, tile)]);
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
            out[gqa_q_index(q_head, d)] = __float2bfloat16(0.0f);
        }
        return;
    }

    float local_l = 0.0f;
    for (int tile = tid; tile < tile_count; tile += blockDim.x) {
        const float tile_l = partial_l[gqa_partial_stat_index(q_head, tile)];
        if (tile_l > 0.0f) {
            local_l += tile_l * expf(partial_m[gqa_partial_stat_index(q_head, tile)] - head_m);
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
        for (int tile = 0; tile < tile_count; ++tile) {
            const float tile_l = partial_l[gqa_partial_stat_index(q_head, tile)];
            if (tile_l <= 0.0f) { continue; }
            const float weight = expf(partial_m[gqa_partial_stat_index(q_head, tile)] - head_m);
            numerator +=
                __bfloat162float(partial_acc[gqa_partial_acc_index(q_head, d, tile)]) * weight;
        }
    }
    const float value      = (head_l > 0.0f) ? numerator / head_l : 0.0f;
    out[gqa_q_index(q_head, d)] = __float2bfloat16(value);
}

} // namespace qus::kernels
