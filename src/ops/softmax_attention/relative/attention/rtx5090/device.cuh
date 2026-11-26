#pragma once

#include "ops/softmax_attention/relative/attention/plan.h"

#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/quant/fp8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace ninfer::ops::relative {

inline constexpr unsigned kWarp = 0xffffffffu;
inline constexpr float kLog2E   = 1.4426950408889634f;
inline constexpr float kLn2     = 0.6931471805599453f;

__device__ __forceinline__ float bf16(unsigned short bits) {
    return __bfloat162float(__ushort_as_bfloat16(bits));
}

__device__ __forceinline__ float sigmoid(float x) {
    const float e   = common::exp2_approx(-fabsf(x) * kLog2E);
    const float inv = common::rcp_approx_ftz(1.0f + e);
    return x < 0.0f ? e * inv : inv;
}

__device__ __forceinline__ std::uint16_t gated(Query q, std::int64_t row, int dim, float value,
                                               float sink_factor) {
    return __bfloat16_as_ushort(
        __float2bfloat16_rn(value * sigmoid(bf16(q.gate[row * kDim + dim])) * sink_factor));
}

__device__ __forceinline__ float sink_factor(Query q, std::int64_t row, float lse2) {
    return sigmoid(lse2 * kLn2 - bf16(q.sink[row]));
}

__device__ __forceinline__ float quad_max(float x) {
    x = fmaxf(x, __shfl_xor_sync(kWarp, x, 1));
    return fmaxf(x, __shfl_xor_sync(kWarp, x, 2));
}

__device__ __forceinline__ float quad_sum(float x, unsigned mask = kWarp) {
    x += __shfl_xor_sync(mask, x, 1);
    return x + __shfl_xor_sync(mask, x, 2);
}

struct KeyRange {
    int first, last, low, high;
};

template <bool Swa, int Keys = kKeyTile, int QueryRows = 16>
__device__ __forceinline__ KeyRange key_range(const int* positions, int count, int split,
                                              int splits, int limit) {
    const int lane = threadIdx.x & 31;
    int lo, hi;
    if constexpr (Swa) {
        if (!count) return {0, 0, 0, -1};
        lo = positions[0];
        hi = positions[count - 1];
        if (lo < 0 || hi < lo) return {0, 0, 0, -1};
    } else if constexpr (QueryRows == 1) {
        if (!count) return {0, 0, 0, -1};
        lo = hi = positions[0];
        if (hi < 0 || hi > limit) return {0, 0, 0, -1};
    } else {
        int local_min = INT_MAX, local_max = -1;
#pragma unroll
        for (int t = lane; t < QueryRows; t += 32) {
            const int p = t < count ? positions[t] : -1;
            if (p >= 0 && p <= limit) {
                local_min = min(local_min, p);
                local_max = max(local_max, p);
            }
        }
        lo = __reduce_min_sync(kWarp, local_min);
        hi = __reduce_max_sync(kWarp, local_max);
        if (hi < 0) return {0, 0, 0, -1};
    }
    const int first     = Swa ? max(0, lo - (kWindow - 1)) / Keys : 0;
    const int tiles     = hi / Keys + 1 - first;
    const auto interval = split_tile_interval(tiles, splits, split);
    return {first + interval.first, first + interval.end, lo, hi};
}

template <bool Swa>
__device__ __forceinline__ int position_limit(CacheView cache) {
    if constexpr (Swa) return INT_MAX;
    return static_cast<int>(min(std::int64_t(INT_MAX), std::int64_t(cache.stride) * 64 - 1));
}

template <bool Swa, bool Fp8>
struct CacheRows {
    static constexpr int Bytes = Fp8 ? kDim : 2 * kDim;
    const unsigned char *k{}, *v{}, *ks{}, *vs{};
    const std::int64_t* pages;
    ChunkView chunk{};
    std::int64_t chunk_first;
    int chunk_start;

    __device__ __forceinline__ CacheRows(CacheView cache, int row, int head, std::int64_t first = 0,
                                         int start = 0, bool history = true)
        : chunk_first(first * kKvHeads + head), chunk_start(start) {
        const int stride          = Swa ? cache.stride : 64;
        const std::int64_t offset = std::int64_t(head) * stride;

        if constexpr (Swa) {
            chunk = cache.chunk;
            if (!history) return;
            k = cache.k.base + cache.k.offsets[row] + offset * Bytes;
            v = cache.v.base + cache.v.offsets[row] + offset * Bytes;
            if constexpr (Fp8) {
                ks = cache.ks.base + cache.ks.offsets[row] + offset;
                vs = cache.vs.base + cache.vs.offsets[row] + offset;
            }
        } else {
            pages = cache.k.offsets + std::int64_t(row) * cache.stride;
            k     = cache.k.base + offset * Bytes;
            v     = cache.v.base + offset * Bytes;
            if constexpr (Fp8) {
                ks = cache.ks.base + offset;
                vs = cache.vs.base + offset;
            }
        }
    }

    __device__ __forceinline__ bool is_chunk(int key) const {
        if constexpr (Swa) return chunk.k && key >= chunk_start;
        return false;
    }

    __device__ __forceinline__ const unsigned char* codes(bool value, int key,
                                                          std::int64_t record_offset) const {
        if (is_chunk(key)) {
            const auto index = chunk_first + std::int64_t(key - chunk_start) * kKvHeads;
            return (value ? chunk.v : chunk.k) + index * Bytes;
        }
        return (value ? v : k) + record_offset + slot(key) * Bytes;
    }

    __device__ __forceinline__ const unsigned char* scale(bool value, int key,
                                                          std::int64_t record_offset) const {
        if (is_chunk(key)) {
            const auto index = chunk_first + std::int64_t(key - chunk_start) * kKvHeads;
            return (value ? chunk.vs : chunk.ks) + index;
        }
        return (value ? vs : ks) + record_offset + slot(key);
    }

    __device__ __forceinline__ std::int64_t record(int key) const {
        if constexpr (Swa) return 0;
        return pages[key / 64];
    }

    __device__ __forceinline__ static int slot(int key) { return key & (Swa ? kWindow - 1 : 63); }
};

template <bool Swa>
__device__ __forceinline__ int cache_row(CacheView cache, int sequence) {
    if constexpr (Swa) return sequence;
    return cache.table_rows[sequence];
}

__device__ __forceinline__ float relative_dot(const unsigned short* r, const unsigned short* proj,
                                              int distance) {
    float bias = 0;
    if (unsigned(distance) < kExtent) {
        proj += distance * kRelDim;
#pragma unroll 1
        for (int u = 0; u < kRelDim; ++u) bias = fmaf(bf16(r[u]), bf16(proj[u]), bias);
    }
    return bias;
}

} // namespace ninfer::ops::relative
