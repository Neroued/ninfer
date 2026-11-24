#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>

namespace ninfer::ops::common {

inline constexpr int kWarpSize          = 32;
inline constexpr unsigned kFullWarpMask = 0xffffffffu;

template <int Width = kWarpSize, class T>
__device__ __forceinline__ T warp_sum(T x, unsigned mask = kFullWarpMask) {
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(mask, x, offset, Width);
    }
    return x;
}

template <int Width = kWarpSize, class T>
__device__ __forceinline__ T warp_reduce_sum(T x, unsigned mask = kFullWarpMask) {
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(mask, x, offset, Width);
    }
    return x;
}

template <int Width = kWarpSize>
__device__ __forceinline__ float warp_max(float x, unsigned mask = kFullWarpMask) {
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(mask, x, offset, Width));
    }
    return x;
}

template <int BlockSize>
__device__ __forceinline__ float block_reduce_sum(float x, float* sums) {
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024);
    static_assert((BlockSize & (BlockSize - 1)) == 0);
    constexpr int Warps = BlockSize / kWarpSize;

    x = warp_reduce_sum(x);
    if constexpr (Warps == 1) { return x; }

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    if (lane == 0) { sums[warp] = x; }
    __syncthreads();

    x = threadIdx.x < Warps ? sums[lane] : 0.0f;
    if (warp == 0) { x = warp_reduce_sum<Warps>(x); }
    return x;
}

struct WelfordMoments {
    float mean;
    float m2;
    int count;
};

__device__ __forceinline__ WelfordMoments welford_merge(WelfordMoments a, WelfordMoments b) {
    if (b.count == 0) return a;
    if (a.count == 0) return b;
    const float delta = b.mean - a.mean;
    const int count   = a.count + b.count;
    const float ratio = static_cast<float>(b.count) / static_cast<float>(count);
    return {a.mean + delta * ratio,
            a.m2 + b.m2 +
                delta * delta *
                    (static_cast<float>(a.count) * static_cast<float>(b.count) /
                     static_cast<float>(count)),
            count};
}

__device__ __forceinline__ void welford_update(WelfordMoments& moments, float value) {
    ++moments.count;
    const float delta = value - moments.mean;
    moments.mean += delta / static_cast<float>(moments.count);
    moments.m2 += delta * (value - moments.mean);
}

__device__ __forceinline__ WelfordMoments warp_reduce_moments(WelfordMoments value) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const WelfordMoments other{
            __shfl_down_sync(kFullWarpMask, value.mean, offset),
            __shfl_down_sync(kFullWarpMask, value.m2, offset),
            __shfl_down_sync(kFullWarpMask, value.count, offset),
        };
        value = welford_merge(value, other);
    }
    return value;
}

// Finite score/id total order: larger scores win, ties prefer lower ids; signed
// zero is canonicalized. Key zero is reserved below every valid score/id pair.
struct ScoreIdOrderGreater {
    __device__ __forceinline__ bool operator()(std::uint64_t lhs, std::uint64_t rhs) const {
        return lhs > rhs;
    }
};

__device__ __forceinline__ std::uint32_t ordered_score_bits(float value) {
    if (value == 0.0f) { value = 0.0f; }
    const std::uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) != 0 ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ std::uint64_t score_id_order_key(float value, std::int32_t id) {
    if (id == INT_MAX) { return 0; }
    return (static_cast<std::uint64_t>(ordered_score_bits(value)) << 32) |
           static_cast<std::uint32_t>(0xffffffffu - static_cast<std::uint32_t>(id));
}

__device__ __forceinline__ float score_from_order_key(std::uint64_t key) {
    const std::uint32_t ordered = static_cast<std::uint32_t>(key >> 32);
    const std::uint32_t bits    = (ordered & 0x80000000u) != 0 ? (ordered ^ 0x80000000u) : ~ordered;
    return __uint_as_float(bits);
}

__device__ __forceinline__ std::int32_t id_from_order_key(std::uint64_t key) {
    if (key == 0) { return INT_MAX; }
    return static_cast<std::int32_t>(0xffffffffu - static_cast<std::uint32_t>(key));
}

__device__ inline unsigned long long block_reduce_max_key(unsigned long long key,
                                                          unsigned long long* warp_keys) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned long long other = __shfl_down_sync(kMask, key, offset);
        if (other > key) { key = other; }
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { warp_keys[warp] = key; }
    __syncthreads();
    key = (warp == 0 && lane < (blockDim.x >> 5)) ? warp_keys[lane] : 0ull;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            const unsigned long long other = __shfl_down_sync(kMask, key, offset);
            if (other > key) { key = other; }
        }
    }
    return key;
}

__device__ __forceinline__ bool score_id_better(float value, std::int32_t index, float best_value,
                                                std::int32_t best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

__device__ __forceinline__ void warp_reduce_argmax(float& value, std::int32_t& index) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value        = __shfl_down_sync(kMask, value, offset);
        const std::int32_t other_index = __shfl_down_sync(kMask, index, offset);
        if (score_id_better(other_value, other_index, value, index)) {
            value = other_value;
            index = other_index;
        }
    }
}

// Requires a multiple of 32 threads, at most 512. Only thread 0 owns the block result.
__device__ __forceinline__ void block_reduce_argmax(float& value, std::int32_t& index) {
    __shared__ float warp_values[16];
    __shared__ std::int32_t warp_indices[16];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    warp_reduce_argmax(value, index);
    if (lane == 0) {
        warp_values[warp]  = value;
        warp_indices[warp] = index;
    }
    __syncthreads();

    value = (lane < (blockDim.x >> 5)) ? warp_values[lane] : -CUDART_INF_F;
    index = (lane < (blockDim.x >> 5)) ? warp_indices[lane] : INT32_MAX;
    if (warp == 0) { warp_reduce_argmax(value, index); }
}

template <int BlockSize>
__device__ __forceinline__ float block_max_broadcast(float value) {
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024);
    static_assert((BlockSize & (BlockSize - 1)) == 0);
    constexpr int kWarps = BlockSize / kWarpSize;
    __shared__ float warp_maxima[kWarps];
    __shared__ float result;

    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    value          = warp_max(value);
    if (lane == 0) { warp_maxima[warp] = value; }
    __syncthreads();

    if (warp == 0) {
        value = lane < kWarps ? warp_maxima[lane] : -CUDART_INF_F;
        value = warp_max(value);
        if (lane == 0) { result = value; }
    }
    __syncthreads();
    return result;
}

} // namespace ninfer::ops::common
