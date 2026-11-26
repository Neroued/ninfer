#pragma once

#include "ops/loop_message/implementation.h"
#include "ops/common/cuda/reduce.cuh"

#include <algorithm>
#include <cfloat>

namespace ninfer::ops::detail::loop_message {

using Bf16 = __nv_bfloat16;

struct Token {
    explicit __device__ Token(const Arguments& a, std::int64_t stream, int token)
        : stream(stream),
          dest(a.dest + ((a.broadcast_dest ? 0 : stream * a.tokens) + token) * a.hidden),
          message(a.message + (stream * a.tokens + token) * a.hidden),
          gamma(a.gamma + (a.broadcast_gamma ? 0 : stream * a.hidden)),
          out(a.out + (stream * a.tokens + token) * a.hidden) {}

    explicit __device__ Token(const Arguments& a, std::int64_t row)
        : stream(row / a.tokens),
          dest(a.dest + (a.broadcast_dest ? row % a.tokens : row) * a.hidden),
          message(a.message + row * a.hidden),
          gamma(a.gamma + (a.broadcast_gamma ? 0 : stream * a.hidden)),
          out(a.out + row * a.hidden) {}

    explicit __device__ Token(const Chunk& c, int row)
        : Token(c.args, c.first_stream + row / c.tokens, c.first_token + row % c.tokens) {}

    std::int64_t stream;
    const float* dest;
    const float* message;
    const Bf16* gamma;
    Bf16* out;
};

__device__ __forceinline__ float blend(float dest, float message, float gain, float forget) {
    return __fadd_rn(__fmul_rn(1.F - tanhf(forget), dest), __fmul_rn(tanhf(gain), message));
}

struct RmsStats {
    float sum_sq;
    float max_abs;

    __device__ __forceinline__ void add(float x) {
        sum_sq  = fmaf(x, x, sum_sq);
        max_abs = fmaxf(max_abs, fabsf(x));
    }

    __device__ __forceinline__ void add(float2 x) {
        sum_sq  = fmaf(x.x, x.x, sum_sq);
        sum_sq  = fmaf(x.y, x.y, sum_sq);
        max_abs = fmaxf(max_abs, fmaxf(fabsf(x.x), fabsf(x.y)));
    }

    __device__ __forceinline__ void add(float4 x) {
        sum_sq         = fmaf(x.x, x.x, sum_sq);
        sum_sq         = fmaf(x.y, x.y, sum_sq);
        sum_sq         = fmaf(x.z, x.z, sum_sq);
        sum_sq         = fmaf(x.w, x.w, sum_sq);
        const float lo = fmaxf(fabsf(x.x), fabsf(x.y));
        const float hi = fmaxf(fabsf(x.z), fabsf(x.w));
        max_abs        = fmaxf(max_abs, fmaxf(lo, hi));
    }

    __device__ __forceinline__ void merge(RmsStats x) {
        sum_sq += x.sum_sq;
        max_abs = fmaxf(max_abs, x.max_abs);
    }

    __device__ __forceinline__ float2 pack() const { return {sum_sq, max_abs}; }
};

__device__ __forceinline__ float warp_max_abs(float max_abs) {
    return __uint_as_float(__reduce_max_sync(common::kFullWarpMask, __float_as_uint(max_abs)));
}

template <int Width = 32>
__device__ __forceinline__ RmsStats warp_rms_stats(RmsStats local) {
    const float sum = common::warp_reduce_sum<Width>(local.sum_sq);
    if constexpr (Width == 32) return {sum, warp_max_abs(local.max_abs)};
    return {sum, common::warp_max<Width>(local.max_abs)};
}

__device__ __forceinline__ void publish_warp_stats(RmsStats local, RmsStats* partials) {
    local = warp_rms_stats(local);
    if (threadIdx.x % 32 == 0) partials[threadIdx.x / 32] = local;
}

template <int Threads>
__device__ __forceinline__ RmsStats gather_rms_stats(const RmsStats* partials) {
    constexpr int Warps  = Threads / 32;
    const RmsStats local = partials[threadIdx.x % Warps];
    return {common::warp_sum<Warps>(local.sum_sq), common::warp_max<Warps>(local.max_abs)};
}

struct NormScale {
    float inverse;
    float retry_max;
};

__device__ __forceinline__ NormScale norm_scale(RmsStats stats, int width, float eps) {
    if (stats.max_abs == 0.F && stats.sum_sq == 0.F) return {rsqrtf(eps), 0.F};

    const float mean = __fdiv_rn(stats.sum_sq, float(width));
    if (__builtin_expect(stats.max_abs > 0.F && isfinite(stats.max_abs) &&
                             (!isfinite(mean) || mean < FLT_MIN),
                         false))
        return {0.F, stats.max_abs};

    const float denom = __fadd_rn(mean, eps);
    // Finite mean and epsilon may still overflow on addition.
    if (__builtin_expect(isinf(denom) && isfinite(mean), false))
        return {rsqrtf(__fadd_rn(0.5F * mean, 0.5F * eps)) * 0x1.6a09e6p-1F, 0.F};
    return {rsqrtf(denom), 0.F};
}

// Normalize x/max_abs with a normal reciprocal even when max_abs is subnormal
// or near FLT_MAX. The power-of-two adjustment keeps the division well scaled.
struct RescaleSquares {
    float shift, reciprocal;

    explicit __device__ __forceinline__ RescaleSquares(float max_abs) {
        shift      = max_abs < FLT_MIN ? 0x1p24F : max_abs > 0x1p126F ? 0x1p-4F : 1.F;
        reciprocal = __fdiv_rn(1.F, max_abs * shift);
    }

    __device__ __forceinline__ void add(float x, float& sum) const {
        const float scaled = (x * shift) * reciprocal;
        sum                = fmaf(scaled, scaled, sum);
    }

    __device__ __forceinline__ void add(float4 x, float& sum) const {
        add(x.x, sum);
        add(x.y, sum);
        add(x.z, sum);
        add(x.w, sum);
    }
};

__device__ __forceinline__ float scaled_inverse_rms(float sum_sq, int width, float eps,
                                                    float max_abs) {
    // Factor out the larger of max_abs and sqrt(eps) before squaring either.
    const float root_eps = sqrtf(eps);
    const float scale    = fmaxf(max_abs, root_eps);
    const float x = __fdiv_rn(max_abs, scale), e = __fdiv_rn(root_eps, scale);
    const float mean = __fdiv_rn(sum_sq, float(width));
    return __fdiv_rn(rsqrtf(fmaf(mean, x * x, e * e)), scale);
}

template <int Threads>
__device__ __forceinline__ float reduce_rescaled_row(const float* input, int width, float max_abs,
                                                     float* scratch = nullptr) {
    const int tid     = threadIdx.x % Threads;
    const auto* packs = reinterpret_cast<const float4*>(input);
    const RescaleSquares rescale(max_abs);
    float sum = 0;
    for (int i = tid; i < width / 4; i += Threads) rescale.add(packs[i], sum);
    if constexpr (Threads == 32) {
        return common::warp_sum(sum);
    } else {
        sum = common::block_reduce_sum<Threads>(sum, scratch);
        if (tid == 0) scratch[0] = sum;
        __syncthreads();
        return scratch[0];
    }
}

// Every lane/thread in the participating group uses the same row statistics.
template <int Threads>
__device__ __forceinline__ float row_inverse_rms(RmsStats stats, const float* input, int width,
                                                 float eps, float* scratch = nullptr) {
    const auto scale = norm_scale(stats, width, eps);
    if (__builtin_expect(scale.retry_max != 0.F, false)) {
        const float sum = reduce_rescaled_row<Threads>(input, width, scale.retry_max, scratch);
        return scaled_inverse_rms(sum, width, eps, scale.retry_max);
    }
    return scale.inverse;
}

template <int Threads>
__device__ __forceinline__ float2 input_scales(const Token& row, int width, float eps,
                                               RmsStats (*stats)[Threads / 32],
                                               float (*retry)[Threads / 32]) {
    RmsStats dest{}, message{};
    const auto* d = reinterpret_cast<const float4*>(row.dest);
    const auto* m = reinterpret_cast<const float4*>(row.message);
    for (int i = threadIdx.x; i < width / 4; i += Threads) {
        dest.add(d[i]);
        message.add(m[i]);
    }
    publish_warp_stats(dest, stats[0]);
    publish_warp_stats(message, stats[1]);
    __syncthreads();
    dest    = gather_rms_stats<Threads>(stats[0]);
    message = gather_rms_stats<Threads>(stats[1]);
    return {row_inverse_rms<Threads>(dest, row.dest, width, eps, retry[0]),
            row_inverse_rms<Threads>(message, row.message, width, eps, retry[1])};
}

__device__ __forceinline__ float tile_inverse_rms(const float2* tiles, int count, const float* row,
                                                  int width, float eps) {
    const int lane = threadIdx.x % 32;
    RmsStats stats{};
    for (int i = lane; i < count; i += 32) {
        const auto value = tiles[i];
        stats.merge({value.x, value.y});
    }
    stats.sum_sq  = common::warp_sum(stats.sum_sq);
    stats.max_abs = warp_max_abs(stats.max_abs);
    return row_inverse_rms<32>(stats, row, width, eps);
}

__device__ __forceinline__ unsigned packed(const Bf16* p) {
    return *reinterpret_cast<const unsigned*>(p);
}

inline unsigned grid_size(std::int64_t blocks, int multiple = 1) {
    constexpr int max_grid_x = 2147483647;
    return unsigned(std::min<std::int64_t>(blocks, max_grid_x / multiple)) * multiple;
}

} // namespace ninfer::ops::detail::loop_message
