#pragma once

#include "ops/loop_message/implementation.h"
#include "ops/common/cuda/reduce.cuh"

#include <algorithm>

namespace ninfer::ops::detail::loop_message {

using Bf16 = __nv_bfloat16;

struct Token {
    explicit __device__ Token(const Arguments& a, std::int64_t row)
        : stream(row / a.tokens),
          dest(a.dest + (a.broadcast_dest ? row % a.tokens : row) * a.hidden),
          message(a.message + row * a.hidden),
          gamma(a.gamma + (a.broadcast_gamma ? 0 : stream * a.hidden)),
          out(a.out + row * a.hidden) {}

    explicit __device__ Token(const Chunk& c, int row)
        : Token(c.args, std::int64_t(c.first_stream + row / c.tokens) * c.args.tokens +
                            c.first_token + row % c.tokens) {}

    std::int64_t stream;
    const float* dest;
    const float* message;
    const Bf16* gamma;
    Bf16* out;
};

__device__ __forceinline__ float blend(float dest, float message, float gain, float forget) {
    return __fadd_rn(__fmul_rn(1.F - tanhf(forget), dest), __fmul_rn(tanhf(gain), message));
}

__device__ __forceinline__ float square_sum(float4 x, float sum) {
    sum = fmaf(x.x, x.x, sum);
    sum = fmaf(x.y, x.y, sum);
    sum = fmaf(x.z, x.z, sum);
    return fmaf(x.w, x.w, sum);
}

__device__ __forceinline__ float wide_inverse_rms(double sum, int width, float epsilon) {
    return float(rsqrt(sum / double(width) + double(epsilon)));
}

__device__ __forceinline__ float rms_denominator(float sum, int width, float epsilon) {
    return __fadd_rn(__fdiv_rn(sum, float(width)), epsilon);
}

__device__ __forceinline__ float inverse_rms(float sum, int width, float epsilon) {
    const float denom = rms_denominator(sum, width, epsilon);
    if (__builtin_expect(!isfinite(denom), false)) return wide_inverse_rms(sum, width, epsilon);
    return rsqrtf(denom);
}

__device__ __forceinline__ bool lost_squares(float sum) { return sum == 0 || isinf(sum); }

// Used when the blended row is not stored contiguously. Both powers of two
// keep the retry's squares and reduction in range for finite FP32 inputs.
__device__ __forceinline__ float square_rescale(float sum) { return sum == 0 ? 0x1p80F : 0x1p-80F; }

__device__ __forceinline__ float scaled_inverse_rms(float sum, int width, float epsilon,
                                                    float factor) {
    const double unscale = factor > 1.F ? 0x1p-160 : 0x1p160;
    return wide_inverse_rms(double(sum) * unscale, width, epsilon);
}

// Recover squares lost to FP32 overflow or underflow before adding epsilon.
// A padded row has a zero sum and no input to rescan.
__device__ __forceinline__ float row_inverse_rms(float sum, const float* input, int width,
                                                 float epsilon) {
    if (__builtin_expect(lost_squares(sum) && input, false)) {
        double wide = 0;
#pragma unroll 1
        for (int i = 0; i < width; ++i) {
            const double x = input[i];
            wide           = fma(x, x, wide);
        }
        return wide_inverse_rms(wide, width, epsilon);
    }
    return inverse_rms(sum, width, epsilon);
}

__device__ __forceinline__ unsigned packed(const Bf16* p) {
    return *reinterpret_cast<const unsigned*>(p);
}

inline unsigned grid_size(std::int64_t blocks, int multiple = 1) {
    constexpr int max_grid_x = 2147483647;
    return unsigned(std::min<std::int64_t>(blocks, max_grid_x / multiple)) * multiple;
}

} // namespace ninfer::ops::detail::loop_message
