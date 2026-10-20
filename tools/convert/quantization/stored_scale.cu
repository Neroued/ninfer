#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __constant__ unsigned nf4_words[16] = {
    0xbf800000, 0xbf3239b1, 0xbf066b30, 0xbeca32a0, 0xbe91a24d, 0xbe3d353f, 0xbdba7871, 0x00000000,
    0x3da2faff, 0x3e24cae3, 0x3e7c04dd, 0x3ead033a, 0x3ee1a4b8, 0x3f1007ab, 0x3f3913b3, 0x3f800000,
};
__device__ __constant__ double e2m1[8] = {0., .5, 1., 1.5, 2., 3., 4., 6.};

__device__ float scale_value(int word) {
    const int exponent = word >> 3;
    return exponent ? __int_as_float(((exponent + 120) << 23) | ((word & 7) << 20))
                    : (word & 7) * (1.f / 512.f);
}

template <bool NF4>
__device__ double code_value(int code) {
    if constexpr (NF4) return double(__uint_as_float(nf4_words[code]));
    return e2m1[code];
}

template <bool NF4>
__device__ int nearest(double x, double s) {
    int low = 0, high = NF4 ? 15 : 7;
    if constexpr (!NF4) x = fabs(x);
#pragma unroll
    for (int i = 0; i < (NF4 ? 4 : 3); ++i) {
        const int mid          = (low + high) / 2;
        const double threshold = (code_value<NF4>(mid) + code_value<NF4>(mid + 1)) * .5 * s;
        const bool above       = x > threshold || (!NF4 && x == threshold && (mid & 1));
        if (above)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

template <bool NF4>
__device__ double error(double x, double s, unsigned mask) {
    const int code = nearest<NF4>(x, s);
    if constexpr (!NF4) x = fabs(x);
    const double delta = x - code_value<NF4>(code) * s;
    double sum         = delta * delta;
#pragma unroll
    for (int offset = 8; offset; offset /= 2) sum += __shfl_xor_sync(mask, sum, offset, 16);
    return __shfl_sync(mask, sum, 0, 16);
}

template <bool NF4>
__device__ float error_lower_bound(float x_low, float x_high, float scale, unsigned mask) {
    // Bracket the closest reconstruction intervals using directed FP32 rounding.
    // Their distance from [x_low,x_high] bounds the exact distance from below.
    int low = 0, high = NF4 ? 15 : 7;
#pragma unroll
    for (int i = 0; i < (NF4 ? 4 : 3); ++i) {
        const int mid = (low + high) / 2;
        if (__fmul_ru(float(code_value<NF4>(mid)), scale) < x_low)
            low = mid + 1;
        else
            high = mid;
    }
    const float value = float(code_value<NF4>(low));
    float distance    = fmaxf(fmaxf(__fsub_rd(__fmul_rd(value, scale), x_high),
                                    __fsub_rd(x_low, __fmul_ru(value, scale))),
                              0.f);
    if (low) {
        const float left =
            fmaxf(__fsub_rd(x_low, __fmul_ru(float(code_value<NF4>(low - 1)), scale)), 0.f);
        distance = fminf(distance, left);
    }
    float sum = __fmul_rd(distance, distance);
#pragma unroll
    for (int offset = 8; offset; offset /= 2)
        sum = __fadd_rd(sum, __shfl_xor_sync(mask, sum, offset, 16));
    return __shfl_sync(mask, sum, 0, 16);
}

__device__ float incumbent_upper_bound(double error) {
    // One outward FP32 ULP also covers the rounding of the FP64 squared-error
    // calculation and its four reduction steps. Overflow disables pruning.
    const float rounded = __double2float_ru(error);
    return isfinite(rounded) ? __int_as_float(__float_as_int(rounded) + 1) : rounded;
}

template <bool NF4>
__global__ void encode(const float* values, unsigned char* codes, unsigned char* scales,
                       int64_t groups, double divisor) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t group = index / 16;
    if (group >= groups) return;
    const int lane = threadIdx.x & 15;
    // Every G16 group takes its own zero/pruning branches. Synchronize only
    // its sixteen lanes, even when the other half warp evaluates another scale.
    const unsigned mask = 0xffffu << (threadIdx.x & 16);
    const double x      = double(values[index]) * divisor;
    double maximum      = fabs(x);
#pragma unroll
    for (int offset = 8; offset; offset /= 2)
        maximum = fmax(maximum, __shfl_xor_sync(mask, maximum, offset, 16));
    int best_word = 0, code = NF4 ? 7 : 0;
    if (maximum != 0.) {
        const double limit  = NF4 ? 1. : 6.;
        const double target = NF4 ? x : fabs(x);
        const float x_low = __double2float_rd(target), x_high = __double2float_ru(target);
        const float maximum_low = __double2float_rd(maximum);
        int low = 1, high = 126;
        while (low < high) {
            const int mid = (low + high) / 2;
            if (scale_value(mid) * limit < maximum)
                low = mid + 1;
            else
                high = mid;
        }
        best_word         = low;
        double best_error = error<NF4>(x, scale_value(best_word), mask);
        float upper       = incumbent_upper_bound(best_error);
        for (int word = 1; word <= 126; ++word) {
            const float s       = scale_value(word);
            const float clipped = fmaxf(__fsub_rd(maximum_low, float(limit) * s), 0.f);
            if (__fmul_rd(clipped, clipped) > upper) continue;
            if (error_lower_bound<NF4>(x_low, x_high, s, mask) > upper) continue;
            const double candidate = error<NF4>(x, s, mask);
            if (candidate < best_error || (candidate == best_error && word < best_word)) {
                best_error = candidate;
                best_word  = word;
                upper      = incumbent_upper_bound(best_error);
            }
        }
        code = nearest<NF4>(x, scale_value(best_word));
        if constexpr (!NF4) code |= signbit(x) ? 8 : 0;
    }
    const int next = __shfl_xor_sync(mask, code, 1, 16);
    if (!(lane & 1)) codes[index / 2] = static_cast<unsigned char>(code | (next << 4));
    if (!lane) scales[group] = static_cast<unsigned char>(best_word);
}

} // namespace

void launch_stored_scale(const float* values, unsigned char* codes, unsigned char* scales,
                         int64_t elements, double divisor, bool nf4, cudaStream_t stream) {
    const auto blocks = (elements + 255) / 256;
    if (nf4)
        encode<true><<<blocks, 256, 0, stream>>>(values, codes, scales, elements / 16, divisor);
    else
        encode<false><<<blocks, 256, 0, stream>>>(values, codes, scales, elements / 16, divisor);
}
