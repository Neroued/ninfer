#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kMaxFp16MagWord = 0x7bff; // 65504.0

__device__ __forceinline__ float fp16_mag_to_float(int word) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(word)));
}

__device__ __forceinline__ int fp16_ceil_word(double target) {
    if (target <= double(fp16_mag_to_float(1))) return 1;
    if (target >= 65504.0) return kMaxFp16MagWord;
    int low = 1, high = kMaxFp16MagWord;
    while (low < high) {
        const int mid = (low + high) >> 1;
        if (double(fp16_mag_to_float(mid)) < target)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

__device__ __forceinline__ int fp16_nearest_word(double target) {
    if (target <= double(fp16_mag_to_float(1))) return 1;
    if (target >= 65504.0) return kMaxFp16MagWord;
    const int hi = fp16_ceil_word(target);
    if (hi == 1) return 1;
    const double d_hi = double(fp16_mag_to_float(hi)) - target;
    const double d_lo = target - double(fp16_mag_to_float(hi - 1));
    return (d_lo <= d_hi) ? (hi - 1) : hi;
}

__device__ __forceinline__ float incumbent_upper_bound(double error) {
    // One outward FP32 ULP covers the FP64 squared-error rounding and five/six
    // binary-tree reduction steps. Overflow safely disables pruning.
    const float rounded = __double2float_ru(error);
    return isfinite(rounded) ? __int_as_float(__float_as_int(rounded) + 1) : rounded;
}

__device__ __forceinline__ double warp_sum_f64(double value) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ float warp_sum_rd_f32(float value) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
        value = __fadd_rd(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
}

__device__ __forceinline__ float warp_max_f32(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
}

template <int QMin, int QMax>
__device__ __forceinline__ int nearest_code_fp64(double y, double a) {
    const double u = y / a;
    if (u <= double(QMin)) return QMin;
    if (u >= double(QMax)) return QMax;
    int q = static_cast<int>(llrint(u));
    if (q < QMin) q = QMin;
    if (q > QMax) q = QMax;
    if (q < QMax) {
        const double hi = (double(q) + 0.5) * a; // exact: <= 20 significand bits
        if (y > hi || (y == hi && (q & 1) != 0)) return q + 1;
    }
    if (q > QMin) {
        const double lo = (double(q) - 0.5) * a; // exact: <= 20 significand bits
        if (y < lo || (y == lo && (q & 1) != 0)) return q - 1;
    }
    return q;
}

template <int QMin, int QMax>
__device__ __forceinline__ float elem_sq_lower_bound_f32(float y, float a, float inv_a) {
    float q = fminf(fmaxf(rintf(y * inv_a), float(QMin)), float(QMax));
    // Both (q +- 0.5f) * a and q * a have at most 20 significand bits -> exact in FP32.
    if (q < float(QMax) && y > (q + 0.5f) * a)
        q += 1.0f;
    else if (q > float(QMin) && y < (q - 0.5f) * a)
        q -= 1.0f;
    const float prod = q * a;
    const float dist = fmaxf((y >= prod) ? __fsub_rd(y, prod) : __fsub_rd(prod, y), 0.0f);
    return __fmul_rd(dist, dist);
}

template <int GroupSize, int QMin, int QMax>
__device__ __forceinline__ double group_error_fp64(const double (&x)[GroupSize / 32], int sign,
                                                   double a) {
    double sum = 0.0;
    if constexpr (GroupSize == 64) {
        const double y0 = sign > 0 ? x[0] : -x[0];
        const double y1 = sign > 0 ? x[1] : -x[1];
        const int q0    = nearest_code_fp64<QMin, QMax>(y0, a);
        const int q1    = nearest_code_fp64<QMin, QMax>(y1, a);
        const double d0 = y0 - double(q0) * a;
        const double d1 = y1 - double(q1) * a;
        sum             = (d0 * d0) + (d1 * d1);
    } else {
        const double y0 = sign > 0 ? x[0] : -x[0];
        const int q0    = nearest_code_fp64<QMin, QMax>(y0, a);
        const double d0 = y0 - double(q0) * a;
        sum             = d0 * d0;
    }
    return warp_sum_f64(sum);
}

template <int GroupSize, int QMin, int QMax>
__device__ __forceinline__ float group_lower_bound_f32(const float (&xf)[GroupSize / 32], int sign,
                                                       float a, float inv_a) {
    float sum;
    if constexpr (GroupSize == 64) {
        const float y0 = sign > 0 ? xf[0] : -xf[0];
        const float y1 = sign > 0 ? xf[1] : -xf[1];
        sum            = __fadd_rd(elem_sq_lower_bound_f32<QMin, QMax>(y0, a, inv_a),
                                   elem_sq_lower_bound_f32<QMin, QMax>(y1, a, inv_a));
    } else {
        const float y0 = sign > 0 ? xf[0] : -xf[0];
        sum            = elem_sq_lower_bound_f32<QMin, QMax>(y0, a, inv_a);
    }
    return warp_sum_rd_f32(sum);
}

__device__ __forceinline__ bool is_better(double cand_err, int cand_sign, int cand_word,
                                          double best_err, int best_sign, int best_word) {
    if (cand_err < best_err) return true;
    if (cand_err > best_err) return false;
    if (cand_sign != best_sign) return cand_sign > best_sign; // positive wins ties
    return cand_word < best_word;                             // smaller magnitude wins ties
}

template <int GroupSize, int QMin, int QMax>
__global__ void encode_groupwise_mse(const float* __restrict__ values,
                                     signed char* __restrict__ codes,
                                     unsigned short* __restrict__ scales, int64_t groups) {
    constexpr int kItems = GroupSize / 32;
    const int64_t group  = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    if (group >= groups) return;
    const int lane       = threadIdx.x & 31;
    const int64_t offset = group * GroupSize;

    float xf[kItems];
    double xd[kItems];
    float local_pos = 0.0f;
    float local_neg = 0.0f;
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
        const float v = values[offset + lane * kItems + i];
        xf[i]         = v;
        xd[i]         = double(v);
        if (v > local_pos) local_pos = v;
        if (-v > local_neg) local_neg = -v;
    }

    const float xmax_f = warp_max_f32(local_pos);
    const float xmin_f = warp_max_f32(local_neg);
    const float max_f  = fmaxf(xmax_f, xmin_f);

    if (max_f == 0.0f) {
#pragma unroll
        for (int i = 0; i < kItems; ++i) codes[offset + lane * kItems + i] = 0;
        if (lane == 0) scales[group] = 0;
        return;
    }

    const double xmax = double(xmax_f);
    const double xmin = double(xmin_f);
    const double maxd = double(max_f);

    const int w_pos0   = fp16_ceil_word(fmax(xmax / double(QMax), xmin / double(-QMin)));
    const int w_neg0   = fp16_ceil_word(fmax(xmax / double(-QMin), xmin / double(QMax)));
    const int w_max    = fp16_ceil_word(maxd);
    const int w_hi_max = fp16_ceil_word(2.0 * maxd);

    double best_err = HUGE_VAL;
    int best_sign   = 1;
    int best_word   = w_pos0;
    float upper     = HUGE_VALF;

    auto try_exact = [&](int sign, int word) {
        const double a = double(fp16_mag_to_float(word));
        const double e = group_error_fp64<GroupSize, QMin, QMax>(xd, sign, a);
        if (is_better(e, sign, word, best_err, best_sign, best_word)) {
            best_err  = e;
            best_sign = sign;
            best_word = word;
            upper     = incumbent_upper_bound(best_err);
        }
    };

    // Fast initial anchors + 3 closed-form AM steps to obtain a tight upper bound immediately.
    try_exact(1, w_pos0);
    if (w_pos0 > 1) try_exact(1, w_pos0 - 1);
    try_exact(-1, w_neg0);
    if (w_neg0 > 1) try_exact(-1, w_neg0 - 1);
    try_exact(1, w_max);

    for (int s_idx = 0; s_idx < 2; ++s_idx) {
        const int sign = s_idx == 0 ? 1 : -1;
        int w_cur      = s_idx == 0 ? w_pos0 : w_neg0;
        for (int iter = 0; iter < 3; ++iter) {
            const double a = double(fp16_mag_to_float(w_cur));
            double xq = 0.0, qq = 0.0;
#pragma unroll
            for (int i = 0; i < kItems; ++i) {
                const double y = sign > 0 ? xd[i] : -xd[i];
                const int q    = nearest_code_fp64<QMin, QMax>(y, a);
                xq += y * double(q);
                qq += double(q * q);
            }
            xq = warp_sum_f64(xq);
            qq = warp_sum_f64(qq);
            if (qq <= 0.0 || xq <= 0.0) break;
            const int w_next = fp16_nearest_word(xq / qq);
            try_exact(sign, w_next);
            if (w_next == w_cur) break;
            w_cur = w_next;
        }
    }

    // 1. Exact downward sweep for positive scales: [w_pos0 - 1 down to 1].
    for (int word = w_pos0 - 1; word >= 1; --word) {
        const float a  = fp16_mag_to_float(word);
        const float cp = fmaxf(__fsub_rd(xmax_f, float(QMax) * a), 0.0f);
        const float cn = fmaxf(__fsub_rd(xmin_f, float(-QMin) * a), 0.0f);
        if (__fadd_rd(__fmul_rd(cp, cp), __fmul_rd(cn, cn)) > upper) break;
        const float inv_a = __fdiv_rn(1.0f, a);
        if (group_lower_bound_f32<GroupSize, QMin, QMax>(xf, 1, a, inv_a) <= upper) {
            try_exact(1, word);
        }
    }

    // 2. Exact downward sweep for negative scales: [w_neg0 - 1 down to 1].
    for (int word = w_neg0 - 1; word >= 1; --word) {
        const float a  = fp16_mag_to_float(word);
        const float cp = fmaxf(__fsub_rd(xmax_f, float(-QMin) * a), 0.0f);
        const float cn = fmaxf(__fsub_rd(xmin_f, float(QMax) * a), 0.0f);
        if (__fadd_rd(__fmul_rd(cp, cp), __fmul_rd(cn, cn)) > upper) break;
        const float inv_a = __fdiv_rn(1.0f, a);
        if (group_lower_bound_f32<GroupSize, QMin, QMax>(xf, -1, a, inv_a) <= upper) {
            try_exact(-1, word);
        }
    }

    // 3. Exact upward sweep from min(w_pos0, w_neg0) up to w_hi_max, stopping when
    //    the monotone zero-rounding energy alone exceeds upper.
    const int w_start = w_pos0 < w_neg0 ? w_pos0 : w_neg0;
    for (int word = w_start; word <= w_hi_max; ++word) {
        const float a    = fp16_mag_to_float(word);
        const float half = 0.5f * a; // exact in FP32 for every finite FP16 a
        float z_local    = 0.0f;
#pragma unroll
        for (int i = 0; i < kItems; ++i) {
            if (fabsf(xf[i]) <= half) z_local = __fadd_rd(z_local, __fmul_rd(xf[i], xf[i]));
        }
        if (warp_sum_rd_f32(z_local) > upper) break;

        const float inv_a = __fdiv_rn(1.0f, a);
        if (word >= w_pos0) {
            if (group_lower_bound_f32<GroupSize, QMin, QMax>(xf, 1, a, inv_a) <= upper) {
                try_exact(1, word);
            }
        }
        if (word >= w_neg0) {
            if (group_lower_bound_f32<GroupSize, QMin, QMax>(xf, -1, a, inv_a) <= upper) {
                try_exact(-1, word);
            }
        }
    }

    const double best_a = double(fp16_mag_to_float(best_word));
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
        const double y = best_sign > 0 ? xd[i] : -xd[i];
        codes[offset + lane * kItems + i] =
            static_cast<signed char>(nearest_code_fp64<QMin, QMax>(y, best_a));
    }
    if (lane == 0) {
        scales[group] =
            static_cast<unsigned short>((best_sign < 0 ? 0x8000u : 0u) | unsigned(best_word));
    }
}

} // namespace

void launch_groupwise_mse(const float* values, signed char* codes, unsigned short* scales,
                          int64_t groups, int bits, cudaStream_t stream) {
    // 256 threads = 8 warps = 8 groups per CTA.
    const int64_t blocks = (groups + 7) / 8;
    if (bits == 4) {
        encode_groupwise_mse<64, -8, 7><<<blocks, 256, 0, stream>>>(values, codes, scales, groups);
    } else if (bits == 5) {
        encode_groupwise_mse<64, -16, 15>
            <<<blocks, 256, 0, stream>>>(values, codes, scales, groups);
    } else if (bits == 6) {
        encode_groupwise_mse<64, -32, 31>
            <<<blocks, 256, 0, stream>>>(values, codes, scales, groups);
    } else {
        encode_groupwise_mse<32, -128, 127>
            <<<blocks, 256, 0, stream>>>(values, codes, scales, groups);
    }
}
