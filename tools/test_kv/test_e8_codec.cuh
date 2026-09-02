#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

namespace ninfer::test_kv {

// Constants for E8 lattice quantization matching Qwen3.8-27B geometry
inline constexpr int kHeadDim = 256;
inline constexpr int kSubDim = 8;
inline constexpr int kNumSubspaces = kHeadDim / kSubDim; // 32
inline constexpr int kTokensPerTile = 64;

// 128-byte aligned packed 2-stage E8 root tile layout (64 bytes codes + 64 bytes scales)
struct alignas(128) E8Packed2BitTile {
    uint8_t codes[kTokensPerTile][kNumSubspaces][2]; // 64 * 32 * 2 = 4,096 bytes
    half scales[kTokensPerTile][kNumSubspaces];       // 64 * 32 * 2 = 4,096 bytes
};

// 128-byte aligned packed E8 4-bit lattice tile layout (128 bytes codes + 8 bytes scales)
struct alignas(128) E8Packed4BitTile {
    uint8_t codes[kTokensPerTile][kHeadDim / 2];      // 64 * 128 = 8,192 bytes
    half scales[kTokensPerTile][kHeadDim / 64];       // 64 * 4 * 2 = 512 bytes
};

// 8x8 Sylvester-Hadamard orthogonal rotation in CUDA registers
__device__ __forceinline__ void hadamard_rot_8d(const float in[8], float out[8]) {
    constexpr float kInvSqrt8 = 0.35355339059327373f; // 1/sqrt(8)
    
    // Fast in-place butterfly stages
    float a0 = in[0] + in[1]; float a1 = in[0] - in[1];
    float a2 = in[2] + in[3]; float a3 = in[2] - in[3];
    float a4 = in[4] + in[5]; float a5 = in[4] - in[5];
    float a6 = in[6] + in[7]; float a7 = in[6] - in[7];

    float b0 = a0 + a2; float b1 = a1 + a3;
    float b2 = a0 - a2; float b3 = a1 - a3;
    float b4 = a4 + a6; float b5 = a5 + a7;
    float b6 = a4 - a6; float b7 = a5 - a7;

    out[0] = (b0 + b4) * kInvSqrt8;
    out[1] = (b1 + b5) * kInvSqrt8;
    out[2] = (b2 + b6) * kInvSqrt8;
    out[3] = (b3 + b7) * kInvSqrt8;
    out[4] = (b0 - b4) * kInvSqrt8;
    out[5] = (b1 - b5) * kInvSqrt8;
    out[6] = (b2 - b6) * kInvSqrt8;
    out[7] = (b3 - b7) * kInvSqrt8;
}

// Algebraic Conway-Sloane E8 nearest root finder for a unit 8D vector u
__device__ __forceinline__ uint8_t e8_quantize_root_8d(const float u[8], float v_out[8]) {
    // --- 1. Best Type A Root (112 candidates: +/- e_i +/- e_j) ---
    int top1 = 0, top2 = 1;
    float abs1 = fabsf(u[0]), abs2 = fabsf(u[1]);
    if (abs1 < abs2) {
        top1 = 1; top2 = 0;
        float tmp = abs1; abs1 = abs2; abs2 = tmp;
    }

    #pragma unroll
    for (int i = 2; i < 8; ++i) {
        float val = fabsf(u[i]);
        if (val > abs1) {
            top2 = top1;
            abs2 = abs1;
            top1 = i;
            abs1 = val;
        } else if (val > abs2) {
            top2 = i;
            abs2 = val;
        }
    }

    const float score_a = abs1 + abs2;

    // --- 2. Best Type B Root (128 candidates: (+/- 0.5, ..., +/- 0.5) with even minus signs) ---
    float abs_min = fabsf(u[0]);
    int min_dim = 0;
    int minus_count = 0;
    float type_b_signs[8];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float val = fabsf(u[i]);
        if (val < abs_min) {
            abs_min = val;
            min_dim = i;
        }
        if (u[i] < 0.0f) {
            type_b_signs[i] = -1.0f;
            minus_count++;
        } else {
            type_b_signs[i] = 1.0f;
        }
    }

    if ((minus_count & 1) != 0) {
        type_b_signs[min_dim] = -type_b_signs[min_dim];
    }

    float score_b = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        score_b += type_b_signs[i] * u[i] * 0.5f;
    }

    // --- 3. Selection & Encoding ---
    if (score_a >= score_b) {
        int i_min = (top1 < top2) ? top1 : top2;
        int i_max = (top1 < top2) ? top2 : top1;
        int pair_idx = (i_min * (15 - i_min)) / 2 + (i_max - i_min - 1);
        int sign_idx = ((u[i_min] < 0.0f ? 1 : 0) << 1) | (u[i_max] < 0.0f ? 1 : 0);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            v_out[i] = 0.0f;
        }
        v_out[i_min] = (u[i_min] < 0.0f) ? -1.0f : 1.0f;
        v_out[i_max] = (u[i_max] < 0.0f) ? -1.0f : 1.0f;

        return static_cast<uint8_t>(pair_idx * 4 + sign_idx);
    } else {
        uint8_t sign_bits = 0;
        #pragma unroll
        for (int i = 0; i < 7; ++i) {
            if (type_b_signs[i] < 0.0f) {
                sign_bits |= (1 << i);
            }
            v_out[i] = type_b_signs[i] * 0.5f;
        }
        v_out[7] = type_b_signs[7] * 0.5f;

        return static_cast<uint8_t>(112 + sign_bits);
    }
}

// Exact dot-product between 8D Query and reconstructed E8 Root from a 1-byte codeword
__device__ __forceinline__ float e8_decode_dot_8d(const float q[8], uint8_t code) {
    constexpr float kInvSqrt2 = 0.7071067811865475f;

    if (code < 112) {
        int pair_idx = code / 4;
        int sign_idx = code % 4;

        int i_min = 0;
        int rem = pair_idx;
        #pragma unroll
        for (int i = 0; i < 7; ++i) {
            int count = 7 - i;
            if (rem < count) {
                i_min = i;
                break;
            }
            rem -= count;
        }
        int i_max = i_min + 1 + rem;

        float s1 = (sign_idx & 2) ? -1.0f : 1.0f;
        float s2 = (sign_idx & 1) ? -1.0f : 1.0f;

        return (s1 * q[i_min] + s2 * q[i_max]) * kInvSqrt2;
    } else {
        int sign_bits = code - 112;
        float sum = 0.0f;
        int minus_count = 0;

        #pragma unroll
        for (int i = 0; i < 7; ++i) {
            if ((sign_bits >> i) & 1) {
                sum -= q[i] * 0.5f;
                minus_count++;
            } else {
                sum += q[i] * 0.5f;
            }
        }
        if ((minus_count & 1) != 0) {
            sum -= q[7] * 0.5f;
        } else {
            sum += q[7] * 0.5f;
        }

        return sum * kInvSqrt2;
    }
}

// General Fast Conway-Sloane E8 Lattice Point Projection for an arbitrary 8D vector
__device__ __forceinline__ void e8_project_8d_general(const float x[8], float out[8]) {
    // 1. Nearest point in D8 (even sum of integers)
    float f_x[8];
    int sum_f = 0;
    float max_err = -1.0f;
    int worst_dim = 0;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        f_x[i] = rintf(x[i]);
        sum_f += static_cast<int>(f_x[i]);
        float err = fabsf(x[i] - f_x[i]);
        if (err > max_err) {
            max_err = err;
            worst_dim = i;
        }
    }

    float d8[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        d8[i] = f_x[i];
    }
    if ((sum_f & 1) != 0) {
        d8[worst_dim] += (x[worst_dim] >= f_x[worst_dim]) ? 1.0f : -1.0f;
    }

    // 2. Nearest point in D8 + 0.5 (Coset 1)
    float f_shift[8];
    int sum_shift = 0;
    float max_err_shift = -1.0f;
    int worst_shift_dim = 0;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float xs = x[i] - 0.5f;
        f_shift[i] = rintf(xs);
        sum_shift += static_cast<int>(f_shift[i]);
        float err = fabsf(xs - f_shift[i]);
        if (err > max_err_shift) {
            max_err_shift = err;
            worst_shift_dim = i;
        }
    }

    float coset1[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        coset1[i] = f_shift[i] + 0.5f;
    }
    if ((sum_shift & 1) != 0) {
        coset1[worst_shift_dim] += ((x[worst_shift_dim] - 0.5f) >= f_shift[worst_shift_dim]) ? 1.0f : -1.0f;
    }

    // 3. Select closer point
    float dist_d8 = 0.0f;
    float dist_coset1 = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float diff0 = x[i] - d8[i];
        float diff1 = x[i] - coset1[i];
        dist_d8 += diff0 * diff0;
        dist_coset1 += diff1 * diff1;
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = (dist_d8 <= dist_coset1) ? d8[i] : coset1[i];
    }
}

} // namespace ninfer::test_kv
