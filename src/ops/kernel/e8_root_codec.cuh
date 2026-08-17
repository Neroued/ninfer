#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#include "ops/kernel/e8_lattice.cuh"

namespace ninfer::ops {

// Algebraic Conway-Sloane E8 Root Quantization (Finds closest root out of 240 minimal vectors)
// Maps unit vector u in R^8 to index in [0, 239]
__device__ __forceinline__ uint8_t e8_quantize_root_8d(const float u[8], float out_root[8]) {
    constexpr float kInvSqrt2 = 0.7071067811865475f;

    // 1. Type A Roots: Permutations of (+-1, +-1, 0, 0, 0, 0, 0, 0) -> 112 roots
    float abs_u[8];
    int signs[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        abs_u[i] = fabsf(u[i]);
        signs[i] = (u[i] >= 0.0f) ? 1 : -1;
    }

    int best_i = 0, best_j = 1;
    float max_pair_sum = -1.0f;
    int pair_idx = 0;
    int best_pair_idx = 0;

    #pragma unroll
    for (int i = 0; i < 7; ++i) {
        #pragma unroll
        for (int j = i + 1; j < 8; ++j) {
            float sum = abs_u[i] + abs_u[j];
            if (sum > max_pair_sum) {
                max_pair_sum = sum;
                best_i = i;
                best_j = j;
                best_pair_idx = pair_idx;
            }
            pair_idx++;
        }
    }

    float best_type_a_score = max_pair_sum;
    int s_i_bit = (signs[best_i] > 0) ? 1 : 0;
    int s_j_bit = (signs[best_j] > 0) ? 1 : 0;
    uint8_t type_a_code = static_cast<uint8_t>(best_pair_idx * 4 + (s_i_bit << 1) + s_j_bit);

    // 2. Type B Roots: 1/2 (+-1, +-1, +-1, +-1, +-1, +-1, +-1, +-1) with even parity -> 128 roots
    float sum_abs = 0.0f;
    int minus_count = 0;
    float min_abs = 1e9f;
    int min_idx = 0;
    int b_signs[8];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum_abs += abs_u[i];
        if (u[i] >= 0.0f) {
            b_signs[i] = 1;
        } else {
            b_signs[i] = -1;
            minus_count++;
        }
        if (abs_u[i] < min_abs) {
            min_abs = abs_u[i];
            min_idx = i;
        }
    }

    float best_type_b_score = 0.5f * sum_abs;
    if ((minus_count & 1) != 0) {
        best_type_b_score -= min_abs;
        b_signs[min_idx] = -b_signs[min_idx];
    }

    uint8_t type_b_code = 0;
    #pragma unroll
    for (int i = 0; i < 7; ++i) {
        if (b_signs[i] > 0) {
            type_b_code |= (1 << i);
        }
    }

    if (best_type_a_score >= best_type_b_score) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) out_root[i] = 0.0f;
        out_root[best_i] = (signs[best_i] > 0) ? 1.0f : -1.0f;
        out_root[best_j] = (signs[best_j] > 0) ? 1.0f : -1.0f;
        return type_a_code;
    } else {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_root[i] = (b_signs[i] > 0) ? 0.5f : -0.5f;
        }
        return static_cast<uint8_t>(112 + type_b_code);
    }
}

// Fast Multiplier-Free Register Dot Product for E8 Root Code
__device__ __forceinline__ float e8_decode_dot_8d(const float q[8], uint8_t code) {
    if (code < 112) {
        // Type A: 28 pairs * 4 sign combinations
        int pair_idx = code >> 2;
        int sign_bits = code & 3;
        float s_i = (sign_bits & 2) ? 1.0f : -1.0f;
        float s_j = (sign_bits & 1) ? 1.0f : -1.0f;

        // Pair lookup table
        constexpr int kPairs[28][2] = {
            {0,1},{0,2},{0,3},{0,4},{0,5},{0,6},{0,7},
            {1,2},{1,3},{1,4},{1,5},{1,6},{1,7},
            {2,3},{2,4},{2,5},{2,6},{2,7},
            {3,4},{3,5},{3,6},{3,7},
            {4,5},{4,6},{4,7},
            {5,6},{5,7},
            {6,7}
        };

        int idx_i = kPairs[pair_idx][0];
        int idx_j = kPairs[pair_idx][1];
        return s_i * q[idx_i] + s_j * q[idx_j];
    } else {
        // Type B: 128 roots (7 independent sign bits, parity for 8th)
        int b_code = code - 112;
        int parity = 0;
        float sum = 0.0f;

        #pragma unroll
        for (int i = 0; i < 7; ++i) {
            int bit = (b_code >> i) & 1;
            parity ^= (1 - bit);
            sum += bit ? q[i] : -q[i];
        }
        sum += (parity == 0) ? q[7] : -q[7];
        return 0.5f * sum;
    }
}

// Cylinder Factorization E8 Root Encoding for one 8D vector (8-bit Root + 4-bit Log-Radius + 4-bit Hyperoctahedral Axis)
__device__ __forceinline__ void e8_encode_cylinder_8d(
    const float rot[8],
    float ks,
    uint8_t& out_root,
    uint8_t& out_rad_axis
) {
    float norm_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        norm_sq += rot[i] * rot[i];
    }
    float out_norm = sqrtf(norm_sq);
    float r_rel = out_norm / (ks * 2.82842712474619f + 1e-8f); // ks * sqrt(8)

    uint32_t rad_idx = 0;
    if (r_rel >= 0.08f) {
        float log_val = 3.0f * (logf(r_rel) * 1.4426950408889634f) + 8.0f; // 3.0 * log2(r_rel) + 8.0
        int q_rad = static_cast<int>(rintf(log_val));
        rad_idx = static_cast<uint32_t>(q_rad < 1 ? 1 : (q_rad > 15 ? 15 : q_rad));
    }

    if (rad_idx == 0) {
        out_root = 0;
        out_rad_axis = 0;
        return;
    }

    float inv_norm = 1.0f / (out_norm + 1e-8f);
    float u[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        u[i] = rot[i] * inv_norm;
    }

    // 1. Primary E8 Root
    float v1[8];
    out_root = e8_quantize_root_8d(u, v1);

    // 2. Residual Axis
    constexpr float kInvSqrt2 = 0.7071067811865475f;
    float dot_u_v1 = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        dot_u_v1 += u[i] * v1[i] * kInvSqrt2;
    }

    float res[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        res[i] = u[i] - dot_u_v1 * (v1[i] * kInvSqrt2);
    }

    int best_dim = 0;
    float max_abs_res = fabsf(res[0]);
    #pragma unroll
    for (int i = 1; i < 8; ++i) {
        float a = fabsf(res[i]);
        if (a > max_abs_res) {
            max_abs_res = a;
            best_dim = i;
        }
    }
    uint32_t sign_bit = (res[best_dim] >= 0.0f) ? 0 : 1;
    uint32_t axis_idx = (static_cast<uint32_t>(best_dim) << 1) | sign_bit;

    out_rad_axis = static_cast<uint8_t>((rad_idx << 4) | (axis_idx & 0x0F));
}

// Warp-Cooperative 8-Lane Subspace Quantizer (100% Lane Occupancy, Zero Loop Divergence)
__device__ __forceinline__ void e8_encode_cylinder_8d_warp(
    float val,
    float ks,
    uint8_t& out_root,
    uint8_t& out_rad_axis,
    int lane
) {
    const int sub_lane = lane & 7;
    constexpr unsigned full_mask = 0xffffffffu;

    // 1. Norm calculation across 8 lanes
    float val_sq = val * val;
    float norm_sq = val_sq;
    #pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) {
        norm_sq += __shfl_xor_sync(full_mask, norm_sq, mask);
    }
    float out_norm = sqrtf(norm_sq);

    // 2. Log-radius index
    float r_rel = out_norm / (ks * 2.82842712474619f + 1e-8f);
    uint32_t rad_idx = 0;
    if (r_rel >= 0.08f) {
        float log_val = 3.0f * (logf(r_rel) * 1.4426950408889634f) + 8.0f;
        int q_rad = static_cast<int>(rintf(log_val));
        rad_idx = static_cast<uint32_t>(q_rad < 1 ? 1 : (q_rad > 15 ? 15 : q_rad));
    }

    // NOTE (correctness hardening of the ported codec): rad_idx is derived from a
    // per-8-lane-subgroup norm (the 8-lane butterfly sum below only straddles lane
    // bits 0-2), so it can differ ACROSS the four 8-lane subgroups of one warp when
    // the warp's 32 lanes hold different dimensions of the same token. We must NOT
    // early-return here: every lane named by full_mask (the whole 32-lane warp) has
    // to converge on each __shfl*_sync below, or the divergent shuffle is undefined
    // behaviour that can corrupt stored key codes (RK2V4E8). A zero-rad_idx subgroup
    // is instead zeroed by the guarded output write at the end of the function; its
    // per-subgroup intermediate values cannot leak into other subgroups because every
    // shuffle uses XOR masks {1,2,4} that stay inside an 8-lane subgroup. Original
    // design credit: UDPSendToFailed/ninfer-4090 (and Don-Chad/ninfer-3090 lineage).

    // 3. Unit vector
    float inv_norm = 1.0f / (out_norm + 1e-8f);
    float u = val * inv_norm;
    float abs_u = fabsf(u);
    int sign_u = (u >= 0.0f) ? 1 : -1;

    // 4. Type A: Top-2 reduction across 8 lanes (3 butterfly shuffles)
    float top1_val = abs_u;
    int   top1_idx = sub_lane;
    float top2_val = -1.0f;
    int   top2_idx = -1;

    #pragma unroll
    for (int mask = 1; mask <= 4; mask <<= 1) {
        float other1_val = __shfl_xor_sync(full_mask, top1_val, mask);
        int   other1_idx = __shfl_xor_sync(full_mask, top1_idx, mask);
        float other2_val = __shfl_xor_sync(full_mask, top2_val, mask);
        int   other2_idx = __shfl_xor_sync(full_mask, top2_idx, mask);

        if (other1_val > top1_val || (other1_val == top1_val && other1_idx < top1_idx)) {
            top2_val = (top1_val > other2_val) ? top1_val : other2_val;
            top2_idx = (top1_val > other2_val) ? top1_idx : other2_idx;
            top1_val = other1_val;
            top1_idx = other1_idx;
        } else {
            if (other1_val > top2_val || (other1_val == top2_val && other1_idx < top2_idx)) {
                top2_val = other1_val;
                top2_idx = other1_idx;
            }
        }
    }

    int best_i = (top1_idx < top2_idx) ? top1_idx : top2_idx;
    int best_j = (top1_idx < top2_idx) ? top2_idx : top1_idx;
    int best_pair_idx = (best_i * (15 - best_i)) / 2 + (best_j - best_i - 1);

    int s_i_val = __shfl_sync(full_mask, sign_u, (lane & ~7) + best_i);
    int s_j_val = __shfl_sync(full_mask, sign_u, (lane & ~7) + best_j);
    int s_i_bit = (s_i_val > 0) ? 1 : 0;
    int s_j_bit = (s_j_val > 0) ? 1 : 0;
    uint8_t type_a_code = static_cast<uint8_t>(best_pair_idx * 4 + (s_i_bit << 1) + s_j_bit);
    float best_type_a_score = top1_val + top2_val;

    // 5. Type B: Sum & Min reduction across 8 lanes
    float sum_abs = abs_u;
    int minus_count = (sign_u < 0) ? 1 : 0;
    float min_abs = abs_u;
    int min_idx = sub_lane;

    #pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) {
        sum_abs += __shfl_xor_sync(full_mask, sum_abs, mask);
        minus_count += __shfl_xor_sync(full_mask, minus_count, mask);
        float other_min = __shfl_xor_sync(full_mask, min_abs, mask);
        int other_idx   = __shfl_xor_sync(full_mask, min_idx, mask);
        if (other_min < min_abs || (other_min == min_abs && other_idx < min_idx)) {
            min_abs = other_min;
            min_idx = other_idx;
        }
    }

    float best_type_b_score = 0.5f * sum_abs;
    if ((minus_count & 1) != 0) {
        best_type_b_score -= min_abs;
    }

    int b_sign = (sub_lane == min_idx && ((minus_count & 1) != 0)) ? -sign_u : sign_u;
    uint32_t b_bit = (sub_lane < 7 && b_sign > 0) ? (1u << sub_lane) : 0u;
    #pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) {
        b_bit += __shfl_xor_sync(full_mask, b_bit, mask);
    }
    uint8_t type_b_code = static_cast<uint8_t>(b_bit);

    // 6. Select Type A vs Type B
    float v1_coord = 0.0f;
    if (best_type_a_score >= best_type_b_score) {
        out_root = type_a_code;
        if (sub_lane == best_i) v1_coord = (s_i_bit ? 1.0f : -1.0f);
        else if (sub_lane == best_j) v1_coord = (s_j_bit ? 1.0f : -1.0f);
        else v1_coord = 0.0f;
    } else {
        out_root = static_cast<uint8_t>(112 + type_b_code);
        v1_coord = (b_sign > 0) ? 0.5f : -0.5f;
    }

    // 7. Residual Axis across 8 lanes
    constexpr float kInvSqrt2 = 0.7071067811865475f;
    float dot_u_v1_lane = u * (v1_coord * kInvSqrt2);
    float dot_u_v1 = dot_u_v1_lane;
    #pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) {
        dot_u_v1 += __shfl_xor_sync(full_mask, dot_u_v1, mask);
    }

    float res = u - dot_u_v1 * (v1_coord * kInvSqrt2);
    float abs_res = fabsf(res);
    float max_abs_res = abs_res;
    int best_dim = sub_lane;

    #pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) {
        float other_res = __shfl_xor_sync(full_mask, max_abs_res, mask);
        int other_dim   = __shfl_xor_sync(full_mask, best_dim, mask);
        if (other_res > max_abs_res || (other_res == max_abs_res && other_dim < best_dim)) {
            max_abs_res = other_res;
            best_dim = other_dim;
        }
    }

    float best_res_sign_val = __shfl_sync(full_mask, res, (lane & ~7) + best_dim);
    uint32_t sign_bit = (best_res_sign_val >= 0.0f) ? 0 : 1;
    uint32_t axis_idx = (static_cast<uint32_t>(best_dim) << 1) | sign_bit;

    // Guarded output write: a zero-rad_idx (near-zero-norm) subgroup must still have
    // participated in every full-mask shuffle above so the whole warp stays converged;
    // it emits the zero code here, matching the pre-fix behaviour but only AFTER all
    // __shfl*_sync calls have completed.
    if (rad_idx == 0) {
        out_root     = 0;
        out_rad_axis = 0;
    } else {
        // out_root was already selected by the Type A/B decision above; only the
        // radius/axis byte depends on rad_idx.
        out_rad_axis = static_cast<uint8_t>((rad_idx << 4) | (axis_idx & 0x0F));
    }
}

__device__ __forceinline__ void e8_encode_root_2stage_8d(
    const float rot[8],
    uint8_t& out_code1,
    uint8_t& out_code2
) {
    e8_encode_cylinder_8d(rot, 1.0f, out_code1, out_code2);
}

// 16-entry Hyperoctahedral Axis Constant Table (128 bytes in __constant__ memory)
__constant__ const std::uint64_t c_axis_i8x8[16] = {
    0x0000000000000001ULL, // +e0 (dim 0, +1)
    0x00000000000000ffULL, // -e0 (dim 0, -1)
    0x0000000000000100ULL, // +e1 (dim 1, +1)
    0x000000000000ff00ULL, // -e1 (dim 1, -1)
    0x0000000000010000ULL, // +e2 (dim 2, +1)
    0x0000000000ff0000ULL, // -e2 (dim 2, -1)
    0x0000000001000000ULL, // +e3 (dim 3, +1)
    0x00000000ff000000ULL, // -e3 (dim 3, -1)
    0x0000000100000000ULL, // +e4 (dim 4, +1)
    0x000000ff00000000ULL, // -e4 (dim 4, -1)
    0x0000010000000000ULL, // +e5 (dim 5, +1)
    0x0000ff0000000000ULL, // -e5 (dim 5, -1)
    0x0001000000000000ULL, // +e6 (dim 6, +1)
    0x00ff000000000000ULL, // -e6 (dim 6, -1)
    0x0100000000000000ULL, // +e7 (dim 7, +1)
    0xff00000000000000ULL  // -e7 (dim 7, -1)
};

// 4-bit Log-Radius Scale Multiplier Table (16 floats, centered at 0.5000 = sqrt(8)/sqrt(32))
__constant__ const float c_radius_scale[16] = {
    0.0000f, // idx 0: Zero vector
    0.0992f, // idx 1: 0.5 * 2^(-7/3)
    0.1250f, // idx 2: 0.5 * 2^(-6/3)
    0.1575f, // idx 3: 0.5 * 2^(-5/3)
    0.1984f, // idx 4: 0.5 * 2^(-4/3)
    0.2500f, // idx 5: 0.5 * 2^(-3/3)
    0.3150f, // idx 6: 0.5 * 2^(-2/3)
    0.3969f, // idx 7: 0.5 * 2^(-1/3)
    0.5000f, // idx 8: 0.5 * 2^(0)  <-- Exact sqrt(8)/sqrt(32) center
    0.6300f, // idx 9: 0.5 * 2^(1/3)
    0.7937f, // idx 10: 0.5 * 2^(2/3)
    1.0000f, // idx 11: 0.5 * 2^(3/3)
    1.2599f, // idx 12: 0.5 * 2^(4/3)
    1.5874f, // idx 13: 0.5 * 2^(5/3)
    2.0000f, // idx 14: 0.5 * 2^(6/3)
    2.5198f  // idx 15: 0.5 * 2^(7/3)
};

// Precomputed 2-Stage E8 Root Tables (2 KiB in __device__ memory, routed through non-blocking L1 cache via __ldg)
__device__ const std::uint64_t c_e8_stage1_i8x8[256] = {
    0x000000000000fcfcULL, // code 0
    0x00000000000004fcULL, // code 1
    0x000000000000fc04ULL, // code 2
    0x0000000000000404ULL, // code 3
    0x0000000000fc00fcULL, // code 4
    0x00000000000400fcULL, // code 5
    0x0000000000fc0004ULL, // code 6
    0x0000000000040004ULL, // code 7
    0x00000000fc0000fcULL, // code 8
    0x00000000040000fcULL, // code 9
    0x00000000fc000004ULL, // code 10
    0x0000000004000004ULL, // code 11
    0x000000fc000000fcULL, // code 12
    0x00000004000000fcULL, // code 13
    0x000000fc00000004ULL, // code 14
    0x0000000400000004ULL, // code 15
    0x0000fc00000000fcULL, // code 16
    0x00000400000000fcULL, // code 17
    0x0000fc0000000004ULL, // code 18
    0x0000040000000004ULL, // code 19
    0x00fc0000000000fcULL, // code 20
    0x00040000000000fcULL, // code 21
    0x00fc000000000004ULL, // code 22
    0x0004000000000004ULL, // code 23
    0xfc000000000000fcULL, // code 24
    0x04000000000000fcULL, // code 25
    0xfc00000000000004ULL, // code 26
    0x0400000000000004ULL, // code 27
    0x0000000000fcfc00ULL, // code 28
    0x000000000004fc00ULL, // code 29
    0x0000000000fc0400ULL, // code 30
    0x0000000000040400ULL, // code 31
    0x00000000fc00fc00ULL, // code 32
    0x000000000400fc00ULL, // code 33
    0x00000000fc000400ULL, // code 34
    0x0000000004000400ULL, // code 35
    0x000000fc0000fc00ULL, // code 36
    0x000000040000fc00ULL, // code 37
    0x000000fc00000400ULL, // code 38
    0x0000000400000400ULL, // code 39
    0x0000fc000000fc00ULL, // code 40
    0x000004000000fc00ULL, // code 41
    0x0000fc0000000400ULL, // code 42
    0x0000040000000400ULL, // code 43
    0x00fc00000000fc00ULL, // code 44
    0x000400000000fc00ULL, // code 45
    0x00fc000000000400ULL, // code 46
    0x0004000000000400ULL, // code 47
    0xfc0000000000fc00ULL, // code 48
    0x040000000000fc00ULL, // code 49
    0xfc00000000000400ULL, // code 50
    0x0400000000000400ULL, // code 51
    0x00000000fcfc0000ULL, // code 52
    0x0000000004fc0000ULL, // code 53
    0x00000000fc040000ULL, // code 54
    0x0000000004040000ULL, // code 55
    0x000000fc00fc0000ULL, // code 56
    0x0000000400fc0000ULL, // code 57
    0x000000fc00040000ULL, // code 58
    0x0000000400040000ULL, // code 59
    0x0000fc0000fc0000ULL, // code 60
    0x0000040000fc0000ULL, // code 61
    0x0000fc0000040000ULL, // code 62
    0x0000040000040000ULL, // code 63
    0x00fc000000fc0000ULL, // code 64
    0x0004000000fc0000ULL, // code 65
    0x00fc000000040000ULL, // code 66
    0x0004000000040000ULL, // code 67
    0xfc00000000fc0000ULL, // code 68
    0x0400000000fc0000ULL, // code 69
    0xfc00000000040000ULL, // code 70
    0x0400000000040000ULL, // code 71
    0x000000fcfc000000ULL, // code 72
    0x00000004fc000000ULL, // code 73
    0x000000fc04000000ULL, // code 74
    0x0000000404000000ULL, // code 75
    0x0000fc00fc000000ULL, // code 76
    0x00000400fc000000ULL, // code 77
    0x0000fc0004000000ULL, // code 78
    0x0000040004000000ULL, // code 79
    0x00fc0000fc000000ULL, // code 80
    0x00040000fc000000ULL, // code 81
    0x00fc000004000000ULL, // code 82
    0x0004000004000000ULL, // code 83
    0xfc000000fc000000ULL, // code 84
    0x04000000fc000000ULL, // code 85
    0xfc00000004000000ULL, // code 86
    0x0400000004000000ULL, // code 87
    0x0000fcfc00000000ULL, // code 88
    0x000004fc00000000ULL, // code 89
    0x0000fc0400000000ULL, // code 90
    0x0000040400000000ULL, // code 91
    0x00fc00fc00000000ULL, // code 92
    0x000400fc00000000ULL, // code 93
    0x00fc000400000000ULL, // code 94
    0x0004000400000000ULL, // code 95
    0xfc0000fc00000000ULL, // code 96
    0x040000fc00000000ULL, // code 97
    0xfc00000400000000ULL, // code 98
    0x0400000400000000ULL, // code 99
    0x00fcfc0000000000ULL, // code 100
    0x0004fc0000000000ULL, // code 101
    0x00fc040000000000ULL, // code 102
    0x0004040000000000ULL, // code 103
    0xfc00fc0000000000ULL, // code 104
    0x0400fc0000000000ULL, // code 105
    0xfc00040000000000ULL, // code 106
    0x0400040000000000ULL, // code 107
    0xfcfc000000000000ULL, // code 108
    0x04fc000000000000ULL, // code 109
    0xfc04000000000000ULL, // code 110
    0x0404000000000000ULL, // code 111
    0xfefefefefefefefeULL, // code 112
    0x02fefefefefefe02ULL, // code 113
    0x02fefefefefe02feULL, // code 114
    0xfefefefefefe0202ULL, // code 115
    0x02fefefefe02fefeULL, // code 116
    0xfefefefefe02fe02ULL, // code 117
    0xfefefefefe0202feULL, // code 118
    0x02fefefefe020202ULL, // code 119
    0x02fefefe02fefefeULL, // code 120
    0xfefefefe02fefe02ULL, // code 121
    0xfefefefe02fe02feULL, // code 122
    0x02fefefe02fe0202ULL, // code 123
    0xfefefefe0202fefeULL, // code 124
    0x02fefefe0202fe02ULL, // code 125
    0x02fefefe020202feULL, // code 126
    0xfefefefe02020202ULL, // code 127
    0x02fefe02fefefefeULL, // code 128
    0xfefefe02fefefe02ULL, // code 129
    0xfefefe02fefe02feULL, // code 130
    0x02fefe02fefe0202ULL, // code 131
    0xfefefe02fe02fefeULL, // code 132
    0x02fefe02fe02fe02ULL, // code 133
    0x02fefe02fe0202feULL, // code 134
    0xfefefe02fe020202ULL, // code 135
    0xfefefe0202fefefeULL, // code 136
    0x02fefe0202fefe02ULL, // code 137
    0x02fefe0202fe02feULL, // code 138
    0xfefefe0202fe0202ULL, // code 139
    0x02fefe020202fefeULL, // code 140
    0xfefefe020202fe02ULL, // code 141
    0xfefefe02020202feULL, // code 142
    0x02fefe0202020202ULL, // code 143
    0x02fe02fefefefefeULL, // code 144
    0xfefe02fefefefe02ULL, // code 145
    0xfefe02fefefe02feULL, // code 146
    0x02fe02fefefe0202ULL, // code 147
    0xfefe02fefe02fefeULL, // code 148
    0x02fe02fefe02fe02ULL, // code 149
    0x02fe02fefe0202feULL, // code 150
    0xfefe02fefe020202ULL, // code 151
    0xfefe02fe02fefefeULL, // code 152
    0x02fe02fe02fefe02ULL, // code 153
    0x02fe02fe02fe02feULL, // code 154
    0xfefe02fe02fe0202ULL, // code 155
    0x02fe02fe0202fefeULL, // code 156
    0xfefe02fe0202fe02ULL, // code 157
    0xfefe02fe020202feULL, // code 158
    0x02fe02fe02020202ULL, // code 159
    0xfefe0202fefefefeULL, // code 160
    0x02fe0202fefefe02ULL, // code 161
    0x02fe0202fefe02feULL, // code 162
    0xfefe0202fefe0202ULL, // code 163
    0x02fe0202fe02fefeULL, // code 164
    0xfefe0202fe02fe02ULL, // code 165
    0xfefe0202fe0202feULL, // code 166
    0x02fe0202fe020202ULL, // code 167
    0x02fe020202fefefeULL, // code 168
    0xfefe020202fefe02ULL, // code 169
    0xfefe020202fe02feULL, // code 170
    0x02fe020202fe0202ULL, // code 171
    0xfefe02020202fefeULL, // code 172
    0x02fe02020202fe02ULL, // code 173
    0x02fe0202020202feULL, // code 174
    0xfefe020202020202ULL, // code 175
    0x0202fefefefefefeULL, // code 176
    0xfe02fefefefefe02ULL, // code 177
    0xfe02fefefefe02feULL, // code 178
    0x0202fefefefe0202ULL, // code 179
    0xfe02fefefe02fefeULL, // code 180
    0x0202fefefe02fe02ULL, // code 181
    0x0202fefefe0202feULL, // code 182
    0xfe02fefefe020202ULL, // code 183
    0xfe02fefe02fefefeULL, // code 184
    0x0202fefe02fefe02ULL, // code 185
    0x0202fefe02fe02feULL, // code 186
    0xfe02fefe02fe0202ULL, // code 187
    0x0202fefe0202fefeULL, // code 188
    0xfe02fefe0202fe02ULL, // code 189
    0xfe02fefe020202feULL, // code 190
    0x0202fefe02020202ULL, // code 191
    0xfe02fe02fefefefeULL, // code 192
    0x0202fe02fefefe02ULL, // code 193
    0x0202fe02fefe02feULL, // code 194
    0xfe02fe02fefe0202ULL, // code 195
    0x0202fe02fe02fefeULL, // code 196
    0xfe02fe02fe02fe02ULL, // code 197
    0xfe02fe02fe0202feULL, // code 198
    0x0202fe02fe020202ULL, // code 199
    0x0202fe0202fefefeULL, // code 200
    0xfe02fe0202fefe02ULL, // code 201
    0xfe02fe0202fe02feULL, // code 202
    0x0202fe0202fe0202ULL, // code 203
    0xfe02fe020202fefeULL, // code 204
    0x0202fe020202fe02ULL, // code 205
    0x0202fe02020202feULL, // code 206
    0xfe02fe0202020202ULL, // code 207
    0xfe0202fefefefefeULL, // code 208
    0x020202fefefefe02ULL, // code 209
    0x020202fefefe02feULL, // code 210
    0xfe0202fefefe0202ULL, // code 211
    0x020202fefe02fefeULL, // code 212
    0xfe0202fefe02fe02ULL, // code 213
    0xfe0202fefe0202feULL, // code 214
    0x020202fefe020202ULL, // code 215
    0x020202fe02fefefeULL, // code 216
    0xfe0202fe02fefe02ULL, // code 217
    0xfe0202fe02fe02feULL, // code 218
    0x020202fe02fe0202ULL, // code 219
    0xfe0202fe0202fefeULL, // code 220
    0x020202fe0202fe02ULL, // code 221
    0x020202fe020202feULL, // code 222
    0xfe0202fe02020202ULL, // code 223
    0x02020202fefefefeULL, // code 224
    0xfe020202fefefe02ULL, // code 225
    0xfe020202fefe02feULL, // code 226
    0x02020202fefe0202ULL, // code 227
    0xfe020202fe02fefeULL, // code 228
    0x02020202fe02fe02ULL, // code 229
    0x02020202fe0202feULL, // code 230
    0xfe020202fe020202ULL, // code 231
    0xfe02020202fefefeULL, // code 232
    0x0202020202fefe02ULL, // code 233
    0x0202020202fe02feULL, // code 234
    0xfe02020202fe0202ULL, // code 235
    0x020202020202fefeULL, // code 236
    0xfe0202020202fe02ULL, // code 237
    0xfe020202020202feULL, // code 238
    0x0202020202020202ULL, // code 239
    0x0000000000000000ULL, // code 240
    0x0000000000000000ULL, // code 241
    0x0000000000000000ULL, // code 242
    0x0000000000000000ULL, // code 243
    0x0000000000000000ULL, // code 244
    0x0000000000000000ULL, // code 245
    0x0000000000000000ULL, // code 246
    0x0000000000000000ULL, // code 247
    0x0000000000000000ULL, // code 248
    0x0000000000000000ULL, // code 249
    0x0000000000000000ULL, // code 250
    0x0000000000000000ULL, // code 251
    0x0000000000000000ULL, // code 252
    0x0000000000000000ULL, // code 253
    0x0000000000000000ULL, // code 254
    0x0000000000000000ULL, // code 255
};

// 1-Cycle Hardware SIMD Decode using Constant Cache, PTX vadd4, and Radius Scaling
__device__ __forceinline__ void e8_root_decode_8d_fast(uint8_t root_code, uint8_t rad_axis_code, int8_t out[8]) {
    const uint32_t rad_idx  = rad_axis_code >> 4;
    const uint32_t axis_idx = rad_axis_code & 0x0F;

    if (rad_idx == 0) {
        *reinterpret_cast<uint64_t*>(out) = 0ULL;
        return;
    }

    const uint64_t w_root = __ldg(&c_e8_stage1_i8x8[root_code]);
    const uint64_t w_axis = c_axis_i8x8[axis_idx];

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    uint32_t dir_lo, dir_hi;
    asm("vadd4.s32.s32.s32.sat %0, %1, %2, %3;"
        : "=r"(dir_lo)
        : "r"(static_cast<uint32_t>(w_root)), "r"(static_cast<uint32_t>(w_axis)), "r"(0));
    asm("vadd4.s32.s32.s32.sat %0, %1, %2, %3;"
        : "=r"(dir_hi)
        : "r"(static_cast<uint32_t>(w_root >> 32)), "r"(static_cast<uint32_t>(w_axis >> 32)), "r"(0));

    const float scale = c_radius_scale[rad_idx];
    const int8_t* p_lo = reinterpret_cast<const int8_t*>(&dir_lo);
    const int8_t* p_hi = reinterpret_cast<const int8_t*>(&dir_hi);

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[i]     = static_cast<int8_t>(__float2int_rn(static_cast<float>(p_lo[i]) * scale));
        out[4 + i] = static_cast<int8_t>(__float2int_rn(static_cast<float>(p_hi[i]) * scale));
    }
#else
    const float scale = c_radius_scale[rad_idx];
    const int8_t* r = reinterpret_cast<const int8_t*>(&w_root);
    const int8_t* a = reinterpret_cast<const int8_t*>(&w_axis);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = static_cast<int8_t>(__float2int_rn(static_cast<float>(r[i] + a[i]) * scale));
    }
#endif
}

__device__ __forceinline__ void e8_root_decode_8d_int8(uint8_t root_code, uint8_t rad_axis_code, int8_t out[8]) {
    e8_root_decode_8d_fast(root_code, rad_axis_code, out);
}

} // namespace ninfer::ops

