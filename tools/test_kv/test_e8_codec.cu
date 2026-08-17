#include "test_e8_codec.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

namespace ninfer::test_kv {

// 1. Two-Stage 240-Root E8 Encoder Kernel
__global__ void e8_encode_tile_kernel(
    const float* __restrict__ keys,
    E8Packed2BitTile* __restrict__ out_tiles,
    int num_tokens
) {
    const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    const int tile_idx = token_idx / kTokensPerTile;
    const int in_tile_idx = token_idx % kTokensPerTile;
    const float* token_key = keys + static_cast<size_t>(token_idx) * kHeadDim;

    #pragma unroll
    for (int sub = 0; sub < kNumSubspaces; ++sub) {
        float raw[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            raw[i] = token_key[sub * 8 + i];
        }

        float rot[8];
        hadamard_rot_8d(raw, rot);

        float norm_sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            norm_sq += rot[i] * rot[i];
        }
        float norm = sqrtf(norm_sq) + 1e-8f;
        float inv_norm = 1.0f / norm;

        float u[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            u[i] = rot[i] * inv_norm;
        }

        // Stage 1
        float v1[8];
        uint8_t code1 = e8_quantize_root_8d(u, v1);

        // Stage 2: Residual
        constexpr float kInvSqrt2 = 0.7071067811865475f;
        float dot_u_v1 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            dot_u_v1 += u[i] * v1[i] * kInvSqrt2;
        }

        float res[8];
        float res_sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            res[i] = u[i] - dot_u_v1 * (v1[i] * kInvSqrt2);
            res_sq += res[i] * res[i];
        }
        float res_norm = sqrtf(res_sq) + 1e-8f;
        float inv_res_norm = 1.0f / res_norm;

        float u_res[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            u_res[i] = res[i] * inv_res_norm;
        }

        float v2[8];
        uint8_t code2 = e8_quantize_root_8d(u_res, v2);

        out_tiles[tile_idx].codes[in_tile_idx][sub][0] = code1;
        out_tiles[tile_idx].codes[in_tile_idx][sub][1] = code2;
        out_tiles[tile_idx].scales[in_tile_idx][sub] = __float2half(norm);
    }
}

// 2. Two-Stage 240-Root E8 Decode Attention Kernel
__global__ void e8_decode_attention_kernel(
    const float* __restrict__ query_raw,
    const E8Packed2BitTile* __restrict__ tiles,
    float* __restrict__ out_scores,
    int num_tiles,
    int num_tokens
) {
    extern __shared__ char smem_raw[];
    E8Packed2BitTile* s_tile = reinterpret_cast<E8Packed2BitTile*>(smem_raw);

    __shared__ float s_q_rot[256];
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    for (int sub = tid; sub < kNumSubspaces; sub += num_threads) {
        float q_in[8], q_out[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            q_in[i] = query_raw[sub * 8 + i];
        }
        hadamard_rot_8d(q_in, q_out);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_q_rot[sub * 8 + i] = q_out[i];
        }
    }
    __syncthreads();

    const int tile_idx = blockIdx.x;
    if (tile_idx >= num_tiles) return;

    constexpr int kTileBytes = sizeof(E8Packed2BitTile);
    constexpr int kVecs = kTileBytes / 16;
    const uint4* src_vecs = reinterpret_cast<const uint4*>(tiles + tile_idx);
    uint4* dst_vecs = reinterpret_cast<uint4*>(s_tile);

    for (int i = tid; i < kVecs; i += num_threads) {
        #if __CUDA_ARCH__ >= 800
        unsigned smem_target = static_cast<unsigned>(__cvta_generic_to_shared(dst_vecs + i));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_target),
                       "l"(src_vecs + i));
        #else
        dst_vecs[i] = src_vecs[i];
        #endif
    }
    #if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    #endif
    __syncthreads();

    constexpr float kC1 = 0.88f;
    constexpr float kC2 = 0.47f;

    for (int t = tid; t < kTokensPerTile; t += num_threads) {
        const int global_token_idx = tile_idx * kTokensPerTile + t;
        if (global_token_idx >= num_tokens) continue;

        float total_score = 0.0f;

        #pragma unroll
        for (int sub = 0; sub < kNumSubspaces; ++sub) {
            float q_sub[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                q_sub[i] = s_q_rot[sub * 8 + i];
            }

            uint8_t code1 = s_tile->codes[t][sub][0];
            uint8_t code2 = s_tile->codes[t][sub][1];
            float scale = __half2float(s_tile->scales[t][sub]);

            float dot1 = e8_decode_dot_8d(q_sub, code1);
            float dot2 = e8_decode_dot_8d(q_sub, code2);

            total_score += scale * (kC1 * dot1 + kC2 * dot2);
        }

        out_scores[global_token_idx] = total_score;
    }
}

// 3. General Conway-Sloane E8 Lattice Point Encoder Kernel (4-bit packing)
__global__ void e8_general_encode_tile_kernel(
    const float* __restrict__ keys,
    E8Packed4BitTile* __restrict__ out_tiles,
    int num_tokens
) {
    const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    const int tile_idx = token_idx / kTokensPerTile;
    const int in_tile_idx = token_idx % kTokensPerTile;
    const float* token_key = keys + static_cast<size_t>(token_idx) * kHeadDim;

    // Process each 64-dim group
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        float rot_64[64];
        // 8x8 rotations across the 8 subspaces in this group
        #pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
            float raw[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                raw[i] = token_key[g * 64 + sub * 8 + i];
            }
            hadamard_rot_8d(raw, &rot_64[sub * 8]);
        }

        // Compute group scale
        float amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            float val = fabsf(rot_64[i]);
            if (val > amax) amax = val;
        }
        amax = fmaxf(amax, 1e-4f);
        float scale = amax / 7.0f;
        float inv_scale = 1.0f / scale;
        out_tiles[tile_idx].scales[in_tile_idx][g] = __float2half(scale);

        // Project each 8D subspace onto E8 lattice and pack into 4-bit
        #pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
            float scaled_x[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                scaled_x[i] = rot_64[sub * 8 + i] * inv_scale;
            }

            float e8_pt[8];
            e8_project_8d_general(scaled_x, e8_pt);

            #pragma unroll
            for (int i = 0; i < 8; i += 2) {
                int q0 = static_cast<int>(rintf(e8_pt[i]));
                int q1 = static_cast<int>(rintf(e8_pt[i + 1]));
                q0 = max(-8, min(7, q0));
                q1 = max(-8, min(7, q1));
                uint8_t packed = (static_cast<uint8_t>(q0 & 0x0F)) |
                                 (static_cast<uint8_t>((q1 & 0x0F) << 4));
                out_tiles[tile_idx].codes[in_tile_idx][g * 32 + sub * 4 + i / 2] = packed;
            }
        }
    }
}

// 4. General Conway-Sloane E8 Decode Attention Kernel (4-bit MMA / registers)
__global__ void e8_general_decode_attention_kernel(
    const float* __restrict__ query_raw,
    const E8Packed4BitTile* __restrict__ tiles,
    float* __restrict__ out_scores,
    int num_tiles,
    int num_tokens
) {
    extern __shared__ char smem_raw[];
    E8Packed4BitTile* s_tile = reinterpret_cast<E8Packed4BitTile*>(smem_raw);

    __shared__ float s_q_rot[256];
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    for (int sub = tid; sub < kNumSubspaces; sub += num_threads) {
        float q_in[8], q_out[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            q_in[i] = query_raw[sub * 8 + i];
        }
        hadamard_rot_8d(q_in, q_out);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_q_rot[sub * 8 + i] = q_out[i];
        }
    }
    __syncthreads();

    const int tile_idx = blockIdx.x;
    if (tile_idx >= num_tiles) return;

    constexpr int kTileBytes = sizeof(E8Packed4BitTile);
    constexpr int kVecs = kTileBytes / 16;
    const uint4* src_vecs = reinterpret_cast<const uint4*>(tiles + tile_idx);
    uint4* dst_vecs = reinterpret_cast<uint4*>(s_tile);

    for (int i = tid; i < kVecs; i += num_threads) {
        #if __CUDA_ARCH__ >= 800
        unsigned smem_target = static_cast<unsigned>(__cvta_generic_to_shared(dst_vecs + i));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_target),
                       "l"(src_vecs + i));
        #else
        dst_vecs[i] = src_vecs[i];
        #endif
    }
    #if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    #endif
    __syncthreads();

    for (int t = tid; t < kTokensPerTile; t += num_threads) {
        const int global_token_idx = tile_idx * kTokensPerTile + t;
        if (global_token_idx >= num_tokens) continue;

        float total_score = 0.0f;

        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            float g_scale = __half2float(s_tile->scales[t][g]);
            float group_sum = 0.0f;

            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                uint8_t packed = s_tile->codes[t][g * 32 + i];
                int s0 = (static_cast<int>(static_cast<int8_t>(packed << 4))) >> 4;
                int s1 = (static_cast<int>(static_cast<int8_t>(packed & 0xF0))) >> 4;

                group_sum += s_q_rot[g * 64 + i * 2] * static_cast<float>(s0);
                group_sum += s_q_rot[g * 64 + i * 2 + 1] * static_cast<float>(s1);
            }

            total_score += group_sum * g_scale;
        }

        out_scores[global_token_idx] = total_score;
    }
}

// 5. Uncompressed FP32 Ground Truth Reference Kernel
__global__ void fp32_reference_attention_kernel(
    const float* __restrict__ query,
    const float* __restrict__ keys,
    float* __restrict__ out_scores,
    int num_tokens
) {
    const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens) return;

    const float* token_key = keys + static_cast<size_t>(token_idx) * kHeadDim;
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < kHeadDim; ++i) {
        sum += query[i] * token_key[i];
    }
    out_scores[token_idx] = sum;
}

} // namespace ninfer::test_kv
