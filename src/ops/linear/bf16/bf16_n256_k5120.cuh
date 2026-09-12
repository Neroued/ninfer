#pragma once

// Small-output BF16 x BF16 MMA for the exact [256,5120] selector projection. A CTA computes one
// 16-row by 8/16-token tile. Sixteen/eight warps split K, double-buffer global-to-shared staging,
// and reduce FP32 fragments in shared memory before the final BF16 store.

#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int KWarps, int TileTokens>
struct Bf16N256K5120MmaSchedule {
    static constexpr int kOutputRowsPerCta  = 16;
    static constexpr int kKWarps            = KWarps;
    static constexpr int kTileTokens        = TileTokens;
    static constexpr int kPipelineStages    = 2;
    static constexpr Cache kWeightCache     = Cache::cg;
    static constexpr Cache kActivationCache = Cache::cg;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kTokenMmas         = kTileTokens / 8;
    static constexpr int kStageElements     = (16 + kTileTokens) * kGroupK;
    static constexpr int kStagingBytes =
        kPipelineStages * kStageElements * static_cast<int>(sizeof(__nv_bfloat16));
    static constexpr int kPartialBytes =
        kKWarps * kTokenMmas * 32 * 4 * static_cast<int>(sizeof(float));
    static_assert(kStagingBytes >= kPartialBytes);
    static constexpr int kSharedBytes = kStagingBytes;

    static_assert((5120 % kGroupK) == 0);
    static_assert(kSharedBytes <= 99 * 1024);
};

__device__ __forceinline__ int bf16_n256_k5120_swizzle(int row, int col) {
    return (col & ~63) + ((((col & 63) >> 3) ^ (row & 7)) << 3) + (col & 7);
}

template <class Schedule>
__global__ __launch_bounds__(Schedule::kThreads, 1) void bf16_n256_k5120_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out, std::int32_t tokens) {
    constexpr int kRows       = 256;
    constexpr int kHidden     = 5120;
    constexpr int kMmaRows    = 16;
    constexpr int kMmaK       = Schedule::kTileKPerWarp;
    constexpr int kKWarps     = Schedule::kKWarps;
    constexpr int kTileTokens = Schedule::kTileTokens;
    constexpr int kTokenMmas  = Schedule::kTokenMmas;
    constexpr int kGroupK     = Schedule::kGroupK;
    constexpr int kGroups     = kHidden / kGroupK;
    static_assert((kMmaK % 16) == 0);

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* staging = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* partial = reinterpret_cast<float*>(shared_raw);

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int gid    = lane >> 2;
    const int lid    = lane & 3;
    const int row0   = static_cast<int>(blockIdx.x) * Schedule::kOutputRowsPerCta;
    const int token0 = static_cast<int>(blockIdx.y) * kTileTokens;

    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_k_offset   = ((lane >> 3) & 1) << 3;
    const int warp_k0      = warp * kMmaK;

    float accum[kTokenMmas][4] = {};

    const auto stage_group = [&](int stage, int group) {
        const int group_k0  = group * kGroupK;
        auto* weight_shared = staging + stage * Schedule::kStageElements;
        auto* x_shared      = weight_shared + kMmaRows * kGroupK;

        constexpr int kWeightVectors = kMmaRows * (kGroupK / 8);
        for (int item = tid; item < kWeightVectors; item += Schedule::kThreads) {
            const int row = item / (kGroupK / 8);
            const int k8  = item - row * (kGroupK / 8);
            cp_async<16, Schedule::kWeightCache>(
                &weight_shared[row * kGroupK + bf16_n256_k5120_swizzle(row, k8 * 8)],
                &weight[static_cast<std::int64_t>(row0 + row) * kHidden + group_k0 + k8 * 8]);
        }

        constexpr int kActivationVectors = kTileTokens * (kGroupK / 8);
        for (int item = tid; item < kActivationVectors; item += Schedule::kThreads) {
            const int token        = item / (kGroupK / 8);
            const int k8           = item - token * (kGroupK / 8);
            const bool valid_token = token0 + token < tokens;
            const int source_token = valid_token ? token0 + token : 0;
            cp_async_zfill<16, Schedule::kActivationCache>(
                &x_shared[token * kGroupK + bf16_n256_k5120_swizzle(token, k8 * 8)],
                &x[static_cast<std::int64_t>(source_token) * kHidden + group_k0 + k8 * 8],
                valid_token ? 16 : 0);
        }
    };

    const auto compute_group = [&](int stage) {
        auto* weight_shared = staging + stage * Schedule::kStageElements;
        auto* x_shared      = weight_shared + kMmaRows * kGroupK;

#pragma unroll
        for (int k_step = 0; k_step < kMmaK / 16; ++k_step) {
            unsigned a_frag[4];
            const int a_col = warp_k0 + k_step * 16 + a_col_offset;
            ldmatrix_x4(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                        smem_addr(&weight_shared[a_row_offset * kGroupK +
                                                 bf16_n256_k5120_swizzle(a_row_offset, a_col)]));
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                unsigned b_frag[2];
                const int b_row = token_mma * 8 + b_inner_row;
                const int b_col = warp_k0 + k_step * 16 + b_k_offset;
                ldmatrix_x2(
                    b_frag[0], b_frag[1],
                    smem_addr(&x_shared[b_row * kGroupK + bf16_n256_k5120_swizzle(b_row, b_col)]));
                mma_bf16(accum[token_mma][0], accum[token_mma][1], accum[token_mma][2],
                         accum[token_mma][3], a_frag[0], a_frag[1], a_frag[2], a_frag[3], b_frag[0],
                         b_frag[1]);
            }
        }
    };

#pragma unroll
    for (int stage = 0; stage < Schedule::kPipelineStages; ++stage) {
        stage_group(stage, stage);
        cp_commit();
    }

#pragma unroll 1
    for (int group = 0; group < kGroups; ++group) {
        if (group + Schedule::kPipelineStages <= kGroups) {
            cp_wait<Schedule::kPipelineStages - 1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const int stage = group % Schedule::kPipelineStages;
        compute_group(stage);
        __syncthreads();

        const int next = group + Schedule::kPipelineStages;
        if (next < kGroups) {
            stage_group(stage, next);
            cp_commit();
        }
    }

    if ((warp & 1) != 0) {
#pragma unroll
        for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
            const std::int64_t base =
                (static_cast<std::int64_t>(warp) * kTokenMmas + token_mma) * 32 * 4;
            store_vec(partial + base + static_cast<std::int64_t>(lane) * 4,
                      make_float4(accum[token_mma][0], accum[token_mma][1], accum[token_mma][2],
                                  accum[token_mma][3]));
        }
    }
    __syncthreads();

    if ((warp & 1) == 0) {
#pragma unroll
        for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
            const std::int64_t partner_base =
                (static_cast<std::int64_t>(warp + 1) * kTokenMmas + token_mma) * 32 * 4;
            const float4 partner =
                load_vec<float4>(partial + partner_base + static_cast<std::int64_t>(lane) * 4);
            accum[token_mma][0] += partner.x;
            accum[token_mma][1] += partner.y;
            accum[token_mma][2] += partner.z;
            accum[token_mma][3] += partner.w;
            if (warp != 0) {
                const std::int64_t base =
                    (static_cast<std::int64_t>(warp) * kTokenMmas + token_mma) * 32 * 4;
                store_vec(partial + base + static_cast<std::int64_t>(lane) * 4,
                          make_float4(accum[token_mma][0], accum[token_mma][1], accum[token_mma][2],
                                      accum[token_mma][3]));
            }
        }
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
            float4 sum = make_float4(accum[token_mma][0], accum[token_mma][1], accum[token_mma][2],
                                     accum[token_mma][3]);
#pragma unroll
            for (int split = 2; split < kKWarps; split += 2) {
                const std::int64_t base =
                    (static_cast<std::int64_t>(split) * kTokenMmas + token_mma) * 32 * 4;
                const float4 value =
                    load_vec<float4>(partial + base + static_cast<std::int64_t>(lane) * 4);
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            const int token  = token0 + token_mma * 8 + 2 * lid;
            const auto store = [&](int local_row, int local_token, float value) {
                if (local_token < tokens) {
                    out[static_cast<std::int64_t>(local_token) * kRows + row0 + local_row] =
                        __float2bfloat16_rn(value);
                }
            };
            store(gid, token, sum.x);
            store(gid, token + 1, sum.y);
            store(gid + 8, token, sum.z);
            store(gid + 8, token + 1, sum.w);
        }
    }
}

} // namespace ninfer::ops::detail
