#pragma once

// SM120 BF16 GDN gating projection for the two exact registered geometries:
//
//   Qwen3.6-27B:     a/b = W[48,5120] @ x[5120,T]
//   Qwen3.6-35B-A3B: a/b = W[32,2048] @ x[2048,T]
//
// A CTA computes the same 16 output rows from both weights over 128 (27B) or
// 64 (35B) tokens.
// Split-K routes use a tuned eight- or sixteen-warp specialization and an
// in-kernel cooperative grid reduction; the unsplit long-context route uses
// eight warps for more independent MMA accumulators. Both preserve a single
// kernel launch.

#include "ops/common/math.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cooperative_groups.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kBf16GdnBlockM     = 16;
inline constexpr int kBf16GdnBlockK     = 64;
inline constexpr int kBf16GdnWarps      = 16;
inline constexpr int kBf16GdnStages     = 2;
inline constexpr int kBf16GdnMFragments = kBf16GdnBlockM / 16;

template <int BlockN>
inline constexpr int kBf16GdnSmemElements =
    kBf16GdnStages * (BlockN * kBf16GdnBlockK + 2 * kBf16GdnBlockM * kBf16GdnBlockK);

template <int BlockN>
inline constexpr int kBf16GdnSmemBytes = kBf16GdnSmemElements<BlockN> * sizeof(__nv_bfloat16);

struct Bf16Gdn27Geometry {
    static constexpr int kHeads  = 48;
    static constexpr int kHidden = 5120;
    static constexpr int kBlockN = 128;
};

struct Bf16Gdn35Geometry {
    static constexpr int kHeads  = 32;
    static constexpr int kHidden = 2048;
    static constexpr int kBlockN = 64;
};

static_assert(Bf16Gdn27Geometry::kHidden % kBf16GdnBlockK == 0);
static_assert(Bf16Gdn35Geometry::kHidden % kBf16GdnBlockK == 0);

__device__ __forceinline__ int bf16_gdn_swizzle(int row, int col) {
    return (col & ~63) + gemm_swz64(row, col & 63);
}

template <class Geometry, int SplitK, bool FullTokens, int Warps, bool NormalizeInput = false,
          int NormTokenCapacity = 0>
__global__ __launch_bounds__(Warps * 32, 1) void bf16_gdn_gating_proj_gemm_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ normalized_x, float norm_eps,
    const __nv_bfloat16* __restrict__ a_weight, const __nv_bfloat16* __restrict__ b_weight,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias, float* __restrict__ partial,
    float* __restrict__ g, float* __restrict__ beta, std::int32_t t) {
    constexpr int kBf16GdnHeads       = Geometry::kHeads;
    constexpr int kBf16GdnHidden      = Geometry::kHidden;
    constexpr int kBf16GdnBlockN      = Geometry::kBlockN;
    constexpr int kBf16GdnLogicalRows = 2 * kBf16GdnHeads;
    constexpr int kBf16GdnKTiles      = kBf16GdnHidden / kBf16GdnBlockK;
    static_assert(SplitK >= 1 && kBf16GdnKTiles % SplitK == 0,
                  "BF16 GDN split-K must divide the exact geometry's K tiles");
    static_assert(Warps == 8 || Warps == 16);
    static_assert(kBf16GdnBlockN % Warps == 0);
    constexpr int kTilesPerSplit = kBf16GdnKTiles / SplitK;
    constexpr int kKSubtiles     = kBf16GdnBlockK / 16;
    constexpr int kThreads       = Warps * 32;
    constexpr int kWarpN         = kBf16GdnBlockN / Warps;
    static_assert(kWarpN % 8 == 0);
    constexpr int kNFragments = kWarpN / 8;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* xs  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    auto* aws = xs + kBf16GdnStages * kBf16GdnBlockN * kBf16GdnBlockK;
    auto* bws = aws + kBf16GdnStages * kBf16GdnBlockM * kBf16GdnBlockK;

    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int head0    = static_cast<int>(blockIdx.y) * kBf16GdnBlockM;
    const int token0   = static_cast<int>(blockIdx.x) * kBf16GdnBlockN;
    const int split    = static_cast<int>(blockIdx.z);
    const int kt_begin = split * kTilesPerSplit;

    if constexpr (NormalizeInput) {
        static_assert(SplitK == 32, "fused input normalization is tuned for split-32");
        static_assert(NormTokenCapacity > 0 && NormTokenCapacity <= 16);
        constexpr int kLocalPairs     = kBf16GdnBlockK / 2;
        const auto* x2                = reinterpret_cast<const __nv_bfloat162*>(x);
        const int token_count         = min(kBf16GdnBlockN, t - token0);
        float sums[NormTokenCapacity] = {};
        // One warp from row tile zero contributes a 64-element norm slice. The existing post-MMA
        // cooperative handoff reduces the 32 slices, so normalization adds no grid-wide barrier.
        if (blockIdx.y == 0 && warp == 0) {
            for (int pair = lane; pair < kLocalPairs; pair += kWarpSize) {
#pragma unroll
                for (int token_local = 0; token_local < NormTokenCapacity; ++token_local) {
                    if (token_local >= token_count) { continue; }
                    const std::int64_t row_base =
                        static_cast<std::int64_t>(token0 + token_local) * (kBf16GdnHidden / 2);
                    const int global_pair = kt_begin * (kBf16GdnBlockK / 2) + pair;
                    const float2 value    = __bfloat1622float2(x2[row_base + global_pair]);
                    sums[token_local] += value.x * value.x + value.y * value.y;
                }
            }
#pragma unroll
            for (int token_local = 0; token_local < NormTokenCapacity; ++token_local) {
                sums[token_local] = warp_reduce_sum(sums[token_local]);
                if (lane == 0 && token_local < token_count) {
                    float* norm_partial =
                        partial + static_cast<std::int64_t>(SplitK) * t * kBf16GdnLogicalRows;
                    norm_partial[static_cast<std::int64_t>(split) * t + token0 + token_local] =
                        sums[token_local];
                }
            }
        }
    }

    float a_acc[kBf16GdnMFragments][kNFragments][4] = {};
    float b_acc[kBf16GdnMFragments][kNFragments][4] = {};

    auto stage_load = [&](int stage, int kt) {
        const int k0 = kt * kBf16GdnBlockK;

        // x is shared by all 16 head rows and by both projections.
        for (int vec = tid; vec < kBf16GdnBlockN * (kBf16GdnBlockK / 8); vec += kThreads) {
            const int token_local = vec / (kBf16GdnBlockK / 8);
            const int k_vec       = vec - token_local * (kBf16GdnBlockK / 8);
            const int kk          = k_vec * 8;
            const int token       = token0 + token_local;
            __nv_bfloat16* dst =
                &xs[stage * kBf16GdnBlockN * kBf16GdnBlockK + token_local * kBf16GdnBlockK +
                    bf16_gdn_swizzle(token_local, kk)];
            if constexpr (NormalizeInput) {
                const bool valid = FullTokens || token < t;
                if (valid) {
                    const auto* source = reinterpret_cast<const __nv_bfloat162*>(
                        &x[static_cast<std::int64_t>(token) * kBf16GdnHidden + k0 + kk]);
                    const auto* gain =
                        reinterpret_cast<const __nv_bfloat162*>(&norm_weight[k0 + kk]);
                    auto* target = reinterpret_cast<__nv_bfloat162*>(dst);
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const float2 value              = __bfloat1622float2(source[pair]);
                        const float2 weight             = __bfloat1622float2(gain[pair]);
                        const __nv_bfloat162 normalized = __floats2bfloat162_rn(
                            value.x * (1.0f + weight.x), value.y * (1.0f + weight.y));
                        target[pair] = normalized;
                    }
                } else {
                    *reinterpret_cast<uint4*>(dst) = uint4{0, 0, 0, 0};
                }
            } else if constexpr (FullTokens) {
                cp_async<16, Cache::cg>(
                    dst, &x[static_cast<std::int64_t>(token) * kBf16GdnHidden + k0 + kk]);
            } else {
                const bool valid     = token < t;
                const int safe_token = valid ? token : 0;
                cp_async_zfill<16, Cache::cg>(
                    dst, &x[static_cast<std::int64_t>(safe_token) * kBf16GdnHidden + k0 + kk],
                    valid ? 16 : 0);
            }
        }

        // Each thread moves one 16-byte vector: 128 vectors from Wa and 128
        // from Wb. Both contiguous weights remain BF16 all the way into ldmatrix.
        const int weight_vecs = kBf16GdnBlockM * (kBf16GdnBlockK / 8);
        for (int all_vec = tid; all_vec < 2 * weight_vecs; all_vec += kThreads) {
            const bool is_b    = all_vec >= weight_vecs;
            const int vec      = is_b ? all_vec - weight_vecs : all_vec;
            const int row      = vec / (kBf16GdnBlockK / 8);
            const int k_vec    = vec - row * (kBf16GdnBlockK / 8);
            const int kk       = k_vec * 8;
            __nv_bfloat16* dst = (is_b ? bws : aws) + stage * kBf16GdnBlockM * kBf16GdnBlockK +
                                 row * kBf16GdnBlockK + bf16_gdn_swizzle(row, kk);
            const __nv_bfloat16* weight = is_b ? b_weight : a_weight;
            cp_async<16, Cache::cg>(
                dst, &weight[static_cast<std::int64_t>(head0 + row) * kBf16GdnHidden + k0 + kk]);
        }
    };

#pragma unroll
    for (int stage = 0; stage < kBf16GdnStages; ++stage) {
        if (stage < kTilesPerSplit) { stage_load(stage, kt_begin + stage); }
        ninfer::ops::cp_commit();
    }

    // ldmatrix fragment lane offsets. A is [M,K], while the token-major x tile
    // is exactly the column-major B representation expected by mma.
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int x_rin    = lane & 7;
    const int x_koff   = ((lane >> 3) & 1) << 3;

#pragma unroll 1
    for (int it = 0; it < kTilesPerSplit; ++it) {
        const int stage = it & 1;
        ninfer::ops::cp_wait<kBf16GdnStages - 1>();
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < kKSubtiles; ++ki) {
            unsigned xf[kNFragments][2];
#pragma unroll
            for (int ni = 0; ni < kNFragments; ++ni) {
                const int xrow = warp * kWarpN + ni * 8 + x_rin;
                const int xcol = ki * 16 + x_koff;
                ldmatrix_x2(xf[ni][0], xf[ni][1],
                            smem_addr(&xs[stage * kBf16GdnBlockN * kBf16GdnBlockK +
                                          xrow * kBf16GdnBlockK + bf16_gdn_swizzle(xrow, xcol)]));
            }
#pragma unroll
            for (int mi = 0; mi < kBf16GdnMFragments; ++mi) {
                unsigned af[4], bf[4];
                const int arow         = mi * 16 + a_rowoff;
                const int acol         = ki * 16 + a_coloff;
                const int weight_stage = stage * kBf16GdnBlockM * kBf16GdnBlockK;
                ldmatrix_x4(
                    af[0], af[1], af[2], af[3],
                    smem_addr(
                        &aws[weight_stage + arow * kBf16GdnBlockK + bf16_gdn_swizzle(arow, acol)]));
                ldmatrix_x4(
                    bf[0], bf[1], bf[2], bf[3],
                    smem_addr(
                        &bws[weight_stage + arow * kBf16GdnBlockK + bf16_gdn_swizzle(arow, acol)]));
#pragma unroll
                for (int ni = 0; ni < kNFragments; ++ni) {
                    mma_bf16(a_acc[mi][ni][0], a_acc[mi][ni][1], a_acc[mi][ni][2], a_acc[mi][ni][3],
                             af[0], af[1], af[2], af[3], xf[ni][0], xf[ni][1]);
                    mma_bf16(b_acc[mi][ni][0], b_acc[mi][ni][1], b_acc[mi][ni][2], b_acc[mi][ni][3],
                             bf[0], bf[1], bf[2], bf[3], xf[ni][0], xf[ni][1]);
                }
            }
        }

        __syncthreads();
        const int next = it + kBf16GdnStages;
        if (next < kTilesPerSplit) { stage_load(stage, kt_begin + next); }
        ninfer::ops::cp_commit();
    }

#pragma unroll
    for (int mi = 0; mi < kBf16GdnMFragments; ++mi) {
        const int row0 = head0 + mi * 16 + gid;
        const int row1 = row0 + 8;
#pragma unroll
        for (int ni = 0; ni < kNFragments; ++ni) {
            const int col0 = token0 + warp * kWarpN + ni * 8 + 2 * lid;
            const int col1 = col0 + 1;
            auto store     = [&](int token, int row, float av, float bv) {
                if constexpr (SplitK == 1) {
                    const std::int64_t out_i =
                        static_cast<std::int64_t>(token) * kBf16GdnHeads + row;
                    g[out_i]    = -expf(A_log[row]) * softplus(av + dt_bias[row]);
                    beta[out_i] = sigmoid(bv);
                } else {
                    const std::int64_t base =
                        (static_cast<std::int64_t>(split) * t + token) * kBf16GdnLogicalRows;
                    partial[base + row]                 = av;
                    partial[base + kBf16GdnHeads + row] = bv;
                }
            };
            if constexpr (FullTokens) {
                store(col0, row0, a_acc[mi][ni][0], b_acc[mi][ni][0]);
                store(col1, row0, a_acc[mi][ni][1], b_acc[mi][ni][1]);
                store(col0, row1, a_acc[mi][ni][2], b_acc[mi][ni][2]);
                store(col1, row1, a_acc[mi][ni][3], b_acc[mi][ni][3]);
            } else {
                if (col0 < t) {
                    store(col0, row0, a_acc[mi][ni][0], b_acc[mi][ni][0]);
                    store(col0, row1, a_acc[mi][ni][2], b_acc[mi][ni][2]);
                }
                if (col1 < t) {
                    store(col1, row0, a_acc[mi][ni][1], b_acc[mi][ni][1]);
                    store(col1, row1, a_acc[mi][ni][3], b_acc[mi][ni][3]);
                }
            }
        }
    }

    if constexpr (SplitK > 1) {
        cooperative_groups::this_grid().sync();
        const int block_linear = (static_cast<int>(blockIdx.z) * static_cast<int>(gridDim.y) +
                                  static_cast<int>(blockIdx.y)) *
                                     static_cast<int>(gridDim.x) +
                                 static_cast<int>(blockIdx.x);
        const int grid_threads = static_cast<int>(gridDim.x) * static_cast<int>(gridDim.y) *
                                 static_cast<int>(gridDim.z) * kThreads;
        const int elems = kBf16GdnHeads * t;
        if constexpr (NormalizeInput) {
            const float* norm_partial =
                partial + static_cast<std::int64_t>(SplitK) * t * kBf16GdnLogicalRows;
            const int hidden_elems = kBf16GdnHidden * t;
            for (int i = block_linear * kThreads + tid; i < hidden_elems; i += grid_threads) {
                const int k     = i % kBf16GdnHidden;
                const int token = i / kBf16GdnHidden;
                float sum       = 0.0F;
#pragma unroll
                for (int s = 0; s < SplitK; ++s) {
                    sum += norm_partial[static_cast<std::int64_t>(s) * t + token];
                }
                const float inv    = rsqrtf(sum / static_cast<float>(kBf16GdnHidden) + norm_eps);
                const float value  = __bfloat162float(x[i]);
                const float weight = __bfloat162float(norm_weight[k]);
                normalized_x[i]    = __float2bfloat16_rn(value * inv * (1.0F + weight));
            }
        }
        for (int i = block_linear * kThreads + tid; i < elems; i += grid_threads) {
            const int row   = i % kBf16GdnHeads;
            const int token = i / kBf16GdnHeads;
            float av        = 0.0f;
            float bv        = 0.0f;
#pragma unroll
            for (int s = 0; s < SplitK; ++s) {
                const std::int64_t base =
                    (static_cast<std::int64_t>(s) * t + token) * kBf16GdnLogicalRows;
                av += partial[base + row];
                bv += partial[base + kBf16GdnHeads + row];
            }
            if constexpr (NormalizeInput) {
                const float* norm_partial =
                    partial + static_cast<std::int64_t>(SplitK) * t * kBf16GdnLogicalRows;
                float sum = 0.0F;
#pragma unroll
                for (int s = 0; s < SplitK; ++s) {
                    sum += norm_partial[static_cast<std::int64_t>(s) * t + token];
                }
                const float inv = rsqrtf(sum / static_cast<float>(kBf16GdnHidden) + norm_eps);
                av *= inv;
                bv *= inv;
            }
            g[i]    = -expf(A_log[row]) * softplus(av + dt_bias[row]);
            beta[i] = sigmoid(bv);
        }
    }
}

} // namespace ninfer::ops::detail
