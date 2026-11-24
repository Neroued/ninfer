#pragma once

#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/cuda/reduce.cuh"

// BF16 control dots, optional input norm, and the complete control epilogue.
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/linear/kernels/rowsplit/rowsplit_mma.cuh"
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>

namespace ninfer::ops::detail::gdn_control {

inline constexpr int kBlockM     = 16;
inline constexpr int kBlockK     = 64;
inline constexpr int kStages     = 2;
inline constexpr int kMFragments = kBlockM / 16;

template <int Heads, int Hidden, int BlockN, bool Paired = true, bool HeadFast = false>
struct MmaGeometry {
    static constexpr bool kPaired   = Paired;
    static constexpr bool kHeadFast = HeadFast;
    static constexpr int kHeads = Heads, kHidden = Hidden, kBlockN = BlockN;
    static constexpr int shared_bytes =
        kStages * (BlockN * kBlockK + (Paired ? 2 : 1) * kBlockM * kBlockK) * sizeof(__nv_bfloat16);
    static_assert(BlockN == 64 || BlockN == 128);
    static_assert(Heads % kBlockM == 0);
    static_assert(Hidden % kBlockK == 0);
};

__device__ __forceinline__ int swizzle(int row, int col) {
    return (col & ~63) + common::swizzle_xor_64(row, col & 63);
}

template <class Geometry, int SplitK, bool FullTokens, int Warps, bool NormalizeInput = false,
          int NormTokenCapacity = 0>
__global__ __launch_bounds__(Warps * 32, 1) void mma_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ norm_weight,
    __nv_bfloat16* __restrict__ normalized_x, float norm_eps,
    const __nv_bfloat16* __restrict__ a_weight, const __nv_bfloat16* __restrict__ b_weight,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias, float* __restrict__ partial,
    float* __restrict__ g, float* __restrict__ beta, std::int32_t t) {
    constexpr bool Paired      = Geometry::kPaired;
    constexpr int kHeads       = Geometry::kHeads;
    constexpr int kHidden      = Geometry::kHidden;
    constexpr int kBlockN      = Geometry::kBlockN;
    constexpr int kLogicalRows = 2 * kHeads;
    constexpr int kKTiles      = kHidden / kBlockK;
    static_assert(SplitK >= 1 && kKTiles % SplitK == 0,
                  "BF16 GDN split-K must divide the exact geometry's K tiles");
    static_assert(Warps == 8 || Warps == 16);
    static_assert(kBlockN % Warps == 0);
    constexpr int kTilesPerSplit = kKTiles / SplitK;
    constexpr int kKSubtiles     = kBlockK / 16;
    constexpr int kThreads       = Warps * 32;
    constexpr int kWarpN         = kBlockN / Warps;
    static_assert(kWarpN % 8 == 0);
    constexpr int kNFragments = kWarpN / 8;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* xs  = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    auto* aws = xs + kStages * kBlockN * kBlockK;
    auto* bws = aws + kStages * kBlockM * kBlockK;

    const int tid            = static_cast<int>(threadIdx.x);
    const int warp           = tid >> 5;
    const int lane           = tid & 31;
    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    constexpr int kHeadTiles = (Paired ? kHeads : 2 * kHeads) / kBlockM;
    const int linear_tile =
        static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) + static_cast<int>(blockIdx.x);
    const int head_tile    = Geometry::kHeadFast ? linear_tile % kHeadTiles : blockIdx.y;
    const int token_tile   = Geometry::kHeadFast ? linear_tile / kHeadTiles : blockIdx.x;
    const int head0        = head_tile * kBlockM;
    const int weight_head0 = Paired ? head0 : head0 % kHeads;
    const int token0       = token_tile * kBlockN;
    const int split        = static_cast<int>(blockIdx.z);
    const int kt_begin     = split * kTilesPerSplit;

    if constexpr (NormalizeInput) {
        static_assert(Paired);
        static_assert(SplitK == 32, "fused input normalization is tuned for split-32");
        static_assert(NormTokenCapacity > 0 && NormTokenCapacity <= 16);
        constexpr int kLocalPairs     = kBlockK / 2;
        const auto* x2                = reinterpret_cast<const __nv_bfloat162*>(x);
        const int token_count         = min(kBlockN, t - token0);
        float sums[NormTokenCapacity] = {};
        // One warp from row tile zero contributes a 64-element norm slice. The existing post-MMA
        // cooperative handoff reduces the 32 slices, so normalization adds no grid-wide barrier.
        if (head_tile == 0 && warp == 0) {
            for (int pair = lane; pair < kLocalPairs; pair += common::kWarpSize) {
#pragma unroll
                for (int token_local = 0; token_local < NormTokenCapacity; ++token_local) {
                    if (token_local >= token_count) { continue; }
                    const std::int64_t row_base =
                        static_cast<std::int64_t>(token0 + token_local) * (kHidden / 2);
                    const int global_pair = kt_begin * (kBlockK / 2) + pair;
                    const float2 value    = __bfloat1622float2(x2[row_base + global_pair]);
                    sums[token_local] += value.x * value.x + value.y * value.y;
                }
            }
#pragma unroll
            for (int token_local = 0; token_local < NormTokenCapacity; ++token_local) {
                sums[token_local] = common::warp_reduce_sum(sums[token_local]);
                if (lane == 0 && token_local < token_count) {
                    float* norm_partial =
                        partial + static_cast<std::int64_t>(SplitK) * t * kLogicalRows;
                    norm_partial[static_cast<std::int64_t>(split) * t + token0 + token_local] =
                        sums[token_local];
                }
            }
        }
    }

    float a_acc[kMFragments][kNFragments][4]                           = {};
    float b_acc[Paired ? kMFragments : 1][Paired ? kNFragments : 1][4] = {};

    auto stage_load = [&](int stage, int kt) {
        const int k0 = kt * kBlockK;

        // x is shared by all 16 head rows and by both projections.
        for (int vec = tid; vec < kBlockN * (kBlockK / 8); vec += kThreads) {
            const int token_local = vec / (kBlockK / 8);
            const int k_vec       = vec - token_local * (kBlockK / 8);
            const int kk          = k_vec * 8;
            const int token       = token0 + token_local;
            __nv_bfloat16* dst =
                &xs[stage * kBlockN * kBlockK + token_local * kBlockK + swizzle(token_local, kk)];
            if constexpr (NormalizeInput) {
                const bool valid = FullTokens || token < t;
                if (valid) {
                    const auto* source = reinterpret_cast<const __nv_bfloat162*>(
                        &x[static_cast<std::int64_t>(token) * kHidden + k0 + kk]);
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
                common::cp_async<16, Cache::cg>(
                    dst, &x[static_cast<std::int64_t>(token) * kHidden + k0 + kk]);
            } else {
                const bool valid     = token < t;
                const int safe_token = valid ? token : 0;
                common::cp_async_zfill<16, Cache::cg>(
                    dst, &x[static_cast<std::int64_t>(safe_token) * kHidden + k0 + kk],
                    valid ? 16 : 0);
            }
        }

        // Each thread moves one 16-byte vector: 128 vectors from Wa and 128
        // from Wb in the paired form. The merged form reads one 16-row tile from A or B.
        // Contiguous weights remain BF16 all the way into ldmatrix.
        const int weight_vecs = kBlockM * (kBlockK / 8);
        for (int all_vec = tid; all_vec < (Paired ? 2 : 1) * weight_vecs; all_vec += kThreads) {
            const bool is_b    = Paired ? all_vec >= weight_vecs : head0 >= kHeads;
            const int vec      = Paired && is_b ? all_vec - weight_vecs : all_vec;
            const int row      = vec / (kBlockK / 8);
            const int k_vec    = vec - row * (kBlockK / 8);
            const int kk       = k_vec * 8;
            __nv_bfloat16* dst = (Paired && is_b ? bws : aws) + stage * kBlockM * kBlockK +
                                 row * kBlockK + swizzle(row, kk);
            const __nv_bfloat16* weight = is_b ? b_weight : a_weight;
            common::cp_async<16, Cache::cg>(
                dst, &weight[static_cast<std::int64_t>(weight_head0 + row) * kHidden + k0 + kk]);
        }
    };

#pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
        if (stage < kTilesPerSplit) { stage_load(stage, kt_begin + stage); }
        common::cp_commit();
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
        common::cp_wait<kStages - 1>();
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < kKSubtiles; ++ki) {
            unsigned xf[kNFragments][2];
#pragma unroll
            for (int ni = 0; ni < kNFragments; ++ni) {
                const int xrow = warp * kWarpN + ni * 8 + x_rin;
                const int xcol = ki * 16 + x_koff;
                common::ldmatrix_x2(
                    xf[ni][0], xf[ni][1],
                    common::smem_addr(
                        &xs[stage * kBlockN * kBlockK + xrow * kBlockK + swizzle(xrow, xcol)]));
            }
#pragma unroll
            for (int mi = 0; mi < kMFragments; ++mi) {
                unsigned af[4], bf[4];
                const int arow         = mi * 16 + a_rowoff;
                const int acol         = ki * 16 + a_coloff;
                const int weight_stage = stage * kBlockM * kBlockK;
                common::ldmatrix_x4(
                    af[0], af[1], af[2], af[3],
                    common::smem_addr(&aws[weight_stage + arow * kBlockK + swizzle(arow, acol)]));
                if constexpr (Paired) {
                    common::ldmatrix_x4(
                        bf[0], bf[1], bf[2], bf[3],
                        common::smem_addr(
                            &bws[weight_stage + arow * kBlockK + swizzle(arow, acol)]));
                }
#pragma unroll
                for (int ni = 0; ni < kNFragments; ++ni) {
                    common::mma_bf16(a_acc[mi][ni][0], a_acc[mi][ni][1], a_acc[mi][ni][2],
                                     a_acc[mi][ni][3], af[0], af[1], af[2], af[3], xf[ni][0],
                                     xf[ni][1]);
                    if constexpr (Paired) {
                        common::mma_bf16(b_acc[mi][ni][0], b_acc[mi][ni][1], b_acc[mi][ni][2],
                                         b_acc[mi][ni][3], bf[0], bf[1], bf[2], bf[3], xf[ni][0],
                                         xf[ni][1]);
                    }
                }
            }
        }

        __syncthreads();
        const int next = it + kStages;
        if (next < kTilesPerSplit) { stage_load(stage, kt_begin + next); }
        common::cp_commit();
    }

#pragma unroll
    for (int mi = 0; mi < kMFragments; ++mi) {
        const int row0 = head0 + mi * 16 + gid;
        const int row1 = row0 + 8;
#pragma unroll
        for (int ni = 0; ni < kNFragments; ++ni) {
            const int col0 = token0 + warp * kWarpN + ni * 8 + 2 * lid;
            const int col1 = col0 + 1;
            auto store     = [&](int token, int row, float av, float bv) {
                if constexpr (SplitK == 1) {
                    const std::int64_t out_i =
                        static_cast<std::int64_t>(token) * kHeads + (Paired ? row : row % kHeads);
                    if constexpr (Paired) {
                        g[out_i]    = -expf(A_log[row]) * common::softplus(av + dt_bias[row]);
                        beta[out_i] = common::sigmoid(bv);
                    } else {
                        if (row >= kHeads)
                            beta[out_i] = common::sigmoid(av);
                        else
                            g[out_i] = -expf(A_log[row]) * common::softplus(av + dt_bias[row]);
                    }
                } else {
                    const std::int64_t base =
                        (static_cast<std::int64_t>(split) * t + token) * kLogicalRows;
                    partial[base + row] = av;
                    if constexpr (Paired) partial[base + kHeads + row] = bv;
                }
            };
            if constexpr (FullTokens) {
                store(col0, row0, a_acc[mi][ni][0], b_acc[Paired ? mi : 0][Paired ? ni : 0][0]);
                store(col1, row0, a_acc[mi][ni][1], b_acc[Paired ? mi : 0][Paired ? ni : 0][1]);
                store(col0, row1, a_acc[mi][ni][2], b_acc[Paired ? mi : 0][Paired ? ni : 0][2]);
                store(col1, row1, a_acc[mi][ni][3], b_acc[Paired ? mi : 0][Paired ? ni : 0][3]);
            } else {
                if (col0 < t) {
                    store(col0, row0, a_acc[mi][ni][0], b_acc[Paired ? mi : 0][Paired ? ni : 0][0]);
                    store(col0, row1, a_acc[mi][ni][2], b_acc[Paired ? mi : 0][Paired ? ni : 0][2]);
                }
                if (col1 < t) {
                    store(col1, row0, a_acc[mi][ni][1], b_acc[Paired ? mi : 0][Paired ? ni : 0][1]);
                    store(col1, row1, a_acc[mi][ni][3], b_acc[Paired ? mi : 0][Paired ? ni : 0][3]);
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
        const int elems = kHeads * t;
        if constexpr (NormalizeInput) {
            const float* norm_partial =
                partial + static_cast<std::int64_t>(SplitK) * t * kLogicalRows;
            const int hidden_elems = kHidden * t;
            for (int i = block_linear * kThreads + tid; i < hidden_elems; i += grid_threads) {
                const int k     = i % kHidden;
                const int token = i / kHidden;
                float sum       = 0.0F;
#pragma unroll
                for (int s = 0; s < SplitK; ++s) {
                    sum += norm_partial[static_cast<std::int64_t>(s) * t + token];
                }
                const float inv    = rsqrtf(sum / static_cast<float>(kHidden) + norm_eps);
                const float value  = __bfloat162float(x[i]);
                const float weight = __bfloat162float(norm_weight[k]);
                normalized_x[i]    = __float2bfloat16_rn(value * inv * (1.0F + weight));
            }
        }
        for (int i = block_linear * kThreads + tid; i < elems; i += grid_threads) {
            const int row   = i % kHeads;
            const int token = i / kHeads;
            float av        = 0.0f;
            float bv        = 0.0f;
#pragma unroll
            for (int s = 0; s < SplitK; ++s) {
                const std::int64_t base = (static_cast<std::int64_t>(s) * t + token) * kLogicalRows;
                av += partial[base + row];
                bv += partial[base + kHeads + row];
            }
            if constexpr (NormalizeInput) {
                const float* norm_partial =
                    partial + static_cast<std::int64_t>(SplitK) * t * kLogicalRows;
                float sum = 0.0F;
#pragma unroll
                for (int s = 0; s < SplitK; ++s) {
                    sum += norm_partial[static_cast<std::int64_t>(s) * t + token];
                }
                const float inv = rsqrtf(sum / static_cast<float>(kHidden) + norm_eps);
                av *= inv;
                bv *= inv;
            }
            g[i]    = -expf(A_log[row]) * common::softplus(av + dt_bias[row]);
            beta[i] = common::sigmoid(bv);
        }
    }
}

} // namespace ninfer::ops::detail::gdn_control
