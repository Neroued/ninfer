#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"
#include "ops/sparse_moe/sparse_moe_route.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden       = 2048;
constexpr int kExperts      = 256;
constexpr int kRouterRows   = 257;
constexpr int kTopK         = 8;
constexpr int kIntermediate = 512;

constexpr int kRouterBM      = 16;
constexpr int kRouterBN      = 64;
constexpr int kRouterBK      = 64;
constexpr int kRouterStages  = 2;
constexpr int kRouterWarps   = 8;
constexpr int kRouterThreads = 32 * kRouterWarps;

__global__ __launch_bounds__(kRouterThreads, 2) void sparse_moe_prefill_router_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ scores, int tokens) {
    __shared__ __align__(16) __nv_bfloat16 Ws[kRouterStages][kRouterBM * kRouterBK];
    __shared__ __align__(16) __nv_bfloat16 Xs[kRouterStages][kRouterBN * kRouterBK];

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int row0   = static_cast<int>(blockIdx.x) * kRouterBM;
    const int token0 = static_cast<int>(blockIdx.y) * kRouterBN;

    float acc[4] = {};

    auto stage_inputs = [&](int stage, int kt) {
        const int k0 = kt * kRouterBK;
        for (int item = tid; item < kRouterBM * (kRouterBK / 8); item += kRouterThreads) {
            const int row   = item / (kRouterBK / 8);
            const int k8    = item - row * (kRouterBK / 8);
            const int rr    = row0 + row;
            auto* dst       = &Ws[stage][row * kRouterBK + gemm_swz64(row, k8 * 8)];
            const auto* src = weight +
                              static_cast<std::int64_t>(rr < kRouterRows ? rr : 0) * kHidden + k0 +
                              k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, rr < kRouterRows ? 16 : 0);
        }
        for (int item = tid; item < kRouterBN * (kRouterBK / 8); item += kRouterThreads) {
            const int col = item / (kRouterBK / 8);
            const int k8  = item - col * (kRouterBK / 8);
            const int tt  = token0 + col;
            auto* dst     = &Xs[stage][col * kRouterBK + gemm_swz64(col, k8 * 8)];
            const auto* src =
                x + static_cast<std::int64_t>(tt < tokens ? tt : 0) * kHidden + k0 + k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, tt < tokens ? 16 : 0);
        }
    };

#pragma unroll
    for (int stage = 0; stage < kRouterStages; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

#pragma unroll 1
    for (int kt = 0; kt < kHidden / kRouterBK; ++kt) {
        const int stage = kt & 1;
        cp_wait<kRouterStages - 1>();
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < kRouterBK / 16; ++ki) {
            unsigned af[4];
            unsigned bf[2];
            const int arow = a_rowoff;
            const int acol = ki * 16 + a_coloff;
            const int brow = warp * 8 + b_rin;
            const int bcol = ki * 16 + b_koff;
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(&Ws[stage][arow * kRouterBK + gemm_swz64(arow, acol)]));
            ldmatrix_x2(bf[0], bf[1],
                        smem_addr(&Xs[stage][brow * kRouterBK + gemm_swz64(brow, bcol)]));
            mma_bf16(acc[0], acc[1], acc[2], acc[3], af[0], af[1], af[2], af[3], bf[0], bf[1]);
        }

        __syncthreads();
        const int next = kt + kRouterStages;
        if (next < kHidden / kRouterBK) { stage_inputs(stage, next); }
        cp_commit();
    }
    cp_wait<0>();

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int r0  = row0 + gid;
    const int r1  = r0 + 8;
    const int c0  = token0 + warp * 8 + 2 * lid;
    const int c1  = c0 + 1;
    if (r0 < kRouterRows && c0 < tokens) {
        scores[static_cast<std::int64_t>(c0) * kSparseMoeRouterScoreRows + r0] = acc[0];
    }
    if (r0 < kRouterRows && c1 < tokens) {
        scores[static_cast<std::int64_t>(c1) * kSparseMoeRouterScoreRows + r0] = acc[1];
    }
    if (r1 < kRouterRows && c0 < tokens) {
        scores[static_cast<std::int64_t>(c0) * kSparseMoeRouterScoreRows + r1] = acc[2];
    }
    if (r1 < kRouterRows && c1 < tokens) {
        scores[static_cast<std::int64_t>(c1) * kSparseMoeRouterScoreRows + r1] = acc[3];
    }
}

__global__ void sparse_moe_prefill_select_count_kernel(const float* __restrict__ scores,
                                                       int* __restrict__ ids,
                                                       float* __restrict__ alpha,
                                                       float* __restrict__ shared_scale,
                                                       int* __restrict__ local_rank,
                                                       int* __restrict__ tile_counts, int tokens) {
    __shared__ int counts[kExperts];
    __shared__ float selected_logits[kSparseMoeRouteTileTokens][kTopK];
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid < kExperts) { counts[tid] = 0; }
    __syncthreads();

    const int token = static_cast<int>(blockIdx.x) * kSparseMoeRouteTileTokens + warp;
    if (token < tokens) {
        sparse_moe_select_top8_warp(scores + static_cast<std::int64_t>(token) *
                                                 kSparseMoeRouterScoreRows,
                                    ids + token * kTopK, alpha + token * kTopK,
                                    shared_scale + token, selected_logits[warp]);
        __syncwarp();
        if (lane == 0) {
#pragma unroll
            for (int rank = 0; rank < kTopK; ++rank) {
                const int assignment   = token * kTopK + rank;
                local_rank[assignment] = atomicAdd(&counts[ids[assignment]], 1);
            }
        }
    }
    __syncthreads();
    if (tid < kExperts) {
        tile_counts[static_cast<std::int64_t>(blockIdx.x) * kExperts + tid] = counts[tid];
    }
}

__global__ void sparse_moe_prefill_scan_kernel(const int* __restrict__ tile_counts,
                                               int* __restrict__ tile_bases,
                                               int* __restrict__ expert_offsets,
                                               int* __restrict__ route_job_experts,
                                               int* __restrict__ route_job_columns,
                                               int* __restrict__ route_job_count, int route_tiles,
                                               int job_bn) {
    __shared__ int scan[kExperts];
    const int expert = static_cast<int>(threadIdx.x);
    int count        = 0;
    for (int tile = 0; tile < route_tiles; ++tile) {
        count += tile_counts[static_cast<std::int64_t>(tile) * kExperts + expert];
    }
    scan[expert] = count;
    __syncthreads();

#pragma unroll
    for (int offset = 1; offset < kExperts; offset <<= 1) {
        const int add = expert >= offset ? scan[expert - offset] : 0;
        __syncthreads();
        scan[expert] += add;
        __syncthreads();
    }

    const int base         = expert == 0 ? 0 : scan[expert - 1];
    expert_offsets[expert] = base;
    if (expert == kExperts - 1) { expert_offsets[kExperts] = scan[expert]; }

    int cursor = base;
    for (int tile = 0; tile < route_tiles; ++tile) {
        const std::int64_t index = static_cast<std::int64_t>(tile) * kExperts + expert;
        tile_bases[index]        = cursor;
        cursor += tile_counts[index];
    }

    scan[expert] = (count + job_bn - 1) / job_bn;
    __syncthreads();
#pragma unroll
    for (int offset = 1; offset < kExperts; offset <<= 1) {
        const int add = expert >= offset ? scan[expert - offset] : 0;
        __syncthreads();
        scan[expert] += add;
        __syncthreads();
    }
    const int job_begin = expert == 0 ? 0 : scan[expert - 1];
    const int jobs      = (count + job_bn - 1) / job_bn;
    for (int job = 0; job < jobs; ++job) {
        route_job_experts[job_begin + job] = expert;
        route_job_columns[job_begin + job] = job * job_bn;
    }
    if (expert == kExperts - 1) { *route_job_count = scan[expert]; }
}

__global__ void sparse_moe_prefill_gather_kernel(const __nv_bfloat16* __restrict__ x,
                                                 const int* __restrict__ ids,
                                                 int* __restrict__ packed_index,
                                                 const int* __restrict__ tile_bases,
                                                 __nv_bfloat16* __restrict__ gathered) {
    const int assignment = static_cast<int>(blockIdx.x);
    const int token      = assignment / kTopK;
    const int expert     = ids[assignment];
    const int tile       = token / kSparseMoeRouteTileTokens;
    const int packed =
        tile_bases[static_cast<std::int64_t>(tile) * kExperts + expert] + packed_index[assignment];
    const int k       = static_cast<int>(threadIdx.x) * 8;
    const uint4 value = load_vec<uint4>(x + static_cast<std::int64_t>(token) * kHidden + k);
    store_vec(gathered + static_cast<std::int64_t>(packed) * kHidden + k, value);
    if (threadIdx.x == 0) { packed_index[assignment] = packed; }
}

constexpr int kExpertBM                = 64;
constexpr int kExpertBN                = 64;
constexpr int kExpertBK                = 64;
constexpr int kExpertStages            = 2;
constexpr int kExpertWarps             = 8;
constexpr int kExpertThreads           = 32 * kExpertWarps;
constexpr int kRtx5090SmCount          = 170;
constexpr int kPrefillBlocksPerSm      = 3;
constexpr int kPrefillPersistentBlocks = kPrefillBlocksPerSm * kRtx5090SmCount;

template <int ExpertWarps, int ExpertBN>
__global__ __launch_bounds__(ExpertWarps * 32, 3) void sparse_moe_prefill_q4_gate_up_kernel(
    const __nv_bfloat16* __restrict__ gathered, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ route_job_count, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ activation) {
    constexpr int ExpertThreads = ExpertWarps * 32;
    constexpr int GroupsPerRow  = kHidden / 64;
    constexpr int WarpCols      = ExpertBN / ExpertWarps;
    constexpr int WarpNT        = WarpCols / 8;
    static_assert(ExpertBN % ExpertWarps == 0 && WarpCols % 8 == 0);
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][ExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertStages][kExpertBM * 32];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * GroupsPerRow * 2];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int a_mat          = lane >> 3;
    const int a_rin          = lane & 7;
    const int a_rowoff       = a_rin + ((a_mat & 1) << 3);
    const int a_coloff       = (a_mat >> 1) << 3;
    const int b_rin          = lane & 7;
    const int b_koff         = ((lane >> 3) & 1) << 3;
    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    constexpr int row_blocks = kIntermediate / (kExpertBM / 2);
    const int total_work     = *route_job_count * row_blocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job     = work / row_blocks;
        const int row_block     = work - route_job * row_blocks;
        const int expert        = route_job_experts[route_job];
        const int logical0      = row_block * (kExpertBM / 2);
        const int begin         = expert_offsets[expert];
        const int count         = expert_offsets[expert + 1] - begin;
        const int column_base   = route_job_columns[route_job];
        const int cols          = count - column_base < ExpertBN ? count - column_base : ExpertBN;
        float acc[4][WarpNT][4] = {};

        auto global_row = [&](int local_row) {
            const int logical = logical0 + (local_row & (kExpertBM / 2 - 1));
            return expert * 1024 + logical + (local_row >= kExpertBM / 2 ? kIntermediate : 0);
        };

        auto stage_scales = [&] {
            constexpr int ScaleBytesPerRow = GroupsPerRow * 2;
            constexpr int ChunksPerRow     = ScaleBytesPerRow / 16;
            for (int item = tid; item < kExpertBM * ChunksPerRow; item += ExpertThreads) {
                const int row   = item / ChunksPerRow;
                const int chunk = item - row * ChunksPerRow;
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(row)) * GroupsPerRow + chunk * 8;
                cp_async<16, Cache::cg>(&Sr[row * ScaleBytesPerRow + chunk * 16], &scales[gi * 2]);
            }
        };

        auto stage_inputs = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < ExpertBN * (kExpertBK / 8); item += ExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int packed_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    gathered +
                    static_cast<std::int64_t>(col < cols ? packed_col : begin) * kHidden + k0 +
                    k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }

            const int group = kt;
            for (int item = tid; item < kExpertBM * 2; item += ExpertThreads) {
                const int row  = item >> 1;
                const int half = item & 1;
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(row)) * GroupsPerRow + group;
                cp_async<16, Cache::cg>(&Cr[stage][row * 32 + half * 16],
                                        &codes[gi * 32 + half * 16]);
            }
        };

        auto decode_weight = [&](int stage) {
            for (int row = warp; row < kExpertBM; row += ExpertWarps) {
                const std::uint8_t packed = Cr[stage][row * 32 + lane];
                const int q0              = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
                const int q1              = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
                const __nv_bfloat162 value =
                    __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
                store_vec(&As[row * kExpertBK + gemm_swz64(row, 2 * lane)], value);
            }
        };

        stage_scales();
        cp_commit();
#pragma unroll
        for (int stage = 0; stage < kExpertStages; ++stage) {
            stage_inputs(stage, stage);
            cp_commit();
        }

#pragma unroll 1
        for (int kt = 0; kt < kHidden / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<kExpertStages - 1>();
            __syncthreads();
            decode_weight(stage);
            __syncthreads();

            if (warp * WarpCols < cols) {
                float partial[4][WarpNT][4] = {};
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    unsigned af[4][4];
                    unsigned bf[WarpNT][2];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        const int brow = warp * WarpCols + ni * 8 + b_rin;
                        const int bcol = ki * 16 + b_koff;
                        ldmatrix_x2(
                            bf[ni][0], bf[ni][1],
                            smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
#pragma unroll
                        for (int mi = 0; mi < 4; ++mi) {
                            mma_bf16(partial[mi][ni][0], partial[mi][ni][1], partial[mi][ni][2],
                                     partial[mi][ni][3], af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                     bf[ni][0], bf[ni][1]);
                        }
                    }
                }
#pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    const int row0 = mi * 16 + gid;
                    const int row1 = row0 + 8;
                    float scale0 =
                        lid == 0
                            ? __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                                  &Sr[(row0 * GroupsPerRow + kt) * 2])))
                            : 0.0f;
                    float scale1 =
                        lid == 0
                            ? __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
                                  &Sr[(row1 * GroupsPerRow + kt) * 2])))
                            : 0.0f;
                    scale0 = __shfl_sync(0xffffffffu, scale0, gid * 4);
                    scale1 = __shfl_sync(0xffffffffu, scale1, gid * 4);
#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        acc[mi][ni][0] = fmaf(partial[mi][ni][0], scale0, acc[mi][ni][0]);
                        acc[mi][ni][1] = fmaf(partial[mi][ni][1], scale0, acc[mi][ni][1]);
                        acc[mi][ni][2] = fmaf(partial[mi][ni][2], scale1, acc[mi][ni][2]);
                        acc[mi][ni][3] = fmaf(partial[mi][ni][3], scale1, acc[mi][ni][3]);
                    }
                }
            }

            __syncthreads();
            const int next = kt + kExpertStages;
            if (next < kHidden / kExpertBK) { stage_inputs(stage, next); }
            cp_commit();
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * WarpCols < cols) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row0 = logical0 + mi * 16 + gid;
                const int row1 = row0 + 8;
#pragma unroll
                for (int ni = 0; ni < WarpNT; ++ni) {
                    const int col0       = begin + column_base + warp * WarpCols + ni * 8 + 2 * lid;
                    const int col1       = col0 + 1;
                    const int local_col0 = warp * WarpCols + ni * 8 + 2 * lid;
                    const int local_col1 = local_col0 + 1;
                    if (local_col0 < cols) {
                        activation[static_cast<std::int64_t>(col0) * kIntermediate + row0] =
                            __float2bfloat16_rn(silu(acc[mi][ni][0]) * acc[mi + 2][ni][0]);
                        activation[static_cast<std::int64_t>(col0) * kIntermediate + row1] =
                            __float2bfloat16_rn(silu(acc[mi][ni][2]) * acc[mi + 2][ni][2]);
                    }
                    if (local_col1 < cols) {
                        activation[static_cast<std::int64_t>(col1) * kIntermediate + row0] =
                            __float2bfloat16_rn(silu(acc[mi][ni][1]) * acc[mi + 2][ni][1]);
                        activation[static_cast<std::int64_t>(col1) * kIntermediate + row1] =
                            __float2bfloat16_rn(silu(acc[mi][ni][3]) * acc[mi + 2][ni][3]);
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <bool Routed>
__global__ __launch_bounds__(kExpertThreads, 1) void sparse_moe_prefill_w8_gate_up_kernel(
    const __nv_bfloat16* __restrict__ input, const int* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ activation, int tokens) {
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][kExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertBM * kExpertBK];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * 16];

    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int expert   = Routed ? static_cast<int>(blockIdx.y) : 0;
    const int logical0 = static_cast<int>(blockIdx.x) * (kExpertBM / 2);
    const int begin    = Routed ? expert_offsets[expert] : static_cast<int>(blockIdx.y) * kExpertBN;
    const int count    = Routed ? expert_offsets[expert + 1] - begin
                                : (tokens - begin < kExpertBN ? tokens - begin : kExpertBN);
    if (count <= 0) { return; }

    const int a_mat              = lane >> 3;
    const int a_rin              = lane & 7;
    const int a_rowoff           = a_rin + ((a_mat & 1) << 3);
    const int a_coloff           = (a_mat >> 1) << 3;
    const int b_rin              = lane & 7;
    const int b_koff             = ((lane >> 3) & 1) << 3;
    const int gid                = lane >> 2;
    const int lid                = lane & 3;
    constexpr int groups_per_row = kHidden / 32;

    const int column_iterations = Routed ? (count + kExpertBN - 1) / kExpertBN : 1;
    for (int iteration = 0; iteration < column_iterations; ++iteration) {
        const int column_base = iteration * kExpertBN;
        const int cols        = count - column_base < kExpertBN ? count - column_base : kExpertBN;
        float acc[4][4]       = {};

        auto global_row = [&](int local_row) {
            const int logical = logical0 + (local_row & (kExpertBM / 2 - 1));
            const int row     = logical + (local_row >= kExpertBM / 2 ? kIntermediate : 0);
            return (Routed ? expert * 1024 : 0) + row;
        };

        auto stage_x = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < kExpertBN * (kExpertBK / 8); item += kExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int source_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    input + static_cast<std::int64_t>(col < cols ? source_col : begin) * kHidden +
                    k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }
        };

        auto stage_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int group0                = kt * groups_per_tile;
            for (int item = tid; item < kExpertBM * (kExpertBK / 16); item += kExpertThreads) {
                const int row   = item / (kExpertBK / 16);
                const int chunk = item - row * (kExpertBK / 16);
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(row)) * groups_per_row + group0;
                cp_async<16, Cache::cg>(&Cr[row * kExpertBK + chunk * 16],
                                        &codes[gi * 32 + chunk * 16]);
            }
            if ((kt % scale_cache_tiles) == 0) {
                for (int row = tid; row < kExpertBM; row += kExpertThreads) {
                    const std::int64_t gi =
                        static_cast<std::int64_t>(global_row(row)) * groups_per_row + group0;
                    cp_async<16, Cache::cg>(&Sr[row * 16], &scales[gi * 2]);
                }
            }
        };

        auto decode_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int scale_offset          = (kt % scale_cache_tiles) * groups_per_tile * 2;
            const int half                  = lane >> 4;
            const int half_lane             = lane & 15;
            for (int row_pair = warp * 2; row_pair < kExpertBM; row_pair += kExpertWarps * 2) {
                const int row = row_pair + half;
                unsigned scale_pair =
                    half_lane == 0
                        ? *reinterpret_cast<const std::uint32_t*>(&Sr[row * 16 + scale_offset])
                        : 0;
                scale_pair = __shfl_sync(0xffffffffu, scale_pair, half * 16);
#pragma unroll
                for (int group = 0; group < groups_per_tile; ++group) {
                    const float scale = __half2float(__ushort_as_half(
                        static_cast<std::uint16_t>((scale_pair >> (group * 16)) & 0xffffu)));
                    const int col     = group * 32 + half_lane * 2;
                    const std::uint16_t packed =
                        *reinterpret_cast<const std::uint16_t*>(&Cr[row * kExpertBK + col]);
                    const int q0 = static_cast<int>(static_cast<std::int8_t>(packed & 0xffu));
                    const int q1 = static_cast<int>(static_cast<std::int8_t>(packed >> 8));
                    const __nv_bfloat162 value = __floats2bfloat162_rn(
                        static_cast<float>(q0) * scale, static_cast<float>(q1) * scale);
                    store_vec(&As[row * kExpertBK + gemm_swz64(row, col)], value);
                }
            }
        };

        stage_x(0, 0);
        stage_weight(0);
        cp_commit();

#pragma unroll 4
        for (int kt = 0; kt < kHidden / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<0>();
            __syncthreads();
            decode_weight(kt);
            __syncthreads();

            const int next = kt + 1;
            if (next < kHidden / kExpertBK) {
                stage_x(next & 1, next);
                stage_weight(next);
                cp_commit();
            }

            if (warp * 8 < cols) {
                unsigned af[2][4][4];
                unsigned bf[2][2];
                auto load_fragments = [&](int slot, int ki) {
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[slot][mi][0], af[slot][mi][1], af[slot][mi][2],
                                    af[slot][mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(bf[slot][0], bf[slot][1],
                                smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
                };
                load_fragments(0, 0);
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    const int slot = ki & 1;
                    if (ki + 1 < kExpertBK / 16) { load_fragments(slot ^ 1, ki + 1); }
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[slot][mi][0],
                                 af[slot][mi][1], af[slot][mi][2], af[slot][mi][3], bf[slot][0],
                                 bf[slot][1]);
                    }
                }
            }
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row0       = logical0 + mi * 16 + gid;
                const int row1       = row0 + 8;
                const int col0       = begin + column_base + warp * 8 + 2 * lid;
                const int col1       = col0 + 1;
                const int local_col0 = warp * 8 + 2 * lid;
                const int local_col1 = local_col0 + 1;
                if (local_col0 < cols) {
                    activation[static_cast<std::int64_t>(col0) * kIntermediate + row0] =
                        __float2bfloat16_rn(silu(acc[mi][0]) * acc[mi + 2][0]);
                    activation[static_cast<std::int64_t>(col0) * kIntermediate + row1] =
                        __float2bfloat16_rn(silu(acc[mi][2]) * acc[mi + 2][2]);
                }
                if (local_col1 < cols) {
                    activation[static_cast<std::int64_t>(col1) * kIntermediate + row0] =
                        __float2bfloat16_rn(silu(acc[mi][1]) * acc[mi + 2][1]);
                    activation[static_cast<std::int64_t>(col1) * kIntermediate + row1] =
                        __float2bfloat16_rn(silu(acc[mi][3]) * acc[mi + 2][3]);
                }
            }
        }
        __syncthreads();
    }
}

struct Q5DownMma {
    static constexpr int kHighBytes = Q5RowSplitStorage::kHighBytesPerGroup;

    __device__ static __forceinline__ __nv_bfloat162 decode(const std::uint8_t* codes,
                                                            const std::uint8_t* high,
                                                            const std::uint8_t* scale, int row,
                                                            int lane) {
        return Q5MmaDecodeAtom::decode_pair(codes, high, scale, row, lane);
    }
};

struct Q6DownMma {
    static constexpr int kHighBytes = Q6RowSplitStorage::kHighBytesPerGroup;

    __device__ static __forceinline__ __nv_bfloat162 decode(const std::uint8_t* codes,
                                                            const std::uint8_t* high,
                                                            const std::uint8_t* scale, int row,
                                                            int lane) {
        return Q6MmaDecodeAtom::decode_pair(codes, high, scale, row, lane);
    }
};

template <class Codec, int ExpertWarps, int ExpertBN>
__global__ __launch_bounds__(ExpertWarps * 32, 3) void sparse_moe_prefill_qx_down_kernel(
    const __nv_bfloat16* __restrict__ activation, const int* __restrict__ expert_offsets,
    const int* __restrict__ route_job_experts, const int* __restrict__ route_job_columns,
    const int* __restrict__ route_job_count, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ output) {
    constexpr int ExpertThreads = ExpertWarps * 32;
    constexpr int GroupsPerRow  = kIntermediate / 64;
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][ExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertStages][kExpertBM * 32];
    __shared__ __align__(16) std::uint8_t Hr[kExpertStages][kExpertBM * Codec::kHighBytes];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * GroupsPerRow * 2];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int a_mat          = lane >> 3;
    const int a_rin          = lane & 7;
    const int a_rowoff       = a_rin + ((a_mat & 1) << 3);
    const int a_coloff       = (a_mat >> 1) << 3;
    const int b_rin          = lane & 7;
    const int b_koff         = ((lane >> 3) & 1) << 3;
    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    constexpr int row_blocks = kHidden / kExpertBM;
    const int total_work     = *route_job_count * row_blocks;
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job   = work / row_blocks;
        const int row_block   = work - route_job * row_blocks;
        const int expert      = route_job_experts[route_job];
        const int row0        = row_block * kExpertBM;
        const int begin       = expert_offsets[expert];
        const int count       = expert_offsets[expert + 1] - begin;
        const int column_base = route_job_columns[route_job];
        const int cols        = count - column_base < ExpertBN ? count - column_base : ExpertBN;
        float acc[4][4]       = {};

        auto stage_scales = [&] {
            for (int row = tid; row < kExpertBM; row += ExpertThreads) {
                const int global_row  = expert * kHidden + row0 + row;
                const std::int64_t gi = static_cast<std::int64_t>(global_row) * GroupsPerRow;
                cp_async<16, Cache::cg>(&Sr[row * GroupsPerRow * 2], &scales[gi * 2]);
            }
        };

        auto stage_inputs = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < ExpertBN * (kExpertBK / 8); item += ExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int packed_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    activation +
                    static_cast<std::int64_t>(col < cols ? packed_col : begin) * kIntermediate +
                    k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }

            for (int item = tid; item < kExpertBM * 2; item += ExpertThreads) {
                const int row         = item >> 1;
                const int half        = item & 1;
                const int global_row  = expert * kHidden + row0 + row;
                const std::int64_t gi = static_cast<std::int64_t>(global_row) * GroupsPerRow + kt;
                cp_async<16, Cache::cg>(&Cr[stage][row * 32 + half * 16],
                                        &codes[gi * 32 + half * 16]);
            }
            for (int row = tid; row < kExpertBM; row += ExpertThreads) {
                const int global_row  = expert * kHidden + row0 + row;
                const std::int64_t gi = static_cast<std::int64_t>(global_row) * GroupsPerRow + kt;
                if constexpr (Codec::kHighBytes == 8) {
                    cp_async<8>(&Hr[stage][row * Codec::kHighBytes], &high[gi * Codec::kHighBytes]);
                } else {
                    cp_async<16, Cache::cg>(&Hr[stage][row * Codec::kHighBytes],
                                            &high[gi * Codec::kHighBytes]);
                }
            }
        };

        auto decode_weight = [&](int stage, int kt) {
            for (int row = warp; row < kExpertBM; row += ExpertWarps) {
                const __nv_bfloat162 value = Codec::decode(
                    Cr[stage], Hr[stage], &Sr[(row * GroupsPerRow + kt) * 2], row, lane);
                store_vec(&As[row * kExpertBK + gemm_swz64(row, 2 * lane)], value);
            }
        };

        stage_scales();
        cp_commit();
#pragma unroll
        for (int stage = 0; stage < kExpertStages; ++stage) {
            stage_inputs(stage, stage);
            cp_commit();
        }

#pragma unroll
        for (int kt = 0; kt < kIntermediate / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<kExpertStages - 1>();
            __syncthreads();
            decode_weight(stage, kt);
            __syncthreads();

            if (warp * 8 < cols) {
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    unsigned af[4][4];
                    unsigned bf[2];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[mi][0],
                                 af[mi][1], af[mi][2], af[mi][3], bf[0], bf[1]);
                    }
                }
            }

            __syncthreads();
            const int next = kt + kExpertStages;
            if (next < kIntermediate / kExpertBK) { stage_inputs(stage, next); }
            cp_commit();
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int output_row0 = row0 + mi * 16 + gid;
                const int output_row1 = output_row0 + 8;
                const int col0        = begin + column_base + warp * 8 + 2 * lid;
                const int col1        = col0 + 1;
                const int local_col0  = warp * 8 + 2 * lid;
                const int local_col1  = local_col0 + 1;
                if (local_col0 < cols) {
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][0]);
                    output[static_cast<std::int64_t>(col0) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][2]);
                }
                if (local_col1 < cols) {
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row0] =
                        __float2bfloat16_rn(acc[mi][1]);
                    output[static_cast<std::int64_t>(col1) * kHidden + output_row1] =
                        __float2bfloat16_rn(acc[mi][3]);
                }
            }
        }
        __syncthreads();
    }
}

template <bool Routed>
__global__ __launch_bounds__(kExpertThreads, 1) void sparse_moe_prefill_w8_down_kernel(
    const __nv_bfloat16* __restrict__ input, const int* __restrict__ expert_offsets,
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ grouped_output, const float* __restrict__ routed_sum,
    const float* __restrict__ shared_scale, __nv_bfloat16* __restrict__ destination, int tokens) {
    __shared__ __align__(16) __nv_bfloat16 As[kExpertBM * kExpertBK];
    __shared__ __align__(16) __nv_bfloat16 Bs[kExpertStages][kExpertBN * kExpertBK];
    __shared__ __align__(16) std::uint8_t Cr[kExpertBM * kExpertBK];
    __shared__ __align__(16) std::uint8_t Sr[kExpertBM * 16];

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int expert = Routed ? static_cast<int>(blockIdx.y) : 0;
    const int row0   = static_cast<int>(blockIdx.x) * kExpertBM;
    const int begin  = Routed ? expert_offsets[expert] : static_cast<int>(blockIdx.y) * kExpertBN;
    const int count  = Routed ? expert_offsets[expert + 1] - begin
                              : (tokens - begin < kExpertBN ? tokens - begin : kExpertBN);
    if (count <= 0) { return; }

    const int a_mat              = lane >> 3;
    const int a_rin              = lane & 7;
    const int a_rowoff           = a_rin + ((a_mat & 1) << 3);
    const int a_coloff           = (a_mat >> 1) << 3;
    const int b_rin              = lane & 7;
    const int b_koff             = ((lane >> 3) & 1) << 3;
    const int gid                = lane >> 2;
    const int lid                = lane & 3;
    constexpr int groups_per_row = kIntermediate / 32;

    const int column_iterations = Routed ? (count + kExpertBN - 1) / kExpertBN : 1;
    for (int iteration = 0; iteration < column_iterations; ++iteration) {
        const int column_base = iteration * kExpertBN;
        const int cols        = count - column_base < kExpertBN ? count - column_base : kExpertBN;
        float acc[4][4]       = {};

        auto stage_x = [&](int stage, int kt) {
            const int k0 = kt * kExpertBK;
            for (int item = tid; item < kExpertBN * (kExpertBK / 8); item += kExpertThreads) {
                const int col        = item / (kExpertBK / 8);
                const int k8         = item - col * (kExpertBK / 8);
                const int source_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * kExpertBK + gemm_swz64(col, k8 * 8)];
                const auto* src =
                    input +
                    static_cast<std::int64_t>(col < cols ? source_col : begin) * kIntermediate +
                    k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }
        };

        auto stage_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int group0                = kt * groups_per_tile;
            for (int item = tid; item < kExpertBM * (kExpertBK / 16); item += kExpertThreads) {
                const int row        = item / (kExpertBK / 16);
                const int chunk      = item - row * (kExpertBK / 16);
                const int global_row = (Routed ? expert * kHidden : 0) + row0 + row;
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row) * groups_per_row + group0;
                cp_async<16, Cache::cg>(&Cr[row * kExpertBK + chunk * 16],
                                        &codes[gi * 32 + chunk * 16]);
            }
            if ((kt % scale_cache_tiles) == 0) {
                for (int row = tid; row < kExpertBM; row += kExpertThreads) {
                    const int global_row = (Routed ? expert * kHidden : 0) + row0 + row;
                    const std::int64_t gi =
                        static_cast<std::int64_t>(global_row) * groups_per_row + group0;
                    cp_async<16, Cache::cg>(&Sr[row * 16], &scales[gi * 2]);
                }
            }
        };

        auto decode_weight = [&](int kt) {
            constexpr int groups_per_tile   = kExpertBK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int scale_offset          = (kt % scale_cache_tiles) * groups_per_tile * 2;
            const int half                  = lane >> 4;
            const int half_lane             = lane & 15;
            for (int row_pair = warp * 2; row_pair < kExpertBM; row_pair += kExpertWarps * 2) {
                const int row = row_pair + half;
                unsigned scale_pair =
                    half_lane == 0
                        ? *reinterpret_cast<const std::uint32_t*>(&Sr[row * 16 + scale_offset])
                        : 0;
                scale_pair = __shfl_sync(0xffffffffu, scale_pair, half * 16);
#pragma unroll
                for (int group = 0; group < groups_per_tile; ++group) {
                    const float scale = __half2float(__ushort_as_half(
                        static_cast<std::uint16_t>((scale_pair >> (group * 16)) & 0xffffu)));
                    const int col     = group * 32 + half_lane * 2;
                    const std::uint16_t packed =
                        *reinterpret_cast<const std::uint16_t*>(&Cr[row * kExpertBK + col]);
                    const int q0 = static_cast<int>(static_cast<std::int8_t>(packed & 0xffu));
                    const int q1 = static_cast<int>(static_cast<std::int8_t>(packed >> 8));
                    const __nv_bfloat162 value = __floats2bfloat162_rn(
                        static_cast<float>(q0) * scale, static_cast<float>(q1) * scale);
                    store_vec(&As[row * kExpertBK + gemm_swz64(row, col)], value);
                }
            }
        };

        stage_x(0, 0);
        stage_weight(0);
        cp_commit();

#pragma unroll
        for (int kt = 0; kt < kIntermediate / kExpertBK; ++kt) {
            const int stage = kt & 1;
            cp_wait<0>();
            __syncthreads();
            decode_weight(kt);
            __syncthreads();

            const int next = kt + 1;
            if (next < kIntermediate / kExpertBK) {
                stage_x(next & 1, next);
                stage_weight(next);
                cp_commit();
            }

            if (warp * 8 < cols) {
                unsigned af[2][4][4];
                unsigned bf[2][2];
                auto load_fragments = [&](int slot, int ki) {
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const int row = mi * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[slot][mi][0], af[slot][mi][1], af[slot][mi][2],
                                    af[slot][mi][3],
                                    smem_addr(&As[row * kExpertBK + gemm_swz64(row, col)]));
                    }
                    const int brow = warp * 8 + b_rin;
                    const int bcol = ki * 16 + b_koff;
                    ldmatrix_x2(bf[slot][0], bf[slot][1],
                                smem_addr(&Bs[stage][brow * kExpertBK + gemm_swz64(brow, bcol)]));
                };
                load_fragments(0, 0);
#pragma unroll
                for (int ki = 0; ki < kExpertBK / 16; ++ki) {
                    const int slot = ki & 1;
                    if (ki + 1 < kExpertBK / 16) { load_fragments(slot ^ 1, ki + 1); }
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        mma_bf16(acc[mi][0], acc[mi][1], acc[mi][2], acc[mi][3], af[slot][mi][0],
                                 af[slot][mi][1], af[slot][mi][2], af[slot][mi][3], bf[slot][0],
                                 bf[slot][1]);
                    }
                }
            }
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int output_row0 = row0 + mi * 16 + gid;
                const int output_row1 = output_row0 + 8;
                const int col0        = begin + column_base + warp * 8 + 2 * lid;
                const int col1        = col0 + 1;
                const int local_col0  = warp * 8 + 2 * lid;
                const int local_col1  = local_col0 + 1;
                auto store            = [&](int col, int row, float value) {
                    if constexpr (Routed) {
                        grouped_output[static_cast<std::int64_t>(col) * kHidden + row] =
                            __float2bfloat16_rn(value);
                    } else {
                        const float merged =
                            __bfloat162float(
                                destination[static_cast<std::int64_t>(col) * kHidden + row]) +
                            routed_sum[static_cast<std::int64_t>(col) * kHidden + row] +
                            shared_scale[col] * value;
                        destination[static_cast<std::int64_t>(col) * kHidden + row] =
                            __float2bfloat16_rn(merged);
                    }
                };
                if (local_col0 < cols) {
                    store(col0, output_row0, acc[mi][0]);
                    store(col0, output_row1, acc[mi][2]);
                }
                if (local_col1 < cols) {
                    store(col1, output_row0, acc[mi][1]);
                    store(col1, output_row1, acc[mi][3]);
                }
            }
        }
        __syncthreads();
    }
}

union alignas(16) SparseMoeBf16x8 {
    uint4 raw;
    __nv_bfloat162 pair[4];
};

__global__ void sparse_moe_prefill_reduce_kernel(const __nv_bfloat16* __restrict__ grouped_output,
                                                 const int* __restrict__ packed_index,
                                                 const float* __restrict__ alpha,
                                                 float* __restrict__ routed_sum) {
    __shared__ int columns[kTopK];
    __shared__ float weights[kTopK];
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    if (tid < kTopK) {
        columns[tid] = packed_index[token * kTopK + tid];
        weights[tid] = alpha[token * kTopK + tid];
    }
    __syncthreads();

    float values[8] = {};
#pragma unroll
    for (int route = 0; route < kTopK; ++route) {
        SparseMoeBf16x8 packed;
        packed.raw = load_vec<uint4>(grouped_output +
                                     static_cast<std::int64_t>(columns[route]) * kHidden + tid * 8);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 decoded = __bfloat1622float2(packed.pair[pair]);
            values[2 * pair] += weights[route] * decoded.x;
            values[2 * pair + 1] += weights[route] * decoded.y;
        }
    }
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        routed_sum[static_cast<std::int64_t>(token) * kHidden + tid * 8 + item] = values[item];
    }
}

} // namespace

void sparse_moe_prefill_launch(const Tensor& x, const SparseMoeWeights& weights,
                               Tensor& destination, const SparseMoePrefillPlan& plan,
                               const SparseMoePrefillWorkspace& workspace, cudaStream_t stream) {
    if (x.ne[1] != plan.tokens || destination.ne[1] != plan.tokens || plan.slice_tokens < 1) {
        throw std::invalid_argument("sparse_moe prefill: launch plan does not match tensors");
    }

    const auto* router = static_cast<const __nv_bfloat16*>(weights.router_shared_gate.qdata);
    const auto* routed_gate_codes = static_cast<const std::uint8_t*>(weights.routed_gate_up.qdata);
    const auto* routed_gate_scales =
        static_cast<const std::uint8_t*>(weights.routed_gate_up.scales);
    const auto* routed_down_codes  = static_cast<const std::uint8_t*>(weights.routed_down.qdata);
    const auto* routed_down_high   = static_cast<const std::uint8_t*>(weights.routed_down.qhigh);
    const auto* routed_down_scales = static_cast<const std::uint8_t*>(weights.routed_down.scales);
    const auto* shared_gate_codes  = static_cast<const std::uint8_t*>(weights.shared_gate_up.qdata);
    const auto* shared_gate_scales =
        static_cast<const std::uint8_t*>(weights.shared_gate_up.scales);
    const auto* shared_down_codes  = static_cast<const std::uint8_t*>(weights.shared_down.qdata);
    const auto* shared_down_scales = static_cast<const std::uint8_t*>(weights.shared_down.scales);

    auto* ids               = static_cast<int*>(workspace.token_ids.data);
    auto* alpha             = static_cast<float*>(workspace.token_alpha.data);
    auto* packed_index      = static_cast<int*>(workspace.packed_index.data);
    auto* shared_scale      = static_cast<float*>(workspace.shared_scale.data);
    auto* tile_counts       = static_cast<int*>(workspace.tile_counts.data);
    auto* tile_bases        = static_cast<int*>(workspace.tile_bases.data);
    auto* offsets           = static_cast<int*>(workspace.expert_offsets.data);
    auto* route_job_experts = static_cast<int*>(workspace.route_job_experts.data);
    auto* route_job_columns = static_cast<int*>(workspace.route_job_columns.data);
    auto* route_job_count   = static_cast<int*>(workspace.route_job_count.data);
    auto* scores            = static_cast<float*>(workspace.score_storage.data);
    auto* shared_activation = static_cast<__nv_bfloat16*>(workspace.shared_activation.data);
    auto* grouped_io        = static_cast<__nv_bfloat16*>(workspace.grouped_io.data);
    auto* routed_activation = static_cast<__nv_bfloat16*>(workspace.routed_storage.data);
    auto* routed_sum        = static_cast<float*>(workspace.routed_sum.data);

    for (std::int32_t token0 = 0; token0 < plan.tokens; token0 += plan.slice_tokens) {
        const std::int32_t tokens =
            std::min(plan.slice_tokens, static_cast<std::int32_t>(plan.tokens - token0));
        const Tensor input_slice = x.slice(1, token0, tokens);
        Tensor output_slice      = destination.slice(1, token0, tokens);
        const auto* input        = static_cast<const __nv_bfloat16*>(input_slice.data);
        auto* output             = static_cast<__nv_bfloat16*>(output_slice.data);
        const int route_tiles =
            (tokens + kSparseMoeRouteTileTokens - 1) / kSparseMoeRouteTileTokens;
        const int assignments = tokens * kTopK;

        sparse_moe_prefill_router_mma_kernel<<<dim3((kRouterRows + kRouterBM - 1) / kRouterBM,
                                                    (tokens + kRouterBN - 1) / kRouterBN),
                                               kRouterThreads, 0, stream>>>(input, router, scores,
                                                                            tokens);
        CUDA_CHECK(cudaGetLastError());

        sparse_moe_prefill_select_count_kernel<<<route_tiles, kRouterThreads, 0, stream>>>(
            scores, ids, alpha, shared_scale, packed_index, tile_counts, tokens);
        CUDA_CHECK(cudaGetLastError());

        const bool wide_plan   = tokens >= kSparseMoePrefillWideMin;
        const int route_job_bn = wide_plan ? 64 : 32;
        sparse_moe_prefill_scan_kernel<<<1, kExpertThreads, 0, stream>>>(
            tile_counts, tile_bases, offsets, route_job_experts, route_job_columns, route_job_count,
            route_tiles, route_job_bn);
        CUDA_CHECK(cudaGetLastError());

        sparse_moe_prefill_gather_kernel<<<assignments, kExpertThreads, 0, stream>>>(
            input, ids, packed_index, tile_bases, grouped_io);
        CUDA_CHECK(cudaGetLastError());

        const dim3 routed_gate_grid(kIntermediate / (kExpertBM / 2), kExperts);
        if (weights.routed_gate_up.qtype == QType::Q4G64_F16S) {
            if (wide_plan) {
                sparse_moe_prefill_q4_gate_up_kernel<8, 64>
                    <<<kPrefillPersistentBlocks, 8 * 32, 0, stream>>>(
                        grouped_io, offsets, route_job_experts, route_job_columns, route_job_count,
                        routed_gate_codes, routed_gate_scales, routed_activation);
            } else {
                sparse_moe_prefill_q4_gate_up_kernel<4, 32>
                    <<<kPrefillPersistentBlocks, 4 * 32, 0, stream>>>(
                        grouped_io, offsets, route_job_experts, route_job_columns, route_job_count,
                        routed_gate_codes, routed_gate_scales, routed_activation);
            }
        } else if (weights.routed_gate_up.qtype == QType::W8G32_F16S) {
            sparse_moe_prefill_w8_gate_up_kernel<true>
                <<<routed_gate_grid, kExpertThreads, 0, stream>>>(
                    grouped_io, offsets, routed_gate_codes, routed_gate_scales, routed_activation,
                    tokens);
        } else {
            throw std::invalid_argument("sparse_moe prefill: unsupported gate/up codec");
        }
        CUDA_CHECK(cudaGetLastError());

        sparse_moe_prefill_w8_gate_up_kernel<false>
            <<<dim3(kIntermediate / (kExpertBM / 2), (tokens + kExpertBN - 1) / kExpertBN),
               kExpertThreads, 0, stream>>>(input, nullptr, shared_gate_codes, shared_gate_scales,
                                            shared_activation, tokens);
        CUDA_CHECK(cudaGetLastError());

        const dim3 routed_down_grid(kHidden / kExpertBM, kExperts);
        switch (weights.routed_down.qtype) {
        case QType::Q5G64_F16S:
            if (wide_plan) {
                sparse_moe_prefill_qx_down_kernel<Q5DownMma, 8, 64>
                    <<<kPrefillPersistentBlocks, 8 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            } else {
                sparse_moe_prefill_qx_down_kernel<Q5DownMma, 4, 32>
                    <<<kPrefillPersistentBlocks, 4 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            }
            break;
        case QType::Q6G64_F16S:
            if (wide_plan) {
                sparse_moe_prefill_qx_down_kernel<Q6DownMma, 8, 64>
                    <<<kPrefillPersistentBlocks, 8 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            } else {
                sparse_moe_prefill_qx_down_kernel<Q6DownMma, 4, 32>
                    <<<kPrefillPersistentBlocks, 4 * 32, 0, stream>>>(
                        routed_activation, offsets, route_job_experts, route_job_columns,
                        route_job_count, routed_down_codes, routed_down_high, routed_down_scales,
                        grouped_io);
            }
            break;
        case QType::W8G32_F16S:
            sparse_moe_prefill_w8_down_kernel<true>
                <<<routed_down_grid, kExpertThreads, 0, stream>>>(
                    routed_activation, offsets, routed_down_codes, routed_down_scales, grouped_io,
                    nullptr, nullptr, nullptr, tokens);
            break;
        default:
            throw std::invalid_argument("sparse_moe prefill: unsupported down codec");
        }
        CUDA_CHECK(cudaGetLastError());

        sparse_moe_prefill_reduce_kernel<<<tokens, kExpertThreads, 0, stream>>>(
            grouped_io, packed_index, alpha, routed_sum);
        CUDA_CHECK(cudaGetLastError());

        sparse_moe_prefill_w8_down_kernel<false>
            <<<dim3(kHidden / kExpertBM, (tokens + kExpertBN - 1) / kExpertBN), kExpertThreads, 0,
               stream>>>(shared_activation, nullptr, shared_down_codes, shared_down_scales, nullptr,
                         routed_sum, shared_scale, output, tokens);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
