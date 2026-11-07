#pragma once

#include "ops/moe/kernels/cuda/groupwise_int/expert.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/mma.cuh"

namespace ninfer::ops::detail::moe_cuda::groupwise_int {
using common::Q4MmaDecodeAtom;
using common::store_vec;
using common::silu;
using common::cp_async;
using common::cp_async_zfill;
using common::cp_commit;
using common::cp_wait;
using common::smem_addr;
using common::mma_bf16;
using common::ldmatrix_x2;
using common::ldmatrix_x4;

enum class GateUpDensity { Any, Spread, Packed };

// One CTA computes `HalfRows = 32 * RowTiles` logical intermediate rows for a column job of one
// expert: the gate and the up weight rows of those logical rows form the A operand (BM = 2 *
// HalfRows rows), the job's token activations form the B operand. Each warp owns 16 logical rows
// (its gate and up rows) and `WarpCols` columns, so a warp's SwiGLU epilogue needs no exchange.
// Weights are decoded once per k tile into shared memory; the group scale is applied to the
// per-tile FP32 partial sums, so the decoded integers are exact in BF16.
template <class Shape, class Access, int ExpertWarps, int ExpertBN, int Stages,
          GateUpDensity Density, int RowTiles>
__global__ __launch_bounds__(ExpertWarps * 32, (RowTiles == 1 ? 3 : 2)) void grouped_q4_gate_up(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ packed_token,
    const int* __restrict__ expert_offsets, const int* __restrict__ route_job_experts,
    const int* __restrict__ route_job_columns, const GroupedInfo* __restrict__ info, Access weights,
    __nv_bfloat16* __restrict__ activation) {
    constexpr int BM = 64 * RowTiles, BK = 64, HalfRows = BM / 2;
    if (info->direct) { return; }
    if constexpr (Density != GateUpDensity::Any) {
        if (bool(info->spread) != (Density == GateUpDensity::Spread)) { return; }
    }
    constexpr int ExpertThreads = ExpertWarps * 32;
    constexpr int GroupsPerRow  = Shape::hidden / 64;
    constexpr int KTiles        = Shape::hidden / BK;
    constexpr int WarpsM        = BM / 32;
    constexpr int WarpsN        = ExpertWarps / WarpsM;
    constexpr int WarpCols      = ExpertBN / WarpsN;
    constexpr int WarpNT        = WarpCols / 8;
    constexpr int StageChunks   = BK / 8;
    constexpr int StageIters    = ExpertBN * StageChunks / ExpertThreads;
    static_assert(ExpertWarps % WarpsM == 0 && ExpertBN % WarpsN == 0 && WarpCols % 16 == 0,
                  "a warp covers whole pairs of 8-column MMA tiles");
    static_assert(StageIters * ExpertThreads == ExpertBN * StageChunks,
                  "the staging loop is unrolled, so every thread must take the same column count");
    static_assert(Stages >= 2 && KTiles >= Stages);
    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[Stages][ExpertBN * BK];
    __shared__ __align__(16) std::uint8_t Cr[Stages][BM * 32];
    __shared__ __align__(16) std::uint8_t Sr[BM * GroupsPerRow * 2];

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int warp_m = warp / WarpsN;
    const int warp_n = warp % WarpsN;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    // x4 ldmatrix serves two adjacent 8-column tiles: lanes 0-15 address the first, 16-31 the next.
    const int b_rowoff       = ((lane >> 4) << 3) + (lane & 7);
    const int b_koff         = ((lane >> 3) & 1) << 3;
    const int gid            = lane >> 2;
    const int lid            = lane & 3;
    constexpr int row_blocks = Shape::intermediate / HalfRows;
    const int total_work     = info->jobs * row_blocks;
    // A scale row holds GroupsPerRow FP16 values as 16-byte chunks. Rotating the chunk by the row
    // spreads the eight rows a quad of lanes reads over distinct banks.
    constexpr int ScaleChunks = GroupsPerRow * 2 / 16;
    auto scale_slot           = [](int row, int chunk) {
        return row * ScaleChunks * 16 + ((chunk ^ ((row >> 1) & (ScaleChunks - 1))) << 4);
    };
    for (int work = static_cast<int>(blockIdx.x); work < total_work;
         work += static_cast<int>(gridDim.x)) {
        const int route_job   = work / row_blocks;
        const int row_block   = work - route_job * row_blocks;
        const int expert      = route_job_experts[route_job];
        const auto matrix     = weights.template resolve<2 * Shape::intermediate>(expert);
        const auto* codes     = matrix.codes;
        const auto* scales    = matrix.scales;
        const int logical0    = row_block * HalfRows;
        const int begin       = expert_offsets[expert];
        const int count       = expert_offsets[expert + 1] - begin;
        const int column_base = route_job_columns[route_job];
        const int cols        = count - column_base < ExpertBN ? count - column_base : ExpertBN;
        // A thread stages the same columns of all Shape::hidden / BK tiles, so it resolves their
        // rows once for the whole job instead of reading the map and redoing the row multiply per
        // tile. A column past the tile is clamped onto a live one and then zero-filled.
        const __nv_bfloat16* src_row[StageIters];
#pragma unroll
        for (int i = 0; i < StageIters; ++i) {
            const int col = (tid + i * ExpertThreads) / StageChunks;
            const int row = packed_token[col < cols ? begin + column_base + col : begin];
            src_row[i]    = x + static_cast<std::int64_t>(row) * Shape::hidden;
        }
        float acc[2][WarpNT][4] = {};

        // Tile rows [0, HalfRows) are gate rows and [HalfRows, BM) the matching up rows.
        auto global_row = [&](int local_row) {
            const int logical = logical0 + (local_row & (HalfRows - 1));
            return matrix.row + logical + (local_row >= HalfRows ? Shape::intermediate : 0);
        };

        auto stage_scales = [&] {
            static_assert(BM * ScaleChunks % ExpertThreads == 0);
#pragma unroll
            for (int i = 0; i < BM * ScaleChunks / ExpertThreads; ++i) {
                const int item  = tid + i * ExpertThreads;
                const int row   = item / ScaleChunks;
                const int chunk = item - row * ScaleChunks;
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(row)) * GroupsPerRow + chunk * 8;
                cp_async<16, Cache::cg>(&Sr[scale_slot(row, chunk)], &scales[gi * 2]);
            }
        };

        auto stage_inputs = [&](int stage, int kt) {
            const int k0 = kt * BK;
#pragma unroll
            for (int i = 0; i < StageIters; ++i) {
                const int item = tid + i * ExpertThreads;
                const int col  = item / StageChunks;
                const int k8   = item - col * StageChunks;
                auto* dst      = &Bs[stage][col * BK + common::swizzle_xor_64(col, k8 * 8)];
                cp_async_zfill<16, Cache::cg>(dst, src_row[i] + k0 + k8 * 8, col < cols ? 16 : 0);
            }

#pragma unroll
            for (int i = 0; i < (BM * 2 + ExpertThreads - 1) / ExpertThreads; ++i) {
                const int item = tid + i * ExpertThreads;
                if (BM * 2 % ExpertThreads == 0 || item < BM * 2) {
                    const int row  = item >> 1;
                    const int half = item & 1;
                    const std::int64_t gi =
                        static_cast<std::int64_t>(global_row(row)) * GroupsPerRow + kt;
                    cp_async<16, Cache::cg>(&Cr[stage][row * 32 + half * 16],
                                            &codes[gi * 32 + half * 16]);
                }
            }
        };

        auto decode_weight = [&](int stage) {
            constexpr int CodeChunksPerRow = common::Q4RowSplitStorage::kCodeBytesPerGroup / 4;
            static_assert(CodeChunksPerRow * 8 == BK,
                          "a row of codes must decode to exactly the tile's k width");
            static_assert(BM * CodeChunksPerRow % ExpertThreads == 0);
#pragma unroll
            for (int i = 0; i < BM * CodeChunksPerRow / ExpertThreads; ++i) {
                const int item  = tid + i * ExpertThreads;
                const int row   = item / CodeChunksPerRow;
                const int chunk = item - row * CodeChunksPerRow;
                unsigned decoded[4];
                Q4MmaDecodeAtom::decode_eight(
                    *reinterpret_cast<const unsigned*>(&Cr[stage][row * 32 + chunk * 4]), decoded);
                store_vec(&As[row * BK + common::swizzle_xor_64(row, chunk * 8)],
                          make_int4(static_cast<int>(decoded[0]), static_cast<int>(decoded[1]),
                                    static_cast<int>(decoded[2]), static_cast<int>(decoded[3])));
            }
        };

        stage_scales();
        cp_commit();
#pragma unroll
        for (int stage = 0; stage < Stages; ++stage) {
            stage_inputs(stage, stage);
            cp_commit();
        }

        // Two barriers per k tile. The first proves every warp has finished the previous tile's
        // MMAs, which frees both the decoded tile and the stage that tile read, so the refill of
        // that stage is issued behind it.
#pragma unroll 1
        for (int kt = 0; kt < KTiles; ++kt) {
            const int stage = kt % Stages;
            if (kt == 0) {
                cp_wait<Stages - 1>();
            } else {
                cp_wait<Stages - 2>();
            }
            __syncthreads();
            if (kt > 0) {
                const int next = kt - 1 + Stages;
                if (next < KTiles) { stage_inputs((kt - 1) % Stages, next); }
                cp_commit();
            }
            decode_weight(stage);
            __syncthreads();

            if (warp_n * WarpCols < cols) {
                float partial[2][WarpNT][4] = {};
#pragma unroll
                for (int ki = 0; ki < BK / 16; ++ki) {
                    unsigned af[2][4];
                    unsigned bf[WarpNT][2];
#pragma unroll
                    for (int mi = 0; mi < 2; ++mi) {
                        const int row = mi * HalfRows + warp_m * 16 + a_rowoff;
                        const int col = ki * 16 + a_coloff;
                        ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                    smem_addr(&As[row * BK + common::swizzle_xor_64(row, col)]));
                    }
#pragma unroll
                    for (int np = 0; np < WarpNT / 2; ++np) {
                        const int brow = warp_n * WarpCols + np * 16 + b_rowoff;
                        const int bcol = ki * 16 + b_koff;
                        ldmatrix_x4(
                            bf[2 * np][0], bf[2 * np][1], bf[2 * np + 1][0], bf[2 * np + 1][1],
                            smem_addr(&Bs[stage][brow * BK + common::swizzle_xor_64(brow, bcol)]));
                    }
#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        // Tiles past the job's last column carry only zero-filled activations.
                        if (warp_n * WarpCols + ni * 8 < cols) {
#pragma unroll
                            for (int mi = 0; mi < 2; ++mi) {
                                mma_bf16(partial[mi][ni][0], partial[mi][ni][1],
                                         partial[mi][ni][2], partial[mi][ni][3], af[mi][0],
                                         af[mi][1], af[mi][2], af[mi][3], bf[ni][0], bf[ni][1]);
                            }
                        }
                    }
                }
#pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    const int row0 = mi * HalfRows + warp_m * 16 + gid;
                    const int row1 = row0 + 8;
                    const int slot = (kt >> 3);
                    const int byte = (kt & 7) * 2;
                    const float scale0 = __half2float(__ushort_as_half(
                        *reinterpret_cast<const std::uint16_t*>(&Sr[scale_slot(row0, slot) + byte])));
                    const float scale1 = __half2float(__ushort_as_half(
                        *reinterpret_cast<const std::uint16_t*>(&Sr[scale_slot(row1, slot) + byte])));
#pragma unroll
                    for (int ni = 0; ni < WarpNT; ++ni) {
                        acc[mi][ni][0] = fmaf(partial[mi][ni][0], scale0, acc[mi][ni][0]);
                        acc[mi][ni][1] = fmaf(partial[mi][ni][1], scale0, acc[mi][ni][1]);
                        acc[mi][ni][2] = fmaf(partial[mi][ni][2], scale1, acc[mi][ni][2]);
                        acc[mi][ni][3] = fmaf(partial[mi][ni][3], scale1, acc[mi][ni][3]);
                    }
                }
            }
        }
        cp_wait<0>();
        // Every warp has finished reading this job's tiles before the next job restages them.
        __syncthreads();

        if (warp_n * WarpCols < cols) {
            const int row0 = logical0 + warp_m * 16 + gid;
            const int row1 = row0 + 8;
#pragma unroll
            for (int ni = 0; ni < WarpNT; ++ni) {
                const int local_col0 = warp_n * WarpCols + ni * 8 + 2 * lid;
                const int local_col1 = local_col0 + 1;
                const int col0       = begin + column_base + local_col0;
                const int col1       = col0 + 1;
                if (local_col0 < cols) {
                    activation[static_cast<std::int64_t>(col0) * Shape::intermediate + row0] =
                        __float2bfloat16_rn(silu(acc[0][ni][0]) * acc[1][ni][0]);
                    activation[static_cast<std::int64_t>(col0) * Shape::intermediate + row1] =
                        __float2bfloat16_rn(silu(acc[0][ni][2]) * acc[1][ni][2]);
                }
                if (local_col1 < cols) {
                    activation[static_cast<std::int64_t>(col1) * Shape::intermediate + row0] =
                        __float2bfloat16_rn(silu(acc[0][ni][1]) * acc[1][ni][1]);
                    activation[static_cast<std::int64_t>(col1) * Shape::intermediate + row1] =
                        __float2bfloat16_rn(silu(acc[0][ni][3]) * acc[1][ni][3]);
                }
            }
        }
    }
}

template <class Shape, class Access, bool Routed, int Columns>
__global__ __launch_bounds__(Columns * 4, 2) void grouped_q8_gate_up(
    const __nv_bfloat16* __restrict__ input, const int* __restrict__ packed_token,
    const int* __restrict__ expert_offsets, const int* __restrict__ job_experts,
    const int* __restrict__ job_columns, Access weights, __nv_bfloat16* __restrict__ activation,
    __nv_bfloat16* __restrict__ activation_low, int tokens, const GroupedInfo* __restrict__ info) {
    constexpr int BM = 64, BK = 64, BN = Columns, Stages = 2, Threads = Columns * 4;
    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[Stages][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[BM * BK];
    __shared__ __align__(16) std::uint8_t Sr[BM * 16];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (info && info->direct) { return; }
    const int a_mat              = lane >> 3;
    const int a_rin              = lane & 7;
    const int a_rowoff           = a_rin + ((a_mat & 1) << 3);
    const int a_coloff           = (a_mat >> 1) << 3;
    const int b_rin              = lane & 7;
    const int b_koff             = ((lane >> 3) & 1) << 3;
    const int gid                = lane >> 2;
    const int lid                = lane & 3;
    constexpr int groups_per_row = Shape::hidden / 32;

    constexpr int RowBlocks = Shape::intermediate / 32;
    const int work_count    = Routed ? info->jobs * RowBlocks : 1;
    const int first         = Routed ? int(blockIdx.x) : 0;
    const int stride        = Routed ? int(gridDim.x) : 1;
    for (int work = first; work < work_count; work += stride) {
        const int job       = work / RowBlocks;
        const int expert    = Routed ? job_experts[job] : 0;
        const int row_block = Routed ? work % RowBlocks : int(blockIdx.x);
        const int logical0  = row_block * (BM / 2);
        const int begin     = Routed ? expert_offsets[expert] : int(blockIdx.y) * BN;
        const int count     = Routed ? expert_offsets[expert + 1] - begin : min(tokens - begin, BN);
        const int column_base = Routed ? job_columns[job] : 0;
        const int cols        = min(count - column_base, BN);
        const auto matrix     = weights.template resolve<2 * Shape::intermediate>(expert);
        const auto* codes     = matrix.codes;
        const auto* scales    = matrix.scales;
        float acc[4][4]       = {};

        auto global_row = [&](int local_row) {
            const int logical = logical0 + (local_row & (BM / 2 - 1));
            const int row     = logical + (local_row >= BM / 2 ? Shape::intermediate : 0);
            return matrix.row + row;
        };

        auto stage_x = [&](int stage, int kt) {
            const int k0 = kt * BK;
            for (int item = tid; item < BN * (BK / 8); item += Threads) {
                const int col        = item / (BK / 8);
                const int k8         = item - col * (BK / 8);
                const int source_col = begin + column_base + col;
                auto* dst            = &Bs[stage][col * BK + common::swizzle_xor_64(col, k8 * 8)];
                const int packed     = col < cols ? source_col : begin;
                const int source     = Routed ? packed_token[packed] : packed;
                const auto* src =
                    input + static_cast<std::int64_t>(source) * Shape::hidden + k0 + k8 * 8;
                cp_async_zfill<16, Cache::cg>(dst, src, col < cols ? 16 : 0);
            }
        };

        auto stage_weight = [&](int kt) {
            constexpr int groups_per_tile   = BK / 32;
            constexpr int scale_cache_tiles = 8 / groups_per_tile;
            const int group0                = kt * groups_per_tile;
            for (int item = tid; item < BM * (BK / 16); item += Threads) {
                const int row   = item / (BK / 16);
                const int chunk = item - row * (BK / 16);
                const std::int64_t gi =
                    static_cast<std::int64_t>(global_row(row)) * groups_per_row + group0;
                cp_async<16, Cache::cg>(&Cr[row * BK + chunk * 16], &codes[gi * 32 + chunk * 16]);
            }
            if ((kt % scale_cache_tiles) == 0) {
                for (int row = tid; row < BM; row += Threads) {
                    const std::int64_t gi =
                        static_cast<std::int64_t>(global_row(row)) * groups_per_row + group0;
                    cp_async<16, Cache::cg>(&Sr[row * 16], &scales[gi * 2]);
                }
            }
        };

        auto decode_weight = [&] {
            for (int item = tid; item < BM * BK / 2; item += Threads) {
                const int row = item / (BK / 2), col = (item % (BK / 2)) * 2;
                const auto packed = *reinterpret_cast<const std::uint16_t*>(&Cr[row * BK + col]);
                const int q0      = static_cast<std::int8_t>(packed & 0xffu);
                const int q1      = static_cast<std::int8_t>(packed >> 8);
                store_vec(&As[row * BK + common::swizzle_xor_64(row, col)],
                          __floats2bfloat162_rn(float(q0), float(q1)));
            }
        };

        stage_x(0, 0);
        stage_weight(0);
        cp_commit();

#pragma unroll 4
        for (int kt = 0; kt < Shape::hidden / BK; ++kt) {
            const int stage = kt & 1;
            cp_wait<0>();
            __syncthreads();
            decode_weight();
            __syncthreads();

            // Capture both groups' scales before the next asynchronous weight stage overwrites Sr.
            unsigned row_scales[4][2];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int row     = mi * 16 + gid;
                const int offset  = (kt % 4) * 4;
                row_scales[mi][0] = *reinterpret_cast<const unsigned*>(&Sr[row * 16 + offset]);
                row_scales[mi][1] =
                    *reinterpret_cast<const unsigned*>(&Sr[(row + 8) * 16 + offset]);
            }

            __syncthreads();
            const int next = kt + 1;
            if (next < Shape::hidden / BK) {
                stage_x(next & 1, next);
                stage_weight(next);
                cp_commit();
            }

            if (warp * 8 < cols) {
#pragma unroll
                for (int group = 0; group < 2; ++group) {
                    float partial[4][4] = {};
#pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        const int ki = group * 2 + k;
                        unsigned bf0, bf1;
                        const int brow = warp * 8 + b_rin, bcol = ki * 16 + b_koff;
                        ldmatrix_x2(
                            bf0, bf1,
                            smem_addr(&Bs[stage][brow * BK + common::swizzle_xor_64(brow, bcol)]));
#pragma unroll
                        for (int mi = 0; mi < 4; ++mi) {
                            const int row = mi * 16 + a_rowoff, col = ki * 16 + a_coloff;
                            unsigned af0, af1, af2, af3;
                            ldmatrix_x4(
                                af0, af1, af2, af3,
                                smem_addr(&As[row * BK + common::swizzle_xor_64(row, col)]));
                            mma_bf16(partial[mi][0], partial[mi][1], partial[mi][2], partial[mi][3],
                                     af0, af1, af2, af3, bf0, bf1);
                        }
                    }
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const float s0 = __half2float(__ushort_as_half(
                            static_cast<unsigned short>(row_scales[mi][0] >> (group * 16))));
                        const float s1 = __half2float(__ushort_as_half(
                            static_cast<unsigned short>(row_scales[mi][1] >> (group * 16))));
                        acc[mi][0]     = fmaf(partial[mi][0], s0, acc[mi][0]);
                        acc[mi][1]     = fmaf(partial[mi][1], s0, acc[mi][1]);
                        acc[mi][2]     = fmaf(partial[mi][2], s1, acc[mi][2]);
                        acc[mi][3]     = fmaf(partial[mi][3], s1, acc[mi][3]);
                    }
                }
            }
        }
        cp_wait<0>();
        __syncthreads();

        if (warp * 8 < cols) {
            const auto store = [&](std::int64_t index, float value) {
                const auto hi         = __float2bfloat16_rn(value);
                activation[index]     = hi;
                activation_low[index] = __float2bfloat16_rn(value - __bfloat162float(hi));
            };
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int row0       = logical0 + mi * 16 + gid;
                const int row1       = row0 + 8;
                const int col0       = begin + column_base + warp * 8 + 2 * lid;
                const int col1       = col0 + 1;
                const int local_col0 = warp * 8 + 2 * lid;
                const int local_col1 = local_col0 + 1;
                if (local_col0 < cols) {
                    store(static_cast<std::int64_t>(col0) * Shape::intermediate + row0,
                          silu(acc[mi][0]) * acc[mi + 2][0]);
                    store(static_cast<std::int64_t>(col0) * Shape::intermediate + row1,
                          silu(acc[mi][2]) * acc[mi + 2][2]);
                }
                if (local_col1 < cols) {
                    store(static_cast<std::int64_t>(col1) * Shape::intermediate + row0,
                          silu(acc[mi][1]) * acc[mi + 2][1]);
                    store(static_cast<std::int64_t>(col1) * Shape::intermediate + row1,
                          silu(acc[mi][3]) * acc[mi + 2][3]);
                }
            }
        }
        __syncthreads();
    }
}

} // namespace ninfer::ops::detail::moe_cuda::groupwise_int
