#pragma once

// Row-scaled E4M3 weight x materialized row-scaled E4M3 activation Tensor Core GEMM.
//
// The MMA tile is oriented as [token,K] x [K,output-row]. This makes the accumulator's
// contiguous axis the public output-row axis. The epilogue stages BF16 pairs and emits aligned
// output vectors, while the output policy remains replaceable by a fused semantic Op.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/vector_output.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_operands.h"
#include "ops/linear/fp8/fp8_shared.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

template <class Schedule>
__device__ __forceinline__ int fp8_mma_shared_byte(int row, int logical_byte) {
    const int logical_segment  = logical_byte >> 4;
    const int byte_in_segment  = logical_byte & 15;
    const int physical_segment = logical_segment ^ (row & (Schedule::kSegmentsPerRow - 1));
    return physical_segment * 16 + byte_in_segment;
}

template <class Schedule>
__device__ __forceinline__ void
fp8_mma_tile_coordinates(std::int32_t linear, std::int32_t row_tiles, std::int32_t token_tiles,
                         std::int32_t& row_tile, std::int32_t& token_tile) {
    if constexpr (Schedule::kRaster == Fp8MmaRaster::RowFast) {
        token_tile = linear / row_tiles;
        row_tile   = linear - token_tile * row_tiles;
    } else if constexpr (Schedule::kRaster == Fp8MmaRaster::TokenFast) {
        row_tile   = linear / token_tiles;
        token_tile = linear - row_tile * token_tiles;
    } else {
        constexpr int group_rows       = Schedule::kRasterGroupRows;
        const std::int32_t group_span  = group_rows * token_tiles;
        const std::int32_t group       = linear / group_span;
        const std::int32_t first_row   = group * group_rows;
        const std::int32_t active_rows = min(group_rows, row_tiles - first_row);
        const std::int32_t within      = linear - group * group_span;
        row_tile                       = first_row + within % active_rows;
        token_tile                     = within / active_rows;
    }
}

template <class Schedule, bool FullTokens, class Epilogue, class Output, class RowPolicy>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a8_mma_kernel(
    Fp8A8Operands operands, Output output, Epilogue epilogue, RowPolicy row_policy,
    int token_offset, int count) {
    constexpr bool PairRows                    = RowPolicy::kPaired;
    const auto* __restrict__ activation_codes  = operands.x;
    const auto* __restrict__ activation_scales = operands.x_scales;
    const auto* __restrict__ weight_codes      = operands.codes;
    const auto* __restrict__ weight_scales     = operands.scales;
    const int K             = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    const int tokens        = token_offset + count;
    const int TILES_K       = K / Schedule::kBlockK;
    constexpr int BM        = Schedule::kBlockTokens;
    constexpr int BN        = Schedule::kBlockRows;
    constexpr int BK        = Schedule::kBlockK;
    constexpr int S         = Schedule::kStages;
    constexpr int THREADS   = Schedule::kThreads;
    auto* shared_raw        = fp8_shared_storage<fp8_mma_shared_bytes<Schedule, Epilogue>>();
    auto* activation_shared = reinterpret_cast<std::uint8_t*>(shared_raw);
    auto* weight_shared     = activation_shared + S * BM * BK;

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int warp_token = warp / Schedule::kWarpsRows;
    const int warp_row   = warp - warp_token * Schedule::kWarpsRows;

    const int row_tiles   = operands.rows / BN;
    const int token_tiles = (count + BM - 1) / BM;
    int row_tile          = 0;
    int token_tile        = 0;
    fp8_mma_tile_coordinates<Schedule>(static_cast<int>(blockIdx.x), row_tiles, token_tiles,
                                       row_tile, token_tile);
    constexpr int rows_per_block = PairRows ? BN / 2 : BN;
    const int row_begin          = row_tile * rows_per_block;
    const int token_begin        = token_offset + token_tile * BM;

    auto stage_inputs = [&](int stage, int k_tile) {
        const int k_begin      = k_tile * BK;
        auto* activation_stage = activation_shared + stage * BM * BK;
        auto* weight_stage     = weight_shared + stage * BN * BK;

#pragma unroll 1
        for (int task = tid; task < BM * Schedule::kSegmentsPerRow; task += THREADS) {
            const int row             = task / Schedule::kSegmentsPerRow;
            const int logical_segment = task - row * Schedule::kSegmentsPerRow;
            const int logical_byte    = logical_segment * 16;
            const int physical_byte   = fp8_mma_shared_byte<Schedule>(row, logical_byte);
            auto* destination         = activation_stage + row * BK + physical_byte;
            const int token           = token_begin + row;
            if constexpr (FullTokens) {
                cp_async<16, Schedule::kActivationCache>(
                    destination, activation_codes + static_cast<std::int64_t>(token) * K + k_begin +
                                     logical_byte);
            } else {
                const bool valid = token < tokens;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    destination,
                    activation_codes + static_cast<std::int64_t>(valid ? token : 0) * K + k_begin +
                        logical_byte,
                    valid ? 16 : 0);
            }
        }

#pragma unroll 1
        for (int task = tid; task < BN * Schedule::kSegmentsPerRow; task += THREADS) {
            const int row             = task / Schedule::kSegmentsPerRow;
            const int logical_segment = task - row * Schedule::kSegmentsPerRow;
            const int logical_byte    = logical_segment * 16;
            const int physical_byte   = fp8_mma_shared_byte<Schedule>(row, logical_byte);
            const int weight_row      = row_policy.weight_row(row_begin, row, operands.rows);
            cp_async<16, Schedule::kWeightCache>(
                weight_stage + row * BK + physical_byte,
                weight_codes + static_cast<std::int64_t>(weight_row) * K + k_begin + logical_byte);
        }
    };

#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

    float accumulators[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
    const int a_matrix                                              = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;

#pragma unroll 1
    for (int k_tile = 0; k_tile < TILES_K; ++k_tile) {
        const int stage = k_tile % S;
        if (k_tile + S <= TILES_K) {
            cp_wait<S - 1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        auto load_fragments = [&](int k_step, unsigned(&a_fragments)[Schedule::kMmaTokens][4],
                                  unsigned(&b_fragments)[Schedule::kMmaRows][2]) {
#pragma unroll
            for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
                const int row = warp_token * Schedule::kWarpTokens + mma_token * 16 + a_row_offset;
                const int logical_byte  = k_step * 32 + a_column_byte;
                const int physical_byte = fp8_mma_shared_byte<Schedule>(row, logical_byte);
                ldmatrix_x4(
                    a_fragments[mma_token][0], a_fragments[mma_token][1], a_fragments[mma_token][2],
                    a_fragments[mma_token][3],
                    smem_addr(activation_shared + stage * BM * BK + row * BK + physical_byte));
            }
#pragma unroll
            for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                const int row = warp_row * Schedule::kWarpRows + mma_row * 8 + b_row_offset;
                const int logical_byte  = k_step * 32 + b_column_byte;
                const int physical_byte = fp8_mma_shared_byte<Schedule>(row, logical_byte);
                ldmatrix_x2(b_fragments[mma_row][0], b_fragments[mma_row][1],
                            smem_addr(weight_shared + stage * BN * BK + row * BK + physical_byte));
            }
        };

        if constexpr (Schedule::kFragmentPipeline == Fp8MmaFragmentPipeline::PingPong) {
            unsigned a_fragments[2][Schedule::kMmaTokens][4];
            unsigned b_fragments[2][Schedule::kMmaRows][2];
            load_fragments(0, a_fragments[0], b_fragments[0]);
#pragma unroll
            for (int k_step = 0; k_step < Schedule::kMmaK; ++k_step) {
                const int slot = k_step & 1;
                if (k_step + 1 < Schedule::kMmaK) {
                    load_fragments(k_step + 1, a_fragments[slot ^ 1], b_fragments[slot ^ 1]);
                }
#pragma unroll
                for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
#pragma unroll
                    for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                        mma_fp8_e4m3(
                            accumulators[mma_token][mma_row][0],
                            accumulators[mma_token][mma_row][1],
                            accumulators[mma_token][mma_row][2],
                            accumulators[mma_token][mma_row][3], a_fragments[slot][mma_token][0],
                            a_fragments[slot][mma_token][1], a_fragments[slot][mma_token][2],
                            a_fragments[slot][mma_token][3], b_fragments[slot][mma_row][0],
                            b_fragments[slot][mma_row][1]);
                    }
                }
            }
        } else {
            unsigned a_fragments[Schedule::kMmaTokens][4];
            unsigned b_fragments[Schedule::kMmaRows][2];
#pragma unroll
            for (int k_step = 0; k_step < Schedule::kMmaK; ++k_step) {
                load_fragments(k_step, a_fragments, b_fragments);
#pragma unroll
                for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
#pragma unroll
                    for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                        mma_fp8_e4m3(accumulators[mma_token][mma_row][0],
                                     accumulators[mma_token][mma_row][1],
                                     accumulators[mma_token][mma_row][2],
                                     accumulators[mma_token][mma_row][3], a_fragments[mma_token][0],
                                     a_fragments[mma_token][1], a_fragments[mma_token][2],
                                     a_fragments[mma_token][3], b_fragments[mma_row][0],
                                     b_fragments[mma_row][1]);
                    }
                }
            }
        }

        __syncthreads();
        const int next_k_tile = k_tile + S;
        if (next_k_tile < TILES_K) {
            stage_inputs(stage, next_k_tile);
            cp_commit();
        }
    }

    constexpr bool collective = requires {
        epilogue.template finish_tile<Schedule, FullTokens>(
            output, shared_raw, accumulators, row_begin, token_begin, operands.rows, tokens);
    };
    static_assert(!PairRows || collective, "paired rows require a collective epilogue");
    const int accumulator_token = lane >> 2;
    const int accumulator_row   = 2 * (lane & 3);
    constexpr int output_stride = BN + 8;
    auto* shared_output         = reinterpret_cast<__nv_bfloat16*>(shared_raw);
#pragma unroll
    for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
        const int token0 =
            token_begin + warp_token * Schedule::kWarpTokens + mma_token * 16 + accumulator_token;
        const int token1 = token0 + 8;
        const float activation_scale0 =
            (FullTokens || token0 < tokens) ? __ldg(activation_scales + token0) : 0.0F;
        const float activation_scale1 =
            (FullTokens || token1 < tokens) ? __ldg(activation_scales + token1) : 0.0F;
#pragma unroll
        for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
            const int local_row0  = warp_row * Schedule::kWarpRows + mma_row * 8 + accumulator_row;
            const int parent_row0 = row_policy.weight_row(row_begin, local_row0, operands.rows);
            const int parent_row1 = row_policy.weight_row(row_begin, local_row0 + 1, operands.rows);
            const float2 weight_scale = [&] {
                if constexpr (requires { RowPolicy::kContiguousPairs; }) {
                    if constexpr (RowPolicy::kContiguousPairs)
                        return bf16x2_bits_to_float2(
                            load_ldg<std::uint32_t>(weight_scales + parent_row0));
                }
                return make_float2(__bfloat162float(weight_scales[parent_row0]),
                                   __bfloat162float(weight_scales[parent_row1]));
            }();
            float value00 =
                accumulators[mma_token][mma_row][0] * activation_scale0 * weight_scale.x;
            float value01 =
                accumulators[mma_token][mma_row][1] * activation_scale0 * weight_scale.y;
            float value10 =
                accumulators[mma_token][mma_row][2] * activation_scale1 * weight_scale.x;
            float value11 =
                accumulators[mma_token][mma_row][3] * activation_scale1 * weight_scale.y;
            if constexpr (collective) {
                accumulators[mma_token][mma_row][0] = value00;
                accumulators[mma_token][mma_row][1] = value01;
                accumulators[mma_token][mma_row][2] = value10;
                accumulators[mma_token][mma_row][3] = value11;
            } else {
                if constexpr (FullTokens) {
                    value00 = epilogue.apply(parent_row0, token0, value00);
                    value01 = epilogue.apply(parent_row1, token0, value01);
                    value10 = epilogue.apply(parent_row0, token1, value10);
                    value11 = epilogue.apply(parent_row1, token1, value11);
                } else {
                    if (token0 < tokens) {
                        value00 = epilogue.apply(parent_row0, token0, value00);
                        value01 = epilogue.apply(parent_row1, token0, value01);
                    }
                    if (token1 < tokens) {
                        value10 = epilogue.apply(parent_row0, token1, value10);
                        value11 = epilogue.apply(parent_row1, token1, value11);
                    }
                }
                auto* destination0 = reinterpret_cast<__nv_bfloat162*>(
                    shared_output + (token0 - token_begin) * output_stride + local_row0);
                auto* destination1 = reinterpret_cast<__nv_bfloat162*>(
                    shared_output + (token1 - token_begin) * output_stride + local_row0);
                *destination0 = __floats2bfloat162_rn(value00, value01);
                *destination1 = __floats2bfloat162_rn(value10, value11);
            }
        }
    }
    __syncthreads();

    if constexpr (collective) {
        epilogue.template finish_tile<Schedule, FullTokens>(
            output, shared_raw, accumulators, row_begin, token_begin, operands.rows, tokens);
    } else {
        constexpr int stored_rows       = PairRows ? BN / 2 : BN;
        constexpr int vectors_per_token = stored_rows / 8;
        constexpr int output_vectors    = BM * vectors_per_token;
        for (int task = tid; task < output_vectors; task += THREADS) {
            const int token_local = task / vectors_per_token;
            const int row_vector  = task - token_local * vectors_per_token;
            const int token       = token_begin + token_local;
            if constexpr (FullTokens) {
                const uint4 values =
                    load_vec<uint4>(shared_output + token_local * output_stride + row_vector * 8);
                linear_store_bf16_vector(output, row_begin + row_vector * 8, token, values);
            } else if (token < tokens) {
                const uint4 values =
                    load_vec<uint4>(shared_output + token_local * output_stride + row_vector * 8);
                linear_store_bf16_vector(output, row_begin + row_vector * 8, token, values);
            }
        }
    }
}

} // namespace ninfer::ops::detail
