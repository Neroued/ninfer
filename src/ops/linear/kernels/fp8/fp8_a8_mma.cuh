#pragma once
#include "ops/common/cuda/pipeline.cuh"

#include "ops/common/cuda/memory.cuh"
#include "ops/linear/common/epilogue.cuh"

// Row-scaled E4M3 weight x materialized row-scaled E4M3 activation Tensor Core GEMM.
//
// The MMA tile is oriented as [token,K] x [K,output-row]. This makes the accumulator's
// contiguous axis the public output-row axis. The epilogue stages BF16 pairs and emits aligned
// output vectors, while the output policy remains replaceable by a fused semantic Op.

#include "ops/linear/common/output.cuh"
#include "ops/linear/kernels/fp8/fp8_schedule.cuh"
#include "ops/linear/kernels/fp8/fp8_operands.h"
#include "ops/linear/kernels/fp8/fp8_a8_mma_common.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

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
    auto* shared_raw        = common::shared_storage<linear_mma_shared_bytes<Schedule, Epilogue>>();
    auto* activation_shared = reinterpret_cast<std::uint8_t*>(shared_raw);
    auto* weight_shared     = activation_shared + S * BM * BK;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

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
                common::cp_async<16, Schedule::kActivationCache>(
                    destination, activation_codes + static_cast<std::int64_t>(token) * K + k_begin +
                                     logical_byte);
            } else {
                const bool valid = token < tokens;
                common::cp_async_zfill<16, Schedule::kActivationCache>(
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
            common::cp_async<16, Schedule::kWeightCache>(
                weight_stage + row * BK + physical_byte,
                weight_codes + static_cast<std::int64_t>(weight_row) * K + k_begin + logical_byte);
        }
    };

#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        stage_inputs(stage, stage);
        common::cp_commit();
    }

    float accumulators[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
#pragma unroll 1
    for (int k_tile = 0; k_tile < TILES_K; ++k_tile) {
        const int stage = k_tile % S;
        if (k_tile + S <= TILES_K) {
            common::cp_wait<S - 1>();
        } else {
            common::cp_wait<0>();
        }
        __syncthreads();

        fp8_mma_compute_stage<Schedule>(activation_shared + stage * BM * BK,
                                        weight_shared + stage * BN * BK, accumulators, warp, lane);

        __syncthreads();
        const int next_k_tile = k_tile + S;
        if (next_k_tile < TILES_K) {
            stage_inputs(stage, next_k_tile);
            common::cp_commit();
        }
    }

    fp8_finish_mma_tile<Schedule, FullTokens>(
        output, epilogue, row_policy, shared_raw, accumulators, activation_scales, weight_scales,
        row_begin, token_begin, operands.rows, tokens, warp, lane);
}

} // namespace ninfer::ops::detail
