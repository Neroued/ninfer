#pragma once

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear/nvfp4/nvfp4_shared.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/vector_output.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <class Schedule>
struct Nvfp4A4SharedStorage {
    alignas(16)
        std::uint8_t a_codes[Schedule::kStages][Schedule::kBlockTokens * Schedule::kCodeRowBytes];
    alignas(
        16) std::uint8_t b_codes[Schedule::kStages][Schedule::kBlockRows * Schedule::kCodeRowBytes];
    alignas(16)
        std::uint32_t a_scale4[Schedule::kStages][Schedule::kBlockTokens * Schedule::kK64PerStage];
    alignas(16)
        std::uint8_t b_scales[Schedule::kStages][Schedule::kBlockRows * Schedule::kK64PerStage * 4];
};

template <class Schedule>
__device__ __forceinline__ int nvfp4_a4_swizzled_byte(int row, int logical_byte) {
    static_assert((Schedule::kSegmentsPerRow & (Schedule::kSegmentsPerRow - 1)) == 0);
    const int logical_segment  = logical_byte >> 4;
    const int byte_in_segment  = logical_byte & 15;
    const int physical_segment = logical_segment ^ (row & (Schedule::kSegmentsPerRow - 1));
    return physical_segment * 16 + byte_in_segment;
}

template <class Schedule>
__device__ __forceinline__ void
stage_nvfp4_a4_activation(const std::uint8_t* __restrict__ x,
                          const std::uint8_t* __restrict__ x_scales, int input_rows,
                          Nvfp4A4SharedStorage<Schedule>& shared, int stage, int k_tile,
                          int token_begin, int active_tokens) {
    const int k              = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    constexpr int kCodeTasks = Schedule::kBlockTokens * Schedule::kSegmentsPerRow;
    for (int task = static_cast<int>(threadIdx.x); task < kCodeTasks; task += Schedule::kThreads) {
        const int row             = task / Schedule::kSegmentsPerRow;
        const int logical_segment = task - row * Schedule::kSegmentsPerRow;
        const int token           = token_begin + row;
        const bool valid          = token < active_tokens;
        const int source_token    = valid ? token : 0;
        const int physical_byte   = nvfp4_a4_swizzled_byte<Schedule>(row, logical_segment * 16);
        auto* destination = shared.a_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
        const auto* input = x + static_cast<std::int64_t>(source_token) * (k / 2) +
                            k_tile * Schedule::kCodeRowBytes + logical_segment * 16;
        cp_async_zfill<16, Schedule::kActivationCache>(destination, input, valid ? 16 : 0);
    }

    if constexpr (Schedule::kK64PerStage <= 4) {
        constexpr int kScaleBytes = Schedule::kK64PerStage * 4;
        for (int row = static_cast<int>(threadIdx.x); row < Schedule::kBlockTokens;
             row += Schedule::kThreads) {
            const int token        = token_begin + row;
            const bool valid       = token < active_tokens;
            const int source_token = valid ? token : 0;
            auto* destination      = &shared.a_scale4[stage][row * Schedule::kK64PerStage];
            const auto* input = x_scales + (static_cast<std::int64_t>(source_token) * (k / 64) +
                                            k_tile * Schedule::kK64PerStage) *
                                               4;
            cp_async_zfill<kScaleBytes>(destination, input, valid ? kScaleBytes : 0);
        }
    } else {
        static_assert((Schedule::kK64PerStage % 4) == 0);
        constexpr int kSegmentsPerRow = Schedule::kK64PerStage / 4;
        constexpr int kScaleTasks     = Schedule::kBlockTokens * kSegmentsPerRow;
        for (int task = static_cast<int>(threadIdx.x); task < kScaleTasks;
             task += Schedule::kThreads) {
            const int row          = task / kSegmentsPerRow;
            const int segment      = task - row * kSegmentsPerRow;
            const int token        = token_begin + row;
            const bool valid       = token < active_tokens;
            const int source_token = valid ? token : 0;
            const int local_k64    = segment * 4;
            auto* destination = &shared.a_scale4[stage][row * Schedule::kK64PerStage + local_k64];
            const auto* input = x_scales + (static_cast<std::int64_t>(source_token) * (k / 64) +
                                            k_tile * Schedule::kK64PerStage + local_k64) *
                                               4;
            cp_async_zfill<16>(destination, input, valid ? 16 : 0);
        }
    }
}

template <class Schedule, class RowPolicy>
__device__ __forceinline__ void
stage_nvfp4_a4_weight(const std::uint8_t* __restrict__ codes,
                      const std::uint8_t* __restrict__ scales,
                      Nvfp4A4SharedStorage<Schedule>& shared, int stage, int k_tile, int row_begin,
                      RowPolicy row_policy, int rows, int input_rows) {
    const int k              = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    constexpr int kCodeTasks = Schedule::kBlockRows * Schedule::kSegmentsPerRow;
    for (int task = static_cast<int>(threadIdx.x); task < kCodeTasks; task += Schedule::kThreads) {
        const int row             = task / Schedule::kSegmentsPerRow;
        const int logical_segment = task - row * Schedule::kSegmentsPerRow;
        const int weight_row      = row_policy.weight_row(row_begin, row, rows);
        const int physical_byte   = nvfp4_a4_swizzled_byte<Schedule>(row, logical_segment * 16);
        auto* destination = shared.b_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
        const auto* input = codes + static_cast<std::int64_t>(weight_row) * (k / 2) +
                            k_tile * Schedule::kCodeRowBytes + logical_segment * 16;
        cp_async<16, Schedule::kWeightCache>(destination, input);
    }

    if constexpr (RowPolicy::kContiguous &&
                  (Schedule::kBlockRows == 64 || Schedule::kBlockRows % 128 == 0)) {
        static_assert((Schedule::kBlockRows % 64) == 0);
        constexpr int kScaleRowTiles = Schedule::kBlockRows >= 128 ? Schedule::kBlockRows / 128 : 1;
        constexpr int kQuartilesPerTile =
            Schedule::kBlockRows >= 128 ? 4 : Schedule::kBlockRows / 32;
        constexpr int kScaleBytesPerTask = kQuartilesPerTile * 4;
        constexpr int kScaleTasks        = kScaleRowTiles * Schedule::kK64PerStage * 32;
        for (int task = static_cast<int>(threadIdx.x); task < kScaleTasks;
             task += Schedule::kThreads) {
            const int row_tile         = task / (Schedule::kK64PerStage * 32);
            const int remainder        = task - row_tile * Schedule::kK64PerStage * 32;
            const int local_k64        = remainder / 32;
            const int row_mod32        = remainder - local_k64 * 32;
            const int global_k64       = k_tile * Schedule::kK64PerStage + local_k64;
            const int global_row_begin = row_begin + row_tile * 128;
            const int persistent_tile  = global_row_begin / 128;
            const int first_quartile   = (global_row_begin & 127) / 32;
            auto* destination          = shared.b_scales[stage] +
                                ((row_tile * Schedule::kK64PerStage + local_k64) * 32 + row_mod32) *
                                    kScaleBytesPerTask;
            const auto* input =
                scales + static_cast<std::int64_t>(persistent_tile * (k / 64) + global_k64) * 512 +
                row_mod32 * 16 + first_quartile * 4;
            cp_async<kScaleBytesPerTask>(destination, input);
        }
    } else {
        constexpr int kScaleTasks = Schedule::kBlockRows * Schedule::kK64PerStage;
        for (int task = static_cast<int>(threadIdx.x); task < kScaleTasks;
             task += Schedule::kThreads) {
            const int row             = task / Schedule::kK64PerStage;
            const int local_k64       = task - row * Schedule::kK64PerStage;
            const int weight_row      = row_policy.weight_row(row_begin, row, rows);
            const int global_k64      = k_tile * Schedule::kK64PerStage + local_k64;
            const int persistent_tile = weight_row / 128;
            const int row_in_tile     = weight_row & 127;
            const int row_mod32       = row_in_tile & 31;
            const int row_quartile    = row_in_tile >> 5;
            auto* destination =
                shared.b_scales[stage] + (row * Schedule::kK64PerStage + local_k64) * 4;
            const auto* input =
                scales + static_cast<std::int64_t>(persistent_tile * (k / 64) + global_k64) * 512 +
                row_mod32 * 16 + row_quartile * 4;
            cp_async<4>(destination, input);
        }
    }
}

template <class Schedule, bool FullTokens, class Epilogue, class OutputPolicy, class RowPolicy>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_a4_mma_kernel(
    const std::uint8_t* __restrict__ x, const std::uint8_t* __restrict__ x_scales,
    const std::uint8_t* __restrict__ weight_codes, const std::uint8_t* __restrict__ weight_scales,
    int rows, int input_rows, float alpha, OutputPolicy output, Epilogue epilogue,
    RowPolicy row_policy, int token_offset, int count) {
    constexpr bool PairRows = RowPolicy::kPaired;
    const int k             = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    const int tokens        = token_offset + count;
    auto* shared_raw        = nvfp4_shared_storage<nvfp4_mma_shared_bytes<Schedule, Epilogue>>();
    auto& shared            = *reinterpret_cast<Nvfp4A4SharedStorage<Schedule>*>(shared_raw);
    const int token_begin   = token_offset + static_cast<int>(blockIdx.y) * Schedule::kBlockTokens;
    constexpr int kRowsPerBlock = PairRows ? Schedule::kBlockRows / 2 : Schedule::kBlockRows;
    const int row_begin         = static_cast<int>(blockIdx.x) * kRowsPerBlock;
    const int kKTiles           = k / Schedule::kBlockK;
    constexpr int kWaitGroups   = Schedule::kStages - 1;

#pragma unroll
    for (int stage = 0; stage < Schedule::kStages; ++stage) {
        if (stage < kKTiles) {
            stage_nvfp4_a4_activation<Schedule>(x, x_scales, k, shared, stage, stage, token_begin,
                                                tokens);
            stage_nvfp4_a4_weight<Schedule>(weight_codes, weight_scales, shared, stage, stage,
                                            row_begin, row_policy, rows, k);
            cp_commit();
        }
    }

    float accumulators[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m = warp / Schedule::kWarpsRows;
    const int warp_n = warp - warp_m * Schedule::kWarpsRows;

    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;

    for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
        const int stage = k_tile % Schedule::kStages;
        cp_wait<kWaitGroups>();
        __syncthreads();

#pragma unroll
        for (int local_k64 = 0; local_k64 < Schedule::kK64PerStage; ++local_k64) {
            unsigned a_fragments[Schedule::kMmaTokens][4];
            unsigned b_fragments[Schedule::kMmaRows][2];
            unsigned a_scales[Schedule::kMmaTokens];
            unsigned b_scales[Schedule::kMmaRows];

#pragma unroll
            for (int mma_m = 0; mma_m < Schedule::kMmaTokens; ++mma_m) {
                const int row          = warp_m * Schedule::kWarpTokens + mma_m * 16 + a_row_offset;
                const int logical_byte = local_k64 * 32 + a_column_byte;
                const int physical_byte = nvfp4_a4_swizzled_byte<Schedule>(row, logical_byte);
                const auto* address =
                    shared.a_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                ldmatrix_x4(a_fragments[mma_m][0], a_fragments[mma_m][1], a_fragments[mma_m][2],
                            a_fragments[mma_m][3], smem_addr(address));
                const int scale_row = warp_m * Schedule::kWarpTokens + mma_m * 16 + sfa_row;
                a_scales[mma_m] =
                    shared.a_scale4[stage][scale_row * Schedule::kK64PerStage + local_k64];
            }

#pragma unroll
            for (int mma_n = 0; mma_n < Schedule::kMmaRows; ++mma_n) {
                const int row           = warp_n * Schedule::kWarpRows + mma_n * 8 + b_row_offset;
                const int logical_byte  = local_k64 * 32 + b_column_byte;
                const int physical_byte = nvfp4_a4_swizzled_byte<Schedule>(row, logical_byte);
                const auto* address =
                    shared.b_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                ldmatrix_x2(b_fragments[mma_n][0], b_fragments[mma_n][1], smem_addr(address));
                const int scale_row = warp_n * Schedule::kWarpRows + mma_n * 8 + sfb_row;
                const std::uint8_t* scale_address;
                if constexpr (RowPolicy::kContiguous &&
                              (Schedule::kBlockRows == 64 || Schedule::kBlockRows % 128 == 0)) {
                    constexpr int kQuartilesPerTile =
                        Schedule::kBlockRows >= 128 ? 4 : Schedule::kBlockRows / 32;
                    constexpr int kScaleBytesPerTask = kQuartilesPerTile * 4;
                    const int row_tile               = scale_row / 128;
                    const int row_in_tile            = scale_row & 127;
                    const int row_mod32              = row_in_tile & 31;
                    const int row_quartile           = row_in_tile >> 5;
                    scale_address =
                        shared.b_scales[stage] +
                        ((row_tile * Schedule::kK64PerStage + local_k64) * 32 + row_mod32) *
                            kScaleBytesPerTask +
                        row_quartile * 4;
                } else {
                    scale_address = shared.b_scales[stage] +
                                    (scale_row * Schedule::kK64PerStage + local_k64) * 4;
                }
                b_scales[mma_n] = load_vec<unsigned>(scale_address);
            }

#pragma unroll
            for (int mma_m = 0; mma_m < Schedule::kMmaTokens; ++mma_m) {
#pragma unroll
                for (int mma_n = 0; mma_n < Schedule::kMmaRows; ++mma_n) {
                    mma_nvfp4_e4m3(accumulators[mma_m][mma_n][0], accumulators[mma_m][mma_n][1],
                                   accumulators[mma_m][mma_n][2], accumulators[mma_m][mma_n][3],
                                   a_fragments[mma_m][0], a_fragments[mma_m][1],
                                   a_fragments[mma_m][2], a_fragments[mma_m][3],
                                   b_fragments[mma_n][0], b_fragments[mma_n][1], a_scales[mma_m],
                                   b_scales[mma_n]);
                }
            }
        }

        __syncthreads();
        const int next_k_tile = k_tile + Schedule::kStages;
        if (next_k_tile < kKTiles) {
            stage_nvfp4_a4_activation<Schedule>(x, x_scales, k, shared, stage, next_k_tile,
                                                token_begin, tokens);
            stage_nvfp4_a4_weight<Schedule>(weight_codes, weight_scales, shared, stage, next_k_tile,
                                            row_begin, row_policy, rows, k);
        }
        cp_commit();
    }

    constexpr bool collective = requires {
        epilogue.template finish_tile<Schedule, FullTokens>(output, shared_raw, accumulators,
                                                            row_begin, token_begin, rows, tokens);
    };
    static_assert(RowPolicy::kContiguous || collective,
                  "reordered rows require a collective epilogue");
    if constexpr (collective) {
#pragma unroll
        for (int mt = 0; mt < Schedule::kMmaTokens; ++mt)
#pragma unroll
            for (int mr = 0; mr < Schedule::kMmaRows; ++mr)
#pragma unroll
                for (int v = 0; v < 4; ++v) accumulators[mt][mr][v] *= alpha;
        epilogue.template finish_tile<Schedule, FullTokens>(output, shared_raw, accumulators,
                                                            row_begin, token_begin, rows, tokens);
    } else {
        const int ar = lane >> 2, ac = 2 * (lane & 3);
        constexpr int stride = Schedule::kBlockRows + 8;
        auto* tile           = reinterpret_cast<__nv_bfloat16*>(shared_raw);
#pragma unroll
        for (int mt = 0; mt < Schedule::kMmaTokens; ++mt) {
            const int t0 = token_begin + warp_m * Schedule::kWarpTokens + mt * 16 + ar;
            const int t1 = t0 + 8;
#pragma unroll
            for (int mr = 0; mr < Schedule::kMmaRows; ++mr) {
                const int local = warp_n * Schedule::kWarpRows + mr * 8 + ac;
                const int r0    = row_policy.weight_row(row_begin, local, rows);
                const int r1    = row_policy.weight_row(row_begin, local + 1, rows);
                float a = accumulators[mt][mr][0] * alpha, b = accumulators[mt][mr][1] * alpha;
                float c = accumulators[mt][mr][2] * alpha, d = accumulators[mt][mr][3] * alpha;
                if (FullTokens || t0 < tokens) {
                    a = epilogue.apply(r0, t0, a);
                    b = epilogue.apply(r1, t0, b);
                }
                if (FullTokens || t1 < tokens) {
                    c = epilogue.apply(r0, t1, c);
                    d = epilogue.apply(r1, t1, d);
                }
                *reinterpret_cast<__nv_bfloat162*>(tile + (t0 - token_begin) * stride + local) =
                    __floats2bfloat162_rn(a, b);
                *reinterpret_cast<__nv_bfloat162*>(tile + (t1 - token_begin) * stride + local) =
                    __floats2bfloat162_rn(c, d);
            }
        }
        __syncthreads();
        for (int item = threadIdx.x; item < Schedule::kBlockTokens * (Schedule::kBlockRows / 8);
             item += Schedule::kThreads) {
            const int t = item / (Schedule::kBlockRows / 8),
                      r = (item % (Schedule::kBlockRows / 8)) * 8;
            if (FullTokens || token_begin + t < tokens)
                linear_store_bf16_vector(output, row_begin + r, token_begin + t,
                                         load_vec<uint4>(tile + t * stride + r));
        }
    }
}
} // namespace ninfer::ops::detail
