#pragma once

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_output.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int BlockM, int BlockN, int BlockK, int WarpsM, int WarpsN, int Stages,
          int MinBlocksPerSm>
struct Nvfp4W4a4MmaSchedule {
    static_assert(BlockM > 0 && (BlockM % 16) == 0);
    static_assert(BlockN > 0 && (BlockN % 8) == 0);
    static_assert(BlockK >= 64 && (BlockK % 64) == 0);
    static_assert(WarpsM > 0 && WarpsN > 0);
    static_assert((BlockM % WarpsM) == 0 && ((BlockM / WarpsM) % 16) == 0);
    static_assert((BlockN % WarpsN) == 0 && ((BlockN / WarpsN) % 8) == 0);
    static_assert(Stages >= 1 && Stages <= 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kBlockM         = BlockM;
    static constexpr int kBlockN         = BlockN;
    static constexpr int kBlockK         = BlockK;
    static constexpr int kWarpsM         = WarpsM;
    static constexpr int kWarpsN         = WarpsN;
    static constexpr int kStages         = Stages;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
    static constexpr int kWarps          = WarpsM * WarpsN;
    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kWarpM          = BlockM / WarpsM;
    static constexpr int kWarpN          = BlockN / WarpsN;
    static constexpr int kMmaM           = kWarpM / 16;
    static constexpr int kMmaN           = kWarpN / 8;
    static constexpr int kK64PerStage    = BlockK / 64;
    static constexpr int kCodeRowBytes   = BlockK / 2;
    static constexpr int kSegmentsPerRow = kCodeRowBytes / 16;
};

struct Nvfp4W4a4MaterializedActivation {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
};

struct Nvfp4W4a4IdentityRows {
    static constexpr bool kContiguous = true;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + local_row;
    }
};

template <class Schedule>
struct Nvfp4W4a4SharedStorage {
    alignas(
        16) std::uint8_t a_codes[Schedule::kStages][Schedule::kBlockM * Schedule::kCodeRowBytes];
    alignas(
        16) std::uint8_t b_codes[Schedule::kStages][Schedule::kBlockN * Schedule::kCodeRowBytes];
    alignas(
        16) std::uint32_t a_scale4[Schedule::kStages][Schedule::kBlockM * Schedule::kK64PerStage];
    alignas(16)
        std::uint8_t b_scales[Schedule::kStages][Schedule::kBlockN * Schedule::kK64PerStage * 4];
};

template <class Schedule>
__device__ __forceinline__ int nvfp4_w4a4_swizzled_byte(int row, int logical_byte) {
    static_assert((Schedule::kSegmentsPerRow & (Schedule::kSegmentsPerRow - 1)) == 0);
    const int logical_segment  = logical_byte >> 4;
    const int byte_in_segment  = logical_byte & 15;
    const int physical_segment = logical_segment ^ (row & (Schedule::kSegmentsPerRow - 1));
    return physical_segment * 16 + byte_in_segment;
}

template <class Geometry, class Schedule>
__device__ __forceinline__ void
stage_nvfp4_w4a4_activation(Nvfp4W4a4MaterializedActivation source,
                            Nvfp4W4a4SharedStorage<Schedule>& shared, int stage, int k_tile,
                            int token_begin, int active_tokens) {
    constexpr int kCodeTasks = Schedule::kBlockM * Schedule::kSegmentsPerRow;
    for (int task = static_cast<int>(threadIdx.x); task < kCodeTasks; task += Schedule::kThreads) {
        const int row             = task / Schedule::kSegmentsPerRow;
        const int logical_segment = task - row * Schedule::kSegmentsPerRow;
        const int token           = token_begin + row;
        const bool valid          = token < active_tokens;
        const int source_token    = valid ? token : 0;
        const int physical_byte   = nvfp4_w4a4_swizzled_byte<Schedule>(row, logical_segment * 16);
        auto* destination = shared.a_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
        const auto* input = source.codes +
                            static_cast<std::int64_t>(source_token) * Geometry::kCodeBytesPerRow +
                            k_tile * Schedule::kCodeRowBytes + logical_segment * 16;
        cp_async_zfill<16, Cache::cg>(destination, input, valid ? 16 : 0);
    }

    if constexpr (Schedule::kK64PerStage <= 4) {
        constexpr int kScaleBytes = Schedule::kK64PerStage * 4;
        for (int row = static_cast<int>(threadIdx.x); row < Schedule::kBlockM;
             row += Schedule::kThreads) {
            const int token        = token_begin + row;
            const bool valid       = token < active_tokens;
            const int source_token = valid ? token : 0;
            auto* destination      = &shared.a_scale4[stage][row * Schedule::kK64PerStage];
            const auto* input      = source.scales + (static_cast<std::int64_t>(source_token) *
                                                     Geometry::kScaleTilesPerRow +
                                                 k_tile * Schedule::kK64PerStage) *
                                                    4;
            cp_async_zfill<kScaleBytes>(destination, input, valid ? kScaleBytes : 0);
        }
    } else {
        static_assert((Schedule::kK64PerStage % 4) == 0);
        constexpr int kSegmentsPerRow = Schedule::kK64PerStage / 4;
        constexpr int kScaleTasks     = Schedule::kBlockM * kSegmentsPerRow;
        for (int task = static_cast<int>(threadIdx.x); task < kScaleTasks;
             task += Schedule::kThreads) {
            const int row          = task / kSegmentsPerRow;
            const int segment      = task - row * kSegmentsPerRow;
            const int token        = token_begin + row;
            const bool valid       = token < active_tokens;
            const int source_token = valid ? token : 0;
            const int local_k64    = segment * 4;
            auto* destination = &shared.a_scale4[stage][row * Schedule::kK64PerStage + local_k64];
            const auto* input = source.scales + (static_cast<std::int64_t>(source_token) *
                                                     Geometry::kScaleTilesPerRow +
                                                 k_tile * Schedule::kK64PerStage + local_k64) *
                                                    4;
            cp_async_zfill<16>(destination, input, valid ? 16 : 0);
        }
    }
}

template <class Geometry, class Schedule, class RowPolicy>
__device__ __forceinline__ void stage_nvfp4_w4a4_weight(const std::uint8_t* __restrict__ codes,
                                                        const std::uint8_t* __restrict__ scales,
                                                        Nvfp4W4a4SharedStorage<Schedule>& shared,
                                                        int stage, int k_tile, int row_begin,
                                                        RowPolicy row_policy) {
    constexpr int kCodeTasks = Schedule::kBlockN * Schedule::kSegmentsPerRow;
    for (int task = static_cast<int>(threadIdx.x); task < kCodeTasks; task += Schedule::kThreads) {
        const int row             = task / Schedule::kSegmentsPerRow;
        const int logical_segment = task - row * Schedule::kSegmentsPerRow;
        const int weight_row      = row_policy.weight_row(row_begin, row);
        const int physical_byte   = nvfp4_w4a4_swizzled_byte<Schedule>(row, logical_segment * 16);
        auto* destination = shared.b_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
        const auto* input = codes +
                            static_cast<std::int64_t>(weight_row) * Geometry::kCodeBytesPerRow +
                            k_tile * Schedule::kCodeRowBytes + logical_segment * 16;
        cp_async<16, Cache::cg>(destination, input);
    }

    if constexpr (RowPolicy::kContiguous) {
        static_assert((Schedule::kBlockN % 64) == 0);
        constexpr int kScaleRowTiles     = Schedule::kBlockN >= 128 ? Schedule::kBlockN / 128 : 1;
        constexpr int kQuartilesPerTile  = Schedule::kBlockN >= 128 ? 4 : Schedule::kBlockN / 32;
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
            const auto* input = scales +
                                static_cast<std::int64_t>(
                                    persistent_tile * Geometry::kScaleTilesPerRow + global_k64) *
                                    512 +
                                row_mod32 * 16 + first_quartile * 4;
            cp_async<kScaleBytesPerTask>(destination, input);
        }
    } else {
        constexpr int kScaleTasks = Schedule::kBlockN * Schedule::kK64PerStage;
        for (int task = static_cast<int>(threadIdx.x); task < kScaleTasks;
             task += Schedule::kThreads) {
            const int row             = task / Schedule::kK64PerStage;
            const int local_k64       = task - row * Schedule::kK64PerStage;
            const int weight_row      = row_policy.weight_row(row_begin, row);
            const int global_k64      = k_tile * Schedule::kK64PerStage + local_k64;
            const int persistent_tile = weight_row / 128;
            const int row_in_tile     = weight_row & 127;
            const int row_mod32       = row_in_tile & 31;
            const int row_quartile    = row_in_tile >> 5;
            auto* destination =
                shared.b_scales[stage] + (row * Schedule::kK64PerStage + local_k64) * 4;
            const auto* input = scales +
                                static_cast<std::int64_t>(
                                    persistent_tile * Geometry::kScaleTilesPerRow + global_k64) *
                                    512 +
                                row_mod32 * 16 + row_quartile * 4;
            cp_async<4>(destination, input);
        }
    }
}

template <class Geometry, class Schedule, class Epilogue, class OutputPolicy,
          class RowPolicy = Nvfp4W4a4IdentityRows, bool PairRows = false>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_w4a4_mma_kernel(
    Nvfp4W4a4MaterializedActivation activation, const std::uint8_t* __restrict__ weight_codes,
    const std::uint8_t* __restrict__ weight_scales, std::int32_t tokens, float alpha,
    Epilogue epilogue, OutputPolicy output, RowPolicy row_policy = {}) {
// mma_bf16/ldmatrix (mma.cuh) already trap below sm_80, but this kernel's
// __shared__ Nvfp4W4a4SharedStorage<Schedule> below is large enough at several of the
// Schedule instantiations this port's callers use that nvlink rejects it against Volta's
// 48KB static-shared cap -- a link-time failure independent of whether the mma-using
// code path is ever reached (nvfp4 is permanently out of scope regardless -- no FP4
// hardware on Volta -- see docs/v100.md). Same fix as
// w8_rowsplit_gemm_medium_t_splitk.cuh: the whole body needs to be behind the guard,
// not just the mma calls within it.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    static_assert((Geometry::kOutputRows % Schedule::kBlockN) == 0);
    static_assert(!PairRows || (Schedule::kBlockN % 2) == 0);
    static_assert(!PairRows || ((Geometry::kOutputRows / 2) % (Schedule::kBlockN / 2)) == 0);

    __shared__ Nvfp4W4a4SharedStorage<Schedule> shared;
    const int token_begin       = static_cast<int>(blockIdx.y) * Schedule::kBlockM;
    constexpr int kRowsPerBlock = PairRows ? Schedule::kBlockN / 2 : Schedule::kBlockN;
    const int row_begin         = static_cast<int>(blockIdx.x) * kRowsPerBlock;
    constexpr int kKTiles       = Geometry::kInputRows / Schedule::kBlockK;
    constexpr int kWaitGroups   = kKTiles < Schedule::kStages ? kKTiles - 1 : Schedule::kStages - 1;

#pragma unroll
    for (int stage = 0; stage < Schedule::kStages; ++stage) {
        if (stage < kKTiles) {
            stage_nvfp4_w4a4_activation<Geometry, Schedule>(activation, shared, stage, stage,
                                                            token_begin, tokens);
            stage_nvfp4_w4a4_weight<Geometry, Schedule>(weight_codes, weight_scales, shared, stage,
                                                        stage, row_begin, row_policy);
            cp_commit();
        }
    }

    float accumulators[Schedule::kMmaM][Schedule::kMmaN][4] = {};
    const int lane                                          = static_cast<int>(threadIdx.x) & 31;
    const int warp                                          = static_cast<int>(threadIdx.x) >> 5;
    const int warp_m                                        = warp / Schedule::kWarpsN;
    const int warp_n                                        = warp - warp_m * Schedule::kWarpsN;

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
            unsigned a_fragments[Schedule::kMmaM][4];
            unsigned b_fragments[Schedule::kMmaN][2];
            unsigned a_scales[Schedule::kMmaM];
            unsigned b_scales[Schedule::kMmaN];

#pragma unroll
            for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
                const int row           = warp_m * Schedule::kWarpM + mma_m * 16 + a_row_offset;
                const int logical_byte  = local_k64 * 32 + a_column_byte;
                const int physical_byte = nvfp4_w4a4_swizzled_byte<Schedule>(row, logical_byte);
                const auto* address =
                    shared.a_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                ldmatrix_x4(a_fragments[mma_m][0], a_fragments[mma_m][1], a_fragments[mma_m][2],
                            a_fragments[mma_m][3], smem_addr(address));
                const int scale_row = warp_m * Schedule::kWarpM + mma_m * 16 + sfa_row;
                a_scales[mma_m] =
                    shared.a_scale4[stage][scale_row * Schedule::kK64PerStage + local_k64];
            }

#pragma unroll
            for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
                const int row           = warp_n * Schedule::kWarpN + mma_n * 8 + b_row_offset;
                const int logical_byte  = local_k64 * 32 + b_column_byte;
                const int physical_byte = nvfp4_w4a4_swizzled_byte<Schedule>(row, logical_byte);
                const auto* address =
                    shared.b_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                ldmatrix_x2(b_fragments[mma_n][0], b_fragments[mma_n][1], smem_addr(address));
                const int scale_row = warp_n * Schedule::kWarpN + mma_n * 8 + sfb_row;
                const std::uint8_t* scale_address;
                if constexpr (RowPolicy::kContiguous) {
                    constexpr int kQuartilesPerTile =
                        Schedule::kBlockN >= 128 ? 4 : Schedule::kBlockN / 32;
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
            for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
#pragma unroll
                for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
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
            stage_nvfp4_w4a4_activation<Geometry, Schedule>(activation, shared, stage, next_k_tile,
                                                            token_begin, tokens);
            stage_nvfp4_w4a4_weight<Geometry, Schedule>(weight_codes, weight_scales, shared, stage,
                                                        next_k_tile, row_begin, row_policy);
        }
        cp_commit();
    }

    const int accumulator_row   = lane >> 2;
    const int accumulator_col   = 2 * (lane & 3);
    constexpr int kOutputStride = Schedule::kBlockN + 8;
    static_assert(sizeof(Nvfp4W4a4SharedStorage<Schedule>) >=
                  Schedule::kBlockM * kOutputStride * sizeof(__nv_bfloat16));
    auto* shared_output = reinterpret_cast<__nv_bfloat16*>(&shared);
#pragma unroll
    for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
        const int token0 = token_begin + warp_m * Schedule::kWarpM + mma_m * 16 + accumulator_row;
        const int token1 = token0 + 8;
#pragma unroll
        for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
            const int local_row0  = warp_n * Schedule::kWarpN + mma_n * 8 + accumulator_col;
            const int parent_row0 = row_policy.weight_row(row_begin, local_row0);
            auto* destination0    = reinterpret_cast<__nv_bfloat162*>(
                shared_output + (token0 - token_begin) * kOutputStride + local_row0);
            auto* destination1 = reinterpret_cast<__nv_bfloat162*>(
                shared_output + (token1 - token_begin) * kOutputStride + local_row0);
            const int parent_row1 = row_policy.weight_row(row_begin, local_row0 + 1);
            float value00         = accumulators[mma_m][mma_n][0] * alpha;
            float value01         = accumulators[mma_m][mma_n][1] * alpha;
            float value10         = accumulators[mma_m][mma_n][2] * alpha;
            float value11         = accumulators[mma_m][mma_n][3] * alpha;
            if (token0 < tokens) {
                value00 = epilogue.apply(parent_row0, token0, value00);
                value01 = epilogue.apply(parent_row1, token0, value01);
            }
            if (token1 < tokens) {
                value10 = epilogue.apply(parent_row0, token1, value10);
                value11 = epilogue.apply(parent_row1, token1, value11);
            }
            *destination0 = __floats2bfloat162_rn(value00, value01);
            *destination1 = __floats2bfloat162_rn(value10, value11);
        }
    }
    __syncthreads();
    constexpr int kStoredRows    = PairRows ? Schedule::kBlockN / 2 : Schedule::kBlockN;
    constexpr int kVectorsPerRow = kStoredRows / 8;
    constexpr int kOutputVectors = Schedule::kBlockM * kVectorsPerRow;
    for (int task = static_cast<int>(threadIdx.x); task < kOutputVectors;
         task += Schedule::kThreads) {
        const int token_local = task / kVectorsPerRow;
        const int row_vector  = task - token_local * kVectorsPerRow;
        const int token       = token_begin + token_local;
        if (token < tokens) {
            const uint4 values =
                load_vec<uint4>(shared_output + token_local * kOutputStride + row_vector * 8);
            if constexpr (PairRows) {
                const uint4 paired = load_vec<uint4>(shared_output + token_local * kOutputStride +
                                                     kStoredRows + row_vector * 8);
                output.store_pair_vector(row_begin + row_vector * 8, token, values, paired);
            } else {
                output.store_vector(row_begin + row_vector * 8, token, values);
            }
        }
    }
#else  // __CUDA_ARCH__ < 800
    (void)activation;
    (void)weight_codes;
    (void)weight_scales;
    (void)tokens;
    (void)alpha;
    (void)epilogue;
    (void)output;
    (void)row_policy;
    __trap();
#endif // __CUDA_ARCH__ >= 800
}

template <class Geometry, int Threads = 256>
__global__ __launch_bounds__(Threads, 512 / Threads) void nvfp4_w4a4_quantize_kernel(
    const __nv_bfloat16* __restrict__ input, std::uint8_t* __restrict__ codes,
    std::uint8_t* __restrict__ scales, std::int32_t tokens, float input_scale_divisor) {
    static_assert(Threads == 128 || Threads == 256 || Threads == 512);
    constexpr int kGroupsPerRow = Geometry::kInputRows / 16;
    const int task =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int tasks = tokens * kGroupsPerRow;
    if (task >= tasks) { return; }

    const int token                   = task / kGroupsPerRow;
    const int group                   = task - token * kGroupsPerRow;
    const Nvfp4QuantizedK16 quantized = quantize_nvfp4_k16(
        input + static_cast<std::int64_t>(token) * Geometry::kInputRows + group * 16,
        input_scale_divisor);
    auto* code_destination =
        codes + static_cast<std::int64_t>(token) * Geometry::kCodeBytesPerRow + group * 8;
    store_vec(code_destination, make_uint2(quantized.codes_lo, quantized.codes_hi));
    scales[static_cast<std::int64_t>(token) * kGroupsPerRow + group] = quantized.scale;
}

} // namespace ninfer::ops::detail
