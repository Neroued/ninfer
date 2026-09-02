#pragma once

#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/nvfp4/nvfp4_output.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

struct alignas(128) Nvfp4W4a4TmaDescriptors {
    CUtensorMap a_codes;
    CUtensorMap b_codes;
    CUtensorMap a_scales;
    CUtensorMap b_scales;
};

inline void nvfp4_check_driver(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS) { return; }
    const char* name = nullptr;
    (void)cuGetErrorName(status, &name);
    throw std::runtime_error(std::string(operation) + ": " +
                             (name != nullptr ? name : "CUDA error"));
}

inline CUtensorMap nvfp4_make_tma_2d(void* address, CUtensorMapDataType data_type,
                                     std::uint64_t columns, std::uint64_t rows,
                                     std::uint64_t row_stride_bytes, std::uint32_t box_columns,
                                     std::uint32_t box_rows, CUtensorMapSwizzle swizzle,
                                     const char* operation) {
    CUtensorMap map{};
    const std::uint64_t global_dim[]     = {columns, rows};
    const std::uint64_t global_stride[]  = {row_stride_bytes};
    const std::uint32_t box_dim[]        = {box_columns, box_rows};
    const std::uint32_t element_stride[] = {1, 1};
    nvfp4_check_driver(
        cuTensorMapEncodeTiled(&map, data_type, 2, address, global_dim, global_stride, box_dim,
                               element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                               CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        operation);
    return map;
}

template <class Geometry, int BlockM>
Nvfp4W4a4TmaDescriptors make_nvfp4_w4a4_tma_descriptors(const std::uint8_t* activation_codes,
                                                        const std::uint8_t* activation_scales,
                                                        const std::uint8_t* weight_codes,
                                                        const std::uint8_t* weight_scales,
                                                        std::int32_t tokens) {
    static_assert(BlockM == 128 || BlockM == 256);
    constexpr std::uint32_t kCodeColumns = 64;
    // TMA's innermost box is at least one 16-byte transaction. A K128 tile consumes the
    // first eight bytes of each row; the second half is harmless look-ahead.
    constexpr std::uint32_t kScaleColumns = 16;
    constexpr std::uint32_t kBlockN       = 128;
    constexpr std::uint64_t kWeightScaleBytes =
        static_cast<std::uint64_t>(Geometry::kOutputRows) * Geometry::kInputRows / 16;

    Nvfp4W4a4TmaDescriptors descriptors{};
    descriptors.a_codes = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(activation_codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        Geometry::kCodeBytesPerRow, tokens, Geometry::kCodeBytesPerRow, kCodeColumns, BlockM,
        CU_TENSOR_MAP_SWIZZLE_64B, "encode activation codes TMA");
    descriptors.b_codes = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(weight_codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        Geometry::kCodeBytesPerRow, Geometry::kOutputRows, Geometry::kCodeBytesPerRow, kCodeColumns,
        kBlockN, CU_TENSOR_MAP_SWIZZLE_64B, "encode weight codes TMA");
    descriptors.a_scales = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(activation_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8,
        Geometry::kGroupsPerRow, tokens, Geometry::kGroupsPerRow, kScaleColumns, BlockM,
        CU_TENSOR_MAP_SWIZZLE_NONE, "encode activation scales TMA");
    descriptors.b_scales = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(weight_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8, 16,
        kWeightScaleBytes / 16, 16, 16, 64, CU_TENSOR_MAP_SWIZZLE_NONE, "encode weight scales TMA");
    return descriptors;
}

template <int BlockM, int Stages, int MinBlocksPerSm>
struct Nvfp4W4a4TmaSchedule {
    static_assert(BlockM == 128 || BlockM == 256);
    static_assert(Stages >= 2 && Stages <= 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kBlockM           = BlockM;
    static constexpr int kBlockN           = 128;
    static constexpr int kBlockK           = 128;
    static constexpr int kStages           = Stages;
    static constexpr int kWarpsM           = 4;
    static constexpr int kWarpsN           = 2;
    static constexpr int kConsumerWarps    = kWarpsM * kWarpsN;
    static constexpr int kConsumerThreads  = kConsumerWarps * 32;
    static constexpr int kProducerThreads  = BlockM == 256 ? 128 : 32;
    static constexpr int kThreads          = kConsumerThreads + kProducerThreads;
    static constexpr int kWarpM            = kBlockM / kWarpsM;
    static constexpr int kWarpN            = kBlockN / kWarpsN;
    static constexpr int kMmaM             = kWarpM / 16;
    static constexpr int kMmaN             = kWarpN / 8;
    static constexpr int kK64PerStage      = 2;
    static constexpr int kScaleWordsPerRow = 4;
    static constexpr int kCodeRowBytes     = 64;
    static constexpr int kMinBlocksPerSm   = MinBlocksPerSm;
};

template <class Schedule>
struct Nvfp4W4a4TmaTensorStorage {
    alignas(
        128) std::uint8_t a_codes[Schedule::kStages][Schedule::kBlockM * Schedule::kCodeRowBytes];
    alignas(
        128) std::uint8_t b_codes[Schedule::kStages][Schedule::kBlockN * Schedule::kCodeRowBytes];
    alignas(16)
        std::uint32_t a_scale4[Schedule::kStages][Schedule::kBlockM * Schedule::kScaleWordsPerRow];
    alignas(16)
        std::uint8_t b_scales[Schedule::kStages][Schedule::kBlockN * Schedule::kK64PerStage * 4];
};

template <class Schedule>
union alignas(128) Nvfp4W4a4TmaScratch {
    Nvfp4W4a4TmaTensorStorage<Schedule> tensors;
    __nv_bfloat16 output[Schedule::kBlockM * (Schedule::kBlockN + 8)];
};

template <class Schedule>
struct Nvfp4W4a4TmaSharedStorage {
    Nvfp4W4a4TmaScratch<Schedule> scratch;
    alignas(8) std::uint64_t full[Schedule::kStages];
    alignas(8) std::uint64_t empty[Schedule::kStages];
};

__device__ __forceinline__ void nvfp4_tma_load_2d(void* destination, const CUtensorMap* descriptor,
                                                  std::int32_t coordinate0,
                                                  std::int32_t coordinate1,
                                                  std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :
                 : "r"(smem_addr(destination)), "l"(descriptor), "r"(coordinate0), "r"(coordinate1),
                   "r"(smem_addr(barrier))
                 : "memory");
}

template <class Geometry, class Schedule, class Epilogue, class OutputPolicy>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_w4a4_tma_kernel(
    const __grid_constant__ Nvfp4W4a4TmaDescriptors descriptors, float alpha,
    const __grid_constant__ Epilogue epilogue, const __grid_constant__ OutputPolicy output) {
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    static_assert((Geometry::kOutputRows % Schedule::kBlockN) == 0);

    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& shared          = *reinterpret_cast<Nvfp4W4a4TmaSharedStorage<Schedule>*>(shared_bytes);
    const int token_begin = static_cast<int>(blockIdx.y) * Schedule::kBlockM;
    const int row_begin   = static_cast<int>(blockIdx.x) * Schedule::kBlockN;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Schedule::kStages; ++stage) {
            cta_mbarrier_init(&shared.full[stage], 1);
            cta_mbarrier_init(&shared.empty[stage], Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

    constexpr int kKTiles = Geometry::kInputRows / Schedule::kBlockK;

    if (threadIdx.x < Schedule::kProducerThreads) {
        if constexpr (Schedule::kProducerThreads == 128) {
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" : : : "memory");
        }
        if (threadIdx.x == 0) {
#pragma unroll 1
            for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
                const int stage                 = k_tile % Schedule::kStages;
                const std::uint32_t empty_phase = 1U ^ ((k_tile / Schedule::kStages) & 1U);
                cta_mbarrier_wait(&shared.empty[stage], empty_phase);
                constexpr std::uint32_t kTransactionBytes =
                    Schedule::kBlockM * Schedule::kCodeRowBytes +
                    Schedule::kBlockN * Schedule::kCodeRowBytes +
                    Schedule::kBlockM * Schedule::kScaleWordsPerRow * 4 +
                    Schedule::kBlockN * Schedule::kK64PerStage * 4;
                cta_mbarrier_arrive_expect_tx(&shared.full[stage], kTransactionBytes);

                auto& tensors = shared.scratch.tensors;
                nvfp4_tma_load_2d(tensors.a_codes[stage], &descriptors.a_codes,
                                  k_tile * Schedule::kCodeRowBytes, token_begin,
                                  &shared.full[stage]);
                nvfp4_tma_load_2d(tensors.b_codes[stage], &descriptors.b_codes,
                                  k_tile * Schedule::kCodeRowBytes, row_begin, &shared.full[stage]);
                nvfp4_tma_load_2d(tensors.a_scale4[stage], &descriptors.a_scales, (k_tile / 2) * 16,
                                  token_begin, &shared.full[stage]);
                const int b_scale_row = ((row_begin / 128) * Geometry::kScaleTilesPerRow +
                                         k_tile * Schedule::kK64PerStage) *
                                        32;
                nvfp4_tma_load_2d(tensors.b_scales[stage], &descriptors.b_scales, 0, b_scale_row,
                                  &shared.full[stage]);
            }
        }
        return;
    }

    if constexpr (Schedule::kProducerThreads == 128) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" : : : "memory");
    }
    auto& tensors             = shared.scratch.tensors;
    const int consumer_thread = static_cast<int>(threadIdx.x) - Schedule::kProducerThreads;
    const int lane            = consumer_thread & 31;
    const int warp            = consumer_thread >> 5;
    const int warp_m          = warp / Schedule::kWarpsN;
    const int warp_n          = warp - warp_m * Schedule::kWarpsN;

    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;

    float accumulators[Schedule::kMmaM][Schedule::kMmaN][4] = {};
#pragma unroll 1
    for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
        const int stage                = k_tile % Schedule::kStages;
        const std::uint32_t full_phase = (k_tile / Schedule::kStages) & 1U;
        cta_mbarrier_wait(&shared.full[stage], full_phase);

#pragma unroll
        for (int local_k64 = 0; local_k64 < Schedule::kK64PerStage; ++local_k64) {
            unsigned a_fragments[Schedule::kMmaM][4];
            unsigned b_fragments[Schedule::kMmaN][2];
            unsigned a_scales[Schedule::kMmaM];
            unsigned b_scales[Schedule::kMmaN];

#pragma unroll
            for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
                const int row          = warp_m * Schedule::kWarpM + mma_m * 16 + a_row_offset;
                const int logical_byte = local_k64 * 32 + a_column_byte;
                const int physical_byte =
                    ((logical_byte >> 4) ^ ((row >> 1) & 3)) * 16 + (logical_byte & 15);
                const auto* address =
                    tensors.a_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                ldmatrix_x4(a_fragments[mma_m][0], a_fragments[mma_m][1], a_fragments[mma_m][2],
                            a_fragments[mma_m][3], smem_addr(address));
                const int scale_row = warp_m * Schedule::kWarpM + mma_m * 16 + sfa_row;
                a_scales[mma_m] =
                    tensors.a_scale4[stage][scale_row * Schedule::kScaleWordsPerRow +
                                            (k_tile & 1) * Schedule::kK64PerStage + local_k64];
            }

#pragma unroll
            for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
                const int row          = warp_n * Schedule::kWarpN + mma_n * 8 + b_row_offset;
                const int logical_byte = local_k64 * 32 + b_column_byte;
                const int physical_byte =
                    ((logical_byte >> 4) ^ ((row >> 1) & 3)) * 16 + (logical_byte & 15);
                const auto* address =
                    tensors.b_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                ldmatrix_x2(b_fragments[mma_n][0], b_fragments[mma_n][1], smem_addr(address));
                const int scale_row    = warp_n * Schedule::kWarpN + mma_n * 8 + sfb_row;
                const int row_mod32    = scale_row & 31;
                const int row_quartile = scale_row >> 5;
                b_scales[mma_n]        = load_vec<unsigned>(
                    tensors.b_scales[stage] + (local_k64 * 32 + row_mod32) * 16 + row_quartile * 4);
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
        if (lane == 0) { cta_mbarrier_arrive(&shared.empty[stage]); }
    }

    // The epilogue reuses the tensor pipeline's shared-memory storage. All consumer
    // warps must finish their final tensor reads before any warp starts overwriting it.
    asm volatile("bar.sync 1, %0;" : : "r"(Schedule::kConsumerThreads) : "memory");

    constexpr int kOutputStride = Schedule::kBlockN + 8;
    auto* shared_output         = shared.scratch.output;
    const int accumulator_row   = lane >> 2;
    const int accumulator_col   = 2 * (lane & 3);
#pragma unroll
    for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
        const int token0 = warp_m * Schedule::kWarpM + mma_m * 16 + accumulator_row;
        const int token1 = token0 + 8;
#pragma unroll
        for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
            const int parent_row = warp_n * Schedule::kWarpN + mma_n * 8 + accumulator_col;
            auto* destination0   = reinterpret_cast<__nv_bfloat162*>(
                shared_output + token0 * kOutputStride + parent_row);
            auto* destination1 = reinterpret_cast<__nv_bfloat162*>(
                shared_output + token1 * kOutputStride + parent_row);
            const int global_row0   = row_begin + parent_row;
            const int global_row1   = global_row0 + 1;
            const int global_token0 = token_begin + token0;
            const int global_token1 = token_begin + token1;
            const float value00 =
                epilogue.apply(global_row0, global_token0, accumulators[mma_m][mma_n][0] * alpha);
            const float value01 =
                epilogue.apply(global_row1, global_token0, accumulators[mma_m][mma_n][1] * alpha);
            const float value10 =
                epilogue.apply(global_row0, global_token1, accumulators[mma_m][mma_n][2] * alpha);
            const float value11 =
                epilogue.apply(global_row1, global_token1, accumulators[mma_m][mma_n][3] * alpha);
            *destination0 = __floats2bfloat162_rn(value00, value01);
            *destination1 = __floats2bfloat162_rn(value10, value11);
        }
    }

    asm volatile("bar.sync 1, %0;" : : "r"(Schedule::kConsumerThreads) : "memory");
    constexpr int kVectorsPerRow = Schedule::kBlockN / 8;
    constexpr int kOutputVectors = Schedule::kBlockM * kVectorsPerRow;
    for (int task = consumer_thread; task < kOutputVectors; task += Schedule::kConsumerThreads) {
        const int token_local = task / kVectorsPerRow;
        const int row_vector  = task - token_local * kVectorsPerRow;
        const int token       = token_begin + token_local;
        const uint4 values =
            load_vec<uint4>(shared_output + token_local * kOutputStride + row_vector * 8);
        output.store_vector(row_begin + row_vector * 8, token, values);
    }
}

} // namespace ninfer::ops::detail
