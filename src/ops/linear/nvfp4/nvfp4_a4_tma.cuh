#pragma once

#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear/nvfp4/nvfp4_shared.cuh"
#include "ops/common/token_slices.h"
#include "ops/common/math.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/vector_output.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

struct alignas(128) Nvfp4A4TmaDescriptors {
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

inline CUtensorMap
nvfp4_make_tma_2d(void* address, CUtensorMapDataType data_type, std::uint64_t columns,
                  std::uint64_t rows, std::uint64_t row_stride_bytes, std::uint32_t box_columns,
                  std::uint32_t box_rows, CUtensorMapSwizzle swizzle, const char* operation,
                  CUtensorMapL2promotion l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE) {
    CUtensorMap map{};
    const std::uint64_t global_dim[]     = {columns, rows};
    const std::uint64_t global_stride[]  = {row_stride_bytes};
    const std::uint32_t box_dim[]        = {box_columns, box_rows};
    const std::uint32_t element_stride[] = {1, 1};
    nvfp4_check_driver(cuTensorMapEncodeTiled(&map, data_type, 2, address, global_dim,
                                              global_stride, box_dim, element_stride,
                                              CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2_promotion,
                                              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                       operation);
    return map;
}

template <class Schedule, class Rows>
Nvfp4A4TmaDescriptors make_nvfp4_a4_tma_descriptors(const Nvfp4A4Operands& p) {
    constexpr int tile      = Schedule::kBlockTokens;
    constexpr int code_rows = Rows::kPaired ? Schedule::kBlockRows / 2 : Schedule::kBlockRows;
    Nvfp4A4TmaDescriptors d{};
    d.a_codes = nvfp4_make_tma_2d(const_cast<std::uint8_t*>(p.x), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                  p.k / 2, p.tokens, p.k / 2, 64, tile, CU_TENSOR_MAP_SWIZZLE_64B,
                                  "encode activation codes TMA");
    d.b_codes =
        nvfp4_make_tma_2d(const_cast<std::uint8_t*>(p.codes), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                          p.k / 2, p.rows, p.k / 2, 64, code_rows, CU_TENSOR_MAP_SWIZZLE_64B,
                          "encode weight codes TMA", Schedule::kWeightCodePromotion);
    // The quantizer and descriptor agree on both token and group tile dimensions.
    d.a_scales = nvfp4_make_tma_2d(
        const_cast<std::uint8_t*>(p.x_scales), CU_TENSOR_MAP_DATA_TYPE_UINT8, tile,
        static_cast<std::uint64_t>(nvfp4_a4_padded_tokens(p.tokens, Schedule::kScaleLayout) /
                                   tile) *
            (p.k / 256) * 16,
        tile, tile, 16, CU_TENSOR_MAP_SWIZZLE_NONE, "encode activation scales TMA");
    d.b_scales =
        nvfp4_make_tma_2d(const_cast<std::uint8_t*>(p.scales), CU_TENSOR_MAP_DATA_TYPE_UINT8, 16,
                          static_cast<std::uint64_t>(p.rows) * p.k / 256, 16, 16, 64,
                          CU_TENSOR_MAP_SWIZZLE_NONE, "encode weight scales TMA");
    return d;
}

template <class Schedule, class Rows>
struct Nvfp4A4TmaTensorStorage {
    alignas(128)
        std::uint8_t a_codes[Schedule::kStages][Schedule::kBlockTokens * Schedule::kCodeRowBytes];
    alignas(128)
        std::uint8_t b_codes[Schedule::kStages][Schedule::kBlockRows * Schedule::kCodeRowBytes];
    alignas(16) std::uint32_t
        a_scale4[Schedule::kScaleSlots][Schedule::kBlockTokens * Schedule::kScaleWordsPerRow];
    alignas(16) std::uint8_t b_scales[Schedule::kStages][Rows::kPaired ? 2 : 1]
                                     [Schedule::kBlockRows * Schedule::kK64PerStage * 4];
};

template <class Schedule, class Rows, class Epilogue>
union alignas(128) Nvfp4A4TmaScratch {
    Nvfp4A4TmaTensorStorage<Schedule, Rows> tensors;
    static constexpr int kEpilogueBytes = [] {
        if constexpr (requires { Epilogue::template kSharedBytes<Schedule>; })
            return Epilogue::template kSharedBytes<Schedule>;
        else
            return 1;
    }();
    unsigned char epilogue_scratch[kEpilogueBytes];
    __nv_bfloat16 output[Schedule::kBlockTokens * (Schedule::kBlockRows + 8)];
};

template <class Schedule, class Rows, class Epilogue>
struct Nvfp4A4TmaSharedStorage {
    Nvfp4A4TmaScratch<Schedule, Rows, Epilogue> scratch;
    alignas(8) std::uint64_t full[Schedule::kStages];
    alignas(8) std::uint64_t empty[Schedule::kStages];
};

// The work distributor hands CTAs to SMs in linear order with blockIdx.x fastest, so the
// stock grid -- x over weight-row tiles, y over token tiles -- puts a different weight tile
// in every CTA that runs at the same time, and the whole weight matrix is re-read from
// memory once per token tile. Walking the token index fastest instead makes the CTAs that
// share a weight tile run together, and the matrix is read once. bf16_gemm_mma_kernel
// already makes this choice; Bf16MmaRaster::TokenFast is the default for every bf16
// schedule in the tree.
__device__ __forceinline__ void nvfp4_tma_raster_blocks(int& block_x, int& block_y) {
    const int rows = static_cast<int>(gridDim.y);
    const int linear =
        static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) + static_cast<int>(blockIdx.x);
    block_y = linear % rows;
    block_x = linear / rows;
}

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

template <class Schedule, class Epilogue, class OutputPolicy, class Rows>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_a4_tma_kernel(
    const __grid_constant__ Nvfp4A4TmaDescriptors descriptors, float alpha,
    const __grid_constant__ Epilogue epilogue, const __grid_constant__ OutputPolicy output,
    int token_count, int output_rows, int input_rows, int token_offset) {
    const int K               = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    constexpr int branches    = Rows::kPaired ? 2 : 1;
    constexpr int loaded_rows = Schedule::kBlockRows / branches;
    static_assert(Schedule::kStages >= 2, "the activation-scale buffer needs two slots");

    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& shared =
        *reinterpret_cast<Nvfp4A4TmaSharedStorage<Schedule, Rows, Epilogue>*>(shared_bytes);
    int block_x = 0;
    int block_y = 0;
    nvfp4_tma_raster_blocks(block_x, block_y);
    const int token_begin = token_offset + block_y * Schedule::kBlockTokens;
    const int row_begin   = block_x * loaded_rows;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Schedule::kStages; ++stage) {
            cta_mbarrier_init(&shared.full[stage], 1);
            cta_mbarrier_init(&shared.empty[stage], Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

    const int kKTiles = K / Schedule::kBlockK;

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
                constexpr std::uint32_t kScaleBytes =
                    Schedule::kBlockTokens * Schedule::kScaleWordsPerRow * 4;
                constexpr std::uint32_t kTransactionBytes =
                    Schedule::kBlockTokens * Schedule::kCodeRowBytes +
                    Schedule::kBlockRows * Schedule::kCodeRowBytes + kScaleBytes +
                    branches * Schedule::kBlockRows * Schedule::kK64PerStage * 4;
                // A scale tile covers kNvfp4ScaleTileGroups groups, which is two K tiles, so the
                // box is fetched on the even tile only and the odd tile expects that many bytes
                // fewer.
                const bool load_scales = (k_tile & 1) == 0;
                cta_mbarrier_arrive_expect_tx(&shared.full[stage],
                                              load_scales ? kTransactionBytes
                                                          : kTransactionBytes - kScaleBytes);

                auto& tensors = shared.scratch.tensors;
                nvfp4_tma_load_2d(tensors.a_codes[stage], &descriptors.a_codes,
                                  k_tile * Schedule::kCodeRowBytes, token_begin,
                                  &shared.full[stage]);
#pragma unroll
                for (int branch = 0; branch < branches; ++branch)
                    nvfp4_tma_load_2d(tensors.b_codes[stage] +
                                          branch * loaded_rows * Schedule::kCodeRowBytes,
                                      &descriptors.b_codes, k_tile * Schedule::kCodeRowBytes,
                                      row_begin + branch * (output_rows / 2), &shared.full[stage]);
                if (load_scales) {
                    // The box is tile-contiguous, so its address is a tile index rather than a
                    // (byte column, token row) pair. Scale slots cover both consumers of each pair.
                    const int kScaleTilesPerPlane = K / 256;
                    const int scale_tile =
                        (token_begin / Schedule::kBlockTokens) * kScaleTilesPerPlane + k_tile / 2;
                    nvfp4_tma_load_2d(tensors.a_scale4[(k_tile / 2) % Schedule::kScaleSlots],
                                      &descriptors.a_scales, 0, scale_tile * 16,
                                      &shared.full[stage]);
                }
#pragma unroll
                for (int branch = 0; branch < branches; ++branch) {
                    const int scale_row =
                        (((row_begin + branch * (output_rows / 2)) / 128) * (K / 64) +
                         k_tile * Schedule::kK64PerStage) *
                        32;
                    nvfp4_tma_load_2d(tensors.b_scales[stage][branch], &descriptors.b_scales, 0,
                                      scale_row, &shared.full[stage]);
                }
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
    const int warp_m          = warp / Schedule::kWarpsRows;
    const int warp_n          = warp - warp_m * Schedule::kWarpsRows;

    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;

    float accumulators[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
#pragma unroll 1
    for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
        const int stage                = k_tile % Schedule::kStages;
        const std::uint32_t full_phase = (k_tile / Schedule::kStages) & 1U;
        cta_mbarrier_wait(&shared.full[stage], full_phase);

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
                const int physical_byte =
                    ((logical_byte >> 4) ^ ((row >> 1) & 3)) * 16 + (logical_byte & 15);
                const auto* address =
                    tensors.a_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                ldmatrix_x4(a_fragments[mma_m][0], a_fragments[mma_m][1], a_fragments[mma_m][2],
                            a_fragments[mma_m][3], smem_addr(address));
                const int scale_row = warp_m * Schedule::kWarpTokens + mma_m * 16 + sfa_row;
                a_scales[mma_m] =
                    tensors.a_scale4[(k_tile / 2) % Schedule::kScaleSlots]
                                    [scale_row * Schedule::kScaleWordsPerRow +
                                     (k_tile & 1) * Schedule::kK64PerStage + local_k64];
            }

#pragma unroll
            for (int mma_n = 0; mma_n < Schedule::kMmaRows; ++mma_n) {
                constexpr int per_branch = Schedule::kMmaRows / branches;
                const int branch = mma_n / per_branch, fragment = mma_n % per_branch;
                const int pair_row     = warp_n * (Schedule::kWarpRows / branches) + fragment * 8;
                const int row          = branch * loaded_rows + pair_row + b_row_offset;
                const int logical_byte = local_k64 * 32 + b_column_byte;
                const int physical_byte =
                    ((logical_byte >> 4) ^ ((row >> 1) & 3)) * 16 + (logical_byte & 15);
                const auto* address =
                    tensors.b_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
                ldmatrix_x2(b_fragments[mma_n][0], b_fragments[mma_n][1], smem_addr(address));
                const int scale_row = (row_begin & 127) + pair_row + sfb_row;
                b_scales[mma_n]     = load_vec<unsigned>(tensors.b_scales[stage][branch] +
                                                         (local_k64 * 32 + (scale_row & 31)) * 16 +
                                                         (scale_row >> 5) * 4);
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
        if (lane == 0) { cta_mbarrier_arrive(&shared.empty[stage]); }
    }

    // The epilogue reuses the tensor pipeline's shared-memory storage. All consumer
    // warps must finish their final tensor reads before any warp starts overwriting it.
    asm volatile("bar.sync 1, %0;" : : "r"(Schedule::kConsumerThreads) : "memory");

    constexpr bool collective = requires {
        epilogue.template finish_tile<Schedule, false>(
            output, shared_bytes, accumulators, row_begin, token_begin, output_rows, token_count);
    };
    static_assert(!Rows::kPaired || collective, "paired TMA rows require a collective epilogue");
    if constexpr (collective) {
#pragma unroll
        for (int mt = 0; mt < Schedule::kMmaTokens; ++mt)
#pragma unroll
            for (int mr = 0; mr < Schedule::kMmaRows; ++mr)
#pragma unroll
                for (int v = 0; v < 4; ++v) accumulators[mt][mr][v] *= alpha;
        epilogue.template finish_tile<Schedule, false>(
            output, shared_bytes, accumulators, row_begin, token_begin, output_rows, token_count);
    } else {
        constexpr int stride = Schedule::kBlockRows + 8;
        auto* tile           = shared.scratch.output;
        const int ar = lane >> 2, ac = 2 * (lane & 3);
#pragma unroll
        for (int mt = 0; mt < Schedule::kMmaTokens; ++mt) {
            const int t0 = warp_m * Schedule::kWarpTokens + mt * 16 + ar, t1 = t0 + 8;
#pragma unroll
            for (int mr = 0; mr < Schedule::kMmaRows; ++mr) {
                const int row = warp_n * Schedule::kWarpRows + mr * 8 + ac;
                float a = accumulators[mt][mr][0] * alpha, b = accumulators[mt][mr][1] * alpha;
                float c = accumulators[mt][mr][2] * alpha, d = accumulators[mt][mr][3] * alpha;
                if (token_begin + t0 < token_count) {
                    a = epilogue.apply(row_begin + row, token_begin + t0, a);
                    b = epilogue.apply(row_begin + row + 1, token_begin + t0, b);
                }
                if (token_begin + t1 < token_count) {
                    c = epilogue.apply(row_begin + row, token_begin + t1, c);
                    d = epilogue.apply(row_begin + row + 1, token_begin + t1, d);
                }
                *reinterpret_cast<__nv_bfloat162*>(tile + t0 * stride + row) =
                    __floats2bfloat162_rn(a, b);
                *reinterpret_cast<__nv_bfloat162*>(tile + t1 * stride + row) =
                    __floats2bfloat162_rn(c, d);
            }
        }
        asm volatile("bar.sync 1, %0;" ::"r"(Schedule::kConsumerThreads) : "memory");
        for (int item = consumer_thread; item < Schedule::kBlockTokens * (Schedule::kBlockRows / 8);
             item += Schedule::kConsumerThreads) {
            const int t = item / (Schedule::kBlockRows / 8),
                      r = (item % (Schedule::kBlockRows / 8)) * 8;
            if (token_begin + t < token_count)
                linear_store_bf16_vector(output, row_begin + r, token_begin + t,
                                         load_vec<uint4>(tile + t * stride + r));
        }
    }
}

template <class Schedule, class Output, class Epilogue, class Rows = Nvfp4IdentityRows>
void launch_nvfp4_a4_tma_mma(const Nvfp4A4Operands& p, Output output, Epilogue epilogue,
                             cudaStream_t stream, Rows = {}) {
    validate_nvfp4_operands<Schedule>(p);
    static_assert(
        Rows::kContiguous ||
            [] {
                if constexpr (requires { Rows::kWarpPaired; })
                    return Rows::kWarpPaired;
                else
                    return false;
            }(),
        "TMA supports contiguous rows or warp-paired branches");
    if (p.rows % Schedule::kBlockRows || p.k % 256 || (Rows::kPaired && p.rows % 256) ||
        p.scale_layout != Schedule::kScaleLayout)
        throw std::invalid_argument(
            "NVFP4 TMA operands do not match row/K tiles or activation-scale layout");
    const auto aligned = [](const void* v) {
        return reinterpret_cast<std::uintptr_t>(v) % 16 == 0;
    };
    if (!aligned(p.x) || !aligned(p.codes) || !aligned(p.scales) || !aligned(p.x_scales))
        throw std::invalid_argument("NVFP4 TMA operands require 16-byte alignment");
    const auto descriptors = make_nvfp4_a4_tma_descriptors<Schedule, Rows>(p);
    constexpr int bytes    = sizeof(Nvfp4A4TmaSharedStorage<Schedule, Rows, Epilogue>);
    constexpr auto kernel  = nvfp4_a4_tma_kernel<Schedule, Epilogue, Output, Rows>;
    (void)nvfp4_prepare_shared<bytes, kernel, true>();
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(p.rows / Schedule::kBlockRows, div_up(count, Schedule::kBlockTokens));
        kernel<<<grid, Schedule::kThreads, bytes, stream>>>(descriptors, p.alpha, epilogue, output,
                                                            offset + count, p.rows, p.k, offset);
        CUDA_CHECK(cudaGetLastError());
    });
}
} // namespace ninfer::ops::detail
