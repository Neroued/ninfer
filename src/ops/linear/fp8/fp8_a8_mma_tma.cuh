#pragma once

// PILOT (TMA activation staging): row-scaled E4M3 weight x materialized row-scaled E4M3
// activation Tensor Core GEMM with the activation tile loaded by cp.async.bulk (TMA, 128B
// swizzle) instead of per-thread cp.async. Weights stay on the cp.async pipeline exactly
// as in fp8_a8_mma_kernel; only the activation copy path and its completion signal
// (mbarrier instead of cp.async groups) differ. Full tiles only.
//
// The 128B TMA swizzle reproduces fp8_mma_shared_byte exactly when BlockK == 128 (eight
// 16-byte segments selected by row & 7) and each activation stage base is 1024-byte
// aligned (the BM*BK stage stride preserves it), so the ldmatrix reader is unchanged and
// outputs must be bit-identical to the baseline kernel.
//
// Contract notes for future callers: the kernel itself is band-agnostic and
// slice-correct (TMA coordinates are absolute), requiring only full BM-wide token
// tiles with BlockK == 128 and a 1024-aligned activation stage base. The Prefill-band
// (T > 144) and single-launch restrictions live in the pilot launcher, which selects
// exact-multiple shapes to match production dispatch.

#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
// Schedule, swizzle, MMA atoms, operands, and epilogue/output policies come from the
// baseline kernel: the pilot changes only the activation copy path, so those definitions
// must stay identical.
#include "ops/linear/fp8/fp8_a8_mma.cuh"

#include <cuda.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

// PILOT-local TMA helpers. If the pilot proceeds, these merge with the Q4/NVFP4 copies
// into one ops/common helper; while the pilot is under evaluation they stay local so no
// existing translation unit changes.
struct Fp8MmaTmaDescriptors {
    CUtensorMap a_tile;
};

inline void fp8_mma_tma_check(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS) { return; }
    const char* name = nullptr;
    (void)cuGetErrorName(status, &name);
    throw std::runtime_error(std::string(operation) + ": " +
                             (name != nullptr ? name : "CUDA error"));
}

inline CUtensorMap fp8_mma_tma_activation_2d(const void* address, std::uint64_t k,
                                            std::uint64_t tokens, std::uint32_t box_k,
                                            std::uint32_t box_tokens) {
    if (address == nullptr || k == 0 || tokens == 0 || box_k == 0 || box_tokens == 0) {
        throw std::invalid_argument("FP8 activation TMA needs a non-empty tensor and box");
    }
    if (reinterpret_cast<std::uintptr_t>(address) % 16 != 0 || k % 16 != 0) {
        throw std::invalid_argument(
            "FP8 activation TMA needs a 16-byte-aligned base and row stride");
    }
    if (box_k > k || box_tokens > tokens) {
        throw std::invalid_argument("FP8 activation TMA box must land inside the tensor");
    }
    if (box_k > 128) {
        throw std::invalid_argument("FP8 activation TMA box exceeds the 128B swizzle span");
    }
    if (k > 0xffffffffULL || tokens > 0xffffffffULL) {
        throw std::invalid_argument("FP8 activation TMA dims exceed the 32-bit element bound");
    }
    CUtensorMap map{};
    const std::uint64_t global_dim[]     = {k, tokens};
    const std::uint64_t global_stride[]  = {k};
    const std::uint32_t box_dim[]        = {box_k, box_tokens};
    const std::uint32_t element_stride[] = {1, 1};
    fp8_mma_tma_check(
        cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(address),
                               global_dim, global_stride, box_dim, element_stride,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        "encode FP8 activation TMA");
    return map;
}

__device__ __forceinline__ void fp8_mma_tma_load_2d(void* destination,
                                                   const CUtensorMap* descriptor,
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

// clang-format off
template <class Schedule, bool FullTokens, class Epilogue, class Output, class RowPolicy>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a8_mma_tma_kernel(
    Fp8A8Operands operands, Output output, Epilogue epilogue, RowPolicy row_policy,
    const __grid_constant__ Fp8MmaTmaDescriptors descriptors, int token_offset, int count) {
    // clang-format on
    static_assert(FullTokens, "FP8 TMA pilot needs full token tiles");
    static_assert(Schedule::kBlockK == 128, "FP8 TMA pilot needs the 128-wide K tile");
    constexpr bool PairRows                    = RowPolicy::kPaired;
    const auto* __restrict__ activation_codes  = operands.x;
    // Retained for a line-by-line mirror of the baseline prologue; the TMA descriptor
    // carries the activation address, so the raw pointer is never dereferenced here.
    (void)activation_codes;
    const auto* __restrict__ activation_scales = operands.x_scales;
    const auto* __restrict__ weight_codes      = operands.codes;
    const auto* __restrict__ weight_scales     = operands.scales;
    const int K                                = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    const int tokens                           = token_offset + count;
    const int TILES_K                          = K / Schedule::kBlockK;
    constexpr int BM                           = Schedule::kBlockTokens;
    constexpr int BN                           = Schedule::kBlockRows;
    constexpr int BK                           = Schedule::kBlockK;
    constexpr int S                            = Schedule::kStages;
    constexpr int THREADS                      = Schedule::kThreads;
    // Every activation stage base must be 1024-byte aligned so the 128B TMA swizzle
    // reproduces fp8_mma_shared_byte with no base-offset term; the shared array base
    // carries alignas(1024) and the BM*BK stage stride preserves it. Barriers sit past
    // the baseline's shared region so the epilogue's shared reuse is untouched.
    static_assert((BM * BK) % 1024 == 0, "FP8 TMA pilot needs 1024-aligned stages");
    static_assert(fp8_mma_shared_bytes<Schedule, Epilogue> % 8 == 0,
                  "FP8 TMA pilot needs 8-byte-aligned barrier storage");
    __shared__ __align__(1024) unsigned char
        tma_shared[fp8_mma_shared_bytes<Schedule, Epilogue> + S * sizeof(std::uint64_t)];
    auto* shared_raw        = tma_shared;
    auto* activation_shared = reinterpret_cast<std::uint8_t*>(shared_raw);
    auto* weight_shared     = activation_shared + S * BM * BK;
    auto* act_full          = reinterpret_cast<std::uint64_t*>(
        tma_shared + fp8_mma_shared_bytes<Schedule, Epilogue>);

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

    // Full tiles only: the TMA box always lands fully inside the activation tensor.
    auto stage_activation = [&](int stage, int k_tile) {
        if (tid == 0) {
            cta_mbarrier_arrive_expect_tx(&act_full[stage], BM * BK);
            fp8_mma_tma_load_2d(activation_shared + stage * BM * BK, &descriptors.a_tile,
                                k_tile * BK, token_begin, &act_full[stage]);
        }
    };

    auto stage_weights = [&](int stage, int k_tile) {
        const int k_begin  = k_tile * BK;
        auto* weight_stage = weight_shared + stage * BN * BK;
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

    auto stage_inputs = [&](int stage, int k_tile) {
        stage_activation(stage, k_tile);
        stage_weights(stage, k_tile);
    };

    if (tid == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) { cta_mbarrier_init(&act_full[stage], 1); }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

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
        const int stage                = k_tile % S;
        const std::uint32_t full_phase = static_cast<std::uint32_t>((k_tile / S) & 1);
        if (k_tile + S <= TILES_K) {
            cp_wait<S - 1>();
        } else {
            cp_wait<0>();
        }
        cta_mbarrier_wait(&act_full[stage], full_phase);
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
        const int token1              = token0 + 8;
        const float activation_scale0 = __ldg(activation_scales + token0);
        const float activation_scale1 = __ldg(activation_scales + token1);
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
                value00 = epilogue.apply(parent_row0, token0, value00);
                value01 = epilogue.apply(parent_row1, token0, value01);
                value10 = epilogue.apply(parent_row0, token1, value10);
                value11 = epilogue.apply(parent_row1, token1, value11);
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
            const uint4 values =
                load_vec<uint4>(shared_output + token_local * output_stride + row_vector * 8);
            linear_store_bf16_vector(output, row_begin + row_vector * 8, token, values);
        }
    }
}
} // namespace ninfer::ops::detail
