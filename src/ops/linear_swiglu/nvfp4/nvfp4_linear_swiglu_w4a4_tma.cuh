#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <class Schedule>
struct Nvfp4LinearSwiGluTmaTensorStorage {
    static_assert(Schedule::kBlockN == 128);
    static constexpr int kPairN = Schedule::kBlockN / 2;

    alignas(
        128) std::uint8_t a_codes[Schedule::kStages][Schedule::kBlockM * Schedule::kCodeRowBytes];
    alignas(
        128) std::uint8_t b_codes[Schedule::kStages][Schedule::kBlockN * Schedule::kCodeRowBytes];
    alignas(16)
        std::uint32_t a_scale4[Schedule::kStages][Schedule::kBlockM * Schedule::kScaleWordsPerRow];
    alignas(16)
        std::uint8_t b_scales[Schedule::kStages][2][Schedule::kBlockN * Schedule::kK64PerStage * 4];
};

template <class Schedule>
union alignas(128) Nvfp4LinearSwiGluTmaScratch {
    static constexpr int kPairN = Schedule::kBlockN / 2;

    Nvfp4LinearSwiGluTmaTensorStorage<Schedule> tensors;
    __nv_bfloat16 output[Schedule::kBlockM * (kPairN + 8)];
};

template <class Schedule>
struct Nvfp4LinearSwiGluTmaSharedStorage {
    Nvfp4LinearSwiGluTmaScratch<Schedule> scratch;
    alignas(8) std::uint64_t full[Schedule::kStages];
    alignas(8) std::uint64_t empty[Schedule::kStages];
};

template <class Geometry, class Schedule>
__global__ __launch_bounds__(
    Schedule::kThreads,
    Schedule::
        kMinBlocksPerSm) void nvfp4_linear_swiglu_w4a4_tma_kernel(const __grid_constant__
                                                                      Nvfp4W4a4TmaDescriptors
                                                                          descriptors,
                                                                  float alpha,
                                                                  __nv_bfloat16* __restrict__ output) {
// Same story as nvfp4_w4a4_tma.cuh: warp-specialized Hopper+/Blackwell code (TMA, mbarrier,
// setmaxnreg), permanently out of scope for Volta. See docs/v100.md.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900
    static_assert(Geometry::kOutputRows == 34816);
    static_assert(Geometry::kInputRows == 5120);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    static_assert(Schedule::kBlockN == 128);
    static_assert(Schedule::kWarpsN == 2);
    static_assert(Schedule::kMmaN == 8);

    constexpr int kIntermediate = Geometry::kOutputRows / 2;
    constexpr int kPairN        = Schedule::kBlockN / 2;
    static_assert((kIntermediate % kPairN) == 0);
    static_assert((kIntermediate % 128) == 0);

    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& shared = *reinterpret_cast<Nvfp4LinearSwiGluTmaSharedStorage<Schedule>*>(shared_bytes);
    const int token_begin = static_cast<int>(blockIdx.y) * Schedule::kBlockM;
    const int pair_begin  = static_cast<int>(blockIdx.x) * kPairN;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < Schedule::kStages; ++stage) {
            nvfp4_mbarrier_init(&shared.full[stage], 1);
            nvfp4_mbarrier_init(&shared.empty[stage], Schedule::kConsumerWarps);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
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
                nvfp4_mbarrier_wait(&shared.empty[stage], empty_phase);
                constexpr std::uint32_t kTransactionBytes =
                    Schedule::kBlockM * Schedule::kCodeRowBytes +
                    Schedule::kBlockN * Schedule::kCodeRowBytes +
                    Schedule::kBlockM * Schedule::kScaleWordsPerRow * 4 +
                    2 * Schedule::kBlockN * Schedule::kK64PerStage * 4;
                nvfp4_mbarrier_arrive_expect_tx(&shared.full[stage], kTransactionBytes);

                auto& tensors = shared.scratch.tensors;
                nvfp4_tma_load_2d(tensors.a_codes[stage], &descriptors.a_codes,
                                  k_tile * Schedule::kCodeRowBytes, token_begin,
                                  &shared.full[stage]);
                nvfp4_tma_load_2d(tensors.b_codes[stage], &descriptors.b_codes,
                                  k_tile * Schedule::kCodeRowBytes, pair_begin,
                                  &shared.full[stage]);
                nvfp4_tma_load_2d(tensors.b_codes[stage] + kPairN * Schedule::kCodeRowBytes,
                                  &descriptors.b_codes, k_tile * Schedule::kCodeRowBytes,
                                  pair_begin + kIntermediate, &shared.full[stage]);
                nvfp4_tma_load_2d(tensors.a_scale4[stage], &descriptors.a_scales, (k_tile / 2) * 16,
                                  token_begin, &shared.full[stage]);

                const int gate_scale_row = ((pair_begin / 128) * Geometry::kScaleTilesPerRow +
                                            k_tile * Schedule::kK64PerStage) *
                                           32;
                const int up_scale_row =
                    (((pair_begin + kIntermediate) / 128) * Geometry::kScaleTilesPerRow +
                     k_tile * Schedule::kK64PerStage) *
                    32;
                nvfp4_tma_load_2d(tensors.b_scales[stage][0], &descriptors.b_scales, 0,
                                  gate_scale_row, &shared.full[stage]);
                nvfp4_tma_load_2d(tensors.b_scales[stage][1], &descriptors.b_scales, 0,
                                  up_scale_row, &shared.full[stage]);
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
    const int pair_mod128   = pair_begin & 127;

    float accumulators[Schedule::kMmaM][Schedule::kMmaN][4] = {};
#pragma unroll 1
    for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
        const int stage                = k_tile % Schedule::kStages;
        const std::uint32_t full_phase = (k_tile / Schedule::kStages) & 1U;
        nvfp4_mbarrier_wait(&shared.full[stage], full_phase);

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
                constexpr int kFragmentsPerBranch = Schedule::kMmaN / 2;
                const int branch                  = mma_n / kFragmentsPerBranch;
                const int pair_fragment           = mma_n - branch * kFragmentsPerBranch;
                const int pair_row =
                    warp_n * (kPairN / Schedule::kWarpsN) + pair_fragment * 8 + b_row_offset;
                const int storage_row  = branch * kPairN + pair_row;
                const int logical_byte = local_k64 * 32 + b_column_byte;
                const int physical_byte =
                    ((logical_byte >> 4) ^ ((storage_row >> 1) & 3)) * 16 + (logical_byte & 15);
                const auto* address =
                    tensors.b_codes[stage] + storage_row * Schedule::kCodeRowBytes + physical_byte;
                ldmatrix_x2(b_fragments[mma_n][0], b_fragments[mma_n][1], smem_addr(address));

                const int scale_pair_row = pair_mod128 + warp_n * (kPairN / Schedule::kWarpsN) +
                                           pair_fragment * 8 + sfb_row;
                const int row_mod32    = scale_pair_row & 31;
                const int row_quartile = scale_pair_row >> 5;
                b_scales[mma_n] =
                    load_vec<unsigned>(tensors.b_scales[stage][branch] +
                                       (local_k64 * 32 + row_mod32) * 16 + row_quartile * 4);
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
        if (lane == 0) { nvfp4_mbarrier_arrive(&shared.empty[stage]); }
    }

    asm volatile("bar.sync 1, %0;" : : "r"(Schedule::kConsumerThreads) : "memory");

    constexpr int kOutputStride     = kPairN + 8;
    constexpr int kGateMmaFragments = Schedule::kMmaN / 2;
    auto* shared_output             = shared.scratch.output;
    const int accumulator_row       = lane >> 2;
    const int accumulator_col       = 2 * (lane & 3);
#pragma unroll
    for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
        const int token0 = warp_m * Schedule::kWarpM + mma_m * 16 + accumulator_row;
        const int token1 = token0 + 8;
#pragma unroll
        for (int mma_n = 0; mma_n < kGateMmaFragments; ++mma_n) {
            const int pair_row =
                warp_n * (kPairN / Schedule::kWarpsN) + mma_n * 8 + accumulator_col;
            auto* destination0 = reinterpret_cast<__nv_bfloat162*>(
                shared_output + token0 * kOutputStride + pair_row);
            auto* destination1 = reinterpret_cast<__nv_bfloat162*>(
                shared_output + token1 * kOutputStride + pair_row);
            const auto& gate = accumulators[mma_m][mma_n];
            const auto& up   = accumulators[mma_m][mma_n + kGateMmaFragments];
            *destination0    = __floats2bfloat162_rn(silu(gate[0] * alpha) * (up[0] * alpha),
                                                     silu(gate[1] * alpha) * (up[1] * alpha));
            *destination1    = __floats2bfloat162_rn(silu(gate[2] * alpha) * (up[2] * alpha),
                                                     silu(gate[3] * alpha) * (up[3] * alpha));
        }
    }

    asm volatile("bar.sync 1, %0;" : : "r"(Schedule::kConsumerThreads) : "memory");
    constexpr int kVectorsPerRow = kPairN / 8;
    constexpr int kOutputVectors = Schedule::kBlockM * kVectorsPerRow;
    for (int task = consumer_thread; task < kOutputVectors; task += Schedule::kConsumerThreads) {
        const int token_local = task / kVectorsPerRow;
        const int row_vector  = task - token_local * kVectorsPerRow;
        const int token       = token_begin + token_local;
        const uint4 values =
            load_vec<uint4>(shared_output + token_local * kOutputStride + row_vector * 8);
        store_vec(output + static_cast<std::int64_t>(token) * kIntermediate + pair_begin +
                      row_vector * 8,
                  values);
    }
#else  // __CUDA_ARCH__ < 900
    (void)descriptors;
    (void)alpha;
    (void)output;
    __trap();
#endif // __CUDA_ARCH__ >= 900
}

} // namespace ninfer::ops::detail
