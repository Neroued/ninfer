#pragma once

#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/host/launch.h"
#include "ops/common/numeric/math.h"
#include "ops/linear/common/epilogue.cuh"

#include "ops/linear/kernels/bf16/bf16_mma_common.cuh"
#include "ops/linear/kernels/bf16/bf16_operands.h"
#include <cuda.h>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

struct alignas(128) Bf16TmaDescriptors {
    CUtensorMap weight;
    CUtensorMap activation;
};

template <class Schedule>
inline CUtensorMap bf16_tma_map(const __nv_bfloat16* pointer, int rows, int k, int block_rows,
                                int block_k) {
    // Factor K into 64-element sectors. The contiguous 128-byte dimension matches the
    // hardware swizzle while the next box dimension permits larger K tiles without repacking.
    CUtensorMap result{};
    std::uint64_t dimensions[]{64, static_cast<std::uint64_t>(k / 64),
                               static_cast<std::uint64_t>(rows)};
    std::uint64_t strides[]{128, static_cast<std::uint64_t>(k) * 2};
    std::uint32_t box[]{64, static_cast<std::uint32_t>(block_k / 64),
                        static_cast<std::uint32_t>(block_rows)};
    if constexpr (bf16_predicated_k<Schedule>) {
        // A logical 2D K axis lets TMA zero-fill its partial final sector, preserving row stride.
        static_assert(Schedule::kBlockK == 64, "K-tail TMA requires a 128-byte inner box");
        dimensions[0] = static_cast<std::uint64_t>(k);
        dimensions[1] = static_cast<std::uint64_t>(rows);
        strides[0]    = static_cast<std::uint64_t>(k) * 2;
        box[1]        = static_cast<std::uint32_t>(block_rows);
    }
    const std::uint32_t steps[]{1, 1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, bf16_predicated_k<Schedule> ? 2 : 3,
        const_cast<__nv_bfloat16*>(pointer), dimensions, strides, box, steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("BF16 TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return result;
}

struct Bf16ContiguousTmaWeightAccess;

template <class Schedule, class WeightAccess = Bf16ContiguousTmaWeightAccess>
Bf16TmaDescriptors make_bf16_tma_descriptors(const Bf16A16Operands& p) {
    validate_bf16_operands<Schedule>(p);
    if ((!bf16_predicated_rows<Schedule> && p.rows % Schedule::kBlockRows) ||
        p.k % (bf16_predicated_k<Schedule> ? 8 : Schedule::kBlockK))
        throw std::invalid_argument("BF16 TMA requires compatible row/K tiles");
    return {WeightAccess::template map<Schedule>(p),
            bf16_tma_map<Schedule>(p.x, p.tokens, p.k, Schedule::kBlockTokens, Schedule::kBlockK)};
}

template <class Schedule>
__device__ __forceinline__ void bf16_tma_load(void* destination, const CUtensorMap* map,
                                              int k_sector, int row, std::uint64_t* barrier) {
    if constexpr (bf16_predicated_k<Schedule>) {
        common::tma_load_2d(destination, map, k_sector * 64, row, barrier);
    } else {
        common::tma_load_3d(destination, map, 0, k_sector, row, barrier);
    }
}

struct Bf16ContiguousTmaWeightAccess {
    template <class Schedule>
    static CUtensorMap map(const Bf16A16Operands& p) {
        return bf16_tma_map<Schedule>(p.weight, p.rows, p.k, Schedule::kBlockRows,
                                      Schedule::kBlockK);
    }

    template <class Schedule>
    __device__ __forceinline__ static void load(void* destination, const CUtensorMap* map,
                                                int k_sector, int row, std::uint64_t* barrier) {
        bf16_tma_load<Schedule>(destination, map, k_sector, row, barrier);
    }
};

template <class Schedule, class Epilogue>
inline constexpr int bf16_tma_scratch_bytes =
    ((Schedule::kTensorBytes > linear_epilogue_bytes<Schedule, Epilogue>
          ? Schedule::kTensorBytes
          : linear_epilogue_bytes<Schedule, Epilogue>)+127) /
    128 * 128;

template <class Schedule, bool FullTokens, class Output, class Epilogue, int Splits = 1,
          class WeightAccess = Bf16ContiguousTmaWeightAccess>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void bf16_a16_tma_mma_kernel(
#if defined(_MSC_VER)
    // MSVC rejects the over-aligned (alignas(128)) descriptors as a by-value kernel parameter
    // (C2719), so they travel as a device pointer on Windows; other compilers keep the
    // __grid_constant__ by-value parameter. The body dereferences them identically.
    const Bf16TmaDescriptors* descriptors,
#else
    const __grid_constant__ Bf16TmaDescriptors descriptors,
#endif
    Output output, Epilogue epilogue, int rows, int input_rows, int token_offset, int count) {
#if defined(_MSC_VER)
    const Bf16TmaDescriptors& tma = *descriptors;
#else
    const Bf16TmaDescriptors& tma = descriptors;
#endif
    constexpr int BR = Schedule::kBlockRows, BT = Schedule::kBlockTokens, BK = Schedule::kBlockK;
    constexpr int S       = Schedule::kStages;
    const int K           = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    const int split_k     = K / Splits;
    const int split_begin = Splits == 1 ? 0 : int(blockIdx.z) * split_k;
    const int tiles_r     = bf16_predicated_rows<Schedule> ? (rows + BR - 1) / BR : rows / BR;
    const int tiles_t     = (count + BT - 1) / BT;
    int tile_r, tile_t;
    bf16_mma_tile_coordinates<Schedule>(blockIdx.x, tiles_r, tiles_t, tile_r, tile_t);
    const int row_begin = tile_r * BR, token_begin = token_offset + tile_t * BT;
    extern __shared__ __align__(128) unsigned char storage[];
    auto* a = reinterpret_cast<__nv_bfloat16*>(storage);
    auto* b = a + S * BR * BK;
    auto* full =
        reinterpret_cast<std::uint64_t*>(storage + bf16_tma_scratch_bytes<Schedule, Epilogue>);
    auto* empty = full + S;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            common::cta_mbarrier_init(full + stage, 1);
            common::cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps);
        }
        common::cta_mbarrier_fence_init();
    }
    __syncthreads();
    const int tiles_k         = split_k / BK + (bf16_predicated_k<Schedule> && split_k % BK != 0);
    constexpr bool collective = requires { Epilogue::kTmaCollective; };
    const int tid             = threadIdx.x - Schedule::kProducerThreads;
    const int warp = tid / 32, lane = tid & 31;
    float accum[Schedule::kMmaRows][Schedule::kMmaTokens][4] = {};
    if (threadIdx.x < Schedule::kProducerThreads) {
        if (threadIdx.x == 0) {
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                common::cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                common::cta_mbarrier_arrive_expect_tx(full + stage, (BR + BT) * BK * 2);
                const int k_sector = split_begin / 64 + kt * (BK / 64);
                WeightAccess::template load<Schedule>(a + stage * BR * BK, &tma.weight, k_sector,
                                                      row_begin, full + stage);
                bf16_tma_load<Schedule>(b + stage * BT * BK, &tma.activation, k_sector,
                                        token_begin, full + stage);
            }
        }
        // Collective consumers reuse staging and need all CTA participants at their barriers.
        if constexpr (!collective) return;
    }
    if (threadIdx.x >= Schedule::kProducerThreads) {
#pragma unroll Schedule::kConsumerKUnroll
        for (int kt = 0; kt < tiles_k; ++kt) {
            const int stage = kt % S;
            common::cta_mbarrier_wait(full + stage, (kt / S) & 1U);
            bf16_mma_compute_stage<Schedule>(a + stage * BR * BK, b + stage * BT * BK, accum, warp,
                                             lane);
            // LDMATRIX reads use the generic proxy; the next TMA write uses the async proxy.
            // Order every lane's reads across that boundary before the elected release arrival.
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (lane == 0) common::cta_mbarrier_arrive(empty + stage);
        }
    }
    bf16_finish_mma_tile<Schedule, FullTokens>(output, epilogue, storage, accum, row_begin,
                                               token_begin, rows, token_offset + count, warp, lane);
}

template <class Schedule, int Splits = 1, class Output, class Epilogue,
          class WeightAccess = Bf16ContiguousTmaWeightAccess>
void launch_bf16_a16_tma_mma(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                             cudaStream_t stream) {
    static_assert(Splits >= 1 && Splits <= 32);
    if constexpr (Splits > 1) {
        if (p.k % (Schedule::kBlockK * Splits))
            throw std::invalid_argument(
                "BF16 split-K TMA requires complete K tiles in every split");
    }
    // Descriptors are launch-owned values, copied into kernel parameters during Graph capture.
    const auto descriptors = make_bf16_tma_descriptors<Schedule, WeightAccess>(p);
#if defined(_MSC_VER)
    Bf16TmaDescriptors* device_descriptors = nullptr;
    CUDA_CHECK(cudaMallocAsync(&device_descriptors, sizeof(Bf16TmaDescriptors), stream));
    CUDA_CHECK(cudaMemcpyAsync(device_descriptors, &descriptors, sizeof(Bf16TmaDescriptors),
                               cudaMemcpyHostToDevice, stream));
#endif
    common::for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const auto blocks =
            static_cast<std::int64_t>(bf16_predicated_rows<Schedule>
                                          ? common::div_up(p.rows, Schedule::kBlockRows)
                                          : p.rows / Schedule::kBlockRows) *
            common::div_up(count, Schedule::kBlockTokens);
        if (blocks > 2147483647LL)
            throw std::invalid_argument("BF16 TMA grid exceeds CUDA grid.x capacity");
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel =
                bf16_a16_tma_mma_kernel<Schedule, Full, Output, Epilogue, Splits, WeightAccess>;
            constexpr int bytes =
                bf16_tma_scratch_bytes<Schedule, Epilogue> + Schedule::kBarrierBytes;
            common::prepare_shared<bytes, kernel, true>();
#if defined(_MSC_VER)
            // MSVC's cudafe cannot launch through the constexpr function-pointer variable
            // captured by this lambda (C2326), so reference the kernel template directly.
            bf16_a16_tma_mma_kernel<Schedule, Full, Output, Epilogue, Splits, WeightAccess>
                <<<dim3(static_cast<unsigned>(blocks), 1, Splits), Schedule::kThreads, bytes,
                   stream>>>(device_descriptors, output, epilogue, p.rows, p.k, offset, count);
#else
            kernel<<<dim3(static_cast<unsigned>(blocks), 1, Splits), Schedule::kThreads, bytes,
                     stream>>>(descriptors, output, epilogue, p.rows, p.k, offset, count);
#endif
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
#if defined(_MSC_VER)
    CUDA_CHECK(cudaFreeAsync(device_descriptors, stream));
#endif
}

} // namespace ninfer::ops::detail