#pragma once

#include "ops/common/mbarrier.cuh"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/fp8/fp8_a8_mma_common.cuh"
#include "ops/linear/fp8/fp8_operands.h"
#include "ops/linear/fp8/fp8_shared.cuh"

#include <cuda.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail {

struct alignas(128) Fp8TmaDescriptors {
    CUtensorMap activation;
    CUtensorMap weight;
};

struct Fp8TmaSplitKPlan {
    int full_tiles = 0;
    int tail_tiles = 0;
    int split_ctas = 0;

    __host__ __device__ int parts(int tail) const {
        return split_ctas / tail_tiles + (tail < split_ctas % tail_tiles);
    }

    __host__ __device__ int first_part(int tail) const {
        const int extra = split_ctas % tail_tiles;
        return tail * (split_ctas / tail_tiles) + (tail < extra ? tail : extra);
    }
};

template <class Schedule>
inline Fp8TmaSplitKPlan fp8_tma_split_k_plan(int tiles, int k) {
    Fp8TmaSplitKPlan plan{tiles, 0, 0};
    if constexpr (Schedule::kSplitWaveCtas > 0) {
        const int tail = tiles % Schedule::kSplitWaveCtas;
        if (tail && tail <= Schedule::kSplitWaveCtas / 2) {
            const int max_parts = std::min(Schedule::kMaxParts, k / Schedule::kBlockK);
            if (max_parts >= 2)
                plan = {tiles - tail, tail, std::min(Schedule::kSplitWaveCtas, tail * max_parts)};
        }
    }
    return plan;
}

inline CUtensorMap fp8_tma_map(const std::uint8_t* pointer, int rows, int k, int block_rows,
                               int block_k) {
    CUtensorMap result{};
    const std::uint64_t dimensions[]{static_cast<std::uint64_t>(k),
                                     static_cast<std::uint64_t>(rows)};
    const std::uint64_t strides[]{static_cast<std::uint64_t>(k)};
    const std::uint32_t box[]{static_cast<std::uint32_t>(block_k),
                              static_cast<std::uint32_t>(block_rows)};
    const std::uint32_t steps[]{1, 1};
    const auto swizzle = block_k == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;
    // Copy the represented E4M3 bytes; the row scales remain explicit FP32 epilogue operands.
    const auto status = cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<std::uint8_t*>(pointer), dimensions,
        strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("FP8 TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return result;
}

__device__ __forceinline__ void fp8_tma_load(void* destination, const CUtensorMap* map, int k,
                                             int row, std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(k), "r"(row), "r"(smem_addr(barrier))
                 : "memory");
}

template <class Schedule, class Epilogue>
inline constexpr int fp8_tma_scratch_bytes = [] {
    constexpr int epilogue_bytes = [] {
        if constexpr (requires { Epilogue::template kSharedBytes<Schedule>; })
            return Epilogue::template kSharedBytes<Schedule>;
        else
            return 0;
    }();
    constexpr int bytes =
        Schedule::kStorageBytes > epilogue_bytes ? Schedule::kStorageBytes : epilogue_bytes;
    return (bytes + 127) / 128 * 128;
}();

template <class Schedule, bool FullTokens, class Output, class Epilogue, bool SplitK = false,
          class RowPolicy = Fp8IdentityRows>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a8_tma_mma_kernel(
#if defined(_MSC_VER)
    // MSVC rejects the over-aligned (alignas(128)) descriptors as a by-value kernel parameter
    // (C2719), so they travel as a device pointer on Windows; other compilers keep the
    // __grid_constant__ by-value parameter. The body dereferences them identically.
    const Fp8TmaDescriptors* descriptors,
#else
    const __grid_constant__ Fp8TmaDescriptors descriptors,
#endif
    Fp8A8Operands operands, Output output,
    Epilogue epilogue, RowPolicy row_policy, int token_offset, int count, Fp8TmaSplitKPlan plan,
    float* partials) {
#if defined(_MSC_VER)
    const Fp8TmaDescriptors& tma = *descriptors;
#else
    const Fp8TmaDescriptors& tma = descriptors;
#endif
    constexpr int BT = Schedule::kBlockTokens, BR = Schedule::kBlockRows;
    constexpr int BK = Schedule::kBlockK, S = Schedule::kStages;
    const int k = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    int tile = blockIdx.x, partial = -1, k_begin = 0, tiles_k = k / BK;
    if constexpr (SplitK) {
        if (tile >= plan.full_tiles) {
            partial          = tile - plan.full_tiles;
            const int base   = plan.split_ctas / plan.tail_tiles;
            const int extra  = plan.split_ctas % plan.tail_tiles;
            const int longer = extra * (base + 1);
            const int tail =
                partial < longer ? partial / (base + 1) : extra + (partial - longer) / base;
            const int part  = partial - plan.first_part(tail);
            const int parts = plan.parts(tail);
            tile            = plan.full_tiles + tail;
            k_begin         = tiles_k * part / parts;
            tiles_k         = tiles_k * (part + 1) / parts - k_begin;
        }
    }
    const int row_tiles = operands.rows / BR, token_tiles = div_up(count, BT);
    int row_tile, token_tile;
    fp8_mma_tile_coordinates<Schedule>(tile, row_tiles, token_tiles, row_tile, token_tile);
    const int row_begin   = row_tile * (BR / (RowPolicy::kPaired ? 2 : 1));
    const int token_begin = token_offset + token_tile * BT;

    extern __shared__ __align__(128) unsigned char fp8_tma_shared[];
    auto* activation = fp8_tma_shared;
    auto* weight     = activation + S * BT * BK;
    auto* full       = reinterpret_cast<std::uint64_t*>(fp8_tma_shared +
                                                        fp8_tma_scratch_bytes<Schedule, Epilogue>);
    auto* empty      = full + S;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            cta_mbarrier_init(full + stage, 1);
            cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

    if (threadIdx.x < Schedule::kProducerThreads) {
        if (threadIdx.x == 0) {
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                cta_mbarrier_arrive_expect_tx(full + stage, (BT + BR) * BK);
                fp8_tma_load(activation + stage * BT * BK, &tma.activation,
                             (k_begin + kt) * BK, token_begin, full + stage);
                if constexpr (RowPolicy::kPaired) {
                    // Each consumer warp owns both gate/up fragments. Load their contiguous
                    // weight spans into that warp's logical shared rows without repacking.
                    constexpr int span = Schedule::kWarpRows / 2;
#pragma unroll
                    for (int local = 0; local < BR; local += span) {
                        fp8_tma_load(weight + (stage * BR + local) * BK, &tma.weight,
                                     (k_begin + kt) * BK,
                                     row_policy.weight_row(row_begin, local, operands.rows),
                                     full + stage);
                    }
                } else {
                    fp8_tma_load(weight + stage * BR * BK, &tma.weight, (k_begin + kt) * BK,
                                 row_begin, full + stage);
                }
            }
        }
        return;
    }

    const int tid  = static_cast<int>(threadIdx.x) - Schedule::kProducerThreads;
    const int warp = tid >> 5, lane = tid & 31;
    float accumulators[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        cta_mbarrier_wait(full + stage, (kt / S) & 1U);
        fp8_mma_compute_stage<Schedule>(activation + stage * BT * BK, weight + stage * BR * BK,
                                        accumulators, warp, lane);
        // Release only after every lane in this consumer warp has finished its shared reads.
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + stage);
    }
    if constexpr (SplitK) {
        if (partial >= 0) {
            // Store raw FP32 accumulators. Scales, epilogue and the final BF16 cast
            // belong after the complete K reduction, including for fused epilogues.
#pragma unroll
            for (int mt = 0; mt < Schedule::kMmaTokens; ++mt) {
                const int token =
                    warp / Schedule::kWarpsRows * Schedule::kWarpTokens + mt * 16 + lane / 4;
#pragma unroll
                for (int mr = 0; mr < Schedule::kMmaRows; ++mr) {
                    const int row =
                        warp % Schedule::kWarpsRows * Schedule::kWarpRows + mr * 8 + 2 * (lane % 4);
                    float* destination =
                        partials + std::size_t(partial) * BT * BR + token * BR + row;
                    *reinterpret_cast<float2*>(destination) =
                        make_float2(accumulators[mt][mr][0], accumulators[mt][mr][1]);
                    *reinterpret_cast<float2*>(destination + 8 * BR) =
                        make_float2(accumulators[mt][mr][2], accumulators[mt][mr][3]);
                }
            }
            return;
        }
    }
    // All consumers must finish reading staged inputs before the epilogue reuses the storage.
    __syncthreads();
    fp8_finish_mma_tile<Schedule, FullTokens>(
        output, epilogue, row_policy, fp8_tma_shared, accumulators, operands.x_scales,
        operands.scales, row_begin, token_begin, operands.rows, token_offset + count, warp, lane);
}

template <class Schedule, class Output, class Epilogue, class RowPolicy>
__global__ void fp8_a8_tma_split_k_reduce(Fp8A8Operands p, const float* partials, Output output,
                                          Epilogue epilogue, RowPolicy row_policy,
                                          Fp8TmaSplitKPlan plan, int token_offset, int count) {
    constexpr int BT = Schedule::kBlockTokens, BR = Schedule::kBlockRows;
    constexpr int stored_rows    = BR / (RowPolicy::kPaired ? 2 : 1);
    constexpr int chunk_elements = BT * stored_rows / Schedule::kReductionBlocks;
    const int tail               = blockIdx.x / Schedule::kReductionBlocks;
    const int chunk              = blockIdx.x % Schedule::kReductionBlocks;
    int row_tile, token_tile;
    fp8_mma_tile_coordinates<Schedule>(plan.full_tiles + tail, p.rows / BR, div_up(count, BT),
                                       row_tile, token_tile);
    const int first = plan.first_part(tail), parts = plan.parts(tail);
    const int row_begin    = row_tile * stored_rows;
    const auto tile_output = linear_output_tile<stored_rows>(output, row_begin);
    for (int index = chunk * chunk_elements + threadIdx.x * 2; index < (chunk + 1) * chunk_elements;
         index += blockDim.x * 2) {
        const int token_local = index / stored_rows;
        const int token       = token_offset + token_tile * BT + token_local;
        const int output_row  = index % stored_rows;
        const int row         = row_begin + output_row;
        if (token < token_offset + count) {
            const auto sum_pair = [&](int local_row) {
                float2 sum{};
                for (int part = 0; part < parts; ++part) {
                    const auto value = *reinterpret_cast<const float2*>(
                        partials + std::size_t(first + part) * BT * BR + token_local * BR +
                        local_row);
                    sum.x += value.x;
                    sum.y += value.y;
                }
                const int parent  = row_policy.weight_row(row_begin, local_row, p.rows);
                const float scale = p.x_scales[token];
                sum.x             = sum.x * scale * __bfloat162float(p.scales[parent]);
                sum.y             = sum.y * scale * __bfloat162float(p.scales[parent + 1]);
                return sum;
            };
            float2 value;
            if constexpr (RowPolicy::kPaired) {
                constexpr int half_warp = Schedule::kWarpRows / 2;
                const int gate_row =
                    (output_row / half_warp) * Schedule::kWarpRows + output_row % half_warp;
                const float2 gate = sum_pair(gate_row);
                const float2 up   = sum_pair(gate_row + half_warp);
                value             = make_float2(epilogue.apply_pair(row, token, gate.x, up.x),
                                                epilogue.apply_pair(row + 1, token, gate.y, up.y));
            } else {
                value = fp8_apply_row_pair(epilogue, row, row + 1, token, sum_pair(output_row));
            }
            if constexpr (std::is_same_v<Output, LinearBf16Output>) {
                *reinterpret_cast<__nv_bfloat162*>(tile_output.at(row, token)) =
                    __floats2bfloat162_rn(value.x, value.y);
            } else {
                tile_output.store(row, token, value.x);
                tile_output.store(row + 1, token, value.y);
            }
        }
    }
}

template <class Schedule, class Output, class Epilogue, class RowPolicy = Fp8IdentityRows>
void launch_fp8_a8_tma_mma(const Fp8A8Operands& p, Output output, Epilogue epilogue,
                           cudaStream_t stream, float* partials = nullptr,
                           RowPolicy row_policy = {}) {
    validate_fp8_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("FP8 TMA requires complete row/K tiles");
    if constexpr (RowPolicy::kPaired) {
        static_assert(RowPolicy::kWarpPaired && RowPolicy::kContiguousPairs);
        static_assert(Schedule::kWarpRows % 16 == 0);
    }
    constexpr int weight_span = RowPolicy::kPaired ? Schedule::kWarpRows / 2 : Schedule::kBlockRows;
    // Descriptors are launch-owned values, copied into kernel parameters during Graph capture.
    const Fp8TmaDescriptors descriptors{
        fp8_tma_map(p.x, p.tokens, p.k, Schedule::kBlockTokens, Schedule::kBlockK),
        fp8_tma_map(p.codes, p.rows, p.k, weight_span, Schedule::kBlockK)};
#if defined(_MSC_VER)
    Fp8TmaDescriptors* device_descriptors = nullptr;
    CUDA_CHECK(cudaMallocAsync(&device_descriptors, sizeof(Fp8TmaDescriptors), stream));
    CUDA_CHECK(cudaMemcpyAsync(device_descriptors, &descriptors, sizeof(Fp8TmaDescriptors),
                               cudaMemcpyHostToDevice, stream));
#endif
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const int blocks  = p.rows / Schedule::kBlockRows * div_up(count, Schedule::kBlockTokens);
        const auto plan   = fp8_tma_split_k_plan<Schedule>(blocks, p.k);
        const auto launch = [&]<bool Full, bool Split>() {
            constexpr auto kernel =
                fp8_a8_tma_mma_kernel<Schedule, Full, Output, Epilogue, Split, RowPolicy>;
            constexpr int bytes =
                fp8_tma_scratch_bytes<Schedule, Epilogue> + Schedule::kBarrierBytes;
            const int dynamic = fp8_prepare_shared<bytes, kernel, true>();
            const int grid    = Split ? plan.full_tiles + plan.split_ctas : blocks;
#if defined(_MSC_VER)
            // MSVC's cudafe cannot launch through the constexpr function-pointer variable
            // captured by this lambda (C2326), so reference the kernel template directly.
            fp8_a8_tma_mma_kernel<Schedule, Full, Output, Epilogue, Split, RowPolicy>
                <<<grid, Schedule::kThreads, dynamic, stream>>>(
                    device_descriptors, p, output, epilogue, row_policy, offset, count, plan,
                    partials);
#else
            kernel<<<grid, Schedule::kThreads, dynamic, stream>>>(
                descriptors, p, output, epilogue, row_policy, offset, count, plan, partials);
#endif
            CUDA_CHECK(cudaGetLastError());
            if constexpr (Split) {
                fp8_a8_tma_split_k_reduce<Schedule>
                    <<<plan.tail_tiles * Schedule::kReductionBlocks, 256, 0, stream>>>(
                        p, partials, output, epilogue, row_policy, plan, offset, count);
                CUDA_CHECK(cudaGetLastError());
            }
        };
        if constexpr (Schedule::kSplitWaveCtas > 0) {
            static_assert(
                (!RowPolicy::kPaired && requires { epilogue.apply(0, 0, 0.0f); }) ||
                    (RowPolicy::kPaired && requires { epilogue.apply_pair(0, 0, 0.0f, 0.0f); }),
                "FP8 TMA split-K requires a scalar or paired epilogue after reduction");
            if (plan.split_ctas) {
                if (!partials || reinterpret_cast<std::uintptr_t>(partials) % 16)
                    throw std::invalid_argument("FP8 TMA split-K requires aligned caller partials");
                if (count % Schedule::kBlockTokens == 0)
                    launch.template operator()<true, true>();
                else
                    launch.template operator()<false, true>();
                return;
            }
        }
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true, false>();
        else
            launch.template operator()<false, false>();
    });
#if defined(_MSC_VER)
    CUDA_CHECK(cudaFreeAsync(device_descriptors, stream));
#endif
}

} // namespace ninfer::ops::detail
