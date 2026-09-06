#pragma once

// Row-scaled E4M3 weight x row-scaled E4M3 activation GEMM, staged by TMA with warp
// specialization. Same arithmetic as fp8_a8_mma.cuh: the accumulator, the m16n8k32 instruction,
// the scale application and the epilogue are unchanged. Only the staging differs - dedicated
// producer threads issue cp.async.bulk.tensor under an mbarrier ring, so one CTA per SM with a
// large token tile replaces two CTAs with a small one.

#include "ops/common/math.cuh"
#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/fp8/fp8_a8_mma.cuh" // Fp8MmaIdentityRows
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <type_traits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

// Highest device ordinal this route keeps per-device state for. Two things are cached by ordinal -
// the shared-memory opt-in in the launcher below, and the multiprocessor count the routing guard
// reads - and both must agree on the bound, or one of them silently stops caching where the other
// still does.
inline constexpr int kFp8A8MaxDevices = 16;

struct alignas(128) Fp8A8TmaDescriptors {
    CUtensorMap a_codes;
    CUtensorMap b_codes;
};

inline void fp8_check_driver(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS) { return; }
    const char* name = nullptr;
    (void)cuGetErrorName(status, &name);
    throw std::runtime_error(std::string(operation) + ": " +
                             (name != nullptr ? name : "CUDA error"));
}

inline CUtensorMap fp8_make_tma_2d(void* address, std::uint64_t columns, std::uint64_t rows,
                                   std::uint64_t row_stride_bytes, std::uint32_t box_columns,
                                   std::uint32_t box_rows, const char* operation) {
    CUtensorMap map{};
    const std::uint64_t global_dim[]     = {columns, rows};
    const std::uint64_t global_stride[]  = {row_stride_bytes};
    const std::uint32_t box_dim[]        = {box_columns, box_rows};
    const std::uint32_t element_stride[] = {1, 1};
    fp8_check_driver(
        cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, address, global_dim,
                               global_stride, box_dim, element_stride,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                               CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
        operation);
    return map;
}

// A is [tokens, K] and B is [output rows, K], both one byte per element and K-contiguous. The
// innermost box is the 64-byte K tile that 64B swizzle expects.
template <class Geometry, int BlockM, int BlockN, bool PairRows = false>
Fp8A8TmaDescriptors make_fp8_a8_tma_descriptors(const std::uint8_t* activation_codes,
                                                const std::uint8_t* weight_codes,
                                                std::int32_t tokens) {
    constexpr std::uint32_t kRowBytes = 64;
    // A paired block computes both branches, so its weight box is one branch tall and the
    // producer issues one request per branch.
    constexpr std::uint32_t kWeightBoxRows = PairRows ? BlockN / 2 : BlockN;
    Fp8A8TmaDescriptors descriptors{};
    descriptors.a_codes =
        fp8_make_tma_2d(const_cast<std::uint8_t*>(activation_codes), Geometry::kInputRows,
                        static_cast<std::uint64_t>(tokens), Geometry::kInputRows, kRowBytes, BlockM,
                        "encode fp8 activation codes TMA");
    descriptors.b_codes = fp8_make_tma_2d(
        const_cast<std::uint8_t*>(weight_codes), Geometry::kInputRows, Geometry::kOutputRows,
        Geometry::kInputRows, kRowBytes, kWeightBoxRows, "encode fp8 weight codes TMA");
    return descriptors;
}

// The route's shape, exported so the host can reason about a launch without restating it. The
// admission model needs the same grid the launcher builds; when those were written out twice, in
// two files, nothing would have caught them drifting apart.
template <class Geometry, class Schedule>
constexpr std::int64_t fp8_a8_tma_token_tiles(std::int32_t tokens) {
    return (static_cast<std::int64_t>(tokens) + Schedule::kBlockM - 1) / Schedule::kBlockM;
}

template <class Geometry, class Schedule>
constexpr std::int64_t fp8_a8_tma_blocks(std::int32_t tokens) {
    return static_cast<std::int64_t>(Geometry::kOutputRows / Schedule::kBlockN) *
           fp8_a8_tma_token_tiles<Geometry, Schedule>(tokens);
}

// Shapes the kernel can express at all. This is a property of the geometry, not a tuning decision,
// and the kernel asserts the same conditions - so it is stated once and consumed by both.
template <class Geometry, class Schedule>
constexpr bool kFp8A8TmaRepresentable = (Geometry::kOutputRows % Schedule::kBlockN) == 0 &&
                                        (Geometry::kInputRows % Schedule::kBlockK) == 0;

template <int BlockM, int Stages, int MinBlocksPerSm>
struct Fp8A8TmaSchedule {
    static_assert(BlockM == 64 || BlockM == 128 || BlockM == 256);
    static_assert(Stages >= 2 && Stages <= 8);
    // Not a free parameter. fp8_a8_tma_cheaper multiplies this by the multiprocessor count to get
    // the slots a wave fills, so a value above 1 divides the predicted wave count while the
    // per-wave work stays whole, and the route starts winning its own comparison at every partial
    // wave. Raising it means checking the shared-memory budget against it and re-fitting the ratio.
    static_assert(MinBlocksPerSm == 1,
                  "the cost model reads this as achieved occupancy; see fp8_a8_tma_cheaper");

    static constexpr int kBlockM          = BlockM;
    static constexpr int kBlockN          = 128;
    static constexpr int kBlockK          = 64;
    static constexpr int kRowBytes        = kBlockK;
    static constexpr int kStages          = Stages;
    static constexpr int kWarpsM          = 4;
    static constexpr int kWarpsN          = 2;
    static constexpr int kConsumerWarps   = kWarpsM * kWarpsN;
    static constexpr int kConsumerThreads = kConsumerWarps * 32;
    // One warp produces. Only lane 0 issues the TMA descriptors; the rest of a warp is the
    // smallest unit the block can be built from. An earlier form padded this to a full 128-thread
    // warpgroup so that setmaxnreg could hand the producer's registers to the consumers, but the
    // kernel allocates 166 registers with no spills, so there was nothing for the donation to buy -
    // and relocatable device code discards the transfer anyway, which is why the build file keeps
    // the nvfp4 warp-specialized kernels out of the RDC archive. Dropping to one warp measured
    // identical (1.0004 / 0.9997 / 0.9996 against the padded form) and removes that conflict.
    static constexpr int kProducerThreads = 32;
    // Raising this to a warpgroup would re-arm the setmaxnreg pair below, and this kernel is
    // compiled into a relocatable archive where that transfer is discarded. Keep it a build error.
    static_assert(
        kProducerThreads == 32,
        "a warpgroup producer re-arms setmaxnreg, which relocatable device code discards");
    static constexpr int kThreads        = kConsumerThreads + kProducerThreads;
    static constexpr int kWarpM          = kBlockM / kWarpsM;
    static constexpr int kWarpN          = kBlockN / kWarpsN;
    static constexpr int kMmaM           = kWarpM / 16;
    static constexpr int kMmaN           = kWarpN / 8;
    static constexpr int kMmaKPerStage   = kBlockK / 32;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;

    static_assert(kWarpM % 16 == 0 && kWarpN % 8 == 0);
};

// The 64B swizzle pattern repeats over eight rows of 64 bytes. fp8_tma_shared_byte reads the
// segment out of a row index taken relative to the tile base, which is the hardware's mapping only
// where that base is itself at the start of an atom - so the requirement is 512 bytes, and 128,
// which is all the TMA store itself needs, is not enough to state it. Every tile base has to hold
// it, not just the first: the stage stride and, for a paired block, the second branch's offset
// both land inside this storage.
inline constexpr std::size_t kFp8A8TmaSwizzleAtomBytes = 512;

template <class Schedule>
struct alignas(kFp8A8TmaSwizzleAtomBytes) Fp8A8TmaTensorStorage {
    alignas(kFp8A8TmaSwizzleAtomBytes)
        std::uint8_t a_codes[Schedule::kStages][Schedule::kBlockM * Schedule::kRowBytes];
    alignas(kFp8A8TmaSwizzleAtomBytes)
        std::uint8_t b_codes[Schedule::kStages][Schedule::kBlockN * Schedule::kRowBytes];

    static_assert((Schedule::kBlockM * Schedule::kRowBytes) % kFp8A8TmaSwizzleAtomBytes == 0,
                  "each activation stage must begin on a swizzle atom");
    static_assert((Schedule::kBlockN * Schedule::kRowBytes) % kFp8A8TmaSwizzleAtomBytes == 0,
                  "each weight stage must begin on a swizzle atom");
    // A paired block loads its second branch at half the weight stage. That offset is a tile base
    // as much as the stage is, so it carries the same requirement.
    static_assert(((Schedule::kBlockN / 2) * Schedule::kRowBytes) % kFp8A8TmaSwizzleAtomBytes == 0,
                  "the second branch of a paired block must begin on a swizzle atom");
};

template <class Schedule>
union alignas(kFp8A8TmaSwizzleAtomBytes) Fp8A8TmaScratch {
    Fp8A8TmaTensorStorage<Schedule> tensors;
    __nv_bfloat16 output[Schedule::kBlockM * (Schedule::kBlockN + 8)];
};

template <class Schedule>
struct alignas(kFp8A8TmaSwizzleAtomBytes) Fp8A8TmaSharedStorage {
    Fp8A8TmaScratch<Schedule> scratch;
    alignas(8) std::uint64_t full[Schedule::kStages];
    alignas(8) std::uint64_t empty[Schedule::kStages];
};

__device__ __forceinline__ void fp8_tma_load_2d(void* destination, const CUtensorMap* descriptor,
                                                std::int32_t coordinate0, std::int32_t coordinate1,
                                                std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :
                 : "r"(smem_addr(destination)), "l"(descriptor), "r"(coordinate0), "r"(coordinate1),
                   "r"(smem_addr(barrier))
                 : "memory");
}

// 64B swizzle places the sixteen-byte segment of a row at segment XOR ((row / 2) % 4). The
// pattern closes after eight rows, so the unit it repeats over is kFp8A8TmaSwizzleAtomBytes, and
// this mapping is the hardware's only for a tile whose base is at the start of one. The row here
// is relative to that base; Fp8A8TmaTensorStorage is what makes the base sit where that is true.
__device__ __forceinline__ int fp8_tma_shared_byte(int row, int logical_byte) {
    return ((logical_byte >> 4) ^ ((row >> 1) & 3)) * 16 + (logical_byte & 15);
}

template <class Geometry, class Schedule, class Epilogue, class Output,
          class RowPolicy = Fp8MmaIdentityRows, bool PairRows = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a8_tma_kernel(
    const __grid_constant__ Fp8A8TmaDescriptors descriptors,
    const float* __restrict__ activation_scales, const __nv_bfloat16* __restrict__ weight_scales,
    std::int32_t tokens, const __grid_constant__ Epilogue epilogue,
    const __grid_constant__ Output output, const __grid_constant__ RowPolicy row_policy) {
    static_assert(kFp8A8TmaRepresentable<Geometry, Schedule>,
                  "the geometry must tile evenly in both K and output rows");
    // The weight scales are fetched one 32-bit word at a time at parent_row0 and applied to
    // parent_row0 and parent_row1, which is right only while the row policy maps adjacent local
    // rows to adjacent parent rows. Fp8SwiGluRows does not, at its branch boundary: there it jumps
    // by the intermediate size. local_row0 is always even, so a pair can only straddle that
    // boundary if the branch width is odd, and the branch width is kBlockN / 2 - hence % 4, not
    // % 2. It cannot fire while kBlockN is fixed at 128; it is here so the invariant is written
    // down where the pair is loaded rather than rediscovered later.
    // The unpaired case needs the identity policy for an unrelated reason: the producer loads
    // B and the store writes at the raw row while the epilogue would apply the mapped one, and
    // nothing reconciles the two.
    static_assert(PairRows || std::is_same_v<RowPolicy, Fp8MmaIdentityRows>,
                  "a non-identity row policy is only wired up for the paired store");
    static_assert(!PairRows || (Schedule::kBlockN % 4) == 0,
                  "paired rows need an even branch width so a scale pair cannot straddle it");

    // The alignment the swizzle indexing needs, declared on the allocation that backs it. The
    // TMA store alone would be satisfied by 128.
    extern __shared__ __align__(kFp8A8TmaSwizzleAtomBytes) unsigned char shared_bytes[];
    auto& shared              = *reinterpret_cast<Fp8A8TmaSharedStorage<Schedule>*>(shared_bytes);
    constexpr int kBranchRows = PairRows ? Schedule::kBlockN / 2 : Schedule::kBlockN;
    const int token_begin     = static_cast<int>(blockIdx.y) * Schedule::kBlockM;
    const int row_begin       = static_cast<int>(blockIdx.x) * kBranchRows;

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
        if (threadIdx.x == 0) {
#pragma unroll 1
            for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
                const int stage                 = k_tile % Schedule::kStages;
                const std::uint32_t empty_phase = 1U ^ ((k_tile / Schedule::kStages) & 1U);
                cta_mbarrier_wait(&shared.empty[stage], empty_phase);
                constexpr std::uint32_t kTransactionBytes =
                    (Schedule::kBlockM + Schedule::kBlockN) * Schedule::kRowBytes;
                cta_mbarrier_arrive_expect_tx(&shared.full[stage], kTransactionBytes);

                auto& tensors = shared.scratch.tensors;
                fp8_tma_load_2d(tensors.a_codes[stage], &descriptors.a_codes,
                                k_tile * Schedule::kRowBytes, token_begin, &shared.full[stage]);
                if constexpr (PairRows) {
                    fp8_tma_load_2d(tensors.b_codes[stage], &descriptors.b_codes,
                                    k_tile * Schedule::kRowBytes,
                                    row_policy.weight_row(row_begin, 0), &shared.full[stage]);
                    fp8_tma_load_2d(tensors.b_codes[stage] + kBranchRows * Schedule::kRowBytes,
                                    &descriptors.b_codes, k_tile * Schedule::kRowBytes,
                                    row_policy.weight_row(row_begin, kBranchRows),
                                    &shared.full[stage]);
                } else {
                    fp8_tma_load_2d(tensors.b_codes[stage], &descriptors.b_codes,
                                    k_tile * Schedule::kRowBytes, row_begin, &shared.full[stage]);
                }
            }
        }
        return;
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

    float accumulators[Schedule::kMmaM][Schedule::kMmaN][4] = {};
#pragma unroll 1
    for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
        const int stage                = k_tile % Schedule::kStages;
        const std::uint32_t full_phase = (k_tile / Schedule::kStages) & 1U;
        cta_mbarrier_wait(&shared.full[stage], full_phase);

#pragma unroll
        for (int k_step = 0; k_step < Schedule::kMmaKPerStage; ++k_step) {
            unsigned a_fragments[Schedule::kMmaM][4];
            unsigned b_fragments[Schedule::kMmaN][2];
#pragma unroll
            for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
                const int row       = warp_m * Schedule::kWarpM + mma_m * 16 + a_row_offset;
                const auto* address = tensors.a_codes[stage] + row * Schedule::kRowBytes +
                                      fp8_tma_shared_byte(row, k_step * 32 + a_column_byte);
                ldmatrix_x4(a_fragments[mma_m][0], a_fragments[mma_m][1], a_fragments[mma_m][2],
                            a_fragments[mma_m][3], smem_addr(address));
            }
#pragma unroll
            for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
                const int row       = warp_n * Schedule::kWarpN + mma_n * 8 + b_row_offset;
                const auto* address = tensors.b_codes[stage] + row * Schedule::kRowBytes +
                                      fp8_tma_shared_byte(row, k_step * 32 + b_column_byte);
                ldmatrix_x2(b_fragments[mma_n][0], b_fragments[mma_n][1], smem_addr(address));
            }
#pragma unroll
            for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
#pragma unroll
                for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
                    mma_fp8_e4m3(accumulators[mma_m][mma_n][0], accumulators[mma_m][mma_n][1],
                                 accumulators[mma_m][mma_n][2], accumulators[mma_m][mma_n][3],
                                 a_fragments[mma_m][0], a_fragments[mma_m][1],
                                 a_fragments[mma_m][2], a_fragments[mma_m][3],
                                 b_fragments[mma_n][0], b_fragments[mma_n][1]);
                }
            }
        }
        if (lane == 0) { cta_mbarrier_arrive(&shared.empty[stage]); }
    }

    // The epilogue reuses the tensor pipeline's storage, so every consumer warp must finish its
    // last tensor read before any warp starts overwriting it.
    asm volatile("bar.sync 1, %0;" : : "r"(Schedule::kConsumerThreads) : "memory");

    constexpr int kOutputStride = Schedule::kBlockN + 8;
    auto* shared_output         = shared.scratch.output;
    const int accumulator_token = lane >> 2;
    const int accumulator_row   = 2 * (lane & 3);
#pragma unroll
    for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
        const int local_token0 = warp_m * Schedule::kWarpM + mma_m * 16 + accumulator_token;
        const int local_token1 = local_token0 + 8;
        const int token0       = token_begin + local_token0;
        const int token1       = token_begin + local_token1;
        // TMA zero-fills the rows past the end, so their accumulators are harmless; they are
        // simply never stored. Nothing may be read on their behalf either: the activation scale is
        // indexed by token, and so is the residual an epilogue may add, so both are gated on the
        // row existing. Rows that do exist take exactly the arithmetic they took before.
        const bool token0_valid       = token0 < tokens;
        const bool token1_valid       = token1 < tokens;
        const float activation_scale0 = token0_valid ? activation_scales[token0] : 0.0F;
        const float activation_scale1 = token1_valid ? activation_scales[token1] : 0.0F;
#pragma unroll
        for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
            const int local_row0  = warp_n * Schedule::kWarpN + mma_n * 8 + accumulator_row;
            const int parent_row0 = row_policy.weight_row(row_begin, local_row0);
            const int parent_row1 = row_policy.weight_row(row_begin, local_row0 + 1);
            const std::uint32_t scale_bits = load_vec<std::uint32_t>(weight_scales + parent_row0);
            const float2 weight_scale      = bf16x2_bits_to_float2(scale_bits);
            const float value00            = token0_valid
                                                 ? epilogue.apply(parent_row0, token0,
                                                                  accumulators[mma_m][mma_n][0] *
                                                                      activation_scale0 * weight_scale.x)
                                                 : 0.0F;
            const float value01            = token0_valid
                                                 ? epilogue.apply(parent_row1, token0,
                                                                  accumulators[mma_m][mma_n][1] *
                                                                      activation_scale0 * weight_scale.y)
                                                 : 0.0F;
            const float value10            = token1_valid
                                                 ? epilogue.apply(parent_row0, token1,
                                                                  accumulators[mma_m][mma_n][2] *
                                                                      activation_scale1 * weight_scale.x)
                                                 : 0.0F;
            const float value11            = token1_valid
                                                 ? epilogue.apply(parent_row1, token1,
                                                                  accumulators[mma_m][mma_n][3] *
                                                                      activation_scale1 * weight_scale.y)
                                                 : 0.0F;
            auto* destination0             = reinterpret_cast<__nv_bfloat162*>(
                shared_output + local_token0 * kOutputStride + local_row0);
            auto* destination1 = reinterpret_cast<__nv_bfloat162*>(
                shared_output + local_token1 * kOutputStride + local_row0);
            *destination0 = __floats2bfloat162_rn(value00, value01);
            *destination1 = __floats2bfloat162_rn(value10, value11);
        }
    }

    asm volatile("bar.sync 1, %0;" : : "r"(Schedule::kConsumerThreads) : "memory");
    constexpr int kVectorsPerRow = kBranchRows / 8;
    constexpr int kOutputVectors = Schedule::kBlockM * kVectorsPerRow;
    for (int task = consumer_thread; task < kOutputVectors; task += Schedule::kConsumerThreads) {
        const int local_token = task / kVectorsPerRow;
        if (token_begin + local_token >= tokens) { continue; }
        const int row_vector = task - local_token * kVectorsPerRow;
        const auto* row_base = shared_output + local_token * kOutputStride + row_vector * 8;
        const uint4 values   = load_vec<uint4>(row_base);
        if constexpr (PairRows) {
            output.store_pair_vector(row_begin + row_vector * 8, token_begin + local_token, values,
                                     load_vec<uint4>(row_base + kBranchRows));
        } else {
            output.store_vector(row_begin + row_vector * 8, token_begin + local_token, values);
        }
    }
}

// One launch path for every Op family: they differ only in the epilogue and output policies.
template <class Geometry, class Schedule, class Epilogue, class Output,
          class RowPolicy = Fp8MmaIdentityRows, bool PairRows = false>
void fp8_a8_tma_launch(const std::uint8_t* activation_codes, const float* activation_scales,
                       const std::uint8_t* weight_codes, const __nv_bfloat16* weight_scales,
                       std::int32_t tokens, Epilogue epilogue, Output output, cudaStream_t stream,
                       RowPolicy row_policy = {}) {
    constexpr std::size_t kSharedBytes = sizeof(Fp8A8TmaSharedStorage<Schedule>);
    static_assert(kSharedBytes <= 99 * 1024);
    // Per device, not per process: the opt-in raises the dynamic shared limit on the current
    // device only, so a process that reaches a second GPU would otherwise launch there without it
    // and fail at 96 KiB. Keyed by ordinal, set once per device.
    static std::array<std::atomic<bool>, kFp8A8MaxDevices> raised{};
    int attribute_device = 0;
    // Throwing here, where the routing guard merely declines: the guard can fall back to the
    // cp.async route and produce the right answer, but by this point the launch is committed and a
    // kernel that needs 96 KiB would fail on the device we could not name.
    if (cudaGetDevice(&attribute_device) != cudaSuccess || attribute_device < 0 ||
        attribute_device >= kFp8A8MaxDevices) {
        throw std::runtime_error("fp8 TMA: cannot identify the current device");
    }
    // Acquire/release rather than relaxed for consistency with the multiprocessor-count cache,
    // which is the same shape of problem. Neither orders the driver-side effect itself - C++
    // atomics cannot - and neither needs to: setting the attribute twice is idempotent, and a
    // thread that reads false when another has already set it simply sets it again.
    if (!raised[attribute_device].load(std::memory_order_acquire)) {
        const cudaError_t attribute = cudaFuncSetAttribute(
            fp8_a8_tma_kernel<Geometry, Schedule, Epilogue, Output, RowPolicy, PairRows>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSharedBytes));
        if (attribute != cudaSuccess) {
            throw std::runtime_error(std::string("fp8 TMA shared memory attribute: ") +
                                     cudaGetErrorName(attribute));
        }
        raised[attribute_device].store(true, std::memory_order_release);
    }

    const Fp8A8TmaDescriptors descriptors =
        make_fp8_a8_tma_descriptors<Geometry, Schedule::kBlockM, Schedule::kBlockN, PairRows>(
            activation_codes, weight_codes, tokens);
    const auto token_tiles = fp8_a8_tma_token_tiles<Geometry, Schedule>(tokens);
    const dim3 blocks(Geometry::kOutputRows / Schedule::kBlockN,
                      static_cast<unsigned>(token_tiles));
    fp8_a8_tma_kernel<Geometry, Schedule, Epilogue, Output, RowPolicy, PairRows>
        <<<blocks, Schedule::kThreads, kSharedBytes, stream>>>(
            descriptors, activation_scales, weight_scales, tokens, epilogue, output, row_policy);
}

} // namespace ninfer::ops::detail
