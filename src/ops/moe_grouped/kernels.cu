// Implements: include/ninfer/ops/moe_grouped.h (moe_grouped_nvfp4)
//
// Derived from giveen/ninfer-ext (commit 9c933f01, Apache License 2.0):
//   * src/ops/offload_moe/a4_kernels.cu - the grouped W4A4 expert pipeline (count, scan, place,
//     quantize, gate/up with fused SwiGLU + NVFP4 requantization, down, k-ordered merge);
//   * src/ops/linear/nvfp4/nvfp4_w4a4_mma.cuh - the routed NVFP4 tensor-core GEMM kernel with
//     row/token/raster policies.
// Changes in this tree: experts are addressed through a slot table over a planar slot pool (any
// slot of any layer, the ExpertCache layout) instead of a staged bank indexed by expert id; routes
// whose expert has no slot are skipped; the weight divisors are looked up by expert id; the
// output is the FP32 routed partial (Store, no shared expert); the activation quantizer is this
// tree's quantize_nvfp4_k16; the TMA/tiled-scale variants are not carried over.
//
// Pipeline per piece of <= kMoeGroupedChunkColumns tokens:
//   * assignments (token, k) with a slot are counted per expert and placed in expert order;
//   * x is quantized once with the layer's gate/up input divisor;
//   * a job is one 64-column tile of one expert: gate/up reads that expert's 1280 rows from its
//     slot and only its columns, applies SwiGLU and writes h as NVFP4 with the down divisor;
//   * down writes one FP32 row per assignment; the merge sums the routes of a token in k order.
#include "ops/moe_grouped/launch.h"
#include "ops/moe_grouped/grouped_scan.cuh"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int H = kMoeSlotsHidden; // 2560
constexpr int E = 512;
constexpr int K = kMoeSlotsTopK;         // 10
constexpr int I = kMoeSlotsIntermediate; // 640

// ---- routed NVFP4 W4A4 MMA kernel (from ninfer-ext nvfp4_w4a4_mma.cuh) ------------------------

template <int BlockM, int BlockN, int BlockK, int WarpsM, int WarpsN, int Stages,
          int MinBlocksPerSm>
struct MmaSchedule {
    static_assert(BlockM > 0 && (BlockM % 16) == 0);
    static_assert(BlockN > 0 && (BlockN % 8) == 0);
    static_assert(BlockK >= 64 && (BlockK % 64) == 0);
    static_assert((BlockM % WarpsM) == 0 && ((BlockM / WarpsM) % 16) == 0);
    static_assert((BlockN % WarpsN) == 0 && ((BlockN / WarpsN) % 8) == 0);
    static_assert(Stages >= 1 && Stages <= 6);

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

// Rows and input width of one weight matrix (per expert); the stored scales are the
// K16M128x4 swizzled plane, so every 128-row tile of a slot is self-contained.
template <int OutputRows, int InputRows>
struct Geometry {
    static_assert((OutputRows % 128) == 0 && (InputRows % 64) == 0);
    static constexpr int kOutputRows       = OutputRows;
    static constexpr int kInputRows        = InputRows;
    static constexpr int kGroupsPerRow     = InputRows / 16;
    static constexpr int kScaleTilesPerRow = InputRows / 64;
    static constexpr int kCodeBytesPerRow  = InputRows / 2;
};

struct Activation {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
};

struct IdentityRows {
    static constexpr bool kContiguous = true;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + local_row;
    }
};

template <class Schedule>
struct SharedStorage {
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
__device__ __forceinline__ int swizzled_byte(int row, int logical_byte) {
    static_assert((Schedule::kSegmentsPerRow & (Schedule::kSegmentsPerRow - 1)) == 0);
    const int logical_segment  = logical_byte >> 4;
    const int byte_in_segment  = logical_byte & 15;
    const int physical_segment = logical_segment ^ (row & (Schedule::kSegmentsPerRow - 1));
    return physical_segment * 16 + byte_in_segment;
}

template <class G, class Schedule, class TokenPolicy>
__device__ __forceinline__ void stage_activation(Activation source, SharedStorage<Schedule>& shared,
                                                 int stage, int k_tile, int token_begin,
                                                 int active_tokens, TokenPolicy token_policy) {
    constexpr int kCodeTasks = Schedule::kBlockM * Schedule::kSegmentsPerRow;
    for (int task = static_cast<int>(threadIdx.x); task < kCodeTasks; task += Schedule::kThreads) {
        const int row             = task / Schedule::kSegmentsPerRow;
        const int logical_segment = task - row * Schedule::kSegmentsPerRow;
        const int token           = token_begin + row;
        const bool valid          = token < active_tokens;
        const int source_token    = valid ? token_policy.source_token(token) : 0;
        const int physical_byte   = swizzled_byte<Schedule>(row, logical_segment * 16);
        auto* destination = shared.a_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
        const auto* input = source.codes +
                            static_cast<std::int64_t>(source_token) * G::kCodeBytesPerRow +
                            k_tile * Schedule::kCodeRowBytes + logical_segment * 16;
        cp_async_zfill<16, Cache::cg>(destination, input, valid ? 16 : 0);
    }
    static_assert(Schedule::kK64PerStage <= 4);
    constexpr int kScaleBytes = Schedule::kK64PerStage * 4;
    for (int row = static_cast<int>(threadIdx.x); row < Schedule::kBlockM;
         row += Schedule::kThreads) {
        const int token        = token_begin + row;
        const bool valid       = token < active_tokens;
        const int source_token = valid ? token_policy.source_token(token) : 0;
        auto* destination      = &shared.a_scale4[stage][row * Schedule::kK64PerStage];
        const auto* input =
            source.scales + (static_cast<std::int64_t>(source_token) * G::kScaleTilesPerRow +
                             k_tile * Schedule::kK64PerStage) *
                                4;
        cp_async_zfill<kScaleBytes>(destination, input, valid ? kScaleBytes : 0);
    }
}

template <class G, class Schedule, class RowPolicy>
__device__ __forceinline__ void stage_weight(const std::uint8_t* __restrict__ codes,
                                             const std::uint8_t* __restrict__ scales,
                                             SharedStorage<Schedule>& shared, int stage, int k_tile,
                                             int row_begin, RowPolicy row_policy) {
    constexpr int kCodeTasks = Schedule::kBlockN * Schedule::kSegmentsPerRow;
    for (int task = static_cast<int>(threadIdx.x); task < kCodeTasks; task += Schedule::kThreads) {
        const int row             = task / Schedule::kSegmentsPerRow;
        const int logical_segment = task - row * Schedule::kSegmentsPerRow;
        const int weight_row      = row_policy.weight_row(row_begin, row);
        const int physical_byte   = swizzled_byte<Schedule>(row, logical_segment * 16);
        auto* destination = shared.b_codes[stage] + row * Schedule::kCodeRowBytes + physical_byte;
        const auto* input = codes + static_cast<std::int64_t>(weight_row) * G::kCodeBytesPerRow +
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
            const auto* input =
                scales +
                (static_cast<std::int64_t>(persistent_tile) * G::kScaleTilesPerRow + global_k64) *
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
            const auto* input =
                scales +
                (static_cast<std::int64_t>(persistent_tile) * G::kScaleTilesPerRow + global_k64) *
                    512 +
                row_mod32 * 16 + row_quartile * 4;
            cp_async<4>(destination, input);
        }
    }
}

// The epilogue stages the tile in shared memory, as BF16 (gate/up: the SwiGLU input is rounded to
// BF16 by the recipe) or as FP32 (down: the per-expert output stays FP32 until the merge, as in
// moe_slots_nvfp4).
template <class Schedule, bool FloatStage>
union KernelShared {
    SharedStorage<Schedule> pipe;
    __nv_bfloat16 out_h[Schedule::kBlockM * (Schedule::kBlockN + 8)];
    float out_f[FloatStage ? Schedule::kBlockM * (Schedule::kBlockN + 4) : 1];
};

template <class G, class Schedule, class Epilogue, class OutputPolicy, class RowPolicy,
          bool PairRows, class TokenPolicy, class RasterPolicy, bool FloatStage = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void grouped_mma_kernel(
    Activation activation, const std::uint8_t* __restrict__ weight_codes,
    const std::uint8_t* __restrict__ weight_scales, float alpha, Epilogue epilogue,
    OutputPolicy output, RowPolicy row_policy, TokenPolicy token_policy,
    RasterPolicy raster_policy) {
    static_assert((G::kInputRows % Schedule::kBlockK) == 0);
    static_assert((G::kOutputRows % Schedule::kBlockN) == 0);
    static_assert(!PairRows || ((G::kOutputRows / 2) % (Schedule::kBlockN / 2)) == 0);

    __shared__ KernelShared<Schedule, FloatStage> storage;
    SharedStorage<Schedule>& shared = storage.pipe;
    // A tile past the device-side job count has no work-list entry: leave before reading one.
    if (!raster_policy.live()) { return; }
    int block_row   = 0;
    int block_token = 0;
    raster_policy.blocks(block_row, block_token);
    const int token_begin       = block_token * Schedule::kBlockM;
    constexpr int kRowsPerBlock = PairRows ? Schedule::kBlockN / 2 : Schedule::kBlockN;
    const int row_begin         = block_row * kRowsPerBlock;
    const int active            = token_policy.active_tokens();
    constexpr int kKTiles       = G::kInputRows / Schedule::kBlockK;
    constexpr int kWaitGroups   = kKTiles < Schedule::kStages ? kKTiles - 1 : Schedule::kStages - 1;

#pragma unroll
    for (int stage = 0; stage < Schedule::kStages; ++stage) {
        if (stage < kKTiles) {
            stage_activation<G, Schedule>(activation, shared, stage, stage, token_begin, active,
                                          token_policy);
            stage_weight<G, Schedule>(weight_codes, weight_scales, shared, stage, stage, row_begin,
                                      row_policy);
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
                const int physical_byte = swizzled_byte<Schedule>(row, logical_byte);
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
                const int physical_byte = swizzled_byte<Schedule>(row, logical_byte);
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
            stage_activation<G, Schedule>(activation, shared, stage, next_k_tile, token_begin,
                                          active, token_policy);
            stage_weight<G, Schedule>(weight_codes, weight_scales, shared, stage, next_k_tile,
                                      row_begin, row_policy);
        }
        cp_commit();
    }

    const int accumulator_row    = lane >> 2;
    const int accumulator_col    = 2 * (lane & 3);
    constexpr int kOutputStride  = FloatStage ? Schedule::kBlockN + 4 : Schedule::kBlockN + 8;
    __nv_bfloat16* shared_output = storage.out_h;
    float* shared_output_f       = storage.out_f;
#pragma unroll
    for (int mma_m = 0; mma_m < Schedule::kMmaM; ++mma_m) {
        const int token0 = token_begin + warp_m * Schedule::kWarpM + mma_m * 16 + accumulator_row;
        const int token1 = token0 + 8;
#pragma unroll
        for (int mma_n = 0; mma_n < Schedule::kMmaN; ++mma_n) {
            const int local_row0  = warp_n * Schedule::kWarpN + mma_n * 8 + accumulator_col;
            const int parent_row0 = row_policy.weight_row(row_begin, local_row0);
            const int parent_row1 = row_policy.weight_row(row_begin, local_row0 + 1);
            float value00         = accumulators[mma_m][mma_n][0] * alpha;
            float value01         = accumulators[mma_m][mma_n][1] * alpha;
            float value10         = accumulators[mma_m][mma_n][2] * alpha;
            float value11         = accumulators[mma_m][mma_n][3] * alpha;
            if (token0 < active) {
                value00 = epilogue.apply(parent_row0, value00);
                value01 = epilogue.apply(parent_row1, value01);
            }
            if (token1 < active) {
                value10 = epilogue.apply(parent_row0, value10);
                value11 = epilogue.apply(parent_row1, value11);
            }
            const int at0 = (token0 - token_begin) * kOutputStride + local_row0;
            const int at1 = (token1 - token_begin) * kOutputStride + local_row0;
            if constexpr (FloatStage) {
                *reinterpret_cast<float2*>(shared_output_f + at0) = make_float2(value00, value01);
                *reinterpret_cast<float2*>(shared_output_f + at1) = make_float2(value10, value11);
            } else {
                *reinterpret_cast<__nv_bfloat162*>(shared_output + at0) =
                    __floats2bfloat162_rn(value00, value01);
                *reinterpret_cast<__nv_bfloat162*>(shared_output + at1) =
                    __floats2bfloat162_rn(value10, value11);
            }
        }
    }
    __syncthreads();
    constexpr int kStoredRows    = PairRows ? Schedule::kBlockN / 2 : Schedule::kBlockN;
    constexpr int kVectorsPerRow = kStoredRows / 8;
    // The gate/up output completes an NVFP4 group across a lane pair: both lanes must hold the
    // same token, which holds while a row's vectors come in pairs.
    static_assert((kVectorsPerRow % 2) == 0);
    constexpr int kOutputVectors = Schedule::kBlockM * kVectorsPerRow;
    for (int task = static_cast<int>(threadIdx.x); task < kOutputVectors;
         task += Schedule::kThreads) {
        const int token_local = task / kVectorsPerRow;
        const int row_vector  = task - token_local * kVectorsPerRow;
        const int token       = token_begin + token_local;
        if (token < active) {
            if constexpr (FloatStage) {
                const float* row = shared_output_f + token_local * kOutputStride + row_vector * 8;
                output.store_vector(row_vector * 8, token, load_vec<float4>(row),
                                    load_vec<float4>(row + 4));
            } else if constexpr (PairRows) {
                const uint4 values =
                    load_vec<uint4>(shared_output + token_local * kOutputStride + row_vector * 8);
                const uint4 paired = load_vec<uint4>(shared_output + token_local * kOutputStride +
                                                     kStoredRows + row_vector * 8);
                output.store_pair_vector(row_vector * 8, token, values, paired);
            } else {
                const uint4 values =
                    load_vec<uint4>(shared_output + token_local * kOutputStride + row_vector * 8);
                output.store_vector(row_vector * 8, token, values);
            }
        }
    }
}

// ---- grouped expert pipeline (from ninfer-ext a4_kernels.cu) ---------------------------------

// The routed schedule ninfer-ext settled on: a 128-deep tile keeps staging small enough for three
// CTAs an SM; a job is one 64-column tile of one expert.
using Schedule = MmaSchedule<64, 128, 128, 2, 4, 2, 3>;
static_assert(Schedule::kBlockM == kMoeGroupedJobColumns);
constexpr int kPairRows = Schedule::kBlockN / 2;
static_assert((kPairRows & (kPairRows - 1)) == 0 && I % kPairRows == 0);
static_assert(H % Schedule::kBlockN == 0);

using GateUpGeometry = Geometry<2 * I, H>;
using DownGeometry   = Geometry<H, I>;

// The grid is sized from a host bound on the work list; the real count lives on the device.
struct Jobs {
    const std::int32_t* __restrict__ experts;
    const std::int32_t* __restrict__ columns;
    const std::int32_t* __restrict__ slots;
    const std::int32_t* __restrict__ count;

    __device__ __forceinline__ int expert() const { return experts[blockIdx.y]; }

    __device__ __forceinline__ int slot() const { return slots[blockIdx.y]; }

    __device__ __forceinline__ int column_base() const { return columns[blockIdx.y]; }

    __device__ __forceinline__ bool live() const { return static_cast<int>(blockIdx.y) < *count; }
};

struct GateUpRaster {
    Jobs jobs;

    __device__ __forceinline__ void blocks(int& block_row, int& block_token) const {
        block_row   = static_cast<int>(blockIdx.x);
        block_token = jobs.column_base() / Schedule::kBlockM;
    }

    __device__ __forceinline__ bool live() const { return jobs.live(); }
};

// Down rows are contiguous inside a slot: folding the slot's row base into the block row lets the
// identity row policy and the contiguous scale staging address the plane.
struct DownRaster {
    Jobs jobs;

    __device__ __forceinline__ void blocks(int& block_row, int& block_token) const {
        block_row   = jobs.slot() * (H / Schedule::kBlockN) + static_cast<int>(blockIdx.x);
        block_token = jobs.column_base() / Schedule::kBlockM;
    }

    __device__ __forceinline__ bool live() const { return jobs.live(); }
};

// A pair tile stages kPairRows gate rows and the matching up rows of the job's slot.
struct GateUpRows {
    static constexpr bool kContiguous = false;
    Jobs jobs;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        const int within =
            row_begin + (local_row & (kPairRows - 1)) + (local_row >= kPairRows ? I : 0);
        return jobs.slot() * (2 * I) + within;
    }
};

// gate/up reads the piece's quantized activations through the grouping.
struct GatherTokens {
    const std::int32_t* __restrict__ packed_token;
    const std::int32_t* __restrict__ offsets;
    Jobs jobs;

    __device__ __forceinline__ int source_token(int column) const {
        return packed_token[offsets[jobs.expert()] + column];
    }

    __device__ __forceinline__ int active_tokens() const {
        const int expert = jobs.expert();
        return offsets[expert + 1] - offsets[expert];
    }
};

// down reads what gate/up wrote, already in grouped order.
struct PackedTokens {
    const std::int32_t* __restrict__ offsets;
    Jobs jobs;

    __device__ __forceinline__ int source_token(int column) const {
        return offsets[jobs.expert()] + column;
    }

    __device__ __forceinline__ int active_tokens() const {
        const int expert = jobs.expert();
        return offsets[expert + 1] - offsets[expert];
    }
};

// T56 wide calls: down reads the call's grouped h through the piece's row map.
struct GatherMiddle {
    const std::int32_t* __restrict__ middle_row;
    const std::int32_t* __restrict__ offsets;
    Jobs jobs;

    __device__ __forceinline__ int source_token(int column) const {
        return middle_row[offsets[jobs.expert()] + column];
    }

    __device__ __forceinline__ int active_tokens() const {
        const int expert = jobs.expert();
        return offsets[expert + 1] - offsets[expert];
    }
};

// Weight divisors by expert id: gate of e at 2e, up at 2e+1 (row of the slot's 1280 rows).
struct GateUpDivisor {
    const float* __restrict__ divisors;
    Jobs jobs;

    __device__ __forceinline__ float apply(int row, float value) const {
        const int within = row % (2 * I);
        return value * __frcp_rn(divisors[2 * jobs.expert() + (within >= I ? 1 : 0)]);
    }
};

struct DownDivisor {
    const float* __restrict__ divisors;
    Jobs jobs;

    __device__ __forceinline__ float apply(int, float value) const {
        return value * __frcp_rn(divisors[jobs.expert()]);
    }
};

union Bf16Pair {
    unsigned bits;
    __nv_bfloat162 values;
};

__device__ __forceinline__ unsigned swiglu_pair(unsigned gate_bits, unsigned up_bits) {
    const Bf16Pair gate{gate_bits};
    const Bf16Pair up{up_bits};
    const float2 g = __bfloat1622float2(gate.values);
    const float2 u = __bfloat1622float2(up.values);
    Bf16Pair result;
    result.values = __floats2bfloat162_rn(silu(g.x) * u.x, silu(g.y) * u.y);
    return result.bits;
}

// Half an NVFP4 group (eight values) packed by one lane.
__device__ __forceinline__ std::uint32_t pack_e2m1x8(const float2 (&values)[4]) {
    std::uint32_t codes = 0;
    asm volatile("{\n"
                 ".reg .b8 b0;\n"
                 ".reg .b8 b1;\n"
                 ".reg .b8 b2;\n"
                 ".reg .b8 b3;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\n"
                 "}\n"
                 : "=r"(codes)
                 : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
                   "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y));
    return codes;
}

// One store is half an NVFP4 group of h; the group maximum is completed across the neighbouring
// lane, which always holds the same token. The CTA covers intermediate rows
// [blockIdx.x * kPairRows, + kPairRows); `local_row` is the row within them.
struct GateUpOutput {
    std::uint8_t* codes;
    std::uint8_t* scales;
    const std::int32_t* __restrict__ offsets;
    Jobs jobs;
    float divisor;

    __device__ __forceinline__ void store_pair_vector(int local_row, int column, uint4 gate,
                                                      uint4 up) const {
        const int row               = static_cast<int>(blockIdx.x) * kPairRows + local_row;
        const std::int64_t packed   = offsets[jobs.expert()] + column;
        const std::uint32_t bits[4] = {swiglu_pair(gate.x, up.x), swiglu_pair(gate.y, up.y),
                                       swiglu_pair(gate.z, up.z), swiglu_pair(gate.w, up.w)};
        float2 values[4];
        float max_abs = 0.0F;
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            values[pair] = bf16x2_bits_to_float2(bits[pair]);
            max_abs      = fmaxf(max_abs, fmaxf(fabsf(values[pair].x), fabsf(values[pair].y)));
        }
        const unsigned pair_mask    = 3U << (threadIdx.x & 31U & ~1U);
        max_abs                     = fmaxf(max_abs, __shfl_xor_sync(pair_mask, max_abs, 1));
        const float scale_unencoded = __fdiv_rn(divisor * max_abs, 6.0F);
        const std::uint8_t scale =
            __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
        std::uint32_t word = 0;
        if (scale != 0) {
            const float reciprocal = __frcp_rn(decode_nvfp4_e4m3(scale));
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                values[pair].x = values[pair].x * divisor * reciprocal;
                values[pair].y = values[pair].y * divisor * reciprocal;
            }
            word = pack_e2m1x8(values);
        }
        const int half      = (row >> 3) & 1;
        const int group_row = row & ~15;
        store_vec(codes + packed * (I / 2) + (group_row >> 1) + half * 4, word);
        if (half == 0) { scales[packed * (I / 16) + (group_row >> 4)] = scale; }
    }
};

struct DownOutput {
    float* data;
    const std::int32_t* __restrict__ offsets;
    Jobs jobs;

    __device__ __forceinline__ void store_vector(int local_row, int column, float4 a,
                                                 float4 b) const {
        const int row             = static_cast<int>(blockIdx.x) * Schedule::kBlockN + local_row;
        const std::int64_t packed = offsets[jobs.expert()] + column;
        float* destination        = data + packed * H + row;
        store_vec(destination, a);
        store_vec(destination + 4, b);
    }
};

// rank[a] orders assignment a among those of its expert; -1 = the expert has no slot.
__global__ void count_kernel(const std::int32_t* __restrict__ ids,
                             const std::int32_t* __restrict__ table, std::int32_t slots,
                             std::int32_t assignments, std::int32_t* __restrict__ counts,
                             std::int32_t* __restrict__ rank) {
    const int a = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (a >= assignments) { return; }
    const std::int32_t e = ids[a];
    bool present         = e >= 0 && e < E;
    if (present) {
        const std::int32_t s = table[e];
        present              = s >= 0 && s < slots;
    }
    rank[a] = present ? atomicAdd(&counts[e], 1) : -1;
}

__global__ void place_kernel(const std::int32_t* __restrict__ ids,
                             const std::int32_t* __restrict__ rank, std::int32_t assignments,
                             const std::int32_t* __restrict__ offsets,
                             std::int32_t* __restrict__ packed_token,
                             std::int32_t* __restrict__ packed_index) {
    const int a = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (a >= assignments) { return; }
    const int r = rank[a];
    if (r < 0) {
        packed_index[a] = -1;
        return;
    }
    const int position     = offsets[ids[a]] + r;
    packed_token[position] = a / K;
    packed_index[a]        = position;
}

// T56: a piece's grouping of a wide call: its position per assignment, and per position the row of
// the call's grouped h (the same assignment's global position).
__global__ void place_piece_kernel(const std::int32_t* __restrict__ ids,
                                   const std::int32_t* __restrict__ rank, std::int32_t assignments,
                                   const std::int32_t* __restrict__ offsets,
                                   const std::int32_t* __restrict__ global_index,
                                   std::int32_t* __restrict__ piece_index,
                                   std::int32_t* __restrict__ middle_row) {
    const int a = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (a >= assignments) { return; }
    const int r = rank[a];
    if (r < 0) {
        piece_index[a] = -1;
        return;
    }
    const int position   = offsets[ids[a]] + r;
    piece_index[a]       = position;
    middle_row[position] = global_index[a];
}

// x [columns, H] BF16 -> NVFP4 codes [columns, H/2], scales [columns, H/16] (row-major).
__global__ void quantize_kernel(const __nv_bfloat16* __restrict__ x,
                                std::uint8_t* __restrict__ codes, std::uint8_t* __restrict__ scales,
                                std::int32_t groups, float divisor) {
    const int task = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (task >= groups) { return; }
    const Nvfp4QuantizedK16 q =
        quantize_nvfp4_k16(x + static_cast<std::int64_t>(task) * 16, divisor);
    store_vec(codes + static_cast<std::int64_t>(task) * 8, make_uint2(q.codes_lo, q.codes_hi));
    scales[task] = q.scale;
}

// One CTA per column; thread i owns features [8i, 8i+8). Routes without a slot add nothing.
__global__ void __launch_bounds__(H / 8)
    merge_kernel(const float* __restrict__ grouped, const std::int32_t* __restrict__ packed_index,
                 const float* __restrict__ weights, float* __restrict__ out) {
    const int column  = static_cast<int>(blockIdx.x);
    const int feature = static_cast<int>(threadIdx.x) * 8;
    float sum[8]      = {};
#pragma unroll
    for (int k = 0; k < K; ++k) {
        const int row = packed_index[column * K + k];
        if (row < 0) { continue; }
        const float weight  = weights[column * K + k];
        const float* source = grouped + static_cast<std::int64_t>(row) * H + feature;
        const float4 a      = load_vec<float4>(source);
        const float4 b      = load_vec<float4>(source + 4);
        const float v[8]    = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) { sum[i] = fmaf(weight, v[i], sum[i]); }
    }
    float* destination = out + static_cast<std::int64_t>(column) * H + feature;
    store_vec(destination, make_float4(sum[0], sum[1], sum[2], sum[3]));
    store_vec(destination + 4, make_float4(sum[4], sum[5], sum[6], sum[7]));
}

} // namespace

namespace {

// count / scan / place of `columns` tokens' assignments: offsets, the job list, packed_token
// (grouped position -> token) and packed_index (assignment -> grouped position).
void group_launch(const std::int32_t* ids, std::int32_t columns, const std::int32_t* table,
                  std::int32_t slots, std::int32_t* counts, std::int32_t* rank,
                  std::int32_t* offsets, const Jobs& jobs, std::int32_t* packed_token,
                  std::int32_t* packed_index, cudaStream_t stream) {
    const int assignments = columns * K;
    const int blocks      = (assignments + 255) / 256;
    CUDA_CHECK(cudaMemsetAsync(counts, 0, sizeof(std::int32_t) * E, stream));
    count_kernel<<<blocks, 256, 0, stream>>>(ids, table, slots, assignments, counts, rank);
    CUDA_CHECK(cudaGetLastError());
    grouped_scan_kernel<E><<<1, E, 0, stream>>>(
        counts, table, offsets, const_cast<std::int32_t*>(jobs.experts),
        const_cast<std::int32_t*>(jobs.columns), const_cast<std::int32_t*>(jobs.slots),
        const_cast<std::int32_t*>(jobs.count));
    CUDA_CHECK(cudaGetLastError());
    if (packed_token != nullptr) {
        place_kernel<<<blocks, 256, 0, stream>>>(ids, rank, assignments, offsets, packed_token,
                                                 packed_index);
        CUDA_CHECK(cudaGetLastError());
    }
}

// x quantized once, then gate/up of every job of the grouping into middle (grouped order).
void gate_up_launch(const MoeGroupedPiece& c, const Jobs& jobs, std::int32_t max_jobs,
                    cudaStream_t stream) {
    const int groups = c.columns * (H / 16);
    quantize_kernel<<<(groups + 255) / 256, 256, 0, stream>>>(c.x, c.input_codes, c.input_scales,
                                                              groups, c.gate_up_input_divisor);
    CUDA_CHECK(cudaGetLastError());
    const Activation input{c.input_codes, c.input_scales};
    const GateUpOutput output{c.middle_codes, c.middle_scales, c.offsets, jobs,
                              c.down_input_divisor};
    const GateUpDivisor epilogue{c.gate_up_divisors, jobs};
    const dim3 grid(I / kPairRows, max_jobs);
    grouped_mma_kernel<GateUpGeometry, Schedule, GateUpDivisor, GateUpOutput, GateUpRows, true,
                       GatherTokens, GateUpRaster><<<grid, Schedule::kThreads, 0, stream>>>(
        input, c.planes[0], c.planes[1], 1.0F / c.gate_up_input_divisor, epilogue, output,
        GateUpRows{jobs}, GatherTokens{c.packed_token, c.offsets, jobs}, GateUpRaster{jobs});
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void moe_grouped_launch(const MoeGroupedPiece& c, cudaStream_t stream) {
    const Jobs jobs{c.job_experts, c.job_columns, c.job_slots, c.job_count};
    const int max_jobs = moe_grouped_max_jobs(c.columns);
    group_launch(c.ids, c.columns, c.table, c.slots, c.counts, c.rank, c.offsets, jobs,
                 c.packed_token, c.packed_index, stream);
    gate_up_launch(c, jobs, max_jobs, stream);
    {
        const Activation middle{c.middle_codes, c.middle_scales};
        const DownOutput output{c.grouped, c.offsets, jobs};
        const DownDivisor epilogue{c.down_divisors, jobs};
        const dim3 grid(H / Schedule::kBlockN, max_jobs);
        grouped_mma_kernel<DownGeometry, Schedule, DownDivisor, DownOutput, IdentityRows, false,
                           PackedTokens, DownRaster, true><<<grid, Schedule::kThreads, 0, stream>>>(
            middle, c.planes[2], c.planes[3], 1.0F / c.down_input_divisor, epilogue, output,
            IdentityRows{}, PackedTokens{c.offsets, jobs}, DownRaster{jobs});
        CUDA_CHECK(cudaGetLastError());
    }
    merge_kernel<<<c.columns, H / 8, 0, stream>>>(c.grouped, c.packed_index, c.weights, c.out);
    CUDA_CHECK(cudaGetLastError());
}

void moe_grouped_wide_launch(const MoeGroupedWide& w, cudaStream_t stream) {
    const MoeGroupedPiece& c = w.call;
    const Jobs jobs{c.job_experts, c.job_columns, c.job_slots, c.job_count};
    group_launch(c.ids, c.columns, c.table, c.slots, c.counts, c.rank, c.offsets, jobs,
                 c.packed_token, c.packed_index, stream);
    gate_up_launch(c, jobs, moe_grouped_max_jobs(c.columns), stream);
    const Jobs piece_jobs{w.piece_job_experts, w.piece_job_columns, w.piece_job_slots,
                          w.piece_job_count};
    const Activation middle{c.middle_codes, c.middle_scales};
    for (std::int32_t begin = 0; begin < c.columns; begin += kMoeGroupedChunkColumns) {
        const std::int32_t columns = std::min(kMoeGroupedChunkColumns, c.columns - begin);
        const std::int64_t k0      = static_cast<std::int64_t>(begin) * K;
        group_launch(c.ids + k0, columns, c.table, c.slots, w.piece_counts, w.piece_rank,
                     w.piece_offsets, piece_jobs, nullptr, nullptr, stream);
        const int assignments = columns * K;
        place_piece_kernel<<<(assignments + 255) / 256, 256, 0, stream>>>(
            c.ids + k0, w.piece_rank, assignments, w.piece_offsets, c.packed_index + k0,
            w.piece_index, w.piece_middle_row);
        CUDA_CHECK(cudaGetLastError());
        const DownOutput output{c.grouped, w.piece_offsets, piece_jobs};
        const DownDivisor epilogue{c.down_divisors, piece_jobs};
        const dim3 grid(H / Schedule::kBlockN, moe_grouped_max_jobs(columns));
        grouped_mma_kernel<DownGeometry, Schedule, DownDivisor, DownOutput, IdentityRows, false,
                           GatherMiddle, DownRaster, true><<<grid, Schedule::kThreads, 0, stream>>>(
            middle, c.planes[2], c.planes[3], 1.0F / c.down_input_divisor, epilogue, output,
            IdentityRows{}, GatherMiddle{w.piece_middle_row, w.piece_offsets, piece_jobs},
            DownRaster{piece_jobs});
        CUDA_CHECK(cudaGetLastError());
        merge_kernel<<<columns, H / 8, 0, stream>>>(c.grouped, w.piece_index, c.weights + k0,
                                                    c.out + static_cast<std::int64_t>(begin) * H);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
