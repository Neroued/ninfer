#pragma once

#include <bit>

#include "ops/moe/kernels/cuda/block_scaled/clamped/nvfp4/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/clamped/output.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/memory.cuh"

#include "ops/common/numeric/math.cuh"
#include "ops/common/quant/nvfp4.cuh"
#include "ops/common/quant/csf32.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4 {
using common::load_vec;
using common::warp_reduce_sum;
using common::silu;
using common::bf16x2_bits_to_float2;
using common::cp_async;
using common::cp_async_zfill;
using common::cp_commit;
using common::cp_wait;
using common::smem_addr;
using common::mma_nvfp4_e4m3;
using common::ldmatrix_x2;
using common::ldmatrix_x4;
using common::nvfp4_scale_byte_offset;
using common::nvfp4_activation_scale;
using common::quantize_nvfp4_activation_pair;
using common::nvfp4_dequant_f16x16;

// A balanced population must cross a tile boundary together. Splitting, for
// example, a 64/65-token pool into two kernels leaves both grids underfilled
// and makes it slower than the larger, uniformly 65-token population.
__device__ __forceinline__ int balanced_group_load(const Groups& groups) {
    const int low = groups.load_range[0], high = groups.load_range[1];
    return high - low <= 1 ? high : 0;
}

template <int Columns, bool Gate, int Stages = 2, int KStep = 128, int Eviction = 0,
          int Intermediate = kIntermediate, bool Native = false, bool KnownCsf = false,
          int WeightRows = 128>
struct Schedule {
    static constexpr bool native      = Native;
    static constexpr bool known_csf   = KnownCsf;
    static constexpr int columns      = Columns;
    static constexpr int stages       = Stages;
    static constexpr int k_step       = KStep;
    static constexpr int eviction     = Eviction;
    static constexpr int weight_rows  = WeightRows;
    static constexpr int threads      = (Columns >= 64 ? 2 : 1) * WeightRows;
    static constexpr int warp_columns = Columns >= 64 ? Columns / 2 : Columns;
    static constexpr int mma_columns  = warp_columns / 8;
    static constexpr int rows         = Gate ? WeightRows / 2 : WeightRows;
    static constexpr int intermediate = Intermediate;
    static constexpr int k            = Gate ? kHidden : Intermediate;
    static constexpr int min_blocks   = Columns == 128 ? 2 : Columns == 64 ? 3 : 1;
};

template <class S, bool Native = S::native>
struct CsfScratch {};

template <class S>
struct CsfScratch<S, true> {
    common::Csf32RowBases128 bases;
    common::Csf32EncodedTile128x4 encoded[S::stages][S::k_step / 64];
};

template <class S>
struct TileStorage : CsfScratch<S> {
    alignas(16) std::uint8_t weight[S::stages][S::weight_rows * (S::k_step / 2)];
    alignas(16) std::uint8_t input[S::stages][S::columns * (S::k_step / 2)];
    alignas(16) unsigned weight_scales[S::stages][128 * (S::k_step / 64)];
    alignas(16) unsigned input_scales[S::stages][S::columns * (S::k_step / 64)];
    int sources[S::columns];
};

template <class S>
__device__ __forceinline__ int tile_offset(int row, int byte) {
    // ldmatrix consumes eight 16-byte rows per wavefront. Mix all three row bits
    // into the bank address; for a K128 tile, XOR the complete offset so crossing
    // a 64-byte row permutes rows rather than aliasing the following row.
    if constexpr (S::native) return (row * (S::k_step / 2) + byte) ^ ((row & 7) << 4);
    return row * (S::k_step / 2) + (byte ^ ((row & 3) << 4));
}

struct Matrix {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    float scale;
    Csf32ScaleView csf;
    bool valid = true;
};

struct ActivationScale {
    const float* values = nullptr;
    float scalar        = 1;

    __device__ __forceinline__ float get(int expert) const {
        return values ? values[expert] : scalar;
    }
};

template <bool Native = false, bool KnownCsf = false>
__device__ __forceinline__ Matrix resolve(const ExpertMatrixDeviceView& matrix, int expert) {
    if constexpr (Native) {
        const auto weight = common::nvfp4_mul_expert_matrix<KnownCsf>(matrix, expert);
        return {weight.codes, weight.scales, weight.multiplier, weight.csf, weight.valid};
    }
    const auto offset = matrix.offsets ? matrix.offsets[expert] : 0;
    return {reinterpret_cast<const std::uint8_t*>(matrix.codes) +
                (matrix.offsets ? offset : expert * matrix.code_stride),
            reinterpret_cast<const std::uint8_t*>(matrix.scales) +
                (matrix.offsets ? offset : expert * matrix.scale_stride),
            matrix.matrix_scales[expert * matrix.matrix_scale_stride]};
}

template <class S, bool Gate>
__device__ __forceinline__ uint4 scale_origins(int row0) {
    if constexpr (Gate && S::weight_rows == 64)
        return make_uint4(row0, row0 + S::intermediate, row0, row0);
    if constexpr (Gate)
        return make_uint4(row0, row0 + 32, row0 + S::intermediate, row0 + S::intermediate + 32);
    return make_uint4(row0, row0 + 32, row0 + 64, row0 + 96);
}

template <class S, bool Gate>
__device__ __forceinline__ void stage_weights(TileStorage<S>& tile, int stage, int kt, int row0,
                                              Matrix weight, int cache_policy, std::uint64_t stream,
                                              std::uint64_t retained) {
    for (int task = threadIdx.x; task < S::weight_rows * (S::k_step / 32); task += S::threads) {
        const int row = task / (S::k_step / 32), segment = task % (S::k_step / 32);
        const int wr =
            row0 +
            (Gate ? (row < S::weight_rows / 2 ? row : row - S::weight_rows / 2 + S::intermediate)
                  : row);
        const bool valid = S::k % S::k_step == 0 || kt * S::k_step + segment * 32 < S::k;
        if constexpr (S::eviction & 4) {
            if (cache_policy & 1)
                common::cp_async_cache_hint<Cache::cg>(
                    tile.weight[stage] + tile_offset<S>(row, segment * 16),
                    weight.codes + std::int64_t(wr) * (S::k / 2) +
                        (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                    valid ? 16 : 0, stream);
            else
                common::cp_async_cache_hint<Cache::ca>(
                    tile.weight[stage] + tile_offset<S>(row, segment * 16),
                    weight.codes + std::int64_t(wr) * (S::k / 2) +
                        (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                    valid ? 16 : 0, stream);
        } else if (cache_policy & 1)
            cp_async_zfill<16, Cache::cg>(tile.weight[stage] + tile_offset<S>(row, segment * 16),
                                          weight.codes + std::int64_t(wr) * (S::k / 2) +
                                              (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                                          valid ? 16 : 0);
        else
            cp_async_zfill<16>(tile.weight[stage] + tile_offset<S>(row, segment * 16),
                               weight.codes + std::int64_t(wr) * (S::k / 2) +
                                   (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                               valid ? 16 : 0);
    }
    if constexpr (S::native) {
        if (S::known_csf || weight.csf.bases) {
            const int warp  = threadIdx.x / 32;
            const auto rows = scale_origins<S, Gate>(row0);
            for (int group = warp * 4; group < S::k_step / 16; group += (S::threads / 32) * 4)
                common::csf32_copy_tile128x4_async<Gate ? 2 * S::intermediate : kHidden, S::k / 16,
                                                   S::weight_rows / 32>(
                    weight.csf, rows, kt * (S::k_step / 16) + group,
                    tile.encoded[stage][group / 4]);
            return;
        }
    }
    constexpr int parts = Gate ? 2 : 1;
    constexpr int bytes = Gate ? S::weight_rows / 16 : S::weight_rows / 8;
    for (int task = threadIdx.x; task < (S::k_step / 64) * parts * 32; task += S::threads) {
        const int row = task % 32, part = (task / 32) % parts, k64 = task / (32 * parts);
        const int wr      = row0 + row + (Gate ? part * S::intermediate : 0);
        auto* destination = reinterpret_cast<std::uint8_t*>(tile.weight_scales[stage]) + k64 * 512 +
                            row * 16 + part * bytes;
        const bool valid = S::k % S::k_step == 0 || kt * S::k_step + k64 * 64 < S::k;
        cp_async_zfill<bytes>(
            destination,
            weight.scales +
                nvfp4_scale_byte_offset(wr, valid ? kt * (S::k_step / 16) + k64 * 4 : 0, S::k),
            valid ? bytes : 0);
    }
}

template <class S, bool Gate>
__device__ __forceinline__ void
stage_inputs(TileStorage<S>& tile, int stage, int kt, int row0, Matrix weight,
             const std::uint8_t* x, const std::uint8_t* scales, int cache_policy,
             std::uint64_t stream, std::uint64_t retained, int active_columns) {
    stage_weights<S, Gate>(tile, stage, kt, row0, weight, cache_policy, stream, retained);
    for (int task = threadIdx.x; task < active_columns * (S::k_step / 32); task += S::threads) {
        const int row = task / (S::k_step / 32), segment = task % (S::k_step / 32);
        const int source = tile.sources[row];
        const bool valid =
            source >= 0 && (S::k % S::k_step == 0 || kt * S::k_step + segment * 32 < S::k);
        if constexpr (S::eviction & 8) {
            if (cache_policy & 2)
                common::cp_async_cache_hint<Cache::cg>(
                    tile.input[stage] + tile_offset<S>(row, segment * 16),
                    x + std::int64_t(source >= 0 ? source : 0) * (S::k / 2) +
                        (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                    valid ? 16 : 0, retained);
            else
                common::cp_async_cache_hint<Cache::ca>(
                    tile.input[stage] + tile_offset<S>(row, segment * 16),
                    x + std::int64_t(source >= 0 ? source : 0) * (S::k / 2) +
                        (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                    valid ? 16 : 0, retained);
        } else if (cache_policy & 2)
            cp_async_zfill<16, Cache::cg>(tile.input[stage] + tile_offset<S>(row, segment * 16),
                                          x + std::int64_t(source >= 0 ? source : 0) * (S::k / 2) +
                                              (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                                          valid ? 16 : 0);
        else
            cp_async_zfill<16>(tile.input[stage] + tile_offset<S>(row, segment * 16),
                               x + std::int64_t(source >= 0 ? source : 0) * (S::k / 2) +
                                   (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                               valid ? 16 : 0);
    }
    for (int task = threadIdx.x; task < active_columns * (S::k_step / 128); task += S::threads) {
        const int row = task / (S::k_step / 128), part = task % (S::k_step / 128);
        const int source = tile.sources[row];
        const bool valid =
            source >= 0 && (S::k % S::k_step == 0 || kt * S::k_step + part * 128 < S::k);
        cp_async_zfill<8>(&tile.input_scales[stage][row * (S::k_step / 64) + part * 2],
                          scales + std::int64_t(source >= 0 ? source : 0) * (S::k / 16) +
                              (valid ? kt * (S::k_step / 16) + part * 8 : 0),
                          valid ? 8 : 0);
    }
    cp_commit();
}

template <class S, bool Gate, bool FullColumns = false>
__device__ __forceinline__ void
contract_tile(TileStorage<S>& tile, int row0, Matrix weight, const std::uint8_t* input,
              const std::uint8_t* input_scales, int begin, int end, int cache_policy,
              std::uint64_t stream, std::uint64_t retained, float (&acc)[2][S::mma_columns][4]) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr int k_tiles = (S::k + S::k_step - 1) / S::k_step;
    // Spread concurrent native tiles over K instead of having every CTA issue
    // the same strided column segment. Each weight and input segment is read once.
    const int rotation  = S::native ? int(blockIdx.x) % k_tiles : 0;
    const auto physical = [&](int tile) {
        if constexpr (S::known_csf) return (tile + rotation) % k_tiles;
        return tile;
    };
    if constexpr (S::native) {
        if ((S::known_csf || weight.csf.bases) && warp == 0)
            common::csf32_copy_row_bases_async<S::weight_rows / 32>(
                weight.csf, scale_origins<S, Gate>(row0), tile.bases);
    }

    const int warp_m = warp % (S::weight_rows / 32), warp_n = warp / (S::weight_rows / 32);
    const int active_columns = FullColumns || (S::columns <= 32 && !S::native)
                                   ? S::columns
                                   : min(S::columns, (end - begin + 7) / 8 * 8);
#pragma unroll
    for (int stage = 0; stage < S::stages; ++stage) {
        stage_inputs<S, Gate>(tile, stage, physical(stage), row0, weight, input, input_scales,
                              cache_policy, stream, retained, active_columns);
    }
#pragma unroll 1
    for (int kt = 0; kt < ((S::k + S::k_step - 1) / S::k_step); ++kt) {
        const int stage = kt % S::stages;
        cp_wait<S::stages - 1>();
        __syncthreads();
        if constexpr (S::native) {
            if (S::known_csf || weight.csf.bases) {
                const auto rows = scale_origins<S, Gate>(row0);
                for (int block = warp; block < (S::weight_rows / 32) * (S::k_step / 32);
                     block += S::threads / 32) {
                    const int row_part = block / (S::k_step / 32);
                    const int group2   = block % (S::k_step / 32) * 2;
                    const int group4   = group2 & ~3;
                    common::csf32_decode_block<S::k_step, Gate ? 2 * S::intermediate : kHidden,
                                               S::k / 16, true>(
                        weight.csf, rows, physical(kt) * (S::k_step / 16) + group4, tile.bases,
                        tile.encoded[stage][group4 / 4], row_part, (group2 & 2) / 2,
                        reinterpret_cast<std::uint8_t*>(tile.weight_scales[stage]), row_part * 32,
                        group2);
                }
                __syncthreads();
            }
        }
#pragma unroll
        for (int ki = 0; ki < S::k_step / 64; ++ki) {
            unsigned a[2][4], as[2];
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const int row_base =
                    Gate ? warp_m * 16 + m * (S::weight_rows / 2) : warp_m * 32 + m * 16;
                const int row = row_base + (lane & 7) + ((lane >> 3 & 1) * 8);
                const int col = ki * 32 + (lane >> 4) * 16;
                ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                            smem_addr(tile.weight[stage] + tile_offset<S>(row, col)));
                const int sr = row_base + ((lane & 1) << 3) + (lane >> 2);
                const int scale_row =
                    S::known_csf || (S::native && weight.csf.bases) ? sr : (sr % 32) * 4 + sr / 32;
                as[m] = tile.weight_scales[stage][ki * 128 + scale_row];
            }
#pragma unroll
            for (int n = 0; n < S::mma_columns; ++n) {
                const int column = warp_n * S::warp_columns + n * 8;
                if (begin + column < end) {
                    const int row = column + (lane & 7);
                    const int col = ki * 32 + ((lane >> 3) & 1) * 16;
                    unsigned b[2];
                    ldmatrix_x2(b[0], b[1],
                                smem_addr(tile.input[stage] + tile_offset<S>(row, col)));
                    const auto bs =
                        tile.input_scales[stage][(column + (lane >> 2)) * (S::k_step / 64) + ki];
#pragma unroll
                    for (int m = 0; m < 2; ++m) {
                        mma_nvfp4_e4m3(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3],
                                       a[m][0], a[m][1], a[m][2], a[m][3], b[0], b[1], as[m], bs);
                    }
                }
            }
        }
        __syncthreads();
        const int next = kt + S::stages;
        if (next < ((S::k + S::k_step - 1) / S::k_step)) {
            stage_inputs<S, Gate>(tile, stage, physical(next), row0, weight, input, input_scales,
                                  cache_policy, stream, retained, active_columns);
        } else {
            // Empty groups drain the fixed-distance pipeline without reusing a live stage.
            cp_commit();
        }
    }
    cp_wait<0>();
}

template <int Intermediate, bool Native = false>
__device__ __forceinline__ void store_product(float lo, float hi, float calibration_value,
                                              std::uint8_t* codes, std::uint8_t* scales,
                                              int assignment, int feature) {
    const int lane = threadIdx.x & 31, r = lane >> 2;
    float maximum = fmaxf(fabsf(lo), fabsf(hi));
#pragma unroll
    for (int delta = 4; delta <= 16; delta *= 2) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
    }
    const auto scale = Native ? common::nvfp4_mul_activation_scale(maximum, calibration_value)
                              : nvfp4_activation_scale(maximum, calibration_value);
    const unsigned packed =
        Native ? common::quantize_nvfp4_mul_activation_pair(make_float2(lo, hi), calibration_value,
                                                            scale)
               : quantize_nvfp4_activation_pair(make_float2(lo, hi), calibration_value, scale);
    unsigned low = (packed & 15) << (r * 4), high = (packed >> 4) << (r * 4);
#pragma unroll
    for (int delta = 4; delta <= 16; delta *= 2) {
        low |= __shfl_xor_sync(0xffffffff, low, delta);
        high |= __shfl_xor_sync(0xffffffff, high, delta);
    }
    if (!r && assignment >= 0) {
        *reinterpret_cast<uint2*>(codes + std::int64_t(assignment) * (Intermediate / 2) +
                                  feature / 2) = make_uint2(low, high);
        scales[std::int64_t(assignment) * (Intermediate / 16) + feature / 16] = scale;
    }
}

template <int Columns, bool Gate, bool Atomic = false, int Stages = 2, int KStep = 128,
          int Eviction = 0, bool CachedExpertMajor = true, bool FilterLoads = false,
          int Intermediate = kIntermediate, bool Native = false, bool KnownCsf = false,
          int WeightRows = 128>
__global__ __launch_bounds__(
    Schedule<Columns, Gate, Stages, KStep, Eviction, Intermediate, Native, KnownCsf,
             WeightRows>::threads,
    Schedule<Columns, Gate, Stages, KStep, Eviction, Intermediate,
             Native>::min_blocks) void grouped_mma(const std::uint8_t* __restrict__ input,
                                                   const std::uint8_t* __restrict__ input_scales,
                                                   ExpertMatrixDeviceView bank,
                                                   ActivationScale activation_scale,
                                                   ActivationScale product_scale, Groups groups,
                                                   std::uint8_t* __restrict__ product,
                                                   std::uint8_t* __restrict__ product_scales,
                                                   float* __restrict__ contributions,
                                                   const float* __restrict__ coefficients,
                                                   float* __restrict__ routed_sum, int subtile_bits,
                                                   float limit, int feature_rows = 0,
                                                   int cache_policy         = 0,
                                                   int minimum_expert_count = 0,
                                                   int maximum_expert_count = 0x7fffffff,
                                                   ExpertLoadRegime regime  = ExpertLoadRegime::All,
                                                   int broad_short_limit    = 0,
                                                   int broad_long_limit = 0, int broad_split = 0,
                                                   int tokens = 0, bool dense_down = false,
                                                   int pool_expert_limit = 0,
                                                   int pool_min_experts  = 0,
                                                   bool grouped_input    = false) {
    using S = Schedule<Columns, Gate, Stages, KStep, Eviction, Intermediate, Native, KnownCsf,
                       WeightRows>;
    static_assert(Native || !CachedExpertMajor || (Stages == 2 && KStep == 128 && Eviction == 0));
    static_assert(S::stages <= (S::k + S::k_step - 1) / S::k_step);
    if constexpr (!Gate) {
        if (pool_expert_limit && groups.load_range[2] > pool_min_experts &&
            groups.load_range[2] <= pool_expert_limit)
            return;
        if (dense_down && groups.load_range && groups.load_range[0] == tokens &&
            groups.load_range[1] == tokens)
            return;
    }
    if constexpr (FilterLoads) {
        // A broad distribution mixes short experts with two to four column jobs.
        // Keep their weight reuse in one cached projection instead of serial
        // count classes. Counts are published by grouping, including Graph replay.
        const bool broad = groups.load_range && groups.load_range[0] <= broad_short_limit &&
                           groups.load_range[1] > broad_long_limit;
        if (regime == ExpertLoadRegime::OnlyBroad) {
            if (!broad) return;
        } else if (broad) {
            if (regime != ExpertLoadRegime::DenseWhenBroad) return;
            minimum_expert_count = broad_split;
        }
    }
    const int population_load = Native && grouped_input ? balanced_group_load(groups) : 0;
    if (groups.load_range &&
        (groups.load_range[1] <= minimum_expert_count ||
         (population_load ? population_load : groups.load_range[0]) > maximum_expert_count))
        return;
    if constexpr (CachedExpertMajor) cache_policy = 0;
    const auto stream   = (S::eviction & 4) ? common::l2_streaming_policy() : 0;
    const auto retained = (S::eviction & 8) ? common::l2_retained_policy() : 0;
    auto& tile =
        *reinterpret_cast<TileStorage<S>*>(common::shared_storage<sizeof(TileStorage<S>)>());
    constexpr int row_tiles = (Gate ? Intermediate : kHidden) / S::rows;
    static_assert((row_tiles & (row_tiles - 1)) == 0);
    constexpr int row_bits = [] {
        int bits = 0;
        for (int rows = row_tiles; rows > 1; rows >>= 1) ++bits;
        return bits;
    }();
    const int subtiles = 1 << subtile_bits;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warp_m = warp % (S::weight_rows / 32), warp_n = warp / (S::weight_rows / 32);
    const int window =
        Gate || CachedExpertMajor || feature_rows == 0 ? row_tiles : min(feature_rows, row_tiles);
    const int rounded_rows = (row_tiles + window - 1) / window * window;
    const int count        = *groups.jobs * rounded_rows * subtiles;
    for (int task = blockIdx.x; task < count; task += gridDim.x) {
        int job, row0, subtile;
        if constexpr (Gate || CachedExpertMajor) {
            job     = task >> (row_bits + subtile_bits);
            row0    = (task & (row_tiles - 1)) * S::rows;
            subtile = (task >> row_bits) & (subtiles - 1);
        } else {
            const int jobs = *groups.jobs;
            job            = (task / (window * subtiles)) % jobs;
            row0           = (task % window + task / (jobs * subtiles * window) * window) * S::rows;
            subtile        = (task / window) % subtiles;
        }
        if (row0 >= row_tiles * S::rows) continue;
        const int expert = groups.job_experts[job];
        const int offset = groups.offsets[expert], end = groups.offsets[expert + 1];
        const int expert_count   = end - offset;
        const int dispatch_count = population_load ? population_load : expert_count;
        if (dispatch_count <= minimum_expert_count || dispatch_count > maximum_expert_count)
            continue;
        if constexpr (!Gate) {
            if (dense_down && expert_count == tokens) continue;
        }
        const int begin = offset + groups.job_columns[job] + subtile * Columns;
        if (begin >= end) { continue; }
        const auto weight = resolve<Native, KnownCsf>(bank, expert);
        if (!weight.valid) continue;
        if (threadIdx.x < Columns) {
            const int packed = begin + threadIdx.x;
            tile.sources[threadIdx.x] =
                packed < end
                    ? (Gate ? (Native ? (grouped_input ? packed : groups.assignments[packed])
                                      : groups.assignments[packed] / kTopK)
                            : packed)
                    : -1;
        }
        __syncthreads();
        float acc[2][S::mma_columns][4] = {};
        contract_tile<S, Gate, CachedExpertMajor && !Native>(tile, row0, weight, input,
                                                             input_scales, begin, end, cache_policy,
                                                             stream, retained, acc);
        const float factor             = Native ? activation_scale.get(expert) * weight.scale
                                                : 1.0F / (activation_scale.scalar * weight.scale);
        const float product_multiplier = Native ? product_scale.get(expert) : product_scale.scalar;
#pragma unroll
        for (int n = 0; n < S::mma_columns; ++n) {
            if (begin + warp_n * S::warp_columns + n * 8 >= end) { continue; }
            const int column = begin + warp_n * S::warp_columns + n * 8 + 2 * (lane & 3);
            if constexpr (Gate) {
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const float g0 = acc[0][n][t] * factor, g1 = acc[0][n][t + 2] * factor;
                    const float h0 =
                        silu(fminf(g0, limit)) * fmaxf(-limit, fminf(acc[1][n][t] * factor, limit));
                    const float h1 = silu(fminf(g1, limit)) *
                                     fmaxf(-limit, fminf(acc[1][n][t + 2] * factor, limit));
                    store_product<Intermediate, Native>(
                        h0, h1, product_multiplier, product, product_scales,
                        column + t < end ? column + t : -1, row0 + warp_m * 16);
                }
            } else {
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const int c = column + t;
                    if (c >= end) continue;
                    // Every row for one column consumes the same immutable route.
                    // Resolve it before atomic writes; otherwise conservative aliasing
                    // reloads the assignment/coefficient after each output element.
                    float* target;
                    float coefficient = 1;
                    if constexpr (Atomic) {
                        const int assignment = groups.assignments[c];
                        target      = routed_sum + std::int64_t(assignment / kTopK) * kHidden;
                        coefficient = coefficients[assignment];
                    } else {
                        target = contributions + std::int64_t(c) * kHidden;
                    }
#pragma unroll
                    for (int m = 0; m < 2; ++m) {
#pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            const int v   = t + 2 * half;
                            const int row = row0 + warp_m * 32 + m * 16 + (lane >> 2) + half * 8;
                            if constexpr (Atomic)
                                atomicAdd(target + row, acc[m][n][v] * factor * coefficient);
                            else
                                target[row] = acc[m][n][v] * factor;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Experts present in every token can be reduced in registers across their down
// contractions. The inverse map retains the original token/route identity even
// when mixed hot/tail grouping publishes assignments in a different order.
template <int Columns, int Intermediate = 1024, bool Native = false, bool KnownCsf = false>
__global__ __launch_bounds__(Schedule<Columns, false>::threads) void dense_token_down(
    const std::uint8_t* __restrict__ input, const std::uint8_t* __restrict__ input_scales,
    ExpertMatrixDeviceView bank, ActivationScale activation_scale, Groups groups,
    const int* __restrict__ ids, const float* __restrict__ coefficients,
    float* __restrict__ destination, int tokens, int pool_expert_limit = 0,
    int pool_min_experts = 0) {
    using S = Schedule<Columns, false, 2, 128, 0, Intermediate, Native, KnownCsf>;
    if (groups.load_range[1] != tokens ||
        (pool_expert_limit && groups.load_range[2] > pool_min_experts &&
         groups.load_range[2] <= pool_expert_limit))
        return;
    __shared__ TileStorage<S> tile;
    __shared__ int full_experts[kTopK], full_count;
    __shared__ float probabilities[Columns];
    if (threadIdx.x == 0) full_count = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < kExperts; e += blockDim.x)
        if (groups.offsets[e + 1] - groups.offsets[e] == tokens) {
            const int ordinal     = atomicAdd(&full_count, 1);
            full_experts[ordinal] = e;
        }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warp_m = warp % (S::weight_rows / 32), warp_n = warp / (S::weight_rows / 32);
    constexpr int row_tiles = kHidden / S::rows;
    const int tasks         = row_tiles * ((tokens + Columns - 1) / Columns);
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int row0                    = task % row_tiles * S::rows;
        const int begin                   = task / row_tiles * Columns;
        float total[2][S::mma_columns][4] = {};
        for (int ordinal = 0; ordinal < full_count; ++ordinal) {
            const int expert = full_experts[ordinal];
            if (threadIdx.x < Columns) {
                const int token = begin + threadIdx.x;
                int assignment  = -1;
                if (token < tokens)
                    for (int j = 0; j < kTopK; ++j)
                        if (ids[token * kTopK + j] == expert) assignment = token * kTopK + j;
                tile.sources[threadIdx.x]  = assignment >= 0 ? groups.inverse[assignment] : -1;
                probabilities[threadIdx.x] = assignment >= 0 ? coefficients[assignment] : 0;
            }
            __syncthreads();
            const auto weight                   = resolve<Native, KnownCsf>(bank, expert);
            float partial[2][S::mma_columns][4] = {};
            contract_tile<S, false, !Native>(tile, row0, weight, input, input_scales, begin, tokens,
                                             0, 0, 0, partial);
            const float factor = Native ? activation_scale.get(expert) * weight.scale
                                        : 1.F / (activation_scale.scalar * weight.scale);
#pragma unroll
            for (int n = 0; n < S::mma_columns; ++n) {
                const int column = warp_n * S::warp_columns + n * 8 + 2 * (lane & 3);
#pragma unroll
                for (int m = 0; m < 2; ++m)
#pragma unroll
                    for (int v = 0; v < 4; ++v)
                        total[m][n][v] +=
                            partial[m][n][v] * factor * probabilities[column + (v & 1)];
            }
            // All readers of sources, probability words and staged operands finish
            // before the next expert reuses this CTA's shared storage.
            __syncthreads();
        }
#pragma unroll
        for (int n = 0; n < S::mma_columns; ++n) {
            const int column = begin + warp_n * S::warp_columns + n * 8 + 2 * (lane & 3);
#pragma unroll
            for (int m = 0; m < 2; ++m)
#pragma unroll
                for (int v = 0; v < 4; ++v)
                    if (column + (v & 1) < tokens) {
                        const int row = row0 + warp_m * 32 + m * 16 + (lane >> 2) + (v / 2) * 8;
                        destination[std::int64_t(column + (v & 1)) * kHidden + row] =
                            total[m][n][v];
                    }
        }
        __syncthreads();
    }
}

template <int Columns, int Intermediate = 1024, bool Native = false, bool KnownCsf = false>
struct PoolContraction {
    using S    = Schedule<Columns, false, 2, 128, 0, Intermediate, Native, KnownCsf>;
    using Tile = TileStorage<S>;
    static constexpr int threads = S::threads;
    const std::uint8_t* input;
    const std::uint8_t* scales;
    ExpertMatrixDeviceView bank;
    ActivationScale activation_scale;

    __device__ __forceinline__ float operator()(Tile& tile, int row, int expert, int begin, int end,
                                                float (&acc)[2][S::mma_columns][4]) const {
        const auto weight = resolve<Native, KnownCsf>(bank, expert);
        contract_tile<S, false, !Native>(tile, row, weight, input, scales, begin, end, 0, 0, 0,
                                         acc);
        return Native ? activation_scale.get(expert) * weight.scale
                      : 1.F / (activation_scale.scalar * weight.scale);
    }
};

__device__ __forceinline__ bool includes(ExpertSubsetKind kind, const int* mask, int expert) {
    return kind == ExpertSubsetKind::All || mask[expert] == (kind == ExpertSubsetKind::MaskOnes);
}

__device__ __forceinline__ void expand_group(Matrix weight, int row, int group, int k,
                                             float* values) {
    const auto decoded =
        nvfp4_dequant_f16x16(weight.codes + std::int64_t(row) * (k / 2) + group * 8,
                             weight.scales[nvfp4_scale_byte_offset(row, group, k)]);
    const int words[8] = {decoded.lo.x, decoded.lo.y, decoded.lo.z, decoded.lo.w,
                          decoded.hi.x, decoded.hi.y, decoded.hi.z, decoded.hi.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const auto pair   = __half22float2(*reinterpret_cast<const __half2*>(&words[i]));
        values[2 * i]     = pair.x;
        values[2 * i + 1] = pair.y;
    }
}

template <int Intermediate>
__global__ void vector_gate(const __nv_bfloat16* x, ExpertMatrixDeviceView bank, const int* ids,
                            ExpertSubsetKind kind, const int* mask, float* activation,
                            float limit) {
    const int assignment = blockIdx.y, expert = ids[assignment];
    if (!includes(kind, mask, expert)) { return; }
    const auto weight = resolve(bank, expert);
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    // Keep the shared input resident while streamed weight codes occupy L1.
    __shared__ __align__(16) __nv_bfloat16 input[kHidden];
    const auto* source = x + std::int64_t(assignment / kTopK) * kHidden;
    for (int i = threadIdx.x; i < kHidden / 8; i += blockDim.x)
        reinterpret_cast<uint4*>(input)[i] = load_vec<uint4>(source + i * 8);
    __syncthreads();
    float gate = 0, up = 0;
#pragma unroll
    for (int group = lane; group < kHidden / 16; group += 32) {
        float g[16], u[16];
        expand_group(weight, row, group, kHidden, g);
        expand_group(weight, row + Intermediate, group, kHidden, u);
        const auto x0           = load_vec<uint4>(input + group * 16);
        const auto x1           = load_vec<uint4>(input + group * 16 + 8);
        const unsigned words[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const float2 v = bf16x2_bits_to_float2(words[pair]);
            gate           = fmaf(g[pair * 2], v.x, gate);
            gate           = fmaf(g[pair * 2 + 1], v.y, gate);
            up             = fmaf(u[pair * 2], v.x, up);
            up             = fmaf(u[pair * 2 + 1], v.y, up);
        }
    }
    gate = warp_reduce_sum(gate) / weight.scale;
    up   = warp_reduce_sum(up) / weight.scale;
    if (!lane) {
        activation[std::int64_t(assignment) * Intermediate + row] =
            silu(fminf(gate, limit)) * fmaxf(-limit, fminf(up, limit));
    }
}

template <int Intermediate, int Rows, int Warps, class Output>
__global__ void vector_down(const float* activation, ExpertMatrixDeviceView bank, const int* ids,
                            const float* coefficients, ExpertSubsetKind kind, const int* mask,
                            Output output) {
    __shared__ float values[Warps][Rows];
    const int token = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float total[Rows] = {};
    // Share route slots across a small warp set before the CTA reduction.
    for (int route = warp; route < kTopK; route += Warps) {
        const int assignment = token * kTopK + route, expert = ids[assignment];
        float sum[Rows] = {};
        if (includes(kind, mask, expert)) {
            const auto weight = resolve(bank, expert);
            for (int group = lane; group < Intermediate / 16; group += 32) {
                float x[16];
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    x[i] = activation[std::int64_t(assignment) * Intermediate + group * 16 + i];
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    float w[16];
                    expand_group(weight, blockIdx.x * Rows + r, group, Intermediate, w);
#pragma unroll
                    for (int i = 0; i < 16; ++i) sum[r] = fmaf(x[i], w[i], sum[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r)
                total[r] += warp_reduce_sum(sum[r]) * (coefficients[assignment] / weight.scale);
        }
    }
    if (!lane) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) values[warp][r] = total[r];
    }
    __syncthreads();
    if (threadIdx.x < Rows) {
        float result = 0;
#pragma unroll
        for (int j = 0; j < Warps; ++j) result += values[j][threadIdx.x];
        const int i = token * kHidden + blockIdx.x * Rows + threadIdx.x;
        output.store(i, output.initial(i) + result);
    }
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4
