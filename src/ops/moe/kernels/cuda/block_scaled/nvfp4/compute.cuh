#pragma once

#include "ops/moe/kernels/cuda/block_scaled/nvfp4/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/output.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/quant/nvfp4.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled::nvfp4 {
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

template <int Columns, bool Gate, int Stages = 2, int KStep = 128, int Eviction = 0>
struct Schedule {
    static constexpr int columns      = Columns;
    static constexpr int stages       = Stages;
    static constexpr int k_step       = KStep;
    static constexpr int eviction     = Eviction;
    static constexpr int threads      = Columns >= 64 ? 256 : 128;
    static constexpr int warp_columns = Columns >= 64 ? Columns / 2 : Columns;
    static constexpr int mma_columns  = warp_columns / 8;
    static constexpr int rows         = Gate ? 64 : 128;
    static constexpr int k            = Gate ? kHidden : kIntermediate;
    static constexpr int min_blocks   = Columns == 128 ? 2 : Columns == 64 ? 3 : 1;
};

template <class S>
struct TileStorage {
    alignas(16) std::uint8_t weight[S::stages][128 * (S::k_step / 2)];
    alignas(16) std::uint8_t input[S::stages][S::columns * (S::k_step / 2)];
    alignas(16) unsigned weight_scales[S::stages][128 * (S::k_step / 64)];
    alignas(16) unsigned input_scales[S::stages][S::columns * (S::k_step / 64)];
    int sources[S::columns];
};

__device__ __forceinline__ int swizzle(int row, int byte) { return byte ^ ((row & 3) << 4); }

struct Matrix {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    float divisor;
};

__device__ __forceinline__ Matrix resolve(const ExpertMatrixDeviceView& matrix, int expert) {
    const auto offset = matrix.offsets ? matrix.offsets[expert] : 0;
    return {reinterpret_cast<const std::uint8_t*>(matrix.codes) +
                (matrix.offsets ? offset : expert * matrix.code_stride),
            reinterpret_cast<const std::uint8_t*>(matrix.scales) +
                (matrix.offsets ? offset : expert * matrix.scale_stride),
            matrix.divisors[expert * matrix.divisor_stride]};
}

template <class S, bool Gate>
__device__ __forceinline__ void stage_inputs(TileStorage<S>& tile, int stage, int kt, int row0,
                                             Matrix weight, const std::uint8_t* x,
                                             const std::uint8_t* scales, int cache_policy,
                                             std::uint64_t stream, std::uint64_t retained) {
    for (int task = threadIdx.x; task < 128 * (S::k_step / 32); task += S::threads) {
        const int row = task / (S::k_step / 32), segment = task % (S::k_step / 32);
        const int wr     = row0 + (Gate ? (row < 64 ? row : row - 64 + kIntermediate) : row);
        const bool valid = S::k % S::k_step == 0 || kt * S::k_step + segment * 32 < S::k;
        if constexpr (S::eviction & 4) {
            if (cache_policy & 1)
                common::cp_async_cache_hint<Cache::cg>(
                    tile.weight[stage] + row * (S::k_step / 2) + swizzle(row, segment * 16),
                    weight.codes + std::int64_t(wr) * (S::k / 2) +
                        (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                    valid ? 16 : 0, stream);
            else
                common::cp_async_cache_hint<Cache::ca>(
                    tile.weight[stage] + row * (S::k_step / 2) + swizzle(row, segment * 16),
                    weight.codes + std::int64_t(wr) * (S::k / 2) +
                        (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                    valid ? 16 : 0, stream);
        } else if (cache_policy & 1)
            cp_async_zfill<16, Cache::cg>(tile.weight[stage] + row * (S::k_step / 2) +
                                              swizzle(row, segment * 16),
                                          weight.codes + std::int64_t(wr) * (S::k / 2) +
                                              (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                                          valid ? 16 : 0);
        else
            cp_async_zfill<16>(tile.weight[stage] + row * (S::k_step / 2) +
                                   swizzle(row, segment * 16),
                               weight.codes + std::int64_t(wr) * (S::k / 2) +
                                   (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                               valid ? 16 : 0);
    }
    constexpr int parts = Gate ? 2 : 1;
    constexpr int bytes = Gate ? 8 : 16;
    for (int task = threadIdx.x; task < (S::k_step / 64) * parts * 32; task += S::threads) {
        const int row = task % 32, part = (task / 32) % parts, k64 = task / (32 * parts);
        const int wr      = row0 + row + (Gate ? part * kIntermediate : 0);
        auto* destination = reinterpret_cast<std::uint8_t*>(tile.weight_scales[stage]) + k64 * 512 +
                            row * 16 + part * bytes;
        const bool valid = S::k % S::k_step == 0 || kt * S::k_step + k64 * 64 < S::k;
        cp_async_zfill<bytes>(
            destination,
            weight.scales +
                nvfp4_scale_byte_offset(wr, valid ? kt * (S::k_step / 16) + k64 * 4 : 0, S::k),
            valid ? bytes : 0);
    }
    for (int task = threadIdx.x; task < S::columns * (S::k_step / 32); task += S::threads) {
        const int row = task / (S::k_step / 32), segment = task % (S::k_step / 32);
        const int source = tile.sources[row];
        const bool valid =
            source >= 0 && (S::k % S::k_step == 0 || kt * S::k_step + segment * 32 < S::k);
        if constexpr (S::eviction & 8) {
            if (cache_policy & 2)
                common::cp_async_cache_hint<Cache::cg>(
                    tile.input[stage] + row * (S::k_step / 2) + swizzle(row, segment * 16),
                    x + std::int64_t(source >= 0 ? source : 0) * (S::k / 2) +
                        (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                    valid ? 16 : 0, retained);
            else
                common::cp_async_cache_hint<Cache::ca>(
                    tile.input[stage] + row * (S::k_step / 2) + swizzle(row, segment * 16),
                    x + std::int64_t(source >= 0 ? source : 0) * (S::k / 2) +
                        (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                    valid ? 16 : 0, retained);
        } else if (cache_policy & 2)
            cp_async_zfill<16, Cache::cg>(tile.input[stage] + row * (S::k_step / 2) +
                                              swizzle(row, segment * 16),
                                          x + std::int64_t(source >= 0 ? source : 0) * (S::k / 2) +
                                              (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                                          valid ? 16 : 0);
        else
            cp_async_zfill<16>(tile.input[stage] + row * (S::k_step / 2) +
                                   swizzle(row, segment * 16),
                               x + std::int64_t(source >= 0 ? source : 0) * (S::k / 2) +
                                   (valid ? kt * (S::k_step / 2) + segment * 16 : 0),
                               valid ? 16 : 0);
    }
    for (int task = threadIdx.x; task < S::columns * (S::k_step / 128); task += S::threads) {
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

__device__ __forceinline__ void store_product(float lo, float hi, float divisor,
                                              std::uint8_t* codes, std::uint8_t* scales,
                                              int assignment, int feature) {
    const int lane = threadIdx.x & 31, r = lane >> 2;
    float maximum = fmaxf(fabsf(lo), fabsf(hi));
#pragma unroll
    for (int delta = 4; delta <= 16; delta *= 2) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
    }
    const auto scale      = nvfp4_activation_scale(maximum, divisor);
    const unsigned packed = quantize_nvfp4_activation_pair(make_float2(lo, hi), divisor, scale);
    unsigned low = (packed & 15) << (r * 4), high = (packed >> 4) << (r * 4);
#pragma unroll
    for (int delta = 4; delta <= 16; delta *= 2) {
        low |= __shfl_xor_sync(0xffffffff, low, delta);
        high |= __shfl_xor_sync(0xffffffff, high, delta);
    }
    if (!r && assignment >= 0) {
        *reinterpret_cast<uint2*>(codes + std::int64_t(assignment) * (kIntermediate / 2) +
                                  feature / 2) = make_uint2(low, high);
        scales[std::int64_t(assignment) * (kIntermediate / 16) + feature / 16] = scale;
    }
}

template <int Columns, bool Gate, bool Atomic = false, int Stages = 2, int KStep = 128,
          int Eviction = 0, bool CachedExpertMajor = true>
__global__ __launch_bounds__(
    Schedule<Columns, Gate, Stages, KStep, Eviction>::threads,
    Schedule<Columns, Gate, Stages, KStep,
             Eviction>::min_blocks) void grouped_mma(const std::uint8_t* input,
                                                     const std::uint8_t* input_scales,
                                                     ExpertMatrixDeviceView bank,
                                                     float activation_divisor,
                                                     float product_divisor, Groups groups,
                                                     std::uint8_t* product,
                                                     std::uint8_t* product_scales,
                                                     float* contributions,
                                                     const float* coefficients, float* routed_sum,
                                                     int subtiles, int feature_rows = 0,
                                                     int cache_policy         = 0,
                                                     int minimum_expert_count = 0,
                                                     int maximum_expert_count = 0x7fffffff) {
    using S = Schedule<Columns, Gate, Stages, KStep, Eviction>;
    static_assert(!CachedExpertMajor || (Stages == 2 && KStep == 128 && Eviction == 0));
    static_assert(S::stages <= (S::k + S::k_step - 1) / S::k_step);
    if (groups.load_range && (groups.load_range[1] <= minimum_expert_count ||
                              groups.load_range[0] > maximum_expert_count))
        return;
    if constexpr (CachedExpertMajor) cache_policy = 0;
    const auto stream   = (S::eviction & 4) ? common::l2_streaming_policy() : 0;
    const auto retained = (S::eviction & 8) ? common::l2_retained_policy() : 0;
    auto& tile =
        *reinterpret_cast<TileStorage<S>*>(common::shared_storage<sizeof(TileStorage<S>)>());
    constexpr int row_tiles = (Gate ? kIntermediate : kHidden) / S::rows;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warp_m = warp % 4, warp_n = warp / 4;
    const int window =
        Gate || CachedExpertMajor || feature_rows == 0 ? row_tiles : min(feature_rows, row_tiles);
    const int rounded_rows = (row_tiles + window - 1) / window * window;
    const int count        = *groups.jobs * rounded_rows * subtiles;
    for (int task = blockIdx.x; task < count; task += gridDim.x) {
        int job, row0, subtile;
        if constexpr (Gate || CachedExpertMajor) {
            job     = task / (row_tiles * subtiles);
            row0    = (task % row_tiles) * S::rows;
            subtile = (task / row_tiles) % subtiles;
        } else {
            const int jobs = *groups.jobs;
            job            = (task / (window * subtiles)) % jobs;
            row0           = (task % window + task / (jobs * subtiles * window) * window) * S::rows;
            subtile        = (task / window) % subtiles;
        }
        if (row0 >= row_tiles * S::rows) continue;
        const int expert = groups.job_experts[job];
        const int offset = groups.offsets[expert], end = groups.offsets[expert + 1];
        const int expert_count = end - offset;
        if (expert_count <= minimum_expert_count || expert_count > maximum_expert_count) continue;
        const int begin = offset + groups.job_columns[job] + subtile * Columns;
        if (begin >= end) { continue; }
        const auto weight = resolve(bank, expert);
        if (threadIdx.x < Columns) {
            const int packed = begin + threadIdx.x;
            tile.sources[threadIdx.x] =
                packed < end ? (Gate ? groups.assignments[packed] / kTopK : packed) : -1;
        }
        __syncthreads();
#pragma unroll
        for (int stage = 0; stage < S::stages; ++stage) {
            stage_inputs<S, Gate>(tile, stage, stage, row0, weight, input, input_scales,
                                  cache_policy, stream, retained);
        }
        float acc[2][S::mma_columns][4] = {};
#pragma unroll 1
        for (int kt = 0; kt < ((S::k + S::k_step - 1) / S::k_step); ++kt) {
            const int stage = kt % S::stages;
            cp_wait<S::stages - 1>();
            __syncthreads();
#pragma unroll
            for (int ki = 0; ki < S::k_step / 64; ++ki) {
                unsigned a[2][4], as[2];
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    const int row_base = Gate ? warp_m * 16 + m * 64 : warp_m * 32 + m * 16;
                    const int row      = row_base + (lane & 7) + ((lane >> 3 & 1) * 8);
                    const int col      = ki * 32 + (lane >> 4) * 16;
                    ldmatrix_x4(
                        a[m][0], a[m][1], a[m][2], a[m][3],
                        smem_addr(tile.weight[stage] + row * (S::k_step / 2) + swizzle(row, col)));
                    const int sr = row_base + ((lane & 1) << 3) + (lane >> 2);
                    as[m]        = tile.weight_scales[stage][ki * 128 + (sr % 32) * 4 + sr / 32];
                }
#pragma unroll
                for (int n = 0; n < S::mma_columns; ++n) {
                    const int column = warp_n * S::warp_columns + n * 8;
                    if (begin + column < end) {
                        const int row = column + (lane & 7);
                        const int col = ki * 32 + ((lane >> 3) & 1) * 16;
                        unsigned b[2];
                        ldmatrix_x2(b[0], b[1],
                                    smem_addr(tile.input[stage] + row * (S::k_step / 2) +
                                              swizzle(row, col)));
                        const auto bs =
                            tile.input_scales[stage]
                                             [(column + (lane >> 2)) * (S::k_step / 64) + ki];
#pragma unroll
                        for (int m = 0; m < 2; ++m) {
                            mma_nvfp4_e4m3(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3],
                                           a[m][0], a[m][1], a[m][2], a[m][3], b[0], b[1], as[m],
                                           bs);
                        }
                    }
                }
            }
            __syncthreads();
            const int next = kt + S::stages;
            if (next < ((S::k + S::k_step - 1) / S::k_step)) {
                stage_inputs<S, Gate>(tile, stage, next, row0, weight, input, input_scales,
                                      cache_policy, stream, retained);
            } else {
                // Empty groups drain the fixed-distance pipeline without reusing a live stage.
                cp_commit();
            }
        }
        cp_wait<0>();
        const float factor = 1.0F / (activation_divisor * weight.divisor);
#pragma unroll
        for (int n = 0; n < S::mma_columns; ++n) {
            if (begin + warp_n * S::warp_columns + n * 8 >= end) { continue; }
            const int column = begin + warp_n * S::warp_columns + n * 8 + 2 * (lane & 3);
            if constexpr (Gate) {
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const float g0 = acc[0][n][t] * factor, g1 = acc[0][n][t + 2] * factor;
                    const float h0 = silu(g0) * (acc[1][n][t] * factor);
                    const float h1 = silu(g1) * (acc[1][n][t + 2] * factor);
                    store_product(h0, h1, product_divisor, product, product_scales,
                                  column + t < end ? column + t : -1, row0 + warp_m * 16);
                }
            } else {
#pragma unroll
                for (int m = 0; m < 2; ++m) {
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const int c = column + (v & 1);
                        if (c < end) {
                            const int row = row0 + warp_m * 32 + m * 16 + (lane >> 2) + (v / 2) * 8;
                            if constexpr (Atomic) {
                                const int assignment = groups.assignments[c];
                                atomicAdd(routed_sum + std::int64_t(assignment / kTopK) * kHidden +
                                              row,
                                          acc[m][n][v] * factor * coefficients[assignment]);
                            } else {
                                contributions[std::int64_t(c) * kHidden + row] =
                                    acc[m][n][v] * factor;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

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

__global__ void vector_gate(const __nv_bfloat16* x, ExpertMatrixDeviceView bank, const int* ids,
                            ExpertSubsetKind kind, const int* mask, float* activation) {
    const int assignment = blockIdx.y, expert = ids[assignment];
    if (!includes(kind, mask, expert)) { return; }
    const auto weight = resolve(bank, expert);
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const auto* input = x + std::int64_t(assignment / kTopK) * kHidden;
    float gate = 0, up = 0;
#pragma unroll
    for (int group = lane; group < kHidden / 16; group += 32) {
        float g[16], u[16];
        expand_group(weight, row, group, kHidden, g);
        expand_group(weight, row + kIntermediate, group, kHidden, u);
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
    gate = warp_reduce_sum(gate) / weight.divisor;
    up   = warp_reduce_sum(up) / weight.divisor;
    if (!lane) { activation[std::int64_t(assignment) * kIntermediate + row] = silu(gate) * up; }
}

template <int Rows, int Warps = kTopK, class Output>
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
            for (int group = lane; group < kIntermediate / 16; group += 32) {
                float x[16];
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    x[i] = activation[std::int64_t(assignment) * kIntermediate + group * 16 + i];
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    float w[16];
                    expand_group(weight, blockIdx.x * Rows + r, group, kIntermediate, w);
#pragma unroll
                    for (int i = 0; i < 16; ++i) sum[r] = fmaf(x[i], w[i], sum[r]);
                }
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r)
                total[r] += warp_reduce_sum(sum[r]) * (coefficients[assignment] / weight.divisor);
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

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::nvfp4
