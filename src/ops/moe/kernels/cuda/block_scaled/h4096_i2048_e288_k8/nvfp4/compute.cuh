#pragma once

#include "ops/moe/kernels/cuda/block_scaled/h4096_i2048_e288_k8/nvfp4/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/h4096_i2048_e288_k8/output.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/quant/nvfp4.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8::nvfp4 {
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

template <int Columns, bool Gate>
struct Schedule {
    static constexpr int columns      = Columns;
    static constexpr int stages       = 2;
    static constexpr int threads      = Columns >= 64 ? 256 : 128;
    static constexpr int warp_columns = Columns >= 64 ? Columns / 2 : Columns;
    static constexpr int mma_columns  = warp_columns / 8;
    static constexpr int rows         = Gate ? 64 : 128;
    static constexpr int k            = Gate ? kHidden : kIntermediate;
    static constexpr int min_blocks   = Columns == 128 ? 2 : Columns == 64 ? 3 : 1;
};

template <class S>
struct TileStorage {
    alignas(16) std::uint8_t weight[S::stages][128 * 64];
    alignas(16) std::uint8_t input[S::stages][S::columns * 64];
    alignas(16) unsigned weight_scales[S::stages][128 * 2];
    alignas(16) unsigned input_scales[S::stages][S::columns * 2];
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
                                             const std::uint8_t* scales) {
    for (int task = threadIdx.x; task < 128 * 4; task += S::threads) {
        const int row = task / 4, segment = task % 4;
        const int wr = row0 + (Gate ? (row < 64 ? row : row - 64 + kIntermediate) : row);
        cp_async<16>(tile.weight[stage] + row * 64 + swizzle(row, segment * 16),
                     weight.codes + std::int64_t(wr) * (S::k / 2) + kt * 64 + segment * 16);
    }
    constexpr int parts = Gate ? 2 : 1;
    constexpr int bytes = Gate ? 8 : 16;
    for (int task = threadIdx.x; task < 2 * parts * 32; task += S::threads) {
        const int row = task % 32, part = (task / 32) % parts, k64 = task / (32 * parts);
        const int wr      = row0 + row + (Gate ? part * kIntermediate : 0);
        auto* destination = reinterpret_cast<std::uint8_t*>(tile.weight_scales[stage]) + k64 * 512 +
                            row * 16 + part * bytes;
        cp_async<bytes>(destination,
                        weight.scales + nvfp4_scale_byte_offset(wr, kt * 8 + k64 * 4, S::k));
    }
    for (int task = threadIdx.x; task < S::columns * 4; task += S::threads) {
        const int row = task / 4, segment = task % 4;
        const int source = tile.sources[row];
        cp_async_zfill<16>(tile.input[stage] + row * 64 + swizzle(row, segment * 16),
                           x + std::int64_t(source >= 0 ? source : 0) * (S::k / 2) + kt * 64 +
                               segment * 16,
                           source >= 0 ? 16 : 0);
    }
    for (int row = threadIdx.x; row < S::columns; row += S::threads) {
        const int source = tile.sources[row];
        cp_async_zfill<8>(&tile.input_scales[stage][row * 2],
                          scales + std::int64_t(source >= 0 ? source : 0) * (S::k / 16) + kt * 8,
                          source >= 0 ? 8 : 0);
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

template <int Columns, bool Gate, bool Atomic = false>
__global__ __launch_bounds__(
    Schedule<Columns, Gate>::threads,
    Schedule<Columns, Gate>::min_blocks) void grouped_mma(const std::uint8_t* input,
                                                          const std::uint8_t* input_scales,
                                                          ExpertMatrixDeviceView bank,
                                                          float activation_divisor,
                                                          float product_divisor, Groups groups,
                                                          std::uint8_t* product,
                                                          std::uint8_t* product_scales,
                                                          float* contributions,
                                                          const float* coefficients,
                                                          float* routed_sum, int subtiles,
                                                          float limit) {
    using S = Schedule<Columns, Gate>;
    __shared__ TileStorage<S> tile;
    constexpr int row_tiles = (Gate ? kIntermediate : kHidden) / S::rows;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int warp_m = warp % 4, warp_n = warp / 4;
    const int count = *groups.jobs * row_tiles * subtiles;
    for (int task = blockIdx.x; task < count; task += gridDim.x) {
        const int job = task / (row_tiles * subtiles), row0 = (task % row_tiles) * S::rows;
        const int subtile = (task / row_tiles) % subtiles;
        const int expert  = groups.job_experts[job];
        const int begin   = groups.offsets[expert] + groups.job_columns[job] + subtile * Columns;
        const int end     = groups.offsets[expert + 1];
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
            stage_inputs<S, Gate>(tile, stage, stage, row0, weight, input, input_scales);
        }
        float acc[2][S::mma_columns][4] = {};
#pragma unroll 1
        for (int kt = 0; kt < S::k / 128; ++kt) {
            const int stage = kt % S::stages;
            cp_wait<S::stages - 1>();
            __syncthreads();
#pragma unroll
            for (int ki = 0; ki < 2; ++ki) {
                unsigned a[2][4], as[2];
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    const int row_base = Gate ? warp_m * 16 + m * 64 : warp_m * 32 + m * 16;
                    const int row      = row_base + (lane & 7) + ((lane >> 3 & 1) * 8);
                    const int col      = ki * 32 + (lane >> 4) * 16;
                    ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                                smem_addr(tile.weight[stage] + row * 64 + swizzle(row, col)));
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
                                    smem_addr(tile.input[stage] + row * 64 + swizzle(row, col)));
                        const auto bs = tile.input_scales[stage][(column + (lane >> 2)) * 2 + ki];
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
            if (next < S::k / 128) {
                stage_inputs<S, Gate>(tile, stage, next, row0, weight, input, input_scales);
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
                    const float h0 =
                        silu(fminf(g0, limit)) * fmaxf(-limit, fminf(acc[1][n][t] * factor, limit));
                    const float h1 = silu(fminf(g1, limit)) *
                                     fmaxf(-limit, fminf(acc[1][n][t + 2] * factor, limit));
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
                            ExpertSubsetKind kind, const int* mask, float* activation,
                            float limit) {
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
    if (!lane) {
        activation[std::int64_t(assignment) * kIntermediate + row] =
            silu(fminf(gate, limit)) * fmaxf(-limit, fminf(up, limit));
    }
}

template <int Rows, class Output>
__global__ void vector_down(const float* activation, ExpertMatrixDeviceView bank, const int* ids,
                            const float* coefficients, ExpertSubsetKind kind, const int* mask,
                            Output output) {
    __shared__ float values[kTopK][Rows];
    const int token = blockIdx.y, route = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int assignment = token * kTopK + route, expert = ids[assignment];
    float sum[Rows] = {};
    if (includes(kind, mask, expert)) {
        const auto weight = resolve(bank, expert);
        for (int group = lane; group < kIntermediate / 16; group += 32) {
            float x[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                x[k] = activation[std::int64_t(assignment) * kIntermediate + group * 16 + k];
            }
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                float w[16];
                expand_group(weight, blockIdx.x * Rows + r, group, kIntermediate, w);
#pragma unroll
                for (int k = 0; k < 16; ++k) { sum[r] = fmaf(x[k], w[k], sum[r]); }
            }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            sum[r] = warp_reduce_sum(sum[r]) * (coefficients[assignment] / weight.divisor);
        }
    }
    if (!lane) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) { values[route][r] = sum[r]; }
    }
    __syncthreads();
    if (threadIdx.x < Rows) {
        float result = 0;
#pragma unroll
        for (int j = 0; j < kTopK; ++j) { result += values[j][threadIdx.x]; }
        const int i = token * kHidden + blockIdx.x * Rows + threadIdx.x;
        output.store(i, output.initial(i) + result);
    }
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8::nvfp4
