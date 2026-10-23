#pragma once

#include "ops/moe/kernels/cuda/block_scaled/h4096_i2048_e288_k8/fp8/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/h4096_i2048_e288_k8/output.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/quant/fp8.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8::fp8 {
using common::load_vec;
using common::warp_reduce_sum;
using common::warp_max;
using common::silu;
using common::bf16x2_bits_to_float2;
using common::cp_async;
using common::cp_async_zfill;
using common::cp_commit;
using common::cp_wait;
using common::smem_addr;
using common::mma_fp8_e4m3;
using common::ldmatrix_x2;
using common::ldmatrix_x4;
using common::fp8_code2_to_float2;
using common::fp8_fp32_scale;
using common::fp8_quant_fp32_pair;

struct Matrix {
    const std::uint8_t* codes;
    const __nv_bfloat16* scales;
};

__device__ __forceinline__ Matrix resolve(const ExpertMatrixDeviceView& matrix, int expert) {
    const auto offset = matrix.offsets ? matrix.offsets[expert] : 0;
    return {reinterpret_cast<const std::uint8_t*>(matrix.codes) +
                (matrix.offsets ? offset : expert * matrix.code_stride),
            reinterpret_cast<const __nv_bfloat16*>(
                matrix.scales + (matrix.offsets ? offset : expert * matrix.scale_stride))};
}

template <int Columns, bool Gate>
struct Schedule {
    static constexpr int columns = Columns, bk = 128;
    static constexpr int warp_rows    = Gate ? 8 : 4;
    static constexpr int warp_groups  = Columns >= 64 ? 2 : 1;
    static constexpr int threads      = warp_rows * warp_groups * 32;
    static constexpr int warp_columns = Columns / warp_groups;
    static constexpr int mma_columns  = warp_columns / 8;
    static constexpr int k            = Gate ? kHidden : kIntermediate;
    static constexpr int weight_rows  = Gate ? 256 : 128;
};

template <class S>
struct TileStorage {
    union {
        struct {
            alignas(16) std::uint8_t weights[2][S::weight_rows * S::bk];
            alignas(16) std::uint8_t input[2][S::columns * S::bk];
        } stage;

        float product[S::weight_rows == 256 ? S::columns : 1][128];
    };

    int sources[S::columns];
};

template <class S>
__device__ __forceinline__ int swizzle(int row, int byte) {
    return byte ^ ((row & (S::bk / 16 - 1)) << 4);
}

template <class S, bool Gate>
__device__ __forceinline__ void stage_inputs(TileStorage<S>& tile, int stage, int kt, int row0,
                                             Matrix weight, const std::uint8_t* input,
                                             int active_columns) {
    for (int task = threadIdx.x; task < S::weight_rows * (S::bk / 16); task += S::threads) {
        const int row = task / (S::bk / 16), col = task % (S::bk / 16) * 16;
        const int wr = row0 + (Gate && row >= 128 ? row - 128 + kIntermediate : row);
        cp_async<16, Cache::cg>(tile.stage.weights[stage] + row * S::bk + swizzle<S>(row, col),
                                weight.codes + std::int64_t(wr) * S::k + kt * S::bk + col);
    }
    for (int task = threadIdx.x; task < active_columns * (S::bk / 16); task += S::threads) {
        const int row = task / (S::bk / 16), col = task % (S::bk / 16) * 16;
        const int source = tile.sources[row];
        cp_async_zfill<16, Cache::cg>(tile.stage.input[stage] + row * S::bk + swizzle<S>(row, col),
                                      input + std::int64_t(source >= 0 ? source : 0) * S::k +
                                          kt * S::bk + col,
                                      source >= 0 ? 16 : 0);
    }
    cp_commit();
}

enum class PopulationRoute { All, Gate, Down, WideDown };

template <int Columns, bool Gate, bool Atomic = false,
          PopulationRoute Population = PopulationRoute::All>
__global__ __launch_bounds__(Schedule<Columns, Gate>::threads) void grouped_mma(
    const std::uint8_t* input, const float* input_scales, ExpertMatrixDeviceView bank,
    Groups groups, std::uint8_t* product, float* product_scales, float* contributions,
    const float* coefficients, float* routed_sum, int subtiles, float limit, int dense_blocks,
    Fp8PopulationTuning population = {}, int dense_assignment_threshold = 0) {
    using S = Schedule<Columns, Gate>;
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    auto& tile              = *reinterpret_cast<TileStorage<S>*>(shared_bytes);
    constexpr int row_tiles = (Gate ? kIntermediate : kHidden) / 128;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp % S::warp_rows, wn = warp / S::warp_rows;
    const int tasks = *groups.jobs * row_tiles * subtiles;
    if (!tasks) return;
    bool few_experts = false;
    if constexpr (Population != PopulationRoute::All) {
        few_experts     = groups.jobs[2] <= population.ordered_expert_limit;
        const bool wide = population.wide_down && !few_experts &&
                          groups.offsets[kExperts] > groups.jobs[2] * population.wide_min_average;
        if constexpr (Population == PopulationRoute::Down) {
            if (wide) return;
        }
        if constexpr (Population == PopulationRoute::WideDown) {
            if (!wide) return;
        }
    }
    const bool sparse = Population == PopulationRoute::All
                            ? groups.jobs[1] <= 2
                            : groups.jobs[3] <= dense_assignment_threshold;
    const int stride  = sparse ? int(gridDim.x) : dense_blocks;
    if (blockIdx.x >= stride) return;
    // Routes have distinct IDs per token, so one selected expert gives every
    // live output element exactly one writer, including after job permutation.
    const bool exclusive_output =
        Atomic && Population == PopulationRoute::Down && groups.jobs[2] == 1;
    for (int task = blockIdx.x; task < tasks; task += stride) {
        int job = task / (row_tiles * subtiles), row = task % row_tiles;
        int subtile = (task / row_tiles) % subtiles;
        if constexpr (Population == PopulationRoute::Gate || Population == PopulationRoute::Down) {
            if (few_experts) {
                const int window = population.row_window;
                if constexpr (Gate) {
                    const int e       = groups.job_experts[job];
                    const int columns = groups.offsets[e + 1] - groups.offsets[e];
                    const int expert_jobs =
                        (columns + Columns * subtiles - 1) / (Columns * subtiles);
                    const int ordinal   = groups.job_columns[job] / (Columns * subtiles);
                    const int first_job = job - ordinal;
                    const int flat   = ordinal * row_tiles * subtiles + subtile * row_tiles + row;
                    const int span   = expert_jobs * subtiles * window;
                    const int within = flat % span;
                    row              = flat / span * window + within % window;
                    const int column_job = within / window;
                    job                  = first_job + column_job / subtiles;
                    subtile              = column_job % subtiles;
                } else {
                    const int span       = *groups.jobs * subtiles * window;
                    const int within     = task % span;
                    row                  = task / span * window + within % window;
                    const int column_job = within / window;
                    job                  = column_job / subtiles;
                    subtile              = column_job % subtiles;
                }
            }
        }
        const int row0   = row * 128;
        const int expert = groups.job_experts[job];
        const int begin  = groups.offsets[expert] + groups.job_columns[job] + subtile * Columns;
        const int end    = groups.offsets[expert + 1];
        if (begin >= end) { continue; }
        // Only populated8-token atoms read B. Retain zero-fill within the final partial atom.
        const int active_columns = min(Columns, ((end - begin + 7) / 8) * 8);
        const auto weight        = resolve(bank, expert);
        if (threadIdx.x < Columns) {
            const int packed = begin + threadIdx.x;
            tile.sources[threadIdx.x] =
                packed < end ? (Gate ? groups.assignments[packed] / kTopK : packed) : -1;
        }
        __syncthreads();
        stage_inputs<S, Gate>(tile, 0, 0, row0, weight, input, active_columns);
        stage_inputs<S, Gate>(tile, 1, 1, row0, weight, input, active_columns);
        float acc[2][S::mma_columns][4] = {};
        for (int kt = 0; kt < S::k / S::bk; ++kt) {
            const int stage = kt % 2;
            cp_wait<1>();
            __syncthreads();
            const float sg = __bfloat162float(weight.scales[(row0 / 128) * (S::k / 128) + kt]);
            const float su =
                Gate ? __bfloat162float(
                           weight.scales[((row0 + kIntermediate) / 128) * (S::k / 128) + kt])
                     : sg;
            if constexpr (Population == PopulationRoute::All) {
#pragma unroll
                for (int n = 0; n < S::mma_columns; ++n) {
                    const int column = wn * S::warp_columns + n * 8;
                    if (begin + column < end) {
                        float p[2][4] = {};
#pragma unroll
                        for (int ki = 0; ki < 4; ++ki) {
                            unsigned a[2][4], b[2];
#pragma unroll
                            for (int m = 0; m < 2; ++m) {
                                const int row_base = Gate ? wm * 16 + m * 128 : wm * 32 + m * 16;
                                const int row      = row_base + (lane & 7) + ((lane >> 3 & 1) * 8);
                                const int col      = ki * 32 + (lane >> 4) * 16;
                                ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                                            smem_addr(tile.stage.weights[stage] + row * S::bk +
                                                      swizzle<S>(row, col)));
                            }
                            const int row = column + (lane & 7),
                                      col = ki * 32 + ((lane >> 3) & 1) * 16;
                            ldmatrix_x2(b[0], b[1],
                                        smem_addr(tile.stage.input[stage] + row * S::bk +
                                                  swizzle<S>(row, col)));
#pragma unroll
                            for (int m = 0; m < 2; ++m) {
                                mma_fp8_e4m3(p[m][0], p[m][1], p[m][2], p[m][3], a[m][0], a[m][1],
                                             a[m][2], a[m][3], b[0], b[1]);
                            }
                        }
#pragma unroll
                        for (int v = 0; v < 4; ++v) {
                            const int source = tile.sources[column + (lane & 3) * 2 + (v & 1)];
                            const float sx =
                                source >= 0 ? input_scales[std::int64_t(source) * (S::k / 128) + kt]
                                            : 0;
                            acc[0][n][v] = fmaf(p[0][v], sg * sx, acc[0][n][v]);
                            acc[1][n][v] = fmaf(p[1][v], su * sx, acc[1][n][v]);
                        }
                    }
                }
            } else if (few_experts || Population == PopulationRoute::WideDown) {
                constexpr int cluster = 2 < S::mma_columns ? 2 : S::mma_columns;
#pragma unroll
                for (int n0 = 0; n0 < S::mma_columns; n0 += cluster) {
                    const int first_column = wn * S::warp_columns + n0 * 8;
                    if (begin + first_column < end) {
                        float p[2][cluster][4] = {};
#pragma unroll
                        for (int ki = 0; ki < 4; ++ki) {
                            unsigned a[2][4];
#pragma unroll
                            for (int m = 0; m < 2; ++m) {
                                const int row_base = Gate ? wm * 16 + m * 128 : wm * 32 + m * 16;
                                const int row      = row_base + (lane & 7) + ((lane >> 3 & 1) * 8);
                                const int col      = ki * 32 + (lane >> 4) * 16;
                                ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                                            smem_addr(tile.stage.weights[stage] + row * S::bk +
                                                      swizzle<S>(row, col)));
                            }
#pragma unroll
                            for (int n = 0; n < cluster; ++n) {
                                const int column = first_column + n * 8;
                                if (begin + column < end) {
                                    unsigned b[2];
                                    const int row = column + (lane & 7),
                                              col = ki * 32 + ((lane >> 3) & 1) * 16;
                                    ldmatrix_x2(b[0], b[1],
                                                smem_addr(tile.stage.input[stage] + row * S::bk +
                                                          swizzle<S>(row, col)));
#pragma unroll
                                    for (int m = 0; m < 2; ++m)
                                        mma_fp8_e4m3(p[m][n][0], p[m][n][1], p[m][n][2], p[m][n][3],
                                                     a[m][0], a[m][1], a[m][2], a[m][3], b[0],
                                                     b[1]);
                                }
                            }
                        }
#pragma unroll
                        for (int n = 0; n < cluster; ++n) {
                            const int column = first_column + n * 8;
                            if (begin + column < end) {
#pragma unroll
                                for (int v = 0; v < 4; ++v) {
                                    const int source =
                                        tile.sources[column + (lane & 3) * 2 + (v & 1)];
                                    const float sx =
                                        source >= 0
                                            ? input_scales[std::int64_t(source) * (S::k / 128) + kt]
                                            : 0;
                                    acc[0][n0 + n][v] =
                                        fmaf(p[0][n][v], sg * sx, acc[0][n0 + n][v]);
                                    acc[1][n0 + n][v] =
                                        fmaf(p[1][n][v], su * sx, acc[1][n0 + n][v]);
                                }
                            }
                        }
                    }
                }
            } else {
#pragma unroll
                for (int n = 0; n < S::mma_columns; ++n) {
                    const int column = wn * S::warp_columns + n * 8;
                    if (begin + column < end) {
                        float p[2][4] = {};
#pragma unroll
                        for (int ki = 0; ki < 4; ++ki) {
                            unsigned a[2][4], b[2];
#pragma unroll
                            for (int m = 0; m < 2; ++m) {
                                const int row_base = Gate ? wm * 16 + m * 128 : wm * 32 + m * 16;
                                const int row      = row_base + (lane & 7) + ((lane >> 3 & 1) * 8);
                                const int col      = ki * 32 + (lane >> 4) * 16;
                                ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                                            smem_addr(tile.stage.weights[stage] + row * S::bk +
                                                      swizzle<S>(row, col)));
                            }
                            const int row = column + (lane & 7),
                                      col = ki * 32 + ((lane >> 3) & 1) * 16;
                            ldmatrix_x2(b[0], b[1],
                                        smem_addr(tile.stage.input[stage] + row * S::bk +
                                                  swizzle<S>(row, col)));
#pragma unroll
                            for (int m = 0; m < 2; ++m) {
                                mma_fp8_e4m3(p[m][0], p[m][1], p[m][2], p[m][3], a[m][0], a[m][1],
                                             a[m][2], a[m][3], b[0], b[1]);
                            }
                        }
#pragma unroll
                        for (int v = 0; v < 4; ++v) {
                            const int source = tile.sources[column + (lane & 3) * 2 + (v & 1)];
                            const float sx =
                                source >= 0 ? input_scales[std::int64_t(source) * (S::k / 128) + kt]
                                            : 0;
                            acc[0][n][v] = fmaf(p[0][v], sg * sx, acc[0][n][v]);
                            acc[1][n][v] = fmaf(p[1][v], su * sx, acc[1][n][v]);
                        }
                    }
                }
            }
            __syncthreads();
            if (kt + 2 < S::k / S::bk) {
                stage_inputs<S, Gate>(tile, stage, kt + 2, row0, weight, input, active_columns);
            } else {
                cp_commit();
            }
        }
        cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int n = 0; n < S::mma_columns; ++n) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int column = wn * S::warp_columns + n * 8 + (lane & 3) * 2 + (v & 1);
                if constexpr (Gate) {
                    const int row = wm * 16 + (lane >> 2) + (v / 2) * 8;
                    if (begin + column < end)
                        tile.product[column][row] = silu(fminf(acc[0][n][v], limit)) *
                                                    fmaxf(-limit, fminf(acc[1][n][v], limit));
                } else {
                    if (begin + column < end) {
#pragma unroll
                        for (int m = 0; m < 2; ++m) {
                            const int row = row0 + wm * 32 + m * 16 + (lane >> 2) + (v / 2) * 8;
                            if constexpr (Atomic) {
                                const int assignment = groups.assignments[begin + column];
                                const auto index = std::int64_t(assignment / kTopK) * kHidden + row;
                                const float value = acc[m][n][v] * coefficients[assignment];
                                if (exclusive_output)
                                    routed_sum[index] = value;
                                else
                                    atomicAdd(routed_sum + index, value);
                            } else {
                                contributions[std::int64_t(begin + column) * kHidden + row] =
                                    acc[m][n][v];
                            }
                        }
                    }
                }
            }
        }
        if constexpr (Gate) {
            __syncthreads();
            for (int column = warp; column < Columns; column += S::threads / 32) {
                if (begin + column < end) {
                    const float4 value =
                        *reinterpret_cast<const float4*>(&tile.product[column][lane * 4]);
                    float maximum     = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                              fmaxf(fabsf(value.z), fabsf(value.w)));
                    maximum           = warp_max(maximum);
                    const float scale = fp8_fp32_scale(maximum);
                    const unsigned packed =
                        fp8_quant_fp32_pair(make_float2(value.x, value.y), scale) |
                        (unsigned(fp8_quant_fp32_pair(make_float2(value.z, value.w), scale)) << 16);
                    *reinterpret_cast<unsigned*>(
                        product + std::int64_t(begin + column) * kIntermediate + row0 + lane * 4) =
                        packed;
                    if (!lane) {
                        product_scales[std::int64_t(begin + column) * (kIntermediate / 128) +
                                       row0 / 128] = scale;
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

__device__ __forceinline__ float dot_four(unsigned codes, float4 x) {
    const float2 lo = fp8_code2_to_float2(codes), hi = fp8_code2_to_float2(codes >> 16);
    return fmaf(hi.y, x.w, fmaf(hi.x, x.z, fmaf(lo.y, x.y, lo.x * x.x)));
}

__global__ void vector_gate(const __nv_bfloat16* x, ExpertMatrixDeviceView bank, const int* ids,
                            ExpertSubsetKind kind, const int* mask, float* activation,
                            float limit) {
    const int assignment = blockIdx.y, expert = ids[assignment];
    if (!includes(kind, mask, expert)) { return; }
    const auto weight = resolve(bank, expert);
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    float gate = 0, up = 0;
    for (int block = 0; block < kHidden / 128; ++block) {
        const int k     = block * 128 + lane * 4;
        const auto bits = load_vec<uint2>(x + std::int64_t(assignment / kTopK) * kHidden + k);
        const float2 a = bf16x2_bits_to_float2(bits.x), b = bf16x2_bits_to_float2(bits.y);
        const float4 input = make_float4(a.x, a.y, b.x, b.y);
        const unsigned g   = load_vec<unsigned>(weight.codes + std::int64_t(row) * kHidden + k);
        const unsigned u =
            load_vec<unsigned>(weight.codes + std::int64_t(row + kIntermediate) * kHidden + k);
        gate = fmaf(dot_four(g, input),
                    __bfloat162float(weight.scales[(row / 128) * (kHidden / 128) + block]), gate);
        up   = fmaf(dot_four(u, input),
                    __bfloat162float(
                      weight.scales[((row + kIntermediate) / 128) * (kHidden / 128) + block]),
                    up);
    }
    gate = warp_reduce_sum(gate);
    up   = warp_reduce_sum(up);
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
        for (int block = 0; block < kIntermediate / 128; ++block) {
            const int k = block * 128 + lane * 4;
            const float4 x =
                load_vec<float4>(activation + std::int64_t(assignment) * kIntermediate + k);
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const int row = blockIdx.x * Rows + r;
                const unsigned w =
                    load_vec<unsigned>(weight.codes + std::int64_t(row) * kIntermediate + k);
                sum[r] = fmaf(
                    dot_four(w, x),
                    __bfloat162float(weight.scales[(row / 128) * (kIntermediate / 128) + block]),
                    sum[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            sum[r] = warp_reduce_sum(sum[r]) * coefficients[assignment];
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
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8::fp8
