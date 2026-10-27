#pragma once

#include "ops/moe/kernels/cuda/block_scaled/fp8/common.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled::fp8 {
template <class S, bool Gate>
__device__ __forceinline__ void stage_inputs(TileStorage<S>& tile, int stage, int kt, int row0,
                                             Matrix weight, const std::uint8_t* input) {
    for (int task = threadIdx.x; task < S::weight_rows * (S::bk / 16); task += S::threads) {
        const int row = task / (S::bk / 16), col = task % (S::bk / 16) * 16;
        const int wr = row0 + (Gate && row >= 128 ? row - 128 + kIntermediate : row);
        cp_async<16, Cache::cg>(tile.stage.weights[stage] + row * S::bk + swizzle<S>(row, col),
                                weight.codes + std::int64_t(wr) * S::k + kt * S::bk + col);
    }
    for (int task = threadIdx.x; task < S::columns * (S::bk / 16); task += S::threads) {
        const int row = task / (S::bk / 16), col = task % (S::bk / 16) * 16;
        const int source = tile.sources[row];
        cp_async_zfill<16, Cache::cg>(tile.stage.input[stage] + row * S::bk + swizzle<S>(row, col),
                                      input + std::int64_t(source >= 0 ? source : 0) * S::k +
                                          kt * S::bk + col,
                                      source >= 0 ? 16 : 0);
    }
    cp_commit();
}

template <int Columns, bool Gate, bool Atomic = false>
__global__ __launch_bounds__(Schedule<Columns, Gate>::threads) void grouped_mma(
    const std::uint8_t* input, const float* input_scales, ExpertMatrixDeviceView bank,
    Groups groups, std::uint8_t* product, float* product_scales, float* contributions,
    const float* coefficients, float* routed_sum, int subtiles) {
    using S = Schedule<Columns, Gate>;
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    auto& tile              = *reinterpret_cast<TileStorage<S>*>(shared_bytes);
    constexpr int row_tiles = (Gate ? kIntermediate : kHidden) / 128;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp % S::warp_rows, wn = warp / S::warp_rows;
    const int tasks = *groups.jobs * row_tiles * subtiles;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int job = task / (row_tiles * subtiles), row0 = (task % row_tiles) * 128;
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
        stage_inputs<S, Gate>(tile, 0, 0, row0, weight, input);
        stage_inputs<S, Gate>(tile, 1, 1, row0, weight, input);
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
                        const int row = column + (lane & 7), col = ki * 32 + ((lane >> 3) & 1) * 16;
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
            __syncthreads();
            if (kt + 2 < S::k / S::bk) {
                stage_inputs<S, Gate>(tile, stage, kt + 2, row0, weight, input);
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
                    const int row             = wm * 16 + (lane >> 2) + (v / 2) * 8;
                    tile.product[column][row] = silu(acc[0][n][v]) * acc[1][n][v];
                } else {
                    if (begin + column < end) {
#pragma unroll
                        for (int m = 0; m < 2; ++m) {
                            const int row = row0 + wm * 32 + m * 16 + (lane >> 2) + (v / 2) * 8;
                            if constexpr (Atomic) {
                                const int assignment = groups.assignments[begin + column];
                                atomicAdd(routed_sum + std::int64_t(assignment / kTopK) * kHidden +
                                              row,
                                          acc[m][n][v] * coefficients[assignment]);
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

__global__ void vector_gate(const __nv_bfloat16* x, ExpertMatrixDeviceView bank, const int* ids,
                            ExpertSubsetKind kind, const int* mask, float* activation) {
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
    if (!lane) { activation[std::int64_t(assignment) * kIntermediate + row] = silu(gate) * up; }
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
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::fp8
