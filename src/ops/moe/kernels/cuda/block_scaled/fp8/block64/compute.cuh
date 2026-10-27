#pragma once

#include "ops/moe/kernels/cuda/block_scaled/fp8/common.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled::fp8::block64 {
inline constexpr int kLocalIntermediate = 320;

template <int Columns, bool Gate>
struct Schedule : fp8::Schedule<Columns, Gate> {
    static constexpr int k      = Gate ? kHidden : kLocalIntermediate;
    static constexpr int groups = (k + 127) / 128;
};

template <class S, bool Gate>
__device__ __forceinline__ void stage_inputs(TileStorage<S>& tile, int stage, int kt, int row0,
                                             Matrix weight, const std::uint8_t* input,
                                             int active_columns) {
    for (int task = threadIdx.x; task < S::weight_rows * 8; task += S::threads) {
        const int row = task / 8, col = task % 8 * 16;
        const int logical_row = row0 + row % 128;
        const int wr = Gate ? logical_row + (row >= 128 ? kLocalIntermediate : 0) : row0 + row;
        const bool row_valid = !Gate || logical_row < kLocalIntermediate;
        if (!row_valid) continue;
        const bool k_valid = kt * 128 + col < S::k;
        cp_async_zfill<16, Cache::cg>(tile.stage.weights[stage] + row * 128 + swizzle<S>(row, col),
                                      weight.codes + std::int64_t(row_valid ? wr : 0) * S::k +
                                          (k_valid ? kt * 128 + col : 0),
                                      row_valid && k_valid ? 16 : 0);
    }
    for (int task = threadIdx.x; task < active_columns * 8; task += S::threads) {
        const int row = task / 8, col = task % 8 * 16;
        const int source = tile.sources[row];
        const bool valid = source >= 0 && kt * 128 + col < S::k;
        cp_async_zfill<16, Cache::cg>(tile.stage.input[stage] + row * 128 + swizzle<S>(row, col),
                                      input + std::int64_t(source >= 0 ? source : 0) * S::k +
                                          (valid ? kt * 128 + col : 0),
                                      valid ? 16 : 0);
    }
    cp_commit();
}

// Keep one A fragment live across two output atoms; scale each input column
// once for its two row halves. Parts is two K32 atoms, or four for equal K64 scales.
template <class S, bool Gate, int Parts>
__device__ __forceinline__ void contract_segment(TileStorage<S>& tile, int stage, int first_k32,
                                                 int kt, int begin, int end,
                                                 const float* input_scales, float sg, float su,
                                                 float (&acc)[2][S::mma_columns][4]) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp % S::warp_rows, wn = warp / S::warp_rows;
    constexpr int cluster = S::mma_columns < 2 ? S::mma_columns : 2;
#pragma unroll
    for (int n0 = 0; n0 < S::mma_columns; n0 += cluster) {
        const int first = wn * S::warp_columns + n0 * 8;
        if (begin + first >= end) continue;
        float p[2][cluster][4]{};
#pragma unroll
        for (int part = 0; part < Parts; ++part) {
            const int ki = first_k32 + part;
            unsigned a[2][4];
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const int row_base = Gate ? wm * 16 + m * 128 : wm * 32 + m * 16;
                const int row      = row_base + (lane & 7) + ((lane >> 3 & 1) * 8);
                const int col      = ki * 32 + (lane >> 4) * 16;
                ldmatrix_x4(
                    a[m][0], a[m][1], a[m][2], a[m][3],
                    smem_addr(tile.stage.weights[stage] + row * 128 + swizzle<S>(row, col)));
            }
#pragma unroll
            for (int n = 0; n < cluster; ++n) {
                const int column = first + n * 8;
                if (begin + column >= end) continue;
                unsigned b[2];
                const int row = column + (lane & 7), col = ki * 32 + ((lane >> 3) & 1) * 16;
                ldmatrix_x2(b[0], b[1],
                            smem_addr(tile.stage.input[stage] + row * 128 + swizzle<S>(row, col)));
#pragma unroll
                for (int m = 0; m < 2; ++m)
                    mma_fp8_e4m3(p[m][n][0], p[m][n][1], p[m][n][2], p[m][n][3], a[m][0], a[m][1],
                                 a[m][2], a[m][3], b[0], b[1]);
            }
        }
#pragma unroll
        for (int n = 0; n < cluster; ++n) {
            const int column = first + n * 8;
            if (begin + column >= end) continue;
            const int c0      = column + (lane & 3) * 2;
            const int source0 = tile.sources[c0], source1 = tile.sources[c0 + 1];
            const float sx0 =
                source0 >= 0 ? input_scales[std::int64_t(source0) * S::groups + kt] : 0;
            const float sx1 =
                source1 >= 0 ? input_scales[std::int64_t(source1) * S::groups + kt] : 0;
            const float gs0 = sg * sx0, gs1 = sg * sx1, us0 = su * sx0, us1 = su * sx1;
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                acc[0][n0 + n][v] = fmaf(p[0][n][v], (v & 1) ? gs1 : gs0, acc[0][n0 + n][v]);
                acc[1][n0 + n][v] = fmaf(p[1][n][v], (v & 1) ? us1 : us0, acc[1][n0 + n][v]);
            }
        }
    }
}

// Weight K64 scales and activation G128 scales are independent. Each K64
// contraction is rescaled before it joins the FP32 sum; equal adjacent scales
// may share that step. The last down tile keeps only its 64 valid K elements.
template <int Columns, bool Gate, bool Atomic>
__global__ __launch_bounds__(Schedule<Columns, Gate>::threads) void grouped_mma(
    const std::uint8_t* __restrict__ input, const float* __restrict__ input_scales,
    ExpertMatrixDeviceView bank, Groups groups, std::uint8_t* __restrict__ product,
    float* __restrict__ product_scales, float* __restrict__ contributions,
    const float* __restrict__ coefficients, float* __restrict__ routed_sum, int subtile_bits,
    int minimum_count, int maximum_count) {
    if (groups.load_range &&
        (groups.load_range[1] <= minimum_count || groups.load_range[0] > maximum_count))
        return;
    using S = Schedule<Columns, Gate>;
    extern __shared__ __align__(16) unsigned char bytes[];
    auto& tile              = *reinterpret_cast<TileStorage<S>*>(bytes);
    constexpr int row_tiles = Gate ? (kLocalIntermediate + 127) / 128 : kHidden / 128;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp % S::warp_rows, wn = warp / S::warp_rows;
    const int subtiles = 1 << subtile_bits;
    const int tasks    = *groups.jobs * row_tiles * subtiles;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int job = (task / row_tiles) >> subtile_bits, row0 = task % row_tiles * 128;
        const int subtile = (task / row_tiles) & (subtiles - 1);
        const int expert  = groups.job_experts[job];
        const int begin   = groups.offsets[expert] + groups.job_columns[job] + subtile * Columns;
        const int end     = groups.offsets[expert + 1];
        if (begin >= end) continue;
        const int count = end - groups.offsets[expert];
        if (count <= minimum_count || count > maximum_count) continue;

        const int active_columns = min(Columns, (end - begin + 7) / 8 * 8);
        const auto weight        = resolve(bank, expert);
        if (threadIdx.x < Columns) {
            const int packed = begin + threadIdx.x;
            tile.sources[threadIdx.x] =
                packed < end ? (Gate ? groups.assignments[packed] / kTopK : packed) : -1;
        }
        __syncthreads();
        stage_inputs<S, Gate>(tile, 0, 0, row0, weight, input, active_columns);
        stage_inputs<S, Gate>(tile, 1, 1, row0, weight, input, active_columns);
        float acc[2][S::mma_columns][4]{};
        for (int kt = 0; kt < S::groups; ++kt) {
            const int stage = kt % 2;
            cp_wait<1>();
            __syncthreads();
            const int row        = row0 + (Gate ? wm * 16 : wm * 32);
            const bool valid_row = !Gate || row < kLocalIntermediate;
            const bool full_k128 = kt * 128 + 128 <= S::k;
            const int scale_k    = kt * 2;
            const float sg0 =
                valid_row ? __bfloat162float(weight.scales[(row / 64) * (S::k / 64) + scale_k]) : 0;
            const float su0 =
                Gate && valid_row
                    ? __bfloat162float(
                          weight.scales[((row + kLocalIntermediate) / 64) * (S::k / 64) + scale_k])
                    : sg0;
            const float sg1 =
                full_k128 && valid_row
                    ? __bfloat162float(weight.scales[(row / 64) * (S::k / 64) + scale_k + 1])
                    : 0;
            const float su1 =
                Gate && full_k128 && valid_row
                    ? __bfloat162float(
                          weight.scales[((row + kLocalIntermediate) / 64) * (S::k / 64) + scale_k +
                                        1])
                    : sg1;
            // The last GU tile has only 64 live rows. Its other warps do not
            // consume staged weights, so their copies and MMA work are unnecessary.
            if (valid_row) {
                if (full_k128 && sg0 == sg1 && su0 == su1) {
                    contract_segment<S, Gate, 4>(tile, stage, 0, kt, begin, end, input_scales, sg0,
                                                 su0, acc);
                } else {
                    contract_segment<S, Gate, 2>(tile, stage, 0, kt, begin, end, input_scales, sg0,
                                                 su0, acc);
                    if (full_k128)
                        contract_segment<S, Gate, 2>(tile, stage, 2, kt, begin, end, input_scales,
                                                     sg1, su1, acc);
                }
            }
            __syncthreads();
            if (kt + 2 < S::groups)
                stage_inputs<S, Gate>(tile, stage, kt + 2, row0, weight, input, active_columns);
            else
                cp_commit();
        }
        cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int n = 0; n < S::mma_columns; ++n) {
            const int base_column = wn * S::warp_columns + n * 8 + (lane & 3) * 2;
            if constexpr (Gate) {
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const int column = base_column + (v & 1);
                    if (begin + column >= end) continue;
                    const int row = wm * 16 + (lane >> 2) + (v / 2) * 8;
                    tile.product[column][row] =
                        row0 + row < kLocalIntermediate ? silu(acc[0][n][v]) * acc[1][n][v] : 0;
                }
            } else {
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const int column = base_column + t;
                    if (begin + column >= end) continue;
                    float* target;
                    float coefficient = 1;
                    if constexpr (Atomic) {
                        const int assignment = groups.assignments[begin + column];
                        coefficient          = coefficients[assignment];
                        target = routed_sum + std::int64_t(assignment / kTopK) * kHidden;
                    } else
                        target = contributions + std::int64_t(begin + column) * kHidden;
#pragma unroll
                    for (int m = 0; m < 2; ++m)
#pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            const int v   = t + 2 * half;
                            const int row = row0 + wm * 32 + m * 16 + (lane >> 2) + half * 8;
                            if constexpr (Atomic)
                                atomicAdd(target + row, acc[m][n][v] * coefficient);
                            else
                                target[row] = acc[m][n][v];
                        }
                }
            }
        }
        if constexpr (Gate) {
            __syncthreads();
            for (int column = warp; column < Columns; column += S::threads / 32) {
                if (begin + column >= end) continue;
                const float4 value =
                    *reinterpret_cast<const float4*>(&tile.product[column][lane * 4]);
                const float maximum = warp_max(fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                                     fmaxf(fabsf(value.z), fabsf(value.w))));
                const float scale   = fp8_fp32_scale(maximum);
                const unsigned packed =
                    fp8_quant_fp32_pair(make_float2(value.x, value.y), scale) |
                    (unsigned(fp8_quant_fp32_pair(make_float2(value.z, value.w), scale)) << 16);
                if (row0 + lane * 4 < kLocalIntermediate)
                    *reinterpret_cast<unsigned*>(product +
                                                 std::int64_t(begin + column) * kLocalIntermediate +
                                                 row0 + lane * 4) = packed;
                if (!lane) product_scales[std::int64_t(begin + column) * 3 + row0 / 128] = scale;
            }
        }
        __syncthreads();
    }
}

__global__ void vector_gate(const __nv_bfloat16* x, ExpertMatrixDeviceView bank, const int* ids,
                            ExpertSubsetKind kind, const int* mask, float* activation) {
    const int assignment = blockIdx.y, expert = ids[assignment];
    if (!includes(kind, mask, expert)) return;
    const auto weight = resolve(bank, expert);
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    float gate = 0, up = 0;
    for (int block = 0; block < kHidden / 128; ++block) {
        const int k     = block * 128 + lane * 4;
        const auto bits = load_vec<uint2>(x + std::int64_t(assignment / kTopK) * kHidden + k);
        const auto a = bf16x2_bits_to_float2(bits.x), b = bf16x2_bits_to_float2(bits.y);
        const auto input = make_float4(a.x, a.y, b.x, b.y);
        const auto g     = load_vec<unsigned>(weight.codes + std::int64_t(row) * kHidden + k);
        const auto u =
            load_vec<unsigned>(weight.codes + std::int64_t(row + kLocalIntermediate) * kHidden + k);
        gate = fmaf(dot_four(g, input),
                    __bfloat162float(weight.scales[(row / 64) * (kHidden / 64) + k / 64]), gate);
        up   = fmaf(dot_four(u, input),
                    __bfloat162float(
                      weight.scales[((row + kLocalIntermediate) / 64) * (kHidden / 64) + k / 64]),
                    up);
    }
    gate = warp_reduce_sum(gate);
    up   = warp_reduce_sum(up);
    if (!lane) activation[std::int64_t(assignment) * kLocalIntermediate + row] = silu(gate) * up;
}

template <int Rows, class Output>
__global__ void vector_down(const float* activation, ExpertMatrixDeviceView bank, const int* ids,
                            const float* coefficients, ExpertSubsetKind kind, const int* mask,
                            Output output) {
    __shared__ float values[kTopK][Rows];
    const int token = blockIdx.y, route = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int assignment = token * kTopK + route, expert = ids[assignment];
    float sum[Rows]{};
    if (includes(kind, mask, expert)) {
        const auto weight = resolve(bank, expert);
        for (int block = 0; block < 3; ++block) {
            const int k = block * 128 + lane * 4;
            if (k >= kLocalIntermediate) continue;
            const auto x =
                load_vec<float4>(activation + std::int64_t(assignment) * kLocalIntermediate + k);
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const int row = blockIdx.x * Rows + r;
                const auto w =
                    load_vec<unsigned>(weight.codes + std::int64_t(row) * kLocalIntermediate + k);
                const float scale = __bfloat162float(weight.scales[(row / 64) * 5 + k / 64]);
                sum[r]            = fmaf(dot_four(w, x), scale, sum[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) sum[r] = warp_reduce_sum(sum[r]) * coefficients[assignment];
    }
    if (!lane) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) values[route][r] = sum[r];
    }
    __syncthreads();
    if (threadIdx.x < Rows) {
        float total = 0;
#pragma unroll
        for (int j = 0; j < kTopK; ++j) total += values[j][threadIdx.x];
        const int i = token * kHidden + blockIdx.x * Rows + threadIdx.x;
        output.store(i, output.initial(i) + total);
    }
}
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::fp8::block64
