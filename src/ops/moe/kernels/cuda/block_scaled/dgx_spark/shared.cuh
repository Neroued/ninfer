#pragma once

#include "ops/moe/kernels/cuda/block_scaled/output.cuh"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/host/launch.h"

#include <cuda.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail::moe_cuda::block_scaled::spark_shared {
using namespace common;

template <int Rows, int Columns, int Threads, class Output>
__device__ void finish_output(float* tile, int row0, int token0, int tokens, const Storage& storage,
                              const float* coefficients, const float* beta, Output output) {
    for (int vector = threadIdx.x; vector < Columns * (Rows / 4); vector += Threads) {
        const int token = token0 + vector / (Rows / 4), row = vector % (Rows / 4) * 4;
        if (token >= tokens) continue;
        const bool live        = output.is_live(token);
        const int physical_row = row ^ (((token - token0) & 7) << 3);
        const float4 a  = common::load_vec<float4>(tile + (token - token0) * Rows + physical_row);
        const int index = token * kHidden + row0 + row;
        float4 routed{};
        if (storage.routed_sum)
            routed = common::load_vec<float4>(storage.routed_sum + index);
        else {
            routed.x = routed_value(token, row0 + row, storage, coefficients);
            routed.y = routed_value(token, row0 + row + 1, storage, coefficients);
            routed.z = routed_value(token, row0 + row + 2, storage, coefficients);
            routed.w = routed_value(token, row0 + row + 3, storage, coefficients);
        }
        const float b = beta[token];
        if constexpr (std::is_same_v<Output, BfloatResult>) {
            if (!live && output.add_residual) continue;
            float4 initial{};
            if (output.add_residual) {
                const uint2 packed = common::load_vec<uint2>(output.data + index);
                const auto lo      = common::bf16x2_bits_to_float2(packed.x),
                           hi      = common::bf16x2_bits_to_float2(packed.y);
                initial            = make_float4(lo.x, lo.y, hi.x, hi.y);
            }
            const auto lo = __floats2bfloat162_rn(live ? initial.x + routed.x + b * a.x : 0,
                                                  live ? initial.y + routed.y + b * a.y : 0);
            const auto hi = __floats2bfloat162_rn(live ? initial.z + routed.z + b * a.z : 0,
                                                  live ? initial.w + routed.w + b * a.w : 0);
            const __nv_bfloat162_raw low = lo, high = hi;
            common::store_vec(output.data + index,
                              make_uint2(unsigned(low.x) | (unsigned(low.y) << 16),
                                         unsigned(high.x) | (unsigned(high.y) << 16)));
        } else {
            common::store_vec(output.data + index, make_float4(live ? routed.x + b * a.x : 0,
                                                               live ? routed.y + b * a.y : 0,
                                                               live ? routed.z + b * a.z : 0,
                                                               live ? routed.w + b * a.w : 0));
        }
    }
}

template <int Columns, int Warps, int Stages>
union SplitKTile {
    struct {
        alignas(16) __nv_bfloat16 weight[Warps][Stages][16 * 64];
        alignas(16) __nv_bfloat16 input[Warps][Stages][Columns * 64];
    } stage;

    float partial[Warps][Columns][16];
};

template <int Intermediate, int Columns, int Warps, int Stages, bool Split, int Blocks>
__global__ __launch_bounds__(Warps * 32, Blocks) void shared_gate_split_k(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* control,
    Storage storage, float* scores, float* shared_gate, int tokens) {
    auto& tile = *reinterpret_cast<SplitKTile<Columns, Warps, Stages>*>(
        shared_storage<sizeof(SplitKTile<Columns, Warps, Stages>)>());
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int control_rows   = control ? (scores ? kExperts + 1 : 1) : 0;
    const int control_blocks = (control_rows + 15) / 16;
    const bool route         = blockIdx.x < control_blocks;
    const int row0           = route ? (control_rows == 1 ? kExperts : blockIdx.x * 16)
                                     : (blockIdx.x - control_blocks) * 8;
    const int token0         = blockIdx.y * Columns;
    if (control && storage.groups.counts && blockIdx.x == 0 && blockIdx.y == 0)
        for (int e = threadIdx.x; e < kExperts; e += Warps * 32) storage.groups.counts[e] = 0;
    auto stage = [&](int s, int kt) {
        for (int item = lane; item < 16 * 8; item += 32) {
            const int row = item / 8, k = item % 8 * 8;
            const int wr     = row0 + (route ? row : (row < 8 ? row : row - 8 + Intermediate));
            const bool valid = (!route || wr < kExperts + 1) && kt < kHidden / 64;
            cp_async_zfill<16>(tile.stage.weight[warp][s] + row * 64 + swizzle_xor_64(row, k),
                               (route ? control : weight) + std::int64_t(valid ? wr : 0) * kHidden +
                                   (valid ? kt * 64 + k : 0),
                               valid ? 16 : 0);
        }
        for (int item = lane; item < Columns * 8; item += 32) {
            const int row = item / 8, k = item % 8 * 8;
            const bool valid = token0 + row < tokens && kt < kHidden / 64;
            cp_async_zfill<16>(tile.stage.input[warp][s] + row * 64 + swizzle_xor_64(row, k),
                               x + std::int64_t(valid ? token0 + row : 0) * kHidden +
                                   (valid ? kt * 64 + k : 0),
                               valid ? 16 : 0);
        }
        cp_commit();
    };
#pragma unroll
    for (int s = 0; s < Stages; ++s) stage(s, warp + s * Warps);
    float acc[Columns / 8][4]{};
    for (int tick = 0; tick < (kHidden / 64 + Warps - 1) / Warps; ++tick) {
        const int s = tick % Stages;
        cp_wait<Stages - 1>();
        __syncwarp();
#pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
            unsigned a[4];
            const int row = (lane & 7) + ((lane >> 3) & 1) * 8, k = ki * 16 + (lane >> 4) * 8;
            ldmatrix_x4(a[0], a[1], a[2], a[3],
                        smem_addr(tile.stage.weight[warp][s] + row * 64 + swizzle_xor_64(row, k)));
#pragma unroll
            for (int n = 0; n < Columns / 8; ++n) {
                const int xr = n * 8 + (lane & 7), xk = ki * 16 + ((lane >> 3) & 1) * 8;
                unsigned b[2];
                ldmatrix_x2(
                    b[0], b[1],
                    smem_addr(tile.stage.input[warp][s] + xr * 64 + swizzle_xor_64(xr, xk)));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], a[0], a[1], a[2], a[3], b[0],
                         b[1]);
            }
        }
        __syncwarp();
        if (tick + Stages < (kHidden / 64 + Warps - 1) / Warps)
            stage(s, warp + (tick + Stages) * Warps);
        else
            cp_commit();
    }
    cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int n = 0; n < Columns / 8; ++n)
#pragma unroll
        for (int v = 0; v < 4; ++v)
            tile.partial[warp][n * 8 + (lane & 3) * 2 + (v & 1)][(lane >> 2) + (v / 2) * 8] =
                acc[n][v];
    __syncthreads();
    if (route) {
        for (int item = threadIdx.x; item < Columns * 16; item += Warps * 32) {
            const int token = token0 + item / 16, row = item % 16;
            if (token >= tokens || row0 + row > kExperts) continue;
            float value = 0;
#pragma unroll
            for (int w = 0; w < Warps; ++w) value += tile.partial[w][item / 16][row];
            if (scores)
                scores[token * kScoreRows + row0 + row] = value;
            else
                shared_gate[token] =
                    storage.token_validity.contains(token) ? 1.F / (1.F + __expf(-value)) : 0;
        }
    } else {
        auto* h = static_cast<__nv_bfloat16*>(storage.shared_activation);
        for (int item = threadIdx.x; item < Columns * 8; item += Warps * 32) {
            const int token = token0 + item / 8, row = item % 8;
            if (token >= tokens) continue;
            float g = 0, u = 0;
#pragma unroll
            for (int w = 0; w < Warps; ++w) {
                g += tile.partial[w][item / 8][row];
                u += tile.partial[w][item / 8][row + 8];
            }
            const float value                    = silu(g) * u;
            const auto hi                        = __float2bfloat16_rn(value);
            h[token * Intermediate + row0 + row] = hi;
            if constexpr (Split)
                h[tokens * Intermediate + token * Intermediate + row0 + row] =
                    __float2bfloat16_rn(value - __bfloat162float(hi));
        }
    }
}

template <int I, int C, int W, int Stages, int Blocks, bool Split>
void launch_split_k_tile(const __nv_bfloat16* x, const __nv_bfloat16* weights,
                         const __nv_bfloat16* control, const Storage& storage, float* scores,
                         float* shared_gate, int tokens, CudaExecutionView cuda) {
    constexpr auto kernel    = shared_gate_split_k<I, C, W, Stages, Split, Blocks>;
    const int bytes          = common::prepare_shared<sizeof(SplitKTile<C, W, Stages>), kernel>();
    const int control_blocks = control ? (scores ? (kExperts + 16) / 16 : 1) : 0;
    kernel<<<dim3(I / 8 + control_blocks, (tokens + C - 1) / C), W * 32, bytes, cuda.stream()>>>(
        x, weights, control, storage, scores, shared_gate, tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <int Intermediate, bool SplitActivation>
void launch_split_k(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                    const __nv_bfloat16* control, const Storage& storage, float* scores,
                    float* shared_gate, int tokens, int columns, CudaExecutionView execution) {
    switch (columns) {
    case 16:
        launch_split_k_tile<Intermediate, 16, 8, 2, 1, SplitActivation>(
            input, weight, control, storage, scores, shared_gate, tokens, execution);
        break;
    case 32:
        launch_split_k_tile<Intermediate, 32, 4, 2, 2, SplitActivation>(
            input, weight, control, storage, scores, shared_gate, tokens, execution);
        break;
    default:
        throw std::logic_error("invalid MoE shared split-K column tile");
    }
}

struct alignas(128) Descriptors {
    CUtensorMap weight, input, low, control;
};

template <bool Gate, int Columns, int Intermediate, bool SplitActivation, int RowWarps, int Stages,
          int Blocks>
struct SharedSchedule {
    static constexpr int columns = Columns, intermediate = Intermediate, warps_rows = RowWarps,
                         stages = Stages, blocks = Blocks;
    static constexpr int weight_rows = RowWarps * 32, rows = Gate ? RowWarps * 16 : RowWarps * 32,
                         k         = Gate ? kHidden : Intermediate,
                         planes    = Gate || !SplitActivation ? 1 : 2;
    static constexpr int consumers = RowWarps * 2, threads = 32 + consumers * 32,
                         atoms          = Columns / 16;
    static constexpr int tensor_bytes   = Stages * (weight_rows + planes * Columns) * 64 * 2;
    static constexpr int epilogue_bytes = Gate ? 0 : rows * Columns * 4;
    static constexpr int bytes =
        (tensor_bytes > epilogue_bytes ? tensor_bytes : epilogue_bytes) + Stages * 16;
};

inline CUtensorMap make_map(const __nv_bfloat16* pointer, int rows, int k, int tile_rows) {
    CUtensorMap map{};
    const std::uint64_t dimensions[]{64, std::uint64_t(k / 64), std::uint64_t(rows)};
    const std::uint64_t strides[]{128, std::uint64_t(k) * 2};
    const std::uint32_t box[]{64, 1, std::uint32_t(tile_rows)}, steps[]{1, 1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<__nv_bfloat16*>(pointer), dimensions,
        strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("MoE shared TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return map;
}

template <class S, bool Gate>
Descriptors make_descriptors(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                             const __nv_bfloat16* control, int tokens) {
    Descriptors maps{};
    maps.weight = make_map(weight, Gate ? 2 * S::intermediate : kHidden, S::k,
                           Gate ? S::weight_rows / 2 : S::weight_rows);
    maps.input  = make_map(input, tokens, S::k, S::columns);
    if constexpr (S::planes == 2)
        maps.low = make_map(input + std::int64_t(tokens) * S::k, tokens, S::k, S::columns);
    if constexpr (Gate)
        if (control) maps.control = make_map(control, kExperts + 1, S::k, S::weight_rows / 2);
    return maps;
}

template <bool Gate, int Columns, int Intermediate, bool SplitActivation, int RowWarps, int Stages,
          int Blocks, class Output>
__global__ __launch_bounds__(
    SharedSchedule<Gate, Columns, Intermediate, SplitActivation, RowWarps, Stages, Blocks>::threads,
    Blocks) void shared_mma(const __grid_constant__ Descriptors descriptors, Storage storage,
                            const float* coefficients, const float* beta, Output output,
                            float* scores, float* shared_gate, int tokens, int control_rows) {
    using S =
        SharedSchedule<Gate, Columns, Intermediate, SplitActivation, RowWarps, Stages, Blocks>;
    extern __shared__ __align__(1024) unsigned char scratch[];
    auto* ws               = reinterpret_cast<__nv_bfloat16*>(scratch);
    auto* xs               = ws + Stages * S::weight_rows * 64;
    auto* full             = reinterpret_cast<std::uint64_t*>(scratch + S::tensor_bytes);
    auto* empty            = full + Stages;
    const int route_blocks = Gate ? (control_rows + S::weight_rows - 1) / S::weight_rows : 0;
    const bool route       = Gate && blockIdx.x < route_blocks;
    const int row0   = route ? (control_rows == 1 ? kExperts : int(blockIdx.x) * S::weight_rows)
                             : (int(blockIdx.x) - route_blocks) * S::rows;
    const int token0 = blockIdx.y * Columns;
    if (!threadIdx.x) {
#pragma unroll
        for (int s = 0; s < Stages; ++s) {
            cta_mbarrier_init(full + s, 1);
            cta_mbarrier_init(empty + s, S::consumers);
        }
        cta_mbarrier_fence_init();
    }
    if constexpr (Gate)
        if (control_rows && storage.groups.counts && blockIdx.x == 0 && blockIdx.y == 0)
            for (int e = threadIdx.x; e < kExperts; e += S::threads) storage.groups.counts[e] = 0;
    __syncthreads();
    float acc[2][S::atoms][4]{};
    if (threadIdx.x < 32) {
        if (!threadIdx.x)
            for (int kt = 0; kt < S::k / 64; ++kt) {
                const int s = kt % Stages;
                cta_mbarrier_wait(empty + s, 1U ^ ((kt / Stages) & 1U));
                cta_mbarrier_arrive_expect_tx(full + s,
                                              (S::weight_rows + S::planes * Columns) * 64 * 2);
                if constexpr (Gate) {
                    const auto* map = route ? &descriptors.control : &descriptors.weight;
                    tma_load_3d(ws + s * S::weight_rows * 64, map, 0, kt, row0, full + s);
                    tma_load_3d(ws + s * S::weight_rows * 64 + (S::weight_rows / 2) * 64, map, 0,
                                kt, row0 + (route ? S::weight_rows / 2 : Intermediate), full + s);
                } else
                    tma_load_3d(ws + s * S::weight_rows * 64, &descriptors.weight, 0, kt, row0,
                                full + s);
                tma_load_3d(xs + s * S::planes * Columns * 64, &descriptors.input, 0, kt, token0,
                            full + s);
                if constexpr (S::planes == 2)
                    tma_load_3d(xs + (s * S::planes + 1) * Columns * 64, &descriptors.low, 0, kt,
                                token0, full + s);
            }
    }
    if (threadIdx.x >= 32) {
        const int tid = threadIdx.x - 32, lane = tid & 31, warp = tid >> 5, wm = warp % RowWarps,
                  wn = warp / RowWarps;
        for (int kt = 0; kt < S::k / 64; ++kt) {
            const int s = kt % Stages;
            cta_mbarrier_wait(full + s, (kt / Stages) & 1U);
#pragma unroll
            for (int ki = 0; ki < 4; ++ki) {
                unsigned a[2][4];
#pragma unroll
                for (int m = 0; m < 2; ++m) {
                    const int row = (Gate ? wm * 16 + m * (S::weight_rows / 2) : wm * 32 + m * 16) +
                                    (lane & 7) + ((lane >> 3 & 1) * 8);
                    const int k = ki * 16 + (lane >> 4) * 8;
                    ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                                smem_addr(ws + s * S::weight_rows * 64 + row * 64 +
                                          swizzle_xor_64(row, k)));
                }
#pragma unroll
                for (int n = 0; n < S::atoms; ++n) {
                    const int row = wn * (Columns / 2) + n * 8 + (lane & 7),
                              k   = ki * 16 + ((lane >> 3) & 1) * 8;
#pragma unroll
                    for (int p = 0; p < S::planes; ++p) {
                        unsigned b[2];
                        ldmatrix_x2(b[0], b[1],
                                    smem_addr(xs + (s * S::planes + p) * Columns * 64 + row * 64 +
                                              swizzle_xor_64(row, k)));
#pragma unroll
                        for (int m = 0; m < 2; ++m)
                            mma_bf16(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3],
                                     a[m][0], a[m][1], a[m][2], a[m][3], b[0], b[1]);
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (!lane) cta_mbarrier_arrive(empty + s);
        }
    }
    // All producer and consumer lanes retire staging before output reuses it.
    __syncthreads();
    if (threadIdx.x >= 32) {
        const int tid = threadIdx.x - 32, lane = tid & 31, warp = tid >> 5, wm = warp % RowWarps,
                  wn = warp / RowWarps;
        auto* h      = static_cast<__nv_bfloat16*>(storage.shared_activation);
#pragma unroll
        for (int n = 0; n < S::atoms; ++n)
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int token = token0 + wn * (Columns / 2) + n * 8 + (lane & 3) * 2 + (v & 1);
                if (token >= tokens) continue;
                if constexpr (Gate) {
                    if (route) {
#pragma unroll
                        for (int m = 0; m < 2; ++m) {
                            const int row = row0 + wm * 16 + m * (S::weight_rows / 2) +
                                            (lane >> 2) + (v / 2) * 8;
                            if (row <= kExperts) {
                                if (scores)
                                    scores[token * kScoreRows + row] = acc[m][n][v];
                                else
                                    shared_gate[token] = storage.token_validity.contains(token)
                                                             ? 1.F / (1.F + __expf(-acc[m][n][v]))
                                                             : 0.F;
                            }
                        }
                    } else {
                        const int row                 = row0 + wm * 16 + (lane >> 2) + (v / 2) * 8;
                        const float value             = silu(acc[0][n][v]) * acc[1][n][v];
                        const auto hi                 = __float2bfloat16_rn(value);
                        h[token * Intermediate + row] = hi;
                        if constexpr (SplitActivation)
                            h[tokens * Intermediate + token * Intermediate + row] =
                                __float2bfloat16_rn(value - __bfloat162float(hi));
                    }
                } else {
#pragma unroll
                    for (int m = 0; m < 2; ++m) {
                        const int row         = row0 + wm * 32 + m * 16 + (lane >> 2) + (v / 2) * 8;
                        const int local_token = token - token0, local_row = row - row0;
                        reinterpret_cast<float*>(scratch)[local_token * S::rows +
                                                          (local_row ^ ((local_token & 7) << 3))] =
                            acc[m][n][v];
                    }
                }
            }
    }
    if constexpr (!Gate) {
        __syncthreads();
        finish_output<S::rows, Columns, S::threads>(reinterpret_cast<float*>(scratch), row0, token0,
                                                    tokens, storage, coefficients, beta, output);
    }
}

template <bool Gate, int Intermediate, bool SplitActivation, int Columns, int RowWarps, int Stages,
          int Blocks, class Output>
void launch(const __nv_bfloat16* input, const __nv_bfloat16* weight, const __nv_bfloat16* control,
            const Storage& storage, const float* coefficients, const float* beta, Output output,
            float* scores, float* shared_gate, int tokens, CudaExecutionView execution) {
    using S =
        SharedSchedule<Gate, Columns, Intermediate, SplitActivation, RowWarps, Stages, Blocks>;
    constexpr auto kernel =
        shared_mma<Gate, Columns, Intermediate, SplitActivation, RowWarps, Stages, Blocks, Output>;
    const int bytes        = common::prepare_shared<S::bytes, kernel, true>();
    const auto descriptors = make_descriptors<S, Gate>(input, weight, control, tokens);
    const int control_rows = control ? (scores ? kExperts + 1 : 1) : 0;
    const int row_tiles =
        Gate ? Intermediate / S::rows + (control_rows + S::weight_rows - 1) / S::weight_rows
             : kHidden / S::rows;
    kernel<<<dim3(row_tiles, (tokens + Columns - 1) / Columns), S::threads, bytes,
             execution.stream()>>>(descriptors, storage, coefficients, beta, output, scores,
                                   shared_gate, tokens, control_rows);
    CUDA_CHECK(cudaGetLastError());
}

template <int Intermediate, bool SplitActivation>
void launch_gate(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                 const __nv_bfloat16* control, const Storage& storage, float* scores,
                 float* shared_gate, int tokens, int columns, CudaExecutionView execution) {
    const auto submit = [&]<int Columns, int RowWarps, int Stages, int Blocks>() {
        launch<true, Intermediate, SplitActivation, Columns, RowWarps, Stages, Blocks>(
            input, weight, control, storage, nullptr, nullptr, FloatContribution{}, scores,
            shared_gate, tokens, execution);
    };
    switch (columns) {
    case 32:
        submit.template operator()<32, 2, 3, 2>();
        break;
    case 64:
        submit.template operator()<64, 2, 2, 2>();
        break;
    case 128:
        submit.template operator()<128, 4, 2, 1>();
        break;
    default:
        throw std::logic_error("invalid MoE shared TMA column tile");
    }
}

template <int Intermediate, bool SplitActivation, class Output>
void launch_down(const __nv_bfloat16* weight, const Storage& storage, const float* coefficients,
                 const float* beta, Output output, int tokens, int columns,
                 CudaExecutionView execution) {
    const auto submit = [&]<int Columns, int RowWarps, int Stages, int Blocks>() {
        launch<false, Intermediate, SplitActivation, Columns, RowWarps, Stages, Blocks>(
            static_cast<const __nv_bfloat16*>(storage.shared_activation), weight, nullptr, storage,
            coefficients, beta, output, nullptr, nullptr, tokens, execution);
    };
    switch (columns) {
    case 32:
        submit.template operator()<32, 2, 3, 2>();
        break;
    case 64:
        submit.template operator()<64, 2, 2, 2>();
        break;
    case 128:
        if constexpr (Intermediate == 320)
            submit.template operator()<64, 2, 2, 2>();
        else
            submit.template operator()<64, 4, 1, 3>();
        break;
    default:
        throw std::logic_error("invalid MoE shared TMA column tile");
    }
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::spark_shared
