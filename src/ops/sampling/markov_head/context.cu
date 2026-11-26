#include "launch.h"

#include "core/pdl.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/reduce.cuh"

#include <algorithm>
#include <cuda_bf16.h>

namespace ninfer::ops::markov {
namespace {
using Bf16             = __nv_bfloat16;
constexpr int kRows    = 16;
constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kStageK  = 256;

struct Problem {
    const Bf16* hidden;
    const Bf16* frnn;
    const Bf16* w_hidden;
    const Bf16* w_frnn;
    const float* bias;
    float* partial;
    float* output;
    int h, rows, batch, steps, splits, hidden_tiles;

    __device__ __forceinline__ std::int64_t columns() const { return std::int64_t(batch) * steps; }

    __device__ __forceinline__ std::int64_t plane() const { return (columns() + batch) * rows; }
};

template <int Columns>
__device__ __forceinline__ void contract(const Problem& p, const Bf16* x, const Bf16* w,
                                         std::int64_t columns, std::int64_t tile_start,
                                         std::int64_t tile_stride, int row, int k0, float* out,
                                         Bf16* weights, Bf16* x_tile, float* reduction) {
    constexpr int kMmas     = Columns / 8;
    constexpr int kWarpK    = kStageK / kWarps;
    constexpr int kVectors  = kStageK / 8;
    constexpr int kFragment = 4 * 32;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int depth = min(kContextKTile, p.h - k0);
    const int a_row = lane & 15, a_k = (lane >> 4) * 8 + warp * kWarpK;
    const int b_row = lane & 7, b_k = ((lane & 8) ? 8 : 0) + warp * kWarpK;
    const auto a_base = common::smem_addr(weights + a_row * kContextKTile);
    const auto b_base = common::smem_addr(x_tile + b_row * kStageK);

    // A CTA owns one weight panel across every column tile assigned to its z-slice.
#pragma unroll
    for (int i = tid; i < kRows * (kContextKTile / 8); i += kThreads) {
        const int r = i / (kContextKTile / 8), k = (i % (kContextKTile / 8)) * 8;
        const bool valid = k < depth;
        common::cp_async_zfill<16, Cache::cg>(
            weights + r * kContextKTile + common::swizzle_xor_64(r, k),
            w + std::int64_t(row + r) * p.h + (valid ? k0 + k : 0), valid ? 16 : 0);
    }
    common::cp_commit();

    const auto stage_x = [&](std::int64_t col0, int k) {
#pragma unroll
        for (int i = tid; i < Columns * kVectors; i += kThreads) {
            const int r = i / kVectors, c = (i % kVectors) * 8;
            const bool valid = col0 + r < columns && k + c < depth;
            common::cp_async_zfill<16, Cache::cg>(
                x_tile + r * kStageK + common::swizzle_xor_64(r, c),
                x + (valid ? (col0 + r) * p.h + k0 + k + c : 0), valid ? 16 : 0);
        }
        common::cp_commit();
    };

    for (std::int64_t col0 = tile_start * Columns; col0 < columns; col0 += tile_stride * Columns) {
        float sum[kMmas][4]{};
        for (int k = 0; k < depth; k += kStageK) {
            stage_x(col0, k);
            common::cp_wait<0>();
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < kWarpK; kk += 16) {
                unsigned a0, a1, a2, a3;
                common::ldmatrix_x4(a0, a1, a2, a3,
                                    a_base + 2 * common::swizzle_xor_64(a_row, k + a_k + kk));
#pragma unroll
                for (int n = 0; n < kMmas; ++n) {
                    unsigned b0, b1;
                    common::ldmatrix_x2(b0, b1,
                                        b_base + 2 * common::swizzle_xor_64(b_row, b_k + kk) +
                                            n * 8 * kStageK * sizeof(Bf16));
                    auto& c = sum[n];
                    common::mma_bf16(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b0, b1);
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int n = 0; n < kMmas; ++n)
            common::store_vec(reduction + (n * kWarps + warp) * kFragment + lane * 4,
                              make_float4(sum[n][0], sum[n][1], sum[n][2], sum[n][3]));
        __syncthreads();
        for (int i = tid; i < min(std::int64_t(Columns), columns - col0) * kRows; i += kThreads) {
            const int col = i / kRows, r = i % kRows;
            const int fragment_lane = (r % 8) * 4 + col % 8 / 2;
            const int component     = (r / 8) * 2 + col % 2;
            const auto* at =
                reduction + (col / 8) * kWarps * kFragment + fragment_lane * 4 + component;
            float value = 0;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) value += at[w * kFragment];
            out[(col0 + col) * p.rows + row + r] = value;
        }
        __syncthreads();
    }
}

template <int Columns>
__global__ void context_partial(const __grid_constant__ Problem p) {
    constexpr int kBaseColumns     = Columns <= 32 ? Columns : 32;
    constexpr int kActivationBytes = Columns * kStageK * sizeof(Bf16);
    constexpr int kReductionBytes  = Columns * kRows * kWarps * sizeof(float);
    __shared__ __align__(16) Bf16 weights[kRows * kContextKTile];
    // Warp reduction starts after the activation tile is retired.
    __shared__ __align__(
        16) char storage[kActivationBytes > kReductionBytes ? kActivationBytes : kReductionBytes];
    auto* act       = reinterpret_cast<Bf16*>(storage);
    auto* reduction = reinterpret_cast<float*>(storage);
    pdl::trigger_dependents();
    const auto cols    = p.columns();
    const int row      = blockIdx.x * kRows;
    const bool is_frnn = static_cast<int>(blockIdx.z) >= p.hidden_tiles;
    const int z_tile   = is_frnn ? static_cast<int>(blockIdx.z) - p.hidden_tiles : blockIdx.z;
    const int z_stride = is_frnn ? static_cast<int>(gridDim.z) - p.hidden_tiles : p.hidden_tiles;
    for (int split = blockIdx.y; split < p.splits; split += gridDim.y) {
        auto* out    = p.partial + split * p.plane();
        const int k0 = split * kContextKTile;
        if (!is_frnn)
            contract<Columns>(p, p.hidden, p.w_hidden, cols, z_tile, z_stride, row, k0, out,
                              weights, act, reduction);
        else
            contract<kBaseColumns>(p, p.frnn, p.w_frnn, p.batch, z_tile, z_stride, row, k0,
                                   out + cols * p.rows, weights, act, reduction);
        if (split + gridDim.y < p.splits) __syncthreads();
    }
}

constexpr int kFinishStepTile = 16;

template <int Splits>
__global__ void context_finish(const __grid_constant__ Problem p) {
    constexpr int kChannels = 8, kStepLanes = 32 / kChannels;
    const int lane      = threadIdx.x;
    const int step_lane = lane / kChannels;
    const int c         = blockIdx.x * kChannels + lane % kChannels;
    const int splits    = Splits ? Splits : p.splits;
    pdl::wait_for_dependencies();
    for (std::int64_t b = blockIdx.y; b < p.batch; b += gridDim.y) {
        const auto reduce = [&](const float* src) {
            float value = 0.0f;
#pragma unroll
            for (int i = 0; i < splits; ++i) value += src[i * p.plane()];
            return value;
        };
        float base = 0.0f;
        if (step_lane == 0) base = reduce(p.partial + (p.columns() + b) * p.rows + c) + p.bias[c];
        base = __shfl_sync(0xffffffff, base, lane % kChannels);
        for (std::int64_t first = blockIdx.z * kFinishStepTile; first < p.steps;
             first += gridDim.z * kFinishStepTile) {
            const auto end = min(first + kFinishStepTile, std::int64_t(p.steps));
            for (std::int64_t s = first + step_lane; s < end; s += kStepLanes) {
                const auto* src = p.partial + (b * p.steps + s) * p.rows + c;
                p.output[(s * p.batch + b) * p.rows + c] = base + reduce(src);
            }
        }
    }
}

constexpr int kDenseK       = 64;
constexpr int kDenseRows    = 32;
constexpr int kDenseColumns = 32;

__global__ void context_base(const __grid_constant__ Problem p) {
    constexpr int warps = 4;
    const int lane      = threadIdx.x % 32;
    const int row       = blockIdx.x * warps + threadIdx.x / 32;
    const auto* w       = p.w_frnn + std::int64_t(row) * p.h;
    for (std::int64_t b = blockIdx.y; b < p.batch; b += gridDim.y) {
        const auto* x = p.frnn + b * p.h;
        float sum     = 0;
        for (std::int64_t k = lane * 8; k < p.h; k += 256) {
            const auto a        = common::load_vec<uint4>(x + k);
            const auto z        = common::load_vec<uint4>(w + k);
            const auto dot_pair = [](unsigned lhs, unsigned rhs, float acc) {
                return fmaf(__uint_as_float(lhs & 0xffff0000U), __uint_as_float(rhs & 0xffff0000U),
                            fmaf(__uint_as_float(lhs << 16), __uint_as_float(rhs << 16), acc));
            };
            sum = dot_pair(a.x, z.x, sum);
            sum = dot_pair(a.y, z.y, sum);
            sum = dot_pair(a.z, z.z, sum);
            sum = dot_pair(a.w, z.w, sum);
        }
        sum = common::warp_reduce_sum(sum);
        if (lane == 0) p.partial[b * p.rows + row] = p.bias[row] + sum;
    }
}

__global__ void context_dense(const __grid_constant__ Problem p) {
    __shared__ __align__(16) Bf16 weights[kDenseRows * kDenseK];
    __shared__ __align__(16) Bf16 activations[kDenseColumns * kDenseK];
    constexpr int vectors = kDenseK / 8;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int row0  = blockIdx.x * kDenseRows;
    const int a_row = (warp % 2) * 16 + (lane & 15), a_k = (lane >> 4) * 8;
    const int x_row = (warp / 2) * 8 + (lane & 7), x_k = (lane & 8) ? 8 : 0;
    const auto a_base = common::smem_addr(weights + a_row * kDenseK);
    const auto x_base = common::smem_addr(activations + x_row * kDenseK);
    for (std::int64_t col0 = std::int64_t(blockIdx.y) * kDenseColumns; col0 < p.columns();
         col0 += std::int64_t(gridDim.y) * kDenseColumns) {
        float4 sum{};
        for (std::int64_t k0 = 0; k0 < p.h; k0 += kDenseK) {
            const int r = tid / vectors, k = (tid % vectors) * 8;
            const auto col         = col0 + r;
            const bool weight_live = row0 + r < p.rows && k0 + k < p.h;
            const bool input_live  = col < p.columns() && k0 + k < p.h;
            const int swizzle      = common::swizzle_xor_64(r, k);
            common::cp_async_zfill<16, Cache::cg>(
                weights + r * kDenseK + swizzle,
                p.w_hidden + (weight_live ? std::int64_t(row0 + r) * p.h + k0 + k : 0),
                weight_live ? 16 : 0);
            common::cp_async_zfill<16, Cache::cg>(activations + r * kDenseK + swizzle,
                                                  p.hidden + (input_live ? col * p.h + k0 + k : 0),
                                                  input_live ? 16 : 0);
            common::cp_commit();
            common::cp_wait<0>();
            __syncthreads();
#pragma unroll
            for (int k_mma = 0; k_mma < kDenseK; k_mma += 16) {
                unsigned a0, a1, a2, a3, b0, b1;
                common::ldmatrix_x4(a0, a1, a2, a3,
                                    a_base + 2 * common::swizzle_xor_64(a_row, k_mma + a_k));
                common::ldmatrix_x2(b0, b1,
                                    x_base + 2 * common::swizzle_xor_64(x_row, k_mma + x_k));
                common::mma_bf16(sum.x, sum.y, sum.z, sum.w, a0, a1, a2, a3, b0, b1);
            }
            __syncthreads();
        }
        const auto quad = common::mma_row_quad(sum);
        const int row   = row0 + (warp % 2) * 16 + quad.row;
        const auto col  = col0 + (warp / 2) * 8 + quad.column;
        if (row < p.rows && col < p.columns()) {
            const auto b    = p.batch == 1 ? 0 : col / p.steps;
            const auto s    = col - b * p.steps;
            const auto base = common::load_vec<float4>(p.partial + b * p.rows + row);
            common::store_vec(p.output + (s * p.batch + b) * p.rows + row,
                              make_float4(quad.values.x + base.x, quad.values.y + base.y,
                                          quad.values.z + base.z, quad.values.w + base.w));
        }
    }
}

void submit(Problem p, cudaStream_t stream) {
    const auto cols = std::uint64_t(p.batch) * p.steps;
    if (cols >= kContextLargeColumns) {
        const dim3 base(p.rows / 4, std::min(p.batch, 65535));
        context_base<<<base, 128, 0, stream>>>(p);
        const dim3 grid(1 + (p.rows - 1) / kDenseRows,
                        std::min<std::uint64_t>(65535, 1 + (cols - 1) / kDenseColumns));
        context_dense<<<grid, kThreads, 0, stream>>>(p);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const auto launch_partial = [&]<int Columns>() {
        constexpr int kBaseColumns = Columns <= 32 ? Columns : 32;
        const int hidden_tiles =
            static_cast<int>(std::min<std::uint64_t>(32767, 1 + (cols - 1) / Columns));
        const int frnn_tiles = std::min(32767, 1 + (p.batch - 1) / kBaseColumns);
        p.hidden_tiles       = hidden_tiles;
        const dim3 grid(p.rows / kRows, std::min(p.splits, 65535), hidden_tiles + frnn_tiles);
        context_partial<Columns><<<grid, kThreads, 0, stream>>>(p);
    };
    if (cols <= 8)
        launch_partial.template operator()<8>();
    else if (cols <= 16)
        launch_partial.template operator()<16>();
    else if (cols <= 32)
        launch_partial.template operator()<32>();
    else
        launch_partial.template operator()<64>();
    CUDA_CHECK(cudaGetLastError());

    const dim3 finish(p.rows / 8, std::min(p.batch, 65535),
                      std::min(65535, 1 + (p.steps - 1) / kFinishStepTile));
    cudaLaunchAttribute dependency{};
    dependency.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    dependency.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t config{};
    config.gridDim  = finish;
    config.blockDim = dim3(32);
    config.stream   = stream;
    config.attrs    = &dependency;
    config.numAttrs = 1;
    if (p.h == 2304)
        CUDA_CHECK(cudaLaunchKernelEx(&config, context_finish<1 + (2304 - 1) / kContextKTile>, p));
    else
        CUDA_CHECK(cudaLaunchKernelEx(&config, context_finish<0>, p));
}
} // namespace

void launch_context(const Tensor& hidden, const Tensor& frnn, const Weight& w_hidden,
                    const Weight& w_frnn, const Tensor& bias, float* partial, Tensor& context,
                    CudaExecutionView execution) {
    const Problem p{static_cast<const Bf16*>(hidden.data),
                    static_cast<const Bf16*>(frnn.data),
                    static_cast<const Bf16*>(w_hidden.qdata),
                    static_cast<const Bf16*>(w_frnn.qdata),
                    static_cast<const float*>(bias.data),
                    partial,
                    static_cast<float*>(context.data),
                    hidden.ne[0],
                    context.ne[0],
                    frnn.ne[1],
                    hidden.ne[1],
                    1 + (hidden.ne[0] - 1) / kContextKTile,
                    1};
    submit(p, execution.stream());
}
} // namespace ninfer::ops::markov
