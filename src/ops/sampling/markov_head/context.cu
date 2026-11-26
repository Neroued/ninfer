#include "launch.h"

#include "core/pdl.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"

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
    const auto cols     = p.columns();
    const int row       = blockIdx.x * kRows;
    const bool is_frnn  = static_cast<int>(blockIdx.z) >= p.hidden_tiles;
    const int z_tile    = is_frnn ? static_cast<int>(blockIdx.z) - p.hidden_tiles : blockIdx.z;
    const int z_stride  = is_frnn ? static_cast<int>(gridDim.z) - p.hidden_tiles : p.hidden_tiles;
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
        for (std::int64_t s = step_lane; s < p.steps; s += kStepLanes) {
            const auto* src                          = p.partial + (b * p.steps + s) * p.rows + c;
            p.output[(s * p.batch + b) * p.rows + c] = base + reduce(src);
        }
    }
}

void submit(Problem p, cudaStream_t stream) {
    const auto cols = std::uint64_t(p.batch) * p.steps;
    const auto launch_partial = [&]<int Columns>() {
        constexpr int kBaseColumns = Columns <= 32 ? Columns : 32;
        const int hidden_tiles =
            static_cast<int>(std::min<std::uint64_t>(32767, 1 + (cols - 1) / Columns));
        const int frnn_tiles =
            std::min(32767, 1 + (p.batch - 1) / kBaseColumns);
        p.hidden_tiles = hidden_tiles;
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

    const dim3 finish(p.rows / 8, std::min(p.batch, 65535));
    cudaLaunchAttribute dependency{};
    dependency.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
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
