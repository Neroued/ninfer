#include "ops/dynamic_grouped_conv/prepare/launch.h"
#include "core/device.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/mma.cuh"

#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail::dynamic_grouped_conv::prepare {
namespace {
constexpr int kBlockK = 64, kStages = 2;

template <int Rows, int Columns>
struct Tile {
    static constexpr int column_warps = Rows == 16 ? Columns / 8 : 4;
    static constexpr int threads      = Rows / 16 * column_warps * 32;
    static constexpr int mmas         = Columns / column_warps / 8;
};

template <int Rows, int Columns, int SplitK>
__launch_bounds__(Tile<Rows, Columns>::threads, 1) __global__
    void partial_kernel(const __nv_bfloat16* __restrict__ input,
                        const __nv_bfloat16* __restrict__ weight, float* __restrict__ partial,
                        int tokens) {
    constexpr int threads         = Tile<Rows, Columns>::threads;
    constexpr int mmas            = Tile<Rows, Columns>::mmas;
    constexpr int tiles_per_split = kHidden / kBlockK / SplitK;
    __shared__ __align__(16) __nv_bfloat16 ws[kStages][Rows * kBlockK];
    __shared__ __align__(16) __nv_bfloat16 xs[kStages][Columns * kBlockK];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, gid = lane / 4, lid = lane % 4;
    const int row0 = blockIdx.x * Rows, col0 = blockIdx.y * Columns, split = blockIdx.z;
    const int warp_row = warp / Tile<Rows, Columns>::column_warps;
    const int warp_col = warp % Tile<Rows, Columns>::column_warps;
    const int kt_begin = split * tiles_per_split;
    float acc[mmas][4] = {};
    auto stage_inputs  = [&](int stage, int kt) {
        for (int item = tid; item < Rows * (kBlockK / 8); item += threads) {
            int row = item / (kBlockK / 8), kk = item % (kBlockK / 8) * 8;
            common::cp_async<16, Cache::cg>(
                &ws[stage][row * kBlockK + common::swizzle_xor_64(row, kk)],
                weight + static_cast<std::int64_t>(row0 + row) * kHidden + kt * kBlockK + kk);
        }
        for (int item = tid; item < Columns * (kBlockK / 8); item += threads) {
            int col = item / (kBlockK / 8), kk = item % (kBlockK / 8) * 8;
            const int safe_col = col0 + col < tokens ? col0 + col : 0;
            common::cp_async_zfill<16, Cache::cg>(
                &xs[stage][col * kBlockK + common::swizzle_xor_64(col, kk)],
                input + static_cast<std::int64_t>(safe_col) * kHidden + kt * kBlockK + kk,
                col0 + col < tokens ? 16 : 0);
        }
    };
    for (int stage = 0; stage < kStages; ++stage) {
        stage_inputs(stage, kt_begin + stage);
        common::cp_commit();
    }
#pragma unroll 1
    for (int tile = 0; tile < tiles_per_split; ++tile) {
        const int stage = tile % kStages;
        if (tile + kStages <= tiles_per_split) {
            common::cp_wait<kStages - 1>();
        } else {
            common::cp_wait<0>();
        }
        __syncthreads();
#pragma unroll
        for (int ki = 0; ki < kBlockK / 16; ++ki) {
            unsigned a[4];
            const int ar = warp_row * 16 + (lane & 7) + (((lane >> 3) & 1) * 8);
            const int ak = ki * 16 + (lane >> 4) * 8;
            common::ldmatrix_x4(
                a[0], a[1], a[2], a[3],
                common::smem_addr(&ws[stage][ar * kBlockK + common::swizzle_xor_64(ar, ak)]));
#pragma unroll
            for (int m = 0; m < mmas; ++m) {
                unsigned b[2];
                const int br = (warp_col * mmas + m) * 8 + (lane & 7);
                const int bk = ki * 16 + ((lane >> 3) & 1) * 8;
                common::ldmatrix_x2(
                    b[0], b[1],
                    common::smem_addr(&xs[stage][br * kBlockK + common::swizzle_xor_64(br, bk)]));
                common::mma_bf16(acc[m][0], acc[m][1], acc[m][2], acc[m][3], a[0], a[1], a[2], a[3],
                                 b[0], b[1]);
            }
        }
        __syncthreads();
        if (tile + kStages < tiles_per_split) stage_inputs(stage, kt_begin + tile + kStages);
        common::cp_commit();
    }
#pragma unroll
    for (int m = 0; m < mmas; ++m) {
        const int row = row0 + warp_row * 16 + gid,
                  col = col0 + (warp_col * mmas + m) * 8 + 2 * lid;
        auto put      = [&](int r, int c, float value) {
            if (c < tokens)
                partial[(static_cast<std::int64_t>(split) * tokens + c) * kCoefficientRows + r] =
                    value;
        };
        put(row, col, acc[m][0]);
        put(row, col + 1, acc[m][1]);
        put(row + 8, col, acc[m][2]);
        put(row + 8, col + 1, acc[m][3]);
    }
}

template <int R, int C, int S>
void launch_partial(const Tensor& input, const Weight& weight, float* partial,
                    cudaStream_t stream) {
    const int tokens = input.ne[1] * input.ne[2];
    const dim3 grid(kCoefficientRows / R, (tokens + C - 1) / C, S);
    partial_kernel<R, C, S><<<grid, Tile<R, C>::threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<const __nv_bfloat16*>(weight.qdata), partial, tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <int Capacity>
__launch_bounds__(Capacity * 16, 4) __global__
    void reduce_kernel(const __nv_bfloat16* base, const float* partial, __nv_bfloat16* prepared,
                       __nv_bfloat16* finish, int width, int batch_size, int splits) {
    __shared__ float projected[4][Capacity];
    __shared__ float normalized[Capacity][16];
    const int tid = threadIdx.x, group = blockIdx.x, batch = blockIdx.y;
    const int tokens = width * batch_size;
    if (tid < 4 * Capacity && tid % Capacity < width) {
        const int coefficient = tid / Capacity, position = tid % Capacity,
                  row = coefficient * kGroups + group;
        float sum     = 0;
        for (int split = 0; split < splits; ++split)
            sum += partial[(static_cast<std::int64_t>(split) * tokens + batch * width + position) *
                               kCoefficientRows +
                           row];
        projected[coefficient][position] = sum;
    }
    const int position = tid / 16, channel = tid % 16, hidden = group * 16 + channel;
    const std::int64_t offset =
        static_cast<std::int64_t>(batch * width + position) * kHidden + hidden;
    if (position < width) { normalized[position][channel] = __bfloat162float(prepared[offset]); }
    __syncthreads();
    if (tid < 2 * Capacity && tid % Capacity < width) {
        const int tap = tid / Capacity, pos = tid % Capacity;
        finish[((batch * width + pos) * 2 + tap) * kGroups + group] =
            __float2bfloat16_rn(projected[2 + tap][pos]);
    }
    if (position < width) {
        float value = (__bfloat162float(base[hidden]) + projected[0][position]) *
                      normalized[position][channel];
        if (position > 0)
            value = fmaf(__bfloat162float(base[kHidden + hidden]) + projected[1][position],
                         normalized[position - 1][channel], value);
        prepared[offset] = __float2bfloat16_rn(value);
    }
}

template <int Capacity>
void launch_reduce(Projection projection, const Tensor& base, const float* partial,
                   Tensor& prepared, Tensor& finish, cudaStream_t stream) {
    const dim3 grid(kGroups, prepared.ne[2]);
    reduce_kernel<Capacity><<<grid, Capacity * 16, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(base.data), partial,
        static_cast<__nv_bfloat16*>(prepared.data), static_cast<__nv_bfloat16*>(finish.data),
        prepared.ne[1], prepared.ne[2], projection.splits);
    CUDA_CHECK(cudaGetLastError());
}

void launch_projection(Projection projection, const Tensor& input, const Weight& weight,
                       float* partial, cudaStream_t stream) {
    if (projection.rows == 16 && projection.splits == 8) {
        switch (projection.columns) {
        case 8:
            return launch_partial<16, 8, 8>(input, weight, partial, stream);
        case 16:
            return launch_partial<16, 16, 8>(input, weight, partial, stream);
        case 32:
            return launch_partial<16, 32, 8>(input, weight, partial, stream);
        case 48:
            return launch_partial<16, 48, 8>(input, weight, partial, stream);
        }
    }
    if (projection.rows == 32 && projection.splits == 4) {
        if (projection.columns == 32)
            return launch_partial<32, 32, 4>(input, weight, partial, stream);
        if (projection.columns == 64)
            return launch_partial<32, 64, 4>(input, weight, partial, stream);
    }
    throw std::logic_error("dynamic grouped conv prepare: invalid production tile");
}
} // namespace

void launch(Plan plan, const Tensor& base, const Weight& projection, float* partials,
            Tensor& prepared, Tensor& finish, cudaStream_t stream) {
    launch_projection(plan.projection, prepared, projection, partials, stream);
    if (plan.reduction_capacity == 8)
        launch_reduce<8>(plan.projection, base, partials, prepared, finish, stream);
    else
        launch_reduce<16>(plan.projection, base, partials, prepared, finish, stream);
}
} // namespace ninfer::ops::detail::dynamic_grouped_conv::prepare
