#pragma once
#include "ops/common/cuda/mma.cuh"
#include "ops/common/host/launch.h"
#include "ops/linear_attention/video_delta_attention/launch.h"
#include "ops/common/cuda/pipeline.cuh"

namespace ninfer::ops::detail::vda::scan_internal {
using BF                   = __nv_bfloat16;
constexpr int MatrixStride = 136;

template <int Values>
constexpr int Threads = Values >= 32 ? 256 : 128;
template <int Values>
constexpr int StateStride = Values == 8 ? 8 : Values + 8;

template <int Values, int Inner, int Stages>
struct Shared {
    struct Stage {
        BF high[Inner * MatrixStride], low[Inner * MatrixStride];
    } stage[Stages];

    BF state_high[128 * StateStride<Values>], state_low[128 * StateStride<Values>];
};

template <int Values, int Inner, int Stages>
__device__ void load_stage(Shared<Values, Inner, Stages>& shared, const BF* high, const BF* low,
                           int slot, int tile, int tid) {
    for (int i = tid * 8; i < Inner * 128; i += Threads<Values> * 8) {
        const int offset = i / 128 * MatrixStride + i % 128;
        common::cp_async<16, Cache::ca>(shared.stage[slot].high + offset, high + tile * 128 + i);
        common::cp_async<16, Cache::ca>(shared.stage[slot].low + offset, low + tile * 128 + i);
    }
    common::cp_commit();
}

__device__ void split_pair(float x, float y, BF* high, BF* low) {
    const auto h = __floats2bfloat162_rn(x, y);
    const auto l = __floats2bfloat162_rn(x - __low2float(h), y - __high2float(h));
    common::store_vec(high, h);
    common::store_vec(low, l);
}

template <int Values, int Inner, int Stages>
__global__ __launch_bounds__(Threads<Values>,
                             2) void boundary_scan(Arguments args, const float* transition,
                                                   const float* injection, float* boundaries,
                                                   int head_begin, int wave_heads, int batch,
                                                   int chunks) {
    extern __shared__ __align__(32) unsigned char storage[];
    auto& shared        = *reinterpret_cast<Shared<Values, Inner, Stages>*>(storage);
    constexpr int Warps = Threads<Values> / 32, RowTiles = 8 / Warps, ColTiles = Values / 8;
    const int head = blockIdx.x, direction = blockIdx.y, value_begin = blockIdx.z * Values;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, group = lane / 4,
              pair = (lane % 4) * 2;
    const auto initial =
        args.initial + (std::int64_t(batch) * args.heads + head_begin + head) * 128 * 128;
    float state[RowTiles][ColTiles][4];
#pragma unroll
    for (int m = 0; m < RowTiles; ++m) {
        const int key = (warp + m * Warps) * 16 + group;
#pragma unroll
        for (int n = 0; n < ColTiles; ++n) {
            const int value = value_begin + n * 8 + pair;
            state[m][n][0]  = initial[value * 128 + key];
            state[m][n][1]  = initial[(value + 1) * 128 + key];
            state[m][n][2]  = initial[value * 128 + key + 8];
            state[m][n][3]  = initial[(value + 1) * 128 + key + 8];
        }
    }
    for (int step = 0; step < args.frames; ++step) {
        const int frame   = direction ? args.frames - 1 - step : step;
        const auto matrix = (std::int64_t(frame) * wave_heads + head) * 128 * 128;
        const auto* high  = reinterpret_cast<const BF*>(transition + matrix);
        const auto* low   = high + 128 * 128;
        load_stage(shared, high, low, 0, 0, tid);
        if constexpr (Stages == 2) load_stage(shared, high, low, 1, Inner, tid);
#pragma unroll
        for (int m = 0; m < RowTiles; ++m) {
            const int key = (warp + m * Warps) * 16 + group;
#pragma unroll
            for (int n = 0; n < ColTiles; ++n) {
                const int value = n * 8 + pair;
                const int a     = key * StateStride<Values> + value,
                          b     = (key + 8) * StateStride<Values> + value;
                split_pair(state[m][n][0], state[m][n][1], shared.state_high + a,
                           shared.state_low + a);
                split_pair(state[m][n][2], state[m][n][3], shared.state_high + b,
                           shared.state_low + b);
                const auto offset = matrix + (value_begin + value) * 128 + key;
                state[m][n][0]    = injection[offset];
                state[m][n][1]    = injection[offset + 128];
                state[m][n][2]    = injection[offset + 8];
                state[m][n][3]    = injection[offset + 136];
            }
        }
        for (int tile = 0; tile < 128 / Inner; ++tile) {
            if constexpr (Stages == 2) {
                if (tile + 1 < 128 / Inner)
                    common::cp_wait<1>();
                else
                    common::cp_wait<0>();
            } else
                common::cp_wait<0>();
            __syncthreads();
            const auto& staged = shared.stage[tile % Stages];
#pragma unroll
            for (int k = 0; k < Inner; k += 16) {
                unsigned bh[ColTiles][2], bl[ColTiles][2];
#pragma unroll
                for (int n = 0; n < ColTiles; ++n) {
                    const int offset =
                        (tile * Inner + k + (lane & 15)) * StateStride<Values> + n * 8;
                    common::ldmatrix_x2_t(bh[n][0], bh[n][1],
                                          common::smem_addr(shared.state_high + offset));
                    common::ldmatrix_x2_t(bl[n][0], bl[n][1],
                                          common::smem_addr(shared.state_low + offset));
                }
#pragma unroll
                for (int m = 0; m < RowTiles; ++m) {
                    const int offset = (k + (lane & 7) + (lane >> 4) * 8) * MatrixStride +
                                       (warp + m * Warps) * 16 + ((lane >> 3) & 1) * 8;
                    unsigned ah[4], al[4];
                    common::ldmatrix_x4_t(ah[0], ah[1], ah[2], ah[3],
                                          common::smem_addr(staged.high + offset));
                    common::ldmatrix_x4_t(al[0], al[1], al[2], al[3],
                                          common::smem_addr(staged.low + offset));
#pragma unroll
                    for (int n = 0; n < ColTiles; ++n) {
                        auto& c = state[m][n];
                        common::mma_bf16(c[0], c[1], c[2], c[3], ah[0], ah[1], ah[2], ah[3],
                                         bh[n][0], bh[n][1]);
                        common::mma_bf16(c[0], c[1], c[2], c[3], ah[0], ah[1], ah[2], ah[3],
                                         bl[n][0], bl[n][1]);
                        common::mma_bf16(c[0], c[1], c[2], c[3], al[0], al[1], al[2], al[3],
                                         bh[n][0], bh[n][1]);
                    }
                }
            }
            __syncthreads();
            if (tile + Stages < 128 / Inner)
                load_stage(shared, high, low, tile % Stages, (tile + Stages) * Inner, tid);
        }
        const auto position = std::int64_t(frame) + args.window.frame_offset;
        const int chunk     = position / args.window.chunk_frames;
        const bool write    = direction ? (frame == 0 || position % args.window.chunk_frames == 0)
                                        : (frame == args.frames - 1 ||
                                        (position + 1) % args.window.chunk_frames == 0);
        if (write) {
            auto* output =
                boundaries +
                ((std::int64_t(direction) * chunks + chunk) * wave_heads + head) * 128 * 128;
#pragma unroll
            for (int m = 0; m < RowTiles; ++m) {
                const int key = (warp + m * Warps) * 16 + group;
#pragma unroll
                for (int n = 0; n < ColTiles; ++n) {
                    const int value                     = value_begin + n * 8 + pair;
                    output[value * 128 + key]           = state[m][n][0];
                    output[(value + 1) * 128 + key]     = state[m][n][1];
                    output[value * 128 + key + 8]       = state[m][n][2];
                    output[(value + 1) * 128 + key + 8] = state[m][n][3];
                }
            }
        }
    }
}

template <int Values, int Inner, int Stages>
void launch(const Arguments& args, const float* p, const float* j, float* boundaries,
            int head_begin, int wave_heads, int batch, int chunks, cudaStream_t stream) {
    constexpr int bytes     = sizeof(Shared<Values, Inner, Stages>);
    constexpr auto selected = boundary_scan<Values, Inner, Stages>;
    common::prepare_shared<bytes, selected, true>();
    selected<<<dim3(wave_heads, 2, 128 / Values), Threads<Values>, bytes, stream>>>(
        args, p, j, boundaries, head_begin, wave_heads, batch, chunks);
}
} // namespace ninfer::ops::detail::vda::scan_internal
