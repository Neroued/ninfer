#pragma once
#include "ops/linear_attention/kimi_delta_attention/input_conv/launch.h"
#include "ops/linear/kernels/q8/q8_a16_tma.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma_k32.cuh"
#include "ops/common/numeric/math.cuh"

namespace ninfer::ops::detail::kda_input {
struct TiledConvolution {
    Arguments args;
    float* edges;
    __nv_bfloat16* tail;
};

__device__ inline void write_convolution(const Arguments& a, int channel, int column, float h0,
                                         float h1, float h2, float current) {
    float value = a.weights[channel] * h0;
    value       = fmaf(a.weights[kProjectionFeatures + channel], h1, value);
    value       = fmaf(a.weights[2 * kProjectionFeatures + channel], h2, value);
    value       = fmaf(a.weights[3 * kProjectionFeatures + channel], current, value);
    auto* out   = channel < kInputFeatures ? a.q : channel < 2 * kInputFeatures ? a.k : a.v;
    out[std::int64_t(column) * kInputFeatures + channel % kInputFeatures] =
        __float2bfloat16_rn(common::silu(value));
}

struct ConvolutionEpilogue {
    template <class S>
    static constexpr int kSharedBytes = 64 * S::kBlockTokens * sizeof(float);

    template <class S, bool Full>
    __device__ void finish_tile(const TiledConvolution& output, unsigned char* scratch,
                                float (&acc)[S::kMmaRows][S::kMmaTokens][4], int row_begin,
                                int token_begin, int rows, int tokens) const {
        static_assert(S::kBlockTokens == kTiledTokens && S::kBlockRows % 64 == 0);
        const auto& a       = output.args;
        constexpr int Block = S::kBlockTokens;
        auto* tile          = reinterpret_cast<float*>(scratch);
        int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
        int wr = warp / S::kWarpGridTokens, wt = warp % S::kWarpGridTokens;
        const auto at = [&](int r, int t) { return tile + t * 64 + (r ^ ((t & 7) << 3)); };
        // Retire contraction reads before reusing its staging. Only 64 rows are
        // transposed at a time, preserving the contraction's shared-memory occupancy.
        for (int first = 0; first < S::kBlockRows; first += 64) {
            __syncthreads();
#pragma unroll
            for (int i = 0; i < S::kMmaRows; i++) {
                int r = wr * S::kWarpRows + i * 16 + lane / 4 - first;
                if (r >= 0 && r < 64) {
#pragma unroll
                    for (int j = 0; j < S::kMmaTokens; j++) {
                        int t             = wt * S::kWarpTokens + j * 8 + 2 * (lane % 4);
                        *at(r, t)         = acc[i][j][0];
                        *at(r, t + 1)     = acc[i][j][1];
                        *at(r + 8, t)     = acc[i][j][2];
                        *at(r + 8, t + 1) = acc[i][j][3];
                    }
                }
            }
            __syncthreads();
            for (int i = tid; i < 64 * Block; i += S::kThreads) {
                const int r = i % 64, local = i / 64, c = row_begin + first + r,
                          column = token_begin + local;
                if (c >= rows || column >= tokens) continue;
                const int b = column / a.width, t = column % a.width,
                          n = a.valid ? a.valid[b] : a.width;
                auto* dst   = c < kInputFeatures ? a.q : c < 2 * kInputFeatures ? a.k : a.v;
                if (t >= n) {
                    dst[std::int64_t(column) * kInputFeatures + c % kInputFeatures] =
                        __float2bfloat16_rn(0.f);
                    continue;
                }
                const float current = *at(r, local);
                if (local < 3 || local >= Block - 3) {
                    const int edge  = local < 3 ? local : 3 + local - (Block - 3);
                    output.edges[(std::int64_t(token_begin / Block) * 6 + edge) *
                                     kProjectionFeatures +
                                 c] = current;
                }
                if (a.records)
                    a.records[std::int64_t(column) * kProjectionFeatures + c] =
                        __float2bfloat16_rn(current);
                if (output.tail && t >= n - 3)
                    output.tail[(std::int64_t(b) * 3 + t - (n - 3)) * kProjectionFeatures + c] =
                        __float2bfloat16_rn(current);
                if (local < 3) continue;
                float h[3];
#pragma unroll
                for (int j = 0; j < 3; j++) {
                    const int p = t - 3 + j;
                    h[j]        = p < 0 ? __bfloat162float(a.source[a.source_offsets[b] / 2 +
                                                             (p + 3LL) * kProjectionFeatures + c])
                                        : *at(r, local - 3 + j);
                }
                write_convolution(a, c, column, h[0], h[1], h[2], current);
            }
        }
    }
};

template <int Block>
__global__ void convolution_boundaries(TiledConvolution output) {
    const auto& a = output.args;
    const int c   = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kProjectionFeatures) return;
    const int total = a.width * a.batch, tiles = total / Block + (total % Block != 0);
    // All old-tail readers for one channel finish before the same channel's
    // in-place update. No CTA consumes another CTA's published history.
    for (int tile = 0; tile < tiles; tile++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            int column = tile * Block + j;
            if (column >= total) break;
            int b = column / a.width, t = column % a.width, n = a.valid ? a.valid[b] : a.width;
            if (t >= n) continue;
            float h[4];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                int p = t - 3 + k, relative = j - 3 + k;
                if (p < 0)
                    h[k] = __bfloat162float(
                        a.source[a.source_offsets[b] / 2 + (p + 3LL) * kProjectionFeatures + c]);
                else {
                    int edge = relative < 0 ? 6 + relative : relative;
                    h[k]     = output.edges[(std::int64_t(tile - (relative < 0)) * 6 + edge) *
                                            kProjectionFeatures +
                                        c];
                }
            }
            write_convolution(a, c, column, h[0], h[1], h[2], h[3]);
        }
    }
    if (output.tail)
        for (int b = 0; b < a.batch; b++) {
            int n = a.valid ? a.valid[b] : a.width;
            __nv_bfloat16 h[3];
#pragma unroll
            for (int j = 0; j < 3; j++) {
                int p = n - 3 + j;
                h[j]  = p < 0
                            ? a.source[a.source_offsets[b] / 2 + (p + 3LL) * kProjectionFeatures + c]
                            : output.tail[(std::int64_t(b) * 3 + j) * kProjectionFeatures + c];
            }
#pragma unroll
            for (int j = 0; j < 3; j++)
                a.destination[a.destination_offsets[b] / 2 + std::int64_t(j) * kProjectionFeatures +
                              c] = h[j];
        }
}
} // namespace ninfer::ops::detail::kda_input
