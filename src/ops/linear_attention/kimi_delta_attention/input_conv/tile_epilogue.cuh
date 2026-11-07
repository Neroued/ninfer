#pragma once
#include "ops/linear_attention/kimi_delta_attention/input_conv/kernels.cuh"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail::kda_input {
struct TiledConvolution {
    Arguments args;
    float* edges;
    __nv_bfloat16* tail;
    std::int64_t column_begin = 0;
};

template <class ConvWeight = float>
__device__ inline void write_convolution(const Arguments& a, int channel, std::int64_t column,
                                         float h0, float h1, float h2, float current) {
    const float value = convolution<ConvWeight>(a.weights, channel, h0, h1, h2, current);
    auto* out         = channel < kInputFeatures ? a.q : channel < 2 * kInputFeatures ? a.k : a.v;
    out[std::int64_t(column) * kInputFeatures + channel % kInputFeatures] =
        __float2bfloat16_rn(value);
}

template <class ConvWeight = float, class Column = int, bool SingleTile = false,
          bool CacheColumns = false>
struct ConvolutionEpilogue {
    template <class S>
    static constexpr int kSharedBytes =
        64 * S::kBlockTokens * sizeof(float) + (CacheColumns ? S::kBlockTokens * sizeof(int4) : 0);

    template <class S, bool Full>
    __device__ void finish_tile(const TiledConvolution& output, unsigned char* scratch,
                                float (&acc)[S::kMmaRows][S::kMmaTokens][4], int row_begin,
                                int token_begin, int rows, int tokens) const {
        static_assert(S::kBlockRows % 64 == 0);
        static_assert(SingleTile || S::kBlockTokens == kTiledTokens);
        const auto& a               = output.args;
        const int local_token_begin = SingleTile ? 0 : token_begin;
        const Column global_begin =
            SingleTile ? 0 : static_cast<Column>(output.column_begin) + token_begin;
        constexpr int Block = S::kBlockTokens;
        auto* tile          = reinterpret_cast<float*>(scratch);
        int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
        int wr = warp / S::kWarpGridTokens, wt = warp % S::kWarpGridTokens;
        const auto at  = [&](int r, int t) { return tile + t * 64 + (r ^ ((t & 7) << 3)); };
        auto* metadata = reinterpret_cast<int4*>(scratch + 64 * Block * sizeof(float));
        if constexpr (CacheColumns) {
            for (int local = tid; local < Block; local += S::kThreads) {
                if (local_token_begin + local < tokens) {
                    const auto column = global_begin + local;
                    const int b = column / a.width, t = column % a.width;
                    metadata[local] = make_int4(b, t, a.valid ? a.valid[b] : a.width, 0);
                } else {
                    metadata[local] = make_int4(0, 0, 0, 0);
                }
            }
        }
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
                const int r = i % 64, local = i / 64, c = row_begin + first + r;
                const auto column = global_begin + local;
                if (c >= rows || local_token_begin + local >= tokens) continue;
                int b, t, n;
                if constexpr (CacheColumns) {
                    const int4 info = common::load_vec<int4>(metadata + local);
                    b               = info.x;
                    t               = info.y;
                    n               = info.z;
                } else {
                    b = column / a.width;
                    t = column % a.width;
                    n = a.valid ? a.valid[b] : a.width;
                }
                auto* dst = c < kInputFeatures ? a.q : c < 2 * kInputFeatures ? a.k : a.v;
                if (t >= n) {
                    dst[std::int64_t(column) * kInputFeatures + c % kInputFeatures] =
                        __float2bfloat16_rn(0.f);
                    continue;
                }
                const float current = *at(r, local);
                if (a.records)
                    a.records[std::int64_t(column) * kProjectionFeatures + c] =
                        __float2bfloat16_rn(current);
                if constexpr (!SingleTile) {
                    if (local < 3 || local >= Block - 3) {
                        const int edge  = local < 3 ? local : 3 + local - (Block - 3);
                        output.edges[(std::int64_t(global_begin / Block) * 6 + edge) *
                                         kProjectionFeatures +
                                     c] = current;
                    }
                    if (output.tail && t >= n - 3)
                        output.tail[(std::int64_t(b) * 3 + t - (n - 3)) * kProjectionFeatures + c] =
                            __float2bfloat16_rn(current);
                    if (local < 3) continue;
                }
                float h[3];
#pragma unroll
                for (int j = 0; j < 3; j++) {
                    const int p = t - 3 + j;
                    h[j]        = p < 0 ? __bfloat162float(a.source[a.source_offsets[b] / 2 +
                                                             (p + 3LL) * kProjectionFeatures + c])
                                        : *at(r, local - 3 + j);
                }
                write_convolution<ConvWeight>(a, c, column, h[0], h[1], h[2], current);
            }
            if constexpr (SingleTile) {
                // Every old-history reader retires before this channel publishes its tail.
                __syncthreads();
                if (a.destination && tid < 64) {
                    const int c = row_begin + first + tid;
                    for (int b = 0; b < a.batch; ++b) {
                        const int n = a.valid ? a.valid[b] : a.width;
                        __nv_bfloat16 history[3];
#pragma unroll
                        for (int j = 0; j < 3; ++j) {
                            const int t = n - 3 + j;
                            history[j]  = t < 0 ? a.source[a.source_offsets[b] / 2 +
                                                          (t + 3LL) * kProjectionFeatures + c]
                                                : __float2bfloat16_rn(*at(tid, b * a.width + t));
                        }
#pragma unroll
                        for (int j = 0; j < 3; ++j)
                            a.destination[a.destination_offsets[b] / 2 +
                                          std::int64_t(j) * kProjectionFeatures + c] = history[j];
                    }
                }
            }
        }
    }
};

template <int Block, class ConvWeight = float, class Column = int>
__global__ void convolution_boundaries(TiledConvolution output) {
    const auto& a = output.args;
    const int c   = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kProjectionFeatures) return;
    const Column total = Column(a.width) * a.batch;
    const auto tiles   = total / Block + (total % Block != 0);
    // All old-tail readers for one channel finish before the same channel's
    // in-place update. No CTA consumes another CTA's published history.
    for (Column tile = 0; tile < tiles; tile++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            const auto column = tile * Block + j;
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
            write_convolution<ConvWeight>(a, c, column, h[0], h[1], h[2], h[3]);
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

template <class Column = int, bool SingleTile = false, bool FastSilu = false>
struct PackedConvolutionEpilogue {
    template <class S>
    static constexpr int kSharedBytes =
        64 * S::kBlockTokens * sizeof(float) + S::kBlockTokens * sizeof(int4);

    template <class S, bool Full>
    __device__ void finish_tile(const TiledConvolution& output, unsigned char* scratch,
                                float (&acc)[S::kMmaRows][S::kMmaTokens][4], int row_begin,
                                int token_begin, int rows, int tokens) const {
        static_assert(S::kBlockRows % 64 == 0);
        static_assert(SingleTile || S::kBlockTokens == kTiledTokens);
        const auto& a         = output.args;
        constexpr int Block   = S::kBlockTokens;
        const int local_begin = SingleTile ? 0 : token_begin;
        const Column begin    = SingleTile ? 0 : Column(output.column_begin) + token_begin;
        const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
        const int wr = warp / S::kWarpGridTokens, wt = warp % S::kWarpGridTokens;
        auto* values    = reinterpret_cast<float*>(scratch);
        auto* metadata  = reinterpret_cast<int4*>(scratch + 64 * Block * sizeof(float));
        const auto at   = [&](int r, int t) { return values + t * 64 + (r ^ ((t & 7) << 3)); };
        const auto pack = [](const float (&v)[4]) {
            return make_uint2(common::pack_bf16x2(v[0], v[1]), common::pack_bf16x2(v[2], v[3]));
        };
        for (int t = tid; t < Block; t += S::kThreads) {
            const Column column = begin + t;
            const int b = column / a.width, offset = column % a.width;
            metadata[t] = local_begin + t < tokens
                              ? make_int4(b, offset, a.valid ? a.valid[b] : a.width, 0)
                              : make_int4(0, 0, 0, 0);
        }
        for (int first = 0; first < S::kBlockRows; first += 64) {
            __syncthreads();
#pragma unroll
            for (int i = 0; i < S::kMmaRows; ++i) {
                const int r = wr * S::kWarpRows + i * 16 + lane / 4 - first;
                if (r >= 0 && r < 64) {
#pragma unroll
                    for (int j = 0; j < S::kMmaTokens; ++j) {
                        const int t       = wt * S::kWarpTokens + j * 8 + 2 * (lane % 4);
                        *at(r, t)         = acc[i][j][0];
                        *at(r, t + 1)     = acc[i][j][1];
                        *at(r + 8, t)     = acc[i][j][2];
                        *at(r + 8, t + 1) = acc[i][j][3];
                    }
                }
            }
            __syncthreads();
            for (int i = tid; i < 16 * Block; i += S::kThreads) {
                const int r = (i % 16) * 4, local = i / 16, c = row_begin + first + r;
                if (c >= rows || local_begin + local >= tokens) continue;
                const int4 info = common::load_vec<int4>(metadata + local);
                const int b = info.x, t = info.y, valid = info.z;
                const Column column = begin + local;
                auto* dst           = c < hidden ? a.q : c < 2 * hidden ? a.k : a.v;
                auto* result        = dst + std::int64_t(column) * hidden + c % hidden;
                if (t >= valid) {
                    common::store_vec(result, make_uint2(0, 0));
                    continue;
                }
                float current[4];
                common::store_vec(current, common::load_vec<float4>(at(r, local)));
                if (a.records)
                    common::store_vec(a.records + std::int64_t(column) * channels + c,
                                      pack(current));
                if constexpr (!SingleTile) {
                    if (local < 3 || local >= Block - 3) {
                        const int edge = local < 3 ? local : 3 + local - (Block - 3);
                        common::store_vec(
                            output.edges + (std::int64_t(begin / Block) * 6 + edge) * channels + c,
                            common::load_vec<float4>(current));
                    }
                    if (output.tail && t >= valid - 3)
                        common::store_vec(
                            output.tail + (std::int64_t(b) * 3 + t - (valid - 3)) * channels + c,
                            pack(current));
                    if (local < 3) continue;
                }
                float h[3][4], weight[4][4], result_values[4];
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    const int previous = t - 3 + j;
                    if (previous < 0)
                        common::load_bf16x4(h[j], a.source + a.source_offsets[b] / 2 +
                                                      (previous + 3LL) * channels + c);
                    else
                        common::store_vec(h[j], common::load_vec<float4>(at(r, local - 3 + j)));
                }
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    common::load_bf16x4(weight[j], static_cast<const __nv_bfloat16*>(a.weights) +
                                                       j * channels + c);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float x          = weight[0][j] * h[0][j];
                    x                = fmaf(weight[1][j], h[1][j], x);
                    x                = fmaf(weight[2][j], h[2][j], x);
                    x                = fmaf(weight[3][j], current[j], x);
                    result_values[j] = FastSilu ? __fdividef(x, 1.F + __expf(-x)) : common::silu(x);
                }
                common::store_vec(result, pack(result_values));
            }
            if constexpr (SingleTile) {
                if (a.destination) {
                    __syncthreads();
                    for (int i = tid; i < 16 * a.batch; i += S::kThreads) {
                        const int r = (i % 16) * 4, b = i / 16, c = row_begin + first + r;
                        if (c >= rows) continue;
                        const int valid = a.valid ? a.valid[b] : a.width;
                        uint2 history[3];
#pragma unroll
                        for (int j = 0; j < 3; ++j) {
                            const int t = valid - 3 + j;
                            if (t < 0)
                                history[j] = common::load_vec<uint2>(
                                    a.source + a.source_offsets[b] / 2 + (t + 3LL) * channels + c);
                            else {
                                float v[4];
                                common::store_vec(v,
                                                  common::load_vec<float4>(at(r, b * a.width + t)));
                                history[j] = pack(v);
                            }
                        }
#pragma unroll
                        for (int j = 0; j < 3; ++j)
                            common::store_vec(a.destination + a.destination_offsets[b] / 2 +
                                                  std::int64_t(j) * channels + c,
                                              history[j]);
                    }
                }
            }
        }
    }
};
} // namespace ninfer::ops::detail::kda_input
