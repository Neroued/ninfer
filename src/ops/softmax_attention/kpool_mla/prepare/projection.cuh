#pragma once

#include "ops/softmax_attention/kpool_mla/prepare/coordinates.cuh"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"

namespace ninfer::ops::detail::kpool {

struct ProjectionRowMap {
    int begin[5]{}, end[5]{}, destination[5]{};

    __device__ __forceinline__ int operator()(int row) const {
#pragma unroll
        for (int i = 0; i < 5; ++i)
            if (row >= begin[i] && row < end[i]) return destination[i] + row - begin[i];
        return -1;
    }
};

inline ProjectionRowMap projection_row_map(const KpoolMlaProjectionLayout& rows, int parent) {
    const KpoolProjectionRows roles[]{rows.latent, rows.index_key, rows.pool_gate, rows.query,
                                      rows.coefficients};
    constexpr int destinations[]{0, 512, 640, 768, 2304};
    ProjectionRowMap out;
    for (int i = 0; i < 5; ++i)
        if (roles[i].rows && roles[i].parent == parent) {
            out.begin[i]       = roles[i].row_begin;
            out.end[i]         = roles[i].row_begin + roles[i].rows;
            out.destination[i] = destinations[i];
        }
    return out;
}

struct ProjectionPartialOutput {
    float* data;
    int rows, columns;
    ProjectionRowMap mapping;

    __device__ __forceinline__ void store(int row, int column, float value) const {
        data[(std::int64_t(blockIdx.z) * columns + column) * rows + mapping(row)] = value;
    }

    __device__ __forceinline__ void store4(int row, int column, float4 value) const {
        *reinterpret_cast<float4*>(data + (std::int64_t(blockIdx.z) * columns + column) * rows +
                                   mapping(row)) = value;
    }
};

struct ProjectionOutput {
    PrepareArguments args;
    std::int64_t begin;
    ProjectionRowMap mapping;

    __device__ bool clear_if_empty(int first_row, int first_column, int rows, int columns) const {
        if (__ldg(args.valid)) return false;
        for (int b = 1; b < args.batch; ++b)
            if (__ldg(args.valid + b)) return false;
        const int first = max(first_row, mapping.begin[4]);
        const int count = min(first_row + rows, mapping.end[4]) - first;
        if (count <= 0) return true;
        for (int i = threadIdx.x; i < count * columns; i += blockDim.x)
            args.coefficients[(begin + first_column + i / count) * 32 + first - mapping.begin[4] +
                              i % count] = 0.F;
        return true;
    }

    __device__ __forceinline__ void store(int row, int column, float value) const {
        row          = mapping(row);
        const auto m = begin + column;
        int b, t;
        coordinate(args, column, b, t);
        const bool live = t < args.valid[b];
        if (row >= 2304) {
            args.coefficients[m * 32 + row - 2304] = live ? value : 0.F;
        } else if (live) {
            if (row >= 640 && row < 768) {
                __nv_bfloat16* dst;
                if (args.update)
                    dst = args.gates + std::int64_t(column) * 128;
                else {
                    const auto& field = args.records[1];
                    dst               = field.data + std::int64_t(b) * field.row_stride +
                          (std::int64_t(args.record_offsets[b]) + t) * field.column_stride;
                }
                dst[row - 640] = __float2bfloat16_rn(value);
            } else {
                const int feature = row < 640 ? row : row - 128;
                args.projected[std::int64_t(column) * args.norm_dim + feature] = value;
            }
        }
    }

    __device__ __forceinline__ void store4(int row, int column, float4 value) const {
        row          = mapping(row);
        const auto m = begin + column;
        int b, t;
        coordinate(args, column, b, t);
        const bool live = t < args.valid[b];
        if (row >= 2304) {
            *reinterpret_cast<float4*>(args.coefficients + m * 32 + row - 2304) =
                live ? value : make_float4(0, 0, 0, 0);
        } else if (live) {
            if (row >= 640 && row < 768) {
                const auto& field = args.records[1];
                auto* dst         = args.update ? args.gates + std::int64_t(column) * 128
                                                : field.data + std::int64_t(b) * field.row_stride +
                                              (std::int64_t(args.record_offsets[b]) + t) *
                                                  field.column_stride;
                dst += row - 640;
                dst[0] = __float2bfloat16_rn(value.x);
                dst[1] = __float2bfloat16_rn(value.y);
                dst[2] = __float2bfloat16_rn(value.z);
                dst[3] = __float2bfloat16_rn(value.w);
            } else {
                const int feature                   = row < 640 ? row : row - 128;
                *reinterpret_cast<float4*>(args.projected + std::int64_t(column) * args.norm_dim +
                                           feature) = value;
            }
        }
    }
};

// Some small paired shapes are faster without the extra device-length dependency.
struct UnconditionalProjectionOutput : ProjectionOutput {
    __device__ bool clear_if_empty(int, int, int, int) const { return false; }
};

// Publish sliced-K reductions with the whole CTA while retaining FP32 norm inputs.
struct ProjectionRowEpilogue {
    static constexpr bool kSwizzleRowStorage = true;

    template <class Output, int Tokens>
    __device__ void apply_row(const Output& output, int row, int begin,
                              const float (&values)[Tokens], int live) const {
        for (int t = 0; t < live; ++t) output.store(row, begin + t, values[t]);
    }

    template <class S, class Output, class Tile>
    __device__ void finish_rows(const Output& output, Tile tile, int row_begin, int token_begin,
                                int rows, int tokens) const {
        constexpr int BR = S::kBlockRows, BT = S::kBlockTokens;
        for (int i = threadIdx.x; i < BR * BT / 4; i += S::kThreads) {
            const int t = i / (BR / 4), r = (i % (BR / 4)) * 4;
            if (row_begin + r + 3 < rows && token_begin + t < tokens)
                output.store4(row_begin + r, token_begin + t,
                              make_float4(tile.at(r, t), tile.at(r + 1, t), tile.at(r + 2, t),
                                          tile.at(r + 3, t)));
        }
    }
};

// Retired operand staging is reused to transpose FP32 accumulators. Unlike the Linear BF16
// vector epilogue this preserves the complete accumulator through normalization.
struct ProjectionTileEpilogue {
    static constexpr bool kTmaCollective  = true;
    static constexpr bool kPredicatedRows = true;
    template <class S>
    static constexpr int kSharedBytes = 64 * S::kBlockTokens * sizeof(float);

    template <class S, bool FullTokens, class Output>
    __device__ __forceinline__ void finish_tile(const Output& output, unsigned char* scratch,
                                                float (&accum)[S::kMmaRows][S::kMmaTokens][4],
                                                int row_begin, int token_begin, int rows,
                                                int tokens) const {
        constexpr int BR = S::kBlockRows, BT = S::kBlockTokens;
        constexpr int Producers = [] {
            if constexpr (requires { S::kProducerThreads; }) return S::kProducerThreads;
            return 0;
        }();
        auto* tile    = reinterpret_cast<float*>(scratch);
        const auto at = [&](int r, int t) { return tile + t * 64 + (r ^ ((t & 7) << 3)); };
#pragma unroll
        for (int first = 0; first < BR; first += 64) {
            if (first) __syncthreads();
            if (threadIdx.x >= Producers) {
                const int tid = threadIdx.x - Producers, warp = tid / 32, lane = tid & 31;
                const int wm = warp / S::kWarpsTokens, wn = warp % S::kWarpsTokens;
#pragma unroll
                for (int mi = 0; mi < S::kMmaRows; ++mi) {
                    const int r = wm * S::kWarpRows + mi * 16 + (lane >> 2) - first;
                    if (r >= 0 && r < 64) {
#pragma unroll
                        for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                            const int t       = wn * S::kWarpTokens + ni * 8 + 2 * (lane & 3);
                            *at(r, t)         = accum[mi][ni][0];
                            *at(r, t + 1)     = accum[mi][ni][1];
                            *at(r + 8, t)     = accum[mi][ni][2];
                            *at(r + 8, t + 1) = accum[mi][ni][3];
                        }
                    }
                }
            }
            __syncthreads();
            const int active = min(64, BR - first);
            for (int i = threadIdx.x; i < active * BT / 4; i += S::kThreads) {
                const int t = i / (active / 4), r = i % (active / 4) * 4;
                if (row_begin + first + r < rows && (FullTokens || token_begin + t < tokens))
                    output.store4(row_begin + first + r, token_begin + t,
                                  *reinterpret_cast<float4*>(at(r, t)));
            }
        }
    }
};
} // namespace ninfer::ops::detail::kpool
