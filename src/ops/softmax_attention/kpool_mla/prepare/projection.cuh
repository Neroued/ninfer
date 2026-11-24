#pragma once

#include "ops/softmax_attention/kpool_mla/prepare/coordinates.cuh"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"

namespace ninfer::ops::detail::kpool {

struct ProjectionPartialOutput {
    float* data;
    int rows, columns;

    __device__ __forceinline__ void store(int row, int column, float value) const {
        data[(std::int64_t(blockIdx.z) * columns + column) * rows + row] = value;
    }

    __device__ __forceinline__ void store4(int row, int column, float4 value) const {
        *reinterpret_cast<float4*>(data + (std::int64_t(blockIdx.z) * columns + column) * rows +
                                   row) = value;
    }
};

struct ProjectionOutput {
    PrepareArguments args;
    std::int64_t begin;

    __device__ __forceinline__ void store(int row, int column, float value) const {
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

// Retired operand staging is reused to transpose FP32 accumulators. Unlike the Linear BF16
// vector epilogue this preserves the complete accumulator through normalization.
struct ProjectionTmaEpilogue {
    static constexpr bool kTmaCollective  = true;
    static constexpr bool kPredicatedRows = true;
    template <class S>
    static constexpr int kSharedBytes = ((S::kBlockRows + 63) / 64 * 64) * S::kBlockTokens * 4;

    template <class S, bool FullTokens, class Output>
    __device__ __forceinline__ void finish_tile(const Output& output, unsigned char* scratch,
                                                float (&accum)[S::kMmaRows][S::kMmaTokens][4],
                                                int row_begin, int token_begin, int rows,
                                                int tokens) const {
        constexpr int BR = S::kBlockRows, BT = S::kBlockTokens, PR = (BR + 63) / 64 * 64;
        auto* tile    = reinterpret_cast<float*>(scratch);
        const auto at = [&](int r, int t) { return tile + t * PR + (r ^ ((t & 7) << 3)); };
        if (threadIdx.x >= S::kProducerThreads) {
            const int tid = threadIdx.x - S::kProducerThreads, warp = tid / 32, lane = tid & 31;
            const int wm = warp / S::kWarpsTokens, wn = warp % S::kWarpsTokens;
#pragma unroll
            for (int mi = 0; mi < S::kMmaRows; ++mi) {
                const int r = wm * S::kWarpRows + mi * 16 + (lane >> 2);
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
        __syncthreads();
        for (int i = threadIdx.x; i < BR * BT / 4; i += S::kThreads) {
            const int t = i / (BR / 4), r = (i % (BR / 4)) * 4;
            if (row_begin + r < rows && (FullTokens || token_begin + t < tokens))
                output.store4(row_begin + r, token_begin + t, *reinterpret_cast<float4*>(at(r, t)));
        }
    }
};
} // namespace ninfer::ops::detail::kpool
