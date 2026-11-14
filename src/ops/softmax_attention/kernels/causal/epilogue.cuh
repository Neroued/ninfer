#pragma once

#include "ops/softmax_attention/kernels/causal/tile_io.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"

namespace ninfer::ops::detail {

__device__ __forceinline__ void causal_store_partial_pair(float* target, float a, float b) {
    *reinterpret_cast<float2*>(target) = make_float2(a, b);
}

template <class G>
__device__ __forceinline__ void causal_store_output_pair(__nv_bfloat16* out, int head, int d,
                                                         int token, float a, float b) {
    *reinterpret_cast<unsigned*>(out + causal_q_index<G>(head, d, token)) =
        common::pack_bf16x2(a, b);
}

__device__ __forceinline__ void causal_store_output(__nv_bfloat16* out, float value) {
    *out = __float2bfloat16(value);
}

// A warp converts one normalized row from the represented rotated V coordinates.
template <class G>
__device__ __forceinline__ void
causal_store_inverse_rotated_row(const float* row, __nv_bfloat16* out, int head, int token) {
    static_assert(G::kHeadDim == kCausalHeadDim);
    const int lane = threadIdx.x & 31;
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) values[r] = row[lane + 32 * r];
    kv::normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r)
        causal_store_output(out + causal_q_index<G>(head, lane + 32 * r, token), values[r]);
}

template <class G, int Threads>
__device__ __forceinline__ void causal_zero_grouped_output(__nv_bfloat16* out, int head,
                                                           std::int64_t column, int tokens) {
    for (int i = threadIdx.x; i < tokens * G::GroupSize * G::kHeadDim; i += Threads) {
        const int row = i / G::kHeadDim, d = i % G::kHeadDim;
        out[causal_q_index<G>(head * G::GroupSize + row % G::GroupSize, d,
                              static_cast<int>(column) + row / G::GroupSize)] =
            __float2bfloat16(0.F);
    }
}

// Grouped quantized mainloops share the same distributed PV accumulator layout. At one
// partition they normalize here; value-basis inversion reuses dead Q/K/V shared storage.
template <class G, int RowTiles, int ValueFragments, int Threads, bool Direct, bool Inverse,
          int FirstScratchRows>
__device__ __forceinline__ void
causal_grouped_epilogue(const float (&acc)[ValueFragments][4], float m0, float m1, float l0,
                        float l1, float* sums, float* first_scratch, float* second_scratch,
                        float* partial_acc, float* partial_m, float* partial_l, __nv_bfloat16* out,
                        int head, int tokens, int valid_tokens, int partial_width,
                        int partial_begin, int split) {
    constexpr int D = G::kHeadDim;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gid = lane >> 2, lid = lane & 3;
    if constexpr (!Direct) {
        if (warp < RowTiles && lid == 0) {
            const int row0 = warp * 16 + gid;
            const int row1 = row0 + 8;
            if (row0 < tokens * G::GroupSize) {
                int q_head = 0;
                int token  = 0;
                causal_row_to_qt<G>(row0, head, q_head, token);
                partial_m[causal_stat_index<G>(q_head, partial_begin + token, split,
                                               partial_width)] = m0;
                partial_l[causal_stat_index<G>(q_head, partial_begin + token, split,
                                               partial_width)] = l0;
            }
            if (row1 < tokens * G::GroupSize) {
                int q_head = 0;
                int token  = 0;
                causal_row_to_qt<G>(row1, head, q_head, token);
                partial_m[causal_stat_index<G>(q_head, partial_begin + token, split,
                                               partial_width)] = m1;
                partial_l[causal_stat_index<G>(q_head, partial_begin + token, split,
                                               partial_width)] = l1;
            }
        }

#pragma unroll
        for (int n = 0; n < ValueFragments; ++n) {
            const int consumer_tile     = warp % RowTiles;
            const int consumer_slice    = warp / RowTiles;
            const int consumer_row_base = consumer_tile * 16;
            const int d0                = (consumer_slice * ValueFragments + n) * 8 + 2 * lid;
            const int row0              = consumer_row_base + gid;
            const int row1              = row0 + 8;
            if (row0 < tokens * G::GroupSize) {
                int q_head = 0;
                int token  = 0;
                causal_row_to_qt<G>(row0, head, q_head, token);
                const auto dst = causal_partial_index<G>(q_head, d0, partial_begin + token, split,
                                                         partial_width);
                causal_store_partial_pair(partial_acc + dst, acc[n][0], acc[n][1]);
            }
            if (row1 < tokens * G::GroupSize) {
                int q_head = 0;
                int token  = 0;
                causal_row_to_qt<G>(row1, head, q_head, token);
                const auto dst = causal_partial_index<G>(q_head, d0, partial_begin + token, split,
                                                         partial_width);
                causal_store_partial_pair(partial_acc + dst, acc[n][2], acc[n][3]);
            }
        }
    } else {
        if (warp < RowTiles && lid == 0) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int row = warp * 16 + gid + 8 * j;
                if (row < tokens * G::GroupSize) { sums[row] = j == 0 ? l0 : l1; }
            }
        }
        __syncthreads();
        const int row_base = warp % RowTiles * 16;
        const int slice    = warp / RowTiles;
#pragma unroll
        for (int n = 0; n < ValueFragments; ++n) {
            const int d = (slice * ValueFragments + n) * 8 + 2 * lid;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int row = row_base + gid + 8 * j;
                if (row >= tokens * G::GroupSize) continue;
                const int token         = row / G::GroupSize;
                const int h             = head * G::GroupSize + row % G::GroupSize;
                const float denominator = sums[row];
                const bool live         = token < valid_tokens && denominator > 0.F;
                const float a           = live ? acc[n][2 * j] / denominator : 0.F;
                const float b           = live ? acc[n][2 * j + 1] / denominator : 0.F;
                if constexpr (Inverse) {
                    auto* destination = row < FirstScratchRows
                                            ? first_scratch + row * D
                                            : second_scratch + (row - FirstScratchRows) * D;
                    causal_store_partial_pair(destination + d, a, b);
                } else {
                    causal_store_output_pair<G>(out, h, d, token, a, b);
                }
            }
        }
        if constexpr (Inverse) {
            __syncthreads();
            for (int row = warp; row < tokens * G::GroupSize; row += Threads / 32) {
                const int token = row / G::GroupSize;
                const int h     = head * G::GroupSize + row % G::GroupSize;
                if (token >= valid_tokens) {
#pragma unroll
                    for (int d = 0; d < 8; ++d)
                        causal_store_output(out + causal_q_index<G>(h, lane + 32 * d, token), 0.F);
                } else {
                    const auto* source = row < FirstScratchRows
                                             ? first_scratch + row * D
                                             : second_scratch + (row - FirstScratchRows) * D;
                    causal_store_inverse_rotated_row<G>(source, out, h, token);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
