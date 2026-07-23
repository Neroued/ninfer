#pragma once

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kBidirectionalGqaHeadDim  = 128;
inline constexpr int kBidirectionalGqaQHeads   = 32;
inline constexpr int kBidirectionalGqaKVHeads  = 8;
inline constexpr int kBidirectionalGqaGroup    = 4;
inline constexpr int kBidirectionalGqaMaxSplit = 85;
inline constexpr int kSwaWindow                = 4096;

__device__ __forceinline__ int bidirectional_gqa_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ unsigned bidirectional_gqa_swz_addr(unsigned lane_base, unsigned ck,
                                                               unsigned as, unsigned r) {
    return lane_base + ((ck | as) ^ r);
}

__device__ __forceinline__ std::int64_t bidirectional_gqa_q_index(int q_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kBidirectionalGqaHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(kBidirectionalGqaQHeads) * token);
}

__device__ __forceinline__ std::int64_t bidirectional_gqa_query_kv_index(int kv_head, int d,
                                                                         int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kBidirectionalGqaHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(kBidirectionalGqaKVHeads) * token);
}

__device__ __forceinline__ std::int64_t
bidirectional_gqa_context_index(int kv_head, int d, int position, int padded_context) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kBidirectionalGqaHeadDim) *
                                              (static_cast<std::int64_t>(position) +
                                               static_cast<std::int64_t>(padded_context) * kv_head);
}

template <int Tokens>
__device__ __forceinline__ std::int64_t bidirectional_gqa_partial_index(int q_head, int d,
                                                                        int token, int split) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kBidirectionalGqaHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(kBidirectionalGqaQHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(Tokens) * split));
}

template <int Tokens>
__device__ __forceinline__ std::int64_t bidirectional_gqa_stat_index(int q_head, int token,
                                                                     int split) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(kBidirectionalGqaQHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(Tokens) * split);
}

__device__ __forceinline__ void noncausal_gqa_row_to_qt(int row, int kv_head, int& q_head,
                                                        int& token) {
    token             = row / kBidirectionalGqaGroup;
    const int q_local = row - token * kBidirectionalGqaGroup;
    q_head            = kv_head * kBidirectionalGqaGroup + q_local;
}

template <bool CyclicSwa, int KeyBlock, int Threads>
__device__ __forceinline__ void
bidirectional_gqa_stage_tile(__nv_bfloat16* dst, const __nv_bfloat16* context,
                             const __nv_bfloat16* query, int key0, int valid_keys, bool query_tile,
                             int kv_head, int padded_context, int tid) {
    constexpr int VecsPerRow = kBidirectionalGqaHeadDim / 8;
    for (int chunk = tid; chunk < KeyBlock * VecsPerRow; chunk += Threads) {
        const int row        = chunk / VecsPerRow;
        const int d          = (chunk - row * VecsPerRow) * 8;
        const bool live      = row < valid_keys;
        int context_position = live ? key0 + row : 0;
        if constexpr (CyclicSwa) { context_position &= kSwaWindow - 1; }
        const std::int64_t src_index =
            query_tile
                ? bidirectional_gqa_query_kv_index(kv_head, d, live ? row : 0)
                : bidirectional_gqa_context_index(kv_head, d, context_position, padded_context);
        const __nv_bfloat16* src = query_tile ? query + src_index : context + src_index;
        __nv_bfloat16* smem = &dst[row * kBidirectionalGqaHeadDim + bidirectional_gqa_swz(row, d)];
        cp_async_zfill<16, Cache::cg>(smem, src, live ? 16 : 0);
    }
}

template <bool CyclicSwa, int Tokens, int WarpsPerCta, int KeyBlock, bool DirectOutput>
__device__ __forceinline__ void noncausal_gqa_split_partial_body(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ context_state,
    const __nv_bfloat16* __restrict__ context_k, const __nv_bfloat16* __restrict__ context_v,
    int padded_context, int max_context, int split_capacity, float scale,
    __nv_bfloat16* __restrict__ partial_acc, float* __restrict__ partial_m,
    float* __restrict__ partial_l, __nv_bfloat16* __restrict__ out) {
    static_assert(Tokens >= 1 && Tokens <= 16);
    static_assert(WarpsPerCta == (Tokens + 3) / 4);
    static_assert(KeyBlock == 32 || KeyBlock == 64);

    constexpr int D             = kBidirectionalGqaHeadDim;
    constexpr int Wc            = WarpsPerCta;
    constexpr int Threads       = Wc * 32;
    constexpr int Br            = Wc * 16;
    constexpr int RowCount      = Tokens * kBidirectionalGqaGroup;
    constexpr int QKNt          = KeyBlock / 8;
    constexpr int QKKs          = D / 16;
    constexpr int PVNt          = D / 8;
    constexpr int PVKs          = KeyBlock / 16;
    constexpr int RowBytes      = D * static_cast<int>(sizeof(__nv_bfloat16));
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(RowCount <= Br);
    static_assert(Br <= 2 * KeyBlock);
    const int kv_head = static_cast<int>(blockIdx.x);
    const int split   = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int length  = context_state[0];
    if (kv_head >= kBidirectionalGqaKVHeads || split >= split_capacity || length < 0 ||
        length > max_context) {
        return;
    }

    const int context_count = CyclicSwa ? min(length, kSwaWindow - 1) : length;
    const int context_start = length - context_count;
    const int context_tiles = (context_count + KeyBlock - 1) / KeyBlock;
    const int active_splits = context_tiles > 0 ? min(context_tiles, split_capacity) : 1;
    if (split >= active_splits) { return; }

    const int tile_begin =
        static_cast<int>((static_cast<std::int64_t>(context_tiles) * split) / active_splits);
    const int tile_end =
        static_cast<int>((static_cast<std::int64_t>(context_tiles) * (split + 1)) / active_splits);
    const bool owns_query        = split == active_splits - 1;
    const int context_tile_count = tile_end - tile_begin;
    const int iterations         = context_tile_count + (owns_query ? 1 : 0);

    extern __shared__ __align__(16) __nv_bfloat16 shared[];
    __nv_bfloat16* k_s = shared;
    __nv_bfloat16* v_s = shared + KeyBlock * D;

    // The two K/V buffers together hold at least Br rows. Use them once as Q staging, then retain
    // all Q MMA fragments in registers for the complete split.
    for (int chunk = tid; chunk < Br * (D / 8); chunk += Threads) {
        const int row = chunk / (D / 8);
        const int d   = (chunk - row * (D / 8)) * 8;
        int q_head = 0, token = 0;
        noncausal_gqa_row_to_qt(row, kv_head, q_head, token);
        const bool live = row < RowCount;
        const __nv_bfloat16* src =
            q + bidirectional_gqa_q_index(live ? q_head : 0, d, live ? token : 0);
        __nv_bfloat16* dst = &shared[row * D + bidirectional_gqa_swz(row, d)];
        cp_async_zfill<16, Cache::cg>(dst, src, live ? 16 : 0);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    const int warp_row0 = warp * 16;
    const int row0      = warp_row0 + gid;
    const int row1      = row0 + 8;
    const int q_position0 =
        CyclicSwa ? context_state[row0 < RowCount ? row0 / kBidirectionalGqaGroup : 0] : 0;
    const int q_position1 =
        CyclicSwa ? context_state[row1 < RowCount ? row1 / kBidirectionalGqaGroup : 0] : 0;
    unsigned af_q[QKKs][4];
#pragma unroll
    for (int ks = 0; ks < QKKs; ++ks) {
        const int row = warp_row0 + a_rowoff;
        const int col = ks * 16 + a_coloff;
        ldmatrix_x4(af_q[ks][0], af_q[ks][1], af_q[ks][2], af_q[ks][3],
                    smem_addr(&shared[row * D + bidirectional_gqa_swz(row, col)]));
    }
    __syncthreads();

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int item = 0; item < 4; ++item) { acc[n][item] = 0.0f; }
    }
    float m0 = -CUDART_INF_F;
    float m1 = -CUDART_INF_F;
    float l0 = 0.0f;
    float l1 = 0.0f;

    const unsigned v_sbase     = smem_addr(v_s);
    const unsigned v_lane_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 8 * RowBytes) +
                                 static_cast<unsigned>(b_rin * RowBytes);
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    auto tile_metadata = [&](int iteration, bool& is_query, int& key0, int& valid_keys) {
        is_query = iteration >= context_tile_count;
        if (is_query) {
            key0       = 0;
            valid_keys = Tokens;
        } else {
            key0       = context_start + (tile_begin + iteration) * KeyBlock;
            valid_keys = min(KeyBlock, length - key0);
        }
    };

    bool current_is_query = false;
    int current_key0      = 0;
    int current_valid     = 0;
    tile_metadata(0, current_is_query, current_key0, current_valid);
    bidirectional_gqa_stage_tile<CyclicSwa, KeyBlock, Threads>(
        k_s, context_k, query_k, current_key0, current_valid, current_is_query, kv_head,
        padded_context, tid);
    cp_commit();

    for (int iteration = 0; iteration < iterations; ++iteration) {
        cp_wait<0>();
        __syncthreads();

        bidirectional_gqa_stage_tile<CyclicSwa, KeyBlock, Threads>(
            v_s, context_v, query_v, current_key0, current_valid, current_is_query, kv_head,
            padded_context, tid);
        cp_commit();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < QKKs; ++ks) {
                unsigned bf[2];
                const int brow = nt * 8 + b_rin;
                const int bcol = ks * 16 + b_koff;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&k_s[brow * D + bidirectional_gqa_swz(brow, bcol)]));
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af_q[ks][0],
                         af_q[ks][1], af_q[ks][2], af_q[ks][3], bf[0], bf[1]);
            }
        }

        cp_wait<0>();
        __syncthreads();

        bool next_is_query = false;
        int next_key0      = 0;
        int next_valid     = 0;
        if (iteration + 1 < iterations) {
            tile_metadata(iteration + 1, next_is_query, next_key0, next_valid);
            bidirectional_gqa_stage_tile<CyclicSwa, KeyBlock, Threads>(
                k_s, context_k, query_k, next_key0, next_valid, next_is_query, kv_head,
                padded_context, tid);
            cp_commit();
        }

        float block_m0 = -CUDART_INF_F;
        float block_m1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            const int col1 = col0 + 1;
            const bool allow00 =
                row0 < RowCount && col0 < current_valid &&
                (!CyclicSwa || current_is_query || current_key0 + col0 >= q_position0 - 4095);
            const bool allow01 =
                row0 < RowCount && col1 < current_valid &&
                (!CyclicSwa || current_is_query || current_key0 + col1 >= q_position0 - 4095);
            const bool allow10 =
                row1 < RowCount && col0 < current_valid &&
                (!CyclicSwa || current_is_query || current_key0 + col0 >= q_position1 - 4095);
            const bool allow11 =
                row1 < RowCount && col1 < current_valid &&
                (!CyclicSwa || current_is_query || current_key0 + col1 >= q_position1 - 4095);
            score[nt][0] = allow00 ? score[nt][0] * scale : -CUDART_INF_F;
            score[nt][1] = allow01 ? score[nt][1] * scale : -CUDART_INF_F;
            score[nt][2] = allow10 ? score[nt][2] * scale : -CUDART_INF_F;
            score[nt][3] = allow11 ? score[nt][3] * scale : -CUDART_INF_F;
            block_m0     = fmaxf(block_m0, fmaxf(score[nt][0], score[nt][1]));
            block_m1     = fmaxf(block_m1, fmaxf(score[nt][2], score[nt][3]));
        }
        block_m0 = warp_max<4>(block_m0, FullMask);
        block_m1 = warp_max<4>(block_m1, FullMask);

        const float next_m0 = fmaxf(m0, block_m0);
        const float next_m1 = fmaxf(m1, block_m1);
        const float alpha0  = m0 == -CUDART_INF_F ? 0.0f : exp2_approx((m0 - next_m0) * Log2E);
        const float alpha1  = m1 == -CUDART_INF_F ? 0.0f : exp2_approx((m1 - next_m1) * Log2E);

        unsigned p_frag[PVKs][4];
        float block_l0 = 0.0f;
        float block_l1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 =
                score[nt][0] > -CUDART_INF_F ? exp2_approx((score[nt][0] - next_m0) * Log2E) : 0.0f;
            const float p01 =
                score[nt][1] > -CUDART_INF_F ? exp2_approx((score[nt][1] - next_m0) * Log2E) : 0.0f;
            const float p10 =
                score[nt][2] > -CUDART_INF_F ? exp2_approx((score[nt][2] - next_m1) * Log2E) : 0.0f;
            const float p11 =
                score[nt][3] > -CUDART_INF_F ? exp2_approx((score[nt][3] - next_m1) * Log2E) : 0.0f;
            block_l0 += p00 + p01;
            block_l1 += p10 + p11;
            const int pk = nt >> 1;
            if ((nt & 1) == 0) {
                p_frag[pk][0] = pack_bf16x2(p00, p01);
                p_frag[pk][1] = pack_bf16x2(p10, p11);
            } else {
                p_frag[pk][2] = pack_bf16x2(p00, p01);
                p_frag[pk][3] = pack_bf16x2(p10, p11);
            }
        }
        block_l0 = warp_sum<4>(block_l0, FullMask);
        block_l1 = warp_sum<4>(block_l1, FullMask);

        l0 = l0 * alpha0 + block_l0;
        l1 = l1 * alpha1 + block_l1;
        m0 = next_m0;
        m1 = next_m1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        constexpr int PVTilePairs = (PVNt + 1) / 2;
        constexpr int PVLoads     = PVKs * PVTilePairs;
        unsigned vf[2][4];
        ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                      bidirectional_gqa_swz_addr(v_lane_base, 0u, v_as, v_r));
#pragma unroll
        for (int load = 0; load < PVLoads; ++load) {
            const int pk   = load / PVTilePairs;
            const int n2   = (load % PVTilePairs) * 2;
            const int cur  = load & 1;
            const int next = cur ^ 1;
            if (load + 1 < PVLoads) {
                const int next_pk = (load + 1) / PVTilePairs;
                const int next_n2 = ((load + 1) % PVTilePairs) * 2;
                ldmatrix_x4_t(vf[next][0], vf[next][1], vf[next][2], vf[next][3],
                              bidirectional_gqa_swz_addr(
                                  v_lane_base + static_cast<unsigned>(next_pk * 16 * RowBytes),
                                  static_cast<unsigned>(next_n2 << 4), v_as, v_r));
            }
            mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[pk][0], p_frag[pk][1],
                     p_frag[pk][2], p_frag[pk][3], vf[cur][0], vf[cur][1]);
            if (n2 + 1 < PVNt) {
                mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3],
                         p_frag[pk][0], p_frag[pk][1], p_frag[pk][2], p_frag[pk][3], vf[cur][2],
                         vf[cur][3]);
            }
        }

        current_is_query = next_is_query;
        current_key0     = next_key0;
        current_valid    = next_valid;
    }

    if constexpr (!DirectOutput) {
        if (lid == 0) {
            const int row0 = warp_row0 + gid;
            const int row1 = row0 + 8;
            if (row0 < RowCount) {
                int q_head = 0, token = 0;
                noncausal_gqa_row_to_qt(row0, kv_head, q_head, token);
                partial_m[bidirectional_gqa_stat_index<Tokens>(q_head, token, split)] = m0;
                partial_l[bidirectional_gqa_stat_index<Tokens>(q_head, token, split)] = l0;
            }
            if (row1 < RowCount) {
                int q_head = 0, token = 0;
                noncausal_gqa_row_to_qt(row1, kv_head, q_head, token);
                partial_m[bidirectional_gqa_stat_index<Tokens>(q_head, token, split)] = m1;
                partial_l[bidirectional_gqa_stat_index<Tokens>(q_head, token, split)] = l1;
            }
        }
    }

#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0   = n * 8 + 2 * lid;
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0, token = 0;
            noncausal_gqa_row_to_qt(row0, kv_head, q_head, token);
            if constexpr (DirectOutput) {
                const float inv_l = l0 > 0.0f ? 1.0f / l0 : 0.0f;
                const auto dst    = bidirectional_gqa_q_index(q_head, d0, token);
                store_vec(&out[dst], pack_bf16x2(acc[n][0] * inv_l, acc[n][1] * inv_l));
            } else {
                const auto dst = bidirectional_gqa_partial_index<Tokens>(q_head, d0, token, split);
                store_vec(&partial_acc[dst], pack_bf16x2(acc[n][0], acc[n][1]));
            }
        }
        if (row1 < RowCount) {
            int q_head = 0, token = 0;
            noncausal_gqa_row_to_qt(row1, kv_head, q_head, token);
            if constexpr (DirectOutput) {
                const float inv_l = l1 > 0.0f ? 1.0f / l1 : 0.0f;
                const auto dst    = bidirectional_gqa_q_index(q_head, d0, token);
                store_vec(&out[dst], pack_bf16x2(acc[n][2] * inv_l, acc[n][3] * inv_l));
            } else {
                const auto dst = bidirectional_gqa_partial_index<Tokens>(q_head, d0, token, split);
                store_vec(&partial_acc[dst], pack_bf16x2(acc[n][2], acc[n][3]));
            }
        }
    }
}

template <int Tokens, int WarpsPerCta, int KeyBlock, bool DirectOutput>
__launch_bounds__(WarpsPerCta * 32, 2) __global__ void bidirectional_gqa_split_partial_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ context_length,
    const __nv_bfloat16* __restrict__ context_k, const __nv_bfloat16* __restrict__ context_v,
    int padded_context, int max_context, int split_capacity, float scale,
    __nv_bfloat16* __restrict__ partial_acc, float* __restrict__ partial_m,
    float* __restrict__ partial_l, __nv_bfloat16* __restrict__ out) {
    noncausal_gqa_split_partial_body<false, Tokens, WarpsPerCta, KeyBlock, DirectOutput>(
        q, query_k, query_v, context_length, context_k, context_v, padded_context, max_context,
        split_capacity, scale, partial_acc, partial_m, partial_l, out);
}

template <int Tokens, int WarpsPerCta, int KeyBlock, bool DirectOutput>
__launch_bounds__(WarpsPerCta * 32, 2) __global__ void swa_split_partial_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const __nv_bfloat16* __restrict__ context_k, const __nv_bfloat16* __restrict__ context_v,
    int padded_context, int max_context, int split_capacity, float scale,
    __nv_bfloat16* __restrict__ partial_acc, float* __restrict__ partial_m,
    float* __restrict__ partial_l, __nv_bfloat16* __restrict__ out) {
    noncausal_gqa_split_partial_body<true, Tokens, WarpsPerCta, KeyBlock, DirectOutput>(
        q, query_k, query_v, positions, context_k, context_v, padded_context, max_context,
        split_capacity, scale, partial_acc, partial_m, partial_l, out);
}

template <bool CyclicSwa, int Tokens, int KeyBlock>
__device__ __forceinline__ void
noncausal_gqa_reduce_body(const __nv_bfloat16* __restrict__ partial_acc,
                          const float* __restrict__ partial_m, const float* __restrict__ partial_l,
                          const std::int32_t* __restrict__ context_state, int max_context,
                          int split_capacity, __nv_bfloat16* __restrict__ out) {
    const int q_head = static_cast<int>(blockIdx.x);
    const int token  = static_cast<int>(blockIdx.y);
    const int tid    = static_cast<int>(threadIdx.x);
    const int length = context_state[0];
    if (q_head >= kBidirectionalGqaQHeads || token >= Tokens) { return; }
    if (length < 0 || length > max_context) {
        if (tid < kBidirectionalGqaHeadDim) {
            out[bidirectional_gqa_q_index(q_head, tid, token)] = __float2bfloat16(0.0f);
        }
        return;
    }

    const int context_count = CyclicSwa ? min(length, kSwaWindow - 1) : length;
    const int context_tiles = (context_count + KeyBlock - 1) / KeyBlock;
    const int active_splits = context_tiles > 0 ? min(context_tiles, split_capacity) : 1;
    __shared__ float reduce[128];

    float local_m = -CUDART_INF_F;
    for (int split = tid; split < active_splits; split += blockDim.x) {
        local_m =
            fmaxf(local_m, partial_m[bidirectional_gqa_stat_index<Tokens>(q_head, token, split)]);
    }
    reduce[tid] = local_m;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]); }
        __syncthreads();
    }
    const float global_m = reduce[0];
    __syncthreads();

    float local_l = 0.0f;
    for (int split = tid; split < active_splits; split += blockDim.x) {
        const auto idx = bidirectional_gqa_stat_index<Tokens>(q_head, token, split);
        local_l += partial_l[idx] * expf(partial_m[idx] - global_m);
    }
    reduce[tid] = local_l;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] += reduce[tid + stride]; }
        __syncthreads();
    }
    const float global_l = reduce[0];

    if (tid < kBidirectionalGqaHeadDim) {
        float numerator = 0.0f;
        for (int split = 0; split < active_splits; ++split) {
            const auto stat    = bidirectional_gqa_stat_index<Tokens>(q_head, token, split);
            const float weight = expf(partial_m[stat] - global_m);
            numerator += __bfloat162float(partial_acc[bidirectional_gqa_partial_index<Tokens>(
                             q_head, tid, token, split)]) *
                         weight;
        }
        const float value = global_l > 0.0f ? numerator / global_l : 0.0f;
        out[bidirectional_gqa_q_index(q_head, tid, token)] = __float2bfloat16(value);
    }
}

template <int Tokens, int KeyBlock>
__launch_bounds__(128, 2) __global__
    void bidirectional_gqa_reduce_kernel(const __nv_bfloat16* __restrict__ partial_acc,
                                         const float* __restrict__ partial_m,
                                         const float* __restrict__ partial_l,
                                         const std::int32_t* __restrict__ context_length,
                                         int max_context, int split_capacity,
                                         __nv_bfloat16* __restrict__ out) {
    noncausal_gqa_reduce_body<false, Tokens, KeyBlock>(
        partial_acc, partial_m, partial_l, context_length, max_context, split_capacity, out);
}

template <int Tokens, int KeyBlock, int WarpsPerBlock>
__launch_bounds__(WarpsPerBlock * 32, 2) __global__
    void swa_reduce_kernel(const __nv_bfloat16* __restrict__ partial_acc,
                           const float* __restrict__ partial_m, const float* __restrict__ partial_l,
                           const std::int32_t* __restrict__ positions, int max_context,
                           int split_capacity, __nv_bfloat16* __restrict__ out) {
    static_assert(WarpsPerBlock >= 1 && WarpsPerBlock <= 8);
    constexpr int MaxSplits = 128;
    constexpr unsigned Mask = 0xffffffffu;
    __shared__ float weights[WarpsPerBlock][MaxSplits];

    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int output_row = static_cast<int>(blockIdx.x) * WarpsPerBlock + warp;
    const int token      = output_row / kBidirectionalGqaQHeads;
    const int q_head     = output_row - token * kBidirectionalGqaQHeads;
    if (warp >= WarpsPerBlock || token >= Tokens) return;

    const int length = positions[0];
    if (length < 0 || length > max_context) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d                                      = lane + item * 32;
            out[bidirectional_gqa_q_index(q_head, d, token)] = __float2bfloat16(0.0f);
        }
        return;
    }
    const int context_count = min(length, kSwaWindow - 1);
    const int context_tiles = (context_count + KeyBlock - 1) / KeyBlock;
    const int active_splits = context_tiles > 0 ? min(context_tiles, split_capacity) : 1;

    float local_m = -CUDART_INF_F;
    for (int split = lane; split < active_splits; split += 32) {
        local_m =
            fmaxf(local_m, partial_m[bidirectional_gqa_stat_index<Tokens>(q_head, token, split)]);
    }
    const float global_m = warp_max<32>(local_m, Mask);

    float local_l = 0.0f;
    for (int split = lane; split < active_splits; split += 32) {
        const auto stat      = bidirectional_gqa_stat_index<Tokens>(q_head, token, split);
        const float weight   = expf(partial_m[stat] - global_m);
        weights[warp][split] = weight;
        local_l += partial_l[stat] * weight;
    }
    const float global_l = warp_sum<32>(local_l, Mask);
    __syncwarp(Mask);

#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d     = lane + item * 32;
        float numerator = 0.0f;
        for (int split = 0; split < active_splits; ++split) {
            numerator +=
                __bfloat162float(
                    partial_acc[bidirectional_gqa_partial_index<Tokens>(q_head, d, token, split)]) *
                weights[warp][split];
        }
        const float value = global_l > 0.0f ? numerator / global_l : 0.0f;
        out[bidirectional_gqa_q_index(q_head, d, token)] = __float2bfloat16(value);
    }
}

} // namespace ninfer::ops
