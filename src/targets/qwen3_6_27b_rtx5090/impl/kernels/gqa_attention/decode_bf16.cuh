#pragma once

// ninfer::kernels - split-KV GQA small-T attention, BF16 KV-cache partial kernel.
// Standalone from the int8 kernel (gqa_attention_decode_i8.cuh): shared scaffolding
// lives in gqa_attention_decode.cuh, but the body/append/load are not shared so the
// bf16 path can be tuned independently. Processes one KV head, one query-head
// subgroup, and one token tile; a reducer combines the split-local partials.

#include <cuda_bf16.h>
#include <math_constants.h>

#include "targets/qwen3_6_27b_rtx5090/impl/kernels/gqa_attention/decode.cuh"

#include <cstdint>

namespace ninfer::kernels {

template <int TokenTile, int WarpsPerCta>
__launch_bounds__(128, 2) __global__ void gqa_attention_small_t_tc_partial_bf16_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
    const std::int32_t* pos, __nv_bfloat16* cache_k, __nv_bfloat16* cache_v, std::int32_t tokens,
    std::int32_t padded_context, std::int32_t max_context, float scale, __nv_bfloat16* partial_acc,
    float* partial_m, float* partial_l) {
    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(WarpsPerCta >= 1 && WarpsPerCta <= 4);

    constexpr int Wc            = WarpsPerCta;
    constexpr int Br            = Wc * 16;
    constexpr int Bc            = 32;
    constexpr int D             = kGqaHeadDim;
    constexpr int Threads       = Wc * 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int QKKs          = D / 16;
    constexpr int PVNt          = D / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr int QkvRows       = 2 * Bc;

    static_assert(QkvRows >= Br);

    __shared__ __align__(16) __nv_bfloat16 qkv_s[QkvRows * D];
    __shared__ __align__(16) __nv_bfloat16 p_s[Wc * 16 * Bc];
    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    const int row_count   = tokens * kGqaGroupSize;

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index(q_head, token, split, tokens)] = -CUDART_INF_F;
                partial_l[gqa_partial_stat_index(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index(q_head, d, token, split, tokens)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= kGqaKVHeads || tokens < 1 || tokens > TokenTile ||
        row_count > Br || split_count <= 0) {
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= max_context) {
        write_neutral();
        return;
    }

    const int window             = last_pos + 1;
    const int active_split_count = gqa_small_t_active_splits(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int kps         = div_up(window, active_split_count);
    const int split_start = split * kps;
    const int split_limit = split_start + kps;
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }

    // Fused append: the split that owns each new token's absolute position writes its
    // bf16 K/V head vector straight into the cache for FUTURE decode steps. The
    // current step still reads the new tokens from k_new/v_new via the from_new path
    // below, so no split reads back what it just wrote in the same launch.
    for (int chunk = tid; chunk < tokens * (D / 8); chunk += Threads) {
        const int token = chunk / (D / 8);
        const int d     = (chunk - token * (D / 8)) * 8;
        const int p_tok = pos[token];
        if (p_tok >= split_start && p_tok < split_end && p_tok >= 0 && p_tok < max_context) {
            const std::int64_t new_off   = gqa_kv_new_index(kv_head, d, token);
            const std::int64_t cache_off = gqa_cache_index(kv_head, d, p_tok, padded_context);
            store_vec(&cache_k[cache_off], load_vec<int4>(&k_new[new_off]));
            store_vec(&cache_v[cache_off], load_vec<int4>(&v_new[new_off]));
        }
    }
    __syncthreads();

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt(row, tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < row_count && gqa_valid_q_head(kv_head, q_head)) {
            value = q[gqa_q_index(q_head, d, token)];
        }
        qkv_s[row * D + gqa_small_t_tc_swz(row, d)] = value;
    }
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
    __nv_bfloat16* p_sw = &p_s[warp * 16 * Bc];

    unsigned af_q[QKKs][4];
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int arow = warp_row0 + a_rowoff;
        const int acol = k * 16 + a_coloff;
        ldmatrix_x4(
            af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
            smem_addr(&qkv_s[arow * D + gqa_small_t_tc_swz(arow, acol)]));
    }
    __syncthreads();

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    const int key_blocks = div_up(split_end - split_start, Bc);
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = split_start + kb * Bc;
        // Stage the bf16 K/V key tile with one cp.async wave (16B/thread, high MLP).
        // Current-step tokens come from k_new/v_new; tail slots are zeroed.
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int key_l      = chunk / (D / 8);
            const int d          = (chunk - key_l * (D / 8)) * 8;
            const int key        = k0 + key_l;
            __nv_bfloat16* k_dst = &k_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            __nv_bfloat16* v_dst = &v_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            if (key < split_end) {
                const int new_token = key - first_pos;
                const bool from_new = new_token >= 0 && new_token < tokens && key >= first_pos;
                if (from_new) {
                    const std::int64_t off = gqa_kv_new_index(kv_head, d, new_token);
                    ninfer::kernels::cp_async<16>(k_dst, &k_new[off]);
                    ninfer::kernels::cp_async<16>(v_dst, &v_new[off]);
                } else {
                    const std::int64_t off = gqa_cache_index(kv_head, d, key, padded_context);
                    ninfer::kernels::cp_async<16>(k_dst, &cache_k[off]);
                    ninfer::kernels::cp_async<16>(v_dst, &cache_v[off]);
                }
            } else {
                store_vec(k_dst, make_int4(0, 0, 0, 0));
                store_vec(v_dst, make_int4(0, 0, 0, 0));
            }
        }
        ninfer::kernels::cp_commit();
        ninfer::kernels::cp_wait<0>();
        __syncthreads();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < QKKs; ++k) {
                unsigned bf[2];
                const int brow = nt * 8 + b_rin;
                const int bcol = k * 16 + b_koff;
                ldmatrix_x2(
                    bf[0], bf[1],
                    smem_addr(&k_s[brow * D + gqa_small_t_tc_swz(brow, bcol)]));
                mma_bf16(score[nt][0], score[nt][1], score[nt][2],
                                                 score[nt][3], af_q[k][0], af_q[k][1], af_q[k][2],
                                                 af_q[k][3], bf[0], bf[1]);
            }
        }

        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
        gqa_small_t_tc_row_to_qt(row0, tokens, kv_head, q_head0, token0);
        gqa_small_t_tc_row_to_qt(row1, tokens, kv_head, q_head1, token1);
        const int qabs0 = (row0 < row_count) ? pos[token0] : -1;
        const int qabs1 = (row1 < row_count) ? pos[token1] : -1;

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            const int col1 = col0 + 1;
            const int key0 = k0 + col0;
            const int key1 = col1 + k0;
            const int new_token0 = key0 - first_pos;
            const int new_token1 = key1 - first_pos;
            const bool from_new0 = new_token0 >= 0 && new_token0 < tokens && key0 >= first_pos;
            const bool from_new1 = new_token1 >= 0 && new_token1 < tokens && key1 >= first_pos;
            score[nt][0]         = (row0 < row_count && key0 < split_end && key0 <= qabs0 &&
                            !(from_new0 && new_token0 > token0))
                                       ? score[nt][0] * scale
                                       : -CUDART_INF_F;
            score[nt][1]         = (row0 < row_count && key1 < split_end && key1 <= qabs0 &&
                            !(from_new1 && new_token1 > token0))
                                       ? score[nt][1] * scale
                                       : -CUDART_INF_F;
            score[nt][2]         = (row1 < row_count && key0 < split_end && key0 <= qabs1 &&
                            !(from_new0 && new_token0 > token1))
                                       ? score[nt][2] * scale
                                       : -CUDART_INF_F;
            score[nt][3]         = (row1 < row_count && key1 < split_end && key1 <= qabs1 &&
                            !(from_new1 && new_token1 > token1))
                                       ? score[nt][3] * scale
                                       : -CUDART_INF_F;
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0    = fmaxf(m0, bm0);
        const float nm1    = fmaxf(m1, bm1);
        const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
        const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0  = nt * 8 + 2 * lid;
            const int col1  = col0 + 1;
            const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                  : 0.0f;
            const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                  : 0.0f;
            const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                  : 0.0f;
            const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                  ? exp2_approx((score[nt][3] - nm1) * Log2E)
                                  : 0.0f;
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col0)]           = __float2bfloat16(p00);
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col1)]           = __float2bfloat16(p01);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0)] = __float2bfloat16(p10);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)] = __float2bfloat16(p11);
        }
        bl0 = warp_sum<4>(bl0, FullMask);
        bl1 = warp_sum<4>(bl1, FullMask);

        l0 = l0 * alpha0 + bl0;
        l1 = l1 * alpha1 + bl1;
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }
        __syncwarp();

#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(
                    pf[0], pf[1], pf[2], pf[3],
                    smem_addr(
                        &p_sw[a_rowoff * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = n * 8;
                ldmatrix_x2_t(
                    vf[0], vf[1],
                    smem_addr(&v_s[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0],
                                                 pf[1], pf[2], pf[3], vf[0], vf[1]);
            }
        }
        __syncthreads();
    }

    if (lid == 0) {
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt(row0, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index(q_head, token, split, tokens)] = m0;
            partial_l[gqa_partial_stat_index(q_head, token, split, tokens)] = l0;
        }
        if (row1 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt(row1, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index(q_head, token, split, tokens)] = m1;
            partial_l[gqa_partial_stat_index(q_head, token, split, tokens)] = l1;
        }
    }

    // MMA fragments hold each row in four-lane groups. Stage the final split-local
    // accumulator through shared memory so partial_acc is written as contiguous d-vector stores.
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0   = n * 8 + 2 * lid;
        const int d1   = d0 + 1;
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            qkv_s[row0 * D + d0] = __float2bfloat16(acc[n][0]);
            qkv_s[row0 * D + d1] = __float2bfloat16(acc[n][1]);
        }
        if (row1 < row_count) {
            qkv_s[row1 * D + d0] = __float2bfloat16(acc[n][2]);
            qkv_s[row1 * D + d1] = __float2bfloat16(acc[n][3]);
        }
    }
    __syncthreads();

    for (int chunk = tid; chunk < row_count * (D / 8); chunk += Threads) {
        const int row = chunk / (D / 8);
        const int d   = (chunk - row * (D / 8)) * 8;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt(row, tokens, kv_head, q_head, token);
        if (gqa_valid_q_head(kv_head, q_head)) {
            const std::int64_t dst = gqa_partial_acc_index(q_head, d, token, split, tokens);
            store_vec(&partial_acc[dst], load_vec<int4>(&qkv_s[row * D + d]));
        }
    }
}

} // namespace ninfer::kernels
