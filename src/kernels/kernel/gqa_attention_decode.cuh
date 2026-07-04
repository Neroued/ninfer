#pragma once

// qus::kernels - split-KV GQA small-T attention. The partial kernel processes
// one KV head, one query-head subgroup, and one token tile. A reducer combines
// split-local online-softmax partials into the final BF16 output.

#include <cuda_bf16.h>
#include <math_constants.h>

#include "kernels/kernel/gdn_common.cuh"

#include <cstdint>

namespace qus::kernels {

inline constexpr int kGqaHeadDim   = 256;
inline constexpr int kGqaQHeads    = 24;
inline constexpr int kGqaKVHeads   = 4;
inline constexpr int kGqaGroupSize = 6;

__device__ __forceinline__ std::int64_t gqa_cache_index(int kv_head, int d, int position,
                                                        int padded_context) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaHeadDim) *
                                              (static_cast<std::int64_t>(position) +
                                               static_cast<std::int64_t>(padded_context) * kv_head);
}

__device__ __forceinline__ std::int64_t gqa_q_index(int q_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaHeadDim) *
               (static_cast<std::int64_t>(q_head) + static_cast<std::int64_t>(kGqaQHeads) * token);
}

__device__ __forceinline__ std::int64_t gqa_kv_new_index(int kv_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaHeadDim) *
                                              (static_cast<std::int64_t>(kv_head) +
                                               static_cast<std::int64_t>(kGqaKVHeads) * token);
}

__device__ __forceinline__ std::int64_t gqa_partial_acc_index(int q_head, int d, int token,
                                                              int split, int tokens) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(kGqaQHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split));
}

__device__ __forceinline__ std::int64_t gqa_partial_stat_index(int q_head, int token, int split,
                                                               int tokens) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(kGqaQHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split);
}

__device__ __forceinline__ float gqa_exp2_fast(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ bool gqa_valid_q_head(int kv_head, int q_head) {
    return kv_head >= 0 && kv_head < kGqaKVHeads && q_head >= kv_head * kGqaGroupSize &&
           q_head < (kv_head + 1) * kGqaGroupSize && q_head < kGqaQHeads;
}

__device__ __forceinline__ int gqa_small_t_active_splits(int window, int max_splits, int tokens) {
    if (tokens <= 1) { return max_splits; }
    if (window <= 0) { return max_splits; }
    const int target_keys_per_split = (tokens <= 5) ? 480 : 512;
    constexpr int kMinSplits        = 4;
    int splits = (window + target_keys_per_split - 1) / target_keys_per_split;
    splits     = max(kMinSplits, splits);
    return min(max_splits, splits);
}

__device__ __forceinline__ int gqa_small_t_tc_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ int gqa_small_t_tc_swz32(int row, int col) {
    return (((col >> 3) ^ (row & 3)) << 3) | (col & 7);
}

__device__ __forceinline__ unsigned gqa_small_t_tc_smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void gqa_small_t_tc_ldmatrix_x4(unsigned& a0, unsigned& a1,
                                                           unsigned& a2, unsigned& a3,
                                                           unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                 : "r"(addr));
}

__device__ __forceinline__ void gqa_small_t_tc_ldmatrix_x2(unsigned& b0, unsigned& b1,
                                                           unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b0), "=r"(b1)
                 : "r"(addr));
}

__device__ __forceinline__ void gqa_small_t_tc_ldmatrix_x2_trans(unsigned& b0, unsigned& b1,
                                                                 unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b0), "=r"(b1)
                 : "r"(addr));
}

__device__ __forceinline__ void
gqa_small_t_tc_mma_m16n8k16_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                 unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                 unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void
gqa_small_t_tc_row_to_qt(int row, int tokens, int kv_head, int& q_head, int& token) {
    token             = row / kGqaGroupSize;
    const int local_q = row - token * kGqaGroupSize;
    q_head            = kv_head * kGqaGroupSize + local_q;
}

template <int TokenTile, int WarpsPerCta>
__launch_bounds__(128, 2) __global__
    void gqa_attention_small_t_tc_partial_kernel(const __nv_bfloat16* q,
                                                 const __nv_bfloat16* k_new,
                                                 const __nv_bfloat16* v_new,
                                                 const std::int32_t* pos,
                                                 __nv_bfloat16* cache_k,
                                                 __nv_bfloat16* cache_v, std::int32_t tokens,
                                                 std::int32_t padded_context,
                                                 std::int32_t max_context, float scale,
                                                 __nv_bfloat16* partial_acc, float* partial_m,
                                                 float* partial_l) {
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
    const int active_split_count = gqa_small_t_active_splits(window, split_count, tokens);
    if (split >= active_split_count) { return; }

    const int kps         = (window + active_split_count - 1) / active_split_count;
    const int split_start = split * kps;
    const int split_limit = split_start + kps;
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }

    for (int chunk = tid; chunk < tokens * (D / 8); chunk += Threads) {
        const int token = chunk / (D / 8);
        const int d     = (chunk - token * (D / 8)) * 8;
        const int p_tok = pos[token];
        if (p_tok >= split_start && p_tok < split_end && p_tok >= 0 && p_tok < max_context) {
            const std::int64_t new_off   = gqa_kv_new_index(kv_head, d, token);
            const std::int64_t cache_off = gqa_cache_index(kv_head, d, p_tok, padded_context);
            *reinterpret_cast<int4*>(&cache_k[cache_off]) =
                *reinterpret_cast<const int4*>(&k_new[new_off]);
            *reinterpret_cast<int4*>(&cache_v[cache_off]) =
                *reinterpret_cast<const int4*>(&v_new[new_off]);
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
        gqa_small_t_tc_ldmatrix_x4(
            af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
            gqa_small_t_tc_smem_addr(&qkv_s[arow * D + gqa_small_t_tc_swz(arow, acol)]));
    }
    __syncthreads();

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    const int key_blocks = (split_end - split_start + Bc - 1) / Bc;
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = split_start + kb * Bc;
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int key_l      = chunk / (D / 8);
            const int d          = (chunk - key_l * (D / 8)) * 8;
            const int key        = k0 + key_l;
            __nv_bfloat16* k_dst = &k_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            __nv_bfloat16* v_dst = &v_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            if (key < split_end) {
                const int new_token = key - first_pos;
                if (new_token >= 0 && new_token < tokens && key >= first_pos) {
                    const std::int64_t off = gqa_kv_new_index(kv_head, d, new_token);
                    qus::kernels::async_copy_global_to_shared<16>(k_dst, &k_new[off]);
                    qus::kernels::async_copy_global_to_shared<16>(v_dst, &v_new[off]);
                } else {
                    const std::int64_t off = gqa_cache_index(kv_head, d, key, padded_context);
                    qus::kernels::async_copy_global_to_shared<16>(k_dst, &cache_k[off]);
                    qus::kernels::async_copy_global_to_shared<16>(v_dst, &cache_v[off]);
                }
            } else {
                *reinterpret_cast<int4*>(k_dst) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4*>(v_dst) = make_int4(0, 0, 0, 0);
            }
        }
        qus::kernels::async_copy_commit();
        qus::kernels::async_copy_wait<0>();
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
                gqa_small_t_tc_ldmatrix_x2(
                    bf[0], bf[1],
                    gqa_small_t_tc_smem_addr(&k_s[brow * D + gqa_small_t_tc_swz(brow, bcol)]));
                gqa_small_t_tc_mma_m16n8k16_bf16(score[nt][0], score[nt][1], score[nt][2],
                                                 score[nt][3], af_q[k][0], af_q[k][1],
                                                 af_q[k][2], af_q[k][3], bf[0], bf[1]);
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
            const int col0       = nt * 8 + 2 * lid;
            const int col1       = col0 + 1;
            const int key0       = k0 + col0;
            const int key1       = col1 + k0;
            const int new_token0 = key0 - first_pos;
            const int new_token1 = key1 - first_pos;
            const bool from_new0 = new_token0 >= 0 && new_token0 < tokens && key0 >= first_pos;
            const bool from_new1 = new_token1 >= 0 && new_token1 < tokens && key1 >= first_pos;
            score[nt][0] =
                (row0 < row_count && key0 < split_end && key0 <= qabs0 &&
                 !(from_new0 && new_token0 > token0))
                    ? score[nt][0] * scale
                    : -CUDART_INF_F;
            score[nt][1] =
                (row0 < row_count && key1 < split_end && key1 <= qabs0 &&
                 !(from_new1 && new_token1 > token0))
                    ? score[nt][1] * scale
                    : -CUDART_INF_F;
            score[nt][2] =
                (row1 < row_count && key0 < split_end && key0 <= qabs1 &&
                 !(from_new0 && new_token0 > token1))
                    ? score[nt][2] * scale
                    : -CUDART_INF_F;
            score[nt][3] =
                (row1 < row_count && key1 < split_end && key1 <= qabs1 &&
                 !(from_new1 && new_token1 > token1))
                    ? score[nt][3] * scale
                    : -CUDART_INF_F;
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = fmaxf(bm0, __shfl_xor_sync(FullMask, bm0, 1));
        bm0 = fmaxf(bm0, __shfl_xor_sync(FullMask, bm0, 2));
        bm1 = fmaxf(bm1, __shfl_xor_sync(FullMask, bm1, 1));
        bm1 = fmaxf(bm1, __shfl_xor_sync(FullMask, bm1, 2));

        const float nm0 = fmaxf(m0, bm0);
        const float nm1 = fmaxf(m1, bm1);
        const float alpha0 =
            (m0 == -CUDART_INF_F) ? 0.0f : gqa_exp2_fast((m0 - nm0) * Log2E);
        const float alpha1 =
            (m1 == -CUDART_INF_F) ? 0.0f : gqa_exp2_fast((m1 - nm1) * Log2E);

        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0  = nt * 8 + 2 * lid;
            const int col1  = col0 + 1;
            const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                  ? gqa_exp2_fast((score[nt][0] - nm0) * Log2E)
                                  : 0.0f;
            const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                  ? gqa_exp2_fast((score[nt][1] - nm0) * Log2E)
                                  : 0.0f;
            const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                  ? gqa_exp2_fast((score[nt][2] - nm1) * Log2E)
                                  : 0.0f;
            const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                  ? gqa_exp2_fast((score[nt][3] - nm1) * Log2E)
                                  : 0.0f;
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col0)]           = __float2bfloat16(p00);
            p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col1)]           = __float2bfloat16(p01);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0)] = __float2bfloat16(p10);
            p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)] = __float2bfloat16(p11);
        }
        bl0 += __shfl_xor_sync(FullMask, bl0, 1);
        bl0 += __shfl_xor_sync(FullMask, bl0, 2);
        bl1 += __shfl_xor_sync(FullMask, bl1, 1);
        bl1 += __shfl_xor_sync(FullMask, bl1, 2);

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
                gqa_small_t_tc_ldmatrix_x4(
                    pf[0], pf[1], pf[2], pf[3],
                    gqa_small_t_tc_smem_addr(
                        &p_sw[a_rowoff * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = n * 8;
                gqa_small_t_tc_ldmatrix_x2_trans(
                    vf[0], vf[1],
                    gqa_small_t_tc_smem_addr(&v_s[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                gqa_small_t_tc_mma_m16n8k16_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3],
                                                 pf[0], pf[1], pf[2], pf[3], vf[0], vf[1]);
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
            *reinterpret_cast<int4*>(&partial_acc[dst]) =
                *reinterpret_cast<const int4*>(&qkv_s[row * D + d]);
        }
    }
}

template <int DChunk>
__launch_bounds__(256) __global__
    void gqa_attention_small_t_reduce_output_kernel(const __nv_bfloat16* partial_acc,
                                                    const float* partial_m, const float* partial_l,
                                                    const std::int32_t* positions,
                                                    std::int32_t tokens, std::int32_t split_count,
                                                    __nv_bfloat16* out) {
    static_assert(DChunk > 0 && DChunk <= kGqaHeadDim);

    const int q_head  = static_cast<int>(blockIdx.x);
    const int d_start = static_cast<int>(blockIdx.y) * DChunk;
    const int token   = static_cast<int>(blockIdx.z);
    const int tid     = threadIdx.x;
    if (q_head >= kGqaQHeads || token >= tokens) { return; }
    const int last_pos           = positions[tokens - 1];
    const int window             = last_pos + 1;
    const int active_split_count = gqa_small_t_active_splits(window, split_count, tokens);

    __shared__ float reduce[256];

    float local_m = -CUDART_INF_F;
    for (int split = tid; split < active_split_count; split += blockDim.x) {
        local_m = fmaxf(local_m, partial_m[gqa_partial_stat_index(q_head, token, split, tokens)]);
    }
    reduce[tid] = local_m;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]); }
        __syncthreads();
    }
    const float head_m = reduce[0];
    __syncthreads();

    if (head_m == -CUDART_INF_F) {
        const int d = d_start + tid;
        if (tid < DChunk && d < kGqaHeadDim) {
            out[gqa_q_index(q_head, d, token)] = __float2bfloat16(0.0f);
        }
        return;
    }

    float local_l = 0.0f;
    for (int split = tid; split < active_split_count; split += blockDim.x) {
        const float tile_l = partial_l[gqa_partial_stat_index(q_head, token, split, tokens)];
        if (tile_l > 0.0f) {
            local_l +=
                tile_l *
                expf(partial_m[gqa_partial_stat_index(q_head, token, split, tokens)] - head_m);
        }
    }
    reduce[tid] = local_l;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] += reduce[tid + stride]; }
        __syncthreads();
    }
    const float head_l = reduce[0];

    const int d = d_start + tid;
    if (tid >= DChunk || d >= kGqaHeadDim) { return; }

    float numerator = 0.0f;
    if (head_l > 0.0f) {
        for (int split = 0; split < active_split_count; ++split) {
            const float tile_l = partial_l[gqa_partial_stat_index(q_head, token, split, tokens)];
            if (tile_l <= 0.0f) { continue; }
            const float weight =
                expf(partial_m[gqa_partial_stat_index(q_head, token, split, tokens)] - head_m);
            numerator += __bfloat162float(
                             partial_acc[gqa_partial_acc_index(q_head, d, token, split, tokens)]) *
                         weight;
        }
    }
    const float value                  = (head_l > 0.0f) ? numerator / head_l : 0.0f;
    out[gqa_q_index(q_head, d, token)] = __float2bfloat16(value);
}

} // namespace qus::kernels
