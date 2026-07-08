#pragma once

// qus::kernels - split-KV GQA small-T attention, int8 KV-cache partial kernel.
// Standalone from the bf16 kernel (gqa_attention_decode_bf16.cuh): shared
// scaffolding lives in gqa_attention_decode.cuh, but the body/append/load are not
// shared so the int8 path can be tuned independently. Processes one KV head, one
// query-head subgroup, and one token tile; a reducer combines the split-local
// partials.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "kernels/kernel/gdn_common.cuh"
#include "kernels/kernel/gqa_attention_decode.cuh"
#include "kernels/kernel/gqa_attention_kv_quant.cuh"

#include <cstdint>

namespace qus::kernels {

// Double-buffered int8 staging for the decode convert pipeline. The Bc=32 key tile
// is loaded as 4 sub-tiles of Bc/2=16 keys (K-lo, K-hi, V-lo, V-hi); two 4 KB
// buffers let the next sub-tile's cp.async overlap the current sub-tile's convert,
// so DRAM stays busy at the low (~4-warp) decode occupancy. Both planes' fp16 group
// scales (2*Bc*groups) are staged once up front so the convert touches only shared
// memory. Code buffers (2*16*256 = 8 KB) + scales (256*2 = 512 B) = 8704 B; with
// 36 KB static this stays under the 48 KB non-opt-in smem limit and keeps 2 blocks/SM
// on sm_120's 100 KB/SM budget, matching the bf16 kernel (which uses 0 dynamic smem).
inline constexpr int kGqaSmallTBc              = 32;
inline constexpr int kGqaSmallTSubKeys         = kGqaSmallTBc / 2;
inline constexpr int kGqaSmallTSubBytesI8      = kGqaSmallTSubKeys * kGqaHeadDim; // 4 KB
inline constexpr int kGqaSmallTStageScaleCount = 2 * kGqaSmallTBc * kGqaKvQuantGroups;
inline constexpr int kGqaSmallTStageBytesI8 =
    2 * kGqaSmallTSubBytesI8 +
    kGqaSmallTStageScaleCount * static_cast<int>(sizeof(__half));

template <int TokenTile, int WarpsPerCta>
__launch_bounds__(128, 2) __global__ void gqa_attention_small_t_tc_partial_i8_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
    const std::int32_t* pos, std::int8_t* cache_k_i8, std::int8_t* cache_v_i8,
    __half* cache_k_scale, __half* cache_v_scale, std::int32_t tokens, std::int32_t padded_context,
    std::int32_t max_context, float scale, __nv_bfloat16* partial_acc, float* partial_m,
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

    // Dynamic staging for int8 codes + scales (see kGqaSmallTStageBytesI8).
    extern __shared__ __align__(16) std::int8_t gqa_decode_stage[];

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
    const int active_split_count = gqa_small_t_active_splits(window, split_count);
    if (split >= active_split_count) { return; }

    const int kps         = (window + active_split_count - 1) / active_split_count;
    const int split_start = split * kps;
    const int split_limit = split_start + kps;
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }

    // Fused int8 append (no separate kernel). The split that owns each new token's
    // absolute position quantizes its K/V head vector per 64-group (warp absmax ->
    // int8 code + fp16 scale) straight into the cache for FUTURE decode steps. The
    // current step still reads the new tokens from k_new/v_new at full precision via
    // the from_new path below, so no split reads back these just-written codes in the
    // same launch: there is no cross-CTA race, exactly like the bf16 fused write.
    for (int pair = warp; pair < tokens * kGqaKvQuantGroups; pair += Wc) {
        const int t        = pair / kGqaKvQuantGroups;
        const int grp      = pair - t * kGqaKvQuantGroups;
        const int position = pos[t];
        if (position < split_start || position >= split_end || position >= max_context) {
            continue;
        }
        const int d0            = grp * kGqaKvQuantGroup + lane;
        const int d1            = d0 + 32;
        const std::int64_t src0 = gqa_kv_new_index(kv_head, d0, t);
        const std::int64_t src1 = gqa_kv_new_index(kv_head, d1, t);
        const float kv0         = __bfloat162float(k_new[src0]);
        const float kv1         = __bfloat162float(k_new[src1]);
        const float vv0         = __bfloat162float(v_new[src0]);
        const float vv1         = __bfloat162float(v_new[src1]);
        float kamax             = fmaxf(fabsf(kv0), fabsf(kv1));
        float vamax             = fmaxf(fabsf(vv0), fabsf(vv1));
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            kamax = fmaxf(kamax, __shfl_xor_sync(FullMask, kamax, off));
            vamax = fmaxf(vamax, __shfl_xor_sync(FullMask, vamax, off));
        }
        // Match the bf16-parity oracle: divide by 127, round the scale to fp16,
        // then quantize with the inverse of that stored fp16 scale.
        const __half ksh  = __float2half_rn(kamax > 0.0f ? kamax / 127.0f : 0.0f);
        const __half vsh  = __float2half_rn(vamax > 0.0f ? vamax / 127.0f : 0.0f);
        const float ks    = __half2float(ksh);
        const float vs    = __half2float(vsh);
        const float k_inv = ks > 0.0f ? 1.0f / ks : 0.0f;
        const float v_inv = vs > 0.0f ? 1.0f / vs : 0.0f;
        cache_k_i8[gqa_kv_quant_code_index(kv_head, d0, position, padded_context)] =
            gqa_kv_quant_code(kv0, k_inv);
        cache_k_i8[gqa_kv_quant_code_index(kv_head, d1, position, padded_context)] =
            gqa_kv_quant_code(kv1, k_inv);
        cache_v_i8[gqa_kv_quant_code_index(kv_head, d0, position, padded_context)] =
            gqa_kv_quant_code(vv0, v_inv);
        cache_v_i8[gqa_kv_quant_code_index(kv_head, d1, position, padded_context)] =
            gqa_kv_quant_code(vv1, v_inv);
        if (lane == 0) {
            const std::int64_t so =
                gqa_kv_quant_scale_index(kv_head, grp, position, padded_context);
            cache_k_scale[so] = ksh;
            cache_v_scale[so] = vsh;
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
        // Double-buffered convert pipeline. The Bc key tile is 4 sub-tiles of
        // SubKeys=Bc/2 keys: (K-lo, K-hi, V-lo, V-hi). Two 4 KB buffers let the next
        // sub-tile's int8 cp.async (half the DRAM bytes of bf16, fully coalesced)
        // overlap the current sub-tile's dequant+store, so DRAM stays busy even at the
        // ~4-warp decode occupancy. Both planes' fp16 group scales are staged up front
        // so the convert touches only shared memory.
        constexpr int kSub    = 4;
        constexpr int SubKeys = Bc / 2;
        std::int8_t* buf0     = gqa_decode_stage;
        std::int8_t* buf1     = gqa_decode_stage + kGqaSmallTSubBytesI8;
        __half* stage_scale = reinterpret_cast<__half*>(&gqa_decode_stage[2 * kGqaSmallTSubBytesI8]);

        for (int i = tid; i < 2 * Bc * kGqaKvQuantGroups; i += Threads) {
            const int pl      = i / (Bc * kGqaKvQuantGroups);
            const int rem     = i - pl * (Bc * kGqaKvQuantGroups);
            const int key_l   = rem / kGqaKvQuantGroups;
            const int grp     = rem - key_l * kGqaKvQuantGroups;
            const int key     = k0 + key_l;
            const __half* csc = (pl == 0) ? cache_k_scale : cache_v_scale;
            stage_scale[i]    = (key < split_end)
                                 ? csc[gqa_kv_quant_scale_index(kv_head, grp, key, padded_context)]
                                 : __float2half(0.0f);
        }

        auto load_sub = [&](int s) {
            const int plane             = s >> 1;
            const int key0              = (s & 1) * SubKeys;
            const std::int8_t* cache_i8 = (plane == 0) ? cache_k_i8 : cache_v_i8;
            std::int8_t* buf            = (s & 1) ? buf1 : buf0;
#pragma unroll 1
            for (int chunk = tid; chunk < SubKeys * (D / 16); chunk += Threads) {
                const int kl  = chunk / (D / 16);
                const int d   = (chunk - kl * (D / 16)) * 16;
                const int key = k0 + key0 + kl;
                if (key < split_end) {
                    const int nt        = key - first_pos;
                    const bool from_new = nt >= 0 && nt < tokens && key >= first_pos;
                    if (!from_new) {
                        const std::int64_t off =
                            gqa_kv_quant_code_index(kv_head, d, key, padded_context);
                        qus::kernels::async_copy_global_to_shared<16>(&buf[kl * D + d],
                                                                      &cache_i8[off]);
                    }
                }
            }
            qus::kernels::async_copy_commit();
        };

        auto convert_sub = [&](int s) {
            const int plane              = s >> 1;
            const int key0               = (s & 1) * SubKeys;
            const __nv_bfloat16* src_new = (plane == 0) ? k_new : v_new;
            __nv_bfloat16* dst_tile      = (plane == 0) ? k_s : v_s;
            const std::int8_t* buf       = (s & 1) ? buf1 : buf0;
            const int scale_base         = plane * (Bc * kGqaKvQuantGroups);
#pragma unroll 1
            for (int chunk = tid; chunk < SubKeys * (D / 8); chunk += Threads) {
                const int kl       = chunk / (D / 8);
                const int d        = (chunk - kl * (D / 8)) * 8;
                const int key_l    = key0 + kl;
                const int key      = k0 + key_l;
                __nv_bfloat16* dst = &dst_tile[key_l * D + gqa_small_t_tc_swz(key_l, d)];
                if (key < split_end) {
                    const int nt        = key - first_pos;
                    const bool from_new = nt >= 0 && nt < tokens && key >= first_pos;
                    if (from_new) {
                        const std::int64_t off = gqa_kv_new_index(kv_head, d, nt);
                        *reinterpret_cast<int4*>(dst) =
                            *reinterpret_cast<const int4*>(&src_new[off]);
                    } else {
                        const int grp = d >> 6;
                        const float s = __half2float(
                            stage_scale[scale_base + key_l * kGqaKvQuantGroups + grp]);
                        *reinterpret_cast<int4*>(dst) =
                            gqa_kv_dequant_i8x8_from(&buf[kl * D + d], s);
                    }
                } else {
                    *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
                }
            }
        };

        load_sub(0);
#pragma unroll 1
        for (int s = 0; s < kSub; ++s) {
            if (s + 1 < kSub) {
                load_sub(s + 1);
                qus::kernels::async_copy_wait<1>();
            } else {
                qus::kernels::async_copy_wait<0>();
            }
            __syncthreads();
            convert_sub(s);
            if (s + 1 < kSub) { __syncthreads(); } // WAR before this buffer is reloaded
        }
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
            // Both bf16 and int8 read the current-step tokens from k_new/v_new, so the
            // causal mask is identical: drop keys past the query and any batch-mate new
            // token that sits after this row's own token.
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
        bm0 = fmaxf(bm0, __shfl_xor_sync(FullMask, bm0, 1));
        bm0 = fmaxf(bm0, __shfl_xor_sync(FullMask, bm0, 2));
        bm1 = fmaxf(bm1, __shfl_xor_sync(FullMask, bm1, 1));
        bm1 = fmaxf(bm1, __shfl_xor_sync(FullMask, bm1, 2));

        const float nm0    = fmaxf(m0, bm0);
        const float nm1    = fmaxf(m1, bm1);
        const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : gqa_exp2_fast((m0 - nm0) * Log2E);
        const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : gqa_exp2_fast((m1 - nm1) * Log2E);

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
                gqa_small_t_tc_mma_m16n8k16_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0],
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
            *reinterpret_cast<int4*>(&partial_acc[dst]) =
                *reinterpret_cast<const int4*>(&qkv_s[row * D + d]);
        }
    }
}

} // namespace qus::kernels
