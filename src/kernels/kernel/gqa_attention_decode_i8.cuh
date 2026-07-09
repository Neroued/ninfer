#pragma once

// qus::kernels - split-KV GQA small-T attention, int8 KV-cache partial kernel.
// int8-native redesign (docs/2026-07-08-gqa-decode-int8-kernel-redesign.md):
//
//   * QK runs on native m16n8k32.s8 tensor cores. Q is quantized on-chip to int8
//     per (row, 64-group); K stays int8 in the cache and is read straight into
//     smem (no dequant). The int32 MMA output is rescaled per 64-group by
//     qs[row,g]*ks[key,g]. This halves the QK MMA count vs bf16 and removes the
//     entire K dequant.
//   * PV stays bf16 (V is quantized per key, so its scale cannot be factored out
//     of a key-contracted int8 accumulation): V int8 is staged, dequanted once to
//     a bf16 tile, then the existing bf16 PV MMA runs. V is still read from DRAM
//     as int8, so the bandwidth win is kept.
//   * All keys (history AND the current/diagonal tokens) are read from the
//     quantized cache; the fused append writes the new tokens first and a
//     __syncthreads orders the in-block readback. No from_new special-casing.
//
// Standalone from the bf16 kernel; shared scaffolding (layout constants, ldmatrix
// helpers, the s8/bf16 MMA helpers, the reducer) lives in gqa_attention_decode.cuh.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "kernels/kernel/gdn_common.cuh"
#include "kernels/kernel/gqa_attention_decode.cuh"
#include "kernels/kernel/gqa_attention_kv_quant.cuh"

#include <cstdint>

namespace qus::kernels {

// Store one int8 code into a d-contiguous-as-b16 swizzled tile so the same
// gqa_small_t_tc_swz / ldmatrix path that serves bf16 tiles serves the int8 tile.
// A b16 lane holds two packed int8 (d even = low byte, d odd = high byte); this
// matches the byte layout a 16 B cp.async of d-contiguous cache bytes produces
// (see the design doc / kernel comments), so Q (byte stores) and K (cp.async)
// agree.
__device__ __forceinline__ void gqa_small_t_i8_store_swz(std::int8_t* tile, int row, int d,
                                                         int d_b16_stride, std::int8_t code) {
    const int c   = d >> 1;
    const int lo  = d & 1;
    const int off = (row * d_b16_stride + gqa_small_t_tc_swz(row, c)) * 2 + lo;
    tile[off]     = code;
}

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
    constexpr int D             = kGqaHeadDim;         // 256
    constexpr int DB16          = D / 2;               // 128 b16 per int8 row
    constexpr int Threads       = Wc * 32;
    constexpr int Groups        = kGqaKvQuantGroups;   // 4 (64-dim groups)
    constexpr int GroupKc       = kGqaKvQuantGroup / 32; // 2 s8 k-chunks per group
    constexpr int QKKs          = D / 32;              // 8 s8 QK k-chunks
    constexpr int QKNt          = Bc / 8;              // 4 QK n-tiles
    constexpr int PVNt          = D / 8;               // 32 PV n-tiles
    constexpr int PVKs          = Bc / 16;             // 2 PV contraction steps
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(QKKs == Groups * GroupKc);
    static_assert(2 * Bc >= Br);

    // One reused byte arena: K int8 | V int8 | V bf16 during the key loop; Q int8
    // in the prologue; the acc bf16 staging in the epilogue. 4*Bc*D bytes covers
    // Q int8 (Br*D <= 4*Bc*D) and acc staging (row_count*D*2 <= 4*Bc*D).
    constexpr int RBytes = 4 * Bc * D;
    __shared__ __align__(16) std::int8_t r_s[RBytes];
    std::int8_t* k_i8       = r_s;              // [Bc, D] swizzled (as b16)
    std::int8_t* v_i8       = r_s + Bc * D;     // [Bc, D] d-contiguous stage
    __nv_bfloat16* v_bf16   = reinterpret_cast<__nv_bfloat16*>(r_s + 2 * Bc * D); // [Bc, D] swz
    std::int8_t* q_i8       = r_s;              // prologue: [Br, D] swizzled (as b16)
    __nv_bfloat16* q_b16    = reinterpret_cast<__nv_bfloat16*>(r_s);
    __nv_bfloat16* k_b16    = reinterpret_cast<__nv_bfloat16*>(k_i8);
    __nv_bfloat16* acc_bf16 = reinterpret_cast<__nv_bfloat16*>(r_s); // epilogue staging

    __shared__ __align__(16) __nv_bfloat16 p_s[Wc * 16 * Bc];
    __shared__ __half k_scale_s[Bc * Groups];
    __shared__ __half v_scale_s[Bc * Groups];
    __shared__ float q_scale_s[2 * Bc * Groups]; // rows < Br <= 2*Bc

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

    // Fused int8 append: the split that owns each new token's absolute position
    // quantizes its K/V head vector per 64-group into the cache. The current step
    // then reads those codes back from the cache (below) after a __syncthreads;
    // each position belongs to exactly one split so there is no cross-CTA race.
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

    // Prologue: quantize Q into the int8 tile (per row, per 64-group) and stash the
    // per-(row,group) fp32 scale. Zero first so invalid rows/scales are 0 (their
    // scores go through 0 code -> 0 -> masked).
    for (int i = tid; i < Br * D; i += Threads) { q_i8[i] = 0; }
    for (int i = tid; i < Br * Groups; i += Threads) { q_scale_s[i] = 0.0f; }
    __syncthreads();

    for (int unit = warp; unit < row_count * Groups; unit += Wc) {
        const int row = unit / Groups;
        const int g   = unit - row * Groups;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt(row, tokens, kv_head, q_head, token);
        if (!gqa_valid_q_head(kv_head, q_head)) { continue; }
        const int d0    = g * kGqaKvQuantGroup + lane;
        const int d1    = d0 + 32;
        const float x0  = __bfloat162float(q[gqa_q_index(q_head, d0, token)]);
        const float x1  = __bfloat162float(q[gqa_q_index(q_head, d1, token)]);
        float amax      = fmaxf(fabsf(x0), fabsf(x1));
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(FullMask, amax, off));
        }
        const float qs  = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
        gqa_small_t_i8_store_swz(q_i8, row, d0, DB16, gqa_kv_quant_code(x0, inv));
        gqa_small_t_i8_store_swz(q_i8, row, d1, DB16, gqa_kv_quant_code(x1, inv));
        if (lane == 0) { q_scale_s[row * Groups + g] = qs; }
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

    // Q A-fragments (int8, loaded once as b16 via ldmatrix.x4).
    unsigned af_q[QKKs][4];
#pragma unroll
    for (int k = 0; k < QKKs; ++k) {
        const int arow = warp_row0 + a_rowoff;
        const int acol = k * 16 + a_coloff;
        gqa_small_t_tc_ldmatrix_x4(af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
                                   gqa_small_t_tc_smem_addr(&q_b16[arow * DB16 +
                                                                   gqa_small_t_tc_swz(arow, acol)]));
    }
    __syncthreads(); // q_i8 (== r_s) is now free for the K/V tiles

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

        // Single-wave stage: K int8 (swizzled) + V int8 (d-contiguous) + fp16
        // scales in one cp.async group.
        for (int i = tid; i < Bc * Groups; i += Threads) {
            const int key_l = i / Groups;
            const int g     = i - key_l * Groups;
            const int key   = k0 + key_l;
            if (key < split_end) {
                k_scale_s[i] = cache_k_scale[gqa_kv_quant_scale_index(kv_head, g, key,
                                                                      padded_context)];
                v_scale_s[i] = cache_v_scale[gqa_kv_quant_scale_index(kv_head, g, key,
                                                                      padded_context)];
            } else {
                k_scale_s[i] = __float2half(0.0f);
                v_scale_s[i] = __float2half(0.0f);
            }
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int key_l = chunk / (D / 16);
            const int dc    = chunk - key_l * (D / 16);
            const int d     = dc * 16;
            const int key   = k0 + key_l;
            if (key < split_end) {
                const std::int64_t off = gqa_kv_quant_code_index(kv_head, d, key, padded_context);
                // K -> swizzled (b16 block start = swz(key_l, dc*8)); V -> d-contiguous.
                std::int8_t* kdst = &k_i8[key_l * D + gqa_small_t_tc_swz(key_l, dc * 8) * 2];
                qus::kernels::async_copy_global_to_shared<16>(kdst, &cache_k_i8[off]);
                qus::kernels::async_copy_global_to_shared<16>(&v_i8[key_l * D + d],
                                                              &cache_v_i8[off]);
            }
        }
        qus::kernels::async_copy_commit();
        qus::kernels::async_copy_wait<0>();
        __syncthreads();

        // Dequant V int8 -> bf16 tile (swizzled, PV B operand).
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int key_l    = chunk / (D / 8);
            const int d        = (chunk - key_l * (D / 8)) * 8;
            __nv_bfloat16* dst = &v_bf16[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            if (k0 + key_l < split_end) {
                const int grp = d >> 6;
                const float s = __half2float(v_scale_s[key_l * Groups + grp]);
                *reinterpret_cast<int4*>(dst) = gqa_kv_dequant_i8x8_from(&v_i8[key_l * D + d], s);
            } else {
                *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // S = Q Kᵀ in int8: per 64-group two m16n8k32.s8 MMAs into int32, then
        // rescale by qs[row,g]*ks[key,g] into the fp32 score.
        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            const int keya = nt * 8 + 2 * lid;
            const int keyb = keya + 1;
#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int k    = g * GroupKc + kk;
                    const int brow = nt * 8 + b_rin;
                    const int bcol = k * 16 + b_koff;
                    unsigned bf[2];
                    gqa_small_t_tc_ldmatrix_x2(
                        bf[0], bf[1],
                        gqa_small_t_tc_smem_addr(&k_b16[brow * DB16 +
                                                        gqa_small_t_tc_swz(brow, bcol)]));
                    gqa_small_t_tc_mma_m16n8k32_s8(c0, c1, c2, c3, af_q[k][0], af_q[k][1],
                                                   af_q[k][2], af_q[k][3], bf[0], bf[1]);
                }
                const float qg0 = q_scale_s[(warp_row0 + gid) * Groups + g];
                const float qg1 = q_scale_s[(warp_row0 + gid + 8) * Groups + g];
                const float ka  = __half2float(k_scale_s[keya * Groups + g]);
                const float kb2 = __half2float(k_scale_s[keyb * Groups + g]);
                s0 += qg0 * ka * static_cast<float>(c0);
                s1 += qg0 * kb2 * static_cast<float>(c1);
                s2 += qg1 * ka * static_cast<float>(c2);
                s3 += qg1 * kb2 * static_cast<float>(c3);
            }
            score[nt][0] = s0;
            score[nt][1] = s1;
            score[nt][2] = s2;
            score[nt][3] = s3;
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
                    gqa_small_t_tc_smem_addr(&v_bf16[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
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

    // Stage the final split-local accumulator through the reused arena so
    // partial_acc is written as contiguous d-vector stores.
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0   = n * 8 + 2 * lid;
        const int d1   = d0 + 1;
        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        if (row0 < row_count) {
            acc_bf16[row0 * D + d0] = __float2bfloat16(acc[n][0]);
            acc_bf16[row0 * D + d1] = __float2bfloat16(acc[n][1]);
        }
        if (row1 < row_count) {
            acc_bf16[row1 * D + d0] = __float2bfloat16(acc[n][2]);
            acc_bf16[row1 * D + d1] = __float2bfloat16(acc[n][3]);
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
                *reinterpret_cast<const int4*>(&acc_bf16[row * D + d]);
        }
    }
}

} // namespace qus::kernels
