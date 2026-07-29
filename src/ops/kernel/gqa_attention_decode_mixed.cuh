#pragma once

// ninfer::ops - split-KV GQA small-T attention, mixed-format KV-cache partial kernel.
// One K/V plane is BF16 and the other is INT8-G64. Built on the BF16 kernel
// skeleton (gqa_attention_decode_bf16.cuh): Q and both MMAs stay BF16; the
// INT8-G64 side is staged as codes+scales and dequantized once into the same
// swizzled BF16 tile the BF16 side fills directly. The fused append stores the
// BF16 side verbatim and quantizes the INT8-G64 side, after which all of its
// keys (history and current tokens) are read back from the cache codes.
//
// Standalone from both symmetric kernels; shared scaffolding lives in
// gqa_attention_decode.cuh, the codec helpers in gqa_attention_kv_quant.cuh.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, int TokenTile, int WarpsPerCta, bool KInt8, bool VInt8,
          typename CacheInput>
__launch_bounds__(128, 2) __global__ void gqa_attention_small_t_tc_partial_mixed_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, void* cache_k,
    void* cache_v, __half* cache_k_scale, __half* cache_v_scale, std::int32_t tokens,
    std::int32_t padded_context, std::int32_t max_context, float scale,
    __nv_bfloat16* partial_acc, float* partial_m, float* partial_l) {
    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(WarpsPerCta >= 1 && WarpsPerCta <= 4);
    static_assert(KInt8 != VInt8, "mixed kernel serves exactly one INT8-G64 side");

    constexpr int Wc            = WarpsPerCta;
    constexpr int Br            = Wc * 16;
    constexpr int Bc            = 32;
    constexpr int D             = kGqaHeadDim;
    constexpr int Threads       = Wc * 32;
    constexpr int Groups        = kGqaKvQuantGroups;
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
    __shared__ __align__(16) std::int8_t i8_codes_s[Bc * D];
    __shared__ __align__(16) __half i8_scale_s[Bc * Groups];
    __nv_bfloat16* k_s = qkv_s;
    __nv_bfloat16* v_s = qkv_s + Bc * D;

    __nv_bfloat16* cache_k_bf16 = static_cast<__nv_bfloat16*>(cache_k);
    __nv_bfloat16* cache_v_bf16 = static_cast<__nv_bfloat16*>(cache_v);
    std::int8_t* cache_k_i8     = static_cast<std::int8_t*>(cache_k);
    std::int8_t* cache_v_i8     = static_cast<std::int8_t*>(cache_v);

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    const int row_count   = tokens * Geometry::GroupSize;

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        row_count > Br || split_count <= 0) {
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= max_context) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, false>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    const int kps         = div_up(window, active_split_count);
    const int split_start = split * kps;
    const int split_limit = split_start + kps;
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }

    if constexpr (CacheInput::writes_cache) {
        // The owning split stores each new row. The BF16 side copies verbatim and is
        // re-read from input below, so no split depends on another split's write.
        if constexpr (!KInt8 || !VInt8) {
            for (int chunk = tid; chunk < tokens * (D / 8); chunk += Threads) {
                const int token = chunk / (D / 8);
                const int d     = (chunk - token * (D / 8)) * 8;
                const int p_tok = pos[token];
                if (p_tok >= split_start && p_tok < split_end && p_tok >= 0 &&
                    p_tok < max_context) {
                    const std::int64_t new_off   = gqa_kv_new_index<Geometry>(kv_head, d, token);
                    const std::int64_t cache_off =
                        gqa_cache_index(kv_head, d, p_tok, padded_context);
                    if constexpr (!KInt8) {
                        store_vec(&cache_k_bf16[cache_off], load_vec<int4>(&input.k[new_off]));
                    }
                    if constexpr (!VInt8) {
                        store_vec(&cache_v_bf16[cache_off], load_vec<int4>(&input.v[new_off]));
                    }
                }
            }
        }
        // The INT8-G64 side quantizes its rows; its attention reads the codes back.
        for (int pair = warp; pair < tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int d0 = grp * kGqaKvQuantGroup + lane;
            const int d1 = d0 + 32;
            if constexpr (KInt8) {
                const std::int64_t src0 = gqa_kv_new_index<Geometry>(kv_head, d0, token);
                const std::int64_t src1 = gqa_kv_new_index<Geometry>(kv_head, d1, token);
                const float x0          = __bfloat162float(input.k[src0]);
                const float x1          = __bfloat162float(input.k[src1]);
                float amax              = fmaxf(fabsf(x0), fabsf(x1));
                amax                    = warp_max(amax, FullMask);
                const __half sh  = __float2half_rn(amax > 0.0f ? amax / 127.0f : 0.0f);
                const float s    = __half2float(sh);
                const float inv  = s > 0.0f ? 1.0f / s : 0.0f;
                cache_k_i8[gqa_kv_quant_code_index(kv_head, d0, position, padded_context)] =
                    gqa_kv_quant_code(x0, inv);
                cache_k_i8[gqa_kv_quant_code_index(kv_head, d1, position, padded_context)] =
                    gqa_kv_quant_code(x1, inv);
                if (lane == 0) {
                    cache_k_scale[gqa_kv_quant_scale_index(kv_head, grp, position,
                                                           padded_context)] = sh;
                }
            }
            if constexpr (VInt8) {
                const std::int64_t src0 = gqa_kv_new_index<Geometry>(kv_head, d0, token);
                const std::int64_t src1 = gqa_kv_new_index<Geometry>(kv_head, d1, token);
                const float x0          = __bfloat162float(input.v[src0]);
                const float x1          = __bfloat162float(input.v[src1]);
                float amax              = fmaxf(fabsf(x0), fabsf(x1));
                amax                    = warp_max(amax, FullMask);
                const __half sh  = __float2half_rn(amax > 0.0f ? amax / 127.0f : 0.0f);
                const float s    = __half2float(sh);
                const float inv  = s > 0.0f ? 1.0f / s : 0.0f;
                cache_v_i8[gqa_kv_quant_code_index(kv_head, d0, position, padded_context)] =
                    gqa_kv_quant_code(x0, inv);
                cache_v_i8[gqa_kv_quant_code_index(kv_head, d1, position, padded_context)] =
                    gqa_kv_quant_code(x1, inv);
                if (lane == 0) {
                    cache_v_scale[gqa_kv_quant_scale_index(kv_head, grp, position,
                                                           padded_context)] = sh;
                }
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < Br * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        __nv_bfloat16 value = __float2bfloat16(0.0f);
        if (row < row_count && gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            value = q[gqa_q_index<Geometry>(q_head, d, token)];
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
        ldmatrix_x4(af_q[k][0], af_q[k][1], af_q[k][2], af_q[k][3],
                    smem_addr(&qkv_s[arow * D + gqa_small_t_tc_swz(arow, acol)]));
    }
    __syncthreads();

    // Stage one BF16 key tile with the same from_new policy as the BF16 kernel.
    auto stage_bf16_tile = [&]<bool IsK>(__nv_bfloat16* tile_s, const __nv_bfloat16* cache_side,
                                         int k0) {
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int key_l    = chunk / (D / 8);
            const int d        = (chunk - key_l * (D / 8)) * 8;
            const int key      = k0 + key_l;
            __nv_bfloat16* dst = &tile_s[key_l * D + gqa_small_t_tc_swz(key_l, d)];
            if (key < split_end) {
                if constexpr (CacheInput::writes_cache) {
                    const __nv_bfloat16* input_side = IsK ? input.k : input.v;
                    const int new_token             = key - first_pos;
                    const bool from_new = new_token >= 0 && new_token < tokens && key >= first_pos;
                    if (from_new) {
                        const std::int64_t off =
                            gqa_kv_new_index<Geometry>(kv_head, d, new_token);
                        ninfer::ops::cp_async<16>(dst, &input_side[off]);
                    } else {
                        const std::int64_t off = gqa_cache_index(kv_head, d, key, padded_context);
                        ninfer::ops::cp_async<16>(dst, &cache_side[off]);
                    }
                } else {
                    const std::int64_t off = gqa_cache_index(kv_head, d, key, padded_context);
                    ninfer::ops::cp_async<16>(dst, &cache_side[off]);
                }
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
    };

    // Stage one INT8-G64 key tile as raw codes + scales (every key, including the
    // current tokens, comes from the cache; see the append note above).
    auto stage_i8_tile = [&](const std::int8_t* cache_side, const __half* scale_side, int k0) {
        for (int key_l = tid; key_l < Bc; key_l += Threads) {
            const int key  = k0 + key_l;
            __half* dst    = &i8_scale_s[key_l * Groups];
            if (key < split_end) {
                const std::int64_t off =
                    gqa_kv_quant_scale_index(kv_head, 0, key, padded_context);
                ninfer::ops::cp_async<8>(dst, &scale_side[off]);
            } else {
                store_vec(dst, make_int2(0, 0));
            }
        }
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int key_l    = chunk / (D / 16);
            const int d        = (chunk - key_l * (D / 16)) * 16;
            const int key      = k0 + key_l;
            std::int8_t* dst   = &i8_codes_s[key_l * D + d];
            if (key < split_end) {
                const std::int64_t off = gqa_kv_quant_code_index(kv_head, d, key, padded_context);
                ninfer::ops::cp_async<16>(dst, &cache_side[off]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
    };

    // Dequantize the staged INT8-G64 tile into the swizzled BF16 tile the MMAs read.
    auto dequant_i8_tile = [&](__nv_bfloat16* tile_s) {
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int key_l = chunk / (D / 8);
            const int d     = (chunk - key_l * (D / 8)) * 8;
            const int grp   = d >> 6;
            const float s   = __half2float(i8_scale_s[key_l * Groups + grp]);
            store_vec(&tile_s[key_l * D + gqa_small_t_tc_swz(key_l, d)],
                      gqa_kv_dequant_i8x8_from(&i8_codes_s[key_l * D + d], s));
        }
    };

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
        if constexpr (KInt8) {
            stage_i8_tile(cache_k_i8, cache_k_scale, k0);
        } else {
            stage_bf16_tile.template operator()<true>(k_s, cache_k_bf16, k0);
        }
        if constexpr (VInt8) {
            stage_i8_tile(cache_v_i8, cache_v_scale, k0);
        } else {
            stage_bf16_tile.template operator()<false>(v_s, cache_v_bf16, k0);
        }
        ninfer::ops::cp_commit();
        ninfer::ops::cp_wait<0>();
        __syncthreads();
        if constexpr (KInt8) { dequant_i8_tile(k_s); }
        if constexpr (VInt8) { dequant_i8_tile(v_s); }
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
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&k_s[brow * D + gqa_small_t_tc_swz(brow, bcol)]));
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af_q[k][0],
                         af_q[k][1], af_q[k][2], af_q[k][3], bf[0], bf[1]);
            }
        }

        const int row0 = warp_row0 + gid;
        const int row1 = row0 + 8;
        int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head0, token0);
        gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head1, token1);
        const int qabs0 = (row0 < row_count) ? pos[token0] : -1;
        const int qabs1 = (row1 < row_count) ? pos[token1] : -1;

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            const int col1 = col0 + 1;
            const int key0 = k0 + col0;
            const int key1 = col1 + k0;
            score[nt][0]   = (row0 < row_count && key0 < split_end && key0 <= qabs0)
                                 ? score[nt][0] * scale
                                 : -CUDART_INF_F;
            score[nt][1]   = (row0 < row_count && key1 < split_end && key1 <= qabs0)
                                 ? score[nt][1] * scale
                                 : -CUDART_INF_F;
            score[nt][2]   = (row1 < row_count && key0 < split_end && key0 <= qabs1)
                                 ? score[nt][2] * scale
                                 : -CUDART_INF_F;
            score[nt][3]   = (row1 < row_count && key1 < split_end && key1 <= qabs1)
                                 ? score[nt][3] * scale
                                 : -CUDART_INF_F;
            bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
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
                ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                            smem_addr(&p_sw[a_rowoff * Bc + gqa_small_t_tc_swz32(a_rowoff, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_s[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
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
            gqa_small_t_tc_row_to_qt<Geometry>(row0, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l0;
        }
        if (row1 < row_count) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, tokens, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = l1;
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
        gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
            store_vec(&partial_acc[dst], load_vec<int4>(&qkv_s[row * D + d]));
        }
    }
}

} // namespace ninfer::ops
