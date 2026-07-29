#pragma once

// Mixed-format GQA prompt kernels: one K/V plane is BF16, the other INT8-G64.
// Built on the BF16 prompt kernel skeleton (gqa_attention_prefill_bf16.cuh): Q, QK,
// and PV stay on BF16 tensor cores; the INT8-G64 side is staged as codes+scales and
// dequantized once into the same swizzled BF16 tile the BF16 side fills directly.
// The fill kernel stores the BF16 side verbatim and quantizes the INT8-G64 side per
// (token, kv_head, 64-group) warp unit, one warp per side format as registered.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/kernel/gqa_attention_kv_quant.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaPrefillMixedSmemBytes = kGqaPrefillSmemBytes;

// One warp per (token, kv_head, 64-d group): the INT8-G64 side quantizes its group
// (two dimensions per lane, matching the symmetric INT8 fill exactly); the BF16 side
// copies its group verbatim (eight 16-byte vectors, one per lane 0..7).
template <typename Geometry, bool KInt8, bool VInt8>
__launch_bounds__(256) __global__ void gqa_attention_prefill_fill_mixed_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, void* __restrict__ cache_k,
    void* __restrict__ cache_v, __half* __restrict__ scale_k, __half* __restrict__ scale_v,
    std::int32_t tokens, std::int32_t padded_context) {
    static_assert(KInt8 != VInt8, "mixed fill serves exactly one INT8-G64 side");

    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads * kGqaKvQuantGroups;
    if (unit >= units) { return; }

    const int group    = unit % kGqaKvQuantGroups;
    const int tmp      = unit / kGqaKvQuantGroups;
    const int kv_head  = tmp % Geometry::KVHeads;
    const int token    = tmp / Geometry::KVHeads;
    const int position = positions[0] + token;
    const int d0       = group * kGqaKvQuantGroup;

    auto quantize_side = [&](const __nv_bfloat16* src, std::int8_t* dst, __half* scale_dst) {
        const int dd0           = d0 + lane;
        const int dd1           = dd0 + 32;
        const std::int64_t src0 = gqa_kv_quant_src_index<Geometry>(kv_head, dd0, token);
        const std::int64_t src1 = gqa_kv_quant_src_index<Geometry>(kv_head, dd1, token);
        const float x0          = __bfloat162float(src[src0]);
        const float x1          = __bfloat162float(src[src1]);
        float absmax            = fmaxf(fabsf(x0), fabsf(x1));
        absmax                  = warp_max(absmax, FullMask);
        const __half sh  = __float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f);
        const float s    = __half2float(sh);
        const float inv  = s > 0.0f ? 1.0f / s : 0.0f;
        dst[gqa_kv_quant_code_index(kv_head, dd0, position, padded_context)] =
            gqa_kv_quant_code(x0, inv);
        dst[gqa_kv_quant_code_index(kv_head, dd1, position, padded_context)] =
            gqa_kv_quant_code(x1, inv);
        if (lane == 0) {
            scale_dst[gqa_kv_quant_scale_index(kv_head, group, position, padded_context)] = sh;
        }
    };

    auto copy_side = [&](const __nv_bfloat16* src, __nv_bfloat16* dst) {
        if (lane < kGqaKvQuantGroup / 8) {
            const int d = d0 + lane * 8;
            const std::int64_t src_off = gqa_kv_quant_src_index<Geometry>(kv_head, d, token);
            const std::int64_t dst_off =
                gqa_kv_quant_code_index(kv_head, d, position, padded_context);
            store_vec(&dst[dst_off], load_vec<int4>(&src[src_off]));
        }
    };

    if constexpr (KInt8) {
        quantize_side(k, static_cast<std::int8_t*>(cache_k), scale_k);
    } else {
        copy_side(k, static_cast<__nv_bfloat16*>(cache_k));
    }
    if constexpr (VInt8) {
        quantize_side(v, static_cast<std::int8_t*>(cache_v), scale_v);
    } else {
        copy_side(v, static_cast<__nv_bfloat16*>(cache_v));
    }
}

// Load/dequantize one INT8-G64 cache tile directly into the swizzled BF16 MMA tile.
// Eight lanes in each warp share one scale load; codes remain one coalesced 64-bit
// load per eight dimensions. This avoids a second shared-memory staging arena.
__device__ __forceinline__ void gqa_prefill_dequant_i8_cache_tile(
    __nv_bfloat16* tile_s, const std::int8_t* cache, const __half* cache_scale, int kv_head,
    int k0, int max_query_abs, int padded_context, int tid) {
    constexpr int D       = kGqaPrefillHeadDim;
    constexpr int Bc      = kGqaPrefillBc;
    constexpr int Threads = kGqaPrefillThreads;
    constexpr unsigned FullMask = 0xffffffffu;
    const int lane = tid & 31;
#pragma unroll 1
    for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
        const int key_l = chunk / (D / 8);
        const int d     = (chunk - key_l * (D / 8)) * 8;
        const int key   = k0 + key_l;
        const int grp   = d >> 6;
        __nv_bfloat16* dst = &tile_s[key_l * D + gqa_prefill_swz(key_l, d)];
        if (key <= max_query_abs) {
            float s = 0.0f;
            if ((lane & 7) == 0) {
                const std::int64_t scale_off =
                    gqa_kv_quant_scale_index(kv_head, grp, key, padded_context);
                s = __half2float(cache_scale[scale_off]);
            }
            s = __shfl_sync(FullMask, s, grp * 8);
            const std::int64_t code_off =
                gqa_kv_quant_code_index(kv_head, d, key, padded_context);
            store_vec(dst, gqa_kv_dequant_i8x8_from(&cache[code_off], s));
        } else {
            store_vec(dst, make_int4(0, 0, 0, 0));
        }
    }
}

// FlashAttention-2 forward over a mixed-format cache; identical tiling and MMA
// schedule to the BF16 prompt kernel, with the INT8-G64 side dequantized in place
// of a direct BF16 stage.
template <typename Geometry, bool KInt8, bool VInt8>
__launch_bounds__(kGqaPrefillThreads, 1) __global__
    void gqa_attention_prefill_mixed_kernel(const __nv_bfloat16* __restrict__ q,
                                            void* __restrict__ cache_k, void* __restrict__ cache_v,
                                            const __half* __restrict__ cache_k_scale,
                                            const __half* __restrict__ cache_v_scale,
                                            const std::int32_t* __restrict__ positions, float scale,
                                            __nv_bfloat16* __restrict__ out, std::int32_t tokens,
                                            std::int32_t padded_context) {
    static_assert(KInt8 != VInt8, "mixed kernel serves exactly one INT8-G64 side");

    constexpr int D             = kGqaPrefillHeadDim; // 256
    constexpr int Br            = kGqaPrefillBr;      // 64 query rows
    constexpr int Bc            = kGqaPrefillBc;      // 64 key cols
    constexpr int Threads       = kGqaPrefillThreads; // 128
    constexpr int QKNt          = Bc / 8;             // 8  QK score n-tiles
    constexpr int QKKs          = D / 16;             // 16 QK contraction steps over head_dim
    constexpr int PVNt          = D / 8;              // 32 PV output n-tiles
    constexpr int PVKs          = Bc / 16;            // 4  PV contraction steps over keys
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(Threads == 128);

    extern __shared__ __align__(16) __nv_bfloat16 gqa_smem[];
    __nv_bfloat16* q_s       = gqa_smem;     // [Br, D] swizzled
    __nv_bfloat16* k_s       = q_s + Br * D; // [Bc, D] swizzled
    __nv_bfloat16* v_s       = k_s + Bc * D; // [Bc, D] swizzled

    const __nv_bfloat16* cache_k_bf16 = static_cast<const __nv_bfloat16*>(cache_k);
    const __nv_bfloat16* cache_v_bf16 = static_cast<const __nv_bfloat16*>(cache_v);
    const std::int8_t* cache_k_i8     = static_cast<const std::int8_t*>(cache_k);
    const std::int8_t* cache_v_i8     = static_cast<const std::int8_t*>(cache_v);

    const int q_block  = static_cast<int>(blockIdx.x);
    const int q_head   = static_cast<int>(blockIdx.y);
    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int q0       = q_block * Br;
    const int kv_head  = q_head / Geometry::GroupSize;
    const int base_pos = positions[0];

    if (q_head >= Geometry::QHeads || q0 >= tokens) { return; }

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16; // this warp owns rows [warp_row0, warp_row0+16)

    // Per-lane precomputed swizzled ldmatrix base addresses (see gqa_prefill_swz_addr).
    const unsigned q_sbase = smem_addr(q_s);
    const unsigned k_sbase = smem_addr(k_s);
    const unsigned v_sbase = smem_addr(v_s);
    // Q A-fragment: row = warp_row0 + a_rowoff, col = k*16 + a_coloff.
    const unsigned q_lane_base = q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * 512);
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
    // K B-fragment via ldmatrix.x4 (2 n-tiles/instr): lanes 16-31 fetch the +8-key
    // half (extra 4096 bytes), lanes with bit3 set fetch the +8 d-contract half.
    const unsigned k_lane_base =
        k_sbase + static_cast<unsigned>(b_rin * 512) + (static_cast<unsigned>(lane >> 4) << 12);
    const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r  = static_cast<unsigned>(b_rin << 4);
    // V B-fragment via ldmatrix.x4.trans (2 n-tiles/instr): row = k*16 + (bit3)*8 + b_rin,
    // col = n*8 + (lane>>4)*8.
    const unsigned v_lane_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
                                 static_cast<unsigned>(b_rin * 512);
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    // Stage Q into smem once via cp.async (overlaps with the K(0) prologue load
    // below); it stays resident for the whole key loop.
    {
        constexpr int VecPerRow      = D / 8;
        constexpr int QRowStride     = D * Geometry::QHeads; // global stride between tokens
        const __nv_bfloat16* q_block = q + gqa_prefill_q_index<Geometry>(q_head, 0, q0);
        if (q0 + Br <= tokens) {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
            }
        } else {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                if (q0 + row < tokens) {
                    cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
                } else {
                    store_vec(p, make_int4(0, 0, 0, 0));
                }
            }
        }
    }

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int n_block_max   = (max_query_abs / Bc) + 1; // n_block_min == 0

    // Fold softmax_scale into the exp2 (FA-style): scores stay raw, so the
    // per-element "* scale" multiply drops out of the QK epilogue entirely.
    const float scale_l2 = scale * Log2E;

    auto issue_k_tile = [&](int k0) {
        if constexpr (!KInt8) {
            gqa_prefill_stage_kv(k_s, cache_k_bf16, kv_head, k0, max_query_abs, padded_context,
                                 tid);
        }
    };
    auto issue_v_tile = [&](int k0) {
        if constexpr (!VInt8) {
            gqa_prefill_stage_kv(v_s, cache_v_bf16, kv_head, k0, max_query_abs, padded_context,
                                 tid);
        }
    };

    // Prologue: commit Q, then kick off K(0). The loop's wait<0> below drains both.
    ninfer::ops::cp_commit();
    if constexpr (!KInt8) {
        issue_k_tile(0);
        ninfer::ops::cp_commit();
    }

    for (int kb = 0; kb < n_block_max; ++kb) {
        const int k0 = kb * Bc;

        ninfer::ops::cp_wait<0>(); // K(kb) landed (also publishes q_s / prev PV done)
        __syncthreads();

        // Overlap the V(kb) load against the K dequant + QK MMA below.
        if constexpr (!VInt8) {
            issue_v_tile(k0);
            ninfer::ops::cp_commit();
        }

        if constexpr (KInt8) {
            gqa_prefill_dequant_i8_cache_tile(k_s, cache_k_i8, cache_k_scale, kv_head, k0,
                                              max_query_abs, padded_context, tid);
            __syncthreads();
        }

        // S = Q Kᵀ for this warp's 16 rows over all Bc keys, in registers.
        // Software-pipelined like cute's gemm: issue the ldmatrix for contraction
        // step k+1 while the m16n8k16 MMAs for step k run, so the LSU (ldmatrix)
        // and tensor pipes overlap instead of stalling on each other.
        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        // Swizzled ldmatrix addresses via precomputed per-lane bases + immediates.
        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        {
            ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                        gqa_prefill_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
            for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                            gqa_prefill_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096),
                                                 0u, k_as, k_r));
            }
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;
            if (k + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            gqa_prefill_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(
                        bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0], bf[nxt][nt2 + 1][1],
                        gqa_prefill_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096), ck,
                                             k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                         af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

        const int row0             = warp_row0 + gid;
        const int row1             = warp_row0 + gid + 8;
        const int qrow0            = q0 + row0;
        const int qrow1            = q0 + row1;
        const int qabs0            = (qrow0 < tokens) ? base_pos + qrow0 : -1;
        const int qabs1            = (qrow1 < tokens) ? base_pos + qrow1 : -1;
        const bool full_score_tile = (q0 + Br <= tokens) && ((k0 + Bc - 1) <= (base_pos + q0));

        // block row-max on raw (unscaled) scores; scale is folded into exp2 below
        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                score[nt][0]   = (qrow0 < tokens && key0 <= qabs0) ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = (qrow0 < tokens && key1 <= qabs0) ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = (qrow1 < tokens && key0 <= qabs1) ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = (qrow1 < tokens && key1 <= qabs1) ? score[nt][3] : -CUDART_INF_F;
                bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(m0, bm0);
        const float nm1        = fmaxf(m1, bm1);
        const float nm0_scaled = nm0 * scale_l2;
        const float nm1_scaled = nm1 * scale_l2;
        const float alpha0     = exp2_approx(__fmaf_rn(m0, scale_l2, -nm0_scaled));
        const float alpha1     = exp2_approx(__fmaf_rn(m1, scale_l2, -nm1_scaled));

        // P = exp2(S - m), repacked into the PV A-fragment layout, plus local block row-sum.
        // The row-sum allreduce is deferred to the epilogue; only row max must be reduced per tile.
        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
                const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
                const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
                const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = (score[nt][0] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p01 = (score[nt][1] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p10 = (score[nt][2] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0f;
                const float p11 = (score[nt][3] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        }

        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        if constexpr (!VInt8) {
            ninfer::ops::cp_wait<0>(); // V(kb) landed; QK done reading k_s
            __syncthreads();
        } else {
            // No V cp.async wait supplies the BF16 path's CTA rendezvous. All warps
            // must finish QK before K(kb+1) is allowed to overwrite k_s.
            __syncthreads();
        }

        if constexpr (VInt8) {
            gqa_prefill_dequant_i8_cache_tile(v_s, cache_v_i8, cache_v_scale, kv_head, k0,
                                              max_query_abs, padded_context, tid);
        }

        // Prefetch BF16 K(kb+1), overlapping the PV MMA. An INT8-G64 K side is
        // loaded directly into the BF16 tile at the start of its next iteration.
        if constexpr (!KInt8) {
            if (kb + 1 < n_block_max) {
                issue_k_tile((kb + 1) * Bc);
                ninfer::ops::cp_commit();
            }
        }

        if constexpr (VInt8) { __syncthreads(); } // publish the dequantized V tile

        // O += P V, contracting over the Bc keys. The (k, n) iteration space is
        // flattened and software-pipelined: the transposed ldmatrix for the next
        // V fragment is issued while the current MMA runs.
        // Each x4.trans load covers 2 output n-tiles (16 dims); pipeline the next
        // load against the current pair of MMAs.
        constexpr int PVHalf  = PVNt / 2;      // 16 n-tile pairs
        constexpr int PVLoads = PVKs * PVHalf; // 64 x4.trans loads
        // Swizzled V x4.trans addresses via precomputed per-lane base + immediates.
        unsigned vf[2][4];
        {
            ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                          gqa_prefill_swz_addr(v_lane_base, 0u, v_as, v_r));
        }
#pragma unroll
        for (int li = 0; li < PVLoads; ++li) {
            const int k   = li / PVHalf;
            const int n2  = (li % PVHalf) * 2;
            const int cur = li & 1;
            const int nxt = cur ^ 1;
            if (li + 1 < PVLoads) {
                const int k2       = (li + 1) / PVHalf;
                const int n2b      = ((li + 1) % PVHalf) * 2;
                const unsigned ckv = static_cast<unsigned>(n2b << 4);
                ldmatrix_x4_t(vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                              gqa_prefill_swz_addr(v_lane_base + static_cast<unsigned>(k2 * 8192),
                                                   ckv, v_as, v_r));
            }
            mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
                     p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_frag[k][0],
                     p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);
        }
    }

    l0 = warp_sum<4>(l0, FullMask);
    l1 = warp_sum<4>(l1, FullMask);

    // Normalize once per row via reciprocal-multiply instead of 128 IEEE divides.
    const float inv_l0 = (l0 > 0.0f) ? __frcp_rn(l0) : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? __frcp_rn(l1) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0    = n * 8 + 2 * lid;
        const int qrow0 = q0 + warp_row0 + gid;
        const int qrow1 = q0 + warp_row0 + gid + 8;
        if (qrow0 < tokens) {
            *reinterpret_cast<unsigned*>(&out[gqa_prefill_q_index<Geometry>(q_head, d0, qrow0)]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (qrow1 < tokens) {
            *reinterpret_cast<unsigned*>(&out[gqa_prefill_q_index<Geometry>(q_head, d0, qrow1)]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
}

} // namespace ninfer::ops
