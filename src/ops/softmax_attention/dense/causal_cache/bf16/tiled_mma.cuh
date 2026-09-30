#pragma once

#include "ops/softmax_attention/dense/causal_cache/bf16/tile_io.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/softmax.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/epilogue.cuh"
#include <math_constants.h>

namespace ninfer::ops::detail {

// Online attention, one CTA per (query tile, query head), with
// bottom-right causal alignment (query row i sees keys [0, base_pos + i]).
template <typename Geometry, class Schedule, typename Metadata>
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocks) __global__
    void bf16_kv_tiled_mma_kernel(const __nv_bfloat16* __restrict__ q,
                                  const __nv_bfloat16* __restrict__ cache_k,
                                  const __half* __restrict__ cache_v, Metadata metadata,
                                  const std::int32_t* __restrict__ positions, float scale,
                                  __nv_bfloat16* __restrict__ out, std::int32_t width) {
    constexpr int D             = Geometry::kHeadDim;   // 256
    constexpr int Br            = Schedule::kQueryRows; // 64 query rows
    constexpr int Bc            = Schedule::kKeyRows;   // 64 key cols
    constexpr int Threads       = Schedule::kThreads;   // 128
    constexpr int QKNt          = Bc / 8;               // 8  QK score n-tiles
    constexpr int QKKs          = D / 16;               // 16 QK contraction steps over head_dim
    constexpr int PVNt          = D / 8;                // 32 PV output n-tiles
    constexpr int PVKs          = Bc / 16;              // 4  PV contraction steps over keys
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(Br == Schedule::kWarps * 16);
    static_assert(bf16_kv_tiled_shared_bytes<Geometry, Schedule> <= 99 * 1024);

    extern __shared__ __align__(16) __nv_bfloat16 causal_smem[];
    __nv_bfloat16* q_s = causal_smem;                             // [Br, D] swizzled
    __nv_bfloat16* k_s = q_s + Br * D;                            // [Bc, D] swizzled
    __half* v_s        = reinterpret_cast<__half*>(k_s + Bc * D); // [Bc, D] swizzled

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);

    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        causal_zero_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid, Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat     = lane >> 3;
    const int a_rin     = lane & 7;
    const int a_rowoff  = a_rin + ((a_mat & 1) << 3);
    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16; // this warp owns rows [warp_row0, warp_row0+16)

    // Per-lane precomputed swizzled ldmatrix base addresses (see causal_swizzle_address).
    const unsigned q_sbase = smem_addr(q_s);
    const unsigned k_sbase = smem_addr(k_s);
    const unsigned v_sbase = smem_addr(v_s);
    // Q A-fragment: row = warp_row0 + a_rowoff, col = k*16 + a_coloff.
    const unsigned q_lane_base = q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * (D * 2));
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
    // K B-fragment via ldmatrix.x4 (2 n-tiles/instr): lanes 16-31 fetch the +8-key
    // half (eight D-wide rows), lanes with bit3 set fetch the +8 d-contract half.
    const unsigned k_lane_base = k_sbase + static_cast<unsigned>(b_rin * (D * 2)) +
                                 (static_cast<unsigned>(lane >> 4) * (8 * D * 2));
    const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r  = static_cast<unsigned>(b_rin << 4);
    // V B-fragment via ldmatrix.x4.trans (2 n-tiles/instr): row = k*16 + (bit3)*8 + b_rin,
    // col = n*8 + (lane>>4)*8.
    const unsigned v_lane_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * (8 * D * 2)) +
                                 static_cast<unsigned>(b_rin * (D * 2));
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    // Stage Q into smem once via cp.async (overlaps with the K(0) prologue load
    // below); it stays resident for the key loop. Global token stride is D*QHeads.
    {
        constexpr int VecPerRow      = D / 8;
        constexpr int QRowStride     = D * Geometry::QHeads; // global stride between tokens
        const __nv_bfloat16* q_block = q + causal_q_index<Geometry>(q_head, 0, q0);
        if (q0 + Br <= tokens) {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk / VecPerRow;
                const int d      = (chunk % VecPerRow) * 8;
                __nv_bfloat16* p = &q_s[row * D + causal_swizzle(row, d)];
                cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
            }
        } else {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk / VecPerRow;
                const int d      = (chunk % VecPerRow) * 8;
                __nv_bfloat16* p = &q_s[row * D + causal_swizzle(row, d)];
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
    Bf16KvSoftmaxRow state0, state1;

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int n_block_max   = (max_query_abs / Bc) + 1; // n_block_min == 0

    // Fold softmax_scale into the exp2 (FA-style): scores stay raw, so the
    // per-element "* scale" multiply drops out of the QK epilogue entirely.
    const float scale_l2 = scale * kLog2E;
    int physical_page    = block_table[0];

    // Prologue: commit Q, then kick off K(0). The loop's wait<0> below drains both.
    ninfer::ops::cp_commit();
    bf16_kv_stage_tile<Geometry, Schedule>(k_s, cache_k, kv_head, 0, max_query_abs, physical_page,
                                           tid);
    ninfer::ops::cp_commit();

    const auto step = [&]<bool FullTile>(int kb) {
        const int k0                 = kb * Bc;
        const int next_physical_page = (kb + 1 < n_block_max)
                                           ? block_table[((kb + 1) * Bc) >> kPagedKVPageShift]
                                           : physical_page;

        ninfer::ops::cp_wait<0>(); // K(kb) landed (also publishes q_s / prev PV done)
        __syncthreads();

        if (paged_kv_page_is_hole(physical_page)) {
            if (kb + 1 < n_block_max) {
                physical_page = next_physical_page;
                bf16_kv_stage_tile<Geometry, Schedule>(k_s, cache_k, kv_head, (kb + 1) * Bc,
                                                        max_query_abs, physical_page, tid);
                ninfer::ops::cp_commit();
            }
            return;
        }

        // Preserve the global FP16 V load/QK overlap.
        bf16_kv_stage_tile<Geometry, Schedule>(v_s, cache_v, kv_head, k0, max_query_abs,
                                               physical_page, tid);
        ninfer::ops::cp_commit();

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
                        causal_swizzle_address(q_lane_base, 0u, q_as, q_r));
#pragma unroll
            for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                ldmatrix_x4(
                    bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                    causal_swizzle_address(k_lane_base + static_cast<unsigned>(nt2 * (8 * D * 2)),
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
                            causal_swizzle_address(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(
                        bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0], bf[nxt][nt2 + 1][1],
                        causal_swizzle_address(
                            k_lane_base + static_cast<unsigned>(nt2 * (8 * D * 2)), ck, k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                         af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

        ninfer::ops::cp_wait<0>(); // V(kb) landed; QK done reading k_s.
        __syncthreads();

        // Prefetch K(kb+1) into the (now-free) K buffer, overlapping the PV MMA.
        if (kb + 1 < n_block_max) {
            physical_page = next_physical_page;
            bf16_kv_stage_tile<Geometry, Schedule>(k_s, cache_k, kv_head, (kb + 1) * Bc,
                                                   max_query_abs, physical_page, tid);
            ninfer::ops::cp_commit();
        }

        const int row0  = warp_row0 + gid;
        const int row1  = warp_row0 + gid + 8;
        const int qrow0 = q0 + row0;
        const int qrow1 = q0 + row1;
        const int qabs0 = (qrow0 < tokens) ? base_pos + qrow0 : -1;
        const int qabs1 = (qrow1 < tokens) ? base_pos + qrow1 : -1;

        // block row-max on raw (unscaled) scores; scale is folded into exp2 below
        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
        if constexpr (FullTile) {
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

        const float alpha0     = state0.update<!FullTile>(bm0, scale_l2);
        const float alpha1     = state1.update<!FullTile>(bm1, scale_l2);
        const float nm0_scaled = state0.scaled_maximum;
        const float nm1_scaled = state1.scaled_maximum;

        // P = exp2(S - m), repacked into the PV A-fragment layout, plus local block row-sum.
        // The row-sum allreduce is deferred to the epilogue; only row max must be reduced per tile.
        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
        if constexpr (FullTile) {
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
                    p_frag[pk][0] = pack_f16x2(p00, p01);
                    p_frag[pk][1] = pack_f16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_f16x2(p00, p01);
                    p_frag[pk][3] = pack_f16x2(p10, p11);
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
                    p_frag[pk][0] = pack_f16x2(p00, p01);
                    p_frag[pk][1] = pack_f16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_f16x2(p00, p01);
                    p_frag[pk][3] = pack_f16x2(p10, p11);
                }
            }
        }

        state0.accumulate(alpha0, bl0);
        state1.accumulate(alpha1, bl1);
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

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
                          causal_swizzle_address(v_lane_base, 0u, v_as, v_r));
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
                ldmatrix_x4_t(
                    vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                    causal_swizzle_address(v_lane_base + static_cast<unsigned>(k2 * (16 * D * 2)),
                                           ckv, v_as, v_r));
            }
            mma_f16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
                    p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_f16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_frag[k][0],
                    p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);
        }
    };
    const int full_blocks = q0 + Br <= tokens ? min(n_block_max, (base_pos + q0 + 1) / Bc) : 0;
    for (int kb = 0; kb < full_blocks; ++kb) step.template operator()<true>(kb);
    for (int kb = full_blocks; kb < n_block_max; ++kb) step.template operator()<false>(kb);

    state0.finish();
    state1.finish();

#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0    = n * 8 + 2 * lid;
        const int qrow0 = q0 + warp_row0 + gid;
        const int qrow1 = q0 + warp_row0 + gid + 8;
        if (qrow0 < tokens) {
            bf16_kv_store_pair<Geometry, false>({}, out, q_head, qrow0, d0, width, 0, acc[n][0],
                                                acc[n][1], state0.scaled_maximum, state0.sum);
        }
        if (qrow1 < tokens) {
            bf16_kv_store_pair<Geometry, false>({}, out, q_head, qrow1, d0, width, 0, acc[n][2],
                                                acc[n][3], state1.scaled_maximum, state1.sum);
        }
    }
    causal_zero_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid, Threads);
}


} // namespace ninfer::ops::detail
