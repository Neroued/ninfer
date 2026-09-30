#pragma once

#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/softmax_attention/common/causal_epilogue.cuh"
#include "ops/softmax_attention/common/causal_operands.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/common/causal_softmax.cuh"
#include "ops/softmax_attention/common/causal_tile_io.cuh"

namespace ninfer::ops::detail {

// Each warp retains 16 query rows through MXFP8 QK, softmax, and FP16 PV.
// Values selects the stored V representation; split outputs remain FP32 with
// maxima in natural scaled-score units for the final merge.
template <class Geometry, class Schedule, class Values, class Metadata>
__global__ __maxnreg__(Schedule::kMaxRegisters) void mxfp8_kv_tiled_mma_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const typename Values::Scale* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, std::int32_t width,
    CausalKvPartition partition, CausalPartialView partial) {
    constexpr int D             = 256;
    constexpr int Br            = Schedule::kQueryRows;
    constexpr int Bc            = Schedule::kKeyRows;
    constexpr int DB16          = 128;
    constexpr int QKKs          = D / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVNtPerWarp   = D / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr unsigned FullMask = 0xffffffffU;
    static_assert(QKKs == 8);
    static_assert(PVNtPerWarp == 32);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::uint8_t* q_fp8 = reinterpret_cast<std::uint8_t*>(smem_raw);
    float* q_scale      = reinterpret_cast<float*>(q_fp8 + Schedule::kQBytes);
    std::uint8_t* k_fp8 = reinterpret_cast<std::uint8_t*>(
        reinterpret_cast<unsigned char*>(q_scale) + Schedule::kQScaleBytes);
    std::uint8_t* v_codes = k_fp8 + Schedule::kKBytes;
    __half* v_f16         = reinterpret_cast<__half*>(v_codes + Schedule::kVBytes);
    __half* k_scale_s =
        reinterpret_cast<__half*>(reinterpret_cast<unsigned char*>(v_f16) + Schedule::kVStageBytes);
    auto* v_scale_s = reinterpret_cast<typename Values::Scale*>(k_scale_s + Bc);

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) return;
    if (q0 >= tokens) return;
    const int split         = blockIdx.z;
    const int visible       = positions[width - 1] + 1;
    const int active_splits = partition.active(visible);
    if (split >= active_splits) return;
    const int logical_tiles         = div_up(visible, Bc);
    const int first_owned_tile      = split * logical_tiles / active_splits;
    const int end_owned_tile        = (split + 1) * logical_tiles / active_splits;
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int tile_rows             = min(Br, tokens - q0);
    const int max_query_abs         = base_pos + q0 + tile_rows - 1;
    const int key_blocks = max(0, min(end_owned_tile, max_query_abs / Bc + 1) - first_owned_tile);

    for (int row = warp; row < Br; row += Schedule::kWarps) {
        float values[8];
        float local_absmax = 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            values[r]   = row < tile_rows
                              ? __bfloat162float(q[causal_q_index<Geometry>(q_head, d, q0 + row)])
                              : 0.0F;
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) local_absmax = fmaxf(local_absmax, fabsf(values[r]));
        const float absmax = warp_max(local_absmax, FullMask);
        const float qs     = absmax > 0.0F ? absmax / kKVCacheFp8MaxFinite : 0.0F;
        const float inv    = qs > 0.0F ? 1.0F / qs : 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            causal_store_query_code(q_fp8, row, d, kv_cache_fp8_quant_code(values[r], inv));
        }
        if (lane == 0) q_scale[row] = qs;
    }
    __syncthreads();

    const int gid              = lane >> 2;
    const int lid              = lane & 3;
    const int a_mat            = lane >> 3;
    const int a_rin            = lane & 7;
    const int a_rowoff         = a_rin + ((a_mat & 1) << 3);
    const int b_rin            = lane & 7;
    const int b_koff           = ((lane >> 3) & 1) << 3;
    const int warp_row0        = warp * 16;
    const float q_scale_r0     = q_scale[warp_row0 + gid];
    const float q_scale_r1     = q_scale[warp_row0 + gid + 8];
    const unsigned q_lane_base = smem_addr(q_fp8) + (warp_row0 + a_rowoff) * D;
    const unsigned q_as        = (a_mat >> 1) << 4;
    const unsigned q_r         = a_rin << 4;
    const unsigned k_lane_base = smem_addr(k_fp8) + b_rin * D + (lane >> 4) * (8 * D);
    const unsigned k_as        = (b_koff >> 3) << 4;
    const unsigned k_r         = b_rin << 4;
    const unsigned v_lane_base = smem_addr(v_f16) + (((lane >> 3) & 1) * 8 + b_rin) * D * 2;
    const unsigned v_as        = (lane >> 4) << 4;
    const unsigned v_r         = b_rin << 4;
    auto issue_kv_scales       = [&](int tile_k0, int cooperative_tid, int cooperative_threads) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        const int page_offset0  = tile_k0 & (kPagedKVPageSize - 1);
        for (int key_l = cooperative_tid; key_l < Bc; key_l += cooperative_threads) {
            const int key = tile_k0 + key_l;
            if (key <= max_query_abs) {
                const std::int64_t off = kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head,
                                                                                  page_offset0 + key_l);
                k_scale_s[key_l]       = cache_k_scale[off];
                if constexpr (Values::kScaleItems == 1) {
                    v_scale_s[key_l] = cache_v_scale[off];
                } else {
                    const auto v_off =
                        paged_kv_element_offset<Values::kScaleItems, Geometry::KVHeads>(
                            physical_page, kv_head, page_offset0 + key_l, 0);
                    cp_async<16>(v_scale_s + key_l * Values::kScaleItems, cache_v_scale + v_off);
                }
            } else {
                k_scale_s[key_l] = __float2half_rn(0.0F);
                if constexpr (Values::kScaleItems == 1)
                    v_scale_s[key_l] = __float2half_rn(0.0F);
                else
                    store_vec(v_scale_s + key_l * Values::kScaleItems, make_int4(0, 0, 0, 0));
            }
        }
    };

    auto issue_kv_codes = [&](int tile_k0, int cooperative_tid, int cooperative_threads) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        const int page_offset0  = tile_k0 & (kPagedKVPageSize - 1);
#pragma unroll 1
        for (int chunk = cooperative_tid; chunk < Bc * (D / 16); chunk += cooperative_threads) {
            const int key_l  = chunk / (D / 16);
            const int dc     = chunk - key_l * (D / 16);
            const int d      = dc * 16;
            const int key    = tile_k0 + key_l;
            std::uint8_t* kd = &k_fp8[(key_l * DB16 + causal_swizzle(key_l, dc * 8)) * 2];
            if (key <= max_query_abs) {
                const std::int64_t off = kv_cache_fp8_code_index<Geometry>(physical_page, kv_head,
                                                                           d, page_offset0 + key_l);
                cp_async<16, Cache::cg>(kd, &cache_k[off]);
                if constexpr (Values::kCodeBytes == D)
                    cp_async<16, Cache::cg>(&v_codes[key_l * D + d], &cache_v[off]);
            } else {
                store_vec(kd, make_int4(0, 0, 0, 0));
                if constexpr (Values::kCodeBytes == D)
                    store_vec(&v_codes[key_l * D + d], make_int4(0, 0, 0, 0));
            }
        }
        if constexpr (Values::kCodeBytes != D) {
#pragma unroll 1
            for (int chunk = cooperative_tid; chunk < Bc * (Values::kCodeBytes / 16);
                 chunk += cooperative_threads) {
                const int row = chunk / (Values::kCodeBytes / 16);
                const int col = (chunk % (Values::kCodeBytes / 16)) * 16;
                auto* dst     = v_codes + row * Values::kCodeBytes + col;
                if (tile_k0 + row <= max_query_abs) {
                    const auto off = paged_kv_element_offset<Values::kCodeBytes, Geometry::KVHeads>(
                        physical_page, kv_head, page_offset0 + row, col);
                    cp_async<16, Cache::cg>(dst, cache_v + off);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        ninfer::ops::cp_commit();
    };

    auto issue_kv_tile = [&](int tile_k0, int cooperative_tid, int cooperative_threads) {
        if (paged_kv_page_is_hole(block_table[tile_k0 >> kPagedKVPageShift])) {
            ninfer::ops::cp_commit();
            return;
        }
        issue_kv_scales(tile_k0, cooperative_tid, cooperative_threads);
        issue_kv_codes(tile_k0, cooperative_tid, cooperative_threads);
    };

    if (key_blocks > 0) issue_kv_tile(first_owned_tile * Bc, tid, Schedule::kThreads);
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    float acc[PVNtPerWarp][4]{};
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0F, l1 = 0.0F;
    const float scale_l2      = scale * kLog2E;
    const auto update_softmax = [&](float& maximum, float tile_maximum) {
        const float previous = maximum;
        maximum              = fmaxf(maximum, tile_maximum);
        return previous == -CUDART_INF_F
                   ? 0.0F
                   : causal_exp_scaled(previous, maximum * scale_l2, scale_l2);
    };
    const auto step = [&]<bool FullTile>(int kb) {
        const int k0 = (first_owned_tile + kb) * Bc;
        if (paged_kv_page_is_hole(block_table[k0 >> kPagedKVPageShift])) {
            if (kb + 1 < key_blocks) issue_kv_tile(k0 + Bc, tid, Schedule::kThreads);
            ninfer::ops::cp_wait<0>();
            __syncthreads();
            return;
        }
        // Conversion runs while all query warps retain their row state in registers.
#pragma unroll 1
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Schedule::kThreads) {
            const int row = chunk / (D / 8);
            const int d   = (chunk % (D / 8)) * 8;
            store_vec(&v_f16[row * D + causal_swizzle(row, d)],
                      Values::expand(
                          v_codes + row * Values::kCodeBytes + d * Values::kCodeBytes / D,
                          v_scale_s[row * Values::kScaleItems + d / (D / Values::kScaleItems)]));
        }
        // S = Q Kᵀ for this warp's 16 rows over all Bc keys, in registers.
        // Software-pipelined like cute's gemm: issue the ldmatrix for contraction
        // step k+1 while the m16n8k32 MMAs for step k run, so the LSU (ldmatrix)
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
                ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                            causal_swizzle_address(
                                k_lane_base + static_cast<unsigned>(nt2 * (8 * D)), 0u, k_as, k_r));
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
                        causal_swizzle_address(k_lane_base + static_cast<unsigned>(nt2 * (8 * D)),
                                               ck, k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_fp8_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                             af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int keya  = nt * 8 + 2 * lid;
            const float ks0 = __half2float(k_scale_s[keya]);
            const float ks1 = __half2float(k_scale_s[keya + 1]);
            score[nt][0] *= q_scale_r0 * ks0;
            score[nt][1] *= q_scale_r0 * ks1;
            score[nt][2] *= q_scale_r1 * ks0;
            score[nt][3] *= q_scale_r1 * ks1;
        }
        // QK is finished with the codes; V has been decoded into its own arena.
        __syncthreads();
        if (kb + 1 < key_blocks) issue_kv_tile(k0 + Bc, tid, Schedule::kThreads);
        const int row0  = warp_row0 + gid;
        const int row1  = warp_row0 + gid + 8;
        const int qrow0 = q0 + row0;
        const int qrow1 = q0 + row1;
        const int qabs0 = (qrow0 < tokens) ? base_pos + qrow0 : -1;
        const int qabs1 = (qrow1 < tokens) ? base_pos + qrow1 : -1;

        // Row maximum before attention scaling; scale is folded into exp2 below.
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

        const float alpha0     = update_softmax(m0, bm0);
        const float alpha1     = update_softmax(m1, bm1);
        const float nm0_scaled = m0 * scale_l2;
        const float nm1_scaled = m1 * scale_l2;

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

        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
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
        constexpr int PVHalf  = PVNtPerWarp / 2; // 16 n-tile pairs
        constexpr int PVLoads = PVKs * PVHalf;   // 64 x4.trans loads
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

        if (kb + 1 < key_blocks) ninfer::ops::cp_wait<0>();
        __syncthreads();
    };
    const int full_blocks =
        q0 + Br <= tokens ? min(key_blocks, max(0, (base_pos + q0 + 1) / Bc - first_owned_tile))
                          : 0;
    for (int kb = 0; kb < full_blocks; ++kb) step.template operator()<true>(kb);
    for (int kb = full_blocks; kb < key_blocks; ++kb) step.template operator()<false>(kb);
    l0             = warp_sum<4>(l0, FullMask);
    l1             = warp_sum<4>(l1, FullMask);
    const int row0 = warp_row0 + gid;
    const int row1 = row0 + 8;
    if (lid == 0) {
        if (row0 < tile_rows) {
            const auto index       = causal_stat_index<Geometry>(q_head, q0 + row0, split, width);
            partial.maximum[index] = m0 * scale;
            partial.sum[index]     = l0;
        }
        if (row1 < tile_rows) {
            const auto index       = causal_stat_index<Geometry>(q_head, q0 + row1, split, width);
            partial.maximum[index] = m1 * scale;
            partial.sum[index]     = l1;
        }
    }
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int d0 = n * 8 + 2 * lid;
        if (row0 < tile_rows)
            causal_store_partial_pair(
                partial.acc + causal_partial_index<Geometry>(q_head, d0, q0 + row0, split, width),
                acc[n][0], acc[n][1]);
        if (row1 < tile_rows)
            causal_store_partial_pair(
                partial.acc + causal_partial_index<Geometry>(q_head, d0, q0 + row1, split, width),
                acc[n][2], acc[n][3]);
    }
}
} // namespace ninfer::ops::detail
