#pragma once
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tile_io.cuh"
#include "ops/common/mbarrier.cuh"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/schedule.cuh"
#include "ops/softmax_attention/common/causal_epilogue.cuh"
#include "ops/softmax_attention/common/causal_operands.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/common/causal_softmax.cuh"

namespace ninfer::ops::detail {
struct Nvfp4KvBarriers {
    alignas(8) std::uint64_t k_full;
    alignas(8) std::uint64_t k_empty;
    alignas(8) std::uint64_t v_full;
    alignas(8) std::uint64_t v_empty;
};

template <typename Geometry, typename Schedule>
__device__ __forceinline__ void
nvfp4_kv_decode_tile(__half* destination, const std::uint8_t* cache,
                     const std::uint8_t* cache_scale, const std::int32_t* block_table, int kv_head,
                     int tile_k0, int max_query_abs, int producer_tid) {
    constexpr int D               = 256;
    constexpr int Bc              = Schedule::kKeyRows;
    constexpr int GroupsPerTile   = Bc * kKVCacheNvfp4Groups;
    constexpr int ProducerThreads = Schedule::kProducerThreads;
    const int physical_page       = block_table[tile_k0 >> kPagedKVPageShift];
    const int page_offset0        = tile_k0 & kPagedKVPageMask;

#pragma unroll 1
    for (int task = producer_tid; task < GroupsPerTile; task += ProducerThreads) {
        const int key_l   = task / kKVCacheNvfp4Groups;
        const int group   = task - key_l * kKVCacheNvfp4Groups;
        const int d       = group * kKVCacheNvfp4Group;
        const int key     = tile_k0 + key_l;
        __half* target_lo = destination + key_l * D + causal_swizzle(key_l, d);
        __half* target_hi = destination + key_l * D + causal_swizzle(key_l, d + 8);
        if (key <= max_query_abs) {
            const std::int64_t code_offset = kv_cache_nvfp4_code_index<Geometry>(
                physical_page, kv_head, d, page_offset0 + key_l);
            const std::int64_t scale_offset = kv_cache_nvfp4_scale_index<Geometry>(
                physical_page, kv_head, group, page_offset0 + key_l);
            const auto represented =
                kv_cache_nvfp4_dequant_f16x16(cache + code_offset, cache_scale[scale_offset]);
            store_vec(target_lo, represented.lo);
            store_vec(target_hi, represented.hi);
        } else {
            store_vec(target_lo, make_int4(0, 0, 0, 0));
            store_vec(target_hi, make_int4(0, 0, 0, 0));
        }
    }
}

template <typename Geometry, typename Schedule, typename Metadata>
__global__ __launch_bounds__(Schedule::kThreads, 1) void nvfp4_kv_tiled_mma_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const std::uint8_t* __restrict__ cache_k_scale,
    const std::uint8_t* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, std::int32_t width,
    CausalKvPartition partition, CausalPartialView partial) {
    constexpr int D             = 256;
    constexpr int Br            = Schedule::kQueryRows;
    constexpr int Bc            = Schedule::kKeyRows;
    constexpr int QKNt          = Bc / 8;
    constexpr int QKKs          = D / 16;
    constexpr int PVNt          = D / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr float Log2E       = kLog2E;
    constexpr unsigned FullMask = 0xffffffffU;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* q_f16  = reinterpret_cast<__half*>(smem_raw);
    __half* k_f16  = q_f16 + Br * D;
    __half* v_f16  = k_f16 + Bc * D;
    auto* barriers = reinterpret_cast<Nvfp4KvBarriers*>(v_f16 + Bc * D);

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

    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int tile_rows             = min(Br, tokens - q0);
    const int max_query_abs         = base_pos + q0 + tile_rows - 1;
    const int logical_tiles         = div_up(visible, Bc);
    const int first_owned_tile      = split * logical_tiles / active_splits;
    const int end_owned_tile        = (split + 1) * logical_tiles / active_splits;
    const int key_blocks = max(0, min(end_owned_tile, max_query_abs / Bc + 1) - first_owned_tile);

    if (tid == 0) {
        cta_mbarrier_init(&barriers->k_full, 1);
        cta_mbarrier_init(&barriers->k_empty, 1);
        cta_mbarrier_init(&barriers->v_full, 1);
        cta_mbarrier_init(&barriers->v_empty, 1);
        cta_mbarrier_fence_init();
    }

    for (int row = warp; row < Br; row += Schedule::kProducerWarps + Schedule::kConsumerWarps) {
        float values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            values[r]   = row < tile_rows
                              ? __bfloat162float(q[causal_q_index<Geometry>(q_head, d, q0 + row)])
                              : 0.0F;
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d                             = lane + 32 * r;
            q_f16[row * D + causal_swizzle(row, d)] = __float2half_rn(values[r]);
        }
    }
    __syncthreads();

    if (tid < Schedule::kProducerThreads) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;"
                     :
                     : "n"(Schedule::kProducerRegisters)
                     : "memory");
        const int producer_tid = tid;
        for (int kb = 0; kb < key_blocks; ++kb) {
            const std::uint32_t empty_phase = 1U ^ static_cast<std::uint32_t>(kb & 1);
            cta_mbarrier_wait(&barriers->k_empty, empty_phase);
            nvfp4_kv_decode_tile<Geometry, Schedule>(k_f16, cache_k, cache_k_scale, block_table,
                                                     kv_head, (first_owned_tile + kb) * Bc,
                                                     max_query_abs, producer_tid);
            asm volatile("bar.sync 1, %0;" : : "r"(Schedule::kProducerThreads) : "memory");
            if (producer_tid == 0) { cta_mbarrier_arrive(&barriers->k_full); }

            cta_mbarrier_wait(&barriers->v_empty, empty_phase);
            nvfp4_kv_decode_tile<Geometry, Schedule>(v_f16, cache_v, cache_v_scale, block_table,
                                                     kv_head, (first_owned_tile + kb) * Bc,
                                                     max_query_abs, producer_tid);
            asm volatile("bar.sync 1, %0;" : : "r"(Schedule::kProducerThreads) : "memory");
            if (producer_tid == 0) { cta_mbarrier_arrive(&barriers->v_full); }
        }
        return;
    }

    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;"
                 :
                 : "n"(Schedule::kConsumerRegisters)
                 : "memory");
    const int consumer_tid  = tid - Schedule::kProducerThreads;
    const int consumer_warp = consumer_tid >> 5;
    const int gid           = lane >> 2;
    const int lid           = lane & 3;
    const int a_mat         = lane >> 3;
    const int a_rin         = lane & 7;
    const int a_rowoff      = a_rin + ((a_mat & 1) << 3);
    const int b_rin         = lane & 7;
    const int b_koff        = ((lane >> 3) & 1) << 3;
    const int warp_row0     = consumer_warp * 16;

    const unsigned q_sbase     = smem_addr(q_f16);
    const unsigned k_sbase     = smem_addr(k_f16);
    const unsigned v_sbase     = smem_addr(v_f16);
    const unsigned q_lane_base = q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * 512);
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
    const unsigned k_lane_base =
        k_sbase + static_cast<unsigned>(b_rin * 512) + (static_cast<unsigned>(lane >> 4) << 12);
    const unsigned k_as        = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r         = static_cast<unsigned>(b_rin << 4);
    const unsigned v_lane_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
                                 static_cast<unsigned>(b_rin * 512);
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[n][i] = 0.0F;
    }
    float m0             = -CUDART_INF_F;
    float m1             = -CUDART_INF_F;
    float l0             = 0.0F;
    float l1             = 0.0F;
    const float scale_l2 = scale * Log2E;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0                   = (first_owned_tile + kb) * Bc;
        const std::uint32_t full_phase = static_cast<std::uint32_t>(kb & 1);
        cta_mbarrier_wait(&barriers->k_full, full_phase);

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt)
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0F;

        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                    causal_swizzle_address(q_lane_base, 0U, q_as, q_r));
#pragma unroll
        for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
            ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                        causal_swizzle_address(k_lane_base + static_cast<unsigned>(nt2 * 4096), 0U,
                                               k_as, k_r));
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
                        causal_swizzle_address(k_lane_base + static_cast<unsigned>(nt2 * 4096), ck,
                                               k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_f16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                        af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

        asm volatile("bar.sync 2, 128;" : : : "memory");
        if (consumer_tid == 0) { cta_mbarrier_arrive(&barriers->k_empty); }

        const int row0             = warp_row0 + gid;
        const int row1             = row0 + 8;
        const int qrow0            = q0 + row0;
        const int qrow1            = q0 + row1;
        const int qabs0            = qrow0 < tokens ? base_pos + qrow0 : -1;
        const int qabs1            = qrow1 < tokens ? base_pos + qrow1 : -1;
        const bool full_score_tile =
            q0 + Br <= tokens && (first_owned_tile + kb) < (base_pos + q0 + 1) / Bc;
        float bm0                  = -CUDART_INF_F;
        float bm1                  = -CUDART_INF_F;
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
                score[nt][0]   = qrow0 < tokens && key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = qrow0 < tokens && key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = qrow1 < tokens && key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = qrow1 < tokens && key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
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
        const float alpha0 =
            m0 == -CUDART_INF_F ? 0.0F : causal_exp_scaled(m0, nm0_scaled, scale_l2);
        const float alpha1 =
            m1 == -CUDART_INF_F ? 0.0F : causal_exp_scaled(m1, nm1_scaled, scale_l2);
        float bl0 = 0.0F;
        float bl1 = 0.0F;
        unsigned p_frag[PVKs][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = score[nt][0] > -CUDART_INF_F
                                  ? causal_exp_scaled(score[nt][0], nm0_scaled, scale_l2)
                                  : 0.0F;
            const float p01 = score[nt][1] > -CUDART_INF_F
                                  ? causal_exp_scaled(score[nt][1], nm0_scaled, scale_l2)
                                  : 0.0F;
            const float p10 = score[nt][2] > -CUDART_INF_F
                                  ? causal_exp_scaled(score[nt][2], nm1_scaled, scale_l2)
                                  : 0.0F;
            const float p11 = score[nt][3] > -CUDART_INF_F
                                  ? causal_exp_scaled(score[nt][3], nm1_scaled, scale_l2)
                                  : 0.0F;
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

        cta_mbarrier_wait(&barriers->v_full, full_phase);
        constexpr int PVHalf  = PVNt / 2;
        constexpr int PVLoads = PVKs * PVHalf;
        unsigned vf[2][4];
        ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                      causal_swizzle_address(v_lane_base, 0U, v_as, v_r));
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
                              causal_swizzle_address(v_lane_base + static_cast<unsigned>(k2 * 8192),
                                                     ckv, v_as, v_r));
            }
            mma_f16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
                    p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_f16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_frag[k][0],
                    p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);
        }
        asm volatile("bar.sync 2, 128;" : : : "memory");
        if (consumer_tid == 0) { cta_mbarrier_arrive(&barriers->v_empty); }
    }

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
    for (int n = 0; n < PVNt; ++n) {
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
