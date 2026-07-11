#pragma once

#include "kernels/common/math.cuh"
#include "kernels/common/mma.cuh"
#include "kernels/common/warp.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace qus::kernels {

inline constexpr int kVisionAttentionHeadDim   = 72;
inline constexpr int kVisionAttentionHeads     = 16;
inline constexpr int kVisionAttentionBr        = 64;
inline constexpr int kVisionAttentionBc        = 64;
inline constexpr int kVisionAttentionPaddedD   = 128;
inline constexpr int kVisionAttentionThreads   = 128;
inline constexpr int kVisionAttentionSmemBytes = (kVisionAttentionBr + 2 * kVisionAttentionBc) *
                                                 kVisionAttentionPaddedD *
                                                 static_cast<int>(sizeof(__nv_bfloat16));

struct alignas(16) VisionAttentionTile {
    std::int32_t q0;
    std::int32_t begin;
    std::int32_t end;
    std::int32_t reserved;
};

static_assert(sizeof(VisionAttentionTile) == 16);

__device__ __forceinline__ const __nv_bfloat16*
vision_attention_ptr(const __nv_bfloat16* data, std::int64_t stride_d, std::int64_t stride_h,
                     std::int64_t stride_t, int d, int h, int t) {
    return data + static_cast<std::int64_t>(d) * stride_d +
           static_cast<std::int64_t>(h) * stride_h + static_cast<std::int64_t>(t) * stride_t;
}

__device__ __forceinline__ int vision_attention_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ unsigned vision_attention_swz_addr(unsigned lane_base, unsigned ck,
                                                              unsigned as, unsigned r) {
    return lane_base + ((ck | as) ^ r);
}

__global__ void vision_attention_prepare_tiles_kernel(const std::int32_t* cu_seqlens,
                                                      std::int32_t segments,
                                                      VisionAttentionTile* tiles,
                                                      std::int32_t max_tiles,
                                                      std::int32_t patches) {
    for (int tile = static_cast<int>(threadIdx.x); tile < max_tiles;
         tile += static_cast<int>(blockDim.x)) {
        tiles[tile] = {-1, 0, 0, 0};
    }
    __syncthreads();
    if (threadIdx.x != 0) { return; }

    int next = 0;
    for (int segment = 0; segment < segments; ++segment) {
        const int begin = cu_seqlens[segment];
        const int end   = cu_seqlens[segment + 1];
        if (begin < 0 || end <= begin || end > patches) { continue; }
        for (int q0 = begin; q0 < end && next < max_tiles; q0 += kVisionAttentionBr) {
            tiles[next++] = {q0, begin, end, 0};
        }
    }
}

__device__ __forceinline__ void vision_attention_stage_q(__nv_bfloat16* dst, const __nv_bfloat16* q,
                                                         int q0, int end, int head, int tid,
                                                         std::int64_t stride_d,
                                                         std::int64_t stride_h,
                                                         std::int64_t stride_t) {
    constexpr int VecsPerRow = kVisionAttentionPaddedD / 8;
    for (int chunk = tid; chunk < kVisionAttentionBr * VecsPerRow;
         chunk += kVisionAttentionThreads) {
        const int row       = chunk / VecsPerRow;
        const int d         = (chunk % VecsPerRow) * 8;
        const bool in_range = q0 + row < end && d < kVisionAttentionHeadDim;
        __nv_bfloat16* smem = &dst[row * kVisionAttentionPaddedD + vision_attention_swz(row, d)];
        const __nv_bfloat16* global = vision_attention_ptr(
            q, stride_d, stride_h, stride_t, in_range ? d : 0, head, in_range ? q0 + row : q0);
        cp_async_zfill<16, Cache::cg>(smem, global, in_range ? 16 : 0);
    }
}

__device__ __forceinline__ void
vision_attention_stage_kv(__nv_bfloat16* dst, const __nv_bfloat16* src, int key0, int end, int head,
                          int tid, std::int64_t stride_d, std::int64_t stride_h,
                          std::int64_t stride_t) {
    constexpr int VecsPerRow = kVisionAttentionPaddedD / 8;
    for (int chunk = tid; chunk < kVisionAttentionBc * VecsPerRow;
         chunk += kVisionAttentionThreads) {
        const int row       = chunk / VecsPerRow;
        const int d         = (chunk % VecsPerRow) * 8;
        const bool in_range = key0 + row < end && d < kVisionAttentionHeadDim;
        __nv_bfloat16* smem = &dst[row * kVisionAttentionPaddedD + vision_attention_swz(row, d)];
        const __nv_bfloat16* global =
            vision_attention_ptr(src, stride_d, stride_h, stride_t, in_range ? d : 0, head,
                                 in_range ? key0 + row : key0);
        cp_async_zfill<16, Cache::cg>(smem, global, in_range ? 16 : 0);
    }
}

__launch_bounds__(kVisionAttentionThreads, 2) __global__ void vision_attention_flash_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const VisionAttentionTile* __restrict__ tiles,
    std::int32_t patches, __nv_bfloat16* __restrict__ out, std::int64_t q_stride_d,
    std::int64_t q_stride_h, std::int64_t q_stride_t, std::int64_t k_stride_d,
    std::int64_t k_stride_h, std::int64_t k_stride_t, std::int64_t v_stride_d,
    std::int64_t v_stride_h, std::int64_t v_stride_t) {
    constexpr int D             = kVisionAttentionHeadDim;
    constexpr int Dp            = kVisionAttentionPaddedD;
    constexpr int Br            = kVisionAttentionBr;
    constexpr int Bc            = kVisionAttentionBc;
    constexpr int QKNt          = Bc / 8;
    constexpr int QKKs          = 5; // ceil(72 / 16)
    constexpr int PVNt          = D / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr int RowBytes      = Dp * static_cast<int>(sizeof(__nv_bfloat16));
    constexpr float ScaleLog2E  = 0.11785113019775792073f * 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    const VisionAttentionTile tile =
        tiles != nullptr
            ? tiles[blockIdx.x]
            : VisionAttentionTile{static_cast<std::int32_t>(blockIdx.x) * Br, 0, patches, 0};
    if (tile.q0 < 0) { return; }
    const int head = static_cast<int>(blockIdx.y);
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    extern __shared__ __align__(16) __nv_bfloat16 shared[];
    __nv_bfloat16* q_s = shared;
    __nv_bfloat16* k_s = q_s + Br * Dp;
    __nv_bfloat16* v_s = k_s + Bc * Dp;

    const int gid       = lane >> 2;
    const int lid       = lane & 3;
    const int a_mat     = lane >> 3;
    const int a_rin     = lane & 7;
    const int a_rowoff  = a_rin + ((a_mat & 1) << 3);
    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16;

    const unsigned q_sbase     = smem_addr(q_s);
    const unsigned k_sbase     = smem_addr(k_s);
    const unsigned v_sbase     = smem_addr(v_s);
    const unsigned q_lane_base = q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * RowBytes);
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
    const unsigned k_lane_base = k_sbase + static_cast<unsigned>(b_rin * RowBytes) +
                                 static_cast<unsigned>((lane >> 4) * 8 * RowBytes);
    const unsigned k_as        = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r         = static_cast<unsigned>(b_rin << 4);
    const unsigned v_lane_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 8 * RowBytes) +
                                 static_cast<unsigned>(b_rin * RowBytes);
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    vision_attention_stage_q(q_s, q, tile.q0, tile.end, head, tid, q_stride_d, q_stride_h,
                             q_stride_t);

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

    cp_commit();
    vision_attention_stage_kv(k_s, k, tile.begin, tile.end, head, tid, k_stride_d, k_stride_h,
                              k_stride_t);
    cp_commit();

    const int key_blocks = (tile.end - tile.begin + Bc - 1) / Bc;
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int key0 = tile.begin + kb * Bc;
        cp_wait<0>();
        __syncthreads();

        vision_attention_stage_kv(v_s, v, key0, tile.end, head, tid, v_stride_d, v_stride_h,
                                  v_stride_t);
        cp_commit();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                    vision_attention_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
        for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
            ldmatrix_x4(
                bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                vision_attention_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 8 * RowBytes),
                                          0u, k_as, k_r));
        }
#pragma unroll
        for (int ks = 0; ks < QKKs; ++ks) {
            const int cur = ks & 1;
            const int nxt = cur ^ 1;
            if (ks + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((ks + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            vision_attention_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0],
                                bf[nxt][nt2 + 1][1],
                                vision_attention_swz_addr(
                                    k_lane_base + static_cast<unsigned>(nt2 * 8 * RowBytes), ck,
                                    k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                         af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

        const int row0       = warp_row0 + gid;
        const int row1       = row0 + 8;
        const int query0     = tile.q0 + row0;
        const int query1     = tile.q0 + row1;
        const bool full_tile = tile.q0 + Br <= tile.end && key0 + Bc <= tile.end;
        float block_max0     = -CUDART_INF_F;
        float block_max1     = -CUDART_INF_F;
        if (full_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                block_max0 = fmaxf(block_max0, fmaxf(score[nt][0], score[nt][1]));
                block_max1 = fmaxf(block_max1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key_a = key0 + nt * 8 + 2 * lid;
                const int key_b = key_a + 1;
                score[nt][0] = query0 < tile.end && key_a < tile.end ? score[nt][0] : -CUDART_INF_F;
                score[nt][1] = query0 < tile.end && key_b < tile.end ? score[nt][1] : -CUDART_INF_F;
                score[nt][2] = query1 < tile.end && key_a < tile.end ? score[nt][2] : -CUDART_INF_F;
                score[nt][3] = query1 < tile.end && key_b < tile.end ? score[nt][3] : -CUDART_INF_F;
                block_max0   = fmaxf(block_max0, fmaxf(score[nt][0], score[nt][1]));
                block_max1   = fmaxf(block_max1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        block_max0 = warp_max<4>(block_max0, FullMask);
        block_max1 = warp_max<4>(block_max1, FullMask);

        const float next_m0 = fmaxf(m0, block_max0);
        const float next_m1 = fmaxf(m1, block_max1);
        const float m0_l2   = next_m0 * ScaleLog2E;
        const float m1_l2   = next_m1 * ScaleLog2E;
        const float alpha0  = exp2_approx(__fmaf_rn(m0, ScaleLog2E, -m0_l2));
        const float alpha1  = exp2_approx(__fmaf_rn(m1, ScaleLog2E, -m1_l2));

        float block_sum0 = 0.0f;
        float block_sum1 = 0.0f;
        unsigned p_frag[PVKs][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = score[nt][0] > -CUDART_INF_F
                                  ? exp2_approx(__fmaf_rn(score[nt][0], ScaleLog2E, -m0_l2))
                                  : 0.0f;
            const float p01 = score[nt][1] > -CUDART_INF_F
                                  ? exp2_approx(__fmaf_rn(score[nt][1], ScaleLog2E, -m0_l2))
                                  : 0.0f;
            const float p10 = score[nt][2] > -CUDART_INF_F
                                  ? exp2_approx(__fmaf_rn(score[nt][2], ScaleLog2E, -m1_l2))
                                  : 0.0f;
            const float p11 = score[nt][3] > -CUDART_INF_F
                                  ? exp2_approx(__fmaf_rn(score[nt][3], ScaleLog2E, -m1_l2))
                                  : 0.0f;
            block_sum0 += p00 + p01;
            block_sum1 += p10 + p11;
            const int pk = nt >> 1;
            if ((nt & 1) == 0) {
                p_frag[pk][0] = pack_bf16x2(p00, p01);
                p_frag[pk][1] = pack_bf16x2(p10, p11);
            } else {
                p_frag[pk][2] = pack_bf16x2(p00, p01);
                p_frag[pk][3] = pack_bf16x2(p10, p11);
            }
        }

        l0 = __fmaf_rn(l0, alpha0, block_sum0);
        l1 = __fmaf_rn(l1, alpha1, block_sum1);
        m0 = next_m0;
        m1 = next_m1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        cp_wait<0>();
        __syncthreads();
        if (kb + 1 < key_blocks) {
            vision_attention_stage_kv(k_s, k, key0 + Bc, tile.end, head, tid, k_stride_d,
                                      k_stride_h, k_stride_t);
            cp_commit();
        }

        constexpr int PVTilePairs = (PVNt + 1) / 2;
        constexpr int PVLoads     = PVKs * PVTilePairs;
        unsigned vf[2][4];
        ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                      vision_attention_swz_addr(v_lane_base, 0u, v_as, v_r));
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
                              vision_attention_swz_addr(
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
    }

    l0                 = warp_sum<4>(l0, FullMask);
    l1                 = warp_sum<4>(l1, FullMask);
    const float inv_l0 = l0 > 0.0f ? __frcp_rn(l0) : 0.0f;
    const float inv_l1 = l1 > 0.0f ? __frcp_rn(l1) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0     = n * 8 + 2 * lid;
        const int query0 = tile.q0 + warp_row0 + gid;
        const int query1 = query0 + 8;
        if (query0 < tile.end) {
            const std::int64_t offset =
                (static_cast<std::int64_t>(query0) * kVisionAttentionHeads + head) * D + d0;
            store_vec(&out[offset], pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0));
        }
        if (query1 < tile.end) {
            const std::int64_t offset =
                (static_cast<std::int64_t>(query1) * kVisionAttentionHeads + head) * D + d0;
            store_vec(&out[offset], pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1));
        }
    }
}

} // namespace qus::kernels
